from dataclasses import replace
import json
import math
import unittest

from studio.antenna_agent import create_default_agent
from studio.antenna_agent_runner import AntennaAgentRunner
from studio.antenna_analysis import EngineeringAnalysisResult, SPEED_OF_LIGHT_MM_GHZ
from studio.antenna_design import BooleanOperation, GeometryObject, TransformSpec
from studio.antenna_llm_planner import (
    AgentLoopBudgets,
    AgentStep,
    AgentStepObservation,
    GeminiSchemaConstrainedPlanner,
    OpenRouterNemotronPlanner,
    PlannedToolCall,
)


def step(*calls, status="execute", message="test step"):
    return AgentStep(
        status, message,
        tuple(PlannedToolCall(name, arguments) for name, arguments in calls),
    )


class ScriptedPlanner:
    def __init__(self, *responses):
        self.responses = list(responses)

    def plan_agent_step(self, **_request):
        return self.responses.pop(0)


class DipoleBaselineAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.agent = create_default_agent()
        self.design = self.agent.create_design("dipole")

    def analyze(self, design=None):
        call = PlannedToolCall("engineering.dipole_baseline", {})
        return self.agent.execute_analysis_batch(design or self.design, (call,))[0]

    @staticmethod
    def values(result):
        return {key: item.value for key, item in result.measurements.items()}

    def update(self, design=None, **values):
        return self.agent.update_parameters(design or self.design, values).design

    def half_wave_design(self, *, scale=1.0):
        wavelength = SPEED_OF_LIGHT_MM_GHZ / self.design.value("frequency_ghz")
        gap = self.design.value("feed_gap_mm")
        arm = (scale * wavelength / 2.0 - gap) / 2.0
        return self.update(arm_length_mm=arm)

    @staticmethod
    def transform_arms(design, transform):
        return replace(
            design,
            geometry=tuple(
                replace(item, transform=transform) if "dipole_arm" in item.tags else item
                for item in design.geometry
            ),
        )

    def unequal_arms(self):
        geometry = []
        for item in self.design.geometry:
            if item.object_id.endswith("upper_arm"):
                dimensions = dict(item.dimensions)
                dimensions["end"] = "feed_gap_mm/2+arm_length_mm*1.25"
                item = replace(item, dimensions=tuple(dimensions.items()))
            geometry.append(item)
        return replace(self.design, geometry=tuple(geometry))

    def modified_arm(self):
        lower = next(item for item in self.design.geometry if item.object_id.endswith("lower_arm"))
        tool = GeometryObject(
            object_id="arm_loading_tool", primitive="cylinder", name="Arm loading tool",
            material_id=lower.material_id,
            dimensions=(
                ("center_1", "conductor_radius_mm"), ("center_2", 0),
                ("radius", "conductor_radius_mm*0.6"),
                ("start", "-feed_gap_mm/2-arm_length_mm"), ("end", "-feed_gap_mm/2"),
            ),
            axis="z", tags=("planner_created", "boolean_tool", "custom_geometry"),
        )
        operation = BooleanOperation(
            "arm_loading_union", "union", lower.object_id, (tool.object_id,),
        )
        return replace(
            self.design,
            geometry=(*self.design.geometry, tool),
            booleans=(*self.design.booleans, operation),
        )

    def test_exact_half_wave_total_and_quarter_wave_arms(self):
        half_wave = self.half_wave_design()
        values = self.values(self.analyze(half_wave))
        self.assertAlmostEqual(values["current_total_dipole_length_lambda_ratio"], 0.5, places=12)
        wavelength = values["lambda0"]
        quarter_arms = self.update(
            arm_length_mm=wavelength / 4.0,
            feed_gap_mm=max(self.design.value("feed_gap_mm"), 1.2),
        )
        values = self.values(self.analyze(quarter_arms))
        self.assertAlmostEqual(values["current_arm_1_length_lambda_ratio"], 0.25, places=12)
        self.assertAlmostEqual(values["current_arm_2_length_lambda_ratio"], 0.25, places=12)

    def test_shorter_and_longer_signed_percent_differences(self):
        for scale in (0.8, 1.2):
            with self.subTest(scale=scale):
                values = self.values(self.analyze(self.half_wave_design(scale=scale)))
                reference = values["half_wave_length_reference"]
                expected = (scale - 1.0) * reference
                self.assertAlmostEqual(values["total_length_difference_from_half_wave"], expected, places=10)
                self.assertAlmostEqual(values["total_length_percent_difference_from_half_wave"], 100 * (scale - 1.0), places=10)

    def test_unequal_arms_are_preserved(self):
        values = self.values(self.analyze(self.unequal_arms()))
        self.assertNotEqual(values["current_arm_1_length"], values["current_arm_2_length"])
        self.assertAlmostEqual(
            values["current_arm_2_length"], values["current_arm_1_length"] * 1.25,
            places=12,
        )

    def test_frequency_change_updates_wavelength_and_normalized_dimensions(self):
        before = self.values(self.analyze())
        changed = self.update(frequency_ghz=5.8)
        after = self.values(self.analyze(changed))
        self.assertLess(after["lambda0"], before["lambda0"])
        self.assertGreater(
            after["current_total_dipole_length_lambda_ratio"],
            before["current_total_dipole_length_lambda_ratio"],
        )

    def test_feed_gap_change_updates_geometry_and_normalized_gap(self):
        before = self.values(self.analyze())
        changed = self.update(feed_gap_mm=2.0)
        after = self.values(self.analyze(changed))
        self.assertAlmostEqual(after["current_feed_gap"], 2.0, places=12)
        self.assertGreater(after["current_feed_gap_lambda_ratio"], before["current_feed_gap_lambda_ratio"])

    def test_radius_change_updates_radius_diameter_and_normalized_thickness(self):
        changed = self.update(conductor_radius_mm=0.5)
        values = self.values(self.analyze(changed))
        self.assertAlmostEqual(values["current_conductor_radius"], 0.5, places=12)
        self.assertAlmostEqual(values["current_conductor_diameter"], 1.0, places=12)
        self.assertAlmostEqual(
            values["current_conductor_radius_lambda_ratio"], 0.5 / values["lambda0"], places=12,
        )

    def test_translation_and_rotation_preserve_physical_dimensions(self):
        original = self.values(self.analyze())
        for transform in (
            TransformSpec(translate_mm=(13.0, -7.0, 21.0)),
            TransformSpec(rotate_deg=(37.0, 28.0, 19.0)),
            TransformSpec(translate_mm=(3.0, 4.0, 5.0), rotate_deg=(90.0, 25.0, 12.0)),
        ):
            with self.subTest(transform=transform):
                values = self.values(self.analyze(self.transform_arms(self.design, transform)))
                for key in (
                    "current_arm_1_length", "current_arm_2_length", "current_feed_gap",
                    "current_total_dipole_length", "current_tip_to_tip_span",
                ):
                    self.assertAlmostEqual(values[key], original[key], places=10)

    def test_modified_conductor_topology_is_reference_only(self):
        result = self.analyze(self.modified_arm())
        self.assertEqual(result.status, "completed")
        self.assertIn("modified topology", result.applicability.casefold())
        self.assertIn("reference-only", result.applicability)
        self.assertTrue(any("composed geometry" in item for item in result.limitations))

    def test_missing_frequency_or_arm_geometry_is_unknown_without_guessing(self):
        no_frequency = replace(
            self.design,
            parameters=tuple(item for item in self.design.parameters if item.key != "frequency_ghz"),
        )
        self.assertEqual(self.analyze(no_frequency).status, "unknown")
        one_arm = replace(
            self.design,
            geometry=tuple(item for item in self.design.geometry if not item.object_id.endswith("upper_arm")),
        )
        result = self.analyze(one_arm)
        self.assertEqual(result.status, "unknown")
        self.assertNotIn("current_total_dipole_length", result.measurements)

    def test_patch_families_are_not_applicable(self):
        for recipe in ("inset_patch", "circular_patch"):
            with self.subTest(recipe=recipe):
                result = self.analyze(self.agent.create_design(recipe))
                self.assertEqual(result.status, "not_applicable")
                self.assertEqual(result.measurements, {})

    def test_analysis_does_not_mutate_design_revision_and_all_units_are_explicit(self):
        snapshot = self.design.to_dict()
        result = self.analyze()
        self.assertEqual(self.design.to_dict(), snapshot)
        self.assertEqual(result.working_design_ref.revision, self.design.revision)
        self.assertTrue(all(item.unit for item in result.measurements.values()))
        self.assertEqual(EngineeringAnalysisResult.from_dict(result.to_dict()), result)

    def test_cache_reuse_and_semantic_change_invalidation(self):
        call = ("engineering.dipole_baseline", {})
        planner = ScriptedPlanner(
            step(call), step(call),
            step(("parameter.set", {"key": "frequency_ghz", "value": 3.0})),
            step(call), step(status="finish", message="Done."),
        )
        result = AntennaAgentRunner(self.agent).run(
            baseline_design=self.design, instruction="Check dipole length.",
            project_memory={"schema_version": 1}, planner=planner,
        )
        self.assertEqual(result.trajectory[0].analysis_cache_hits, 0)
        self.assertEqual(result.trajectory[1].analysis_cache_hits, 1)
        self.assertEqual(result.trajectory[3].analysis_cache_hits, 0)
        self.assertNotEqual(
            result.trajectory[1].analysis_results[0].analysis_id,
            result.trajectory[3].analysis_results[0].analysis_id,
        )

    def test_manifest_exposes_read_only_empty_argument_schema(self):
        tool = next(
            item for item in self.agent.capability_manifest(self.design)["callable_tools"]
            if item["name"] == "engineering.dipole_baseline"
        )
        self.assertEqual(tool["effect"], "analysis")
        self.assertEqual(tool["arguments"], {
            "type": "object", "additionalProperties": False, "properties": {},
        })

    def test_gemini_and_nemotron_receive_identical_schema_and_result(self):
        design = self.unequal_arms()
        result = self.analyze(design)
        call = PlannedToolCall("engineering.dipole_baseline", {})
        observation = AgentStepObservation(
            "accepted_analysis", (call,), {"design_id": design.design_id, "revision": design.revision},
            result.semantic_design_hash, AgentLoopBudgets(4, 3, 2, 23, 2),
            executed_calls=(call,), validation_result={"status": "read_only", "design_unchanged": True},
            analysis_results=(result,),
        )
        exchanges = []
        for cls in (GeminiSchemaConstrainedPlanner, OpenRouterNemotronPlanner):
            planner = cls.__new__(cls)
            planner._agent_loop_remaining_budgets = observation.remaining_budgets
            planner._agent_loop_engineering_report = None
            exchanges.append(planner._exchange(
                instruction="Check the electrical length of this dipole.", current_design=design.to_dict(),
                capability_manifest=self.agent.capability_manifest(design), project_memory=None,
                agent_observation=observation, execution_feedback=None,
            ))
        self.assertEqual(exchanges[0], exchanges[1])
        payload = json.loads(exchanges[0].user_content)
        self.assertEqual(payload["agent_observation"]["analysis_results"], [result.to_dict()])


if __name__ == "__main__":
    unittest.main()
