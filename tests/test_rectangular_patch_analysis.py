from dataclasses import replace
import json
import math
import unittest

from studio.antenna_agent import create_default_agent
from studio.antenna_agent_runner import AntennaAgentRunner
from studio.antenna_analysis import (
    EngineeringAnalysisResult,
    SPEED_OF_LIGHT_MM_GHZ,
    rectangular_patch_baseline_dimensions,
)
from studio.antenna_llm_planner import (
    AgentLoopBudgets,
    AgentStep,
    AgentStepObservation,
    GeminiSchemaConstrainedPlanner,
    LLMToolPlan,
    OpenRouterNemotronPlanner,
    PlannedToolCall,
)


def plan(*calls):
    return LLMToolPlan(
        "execute", "test plan",
        tuple(PlannedToolCall(name, arguments) for name, arguments in calls),
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


class RectangularPatchBaselineTests(unittest.TestCase):
    def setUp(self):
        self.agent = create_default_agent()
        self.design = self.agent.create_design("inset_patch")

    def analyze(self, design=None):
        call = PlannedToolCall("engineering.rectangular_patch_baseline", {})
        return self.agent.execute_analysis_batch(design or self.design, (call,))[0]

    @staticmethod
    def values(result):
        return {key: item.value for key, item in result.measurements.items()}

    def set_parameters(self, design, **values):
        return self.agent.execute_llm_plan(
            design,
            plan(*( ("parameter.set", {"key": key, "value": value}) for key, value in values.items() )),
        ).design

    def center_slot(self, design=None):
        return self.agent.execute_llm_plan(design or self.design, plan(
            ("parameter.create", {
                "key": "slot_radius_mm", "label": "Slot radius", "value": 3.0,
                "unit": "mm", "sweepable": True,
            }),
            ("geometry.cylinder", {
                "object_id": "center_slot_tool", "material_id": "copper", "axis": "z",
                "tags": ["planner_created", "boolean_tool", "slot"],
                "dimensions": {
                    "center_1": 0, "center_2": 0, "radius": "slot_radius_mm",
                    "start": "substrate_thickness_mm",
                    "end": "substrate_thickness_mm+copper_thickness_mm",
                },
            }),
            ("boolean.subtract", {
                "operation_id": "center_slot_subtract",
                "target_id": "element_1_1_patch",
                "tool_ids": ["center_slot_tool"],
            }),
        )).design

    def edge_union(self):
        return self.agent.execute_llm_plan(self.design, plan(
            ("parameter.create", {
                "key": "edge_radius_mm", "label": "Edge radius", "value": 2.0,
                "unit": "mm", "sweepable": True,
            }),
            ("geometry.cylinder", {
                "object_id": "edge_union_tool", "material_id": "copper", "axis": "z",
                "tags": ["planner_created", "boolean_tool"],
                "dimensions": {
                    "center_1": "patch_width_mm/2", "center_2": 0,
                    "radius": "edge_radius_mm", "start": "substrate_thickness_mm",
                    "end": "substrate_thickness_mm+copper_thickness_mm",
                },
            }),
            ("boolean.union", {
                "operation_id": "edge_circle_union",
                "target_id": "element_1_1_patch",
                "tool_ids": ["edge_union_tool"],
            }),
        )).design

    def test_equations_are_numerically_verified(self):
        frequency, epsilon_r, height = 2.45, 4.3, 1.6
        result = rectangular_patch_baseline_dimensions(frequency, epsilon_r, height)
        expected_width = SPEED_OF_LIGHT_MM_GHZ / (2 * frequency) * math.sqrt(2 / (epsilon_r + 1))
        expected_effective = ((epsilon_r + 1) / 2
                              + (epsilon_r - 1) / 2 / math.sqrt(1 + 12 * height / expected_width))
        expected_extension = 0.412 * height * (
            (expected_effective + 0.3) * (expected_width / height + 0.264)
            / ((expected_effective - 0.258) * (expected_width / height + 0.8))
        )
        expected_length = SPEED_OF_LIGHT_MM_GHZ / (2 * frequency * math.sqrt(expected_effective)) - 2 * expected_extension
        self.assertAlmostEqual(result.width_mm, expected_width, places=12)
        self.assertAlmostEqual(result.effective_epsilon_r, expected_effective, places=12)
        self.assertAlmostEqual(result.fringing_extension_mm, expected_extension, places=12)
        self.assertAlmostEqual(result.length_mm, expected_length, places=12)

    def test_default_fr4_patch_returns_estimated_and_current_dimensions(self):
        result = self.analyze()
        values = self.values(result)
        self.assertEqual(result.status, "completed")
        self.assertEqual(values["frequency"], 2.45)
        self.assertEqual(values["epsilon_r"], 4.4)
        self.assertAlmostEqual(values["estimated_patch_width"], 37.234261, places=6)
        self.assertAlmostEqual(values["estimated_patch_length"], 28.809290, places=6)
        self.assertEqual(values["current_patch_width"], self.design.value("patch_width_mm"))
        self.assertEqual(values["current_patch_length"], self.design.value("patch_length_mm"))
        self.assertIn("Conventional unmodified", result.applicability)

    def test_frequency_change_updates_estimated_dimensions(self):
        before = self.values(self.analyze())
        changed = self.set_parameters(self.design, frequency_ghz=5.8)
        after = self.values(self.analyze(changed))
        self.assertLess(after["estimated_patch_width"], before["estimated_patch_width"])
        self.assertLess(after["estimated_patch_length"], before["estimated_patch_length"])

    def test_epsilon_change_updates_estimated_dimensions(self):
        before = self.values(self.analyze())
        materials = tuple(
            replace(item, epsilon_r=2.2) if item.kind == "dielectric" else item
            for item in self.design.materials
        )
        changed = replace(self.design, materials=materials)
        after = self.values(self.analyze(changed))
        self.assertGreater(after["estimated_patch_width"], before["estimated_patch_width"])
        self.assertGreater(after["estimated_patch_length"], before["estimated_patch_length"])

    def test_substrate_thickness_updates_effective_epsilon_and_fringing(self):
        before = self.values(self.analyze())
        changed = self.set_parameters(self.design, substrate_thickness_mm=3.2)
        after = self.values(self.analyze(changed))
        self.assertNotEqual(after["effective_dielectric_constant"], before["effective_dielectric_constant"])
        self.assertGreater(after["fringing_extension"], before["fringing_extension"])

    def test_array_analyzes_one_element_not_array_footprint(self):
        array = self.set_parameters(self.design, array_rows=2, array_columns=3)
        result = self.analyze(array)
        values = self.values(result)
        self.assertEqual(values["current_patch_width"], array.value("patch_width_mm"))
        self.assertNotEqual(values["current_patch_width"], array.value("board_width_mm"))
        self.assertIn("one element", result.applicability)
        self.assertIn("total array footprint is not analyzed", result.applicability)

    def test_center_slot_is_generic_modified_topology(self):
        result = self.analyze(self.center_slot())
        self.assertEqual(result.status, "completed")
        self.assertIn("modified topology", result.applicability.casefold())
        self.assertIn("reference-only", result.applicability)
        self.assertTrue(any("composed geometry" in item for item in result.limitations))

    def test_edge_union_is_generic_modified_topology(self):
        result = self.analyze(self.edge_union())
        self.assertIn("modified topology", result.applicability.casefold())
        self.assertIn("reference-only", result.applicability)

    def test_missing_epsilon_is_unknown_without_material_name_guess(self):
        no_dielectric = replace(
            self.design,
            materials=tuple(item for item in self.design.materials if item.kind != "dielectric"),
        )
        result = self.analyze(no_dielectric)
        self.assertEqual(result.status, "unknown")
        self.assertNotIn("epsilon_r", result.measurements)
        self.assertIn("relative permittivity", result.applicability)

    def test_unsupported_family_is_not_applicable(self):
        result = self.analyze(self.agent.create_design("dipole"))
        self.assertEqual(result.status, "not_applicable")
        self.assertEqual(result.measurements, {})

    def test_analysis_does_not_mutate_design_or_revision_and_units_are_explicit(self):
        snapshot = self.design.to_dict()
        result = self.analyze()
        self.assertEqual(self.design.to_dict(), snapshot)
        self.assertEqual(result.working_design_ref.revision, self.design.revision)
        self.assertTrue(all(item.unit for item in result.measurements.values()))
        self.assertEqual(EngineeringAnalysisResult.from_dict(result.to_dict()), result)

    def test_cache_reuse_and_design_change_invalidation(self):
        call = ("engineering.rectangular_patch_baseline", {})
        runner = AntennaAgentRunner(self.agent)
        planner = ScriptedPlanner(
            step(call), step(call),
            step(("parameter.set", {"key": "frequency_ghz", "value": 3.0})),
            step(call), step(status="finish", message="Done."),
        )
        result = runner.run(
            baseline_design=self.design, instruction="Check patch sizing.",
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
            if item["name"] == "engineering.rectangular_patch_baseline"
        )
        self.assertEqual(tool["effect"], "analysis")
        self.assertEqual(tool["arguments"], {
            "type": "object", "additionalProperties": False, "properties": {},
        })

    def test_gemini_and_nemotron_receive_identical_schema_and_result(self):
        design = self.center_slot()
        result = self.analyze(design)
        call = PlannedToolCall("engineering.rectangular_patch_baseline", {})
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
                instruction="Check the analytical patch sizing.", current_design=design.to_dict(),
                capability_manifest=self.agent.capability_manifest(design), project_memory=None,
                agent_observation=observation, execution_feedback=None,
            ))
        self.assertEqual(exchanges[0], exchanges[1])
        payload = json.loads(exchanges[0].user_content)
        self.assertEqual(payload["agent_observation"]["analysis_results"], [result.to_dict()])


if __name__ == "__main__":
    unittest.main()
