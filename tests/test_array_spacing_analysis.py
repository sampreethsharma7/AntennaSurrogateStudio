import json
import math
import unittest
from dataclasses import replace

from studio.antenna_agent import AgentInstructionError, create_default_agent
from studio.antenna_agent_runner import AntennaAgentRunner, semantic_design_hash
from studio.antenna_analysis import EngineeringAnalysisResult, SPEED_OF_LIGHT_MM_GHZ
from studio.antenna_llm_planner import (
    AgentLoopBudgets, AgentStep, AgentStepObservation, GeminiSchemaConstrainedPlanner,
    OpenRouterNemotronPlanner, PlannedToolCall,
)


def step(*calls, status="execute", message="test step"):
    return AgentStep(status, message, tuple(PlannedToolCall(name, arguments) for name, arguments in calls))


class ScriptedPlanner:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.last_run_metadata = {}

    def plan_agent_step(self, **request):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ArraySpacingAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.agent = create_default_agent()
        self.single = self.agent.create_design("inset_patch")

    def array(self, rows, columns, spacing_lambda=0.55):
        return self.agent.update_parameters(self.single, {
            "array_rows": rows,
            "array_columns": columns,
            "element_spacing_lambda": spacing_lambda,
        }).design

    def analyze(self, design, **arguments):
        call = PlannedToolCall("engineering.array_spacing", arguments)
        return self.agent.execute_analysis_batch(design, (call,))[0]

    @staticmethod
    def values(result):
        return {key: item.value for key, item in result.measurements.items()}

    def test_single_element_is_not_applicable(self):
        result = self.analyze(self.single)
        values = self.values(result)
        self.assertEqual(result.status, "not_applicable")
        self.assertEqual((values["rows"], values["columns"]), (1, 1))
        self.assertNotIn("row_spacing", values)
        self.assertNotIn("column_spacing", values)

    def test_linear_column_array_has_only_column_axis(self):
        design = self.array(1, 4)
        result = self.analyze(design)
        values = self.values(result)
        self.assertEqual(result.status, "completed")
        self.assertNotIn("row_spacing", values)
        self.assertAlmostEqual(values["column_spacing"], design.array.spacing_mm)
        self.assertAlmostEqual(values["column_spacing_lambda"], 0.55, places=12)
        self.assertIn("row axis is not applicable", result.applicability)

    def test_linear_row_array_has_only_row_axis(self):
        result = self.analyze(self.array(3, 1, 0.7))
        values = self.values(result)
        self.assertIn("row_spacing", values)
        self.assertAlmostEqual(values["row_spacing_lambda"], 0.7, places=12)
        self.assertNotIn("column_spacing", values)
        self.assertIn("column axis is not applicable", result.applicability)

    def test_rectangular_array_reports_both_axes_and_free_space_wavelength(self):
        design = self.array(2, 3, 0.6)
        result = self.analyze(design)
        values = self.values(result)
        expected_lambda = SPEED_OF_LIGHT_MM_GHZ / design.frequency_ghz
        self.assertAlmostEqual(values["lambda0"], expected_lambda, places=12)
        self.assertAlmostEqual(values["row_spacing"], design.array.spacing_mm, places=12)
        self.assertAlmostEqual(values["column_spacing"], design.array.spacing_mm, places=12)
        self.assertAlmostEqual(values["row_spacing_lambda"], 0.6, places=12)
        self.assertAlmostEqual(values["column_spacing_lambda"], 0.6, places=12)
        self.assertEqual(result.measurements["frequency"].unit, "GHz")
        self.assertEqual(result.measurements["lambda0"].unit, "mm")

    def test_frequency_change_updates_wavelength_and_ratio_for_fixed_physical_spacing(self):
        design = self.array(2, 3, 0.6)
        before = self.analyze(design)
        parameters = tuple(
            replace(item, value=item.value * 2) if item.key == "frequency_ghz" else item
            for item in design.parameters
        )
        changed = replace(design, revision=design.revision + 1, parameters=parameters)
        after = self.analyze(changed)
        self.assertAlmostEqual(self.values(after)["lambda0"], self.values(before)["lambda0"] / 2, places=12)
        self.assertAlmostEqual(
            self.values(after)["row_spacing_lambda"], self.values(before)["row_spacing_lambda"] * 2,
            places=12,
        )

    def test_spacing_change_updates_normalized_spacing(self):
        design = self.array(2, 3, 0.6)
        changed = replace(design, array=replace(design.array, spacing_mm=design.array.spacing_mm * 1.5))
        self.assertAlmostEqual(self.values(self.analyze(changed))["row_spacing_lambda"], 0.9, places=12)

    def test_half_lambda_without_scan_makes_no_grating_lobe_claim(self):
        result = self.analyze(self.array(1, 4, 0.5))
        self.assertAlmostEqual(self.values(result)["column_spacing_lambda"], 0.5, places=12)
        self.assertNotIn("visible_nonzero_order_count", " ".join(result.measurements))
        self.assertIn("No grating-lobe conclusion", " ".join(result.limitations))

    def test_above_half_lambda_without_scan_still_makes_no_claim(self):
        result = self.analyze(self.array(1, 4, 0.8))
        self.assertAlmostEqual(self.values(result)["column_spacing_lambda"], 0.8, places=12)
        self.assertIn("not classified", result.message)
        self.assertNotIn("has grating", result.message.casefold())

    def test_scan_angle_with_no_visible_nonzero_order(self):
        result = self.analyze(self.array(1, 4, 0.5), scan_angle_deg=45, principal_axis="column")
        values = self.values(result)
        self.assertEqual(values["column_visible_nonzero_order_count"], 0)
        self.assertIn("no visible nonzero integer order", result.message)

    def test_scan_angle_with_visible_nonzero_order_and_angle(self):
        result = self.analyze(self.array(1, 4, 0.8), scan_angle_deg=45, principal_axis="column")
        values = self.values(result)
        expected = math.degrees(math.asin(math.sin(math.radians(45)) - 1 / 0.8))
        self.assertEqual(values["column_visible_nonzero_order_count"], 1)
        self.assertAlmostEqual(values["column_visible_order_m_neg_1_angle"], expected, places=12)
        self.assertIn("m=-1", result.message)
        self.assertTrue(any("mutual coupling" in item for item in result.limitations))
        self.assertTrue(any("full-wave" in item for item in result.limitations))

    def test_unresolved_frequency_and_spacing_return_unknown(self):
        array = self.array(2, 3)
        no_frequency = replace(array, parameters=tuple(
            item for item in array.parameters if item.key != "frequency_ghz"
        ))
        self.assertEqual(self.analyze(no_frequency).status, "unknown")
        no_spacing = replace(array, array=replace(array.array, spacing_mm=0.0))
        self.assertEqual(self.analyze(no_spacing).status, "unknown")

    def test_inactive_requested_scan_axis_is_not_applicable(self):
        result = self.analyze(self.array(1, 4), scan_angle_deg=20, principal_axis="row")
        self.assertEqual(result.status, "not_applicable")
        self.assertIn("inactive", result.applicability)

    def test_scan_range_is_not_exposed_or_silently_approximated(self):
        manifest = self.agent.capability_manifest(self.array(1, 4))
        tool = next(item for item in manifest["callable_tools"]
                    if item["name"] == "engineering.array_spacing")
        self.assertNotIn("scan_range_deg", tool["arguments"]["properties"])
        with self.assertRaisesRegex(AgentInstructionError, "schema validation"):
            self.analyze(self.array(1, 4), scan_range_deg=45)

    def test_analysis_does_not_mutate_design_or_revision_and_round_trips(self):
        design = self.array(2, 3)
        snapshot = design.to_dict()
        result = self.analyze(design, scan_angle_deg=20)
        self.assertEqual(design.to_dict(), snapshot)
        self.assertEqual(result.working_design_ref.revision, design.revision)
        self.assertEqual(result.semantic_design_hash, semantic_design_hash(design))
        self.assertEqual(EngineeringAnalysisResult.from_dict(result.to_dict()), result)

    def test_cache_reuse_and_semantic_design_change_invalidation(self):
        design = self.array(1, 4)
        analysis_call = ("engineering.array_spacing", {})
        planner = ScriptedPlanner(
            step(analysis_call), step(analysis_call),
            step(("parameter.set", {"key": "frequency_ghz", "value": 3.0})),
            step(analysis_call), step(status="clarify", message="Inspection complete."),
        )
        result = AntennaAgentRunner(self.agent).run(
            baseline_design=design, instruction="Inspect array spacing.",
            project_memory={"schema_version": 1}, planner=planner,
        )
        self.assertEqual(result.trajectory[0].analysis_cache_hits, 0)
        self.assertEqual(result.trajectory[1].analysis_cache_hits, 1)
        self.assertEqual(result.trajectory[3].analysis_cache_hits, 0)
        self.assertNotEqual(
            result.trajectory[1].analysis_results[0].semantic_design_hash,
            result.trajectory[3].analysis_results[0].semantic_design_hash,
        )
        self.assertIsNone(result.final_design)

    def test_equivalent_integer_and_float_scan_arguments_share_identity_and_cache(self):
        design = self.array(1, 4, 0.6)
        integer = self.analyze(design, scan_angle_deg=45, principal_axis="column")
        floating = self.analyze(design, scan_angle_deg=45.0, principal_axis="column")
        self.assertEqual(integer.analysis_id, floating.analysis_id)
        planner = ScriptedPlanner(
            step(("engineering.array_spacing", {"scan_angle_deg": 45, "principal_axis": "column"})),
            step(("engineering.array_spacing", {"scan_angle_deg": 45.0, "principal_axis": "column"})),
            step(status="clarify", message="Done."),
        )
        result = AntennaAgentRunner(self.agent).run(
            baseline_design=design, instruction="Inspect scan spacing.",
            project_memory={"schema_version": 1}, planner=planner,
        )
        self.assertEqual(result.trajectory[1].analysis_cache_hits, 1)

    def test_gemini_and_nemotron_receive_identical_schema_and_result(self):
        design = self.array(2, 3, 0.6)
        result = self.analyze(design)
        call = PlannedToolCall("engineering.array_spacing", {})
        observation = AgentStepObservation(
            "accepted_analysis", (call,), {"design_id": design.design_id, "revision": design.revision},
            semantic_design_hash(design), AgentLoopBudgets(4, 3, 2, 23, 2),
            executed_calls=(call,), validation_result={"status": "read_only", "design_unchanged": True},
            analysis_results=(result,),
        )
        exchanges = []
        for cls in (GeminiSchemaConstrainedPlanner, OpenRouterNemotronPlanner):
            planner = cls.__new__(cls)
            planner._agent_loop_remaining_budgets = observation.remaining_budgets
            planner._agent_loop_engineering_report = None
            exchanges.append(planner._exchange(
                instruction="Check the electrical spacing.", current_design=design.to_dict(),
                capability_manifest=self.agent.capability_manifest(design), project_memory=None,
                agent_observation=observation, execution_feedback=None,
            ))
        self.assertEqual(exchanges[0], exchanges[1])
        payload = json.loads(exchanges[0].user_content)
        tool = next(item for item in payload["runtime_capabilities"]["callable_tools"]
                    if item["name"] == "engineering.array_spacing")
        self.assertEqual(tool["effect"], "analysis")
        self.assertEqual(payload["agent_observation"]["analysis_results"], [result.to_dict()])


if __name__ == "__main__":
    unittest.main()
