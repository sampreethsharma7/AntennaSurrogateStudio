import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from studio.antenna_agent import create_default_agent
from studio.antenna_agent_runner import AntennaAgentRunner, semantic_design_hash
from studio.antenna_engineering_checks import run_engineering_checks
from studio.antenna_llm_planner import (
    AgentLoopBudgets, AgentStep, AgentStepObservation, DeferredEngineeringFinding,
    EngineeringDisposition, GeminiSchemaConstrainedPlanner, OpenRouterNemotronPlanner,
    build_agent_step_exchange, parse_agent_step, parse_llm_tool_plan,
    plan_json_schema, validate_engineering_disposition,
)
from studio.antenna_tools import CapabilityError
from tests.test_antenna_agent_runner import ScriptedPlanner, step


class EngineeringDispositionTests(unittest.TestCase):
    def setUp(self):
        self.agent = create_default_agent()
        self.design = self.agent.create_design("inset_patch")
        self.clean = run_engineering_checks(self.design)
        self.warning = replace(self.clean.findings[0], severity="warning")
        self.report = replace(self.clean, findings=(self.warning, *self.clean.findings[1:]))

    def acknowledged(self, report=None):
        report = report or self.report
        return EngineeringDisposition(report.semantic_design_hash,
            tuple(f.observation_id for f in report.findings if f.severity == "warning"))

    def validate(self, disposition, report=None):
        return validate_engineering_disposition(disposition, report or self.report)

    def test_empty_disposition_and_omission_valid_without_warnings(self):
        for disposition in (None, EngineeringDisposition()):
            self.assertEqual(self.validate(disposition, self.clean)["status"], "valid")

    def test_one_warning_acknowledged_and_info_not_required(self):
        result = self.validate(self.acknowledged())
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["missing_warning_observation_ids"], [])
        self.assertGreater(len(self.report.findings), 1)

    def test_deferred_with_reason_allows_finish(self):
        disposition = EngineeringDisposition(self.report.semantic_design_hash, deferred=(
            DeferredEngineeringFinding(self.warning.observation_id, "User accepts retaining this geometry."),))
        self.assertEqual(self.validate(disposition)["status"], "valid")
        self.assertEqual(EngineeringDisposition.from_dict(disposition.to_dict()), disposition)

    def test_omission_and_unknown_ids_are_rejected(self):
        result = self.validate(None)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["missing_warning_observation_ids"], [self.warning.observation_id])
        disposition = replace(self.acknowledged(), acknowledged_observation_ids=(self.warning.observation_id, "invented-id"))
        result = self.validate(disposition)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["unknown_observation_ids"], ["invented-id"])

    def test_stale_removed_id_and_old_hash_with_stable_ids_rejected(self):
        old = self.acknowledged()
        removed = replace(self.report, findings=())
        self.assertEqual(self.validate(old, removed)["unknown_observation_ids"], [self.warning.observation_id])
        # Observation IDs deliberately survive parameter changes. Hash binding
        # prevents reusing an old acknowledgement of changed measurements.
        current = replace(self.report, semantic_design_hash="b" * 64)
        feedback = self.validate(old, current)
        self.assertTrue(feedback["stale_design_hash"])
        self.assertEqual(feedback["status"], "rejected")
        self.assertEqual(feedback["unknown_observation_ids"], [])

    def test_duplicates_in_each_list_and_across_lists_rejected(self):
        identity = self.warning.observation_id
        deferred = DeferredEngineeringFinding(identity, "Preserved by request.")
        cases = (
            EngineeringDisposition(self.report.semantic_design_hash, (identity, identity)),
            EngineeringDisposition(self.report.semantic_design_hash, deferred=(deferred, deferred)),
            EngineeringDisposition(self.report.semantic_design_hash, (identity,), (deferred,)),
        )
        for disposition in cases:
            with self.subTest(disposition=disposition):
                result = self.validate(disposition)
                self.assertEqual(result["status"], "rejected")
                self.assertEqual(result["duplicate_observation_ids"], [identity])

    def test_reason_is_required_short_and_not_whitespace(self):
        for reason in (None, "", "  ", "x" * 401):
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                DeferredEngineeringFinding(self.warning.observation_id, reason)

    def test_strict_parser_accepts_finish_disposition_only(self):
        finish = AgentStep("finish", "Retained with concerns.", engineering_disposition=self.acknowledged())
        parsed = parse_agent_step(finish.to_dict(), callable_tool_names=())
        self.assertEqual(parsed, finish)
        for status in ("clarify", "refuse", "execute"):
            payload = finish.to_dict()
            payload["status"] = status
            with self.assertRaises(CapabilityError):
                parse_agent_step(payload, callable_tool_names=())
        payload = finish.to_dict()
        payload["engineering_disposition"]["invented"] = 1
        with self.assertRaises(CapabilityError):
            parse_agent_step(payload, callable_tool_names=())

    def test_rejected_finish_corrects_with_normal_decision_budget_no_tools(self):
        planner = ScriptedPlanner(step(status="finish"),
            AgentStep("finish", "Done.", engineering_disposition=self.acknowledged()))
        with patch("studio.antenna_agent_runner.run_engineering_checks", return_value=self.report) as checks:
            result = AntennaAgentRunner(self.agent).run(baseline_design=self.design, instruction="Preserve geometry.",
                project_memory=None, planner=planner)
        self.assertEqual(result.outcome, "finished")
        self.assertEqual([e.execution_status for e in result.trajectory], ["finish_rejected", "finished"])
        self.assertEqual(checks.call_count, 1)
        feedback = planner.requests[1]["agent_observation"]
        self.assertEqual(feedback.failure_category, "engineering_disposition")
        self.assertEqual(feedback.validation_result["missing_warning_observation_ids"], [self.warning.observation_id])
        self.assertEqual(feedback.planned_calls, ())
        self.assertEqual(AgentStepObservation.from_dict(feedback.to_dict()), feedback)
        self.assertEqual(feedback.remaining_budgets, AgentLoopBudgets(4, 3, 2, 24))
        self.assertEqual(planner.requests[0]["current_design"], planner.requests[1]["current_design"])
        self.assertIs(planner.requests[0]["engineering_report"], planner.requests[1]["engineering_report"])
        self.assertFalse(result.has_publishable_change)
        self.assertEqual(result.aggregate_calls, ())

    def test_repeated_incomplete_finish_exhausts_budget_and_preserves_baseline(self):
        planner = ScriptedPlanner(*[step(status="finish") for _ in range(3)])
        with patch("studio.antenna_agent_runner.run_engineering_checks", return_value=self.report):
            result = AntennaAgentRunner(self.agent).run(baseline_design=self.design, instruction="Preserve geometry.",
                project_memory=None, planner=planner, budgets=AgentLoopBudgets(3, 3, 2, 24))
        self.assertEqual(result.outcome, "agent_limit")
        self.assertEqual(len(planner.requests), 3)
        self.assertTrue(all(e.execution_status == "finish_rejected" for e in result.trajectory))
        self.assertIsNone(result.final_design)
        self.assertEqual(result.working_design_hash, semantic_design_hash(self.design))

    def test_warning_disposition_publishes_accepted_change_without_message_policing(self):
        change = step(("parameter.set", {"key": "array_rows", "value": 2}),
                      ("parameter.set", {"key": "array_columns", "value": 3}))
        candidate = self.agent.update_parameters(self.design, {"array_rows": 2, "array_columns": 3}).design
        report = run_engineering_checks(candidate)
        planner = ScriptedPlanner(change, step(status="finish"),
            AgentStep("finish", "Done.", engineering_disposition=self.acknowledged(report)))
        result = AntennaAgentRunner(self.agent).run(baseline_design=self.design, instruction="Make an array.",
            project_memory=None, planner=planner)
        self.assertEqual(result.outcome, "finished")
        self.assertTrue(result.has_publishable_change)
        self.assertEqual((result.final_design.array.rows, result.final_design.array.columns), (2, 3))
        self.assertEqual([e.execution_status for e in result.trajectory], ["accepted", "finish_rejected", "finished"])
        self.assertEqual(result.final_step.engineering_disposition, self.acknowledged(report))

    def test_provider_neutral_schema_and_instructions_are_identical(self):
        args = dict(instruction="Preserve the requested geometry.", current_design=self.design.to_dict(),
            capability_manifest=self.agent.capability_manifest(self.design), project_memory=None,
            agent_observation=None, remaining_budgets=AgentLoopBudgets(), engineering_report=self.report)
        expected = build_agent_step_exchange(**args)
        for cls in (GeminiSchemaConstrainedPlanner, OpenRouterNemotronPlanner):
            planner = cls.__new__(cls)
            captured = []
            def plan(**kwargs):
                captured.append(planner._exchange(**kwargs, execution_feedback=None))
                return AgentStep("finish", "Done.", engineering_disposition=self.acknowledged())
            with patch.object(planner, "plan", side_effect=plan):
                planner.plan_agent_step(**args)
            self.assertEqual(captured, [expected])
        self.assertIn("engineering_disposition", expected.schema["properties"])
        self.assertIn("exactly once", expected.system_instruction)

    def test_legacy_toolplan_and_one_shot_schema_unchanged(self):
        manifest = self.agent.capability_manifest(self.design)
        schema = plan_json_schema(tuple(manifest["callable_tools"]))
        self.assertNotIn("engineering_disposition", schema["properties"])
        payload = {"schema_version": 1, "status": "execute", "message": "Set width.",
            "calls": [{"name": "parameter.set", "arguments": {"key": "patch_width_mm", "value": 40}}]}
        plan = parse_llm_tool_plan(payload, callable_tool_names=("parameter.set",))
        result = self.agent.execute_llm_plan(self.design, plan)
        self.assertEqual(result.design.value("patch_width_mm"), 40)
        with self.assertRaises(CapabilityError):
            parse_llm_tool_plan({**payload, "engineering_disposition": self.acknowledged().to_dict()},
                               callable_tool_names=("parameter.set",))

    def test_measurement_labels_distinguish_whole_and_constituent_without_changing_values(self):
        array = self.agent.update_parameters(self.design, {"array_rows": 2, "array_columns": 3}).design
        report = run_engineering_checks(array)
        contacts = [f for f in report.findings if f.category == "conductor_contact"]
        self.assertEqual(len(contacts), 3)
        for f in contacts:
            self.assertAlmostEqual(f.measured_values["resolved_conductor_xy_overlap_area_mm2"].value, 109.5406487)
            self.assertAlmostEqual(f.measured_values["retained_constituent_xy_overlap_area_mm2"].value, 61.6893235)
            self.assertIn("complete final resolved conductors", f.message)
            self.assertIn("not the complete resolved-conductor overlap", f.message)
            self.assertEqual(f.check_version, "2")


if __name__ == "__main__":
    unittest.main()
