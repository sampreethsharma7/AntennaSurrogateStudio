"""Engineering evidence in the existing bounded runner, without live providers."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from studio.antenna_agent import create_default_agent
from studio.antenna_agent_runner import AntennaAgentRunner, semantic_design_hash, _TurnEngineeringReports, _stable_hash
from studio.antenna_builder import BuilderProjectSession, execute_builder_turn, load_builder_session, save_builder_session
from studio.antenna_engineering_checks import run_engineering_checks
from studio.antenna_llm_planner import (
    AgentLoopBudgets, AgentStep, EngineeringDisposition, GeminiSchemaConstrainedPlanner, GroqSchemaConstrainedPlanner,
    OpenRouterNemotronPlanner, SchemaConstrainedLLMPlanner, build_agent_step_exchange,
)
from tests.test_antenna_agent_runner import ScriptedPlanner, step


def array_step():
    return step(("parameter.set", {"key": "array_rows", "value": 2}),
                ("parameter.set", {"key": "array_columns", "value": 3}))


class AgentEngineeringIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.agent = create_default_agent()
        self.baseline = self.agent.create_design("inset_patch")
        self.runner = AntennaAgentRunner(self.agent)

    def array_finish(self, message="Preserved with contact concerns."):
        design = self.agent.update_parameters(self.baseline, {"array_rows": 2, "array_columns": 3}).design
        report = run_engineering_checks(design)
        return AgentStep("finish", message, engineering_disposition=EngineeringDisposition(
            report.semantic_design_hash, tuple(f.observation_id for f in report.findings if f.severity == "warning")))

    def run_turn(self, planner, *, empty=False, budgets=None):
        return self.runner.run(baseline_design=None if empty else self.baseline,
            instruction="Apply the requested change.", project_memory={"requirements": []},
            planner=planner, budgets=budgets)

    def test_initial_report_is_current_info_and_does_not_trigger_behavior(self):
        planner = ScriptedPlanner(step(status="finish"))
        result = self.run_turn(planner)
        report = planner.requests[0]["engineering_report"]
        self.assertEqual(report, run_engineering_checks(self.baseline))
        self.assertTrue(report.findings)
        self.assertEqual({f.severity for f in report.findings}, {"info"})
        self.assertEqual(result.outcome, "finished")
        self.assertFalse(result.has_publishable_change)
        self.assertIsNone(planner.requests[0]["agent_observation"])

    def test_accepted_batch_refreshes_report_and_warning_does_not_block_finish(self):
        planner = ScriptedPlanner(array_step(), self.array_finish())
        with patch("studio.antenna_agent_runner.run_engineering_checks", wraps=run_engineering_checks) as checks:
            result = self.run_turn(planner)
        self.assertEqual(checks.call_count, 2)  # baseline + accepted array; finish reuses
        self.assertEqual(result.outcome, "finished")
        self.assertTrue(result.has_publishable_change)
        report = planner.requests[1]["engineering_report"]
        self.assertEqual(report.semantic_design_hash, semantic_design_hash(result.final_design))
        self.assertEqual(report.working_design_ref.revision, result.final_design.revision)
        self.assertEqual(len([f for f in report.findings if f.category == "conductor_contact"]), 3)
        self.assertEqual(len([f for f in report.findings if f.category == "coincident_port_segments"]), 3)
        self.assertNotIn("engineering_report", planner.requests[1]["agent_observation"].to_dict())

    def test_unchanged_rejected_batch_reuses_report(self):
        planner = ScriptedPlanner(step(("parameter.set", {"key": "missing_parameter", "value": 1})), step(status="finish"))
        with patch("studio.antenna_agent_runner.run_engineering_checks", wraps=run_engineering_checks) as checks:
            result = self.run_turn(planner)
        self.assertEqual(checks.call_count, 1)
        self.assertIs(planner.requests[0]["engineering_report"], planner.requests[1]["engineering_report"])
        self.assertEqual([e.execution_status for e in result.trajectory], ["rejected", "finished"])

    def test_revision_rebind_and_suite_version_cache_invalidation(self):
        cache = _TurnEngineeringReports()
        revised = replace(self.baseline, revision=41)
        with patch("studio.antenna_agent_runner.run_engineering_checks", wraps=run_engineering_checks) as checks:
            initial = cache.current(self.baseline)
            report = cache.current(revised)
            self.assertEqual(checks.call_count, 1)
            self.assertEqual(report.working_design_ref.revision, 41)
            self.assertEqual(report.findings, initial.findings)
            self.assertEqual(report.semantic_design_hash, initial.semantic_design_hash)
            with patch("studio.antenna_agent_runner.SUITE_VERSION", "future"):
                cache.current(revised)
            self.assertEqual(checks.call_count, 2)

    def test_unknown_coverage_is_visible_without_automatic_refusal(self):
        original = run_engineering_checks(self.baseline)
        unknown = replace(original, findings=(), coverage=tuple(replace(c, status="unknown", reason="Unresolved geometry.") for c in original.coverage))
        planner = ScriptedPlanner(step(status="finish"))
        with patch("studio.antenna_agent_runner.run_engineering_checks", return_value=unknown):
            result = self.run_turn(planner)
        self.assertEqual(result.outcome, "finished")
        self.assertEqual({c.status for c in planner.requests[0]["engineering_report"].coverage}, {"unknown"})

    def test_finish_rejects_blocking_but_does_not_police_warning_prose(self):
        original = run_engineering_checks(self.baseline)
        for severity, expected in (("warning", "finished"), ("blocking", "invalid_plan")):
            with self.subTest(severity=severity):
                report = replace(original, findings=(replace(original.findings[0], severity=severity),))
                planner = ScriptedPlanner(AgentStep("finish", "Done.", engineering_disposition=EngineeringDisposition(
                    report.semantic_design_hash, (report.findings[0].observation_id,))))
                with patch("studio.antenna_agent_runner.run_engineering_checks", return_value=report):
                    result = self.run_turn(planner)
                self.assertEqual(result.outcome, expected)

    def test_clarify_and_refuse_discard_warning_candidate_but_keep_audit(self):
        snapshot = self.baseline.to_dict()
        for status in ("clarify", "refuse"):
            with self.subTest(status=status):
                planner = ScriptedPlanner(array_step(), step(status=status))
                result = self.run_turn(planner)
                self.assertEqual(result.outcome, status)
                self.assertIsNone(result.final_design)
                self.assertFalse(result.has_publishable_change)
                self.assertEqual(result.working_design_hash, semantic_design_hash(self.baseline))
                self.assertTrue(any(f.severity == "warning" for f in planner.requests[1]["engineering_report"].findings))
                records = [e.to_dict()["engineering_audit"] for e in result.trajectory]
                self.assertEqual(records[0]["after_report_hash"], records[1]["before_report_hash"])
                self.assertEqual(records[0]["before_report_hash"], records[1]["after_report_hash"])
                self.assertEqual(records[1]["reports"], {})
                self.assertTrue(records[1]["finding_delta"]["removed"])
        self.assertEqual(snapshot, self.baseline.to_dict())

    def test_empty_project_has_no_fabricated_findings_and_refreshes_after_creation(self):
        planner = ScriptedPlanner(step(("recipe.select", {"recipe_id": "inset_patch_v2"})), step(status="finish"))
        with patch("studio.antenna_agent_runner.run_engineering_checks", wraps=run_engineering_checks) as checks:
            result = self.run_turn(planner, empty=True)
        initial = planner.requests[0]["engineering_report"]
        self.assertIsNone(initial.working_design_ref)
        self.assertEqual(initial.semantic_design_hash, semantic_design_hash(None))
        self.assertEqual(initial.findings, ())
        self.assertEqual({c.status for c in initial.coverage}, {"not_applicable"})
        self.assertEqual(checks.call_count, 1)
        self.assertEqual(result.outcome, "finished")

    def test_audit_is_reconstructable_deduplicated_and_hash_verified(self):
        planner = ScriptedPlanner(array_step(), self.array_finish())
        result = self.run_turn(planner)
        store = {}
        for entry in result.trajectory:
            audit = entry.to_dict()["engineering_audit"]
            for key, payload in audit["reports"].items():
                self.assertNotIn(key, store)
                self.assertEqual(_stable_hash(payload), key)
                store[key] = payload
            for side, state in (("before", entry.working_state_before), ("after", entry.working_state_after)):
                report = store[audit[f"{side}_report_hash"]]
                self.assertEqual(report["semantic_design_hash"], state["semantic_hash"])
                self.assertEqual(report["working_design_ref"], {k: state[k] for k in ("design_id", "revision")})
            self.assertTrue(audit["after_geometry_hashes"])
        self.assertEqual(len(store), 2)
        self.assertEqual(result.trajectory[-1].engineering_audit["reports"], {})

    def test_failed_inspection_and_stale_report_are_explicit_failed_coverage(self):
        stale = replace(run_engineering_checks(self.baseline), semantic_design_hash="a" * 64)
        for mock_args in ({"return_value": stale}, {"side_effect": RuntimeError("api_key=do-not-print")}):
            planner = ScriptedPlanner(step(status="finish"))
            with patch("studio.antenna_agent_runner.run_engineering_checks", **mock_args):
                result = self.run_turn(planner)
            report = planner.requests[0]["engineering_report"]
            self.assertEqual(report.semantic_design_hash, semantic_design_hash(self.baseline))
            self.assertEqual(report.findings, ())
            self.assertEqual({c.status for c in report.coverage}, {"failed"})
            self.assertNotIn("do-not-print", json.dumps(result.to_dict()))

    def test_providers_share_report_and_instructions_through_actual_mixin(self):
        report = run_engineering_checks(self.baseline)
        args = dict(instruction="Inspect this design.", current_design=self.baseline.to_dict(),
            capability_manifest=self.agent.capability_manifest(self.baseline), remaining_budgets=AgentLoopBudgets(),
            project_memory={"requirements": []}, engineering_report=report)
        expected = build_agent_step_exchange(**args)
        for cls in (SchemaConstrainedLLMPlanner, GeminiSchemaConstrainedPlanner, GroqSchemaConstrainedPlanner, OpenRouterNemotronPlanner):
            with self.subTest(provider=cls.__name__):
                planner = cls.__new__(cls)
                captured = []
                def plan(**kwargs):
                    captured.append(planner._exchange(**kwargs, execution_feedback=None))
                    return step(status="finish")
                with patch.object(planner, "plan", side_effect=plan):
                    planner.plan_agent_step(**args)
                self.assertEqual(captured, [expected])
                self.assertIsNone(planner._agent_loop_engineering_report)
                self.assertIsNone(planner._agent_loop_remaining_budgets)
        self.assertIn("not predetermined repair instructions", expected.system_instruction)
        self.assertIn("warning indicates", expected.system_instruction)
        self.assertIn("full-wave validity", expected.system_instruction)

    def test_provider_failure_also_retains_current_report(self):
        planner = ScriptedPlanner(RuntimeError("Provider unavailable."))
        result = self.run_turn(planner)
        self.assertEqual(result.outcome, "provider_error")
        self.assertEqual(len(result.trajectory[0].engineering_audit["reports"]), 1)

    def test_budgets_cycles_and_no_progress_still_apply(self):
        planner = ScriptedPlanner(array_step(), self.array_finish())
        result = self.run_turn(planner, budgets=AgentLoopBudgets(1, 3, 2, 24))
        self.assertEqual(result.outcome, "agent_limit")
        self.assertIsNone(result.final_design)
        self.assertTrue(result.trajectory[0].engineering_audit["finding_delta"]["added"])
        planner = ScriptedPlanner(step(("parameter.set", {"key": "patch_width_mm", "value": self.baseline.value("patch_width_mm")})), step(status="finish"))
        result = self.run_turn(planner)
        self.assertEqual(result.trajectory[0].execution_status, "no_progress")
        self.assertEqual(result.trajectory[0].engineering_audit["finding_delta"], {"added": [], "removed": [], "changed": []})

    def test_publication_once_and_abandoned_reports_not_in_project_memory(self):
        root = Path(__file__).resolve().parents[1] / ".test_runs"
        root.mkdir(exist_ok=True)
        for status in ("finish", "clarify", "refuse"):
            with self.subTest(status=status), tempfile.TemporaryDirectory(dir=root) as directory:
                session = BuilderProjectSession(design=self.baseline)
                save_builder_session(directory, session)
                previous = load_builder_session(directory)
                terminal_step = self.array_finish() if status == "finish" else step(status=status, message="Please review the requested candidate.")
                planner = ScriptedPlanner(array_step(), terminal_step)
                import studio.antenna_builder as builder
                with patch.object(builder, "_publish_builder_outcome", wraps=builder._publish_builder_outcome) as publish:
                    result = execute_builder_turn(directory, session, "Create an array.", planner=planner)
                    self.assertEqual(result.terminal_result.outcome, "finished" if status == "finish" else status)
                    self.assertEqual(publish.call_count, 1)
                restored = load_builder_session(directory)
                if status != "finish":
                    self.assertEqual(restored.design.to_dict(), previous.design.to_dict())
                    self.assertEqual(restored.memory.canonical_ref, previous.memory.canonical_ref)
                    self.assertEqual(restored.memory.important_changes, previous.memory.important_changes)
                else:
                    self.assertEqual(restored.design.revision, self.baseline.revision + 1)
                    self.assertEqual((restored.design.array.rows, restored.design.array.columns), (2, 3))
                memory_text = json.dumps(restored.memory.to_dict())
                for marker in ("engineering_report", "evaluated_geometry_hash", "retained_constituent_xy_overlap_area_mm2"):
                    self.assertNotIn(marker, memory_text)
                records = [json.loads(line) for line in (Path(directory) / "design/planner_ab.jsonl").read_text().splitlines()]
                self.assertEqual(len(records), 1)
                self.assertEqual(len(records[0]["trajectory"][0]["engineering_audit"]["reports"]), 2)


if __name__ == "__main__":
    unittest.main()
