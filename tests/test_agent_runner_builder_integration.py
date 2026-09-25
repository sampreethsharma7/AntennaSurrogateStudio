import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from studio.antenna_agent import create_default_agent
from studio.antenna_builder import (
    BuilderProjectSession,
    ProjectMemoryReducer,
    apply_text_instruction,
    execute_builder_turn,
    load_builder_session,
    save_builder_session,
)
from studio.antenna_llm_planner import AgentLoopBudgets, AgentStep, LLMToolPlan, PlannedToolCall
from studio.antenna_tools import CapabilityError


def step(*calls, status="execute", message="Continue the requested design."):
    return AgentStep(
        status,
        message,
        tuple(PlannedToolCall(name, arguments) for name, arguments in calls),
    )


def initial_patch_step():
    return step(
        ("recipe.select", {"recipe_id": "inset_patch_v2"}),
        ("parameter.set", {"key": "frequency_ghz", "value": 2.45}),
        ("parameter.set", {"key": "material", "value": "FR4"}),
    )


def center_slot_step():
    return step(
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
    )


class ScriptedAgentPlanner:
    backend_id = "scripted_agent"
    model = "test-model"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.last_run_metadata = {}

    def plan_agent_step(self, **request):
        self.requests.append(request)
        self.last_run_metadata = {
            "backend": self.backend_id,
            "model": self.model,
            "iteration": len(self.requests),
        }
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class AgentRunnerBuilderIntegrationTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        self.project_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def audit_records(self):
        path = self.project_path / "design" / "planner_ab.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_execute_then_finish_publishes_once_with_two_conversation_records(self):
        session = BuilderProjectSession(design=create_default_agent().create_design("inset_patch"))
        baseline_revision = session.design.revision
        planner = ScriptedAgentPlanner(
            step(("parameter.set", {"key": "patch_width_mm", "value": 40.0})),
            step(status="finish", message="Patch width updated."),
        )

        result = execute_builder_turn(
            self.project_path, session, "Set PatchW to 40 mm.", planner=planner, turn_id="turn-1"
        )

        self.assertEqual(result.session.design.revision, baseline_revision + 1)
        self.assertEqual(result.session.design.value("patch_width_mm"), 40.0)
        self.assertEqual(len(result.session.conversation), 2)
        self.assertEqual([item["role"] for item in result.session.conversation], ["user", "builder"])
        self.assertEqual(len(self.audit_records()), 1)
        self.assertEqual(len(self.audit_records()[0]["trajectory"]), 2)

    def test_two_execute_batches_from_empty_project_publish_revision_zero_once(self):
        session = BuilderProjectSession()
        planner = ScriptedAgentPlanner(
            initial_patch_step(),
            center_slot_step(),
            step(status="finish", message="Patch and center slot are complete."),
        )
        original_reduce = ProjectMemoryReducer.reduce
        reduce_calls = []

        def recording_reduce(reducer, *args, **kwargs):
            reduce_calls.append((args, kwargs))
            return original_reduce(reducer, *args, **kwargs)

        with patch.object(ProjectMemoryReducer, "reduce", recording_reduce):
            result = execute_builder_turn(
                self.project_path,
                session,
                "Create an inset-fed rectangular patch antenna at 2.45 GHz on FR4 and add a circular slot of radius 3 mm at the patch center.",
                planner=planner,
                turn_id="turn-empty",
            )

        self.assertEqual(len(reduce_calls), 1)
        self.assertEqual(result.session.design.revision, 0)
        self.assertEqual(result.session.design.value("slot_radius_mm"), 3.0)
        self.assertEqual(len(result.session.design.composed_operations), 1)
        self.assertEqual(result.session.memory.canonical_ref.revision, 0)
        self.assertEqual(len(result.session.memory.important_changes), 1)
        self.assertEqual(len(result.session.conversation), 2)
        audit = self.audit_records()[0]
        self.assertEqual(audit["turn_id"], "turn-empty")
        self.assertEqual(audit["terminal_status"], "finished")
        self.assertEqual(len(audit["trajectory"]), 3)
        executed_names = [item["name"] for item in audit["aggregate_executed_tools"]]
        self.assertIn("recipe.select", [call.name for call in result.terminal_result.aggregate_calls])
        self.assertIn("boolean.subtract", [call.name for call in result.terminal_result.aggregate_calls])
        self.assertGreater(len(executed_names), 6)
        self.assertFalse(audit["intermediate_changes_discarded"])

        restored = load_builder_session(self.project_path)
        self.assertEqual(restored.design, result.session.design)
        self.assertEqual(restored.conversation, result.session.conversation)
        self.assertEqual(restored.memory, result.session.memory)

    def test_existing_design_two_batches_increment_revision_exactly_once(self):
        baseline = create_default_agent().create_design("inset_patch")
        session = BuilderProjectSession(design=baseline)
        result = execute_builder_turn(
            self.project_path,
            session,
            "Change width and length.",
            planner=ScriptedAgentPlanner(
                step(("parameter.set", {"key": "patch_width_mm", "value": 40.0})),
                step(("parameter.set", {"key": "patch_length_mm", "value": 31.0})),
                step(status="finish", message="Both dimensions are updated."),
            ),
            turn_id="turn-existing",
        )
        self.assertEqual(result.session.design.revision, baseline.revision + 1)
        self.assertEqual(result.session.design.value("patch_width_mm"), 40.0)
        self.assertEqual(result.session.design.value("patch_length_mm"), 31.0)

    def test_finish_without_change_preserves_revision_and_empty_design(self):
        baseline = create_default_agent().create_design("inset_patch")
        session = BuilderProjectSession(design=baseline)
        result = execute_builder_turn(
            self.project_path,
            session,
            "What can you build?",
            planner=ScriptedAgentPlanner(step(status="finish", message="Three antenna families are available.")),
            turn_id="turn-info",
        )
        self.assertIs(result.session.design, baseline)
        self.assertEqual(result.session.design.revision, baseline.revision)
        self.assertEqual(result.session.conversation[-1]["outcome"], "completed")

        empty_path = self.project_path / "empty"
        empty_result = execute_builder_turn(
            empty_path,
            BuilderProjectSession(),
            "What can you build?",
            planner=ScriptedAgentPlanner(step(status="finish", message="Three antenna families are available.")),
            turn_id="turn-empty-info",
        )
        self.assertIsNone(empty_result.session.design)
        self.assertFalse((empty_path / "design" / "antenna_state.json").exists())

    def _assert_rollback(self, planner, expected_exception=None, *, expected_outcome=None, budgets=None):
        baseline = create_default_agent().create_design("inset_patch")
        session = BuilderProjectSession(design=baseline)
        save_builder_session(self.project_path, session)
        state_path = self.project_path / "design" / "antenna_state.json"
        before_bytes = state_path.read_bytes()
        if expected_exception is None:
            result = execute_builder_turn(
                self.project_path,
                session,
                "Attempt a multi-step edit.",
                planner=planner,
                budgets=budgets,
                turn_id="turn-rollback",
            )
            self.assertEqual(result.terminal_result.outcome, expected_outcome)
        else:
            with self.assertRaises(expected_exception):
                execute_builder_turn(
                    self.project_path,
                    session,
                    "Attempt a multi-step edit.",
                    planner=planner,
                    budgets=budgets,
                    turn_id="turn-rollback",
                )
        self.assertEqual(session.design, baseline)
        self.assertEqual(state_path.read_bytes(), before_bytes)
        self.assertEqual(session.memory.important_changes, ())
        self.assertEqual(len(session.conversation), 2)
        audit = self.audit_records()[0]
        self.assertTrue(audit["intermediate_changes_discarded"])
        self.assertEqual(audit["published_design_ref"]["revision"], baseline.revision)

    def test_intermediate_success_then_clarify_rolls_back(self):
        self._assert_rollback(
            ScriptedAgentPlanner(
                step(("parameter.set", {"key": "patch_width_mm", "value": 40.0})),
                step(status="clarify", message="Which excitation should I use?"),
            ),
            expected_outcome="clarify",
        )
        self.assertEqual(self.audit_records()[0]["terminal_status"], "clarify")

    def test_intermediate_success_then_refuse_rolls_back(self):
        self._assert_rollback(
            ScriptedAgentPlanner(
                step(("parameter.set", {"key": "patch_width_mm", "value": 40.0})),
                step(status="refuse", message="That capability is unavailable."),
            ),
            expected_outcome="refuse",
        )
        self.assertEqual(self.audit_records()[0]["terminal_status"], "refuse")

    def test_intermediate_success_then_provider_error_rolls_back_without_limitation(self):
        self._assert_rollback(
            ScriptedAgentPlanner(
                step(("parameter.set", {"key": "patch_width_mm", "value": 40.0})),
                RuntimeError("provider unavailable api_key=never-store-this"),
            ),
            CapabilityError,
        )
        self.assertEqual(self.audit_records()[0]["terminal_status"], "provider_error")
        self.assertEqual(load_builder_session(self.project_path).memory.limitations, ())
        self.assertNotIn("never-store-this", json.dumps(self.audit_records()))

    def test_intermediate_success_then_agent_limit_rolls_back(self):
        self._assert_rollback(
            ScriptedAgentPlanner(
                step(("parameter.set", {"key": "patch_width_mm", "value": 40.0})),
                step(("parameter.set", {"key": "patch_length_mm", "value": 31.0})),
            ),
            CapabilityError,
            budgets=AgentLoopBudgets(5, 1, 2, 24),
        )
        self.assertEqual(self.audit_records()[0]["terminal_status"], "agent_limit")

    def test_cycle_and_duplicate_failures_preserve_baseline(self):
        baseline = create_default_agent().create_design("inset_patch")
        original_width = baseline.value("patch_width_mm")
        for name, planner, status in (
            (
                "cycle",
                ScriptedAgentPlanner(
                    step(("parameter.set", {"key": "patch_width_mm", "value": original_width + 1})),
                    step(("parameter.set", {"key": "patch_width_mm", "value": original_width})),
                ),
                "cycle_detected",
            ),
            (
                "duplicate",
                ScriptedAgentPlanner(
                    step(("parameter.set", {"key": "missing", "value": 1})),
                    step(("parameter.set", {"key": "missing", "value": 1})),
                ),
                "duplicate_rejection",
            ),
        ):
            with self.subTest(name=name):
                path = self.project_path / name
                session = BuilderProjectSession(design=baseline)
                with self.assertRaises(CapabilityError):
                    execute_builder_turn(path, session, "Try edit.", planner=planner, turn_id=f"turn-{name}")
                self.assertEqual(session.design, baseline)
                audit = json.loads((path / "design" / "planner_ab.jsonl").read_text().splitlines()[0])
                self.assertEqual(audit["terminal_status"], status)

    def test_legacy_apply_text_instruction_remains_one_shot(self):
        baseline = create_default_agent().create_design("inset_patch")

        class LegacyPlanner:
            def plan(self, **_request):
                return LLMToolPlan(
                    "execute", "Set width.",
                    (PlannedToolCall("parameter.set", {"key": "patch_width_mm", "value": 40.0}),),
                )

        result = apply_text_instruction(baseline, "Set width.", planner=LegacyPlanner())
        self.assertEqual(result.state.revision, baseline.revision + 1)
        self.assertEqual(result.state.value("patch_width_mm"), 40.0)


if __name__ == "__main__":
    unittest.main()
