import json
import unittest

from studio.antenna_agent import create_default_agent
from studio.antenna_agent_runner import AntennaAgentRunner, semantic_design_hash
from studio.antenna_llm_planner import (
    AgentLoopBudgets,
    AgentStep,
    LLMToolPlan,
    PlannedToolCall,
)


def step(*calls, status="execute", message="test step"):
    return AgentStep(
        status=status,
        message=message,
        calls=tuple(PlannedToolCall(name, arguments) for name, arguments in calls),
    )


class ScriptedPlanner:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.last_run_metadata = {}

    def plan_agent_step(self, **request):
        self.requests.append(request)
        self.last_run_metadata = {
            "backend": "scripted",
            "iteration": len(self.requests),
            "authorization": "Bearer test-secret",
        }
        if not self.responses:
            raise RuntimeError("The scripted planner has no response.")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


_DEFAULT_BASELINE = object()


class AntennaAgentRunnerTests(unittest.TestCase):
    def setUp(self):
        self.agent = create_default_agent()
        self.runner = AntennaAgentRunner(self.agent)
        self.baseline = self.agent.create_design("inset_patch")
        self.baseline_snapshot = self.baseline.to_dict()

    def _run(self, planner, *, baseline=_DEFAULT_BASELINE, budgets=None, memory=None):
        if baseline is _DEFAULT_BASELINE:
            baseline = self.baseline
        return self.runner.run(
            baseline_design=baseline,
            instruction="Apply the requested antenna change.",
            project_memory=memory or {"schema_version": 1, "requirements": []},
            planner=planner,
            budgets=budgets,
        )

    def test_one_execute_then_finish_returns_unpublished_candidate(self):
        planner = ScriptedPlanner(
            step(("parameter.set", {"key": "patch_width_mm", "value": 40.0})),
            step(status="finish", message="The requested edit is complete."),
        )
        result = self._run(planner)

        self.assertEqual(result.outcome, "finished")
        self.assertEqual(result.final_design.value("patch_width_mm"), 40.0)
        self.assertTrue(result.has_publishable_change)
        self.assertEqual([item.execution_status for item in result.trajectory], ["accepted", "finished"])
        self.assertEqual(self.baseline.to_dict(), self.baseline_snapshot)

    def test_two_successful_batches_then_finish(self):
        memory = {"schema_version": 1, "decisions": [{"key": "antenna_family", "value": "inset_patch_v2"}]}
        planner = ScriptedPlanner(
            step(("parameter.set", {"key": "patch_width_mm", "value": 40.0})),
            step(("parameter.set", {"key": "patch_length_mm", "value": 31.0})),
            step(status="finish", message="Complete."),
        )
        result = self._run(planner, memory=memory)

        self.assertEqual(result.outcome, "finished")
        self.assertEqual(result.final_design.value("patch_width_mm"), 40.0)
        self.assertEqual(result.final_design.value("patch_length_mm"), 31.0)
        self.assertEqual(len(result.aggregate_calls), 2)
        self.assertEqual(planner.requests[1]["agent_observation"].outcome, "accepted")
        self.assertEqual(planner.requests[0]["remaining_budgets"], AgentLoopBudgets())
        self.assertEqual(
            planner.requests[1]["remaining_budgets"],
            AgentLoopBudgets(4, 2, 2, 23),
        )
        self.assertEqual(planner.requests[0]["project_memory"], memory)
        self.assertEqual(planner.requests[1]["project_memory"], memory)
        self.assertEqual(planner.requests[1]["current_design"]["revision"], self.baseline.revision + 1)
        self.assertEqual(self.baseline.to_dict(), self.baseline_snapshot)

    def test_rejected_batch_can_be_replanned_to_success(self):
        planner = ScriptedPlanner(
            step(("parameter.set", {"key": "not_installed", "value": 1.0})),
            step(("parameter.set", {"key": "patch_width_mm", "value": 40.0})),
            step(status="finish", message="Recovered."),
        )
        result = self._run(planner)

        self.assertEqual(result.outcome, "finished")
        self.assertEqual([item.execution_status for item in result.trajectory], ["rejected", "accepted", "finished"])
        self.assertEqual(planner.requests[1]["agent_observation"].outcome, "rejected")
        self.assertEqual(result.final_design.value("patch_width_mm"), 40.0)

    def test_identical_rejected_action_is_skipped_without_second_execution_budget(self):
        rejected = step(("parameter.set", {"key": "not_installed", "value": 1.0}))
        planner = ScriptedPlanner(rejected, rejected)
        result = self._run(planner)

        self.assertEqual(result.outcome, "duplicate_rejection")
        self.assertEqual([item.execution_status for item in result.trajectory], ["rejected", "skipped_duplicate"])
        self.assertEqual(result.trajectory[0].budget_counters.rejected_action_batches, 1)
        self.assertEqual(result.trajectory[1].budget_counters.rejected_action_batches, 1)
        self.assertIsNone(result.final_design)

    def test_accepted_executor_no_op_is_reported_as_no_progress(self):
        current = self.baseline.value("patch_width_mm")
        planner = ScriptedPlanner(
            step(("parameter.set", {"key": "patch_width_mm", "value": current})),
            step(status="finish", message="No change was needed."),
        )
        result = self._run(planner)

        self.assertEqual(result.outcome, "finished")
        self.assertEqual(result.trajectory[0].execution_status, "no_progress")
        self.assertEqual(planner.requests[1]["agent_observation"].failure_category, "no_progress")
        self.assertFalse(result.has_publishable_change)
        self.assertEqual(result.final_design.revision, self.baseline.revision)

    def test_semantic_state_cycle_terminates_and_rolls_back(self):
        original = self.baseline.value("patch_width_mm")
        planner = ScriptedPlanner(
            step(("parameter.set", {"key": "patch_width_mm", "value": original + 1.0})),
            step(("parameter.set", {"key": "patch_width_mm", "value": original})),
        )
        result = self._run(planner)

        self.assertEqual(result.outcome, "cycle_detected")
        self.assertEqual(result.trajectory[-1].execution_status, "cycle_detected")
        self.assertIsNone(result.final_design)
        self.assertEqual(result.working_design_hash, semantic_design_hash(self.baseline))
        self.assertEqual(self.baseline.to_dict(), self.baseline_snapshot)

    def test_accepted_batch_then_clarify_discards_candidate(self):
        planner = ScriptedPlanner(
            step(("parameter.set", {"key": "patch_width_mm", "value": 40.0})),
            step(status="clarify", message="Which excitation should I use?"),
        )
        result = self._run(planner)

        self.assertEqual(result.outcome, "clarify")
        self.assertIsNone(result.final_design)
        self.assertFalse(result.has_publishable_change)
        self.assertEqual(result.working_design_hash, semantic_design_hash(self.baseline))
        self.assertEqual(self.baseline.to_dict(), self.baseline_snapshot)

    def test_accepted_batch_then_refuse_discards_candidate(self):
        planner = ScriptedPlanner(
            step(("parameter.set", {"key": "patch_width_mm", "value": 40.0})),
            step(status="refuse", message="The next operation is unavailable."),
        )
        result = self._run(planner)

        self.assertEqual(result.outcome, "refuse")
        self.assertIsNone(result.final_design)
        self.assertEqual(result.working_design_hash, semantic_design_hash(self.baseline))
        self.assertEqual(self.baseline.to_dict(), self.baseline_snapshot)

    def test_accepted_batch_then_provider_failure_discards_and_sanitizes(self):
        planner = ScriptedPlanner(
            step(("parameter.set", {"key": "patch_width_mm", "value": 40.0})),
            RuntimeError("network failed api_key=do-not-record Bearer hidden-value"),
        )
        result = self._run(planner)
        serialized = json.dumps(result.to_dict())

        self.assertEqual(result.outcome, "provider_error")
        self.assertIsNone(result.final_design)
        self.assertNotIn("do-not-record", serialized)
        self.assertNotIn("hidden-value", serialized)
        self.assertNotIn("test-secret", serialized)
        self.assertEqual(self.baseline.to_dict(), self.baseline_snapshot)

    def test_action_batch_budget_exhaustion_rolls_back(self):
        planner = ScriptedPlanner(
            step(("parameter.set", {"key": "patch_width_mm", "value": 40.0})),
            step(("parameter.set", {"key": "patch_length_mm", "value": 31.0})),
        )
        result = self._run(
            planner,
            budgets=AgentLoopBudgets(
                decision_iterations=5,
                accepted_action_batches=1,
                rejected_action_batches=2,
                total_proposed_tool_calls=24,
            ),
        )

        self.assertEqual(result.outcome, "agent_limit")
        self.assertIn("accepted action-batch", result.message)
        self.assertEqual(result.trajectory[-1].execution_status, "agent_limit")
        self.assertIsNone(result.final_design)
        self.assertEqual(self.baseline.to_dict(), self.baseline_snapshot)

    def test_rejected_batch_budget_blocks_another_execution(self):
        planner = ScriptedPlanner(
            step(("parameter.set", {"key": "not_installed", "value": 1.0})),
            step(("parameter.set", {"key": "patch_width_mm", "value": 40.0})),
        )
        result = self._run(
            planner,
            budgets=AgentLoopBudgets(
                decision_iterations=5,
                accepted_action_batches=3,
                rejected_action_batches=1,
                total_proposed_tool_calls=24,
            ),
        )

        self.assertEqual(result.outcome, "agent_limit")
        self.assertIn("rejected action-batch", result.message)
        self.assertEqual(result.trajectory[-1].executed_calls, ())
        self.assertEqual(self.baseline.to_dict(), self.baseline_snapshot)

    def test_empty_project_can_create_recipe_then_composition_then_finish(self):
        planner = ScriptedPlanner(
            step(
                ("recipe.select", {"recipe_id": "inset_patch_v2"}),
                ("parameter.set", {"key": "frequency_ghz", "value": 2.45}),
                ("parameter.set", {"key": "material", "value": "FR4"}),
            ),
            step(
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
            ),
            step(status="finish", message="Patch and slot are complete."),
        )
        result = self._run(planner, baseline=None)

        self.assertEqual(result.outcome, "finished")
        self.assertEqual(result.final_design.recipe_id, "inset_patch_v2")
        self.assertEqual(result.final_design.revision, 1)
        self.assertEqual(result.final_design.value("slot_radius_mm"), 3.0)
        self.assertEqual(len(result.final_design.composed_operations), 1)
        first_names = [item["name"] for item in planner.requests[0]["capability_manifest"]["callable_tools"]]
        second_names = [item["name"] for item in planner.requests[1]["capability_manifest"]["callable_tools"]]
        self.assertEqual(first_names, ["recipe.select", "parameter.set"])
        self.assertIn("geometry.cylinder", second_names)
        self.assertTrue(result.has_publishable_change)

    def test_empty_project_later_failure_returns_no_design(self):
        planner = ScriptedPlanner(
            step(("recipe.select", {"recipe_id": "inset_patch_v2"})),
            RuntimeError("provider unavailable"),
        )
        result = self._run(planner, baseline=None)

        self.assertEqual(result.outcome, "provider_error")
        self.assertIsNone(result.final_design)
        self.assertIsNone(result.working_design_ref)
        self.assertEqual(result.working_design_hash, semantic_design_hash(None))

    def test_finish_without_design_supports_information_request(self):
        planner = ScriptedPlanner(step(status="finish", message="I can build three validated families."))
        result = self._run(planner, baseline=None)

        self.assertEqual(result.outcome, "finished")
        self.assertIsNone(result.final_design)
        self.assertFalse(result.has_publishable_change)
        self.assertEqual(len(result.trajectory), 1)

    def test_total_proposed_tool_call_limit_prevents_execution(self):
        planner = ScriptedPlanner(step(
            ("parameter.set", {"key": "patch_width_mm", "value": 40.0}),
            ("parameter.set", {"key": "patch_length_mm", "value": 31.0}),
        ))
        result = self._run(
            planner,
            budgets=AgentLoopBudgets(
                decision_iterations=5,
                accepted_action_batches=3,
                rejected_action_batches=2,
                total_proposed_tool_calls=1,
            ),
        )

        self.assertEqual(result.outcome, "agent_limit")
        self.assertIn("tool-call", result.message)
        self.assertEqual(result.trajectory[0].executed_calls, ())
        self.assertEqual(self.baseline.to_dict(), self.baseline_snapshot)

    def test_decision_iteration_limit_discards_unfinished_candidate(self):
        planner = ScriptedPlanner(
            step(("parameter.set", {"key": "patch_width_mm", "value": 40.0})),
        )
        result = self._run(
            planner,
            budgets=AgentLoopBudgets(
                decision_iterations=1,
                accepted_action_batches=3,
                rejected_action_batches=2,
                total_proposed_tool_calls=24,
            ),
        )

        self.assertEqual(result.outcome, "agent_limit")
        self.assertIn("decision-iteration", result.message)
        self.assertIsNone(result.final_design)
        self.assertEqual(self.baseline.to_dict(), self.baseline_snapshot)

    def test_non_agent_step_is_invalid_plan_and_preserves_baseline(self):
        planner = ScriptedPlanner({"status": "finish"})
        result = self._run(planner)

        self.assertEqual(result.outcome, "invalid_plan")
        self.assertEqual(result.trajectory[0].execution_status, "invalid_plan")
        self.assertIsNone(result.final_design)
        self.assertEqual(self.baseline.to_dict(), self.baseline_snapshot)

    def test_existing_executor_behavior_remains_unchanged(self):
        direct_plan = step(("parameter.set", {"key": "patch_width_mm", "value": 40.0}))
        one_shot = self.agent.execute_llm_plan(
            self.baseline,
            LLMToolPlan("execute", direct_plan.message, direct_plan.calls),
        )
        planner = ScriptedPlanner(direct_plan, step(status="finish", message="Done."))
        looped = self._run(planner)

        self.assertEqual(one_shot.design.to_dict(), looped.final_design.to_dict())
        self.assertEqual(self.baseline.to_dict(), self.baseline_snapshot)


if __name__ == "__main__":
    unittest.main()
