import json
import unittest

from studio.antenna_agent import AntennaDesignAgent, AgentInstructionError
from studio.antenna_agent_runner import AntennaAgentRunner, semantic_design_hash
from studio.antenna_analysis import (
    AnalysisMeasurement, EngineeringAnalysisResult, stable_analysis_id,
)
from studio.antenna_engineering import EngineeringDesignRef
from studio.antenna_llm_planner import (
    AgentLoopBudgets, AgentStep, AgentStepObservation, LLMToolPlan, PlannedToolCall,
    build_agent_step_exchange,
)
from studio.antenna_tools import ToolDefinition, create_tool_registry


def step(*calls, status="execute", message="test step"):
    return AgentStep(status, message, tuple(PlannedToolCall(name, arguments) for name, arguments in calls))


class ScriptedPlanner:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.last_run_metadata = {}

    def plan_agent_step(self, **request):
        self.requests.append(request)
        if not self.responses:
            raise RuntimeError("No scripted response remains.")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class AnalysisToolRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.invocations = 0
        registry = create_tool_registry()

        def counting(design, arguments):
            self.invocations += 1
            digest = semantic_design_hash(design)
            return EngineeringAnalysisResult(
                "engineering.counting_summary", "test-1",
                stable_analysis_id("engineering.counting_summary", "test-1", arguments, digest),
                EngineeringDesignRef(design.design_id, design.revision), digest, "completed",
                {"geometry_object_count": AnalysisMeasurement(len(design.geometry), "count")},
                ("Canonical geometry is counted directly.",),
                "Current canonical design inventory.",
                ("No RF behavior is evaluated.",),
                "Counted canonical geometry. api_key=analysis-secret",
            )

        registry.register_tool(ToolDefinition(
            "engineering.counting_summary", "engineering_analysis", "Test-only counting analysis.",
            "analysis", counting, planner_exposed=True,
            argument_schema={"type": "object", "additionalProperties": False, "properties": {}},
            effect="analysis", tool_version="test-1", requires_design=True,
        ))
        self.agent = AntennaDesignAgent(registry)
        self.runner = AntennaAgentRunner(self.agent)
        self.design = self.agent.create_design("inset_patch")
        self.snapshot = self.design.to_dict()

    @staticmethod
    def analysis_step(name="engineering.counting_summary"):
        return step((name, {}), message="Inspect the current design.")

    @staticmethod
    def design_step(value=40.0):
        return step(("parameter.set", {"key": "patch_width_mm", "value": value}))

    def run_agent(self, planner, *, baseline=None, budgets=None):
        return self.runner.run(
            baseline_design=self.design if baseline is None else baseline,
            instruction="Inspect or update the antenna.", project_memory={"schema_version": 1},
            planner=planner, budgets=budgets,
        )

    def test_registry_effect_and_existing_design_action_behavior(self):
        self.assertEqual(self.agent.registry.tool("parameter.create").effect, "design_action")
        self.assertEqual(self.agent.registry.tool("engineering.design_summary").effect, "analysis")
        manifest = self.agent.capability_manifest(self.design)
        effects = {item["name"]: item["effect"] for item in manifest["callable_tools"]}
        self.assertEqual(effects["parameter.set"], "design_action")
        self.assertEqual(effects["engineering.design_summary"], "analysis")
        result = self.agent.execute_llm_plan(
            self.design, LLMToolPlan("execute", "edit", self.design_step().calls),
        )
        self.assertEqual(result.design.value("patch_width_mm"), 40.0)

    def test_analysis_result_round_trip_and_no_design_change(self):
        result = self.agent.execute_analysis_batch(
            self.design, self.analysis_step("engineering.design_summary").calls,
        )[0]
        self.assertEqual(EngineeringAnalysisResult.from_dict(result.to_dict()), result)
        self.assertEqual(result.semantic_design_hash, semantic_design_hash(self.design))
        self.assertEqual(result.working_design_ref.revision, self.design.revision)
        self.assertEqual(self.design.to_dict(), self.snapshot)

    def test_analysis_is_accepted_not_no_progress_and_reaches_next_iteration(self):
        planner = ScriptedPlanner(self.analysis_step(), step(status="finish", message="Inspection complete."))
        result = self.run_agent(planner)
        observation = planner.requests[1]["agent_observation"]
        self.assertEqual(result.outcome, "finished")
        self.assertEqual(result.trajectory[0].execution_status, "accepted_analysis")
        self.assertEqual(observation.outcome, "accepted_analysis")
        self.assertEqual(len(observation.analysis_results), 1)
        self.assertEqual(observation.working_design_hash, semantic_design_hash(self.design))
        self.assertEqual(result.final_design.revision, self.design.revision)
        self.assertFalse(result.has_publishable_change)

    def test_provider_neutral_context_serializes_latest_analysis_results(self):
        analysis = self.agent.execute_analysis_batch(
            self.design, self.analysis_step("engineering.design_summary").calls,
        )[0]
        call = PlannedToolCall("engineering.design_summary", {})
        observation = AgentStepObservation(
            outcome="accepted_analysis", planned_calls=(call,), executed_calls=(call,),
            working_design_ref={"design_id": self.design.design_id, "revision": self.design.revision},
            working_design_hash=semantic_design_hash(self.design),
            remaining_budgets=AgentLoopBudgets(4, 3, 2, 23, 2),
            validation_result={"status": "read_only", "design_unchanged": True},
            analysis_results=(analysis,),
        )
        exchange = build_agent_step_exchange(
            instruction="Use the inspection result.", current_design=self.design.to_dict(),
            capability_manifest=self.agent.capability_manifest(self.design),
            remaining_budgets=observation.remaining_budgets, agent_observation=observation,
        )
        payload = json.loads(exchange.user_content)
        self.assertEqual(payload["agent_observation"]["outcome"], "accepted_analysis")
        self.assertEqual(payload["agent_observation"]["analysis_results"], [analysis.to_dict()])
        self.assertIn("without modifying it", exchange.system_instruction)
        self.assertIn("not as full-wave validation", exchange.system_instruction)

    def test_analysis_then_design_action_and_design_action_then_analysis(self):
        for responses, expected in (
            ((self.analysis_step(), self.design_step(), step(status="finish", message="Done.")),
             ["accepted_analysis", "accepted", "finished"]),
            ((self.design_step(), self.analysis_step(), step(status="finish", message="Done.")),
             ["accepted", "accepted_analysis", "finished"]),
        ):
            with self.subTest(expected=expected):
                result = self.run_agent(ScriptedPlanner(*responses))
                self.assertEqual([item.execution_status for item in result.trajectory], expected)
                self.assertEqual(result.final_design.value("patch_width_mm"), 40.0)

    def test_mixed_effect_batch_is_rejected_transactionally(self):
        mixed = step(
            ("engineering.counting_summary", {}),
            ("parameter.set", {"key": "patch_width_mm", "value": 40.0}),
        )
        planner = ScriptedPlanner(mixed, step(status="finish", message="No changes applied."))
        result = self.run_agent(planner)
        self.assertEqual(result.trajectory[0].execution_status, "rejected")
        self.assertEqual(result.trajectory[0].observation.failure_category, "mixed_tool_effects")
        self.assertEqual(self.invocations, 0)
        self.assertEqual(result.final_design.to_dict(), self.snapshot)

    def test_analysis_budget_is_enforced_and_shared_budgets_decrement(self):
        planner = ScriptedPlanner(self.analysis_step(), self.analysis_step())
        result = self.run_agent(planner, budgets=AgentLoopBudgets(5, 3, 2, 24, analysis_batches=1))
        self.assertEqual(result.outcome, "agent_limit")
        self.assertIn("analysis-batch", result.message)
        self.assertEqual(result.trajectory[0].budget_counters.analysis_batches, 0)
        self.assertEqual(result.trajectory[0].budget_counters.total_proposed_tool_calls, 23)
        self.assertEqual(result.trajectory[0].budget_counters.accepted_action_batches, 3)

    def test_repeated_analysis_uses_cache_but_still_consumes_batch(self):
        planner = ScriptedPlanner(
            self.analysis_step(), self.analysis_step(), step(status="finish", message="Done."),
        )
        result = self.run_agent(planner)
        self.assertEqual(self.invocations, 1)
        self.assertEqual(result.trajectory[0].analysis_cache_hits, 0)
        self.assertEqual(result.trajectory[1].analysis_cache_hits, 1)
        self.assertEqual(result.trajectory[1].budget_counters.analysis_batches, 1)
        self.assertEqual(
            result.trajectory[0].analysis_results[0].analysis_id,
            result.trajectory[1].analysis_results[0].analysis_id,
        )

    def test_design_change_invalidates_analysis_cache(self):
        planner = ScriptedPlanner(
            self.analysis_step(), self.design_step(), self.analysis_step(),
            step(status="finish", message="Done."),
        )
        result = self.run_agent(planner)
        self.assertEqual(result.outcome, "finished")
        self.assertEqual(self.invocations, 2)
        first = result.trajectory[0].analysis_results[0]
        second = result.trajectory[2].analysis_results[0]
        self.assertNotEqual(first.semantic_design_hash, second.semantic_design_hash)
        self.assertNotEqual(first.analysis_id, second.analysis_id)

    def test_analysis_is_unavailable_without_a_design(self):
        manifest = self.agent.capability_manifest(None)
        self.assertNotIn("engineering.design_summary", {item["name"] for item in manifest["callable_tools"]})
        with self.assertRaisesRegex(AgentInstructionError, "requires a current canonical design"):
            self.agent.execute_analysis_batch(None, self.analysis_step().calls)

    def test_clarify_refuse_and_provider_error_after_analysis_preserve_baseline(self):
        endings = (
            step(status="clarify", message="Which option?"),
            step(status="refuse", message="Unavailable."),
            RuntimeError("network error api_key=provider-secret"),
        )
        for ending in endings:
            with self.subTest(ending=type(ending).__name__):
                result = self.run_agent(ScriptedPlanner(self.analysis_step(), ending))
                self.assertIsNone(result.final_design)
                self.assertEqual(result.working_design_hash, semantic_design_hash(self.design))
                self.assertEqual(self.design.to_dict(), self.snapshot)

    def test_trajectory_records_results_and_redacts_secrets(self):
        result = self.run_agent(ScriptedPlanner(self.analysis_step(), step(status="finish", message="Done.")))
        serialized = json.dumps(result.to_dict())
        self.assertIn("analysis_results", serialized)
        self.assertIn("[REDACTED]", serialized)
        self.assertNotIn("analysis-secret", serialized)
        self.assertNotIn("reasoning", serialized.casefold())

    def test_legacy_one_shot_design_action_unchanged_and_analysis_rejected(self):
        plan = LLMToolPlan("execute", "edit", self.design_step().calls)
        result = self.agent.execute_llm_plan(self.design, plan)
        self.assertEqual(result.design.value("patch_width_mm"), 40.0)
        with self.assertRaises(AgentInstructionError):
            self.agent.execute_llm_plan(
                self.design, LLMToolPlan("execute", "inspect", self.analysis_step().calls),
            )


if __name__ == "__main__":
    unittest.main()
