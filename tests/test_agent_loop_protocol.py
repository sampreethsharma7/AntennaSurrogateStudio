import json
import unittest

from studio.antenna_agent import create_default_agent
from studio.antenna_llm_planner import (
    AgentLoopBudgets,
    AgentStep,
    AgentStepObservation,
    AgentTerminalResult,
    AgentTrajectoryEntry,
    LLMToolPlan,
    PlannedToolCall,
    SchemaConstrainedLLMPlanner,
    agent_step_json_schema,
    build_agent_step_exchange,
    parse_agent_step,
    parse_llm_tool_plan,
    plan_json_schema,
)
from studio.antenna_tools import CapabilityError


class _Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class AgentLoopProtocolTests(unittest.TestCase):
    def setUp(self):
        agent = create_default_agent()
        state = agent.create_design("inset_patch")
        self.manifest = agent.capability_manifest(state)
        self.names = tuple(item["name"] for item in self.manifest["callable_tools"])
        self.call = {
            "name": "parameter.set",
            "arguments": {"key": "patch_width_mm", "value": 40},
        }

    def _payload(self, status, calls=None):
        return {
            "schema_version": 1,
            "status": status,
            "message": f"{status} message",
            "calls": list(calls or ()),
        }

    def test_execute_requires_at_least_one_call(self):
        parsed = parse_agent_step(
            self._payload("execute", [self.call]),
            callable_tool_names=self.names,
        )
        self.assertEqual(parsed.status, "execute")
        self.assertEqual(parsed.calls[0].name, "parameter.set")
        with self.assertRaisesRegex(CapabilityError, "at least one"):
            parse_agent_step(self._payload("execute"), callable_tool_names=self.names)

    def test_finish_requires_zero_calls(self):
        parsed = parse_agent_step(self._payload("finish"), callable_tool_names=self.names)
        self.assertEqual(parsed.calls, ())
        with self.assertRaisesRegex(CapabilityError, "zero tool calls"):
            parse_agent_step(self._payload("finish", [self.call]), callable_tool_names=self.names)

    def test_clarify_requires_zero_calls(self):
        parsed = parse_agent_step(self._payload("clarify"), callable_tool_names=self.names)
        self.assertEqual(parsed.calls, ())
        with self.assertRaisesRegex(CapabilityError, "zero tool calls"):
            parse_agent_step(self._payload("clarify", [self.call]), callable_tool_names=self.names)

    def test_refuse_requires_zero_calls(self):
        parsed = parse_agent_step(self._payload("refuse"), callable_tool_names=self.names)
        self.assertEqual(parsed.calls, ())
        with self.assertRaisesRegex(CapabilityError, "zero tool calls"):
            parse_agent_step(self._payload("refuse", [self.call]), callable_tool_names=self.names)

    def test_runtime_and_unknown_statuses_are_not_model_emittable(self):
        for status in ("agent_limit", "provider_error", "cycle_detected", "invalid_plan", "duplicate_rejection", "unknown"):
            with self.subTest(status=status), self.assertRaisesRegex(CapabilityError, "invalid AgentStep status"):
                parse_agent_step(self._payload(status), callable_tool_names=self.names)
        with self.assertRaisesRegex(ValueError, "Unknown model-emitted"):
            AgentStep("agent_limit", "Runner only.")

    def test_registered_tool_argument_schemas_are_unchanged(self):
        old_schema = plan_json_schema(tuple(self.manifest["callable_tools"]))
        agent_schema = agent_step_json_schema(tuple(self.manifest["callable_tools"]))
        old_variants = old_schema["properties"]["calls"]["items"]["oneOf"]
        agent_variants = agent_schema["properties"]["calls"]["items"]["oneOf"]
        self.assertEqual(agent_variants, old_variants)
        self.assertEqual(
            agent_schema["properties"]["status"]["enum"],
            ["execute", "finish", "clarify", "refuse"],
        )
        json.dumps(agent_schema)

    def test_accepted_and_rejected_observations_round_trip(self):
        budgets = AgentLoopBudgets(4, 2, 1, 20)
        accepted = AgentStepObservation(
            outcome="accepted",
            planned_calls=(PlannedToolCall("parameter.set", {"key": "patch_width_mm", "value": 40}),),
            executed_calls=(PlannedToolCall("parameter.set", {"key": "patch_width_mm", "value": 40}),),
            deterministic_changes=({"kind": "parameter", "key": "patch_width_mm"},),
            validation_result={"status": "valid"},
            working_design_ref={"design_id": "design-1", "revision": 2},
            working_design_hash="abc123",
            remaining_budgets=budgets,
        )
        self.assertEqual(AgentStepObservation.from_dict(accepted.to_dict()), accepted)

        rejected = AgentStepObservation(
            outcome="rejected",
            planned_calls=(PlannedToolCall("parameter.set", {"key": "missing", "value": 1}),),
            validation_result={"status": "rejected"},
            working_design_ref={"design_id": "design-1", "revision": 1},
            working_design_hash="unchanged",
            failure_category="applicability",
            sanitized_error="The parameter is unavailable.",
            failed_action_fingerprint="failure-1",
            remaining_budgets=budgets,
        )
        self.assertEqual(AgentStepObservation.from_dict(rejected.to_dict()), rejected)

    def test_budget_defaults_round_trip(self):
        budgets = AgentLoopBudgets()
        self.assertEqual(
            budgets.to_dict(),
            {
                "decision_iterations": 5,
                "accepted_action_batches": 3,
                "rejected_action_batches": 2,
                "total_proposed_tool_calls": 24,
                "analysis_batches": 3,
            },
        )
        self.assertEqual(AgentLoopBudgets.from_dict(budgets.to_dict()), budgets)

    def test_trajectory_serialization_redacts_secret_metadata(self):
        step = AgentStep("finish", "Done.")
        entry = AgentTrajectoryEntry(
            iteration_index=1,
            working_state_before={"design_id": "design-1", "revision": 1},
            returned_step=step,
            planner_metadata={
                "backend": "test",
                "authorization": "Bearer should-not-appear",
                "nested": {"api_key": "also-secret", "schema_repairs": 0},
            },
            execution_status="not_executed",
            executed_calls=(),
            deterministic_changes=(),
            sanitized_failure={"category": "none", "access_token": "hidden"},
            working_state_after={"design_id": "design-1", "revision": 1},
            observation=None,
            budget_counters=AgentLoopBudgets(),
        )
        terminal = AgentTerminalResult(
            outcome="finish",
            message="Done.",
            working_design_ref={"design_id": "design-1", "revision": 1},
            working_design_hash="hash-1",
            final_step=step,
            trajectory=(entry,),
        )
        serialized = json.dumps(terminal.to_dict())
        self.assertNotIn("should-not-appear", serialized)
        self.assertNotIn("also-secret", serialized)
        self.assertNotIn("hidden", serialized)
        self.assertIn("[REDACTED]", serialized)

    def test_existing_tool_plan_v1_contract_remains_separate(self):
        plan = parse_llm_tool_plan(
            self._payload("execute", [self.call]),
            callable_tool_names=self.names,
        )
        self.assertEqual(plan.status, "execute")
        with self.assertRaisesRegex(CapabilityError, "invalid plan status"):
            parse_llm_tool_plan(self._payload("finish"), callable_tool_names=self.names)

    def test_existing_provider_exchange_accepts_structured_agent_observation(self):
        captured = {}
        plan = self._payload("execute", [self.call])
        observation = AgentStepObservation(
            outcome="rejected",
            planned_calls=(PlannedToolCall("parameter.set", {"key": "missing", "value": 1}),),
            validation_result={"status": "rejected"},
            working_design_ref={"design_id": "design-1", "revision": 1},
            working_design_hash="unchanged",
            failure_category="applicability",
            sanitized_error="The parameter is unavailable.",
            failed_action_fingerprint="failure-1",
            remaining_budgets=AgentLoopBudgets(),
        )

        def opener(request, timeout):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _Response({"message": {"content": json.dumps(plan)}})

        planner = SchemaConstrainedLLMPlanner(
            "test-model",
            base_url="http://127.0.0.1:11434",
            opener=opener,
        )
        planner.plan(
            instruction="Try another valid approach",
            current_design=None,
            capability_manifest=self.manifest,
            agent_observation=observation,
        )
        user_payload = json.loads(captured["body"]["messages"][1]["content"])
        self.assertEqual(user_payload["agent_observation"], observation.to_dict())

    def test_existing_provider_transport_can_decode_agent_step_contract(self):
        captured = {}

        def opener(request, timeout):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _Response({"message": {"content": json.dumps(self._payload("finish"))}})

        planner = SchemaConstrainedLLMPlanner(
            "test-model",
            base_url="http://127.0.0.1:11434",
            opener=opener,
        )
        budgets = AgentLoopBudgets(4, 2, 1, 20)
        result = planner.plan_agent_step(
            instruction="Report when the requested design is complete.",
            current_design=None,
            capability_manifest=self.manifest,
            project_memory={"schema_version": 1},
            remaining_budgets=budgets,
        )

        self.assertEqual(result.status, "finish")
        self.assertEqual(
            captured["body"]["format"]["properties"]["status"]["enum"],
            ["execute", "finish", "clarify", "refuse"],
        )
        user_payload = json.loads(captured["body"]["messages"][1]["content"])
        self.assertEqual(user_payload["remaining_budgets"], budgets.to_dict())
        self.assertIn("agent_step_schema", user_payload)

    def test_agent_step_context_supports_progressive_state_dependent_capabilities(self):
        agent = create_default_agent()
        budgets = AgentLoopBudgets()

        empty_manifest = agent.capability_manifest(None)
        first_exchange = build_agent_step_exchange(
            instruction="Create a supported antenna and add a composed feature.",
            current_design=None,
            capability_manifest=empty_manifest,
            remaining_budgets=budgets,
        )
        first_payload = json.loads(first_exchange.user_content)
        first_names = [
            item["name"]
            for item in first_payload["runtime_capabilities"]["callable_tools"]
        ]
        self.assertEqual(first_names, ["recipe.select", "parameter.set"])
        self.assertIn("CURRENT working state", first_exchange.system_instruction)
        self.assertIn("newly generated capability manifest", first_exchange.system_instruction)
        self.assertIn("prefer execute over refuse", first_exchange.system_instruction)
        self.assertIn("Do not refuse solely because a later step is unavailable", first_exchange.system_instruction)

        design = agent.execute_llm_plan(
            None,
            parse_llm_tool_plan(
                {
                    "schema_version": 1,
                    "status": "execute",
                    "message": "Create the base antenna first.",
                    "calls": [
                        {"name": "recipe.select", "arguments": {"recipe_id": "inset_patch_v2"}},
                        {"name": "parameter.set", "arguments": {"key": "frequency_ghz", "value": 2.45}},
                        {"name": "parameter.set", "arguments": {"key": "material", "value": "FR4"}},
                    ],
                },
                callable_tool_names=tuple(first_names),
            ),
        ).design
        second_manifest = agent.capability_manifest(design)
        accepted_calls = (
            PlannedToolCall("recipe.select", {"recipe_id": "inset_patch_v2"}),
            PlannedToolCall("parameter.set", {"key": "frequency_ghz", "value": 2.45}),
            PlannedToolCall("parameter.set", {"key": "material", "value": "FR4"}),
        )
        second_exchange = build_agent_step_exchange(
            instruction="Create a supported antenna and add a composed feature.",
            current_design=design.to_dict(),
            capability_manifest=second_manifest,
            remaining_budgets=AgentLoopBudgets(4, 2, 2, 21),
            agent_observation=AgentStepObservation(
                outcome="accepted",
                planned_calls=accepted_calls,
                executed_calls=accepted_calls,
                working_design_ref={"design_id": design.design_id, "revision": design.revision},
                working_design_hash="accepted-state",
                remaining_budgets=AgentLoopBudgets(4, 2, 2, 21),
            ),
        )
        second_payload = json.loads(second_exchange.user_content)
        second_names = [
            item["name"]
            for item in second_payload["runtime_capabilities"]["callable_tools"]
        ]
        self.assertIn("geometry.cylinder", second_names)
        self.assertIn("boolean.subtract", second_names)
        self.assertEqual(second_payload["agent_observation"]["outcome"], "accepted")

        primitive_step = parse_agent_step(
            {
                "schema_version": 1,
                "status": "execute",
                "message": "Use the capabilities unlocked by the accepted base design.",
                "calls": [
                    {
                        "name": "parameter.create",
                        "arguments": {
                            "key": "feature_radius_mm", "label": "Feature radius",
                            "value": 3.0, "unit": "mm", "sweepable": True,
                        },
                    },
                    {
                        "name": "geometry.cylinder",
                        "arguments": {
                            "object_id": "feature_tool", "material_id": "copper", "axis": "z",
                            "tags": ["planner_created", "boolean_tool", "slot"],
                            "dimensions": {
                                "center_1": 0, "center_2": 0, "radius": "feature_radius_mm",
                                "start": "substrate_thickness_mm",
                                "end": "substrate_thickness_mm+copper_thickness_mm",
                            },
                        },
                    },
                    {
                        "name": "boolean.subtract",
                        "arguments": {
                            "operation_id": "feature_subtract",
                            "target_id": "element_1_1_patch",
                            "tool_ids": ["feature_tool"],
                        },
                    },
                ],
            },
            callable_tool_names=tuple(second_names),
        )
        self.assertEqual(primitive_step.status, "execute")
        completed_design = agent.execute_llm_plan(
            design,
            LLMToolPlan(primitive_step.status, primitive_step.message, primitive_step.calls),
        ).design
        third_manifest = agent.capability_manifest(completed_design)
        third_exchange = build_agent_step_exchange(
            instruction="Create a supported antenna and add a composed feature.",
            current_design=completed_design.to_dict(),
            capability_manifest=third_manifest,
            remaining_budgets=AgentLoopBudgets(3, 1, 2, 18),
        )
        third_names = tuple(
            item["name"] for item in third_manifest["callable_tools"]
        )
        self.assertIn("composition.set_scope", third_names)

        finish = parse_agent_step(
            {
                "schema_version": 1,
                "status": "finish",
                "message": "The complete request is now satisfied.",
                "calls": [],
            },
            callable_tool_names=third_names,
        )
        self.assertEqual(finish.status, "finish")
        self.assertEqual(finish.calls, ())
        self.assertIn("complete user request", third_exchange.system_instruction)
        self.assertIn("Do not finish merely because", third_exchange.system_instruction)


if __name__ == "__main__":
    unittest.main()
