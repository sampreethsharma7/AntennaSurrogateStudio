import json
import tempfile
import unittest
from pathlib import Path

from studio.antenna_agent import create_default_agent
from studio.antenna_builder import (
    BuilderProjectSession,
    ProjectMemory,
    execute_builder_turn,
    load_builder_session,
    save_builder_session,
)
from studio.antenna_llm_planner import (
    AgentLoopBudgets,
    AgentStep,
    LLMToolPlan,
    PlannedToolCall,
    SemanticMemoryProposal,
    build_agent_step_exchange,
    parse_agent_step,
)
from studio.antenna_tools import CapabilityError


def proposal(
    action,
    semantic_kind,
    key,
    value,
    evidence_quote,
    *,
    unit=None,
    reference_id=None,
):
    return SemanticMemoryProposal(
        action=action,
        semantic_kind=semantic_kind,
        key=key,
        value=value,
        unit=unit,
        evidence_quote=evidence_quote,
        reference_id=reference_id,
    )


def step(*calls, status="execute", message="Continue.", proposals=()):
    return AgentStep(
        status=status,
        message=message,
        calls=tuple(PlannedToolCall(name, arguments) for name, arguments in calls),
        memory_proposals=tuple(proposals),
    )


class Planner:
    backend_id = "semantic-memory-test"
    model = "mock"

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


class SemanticProjectMemoryTests(unittest.TestCase):
    def setUp(self):
        root = Path.cwd() / ".test_runs"
        root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=root)
        self.project_path = Path(self.temp_dir.name)
        self.session = BuilderProjectSession(
            design=create_default_agent().create_design("inset_patch")
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def audit(self):
        path = self.project_path / "design" / "planner_ab.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def run_terminal(self, instruction, *proposals_, status="finish", turn_id="turn-1"):
        planner = Planner(step(
            status=status,
            message="Terminal response.",
            proposals=proposals_,
        ))
        return execute_builder_turn(
            self.project_path,
            self.session,
            instruction,
            planner=planner,
            turn_id=turn_id,
        )

    @staticmethod
    def semantic_items(memory):
        return tuple(
            item
            for collection in (memory.requirements, memory.decisions)
            for item in collection
            if item.source == "user_semantic"
        )

    def test_rhcp_goal_is_grounded_and_persisted_with_provenance(self):
        text = "I eventually want RHCP."
        self.run_terminal(
            text,
            proposal("upsert", "goal", "target_polarization", "RHCP", text),
        )
        item = self.semantic_items(self.session.memory)[0]
        self.assertEqual(item.key, "target_polarization")
        self.assertEqual(item.value, "rhcp")
        self.assertEqual(item.evidence_quote, text)
        self.assertEqual(item.source, "user_semantic")
        self.assertEqual(item.source_turn_id, "turn-1")
        self.assertEqual(item.created_design_revision, self.session.design.revision)

    def test_executed_canonical_value_is_rejected_as_redundant_semantic_memory(self):
        text = "Use independent ports for now."
        execute_builder_turn(
            self.project_path,
            self.session,
            text,
            planner=Planner(
                step(("excitation.set_strategy", {"strategy": "independent_ports"})),
                step(
                    status="finish",
                    message="Independent ports are realized.",
                    proposals=(proposal(
                        "upsert", "preference", "excitation_preference",
                        "independent_ports", text,
                    ),),
                ),
            ),
            turn_id="turn-independent",
        )
        self.assertEqual(self.semantic_items(self.session.memory), ())
        rejected = self.audit()[-1]["semantic_memory"]["rejected"]
        self.assertEqual(rejected[0]["reason"], "duplicates_current_canonical_action")
        self.assertEqual(self.session.design.excitation.strategy, "independent_ports")

    def test_future_corporate_intent_does_not_change_canonical_excitation(self):
        text = "I may want a corporate feed later."
        before = self.session.design
        self.run_terminal(
            text,
            proposal(
                "upsert", "future_intent", "feed_network_future_goal",
                "corporate_feed", text,
            ),
        )
        self.assertIs(self.session.design, before)
        self.assertEqual(self.session.design.excitation.strategy, "single_element_feed")
        self.assertEqual(self.semantic_items(self.session.memory)[0].value, "corporate_feed")

    def test_constraint_supersede_and_resolve_preserve_history(self):
        first = "Keep board width below 80 mm."
        self.run_terminal(
            first,
            proposal(
                "upsert", "constraint", "board_width_limit", 80, first, unit="mm",
            ),
            turn_id="turn-limit-1",
        )
        original = self.semantic_items(self.session.memory)[0]

        second = "Actually 100 mm is okay."
        self.run_terminal(
            second,
            proposal(
                "supersede", "constraint", "board_width_limit", 100, second,
                unit="mm", reference_id=original.item_id,
            ),
            turn_id="turn-limit-2",
        )
        items = self.semantic_items(self.session.memory)
        self.assertEqual([item.status for item in items], ["superseded", "active"])
        self.assertEqual(items[1].value, 100)
        self.assertEqual(items[1].supersedes_item_id, original.item_id)

        third = "I no longer care about that board-width limit."
        self.run_terminal(
            third,
            proposal(
                "resolve", "constraint", "board_width_limit", None, third,
                reference_id=items[1].item_id,
            ),
            turn_id="turn-limit-3",
        )
        items = self.semantic_items(self.session.memory)
        self.assertEqual([item.status for item in items], ["superseded", "resolved", "resolved"])
        self.assertEqual(items[-1].supersedes_item_id, items[1].item_id)
        self.assertNotIn("board_width_limit", {
            item["key"] for item in self.session.memory.to_planner_dict().get("project_intent", [])
        })

    def test_priority_is_stored_and_transient_command_needs_no_proposal(self):
        text = "Bandwidth matters more than size."
        self.run_terminal(
            text,
            proposal(
                "upsert", "priority", "optimization_priority",
                "Bandwidth matters more than size", text,
            ),
            turn_id="turn-priority",
        )
        self.assertEqual(self.semantic_items(self.session.memory)[0].semantic_kind, "priority")

    def test_transient_slot_move_is_canonical_only(self):
        agent = create_default_agent()
        base = agent.create_design("inset_patch")
        slotted = agent.execute_llm_plan(base, LLMToolPlan(
            "execute",
            "Create center slot.",
            (
                PlannedToolCall("parameter.create", {
                    "key": "slot_radius_mm", "label": "Slot radius", "value": 3.0,
                    "unit": "mm", "sweepable": True,
                }),
                PlannedToolCall("geometry.cylinder", {
                    "object_id": "center_slot_tool", "material_id": "copper", "axis": "z",
                    "tags": ["planner_created", "boolean_tool", "slot"],
                    "dimensions": {
                        "center_1": 0, "center_2": 0, "radius": "slot_radius_mm",
                        "start": "substrate_thickness_mm",
                        "end": "substrate_thickness_mm+copper_thickness_mm",
                    },
                }),
                PlannedToolCall("boolean.subtract", {
                    "operation_id": "center_slot_subtract",
                    "target_id": "element_1_1_patch",
                    "tool_ids": ["center_slot_tool"],
                }),
            ),
        )).design
        group = slotted.composed_operations[0]
        cylinder = next(item for item in group.calls if item.name == "geometry.cylinder")
        transient_path = self.project_path / "transient"
        transient = BuilderProjectSession(design=slotted)
        execute_builder_turn(
            transient_path,
            transient,
            "Move the slot 3 mm right.",
            planner=Planner(
                step(("composition.update_operation", {
                    "group_id": group.group_id,
                    "operation_id": cylinder.operation_id,
                    "argument_updates": {"dimensions": {"center_1": 3}},
                })),
                step(status="finish", message="Slot moved."),
            ),
            turn_id="turn-transient",
        )
        updated_group = transient.design.composed_operations[0]
        updated_cylinder = next(
            item for item in updated_group.calls if item.operation_id == cylinder.operation_id
        )
        self.assertEqual(updated_cylinder.arguments()["dimensions"]["center_1"], 3)
        self.assertEqual(self.semantic_items(transient.memory), ())

    def test_invalid_grounding_and_nonexact_quote_are_rejected(self):
        text = "Make this a 2x3 array."
        self.run_terminal(
            text,
            proposal("upsert", "preference", "array_preference", "compact", text),
            proposal("upsert", "goal", "invented_goal", "high_gain", "User wants high gain."),
            turn_id="turn-invalid",
        )
        items = self.semantic_items(self.session.memory)
        self.assertEqual(items, ())
        rejected = self.audit()[-1]["semantic_memory"]["rejected"]
        self.assertEqual(
            {item["reason"] for item in rejected},
            {
                "value_not_lexically_grounded_in_evidence_quote",
                "evidence_quote_not_exact_current_turn_substring",
            },
        )

    def test_duplicate_key_and_mixed_valid_invalid_are_deterministic(self):
        text = "I eventually want RHCP, and bandwidth matters more than size."
        self.run_terminal(
            text,
            proposal("upsert", "goal", "target_polarization", "RHCP", "I eventually want RHCP"),
            proposal("upsert", "goal", "desired_polarization", "RHCP", "I eventually want RHCP"),
            proposal("upsert", "priority", "optimization_priority", "bandwidth matters more than size", "bandwidth matters more than size"),
            proposal("upsert", "goal", "unsupported_inference", "high_gain", "not present"),
            turn_id="turn-mixed",
        )
        active = [item for item in self.semantic_items(self.session.memory) if item.status == "active"]
        self.assertEqual({item.key for item in active}, {"target_polarization", "optimization_priority"})
        rejected = self.audit()[-1]["semantic_memory"]["rejected"]
        self.assertEqual(
            {item["reason"] for item in rejected},
            {"duplicate_key_in_turn", "evidence_quote_not_exact_current_turn_substring"},
        )

    def test_design_finish_and_semantic_memory_publish_once(self):
        text = "Set the patch width to 40 mm; bandwidth remains my main priority."
        result = execute_builder_turn(
            self.project_path,
            self.session,
            text,
            planner=Planner(
                step(("parameter.set", {"key": "patch_width_mm", "value": 40.0})),
                step(
                    status="finish",
                    message="Width updated.",
                    proposals=(proposal(
                        "upsert", "priority", "optimization_priority",
                        "bandwidth", "bandwidth remains my main priority",
                    ),),
                ),
            ),
            turn_id="turn-design-memory",
        )
        self.assertEqual(result.session.design.value("patch_width_mm"), 40.0)
        self.assertEqual(len(self.semantic_items(result.session.memory)), 1)
        audit = self.audit()[-1]["semantic_memory"]
        self.assertEqual(len(audit["applied_ids"]), 1)
        self.assertEqual(audit["rejected"], [])

    def test_invalid_proposal_does_not_block_valid_design_publication(self):
        text = "Set the patch width to 41 mm."
        result = execute_builder_turn(
            self.project_path,
            self.session,
            text,
            planner=Planner(
                step(("parameter.set", {"key": "patch_width_mm", "value": 41.0})),
                step(
                    status="finish",
                    message="Width updated despite auxiliary-memory rejection.",
                    proposals=(proposal(
                        "invented_action", "goal", "invented_goal", "gain", text,
                    ),),
                ),
            ),
            turn_id="turn-invalid-auxiliary",
        )
        self.assertEqual(result.session.design.value("patch_width_mm"), 41.0)
        self.assertEqual(self.semantic_items(result.session.memory), ())
        rejected = self.audit()[-1]["semantic_memory"]["rejected"]
        self.assertEqual(rejected[0]["reason"], "unsupported_action")

    def test_clarify_and_refuse_persist_grounded_intent_without_design_change(self):
        baseline = self.session.design
        clarify_text = "I eventually want RHCP, but which supported antenna should I start with?"
        clarify = self.run_terminal(
            clarify_text,
            proposal("upsert", "goal", "target_polarization", "RHCP", "I eventually want RHCP"),
            status="clarify",
            turn_id="turn-clarify",
        )
        self.assertEqual(clarify.terminal_result.outcome, "clarify")
        self.assertIs(self.session.design, baseline)

        refuse_text = "I may want a corporate feed later, even if it is unavailable now."
        refused = self.run_terminal(
            refuse_text,
            proposal(
                "upsert", "future_intent", "feed_network_future_goal",
                "corporate_feed", "I may want a corporate feed later",
            ),
            status="refuse",
            turn_id="turn-refuse",
        )
        self.assertEqual(refused.terminal_result.outcome, "refuse")
        self.assertIs(self.session.design, baseline)
        self.assertEqual(
            {item.key for item in self.semantic_items(self.session.memory) if item.status == "active"},
            {"target_polarization", "feed_network_future_goal"},
        )

    def test_provider_error_after_provisional_action_publishes_no_semantic_memory(self):
        baseline = self.session.design
        with self.assertRaises(CapabilityError):
            execute_builder_turn(
                self.project_path,
                self.session,
                "Set the patch width, and remember I eventually want RHCP.",
                planner=Planner(
                    step(("parameter.set", {"key": "patch_width_mm", "value": 40.0})),
                    RuntimeError("provider unavailable"),
                ),
                turn_id="turn-provider-error",
            )
        self.assertIs(self.session.design, baseline)
        self.assertEqual(self.semantic_items(self.session.memory), ())
        self.assertEqual(self.audit()[-1]["semantic_memory"]["proposed"], [])

    def test_save_reopen_and_future_planner_context_keep_state_and_intent_separate(self):
        text = "I may want a corporate feed later."
        self.run_terminal(
            text,
            proposal(
                "upsert", "future_intent", "feed_network_future_goal",
                "corporate_feed", text,
            ),
            turn_id="turn-save",
        )
        restored = load_builder_session(self.project_path)
        self.assertEqual(restored.memory, self.session.memory)

        planner = Planner(step(status="finish", message="Current state described."))
        execute_builder_turn(
            self.project_path,
            restored,
            "What is current, and what is only a future goal?",
            planner=planner,
            turn_id="turn-context",
        )
        request = planner.requests[0]
        self.assertEqual(request["current_design"]["excitation"]["strategy"], "single_element_feed")
        intent = request["project_memory"]["project_intent"]
        self.assertEqual(intent[0]["value"], "corporate_feed")
        self.assertIn("supplied separately", request["project_memory"]["context_separation"]["canonical_current_state"])

    def test_legacy_memory_without_semantic_fields_loads_unchanged(self):
        save_builder_session(self.project_path, self.session)
        path = self.project_path / "design" / "project_memory.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "canonical_ref": {
                "design_id": self.session.design.design_id,
                "revision": self.session.design.revision,
            },
            "requirements": [{
                "item_id": "legacy-requirement",
                "key": "target_frequency",
                "value": 2.45,
                "unit": "GHz",
                "status": "active",
                "source_turn_id": "legacy-turn",
                "design_revision": 0,
            }],
            "decisions": [], "assumptions": [], "limitations": [],
            "open_questions": [], "important_changes": [], "recent_context": [],
        }), encoding="utf-8")
        restored = load_builder_session(self.project_path)
        item = restored.memory.requirements[0]
        self.assertEqual(item.source, "deterministic")
        self.assertIsNone(item.semantic_kind)
        self.assertEqual(item.value, 2.45)

    def test_contract_schema_guidance_and_terminal_only_rule(self):
        manifest = create_default_agent().capability_manifest(self.session.design)
        exchange = build_agent_step_exchange(
            instruction="I eventually want RHCP.",
            current_design=self.session.design.to_dict(),
            capability_manifest=manifest,
            remaining_budgets=AgentLoopBudgets(),
        )
        schema = exchange.schema["properties"]["memory_proposals"]
        self.assertEqual(schema["maxItems"], 6)
        self.assertIn("exact nonempty substring", exchange.system_instruction)
        self.assertIn("never substitute true or false", exchange.system_instruction)
        value_types = schema["items"]["properties"]["value"]["type"]
        self.assertNotIn("boolean", value_types)
        parsed = parse_agent_step(
            step(
                status="finish",
                proposals=(proposal(
                    "upsert", "goal", "target_polarization", "RHCP",
                    "I eventually want RHCP.",
                ),),
            ).to_dict(),
            callable_tool_names=(),
        )
        self.assertEqual(parsed.memory_proposals[0].key, "target_polarization")
        with self.assertRaisesRegex(CapabilityError, "boolean placeholders"):
            parse_agent_step(
                step(
                    status="finish",
                    proposals=(proposal(
                        "upsert", "goal", "target_polarization", True,
                        "I eventually want RHCP.",
                    ),),
                ).to_dict(),
                callable_tool_names=(),
            )
        with self.assertRaisesRegex(ValueError, "terminal"):
            step(
                ("parameter.set", {"key": "patch_width_mm", "value": 40}),
                proposals=(proposal(
                    "upsert", "goal", "target_polarization", "RHCP",
                    "I eventually want RHCP.",
                ),),
            )

    def test_four_turn_scenario_keeps_future_goals_out_of_current_design(self):
        first = "I eventually want RHCP, but use linear polarization for now."
        self.run_terminal(
            first,
            proposal("upsert", "goal", "target_polarization", "RHCP", "I eventually want RHCP"),
            proposal("upsert", "preference", "current_polarization_preference", "linear", "use linear polarization for now"),
            turn_id="scenario-1",
        )
        execute_builder_turn(
            self.project_path,
            self.session,
            "Make this a 2x3 independently excited array.",
            planner=Planner(
                step(
                    ("parameter.set", {"key": "array_rows", "value": 2}),
                    ("parameter.set", {"key": "array_columns", "value": 3}),
                    ("excitation.set_strategy", {"strategy": "independent_ports"}),
                ),
                step(status="finish", message="2x3 independent array complete."),
            ),
            turn_id="scenario-2",
        )
        third = "Corporate feed can come later."
        self.run_terminal(
            third,
            proposal(
                "upsert", "future_intent", "feed_network_future_goal",
                "corporate_feed", third,
            ),
            turn_id="scenario-3",
        )
        planner = Planner(step(status="finish", message="Goals listed."))
        execute_builder_turn(
            self.project_path,
            self.session,
            "What are the design goals I've given you?",
            planner=planner,
            turn_id="scenario-4",
        )
        request = planner.requests[0]
        self.assertEqual(request["current_design"]["array"]["rows"], 2)
        self.assertEqual(request["current_design"]["array"]["columns"], 3)
        self.assertEqual(request["current_design"]["excitation"]["strategy"], "independent_ports")
        intent = {item["key"]: item["value"] for item in request["project_memory"]["project_intent"]}
        self.assertEqual(intent["target_polarization"], "rhcp")
        self.assertEqual(intent["feed_network_future_goal"], "corporate_feed")


if __name__ == "__main__":
    unittest.main()
