import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.stage5e_memory import (
    EXPECTED_TEST_SET_SHA256,
    _start_or_verify_live_freeze,
    evaluate_case,
    finalize_outputs,
    load_frozen_definition,
    prepare_outputs,
    provider_counts,
    replay_recorded_stage5b,
    run_live_case,
    run_live_provider,
)
from studio.antenna_builder import ProjectMemory, apply_semantic_memory_proposals
from studio.antenna_llm_planner import AgentStep, SemanticMemoryProposal


def proposal(action, kind, key, value, quote, *, unit=None, reference_id=None):
    return SemanticMemoryProposal(
        action=action,
        semantic_kind=kind,
        key=key,
        value=value,
        evidence_quote=quote,
        unit=unit,
        reference_id=reference_id,
    )


class Stage5EMemoryValidationTests(unittest.TestCase):
    def setUp(self):
        self.definition = load_frozen_definition()
        self.cases = {case["case_id"]: case for case in self.definition["cases"]}

    def _apply_turn(self, memory, instruction, proposals, turn_id):
        updated, audit = apply_semantic_memory_proposals(
            memory,
            tuple(proposals),
            instruction=instruction,
            planner_calls=(),
            design=None,
            turn_id=turn_id,
        )
        return updated, {
            "instruction": instruction,
            "semantic_memory": audit,
            "memory": updated.to_dict(),
        }

    @staticmethod
    def _observation(memory, turns, *, excitation="single_element_feed", response=""):
        return {
            "turn_results": turns,
            "final_memory": memory.to_dict(),
            "planner_context": memory.to_planner_dict(),
            "final_canonical_facts": {"excitation_strategy": excitation},
            "final_user_facing_response": response,
        }

    def test_frozen_definition_has_eleven_cases_seventeen_turns_and_exact_sha(self):
        self.assertEqual(self.definition["case_count"], 11)
        self.assertEqual(self.definition["prompt_turn_count"], 17)
        self.assertEqual(
            set(self.cases),
            {"P01", "P02", "P03", "B01", "F01", "F02", "U01", "U02", "A01", "A02", "M01"},
        )
        self.assertEqual(len(EXPECTED_TEST_SET_SHA256), 64)

    def test_polarization_expectations_keep_generic_rhcp_and_lhcp_distinct(self):
        values = {
            case_id: self.cases[case_id]["expected"]["required_active"][0]["value"]
            for case_id in ("P01", "P02", "P03")
        }
        self.assertEqual(values["P01"], "circular_polarization")
        self.assertEqual(values["P02"], "rhcp")
        self.assertEqual(values["P03"], "lhcp")
        self.assertEqual(len(set(values.values())), 3)

    def test_recorded_g01_g02_g03_replay_preserves_canonical_results_and_provenance(self):
        replay = replay_recorded_stage5b()
        g01 = replay["cases"]["G01"]["semantic_items"]
        self.assertEqual((g01[0]["key"], g01[0]["value"]), ("target_polarization", "circular_polarization"))
        self.assertEqual((g01[0]["proposed_key"], g01[0]["proposed_value"]), ("target_polarization", "circular"))
        self.assertEqual(g01[0]["evidence_quote"], "Long term I want circular polarization")

        g02 = replay["cases"]["G02"]["semantic_items"]
        active = [item for item in g02 if item["status"] == "active"]
        self.assertEqual(len(active), 1)
        self.assertEqual(
            (active[0]["key"], active[0]["value"], active[0]["unit"], active[0]["constraint_operator"]),
            ("board_width_limit", 90.0, "mm", "max"),
        )
        self.assertEqual(sum(item["status"] == "superseded" for item in g02), 1)
        self.assertTrue(all(item["proposed_key"] == "max_board_width_mm" for item in g02))

        g03 = replay["cases"]["G03"]["semantic_items"]
        self.assertEqual((g03[0]["key"], g03[0]["value"]), ("feed_network_future_goal", "corporate_feed"))
        self.assertEqual(g03[0]["proposed_key"], "corporate_distribution_network")
        self.assertEqual(g03[0]["proposed_value"], "corporate distribution network")
        self.assertIn(g03[0]["evidence_quote"], replay["cases"]["G03"]["turns"][0]["instruction"])
        self.assertEqual(replay["cloud_calls"], 0)

    def test_board_lifecycle_evaluator_checks_supersede_resolve_and_each_turn(self):
        memory = ProjectMemory.empty()
        turns = []
        first = self.cases["B01"]["turns"][0]
        memory, turn = self._apply_turn(
            memory,
            first,
            [proposal("upsert", "constraint", "max_board_width_mm", 75, first, unit="mm")],
            "B01-1",
        )
        turns.append(turn)
        active = next(item for item in memory.requirements if item.source == "user_semantic" and item.status == "active")
        second = self.cases["B01"]["turns"][1]
        memory, turn = self._apply_turn(
            memory,
            second,
            [proposal(
                "supersede", "constraint", "board_width_max", 95, second,
                unit="mm", reference_id=active.item_id,
            )],
            "B01-2",
        )
        turns.append(turn)
        active = next(item for item in memory.requirements if item.source == "user_semantic" and item.status == "active")
        third = self.cases["B01"]["turns"][2]
        memory, turn = self._apply_turn(
            memory,
            third,
            [proposal(
                "resolve", "constraint", "board_width_limit", None, third,
                reference_id=active.item_id,
            )],
            "B01-3",
        )
        turns.append(turn)
        evaluation = evaluate_case(self.cases["B01"], self._observation(memory, turns))
        self.assertTrue(evaluation["passed"], evaluation)
        self.assertTrue(evaluation["dimensions"]["supersede_resolve_correctness"]["passed"])

    def test_unknown_concept_is_preserved_without_registered_mapping(self):
        text = self.cases["U01"]["turns"][0]
        memory, turn = self._apply_turn(
            ProjectMemory.empty(),
            text,
            [proposal("upsert", "priority", "low_visual_profile", "important", text)],
            "U01-1",
        )
        evaluation = evaluate_case(self.cases["U01"], self._observation(memory, [turn]))
        self.assertTrue(evaluation["passed"], evaluation)
        active = memory.to_planner_dict()["project_intent"]
        self.assertEqual(active[0]["key"], "low_visual_profile")

    def test_anti_merge_cases_allow_no_proposal_and_forbid_wrong_concepts(self):
        for case_id in ("A01", "A02"):
            text = self.cases[case_id]["turns"][0]
            memory, turn = self._apply_turn(ProjectMemory.empty(), text, [], f"{case_id}-1")
            evaluation = evaluate_case(self.cases[case_id], self._observation(memory, [turn]))
            self.assertTrue(evaluation["passed"], {case_id: evaluation})

    def test_multi_turn_evaluator_separates_current_excitation_and_future_intent(self):
        case = self.cases["M01"]
        memory = ProjectMemory.empty()
        turns = []
        instructions = case["turns"]
        memory, turn = self._apply_turn(
            memory, instructions[0],
            [proposal("upsert", "goal", "desired_polarization", "RHCP", instructions[0])],
            "M01-1",
        )
        turns.append(turn)
        memory, turn = self._apply_turn(
            memory, instructions[1],
            [proposal("upsert", "constraint", "board_width_max", 85, instructions[1], unit="mm")],
            "M01-2",
        )
        turns.append(turn)
        memory, turn = self._apply_turn(memory, instructions[2], [], "M01-3")
        turns.append(turn)
        memory, turn = self._apply_turn(
            memory, instructions[3],
            [proposal("upsert", "future_intent", "future_feed_network", "Corporate feed", instructions[3])],
            "M01-4",
        )
        turns.append(turn)
        memory, turn = self._apply_turn(memory, instructions[4], [], "M01-5")
        turns.append(turn)
        observation = self._observation(
            memory,
            turns,
            excitation="independent_ports",
            response="Your goals are RHCP, an 85 mm board maximum, and a corporate feed later.",
        )
        evaluation = evaluate_case(case, observation)
        self.assertTrue(evaluation["passed"], evaluation)
        self.assertTrue(evaluation["dimensions"]["current_state_future_intent_separation"]["passed"])
        self.assertTrue(evaluation["dimensions"]["final_conversational_recall"]["passed"])

    def test_provider_counts_keep_provider_failures_separate(self):
        counts = provider_counts([
            {"evaluable": True, "evaluation": {"passed": True}, "provider_api_failure_count": 0},
            {"evaluable": True, "evaluation": {"passed": False}, "provider_api_failure_count": 0},
            {"evaluable": False, "evaluation": {"passed": False}, "provider_api_failure_count": 2},
        ])
        self.assertEqual(
            counts,
            {
                "total_cases": 3,
                "evaluable": 2,
                "passed": 1,
                "failed": 1,
                "provider_failure_cases": 1,
                "provider_failure_attempts": 2,
                "dimension_failure_counts": {},
            },
        )

    def test_live_provider_requires_explicit_authorization_before_any_setup(self):
        with self.assertRaisesRegex(PermissionError, "explicit user authorization"):
            run_live_provider("gemini", authorized=False, run_root=Path("unused"))

    def test_live_case_uses_production_turn_and_objective_evaluator(self):
        case = self.cases["P01"]
        text = case["turns"][0]

        class Planner:
            backend_id = "stage5e-mock"
            model = "mock"
            last_run_metadata = {}

            def plan_agent_step(self, **_request):
                return AgentStep(
                    "finish",
                    "Circular polarization is recorded as a future goal.",
                    memory_proposals=(proposal(
                        "upsert", "future_intent", "desired_polarization", "circular", text,
                    ),),
                )

        root = Path.cwd() / ".stage5e_live_case_test"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir()
        try:
            with patch("benchmarks.stage5e_memory._planner_factory", return_value=Planner()):
                result = run_live_case(
                    case,
                    provider="gemini",
                    model="mock",
                    run_root=root,
                    env_file=Path("unused"),
                )
            self.assertTrue(result["evaluable"])
            self.assertTrue(result["evaluation"]["passed"], result["evaluation"])
            provenance = result["contract_provenance"]
            self.assertEqual(provenance["agent_step_contract_version"], 3)
            self.assertEqual(provenance["benchmark_sha256"], EXPECTED_TEST_SET_SHA256)
            self.assertEqual(len(provenance["contract_fingerprints"]), 4)
            self.assertTrue(provenance["captured_at_utc"])
            self.assertTrue((root / "gemini" / "P01" / "observation.json").exists())
        finally:
            shutil.rmtree(root)

    def test_freeze_guard_rejects_changed_sources(self):
        root = Path.cwd() / ".stage5e_freeze_test"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir()
        try:
            (root / "run_manifest.json").write_text(json.dumps({
                "test_set_sha256": EXPECTED_TEST_SET_SHA256,
                "normalization_source_sha256": "wrong",
                "builder_source_sha256": "wrong",
                "harness_source_sha256": "wrong",
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "freeze violation"):
                _start_or_verify_live_freeze(root, "gemini")
        finally:
            shutil.rmtree(root)

    def test_prepare_writes_deterministic_artifacts_without_provider_results(self):
        root = Path.cwd() / ".stage5e_prepare_test"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir()
        try:
            prepared = prepare_outputs(root)
            self.assertEqual(prepared["manifest"]["cloud_calls_during_prepare"], 0)
            self.assertEqual(prepared["manifest"]["live_status"], "awaiting_explicit_cloud_authorization")
            self.assertTrue((root / "deterministic_g01_g02_g03_replay.json").exists())
            self.assertEqual((root / "test_set_sha256.txt").read_text().strip(), EXPECTED_TEST_SET_SHA256)
            self.assertFalse((root / "gemini_results.json").exists())
            self.assertFalse((root / "nemotron_results.json").exists())
        finally:
            shutil.rmtree(root)

    def test_finalize_separates_provider_absence_and_sets_objective_status(self):
        root = Path.cwd() / ".stage5e_finalize_test"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir()
        try:
            passed = {
                "case_id": "P01", "evaluable": True,
                "evaluation": {"passed": True, "dimensions": {}},
                "provider_api_failure_count": 0,
            }
            unavailable = {
                "case_id": "P01", "evaluable": False,
                "evaluation": {"passed": False, "dimensions": {}, "unevaluable_reason": "provider_api_failure"},
                "provider_api_failure_count": 2,
            }
            gemini = {"results": [passed], "counts": provider_counts([passed])}
            nemotron = {"results": [unavailable], "counts": provider_counts([unavailable])}
            (root / "gemini_results.json").write_text(json.dumps(gemini), encoding="utf-8")
            (root / "nemotron_results.json").write_text(json.dumps(nemotron), encoding="utf-8")
            (root / "run_manifest.json").write_text(json.dumps({"live_status": "captured"}), encoding="utf-8")
            result = finalize_outputs(root)
            self.assertEqual(result["validation_status"], "validated")
            inventory = json.loads((root / "failure_inventory.json").read_text(encoding="utf-8"))
            self.assertTrue(inventory["entries"][0]["provider_api_failure"])
            self.assertTrue((root / "cross_provider_comparison.md").exists())
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
