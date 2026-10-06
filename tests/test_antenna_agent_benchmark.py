import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.antenna_agent_benchmark import (
    BenchmarkDefinitionError,
    BenchmarkRunner,
    BenchmarkSelection,
    DEFAULT_DEFINITION_PATH,
    FAILURE_TAXONOMY,
    ReplayExecutor,
    evaluate_case,
    load_benchmark,
    markdown_summary,
    write_run_outputs,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def successful_observation(case):
    acceptance = case["acceptance"]
    changed = acceptance["mutation"] == "required"
    warning_count = int(acceptance["engineering"].get("minimum_warning_count", 0))
    warnings = [f"warning-{index + 1}" for index in range(warning_count)]
    canonical = {}
    for criterion in acceptance["canonical_facts"]:
        if criterion["op"] in {"eq", "gte", "lte", "contains"}:
            canonical[criterion["key"]] = criterion["value"]
        elif criterion["op"] == "contains_all":
            canonical[criterion["key"]] = criterion["value"]
        elif criterion["op"] == "exists":
            canonical[criterion["key"]] = "present"
        elif criterion["op"] == "absent":
            canonical[criterion["key"]] = None
        elif criterion["op"] == "ne":
            canonical[criterion["key"]] = object()
    memory_items = [dict(item) for item in acceptance["memory"].get("required_items", [])]
    for item in memory_items:
        item.setdefault("source", "user_semantic")
    terminal = acceptance["allowed_terminal_outcomes"][0]
    failure_signals = []
    provider_failure = terminal == "provider_error"
    if provider_failure:
        failure_signals.append("provider_api_failure")
    return {
        "case_id": case["case_id"],
        "provider": "mock",
        "model": "deterministic",
        "initial_semantic_hash": HASH_A,
        "final_semantic_hash": HASH_B if changed else HASH_A,
        "initial_memory_hash": HASH_A,
        "final_memory_hash": HASH_C if memory_items else HASH_A,
        "engineering_report_hashes": [HASH_C],
        "trajectory_reference": f"mock://{case['case_id']}",
        "terminal_result": {"outcome": terminal},
        "terminal_outcome": terminal,
        "task_completed": acceptance["completion"],
        "canonical_facts": canonical,
        "selected_capabilities": acceptance["required_capabilities"],
        "hallucinated_capabilities": [],
        "engineering_warning_ids": warnings,
        "acknowledged_warning_ids": warnings,
        "deferred_warning_ids": [],
        "warnings_mentioned_in_prose": [],
        "analysis_tools": acceptance["analysis"].get("required_tools", []),
        "unsupported_model_claims": [],
        "memory_items": memory_items,
        "schema_agent_step_failures": 0,
        "provider_api_failure": provider_failure,
        "execution_rejections": 0,
        "decision_iterations": len(case["turns"]),
        "design_action_batches": int(changed),
        "analysis_batches": int(bool(acceptance["analysis"].get("required_tools"))),
        "failure_signals": failure_signals,
    }


class AntennaAgentBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.definition = load_benchmark()
        cls.by_id = {case["case_id"]: case for case in cls.definition["cases"]}

    def test_frozen_v1_has_thirty_unique_cases_and_all_categories(self):
        self.assertEqual(self.definition["benchmark_version"], "antenna-agent-unseen-v1")
        self.assertEqual(len(self.definition["cases"]), 30)
        self.assertEqual(len(self.by_id), 30)
        self.assertEqual({case["category"] for case in self.definition["cases"]}, set("ABCDEFGH"))
        digest = hashlib.sha256(DEFAULT_DEFINITION_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, "281efa92743b03c69f2a16ad3ee8d6d01dafd339541e43cbcd0ac675ee86b868")

    def test_cases_use_observable_acceptance_and_do_not_prescribe_call_sequences(self):
        for case in self.definition["cases"]:
            acceptance = case["acceptance"]
            self.assertNotIn("tool_calls", acceptance)
            self.assertNotIn("exact_plan", acceptance)
            self.assertIn("canonical_facts", acceptance)
            self.assertIn("allowed_terminal_outcomes", acceptance)
            self.assertIn("rollback_required", acceptance)

    def test_full_mocked_suite_exercises_acceptance_logic(self):
        observations = {
            case["case_id"]: successful_observation(case)
            for case in self.definition["cases"]
        }
        run = BenchmarkRunner(self.definition, ReplayExecutor(observations)).run(
            provider="mock", model="deterministic"
        )
        self.assertEqual(run["dimension_totals"]["cases"], 30)
        self.assertTrue(all(item["acceptance_checks_passed"] for item in run["results"]))
        self.assertEqual(run["dimension_totals"]["provider_api_failures"], 1)

    def test_case_and_category_selection(self):
        observations = {
            case["case_id"]: successful_observation(case)
            for case in self.definition["cases"]
        }
        runner = BenchmarkRunner(self.definition, ReplayExecutor(observations))
        one = runner.run(
            provider="mock", model="deterministic", selection=BenchmarkSelection(case_id="F03")
        )
        category = runner.run(
            provider="mock", model="deterministic", selection=BenchmarkSelection(category="G")
        )
        self.assertEqual([item["case_id"] for item in one["results"]], ["F03"])
        self.assertEqual(len(category["results"]), 3)
        self.assertTrue(all(item["category"] == "G" for item in category["results"]))

    def test_provider_failure_is_separate_from_agent_dimensions(self):
        case = self.by_id["H03"]
        observation = successful_observation(case)
        result = evaluate_case(case, observation)
        self.assertTrue(result["provider_api_failure"])
        self.assertIn("provider_api_failure", result["failure_classifications"])
        self.assertTrue(result["transaction_rollback_correct"])
        self.assertEqual(result["task_completed"], "no")

    def test_schema_and_planner_failures_remain_distinct(self):
        case = self.by_id["A01"]
        observation = successful_observation(case)
        observation["schema_agent_step_failures"] = 1
        observation["failure_signals"] = ["planner_reasoning_failure"]
        result = evaluate_case(case, observation)
        self.assertIn("planner_reasoning_failure", result["failure_classifications"])
        self.assertIn("schema_structured_output_failure", result["failure_classifications"])
        self.assertFalse(result["provider_api_failure"])

    def test_grounding_claim_failure_is_separate_from_structured_analysis(self):
        case = self.by_id["F04"]
        observation = successful_observation(case)
        observation["unsupported_model_claims"] = ["The half-wave estimate proves resonance."]
        result = evaluate_case(case, observation)
        self.assertTrue(result["required_tool_capability_selected"])
        self.assertFalse(result["analytical_result_grounded_correctly"])
        self.assertEqual(len(result["unsupported_model_claims"]), 1)

    def test_structured_disposition_and_prose_warning_are_scored_separately(self):
        case = self.by_id["E01"]
        observation = successful_observation(case)
        result = evaluate_case(case, observation)
        self.assertTrue(result["engineering_findings_accounted_for"])
        self.assertEqual(result["presentation_warning_omissions"], ["warning-1"])
        self.assertTrue(result["acceptance_checks_passed"])

    def test_semantic_memory_history_matches_status_and_value(self):
        case = self.by_id["G02"]
        result = evaluate_case(case, successful_observation(case))
        self.assertTrue(result["semantic_memory_correct"])
        self.assertEqual(len(result["memory_fact_results"]), 2)

    def test_canonical_rollback_does_not_require_context_memory_rollback(self):
        case = self.by_id["H01"]
        observation = successful_observation(case)
        observation["final_memory_hash"] = HASH_B
        result = evaluate_case(case, observation)
        self.assertTrue(result["transaction_rollback_correct"])
        self.assertEqual(result["initial_semantic_hash"], result["final_semantic_hash"])

    def test_invalid_failure_classification_is_rejected(self):
        case = self.by_id["A01"]
        observation = successful_observation(case)
        observation["failure_signals"] = ["made_up_failure"]
        with self.assertRaises(BenchmarkDefinitionError):
            evaluate_case(case, observation)
        self.assertEqual(len(FAILURE_TAXONOMY), 9)

    def test_json_and_markdown_outputs_are_deterministic(self):
        case = self.by_id["F01"]
        run = BenchmarkRunner(
            self.definition,
            ReplayExecutor({"F01": successful_observation(case)}),
        ).run(
            provider="mock", model="deterministic", selection=BenchmarkSelection(case_id="F01")
        )
        first = markdown_summary(run)
        second = markdown_summary(run)
        self.assertEqual(first, second)
        self.assertIn("No single quality score", first)
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "result.json"
            markdown_path = Path(directory) / "result.md"
            write_run_outputs(run, json_path=json_path, markdown_path=markdown_path)
            restored = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(restored["definition_hash"], run["definition_hash"])
            self.assertEqual(markdown_path.read_text(encoding="utf-8"), first)


if __name__ == "__main__":
    unittest.main()
