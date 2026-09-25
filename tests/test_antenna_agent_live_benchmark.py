import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.antenna_agent_benchmark import (
    BenchmarkRunner, BenchmarkSelection, ReplayExecutor, load_benchmark,
)
from benchmarks.antenna_agent_live import (
    EXPECTED_BENCHMARK_SHA256,
    LiveBenchmarkExecutor,
    build_fixture,
    canonical_facts,
    configured_credentials,
    cross_provider_markdown,
    failure_inventory,
    new_manifest,
    verify_frozen_benchmark,
)
from studio.antenna_engineering_checks import run_engineering_checks
from studio.antenna_llm_planner import AgentStep
from studio.antenna_tools import CapabilityError


class FinishPlanner:
    backend_id = "mock"
    model = "mock"
    last_run_metadata = {"schema_repairs": 0}

    def plan_agent_step(self, **_request):
        return AgentStep("finish", "The unchanged fixture is complete.")


class ProviderErrorPlanner(FinishPlanner):
    def plan_agent_step(self, **_request):
        raise CapabilityError("simulated provider transport failure")


class LiveAntennaBenchmarkTests(unittest.TestCase):
    def test_frozen_sha_is_verified_before_live_execution(self):
        self.assertEqual(verify_frozen_benchmark(), EXPECTED_BENCHMARK_SHA256)

    def test_all_fixture_definitions_build_in_isolation(self):
        definition = load_benchmark()
        for fixture_id in definition["fixtures"]:
            session = build_fixture(fixture_id)
            if fixture_id == "empty_project":
                self.assertIsNone(session.design)
            else:
                self.assertIsNotNone(session.design)
                self.assertEqual(session.design.design_id, f"benchmark_{fixture_id}")

    def test_warning_fixtures_are_deterministic(self):
        overlap = run_engineering_checks(build_fixture("warning_patch_overlap").design)
        heavy = run_engineering_checks(build_fixture("warning_patch_heavy").design)
        self.assertGreaterEqual(sum(item.severity == "warning" for item in overlap.findings), 1)
        self.assertGreaterEqual(sum(item.severity == "warning" for item in heavy.findings), 2)

    def test_canonical_fact_capture_uses_final_design_state(self):
        facts = canonical_facts(build_fixture("inset_patch_2x3_decorated").design)
        self.assertEqual((facts["array_rows"], facts["array_columns"]), (2, 3))
        self.assertEqual(facts["excitation_strategy"], "independent_ports")
        self.assertEqual(facts["physical_port_count"], 6)
        self.assertEqual(facts["physical_circle_subtraction_count"], 6)
        self.assertEqual(facts["logical_composed_feature_count"], 1)

    def test_provider_error_retry_reuses_baseline_and_records_both_attempts(self):
        case = {
            "case_id": "T01",
            "initial_fixture": "inset_patch_clean",
            "turns": ["Inspect the unchanged fixture."],
        }
        planners = iter((ProviderErrorPlanner(), FinishPlanner()))
        with tempfile.TemporaryDirectory() as directory, patch(
            "benchmarks.antenna_agent_live._planner_factory",
            side_effect=lambda *_args, **_kwargs: next(planners),
        ):
            executor = LiveBenchmarkExecutor(
                Path(directory), Path(directory) / ".env", retry_delay=lambda _seconds: None
            )
            observation = executor.execute(case, provider="gemini", model="mock")
        self.assertFalse(observation["provider_api_failure"])
        self.assertEqual(observation["terminal_outcome"], "finished")
        provenance = observation["contract_provenance"]
        self.assertEqual(provenance["agent_step_contract_version"], 2)
        self.assertEqual(provenance["benchmark_sha256"], EXPECTED_BENCHMARK_SHA256)
        self.assertEqual(len(provenance["contract_fingerprints"]), 4)
        self.assertTrue(provenance["captured_at_utc"])
        self.assertEqual(len(observation["attempts"]), 2)
        self.assertEqual(observation["attempts"][0]["terminal_outcome"], "provider_error")
        self.assertEqual(
            observation["initial_semantic_hash"], observation["final_semantic_hash"]
        )

    def test_exhausted_provider_retry_preserves_design(self):
        case = {
            "case_id": "T02",
            "initial_fixture": "inset_patch_clean",
            "turns": ["Inspect the unchanged fixture."],
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "benchmarks.antenna_agent_live._planner_factory",
            return_value=ProviderErrorPlanner(),
        ):
            executor = LiveBenchmarkExecutor(
                Path(directory), Path(directory) / ".env", retry_delay=lambda _seconds: None
            )
            observation = executor.execute(case, provider="openrouter", model="mock")
        self.assertTrue(observation["provider_api_failure"])
        self.assertEqual(len(observation["attempts"]), 2)
        self.assertEqual(
            observation["initial_semantic_hash"], observation["final_semantic_hash"]
        )
        self.assertIn("provider_api_failure", observation["failure_signals"])

    def test_manifest_records_frozen_configuration_without_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = new_manifest(Path(directory), "test-run")
        serialized = json.dumps(manifest).casefold()
        self.assertEqual(manifest["benchmark_sha256"], EXPECTED_BENCHMARK_SHA256)
        self.assertEqual(manifest["retry_policy"]["maximum_attempts"], 2)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("secret-one", serialized)
        self.assertNotIn("secret-two", serialized)

    def test_credential_probe_returns_presence_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("GEMINI_API_KEY=secret-one\nOPEN_ROUTER_API_KEY=secret-two\n", encoding="utf-8")
            status = configured_credentials(path)
        self.assertEqual(status, {"gemini": True, "openrouter": True})
        self.assertNotIn("secret", json.dumps(status))

    def test_cross_provider_comparison_excludes_provider_failures(self):
        definition = load_benchmark()
        case = next(item for item in definition["cases"] if item["case_id"] == "A04")
        base = {
            "case_id": "A04", "provider": "gemini", "model": "mock",
            "initial_semantic_hash": "a" * 64, "final_semantic_hash": "a" * 64,
            "initial_memory_hash": "a" * 64, "final_memory_hash": "a" * 64,
            "engineering_report_hashes": [], "trajectory_reference": "mock://A04",
            "terminal_result": {"outcome": "refuse"}, "terminal_outcome": "refuse",
            "task_completed": "no", "canonical_facts": {"design_exists": False},
            "selected_capabilities": [], "hallucinated_capabilities": [],
            "engineering_warning_ids": [], "acknowledged_warning_ids": [],
            "deferred_warning_ids": [], "warnings_mentioned_in_prose": [],
            "analysis_tools": [], "unsupported_model_claims": [], "memory_items": [],
            "schema_agent_step_failures": 0, "provider_api_failure": False,
            "execution_rejections": 0, "decision_iterations": 1,
            "design_action_batches": 0, "analysis_batches": 0, "failure_signals": [],
        }
        gemini = BenchmarkRunner(definition, ReplayExecutor({"A04": base})).run(
            provider="gemini", model="mock",
            selection=BenchmarkSelection(case_id="A04"),
        )
        failed = dict(base)
        failed.update({
            "provider": "openrouter", "terminal_outcome": "provider_error",
            "terminal_result": {"outcome": "provider_error"}, "provider_api_failure": True,
            "failure_signals": ["provider_api_failure"],
        })
        nemotron = BenchmarkRunner(definition, ReplayExecutor({"A04": failed})).run(
            provider="openrouter", model="mock",
            selection=BenchmarkSelection(case_id="A04"),
        )
        report = cross_provider_markdown(gemini, nemotron)
        self.assertIn("Paired evaluable cases: 0", report)
        inventory = failure_inventory({"gemini": gemini, "nemotron": nemotron})
        self.assertEqual(inventory["taxonomy_totals"]["provider_api_failure"], 1)


if __name__ == "__main__":
    unittest.main()
