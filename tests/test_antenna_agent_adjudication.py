from __future__ import annotations

import unittest
from pathlib import Path

from benchmarks.antenna_agent_adjudication import (
    EVALUATOR_VERSION,
    SOURCE_BENCHMARK_SHA256,
    adjudicate_run,
    canonical_facts_v2,
    corrected_case,
    sha256_file,
)
from studio.antenna_builder import load_builder_session


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "benchmarks" / "results" / "v1" / "20260924_stage5b_hosted_v1"
DEFINITION = ROOT / "benchmarks" / "antenna_agent_v1.json"


class AntennaAgentAdjudicationTests(unittest.TestCase):
    def test_source_benchmark_is_still_frozen(self):
        self.assertEqual(sha256_file(DEFINITION), SOURCE_BENCHMARK_SHA256)

    def test_exact_aliases_follow_production_identity(self):
        case = {
            "acceptance": {
                "canonical_facts": [
                    {"key": "family", "op": "eq", "value": "inset_patch"},
                    {"key": "material", "op": "eq", "value": "Rogers 4003C"},
                    {"key": "feature_target_role", "op": "eq", "value": "radiating_patch"},
                ],
                "required_capabilities": ["analysis.array_spacing"],
                "forbidden_capabilities": [],
                "analysis": {"required_tools": ["analysis.dipole_electrical_length"]},
                "clarification": "forbidden",
                "allowed_terminal_outcomes": ["finished"],
            }
        }
        fixed = corrected_case(case)
        criteria = fixed["acceptance"]["canonical_facts"]
        self.assertEqual(criteria[0]["value"], "rectangular_inset_patch")
        self.assertEqual(criteria[1]["value"], "Rogers RO4003C")
        self.assertEqual(criteria[2]["value"], "radiating_patch_conductor")
        self.assertEqual(fixed["acceptance"]["required_capabilities"], ["engineering.array_spacing"])
        self.assertEqual(fixed["acceptance"]["analysis"]["required_tools"], ["engineering.dipole_baseline"])

    def test_composition_extraction_counts_physical_cutters(self):
        design = load_builder_session(RUN_ROOT / "gemini" / "B03").design
        facts = canonical_facts_v2(design)
        self.assertEqual(facts["circle_subtraction_count"], 2)
        self.assertTrue(facts["distinct_feature_offsets"])

    def test_parameter_backed_feature_values_are_resolved(self):
        circle = load_builder_session(RUN_ROOT / "gemini" / "C01").design
        rectangle = load_builder_session(RUN_ROOT / "gemini" / "C03").design
        self.assertAlmostEqual(canonical_facts_v2(circle)["circle_radius_mm"], 4.25)
        self.assertAlmostEqual(canonical_facts_v2(rectangle)["rectangle_width_mm"], 8.0)

    def test_adjudication_uses_v2_and_preserves_provider_failures(self):
        payload = adjudicate_run(RUN_ROOT, DEFINITION)
        self.assertEqual(payload["evaluator_version"], EVALUATOR_VERSION)
        records = {(item["provider"], item["case_id"]): item for item in payload["adjudications"]}
        self.assertEqual(records[("gemini", "A03")]["adjudication"], "benchmark_contract_mismatch")
        self.assertEqual(records[("gemini", "C01")]["adjudication"], "evaluator_extraction_bug")
        self.assertEqual(records[("openrouter", "C01")]["adjudication"], "confirmed_schema_failure")
        self.assertEqual(records[("openrouter", "F02")]["adjudication"], "confirmed_provider_failure")
        self.assertTrue(all(
            item["transaction_rollback_correct"] for item in payload["adjudications"]
        ))


if __name__ == "__main__":
    unittest.main()
