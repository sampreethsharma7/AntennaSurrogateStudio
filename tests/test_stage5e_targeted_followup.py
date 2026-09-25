import unittest

from benchmarks.stage5e_targeted_followup import (
    FAILED_CASES,
    replay_provider_recorded_outputs,
)


class Stage5ETargetedFollowupTests(unittest.TestCase):
    def test_targeted_sets_are_exactly_original_provider_failures(self):
        self.assertEqual(FAILED_CASES["gemini"], ("P03", "F02", "U02", "M01"))
        self.assertEqual(
            FAILED_CASES["openrouter"],
            ("P01", "P03", "F01", "F02", "U01", "U02", "M01"),
        )

    def test_recorded_outputs_replay_all_frozen_turns(self):
        gemini = replay_provider_recorded_outputs("gemini")
        nemotron = replay_provider_recorded_outputs("openrouter")
        self.assertEqual((gemini["replayed_case_count"], gemini["replayed_turn_count"]), (11, 17))
        self.assertEqual((nemotron["replayed_case_count"], nemotron["replayed_turn_count"]), (11, 17))

    def test_recorded_replay_fixes_only_grounded_alias_coverage(self):
        gemini = replay_provider_recorded_outputs("gemini")
        results = {item["case_id"]: item for item in gemini["results"]}
        self.assertTrue(results["P03"]["evaluation"]["passed"])
        self.assertTrue(results["F02"]["evaluation"]["passed"])
        self.assertTrue(results["M01"]["evaluation"]["passed"])
        self.assertFalse(results["U02"]["evaluation"]["passed"])
        self.assertEqual(gemini["counts"]["passed"], 10)

    def test_boolean_recorded_outputs_remain_contract_failures(self):
        nemotron = replay_provider_recorded_outputs("openrouter")
        self.assertEqual(len(nemotron["boolean_contract_violations"]), 7)
        self.assertEqual(nemotron["counts"]["passed"], 4)
        reasons = {
            rejected["reason"]
            for result in nemotron["results"]
            for turn in result["turn_results"]
            for rejected in turn["semantic_memory"]["rejected"]
        }
        self.assertIn("boolean_placeholder_not_allowed", reasons)


if __name__ == "__main__":
    unittest.main()
