import unittest

from studio.antenna_agent import AgentInstructionError, create_default_agent
from studio.antenna_llm_planner import LLMToolPlan, PlannedToolCall
from studio.antenna_recipes import MATERIAL_ALIASES, MATERIALS, canonical_choice


def plan(*calls):
    return LLMToolPlan(
        "execute",
        "planner choice value test",
        tuple(PlannedToolCall(name, arguments) for name, arguments in calls),
    )


class CanonicalChoiceTests(unittest.TestCase):
    def test_matches_ignore_case_spacing_and_known_aliases(self):
        choices = tuple(MATERIALS)
        self.assertEqual(canonical_choice("FR4", choices, MATERIAL_ALIASES), "FR4")
        self.assertEqual(canonical_choice("fr4", choices, MATERIAL_ALIASES), "FR4")
        self.assertEqual(canonical_choice("  Fr4 ", choices, MATERIAL_ALIASES), "FR4")
        self.assertEqual(canonical_choice("FR-4", choices, MATERIAL_ALIASES), "FR4")
        self.assertEqual(canonical_choice("rogers  ro4003c", choices, MATERIAL_ALIASES), "Rogers RO4003C")
        self.assertEqual(canonical_choice("RT5880", choices, MATERIAL_ALIASES), "Rogers RT5880")

    def test_unknown_or_non_string_values_are_not_matched(self):
        choices = tuple(MATERIALS)
        self.assertIsNone(canonical_choice("Rogers RO4350B", choices, MATERIAL_ALIASES))
        self.assertIsNone(canonical_choice(4.4, choices, MATERIAL_ALIASES))
        self.assertIsNone(canonical_choice("FR-4", choices))


class PlannerMaterialValueTests(unittest.TestCase):
    def setUp(self):
        self.agent = create_default_agent()

    def _set_material(self, design, value):
        return self.agent.execute_llm_plan(
            design, plan(("parameter.set", {"key": "material", "value": value})),
        ).design

    def test_common_material_spellings_set_the_installed_material(self):
        for family in ("inset_patch", "circular_patch"):
            design = self.agent.create_design(family)
            for spelling, expected in (
                ("fr4", "FR4"),
                ("FR-4", "FR4"),
                ("rt5880", "Rogers RT5880"),
                ("Rogers RO4003C", "Rogers RO4003C"),
            ):
                with self.subTest(family=family, spelling=spelling):
                    updated = self._set_material(design, spelling)
                    self.assertEqual(updated.metadata_map()["substrate_material"], expected)

    def test_unknown_material_is_a_recoverable_instruction_error_listing_choices(self):
        design = self.agent.create_design("inset_patch")
        with self.assertRaises(AgentInstructionError) as caught:
            self._set_material(design, "Rogers RO4350B")
        message = str(caught.exception)
        for name in MATERIALS:
            self.assertIn(name, message)
        self.assertNotIsInstance(caught.exception, KeyError)


if __name__ == "__main__":
    unittest.main()
