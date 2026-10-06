"""Element spacing must have exactly one independent variable per mode.

`ElementSpacing` was always `299.792458/FreqGHz*SpacingLambda`, so an LHS
column over frequency would have varied the physical array at the same time and
confounded any surrogate fitted to it. Frequency was therefore barred from
sweeps outright, which also removed the legitimate case: sweeping the operating
point of a fixed piece of hardware.

These tests pin both halves. In the default electrical mode frequency still
moves the array, and is still refused as a sweep variable. In the physical mode
the millimetre spacing is independent, frequency is sweepable, and sweeping it
leaves the geometry alone.
"""

from __future__ import annotations

import json
import unittest

from studio.antenna_agent import create_default_agent
from studio.antenna_builder import (
    AntennaBuilderError,
    AntennaState,
    apply_text_instruction,
    cst_macro,
    lhs_variables_for_state,
    recipe_parameter_definitions,
)
from studio.antenna_llm_planner import LLMToolPlan, PlannedToolCall
from studio.antenna_recipes import (
    SPACING_MODE_FIXED_MM,
    SPACING_MODE_LAMBDA,
    SPEED_OF_LIGHT_MM_GHZ,
)
from studio.cst_antenna_adapter import CSTAdapter


ARRAY_RECIPES = ("inset_patch", "circular_patch", "dipole")


class _StaticPlanner:
    def __init__(self, result):
        self.result = result

    def plan(self, **_kwargs):
        return self.result


def apply_plan(state, *calls):
    plan = LLMToolPlan(
        "execute",
        "test plan",
        tuple(PlannedToolCall(name, arguments) for name, arguments in calls),
    )
    return apply_text_instruction(
        state, "planned instruction", planner=_StaticPlanner(plan)
    )


def build(recipe_id: str, **values):
    agent = create_default_agent()
    built = agent.create_design(recipe_id, {"array_columns": 3, **values})
    return getattr(built, "design", built)


def parameters(design) -> dict:
    return {parameter.key: parameter for parameter in design.parameters}


def set_value(design, key, value):
    agent = create_default_agent()
    result = agent.execute_llm_plan(
        design,
        LLMToolPlan(
            status="execute",
            message="",
            calls=(PlannedToolCall("parameter.set", {"key": key, "value": value}),),
        ),
    )
    return getattr(result, "design", result)


class ElectricalSpacingModeTests(unittest.TestCase):
    """The default mode keeps the historical, frequency-locked behaviour."""

    def test_default_mode_is_electrical(self):
        for recipe_id in ARRAY_RECIPES:
            with self.subTest(recipe_id=recipe_id):
                design = build(recipe_id)
                self.assertEqual(
                    design.metadata_map().get("spacing_mode"), SPACING_MODE_LAMBDA
                )

    def test_frequency_still_moves_the_array(self):
        design = build("inset_patch")
        before = parameters(design)["element_spacing_mm"].value
        moved = set_value(design, "frequency_ghz", 5.0)
        after = parameters(moved)["element_spacing_mm"].value
        self.assertAlmostEqual(before, SPEED_OF_LIGHT_MM_GHZ / 2.45 * 0.55, places=3)
        self.assertAlmostEqual(after, SPEED_OF_LIGHT_MM_GHZ / 5.0 * 0.55, places=3)
        self.assertAlmostEqual(moved.array.spacing_mm, after, places=6)

    def test_frequency_is_refused_as_a_sweep_variable(self):
        for recipe_id in ARRAY_RECIPES:
            with self.subTest(recipe_id=recipe_id):
                design = build(recipe_id)
                with self.assertRaisesRegex(AntennaBuilderError, "not a sweepable"):
                    lhs_variables_for_state(design, ["FreqGHz"])

    def test_electrical_spacing_stays_sweepable(self):
        design = build("inset_patch")
        variable = lhs_variables_for_state(design, ["SpacingLambda"])[0]
        self.assertEqual(variable.name, "SpacingLambda")
        self.assertLess(variable.minimum, variable.maximum)


class PhysicalSpacingModeTests(unittest.TestCase):
    """The physical mode is what makes a frequency sweep unambiguous."""

    def test_frequency_is_sweepable(self):
        for recipe_id in ARRAY_RECIPES:
            with self.subTest(recipe_id=recipe_id):
                design = build(recipe_id, spacing_mode=SPACING_MODE_FIXED_MM)
                variable = lhs_variables_for_state(design, ["FreqGHz"])[0]
                self.assertEqual(variable.name, "FreqGHz")
                self.assertAlmostEqual(variable.minimum, 2.45 * 0.9, places=6)
                self.assertAlmostEqual(variable.maximum, 2.45 * 1.1, places=6)

    def test_sweeping_frequency_does_not_move_the_array(self):
        design = build("inset_patch", spacing_mode=SPACING_MODE_FIXED_MM)
        spacing = parameters(design)["element_spacing_mm"].value
        positions = [port.element_index for port in design.ports]
        for frequency in (2.205, 2.45, 2.695, 5.0):
            with self.subTest(frequency=frequency):
                moved = set_value(design, "frequency_ghz", frequency)
                self.assertAlmostEqual(
                    parameters(moved)["element_spacing_mm"].value, spacing, places=9
                )
                self.assertAlmostEqual(moved.array.spacing_mm, spacing, places=9)
                self.assertEqual(
                    [port.element_index for port in moved.ports], positions
                )

    def test_physical_spacing_is_the_independent_variable(self):
        design = build("inset_patch", spacing_mode=SPACING_MODE_FIXED_MM)
        variable = lhs_variables_for_state(design, ["ElementSpacing"])[0]
        self.assertEqual(variable.name, "ElementSpacing")
        edited = set_value(design, "element_spacing_mm", 40.0)
        self.assertAlmostEqual(edited.array.spacing_mm, 40.0, places=9)
        # And it survives a later frequency change.
        after = set_value(edited, "frequency_ghz", 5.0)
        self.assertAlmostEqual(after.array.spacing_mm, 40.0, places=9)

    def test_electrical_spacing_is_reported_not_assumed(self):
        design = build("inset_patch", spacing_mode=SPACING_MODE_FIXED_MM)
        spacing = parameters(design)["element_spacing_mm"].value
        for frequency in (2.45, 5.0):
            with self.subTest(frequency=frequency):
                moved = set_value(design, "frequency_ghz", frequency)
                reported = parameters(moved)["element_spacing_lambda"]
                self.assertFalse(reported.editable)
                self.assertFalse(reported.sweepable)
                self.assertAlmostEqual(
                    reported.value,
                    spacing * frequency / SPEED_OF_LIGHT_MM_GHZ,
                    places=5,
                )

    def test_mode_survives_later_edits(self):
        design = build("inset_patch", spacing_mode=SPACING_MODE_FIXED_MM)
        design = set_value(design, "frequency_ghz", 3.0)
        design = set_value(design, "patch_width_mm", 36.0)
        self.assertEqual(
            design.metadata_map().get("spacing_mode"), SPACING_MODE_FIXED_MM
        )
        self.assertTrue(parameters(design)["frequency_ghz"].sweepable)


class DependentSpacingEditTests(unittest.TestCase):
    """Editing the derived variable must fail loudly, not silently revert."""

    def test_electrical_mode_refuses_a_millimetre_edit(self):
        design = build("inset_patch")
        with self.assertRaisesRegex(Exception, "derived rather than set"):
            set_value(design, "element_spacing_mm", 40.0)

    def test_physical_mode_refuses_a_wavelength_edit(self):
        design = build("inset_patch", spacing_mode=SPACING_MODE_FIXED_MM)
        with self.assertRaisesRegex(Exception, "reported rather than set"):
            set_value(design, "element_spacing_lambda", 0.7)

    def test_each_mode_accepts_its_own_independent_variable(self):
        electrical = set_value(build("inset_patch"), "element_spacing_lambda", 0.7)
        self.assertAlmostEqual(
            electrical.array.spacing_mm, SPEED_OF_LIGHT_MM_GHZ / 2.45 * 0.7, places=3
        )
        physical = set_value(
            build("inset_patch", spacing_mode=SPACING_MODE_FIXED_MM),
            "element_spacing_mm",
            40.0,
        )
        self.assertAlmostEqual(physical.array.spacing_mm, 40.0, places=9)


class ParameterTableTests(unittest.TestCase):
    """The Vary boxes must agree with what LHS selection accepts."""

    def _vary(self, design) -> dict:
        return {
            row.key: row.sweepable
            for row in recipe_parameter_definitions(design)
            if row.key
            in {"frequency_ghz", "element_spacing_lambda", "element_spacing_mm"}
        }

    def test_electrical_mode_offers_wavelength_spacing_only(self):
        vary = self._vary(build("inset_patch"))
        self.assertEqual(
            vary,
            {
                "frequency_ghz": False,
                "element_spacing_lambda": True,
                "element_spacing_mm": False,
            },
        )

    def test_physical_mode_offers_frequency_and_millimetre_spacing(self):
        vary = self._vary(build("inset_patch", spacing_mode=SPACING_MODE_FIXED_MM))
        self.assertEqual(
            vary,
            {
                "frequency_ghz": True,
                "element_spacing_lambda": False,
                "element_spacing_mm": True,
            },
        )

    def test_every_offered_vary_box_is_accepted_by_lhs(self):
        for mode in (SPACING_MODE_LAMBDA, SPACING_MODE_FIXED_MM):
            with self.subTest(mode=mode):
                design = build("inset_patch", spacing_mode=mode)
                names = {
                    row.name
                    for row in recipe_parameter_definitions(design)
                    if row.sweepable
                }
                for name in names:
                    lhs_variables_for_state(design, [name])

    def test_the_spacing_basis_is_not_rendered_as_a_numeric_row(self):
        rows = {row.key: row for row in recipe_parameter_definitions(build("inset_patch"))}
        self.assertEqual(rows["spacing_mode"].kind, "choice")


class SpacingModeEquivalenceTests(unittest.TestCase):
    def test_both_modes_start_from_the_same_geometry(self):
        for recipe_id in ARRAY_RECIPES:
            with self.subTest(recipe_id=recipe_id):
                electrical = build(recipe_id)
                physical = build(recipe_id, spacing_mode=SPACING_MODE_FIXED_MM)
                self.assertAlmostEqual(
                    electrical.array.spacing_mm, physical.array.spacing_mm, places=3
                )

    def test_switching_mode_keeps_the_current_spacing(self):
        """Switching at 5 GHz must not snap back to the 2.45 GHz spacing."""

        design = build("inset_patch")
        design = set_value(design, "frequency_ghz", 5.0)
        spacing = design.array.spacing_mm
        switched = set_value(design, "spacing_mode", SPACING_MODE_FIXED_MM)
        self.assertAlmostEqual(switched.array.spacing_mm, spacing, places=6)


class BackwardCompatibilityTests(unittest.TestCase):
    def test_values_without_a_mode_normalize_to_electrical(self):
        agent = create_default_agent()
        for recipe_id in ARRAY_RECIPES:
            with self.subTest(recipe_id=recipe_id):
                recipe = agent.registry.recipe(
                    build(recipe_id).recipe_id
                )
                normalized = recipe.normalize({"array_columns": 3})
                self.assertEqual(normalized["spacing_mode"], SPACING_MODE_LAMBDA)

    def test_a_saved_electrical_design_reloads_unchanged(self):
        design = build("inset_patch")
        restored = type(design).from_dict(design.to_dict())
        self.assertEqual(restored, design)
        self.assertEqual(
            restored.metadata_map().get("spacing_mode"), SPACING_MODE_LAMBDA
        )
        with self.assertRaisesRegex(AntennaBuilderError, "not a sweepable"):
            lhs_variables_for_state(restored, ["FreqGHz"])

    def test_a_saved_physical_design_reloads_unchanged(self):
        design = build("inset_patch", spacing_mode=SPACING_MODE_FIXED_MM)
        restored = type(design).from_dict(design.to_dict())
        self.assertEqual(restored, design)
        self.assertEqual(
            restored.metadata_map().get("spacing_mode"), SPACING_MODE_FIXED_MM
        )
        self.assertEqual(
            lhs_variables_for_state(restored, ["FreqGHz"])[0].name, "FreqGHz"
        )

    def test_a_design_saved_before_the_mode_existed_stays_electrical(self):
        """Projects on disk carry no spacing_mode, and must not change meaning."""

        design = build("inset_patch")
        payload = json.loads(json.dumps(design.to_dict()))
        payload["metadata"] = [
            entry for entry in payload["metadata"] if entry[0] != "spacing_mode"
        ]
        legacy = type(design).from_dict(payload)
        self.assertNotIn("spacing_mode", legacy.metadata_map())
        self.assertAlmostEqual(
            legacy.array.spacing_mm, SPEED_OF_LIGHT_MM_GHZ / 2.45 * 0.55, places=3
        )
        with self.assertRaisesRegex(AntennaBuilderError, "not a sweepable"):
            lhs_variables_for_state(legacy, ["FreqGHz"])
        # Editing it keeps the old meaning and records the mode from then on.
        edited = set_value(legacy, "frequency_ghz", 5.0)
        self.assertAlmostEqual(
            edited.array.spacing_mm, SPEED_OF_LIGHT_MM_GHZ / 5.0 * 0.55, places=3
        )
        self.assertEqual(
            edited.metadata_map().get("spacing_mode"), SPACING_MODE_LAMBDA
        )

    def test_the_default_builder_state_is_unchanged(self):
        state = AntennaState.starting_design()
        with self.assertRaisesRegex(AntennaBuilderError, "not a sweepable"):
            lhs_variables_for_state(state, ["FreqGHz"])


class CSTHandoffTests(unittest.TestCase):
    """The exported model must say which basis it uses."""

    def test_electrical_mode_exports_the_coupling_and_declares_it(self):
        design = build("inset_patch")
        macro = CSTAdapter().macro(design)
        self.assertIn(
            'StoreParameter "ElementSpacing", "299.792458/FreqGHz*SpacingLambda"', macro
        )
        self.assertIn("also moves every element and resizes the board", macro)

    def test_physical_mode_exports_a_fixed_dimension_and_declares_it(self):
        design = build("inset_patch", spacing_mode=SPACING_MODE_FIXED_MM)
        macro = CSTAdapter().macro(design)
        self.assertNotIn("FreqGHz*SpacingLambda", macro)
        self.assertIn("changes the operating point only", macro)
        spacing = parameters(design)["element_spacing_mm"].value
        stored = dict(CSTAdapter().parameter_values(design))
        self.assertAlmostEqual(float(stored["ElementSpacing"]), spacing, places=6)

    def test_builder_macro_keeps_the_declared_basis(self):
        state = AntennaState.starting_design()
        state = apply_plan(state, ("parameter.set", {"key": "array_columns", "value": 3})).state
        self.assertIn("ElementSpacing is locked to SpacingLambda", cst_macro(state))


if __name__ == "__main__":
    unittest.main()
