import json
import tempfile
import unittest
from pathlib import Path

from studio.antenna_agent import PlannerRefusal
from studio.antenna_builder import (
    AntennaBuilderError, AntennaState, apply_structured_state_patch,
    apply_text_instruction, build_geometry_scene, cst_macro,
    lhs_variables_for_state, load_project_design, save_cst_package,
    save_project_design,
)
from studio.antenna_llm_planner import LLMToolPlan, PlannedToolCall


class _StaticPlanner:
    def __init__(self, result):
        self.result = result

    def plan(self, **_kwargs):
        return self.result


class _RepairingPlanner:
    def __init__(self, first, repaired):
        self.first = first
        self.repaired = repaired
        self.feedback = None

    def plan(self, **_kwargs):
        return self.first

    def repair(self, **feedback):
        self.feedback = feedback
        return self.repaired


def apply_plan(state, *calls, status="execute", message="test plan"):
    result = LLMToolPlan(status, message, tuple(PlannedToolCall(name, arguments) for name, arguments in calls))
    return apply_text_instruction(state, "planned instruction", planner=_StaticPlanner(result))


class AntennaBuilderTests(unittest.TestCase):
    def test_planner_audit_records_returned_repair_and_executed_sequences(self):
        state = AntennaState.starting_design()
        first = LLMToolPlan("execute", "wrong order", (
            PlannedToolCall("parameter.set", {"key": "corner_radius_ratio", "value": 0.25}),
            PlannedToolCall("modifier.apply", {"modifier_id": "corner_circle_cutouts_v1"}),
        ))
        repaired = LLMToolPlan("execute", "fixed order", (
            PlannedToolCall("modifier.apply", {"modifier_id": "corner_circle_cutouts_v1"}),
            PlannedToolCall("parameter.set", {"key": "corner_radius_ratio", "value": 0.25}),
        ))
        planner = _RepairingPlanner(first, repaired)
        planner.backend_id = "test_backend"
        planner.model = "test-model"
        planner.last_run_metadata = {"schema_repairs": 0}
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as folder:
            audit = Path(folder) / "planner_ab.jsonl"
            apply_text_instruction(
                state,
                "Add quarter-width corner cutouts",
                planner=planner,
                audit_log_path=audit,
            )
            records = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["planner_backend"], "test_backend")
        self.assertEqual(record["user_request"], "Add quarter-width corner cutouts")
        self.assertEqual(record["returned_plan"]["message"], "wrong order")
        self.assertEqual(record["repair_attempt"]["plan"]["message"], "fixed order")
        self.assertEqual(record["validation_result"]["status"], "accepted")
        self.assertTrue(record["final_executed_tool_sequence"])

    def test_executor_rejection_gets_one_llm_repair_without_partial_state(self):
        state = AntennaState.starting_design()
        first = LLMToolPlan("execute", "wrong order", (
            PlannedToolCall("parameter.set", {"key": "corner_radius_ratio", "value": 0.25}),
            PlannedToolCall("modifier.apply", {"modifier_id": "corner_circle_cutouts_v1"}),
        ))
        repaired = LLMToolPlan("execute", "fixed order", (
            PlannedToolCall("modifier.apply", {"modifier_id": "corner_circle_cutouts_v1"}),
            PlannedToolCall("parameter.set", {"key": "corner_radius_ratio", "value": 0.25}),
        ))
        planner = _RepairingPlanner(first, repaired)

        update = apply_text_instruction(state, "Add quarter-width corner cutouts", planner=planner)

        self.assertEqual(update.state.revision, 1)
        self.assertEqual(state.revision, 0)
        self.assertEqual(update.state.metadata_map()["active_modifiers"], "corner_circle_cutouts_v1")
        self.assertIn("unavailable", str(planner.feedback["error"]))

    def test_example_conversation_updates_one_live_state(self):
        state = AntennaState.starting_design()
        first = apply_plan(state,
            ("recipe.select", {"recipe_id": "inset_patch_v2"}),
            ("parameter.set", {"key": "frequency_ghz", "value": 2.45}),
            ("parameter.set", {"key": "material", "value": "FR4"}))
        second = apply_plan(first.state, ("parameter.set", {"key": "substrate_thickness_mm", "value": 1.6}))
        third = apply_plan(second.state, ("parameter.set", {"key": "patch_width_mm", "value": 38}))
        fourth = apply_plan(third.state,
            ("parameter.set", {"key": "array_rows", "value": 1}),
            ("parameter.set", {"key": "array_columns", "value": 4}),
            ("parameter.set", {"key": "element_spacing_lambda", "value": 0.55}))
        self.assertEqual(fourth.state.frequency_ghz, 2.45)
        self.assertEqual(fourth.state.material, "FR4")
        self.assertEqual(fourth.state.patch_width_mm, 38)
        self.assertEqual((fourth.state.array_rows, fourth.state.array_columns), (1, 4))
        self.assertEqual({state.design_id, fourth.state.design_id}, {state.design_id})

    def test_frequency_and_material_recalculate_starting_patch(self):
        original = AntennaState.starting_design()
        updated = apply_plan(original,
            ("recipe.select", {"recipe_id": "inset_patch_v2"}),
            ("parameter.set", {"key": "frequency_ghz", "value": 5.8}),
            ("parameter.set", {"key": "material", "value": "Rogers RT5880"})).state
        self.assertEqual(updated.frequency_ghz, 5.8)
        self.assertEqual(updated.material, "Rogers RT5880")
        self.assertLess(updated.patch_length_mm, original.patch_length_mm)

    def test_refusal_and_invalid_geometry_are_rejected(self):
        state = AntennaState.starting_design()
        with self.assertRaisesRegex(PlannerRefusal, "not installed"):
            apply_plan(state, status="refuse", message="A horn antenna recipe is not installed.")
        with self.assertRaisesRegex(AntennaBuilderError, "less than half"):
            apply_structured_state_patch(state, {"inset_depth_mm": state.patch_length_mm})
        self.assertEqual(state.revision, 0)

    def test_geometry_scene_uses_exact_array_replication(self):
        state = apply_plan(AntennaState.starting_design(),
            ("parameter.set", {"key": "array_rows", "value": 2}),
            ("parameter.set", {"key": "array_columns", "value": 3}),
            ("parameter.set", {"key": "element_spacing_lambda", "value": 0.6})).state
        scene = build_geometry_scene(state)
        self.assertEqual(len(scene.element_centers), 6)
        self.assertEqual(len(scene.solids), 2 + 6)
        self.assertEqual(len(scene.evaluated_operations), 6)
        self.assertEqual(scene.bounds[1] - scene.bounds[0], state.board_width_mm)

    def test_cst_macro_preserves_parameters_and_independent_ports(self):
        state = apply_plan(AntennaState.starting_design(),
            ("parameter.set", {"key": "array_columns", "value": 4}),
            ("parameter.set", {"key": "element_spacing_lambda", "value": 0.55})).state
        script = cst_macro(state)
        for parameter in ("FreqGHz", "SubH", "PatchL", "PatchW", "Inset", "ElementSpacing"):
            self.assertIn(f'StoreParameter "{parameter}"', script)
        self.assertEqual(script.count("With DiscretePort"), 4)
        self.assertNotIn("Solver.Start", script)

    def test_project_state_and_export_package_round_trip(self):
        state = apply_plan(AntennaState.starting_design(), ("parameter.set", {"key": "patch_width_mm", "value": 38})).state
        messages = [{"role": "user", "content": "Change width"}, {"role": "builder", "content": "Updated."}]
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as folder:
            root = Path(folder)
            save_project_design(root, state, messages)
            restored, restored_messages = load_project_design(root)
            macro_path, manifest_path = save_cst_package(root / "candidate.bas", restored, conversation=restored_messages)
            self.assertEqual(restored, state)
            self.assertTrue(macro_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["derived"]["port_count"], 1)
            self.assertFalse(manifest["solver_adapter"]["solver_started"])

    def test_composed_operation_history_survives_project_save_and_reopen(self):
        state = apply_plan(
            AntennaState.starting_design(),
            ("parameter.create", {
                "key": "slot_radius_mm", "label": "Slot radius",
                "value": 3.0, "unit": "mm", "sweepable": True,
            }),
            ("geometry.cylinder", {
                "object_id": "center_slot_tool", "material_id": "copper", "axis": "z",
                "tags": ["planner_created", "boolean_tool", "slot"],
                "dimensions": {
                    "center_1": 0, "center_2": 0, "radius": "slot_radius_mm",
                    "start": "substrate_thickness_mm", "end": "substrate_thickness_mm+copper_thickness_mm",
                },
            }),
            ("boolean.subtract", {
                "operation_id": "center_slot_subtract", "target_id": "element_1_1_patch",
                "tool_ids": ["center_slot_tool"],
            }),
        ).state
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as folder:
            root = Path(folder)
            save_project_design(root, state, [])
            restored, _messages = load_project_design(root)

        self.assertEqual(restored, state)
        self.assertEqual(len(restored.composed_operations), 1)
        target = restored.composed_operations[0].calls[-1].semantic_target
        self.assertEqual((target.role, target.element_row, target.element_column), (
            "radiating_patch_conductor", 1, 1,
        ))
        rebuilt = apply_plan(
            restored,
            ("parameter.set", {"key": "patch_width_mm", "value": 40.0}),
        ).state
        self.assertEqual(rebuilt.booleans[-1].operation_id, "center_slot_subtract")

    def test_selected_parameters_become_lhs_variables(self):
        state = AntennaState.starting_design()
        variables = lhs_variables_for_state(state, ["PatchL", "PatchW", "Inset"])
        self.assertEqual([variable.name for variable in variables], ["PatchL", "PatchW", "Inset"])
        with self.assertRaisesRegex(AntennaBuilderError, "not a sweepable"):
            lhs_variables_for_state(state, ["ArrayRows"])
        with self.assertRaisesRegex(AntennaBuilderError, "not a sweepable"):
            lhs_variables_for_state(state, ["FreqGHz"])

    def test_ai_structured_patch_remains_allowlisted_and_revalidated(self):
        state = AntennaState.starting_design()
        update = apply_structured_state_patch(state, {"material": "Rogers RO4003C", "patch_width_mm": 39, "array_columns": 4})
        self.assertEqual(update.state.material, "Rogers RO4003C")
        self.assertEqual(update.state.array_columns, 4)
        with self.assertRaisesRegex(AntennaBuilderError, "unavailable"):
            apply_structured_state_patch(state, {"cst_command": "Solver.Start"})

    def test_corner_cutouts_preview_cst_and_lhs_share_state(self):
        state = apply_plan(AntennaState.starting_design(),
            ("design.reset", {}),
            ("modifier.apply", {"modifier_id": "corner_circle_cutouts_v1"}),
            ("parameter.set", {"key": "corner_radius_ratio", "value": 0.25})).state
        scene = build_geometry_scene(state)
        script = cst_macro(state)
        self.assertEqual(len(scene.solids), 3)
        self.assertEqual(len(scene.evaluated_operations), 2)
        self.assertIn('StoreParameter "CornerRadius", "PatchW*CornerRadiusRatio"', script)
        self.assertEqual(script.count("Solid.Subtract"), 4)
        variable = lhs_variables_for_state(state, ["CornerRadiusRatio"])[0]
        self.assertAlmostEqual(variable.minimum, 0.2)
        self.assertAlmostEqual(variable.maximum, 0.3)


if __name__ == "__main__":
    unittest.main()
