import tempfile
import unittest
from pathlib import Path

from studio.antenna_agent import AgentInstructionError, CapabilityUnavailableError, create_default_agent
from studio.antenna_agent_runner import AntennaAgentRunner
from studio.antenna_builder import load_project_design, save_project_design
from studio.antenna_design import object_bounds, resolved_dimensions
from studio.antenna_engineering_checks import run_engineering_checks
from studio.antenna_llm_planner import AgentStep, LLMToolPlan, PlannedToolCall


def plan(*calls):
    return LLMToolPlan(
        "execute",
        "generic persisted-composition edit",
        tuple(PlannedToolCall(name, arguments) for name, arguments in calls),
    )


class CompositionEditingTests(unittest.TestCase):
    def setUp(self):
        self.agent = create_default_agent()

    def _circle(self, design=None, *, prefix="slot", x=0.0, y=0.0):
        design = design or self.agent.create_design("inset_patch")
        return self.agent.execute_llm_plan(
            design,
            plan(
                ("parameter.create", {
                    "key": f"{prefix}_radius_mm", "label": f"{prefix} radius", "value": 3.0,
                    "unit": "mm", "sweepable": True,
                }),
                ("geometry.cylinder", {
                    "object_id": f"{prefix}_tool", "material_id": "copper", "axis": "z",
                    "tags": ["planner_created", "boolean_tool", "slot"],
                    "dimensions": {
                        "center_1": x, "center_2": y, "radius": f"{prefix}_radius_mm",
                        "start": "substrate_thickness_mm",
                        "end": "substrate_thickness_mm+copper_thickness_mm",
                    },
                }),
                ("boolean.subtract", {
                    "operation_id": f"{prefix}_subtract", "target_id": "element_1_1_patch",
                    "tool_ids": [f"{prefix}_tool"],
                }),
            ),
        ).design

    def _rectangle(self, design=None, *, prefix="rect"):
        design = design or self.agent.create_design("inset_patch")
        return self.agent.execute_llm_plan(
            design,
            plan(
                ("parameter.create", {
                    "key": f"{prefix}_width_mm", "label": "Width", "value": 6.0,
                    "unit": "mm", "sweepable": True,
                }),
                ("parameter.create", {
                    "key": f"{prefix}_height_mm", "label": "Height", "value": 1.0,
                    "unit": "mm", "sweepable": True,
                }),
                ("geometry.rectangle_sheet", {
                    "object_id": f"{prefix}_tool", "material_id": "copper",
                    "tags": ["planner_created", "boolean_tool", "slot"],
                    "dimensions": {
                        "x_min": f"-{prefix}_width_mm/2", "x_max": f"{prefix}_width_mm/2",
                        "y_min": f"-{prefix}_height_mm/2", "y_max": f"{prefix}_height_mm/2",
                        "z_min": "substrate_thickness_mm",
                        "z_max": "substrate_thickness_mm+copper_thickness_mm",
                    },
                }),
                ("boolean.subtract", {
                    "operation_id": f"{prefix}_subtract", "target_id": "element_1_1_patch",
                    "tool_ids": [f"{prefix}_tool"],
                }),
            ),
        ).design

    @staticmethod
    def _call(design, group_id, name, occurrence=0):
        group = next(group for group in design.composed_operations if group.group_id == group_id)
        return [call for call in group.calls if call.name == name][occurrence]

    def _edit(self, design, group_id, operation_id, updates):
        return self.agent.execute_llm_plan(
            design,
            plan(("composition.update_operation", {
                "group_id": group_id,
                "operation_id": operation_id,
                "argument_updates": updates,
            })),
        ).design

    @staticmethod
    def _tools(design):
        return [item for item in design.geometry if "boolean_tool" in item.tags]

    def test_circle_radius_edit_preserves_parameter_expression_and_stable_ids(self):
        design = self._circle()
        group = design.composed_operations[0]
        original_ids = [call.operation_id for call in group.calls]
        cylinder = self._call(design, group.group_id, "geometry.cylinder")
        edited = self._edit(
            design, group.group_id, cylinder.operation_id, {"dimensions": {"radius": 4.0}},
        )

        self.assertEqual(edited.value("slot_radius_mm"), 4.0)
        self.assertEqual(
            self._call(edited, group.group_id, "geometry.cylinder").arguments()["dimensions"]["radius"],
            "slot_radius_mm",
        )
        self.assertEqual([call.operation_id for call in edited.composed_operations[0].calls], original_ids)
        self.assertEqual(resolved_dimensions(edited, self._tools(edited)[0])["radius"], 4.0)

    def test_circle_move_and_rectangle_resize_edit_persisted_structure(self):
        circle = self._circle()
        group = circle.composed_operations[0]
        cylinder = self._call(circle, group.group_id, "geometry.cylinder")
        moved = self._edit(circle, group.group_id, cylinder.operation_id, {
            "dimensions": {"center_1": 3.0, "center_2": -2.0},
        })
        bounds = object_bounds(moved, self._tools(moved)[0])
        self.assertAlmostEqual((bounds[0] + bounds[1]) / 2, 3.0)
        self.assertAlmostEqual((bounds[2] + bounds[3]) / 2, -2.0)

        rectangle = self._rectangle()
        rect_group = rectangle.composed_operations[0]
        width = next(
            call for call in rect_group.calls
            if call.name == "parameter.create" and call.arguments()["key"] == "rect_width_mm"
        )
        height = next(
            call for call in rect_group.calls
            if call.name == "parameter.create" and call.arguments()["key"] == "rect_height_mm"
        )
        resized = self.agent.execute_llm_plan(
            rectangle,
            plan(
                ("composition.update_operation", {
                    "group_id": rect_group.group_id, "operation_id": width.operation_id,
                    "argument_updates": {"value": 8.0},
                }),
                ("composition.update_operation", {
                    "group_id": rect_group.group_id, "operation_id": height.operation_id,
                    "argument_updates": {"value": 2.0},
                }),
            ),
        ).design
        rect_bounds = object_bounds(resized, self._tools(resized)[0])
        self.assertAlmostEqual(rect_bounds[1] - rect_bounds[0], 8.0)
        self.assertAlmostEqual(rect_bounds[3] - rect_bounds[2], 2.0)

    def test_translate_and_rotate_operations_are_generically_editable(self):
        base = self.agent.create_design("inset_patch")
        transformed = self.agent.execute_llm_plan(
            base,
            plan(
                ("parameter.create", {
                    "key": "shape_radius_mm", "label": "Radius", "value": 2.0,
                    "unit": "mm", "sweepable": True,
                }),
                ("geometry.cylinder", {
                    "object_id": "shape_tool", "material_id": "copper", "axis": "z",
                    "tags": ["planner_created", "boolean_tool", "slot"],
                    "dimensions": {
                        "center_1": 0, "center_2": 0, "radius": "shape_radius_mm",
                        "start": "substrate_thickness_mm",
                        "end": "substrate_thickness_mm+copper_thickness_mm",
                    },
                }),
                ("geometry.translate", {"object_id": "shape_tool", "offset_mm": [1, 0, 0]}),
                ("geometry.rotate", {"object_id": "shape_tool", "angles_deg": [0, 0, 0]}),
                ("boolean.subtract", {
                    "operation_id": "shape_subtract", "target_id": "element_1_1_patch",
                    "tool_ids": ["shape_tool"],
                }),
            ),
        ).design
        group = transformed.composed_operations[0]
        translate = self._call(transformed, group.group_id, "geometry.translate")
        rotate = self._call(transformed, group.group_id, "geometry.rotate")
        edited = self.agent.execute_llm_plan(
            transformed,
            plan(
                ("composition.update_operation", {
                    "group_id": group.group_id, "operation_id": translate.operation_id,
                    "argument_updates": {"offset_mm": [2, 1, 0]},
                }),
                ("composition.update_operation", {
                    "group_id": group.group_id, "operation_id": rotate.operation_id,
                    "argument_updates": {"angles_deg": [0, 0, 15]},
                }),
            ),
        ).design
        tool = self._tools(edited)[0]
        self.assertEqual(tool.transform.translate_mm, (2.0, 1.0, 0.0))
        self.assertEqual(tool.transform.rotate_deg, (0.0, 0.0, 15.0))

    def test_all_and_selected_scope_edits_replay_in_target_local_frames(self):
        design = self._circle(x=1.0)
        group = design.composed_operations[0]
        design = self.agent.execute_llm_plan(
            design,
            plan(
                ("parameter.set", {"key": "array_rows", "value": 2}),
                ("parameter.set", {"key": "array_columns", "value": 3}),
                ("composition.set_scope", {"group_id": group.group_id, "scope": "all", "elements": []}),
            ),
        ).design
        parameter = self._call(design, group.group_id, "parameter.create")
        cylinder = self._call(design, group.group_id, "geometry.cylinder")
        edited = self.agent.execute_llm_plan(
            design,
            plan(
                ("composition.update_operation", {
                    "group_id": group.group_id, "operation_id": parameter.operation_id,
                    "argument_updates": {"value": 4.0},
                }),
                ("composition.update_operation", {
                    "group_id": group.group_id, "operation_id": cylinder.operation_id,
                    "argument_updates": {"dimensions": {"center_1": 3.0, "center_2": 2.0}},
                }),
            ),
        ).design
        self.assertEqual(len(self._tools(edited)), 6)
        self.assertTrue(all(resolved_dimensions(edited, item)["radius"] == 4.0 for item in self._tools(edited)))
        spacing = edited.element_spacing_mm
        expected = {
            ((column - 2) * spacing + 3.0, (row - 1.5) * spacing + 2.0)
            for row in (1, 2) for column in (1, 2, 3)
        }
        centers = {
            ((b := object_bounds(edited, item))[0] + (b[1] - b[0]) / 2,
             b[2] + (b[3] - b[2]) / 2)
            for item in self._tools(edited)
        }
        self.assertEqual(centers, expected)

        selected = self.agent.execute_llm_plan(
            edited,
            plan(("composition.set_scope", {
                "group_id": group.group_id, "scope": "selected",
                "elements": [{"row": 1, "column": 2}, {"row": 2, "column": 3}],
            })),
        ).design
        parameter = self._call(selected, group.group_id, "parameter.create")
        selected = self._edit(selected, group.group_id, parameter.operation_id, {"value": 2.5})
        self.assertEqual(len(self._tools(selected)), 2)
        self.assertTrue(all(resolved_dimensions(selected, item)["radius"] == 2.5 for item in self._tools(selected)))

    def test_array_layout_changes_follow_edited_feature(self):
        design = self._circle()
        group = design.composed_operations[0]
        design = self.agent.execute_llm_plan(
            design,
            plan(
                ("parameter.set", {"key": "array_columns", "value": 3}),
                ("composition.set_scope", {"group_id": group.group_id, "scope": "all", "elements": []}),
            ),
        ).design
        cylinder = self._call(design, group.group_id, "geometry.cylinder")
        moved = self._edit(
            design, group.group_id, cylinder.operation_id, {"dimensions": {"center_1": 2.0}},
        )
        relaid = self.agent.execute_llm_plan(
            moved,
            plan(
                ("parameter.set", {"key": "array_rows", "value": 2}),
                ("parameter.set", {"key": "array_columns", "value": 2}),
                ("parameter.set", {"key": "element_spacing_lambda", "value": 0.7}),
            ),
        ).design
        self.assertEqual(len(self._tools(relaid)), 4)
        spacing = relaid.element_spacing_mm
        centers = sorted(
            ((b := object_bounds(relaid, item))[0] + (b[1] - b[0]) / 2,
             b[2] + (b[3] - b[2]) / 2)
            for item in self._tools(relaid)
        )
        expected = sorted(
            ((column - 1.5) * spacing + 2.0, (row - 1.5) * spacing)
            for row in (1, 2) for column in (1, 2)
        )
        self.assertEqual(centers, expected)

    def test_delete_one_multiple_and_array_wide_compositions(self):
        design = self._circle(prefix="first")
        design = self._circle(design, prefix="second", x=8.0)
        first_id, second_id = [group.group_id for group in design.composed_operations]
        deleted = self.agent.execute_llm_plan(
            design, plan(("composition.delete", {"group_id": first_id})),
        ).design
        self.assertEqual([group.group_id for group in deleted.composed_operations], [second_id])
        self.assertNotIn("first_radius_mm", deleted.parameter_map())
        self.assertIn("second_radius_mm", deleted.parameter_map())
        self.assertEqual([item.object_id for item in self._tools(deleted)], ["second_tool"])

        arrayed = self.agent.execute_llm_plan(
            deleted,
            plan(
                ("parameter.set", {"key": "array_columns", "value": 3}),
                ("composition.set_scope", {"group_id": second_id, "scope": "all", "elements": []}),
            ),
        ).design
        self.assertEqual(len(self._tools(arrayed)), 3)
        empty = self.agent.execute_llm_plan(
            arrayed, plan(("composition.delete", {"group_id": second_id})),
        ).design
        self.assertFalse(empty.composed_operations)
        self.assertFalse(self._tools(empty))
        self.assertIn("patch_width_mm", empty.parameter_map())

    def test_edit_delete_round_trip_and_legacy_call_ids(self):
        design = self._circle()
        group = design.composed_operations[0]
        parameter = self._call(design, group.group_id, "parameter.create")
        edited = self._edit(design, group.group_id, parameter.operation_id, {"value": 4.0})
        root = Path(__file__).resolve().parents[1] / ".test_runs"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            save_project_design(folder, edited, [])
            restored, _ = load_project_design(folder)
        self.assertEqual(restored, edited)
        self.assertEqual(restored.value("slot_radius_mm"), 4.0)

        payload = edited.to_dict()
        for call in payload["composed_operations"][0]["calls"]:
            call.pop("operation_id")
        legacy = type(edited).from_dict(payload)
        self.assertTrue(all(call.operation_id.startswith("call_") for call in legacy.composed_operations[0].calls))
        self.assertEqual(legacy.to_dict(), type(legacy).from_dict(legacy.to_dict()).to_dict())

        deleted = self.agent.execute_llm_plan(
            restored, plan(("composition.delete", {"group_id": group.group_id})),
        ).design
        with tempfile.TemporaryDirectory(dir=root) as folder:
            save_project_design(folder, deleted, [])
            reopened, _ = load_project_design(folder)
        self.assertFalse(reopened.composed_operations)
        self.assertNotIn("slot_radius_mm", reopened.parameter_map())

    def test_invalid_ids_schema_and_boolean_replay_are_atomic(self):
        design = self._circle()
        before = design.to_dict()
        group = design.composed_operations[0]
        cylinder = self._call(design, group.group_id, "geometry.cylinder")
        cases = (
            ("composition.update_operation", {
                "group_id": group.group_id, "operation_id": "missing_call", "argument_updates": {"dimensions": {"center_1": 1}},
            }),
            ("composition.update_operation", {
                "group_id": group.group_id, "operation_id": cylinder.operation_id, "argument_updates": {"axis": "x"},
            }),
            ("composition.update_operation", {
                "group_id": "missing_group", "operation_id": cylinder.operation_id, "argument_updates": {"dimensions": {"center_1": 1}},
            }),
            ("composition.update_operation", {
                "group_id": group.group_id, "operation_id": cylinder.operation_id, "argument_updates": {"dimensions": {"center_1": 1000}},
            }),
        )
        for name, arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises((AgentInstructionError, CapabilityUnavailableError)):
                    self.agent.execute_llm_plan(design, plan((name, arguments)))
                self.assertEqual(design.to_dict(), before)
        with self.assertRaises((AgentInstructionError, CapabilityUnavailableError)):
            self.agent.execute_llm_plan(
                design, plan(("composition.delete", {"group_id": "missing_group"})),
            )
        self.assertEqual(design.to_dict(), before)

    def test_manifest_exposes_stable_editable_records_and_dynamic_tools(self):
        design = self._circle()
        manifest = self.agent.capability_manifest(design)
        names = [item["name"] for item in manifest["callable_tools"]]
        self.assertIn("composition.update_operation", names)
        self.assertIn("composition.delete", names)
        feature = manifest["composed_features"][0]
        self.assertEqual(feature["group_id"], "composition_1")
        self.assertEqual(feature["target_selector"]["role"], "radiating_patch_conductor")
        self.assertEqual(feature["coordinate_frame"], "target_local")
        self.assertTrue(feature["parameters"])
        self.assertTrue(all(item["operation_id"] for item in feature["operations"]))
        self.assertTrue(all("editable_arguments" in item for item in feature["operations"]))

    def test_engineering_report_changes_after_accepted_semantic_edit(self):
        design = self._circle()
        before = run_engineering_checks(design)
        group = design.composed_operations[0]
        cylinder = self._call(design, group.group_id, "geometry.cylinder")
        edited = self._edit(
            design, group.group_id, cylinder.operation_id, {"dimensions": {"center_1": 4.0}},
        )
        after = run_engineering_checks(edited)
        self.assertNotEqual(before.semantic_design_hash, after.semantic_design_hash)
        self.assertEqual(after.working_design_ref.revision, edited.revision)

    def test_existing_set_scope_behavior_remains_unchanged(self):
        design = self._circle()
        group = design.composed_operations[0]
        updated = self.agent.execute_llm_plan(
            design,
            plan(
                ("parameter.set", {"key": "array_columns", "value": 2}),
                ("composition.set_scope", {"group_id": group.group_id, "scope": "all", "elements": []}),
            ),
        ).design
        self.assertEqual(updated.composed_operations[0].target_selector.scope, "all")
        self.assertEqual(len(self._tools(updated)), 2)

    def test_provisional_composition_edit_rolls_back_on_clarify_or_refuse(self):
        design = self._circle()
        baseline = design.to_dict()
        group = design.composed_operations[0]
        cylinder = self._call(design, group.group_id, "geometry.cylinder")
        edit_step = AgentStep("execute", "Provisional move.", (PlannedToolCall(
            "composition.update_operation",
            {
                "group_id": group.group_id,
                "operation_id": cylinder.operation_id,
                "argument_updates": {"dimensions": {"center_1": 3.0}},
            },
        ),))

        class ScriptedPlanner:
            last_run_metadata = {}

            def __init__(self, terminal_status):
                self.responses = [edit_step, AgentStep(terminal_status, f"{terminal_status} after edit.")]

            def plan_agent_step(self, **_request):
                return self.responses.pop(0)

        for status in ("clarify", "refuse"):
            with self.subTest(status=status):
                result = AntennaAgentRunner(self.agent).run(
                    baseline_design=design,
                    instruction="Move the feature, then stop.",
                    project_memory={"schema_version": 1},
                    planner=ScriptedPlanner(status),
                )
                self.assertEqual(result.outcome, status)
                self.assertIsNone(result.final_design)
                self.assertEqual(design.to_dict(), baseline)
                self.assertEqual(
                    [item.execution_status for item in result.trajectory],
                    ["accepted", status],
                )


if __name__ == "__main__":
    unittest.main()
