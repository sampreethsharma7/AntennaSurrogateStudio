import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from shapely.geometry import Point, Polygon

from studio.antenna_agent import CapabilityUnavailableError, create_default_agent
from studio.antenna_builder import load_project_design, save_project_design
from studio.antenna_design import object_bounds, resolved_dimensions
from studio.antenna_geometry import build_geometry_scene
from studio.antenna_llm_planner import LLMToolPlan, PlannedToolCall
from studio.cst_antenna_adapter import CSTAdapter


def plan(*calls):
    return LLMToolPlan(
        "execute",
        "generic composed-feature propagation test",
        tuple(PlannedToolCall(name, arguments) for name, arguments in calls),
    )


class ComposedArrayPropagationTests(unittest.TestCase):
    def setUp(self):
        self.agent = create_default_agent()

    def _center_slot(self):
        return self.agent.execute_llm_plan(
            self.agent.create_design("inset_patch"),
            plan(
                (
                    "parameter.create",
                    {
                        "key": "slot_radius_mm",
                        "label": "Slot radius",
                        "value": 3.0,
                        "unit": "mm",
                        "sweepable": True,
                    },
                ),
                (
                    "geometry.cylinder",
                    {
                        "object_id": "center_slot_tool",
                        "material_id": "copper",
                        "axis": "z",
                        "tags": ["planner_created", "boolean_tool", "slot"],
                        "dimensions": {
                            "center_1": 0,
                            "center_2": 0,
                            "radius": "slot_radius_mm",
                            "start": "substrate_thickness_mm",
                            "end": "substrate_thickness_mm+copper_thickness_mm",
                        },
                    },
                ),
                (
                    "boolean.subtract",
                    {
                        "operation_id": "center_slot_subtract",
                        "target_id": "element_1_1_patch",
                        "tool_ids": ["center_slot_tool"],
                    },
                ),
            ),
        ).design

    def _all_2x3(self):
        return self.agent.execute_llm_plan(
            self._center_slot(),
            plan(
                ("parameter.set", {"key": "array_rows", "value": 2}),
                ("parameter.set", {"key": "array_columns", "value": 3}),
                (
                    "composition.set_scope",
                    {"group_id": "composition_1", "scope": "all", "elements": []},
                ),
            ),
        ).design

    @staticmethod
    def _slot_tools(design):
        return [item for item in design.geometry if "boolean_tool" in item.tags]

    @staticmethod
    def _expected_centers(design):
        spacing = design.element_spacing_mm
        return {
            (
                (column - 1 - (design.array.columns - 1) / 2) * spacing,
                (row - 1 - (design.array.rows - 1) / 2) * spacing,
            )
            for row in range(1, design.array.rows + 1)
            for column in range(1, design.array.columns + 1)
        }

    @staticmethod
    def _tool_centers(design):
        return {
            (
                (bounds := object_bounds(design, item))[0] + (bounds[1] - bounds[0]) / 2,
                bounds[2] + (bounds[3] - bounds[2]) / 2,
            )
            for item in ComposedArrayPropagationTests._slot_tools(design)
        }

    def test_exact_live_sequence_creates_one_all_scope_feature_and_six_local_instances(self):
        design = self._all_2x3()
        group = design.composed_operations[0]
        undecorated = self.agent.update_parameters(
            self.agent.create_design("inset_patch"),
            {"array_rows": 2, "array_columns": 3},
        ).design

        self.assertEqual((design.array.rows, design.array.columns), (2, 3))
        self.assertEqual(len(design.composed_operations), 1)
        self.assertEqual(group.target_selector.role, "radiating_patch_conductor")
        self.assertEqual(group.target_selector.scope, "all")
        self.assertEqual(group.target_selector.elements, ())
        self.assertEqual(group.coordinate_frame, "target_local")
        self.assertEqual(len(self._slot_tools(design)), 6)
        self.assertEqual(self._tool_centers(design), self._expected_centers(design))

        subtracts = [item for item in design.booleans if item.operation == "subtract"]
        self.assertEqual(len(subtracts), 6)
        self.assertEqual(
            {item.target_id for item in subtracts},
            {f"element_{row}_{column}_patch" for row in (1, 2) for column in (1, 2, 3)},
        )
        self.assertEqual(len(design.ports), 6)
        self.assertEqual(sum("substrate" in item.tags for item in design.geometry), 1)
        self.assertEqual(
            tuple(item for item in design.geometry if "planner_created" not in item.tags),
            undecorated.geometry,
        )
        self.assertEqual(design.ports, undecorated.ports)

    def test_vtk_mesh_input_and_cst_history_contain_six_real_slots(self):
        design = self._all_2x3()
        scene = build_geometry_scene(design)
        radiators = [item for item in scene.solids if "patch_element" in item.tags]
        self.assertEqual(len(radiators), 6)
        self.assertFalse(scene.warnings)
        self.assertEqual(len(scene.evaluated_operations), 12)

        for center, radiator in zip(sorted(self._expected_centers(design)), sorted(radiators, key=lambda item: item.bounds[:4])):
            top_z = radiator.bounds[5]
            top_faces = [
                Polygon([(radiator.vertices[index][0], radiator.vertices[index][1]) for index in face])
                for face in radiator.faces
                if len(face) == 3 and all(abs(radiator.vertices[index][2] - top_z) < 1e-9 for index in face)
            ]
            self.assertFalse(any(face.covers(Point(*center)) for face in top_faces))

        operations = CSTAdapter().history_operations(design)
        cylinders = [
            item for item in operations
            if item.category == "primitive" and "center_slot" in item.name
        ]
        subtracts = [
            item for item in operations
            if item.category == "boolean" and item.script.startswith("Solid.Subtract")
        ]
        self.assertEqual(len(cylinders), 6)
        self.assertEqual(len(subtracts), 6)
        self.assertTrue(all('.OuterRadius "SlotRadius"' in item.script for item in cylinders))
        self.assertTrue(any("ElementSpacing" in item.script for item in cylinders))

    def test_composed_parameter_and_spacing_updates_replay_every_instance(self):
        design = self._all_2x3()
        before_volume = sum(
            item.evaluated_volume_mm3
            for item in build_geometry_scene(design).solids
            if "patch_element" in item.tags
        )
        design = self.agent.execute_llm_plan(
            design,
            plan(("parameter.set", {"key": "slot_radius_mm", "value": 5.0})),
        ).design
        after_volume = sum(
            item.evaluated_volume_mm3
            for item in build_geometry_scene(design).solids
            if "patch_element" in item.tags
        )

        self.assertEqual(design.value("slot_radius_mm"), 5.0)
        self.assertTrue(all(resolved_dimensions(design, item)["radius"] == 5.0 for item in self._slot_tools(design)))
        self.assertAlmostEqual(
            before_volume - after_volume,
            6 * math.pi * (5.0**2 - 3.0**2) * design.copper_thickness_mm,
            delta=0.02,
        )

        moved = self.agent.execute_llm_plan(
            design,
            plan(("parameter.set", {"key": "element_spacing_lambda", "value": 0.7})),
        ).design
        self.assertEqual(self._tool_centers(moved), self._expected_centers(moved))

    def test_array_resize_and_selected_scope_rederive_physical_instances(self):
        design = self._all_2x3()
        resized = self.agent.execute_llm_plan(
            design,
            plan(
                ("parameter.set", {"key": "array_rows", "value": 1}),
                ("parameter.set", {"key": "array_columns", "value": 4}),
            ),
        ).design
        self.assertEqual(len(resized.composed_operations), 1)
        self.assertEqual(len(self._slot_tools(resized)), 4)
        self.assertEqual(self._tool_centers(resized), self._expected_centers(resized))

        selected_pair = self.agent.execute_llm_plan(
            resized,
            plan((
                "composition.set_scope",
                {
                    "group_id": "composition_1",
                    "scope": "selected",
                    "elements": [{"row": 1, "column": 2}, {"row": 1, "column": 4}],
                },
            )),
        ).design
        self.assertEqual(len(self._slot_tools(selected_pair)), 2)
        self.assertEqual(
            {item.target_id for item in selected_pair.booleans if item.operation == "subtract"},
            {"element_1_2_patch", "element_1_4_patch"},
        )

        selected = self.agent.execute_llm_plan(
            selected_pair,
            plan((
                "composition.set_scope",
                {
                    "group_id": "composition_1",
                    "scope": "single",
                    "elements": [{"row": 1, "column": 3}],
                },
            )),
        ).design
        self.assertEqual(len(self._slot_tools(selected)), 1)
        self.assertEqual(selected.booleans[-1].target_id, "element_1_3_patch")
        self.assertEqual(selected.composed_operations[0].target_selector.elements, ((1, 3),))

    def test_target_local_offset_is_reapplied_from_each_element_center(self):
        design = self.agent.execute_llm_plan(
            self.agent.create_design("inset_patch"),
            plan(
                (
                    "parameter.create",
                    {"key": "offset_radius_mm", "label": "Radius", "value": 2.0, "unit": "mm", "sweepable": True},
                ),
                (
                    "geometry.cylinder",
                    {
                        "object_id": "offset_circle_tool",
                        "material_id": "copper",
                        "axis": "z",
                        "tags": ["planner_created", "boolean_tool", "slot"],
                        "dimensions": {
                            "center_1": 5,
                            "center_2": 0,
                            "radius": "offset_radius_mm",
                            "start": "substrate_thickness_mm",
                            "end": "substrate_thickness_mm+copper_thickness_mm",
                        },
                    },
                ),
                (
                    "boolean.subtract",
                    {
                        "operation_id": "offset_circle_subtract",
                        "target_id": "element_1_1_patch",
                        "tool_ids": ["offset_circle_tool"],
                    },
                ),
            ),
        ).design
        design = self.agent.execute_llm_plan(
            design,
            plan(
                ("parameter.set", {"key": "array_columns", "value": 3}),
                ("composition.set_scope", {"group_id": "composition_1", "scope": "all", "elements": []}),
            ),
        ).design
        expected = {(x + 5.0, y) for x, y in self._expected_centers(design)}
        self.assertEqual(self._tool_centers(design), expected)

    def test_scope_round_trip_and_failed_resolution_are_transactional(self):
        design = self._all_2x3()
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as folder:
            save_project_design(folder, design, [])
            restored, _messages = load_project_design(folder)
        self.assertEqual(restored.composed_operations[0].target_selector.scope, "all")
        self.assertEqual(restored, design)

        before = restored.to_dict()
        with self.assertRaisesRegex(CapabilityUnavailableError, "cannot resolve selected element"):
            self.agent.execute_llm_plan(
                restored,
                plan((
                    "composition.set_scope",
                    {
                        "group_id": "composition_1",
                        "scope": "single",
                        "elements": [{"row": 3, "column": 1}],
                    },
                )),
            )
        self.assertEqual(restored.to_dict(), before)

    def test_manifest_exposes_generic_scope_tool_and_logical_feature(self):
        design = self._center_slot()
        manifest = self.agent.capability_manifest(design)
        scope_tool = next(item for item in manifest["callable_tools"] if item["name"] == "composition.set_scope")
        self.assertEqual(scope_tool["arguments"]["properties"]["group_id"]["enum"], ["composition_1"])
        self.assertEqual(manifest["composed_features"][0]["target_selector"]["scope"], "single")
        self.assertEqual(manifest["composed_features"][0]["coordinate_frame"], "target_local")

    def test_legacy_world_frame_feature_is_promoted_when_scope_changes(self):
        design = self._center_slot()
        legacy_group = replace(
            design.composed_operations[0],
            target_selector=None,
            coordinate_frame="world",
        )
        legacy = replace(design, composed_operations=(legacy_group,))
        propagated = self.agent.execute_llm_plan(
            legacy,
            plan(
                ("parameter.set", {"key": "array_rows", "value": 2}),
                ("parameter.set", {"key": "array_columns", "value": 3}),
                ("composition.set_scope", {"group_id": "composition_1", "scope": "all", "elements": []}),
            ),
        ).design
        self.assertEqual(propagated.composed_operations[0].coordinate_frame, "target_local")
        self.assertEqual(self._tool_centers(propagated), self._expected_centers(propagated))


if __name__ == "__main__":
    unittest.main()
