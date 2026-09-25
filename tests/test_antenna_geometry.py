import math
import unittest
from collections import Counter
from dataclasses import replace

from shapely.geometry import Point, Polygon

from studio.antenna_agent import create_default_agent
from studio.antenna_builder import apply_text_instruction, build_geometry_scene, cst_history
from studio.antenna_design import DesignValidationError, evaluate_scalar
from studio.antenna_llm_planner import LLMToolPlan, PlannedToolCall
from studio.antenna_validation import validate_design


class _Planner:
    def __init__(self, calls):
        self.result = LLMToolPlan("execute", "geometry audit", tuple(
            PlannedToolCall(name, arguments) for name, arguments in calls
        ))

    def plan(self, **_kwargs):
        return self.result


def _apply(state, *calls):
    return apply_text_instruction(state, "geometry audit", planner=_Planner(calls)).state


def _radiator(scene):
    return next(solid for solid in scene.solids if "patch_element" in solid.tags)


class CanonicalGeometryEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.agent = create_default_agent()

    def test_overly_deep_scalar_expression_is_a_validation_error(self):
        expression = "+".join(["1"] * 5000)
        with self.assertRaisesRegex(DesignValidationError, "too deeply nested"):
            evaluate_scalar(expression, {})

    def test_inset_patch_unions_recipe_pieces_before_rendering(self):
        state = self.agent.create_design("inset_patch")
        scene = build_geometry_scene(state)
        radiator = _radiator(scene)
        self.assertEqual(len(scene.solids), 3)
        self.assertEqual(len(radiator.source_ids), 4)
        self.assertEqual(scene.evaluated_operations, ("element_1_1_conductor_union",))
        self.assertFalse(scene.warnings)
        self.assertEqual(cst_history(state).count("Solid.Add"), 3)

    def test_circular_patch_and_probe_match_canonical_primitives(self):
        state = self.agent.create_design("circular_patch")
        scene = build_geometry_scene(state)
        self.assertEqual(len(scene.solids), 4)
        self.assertEqual(sum("circular_patch_element" in item.tags for item in scene.solids), 1)
        self.assertEqual(sum("probe_feed" in item.tags for item in scene.solids), 1)
        self.assertFalse(scene.evaluated_operations)
        self.assertFalse(scene.warnings)

    def test_dipole_preserves_two_separate_cylindrical_arms(self):
        state = self.agent.create_design("dipole")
        scene = build_geometry_scene(state)
        self.assertEqual(len(scene.solids), 2)
        self.assertEqual(sum("dipole_arm" in item.tags for item in scene.solids), 2)
        self.assertFalse(scene.warnings)

    def test_center_circle_subtraction_reduces_exact_evaluated_volume(self):
        state = self.agent.create_design("inset_patch")
        baseline = _radiator(build_geometry_scene(state)).evaluated_volume_mm3
        state = _apply(
            state,
            ("parameter.create", {"key": "slot_radius_mm", "label": "Slot radius", "value": 3.0, "unit": "mm", "sweepable": True}),
            ("geometry.cylinder", {"object_id": "center_slot_tool", "material_id": "copper", "axis": "z", "tags": ["planner_created", "boolean_tool", "slot"], "dimensions": {"center_1": 0, "center_2": 0, "radius": "slot_radius_mm", "start": "substrate_thickness_mm", "end": "substrate_thickness_mm+copper_thickness_mm"}}),
            ("boolean.subtract", {"operation_id": "center_slot_subtract", "target_id": "element_1_1_patch", "tool_ids": ["center_slot_tool"]}),
        )
        scene = build_geometry_scene(state)
        radiator = _radiator(scene)
        changed = radiator.evaluated_volume_mm3
        self.assertAlmostEqual(baseline - changed, math.pi * 3.0**2 * state.copper_thickness_mm, delta=0.003)
        top_z = radiator.bounds[5]
        top_faces = [
            Polygon([(radiator.vertices[index][0], radiator.vertices[index][1]) for index in face])
            for face in radiator.faces
            if len(face) == 3 and all(abs(radiator.vertices[index][2] - top_z) < 1e-9 for index in face)
        ]
        self.assertFalse(any(face.covers(Point(0, 0)) for face in top_faces))
        self.assertIn("center_slot_subtract", scene.evaluated_operations)
        self.assertIn('Solid.Subtract "Antenna:element_1_1_patch", "Antenna:center_slot_tool"', cst_history(state))

    def test_arbitrary_rectangle_subtraction_is_evaluated(self):
        state = self.agent.create_design("inset_patch")
        baseline = _radiator(build_geometry_scene(state)).evaluated_volume_mm3
        state = _apply(
            state,
            ("parameter.create", {"key": "rect_slot_w_mm", "label": "Slot width", "value": 6.0, "unit": "mm", "sweepable": True}),
            ("parameter.create", {"key": "rect_slot_h_mm", "label": "Slot height", "value": 1.0, "unit": "mm", "sweepable": True}),
            ("geometry.rectangle_sheet", {"object_id": "rect_slot_tool", "material_id": "copper", "tags": ["planner_created", "boolean_tool", "slot"], "dimensions": {"x_min": "-rect_slot_w_mm/2", "x_max": "rect_slot_w_mm/2", "y_min": "-rect_slot_h_mm/2", "y_max": "rect_slot_h_mm/2", "z_min": "substrate_thickness_mm", "z_max": "substrate_thickness_mm+copper_thickness_mm"}}),
            ("boolean.subtract", {"operation_id": "rect_slot_subtract", "target_id": "element_1_1_patch", "tool_ids": ["rect_slot_tool"]}),
        )
        changed = _radiator(build_geometry_scene(state)).evaluated_volume_mm3
        self.assertAlmostEqual(baseline - changed, 6.0 * 1.0 * state.copper_thickness_mm, places=5)
        self.assertEqual(cst_history(state).count("Solid.Subtract"), 1)

    def test_duplicate_and_translate_create_two_visible_holes(self):
        state = self.agent.create_design("inset_patch")
        state = _apply(
            state,
            ("parameter.create", {"key": "pair_slot_r_mm", "label": "Pair slot radius", "value": 2.0, "unit": "mm", "sweepable": True}),
            ("geometry.cylinder", {"object_id": "left_slot_tool", "material_id": "copper", "axis": "z", "tags": ["planner_created", "boolean_tool", "slot"], "dimensions": {"center_1": -5, "center_2": 0, "radius": "pair_slot_r_mm", "start": "substrate_thickness_mm", "end": "substrate_thickness_mm+copper_thickness_mm"}}),
            ("geometry.duplicate", {"source_id": "left_slot_tool", "new_id": "right_slot_tool", "offset_mm": [10, 0, 0]}),
            ("boolean.subtract", {"operation_id": "pair_slot_subtract", "target_id": "element_1_1_patch", "tool_ids": ["left_slot_tool", "right_slot_tool"]}),
        )
        radiator = _radiator(build_geometry_scene(state))
        top_z = radiator.bounds[5]
        top_faces = [
            Polygon([(radiator.vertices[index][0], radiator.vertices[index][1]) for index in face])
            for face in radiator.faces
            if len(face) == 3 and all(abs(radiator.vertices[index][2] - top_z) < 1e-9 for index in face)
        ]
        for center in (Point(-5, 0), Point(5, 0)):
            self.assertFalse(any(face.covers(center) for face in top_faces))

    def test_circle_union_at_edge_expands_result(self):
        state = self.agent.create_design("inset_patch")
        baseline = _radiator(build_geometry_scene(state)).evaluated_volume_mm3
        state = _apply(
            state,
            ("parameter.create", {"key": "edge_circle_r_mm", "label": "Edge circle radius", "value": 2.0, "unit": "mm", "sweepable": True}),
            ("geometry.cylinder", {"object_id": "edge_circle_tool", "material_id": "copper", "axis": "z", "tags": ["planner_created", "boolean_tool", "custom_geometry"], "dimensions": {"center_1": "patch_width_mm/2", "center_2": "0*patch_width_mm", "radius": "edge_circle_r_mm", "start": "substrate_thickness_mm", "end": "substrate_thickness_mm+copper_thickness_mm"}}),
            ("boolean.union", {"operation_id": "edge_circle_union", "target_id": "element_1_1_patch", "tool_ids": ["edge_circle_tool"]}),
        )
        scene = build_geometry_scene(state)
        self.assertGreater(_radiator(scene).evaluated_volume_mm3, baseline)
        self.assertIn("edge_circle_union", scene.evaluated_operations)
        self.assertEqual(cst_history(state).count("Solid.Add"), 4)

    def test_simple_array_evaluates_each_element_union(self):
        state = self.agent.update_parameters(
            self.agent.create_design("inset_patch"),
            {"array_rows": 1, "array_columns": 4, "element_spacing_lambda": 0.55},
        ).design
        scene = build_geometry_scene(state)
        self.assertEqual(len(scene.solids), 6)
        self.assertEqual(len(scene.evaluated_operations), 4)
        self.assertEqual(len(scene.element_centers), 4)
        self.assertEqual(cst_history(state).count("Solid.Add"), 12)
        self.assertFalse(scene.warnings)

    def test_rotation_is_shared_by_canonical_bounds_preview_and_cst(self):
        state = self.agent.create_design("dipole")
        first = state.geometry[0]
        rotated = replace(
            first,
            transform=replace(first.transform, rotate_deg=(90.0, 0.0, 0.0)),
        )
        state = validate_design(replace(state, geometry=(rotated, *state.geometry[1:])))
        scene = build_geometry_scene(state)
        solid = next(item for item in scene.solids if item.source_ids == (first.object_id,))
        y_span = solid.bounds[3] - solid.bounds[2]
        z_span = solid.bounds[5] - solid.bounds[4]
        self.assertGreater(y_span, z_span * 10)
        self.assertIn('.Angle "90", "0", "0"', cst_history(state))

    def test_representative_render_meshes_are_closed_and_consistently_wound(self):
        inset = self.agent.create_design("inset_patch")
        center_hole = _apply(
            inset,
            ("parameter.create", {"key": "audit_slot_radius", "label": "Slot radius", "value": 3.0, "unit": "mm", "sweepable": True}),
            ("geometry.cylinder", {"object_id": "audit_center_tool", "material_id": "copper", "axis": "z", "tags": ["planner_created", "boolean_tool", "slot"], "dimensions": {"center_1": 0, "center_2": 0, "radius": "audit_slot_radius", "start": "substrate_thickness_mm", "end": "substrate_thickness_mm+copper_thickness_mm"}}),
            ("boolean.subtract", {"operation_id": "audit_center_subtract", "target_id": "element_1_1_patch", "tool_ids": ["audit_center_tool"]}),
        )
        rectangle_hole = _apply(
            inset,
            ("geometry.rectangle_sheet", {"object_id": "audit_rectangle_tool", "material_id": "copper", "tags": ["planner_created", "boolean_tool", "slot"], "dimensions": {"x_min": "0*patch_width_mm-3", "x_max": "0*patch_width_mm+3", "y_min": "0*patch_length_mm-0.5", "y_max": "0*patch_length_mm+0.5", "z_min": "substrate_thickness_mm", "z_max": "substrate_thickness_mm+copper_thickness_mm"}}),
            ("boolean.subtract", {"operation_id": "audit_rectangle_subtract", "target_id": "element_1_1_patch", "tool_ids": ["audit_rectangle_tool"]}),
        )
        edge_union = _apply(
            inset,
            ("geometry.cylinder", {"object_id": "audit_edge_tool", "material_id": "copper", "axis": "z", "tags": ["planner_created", "boolean_tool", "custom_geometry"], "dimensions": {"center_1": "patch_width_mm/2", "center_2": 0, "radius": "0*patch_width_mm+3", "start": "substrate_thickness_mm", "end": "substrate_thickness_mm+copper_thickness_mm"}}),
            ("boolean.union", {"operation_id": "audit_edge_union", "target_id": "element_1_1_patch", "tool_ids": ["audit_edge_tool"]}),
        )
        array = self.agent.update_parameters(
            inset, {"array_rows": 2, "array_columns": 3}
        ).design
        cases = (
            inset,
            center_hole,
            rectangle_hole,
            edge_union,
            self.agent.create_design("circular_patch"),
            self.agent.create_design("dipole"),
            array,
        )
        for state in cases:
            with self.subTest(family=state.family, elements=state.array.element_count):
                for solid in build_geometry_scene(state).solids:
                    edge_counts = Counter()
                    signed_volume = 0.0
                    for face in solid.faces:
                        for first, second in zip(face, (*face[1:], face[0])):
                            edge_counts[tuple(sorted((first, second)))] += 1
                        origin = solid.vertices[face[0]]
                        for index in range(1, len(face) - 1):
                            b = solid.vertices[face[index]]
                            c = solid.vertices[face[index + 1]]
                            signed_volume += (
                                origin[0] * (b[1] * c[2] - b[2] * c[1])
                                + origin[1] * (b[2] * c[0] - b[0] * c[2])
                                + origin[2] * (b[0] * c[1] - b[1] * c[0])
                            ) / 6
                    self.assertTrue(all(count == 2 for count in edge_counts.values()))
                    self.assertGreater(signed_volume, 0)
                    if solid.evaluated_volume_mm3 is not None:
                        self.assertAlmostEqual(
                            signed_volume, solid.evaluated_volume_mm3, places=6
                        )


if __name__ == "__main__":
    unittest.main()
