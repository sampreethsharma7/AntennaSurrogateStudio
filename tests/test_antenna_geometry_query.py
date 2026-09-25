import json
import math
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

from shapely.geometry import Point

from studio.antenna_design import BooleanOperation, TransformSpec, evaluate_scalar, resolve_parameter_values
from studio.antenna_geometry import build_geometry_scene, evaluate_geometry, transformed_point
from studio.antenna_geometry_query import GEOMETRY_TOLERANCES
from tests.geometry_query_fixtures import mesh_signature, representative_designs


class EvaluatedGeometryQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.states = representative_designs()

    def test_exact_pre_refactor_mesh_fingerprints(self):
        # Captured from the original evaluator BEFORE it was refactored. Includes
        # every vertex/face/order, color, source ID, volume, bounds, port and warning.
        baseline = json.loads(Path(__file__).with_name("geometry_query_mesh_baseline.json").read_text())
        self.assertEqual(set(baseline), set(self.states))
        for name, state in self.states.items():
            with self.subTest(name=name):
                self.assertEqual(mesh_signature(build_geometry_scene(state)), baseline[name])

    def test_queries_never_mesh_and_viewer_uses_the_shared_query_once(self):
        with patch("studio.antenna_geometry._extrude_shape", side_effect=AssertionError("meshing")), patch(
            "studio.antenna_geometry._cylinder_mesh", side_effect=AssertionError("meshing")
        ), patch("studio.antenna_geometry._box_mesh", side_effect=AssertionError("meshing")):
            for state in self.states.values():
                evaluate_geometry(state)
        with patch("studio.antenna_geometry.evaluate_geometry", wraps=evaluate_geometry) as shared:
            build_geometry_scene(self.states["circle_hole"])
            self.assertEqual(shared.call_count, 1)

    def test_inset_roles_materials_and_consumed_union_constituents(self):
        result = evaluate_geometry(self.states["inset"])
        radiator = result.object("element_1_1_patch")
        feed = result.object("element_1_1_feed")
        self.assertEqual(radiator.semantic_role, "radiating_patch_conductor")
        self.assertEqual(feed.semantic_role, "feed")
        self.assertEqual(feed.element, (1, 1))
        self.assertEqual(radiator.component, "Antenna")
        self.assertEqual(radiator.material.kind, "conductor")
        self.assertEqual(result.object("substrate").material.kind, "dielectric")
        self.assertEqual(feed.consumed_by, "element_1_1_conductor_union")
        self.assertIs(feed.is_physical, False)
        self.assertIsNone(feed.planar_shape)
        self.assertIsNotNone(feed.source_planar_shape)
        self.assertIn(feed.object_id, radiator.source_primitive_ids)
        self.assertEqual(radiator.boolean_history, ("element_1_1_conductor_union",))
        self.assertTrue(radiator.require_planar_shape().covers(feed.source_planar_shape))
        with self.assertRaises(ValueError):
            feed.require_planar_shape()
        self.assertEqual(len(result.physical_objects), 3)

    def test_resolved_holes_and_public_planar_predicates(self):
        for name, area in (("circle_hole", None), ("rectangle_hole", 6.0)):
            with self.subTest(name=name):
                result = evaluate_geometry(self.states[name])
                radiator = result.object("element_1_1_patch")
                shape = radiator.require_planar_shape()
                tool = result.object("tool")
                self.assertFalse(shape.covers(Point(0, 0)))
                self.assertEqual(len(shape.interiors), 1)
                self.assertGreater(shape.distance(Point(0, 0)), 0)
                self.assertGreater(shape.boundary.distance(Point(0, 0)), 0)
                self.assertAlmostEqual(shape.intersection(tool.source_planar_shape).area, 0)
                self.assertFalse(shape.contains(Point(0, 0)))
                self.assertIs(tool.is_physical, False)
                self.assertNotIn(tool, result.physical_objects)
                self.assertEqual(tool.consumed_by, "custom_boolean")
                self.assertIn("custom_boolean", radiator.boolean_history)
                if area is not None:
                    base = evaluate_geometry(self.states["inset"]).object(radiator.object_id)
                    self.assertAlmostEqual(base.planar_shape.area - shape.area, area)

    def test_edge_union_preserves_source_and_boolean_provenance(self):
        result = evaluate_geometry(self.states["edge_union"])
        radiator, tool = result.object("element_1_1_patch"), result.object("tool")
        self.assertIn("tool", radiator.source_primitive_ids)
        self.assertTrue(radiator.planar_shape.covers(tool.source_planar_shape))
        self.assertEqual(radiator.boolean_history, ("element_1_1_conductor_union", "custom_boolean"))
        self.assertEqual(result.booleans[-1].tool_ids, ("tool",))
        self.assertEqual(result.booleans[-1].operation, "union")

    def test_circular_patch_and_dipole_curvature_and_intervals(self):
        for name, expected_solids in (("circular", 4), ("dipole", 2)):
            result = evaluate_geometry(self.states[name])
            self.assertEqual(len(result.physical_objects), expected_solids)
            self.assertTrue(all(c.status == "completed" for c in result.coverage))
            for item in result.physical_objects:
                if item.primitive == "cylinder":
                    self.assertEqual(item.approximation.circle_segments, 64)
                    self.assertGreater(item.approximation.maximum_chord_error_mm, 0)
                    self.assertEqual((item.z_min, item.z_max), (item.bounds[4], item.bounds[5]))
        dipole = evaluate_geometry(self.states["dipole"])
        self.assertEqual({obj.semantic_role for obj in dipole.physical_objects}, {"radiating_arm"})
        first, second = dipole.physical_objects
        self.assertTrue(first.planar_shape.equals(second.planar_shape))
        self.assertTrue(first.z_max < second.z_min or second.z_max < first.z_min)

    def test_array_frames_and_membership_are_independent_of_ports(self):
        state = self.states["array_2x3"]
        result = evaluate_geometry(state)
        self.assertEqual(len(result.elements), 6)
        spacing = state.array.spacing_mm
        self.assertEqual(result.elements[0].origin_mm, (-spacing, -spacing / 2, 0))
        self.assertEqual(result.elements[-1].origin_mm, (spacing, spacing / 2, 0))
        for element in result.elements:
            row, col = element.index
            self.assertIn(f"element_{row}_{col}_feed", element.object_ids)
            feed = result.object(f"element_{row}_{col}_feed")
            radiator = result.object(f"element_{row}_{col}_patch")
            self.assertEqual(feed.element, radiator.element)
            self.assertEqual(element.world_to_local(element.local_to_world((3, 4, 5))), (3, 4, 5))
        moved_ports = tuple(replace(p, positive_point=(100, 200, 300), negative_point=(4, 5, 6)) for p in state.ports)
        self.assertEqual(result.elements, evaluate_geometry(replace(state, ports=moved_ports)).elements)
        self.assertEqual(result.elements, evaluate_geometry(replace(state, ports=())).elements)
        vector_state = replace(state, array=replace(state.array, row_vector=(0, 0, 1), column_vector=(0, 1, 0)))
        vector_result = evaluate_geometry(vector_state)
        self.assertEqual(vector_result.elements[0].origin_mm, (0, -spacing, -spacing / 2))

    def test_ports_preserve_evaluated_endpoints_and_only_canonical_metadata(self):
        for state in self.states.values():
            result = evaluate_geometry(state)
            values = resolve_parameter_values(state)
            for canonical, evaluated in zip(state.ports, result.ports):
                self.assertEqual(evaluated.positive_point, tuple(evaluate_scalar(v, values) for v in canonical.positive_point))
                self.assertEqual(evaluated.negative_point, tuple(evaluate_scalar(v, values) for v in canonical.negative_point))
                self.assertEqual((evaluated.port_id, evaluated.name, evaluated.kind, evaluated.impedance_ohms),
                                 (canonical.port_id, canonical.name, canonical.kind, canonical.impedance_ohms))
                self.assertEqual(evaluated.canonical_references, ())
                self.assertEqual(evaluated.element_index, canonical.element_index)
                self.assertIsNotNone(evaluated.element)

    def test_translate_duplicate_and_rotation_are_resolved_before_queries(self):
        result = evaluate_geometry(self.states["translated_duplicate"])
        self.assertAlmostEqual(result.object("tool").planar_shape.centroid.x, -5)
        self.assertAlmostEqual(result.object("tool").planar_shape.centroid.y, 2)
        self.assertAlmostEqual(result.object("copy").planar_shape.centroid.x, 10)
        self.assertEqual(result.object("copy").source_primitive_ids, ("copy",))
        # Canonical duplicate is a new primitive, and stores no source link. Do not
        # fabricate a historical duplication relationship unavailable in state.
        state = self.states["rotated"]
        canonical = state.geometry[-1]
        obj = evaluate_geometry(state).object("tool")
        self.assertEqual(obj.transform, canonical.transform)
        self.assertAlmostEqual(obj.planar_shape.area, 6)
        for x, y in obj.local_planar_shape.exterior.coords:
            world = transformed_point(canonical, (x, y, 0))
            self.assertLess(obj.planar_shape.boundary.distance(Point(*world[:2])), 1e-12)

    def test_projected_overlap_does_not_collapse_separate_z_layers(self):
        result = evaluate_geometry(self.states["separate_z"])
        tool = result.object("tool")
        radiator = result.object("element_1_1_patch")
        self.assertTrue(radiator.planar_shape.intersects(tool.planar_shape))
        self.assertGreater(radiator.planar_shape.intersection(tool.planar_shape).area, 0)
        self.assertGreater(tool.z_min, radiator.z_max)
        self.assertAlmostEqual(tool.z_min, radiator.z_min + 10)

    def test_unsupported_boolean_and_nonplanar_primitive_never_claim_clearance(self):
        result = evaluate_geometry(self.states["nonplanar_boolean"])
        self.assertEqual(result.booleans[-1].status, "unsupported")
        self.assertEqual({obj.object_id for obj in result.unresolved_objects}, {"tool", "element_1_1_patch"})
        for obj in result.unresolved_objects:
            with self.assertRaises(ValueError):
                obj.require_planar_shape()
        tilted = evaluate_geometry(self.states["tilted_primitive"]).object("tool")
        self.assertEqual(tilted.status, "partially_evaluated")
        self.assertTrue(tilted.is_physical)
        self.assertIsNotNone(tilted.local_planar_shape)
        self.assertIsNotNone(tilted.bounds)
        with self.assertRaises(ValueError):
            tilted.require_planar_shape()

    def test_unresolved_expression_primitive_port_and_boolean_dependency(self):
        state = self.states["circle_hole"]
        tool = state.geometry[-1]
        for bad, expected in ((replace(tool, primitive="unknown"), "unsupported"),
                              (replace(tool, dimensions=tuple((key, "missing" if key == "radius" else value) for key, value in tool.dimensions)), "failed")):
            with self.subTest(expected=expected):
                changed = replace(state, geometry=(*state.geometry[:-1], bad), ports=(replace(state.ports[0], positive_point=("missing", 0, 0)),))
                result = evaluate_geometry(changed)
                self.assertEqual(result.object("tool").status, expected)
                self.assertIsNone(result.object("element_1_1_patch").is_physical)
                self.assertIsNone(result.ports[0].positive_point)
                self.assertEqual(result.ports[0].status, "failed")
                self.assertTrue(result.warnings)

    def test_unknown_propagates_through_dependent_booleans(self):
        state = self.states["nonplanar_boolean"]
        tool2 = replace(self.states["circle_hole"].geometry[-1], object_id="second", name="second")
        changed = replace(state, geometry=(*state.geometry, tool2), booleans=(*state.booleans,
            BooleanOperation("dependent", "subtract", "element_1_1_patch", ("second",))))
        result = evaluate_geometry(changed)
        self.assertEqual(result.booleans[-1].status, "unsupported")
        self.assertIn("unresolved earlier", result.booleans[-1].reason)
        self.assertIsNone(result.object("element_1_1_patch").planar_shape)
        # A prior approximate success must not hide a subsequent unsupported step.
        edge = self.states["edge_union"]
        near = replace(edge.geometry[-1], transform=TransformSpec((0, 0, 1e-8)))
        tilted = replace(tool2, transform=TransformSpec(rotate_deg=(90, 0, 0)))
        result = evaluate_geometry(replace(edge, geometry=(*edge.geometry[:-1], near, tilted),
            booleans=(*edge.booleans, BooleanOperation("later_failure", "subtract", "element_1_1_patch", ("second",)))))
        self.assertEqual(result.object("element_1_1_patch").status, "unsupported")
        self.assertIsNone(result.object("element_1_1_patch").is_physical)

    def test_tolerances_and_small_layer_ambiguity_are_explicit(self):
        result = evaluate_geometry(self.states["circle_hole"])
        self.assertEqual(result.tolerances, GEOMETRY_TOLERANCES)
        self.assertEqual(result.tolerances.layer_mm, 1e-7)
        self.assertEqual(result.tolerances.mesh_vertex_decimal_places, 12)
        radiator = result.object("element_1_1_patch")
        self.assertAlmostEqual(radiator.approximation.maximum_chord_error_mm, 3 * (1 - math.cos(math.pi / 64)))
        state = self.states["edge_union"]
        tool = replace(state.geometry[-1], transform=TransformSpec((0, 0, 1e-8)))
        result = evaluate_geometry(replace(state, geometry=(*state.geometry[:-1], tool)))
        self.assertEqual(result.booleans[-1].status, "partially_evaluated")
        self.assertIn("tolerance", result.booleans[-1].reason)
        self.assertEqual(result.object("element_1_1_patch").status, "partially_evaluated")

    def test_geometry_hash_tracks_geometry_provenance_and_excludes_bookkeeping(self):
        state = self.states["circle_hole"]
        baseline = evaluate_geometry(state)
        revised = replace(state, revision=19, design_id="other", metadata=(*state.metadata, ("timestamp", "tomorrow"), ("audit", "new log")), validation=())
        again = evaluate_geometry(revised)
        self.assertEqual(again.geometry_hash, baseline.geometry_hash)
        self.assertEqual(again.semantic_design_hash, baseline.semantic_design_hash)
        self.assertNotEqual(again.design_ref, baseline.design_ref)
        self.assertEqual(baseline.geometry_hash, evaluate_geometry(state).geometry_hash)
        self.assertNotEqual(baseline.geometry_hash, evaluate_geometry(self.states["rectangle_hole"]).geometry_hash)
        for changed in (
            replace(state, geometry=(*state.geometry[:-1], replace(state.geometry[-1], transform=TransformSpec((1, 0, 0))))),
            replace(state, geometry=tuple(replace(obj, tags=(*obj.tags, "new_role")) if obj.object_id == "tool" else obj for obj in state.geometry)),
            replace(state, array=replace(state.array, spacing_mm=state.array.spacing_mm + 1, columns=2)),
            replace(state, ports=(replace(state.ports[0], impedance_ohms=75),)),
            replace(state, materials=tuple(replace(mat, epsilon_r=5) if mat.kind == "dielectric" else mat for mat in state.materials)),
        ):
            self.assertNotEqual(baseline.geometry_hash, evaluate_geometry(changed).geometry_hash)

    def test_result_and_shapes_are_read_only_and_canonical_state_is_untouched(self):
        state = self.states["circle_hole"]
        before = state.to_dict()
        result = evaluate_geometry(state)
        with self.assertRaises(FrozenInstanceError):
            result.geometry_hash = "changed"
        obj = result.object("element_1_1_patch")
        with self.assertRaises(FrozenInstanceError):
            obj.is_physical = False
        with self.assertRaises(AttributeError):
            obj.planar_shape.exterior.coords = [(0, 0)]
        obj.planar_shape.buffer(100)  # Shapely operations return new shapes.
        self.assertEqual(state.to_dict(), before)
        self.assertEqual(evaluate_geometry(state).geometry_hash, result.geometry_hash)


if __name__ == "__main__":
    unittest.main()
