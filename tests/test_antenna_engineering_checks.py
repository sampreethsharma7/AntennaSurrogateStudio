import json
import math
import unittest
from dataclasses import replace
from unittest.mock import patch

from studio.antenna_design import AntennaDesign, ArraySpec, BooleanOperation, GeometryObject, MaterialSpec, PortSpec, TransformSpec
from studio.antenna_engineering import EngineeringReport
from studio.antenna_engineering_checks import (
    CONTACT_CHECK, PORT_CHECK, CHECK_VERSIONS, EngineeringCheckConfig, check_conductor_contact_clearance,
    check_port_attachment_distinctness, engineering_check_cache_key, run_engineering_checks,
)
from studio.antenna_geometry import evaluate_geometry
from tests.geometry_query_fixtures import representative_designs


def brick(name, x0=0, x1=2, y0=0, y1=2, z0=0, z1=1, *, element=1, tags=("planner_created",)):
    return GeometryObject(name, "box", name, "metal", (
        ("x_min", x0), ("x_max", x1), ("y_min", y0), ("y_max", y1), ("z_min", z0), ("z_max", z1)),
        tags=(*tags, f"element_{element}"))


def design(*objects, ports=(), booleans=(), columns=1):
    return replace(AntennaDesign.empty(recipe_id="geometry_fixture", family="geometry_fixture", display_name="Geometry fixture"),
        geometry=objects, materials=(MaterialSpec("metal", "Metal", "conductor"),), ports=ports,
        booleans=booleans, array=ArraySpec(columns=columns, spacing_mm=10))


def findings(report, category):
    return [f for f in report.findings if f.category == category]


def statuses(report):
    return {c.check_id: c.status for c in report.coverage}


class GeometryEngineeringChecksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = representative_designs()

    def test_single_patch_internal_union_contact_is_informational(self):
        report = run_engineering_checks(self.examples["inset"])
        self.assertEqual(set(statuses(report).values()), {"completed"})
        self.assertFalse([f for f in report.findings if f.severity != "info"])
        contacts = findings(report, "union_constituent_contact")
        feed = next(f for f in contacts if "element_1_1_feed" in f.affected_objects)
        self.assertEqual(feed.affected_elements, ((1, 1),))
        self.assertEqual(feed.measured_values["contact_within_tolerance"].value, 1)
        self.assertEqual(len(findings(report, "port_terminal_attached")), 2)

    def test_existing_array_contact_overlap_and_duplicate_ports(self):
        report = run_engineering_checks(self.examples["array_2x3"])
        self.assertEqual(set(statuses(report).values()), {"completed"})
        contacts = findings(report, "conductor_contact")
        self.assertEqual(len(contacts), 3)
        for f in contacts:
            self.assertEqual(f.severity, "warning")
            self.assertAlmostEqual(f.measured_values["retained_constituent_xy_overlap_area_mm2"].value, 60.4995)
            self.assertAlmostEqual(f.measured_values["resolved_conductor_xy_overlap_area_mm2"].value, 107.4279)
            self.assertEqual(len(f.affected_elements), 2)
            self.assertIn("feed", f.message)
            self.assertIn("radiating_patch_conductor", f.message)
        duplicates = findings(report, "coincident_port_segments")
        self.assertEqual(len(duplicates), 3)
        for f in duplicates:
            self.assertEqual(f.measured_values["cross_element"].value, 1)
            self.assertEqual(len(f.affected_ports), 2)
            self.assertEqual(len(f.affected_elements), 2)
        summary = findings(report, "port_segment_summary")[0]
        self.assertEqual(summary.measured_values["canonical_port_count"].value, 6)
        self.assertEqual(summary.measured_values["distinct_segment_groups"].value, 3)
        self.assertEqual(len(findings(report, "port_element_association")), 6)

    def test_other_layouts_and_renamed_objects_use_the_same_generic_checks(self):
        from studio.antenna_agent import create_default_agent
        agent = create_default_agent()
        for rows, cols, contacts in ((2, 2, 2), (3, 1, 3)):
            state = agent.update_parameters(self.examples["inset"], {"array_rows": rows, "array_columns": cols}).design
            mapping = {obj.object_id: f"arbitrary_{i}_{obj.object_id.rsplit('_', 1)[-1]}" for i, obj in enumerate(state.geometry)}
            state = replace(state,
                geometry=tuple(replace(obj, object_id=mapping[obj.object_id], name=mapping[obj.object_id]) for obj in state.geometry),
                booleans=tuple(replace(op, target_id=mapping[op.target_id], tool_ids=tuple(mapping[name] for name in op.tool_ids)) for op in state.booleans))
            report = run_engineering_checks(state)
            self.assertEqual(len(findings(report, "conductor_contact")), contacts)
            self.assertEqual(findings(report, "port_segment_summary")[0].measured_values["distinct_segment_groups"].value, cols)

    def test_separate_z_is_clearance_not_physical_intersection(self):
        report = run_engineering_checks(design(brick("a"), brick("b", z0=2, z1=3, element=2), columns=2))
        self.assertFalse(findings(report, "conductor_contact"))
        clearance = findings(report, "conductor_clearance")[0]
        self.assertEqual(clearance.measured_values["xy_intersection_area_mm2"].value, 4)
        self.assertEqual(clearance.measured_values["z_overlap_mm"].value, 0)
        self.assertEqual(clearance.measured_values["z_clearance_mm"].value, 1)
        self.assertEqual(clearance.measured_values["contact_within_tolerance"].value, 0)

    def test_small_positive_clearance_and_numerical_contact_tolerance(self):
        for gap, contact in ((1e-4, False), (5e-8, True)):
            with self.subTest(gap=gap):
                report = run_engineering_checks(design(brick("a"), brick("b", x0=2+gap, x1=3, element=2), columns=2))
                relations = findings(report, "conductor_contact" if contact else "conductor_clearance")
                self.assertEqual(len(relations), 1)
                self.assertAlmostEqual(relations[0].measured_values["planar_clearance_mm"].value, gap)
                self.assertEqual(relations[0].measured_values["contact_within_tolerance"].value, int(contact))
        for gap, contact in ((1e-4, False), (5e-8, True)):
            report = run_engineering_checks(design(brick("a"), brick("b", z0=1+gap, z1=2)))
            self.assertEqual(bool(findings(report, "conductor_contact")), contact)

    def test_clearance_output_is_bounded_by_nearest_neighbors(self):
        state = design(*(brick(f"c{i}", x0=i*10, x1=i*10+1) for i in range(30)))
        report = run_engineering_checks(state)
        self.assertLessEqual(len(findings(report, "conductor_clearance")), 30)
        self.assertFalse(findings(report, "conductor_contact"))
        disabled = run_engineering_checks(state, config=EngineeringCheckConfig(clearance_neighbors=0))
        self.assertFalse(findings(disabled, "conductor_clearance"))

    def test_object_in_circular_boolean_hole_does_not_touch_survivor(self):
        state = self.examples["circle_hole"]
        insert = replace(brick("island", -.5, .5, -.5, .5, state.substrate_thickness_mm,
                               state.substrate_thickness_mm + state.copper_thickness_mm), material_id="copper")
        report = run_engineering_checks(replace(state, geometry=(*state.geometry, insert)), checks=[CONTACT_CHECK])
        self.assertFalse(findings(report, "conductor_contact"))
        self.assertFalse(any("tool" in f.affected_objects for f in report.findings))
        # No original cutter is considered physical, even though its XY projection
        # encloses the island. The surviving conductor retains an actual hole.
        query = evaluate_geometry(replace(state, geometry=(*state.geometry, insert)))
        self.assertGreater(query.object("island").planar_shape.distance(query.object("element_1_1_patch").planar_shape), 2)

    def test_removed_union_constituent_is_not_resurrected_for_contact(self):
        radiator = brick("a_patch", 0, 2, tags=("patch_element",))
        feed = brick("b_feed", -1, 0, tags=("patch_element",))
        cutter = brick("cutter", -2, .2)
        state = design(radiator, feed, cutter, booleans=(
            BooleanOperation("join", "union", "a_patch", ("b_feed",)),
            BooleanOperation("cut", "subtract", "a_patch", ("cutter",))))
        report = run_engineering_checks(state, checks=[CONTACT_CHECK])
        self.assertEqual(statuses(report)[CONTACT_CHECK], "completed")
        self.assertFalse(findings(report, "union_constituent_contact"))
        self.assertFalse(any("b_feed" in f.affected_objects or "cutter" in f.affected_objects for f in report.findings))

    def test_transform_contact_follows_resolved_shapes_not_boxes(self):
        a = brick("a", -2, 2, -.25, .25)
        b = brick("b", -.1, .1, 1, 1.5, element=2)
        self.assertFalse(findings(run_engineering_checks(design(a, b, columns=2)), "conductor_contact"))
        rotated = replace(a, transform=TransformSpec(rotate_deg=(0, 0, 90)))
        self.assertTrue(findings(run_engineering_checks(design(rotated, b, columns=2)), "conductor_contact"))
        moved = replace(rotated, transform=replace(rotated.transform, translate_mm=(10, 0, 0)))
        self.assertFalse(findings(run_engineering_checks(design(moved, b, columns=2)), "conductor_contact"))
        # Two parallel rotated narrow rectangles can have overlapping bounding
        # boxes while their actual polygons are separated.
        parallel = replace(a, object_id="parallel", name="parallel", transform=TransformSpec((0, 1, 0), (0, 0, 45)))
        diagonal = replace(a, transform=TransformSpec(rotate_deg=(0, 0, 45)))
        report = run_engineering_checks(design(diagonal, parallel))
        self.assertFalse(findings(report, "conductor_contact"))

    def test_endpoint_in_hole_reports_unattached_with_survivor_provenance(self):
        state = self.examples["circle_hole"]
        port = replace(state.ports[0], positive_point=(0, 0, state.substrate_thickness_mm))
        report = run_engineering_checks(replace(state, ports=(port,)), checks=[PORT_CHECK])
        failed = findings(report, "port_terminal_unattached")
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].relationship_key, "positive")
        self.assertEqual(failed[0].measured_values["inside_resolved_hole"].value, 1)
        self.assertIn("element_1_1_patch", failed[0].affected_objects)
        self.assertNotIn("tool", failed[0].affected_objects)

    def test_correct_attachment_and_duplicate_or_reversed_segments(self):
        state = self.examples["inset"]
        port = state.ports[0]
        for reversed_order in (False, True):
            duplicate = replace(port, port_id="another", name="another",
                positive_point=port.negative_point if reversed_order else port.positive_point,
                negative_point=port.positive_point if reversed_order else port.negative_point)
            report = run_engineering_checks(replace(state, ports=(port, duplicate)), checks=[PORT_CHECK])
            self.assertEqual(statuses(report)[PORT_CHECK], "completed")
            self.assertEqual(len(findings(report, "port_terminal_attached")), 4)
            conflict = findings(report, "coincident_port_segments")[0]
            self.assertEqual(set(conflict.affected_ports), {port.port_id, duplicate.port_id})
            self.assertEqual(conflict.measured_values["reversed_endpoint_order"].value, int(reversed_order))
            self.assertEqual(conflict.severity, "warning")
            self.assertFalse(findings(report, "port_same_conductor_component"))

    def test_port_numerical_endpoint_and_coincidence_tolerances(self):
        a, b = brick("lower", z0=-1, z1=0), brick("upper", z0=1, z1=2)
        original = PortSpec("p", "p", "discrete", (1, 1, 1-4e-8), (1, 1, 4e-8))
        for offset, duplicate in ((5e-8, True), (2e-7, False)):
            other = replace(original, port_id="q", positive_point=(1+offset, 1, 1-4e-8), negative_point=(1+offset, 1, 4e-8))
            report = run_engineering_checks(design(a, b, ports=(original, other)), checks=[PORT_CHECK])
            self.assertEqual(len(findings(report, "port_terminal_attached")), 4)
            self.assertEqual(bool(findings(report, "coincident_port_segments")), duplicate)

    def test_same_component_across_objects_and_disconnected_union_islands(self):
        a = brick("a", 0, 1)
        b = brick("b", 1, 2)
        port = PortSpec("p", "p", "discrete", (.5, 1, .5), (1.5, 1, .5))
        report = run_engineering_checks(design(a, b, ports=(port,)), checks=[PORT_CHECK])
        self.assertEqual(len(findings(report, "port_same_conductor_component")), 1)
        far = replace(b, dimensions=tuple((k, v+2 if k in ("x_min", "x_max") else v) for k, v in b.dimensions))
        port = replace(port, negative_point=(3.5, 1, .5))
        disconnected = design(a, far, ports=(port,), booleans=(BooleanOperation("join", "union", "a", ("b",)),))
        report = run_engineering_checks(disconnected, checks=[PORT_CHECK])
        self.assertEqual(len(findings(report, "port_terminal_attached")), 2)
        self.assertFalse(findings(report, "port_same_conductor_component"))

    def test_port_membership_checks_use_local_retained_constituents(self):
        a = brick("a_patch", 0, 1, tags=("patch_element",), element=1)
        b = brick("b_patch", 1, 2, tags=("patch_element",), element=2)
        ground = replace(brick("ground", 0, 2, z0=-2, z1=-1), tags=("ground",))
        p = PortSpec("p", "p", "discrete", (1.5, 1, .5), (1.5, 1, -1), element_index=2)
        report = run_engineering_checks(design(a, b, ground, ports=(p,), columns=2,
            booleans=(BooleanOperation("join", "union", "a_patch", ("b_patch",)),)), checks=[PORT_CHECK])
        self.assertFalse(findings(report, "port_element_association"))
        self.assertEqual(len(findings(report, "port_terminal_attached")), 2)

    def test_partial_geometry_has_unknown_coverage_and_no_false_clean_result(self):
        for name in ("nonplanar_boolean", "tilted_primitive"):
            report = run_engineering_checks(self.examples[name])
            self.assertEqual(set(statuses(report).values()), {"unknown"})
            self.assertTrue(findings(report, "geometry_unknown"))
        state = self.examples["nonplanar_boolean"]
        report = run_engineering_checks(state)
        self.assertFalse(findings(report, "port_terminal_unattached"))
        self.assertTrue(findings(report, "port_attachment_unknown"))

    def test_curve_boundary_uncertainty_never_becomes_false_clear_or_attachment(self):
        circle = GeometryObject("disk", "cylinder", "disk", "metal", (
            ("center_1", 0), ("center_2", 0), ("radius", 3), ("start", 0), ("end", 1)))
        # True circle at the midpoint of a tessellated chord: the polygon alone
        # would incorrectly classify this as an unattached point.
        angle = math.pi / 64
        p = PortSpec("p", "p", "discrete", (3*math.cos(angle), 3*math.sin(angle), .5), (0, 0, .5))
        report = run_engineering_checks(design(circle, ports=(p,)), checks=[PORT_CHECK])
        self.assertEqual(statuses(report)[PORT_CHECK], "unknown")
        self.assertEqual(findings(report, "port_attachment_unknown")[0].relationship_key, "positive")
        self.assertFalse(findings(report, "port_terminal_unattached"))
        tangent = brick("other", 3, 4, -.5, .5)
        report = run_engineering_checks(design(circle, tangent))
        self.assertEqual(statuses(report)[CONTACT_CHECK], "unknown")
        self.assertFalse(findings(report, "conductor_contact"))
        self.assertTrue(findings(report, "contact_unknown"))
        p = replace(p, positive_point=(0, 0, .5), negative_point=(3.5, 0, .5))
        report = run_engineering_checks(design(circle, tangent, ports=(p,)), checks=[PORT_CHECK])
        self.assertEqual(statuses(report)[PORT_CHECK], "unknown")
        self.assertTrue(findings(report, "conductor_connectivity_unknown"))
        self.assertFalse(findings(report, "port_same_conductor_component"))

    def test_empty_unresolved_ports_and_unsupported_port_kinds_have_coverage(self):
        self.assertEqual(set(statuses(run_engineering_checks(design())).values()), {"not_applicable"})
        state = self.examples["inset"]
        for port in (replace(state.ports[0], positive_point=("missing", 0, 0)),
                     replace(state.ports[0], kind="waveguide"), replace(state.ports[0], element_index=999)):
            report = run_engineering_checks(replace(state, ports=(port,)), checks=[PORT_CHECK])
            self.assertEqual(statuses(report)[PORT_CHECK], "unknown")

    def test_evaluates_once_shares_query_and_never_mutates_canonical_design(self):
        state = self.examples["array_2x3"]
        before = state.to_dict()
        with patch("studio.antenna_engineering_checks.evaluate_geometry", wraps=evaluate_geometry) as evaluate, \
             patch("studio.antenna_engineering_checks.check_conductor_contact_clearance", wraps=check_conductor_contact_clearance) as contacts, \
             patch("studio.antenna_engineering_checks.check_port_attachment_distinctness", wraps=check_port_attachment_distinctness) as ports:
            report = run_engineering_checks(state)
            self.assertEqual(evaluate.call_count, 1)
            self.assertIs(contacts.call_args.args[0], ports.call_args.args[0])
        self.assertEqual(state.to_dict(), before)
        self.assertEqual(EngineeringReport.from_dict(json.loads(json.dumps(report.to_dict()))), report)
        self.assertFalse(any(f.capability_refs for f in report.findings))
        self.assertFalse(any(f.severity == "blocking" for f in report.findings))

    def test_ids_and_geometry_hashes_repeat_and_survive_revision_only_changes(self):
        state = self.examples["array_2x3"]
        first, again = run_engineering_checks(state), run_engineering_checks(state)
        self.assertEqual(first.to_dict(), again.to_dict())
        revision = run_engineering_checks(replace(state, revision=state.revision+1, metadata=(*state.metadata, ("audit", "later"))))
        self.assertEqual(first.findings, revision.findings)
        self.assertNotEqual(first.working_design_ref, revision.working_design_ref)
        query = evaluate_geometry(state)
        self.assertTrue(all(f.source.evaluated_geometry_hash == query.geometry_hash for f in first.findings))
        key = engineering_check_cache_key(query)
        self.assertEqual(len(key), 64)
        self.assertEqual(key, engineering_check_cache_key(evaluate_geometry(replace(state, revision=100))))
        self.assertNotEqual(key, engineering_check_cache_key(query, config=EngineeringCheckConfig(0)))
        self.assertNotEqual(key, engineering_check_cache_key(query, checks=[PORT_CHECK]))
        with patch.dict("studio.antenna_engineering_checks.CHECK_VERSIONS", {CONTACT_CHECK: "future-test-version"}):
            self.assertNotEqual(key, engineering_check_cache_key(query))

    def test_failed_inspections_are_explicit_and_do_not_publish_partial_findings(self):
        state = self.examples["inset"]
        with patch("studio.antenna_engineering_checks.check_conductor_contact_clearance", side_effect=RuntimeError("failure")):
            report = run_engineering_checks(state)
            self.assertEqual(statuses(report), {name: "failed" if name == CONTACT_CHECK else "completed" for name in CHECK_VERSIONS})
            self.assertTrue(all(f.check_id != CONTACT_CHECK for f in report.findings))
        with patch("studio.antenna_engineering_checks.evaluate_geometry", side_effect=RuntimeError("failure")):
            report = run_engineering_checks(state)
            self.assertEqual(set(statuses(report).values()), {"failed"})
            self.assertFalse(report.findings)
        for checks in (["unregistered"], [CONTACT_CHECK, CONTACT_CHECK]):
            with self.assertRaises(ValueError):
                run_engineering_checks(state, checks=checks)


if __name__ == "__main__":
    unittest.main()
