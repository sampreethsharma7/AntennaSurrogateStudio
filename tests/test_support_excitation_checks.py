import unittest
from dataclasses import replace
from unittest.mock import patch

from studio.antenna_design import ArraySpec, BooleanOperation, MaterialSpec, PortSpec, TransformSpec
from studio.antenna_engineering_checks import (
    CHECK_VERSIONS, CONTAINMENT_CHECK, EXCITATION_CHECK, CONTACT_CHECK, PORT_CHECK,
    check_substrate_support_containment, check_excitation_consistency,
    check_conductor_contact_clearance, check_port_attachment_distinctness, run_engineering_checks,
)
from studio.antenna_geometry import evaluate_geometry
from studio.antenna_llm_planner import EngineeringDisposition, validate_engineering_disposition
from tests.geometry_query_fixtures import representative_designs
from tests.test_antenna_engineering_checks import brick, design, findings, statuses


def supported(conductor, *, board=None, extra=(), booleans=()):
    board = board or replace(brick("arbitrary_support", 0, 10, 0, 10, 0, 1),
                            material_id="dielectric", tags=("substrate",))
    state = design(board, conductor, *extra, booleans=booleans)
    return replace(state, materials=(*state.materials, MaterialSpec("dielectric", "Board", "dielectric")))


def independent_elements(count=6):
    objects = tuple(brick(f"conductor_{i}", i*10, i*10+2, 0, 2, 0, 1, element=i+1) for i in range(count))
    ports = tuple(PortSpec(f"excitation_{i}", f"Excitation {i}", "discrete",
        (i*10+1, 1, 1), (i*10+1, 1, 0), element_index=i+1) for i in range(count))
    return design(*objects, ports=ports, columns=count)


class SupportExcitationCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = representative_designs()

    def test_single_patch_exact_board_edge_is_contained_and_excitation_coherent(self):
        report = run_engineering_checks(self.examples["inset"])
        self.assertEqual(set(statuses(report).values()), {"completed"})
        self.assertFalse([f for f in report.findings if f.severity != "info"])
        containment = findings(report, "support_contained")
        self.assertEqual(len(containment), 2)  # physical radiator/feed union and ground
        for f in containment:
            self.assertEqual(f.measured_values["outside_tolerance_area_mm2"].value, 0)
            self.assertEqual(f.measured_values["minimum_support_edge_clearance_mm"].value, 0)
        summary = findings(report, "excitation_collection_summary")[0]
        self.assertEqual(summary.measured_values["physical_segment_group_count"].value, 1)
        self.assertEqual(summary.measured_values["elements_with_indexed_port"].value, 1)

    def test_small_overhang_area_and_boundary_measurement(self):
        state = supported(brick("metal", 9, 10.1, 2, 4, 1, 1.1))
        report = run_engineering_checks(state, checks=[CONTAINMENT_CHECK])
        f = findings(report, "support_overhang")[0]
        self.assertEqual(f.severity, "warning")
        self.assertAlmostEqual(f.measured_values["outside_support_area_mm2"].value, .2)
        self.assertAlmostEqual(f.measured_values["maximum_sampled_boundary_overhang_mm"].value, .1)
        self.assertGreater(f.measured_values["overhang_sampling_error_bound_mm"].value, 0)
        self.assertIn("metal", f.affected_objects)

    def test_boundary_contact_within_tolerance_not_overhang(self):
        for delta in (0, 5e-8, -5e-8):
            with self.subTest(delta=delta):
                report = run_engineering_checks(supported(brick("metal", 9, 10+delta, 2, 4, 1, 1.1)), checks=[CONTAINMENT_CHECK])
                self.assertEqual(statuses(report)[CONTAINMENT_CHECK], "completed")
                self.assertFalse(findings(report, "support_overhang"))
                self.assertEqual(len(findings(report, "support_contained")), 1)

    def test_circular_and_boolean_modified_radiators_contained(self):
        for name in ("circular", "circle_hole", "rectangle_hole", "edge_union"):
            with self.subTest(name=name):
                report = run_engineering_checks(self.examples[name], checks=[CONTAINMENT_CHECK])
                self.assertEqual(statuses(report)[CONTAINMENT_CHECK], "completed")
                self.assertFalse([f for f in report.findings if f.severity == "warning"])

    def test_composed_union_overhang_attributes_retained_feature(self):
        state = self.examples["edge_union"]
        tool = state.geometry[-1]
        tool = replace(tool, dimensions=tuple((k, 10 if k == "radius" else v) for k, v in tool.dimensions))
        state = replace(state, geometry=(*state.geometry[:-1], tool))
        report = run_engineering_checks(state, checks=[CONTAINMENT_CHECK])
        overhang = findings(report, "support_overhang")[0]
        self.assertIn(tool.object_id, overhang.affected_objects)
        self.assertIn("element_1_1_patch", overhang.affected_objects)
        self.assertGreater(overhang.measured_values["outside_support_area_mm2"].value, 0)
        # A consumed subtraction cutter cannot become a physical overhang.
        cut = replace(state, booleans=(*state.booleans[:-1], replace(state.booleans[-1], operation="subtract")))
        self.assertFalse(findings(run_engineering_checks(cut, checks=[CONTAINMENT_CHECK]), "support_overhang"))

    def test_support_hole_uses_resolved_footprint_not_outer_bounds(self):
        metal = brick("metal", 4, 6, 4, 6, 1, 1.1)
        cutter = brick("hole_tool", 3, 7, 3, 7, 0, 1)
        state = supported(metal, extra=(cutter,), booleans=(BooleanOperation("hole", "subtract", "arbitrary_support", ("hole_tool",)),))
        report = run_engineering_checks(state, checks=[CONTAINMENT_CHECK])
        f = findings(report, "support_overhang")[0]
        self.assertEqual(f.measured_values["outside_support_area_mm2"].value, 4)
        self.assertNotIn("hole_tool", f.affected_objects)

    def test_translated_rotated_geometry_preserves_containment_measurements(self):
        state = supported(brick("metal", 9, 10.1, 2, 4, 1, 1.1))
        original = run_engineering_checks(state, checks=[CONTAINMENT_CHECK])
        moved = replace(state, geometry=tuple(replace(obj, transform=TransformSpec((20, -40, 3), (0, 0, 37))) for obj in state.geometry))
        report = run_engineering_checks(moved, checks=[CONTAINMENT_CHECK])
        self.assertAlmostEqual(findings(report, "support_overhang")[0].measured_values["outside_support_area_mm2"].value,
                               findings(original, "support_overhang")[0].measured_values["outside_support_area_mm2"].value)
        board = replace(brick("support", -5, 5, -1, 1, 0, 1), material_id="dielectric", tags=("support",),
                        transform=TransformSpec(rotate_deg=(0, 0, 45)))
        # Inside support AABB, outside its actual rotated polygon.
        report = run_engineering_checks(supported(brick("metal", -2.5, -1.5, 1.5, 2.5, 1, 1.1), board=board), checks=[CONTAINMENT_CHECK])
        self.assertEqual(findings(report, "support_overhang")[0].measured_values["outside_support_area_mm2"].value, 1)

    def test_unsupported_geometry_and_ambiguous_support_are_unknown(self):
        for name in ("nonplanar_boolean", "tilted_primitive"):
            with self.subTest(name=name):
                report = run_engineering_checks(self.examples[name])
                self.assertEqual(statuses(report)[CONTAINMENT_CHECK], "unknown")
                self.assertEqual(statuses(report)[EXCITATION_CHECK], "unknown")
        state = supported(brick("metal", 2, 3, 2, 3, 1, 1.1))
        state = replace(state, geometry=(*state.geometry, replace(state.geometry[0], object_id="other_support")))
        self.assertEqual(statuses(run_engineering_checks(state, checks=[CONTAINMENT_CHECK]))[CONTAINMENT_CHECK], "unknown")
        state = supported(brick("metal", 2, 3, 2, 3, 5, 6))
        self.assertEqual(statuses(run_engineering_checks(state, checks=[CONTAINMENT_CHECK]))[CONTAINMENT_CHECK], "unknown")
        self.assertEqual(statuses(run_engineering_checks(self.examples["dipole"], checks=[CONTAINMENT_CHECK]))[CONTAINMENT_CHECK], "not_applicable")

    def test_problematic_array_adds_group_level_warnings_without_replacing_port_check(self):
        report = run_engineering_checks(self.examples["array_2x3"])
        self.assertEqual(len(findings(report, "conductor_contact")), 3)
        self.assertEqual(len(findings(report, "coincident_port_segments")), 3)
        self.assertEqual(len(findings(report, "port_element_association")), 6)
        grouped = findings(report, "excitation_shared_segment")
        self.assertEqual(len(grouped), 3)
        for f in grouped:
            self.assertEqual(f.measured_values["nominal_port_count"].value, 2)
            self.assertEqual(f.measured_values["associated_element_count"].value, 2)
        self.assertFalse(findings(report, "support_overhang"))
        summary = findings(report, "excitation_collection_summary")[0]
        self.assertEqual(summary.measured_values["expected_element_count"].value, 6)
        self.assertEqual(summary.measured_values["physical_segment_group_count"].value, 3)
        disposition = EngineeringDisposition(report.semantic_design_hash, tuple(f.observation_id for f in report.findings if f.severity == "warning"))
        self.assertEqual(validate_engineering_disposition(disposition, report)["status"], "valid")
        only_old = replace(disposition, acknowledged_observation_ids=tuple(f.observation_id for f in report.findings
            if f.severity == "warning" and f.check_id in {CONTACT_CHECK, PORT_CHECK}))
        self.assertEqual(len(validate_engineering_disposition(only_old, report)["missing_warning_observation_ids"]), 3)

    def test_six_distinct_ports_have_coherent_collection(self):
        state = replace(independent_elements(), array=ArraySpec(rows=2, columns=3, spacing_mm=10))
        report = run_engineering_checks(state, checks=[EXCITATION_CHECK])
        self.assertEqual(statuses(report)[EXCITATION_CHECK], "completed")
        self.assertFalse([f for f in report.findings if f.severity == "warning"])
        summary = findings(report, "excitation_collection_summary")[0]
        self.assertEqual(summary.measured_values["physical_segment_group_count"].value, 6)

    def test_missing_element_port_is_specific_collection_warning(self):
        state = independent_elements()
        report = run_engineering_checks(replace(state, ports=state.ports[:-1]), checks=[EXCITATION_CHECK])
        missing = findings(report, "excitation_missing_element_port")
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0].affected_elements, ((1, 6),))
        self.assertEqual(missing[0].severity, "warning")
        self.assertEqual(statuses(report)[EXCITATION_CHECK], "completed")

    def test_shared_segment_and_one_port_touching_multiple_elements(self):
        state = independent_elements(2)
        same = replace(state.ports[1], positive_point=state.ports[0].negative_point, negative_point=state.ports[0].positive_point)
        report = run_engineering_checks(replace(state, ports=(state.ports[0], same)), checks=[EXCITATION_CHECK])
        self.assertEqual(len(findings(report, "excitation_shared_segment")), 1)
        # One indexed port touching two overlapping elements: no pairwise port
        # duplicate exists, but its collection-level element mapping is ambiguous.
        overlap = replace(state.geometry[1], dimensions=state.geometry[0].dimensions)
        report = run_engineering_checks(replace(state, geometry=(state.geometry[0], overlap), ports=state.ports[:1]), checks=[EXCITATION_CHECK])
        f = findings(report, "excitation_shared_segment")[0]
        self.assertEqual(f.measured_values["nominal_port_count"].value, 1)
        self.assertEqual(f.measured_values["resolved_contact_element_count"].value, 2)

    def test_unresolved_port_or_index_and_unsupported_port_kind_are_unknown(self):
        state = independent_elements(1)
        for port in (replace(state.ports[0], element_index=100), replace(state.ports[0], kind="waveguide"),
                     replace(state.ports[0], positive_point=("missing", 0, 0))):
            with self.subTest(port=port):
                report = run_engineering_checks(replace(state, ports=(port,)), checks=[EXCITATION_CHECK])
                self.assertEqual(statuses(report)[EXCITATION_CHECK], "unknown")
                self.assertTrue(findings(report, "excitation_association_unknown"))

    def test_evaluate_once_shared_query_independent_failure_and_no_mutation(self):
        state = self.examples["array_2x3"]
        snapshot = state.to_dict()
        names = ("check_conductor_contact_clearance", "check_port_attachment_distinctness",
                 "check_substrate_support_containment", "check_excitation_consistency")
        from contextlib import ExitStack
        import studio.antenna_engineering_checks as module
        with ExitStack() as stack:
            evaluate = stack.enter_context(patch.object(module, "evaluate_geometry", wraps=evaluate_geometry))
            checks = [stack.enter_context(patch.object(module, name, wraps=getattr(module, name))) for name in names]
            report = run_engineering_checks(state)
            self.assertEqual(evaluate.call_count, 1)
            self.assertTrue(all(check.call_args.args[0] is checks[0].call_args.args[0] for check in checks))
        self.assertEqual(snapshot, state.to_dict())
        self.assertEqual(set(statuses(report)), set(CHECK_VERSIONS))
        self.assertFalse(any(f.severity == "blocking" or f.capability_refs for f in report.findings))
        for name, key in ((names[2], CONTAINMENT_CHECK), (names[3], EXCITATION_CHECK)):
            with patch.object(module, name, side_effect=RuntimeError("private detail")):
                failed = run_engineering_checks(state)
            self.assertEqual(statuses(failed)[key], "failed")
            self.assertTrue(all(c.status == "completed" for c in failed.coverage if c.check_id != key))
            self.assertFalse(any(f.check_id == key for f in failed.findings))

    def test_findings_are_stable_under_repeat_revision_and_input_order(self):
        state = self.examples["array_2x3"]
        first = run_engineering_checks(state)
        for other in (state, replace(state, revision=99), replace(state, ports=tuple(reversed(state.ports)))):
            again = run_engineering_checks(other)
            self.assertEqual([f.observation_id for f in first.findings], [f.observation_id for f in again.findings])
            self.assertEqual([f.measured_values for f in first.findings], [f.measured_values for f in again.findings])


if __name__ == "__main__":
    unittest.main()
