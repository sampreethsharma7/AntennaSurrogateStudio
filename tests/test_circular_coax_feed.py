import json
import unittest
import shutil
import uuid
from dataclasses import replace
from pathlib import Path

from studio.antenna_agent import create_default_agent
from studio.antenna_builder import load_project_design, save_project_design
from studio.antenna_design import DesignValidationError, PortCrossSection, PortSpec, evaluate_scalar, object_bounds, resolve_parameter_values
from studio.antenna_validation import validate_design
from studio.cst_antenna_adapter import CSTAdapter
from tests.geometry_query_fixtures import mesh_signature
from studio.antenna_geometry import build_geometry_scene


class CircularCoaxFeedTests(unittest.TestCase):
    def setUp(self):
        self.agent = create_default_agent()
        self.adapter = CSTAdapter()

    def _design(self, **updates):
        design = self.agent.create_design("circular_patch")
        return self.agent.update_parameters(design, updates).design if updates else design

    def _object(self, design, object_id):
        return next(item for item in design.geometry if item.object_id == object_id)

    def test_default_feed_is_a_valid_solver_neutral_coax_transition(self):
        design = self._design()
        port = design.ports[0]
        values = resolve_parameter_values(design)

        self.assertEqual(port.kind, "modal_cross_section")
        self.assertEqual(port.signal_terminal, "element_1_probe")
        self.assertEqual(port.reference_terminal, "element_1_coax_outer")
        self.assertEqual(port.cross_section.geometry_id, "element_1_coax_dielectric")
        self.assertEqual(port.cross_section.face, "z_min")
        self.assertEqual(port.mode_count, 1)
        self.assertEqual(design.parameter("coax_dielectric_radius_mm").expression, "3.5*probe_radius_mm")
        self.assertEqual(design.parameter("coax_outer_radius_mm").expression, "4*probe_radius_mm")
        self.assertEqual(design.parameter("coax_launch_length_mm").expression, "3*substrate_thickness_mm")
        self.assertAlmostEqual(values["coax_dielectric_radius_mm"], 3.5 * values["probe_radius_mm"])
        self.assertTrue(all(record.passed for record in design.validation))

    def test_ground_clearance_is_present_and_probe_is_isolated_but_contacts_patch(self):
        design = self._design()
        values = resolve_parameter_values(design)
        ground = self._object(design, "ground")
        patch = self._object(design, "element_1_patch")
        probe = self._object(design, "element_1_probe")
        clearance = self._object(design, "element_1_ground_clearance_tool")
        outer = self._object(design, "element_1_coax_outer")

        self.assertTrue(any(
            item.operation == "subtract" and item.target_id == ground.object_id
            and clearance.object_id in item.tool_ids
            for item in design.booleans
        ))
        self.assertGreater(
            evaluate_scalar(clearance.dimension_map()["radius"], values),
            evaluate_scalar(probe.dimension_map()["radius"], values),
        )
        ground_bounds = object_bounds(design, ground)
        clearance_bounds = object_bounds(design, clearance)
        probe_bounds = object_bounds(design, probe)
        patch_bounds = object_bounds(design, patch)
        outer_bounds = object_bounds(design, outer)
        self.assertLessEqual(clearance_bounds[4], ground_bounds[4])
        self.assertGreaterEqual(clearance_bounds[5], ground_bounds[5])
        self.assertGreaterEqual(probe_bounds[5], patch_bounds[4])
        self.assertEqual(outer_bounds[5], ground_bounds[5])

    def test_feed_offset_probe_radius_and_substrate_height_rebuild_coax(self):
        design = self._design(feed_offset_mm=4.25, probe_radius_mm=0.8, substrate_thickness_mm=2.0)
        values = resolve_parameter_values(design)
        probe = self._object(design, "element_1_probe")
        dielectric = self._object(design, "element_1_coax_dielectric")
        port = design.ports[0]

        self.assertAlmostEqual(evaluate_scalar(probe.dimension_map()["center_1"], values), 4.25)
        self.assertAlmostEqual(evaluate_scalar(probe.dimension_map()["radius"], values), 0.8)
        self.assertAlmostEqual(values["coax_dielectric_radius_mm"], 2.8)
        self.assertAlmostEqual(values["coax_outer_radius_mm"], 3.2)
        self.assertAlmostEqual(values["coax_launch_length_mm"], 6.0)
        self.assertEqual(object_bounds(design, dielectric)[4], -6.0)
        self.assertEqual(evaluate_scalar(port.positive_point[2], values), -6.0)
        self.assertTrue(all(record.passed for record in design.validation))

    def test_project_save_reopen_preserves_modal_port_contract(self):
        design = self._design(feed_offset_mm=4.0, probe_radius_mm=0.75)
        folder = Path(__file__).resolve().parents[1] / ".test_runs" / f"circular_coax_{uuid.uuid4().hex}"
        folder.mkdir(parents=True)
        try:
            save_project_design(folder, design, [{"role": "user", "content": "Move the circular-patch feed."}])
            restored, conversation = load_project_design(folder)
        finally:
            shutil.rmtree(folder)

        self.assertEqual(restored.to_dict(), design.to_dict())
        self.assertEqual(restored.ports[0].cross_section, design.ports[0].cross_section)
        self.assertEqual(conversation[0]["content"], "Move the circular-patch feed.")

    def test_cst_export_uses_symbolic_native_coax_geometry_and_waveguide_port(self):
        design = self._design()
        operations = self.adapter.history_operations(design)
        script = "\n".join(item.script for item in operations)
        port = next(item for item in operations if item.category == "port")

        self.assertNotIn("With DiscretePort", script)
        self.assertIn("With Port", port.script)
        self.assertIn('.NumberOfModes "1"', port.script)
        self.assertIn('.Coordinates "Free"', port.script)
        self.assertIn('.Orientation "zmin"', port.script)
        self.assertIn('.Zrange "-CoaxLength", "-CoaxLength"', port.script)
        self.assertIn("CoaxOuterRadius", port.script)
        self.assertTrue(any(item.name == "Subtract: Antenna:GroundClearanceTool_1 from Antenna:Ground" for item in operations))
        self.assertTrue(any(item.name == "Subtract: Antenna:CoaxOuterBoreTool_1 from Antenna:CoaxOuter_1" for item in operations))

    def test_previous_axial_discrete_port_and_solid_ground_are_rejected(self):
        design = self._design()
        keep = {"ground", "substrate", "element_1_patch", "element_1_probe"}
        geometry = tuple(item for item in design.geometry if item.object_id in keep)
        probe_index = next(index for index, item in enumerate(geometry) if item.object_id == "element_1_probe")
        probe = geometry[probe_index]
        dimensions = dict(probe.dimensions)
        dimensions["start"] = 0
        geometry = tuple(
            replace(item, dimensions=tuple(dimensions.items())) if index == probe_index else item
            for index, item in enumerate(geometry)
        )
        old_port = PortSpec(
            "port_1", "Port 1", "discrete",
            ("0+feed_offset_mm", "0", "substrate_thickness_mm"),
            ("0+feed_offset_mm", "0", 0),
        )
        invalid = replace(design, geometry=geometry, booleans=(), ports=(old_port,), validation=())

        with self.assertRaisesRegex(DesignValidationError, "invalid axial discrete-port excitation"):
            validate_design(invalid)
        with self.assertRaisesRegex(DesignValidationError, "invalid axial discrete-port excitation"):
            self.adapter.history_operations(invalid)

    def test_coax_validation_rejects_missing_clearance_bad_terminals_and_bad_cross_section(self):
        design = self._design()
        port = design.ports[0]
        no_clearance = replace(
            design,
            booleans=tuple(item for item in design.booleans if item.target_id != "ground"),
            validation=(),
        )
        same_terminal = replace(
            design,
            ports=(replace(port, reference_terminal=port.signal_terminal),),
            validation=(),
        )
        metal_cross_section = replace(
            design,
            ports=(replace(port, cross_section=PortCrossSection("element_1_probe", "z_min")),),
            validation=(),
        )
        probe = self._object(design, "element_1_probe")
        disconnected_dimensions = dict(probe.dimensions)
        disconnected_dimensions["end"] = "substrate_thickness_mm-copper_thickness_mm"
        disconnected = replace(
            design,
            geometry=tuple(
                replace(item, dimensions=tuple(disconnected_dimensions.items()))
                if item.object_id == probe.object_id else item
                for item in design.geometry
            ),
            validation=(),
        )

        cases = (
            (no_clearance, "lacks a Boolean ground-clearance hole"),
            (same_terminal, "signal and reference terminals must be different"),
            (metal_cross_section, "modal cross-section must resolve to dielectric geometry"),
            (disconnected, "probe does not contact the circular patch conductor"),
        )
        for invalid, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(DesignValidationError, message):
                validate_design(invalid)

    def test_inset_patch_discrete_port_and_geometry_fingerprint_are_unchanged(self):
        inset = self.agent.create_design("inset_patch")
        baseline = json.loads(Path(__file__).with_name("geometry_query_mesh_baseline.json").read_text())

        self.assertEqual(mesh_signature(build_geometry_scene(inset)), baseline["inset"])
        self.assertTrue(all(port.kind == "discrete" and port.cross_section is None for port in inset.ports))
        self.assertEqual(self.adapter.macro(inset).count("With DiscretePort"), 1)
        self.assertNotIn("With Port\n", self.adapter.macro(inset))


if __name__ == "__main__":
    unittest.main()
