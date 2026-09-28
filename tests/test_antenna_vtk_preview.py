import unittest

from studio.antenna_agent import create_default_agent
from studio.antenna_builder import apply_text_instruction, build_geometry_scene, cst_history
from studio.antenna_llm_planner import LLMToolPlan, PlannedToolCall
from studio.antenna_vtk_preview import (
    VTK_IMPORT_ERROR,
    physical_render_size,
    polydata_from_solid,
    polydata_matches_solid,
)


class _Planner:
    def __init__(self, calls):
        self.result = LLMToolPlan(
            "execute",
            "vtk topology audit",
            tuple(PlannedToolCall(name, arguments) for name, arguments in calls),
        )

    def plan(self, **_kwargs):
        return self.result


class VtkPhysicalResolutionTests(unittest.TestCase):
    def test_logical_viewport_is_scaled_to_physical_framebuffer(self):
        self.assertEqual(physical_render_size(700, 500, 1.0), (700, 500))
        self.assertEqual(physical_render_size(700, 500, 1.25), (875, 625))
        self.assertEqual(physical_render_size(700, 500, 1.5), (1050, 750))


def _apply(state, *calls):
    return apply_text_instruction(
        state, "vtk topology audit", planner=_Planner(calls)
    ).state


@unittest.skipIf(VTK_IMPORT_ERROR is not None, "VTK is not installed")
class VtkCanonicalMeshTests(unittest.TestCase):
    def setUp(self):
        self.agent = create_default_agent()

    def _representative_states(self):
        inset = self.agent.create_design("inset_patch")
        center_hole = _apply(
            inset,
            ("parameter.create", {"key": "vtk_slot_radius", "label": "Slot radius", "value": 3.0, "unit": "mm", "sweepable": True}),
            ("geometry.cylinder", {"object_id": "vtk_center_tool", "material_id": "copper", "axis": "z", "tags": ["planner_created", "boolean_tool", "slot"], "dimensions": {"center_1": 0, "center_2": 0, "radius": "vtk_slot_radius", "start": "substrate_thickness_mm", "end": "substrate_thickness_mm+copper_thickness_mm"}}),
            ("boolean.subtract", {"operation_id": "vtk_center_subtract", "target_id": "element_1_1_patch", "tool_ids": ["vtk_center_tool"]}),
        )
        rectangle_hole = _apply(
            inset,
            ("geometry.rectangle_sheet", {"object_id": "vtk_rectangle_tool", "material_id": "copper", "tags": ["planner_created", "boolean_tool", "slot"], "dimensions": {"x_min": "0*patch_width_mm-3", "x_max": "0*patch_width_mm+3", "y_min": "0*patch_length_mm-0.5", "y_max": "0*patch_length_mm+0.5", "z_min": "substrate_thickness_mm", "z_max": "substrate_thickness_mm+copper_thickness_mm"}}),
            ("boolean.subtract", {"operation_id": "vtk_rectangle_subtract", "target_id": "element_1_1_patch", "tool_ids": ["vtk_rectangle_tool"]}),
        )
        edge_union = _apply(
            inset,
            ("geometry.cylinder", {"object_id": "vtk_edge_tool", "material_id": "copper", "axis": "z", "tags": ["planner_created", "boolean_tool", "custom_geometry"], "dimensions": {"center_1": "patch_width_mm/2", "center_2": 0, "radius": "0*patch_width_mm+3", "start": "substrate_thickness_mm", "end": "substrate_thickness_mm+copper_thickness_mm"}}),
            ("boolean.union", {"operation_id": "vtk_edge_union", "target_id": "element_1_1_patch", "tool_ids": ["vtk_edge_tool"]}),
        )
        array = self.agent.update_parameters(
            inset, {"array_rows": 2, "array_columns": 3}
        ).design
        return (
            ("plain inset", inset),
            ("center circular hole", center_hole),
            ("rectangular slot", rectangle_hole),
            ("edge-circle union", edge_union),
            ("circular patch", self.agent.create_design("circular_patch")),
            ("dipole", self.agent.create_design("dipole")),
            ("2x3 array", array),
        )

    def test_vtk_receives_exact_evaluated_vertices_and_faces(self):
        for name, state in self._representative_states():
            with self.subTest(case=name):
                scene = build_geometry_scene(state)
                self.assertFalse(scene.warnings)
                for solid in scene.solids:
                    data = polydata_from_solid(solid)
                    self.assertTrue(polydata_matches_solid(solid, data))
                    for index, expected in enumerate(solid.vertices):
                        actual = data.GetPoint(index)
                        for actual_value, expected_value in zip(actual, expected):
                            self.assertAlmostEqual(actual_value, expected_value, places=5)

    def test_vtk_scene_and_cst_adapter_follow_the_same_canonical_graph(self):
        for name, state in self._representative_states():
            with self.subTest(case=name):
                scene = build_geometry_scene(state)
                history = cst_history(state)
                self.assertEqual(
                    scene.evaluated_operations,
                    tuple(operation.operation_id for operation in state.booleans),
                )
                by_id = {item.object_id: item for item in state.geometry}
                for solid in scene.solids:
                    for source_id in solid.source_ids:
                        self.assertIn(f'.Name "{by_id[source_id].name}"', history)
                expected_subtracts = sum(
                    len(operation.tool_ids)
                    for operation in state.booleans
                    if operation.operation == "subtract"
                )
                expected_unions = sum(
                    len(operation.tool_ids)
                    for operation in state.booleans
                    if operation.operation == "union"
                )
                self.assertEqual(history.count("Solid.Subtract"), expected_subtracts)
                self.assertEqual(history.count("Solid.Add"), expected_unions)


if __name__ == "__main__":
    unittest.main()
