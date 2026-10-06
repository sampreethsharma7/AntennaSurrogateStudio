import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from studio.antenna_agent import create_default_agent
from studio.antenna_builder import (
    _populate_native_cst_project,
    apply_text_instruction,
    build_geometry_scene,
    create_native_cst_project,
)
from studio.antenna_llm_planner import LLMToolPlan, PlannedToolCall
from studio.cst_antenna_adapter import CSTAdapter


class _Planner:
    def __init__(self, calls):
        self.result = LLMToolPlan(
            "execute",
            "native CST history test",
            tuple(PlannedToolCall(name, arguments) for name, arguments in calls),
        )

    def plan(self, **_kwargs):
        return self.result


class _RecordingModel:
    def __init__(self):
        self.events = []

    def StoreParameter(self, name, value):
        self.events.append(("parameter", name, value))

    def AddToHistory(self, name, script):
        self.events.append(("history", name, script))


class _NativeRecordingModel(_RecordingModel):
    def __init__(self):
        super().__init__()
        self.saved_path = None
        self.quit_calls = 0

    def SaveAs(self, path, _include_results):
        self.saved_path = path
        Path(path).write_bytes(b"mock cst project")

    def Quit(self):
        self.quit_calls += 1


class _RecordingApplication:
    def __init__(self, model):
        self.model = model

    def NewMWS(self):
        return self.model


def _apply(state, *calls):
    return apply_text_instruction(
        state,
        "native CST history test",
        planner=_Planner(calls),
    ).state


def _center_circle_slot(state):
    return _apply(
        state,
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
    )


def _center_rectangle_slot(state):
    return _apply(
        state,
        (
            "parameter.create",
            {
                "key": "rect_slot_w_mm",
                "label": "Slot width",
                "value": 6.0,
                "unit": "mm",
                "sweepable": True,
            },
        ),
        (
            "parameter.create",
            {
                "key": "rect_slot_h_mm",
                "label": "Slot height",
                "value": 1.0,
                "unit": "mm",
                "sweepable": True,
            },
        ),
        (
            "geometry.rectangle_sheet",
            {
                "object_id": "rect_slot_tool",
                "material_id": "copper",
                "tags": ["planner_created", "boolean_tool", "slot"],
                "dimensions": {
                    "x_min": "-rect_slot_w_mm/2",
                    "x_max": "rect_slot_w_mm/2",
                    "y_min": "-rect_slot_h_mm/2",
                    "y_max": "rect_slot_h_mm/2",
                    "z_min": "substrate_thickness_mm",
                    "z_max": "substrate_thickness_mm+copper_thickness_mm",
                },
            },
        ),
        (
            "boolean.subtract",
            {
                "operation_id": "rect_slot_subtract",
                "target_id": "element_1_1_patch",
                "tool_ids": ["rect_slot_tool"],
            },
        ),
    )


class CSTNativeHistoryTests(unittest.TestCase):
    def setUp(self):
        self.agent = create_default_agent()
        self.adapter = CSTAdapter()

    def test_native_project_stores_parameters_before_individual_history(self):
        state = self.agent.create_design("inset_patch")
        model = _RecordingModel()

        _populate_native_cst_project(model, state)

        parameter_count = len(state.parameters)
        self.assertTrue(all(event[0] == "parameter" for event in model.events[:parameter_count]))
        self.assertTrue(all(event[0] == "history" for event in model.events[parameter_count:]))
        self.assertEqual(model.events[parameter_count][1], "change units")
        self.assertFalse(any(
            "StoreParameter" in event[2]
            for event in model.events
            if event[0] == "history"
        ))
        self.assertIn(("parameter", "PatchW", f"{state.patch_width_mm:.12g}"), model.events)

    def test_native_project_uses_isolated_cst_and_releases_it_after_save(self):
        state = self.agent.create_design("inset_patch")
        model = _NativeRecordingModel()
        application = _RecordingApplication(model)
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as folder:
            destination = Path(folder) / "released.cst"
            with (
                patch("win32com.client.DispatchEx", return_value=application) as dispatch_ex,
                patch("win32com.client.Dispatch") as shared_dispatch,
                patch("pythoncom.CoInitialize") as co_initialize,
                patch("pythoncom.CoUninitialize") as co_uninitialize,
            ):
                result = create_native_cst_project(destination, state)

            self.assertEqual(result, destination.resolve())
            self.assertTrue(destination.is_file())
        dispatch_ex.assert_called_once_with("CSTStudio.Application")
        shared_dispatch.assert_not_called()
        self.assertEqual(model.quit_calls, 1)
        co_initialize.assert_called_once_with()
        co_uninitialize.assert_called_once_with()

    def test_each_logical_operation_has_a_native_history_entry(self):
        state = self.agent.create_design("inset_patch")
        operations = self.adapter.history_operations(state)
        categories = [operation.category for operation in operations]
        names = [operation.name for operation in operations]

        self.assertEqual(categories.count("primitive"), len(state.geometry))
        self.assertEqual(categories.count("boolean"), sum(len(item.tool_ids) for item in state.booleans))
        self.assertEqual(categories.count("port"), len(state.ports))
        self.assertEqual(categories.count("simulation"), 3)
        self.assertIn("define brick: Antenna:element_1_1_patch", names)
        self.assertIn("define port: Port 1", names)
        self.assertIn("Set frequency range", names)
        self.assertIn("Set boundaries", names)

    def test_center_slot_uses_symbolic_cylinder_then_separate_subtraction(self):
        state = _center_circle_slot(self.agent.create_design("inset_patch"))
        operations = self.adapter.history_operations(state)
        cylinder = next(item for item in operations if item.name == "define cylinder: Antenna:center_slot_tool")
        subtraction = next(item for item in operations if item.name.startswith("Subtract: Antenna:center_slot_tool"))

        self.assertIn('.OuterRadius "SlotRadius"', cylinder.script)
        self.assertIn('.Zrange "SubH", "SubH+CopperT"', cylinder.script)
        self.assertEqual(
            subtraction.script,
            'Solid.Subtract "Antenna:element_1_1_patch", "Antenna:center_slot_tool"',
        )
        self.assertNotIn("StoreParameter", cylinder.script + subtraction.script)

    def test_rectangle_slot_remains_a_native_brick_and_separate_subtraction(self):
        state = _center_rectangle_slot(self.agent.create_design("inset_patch"))
        operations = self.adapter.history_operations(state)
        brick = next(item for item in operations if item.name == "define brick: Antenna:rect_slot_tool")

        self.assertIn('.Xrange "-RectSlotW/2", "RectSlotW/2"', brick.script)
        self.assertIn('.Yrange "-RectSlotH/2", "RectSlotH/2"', brick.script)
        self.assertTrue(any(
            item.script == 'Solid.Subtract "Antenna:element_1_1_patch", "Antenna:rect_slot_tool"'
            for item in operations
        ))

    def test_parameterized_recipe_dimensions_remain_symbolic(self):
        state = self.agent.create_design("inset_patch")
        text = "\n".join(item.script for item in self.adapter.history_operations(state))

        for expression in (
            "PatchW/2",
            "PatchL/2",
            "SubH",
            "Inset",
            "FeedW/2",
        ):
            with self.subTest(expression=expression):
                self.assertIn(expression, text)

    def test_native_history_generation_does_not_mutate_state_or_preview(self):
        state = _center_circle_slot(self.agent.create_design("inset_patch"))
        before_state = state.to_dict()
        before_scene = build_geometry_scene(state)

        self.adapter.parameter_values(state)
        self.adapter.history_operations(state)

        self.assertEqual(state.to_dict(), before_state)
        self.assertEqual(build_geometry_scene(state), before_scene)

    def test_representative_designs_generate_split_native_history(self):
        inset = self.agent.create_design("inset_patch")
        array = self.agent.update_parameters(
            inset,
            {"array_rows": 1, "array_columns": 2},
        ).design
        cases = {
            "plain inset": inset,
            "center circular slot": _center_circle_slot(inset),
            "rectangular slot": _center_rectangle_slot(inset),
            "circular patch": self.agent.create_design("circular_patch"),
            "dipole": self.agent.create_design("dipole"),
            "simple array": array,
        }
        for label, state in cases.items():
            with self.subTest(label=label):
                operations = self.adapter.history_operations(state)
                self.assertTrue(operations)
                self.assertEqual(operations[0].name, "change units")
                self.assertFalse(any("StoreParameter" in item.script for item in operations))
                self.assertEqual(
                    sum(item.category == "primitive" for item in operations),
                    len(state.geometry),
                )


if __name__ == "__main__":
    unittest.main()
