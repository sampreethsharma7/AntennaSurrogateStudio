import csv
import json
import os
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from studio import __version__
from studio.dataset_registry import register_dataset
from studio.dataset_validation import validate_dataset
from studio.inference import (
    INFERENCE_COMPLETED,
    INFERENCE_FAILED,
    InferenceResult,
)
from studio.model_book import save_model_book, set_active_model_book
from studio.model_training import (
    ModelTrainingRequest,
    submit_model_training_request,
)
from studio.parser_engine import TrainingRequest
from studio.project_store import ProjectStore
from studio.ui import StudioApp, responsive_window_layout


GUI_MAY_BE_AVAILABLE = (
    os.name == "nt"
    or os.sys.platform == "darwin"
    or bool(os.environ.get("DISPLAY"))
)


class ResponsiveWindowLayoutTests(unittest.TestCase):
    def test_1366_by_768_is_physical_and_usable_at_common_windows_scaling(self):
        for dpi_scaling in (1.0, 1.25, 1.5):
            with self.subTest(dpi_scaling=dpi_scaling):
                layout = responsive_window_layout(
                    1366 / dpi_scaling,
                    768 / dpi_scaling,
                    dpi_scaling,
                )
                self.assertEqual((layout.width, layout.height), (1366, 768))
                self.assertLessEqual(layout.min_width, 1366)
                self.assertLessEqual(layout.min_height, 768)
                self.assertLessEqual(layout.ui_scaling, 1.08)
                self.assertAlmostEqual(
                    layout.window_scaling_factor * dpi_scaling,
                    1.0,
                )
                self.assertTrue(layout.compact)


def create_active_book(project, *, output_count=1, name="Page Model"):
    input_path = project.path / "data" / "prepared" / "inputs.csv"
    output_path = project.path / "data" / "prepared" / "outputs.csv"
    target_columns = [f"theta_{index}" for index in range(output_count)]
    if output_count == 1:
        target_columns = ["gain"]
    with input_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Sample ID", "P2", "P3", "P4"])
        for index in range(1, 25):
            writer.writerow(
                [
                    f"Design_{index:03d}",
                    float(index),
                    float((index * 2) % 7),
                    float((index * 3) % 5),
                ]
            )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Sample ID", *target_columns])
        for index in range(1, 25):
            p2 = float(index)
            p3 = float((index * 2) % 7)
            p4 = float((index * 3) % 5)
            values = [
                (target_index + 1) * p2 + 0.5 * p3 - 0.25 * p4
                for target_index in range(output_count)
            ]
            writer.writerow([f"Design_{index:03d}", *values])

    validation = validate_dataset(
        TrainingRequest(
            input_csv_path=input_path,
            output_csv_path=output_path,
            feature_columns=["P2", "P3", "P4"],
            target_columns=target_columns,
            sample_id_column="Sample ID",
        )
    )
    register_dataset(project.path, validation)
    run = submit_model_training_request(
        ModelTrainingRequest(
            model_name="linear_regression",
            training_mode="auto",
            search_level="medium",
        ),
        project_path=project.path,
    )
    if not run.success:
        raise AssertionError(run.error_message)
    book = save_model_book(project.path, run.run_id, name)
    set_active_model_book(project.path, book.book_id)
    return book


@unittest.skipUnless(GUI_MAY_BE_AVAILABLE, "A desktop display is required.")
class InferencePageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        cls.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        cls.store = ProjectStore(Path(cls.temp_dir.name) / "library")
        cls.single_project = cls.store.create_project("Inference Page Single")
        cls.single_book = create_active_book(cls.single_project)
        cls.multi_project = cls.store.create_project("Inference Page Multi")
        cls.multi_book = create_active_book(
            cls.multi_project,
            output_count=12,
            name="Response Model",
        )
        cls.empty_project = cls.store.create_project("Inference Page Empty")
        cls.corrupt_project = cls.store.create_project("Inference Page Corrupt")
        cls.corrupt_book = create_active_book(
            cls.corrupt_project,
            name="Corrupt Model",
        )
        with cls.corrupt_book.model_artifact_path.open("ab") as handle:
            handle.write(b"tampered")
        try:
            cls.app = StudioApp(project_store=cls.store)
        except tk.TclError as exc:
            cls.temp_dir.cleanup()
            raise unittest.SkipTest(
                f"A desktop display is not available: {exc}"
            ) from exc
        cls.app.withdraw()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "app"):
            cls.app.destroy()
        if hasattr(cls, "temp_dir"):
            cls.temp_dir.cleanup()

    def setUp(self):
        self.app.set_sidebar_collapsed(False)
        self.app.set_snowbuddy_collapsed(False)
        project = self.store.open_project(self.single_project.path, touch=False)
        self.app.set_project(project, target_page="inference")
        self.app.update_idletasks()
        self.page = self.app.inference_page

    def _fill_inputs(self, values=None):
        resolved = values or {"P2": 4.0, "P3": 2.0, "P4": 3.0}
        for name, value in resolved.items():
            entry = self.page.input_entries[name]
            entry.delete(0, "end")
            entry.insert(0, str(value))

    def test_page_loads_with_active_model_and_navigation(self):
        self.assertIn("inference", self.app.pages)
        self.assertEqual(self.app.active_page, "inference")
        self.assertEqual(self.page.active_book.book_id, self.single_book.book_id)
        self.assertEqual(self.page.active_badge.winfo_manager(), "")
        self.assertEqual(self.page.model_summary.cget("text"), "3 inputs → 1 outputs")
        self.assertEqual(self.page.result_model_name.cget("text"), "Page Model")
        self.assertIn(
            "Linear Regression",
            self.page.result_model_type.cget("text"),
        )
        self.assertEqual(self.page.predict_button.cget("state"), "normal")
        self.assertEqual(self.page.raw_values_button.cget("state"), "disabled")
        self.assertEqual(self.page.export_button.cget("state"), "disabled")
        self.assertEqual(self.app.nav_buttons["inference"].cget("state"), "normal")

    def test_required_features_generate_numeric_input_fields(self):
        self.assertEqual(list(self.page.input_entries), ["P2", "P3", "P4"])
        self.assertEqual(list(self.page.input_shells), ["P2", "P3", "P4"])
        self.assertTrue(
            all(
                entry.cget("placeholder_text") == "Numeric value"
                for entry in self.page.input_entries.values()
            )
        )
        self.assertEqual(self.page.input_pager.winfo_manager(), "")

    def test_many_required_inputs_are_paged_without_losing_values(self):
        self.page.active_book.feature_columns = [f"P{index}" for index in range(1, 11)]
        self.page._refresh()

        self.assertEqual(len(self.page.input_entries), 10)
        self.assertEqual(self.page.input_pager.winfo_manager(), "grid")
        self.assertEqual(self.page.input_page_label.cget("text"), "Inputs 1–8 of 10")
        self.page.input_entries["P1"].insert(0, "1.25")

        self.page._change_input_page(1)

        self.assertEqual(self.page.input_page_label.cget("text"), "Inputs 9–10 of 10")
        self.assertEqual(self.page.input_entries["P1"].get(), "1.25")

    def test_successful_prediction_uses_existing_backend_and_displays_value(self):
        self._fill_inputs()

        self.page.predict_button.invoke()

        self.assertIsNotNone(self.page.last_result)
        self.assertTrue(self.page.last_result.success)
        self.assertEqual(self.page.result_title.cget("text"), "gain")
        self.assertIn("Predicted value", self.page.result_summary.cget("text"))
        predicted_value = next(iter(self.page.last_result.predictions.values()))
        self.assertEqual(self.page.result_summary_values["count"].cget("text"), "1")
        self.assertEqual(
            self.page.result_summary_values["minimum"].cget("text"),
            f"{predicted_value:.6g}",
        )
        self.assertEqual(
            self.page.result_summary_values["maximum"].cget("text"),
            f"{predicted_value:.6g}",
        )
        self.assertEqual(
            self.page.inputs_used_value.cget("text"),
            "P2 = 4 · P3 = 2 · P4 = 3",
        )
        self.assertEqual(self.page.raw_values_button.cget("state"), "normal")
        self.assertEqual(self.page.export_button.cget("state"), "normal")
        self.assertEqual(self.page.predict_button.cget("state"), "normal")
        self.assertEqual(self.page.predict_button.cget("text"), "Predict")

    def test_multi_output_prediction_uses_compact_summary_and_ordered_plot(self):
        project = self.store.open_project(self.multi_project.path, touch=False)
        self.app.set_project(project, target_page="inference")
        self.app.update_idletasks()
        self.page = self.app.inference_page
        self._fill_inputs()

        self.page.predict_button.invoke()
        self.app.update_idletasks()

        self.assertTrue(self.page.last_result.success)
        self.assertIn("12 outputs", self.page.result_title.cget("text"))
        summary = self.page.result_summary.cget("text")
        self.assertIn("saved Model Book output interface", summary)
        self.assertNotIn("First", summary)
        self.assertNotIn("Last", summary)
        self.assertNotIn("theta_5", summary)
        values = tuple(self.page.last_result.predictions.values())
        self.assertEqual(self.page.result_summary_values["count"].cget("text"), "12")
        self.assertEqual(
            self.page.result_summary_values["minimum"].cget("text"),
            f"{min(values):.6g}",
        )
        self.assertEqual(
            self.page.result_summary_values["maximum"].cget("text"),
            f"{max(values):.6g}",
        )
        self.assertEqual(len(self.page.response_plot.target_names), 12)
        self.assertEqual(
            self.page.response_plot.target_names,
            tuple(self.page.last_result.target_order),
        )
        self.assertEqual(len(self.page.response_plot.y_values), 12)
        self.assertGreater(len(self.page.response_plot.canvas.find_all()), 0)

    def test_replace_and_add_modes_manage_multiple_prediction_curves(self):
        project = self.store.open_project(self.multi_project.path, touch=False)
        self.app.set_project(project, target_page="inference")
        self.app.update_idletasks()
        self.page = self.app.inference_page
        self._fill_inputs({"P2": 4.0, "P3": 2.0, "P4": 3.0})
        self.page.predict_button.invoke()
        first = self.page.response_plot.state.selected_curve

        self.page.prediction_plot_mode.set("Add to plot")
        self._fill_inputs({"P2": 7.0, "P3": 1.0, "P4": 2.0})
        self.page.predict_button.invoke()
        second = self.page.response_plot.state.selected_curve

        self.assertEqual(len(self.page.response_plot.state.curves), 2)
        self.assertEqual(first.inputs, {"P2": 4.0, "P3": 2.0, "P4": 3.0})
        self.assertEqual(second.inputs, {"P2": 7.0, "P3": 1.0, "P4": 2.0})
        self.assertNotEqual(first.curve_id, second.curve_id)

        self.page.prediction_plot_mode.set("Replace current curve")
        self._fill_inputs({"P2": 8.0, "P3": 2.0, "P4": 1.0})
        self.page.predict_button.invoke()

        self.assertEqual(len(self.page.response_plot.state.curves), 2)
        self.assertEqual(
            self.page.response_plot.state.selected_curve.curve_id,
            second.curve_id,
        )
        self.assertEqual(
            self.page.response_plot.state.selected_curve.inputs,
            {"P2": 8.0, "P3": 2.0, "P4": 1.0},
        )

    def test_workbench_visibility_rename_delete_annotations_and_navigation(self):
        project = self.store.open_project(self.multi_project.path, touch=False)
        self.app.set_project(project, target_page="inference")
        self.app.update_idletasks()
        self.page = self.app.inference_page
        self._fill_inputs()
        self.page.predict_button.invoke()
        workbench = self.page.response_plot
        curve = workbench.state.selected_curve
        original_limits = workbench.state.view_limits

        workbench.rename_selected("Nominal Design")
        workbench.set_curve_visible(curve.curve_id, False)
        self.assertEqual(curve.name, "Nominal Design")
        self.assertFalse(curve.visible)
        workbench.set_curve_visible(curve.curve_id, True)

        marker = workbench.add_annotation(curve.x_values[0], curve.y_values[0], "Mpeak")
        self.assertEqual(marker.curve_id, curve.curve_id)
        self.assertEqual(len(workbench.state.annotations), 1)

        workbench.zoom_in()
        self.assertNotEqual(workbench.state.view_limits, original_limits)
        workbench.state.pan(0.1, 0.1)
        workbench.reset_view()
        self.assertEqual(workbench.state.view_limits, original_limits)

        workbench.delete_selected()
        self.assertEqual(workbench.state.curves, [])
        self.assertEqual(workbench.state.annotations, [])

    def test_hover_crosshair_and_movable_legend_report_curve_coordinates(self):
        project = self.store.open_project(self.multi_project.path, touch=False)
        self.app.set_project(project, target_page="inference")
        self.app.update_idletasks()
        self.page = self.app.inference_page
        self._fill_inputs()
        self.page.predict_button.invoke()
        workbench = self.page.response_plot
        workbench.redraw()
        curve = workbench.state.selected_curve
        left, top, right, bottom = workbench._plot_bounds
        x_min, x_max, y_min, y_max = workbench.state.view_limits
        point_x = left + (curve.x_values[0] - x_min) / (x_max - x_min) * (right - left)
        point_y = bottom - (curve.y_values[0] - y_min) / (y_max - y_min) * (bottom - top)

        workbench._on_motion(SimpleNamespace(x=point_x, y=point_y))
        self.assertIn(curve.name, workbench.hover_details)
        self.assertIn("X", workbench.hover_details)
        self.assertIn("Y", workbench.hover_details)
        self.assertIsNotNone(workbench._hover_point)

        old_position = workbench.state.legend_position
        x1, y1, x2, y2 = workbench._legend_bounds
        workbench._on_press(SimpleNamespace(x=(x1 + x2) / 2, y=(y1 + y2) / 2))
        workbench._on_drag(SimpleNamespace(x=left + 80, y=top + 80))
        workbench._on_release(SimpleNamespace())
        self.assertNotEqual(workbench.state.legend_position, old_position)

    def test_snowbuddy_can_minimize_and_restore_without_losing_context(self):
        project_path = self.app.snowbuddy_panel.current_project.path

        self.app.snowbuddy_panel.collapse_button.invoke()
        self.app.update_idletasks()

        self.assertTrue(self.app.snowbuddy_collapsed)
        self.assertEqual(self.app.snowbuddy_panel.winfo_manager(), "")
        self.assertEqual(self.app.snowbuddy_restore_button.winfo_manager(), "grid")
        self.assertEqual(self.app.snowbuddy_panel.current_project.path, project_path)

        self.app.snowbuddy_restore_button.invoke()
        self.app.update_idletasks()
        self.assertFalse(self.app.snowbuddy_collapsed)
        self.assertEqual(self.app.snowbuddy_panel.winfo_manager(), "grid")

    def test_workflow_sidebar_collapses_to_icons_and_navigation_still_works(self):
        self.app.sidebar_toggle_button.invoke()
        self.app.update_idletasks()

        self.assertTrue(self.app.sidebar_collapsed)
        self.assertEqual(self.app.sidebar.cget("width"), 76)
        self.assertEqual(self.app.sidebar_workflow_label.winfo_manager(), "")
        self.assertEqual(self.app.sidebar_project_shell.winfo_manager(), "")
        for name, button in self.app.nav_buttons.items():
            icon, label = self.app.nav_specs[name]
            self.assertEqual(button.cget("text"), icon)
            self.assertNotIn(label, button.cget("text"))
            self.assertEqual(button.accessible_name, label)
            self.assertEqual(self.app.nav_tooltips[label].text, label)
            self.assertTrue(self.app.nav_tooltips[label].enabled())

        self.app.nav_buttons["data"].invoke()
        self.assertEqual(self.app.active_page, "data")
        self.assertTrue(self.app.sidebar_collapsed)
        self.app.nav_buttons["inference"].invoke()
        self.assertEqual(self.app.active_page, "inference")
        self.assertTrue(self.app.sidebar_collapsed)

        self.app.sidebar_toggle_button.invoke()
        self.app.update_idletasks()
        self.assertFalse(self.app.sidebar_collapsed)
        self.assertEqual(self.app.sidebar.cget("width"), 226)
        self.assertIn("Inference", self.app.nav_buttons["inference"].cget("text"))

    def test_snowbuddy_uses_top_bar_launcher_and_docked_non_overlay_panel(self):
        self.app.geometry("1920x900+0+0")
        self.app.deiconify()
        self.app.set_sidebar_collapsed(True)
        self.app.set_snowbuddy_collapsed(True)
        self.app.update()

        self.assertEqual(self.app.snowbuddy_panel.winfo_manager(), "")
        self.assertEqual(self.app.snowbuddy_restore_button.winfo_manager(), "grid")
        launcher_grid = self.app.snowbuddy_restore_button.grid_info()
        self.assertEqual(int(launcher_grid["row"]), 0)
        self.assertEqual(int(launcher_grid["column"]), 4)
        self.assertEqual(set(launcher_grid["sticky"]), {"e"})
        self.assertEqual(self.app.snowbuddy_restore_button.accessible_name, "Open SnowBuddy")
        self.assertEqual(
            int(self.app.workspace.grid_columnconfigure(1)["minsize"]),
            0,
        )

        self.app.snowbuddy_restore_button.invoke()
        self.app.update()
        self.assertEqual(self.app.snowbuddy_panel.winfo_manager(), "grid")
        self.assertEqual(self.app.snowbuddy_display_mode, "docked")
        drawer_grid = self.app.snowbuddy_panel.grid_info()
        self.assertEqual(int(drawer_grid["column"]), 1)
        self.assertEqual(set(drawer_grid["sticky"]), {"n", "s", "e", "w"})
        self.assertEqual(
            int(self.app.workspace.grid_columnconfigure(1)["minsize"]),
            390,
        )
        self.assertEqual(self.app.snowbuddy_restore_button.accessible_name, "Close SnowBuddy")
        self.app.withdraw()

    def test_laptop_layout_keeps_snowbuddy_and_plot_settings_clear_of_actions(self):
        self.app.geometry("1366x768+0+0")
        self.app.deiconify()
        self.app.set_sidebar_collapsed(True)
        self.app.set_snowbuddy_collapsed(True)
        self.app.update()
        self.assertGreaterEqual(
            self.app.workspace.winfo_rooty(),
            self.app.snowbuddy_restore_button.winfo_rooty()
            + self.app.snowbuddy_restore_button.winfo_height(),
        )

        project = self.store.open_project(self.multi_project.path, touch=False)
        self.app.set_project(project, target_page="inference")
        self._fill_inputs()
        self.page.predict_button.invoke()
        self.app.update()
        workbench = self.page.response_plot
        plot_width = workbench.canvas.winfo_width()
        self.assertGreaterEqual(plot_width, 420)
        self.assertLess(
            workbench._plot_bounds[3],
            workbench.canvas.winfo_height(),
        )

        self.app.set_snowbuddy_collapsed(False)
        self.app.update()
        self.assertEqual(self.app.snowbuddy_display_mode, "focus")
        self.assertEqual(self.app.page_host.winfo_manager(), "")
        self.assertEqual(int(self.app.snowbuddy_panel.grid_info()["column"]), 0)
        self.app.set_snowbuddy_collapsed(True)
        self.app.update()
        self.assertEqual(self.app.page_host.winfo_manager(), "grid")
        self.assertGreaterEqual(workbench.canvas.winfo_width(), plot_width)

        workbench.plot_settings_button.invoke()
        dialog = workbench.plot_settings_dialog
        dialog.geometry("700x540+20+20")
        for tab_name in ("Axes & Grid", "Text & Legend", "Selected Curve"):
            dialog.settings_tabs.set(tab_name)
            scroll = dialog.settings_scrolls[tab_name]
            dialog.update()
            scroll._parent_canvas.yview_moveto(1.0)
            dialog.update()
            actions_bottom = dialog.actions.winfo_y() + dialog.actions.winfo_height()
            self.assertLessEqual(actions_bottom, dialog.winfo_height())
            self.assertEqual(dialog.apply_button.winfo_manager(), "pack")
            self.assertEqual(dialog.cancel_button.winfo_manager(), "pack")
            self.assertGreater(dialog.apply_button.winfo_reqwidth(), 1)
            self.assertGreater(dialog.cancel_button.winfo_reqwidth(), 1)
        dialog.settings_tabs.set("Text & Legend")
        text_scroll = dialog.settings_scrolls["Text & Legend"]
        dialog.update()
        text_scroll._parent_canvas.yview_moveto(1.0)
        dialog.update()
        self.assertLessEqual(
            dialog.show_legend_checkbox.winfo_rooty()
            + dialog.show_legend_checkbox.winfo_height(),
            text_scroll.winfo_rooty() + text_scroll.winfo_height(),
        )
        dialog.destroy()
        workbench.plot_settings_dialog = None
        self.app.withdraw()

    def test_success_closes_snowbuddy_only_when_plot_canvas_cannot_render(self):
        self.app.geometry("1260x800+0+0")
        self.app.deiconify()
        self.app.set_sidebar_collapsed(False)
        self.app.set_snowbuddy_collapsed(False)
        project = self.store.open_project(self.multi_project.path, touch=False)
        self.app.set_project(project, target_page="inference")
        self.app.update()
        self.page = self.app.inference_page
        self._fill_inputs()

        self.page.predict_button.invoke()
        self.app.update_idletasks()

        self.assertTrue(self.page.last_result.success)
        self.assertTrue(self.app.snowbuddy_collapsed)
        self.assertTrue(self.page.response_plot.canvas.winfo_ismapped())
        self.assertGreater(self.page.response_plot.canvas.winfo_width(), 1)
        self.assertIn(
            "SnowBuddy closed to show plot",
            self.page.footer_status.cget("text"),
        )
        self.app.withdraw()

    def test_success_keeps_snowbuddy_open_when_plot_has_rendering_space(self):
        self.app.geometry("1920x900+0+0")
        self.app.deiconify()
        self.app.set_sidebar_collapsed(True)
        self.app.set_snowbuddy_collapsed(False)
        project = self.store.open_project(self.multi_project.path, touch=False)
        self.app.set_project(project, target_page="inference")
        self.app.update()
        self.page = self.app.inference_page
        self._fill_inputs()

        self.page.predict_button.invoke()
        self.app.update_idletasks()

        self.assertTrue(self.page.last_result.success)
        self.assertFalse(self.app.snowbuddy_collapsed)
        self.assertTrue(self.page.response_plot.canvas.winfo_ismapped())
        self.assertGreater(self.page.response_plot.canvas.winfo_width(), 1)
        self.assertNotIn(
            "SnowBuddy closed to show plot",
            self.page.footer_status.cget("text"),
        )
        self.app.withdraw()

    def test_plot_settings_dialog_exposes_global_and_selected_curve_controls(self):
        project = self.store.open_project(self.multi_project.path, touch=False)
        self.app.set_project(project, target_page="inference")
        self.app.update_idletasks()
        self.page = self.app.inference_page
        self._fill_inputs()
        self.page.predict_button.invoke()
        workbench = self.page.response_plot

        workbench.plot_settings_button.invoke()
        self.app.update_idletasks()
        dialog = workbench.plot_settings_dialog

        self.assertIsNotNone(dialog)
        self.assertEqual(dialog.title(), "Plot Settings")
        self.assertEqual(
            set(dialog.settings_tabs._tab_dict),
            {"Axes & Grid", "Text & Legend", "Selected Curve"},
        )
        self.assertEqual(dialog.x_scale_menu.get(), "Linear")
        self.assertEqual(dialog.y_scale_menu.get(), "Linear")
        self.assertEqual(dialog.legend_location_menu.get(), "Upper right")
        self.assertEqual(dialog.x_label_font_size_entry.get(), "14")
        self.assertEqual(dialog.y_label_font_size_entry.get(), "14")
        self.assertEqual(dialog.x_value_font_size_entry.get(), "11")
        self.assertEqual(dialog.y_value_font_size_entry.get(), "11")
        self.assertEqual(dialog.legend_font_size_entry.get(), "11")
        self.assertEqual(dialog.legend_line_width_entry.get(), "2")
        self.assertEqual(dialog.line_style_menu.get(), "Solid")
        self.assertEqual(dialog.marker_style_menu.get(), "Circle")
        self.assertEqual(dialog.apply_button.winfo_manager(), "pack")
        self.assertEqual(dialog.cancel_button.winfo_manager(), "pack")

        dialog.plot_title_entry.delete(0, "end")
        dialog.plot_title_entry.insert(0, "Custom Response")
        dialog.legend_visible_var.set(False)
        dialog.legend_location_menu.set("Lower left")
        for entry, value in (
            (dialog.plot_title_font_size_entry, "19"),
            (dialog.x_label_font_size_entry, "14"),
            (dialog.y_label_font_size_entry, "15"),
            (dialog.x_value_font_size_entry, "10"),
            (dialog.y_value_font_size_entry, "11"),
            (dialog.legend_font_size_entry, "13"),
            (dialog.legend_line_width_entry, "4.5"),
        ):
            entry.delete(0, "end")
            entry.insert(0, value)
        dialog.line_width_entry.delete(0, "end")
        dialog.line_width_entry.insert(0, "4")
        dialog.line_style_menu.set("Dashed")
        dialog.marker_style_menu.set("Square")
        dialog.marker_size_entry.delete(0, "end")
        dialog.marker_size_entry.insert(0, "5")
        dialog._apply()

        curve = workbench.state.selected_curve
        self.assertEqual(workbench.state.plot_title, "Custom Response")
        self.assertFalse(workbench.state.legend_visible)
        self.assertEqual(workbench.state.legend_location, "Lower left")
        self.assertEqual(workbench.state.plot_title_font_size, 19)
        self.assertEqual(workbench.state.x_label_font_size, 14)
        self.assertEqual(workbench.state.y_label_font_size, 15)
        self.assertEqual(workbench.state.x_value_font_size, 10)
        self.assertEqual(workbench.state.y_value_font_size, 11)
        self.assertEqual(workbench.state.legend_font_size, 13)
        self.assertEqual(workbench.state.legend_line_width, 4.5)
        self.assertEqual(curve.line_width, 4.0)
        self.assertEqual(curve.line_style, "Dashed")
        self.assertEqual(curve.marker_style, "Square")
        self.assertEqual(curve.marker_size, 5.0)

    def test_raw_values_show_complete_inputs_and_saved_output_order(self):
        project = self.store.open_project(self.multi_project.path, touch=False)
        self.app.set_project(project, target_page="inference")
        self.app.update_idletasks()
        self.page = self.app.inference_page
        self._fill_inputs()
        self.page.predict_button.invoke()

        self.page.raw_values_button.invoke()
        self.app.update_idletasks()

        dialog = self.page.raw_values_dialog
        self.assertIsNotNone(dialog)
        raw_text = dialog.raw_text
        self.assertIn("INPUTS USED (3)", raw_text)
        self.assertIn("P2 = 4", raw_text)
        self.assertIn("P3 = 2", raw_text)
        self.assertIn("P4 = 3", raw_text)
        self.assertIn("PREDICTED OUTPUTS (12)", raw_text)
        positions = [raw_text.index(f"theta_{index} =") for index in range(12)]
        self.assertEqual(positions, sorted(positions))
        dialog.destroy()
        self.page.raw_values_dialog = None

    def test_export_prediction_contains_model_inputs_and_ordered_outputs(self):
        project = self.store.open_project(self.multi_project.path, touch=False)
        self.app.set_project(project, target_page="inference")
        self.app.update_idletasks()
        self.page = self.app.inference_page
        self._fill_inputs()
        self.page.predict_button.invoke()
        export_path = Path(self.temp_dir.name) / "ordered_prediction.json"

        with (
            patch(
                "studio.inference_ui.filedialog.asksaveasfilename",
                return_value=str(export_path),
            ),
            patch("studio.inference_ui.messagebox.showinfo") as showinfo,
        ):
            self.page.export_button.invoke()

        payload = json.loads(export_path.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["export_type"],
            "antenna_surrogate_studio_prediction",
        )
        self.assertEqual(payload["model_book"]["book_id"], self.multi_book.book_id)
        self.assertEqual(payload["model_book"]["name"], "Response Model")
        self.assertEqual(
            [item["name"] for item in payload["inputs"]],
            ["P2", "P3", "P4"],
        )
        self.assertEqual(
            [item["value"] for item in payload["inputs"]],
            [4.0, 2.0, 3.0],
        )
        self.assertEqual(payload["output_count"], 12)
        self.assertEqual(
            [item["target"] for item in payload["predicted_outputs"]],
            self.page.last_result.target_order,
        )
        self.assertEqual(
            [item["value"] for item in payload["predicted_outputs"]],
            list(self.page.last_result.predictions.values()),
        )
        self.assertEqual(payload["output_axis"]["label"], "Theta")
        self.assertEqual(payload["output_axis"]["source"], "target_columns")
        self.assertEqual(
            payload["output_axis"]["values"],
            [float(index) for index in range(12)],
        )
        showinfo.assert_called_once()

    def test_curve_csv_export_uses_saved_axis_and_target_order(self):
        project = self.store.open_project(self.multi_project.path, touch=False)
        self.app.set_project(project, target_page="inference")
        self.app.update_idletasks()
        self.page = self.app.inference_page
        self._fill_inputs()
        self.page.predict_button.invoke()
        export_path = Path(self.temp_dir.name) / "ordered_curve.csv"

        with (
            patch(
                "studio.inference_ui.filedialog.asksaveasfilename",
                return_value=str(export_path),
            ),
            patch("studio.inference_ui.messagebox.showinfo"),
        ):
            self.page.export_button.invoke()

        with export_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0], ["Theta", "Predicted value", "Output variable"])
        self.assertEqual([row[2] for row in rows[1:]], self.page.last_result.target_order)
        self.assertEqual([float(row[0]) for row in rows[1:]], list(range(12)))
        self.assertEqual(
            [float(row[1]) for row in rows[1:]],
            list(self.page.last_result.predictions.values()),
        )

    def test_missing_and_invalid_numeric_values_show_clear_errors(self):
        with patch("studio.inference_ui.submit_inference_request") as submit:
            self.page.predict_button.invoke()
            self.assertIn("Enter a value for P2", self.page.input_error.cget("text"))
            submit.assert_not_called()

            self._fill_inputs()
            self.page.input_entries["P3"].delete(0, "end")
            self.page.input_entries["P3"].insert(0, "not-a-number")
            self.page.predict_button.invoke()
            self.assertIn("P3 must be a numeric value", self.page.input_error.cget("text"))
            submit.assert_not_called()

    def test_no_active_model_disables_prediction_with_guidance(self):
        project = self.store.open_project(self.empty_project.path, touch=False)
        self.app.set_project(project, target_page="inference")
        self.app.update_idletasks()
        page = self.app.inference_page

        self.assertIsNone(page.active_book)
        self.assertEqual(page.predict_button.cget("state"), "disabled")
        self.assertEqual(page.raw_values_button.cget("state"), "disabled")
        self.assertEqual(page.export_button.cget("state"), "disabled")
        self.assertIn("No active Model Book", page.input_error.cget("text"))
        self.assertEqual(page.result_title.cget("text"), "Prediction unavailable")

    def test_corrupted_active_model_is_reported_without_traceback(self):
        project = self.store.open_project(self.corrupt_project.path, touch=False)
        self.app.set_project(project, target_page="inference")
        self.app.update_idletasks()
        page = self.app.inference_page

        self.assertIsNone(page.active_book)
        self.assertEqual(page.predict_button.cget("state"), "disabled")
        self.assertEqual(page.raw_values_button.cget("state"), "disabled")
        self.assertEqual(page.export_button.cget("state"), "disabled")
        message = page.result_summary.cget("text")
        self.assertIn("failed its integrity check", message)
        self.assertNotIn("Traceback", message)

    def test_backend_failure_is_shown_and_predict_is_reenabled(self):
        self._fill_inputs()
        failure = InferenceResult(
            success=False,
            status=INFERENCE_FAILED,
            model_book_id=self.single_book.book_id,
            model_book_name=None,
            model_name=None,
            error_message="The active model could not predict this sample.",
        )

        with patch(
            "studio.inference_ui.submit_inference_request",
            return_value=failure,
        ):
            self.page.predict_button.invoke()

        self.assertEqual(self.page.result_title.cget("text"), "Prediction failed")
        self.assertIn("could not predict", self.page.result_summary.cget("text"))
        self.assertEqual(self.page.predict_button.cget("state"), "normal")
        self.assertEqual(self.page.raw_values_button.cget("state"), "disabled")
        self.assertEqual(self.page.export_button.cget("state"), "disabled")
        self.assertFalse(self.page.prediction_in_progress)

    def test_predict_button_lifecycle(self):
        self._fill_inputs()
        observed = {}
        success = InferenceResult(
            success=True,
            status=INFERENCE_COMPLETED,
            model_book_id=self.single_book.book_id,
            model_book_name=self.single_book.name,
            model_name="linear_regression",
            feature_order=["P2", "P3", "P4"],
            target_order=["gain"],
            input_values={"P2": 4.0, "P3": 2.0, "P4": 3.0},
            predictions={"gain": 13.5},
        )

        def observe(request, *, project_path):
            observed["state"] = self.page.predict_button.cget("state")
            observed["text"] = self.page.predict_button.cget("text")
            observed["busy"] = self.page.prediction_in_progress
            return success

        with patch(
            "studio.inference_ui.submit_inference_request",
            side_effect=observe,
        ):
            self.page.predict_button.invoke()

        self.assertEqual(observed["state"], "disabled")
        self.assertEqual(observed["text"], "Predicting…")
        self.assertTrue(observed["busy"])
        self.assertEqual(self.page.predict_button.cget("state"), "normal")
        self.assertEqual(self.page.predict_button.cget("text"), "Predict")
        self.assertFalse(self.page.prediction_in_progress)

    def test_last_page_inference_is_restored_after_project_reopen(self):
        self.app.show_page("inference")
        reopened = self.store.open_project(self.single_project.path, touch=False)

        self.assertEqual(reopened.manifest["ui"]["last_page"], "inference")
        self.app.set_project(reopened)
        self.app.update_idletasks()
        self.assertEqual(self.app.active_page, "inference")

    def test_start_resume_button_uses_model_saved_stage(self):
        self.app.show_page("start")
        self.app.start_page.refresh()

        self.assertIn("Run Inference", self.app.start_page.continue_project_button.cget("text"))
        self.assertIn("Run Inference", self.app.start_page.next_page_button.cget("text"))
        self.assertIn(__version__, self.app.sidebar_version_label.cget("text"))
        self.app.start_page.continue_project_button.invoke()

        self.assertEqual(self.app.active_page, "inference")

        self.app.show_page("start")
        self.app.start_page.next_page_button.invoke()
        self.assertEqual(self.app.active_page, "inference")

    def test_in_memory_plot_workspace_survives_page_navigation_for_same_book(self):
        self._fill_inputs()
        self.page.predict_button.invoke()
        curve_id = self.page.response_plot.state.selected_curve_id

        self.app.show_page("library")
        self.app.show_page("inference")
        self.app.update_idletasks()

        self.assertEqual(len(self.page.response_plot.state.curves), 1)
        self.assertEqual(self.page.response_plot.state.selected_curve_id, curve_id)
        self.assertTrue(self.page.last_result.success)


if __name__ == "__main__":
    unittest.main()
