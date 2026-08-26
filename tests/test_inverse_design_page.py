import os
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

from studio.inverse_design import (
    INVERSE_DESIGN_COMPLETED,
    INVERSE_DESIGN_FAILED,
    InverseDesignResult,
)
from studio.inverse_design_ui import CONFIGURATION_MIN_WIDTH, RESULT_MIN_WIDTH
from studio.project_store import ProjectStore
from studio.scientific_plot import CURVE_MANAGER_MIN_WIDTH, PLOT_PANE_MIN_WIDTH
from studio.ui import StudioApp
from tests.test_inference_page import create_active_book


GUI_MAY_BE_AVAILABLE = (
    os.name == "nt"
    or os.sys.platform == "darwin"
    or bool(os.environ.get("DISPLAY"))
)


class _ImmediateThread:
    def __init__(self, *, target, daemon):
        self.target = target
        self.daemon = daemon

    def start(self):
        self.target()


@unittest.skipUnless(GUI_MAY_BE_AVAILABLE, "A desktop display is required.")
class InverseDesignPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        cls.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        cls.store = ProjectStore(Path(cls.temp_dir.name) / "library")
        cls.project = cls.store.create_project("Inverse Design Page")
        cls.book = create_active_book(
            cls.project,
            output_count=8,
            name="Inverse Response Model",
        )
        cls.empty_project = cls.store.create_project("Inverse Design Empty")
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
        project = self.store.open_project(self.project.path, touch=False)
        self.app.set_project(project, target_page="inverse_design")
        self.app.update_idletasks()
        self.page = self.app.inverse_design_page

    def _fill_valid_form(self):
        for index, (name, widgets) in enumerate(self.page.input_widgets.items()):
            if index == 0:
                widgets.mode_control.set("Variable")
                widgets.mode.set("Variable")
                self.page._input_mode_changed(name)
                widgets.lower.delete(0, "end")
                widgets.lower.insert(0, "0")
                widgets.upper.delete(0, "end")
                widgets.upper.insert(0, "10")
            else:
                widgets.mode_control.set("Fixed")
                widgets.mode.set("Fixed")
                self.page._input_mode_changed(name)
                widgets.fixed.delete(0, "end")
                widgets.fixed.insert(0, str(index + 1))

    def test_page_loads_active_book_and_dynamic_saved_inputs(self):
        self.assertIn("inverse_design", self.app.pages)
        self.assertIn("inverse_design", self.app.nav_buttons)
        self.assertEqual(self.app.active_page, "inverse_design")
        self.assertEqual(self.page.active_book.book_id, self.book.book_id)
        self.assertEqual(list(self.page.input_widgets), ["P2", "P3", "P4"])
        self.assertFalse(hasattr(self.page, "objective_output"))
        self.assertEqual(self.page.objective_scope.get(), "Single point")
        self.assertEqual(self.page.single_coordinate.get(), "0")
        self.assertIn("enter coordinates directly", self.page.axis_help.cget("text"))
        self.assertEqual(self.page.objective_goal.get(), "Minimize")
        self.assertIn("Differential Evolution", self.page.footer_status.cget("text"))
        self.assertEqual(self.page.run_button.cget("state"), "normal")

    def test_form_builds_variable_fixed_target_and_generic_constraint_request(self):
        self._fill_valid_form()
        self.page.single_coordinate.delete(0, "end")
        self.page.single_coordinate.insert(0, "3")
        self.page.objective_goal.set("Target value")
        self.page._goal_changed("Target value")
        self.page.target_value.insert(0, "4.5")
        self.page._add_constraint()
        row = self.page.constraint_widgets[0]
        row.coordinate_start.delete(0, "end")
        row.coordinate_start.insert(0, "7")
        row.operator.set("Within range")
        self.page._constraint_operator_changed(0)
        row.first_value.insert(0, "1")
        row.second_value.insert(0, "8")

        request = self.page.build_request()

        self.assertEqual(request.variable_bounds, {"P2": (0.0, 10.0)})
        self.assertEqual(request.fixed_inputs, {"P3": 2.0, "P4": 3.0})
        self.assertEqual(request.objective.output_name, "theta_3")
        self.assertEqual(request.objective.output_names, ["theta_3"])
        self.assertEqual(request.objective.aggregation, "single")
        self.assertEqual(request.objective.goal, "target")
        self.assertEqual(request.objective.target_value, 4.5)
        self.assertEqual(request.constraints[0].operator, "within_range")
        self.assertEqual(request.constraints[0].output_name, "theta_7")
        self.assertEqual(request.constraints[0].aggregation, "single")
        self.assertEqual(request.constraints[0].lower_bound, 1.0)
        self.assertEqual(request.constraints[0].upper_bound, 8.0)

    def test_mean_over_coordinate_range_builds_one_ordered_scalar_objective(self):
        self._fill_valid_form()
        self.page.objective_scope.set("Mean over range")
        self.page._objective_scope_changed("Mean over range")
        self.page.range_start.delete(0, "end")
        self.page.range_start.insert(0, "2")
        self.page.range_end.delete(0, "end")
        self.page.range_end.insert(0, "5")

        request = self.page.build_request()

        self.assertEqual(request.objective.aggregation, "mean")
        self.assertIsNone(request.objective.output_name)
        self.assertEqual(
            request.objective.output_names,
            ["theta_2", "theta_3", "theta_4", "theta_5"],
        )

    def test_invalid_coordinate_range_is_rejected_without_output_scrolling(self):
        self._fill_valid_form()
        self.page.objective_scope.set("Mean over range")
        self.page._objective_scope_changed("Mean over range")
        self.page.range_start.delete(0, "end")
        self.page.range_start.insert(0, "9")
        self.page.range_end.delete(0, "end")
        self.page.range_end.insert(0, "3")

        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            self.page.build_request()

    def test_mean_constraint_uses_ordered_axis_range_without_output_scrolling(self):
        self._fill_valid_form()
        self.page._add_constraint()
        row = self.page.constraint_widgets[0]
        row.scope.set("Mean over range")
        self.page._constraint_scope_changed(0)
        row.coordinate_start.delete(0, "end")
        row.coordinate_start.insert(0, "2")
        row.coordinate_end.delete(0, "end")
        row.coordinate_end.insert(0, "5")
        row.operator.set("At least (≥)")
        row.first_value.insert(0, "-10")

        request = self.page.build_request()

        constraint = request.constraints[0]
        self.assertEqual(constraint.aggregation, "mean")
        self.assertIsNone(constraint.output_name)
        self.assertEqual(
            constraint.output_names,
            ["theta_2", "theta_3", "theta_4", "theta_5"],
        )

    def test_invalid_form_does_not_submit_and_shows_friendly_message(self):
        with patch("studio.inverse_design_ui.submit_inverse_design_request") as submit:
            self.page.run_button.invoke()

        submit.assert_not_called()
        self.assertIn("lower bound", self.page.footer_status.cget("text"))
        self.assertNotIn("Traceback", self.page.footer_status.cget("text"))

    def test_run_button_lifecycle_and_success_result_plot(self):
        self._fill_valid_form()
        observed = {}
        predictions = {f"theta_{index}": float(index + 1) for index in range(8)}
        success = InverseDesignResult(
            success=True,
            status=INVERSE_DESIGN_COMPLETED,
            model_book_id=self.book.book_id,
            model_book_name=self.book.name,
            model_name="linear_regression",
            run_id="inverse-0001",
            best_inputs={"P2": 4.0, "P3": 2.0, "P4": 3.0},
            predicted_outputs=predictions,
            objective={
                "output_name": "theta_0",
                "output_names": ["theta_0"],
                "aggregation": "single",
                "goal": "minimize",
                "target_value": None,
            },
            objective_value=1.0,
            objective_score=1.0,
            feasible=True,
            evaluations=120,
            iterations=18,
        )

        def observe(request, *, project_path):
            observed["state"] = self.page.run_button.cget("state")
            observed["text"] = self.page.run_button.cget("text")
            observed["busy"] = self.page.optimization_in_progress
            return success

        with (
            patch(
                "studio.inverse_design_ui.submit_inverse_design_request",
                side_effect=observe,
            ),
            patch("studio.inverse_design_ui.threading.Thread", _ImmediateThread),
        ):
            self.page.run_button.invoke()
            self.app.update()

        self.assertEqual(observed["state"], "disabled")
        self.assertEqual(observed["text"], "Optimizing…")
        self.assertTrue(observed["busy"])
        self.assertEqual(self.page.run_button.cget("state"), "normal")
        self.assertEqual(self.page.run_button.cget("text"), "Run Inverse Design")
        self.assertFalse(self.page.optimization_in_progress)
        self.assertEqual(self.page.config_section_control.get(), "Inputs")
        self.assertEqual(self.page.configuration_card.winfo_manager(), "panedwindow")
        self.assertEqual(self.page.response_plot.winfo_manager(), "grid")
        self.assertEqual(self.page.result_badge.cget("text"), "OPTIMIZED")
        self.assertEqual(self.page.feasible_metric.cget("text"), "Not used")
        self.assertEqual(len(self.page.response_plot.state.curves), 1)
        self.assertIn("inverse-0001", self.page.response_plot.state.selected_curve.name)
        self.assertEqual(
            self.page.response_plot.state.selected_curve.inputs,
            success.best_inputs,
        )
        legend_text = {
            self.page.response_plot.canvas.itemcget(item, "text")
            for item in self.page.response_plot.canvas.find_all()
            if self.page.response_plot.canvas.type(item) == "text"
        }
        self.assertIn(
            self.page.response_plot.state.selected_curve.name,
            legend_text,
        )

    def test_repeated_search_results_add_or_replace_curves_without_leaving_configuration(self):
        predictions = {f"theta_{index}": float(index + 1) for index in range(8)}
        first = InverseDesignResult(
            success=True,
            status=INVERSE_DESIGN_COMPLETED,
            model_book_id=self.book.book_id,
            run_id="inverse-0001",
            best_inputs={"P2": 1.0, "P3": 2.0, "P4": 3.0},
            predicted_outputs=predictions,
            objective={
                "output_name": "theta_0",
                "output_names": ["theta_0"],
                "aggregation": "single",
                "goal": "minimize",
                "target_value": None,
            },
            objective_value=1.0,
            feasible=True,
        )
        second = InverseDesignResult(
            success=True,
            status=INVERSE_DESIGN_COMPLETED,
            model_book_id=self.book.book_id,
            run_id="inverse-0002",
            best_inputs={"P2": 5.0, "P3": 2.0, "P4": 3.0},
            predicted_outputs={name: value + 2 for name, value in predictions.items()},
            objective={
                "output_name": None,
                "output_names": ["theta_2", "theta_3", "theta_4"],
                "aggregation": "mean",
                "goal": "maximize",
                "target_value": None,
            },
            objective_value=6.0,
            feasible=True,
        )

        self.page.plot_action.set("Add to plot")
        self.page._show_success(first)
        self.page._show_success(second)
        self.assertEqual(len(self.page.response_plot.state.curves), 2)
        self.assertIn("Mean", self.page.response_plot.state.selected_curve.name)
        self.assertEqual(self.page.configuration_card.winfo_manager(), "panedwindow")

        self.page.plot_action.set("Replace selected curve")
        second.run_id = "inverse-0003"
        self.page._show_success(second)
        self.assertEqual(len(self.page.response_plot.state.curves), 2)
        self.assertEqual(
            self.page.response_plot.state.selected_curve.inputs,
            second.best_inputs,
        )
        self.assertEqual(
            self.page.response_plot.state.selected_curve.y_values,
            tuple(second.predicted_outputs.values()),
        )

    def test_target_result_shows_achieved_value_gap_without_claiming_target_met(self):
        result = InverseDesignResult(
            success=True,
            status=INVERSE_DESIGN_COMPLETED,
            model_book_id=self.book.book_id,
            run_id="inverse-0010",
            best_inputs={"P2": 10.0, "P3": 2.0, "P4": 3.0},
            predicted_outputs={
                f"theta_{index}": float(index + 1) for index in range(8)
            },
            objective={
                "output_name": "theta_0",
                "output_names": ["theta_0"],
                "aggregation": "single",
                "goal": "target",
                "target_value": 5.0,
            },
            objective_value=1.0,
            objective_score=4.0,
            target_gap=4.0,
            feasible=True,
            evaluations=80,
            iterations=12,
        )

        self.page.last_result = result
        self.page._show_success(result)

        self.assertEqual(self.page.result_title.cget("text"), "Closest predicted design")
        self.assertEqual(self.page.result_badge.cget("text"), "CLOSEST FOUND")
        self.assertEqual(self.page.objective_metric_title.cget("text"), "ACHIEVED")
        self.assertEqual(self.page.objective_metric.cget("text"), "1")
        self.assertEqual(self.page.iteration_metric_title.cget("text"), "TARGET GAP")
        self.assertEqual(self.page.iteration_metric.cget("text"), "4")
        self.assertEqual(self.page.feasible_metric.cget("text"), "Not used")
        self.assertIn("requested 5", self.page.result_summary.cget("text"))
        self.assertNotIn("target met", self.page.result_summary.cget("text").lower())
        snowbuddy_state = "\n".join(self.page.describe_ui_state())
        self.assertIn("Latest inverse-design outcome: CLOSEST FOUND", snowbuddy_state)
        self.assertIn("Latest target gap: 4", snowbuddy_state)

        result.constraint_evaluations = [
            {
                "output_name": "theta_7",
                "operator": "less_than_or_equal",
                "predicted_value": 8.0,
                "satisfied": True,
                "violation": 0.0,
                "value": 10.0,
            }
        ]
        self.page._show_success(result)
        self.assertEqual(self.page.result_badge.cget("text"), "CONSTRAINTS MET")
        self.assertEqual(self.page.iteration_metric_title.cget("text"), "TARGET GAP")
        self.assertEqual(self.page.iteration_metric.cget("text"), "4")
        self.assertNotIn("target met", self.page.result_summary.cget("text").lower())

    def test_legacy_saved_target_result_derives_missing_target_gap(self):
        payload = {
            "success": True,
            "status": INVERSE_DESIGN_COMPLETED,
            "model_book_id": self.book.book_id,
            "run_id": "inverse-legacy-target",
            "best_inputs": {"P2": 10.0, "P3": 2.0, "P4": 3.0},
            "predicted_outputs": {
                f"theta_{index}": float(index + 1) for index in range(8)
            },
            "objective": {
                "output_name": "theta_0",
                "output_names": ["theta_0"],
                "aggregation": "single",
                "goal": "target",
                "target_value": 5.0,
            },
            "objective_value": 1.0,
            "objective_score": 4.0,
            "constraint_evaluations": [],
            "feasible": True,
        }

        restored = self.page._restored_result(payload)

        self.assertEqual(restored.target_gap, 4.0)

    def test_constraint_success_is_labeled_as_constraints_met(self):
        result = InverseDesignResult(
            success=True,
            status=INVERSE_DESIGN_COMPLETED,
            model_book_id=self.book.book_id,
            run_id="inverse-0011",
            best_inputs={"P2": 4.0, "P3": 2.0, "P4": 3.0},
            predicted_outputs={
                f"theta_{index}": float(index + 1) for index in range(8)
            },
            objective={
                "output_name": "theta_0",
                "output_names": ["theta_0"],
                "aggregation": "single",
                "goal": "minimize",
                "target_value": None,
            },
            objective_value=1.0,
            constraint_evaluations=[
                {
                    "output_name": "theta_7",
                    "operator": "less_than_or_equal",
                    "predicted_value": 8.0,
                    "satisfied": True,
                    "violation": 0.0,
                    "value": 10.0,
                }
            ],
            feasible=True,
        )

        self.page._show_success(result)

        self.assertEqual(
            self.page.result_title.cget("text"),
            "Best constraint-satisfying design",
        )
        self.assertEqual(self.page.result_badge.cget("text"), "CONSTRAINTS MET")
        self.assertEqual(self.page.feasible_metric.cget("text"), "1/1 met")

    def test_effectively_duplicate_response_warns_without_discarding_curve(self):
        predictions = {f"theta_{index}": float(index + 1) for index in range(8)}
        first = InverseDesignResult(
            success=True,
            status=INVERSE_DESIGN_COMPLETED,
            model_book_id=self.book.book_id,
            run_id="inverse-0020",
            best_inputs={"P2": 1.0, "P3": 2.0, "P4": 3.0},
            predicted_outputs=predictions,
            objective={
                "output_name": "theta_0",
                "output_names": ["theta_0"],
                "aggregation": "single",
                "goal": "minimize",
                "target_value": None,
            },
            objective_value=1.0,
            feasible=True,
        )
        second = InverseDesignResult(
            success=True,
            status=INVERSE_DESIGN_COMPLETED,
            model_book_id=self.book.book_id,
            run_id="inverse-0021",
            best_inputs={"P2": 1.00001, "P3": 2.0, "P4": 3.0},
            predicted_outputs={
                name: value + 0.00001 for name, value in predictions.items()
            },
            objective=dict(first.objective),
            objective_value=1.00001,
            feasible=True,
        )

        self.page.response_plot.state.clear_curves()
        self.page.response_plot._refresh_manager()
        self.page.plot_action.set("Add to plot")
        self.page._show_success(first)
        self.page._show_success(second)

        self.assertEqual(len(self.page.response_plot.state.curves), 2)
        self.assertIn("same design and response", self.page.result_summary.cget("text"))
        self.assertIn("matches plotted curve", self.page.footer_status.cget("text"))
        self.assertIn("0.01%", self.page.footer_status.cget("text"))

    def test_constraint_failure_clears_stale_metrics_but_preserves_plotted_curves(self):
        existing_curve_count = len(self.page.response_plot.state.curves)
        failure = InverseDesignResult(
            success=False,
            status=INVERSE_DESIGN_FAILED,
            model_book_id=self.book.book_id,
            error_message=(
                "No design satisfying all output constraints was found within the "
                "configured input bounds and search budget."
            ),
        )

        self.page._finish_optimization(failure)

        self.assertEqual(self.page.result_badge.cget("text"), "NO CONSTRAINT MATCH")
        self.assertEqual(self.page.objective_metric.cget("text"), "—")
        self.assertEqual(self.page.feasible_metric.cget("text"), "—")
        self.assertIn("No new design", self.page.latest_inputs.cget("text"))
        self.assertEqual(
            len(self.page.response_plot.state.curves),
            existing_curve_count,
        )
        self.assertNotIn("Traceback", self.page.footer_status.cget("text"))

    def test_latest_saved_result_rehydrates_live_state_and_snowbuddy_context(self):
        payload = {
            "success": True,
            "status": INVERSE_DESIGN_COMPLETED,
            "model_book_id": self.book.book_id,
            "model_book_name": self.book.name,
            "model_name": "linear_regression",
            "run_id": "inverse-0042",
            "best_inputs": {"P2": 4.0, "P3": 2.0, "P4": 3.0},
            "predicted_outputs": {
                f"theta_{index}": float(index + 1) for index in range(8)
            },
            "objective": {
                "output_name": "theta_0",
                "output_names": ["theta_0"],
                "aggregation": "single",
                "goal": "minimize",
                "target_value": None,
            },
            "objective_value": 1.0,
            "objective_score": 1.0,
            "constraint_evaluations": [],
            "feasible": True,
            "evaluations": 140,
            "iterations": 21,
            "artifact_directory": "inverse_design/runs/inverse-0042",
        }
        self.page.last_result = None
        self.page.response_plot.state.clear_curves()
        self.page.response_plot._refresh_manager()

        with patch(
            "studio.inverse_design_ui.load_inverse_design_run",
            return_value=payload,
        ):
            self.page._restore_latest_result()

        self.assertIsNotNone(self.page.last_result)
        self.assertEqual(self.page.last_result.run_id, "inverse-0042")
        self.assertEqual(self.page.last_result.status, INVERSE_DESIGN_COMPLETED)
        self.assertTrue(self.page.last_result.feasible)
        self.assertEqual(self.page.result_badge.cget("text"), "OPTIMIZED")
        self.assertEqual(len(self.page.response_plot.state.curves), 1)
        self.assertIn(
            f"Latest inverse-design result: {INVERSE_DESIGN_COMPLETED}",
            self.app.snowbuddy_ui_state(),
        )

    def test_optional_constraints_can_be_added_removed_and_are_bounded(self):
        for _ in range(6):
            self.page._add_constraint()
        self.assertEqual(len(self.page.constraint_widgets), 4)
        self.assertEqual(self.page.add_constraint_button.cget("state"), "disabled")
        self.page._remove_constraint(1)
        self.assertEqual(len(self.page.constraint_widgets), 3)
        self.assertEqual(self.page.add_constraint_button.cget("state"), "normal")

    def test_no_active_book_disables_search_with_guidance(self):
        project = self.store.open_project(self.empty_project.path, touch=False)
        self.app.set_project(project, target_page="inverse_design")
        self.app.update_idletasks()

        self.assertIsNone(self.page.active_book)
        self.assertEqual(self.page.run_button.cget("state"), "disabled")
        self.assertIn("No active Model Book", self.page.footer_status.cget("text"))

    def test_inference_has_forward_action_and_last_page_is_restored(self):
        project = self.store.open_project(self.project.path, touch=False)
        self.app.set_project(project, target_page="inference")
        self.app.update_idletasks()
        self.assertEqual(
            self.app.inference_page.inverse_design_button.cget("state"),
            "normal",
        )
        self.app.inference_page.inverse_design_button.invoke()
        self.assertEqual(self.app.active_page, "inverse_design")
        reopened = self.store.open_project(self.project.path, touch=False)
        self.assertEqual(reopened.manifest["ui"]["last_page"], "inverse_design")
        self.app.set_project(reopened)
        self.assertEqual(self.app.active_page, "inverse_design")

    def test_laptop_layout_keeps_configuration_and_actions_reachable(self):
        self.app.geometry("1366x768+0+0")
        self.app.deiconify()
        self.app.set_sidebar_collapsed(True)
        self.app.set_snowbuddy_collapsed(True)
        self.app.show_page("inverse_design")
        self.page.config_section_control.set("Constraints")
        self.page._show_config_section("Constraints")
        for _ in range(4):
            self.page._add_constraint()
        self.app.update()

        footer_top = self.page.run_button.winfo_rooty()
        last_row = self.page.constraint_widgets[-1].frame
        self.assertLessEqual(
            last_row.winfo_rooty() + last_row.winfo_height(),
            footer_top,
        )
        self.assertGreaterEqual(self.page.run_button.winfo_width(), 170)
        self.assertGreaterEqual(
            self.page.configuration_card.winfo_width(),
            CONFIGURATION_MIN_WIDTH,
        )
        self.assertGreaterEqual(self.page.result_card.winfo_width(), RESULT_MIN_WIDTH)
        self.assertGreaterEqual(self.page.response_plot.canvas.winfo_width(), 400)
        self.assertGreaterEqual(
            self.page.response_plot.curve_manager.winfo_width(),
            CURVE_MANAGER_MIN_WIDTH,
        )

        # Dragging fully left must stop at a width where labels and fields remain
        # readable; child controls must resize with the divider rather than clip.
        self.page.workspace_split.sash_place(0, 1, 1)
        self.page.config_section_control.set("Inputs")
        self.page._show_config_section("Inputs")
        self.app.update()
        minimum_configuration_width = self.page.configuration_card.winfo_width()
        self.assertGreaterEqual(
            minimum_configuration_width,
            CONFIGURATION_MIN_WIDTH,
        )
        first_input = next(iter(self.page.input_widgets.values()))
        self.assertGreaterEqual(first_input.mode_control.winfo_width(), 165)
        self.assertGreaterEqual(first_input.lower.winfo_width(), 54)
        self.assertGreaterEqual(first_input.upper.winfo_width(), 54)
        self.assertGreaterEqual(first_input.fixed.winfo_width(), 54)
        self.assertGreaterEqual(self.page.config_section_control.winfo_width(), 450)
        self.assertEqual(
            int(float(self.page.configuration_intro.cget("wraplength"))),
            minimum_configuration_width - 28,
        )

        # Mean-range controls and constraint menus are the longest controls in
        # the form, so verify them at the smallest permitted pane width.
        self.page.config_section_control.set("Objective")
        self.page._show_config_section("Objective")
        self.page.objective_scope.set("Mean over range")
        self.page._objective_scope_changed("Mean over range")
        self.app.update()
        self.assertGreaterEqual(self.page.objective_scope.winfo_width(), 450)
        self.assertGreaterEqual(self.page.range_start.winfo_width(), 210)
        self.assertGreaterEqual(self.page.range_end.winfo_width(), 210)

        self.page.config_section_control.set("Constraints")
        self.page._show_config_section("Constraints")
        self.app.update()
        first_constraint = self.page.constraint_widgets[0]
        first_constraint.scope.set("Mean over range")
        self.page._constraint_scope_changed(0)
        self.app.update()
        self.assertGreaterEqual(first_constraint.scope.winfo_width(), 145)
        self.assertGreaterEqual(first_constraint.operator.winfo_width(), 145)
        self.assertGreaterEqual(first_constraint.coordinate_start.winfo_width(), 110)
        self.assertGreaterEqual(first_constraint.coordinate_end.winfo_width(), 110)

        # The same responsive children must grow when the user gives the pane
        # more room.
        self.page.workspace_split.sash_place(
            0, self.page.workspace_split.winfo_width() - 1, 1
        )
        self.app.update()
        self.assertGreaterEqual(
            self.page.result_card.winfo_width(),
            RESULT_MIN_WIDTH,
        )
        self.assertGreaterEqual(
            self.page.response_plot.canvas.winfo_width(),
            PLOT_PANE_MIN_WIDTH,
        )
        self.page.workspace_split.sash_place(
            0, minimum_configuration_width + 40, 1
        )
        self.app.update()
        self.assertGreater(
            self.page.configuration_card.winfo_width(),
            minimum_configuration_width,
        )
        self.assertGreater(
            int(float(self.page.configuration_intro.cget("wraplength"))),
            minimum_configuration_width - 28,
        )

        # The curve manager has its own readable minimum, and its details text
        # follows the divider in both directions.
        plot_split_width = self.page.response_plot.plot_split.winfo_width()
        self.page.response_plot.plot_split.sash_place(
            0, plot_split_width - 1, 1
        )
        self.app.update()
        minimum_manager_width = self.page.response_plot.curve_manager.winfo_width()
        self.assertGreaterEqual(minimum_manager_width, CURVE_MANAGER_MIN_WIDTH)
        self.assertGreaterEqual(
            self.page.response_plot.canvas.winfo_width(),
            PLOT_PANE_MIN_WIDTH,
        )
        self.assertEqual(
            int(float(self.page.response_plot.selected_inputs.cget("wraplength"))),
            minimum_manager_width - 24,
        )
        self.page.response_plot.plot_split.sash_place(
            0, plot_split_width - minimum_manager_width - 70, 1
        )
        self.app.update()
        self.assertGreater(
            self.page.response_plot.curve_manager.winfo_width(),
            minimum_manager_width,
        )
        self.assertGreater(
            int(float(self.page.response_plot.selected_inputs.cget("wraplength"))),
            minimum_manager_width - 24,
        )

        # SnowBuddy must use the focused presentation on this compact page;
        # docking it would violate the verified form/result minima.
        self.app.set_snowbuddy_collapsed(False)
        self.app.update()
        self.assertEqual(self.app.snowbuddy_display_mode, "focus")
        self.assertEqual(self.app.page_host.winfo_manager(), "")
        self.app.set_snowbuddy_collapsed(True)
        self.app.update()
        self.assertEqual(self.app.page_host.winfo_manager(), "grid")

        # Normal development size uses the same constraints without clipping.
        self.app.geometry("1600x1000+0+0")
        self.app.update()
        self.assertGreaterEqual(
            self.page.configuration_card.winfo_width(), CONFIGURATION_MIN_WIDTH
        )
        self.assertGreaterEqual(
            self.page.response_plot.curve_manager.winfo_width(),
            CURVE_MANAGER_MIN_WIDTH,
        )
        self.app.withdraw()


if __name__ == "__main__":
    unittest.main()
