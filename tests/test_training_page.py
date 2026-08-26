import csv
import os
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

from studio.dataset_registry import register_dataset
from studio.dataset_validation import validate_dataset
from studio.model_training import (
    NEURAL_NETWORK_CUSTOM_DEFAULTS,
    NEURAL_NETWORK_CUSTOM_PARAMETER_NAMES,
    TRAINING_COMPLETED,
    XGBOOST_CUSTOM_DEFAULTS,
    XGBOOST_CUSTOM_PARAMETER_NAMES,
    ModelTrainingRequest,
    ModelTrainingResult,
)
from studio.parser_engine import TrainingRequest
from studio.project_store import ProjectStore
from studio.training_ui import (
    AUTO_SEARCH_LEVELS,
    SUPPORTED_MODELS,
    TRAIN_BUTTON_LABEL,
)
from studio.theme import COLORS
from studio.ui import StudioApp


GUI_MAY_BE_AVAILABLE = (
    os.name == "nt"
    or os.sys.platform == "darwin"
    or bool(os.environ.get("DISPLAY"))
)


@unittest.skipUnless(GUI_MAY_BE_AVAILABLE, "A desktop display is required.")
class ModelTrainingPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        cls.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        cls.store = ProjectStore(Path(cls.temp_dir.name) / "library")
        try:
            cls.app = StudioApp(project_store=cls.store)
        except tk.TclError as exc:
            cls.temp_dir.cleanup()
            raise unittest.SkipTest(
                f"A desktop display is not available: {exc}"
            ) from exc
        cls.app.withdraw()
        cls.project = cls.store.create_project("Training UI Test")
        input_path = cls.project.path / "data" / "prepared" / "inputs.csv"
        output_path = cls.project.path / "data" / "prepared" / "outputs.csv"
        with input_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Sample ID", "x1", "x2"])
            for index in range(1, 21):
                writer.writerow(
                    [f"Design_{index:03d}", index, (index * 3) % 7]
                )
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Sample ID", "target"])
            for index in range(1, 21):
                x2 = (index * 3) % 7
                writer.writerow(
                    [f"Design_{index:03d}", 2.0 * index - 0.5 * x2 + 3.0]
                )
        validation = validate_dataset(
            TrainingRequest(
                input_csv_path=input_path,
                output_csv_path=output_path,
                feature_columns=["x1", "x2"],
                target_columns=["target"],
                sample_id_column="Sample ID",
            )
        )
        register_dataset(cls.project.path, validation)
        cls.project = cls.store.open_project(cls.project.path, touch=False)
        cls.app.set_project(cls.project, target_page="training")
        cls.app.update_idletasks()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "app"):
            cls.app.destroy()
        if hasattr(cls, "temp_dir"):
            cls.temp_dir.cleanup()

    def setUp(self):
        self.page = self.app.training_page
        self.page._reset_ui_state()
        self.app.show_page("training", persist=False)
        self.app.update_idletasks()

    @staticmethod
    def _successful_result(
        run_number: int | None = None,
        *,
        training_mode: str = "auto",
        parameters_used: dict[str, bool] | None = None,
        search_level: str | None = None,
        configurations_evaluated: int | None = None,
        cross_validation_folds: int | None = None,
        best_validation_rmse: float | None = None,
    ) -> ModelTrainingResult:
        resolved_parameters = parameters_used or {
            "fit_intercept": True,
            "positive": False,
        }
        return ModelTrainingResult(
            success=True,
            status=TRAINING_COMPLETED,
            model_name="linear_regression",
            training_rows=16,
            test_rows=4,
            metrics={"MAE": 0.1, "RMSE": 0.2, "R²": 0.9},
            predictions=[],
            error_message=None,
            training_mode=training_mode,
            parameters_used=resolved_parameters,
            search_level=(
                search_level or "medium" if training_mode == "auto" else None
            ),
            configurations_evaluated=(
                configurations_evaluated
                if configurations_evaluated is not None
                else (2 if training_mode == "auto" else 0)
            ),
            cross_validation_folds=(
                cross_validation_folds
                if cross_validation_folds is not None
                else (3 if training_mode == "auto" else None)
            ),
            best_parameters=resolved_parameters,
            best_validation_rmse=(
                best_validation_rmse
                if best_validation_rmse is not None
                else (0.15 if training_mode == "auto" else None)
            ),
            test_metrics={"MAE": 0.1, "RMSE": 0.2, "R²": 0.9},
            run_number=run_number,
            run_id=f"run-{run_number:04d}" if run_number else None,
        )

    def test_model_training_page_renders(self):
        self.assertIn("training", self.app.pages)
        self.assertEqual(self.app.active_page, "training")
        self.assertTrue(self.page.winfo_exists())

    def test_training_page_uses_current_product_copy(self):
        texts: list[str] = []
        pending = [self.page]
        while pending:
            widget = pending.pop()
            pending.extend(widget.winfo_children())
            try:
                value = widget.cget("text")
            except (AttributeError, tk.TclError, ValueError):
                continue
            if value:
                texts.append(str(value))
        rendered = "\n".join(texts)
        self.assertIn("LOCAL  ·  MODEL TRAINING", rendered)
        self.assertNotIn("BASIC TRAINING", rendered)
        self.assertNotIn("More models can be added later", rendered)

    def test_linear_regression_appears_in_model_dropdown(self):
        self.assertEqual(
            SUPPORTED_MODELS,
            (
                "Linear Regression",
                "XGBoost",
                "Neural Network",
                "Ensemble AI Engine",
            ),
        )
        self.assertEqual(
            list(self.page.model_dropdown.cget("values")),
            [
                "Linear Regression",
                "XGBoost",
                "Neural Network",
                "Ensemble AI Engine",
            ],
        )

    def test_ensemble_selection_locks_auto_high_and_builds_request(self):
        self.page._model_changed("Ensemble AI Engine")

        request = self.page.build_model_training_request()

        self.assertEqual(request.model_name, "ensemble_ai_engine")
        self.assertEqual(request.training_mode, "auto")
        self.assertEqual(request.search_level, "high")
        self.assertIsNone(request.custom_hyperparameters)
        self.assertEqual(self.page.training_mode_control.cget("state"), "disabled")
        self.assertEqual(self.page.auto_search_frame.winfo_manager(), "")
        self.assertEqual(self.page.advanced_card.winfo_manager(), "")
        self.assertEqual(self.page.ensemble_mode_note.winfo_manager(), "grid")

    def test_neural_network_auto_uses_medium_search_request(self):
        self.page._model_changed("Neural Network")

        request = self.page.build_model_training_request()

        self.assertEqual(request.model_name, "neural_network")
        self.assertEqual(request.training_mode, "auto")
        self.assertEqual(request.search_level, "medium")
        self.assertIsNone(request.custom_hyperparameters)
        self.assertEqual(self.page.auto_search_frame.winfo_manager(), "grid")
        self.assertEqual(self.page.advanced_card.winfo_manager(), "")

    def test_neural_network_custom_controls_build_valid_request(self):
        self.page._model_changed("Neural Network")
        self.page._training_mode_changed("Custom")
        values = {
            "hidden_layer_sizes": "48, 24",
            "activation": "tanh",
            "learning_rate_init": "0.002",
            "batch_size": "4",
            "max_iter": "90",
        }
        for name, value in values.items():
            self.page.neural_network_parameter_vars[name].set(value)

        request = self.page.build_model_training_request()

        self.assertEqual(request.model_name, "neural_network")
        self.assertEqual(request.training_mode, "custom")
        self.assertIsNone(request.search_level)
        self.assertEqual(
            request.custom_hyperparameters,
            {
                "hidden_layer_sizes": [48, 24],
                "activation": "tanh",
                "learning_rate_init": 0.002,
                "batch_size": 4,
                "max_iter": 90,
            },
        )
        self.assertEqual(
            self.page.neural_network_settings_grid.winfo_manager(),
            "grid",
        )
        self.assertEqual(self.page.xgboost_settings_grid.winfo_manager(), "")
        for control in self.page.neural_network_parameter_controls.values():
            self.assertEqual(control.cget("state"), "normal")

    def test_neural_network_custom_defaults_match_contract(self):
        self.page._model_changed("Neural Network")
        self.page._training_mode_changed("Custom")

        request = self.page.build_model_training_request()

        self.assertEqual(
            request.custom_hyperparameters,
            NEURAL_NETWORK_CUSTOM_DEFAULTS,
        )
        self.assertEqual(
            tuple(self.page.neural_network_parameter_controls),
            NEURAL_NETWORK_CUSTOM_PARAMETER_NAMES,
        )

    def test_invalid_neural_network_custom_value_does_not_call_backend(self):
        self.page._model_changed("Neural Network")
        self.page._training_mode_changed("Custom")
        self.page.neural_network_parameter_vars["hidden_layer_sizes"].set("64, bad")

        with (
            patch("studio.ui.submit_model_training_request") as submit_request,
            patch("studio.ui.messagebox.showerror") as show_error,
        ):
            self.page.train_button.invoke()

        submit_request.assert_not_called()
        show_error.assert_called_once()
        self.assertIn("hidden layers", show_error.call_args.args[1])

    def test_xgboost_auto_selection_uses_medium_search_request(self):
        self.page._model_changed("XGBoost")

        request = self.page.build_model_training_request()

        self.assertEqual(request.model_name, "xgboost")
        self.assertEqual(request.training_mode, "auto")
        self.assertEqual(request.search_level, "medium")
        self.assertIsNone(request.custom_hyperparameters)
        self.assertEqual(self.page.training_mode_control.cget("state"), "normal")
        self.assertEqual(self.page.auto_search_frame.winfo_manager(), "grid")
        self.assertEqual(self.page.advanced_card.winfo_manager(), "")
        self.assertEqual(self.page.fixed_baseline_note.winfo_manager(), "")

    def test_xgboost_custom_mode_shows_numeric_settings_and_builds_request(self):
        self.page._model_changed("XGBoost")
        self.page._training_mode_changed("Custom")
        values = {
            "n_estimators": "90",
            "max_depth": "5",
            "learning_rate": "0.04",
            "subsample": "0.85",
            "colsample_bytree": "0.8",
        }
        for name, value in values.items():
            self.page.xgboost_parameter_vars[name].set(value)

        request = self.page.build_model_training_request()

        self.assertEqual(request.model_name, "xgboost")
        self.assertEqual(request.training_mode, "custom")
        self.assertIsNone(request.search_level)
        self.assertEqual(
            request.custom_hyperparameters,
            {
                "n_estimators": 90,
                "max_depth": 5,
                "learning_rate": 0.04,
                "subsample": 0.85,
                "colsample_bytree": 0.8,
            },
        )
        self.assertEqual(self.page.advanced_card.winfo_manager(), "grid")
        self.assertEqual(self.page.xgboost_settings_grid.winfo_manager(), "grid")
        self.assertEqual(self.page.linear_settings_row.winfo_manager(), "")
        for entry in self.page.xgboost_parameter_entries.values():
            self.assertEqual(entry.cget("state"), "normal")

    def test_xgboost_custom_defaults_match_fixed_baseline_controls(self):
        self.page._model_changed("XGBoost")
        self.page._training_mode_changed("Custom")

        request = self.page.build_model_training_request()

        self.assertEqual(request.custom_hyperparameters, XGBOOST_CUSTOM_DEFAULTS)
        self.assertEqual(
            tuple(self.page.xgboost_parameter_entries),
            XGBOOST_CUSTOM_PARAMETER_NAMES,
        )

    def test_invalid_xgboost_custom_ui_value_does_not_call_backend(self):
        self.page._model_changed("XGBoost")
        self.page._training_mode_changed("Custom")
        self.page.xgboost_parameter_vars["subsample"].set("1.5")

        with (
            patch("studio.ui.submit_model_training_request") as submit_request,
            patch("studio.ui.messagebox.showerror") as show_error,
        ):
            self.page.train_button.invoke()

        submit_request.assert_not_called()
        show_error.assert_called_once()
        self.assertIn("subsample", show_error.call_args.args[1])
        self.assertNotIn("Traceback", show_error.call_args.args[1])

    def test_clicking_train_with_xgboost_calls_backend_and_reports_metrics(self):
        self.page._model_changed("XGBoost")
        parameters = {
            "objective": "reg:squarederror",
            "n_estimators": 64,
            "learning_rate": 0.1,
            "max_depth": 4,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        }
        result = ModelTrainingResult(
            success=True,
            status=TRAINING_COMPLETED,
            model_name="xgboost",
            training_rows=16,
            test_rows=4,
            metrics={"MAE": 0.1, "RMSE": 0.2, "R²": 0.9},
            predictions=[],
            error_message=None,
            training_mode="auto",
            parameters_used=parameters,
            search_level="medium",
            configurations_evaluated=3,
            cross_validation_folds=3,
            best_validation_rmse=0.15,
            test_metrics={"MAE": 0.1, "RMSE": 0.2, "R²": 0.9},
            run_number=2,
            run_id="run-0002",
        )

        with (
            patch(
                "studio.ui.submit_model_training_request",
                return_value=result,
            ) as submit_request,
            patch("studio.ui.messagebox.showinfo") as show_info,
        ):
            self.page.train_button.invoke()

        request = submit_request.call_args.args[0]
        self.assertEqual(request.model_name, "xgboost")
        self.assertEqual(request.search_level, "medium")
        self.assertEqual(show_info.call_args.args[0], "Auto Search Completed")
        self.assertIn("Model: XGBoost", show_info.call_args.args[1])
        self.assertIn("Search Level: Medium", show_info.call_args.args[1])
        self.assertIn("Configurations Evaluated: 3", show_info.call_args.args[1])
        self.assertIn("n_estimators: 64", show_info.call_args.args[1])
        self.assertEqual(self.page.train_button.cget("state"), "normal")

    def test_clicking_train_with_custom_xgboost_passes_values_and_reports_them(self):
        self.page._model_changed("XGBoost")
        self.page._training_mode_changed("Custom")
        custom_values = {
            "n_estimators": 40,
            "max_depth": 3,
            "learning_rate": 0.06,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
        }
        for name, value in custom_values.items():
            self.page.xgboost_parameter_vars[name].set(str(value))
        parameters = {
            "objective": "reg:squarederror",
            **custom_values,
            "random_state": 42,
            "n_jobs": 1,
        }
        result = ModelTrainingResult(
            success=True,
            status=TRAINING_COMPLETED,
            model_name="xgboost",
            training_rows=16,
            test_rows=4,
            metrics={"MAE": 0.1, "RMSE": 0.2, "R²": 0.9},
            predictions=[],
            error_message=None,
            training_mode="custom",
            parameters_used=parameters,
            test_metrics={"MAE": 0.1, "RMSE": 0.2, "R²": 0.9},
            run_number=3,
            run_id="run-0003",
        )

        with (
            patch("studio.ui.submit_model_training_request", return_value=result) as submit,
            patch("studio.ui.messagebox.showinfo") as show_info,
        ):
            self.page.train_button.invoke()

        request = submit.call_args.args[0]
        self.assertEqual(request.training_mode, "custom")
        self.assertEqual(request.custom_hyperparameters, custom_values)
        message = show_info.call_args.args[1]
        self.assertIn("Training Mode: Custom", message)
        for name, value in custom_values.items():
            self.assertIn(f"{name}: {value}", message)
        self.assertEqual(self.page.train_button.cget("state"), "normal")

    def test_training_dropdowns_use_the_studio_theme(self):
        for shell, dropdown in (
            (self.page.model_dropdown_shell, self.page.model_dropdown),
            (
                self.page.search_level_dropdown_shell,
                self.page.search_level_dropdown,
            ),
        ):
            self.assertEqual(shell.cget("border_color"), COLORS["border"])
            self.assertEqual(dropdown.cget("fg_color"), COLORS["control"])
            self.assertEqual(
                dropdown.cget("button_color"),
                COLORS["surface_elevated"],
            )
            self.assertEqual(
                dropdown.cget("dropdown_fg_color"),
                COLORS["surface"],
            )
            self.assertEqual(
                dropdown.cget("dropdown_hover_color"),
                COLORS["nav_active"],
            )
            self.assertEqual(
                dropdown.cget("dropdown_text_color"),
                COLORS["ink"],
            )

    def test_auto_is_selected_by_default(self):
        self.assertEqual(self.page.training_mode.get(), "Auto")
        self.assertEqual(self.page.state.training_mode, "Auto")

    def test_medium_is_selected_by_default(self):
        self.assertEqual(AUTO_SEARCH_LEVELS, ("Medium", "High"))
        self.assertEqual(self.page.search_level.get(), "Medium")
        self.assertEqual(self.page.state.search_level, "Medium")

    def test_auto_search_level_is_available_in_auto_mode(self):
        self.assertEqual(self.page.auto_search_frame.winfo_manager(), "grid")
        self.assertEqual(self.page.search_level_dropdown.cget("state"), "normal")

    def test_advanced_settings_are_hidden_and_disabled_in_auto_mode(self):
        self.assertEqual(self.page.advanced_card.winfo_manager(), "")
        self.assertEqual(self.page.fit_intercept_switch.cget("state"), "disabled")
        self.assertEqual(self.page.positive_switch.cget("state"), "disabled")
        self.assertTrue(self.page.fit_intercept.get())
        self.assertFalse(self.page.positive.get())

    def test_selecting_custom_enables_advanced_settings(self):
        self.page._training_mode_changed("Custom")
        self.app.update_idletasks()

        self.assertEqual(self.page.training_mode.get(), "Custom")
        self.assertEqual(self.page.advanced_card.winfo_manager(), "grid")
        self.assertEqual(self.page.fit_intercept_switch.cget("state"), "normal")
        self.assertEqual(self.page.positive_switch.cget("state"), "normal")
        vertical_gap = self.page.advanced_card.winfo_y() - (
            self.page.training_mode_card.winfo_y()
            + self.page.training_mode_card.winfo_height()
        )
        self.assertGreaterEqual(vertical_gap, -2)
        self.assertLessEqual(vertical_gap, 12)

    def test_selecting_custom_hides_auto_search_level(self):
        self.page._training_mode_changed("Custom")

        self.assertEqual(self.page.auto_search_frame.winfo_manager(), "")
        self.assertEqual(self.page.custom_mode_note.winfo_manager(), "grid")

    def test_selecting_auto_again_disables_advanced_settings(self):
        self.page._training_mode_changed("Custom")
        self.page._training_mode_changed("Auto")

        self.assertEqual(self.page.auto_search_frame.winfo_manager(), "grid")
        self.assertEqual(self.page.advanced_card.winfo_manager(), "")
        self.assertEqual(self.page.fit_intercept_switch.cget("state"), "disabled")
        self.assertEqual(self.page.positive_switch.cget("state"), "disabled")

    def test_train_model_button_exists(self):
        self.assertEqual(self.page.train_button.cget("text"), TRAIN_BUTTON_LABEL)
        self.assertEqual(self.page.train_button.cget("state"), "normal")

    def test_train_model_button_state_lifecycle(self):
        observed_during_training = {}

        def observe_training_state(request, *, project_path):
            observed_during_training["state"] = self.page.train_button.cget(
                "state"
            )
            observed_during_training["text"] = self.page.train_button.cget(
                "text"
            )
            observed_during_training["in_progress"] = (
                self.page.training_in_progress
            )
            return self._successful_result()

        with (
            patch(
                "studio.ui.submit_model_training_request",
                side_effect=observe_training_state,
            ),
            patch("studio.ui.messagebox.showinfo"),
        ):
            self.page.train_button.invoke()

        self.assertEqual(observed_during_training["state"], "disabled")
        self.assertEqual(observed_during_training["text"], "Training…")
        self.assertTrue(observed_during_training["in_progress"])
        self.assertEqual(self.page.train_button.cget("state"), "normal")
        self.assertEqual(self.page.train_button.cget("text"), TRAIN_BUTTON_LABEL)
        self.assertFalse(self.page.training_in_progress)

    def test_clicking_train_in_auto_mode_creates_the_correct_request(self):
        with (
            patch(
                "studio.ui.submit_model_training_request",
                return_value=self._successful_result(),
            ) as submit_request,
            patch("studio.ui.messagebox.showinfo") as show_info,
        ):
            self.page.train_button.invoke()

        submit_request.assert_called_once()
        request = submit_request.call_args.args[0]
        self.assertEqual(
            request,
            ModelTrainingRequest(
                model_name="linear_regression",
                training_mode="auto",
                search_level="medium",
                custom_hyperparameters=None,
            ),
        )
        self.assertIs(self.page.last_training_request, request)
        self.assertEqual(
            submit_request.call_args.kwargs["project_path"],
            self.project.path,
        )
        self.assertTrue(self.page.last_training_result.success)
        show_info.assert_called_once()
        self.assertEqual(show_info.call_args.args[0], "Auto Search Completed")
        self.assertIn("Search Level: Medium", show_info.call_args.args[1])
        self.assertIn("Configurations Evaluated: 2", show_info.call_args.args[1])
        self.assertIn("Cross-Validation Folds: 3", show_info.call_args.args[1])
        self.assertIn("fit_intercept: True", show_info.call_args.args[1])
        self.assertIn("positive: False", show_info.call_args.args[1])
        self.assertIn("Validation RMSE: 0.15", show_info.call_args.args[1])
        self.assertIn("Test MAE: 0.1", show_info.call_args.args[1])

    def test_high_auto_result_displays_search_summary(self):
        self.page.search_level.set("High")

        with (
            patch(
                "studio.ui.submit_model_training_request",
                return_value=self._successful_result(
                    search_level="high",
                    configurations_evaluated=4,
                    cross_validation_folds=5,
                    best_validation_rmse=0.125,
                ),
            ),
            patch("studio.ui.messagebox.showinfo") as show_info,
        ):
            self.page.train_button.invoke()

        self.assertEqual(show_info.call_args.args[0], "Auto Search Completed")
        message = show_info.call_args.args[1]
        self.assertIn("Search Level: High", message)
        self.assertIn("Configurations Evaluated: 4", message)
        self.assertIn("Cross-Validation Folds: 5", message)
        self.assertIn("Validation RMSE: 0.125", message)
        self.assertIn("Test RMSE: 0.2", message)
        self.assertEqual(self.page.train_button.cget("state"), "normal")

    def test_successful_click_displays_latest_run(self):
        with (
            patch(
                "studio.ui.submit_model_training_request",
                return_value=self._successful_result(run_number=3),
            ),
            patch("studio.ui.messagebox.showinfo"),
        ):
            self.page.train_button.invoke()

        self.assertEqual(self.page.latest_run_number, 3)
        self.assertEqual(self.page.latest_run_var.get(), "Latest Run: Run 3")
        self.assertEqual(
            str(self.page.latest_run_label.cget("textvariable")),
            str(self.page.latest_run_var),
        )

    def test_clicking_train_in_custom_mode_creates_the_correct_request(self):
        self.page._training_mode_changed("Custom")
        self.page.fit_intercept.set(False)
        self.page.positive.set(True)

        with (
            patch(
                "studio.ui.submit_model_training_request",
                return_value=self._successful_result(
                    training_mode="custom",
                    parameters_used={
                        "fit_intercept": False,
                        "positive": True,
                    },
                ),
            ) as submit_request,
            patch("studio.ui.messagebox.showinfo") as show_info,
        ):
            self.page.train_button.invoke()

        submit_request.assert_called_once()
        request = submit_request.call_args.args[0]
        self.assertEqual(
            request,
            ModelTrainingRequest(
                model_name="linear_regression",
                training_mode="custom",
                search_level=None,
                custom_hyperparameters={
                    "fit_intercept": False,
                    "positive": True,
                },
            ),
        )
        self.assertIs(self.page.last_training_request, request)
        self.assertTrue(self.page.last_training_result.success)
        show_info.assert_called_once()
        self.assertIn("Training Mode: Custom", show_info.call_args.args[1])
        self.assertIn("fit_intercept: False", show_info.call_args.args[1])
        self.assertIn("positive: True", show_info.call_args.args[1])
        self.assertEqual(self.page.train_button.cget("state"), "normal")
        self.assertEqual(self.page.train_button.cget("text"), TRAIN_BUTTON_LABEL)
        self.assertFalse(self.page.training_in_progress)

    def test_invalid_page_state_does_not_call_training_backend(self):
        self.page.search_level.set("")

        with (
            patch("studio.ui.submit_model_training_request") as submit_request,
            patch("studio.ui.messagebox.showerror") as show_error,
            patch("studio.ui.messagebox.showinfo") as show_info,
        ):
            self.page.train_button.invoke()

        submit_request.assert_not_called()
        show_info.assert_not_called()
        show_error.assert_called_once()
        self.assertEqual(
            show_error.call_args.args[0],
            "Invalid training configuration",
        )
        self.assertIn(
            "Auto mode requires a search level",
            show_error.call_args.args[1],
        )
        self.assertNotIn("Traceback", show_error.call_args.args[1])
        self.assertIsNone(self.page.last_training_request)

    def test_clicking_train_model_calls_real_backend(self):
        with patch("studio.ui.messagebox.showinfo") as show_info:
            self.page.train_button.invoke()

        show_info.assert_called_once()
        self.assertEqual(show_info.call_args.args[0], "Auto Search Completed")
        self.assertIn("Validation RMSE:", show_info.call_args.args[1])
        self.assertIn("Test MAE:", show_info.call_args.args[1])
        self.assertIn("Test RMSE:", show_info.call_args.args[1])
        self.assertIn("Test R²:", show_info.call_args.args[1])
        result = self.page.last_training_result
        self.assertTrue(result.success)
        self.assertTrue(result.model_artifact_path.is_file())
        self.assertTrue(result.metrics_artifact_path.is_file())
        self.assertTrue(result.predictions_artifact_path.is_file())
        self.assertEqual(result.run_directory.name, result.run_id)
        self.assertEqual(
            self.page.latest_run_var.get(),
            f"Latest Run: Run {result.run_number}",
        )
        reopened = self.store.open_project(self.project.path, touch=False)
        self.page.set_project(reopened)
        self.assertEqual(
            self.page.latest_run_var.get(),
            f"Latest Run: Run {result.run_number}",
        )
        self.assertEqual(
            self.page.last_training_request,
            ModelTrainingRequest(
                model_name="linear_regression",
                training_mode="auto",
                search_level="medium",
                custom_hyperparameters=None,
            ),
        )


if __name__ == "__main__":
    unittest.main()
