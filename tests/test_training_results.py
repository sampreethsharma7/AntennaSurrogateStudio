import csv
import json
import os
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

from studio.assistant import build_project_context
from studio.dataset_registry import register_dataset
from studio.dataset_validation import validate_dataset
from studio.model_book import list_model_books
from studio.model_training import (
    ModelTrainingRequest,
    submit_model_training_request,
)
from studio.parser_engine import TrainingRequest
from studio.project_store import ProjectStore, atomic_write_json
from studio.results_ui import (
    RESULT_SECTIONS,
    infer_curve_axis,
)
from studio.scientific_plot import ScientificPlotWorkbench
from studio.theme import COLORS
from studio.training_results import (
    CUSTOM_VALIDATION_RMSE_TOLERANCE,
    EXPECTED_PREDICTION_COLUMNS,
    TrainingResultsError,
    load_latest_training_results,
    metric_card_data,
    prediction_table_rows,
)
from studio.ui import StudioApp


def register_test_dataset(project, *, offset: float = 0.0, rows: int = 20):
    input_path = project.path / "data" / "prepared" / "inputs.csv"
    output_path = project.path / "data" / "prepared" / "outputs.csv"
    with input_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Sample ID", "x1", "x2"])
        for index in range(1, rows + 1):
            x1 = float(index) + offset
            x2 = float((index * 3) % 7)
            writer.writerow([f"Design_{index:03d}", x1, x2])
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Sample ID", "target"])
        for index in range(1, rows + 1):
            x1 = float(index) + offset
            x2 = float((index * 3) % 7)
            writer.writerow([f"Design_{index:03d}", 2.0 * x1 - 0.5 * x2 + 3.0])
    validation = validate_dataset(
        TrainingRequest(
            input_csv_path=input_path,
            output_csv_path=output_path,
            feature_columns=["x1", "x2"],
            target_columns=["target"],
            sample_id_column="Sample ID",
        )
    )
    return register_dataset(project.path, validation)


def register_curve_dataset(project, *, rows: int = 20):
    input_path = project.path / "data" / "prepared" / "inputs.csv"
    output_path = project.path / "data" / "prepared" / "outputs.csv"
    coordinates = (1.0, 1.5, 2.0, 2.5, 3.0)
    target_columns = [
        f"S11_Frequency_{coordinate:g}_GHz"
        for coordinate in coordinates
    ]
    with input_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Sample ID", "patch_length", "patch_width"])
        for index in range(1, rows + 1):
            writer.writerow(
                [
                    f"Design_{index:03d}",
                    10.0 + index * 0.2,
                    7.0 + index * 0.1,
                ]
            )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Sample ID", *target_columns])
        for index in range(1, rows + 1):
            length = 10.0 + index * 0.2
            width = 7.0 + index * 0.1
            values = [
                -18.0 + 0.4 * length - 0.2 * width + coordinate
                for coordinate in coordinates
            ]
            writer.writerow([f"Design_{index:03d}", *values])
    validation = validate_dataset(
        TrainingRequest(
            input_csv_path=input_path,
            output_csv_path=output_path,
            feature_columns=["patch_length", "patch_width"],
            target_columns=target_columns,
            sample_id_column="Sample ID",
        )
    )
    return register_dataset(project.path, validation)


def auto_request(level: str = "high") -> ModelTrainingRequest:
    return ModelTrainingRequest(
        model_name="linear_regression",
        training_mode="auto",
        search_level=level,
    )


def custom_request(fit_intercept: bool, positive: bool) -> ModelTrainingRequest:
    return ModelTrainingRequest(
        model_name="linear_regression",
        training_mode="custom",
        custom_hyperparameters={
            "fit_intercept": fit_intercept,
            "positive": positive,
        },
    )


class TrainingResultsAnalysisTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        self.store = ProjectStore(Path(self.temp_dir.name) / "library")
        self.project = self.store.create_project("Results Analysis")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _train_auto(self, level: str = "high"):
        return submit_model_training_request(
            auto_request(level),
            project_path=self.project.path,
        )

    def _train_custom(self, fit_intercept: bool, positive: bool):
        return submit_model_training_request(
            custom_request(fit_intercept, positive),
            project_path=self.project.path,
        )

    def test_auto_results_load_selected_configuration_and_validation_reason(self):
        register_test_dataset(self.project)
        trained = self._train_auto("high")

        view = load_latest_training_results(self.project.path)

        self.assertIsNotNone(view)
        self.assertEqual(view.recommendation_title, "Recommended Configuration Selected")
        self.assertEqual(
            view.recommendation_statement,
            "Recommended Linear Regression configuration selected",
        )
        self.assertEqual(view.parameters_used, trained.best_parameters)
        self.assertEqual(view.search_level, "high")
        self.assertEqual(view.configurations_evaluated, 4)
        self.assertEqual(view.cross_validation_folds, 5)
        self.assertEqual(view.validation_rmse, trained.best_validation_rmse)
        self.assertTrue(any(candidate.selected for candidate in view.auto_candidates))

    def test_metric_cards_use_saved_values_without_inventing_units(self):
        register_test_dataset(self.project)
        trained = self._train_auto("medium")

        view = load_latest_training_results(self.project.path)
        cards = {card["name"]: card for card in metric_card_data(view)}

        self.assertEqual(cards["R²"]["value"], trained.metrics["R²"])
        self.assertEqual(cards["RMSE"]["value"], trained.metrics["RMSE"])
        self.assertEqual(cards["MAE"]["value"], trained.metrics["MAE"])
        self.assertEqual(
            cards["Validation RMSE"]["value"],
            trained.best_validation_rmse,
        )
        self.assertIsNone(view.target_unit)
        self.assertNotIn("GHz", cards["RMSE"]["display_value"])
        self.assertIn("Higher is better", cards["R²"]["direction"])
        self.assertIn("Lower is better", cards["RMSE"]["direction"])

    def test_prediction_data_residuals_and_largest_error_come_from_csv(self):
        register_test_dataset(self.project)
        trained = self._train_auto("medium")

        view = load_latest_training_results(self.project.path)
        with trained.predictions_artifact_path.open(
            newline="",
            encoding="utf-8",
        ) as handle:
            artifact_rows = list(csv.DictReader(handle))

        self.assertEqual(len(view.predictions), len(artifact_rows))
        for prediction, saved in zip(view.predictions, artifact_rows):
            self.assertEqual(prediction.sample_id, saved["sample_id"])
            self.assertEqual(prediction.actual_value, float(saved["actual_value"]))
            self.assertEqual(
                prediction.predicted_value,
                float(saved["predicted_value"]),
            )
            self.assertAlmostEqual(
                prediction.residual,
                prediction.actual_value - prediction.predicted_value,
            )
        expected_largest = max(
            view.predictions,
            key=lambda row: (row.absolute_error, row.sample_id),
        )
        self.assertEqual(view.largest_error_prediction, expected_largest)
        self.assertEqual(
            tuple(prediction_table_rows(view)[0]),
            EXPECTED_PREDICTION_COLUMNS,
        )
        for sample_id in view.prediction_sample_ids:
            self.assertEqual(
                set(view.sample_input_values[sample_id]),
                {"x1", "x2"},
            )

    def test_curve_axis_inference_is_deterministic(self):
        label, values = infer_curve_axis(
            (
                "S11_Frequency_1_GHz",
                "S11_Frequency_1.5_GHz",
                "S11_Frequency_2_GHz",
            )
        )
        self.assertEqual(label, "Frequency (GHz)")
        self.assertEqual(values, (1.0, 1.5, 2.0))

    def test_auto_candidates_are_ordered_and_selected_candidate_is_marked(self):
        register_test_dataset(self.project)
        self._train_auto("high")

        view = load_latest_training_results(self.project.path)
        successful_scores = [
            candidate.mean_validation_rmse
            for candidate in view.auto_candidates
            if candidate.success
        ]

        self.assertEqual(successful_scores, sorted(successful_scores))
        selected = [candidate for candidate in view.auto_candidates if candidate.selected]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].parameters, view.parameters_used)

    def test_failed_auto_candidates_remain_visible(self):
        register_test_dataset(self.project)

        def fail_positive(features, targets, parameters, folds):
            if parameters["positive"]:
                raise ValueError("positive candidate failed")
            return [1.0 if parameters["fit_intercept"] else 2.0] * folds

        with patch(
            "studio.model_training._cross_validate_linear_regression_configuration",
            side_effect=fail_positive,
        ):
            self._train_auto("high")

        view = load_latest_training_results(self.project.path)
        failed = [candidate for candidate in view.auto_candidates if not candidate.success]

        self.assertEqual(len(failed), 2)
        self.assertTrue(
            all("positive candidate failed" in candidate.error_message for candidate in failed)
        )
        self.assertTrue(view.auto_candidates[0].selected)

    def test_custom_result_uses_compatible_auto_validation_for_suggestion(self):
        register_test_dataset(self.project)
        self._train_auto("high")
        custom = self._train_custom(False, False)

        view = load_latest_training_results(self.project.path)

        self.assertEqual(view.run_id, custom.run_id)
        self.assertEqual(view.training_mode, "custom")
        self.assertEqual(
            view.recommendation_statement,
            "Your custom configuration was evaluated",
        )
        self.assertEqual(
            view.parameters_used,
            {"fit_intercept": False, "positive": False},
        )
        self.assertIsNotNone(view.custom_recommendation)
        self.assertEqual(
            view.custom_recommendation.recommendation,
            "Suggestion: Use the Auto-selected configuration.",
        )
        self.assertIn("lower validation RMSE", view.custom_recommendation.explanation)

    def test_different_dataset_fingerprints_are_not_compared(self):
        first_dataset = register_test_dataset(self.project)
        self._train_auto("high")
        second_dataset = register_test_dataset(self.project, offset=100.0)
        self._train_custom(False, False)

        view = load_latest_training_results(self.project.path)

        self.assertNotEqual(
            first_dataset.fingerprint_sha256,
            second_dataset.fingerprint_sha256,
        )
        self.assertIsNone(view.custom_recommendation)
        self.assertEqual(view.dataset_fingerprint, second_dataset.fingerprint_sha256)
        self.assertIn("No comparable Auto result", view.custom_guidance)
        self.assertIn("Run Auto training", view.custom_guidance)

    def test_custom_without_auto_displays_run_auto_guidance(self):
        register_test_dataset(self.project)
        self._train_custom(False, True)

        view = load_latest_training_results(self.project.path)

        self.assertIsNone(view.custom_recommendation)
        self.assertEqual(
            view.custom_guidance,
            "No comparable Auto result is available for this dataset. "
            "Run Auto training to generate a recommendation.",
        )

    def test_recommendation_uses_validation_not_test_performance(self):
        register_test_dataset(self.project)
        auto = self._train_auto("high")
        self._train_custom(False, False)
        atomic_write_json(
            auto.metrics_artifact_path,
            {"MAE": 9999.0, "RMSE": 9999.0, "R²": -9999.0},
        )

        view = load_latest_training_results(self.project.path)

        self.assertEqual(
            view.custom_recommendation.recommendation,
            "Suggestion: Use the Auto-selected configuration.",
        )
        self.assertEqual(view.custom_recommendation.auto_test_metrics["RMSE"], 9999.0)

    def test_negligible_validation_difference_uses_named_tolerance(self):
        register_test_dataset(self.project)
        auto = self._train_auto("high")
        self._train_custom(False, False)
        payload = json.loads(
            auto.auto_search_results_artifact_path.read_text(encoding="utf-8")
        )
        payload["best_validation_rmse"] = 1.0
        for result in payload["search_results"]:
            parameters = result["parameters"]
            if parameters == payload["best_parameters"]:
                result["mean_validation_rmse"] = 1.0
                result["fold_rmse"] = [1.0] * len(result["fold_rmse"])
            elif parameters == {"fit_intercept": False, "positive": False}:
                result["mean_validation_rmse"] = 1.005
                result["fold_rmse"] = [1.005] * len(result["fold_rmse"])
            elif result["success"]:
                result["mean_validation_rmse"] = 2.0
                result["fold_rmse"] = [2.0] * len(result["fold_rmse"])
        atomic_write_json(auto.auto_search_results_artifact_path, payload)

        view = load_latest_training_results(self.project.path)

        self.assertEqual(
            view.custom_recommendation.recommendation,
            "Your Custom configuration performs similarly to the "
            "Auto-selected configuration.",
        )
        self.assertAlmostEqual(
            view.custom_recommendation.relative_validation_difference,
            0.005,
        )
        self.assertEqual(
            view.custom_recommendation.tolerance,
            CUSTOM_VALIDATION_RMSE_TOLERANCE,
        )

    def test_latest_completed_run_is_loaded_and_previous_run_is_preserved(self):
        register_test_dataset(self.project)
        first = self._train_auto("medium")
        second = self._train_custom(True, False)

        view = load_latest_training_results(self.project.path)

        self.assertEqual(view.run_id, second.run_id)
        self.assertEqual(view.run_number, 2)
        self.assertTrue(first.run_directory.is_dir())
        self.assertTrue(second.run_directory.is_dir())
        self.assertTrue(first.auto_search_results_artifact_path.is_file())

    def test_plain_language_insights_are_deterministic_and_limited(self):
        register_test_dataset(self.project)
        self._train_auto("high")

        first = load_latest_training_results(self.project.path)
        second = load_latest_training_results(self.project.path)

        self.assertEqual(first.insights, second.insights)
        self.assertLessEqual(len(first.insights), 3)
        self.assertTrue(first.insights)
        self.assertTrue(first.residual_interpretation)

    def test_calculated_findings_reach_snowbuddy_project_context(self):
        register_test_dataset(self.project)
        self._train_auto("high")
        reopened = self.store.open_project(self.project.path, touch=False)
        view = load_latest_training_results(reopened.path)

        context = build_project_context(reopened)

        self.assertIn("SnowBuddy result evidence", context)
        self.assertIn(f"Latest results run ID: {view.run_id}", context)
        self.assertIn("Latest calculated findings:", context)
        for insight in view.insights:
            self.assertIn(insight, context)

    def test_missing_or_malformed_artifacts_raise_friendly_errors(self):
        register_test_dataset(self.project)
        trained = self._train_auto("medium")
        trained.metrics_artifact_path.write_text("{not json", encoding="utf-8")

        with self.assertRaisesRegex(
            TrainingResultsError,
            "malformed or unreadable",
        ) as context:
            load_latest_training_results(self.project.path)

        self.assertNotIn("Traceback", str(context.exception))


GUI_MAY_BE_AVAILABLE = (
    os.name == "nt"
    or os.sys.platform == "darwin"
    or bool(os.environ.get("DISPLAY"))
)


@unittest.skipUnless(GUI_MAY_BE_AVAILABLE, "A desktop display is required.")
class TrainingResultsPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        cls.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        cls.store = ProjectStore(Path(cls.temp_dir.name) / "library")
        cls.auto_project = cls.store.create_project("Auto Results UI")
        register_test_dataset(cls.auto_project)
        submit_model_training_request(
            auto_request("high"), project_path=cls.auto_project.path
        )
        cls.custom_project = cls.store.create_project("Custom Results UI")
        register_test_dataset(cls.custom_project)
        submit_model_training_request(
            auto_request("high"), project_path=cls.custom_project.path
        )
        submit_model_training_request(
            custom_request(False, False), project_path=cls.custom_project.path
        )
        cls.comparison_project = cls.store.create_project("Comparison Results UI")
        register_test_dataset(cls.comparison_project)
        submit_model_training_request(
            auto_request("medium"), project_path=cls.comparison_project.path
        )
        submit_model_training_request(
            ModelTrainingRequest(
                model_name="xgboost",
                training_mode="auto",
                search_level="medium",
                custom_hyperparameters=None,
            ),
            project_path=cls.comparison_project.path,
        )
        cls.curve_project = cls.store.create_project("Curve Results UI")
        register_curve_dataset(cls.curve_project)
        submit_model_training_request(
            auto_request("high"), project_path=cls.curve_project.path
        )
        cls.empty_project = cls.store.create_project("Empty Results UI")
        cls.malformed_project = cls.store.create_project("Malformed Results UI")
        register_test_dataset(cls.malformed_project)
        malformed_run = submit_model_training_request(
            auto_request("medium"), project_path=cls.malformed_project.path
        )
        malformed_run.metrics_artifact_path.write_text(
            "{not valid json",
            encoding="utf-8",
        )
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
        project = self.store.open_project(self.auto_project.path, touch=False)
        self.app.set_project(project, target_page="results")
        self.app.update_idletasks()
        self.page = self.app.results_page

    def test_auto_page_displays_recommendation_metrics_and_latest_run(self):
        self.assertEqual(self.app.active_page, "results")
        self.assertEqual(self.page.recommendation_title.cget("text"), "AUTO BEST")
        body = self.page.recommendation_body.cget("text")
        self.assertIn("fit_intercept=", body)
        self.assertIn("positive=", body)
        self.assertNotIn("\n", body)
        self.assertEqual(self.page.recommendation_card.cget("height"), 44)
        self.assertFalse(hasattr(self.page, "recommendation_meta"))
        self.assertFalse(hasattr(self.page, "recommendation_action"))

        self.page.show_section("configuration")
        configuration_text = []

        def collect_text(widget):
            try:
                value = widget.cget("text")
            except (tk.TclError, ValueError):
                value = ""
            if value:
                configuration_text.append(str(value))
            for child in widget.winfo_children():
                collect_text(child)

        collect_text(self.page.content_card)
        configuration_summary = " ".join(configuration_text)
        self.assertIn("lowest validation RMSE", configuration_summary)
        self.assertIn("High", configuration_summary)
        self.assertIn("4 configurations", configuration_summary)
        self.assertIn("5 folds", configuration_summary)
        displayed = [value.cget("text") for _, value, _ in self.page.metric_widgets]
        expected = [card["display_value"] for card in metric_card_data(self.page.result)]
        self.assertEqual(displayed, expected)
        self.assertIn(self.page.result.run_id.upper(), self.page.run_badge.cget("text"))

    def test_metric_cards_are_compact_and_expose_help_on_hover_or_focus(self):
        cards = metric_card_data(self.page.result)
        self.app.update_idletasks()
        for (_, _, help_button), card in zip(self.page.metric_widgets, cards):
            self.assertEqual(help_button.cget("text"), "?")
            self.assertEqual(
                help_button.help_text,
                f"{card['meaning']}\n{card['direction']}",
            )
            self.assertEqual(help_button.master.cget("height"), 62)

        first_help = self.page.metric_widgets[0][2]
        first_help._show_tooltip()
        self.app.update_idletasks()
        self.assertIsNotNone(first_help.tooltip_window)
        first_help._hide_tooltip()
        self.assertIsNone(first_help.tooltip_window)

    def test_plots_use_loaded_test_predictions_and_backend_residuals(self):
        section_labels = dict(RESULT_SECTIONS)
        self.assertEqual(section_labels["fit"], "Predictions")
        self.assertNotIn("predictions", section_labels)
        self.assertNotIn("insights", section_labels)
        self.assertNotIn("What This Means", section_labels.values())
        self.assertFalse(hasattr(self.page, "_render_insights"))
        self.page.show_section("fit")
        self.assertIsInstance(self.page.current_chart, ScientificPlotWorkbench)
        self.assertEqual(
            [curve.name for curve in self.page.current_chart.state.curves],
            ["Actual", "Predicted"],
        )
        self.page.show_section("residuals")
        self.assertIsInstance(self.page.current_chart, ScientificPlotWorkbench)
        residual_curve, zero_curve = self.page.current_chart.state.curves
        self.assertEqual(residual_curve.name, "Residual")
        self.assertEqual(residual_curve.line_style, "None")
        self.assertEqual(zero_curve.name, "Zero error")
        for prediction in self.page.result.predictions:
            self.assertAlmostEqual(
                prediction.residual,
                prediction.actual_value - prediction.predicted_value,
            )

    def test_model_comparison_shows_both_families_metrics_and_run_access(self):
        project = self.store.open_project(self.comparison_project.path, touch=False)
        self.app.set_project(project, target_page="results")
        self.app.update_idletasks()
        page = self.app.results_page

        page.show_section("comparison")
        self.app.update_idletasks()

        self.assertIn("comparison", dict(RESULT_SECTIONS))
        self.assertIsNotNone(page.model_comparison)
        self.assertIsNotNone(page.comparison_metric_chart)
        self.assertEqual(
            tuple(family.model_name for family in page.model_comparison.families),
            (
                "linear_regression",
                "xgboost",
                "neural_network",
                "ensemble_ai_engine",
            ),
        )
        self.assertIsNone(page.model_comparison.family("neural_network").best_run)
        self.assertEqual(
            set(page.comparison_run_buttons),
            {"linear_regression", "xgboost"},
        )
        self.assertEqual(
            set(page.comparison_metric_chart.metric_values),
            {"Validation RMSE", "Test RMSE", "MAE", "R²"},
        )
        for values in page.comparison_metric_chart.metric_values.values():
            self.assertEqual(set(values), {"linear_regression", "xgboost"})
        self.assertIn(
            "LONGER BAR IS BETTER",
            page.comparison_metric_chart.quality_caption.cget("text"),
        )
        for metric_name in ("Validation RMSE", "Test RMSE", "MAE"):
            values = page.comparison_metric_chart.metric_values[metric_name]
            bars = page.comparison_metric_chart.metric_bar_values[metric_name]
            better = min(values, key=values.get)
            worse = max(values, key=values.get)
            self.assertGreater(bars[better], bars[worse])
        r_squared = page.comparison_metric_chart.metric_values["R²"]
        r_squared_bars = page.comparison_metric_chart.metric_bar_values["R²"]
        self.assertGreater(
            r_squared_bars[max(r_squared, key=r_squared.get)],
            r_squared_bars[min(r_squared, key=r_squared.get)],
        )
        self.assertIn(
            "LATEST SELECTED RUN METRICS",
            page.metrics_context_label.cget("text"),
        )

        linear_run_id = page.model_comparison.family(
            "linear_regression"
        ).best_run.run_id
        page.comparison_run_buttons["linear_regression"].invoke()
        self.app.update_idletasks()
        self.assertEqual(page.result.run_id, linear_run_id)
        self.assertEqual(page.active_section, "fit")
        self.assertIn("SELECTED", page.run_badge.cget("text"))

    def test_loading_another_project_resets_results_to_predictions(self):
        self.page.show_section("configuration")
        self.assertEqual(self.page.active_section, "configuration")

        project = self.store.open_project(self.curve_project.path, touch=False)
        self.app.set_project(project, target_page="results")
        self.app.update_idletasks()

        self.assertEqual(self.page.active_section, "fit")
        self.assertIsInstance(self.page.current_chart, ScientificPlotWorkbench)
        self.assertEqual(
            self.page.section_buttons["fit"].cget("fg_color"),
            COLORS["nav_active"],
        )

    def test_loading_a_new_run_resets_results_to_predictions(self):
        project = self.store.create_project("Results New Run Reset")
        register_test_dataset(project)
        submit_model_training_request(
            auto_request("high"), project_path=project.path
        )
        project = self.store.open_project(project.path, touch=False)
        self.app.set_project(project, target_page="results")
        self.app.update_idletasks()
        self.page.show_section("errors")
        self.assertEqual(self.page.active_section, "errors")

        submit_model_training_request(
            auto_request("medium"), project_path=project.path
        )
        project = self.store.open_project(project.path, touch=False)
        self.page.set_project(project)
        self.app.update_idletasks()

        self.assertEqual(self.page.result.run_id, "run-0002")
        self.assertEqual(self.page.active_section, "fit")
        self.assertIsInstance(self.page.current_chart, ScientificPlotWorkbench)

    def test_predictions_section_is_plot_only_with_csv_access(self):
        self.page.show_section("fit")
        self.assertIsInstance(self.page.current_chart, ScientificPlotWorkbench)
        self.assertEqual(
            self.page.open_predictions_button.cget("text"),
            "Open Test Data CSV",
        )
        self.assertFalse(hasattr(self.page, "prediction_table"))

    def test_multi_output_curves_sample_selection_and_plot_settings(self):
        project = self.store.open_project(self.curve_project.path, touch=False)
        self.app.set_project(project, target_page="results")
        self.app.update_idletasks()
        page = self.app.results_page
        page.show_section("fit")
        chart = page.current_chart
        self.assertIsInstance(chart, ScientificPlotWorkbench)
        self.assertEqual(chart.x_label, "Frequency (GHz)")
        self.assertEqual(chart.y_label, "Response value")
        self.assertEqual(chart.x_values, (1.0, 1.5, 2.0, 2.5, 3.0))
        self.assertEqual(len(chart.state.curves), 2)
        self.assertEqual(chart.state.curves[0].name, "Actual")
        self.assertEqual(chart.state.curves[1].name, "Predicted")
        self.assertEqual(
            tuple(page.curve_sample_menu.cget("values")),
            page.result.prediction_sample_ids,
        )
        self.assertIs(page.curve_sample_menu.master, chart.manager_context)
        self.assertEqual(int(page.curve_sample_menu.grid_info()["row"]), 1)
        self.assertFalse(hasattr(page, "curve_controls"))
        self.assertFalse(hasattr(page, "curve_axis_min_entry"))
        self.assertFalse(hasattr(page, "curve_axis_max_entry"))
        self.assertFalse(hasattr(page, "curve_axis_step_entry"))
        self.assertFalse(hasattr(page, "curve_axis_apply_button"))
        input_text = page.input_values_label.cget("text")
        self.assertIn("patch_length =", input_text)
        self.assertIn("patch_width =", input_text)

        self.assertEqual(chart.plot_settings_button.cget("text"), "Plot Settings")
        self.assertEqual(chart.state.curves[0].line_style, "Solid")
        self.assertEqual(chart.state.curves[1].line_style, "Dashed")
        self.assertTrue(chart.state.legend_visible)

        page.prediction_plot_state.x_label = "Theta (deg)"
        page.prediction_plot_state.y_label = "S11 (dB)"
        page.prediction_plot_state.axis_labels_user_defined = True
        page.prediction_plot_state.set_limits(
            0.75,
            3.25,
            -10.0,
            10.0,
            user_defined=True,
        )

        second_sample = page.result.prediction_sample_ids[1]
        page._curve_sample_changed(second_sample)
        self.assertEqual(page.curve_sample.get(), second_sample)
        expected_inputs = page.result.sample_input_values[second_sample]
        for name, value in expected_inputs.items():
            self.assertIn(
                f"{name} = {value:.6g}",
                page.input_values_label.cget("text"),
            )
        self.assertEqual(page.current_chart.x_label, "Theta (deg)")
        self.assertEqual(page.current_chart.y_label, "S11 (dB)")
        self.assertEqual(
            page.current_chart.x_values,
            (1.0, 1.5, 2.0, 2.5, 3.0),
        )
        self.assertEqual(
            page.current_chart.state.view_limits,
            (0.75, 3.25, -10.0, 10.0),
        )

    def test_auto_configuration_table_marks_selected_and_keeps_all_candidates(self):
        self.page.show_section("configuration")
        text_values = []

        def collect_text(widget):
            try:
                value = widget.cget("text")
            except (tk.TclError, ValueError):
                value = ""
            if value:
                text_values.append(str(value))
            for child in widget.winfo_children():
                collect_text(child)

        collect_text(self.page.content_card)
        table_text = " ".join(text_values)
        self.assertIn("✓ Selected", table_text)
        self.assertGreaterEqual(table_text.count("Completed"), 4)

    def test_custom_page_displays_user_parameters_and_suggestion(self):
        project = self.store.open_project(self.custom_project.path, touch=False)
        self.app.set_project(project, target_page="results")
        self.app.update_idletasks()

        title = self.app.results_page.recommendation_title.cget("text")
        body = self.app.results_page.recommendation_body.cget("text")

        self.assertEqual(title, "CUSTOM USED")
        self.assertIn("fit_intercept=False", body)
        self.assertIn("positive=False", body)
        page = self.app.results_page
        self.assertFalse(hasattr(page, "recommendation_meta"))
        self.assertFalse(hasattr(page, "recommendation_action"))
        page.show_section("configuration")
        configuration_text = []

        def collect_text(widget):
            try:
                value = widget.cget("text")
            except (tk.TclError, ValueError):
                value = ""
            if value:
                configuration_text.append(str(value))
            for child in widget.winfo_children():
                collect_text(child)

        collect_text(page.content_card)
        comparison = " ".join(configuration_text)
        self.assertIn("Your Custom Configuration", comparison)
        self.assertIn("Suggested Auto Configuration", comparison)
        self.assertIn("fit_intercept: True", comparison)

    def test_xgboost_results_page_displays_auto_search_summary(self):
        project = self.store.create_project("XGBoost Results UI")
        register_test_dataset(project)
        result = submit_model_training_request(
            ModelTrainingRequest(
                model_name="xgboost",
                training_mode="auto",
                search_level="medium",
            ),
            project_path=project.path,
        )
        self.assertTrue(result.success, result.error_message)
        project = self.store.open_project(project.path, touch=False)
        self.app.set_project(project, target_page="results")
        self.app.update_idletasks()

        page = self.app.results_page
        self.assertEqual(page.recommendation_title.cget("text"), "AUTO BEST")
        self.assertIn("trees", page.recommendation_body.cget("text"))
        page.show_section("configuration")
        configuration_text = []

        def collect_text(widget):
            try:
                value = widget.cget("text")
            except (tk.TclError, ValueError):
                value = ""
            if value:
                configuration_text.append(str(value))
            for child in widget.winfo_children():
                collect_text(child)

        collect_text(page.content_card)
        rendered = " ".join(configuration_text)
        self.assertIn("Auto-search comparison", rendered)
        self.assertIn("Medium", rendered)
        self.assertIn("3 configurations", rendered)
        self.assertIn("n=", rendered)
        self.assertIn("Selected", rendered)

    def test_xgboost_results_page_displays_exact_custom_parameters(self):
        project = self.store.create_project("XGBoost Custom Results UI")
        register_test_dataset(project)
        custom_parameters = {
            "n_estimators": 22,
            "max_depth": 2,
            "learning_rate": 0.07,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
        }
        result = submit_model_training_request(
            ModelTrainingRequest(
                model_name="xgboost",
                training_mode="custom",
                custom_hyperparameters=custom_parameters,
            ),
            project_path=project.path,
        )
        self.assertTrue(result.success, result.error_message)
        project = self.store.open_project(project.path, touch=False)
        self.app.set_project(project, target_page="results")
        self.app.update_idletasks()

        page = self.app.results_page
        self.assertEqual(page.recommendation_title.cget("text"), "CUSTOM USED")
        self.assertIn("22 trees", page.recommendation_body.cget("text"))
        page.show_section("configuration")
        configuration_text = []

        def collect_text(widget):
            try:
                value = widget.cget("text")
            except (tk.TclError, ValueError):
                value = ""
            if value:
                configuration_text.append(str(value))
            for child in widget.winfo_children():
                collect_text(child)

        collect_text(page.content_card)
        rendered = " ".join(configuration_text)
        self.assertIn("Custom XGBoost configuration", rendered)
        self.assertIn("exact estimator parameters", rendered)
        for name, value in custom_parameters.items():
            self.assertIn(name, rendered)
            self.assertIn(str(value), rendered)

    def test_neural_network_results_use_saved_search_and_scientific_plot(self):
        project = self.store.create_project("Neural Network Results UI")
        register_test_dataset(project)
        result = submit_model_training_request(
            ModelTrainingRequest(
                model_name="neural_network",
                training_mode="auto",
                search_level="medium",
            ),
            project_path=project.path,
        )
        self.assertTrue(result.success, result.error_message)
        project = self.store.open_project(project.path, touch=False)
        self.app.set_project(project, target_page="results")
        self.app.update_idletasks()

        page = self.app.results_page
        self.assertEqual(page.result.model_name, "neural_network")
        self.assertEqual(page.recommendation_title.cget("text"), "AUTO BEST")
        self.assertIn("layers", page.recommendation_body.cget("text"))
        self.assertIsInstance(page.current_chart, ScientificPlotWorkbench)
        page.show_section("configuration")
        configuration_text: list[str] = []

        def collect_text(widget):
            try:
                value = widget.cget("text")
            except (tk.TclError, ValueError):
                value = ""
            if value:
                configuration_text.append(str(value))
            for child in widget.winfo_children():
                collect_text(child)

        collect_text(page.content_card)
        rendered = " ".join(configuration_text)
        self.assertIn("Auto-search comparison", rendered)
        self.assertIn("Neural Network parameters", rendered)
        self.assertIn("Selected", rendered)

    def test_train_button_remains_usable_from_results_workflow(self):
        self.assertEqual(self.app.training_page.train_button.cget("state"), "normal")
        self.assertEqual(self.app.training_page.train_button.cget("text"), "Train Model")
        self.assertEqual(
            self.page.train_again_button.cget("text"),
            "Adjust & Train Again",
        )
        self.app.training_page._training_mode_changed("Custom")
        self.page.train_again_button.invoke()
        self.assertEqual(self.app.active_page, "training")
        self.assertEqual(self.app.training_page.training_mode.get(), "Custom")
        self.assertEqual(self.page.winfo_y(), 0)

    def test_completed_result_can_be_saved_as_named_model_book(self):
        project = self.store.create_project("Results Model Book Save")
        register_test_dataset(project)
        submit_model_training_request(
            auto_request("medium"), project_path=project.path
        )
        project = self.store.open_project(project.path, touch=False)
        self.app.set_project(project, target_page="results")
        self.app.update_idletasks()
        page = self.app.results_page

        self.assertEqual(
            page.save_model_button.cget("text"),
            "Create Model Book  →",
        )
        self.assertEqual(page.save_model_button.cget("state"), "normal")
        with (
            patch.object(
                page,
                "_ask_model_book_name",
                return_value="UI Saved Surrogate",
            ),
            patch("studio.results_ui.messagebox.showinfo") as showinfo,
        ):
            page._save_as_model()

        books = list_model_books(project.path)
        self.assertEqual(len(books), 1)
        self.assertEqual(books[0].name, "UI Saved Surrogate")
        self.assertEqual(books[0].source_run_id, page.result.run_id)
        self.assertEqual(page.save_model_button.cget("state"), "normal")
        self.assertEqual(
            page.save_model_button.cget("text"),
            "Open Model Library  →",
        )
        self.assertIn("UI Saved Surrogate", page.footer_status_label.cget("text"))
        self.assertEqual(
            self.app.snowbuddy_panel.current_project.manifest["model_library"][
                "book_count"
            ],
            1,
        )
        self.assertIsNone(
            self.app.snowbuddy_panel.current_project.manifest["model_library"][
                "active_book_id"
            ]
        )
        showinfo.assert_called_once()

        page._primary_model_action()
        self.assertEqual(self.app.active_page, "library")
        self.assertEqual(
            self.app.library_page.selected_book_id,
            books[0].book_id,
        )
        self.assertIsNone(self.app.library_page.library.active_book_id)

        reopened = self.store.open_project(project.path, touch=False)
        self.app.set_project(reopened, target_page="results")
        self.assertEqual(
            self.app.results_page.save_model_button.cget("text"),
            "Open Model Library  →",
        )
        self.assertEqual(
            self.app.results_page.saved_model_book_id,
            books[0].book_id,
        )

    def test_empty_failure_and_malformed_states_are_friendly(self):
        empty = self.store.open_project(self.empty_project.path, touch=False)
        self.app.set_project(empty, target_page="results")
        self.assertIn(
            "No completed training run is available yet",
            self.app.results_page.empty_label.cget("text"),
        )
        self.assertEqual(
            self.app.results_page.save_model_button.cget("state"),
            "disabled",
        )

        self.app.results_page.show_training_failure()
        self.assertIn(
            "Training did not complete",
            self.app.results_page.empty_label.cget("text"),
        )

        malformed = self.store.open_project(
            self.malformed_project.path,
            touch=False,
        )
        self.app.set_project(malformed, target_page="results")
        error_text = self.app.results_page.empty_label.cget("text")
        self.assertIn("could not be loaded", error_text)
        self.assertIn("malformed or unreadable", error_text)
        self.assertNotIn("Traceback", error_text)


if __name__ == "__main__":
    unittest.main()
