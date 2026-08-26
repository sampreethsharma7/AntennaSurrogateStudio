import json
import math
import os
import tempfile
import tkinter as tk
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

from studio.assistant import build_latest_run_evidence
from studio.ensemble import (
    WeightedEnsembleRegressor,
    normalize_inverse_rmse_weights,
)
from studio.inference import InferenceRequest, submit_inference_request
from studio.model_book import load_model_book, save_model_book, set_active_model_book
from studio.model_comparison import compare_compatible_model_runs
from studio.model_training import (
    ModelTrainingError,
    ModelTrainingRequest,
    submit_model_training_request,
)
from studio.project_store import ProjectStore, atomic_write_json
from studio.training_results import load_latest_training_results
from studio.ui import StudioApp
from tests.test_neural_network_training import register_neural_network_dataset


def ensemble_request() -> ModelTrainingRequest:
    return ModelTrainingRequest(
        model_name="ensemble_ai_engine",
        training_mode="auto",
        search_level="high",
        custom_hyperparameters=None,
    )


class EnsembleContractTests(unittest.TestCase):
    def test_contract_requires_auto_high(self):
        self.assertEqual(ensemble_request().search_level, "high")
        with self.assertRaisesRegex(ValueError, "Auto High"):
            ModelTrainingRequest(
                model_name="ensemble_ai_engine",
                training_mode="auto",
                search_level="medium",
            )
        with self.assertRaisesRegex(ValueError, "Auto High"):
            ModelTrainingRequest(
                model_name="ensemble_ai_engine",
                training_mode="custom",
                custom_hyperparameters={"anything": True},
            )

    def test_inverse_validation_rmse_weights_are_normalized(self):
        weights = normalize_inverse_rmse_weights(
            {"linear_regression": 1.0, "xgboost": 2.0, "neural_network": 4.0}
        )
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertAlmostEqual(weights["linear_regression"], 4.0 / 7.0)
        self.assertAlmostEqual(weights["xgboost"], 2.0 / 7.0)
        self.assertAlmostEqual(weights["neural_network"], 1.0 / 7.0)
        self.assertGreater(weights["linear_regression"], weights["xgboost"])


class EnsembleEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        cls.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        cls.store = ProjectStore(Path(cls.temp_dir.name) / "library")
        cls.project = cls.store.create_project("Ensemble AI Engine End To End")
        register_neural_network_dataset(cls.project)
        cls.result = submit_model_training_request(
            ensemble_request(),
            project_path=cls.project.path,
        )
        if not cls.result.success:
            raise AssertionError(cls.result.error_message)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_all_available_components_train_in_auto_high(self):
        manifest = json.loads(
            (self.project.path / "project.json").read_text(encoding="utf-8")
        )
        runs = manifest["model_training"]["runs"]
        self.assertEqual(
            [run["model_name"] for run in runs],
            [
                "linear_regression",
                "xgboost",
                "neural_network",
                "ensemble_ai_engine",
            ],
        )
        for run in runs[:3]:
            self.assertEqual(run["configuration"]["training_mode"], "auto")
            self.assertEqual(run["configuration"]["search_level"], "high")
        self.assertEqual(self.result.run_id, "run-0004")

    def test_weights_use_component_validation_rmse_only(self):
        validation = {
            component["model_name"]: component["validation_rmse"]
            for component in self.result.component_results
        }
        expected = normalize_inverse_rmse_weights(validation)
        self.assertEqual(set(expected), set(self.result.ensemble_weights))
        for model_name, weight in expected.items():
            self.assertAlmostEqual(self.result.ensemble_weights[model_name], weight)
        self.assertAlmostEqual(sum(self.result.ensemble_weights.values()), 1.0)

    def test_ensemble_test_predictions_are_weighted_component_predictions(self):
        component_predictions = {}
        for component in self.result.component_results:
            view = load_latest_training_results(
                self.project.path,
                run_id=component["source_run_id"],
            )
            component_predictions[component["model_name"]] = {
                (row.sample_id, row.target_name): row.predicted_value
                for row in view.predictions
            }
        for row in self.result.predictions:
            key = (str(row["sample_id"]), str(row["target_name"]))
            expected = sum(
                component_predictions[name][key] * weight
                for name, weight in self.result.ensemble_weights.items()
            )
            self.assertAlmostEqual(float(row["predicted_value"]), expected)

    def test_validation_evidence_and_recommendation_are_recorded(self):
        best_component = min(
            self.result.component_results,
            key=lambda component: component["validation_rmse"],
        )
        self.assertEqual(
            self.result.best_individual_model,
            best_component["model_name"],
        )
        self.assertAlmostEqual(
            self.result.best_individual_validation_rmse,
            best_component["validation_rmse"],
        )
        self.assertEqual(
            self.result.ensemble_improved_on_best,
            self.result.ensemble_validation_rmse
            < self.result.best_individual_validation_rmse - 1e-12,
        )

    def test_artifacts_and_training_results_preserve_components(self):
        self.assertTrue(self.result.model_artifact_path.is_file())
        self.assertTrue(self.result.ensemble_results_artifact_path.is_file())
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Setting the shape on a NumPy array has been deprecated.*",
                category=DeprecationWarning,
            )
            saved_model = joblib.load(self.result.model_artifact_path)
        self.assertIsInstance(saved_model, WeightedEnsembleRegressor)
        self.assertEqual(saved_model.weights, self.result.ensemble_weights)
        view = load_latest_training_results(self.project.path)
        self.assertEqual(view.model_name, "ensemble_ai_engine")
        self.assertEqual(len(view.ensemble_components), 3)
        self.assertAlmostEqual(view.validation_rmse, self.result.ensemble_validation_rmse)
        for component in self.result.component_results:
            path = self.result.run_directory / "components" / {
                "linear_regression": "linear_regression_model.joblib",
                "xgboost": "xgboost_model.joblib",
                "neural_network": "neural_network_model.joblib",
            }[component["model_name"]]
            self.assertTrue(path.is_file())

    def test_model_book_load_and_inference_reproduce_ensemble(self):
        book = save_model_book(
            self.project.path,
            self.result.run_id,
            "Reusable Ensemble",
        )
        loaded = load_model_book(self.project.path, book.book_id)
        self.assertEqual(loaded.model_name, "ensemble_ai_engine")
        self.assertEqual(len(loaded.ensemble_components), 3)
        self.assertEqual(set(loaded.component_artifact_paths), set(self.result.ensemble_weights))
        set_active_model_book(self.project.path, loaded.book_id)
        inputs = {"width": 1.0, "height": 0.5, "gap": 0.2}
        inference = submit_inference_request(
            InferenceRequest(inputs=inputs),
            project_path=self.project.path,
        )
        self.assertTrue(inference.success, inference.error_message)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Setting the shape on a NumPy array has been deprecated.*",
                category=DeprecationWarning,
            )
            estimator = joblib.load(loaded.model_artifact_path)
        expected = np.asarray(estimator.predict([[1.0, 0.5, 0.2]]), dtype=float)
        if expected.ndim == 2:
            expected = expected[0]
        self.assertEqual(list(inference.predictions), list(loaded.target_columns))
        for value, target in zip(expected.tolist(), loaded.target_columns, strict=True):
            self.assertAlmostEqual(inference.predictions[target], value)

    def test_model_comparison_recommends_ensemble_only_when_validation_is_lower(self):
        comparison = compare_compatible_model_runs(self.project.path)
        self.assertEqual(
            tuple(family.model_name for family in comparison.families),
            (
                "linear_regression",
                "xgboost",
                "neural_network",
                "ensemble_ai_engine",
            ),
        )
        ensemble = comparison.family("ensemble_ai_engine").best_run
        individual = min(
            (
                comparison.family(name).best_run
                for name in ("linear_regression", "xgboost", "neural_network")
            ),
            key=lambda run: run.validation_rmse,
        )
        if ensemble.validation_rmse < individual.validation_rmse - 1e-12:
            self.assertEqual(comparison.recommended_model, "ensemble_ai_engine")
        else:
            self.assertNotEqual(comparison.recommended_model, "ensemble_ai_engine")

    def test_validation_backed_comparison_can_recommend_ensemble(self):
        path = self.result.ensemble_results_artifact_path
        original = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(original)
            payload["ensemble_validation_rmse"] = (
                payload["best_individual_validation_rmse"] * 0.5
            )
            payload["ensemble_improved_on_best"] = True
            atomic_write_json(path, payload)
            comparison = compare_compatible_model_runs(self.project.path)
            self.assertEqual(comparison.recommended_model, "ensemble_ai_engine")
            self.assertIn(
                "lower compatible validation RMSE",
                comparison.recommendation_reason,
            )
        finally:
            path.write_text(original, encoding="utf-8")

    def test_snowbuddy_evidence_includes_ensemble_weights_and_decision(self):
        project = self.store.open_project(self.project.path, touch=False)
        evidence = build_latest_run_evidence(project)
        self.assertIn("Model: ensemble_ai_engine", evidence)
        self.assertIn("weights:", evidence)
        self.assertIn("Validation RMSE:", evidence)

    @unittest.skipUnless(os.name == "nt" or os.environ.get("DISPLAY"), "GUI required")
    def test_training_results_ui_renders_ensemble_configuration_and_four_families(self):
        try:
            app = StudioApp(project_store=self.store)
        except tk.TclError as exc:
            self.skipTest(f"A desktop display is not available: {exc}")
        try:
            app.withdraw()
            project = self.store.open_project(self.project.path, touch=False)
            app.set_project(project, target_page="results")
            app.update_idletasks()
            page = app.results_page
            self.assertEqual(page.result.model_name, "ensemble_ai_engine")
            page.show_section("configuration")
            app.update_idletasks()
            texts = []
            pending = [page.content_card]
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
            self.assertIn("Ensemble composition", rendered)
            page.show_section("comparison")
            app.update_idletasks()
            self.assertEqual(
                set(page.comparison_metric_chart.metric_values["Validation RMSE"]),
                {
                    "linear_regression",
                    "xgboost",
                    "neural_network",
                    "ensemble_ai_engine",
                },
            )
        finally:
            app.destroy()


class EnsembleFailureToleranceTests(unittest.TestCase):
    def test_one_failed_component_is_recorded_and_two_models_still_form_ensemble(self):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as temp_dir:
            store = ProjectStore(Path(temp_dir) / "library")
            project = store.create_project("Ensemble Failure Tolerance")
            register_neural_network_dataset(project)
            with patch(
                "studio.model_training._run_xgboost_auto_search",
                side_effect=ModelTrainingError("Injected XGBoost failure."),
            ):
                result = submit_model_training_request(
                    ensemble_request(),
                    project_path=project.path,
                )
            self.assertTrue(result.success, result.error_message)
            self.assertEqual(
                set(result.ensemble_weights),
                {"linear_regression", "neural_network"},
            )
            self.assertAlmostEqual(sum(result.ensemble_weights.values()), 1.0)
            self.assertEqual(
                result.component_failures,
                [
                    {
                        "model_name": "xgboost",
                        "error_message": "Injected XGBoost failure.",
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
