import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from studio.dataset_registry import register_dataset
from studio.dataset_validation import validate_dataset
from studio.inference import InferenceRequest, submit_inference_request
from studio.model_book import save_model_book, set_active_model_book
from studio.model_training import (
    TRAINING_COMPLETED,
    XGBOOST_AUTO_SEARCH_CONFIGURATIONS,
    XGBOOST_CUSTOM_DEFAULTS,
    XGBOOST_CUSTOM_PARAMETER_NAMES,
    ModelTrainingRequest,
    resolve_xgboost_parameters,
    submit_model_training_request,
)
from studio.parser_engine import TrainingRequest
from studio.project_store import ProjectStore
from studio.training_results import load_latest_training_results


class BasicXGBoostTrainingTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        self.store = ProjectStore(Path(self.temp_dir.name) / "library")
        self.project = self.store.create_project("XGBoost Training Test")
        self._register_multi_output_dataset()

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _request(search_level: str = "medium") -> ModelTrainingRequest:
        return ModelTrainingRequest(
            model_name="xgboost",
            training_mode="auto",
            search_level=search_level,
            custom_hyperparameters=None,
        )

    @staticmethod
    def _custom_request(**overrides) -> ModelTrainingRequest:
        parameters = dict(XGBOOST_CUSTOM_DEFAULTS)
        parameters.update(overrides)
        return ModelTrainingRequest(
            model_name="xgboost",
            training_mode="custom",
            search_level=None,
            custom_hyperparameters=parameters,
        )

    def _register_multi_output_dataset(self, row_count: int = 30) -> None:
        input_path = self.project.path / "data" / "prepared" / "inputs.csv"
        output_path = self.project.path / "data" / "prepared" / "outputs.csv"
        with input_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Sample ID", "width", "height", "gap"])
            for index in range(1, row_count + 1):
                writer.writerow(
                    [
                        f"Design_{index:03d}",
                        index / 10,
                        (index % 7) / 5,
                        (index % 5) / 8,
                    ]
                )
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Sample ID", "gain_0", "gain_1"])
            for index in range(1, row_count + 1):
                width = index / 10
                height = (index % 7) / 5
                gap = (index % 5) / 8
                writer.writerow(
                    [
                        f"Design_{index:03d}",
                        width * width + 0.5 * height - gap,
                        1.5 * width + height * gap,
                    ]
                )
        validation = validate_dataset(
            TrainingRequest(
                input_csv_path=input_path,
                output_csv_path=output_path,
                feature_columns=["width", "height", "gap"],
                target_columns=["gain_0", "gain_1"],
                sample_id_column="Sample ID",
            )
        )
        register_dataset(self.project.path, validation)

    def _train(self, request: ModelTrainingRequest | None = None):
        return submit_model_training_request(
            request or self._request(),
            project_path=self.project.path,
        )

    def test_xgboost_request_supports_auto_search_and_valid_custom(self):
        request = self._request()
        self.assertEqual(request.model_name, "xgboost")
        self.assertEqual(request.search_level, "medium")
        custom = self._custom_request(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
        )
        self.assertEqual(custom.training_mode, "custom")
        self.assertEqual(
            custom.custom_hyperparameters,
            {
                "n_estimators": 80,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
            },
        )
        resolved = resolve_xgboost_parameters(custom)
        for name in XGBOOST_CUSTOM_PARAMETER_NAMES:
            self.assertEqual(resolved[name], custom.custom_hyperparameters[name])
        self.assertEqual(resolved["random_state"], 42)
        self.assertEqual(resolved["n_jobs"], 1)
        with self.assertRaisesRegex(ValueError, "requires a search level"):
            ModelTrainingRequest(
                model_name="xgboost",
                training_mode="auto",
                search_level=None,
            )

    def test_invalid_custom_xgboost_parameters_are_rejected(self):
        invalid_values = (
            ("n_estimators", 0, "between 1 and 5000"),
            ("n_estimators", 2.5, "must be an integer"),
            ("max_depth", 65, "between 1 and 64"),
            ("learning_rate", 0.0, "greater than 0"),
            ("subsample", 1.1, "no greater than 1"),
            ("colsample_bytree", float("nan"), "finite number"),
        )
        for name, value, message in invalid_values:
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(ValueError, message):
                    self._custom_request(**{name: value})

        missing = dict(XGBOOST_CUSTOM_DEFAULTS)
        missing.pop("max_depth")
        with self.assertRaisesRegex(ValueError, "requires all parameters"):
            ModelTrainingRequest(
                model_name="xgboost",
                training_mode="custom",
                custom_hyperparameters=missing,
            )
        unknown = dict(XGBOOST_CUSTOM_DEFAULTS)
        unknown["gamma"] = 0.2
        with self.assertRaisesRegex(ValueError, "Unsupported XGBoost parameter: gamma"):
            ModelTrainingRequest(
                model_name="xgboost",
                training_mode="custom",
                custom_hyperparameters=unknown,
            )

    def test_xgboost_trains_multi_output_and_saves_core_artifacts(self):
        result = self._train()

        self.assertTrue(result.success, result.error_message)
        self.assertEqual(result.status, TRAINING_COMPLETED)
        self.assertEqual(result.model_name, "xgboost")
        self.assertEqual(result.training_rows, 24)
        self.assertEqual(result.test_rows, 6)
        self.assertEqual(result.parameters_used, result.best_parameters)
        self.assertEqual(result.search_level, "medium")
        self.assertEqual(result.configurations_evaluated, 3)
        self.assertEqual(result.cross_validation_folds, 3)
        self.assertEqual(len(result.search_results), 3)
        self.assertIsNotNone(result.best_validation_rmse)
        self.assertEqual(set(result.metrics), {"MAE", "RMSE", "R²"})
        self.assertEqual(len(result.predictions), 12)
        self.assertEqual(
            {row["target_name"] for row in result.predictions},
            {"gain_0", "gain_1"},
        )
        self.assertTrue(result.model_artifact_path.is_file())
        self.assertEqual(result.model_artifact_path.name, "xgboost_model.joblib")
        self.assertTrue(result.metrics_artifact_path.is_file())
        self.assertTrue(result.predictions_artifact_path.is_file())
        self.assertTrue(result.training_config_artifact_path.is_file())
        self.assertTrue(result.auto_search_results_artifact_path.is_file())
        self.assertIsInstance(joblib.load(result.model_artifact_path), XGBRegressor)

    def test_medium_and_high_use_bounded_candidate_counts_and_fold_counts(self):
        medium = self._train(self._request("medium"))
        high = self._train(self._request("high"))

        self.assertTrue(medium.success and high.success)
        self.assertEqual(len(XGBOOST_AUTO_SEARCH_CONFIGURATIONS["medium"]), 3)
        self.assertEqual(len(XGBOOST_AUTO_SEARCH_CONFIGURATIONS["high"]), 6)
        self.assertEqual(medium.configurations_evaluated, 3)
        self.assertEqual(medium.cross_validation_folds, 3)
        self.assertEqual(high.configurations_evaluated, 6)
        self.assertEqual(high.cross_validation_folds, 5)

    def test_repeated_xgboost_runs_are_deterministic_and_separate(self):
        first = self._train()
        second = self._train()

        self.assertTrue(first.success and second.success)
        self.assertEqual(first.search_results, second.search_results)
        self.assertEqual(first.best_parameters, second.best_parameters)
        self.assertEqual(first.best_validation_rmse, second.best_validation_rmse)
        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(first.predictions, second.predictions)
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertNotEqual(first.run_directory, second.run_directory)
        self.assertTrue(first.run_directory.is_dir())
        self.assertTrue(second.run_directory.is_dir())

    def test_xgboost_cross_validation_receives_only_training_partition(self):
        observed_rows = []

        def deterministic_scores(features, targets, parameters, fold_count):
            observed_rows.append((len(features), len(targets), fold_count))
            return [float(parameters["max_depth"])] * fold_count

        with patch(
            "studio.model_training._cross_validate_xgboost_configuration",
            side_effect=deterministic_scores,
        ):
            result = self._train()

        self.assertTrue(result.success, result.error_message)
        self.assertEqual(observed_rows, [(24, 24, 3)] * 3)
        self.assertEqual(result.test_rows, 6)

    def test_xgboost_auto_selection_uses_validation_not_test_data(self):
        first = self._train()
        self.assertTrue(first.success, first.error_message)

        _train_indices, test_indices = train_test_split(
            np.arange(30),
            test_size=0.20,
            random_state=42,
            shuffle=True,
        )
        output_path = self.project.path / "data" / "prepared" / "outputs.csv"
        with output_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        for index in test_indices:
            rows[index + 1][1] = str(float(rows[index + 1][1]) + 10000.0)
            rows[index + 1][2] = str(float(rows[index + 1][2]) - 10000.0)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(rows)
        input_path = self.project.path / "data" / "prepared" / "inputs.csv"
        validation = validate_dataset(
            TrainingRequest(
                input_csv_path=input_path,
                output_csv_path=output_path,
                feature_columns=["width", "height", "gap"],
                target_columns=["gain_0", "gain_1"],
                sample_id_column="Sample ID",
            )
        )
        register_dataset(self.project.path, validation)

        second = self._train()

        self.assertTrue(second.success, second.error_message)
        self.assertEqual(first.search_results, second.search_results)
        self.assertEqual(first.best_parameters, second.best_parameters)
        self.assertEqual(first.best_validation_rmse, second.best_validation_rmse)
        self.assertNotEqual(first.test_metrics, second.test_metrics)

    def test_lowest_xgboost_validation_rmse_is_selected(self):
        def configured_scores(features, targets, parameters, fold_count):
            score = 0.2 if parameters["max_depth"] == 3 else 1.0
            return [score] * fold_count

        with patch(
            "studio.model_training._cross_validate_xgboost_configuration",
            side_effect=configured_scores,
        ):
            result = self._train()

        self.assertTrue(result.success, result.error_message)
        self.assertEqual(result.best_parameters["max_depth"], 3)
        self.assertEqual(result.parameters_used, result.best_parameters)
        self.assertAlmostEqual(result.best_validation_rmse, 0.2)

    def test_failed_xgboost_candidate_does_not_stop_search(self):
        def fail_one(features, targets, parameters, fold_count):
            if parameters["n_estimators"] == 64:
                raise ValueError("candidate failure")
            return [0.5] * fold_count

        with patch(
            "studio.model_training._cross_validate_xgboost_configuration",
            side_effect=fail_one,
        ):
            result = self._train()

        self.assertTrue(result.success, result.error_message)
        self.assertFalse(result.search_results[0]["success"])
        self.assertIn("candidate failure", result.search_results[0]["error_message"])
        self.assertTrue(all(item["success"] for item in result.search_results[1:]))
        saved_search = json.loads(
            result.auto_search_results_artifact_path.read_text(encoding="utf-8")
        )
        self.assertFalse(saved_search["search_results"][0]["success"])
        self.assertIn(
            "candidate failure",
            saved_search["search_results"][0]["error_message"],
        )

    def test_all_failed_xgboost_candidates_return_clear_failure(self):
        with patch(
            "studio.model_training._cross_validate_xgboost_configuration",
            side_effect=ValueError("candidate failure"),
        ):
            result = self._train()

        self.assertFalse(result.success)
        self.assertIn("all XGBoost configurations", result.error_message)
        self.assertNotIn("Traceback", result.error_message)

    def test_custom_xgboost_parameters_reach_estimator_and_artifacts(self):
        request = self._custom_request(
            n_estimators=35,
            max_depth=2,
            learning_rate=0.075,
            subsample=0.85,
            colsample_bytree=0.9,
        )
        result = self._train(request)

        self.assertTrue(result.success, result.error_message)
        self.assertEqual(result.training_mode, "custom")
        expected = resolve_xgboost_parameters(request)
        self.assertEqual(result.parameters_used, expected)
        estimator = joblib.load(result.model_artifact_path)
        estimator_parameters = estimator.get_params()
        for name in XGBOOST_CUSTOM_PARAMETER_NAMES:
            self.assertEqual(estimator_parameters[name], expected[name])

        run_manifest = json.loads(
            (result.run_directory / "run.json").read_text(encoding="utf-8")
        )
        training_config = json.loads(
            result.training_config_artifact_path.read_text(encoding="utf-8")
        )
        self.assertEqual(run_manifest["parameters_used"], expected)
        self.assertEqual(training_config["parameters_used"], expected)
        self.assertEqual(training_config["training_mode"], "custom")

    def test_repeated_custom_xgboost_runs_are_deterministic_and_separate(self):
        request = self._custom_request(
            n_estimators=30,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.85,
        )
        first = self._train(request)
        second = self._train(request)

        self.assertTrue(first.success and second.success)
        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(first.predictions, second.predictions)
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertTrue(first.run_directory.is_dir())
        self.assertTrue(second.run_directory.is_dir())

    def test_training_results_load_xgboost_auto_search(self):
        trained = self._train()

        view = load_latest_training_results(self.project.path)

        self.assertIsNotNone(view)
        self.assertEqual(view.model_name, "xgboost")
        self.assertEqual(view.parameters_used, trained.parameters_used)
        self.assertEqual(view.search_level, "medium")
        self.assertEqual(view.configurations_evaluated, 3)
        self.assertEqual(len(view.auto_candidates), 3)
        self.assertEqual(sum(item.selected for item in view.auto_candidates), 1)
        self.assertEqual(view.metrics, trained.metrics)
        self.assertEqual(view.recommendation_title, "Recommended Configuration Selected")

    def test_training_results_load_custom_xgboost_parameters(self):
        request = self._custom_request(
            n_estimators=28,
            max_depth=2,
            learning_rate=0.06,
            subsample=0.8,
            colsample_bytree=0.75,
        )
        trained = self._train(request)

        view = load_latest_training_results(self.project.path)

        self.assertIsNotNone(view)
        self.assertEqual(view.training_mode, "custom")
        self.assertEqual(view.parameters_used, trained.parameters_used)
        self.assertEqual(view.recommendation_title, "Custom Configuration Evaluated")
        self.assertEqual(
            view.recommendation_statement,
            "Your custom XGBoost configuration was evaluated",
        )

    def test_xgboost_run_can_be_saved_and_used_as_a_model_book(self):
        trained = self._train()
        book = save_model_book(
            self.project.path,
            trained.run_id,
            "XGBoost Auto",
        )
        set_active_model_book(self.project.path, book.book_id)

        self.assertEqual(book.model_name, "xgboost")
        self.assertEqual(book.model_type, "xgboost.sklearn.XGBRegressor")
        self.assertEqual(book.parameters_used, trained.parameters_used)
        self.assertEqual(book.search_level, "medium")
        self.assertEqual(book.validation_metrics["RMSE"], trained.best_validation_rmse)
        inference = submit_inference_request(
            InferenceRequest(
                model_book_id=book.book_id,
                inputs={"width": 1.25, "height": 0.8, "gap": 0.2},
            ),
            project_path=self.project.path,
        )
        self.assertTrue(inference.success, inference.error_message)
        self.assertEqual(inference.target_order, ["gain_0", "gain_1"])
        self.assertEqual(list(inference.predictions), ["gain_0", "gain_1"])

    def test_custom_xgboost_run_can_be_saved_as_a_model_book(self):
        trained = self._train(
            self._custom_request(
                n_estimators=24,
                max_depth=2,
                learning_rate=0.07,
                subsample=0.85,
                colsample_bytree=0.8,
            )
        )
        book = save_model_book(
            self.project.path,
            trained.run_id,
            "XGBoost Custom",
        )

        self.assertEqual(book.training_mode, "custom")
        self.assertEqual(book.parameters_used, trained.parameters_used)

    def test_xgboost_run_manifest_records_selected_parameters_and_search(self):
        result = self._train()
        run_manifest = json.loads(
            (result.run_directory / "run.json").read_text(encoding="utf-8")
        )
        training_config = json.loads(
            result.training_config_artifact_path.read_text(encoding="utf-8")
        )

        self.assertEqual(run_manifest["model_name"], "xgboost")
        self.assertEqual(run_manifest["parameters_used"], result.parameters_used)
        self.assertEqual(run_manifest["auto_search"]["search_level"], "medium")
        self.assertEqual(run_manifest["auto_search"]["configurations_evaluated"], 3)
        self.assertEqual(training_config["model_name"], "xgboost")
        self.assertEqual(training_config["search_level"], "medium")
        saved_search = json.loads(
            result.auto_search_results_artifact_path.read_text(encoding="utf-8")
        )
        self.assertEqual(saved_search["best_parameters"], result.parameters_used)
        self.assertEqual(len(saved_search["configurations_tested"]), 3)
        self.assertEqual(saved_search["test_metrics"], result.test_metrics)
