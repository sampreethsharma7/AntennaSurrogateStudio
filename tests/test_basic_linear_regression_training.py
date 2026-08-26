import csv
import hashlib
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import joblib

from studio.dataset_registry import register_dataset
from studio.dataset_validation import validate_dataset
from studio.model_training import (
    TRAINING_COMPLETED,
    TRAINING_FAILED,
    ModelTrainingRequest,
    resolve_linear_regression_parameters,
    submit_model_training_request,
)
from studio.parser_engine import (
    IMPORTED_OUTPUT_LABEL,
    TrainingRequest,
    prepare,
)
from studio.project_store import ProjectStore, atomic_write_json


class BasicLinearRegressionTrainingTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        self.store = ProjectStore(Path(self.temp_dir.name) / "library")
        self.project = self.store.create_project("Basic Training Test")

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _auto_request(search_level: str = "medium") -> ModelTrainingRequest:
        return ModelTrainingRequest(
            model_name="linear_regression",
            training_mode="auto",
            search_level=search_level,
        )

    @staticmethod
    def _custom_request(
        *, fit_intercept: bool, positive: bool
    ) -> ModelTrainingRequest:
        return ModelTrainingRequest(
            model_name="linear_regression",
            training_mode="custom",
            custom_hyperparameters={
                "fit_intercept": fit_intercept,
                "positive": positive,
            },
        )

    def _register_dataset(self, row_count: int = 20, *, sample_ids: bool = True):
        input_path = self.project.path / "data" / "prepared" / "inputs.csv"
        output_path = self.project.path / "data" / "prepared" / "outputs.csv"
        input_headers = ["x1", "x2"]
        output_headers = ["target"]
        if sample_ids:
            input_headers.insert(0, "Sample ID")
            output_headers.insert(0, "Sample ID")

        with input_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(input_headers)
            for index in range(1, row_count + 1):
                values = [float(index), float((index * 3) % 7)]
                if sample_ids:
                    values.insert(0, f"Design_{index:03d}")
                writer.writerow(values)

        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(output_headers)
            for index in range(1, row_count + 1):
                x2 = float((index * 3) % 7)
                values = [2.0 * index - 0.5 * x2 + 3.0]
                if sample_ids:
                    values.insert(0, f"Design_{index:03d}")
                writer.writerow(values)

        validation = validate_dataset(
            TrainingRequest(
                input_csv_path=input_path,
                output_csv_path=output_path,
                feature_columns=["x1", "x2"],
                target_columns=["target"],
                sample_id_column="Sample ID" if sample_ids else None,
            )
        )
        return register_dataset(self.project.path, validation)

    def _train(self, search_level: str = "medium"):
        return submit_model_training_request(
            self._auto_request(search_level),
            project_path=self.project.path,
        )

    def test_linear_regression_trains_successfully(self):
        self._register_dataset()

        result = self._train()

        self.assertTrue(result.success)
        self.assertEqual(result.status, TRAINING_COMPLETED)
        self.assertEqual(result.model_name, "linear_regression")
        self.assertEqual(result.training_rows, 16)
        self.assertEqual(result.test_rows, 4)
        self.assertIsNone(result.error_message)

    def test_csv_pair_sample_ids_survive_prepare_register_and_train(self):
        source_input = self.project.path / "source_inputs.csv"
        source_output = self.project.path / "source_outputs.csv"
        original_ids = [f"Antenna_{index:03d}" for index in range(1, 21)]

        with source_input.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Sample ID", "x1", "x2"])
            for index, sample_id in enumerate(original_ids, start=1):
                writer.writerow([sample_id, float(index), float(index % 4)])

        with source_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Sample ID", "target"])
            for index, sample_id in enumerate(original_ids, start=1):
                writer.writerow([sample_id, 2.5 * index - float(index % 4)])

        prepared_input = self.project.path / "data" / "prepared" / "inputs.csv"
        prepared_output = self.project.path / "data" / "prepared" / "outputs.csv"
        prepared = prepare(
            "pair",
            source_input,
            ["x1", "x2"],
            IMPORTED_OUTPUT_LABEL,
            prepared_input,
            prepared_output,
            source_output_path=source_output,
        )
        validation = validate_dataset(
            TrainingRequest(
                input_csv_path=prepared_input,
                output_csv_path=prepared_output,
                feature_columns=prepared.inputs,
                target_columns=prepared.target_columns,
                sample_id_column=prepared.sample_id_column,
            )
        )
        registered = register_dataset(self.project.path, validation)
        result = self._train()

        self.assertEqual(prepared.sample_id_column, "Sample ID")
        self.assertEqual(registered.sample_id_column, "Sample ID")
        with registered.input_csv_path.open(
            newline="", encoding="utf-8"
        ) as handle:
            registered_input_rows = list(csv.reader(handle))
        self.assertEqual(registered_input_rows[0][0], "Sample ID")
        self.assertEqual(
            [row[0] for row in registered_input_rows[1:]],
            original_ids,
        )
        self.assertTrue(result.success)
        self.assertTrue(
            {row["sample_id"] for row in result.predictions}.issubset(
                set(original_ids)
            )
        )

    def test_split_is_deterministic(self):
        self._register_dataset()

        first = self._train()
        second = self._train()

        self.assertEqual(
            [row["sample_id"] for row in first.predictions],
            [row["sample_id"] for row in second.predictions],
        )
        self.assertEqual(first.training_rows, second.training_rows)
        self.assertEqual(first.test_rows, second.test_rows)

    def test_repeated_runs_produce_identical_metrics_and_predictions(self):
        self._register_dataset()

        first = self._train()
        first_model_bytes = first.model_artifact_path.read_bytes()
        second = self._train()

        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(first.predictions, second.predictions)
        self.assertEqual(first.search_results, second.search_results)
        self.assertEqual(first.best_parameters, second.best_parameters)
        self.assertEqual(
            first.best_validation_rmse,
            second.best_validation_rmse,
        )
        self.assertEqual(first.run_number, 1)
        self.assertEqual(second.run_number, 2)
        self.assertEqual(first.run_id, "run-0001")
        self.assertEqual(second.run_id, "run-0002")
        self.assertNotEqual(first.run_directory, second.run_directory)
        self.assertTrue(first.model_artifact_path.is_file())
        self.assertEqual(first.model_artifact_path.read_bytes(), first_model_bytes)
        self.assertTrue(second.model_artifact_path.is_file())
        manifest = json.loads(
            (self.project.path / "project.json").read_text(encoding="utf-8")
        )
        training = manifest["model_training"]
        self.assertEqual(training["run_count"], 2)
        self.assertEqual(training["latest_run_number"], 2)
        self.assertEqual(training["latest_run_id"], "run-0002")
        self.assertEqual(
            [record["display_name"] for record in training["runs"]],
            ["Run 1", "Run 2"],
        )

    def test_metrics_are_numeric(self):
        self._register_dataset()

        result = self._train()

        self.assertEqual(set(result.metrics), {"MAE", "RMSE", "R²"})
        for value in result.metrics.values():
            self.assertIsInstance(value, float)
            self.assertTrue(value == value)

    def test_predictions_preserve_sample_ids_and_required_values(self):
        self._register_dataset()

        result = self._train()

        self.assertEqual(len(result.predictions), result.test_rows)
        for prediction in result.predictions:
            self.assertTrue(str(prediction["sample_id"]).startswith("Design_"))
            self.assertTrue(
                {
                    "sample_id",
                    "actual_value",
                    "predicted_value",
                    "residual",
                }.issubset(prediction)
            )
            self.assertAlmostEqual(
                prediction["residual"],
                prediction["actual_value"] - prediction["predicted_value"],
            )

    def test_missing_values_stop_training(self):
        dataset = self._register_dataset()
        with dataset.output_csv_path.open(encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        rows[3][1] = ""
        with dataset.output_csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            csv.writer(handle).writerows(rows)
        dataset_manifest = json.loads(
            dataset.manifest_path.read_text(encoding="utf-8")
        )
        dataset_manifest["files"]["output"]["sha256"] = hashlib.sha256(
            dataset.output_csv_path.read_bytes()
        ).hexdigest()
        atomic_write_json(dataset.manifest_path, dataset_manifest)

        result = self._train()

        self.assertFalse(result.success)
        self.assertEqual(result.status, TRAINING_FAILED)
        self.assertIn("empty value", result.error_message)
        self.assertEqual(
            list((self.project.path / "models").rglob("metrics.json")),
            [],
        )

    def test_too_few_rows_stop_training(self):
        self._register_dataset(row_count=4)

        result = self._train()

        self.assertFalse(result.success)
        self.assertIn("At least 5 usable rows", result.error_message)
        self.assertIn("2-fold validation", result.error_message)

    def test_training_requires_a_stage_zero_registered_dataset(self):
        result = self._train()

        self.assertFalse(result.success)
        self.assertIn("no Stage 0 validated dataset", result.error_message)

    def test_missing_selected_feature_stops_training(self):
        dataset = self._register_dataset()
        manifest = json.loads(dataset.manifest_path.read_text(encoding="utf-8"))
        manifest["contract"]["feature_columns"] = ["missing_feature"]
        atomic_write_json(dataset.manifest_path, manifest)

        result = self._train()

        self.assertFalse(result.success)
        self.assertIn("missing_feature", result.error_message)

    def test_missing_selected_target_stops_training(self):
        dataset = self._register_dataset()
        manifest = json.loads(dataset.manifest_path.read_text(encoding="utf-8"))
        manifest["contract"]["target_columns"] = ["missing_target"]
        atomic_write_json(dataset.manifest_path, manifest)

        result = self._train()

        self.assertFalse(result.success)
        self.assertIn("missing_target", result.error_message)

    def test_unsupported_model_stops_training(self):
        self._register_dataset()
        request = self._auto_request()
        request.model_name = "random_forest"

        result = submit_model_training_request(
            request,
            project_path=self.project.path,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status, TRAINING_FAILED)
        self.assertIn("Unsupported model", result.error_message)

    def test_auto_mode_uses_selected_parameters(self):
        self._register_dataset()

        result = self._train()

        self.assertTrue(result.success)
        self.assertEqual(result.training_mode, "auto")
        self.assertEqual(result.search_level, "medium")
        self.assertEqual(
            result.parameters_used,
            {"fit_intercept": True, "positive": False},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model = joblib.load(result.model_artifact_path)
        self.assertTrue(model.fit_intercept)
        self.assertFalse(model.positive)

    def test_medium_search_evaluates_exact_configurations_and_three_folds(self):
        self._register_dataset()

        result = self._train("medium")

        self.assertTrue(result.success)
        self.assertEqual(result.configurations_evaluated, 2)
        self.assertEqual(result.cross_validation_folds, 3)
        self.assertEqual(
            [entry["parameters"] for entry in result.search_results],
            [
                {"fit_intercept": True, "positive": False},
                {"fit_intercept": False, "positive": False},
            ],
        )
        self.assertTrue(all(len(entry["fold_rmse"]) == 3 for entry in result.search_results))
        self.assertEqual(result.test_metrics, result.metrics)
        self.assertIsInstance(result.best_validation_rmse, float)

    def test_high_search_evaluates_exact_configurations_and_five_folds(self):
        self._register_dataset()

        result = self._train("high")

        self.assertTrue(result.success)
        self.assertEqual(result.search_level, "high")
        self.assertEqual(result.configurations_evaluated, 4)
        self.assertEqual(result.cross_validation_folds, 5)
        self.assertEqual(
            [entry["parameters"] for entry in result.search_results],
            [
                {"fit_intercept": True, "positive": False},
                {"fit_intercept": False, "positive": False},
                {"fit_intercept": True, "positive": True},
                {"fit_intercept": False, "positive": True},
            ],
        )

    def test_high_fold_count_is_reduced_for_a_smaller_training_set(self):
        self._register_dataset(row_count=6)

        result = self._train("high")

        self.assertTrue(result.success)
        self.assertEqual(result.training_rows, 4)
        self.assertEqual(result.cross_validation_folds, 4)

    def test_auto_cross_validation_receives_only_the_training_partition(self):
        self._register_dataset()
        observed_row_counts = []

        def observe_training_rows(features, targets, parameters, fold_count):
            observed_row_counts.append((len(features), len(targets), fold_count))
            return [1.0] * fold_count

        with patch(
            "studio.model_training._cross_validate_linear_regression_configuration",
            side_effect=observe_training_rows,
        ):
            result = self._train("medium")

        self.assertTrue(result.success)
        self.assertEqual(observed_row_counts, [(16, 16, 3), (16, 16, 3)])
        self.assertEqual(result.test_rows, 4)

    def test_test_set_is_not_used_for_model_selection(self):
        from sklearn.model_selection import train_test_split

        dataset = self._register_dataset()
        first = self._train("medium")
        _, test_indices = train_test_split(
            list(range(20)),
            test_size=0.20,
            random_state=42,
            shuffle=True,
        )
        with dataset.output_csv_path.open(encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        for index in test_indices:
            rows[index + 1][1] = str(float(rows[index + 1][1]) + 10000.0)
        with dataset.output_csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            csv.writer(handle).writerows(rows)
        dataset_manifest = json.loads(
            dataset.manifest_path.read_text(encoding="utf-8")
        )
        dataset_manifest["files"]["output"]["sha256"] = hashlib.sha256(
            dataset.output_csv_path.read_bytes()
        ).hexdigest()
        atomic_write_json(dataset.manifest_path, dataset_manifest)

        second = self._train("medium")

        self.assertTrue(second.success)
        self.assertEqual(first.search_results, second.search_results)
        self.assertEqual(first.best_parameters, second.best_parameters)
        self.assertEqual(
            first.best_validation_rmse,
            second.best_validation_rmse,
        )
        self.assertNotEqual(first.test_metrics, second.test_metrics)

    def test_lowest_validation_rmse_selects_the_configuration(self):
        self._register_dataset()

        def configured_scores(features, targets, parameters, fold_count):
            score = 0.25 if not parameters["fit_intercept"] else 1.0
            return [score] * fold_count

        with patch(
            "studio.model_training._cross_validate_linear_regression_configuration",
            side_effect=configured_scores,
        ):
            result = self._train("medium")

        self.assertTrue(result.success)
        self.assertEqual(
            result.best_parameters,
            {"fit_intercept": False, "positive": False},
        )
        self.assertEqual(result.parameters_used, result.best_parameters)
        self.assertEqual(result.best_validation_rmse, 0.25)

    def test_auto_search_tie_breaking_is_deterministic(self):
        self._register_dataset()

        with patch(
            "studio.model_training._cross_validate_linear_regression_configuration",
            side_effect=lambda features, targets, parameters, folds: [1.0] * folds,
        ):
            result = self._train("high")

        self.assertTrue(result.success)
        self.assertEqual(
            result.best_parameters,
            {"fit_intercept": True, "positive": False},
        )

    def test_failed_configuration_does_not_stop_auto_search(self):
        self._register_dataset()

        def fail_one_configuration(features, targets, parameters, fold_count):
            if parameters == {"fit_intercept": True, "positive": False}:
                raise ValueError("candidate failure")
            return [0.5] * fold_count

        with patch(
            "studio.model_training._cross_validate_linear_regression_configuration",
            side_effect=fail_one_configuration,
        ):
            result = self._train("high")

        self.assertTrue(result.success)
        self.assertFalse(result.search_results[0]["success"])
        self.assertIn("candidate failure", result.search_results[0]["error_message"])
        self.assertTrue(all(entry["success"] for entry in result.search_results[1:]))
        self.assertEqual(
            result.best_parameters,
            {"fit_intercept": False, "positive": False},
        )

    def test_all_failed_configurations_return_a_clear_failure(self):
        self._register_dataset()

        with patch(
            "studio.model_training._cross_validate_linear_regression_configuration",
            side_effect=ValueError("candidate failure"),
        ):
            result = self._train("medium")

        self.assertFalse(result.success)
        self.assertEqual(result.status, TRAINING_FAILED)
        self.assertIn("all Linear Regression configurations", result.error_message)
        self.assertNotIn("Traceback", result.error_message)
        self.assertFalse((self.project.path / "models" / "runs").exists())

    def test_unsupported_mutated_search_level_returns_a_clear_failure(self):
        self._register_dataset()
        request = self._auto_request()
        request.search_level = "extreme"

        result = submit_model_training_request(
            request,
            project_path=self.project.path,
        )

        self.assertFalse(result.success)
        self.assertIn("Unsupported Auto search level", result.error_message)
        self.assertFalse((self.project.path / "models" / "runs").exists())

    def test_custom_mode_uses_true_false_parameters(self):
        self._assert_custom_parameters(
            fit_intercept=True,
            positive=False,
        )

    def test_custom_mode_uses_false_false_parameters(self):
        self._assert_custom_parameters(
            fit_intercept=False,
            positive=False,
        )

    def test_custom_mode_uses_true_true_parameters(self):
        self._assert_custom_parameters(
            fit_intercept=True,
            positive=True,
        )

    def _assert_custom_parameters(
        self, *, fit_intercept: bool, positive: bool
    ) -> None:
        self._register_dataset()
        request = self._custom_request(
            fit_intercept=fit_intercept,
            positive=positive,
        )

        result = submit_model_training_request(
            request,
            project_path=self.project.path,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.training_mode, "custom")
        self.assertIsNone(result.search_level)
        self.assertEqual(result.configurations_evaluated, 0)
        self.assertIsNone(result.cross_validation_folds)
        self.assertEqual(result.search_results, [])
        self.assertIsNone(result.best_validation_rmse)
        self.assertIsNone(result.auto_search_results_artifact_path)
        self.assertEqual(
            result.parameters_used,
            {
                "fit_intercept": fit_intercept,
                "positive": positive,
            },
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model = joblib.load(result.model_artifact_path)
        self.assertIs(model.fit_intercept, fit_intercept)
        self.assertIs(model.positive, positive)

    def test_parameter_resolver_returns_exact_auto_and_custom_values(self):
        self.assertEqual(
            resolve_linear_regression_parameters(self._auto_request()),
            {"fit_intercept": True, "positive": False},
        )
        self.assertEqual(
            resolve_linear_regression_parameters(
                self._custom_request(fit_intercept=False, positive=True)
            ),
            {"fit_intercept": False, "positive": True},
        )

    def test_separate_custom_configurations_create_separate_runs(self):
        self._register_dataset()

        first = submit_model_training_request(
            self._custom_request(fit_intercept=False, positive=False),
            project_path=self.project.path,
        )
        first_config_path = first.training_config_artifact_path
        second = submit_model_training_request(
            self._custom_request(fit_intercept=True, positive=True),
            project_path=self.project.path,
        )
        first_config = json.loads(
            first_config_path.read_text(encoding="utf-8")
        )

        self.assertTrue(first.success)
        self.assertTrue(second.success)
        self.assertEqual(first.run_number, 1)
        self.assertEqual(second.run_number, 2)
        self.assertNotEqual(first.run_directory, second.run_directory)
        self.assertEqual(
            first_config["parameters_used"],
            {"fit_intercept": False, "positive": False},
        )
        self.assertEqual(
            json.loads(
                second.training_config_artifact_path.read_text(
                    encoding="utf-8"
                )
            )["parameters_used"],
            {"fit_intercept": True, "positive": True},
        )
        self.assertTrue(first_config_path.is_file())

    def test_invalid_mutated_custom_request_does_not_start_training(self):
        self._register_dataset()
        request = self._custom_request(fit_intercept=True, positive=False)
        request.custom_hyperparameters = {
            "fit_intercept": "not-a-boolean",
            "positive": False,
        }

        result = submit_model_training_request(
            request,
            project_path=self.project.path,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status, TRAINING_FAILED)
        self.assertIn("must be Boolean", result.error_message)
        self.assertNotIn("Traceback", result.error_message)
        self.assertFalse((self.project.path / "models" / "runs").exists())

    def test_high_search_level_trains_with_auto_search(self):
        self._register_dataset()
        request = ModelTrainingRequest(
            model_name="linear_regression",
            training_mode="auto",
            search_level="high",
        )

        result = submit_model_training_request(
            request,
            project_path=self.project.path,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.search_level, "high")
        self.assertEqual(result.configurations_evaluated, 4)
        self.assertEqual(result.cross_validation_folds, 5)

    def test_artifacts_and_project_manifest_are_saved(self):
        dataset = self._register_dataset()

        result = self._train()

        self.assertTrue(result.model_artifact_path.is_file())
        self.assertTrue(result.metrics_artifact_path.is_file())
        self.assertTrue(result.predictions_artifact_path.is_file())
        self.assertTrue(result.training_config_artifact_path.is_file())
        self.assertTrue(result.auto_search_results_artifact_path.is_file())
        self.assertEqual(result.run_number, 1)
        self.assertEqual(result.run_id, "run-0001")
        self.assertEqual(
            result.run_directory,
            self.project.path / "models" / "runs" / "run-0001",
        )
        self.assertTrue((result.run_directory / "run.json").is_file())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model = joblib.load(result.model_artifact_path)
        self.assertTrue(model.fit_intercept)
        self.assertFalse(model.positive)
        saved_metrics = json.loads(
            result.metrics_artifact_path.read_text(encoding="utf-8")
        )
        self.assertEqual(saved_metrics, result.metrics)
        saved_training_config = json.loads(
            result.training_config_artifact_path.read_text(encoding="utf-8")
        )
        self.assertEqual(
            saved_training_config,
            {
                "model_name": "linear_regression",
                "training_mode": "auto",
                "search_level": "medium",
                "parameters_used": {
                    "fit_intercept": True,
                    "positive": False,
                },
            },
        )
        run_manifest = json.loads(
            (result.run_directory / "run.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            run_manifest["parameters_used"],
            {"fit_intercept": True, "positive": False},
        )
        saved_search = json.loads(
            result.auto_search_results_artifact_path.read_text(encoding="utf-8")
        )
        self.assertEqual(saved_search["search_level"], "medium")
        self.assertEqual(saved_search["configurations_evaluated"], 2)
        self.assertEqual(
            saved_search["configurations_tested"],
            [entry["parameters"] for entry in result.search_results],
        )
        self.assertEqual(saved_search["cross_validation_folds"], 3)
        self.assertEqual(saved_search["search_results"], result.search_results)
        self.assertEqual(saved_search["best_parameters"], result.best_parameters)
        self.assertEqual(saved_search["test_metrics"], result.test_metrics)
        self.assertEqual(
            run_manifest["artifacts"]["auto_search_results"],
            "models/runs/run-0001/auto_search_results.json",
        )
        with result.predictions_artifact_path.open(
            newline="", encoding="utf-8"
        ) as handle:
            saved_predictions = list(csv.DictReader(handle))
        self.assertEqual(len(saved_predictions), len(result.predictions))

        project_manifest = json.loads(
            (self.project.path / "project.json").read_text(encoding="utf-8")
        )
        self.assertEqual(project_manifest["workflow"]["stage"], "model_trained")
        training = project_manifest["model_training"]
        self.assertEqual(training["status"], TRAINING_COMPLETED)
        self.assertEqual(training["dataset_id"], dataset.dataset_id)
        self.assertEqual(training["run_count"], 1)
        self.assertEqual(training["latest_run_number"], 1)
        self.assertEqual(training["latest_run_id"], "run-0001")
        self.assertEqual(len(training["runs"]), 1)
        self.assertEqual(training["configuration"]["test_size"], 0.20)
        self.assertEqual(training["configuration"]["random_state"], 42)
        self.assertEqual(
            training["parameters_used"],
            {"fit_intercept": True, "positive": False},
        )
        self.assertEqual(
            training["auto_search"]["best_parameters"],
            result.best_parameters,
        )

    def test_legacy_flat_artifacts_are_preserved_as_run_one(self):
        dataset = self._register_dataset()
        models_root = self.project.path / "models"
        legacy_model = models_root / "linear_regression_model.joblib"
        legacy_metrics = models_root / "metrics.json"
        legacy_predictions = models_root / "test_predictions.csv"
        legacy_model.write_bytes(b"legacy-model-artifact")
        legacy_metrics.write_text('{"MAE": 1.0}\n', encoding="utf-8")
        legacy_predictions.write_text(
            "sample_id,actual_value,predicted_value,residual\n"
            "Legacy_001,1,1,0\n",
            encoding="utf-8",
        )
        self.store.update_project(
            self.project,
            {
                "workflow": {"stage": "model_trained", "completed_steps": 3},
                "model_training": {
                    "status": TRAINING_COMPLETED,
                    "model_name": "linear_regression",
                    "dataset_id": dataset.dataset_id,
                    "trained_at": "2026-08-05T00:00:00+00:00",
                    "configuration": {"training_mode": "auto"},
                    "training_rows": 16,
                    "test_rows": 4,
                    "metrics": {"MAE": 1.0, "RMSE": 1.0, "R²": 0.0},
                    "artifacts": {
                        "model": "models/linear_regression_model.joblib",
                        "metrics": "models/metrics.json",
                        "predictions": "models/test_predictions.csv",
                    },
                },
            },
        )

        result = self._train()

        self.assertTrue(result.success)
        self.assertEqual(result.run_number, 2)
        migrated = models_root / "runs" / "run-0001"
        self.assertEqual(
            (migrated / "linear_regression_model.joblib").read_bytes(),
            b"legacy-model-artifact",
        )
        self.assertTrue((migrated / "metrics.json").is_file())
        self.assertTrue((migrated / "test_predictions.csv").is_file())
        self.assertTrue((migrated / "training_config.json").is_file())
        migrated_record = json.loads(
            (migrated / "run.json").read_text(encoding="utf-8")
        )
        self.assertTrue(migrated_record["migrated_from_legacy_layout"])
        manifest = json.loads(
            (self.project.path / "project.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["model_training"]["run_count"], 2)
        self.assertEqual(manifest["model_training"]["latest_run_number"], 2)


if __name__ == "__main__":
    unittest.main()
