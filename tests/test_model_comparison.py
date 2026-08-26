import csv
import json
import tempfile
import unittest
from pathlib import Path

from studio.dataset_registry import register_dataset
from studio.dataset_validation import validate_dataset
from studio.model_comparison import (
    VALIDATION_TIE_ABSOLUTE_TOLERANCE,
    compare_compatible_model_runs,
)
from studio.model_training import (
    XGBOOST_CUSTOM_DEFAULTS,
    ModelTrainingRequest,
    submit_model_training_request,
)
from studio.parser_engine import TrainingRequest
from studio.project_store import ProjectStore, atomic_write_json
from studio.training_results import load_latest_training_results


def register_comparison_dataset(project, *, offset: float = 0.0) -> None:
    input_path = project.path / "data" / "prepared" / "inputs.csv"
    output_path = project.path / "data" / "prepared" / "outputs.csv"
    with input_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Sample ID", "width", "height", "gap"])
        for index in range(1, 25):
            writer.writerow(
                [
                    f"Design_{index:03d}",
                    index / 10 + offset,
                    (index % 7) / 5,
                    (index % 5) / 8,
                ]
            )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Sample ID", "gain_0", "gain_1"])
        for index in range(1, 25):
            width = index / 10 + offset
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
    register_dataset(project.path, validation)


def auto_request(model_name: str) -> ModelTrainingRequest:
    return ModelTrainingRequest(
        model_name=model_name,
        training_mode="auto",
        search_level="medium",
        custom_hyperparameters=None,
    )


def set_validation_rmse(result, value: float) -> None:
    path = result.auto_search_results_artifact_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["best_validation_rmse"] = value
    atomic_write_json(path, payload)


class ModelComparisonTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        self.store = ProjectStore(Path(self.temp_dir.name) / "library")
        self.project = self.store.create_project("Model Comparison")
        register_comparison_dataset(self.project)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _train(self, model_name: str):
        result = submit_model_training_request(
            auto_request(model_name),
            project_path=self.project.path,
        )
        self.assertTrue(result.success, result.error_message)
        return result

    def test_detects_compatible_linear_and_xgboost_runs(self):
        linear = self._train("linear_regression")
        xgboost = self._train("xgboost")

        comparison = compare_compatible_model_runs(self.project.path)

        self.assertEqual(comparison.compatible_run_count, 2)
        self.assertEqual(comparison.incompatible_run_count, 0)
        self.assertEqual(
            comparison.family("linear_regression").best_run.run_id,
            linear.run_id,
        )
        self.assertEqual(comparison.family("xgboost").best_run.run_id, xgboost.run_id)
        self.assertEqual(comparison.feature_columns, ("width", "height", "gap"))
        self.assertEqual(comparison.target_columns, ("gain_0", "gain_1"))
        self.assertEqual(comparison.test_size, 0.20)
        self.assertEqual(comparison.random_state, 42)

    def test_incompatible_split_is_excluded(self):
        linear = self._train("linear_regression")
        xgboost = self._train("xgboost")
        run_manifest_path = xgboost.run_directory / "run.json"
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        run_manifest["configuration"]["test_size"] = 0.25
        atomic_write_json(run_manifest_path, run_manifest)

        comparison = compare_compatible_model_runs(
            self.project.path,
            anchor_run_id=linear.run_id,
        )

        self.assertEqual(comparison.compatible_run_count, 1)
        self.assertEqual(comparison.incompatible_run_count, 1)
        self.assertIsNone(comparison.family("xgboost").best_run)
        self.assertIsNone(comparison.recommended_model)

    def test_different_registered_dataset_is_excluded(self):
        linear = self._train("linear_regression")
        register_comparison_dataset(self.project, offset=10.0)
        xgboost = self._train("xgboost")

        comparison = compare_compatible_model_runs(
            self.project.path,
            anchor_run_id=linear.run_id,
        )

        self.assertEqual(comparison.compatible_run_count, 1)
        self.assertEqual(comparison.incompatible_run_count, 1)
        self.assertIsNone(comparison.family("xgboost").best_run)
        self.assertNotEqual(
            load_latest_training_results(
                self.project.path,
                run_id=linear.run_id,
            ).dataset_id,
            load_latest_training_results(
                self.project.path,
                run_id=xgboost.run_id,
            ).dataset_id,
        )

    def test_recommendation_uses_lower_validation_rmse_not_test_metrics(self):
        linear = self._train("linear_regression")
        xgboost = self._train("xgboost")
        set_validation_rmse(linear, 0.8)
        set_validation_rmse(xgboost, 0.2)

        comparison = compare_compatible_model_runs(self.project.path)

        self.assertEqual(comparison.recommended_model, "xgboost")
        self.assertEqual(comparison.recommendation_title, "Recommended Model: XGBoost")
        self.assertIn("lower compatible validation RMSE", comparison.recommendation_reason)
        self.assertIn("not used", comparison.recommendation_reason)

    def test_tie_is_deterministically_resolved_to_linear_regression(self):
        linear = self._train("linear_regression")
        xgboost = self._train("xgboost")
        score = 0.4
        set_validation_rmse(linear, score)
        set_validation_rmse(
            xgboost,
            score + VALIDATION_TIE_ABSOLUTE_TOLERANCE / 2,
        )

        first = compare_compatible_model_runs(self.project.path)
        second = compare_compatible_model_runs(self.project.path)

        self.assertEqual(first.recommended_model, "linear_regression")
        self.assertEqual(first, second)
        self.assertIn("simpler model family", first.recommendation_reason)

    def test_missing_validation_does_not_fall_back_to_test_performance(self):
        self._train("linear_regression")
        custom = ModelTrainingRequest(
            model_name="xgboost",
            training_mode="custom",
            search_level=None,
            custom_hyperparameters=dict(XGBOOST_CUSTOM_DEFAULTS),
        )
        result = submit_model_training_request(custom, project_path=self.project.path)
        self.assertTrue(result.success, result.error_message)

        comparison = compare_compatible_model_runs(self.project.path)

        xgboost = comparison.family("xgboost")
        self.assertEqual(xgboost.compatible_run_count, 1)
        self.assertEqual(xgboost.missing_validation_count, 1)
        self.assertIsNone(xgboost.best_run)
        self.assertIsNone(comparison.recommended_model)
        self.assertIn("Test metrics alone", comparison.recommendation_reason)

    def test_best_valid_run_per_family_is_lowest_validation_run(self):
        first_linear = self._train("linear_regression")
        second_linear = self._train("linear_regression")
        xgboost = self._train("xgboost")
        set_validation_rmse(first_linear, 0.25)
        set_validation_rmse(second_linear, 0.60)
        set_validation_rmse(xgboost, 0.50)

        comparison = compare_compatible_model_runs(self.project.path)

        linear_family = comparison.family("linear_regression")
        self.assertEqual(linear_family.compatible_run_count, 2)
        self.assertEqual(linear_family.best_run.run_id, first_linear.run_id)
        self.assertEqual(comparison.recommended_model, "linear_regression")

    def test_specific_completed_run_can_be_loaded_without_changing_latest_default(self):
        first = self._train("linear_regression")
        second = self._train("linear_regression")

        selected = load_latest_training_results(self.project.path, run_id=first.run_id)
        latest = load_latest_training_results(self.project.path)

        self.assertEqual(selected.run_id, first.run_id)
        self.assertEqual(latest.run_id, second.run_id)


if __name__ == "__main__":
    unittest.main()
