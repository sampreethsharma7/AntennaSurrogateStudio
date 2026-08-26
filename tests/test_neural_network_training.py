import csv
import hashlib
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from studio.dataset_registry import register_dataset
from studio.dataset_validation import validate_dataset
from studio.inference import InferenceRequest, submit_inference_request
from studio.model_book import (
    load_model_book,
    save_model_book,
    set_active_model_book,
)
from studio.model_comparison import compare_compatible_model_runs
from studio.model_training import (
    NEURAL_NETWORK_AUTO_SEARCH_CONFIGURATIONS,
    NEURAL_NETWORK_CUSTOM_DEFAULTS,
    ModelTrainingRequest,
    submit_model_training_request,
)
from studio.parser_engine import TrainingRequest
from studio.project_store import ProjectStore, atomic_write_json
from studio.training_results import load_latest_training_results


def register_neural_network_dataset(project) -> None:
    input_path = project.path / "data" / "prepared" / "inputs.csv"
    output_path = project.path / "data" / "prepared" / "outputs.csv"
    with input_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Sample ID", "width", "height", "gap"])
        for index in range(1, 31):
            writer.writerow(
                [
                    f"Design_{index:03d}",
                    index / 10,
                    (index % 7) / 4,
                    (index % 5) / 6,
                ]
            )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Sample ID", "gain_0", "gain_1", "gain_2"])
        for index in range(1, 31):
            width = index / 10
            height = (index % 7) / 4
            gap = (index % 5) / 6
            writer.writerow(
                [
                    f"Design_{index:03d}",
                    0.8 * width + 0.3 * height - gap,
                    width * height + 0.2 * gap,
                    width * width - 0.5 * height + gap,
                ]
            )
    validation = validate_dataset(
        TrainingRequest(
            input_csv_path=input_path,
            output_csv_path=output_path,
            feature_columns=["width", "height", "gap"],
            target_columns=["gain_0", "gain_1", "gain_2"],
            sample_id_column="Sample ID",
        )
    )
    register_dataset(project.path, validation)


def neural_auto(level: str) -> ModelTrainingRequest:
    return ModelTrainingRequest(
        model_name="neural_network",
        training_mode="auto",
        search_level=level,
        custom_hyperparameters=None,
    )


def neural_custom(**overrides) -> ModelTrainingRequest:
    parameters = {
        name: (list(value) if name == "hidden_layer_sizes" else value)
        for name, value in NEURAL_NETWORK_CUSTOM_DEFAULTS.items()
    }
    parameters.update(overrides)
    return ModelTrainingRequest(
        model_name="neural_network",
        training_mode="custom",
        search_level=None,
        custom_hyperparameters=parameters,
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NeuralNetworkContractTests(unittest.TestCase):
    def test_auto_medium_and_high_requests_are_valid(self):
        self.assertEqual(neural_auto("medium").search_level, "medium")
        self.assertEqual(neural_auto("high").search_level, "high")

    def test_custom_request_normalizes_supported_values(self):
        request = neural_custom(
            hidden_layer_sizes=(48, 24),
            activation="tanh",
            learning_rate_init=0.002,
            batch_size=4,
            max_iter=80,
        )
        self.assertEqual(request.custom_hyperparameters["hidden_layer_sizes"], [48, 24])
        self.assertEqual(request.custom_hyperparameters["activation"], "tanh")

    def test_invalid_custom_values_fail_in_backend_contract(self):
        invalid = (
            ({"hidden_layer_sizes": []}, "hidden_layer_sizes"),
            ({"hidden_layer_sizes": [32, 0]}, "hidden_layer_sizes"),
            ({"activation": "swish"}, "activation"),
            ({"learning_rate_init": 0.0}, "learning_rate_init"),
            ({"batch_size": 0}, "batch_size"),
            ({"max_iter": 0}, "max_iter"),
        )
        for overrides, message in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    neural_custom(**overrides)

    def test_unknown_custom_parameter_fails(self):
        parameters = dict(NEURAL_NETWORK_CUSTOM_DEFAULTS)
        parameters["dropout"] = 0.2
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported Neural Network parameter: dropout",
        ):
            ModelTrainingRequest(
                model_name="neural_network",
                training_mode="custom",
                search_level=None,
                custom_hyperparameters=parameters,
            )


class NeuralNetworkEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        cls.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        cls.store = ProjectStore(Path(cls.temp_dir.name) / "library")
        cls.project = cls.store.create_project("Neural Network End To End")
        register_neural_network_dataset(cls.project)

        cls.medium = submit_model_training_request(
            neural_auto("medium"), project_path=cls.project.path
        )
        if not cls.medium.success:
            raise AssertionError(cls.medium.error_message)
        cls.medium_repeat = submit_model_training_request(
            neural_auto("medium"), project_path=cls.project.path
        )
        if not cls.medium_repeat.success:
            raise AssertionError(cls.medium_repeat.error_message)
        cls.high = submit_model_training_request(
            neural_auto("high"), project_path=cls.project.path
        )
        if not cls.high.success:
            raise AssertionError(cls.high.error_message)
        cls.custom = submit_model_training_request(
            neural_custom(
                hidden_layer_sizes=[24, 12],
                activation="tanh",
                learning_rate_init=0.002,
                batch_size=4,
                max_iter=90,
            ),
            project_path=cls.project.path,
        )
        if not cls.custom.success:
            raise AssertionError(cls.custom.error_message)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_medium_and_high_use_documented_candidate_counts(self):
        self.assertEqual(
            self.medium.configurations_evaluated,
            len(NEURAL_NETWORK_AUTO_SEARCH_CONFIGURATIONS["medium"]),
        )
        self.assertEqual(
            self.high.configurations_evaluated,
            len(NEURAL_NETWORK_AUTO_SEARCH_CONFIGURATIONS["high"]),
        )
        self.assertEqual(self.medium.cross_validation_folds, 3)
        self.assertEqual(self.high.cross_validation_folds, 5)

    def test_multi_output_training_metrics_predictions_and_artifacts(self):
        self.assertEqual(self.medium.model_name, "neural_network")
        self.assertEqual(set(self.medium.metrics), {"MAE", "RMSE", "R²"})
        self.assertEqual(len(self.medium.predictions), self.medium.test_rows * 3)
        self.assertEqual(
            {row["target_name"] for row in self.medium.predictions},
            {"gain_0", "gain_1", "gain_2"},
        )
        self.assertEqual(self.medium.model_artifact_path.name, "neural_network_model.joblib")
        self.assertTrue(self.medium.model_artifact_path.is_file())
        self.assertTrue(self.medium.metrics_artifact_path.is_file())
        self.assertTrue(self.medium.predictions_artifact_path.is_file())
        self.assertTrue(self.medium.auto_search_results_artifact_path.is_file())

    def test_auto_training_is_reproducible(self):
        self.assertEqual(self.medium.metrics, self.medium_repeat.metrics)
        self.assertEqual(self.medium.predictions, self.medium_repeat.predictions)
        self.assertEqual(self.medium.search_results, self.medium_repeat.search_results)
        self.assertEqual(self.medium.best_parameters, self.medium_repeat.best_parameters)

    def test_custom_parameters_reach_the_saved_real_estimator(self):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Setting the shape on a NumPy array has been deprecated.*",
                category=DeprecationWarning,
            )
            estimator = joblib.load(self.custom.model_artifact_path)
        self.assertIsInstance(estimator, Pipeline)
        network = estimator.named_steps["neural_network"]
        self.assertIsInstance(network, MLPRegressor)
        self.assertEqual(tuple(network.hidden_layer_sizes), (24, 12))
        self.assertEqual(network.activation, "tanh")
        self.assertEqual(network.learning_rate_init, 0.002)
        self.assertEqual(network.batch_size, 4)
        self.assertEqual(network.max_iter, 90)
        self.assertEqual(network.random_state, 42)
        self.assertFalse(network.shuffle)

    def test_training_results_load_auto_and_custom_runs(self):
        auto_view = load_latest_training_results(
            self.project.path, run_id=self.high.run_id
        )
        custom_view = load_latest_training_results(
            self.project.path, run_id=self.custom.run_id
        )
        self.assertEqual(auto_view.model_name, "neural_network")
        self.assertEqual(len(auto_view.auto_candidates), self.high.configurations_evaluated)
        self.assertEqual(auto_view.parameters_used, self.high.parameters_used)
        self.assertEqual(custom_view.model_name, "neural_network")
        self.assertEqual(custom_view.parameters_used["hidden_layer_sizes"], [24, 12])
        self.assertEqual(
            custom_view.recommendation_statement,
            "Your custom Neural Network configuration was evaluated",
        )

    def test_model_book_and_multi_output_inference_work_end_to_end(self):
        source_hash = sha256(self.custom.model_artifact_path)
        book = save_model_book(
            self.project.path,
            self.custom.run_id,
            "Reusable Neural Surrogate",
        )
        loaded = load_model_book(self.project.path, book.book_id)
        self.assertEqual(loaded.model_name, "neural_network")
        self.assertEqual(
            loaded.model_type,
            "sklearn.pipeline.Pipeline[StandardScaler,MLPRegressor]",
        )
        self.assertEqual(loaded.target_columns, ["gain_0", "gain_1", "gain_2"])
        self.assertEqual(loaded.parameters_used["hidden_layer_sizes"], [24, 12])
        self.assertEqual(sha256(self.custom.model_artifact_path), source_hash)
        set_active_model_book(self.project.path, book.book_id)
        inference = submit_inference_request(
            InferenceRequest(
                model_book_id=book.book_id,
                inputs={"gap": 0.2, "width": 1.25, "height": 0.75},
            ),
            project_path=self.project.path,
        )
        self.assertTrue(inference.success, inference.error_message)
        self.assertEqual(inference.feature_order, ["width", "height", "gap"])
        self.assertEqual(inference.target_order, ["gain_0", "gain_1", "gain_2"])
        self.assertEqual(list(inference.predictions), inference.target_order)

    def test_three_family_comparison_can_recommend_neural_network(self):
        linear = submit_model_training_request(
            ModelTrainingRequest(
                model_name="linear_regression",
                training_mode="auto",
                search_level="medium",
                custom_hyperparameters=None,
            ),
            project_path=self.project.path,
        )
        xgboost = submit_model_training_request(
            ModelTrainingRequest(
                model_name="xgboost",
                training_mode="auto",
                search_level="medium",
                custom_hyperparameters=None,
            ),
            project_path=self.project.path,
        )
        self.assertTrue(linear.success, linear.error_message)
        self.assertTrue(xgboost.success, xgboost.error_message)
        for result, score in ((linear, 0.6), (xgboost, 0.4), (self.high, 0.2)):
            payload = json.loads(
                result.auto_search_results_artifact_path.read_text(encoding="utf-8")
            )
            payload["best_validation_rmse"] = score
            atomic_write_json(result.auto_search_results_artifact_path, payload)
        comparison = compare_compatible_model_runs(
            self.project.path,
            anchor_run_id=self.high.run_id,
        )
        self.assertEqual(
            tuple(family.model_name for family in comparison.families),
            (
                "linear_regression",
                "xgboost",
                "neural_network",
                "ensemble_ai_engine",
            ),
        )
        self.assertEqual(comparison.recommended_model, "neural_network")
        self.assertEqual(
            comparison.recommendation_title,
            "Recommended Model: Neural Network",
        )


if __name__ == "__main__":
    unittest.main()
