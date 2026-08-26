import unittest

from studio.model_training import (
    TRAINING_FAILED,
    ModelTrainingRequest,
    submit_model_training_request,
)


class ModelTrainingRequestTests(unittest.TestCase):
    def test_valid_auto_request_with_medium_search(self):
        request = ModelTrainingRequest(
            model_name="linear_regression",
            training_mode="auto",
            search_level="medium",
            custom_hyperparameters=None,
        )

        self.assertEqual(request.model_name, "linear_regression")
        self.assertEqual(request.training_mode, "auto")
        self.assertEqual(request.search_level, "medium")
        self.assertIsNone(request.custom_hyperparameters)

    def test_valid_auto_request_with_high_search(self):
        request = ModelTrainingRequest(
            model_name="linear_regression",
            training_mode="auto",
            search_level="high",
        )

        self.assertEqual(request.search_level, "high")

    def test_auto_request_allows_empty_custom_parameters(self):
        request = ModelTrainingRequest(
            model_name="linear_regression",
            training_mode="auto",
            search_level="medium",
            custom_hyperparameters={},
        )

        self.assertEqual(request.custom_hyperparameters, {})

    def test_auto_request_without_search_level_fails(self):
        with self.assertRaisesRegex(
            ValueError,
            "Auto mode requires a search level",
        ):
            ModelTrainingRequest(
                model_name="linear_regression",
                training_mode="auto",
            )

    def test_auto_request_with_unknown_search_level_fails(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported Auto search level: exhaustive",
        ):
            ModelTrainingRequest(
                model_name="linear_regression",
                training_mode="auto",
                search_level="exhaustive",
            )

    def test_auto_request_with_custom_parameters_fails(self):
        with self.assertRaisesRegex(
            ValueError,
            "Custom hyperparameters cannot be provided in Auto mode",
        ):
            ModelTrainingRequest(
                model_name="linear_regression",
                training_mode="auto",
                search_level="medium",
                custom_hyperparameters={
                    "fit_intercept": True,
                    "positive": False,
                },
            )

    def test_valid_custom_request(self):
        request = ModelTrainingRequest(
            model_name="linear_regression",
            training_mode="custom",
            search_level=None,
            custom_hyperparameters={
                "fit_intercept": True,
                "positive": False,
            },
        )

        self.assertIsNone(request.search_level)
        self.assertEqual(
            request.custom_hyperparameters,
            {"fit_intercept": True, "positive": False},
        )

    def test_custom_request_with_search_level_fails(self):
        with self.assertRaisesRegex(
            ValueError,
            "Search level cannot be used in Custom mode",
        ):
            ModelTrainingRequest(
                model_name="linear_regression",
                training_mode="custom",
                search_level="medium",
                custom_hyperparameters={
                    "fit_intercept": True,
                    "positive": False,
                },
            )

    def test_custom_request_without_hyperparameters_fails(self):
        with self.assertRaisesRegex(
            ValueError,
            "Custom mode requires hyperparameters",
        ):
            ModelTrainingRequest(
                model_name="linear_regression",
                training_mode="custom",
            )

    def test_missing_model_fails(self):
        with self.assertRaisesRegex(ValueError, "A model name is required"):
            ModelTrainingRequest(
                model_name="",
                training_mode="auto",
                search_level="medium",
            )

    def test_unknown_model_fails(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported model: random_forest",
        ):
            ModelTrainingRequest(
                model_name="random_forest",
                training_mode="auto",
                search_level="medium",
            )

    def test_missing_training_mode_fails(self):
        with self.assertRaisesRegex(ValueError, "A training mode is required"):
            ModelTrainingRequest(
                model_name="linear_regression",
                training_mode="",
                search_level="medium",
            )

    def test_unknown_training_mode_fails(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported training mode: guided",
        ):
            ModelTrainingRequest(
                model_name="linear_regression",
                training_mode="guided",
            )

    def test_unknown_custom_parameter_fails(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported Linear Regression parameter: normalize",
        ):
            ModelTrainingRequest(
                model_name="linear_regression",
                training_mode="custom",
                custom_hyperparameters={
                    "fit_intercept": True,
                    "positive": False,
                    "normalize": True,
                },
            )

    def test_custom_request_requires_both_parameters(self):
        with self.assertRaisesRegex(
            ValueError,
            "Custom mode requires both fit_intercept and positive",
        ):
            ModelTrainingRequest(
                model_name="linear_regression",
                training_mode="custom",
                custom_hyperparameters={"fit_intercept": True},
            )

    def test_non_boolean_linear_regression_parameters_fail(self):
        invalid_values = (
            {"fit_intercept": 1, "positive": False},
            {"fit_intercept": True, "positive": "false"},
        )
        for parameters in invalid_values:
            with self.subTest(parameters=parameters):
                with self.assertRaisesRegex(ValueError, "must be Boolean"):
                    ModelTrainingRequest(
                        model_name="linear_regression",
                        training_mode="custom",
                        custom_hyperparameters=parameters,
                    )

    def test_submission_requires_project_context_before_training(self):
        request = ModelTrainingRequest(
            model_name="linear_regression",
            training_mode="auto",
            search_level="medium",
        )

        result = submit_model_training_request(request)

        self.assertFalse(result.success)
        self.assertEqual(result.status, TRAINING_FAILED)
        self.assertIn("open Antenna Surrogate Studio project", result.error_message)


if __name__ == "__main__":
    unittest.main()
