import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from studio.dataset_registry import register_dataset
from studio.dataset_validation import validate_dataset
from studio.inverse_design import (
    INVERSE_DESIGN_COMPLETED,
    INVERSE_DESIGN_FAILED,
    InverseDesignObjective,
    InverseDesignRequest,
    OutputConstraint,
    load_inverse_design_run,
    load_inverse_design_runs,
    submit_inverse_design_request,
)
from studio.model_book import save_model_book, set_active_model_book
from studio.model_training import ModelTrainingRequest, submit_model_training_request
from studio.parser_engine import TrainingRequest
from studio.project_store import ProjectStore


class InverseDesignContractTests(unittest.TestCase):
    def test_valid_contract_preserves_variable_fixed_objective_and_constraints(self):
        request = InverseDesignRequest(
            model_book_id="book-0001",
            variable_bounds={"P2": (0, 5)},
            fixed_inputs={"P3": 2, "P4": 3},
            objective=InverseDesignObjective("gain", "target", 13.5),
            constraints=[OutputConstraint("loss", "less_than_or_equal", value=8)],
        )

        self.assertEqual(request.variable_bounds, {"P2": (0.0, 5.0)})
        self.assertEqual(request.fixed_inputs, {"P3": 2.0, "P4": 3.0})
        self.assertEqual(request.objective.target_value, 13.5)
        self.assertEqual(request.constraints[0].value, 8.0)

    def test_contract_rejects_missing_variables_overlap_and_invalid_bounds(self):
        objective = InverseDesignObjective("gain", "minimize")
        with self.assertRaisesRegex(ValueError, "At least one input"):
            InverseDesignRequest({}, {"P2": 1}, objective)
        with self.assertRaisesRegex(ValueError, "both variable and fixed"):
            InverseDesignRequest({"P2": (0, 1)}, {"P2": 0.5}, objective)
        with self.assertRaisesRegex(ValueError, "must be less"):
            InverseDesignRequest({"P2": (1, 1)}, {}, objective)
        with self.assertRaisesRegex(ValueError, "finite"):
            InverseDesignRequest({"P2": (0, math.inf)}, {}, objective)

    def test_objective_contract_enforces_goal_and_target_value(self):
        with self.assertRaisesRegex(ValueError, "minimize, maximize, or target"):
            InverseDesignObjective("gain", "closest")
        with self.assertRaisesRegex(ValueError, "requires a target"):
            InverseDesignObjective("gain", "target")
        with self.assertRaisesRegex(ValueError, "only be used"):
            InverseDesignObjective("gain", "minimize", 4)

    def test_mean_objective_requires_two_unique_ordered_outputs(self):
        objective = InverseDesignObjective(
            None,
            "maximize",
            aggregation="mean",
            output_names=["theta_0", "theta_1", "theta_2"],
        )

        self.assertEqual(objective.aggregation, "mean")
        self.assertEqual(
            objective.selected_outputs,
            ["theta_0", "theta_1", "theta_2"],
        )
        self.assertEqual(
            objective.display_name,
            "Mean of theta_0 through theta_2",
        )
        with self.assertRaisesRegex(ValueError, "at least two"):
            InverseDesignObjective(
                None,
                "minimize",
                aggregation="mean",
                output_names=["theta_0"],
            )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            InverseDesignObjective(
                None,
                "minimize",
                aggregation="mean",
                output_names=["theta_0", "theta_0"],
            )

    def test_constraint_contract_enforces_threshold_and_range(self):
        with self.assertRaisesRegex(ValueError, "requires a threshold"):
            OutputConstraint("gain", "greater_than_or_equal")
        with self.assertRaisesRegex(ValueError, "requires lower and upper"):
            OutputConstraint("gain", "within_range", lower_bound=1)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            OutputConstraint(
                "gain",
                "within_range",
                lower_bound=3,
                upper_bound=2,
            )

    def test_mean_constraint_requires_two_unique_ordered_outputs(self):
        constraint = OutputConstraint(
            None,
            "greater_than_or_equal",
            value=4,
            aggregation="mean",
            output_names=["theta_0", "theta_1", "theta_2"],
        )

        self.assertEqual(constraint.aggregation, "mean")
        self.assertEqual(
            constraint.selected_outputs,
            ["theta_0", "theta_1", "theta_2"],
        )
        with self.assertRaisesRegex(ValueError, "at least two"):
            OutputConstraint(
                None,
                "less_than_or_equal",
                value=4,
                aggregation="mean",
                output_names=["theta_0"],
            )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            OutputConstraint(
                None,
                "less_than_or_equal",
                value=4,
                aggregation="mean",
                output_names=["theta_0", "theta_0"],
            )


class InverseDesignBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        cls.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        cls.store = ProjectStore(Path(cls.temp_dir.name) / "library")
        cls.project = cls.store.create_project("Inverse Design Backend")
        input_path = cls.project.path / "data" / "prepared" / "inputs.csv"
        output_path = cls.project.path / "data" / "prepared" / "outputs.csv"
        with input_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Sample ID", "P2", "P3", "P4"])
            for index in range(1, 31):
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
            writer.writerow(["Sample ID", "gain", "loss"])
            for index in range(1, 31):
                p2 = float(index)
                p3 = float((index * 2) % 7)
                p4 = float((index * 3) % 5)
                writer.writerow(
                    [
                        f"Design_{index:03d}",
                        2.0 * p2 + 3.0 * p3 - 0.5 * p4 + 1.0,
                        -p2 + 0.25 * p3 + 4.0 * p4 - 2.0,
                    ]
                )
        validation = validate_dataset(
            TrainingRequest(
                input_csv_path=input_path,
                output_csv_path=output_path,
                feature_columns=["P2", "P3", "P4"],
                target_columns=["gain", "loss"],
                sample_id_column="Sample ID",
            )
        )
        register_dataset(cls.project.path, validation)
        run = submit_model_training_request(
            ModelTrainingRequest(
                model_name="linear_regression",
                training_mode="custom",
                search_level=None,
                custom_hyperparameters={
                    "fit_intercept": True,
                    "positive": False,
                },
            ),
            project_path=cls.project.path,
        )
        if not run.success:
            raise AssertionError(run.error_message)
        cls.book = save_model_book(cls.project.path, run.run_id, "Inverse Evaluator")
        set_active_model_book(cls.project.path, cls.book.book_id)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def _request(
        self,
        goal="minimize",
        *,
        target=None,
        constraints=None,
        max_iterations=20,
    ):
        return InverseDesignRequest(
            model_book_id=self.book.book_id,
            variable_bounds={"P2": (0.0, 5.0)},
            fixed_inputs={"P3": 2.0, "P4": 3.0},
            objective=InverseDesignObjective("gain", goal, target),
            constraints=list(constraints or []),
            max_iterations=max_iterations,
            population_multiplier=5,
            random_seed=42,
        )

    def test_minimize_and_maximize_use_saved_surrogate(self):
        minimum = submit_inverse_design_request(
            self._request("minimize"),
            project_path=self.project.path,
        )
        maximum = submit_inverse_design_request(
            self._request("maximize"),
            project_path=self.project.path,
        )

        self.assertTrue(minimum.success, minimum.error_message)
        self.assertTrue(maximum.success, maximum.error_message)
        self.assertEqual(minimum.status, INVERSE_DESIGN_COMPLETED)
        self.assertAlmostEqual(minimum.best_inputs["P2"], 0.0, places=5)
        self.assertAlmostEqual(minimum.objective_value, 5.5, places=5)
        self.assertAlmostEqual(maximum.best_inputs["P2"], 5.0, places=5)
        self.assertAlmostEqual(maximum.objective_value, 15.5, places=5)
        self.assertEqual(
            list(minimum.best_inputs),
            ["P2", "P3", "P4"],
        )

    def test_target_objective_is_reproducible(self):
        first = submit_inverse_design_request(
            self._request("target", target=13.5),
            project_path=self.project.path,
        )
        second = submit_inverse_design_request(
            self._request("target", target=13.5),
            project_path=self.project.path,
        )

        self.assertTrue(first.success, first.error_message)
        self.assertTrue(second.success, second.error_message)
        self.assertAlmostEqual(first.best_inputs["P2"], 4.0, places=5)
        self.assertAlmostEqual(first.objective_value, 13.5, places=5)
        self.assertAlmostEqual(first.objective_score, 0.0, places=5)
        self.assertAlmostEqual(first.target_gap, 0.0, places=5)
        self.assertEqual(first.best_inputs, second.best_inputs)
        self.assertEqual(first.predicted_outputs, second.predicted_outputs)
        saved = json.loads(
            (first.artifact_directory / "result.json").read_text(encoding="utf-8")
        )
        self.assertAlmostEqual(saved["target_gap"], 0.0, places=5)

    def test_unattainable_target_returns_closest_value_and_explicit_gap(self):
        result = submit_inverse_design_request(
            self._request("target", target=100.0),
            project_path=self.project.path,
        )

        self.assertTrue(result.success, result.error_message)
        self.assertTrue(result.feasible)
        self.assertEqual(result.constraint_evaluations, [])
        self.assertAlmostEqual(result.objective_value, 15.5, places=5)
        self.assertAlmostEqual(result.target_gap, 84.5, places=5)
        self.assertAlmostEqual(result.objective_score, result.target_gap, places=8)

    def test_mean_over_ordered_output_range_is_one_scalar_objective(self):
        request = InverseDesignRequest(
            model_book_id=self.book.book_id,
            variable_bounds={"P2": (0.0, 5.0)},
            fixed_inputs={"P3": 2.0, "P4": 3.0},
            objective=InverseDesignObjective(
                None,
                "minimize",
                aggregation="mean",
                output_names=["gain", "loss"],
            ),
            max_iterations=20,
            population_multiplier=5,
            random_seed=42,
        )

        result = submit_inverse_design_request(
            request,
            project_path=self.project.path,
        )

        self.assertTrue(result.success, result.error_message)
        self.assertAlmostEqual(result.best_inputs["P2"], 0.0, places=5)
        self.assertAlmostEqual(result.objective_value, 8.0, places=5)
        self.assertEqual(result.objective["aggregation"], "mean")
        self.assertEqual(result.objective["output_names"], ["gain", "loss"])

    def test_output_constraint_is_enforced_and_reported(self):
        result = submit_inverse_design_request(
            self._request(
                "minimize",
                constraints=[
                    OutputConstraint(
                        "gain",
                        "greater_than_or_equal",
                        value=12.0,
                    )
                ],
            ),
            project_path=self.project.path,
        )

        self.assertTrue(result.success, result.error_message)
        self.assertGreaterEqual(result.objective_value, 12.0 - 1e-5)
        self.assertTrue(result.feasible)
        self.assertTrue(result.constraint_evaluations[0]["satisfied"])

    def test_mean_over_output_range_constraint_is_enforced_and_reported(self):
        result = submit_inverse_design_request(
            self._request(
                "minimize",
                constraints=[
                    OutputConstraint(
                        None,
                        "greater_than_or_equal",
                        value=9.0,
                        aggregation="mean",
                        output_names=["gain", "loss"],
                    )
                ],
            ),
            project_path=self.project.path,
        )

        self.assertTrue(result.success, result.error_message)
        evaluation = result.constraint_evaluations[0]
        self.assertTrue(evaluation["satisfied"])
        self.assertEqual(evaluation["aggregation"], "mean")
        self.assertEqual(evaluation["output_names"], ["gain", "loss"])
        self.assertGreaterEqual(evaluation["predicted_value"], 9.0 - 1e-5)
        saved_request = json.loads(
            (result.artifact_directory / "request.json").read_text(encoding="utf-8")
        )
        self.assertEqual(saved_request["schema_version"], 2)
        self.assertEqual(saved_request["constraints"][0]["aggregation"], "mean")
        self.assertEqual(
            saved_request["constraints"][0]["output_names"],
            ["gain", "loss"],
        )

    def test_infeasible_constraint_returns_clear_failure_without_artifacts(self):
        before = len(
            list((self.project.path / "inverse_design" / "runs").glob("inverse-*"))
        )
        result = submit_inverse_design_request(
            self._request(
                constraints=[
                    OutputConstraint(
                        "gain",
                        "greater_than_or_equal",
                        value=1000.0,
                    )
                ],
                max_iterations=4,
            ),
            project_path=self.project.path,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status, INVERSE_DESIGN_FAILED)
        self.assertIn("satisfying all output constraints", result.error_message)
        self.assertIn("search budget", result.error_message)
        self.assertNotIn("Traceback", result.error_message)
        after = len(
            list((self.project.path / "inverse_design" / "runs").glob("inverse-*"))
        )
        self.assertEqual(after, before)

    def test_exact_model_book_interface_is_required(self):
        missing = InverseDesignRequest(
            model_book_id=self.book.book_id,
            variable_bounds={"P2": (0, 5)},
            fixed_inputs={"P3": 2},
            objective=InverseDesignObjective("gain", "minimize"),
            max_iterations=2,
            population_multiplier=4,
        )
        unknown_output = self._request()
        unknown_output.objective.output_name = "not_saved"
        unknown_output.objective.output_names = ["not_saved"]

        missing_result = submit_inverse_design_request(
            missing,
            project_path=self.project.path,
        )
        output_result = submit_inverse_design_request(
            unknown_output,
            project_path=self.project.path,
        )

        self.assertFalse(missing_result.success)
        self.assertIn("Missing: P4", missing_result.error_message)
        self.assertFalse(output_result.success)
        self.assertIn("not saved", output_result.error_message)

    def test_completed_runs_are_separate_and_loadable(self):
        first = submit_inverse_design_request(
            self._request("minimize"),
            project_path=self.project.path,
        )
        second = submit_inverse_design_request(
            self._request("maximize"),
            project_path=self.project.path,
        )

        self.assertNotEqual(first.run_id, second.run_id)
        self.assertTrue(first.artifact_directory.exists())
        self.assertTrue(second.artifact_directory.exists())
        for result in (first, second):
            self.assertTrue((result.artifact_directory / "request.json").exists())
            self.assertTrue((result.artifact_directory / "result.json").exists())
            self.assertTrue(
                (result.artifact_directory / "best_prediction.csv").exists()
            )
            self.assertTrue(
                (result.artifact_directory / "evaluation_trace.csv").exists()
            )
        loaded_first = load_inverse_design_run(self.project.path, first.run_id)
        latest = load_inverse_design_run(self.project.path)
        self.assertEqual(loaded_first["run_id"], first.run_id)
        self.assertEqual(latest["run_id"], second.run_id)
        self.assertEqual(
            list(latest["predicted_outputs"]),
            ["gain", "loss"],
        )
        manifest = json.loads(
            (self.project.path / "project.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["inverse_design"]["latest_run_id"], second.run_id)
        self.assertGreaterEqual(manifest["inverse_design"]["run_count"], 2)

        history = load_inverse_design_runs(
            self.project.path,
            model_book_id=self.book.book_id,
        )
        self.assertEqual(
            [payload["run_id"] for payload in history.runs][-2:],
            [first.run_id, second.run_id],
        )
        self.assertEqual(history.errors, [])

    def test_history_skips_a_corrupt_inverse_run(self):
        first = submit_inverse_design_request(
            self._request("minimize"),
            project_path=self.project.path,
        )
        second = submit_inverse_design_request(
            self._request("maximize"),
            project_path=self.project.path,
        )
        (second.artifact_directory / "result.json").write_text(
            "{invalid json", encoding="utf-8"
        )

        history = load_inverse_design_runs(
            self.project.path,
            model_book_id=self.book.book_id,
        )

        restored_ids = [payload["run_id"] for payload in history.runs]
        self.assertIn(first.run_id, restored_ids)
        self.assertNotIn(second.run_id, restored_ids)
        self.assertEqual(len(history.errors), 1)
        self.assertIn(second.run_id, history.errors[0])

    def test_history_filter_excludes_runs_from_another_model_book(self):
        first_book = self.book
        first = submit_inverse_design_request(
            self._request("minimize"),
            project_path=self.project.path,
        )
        second_book = save_model_book(
            self.project.path,
            first_book.source_run_id,
            "Second Inverse Evaluator",
        )
        set_active_model_book(self.project.path, second_book.book_id)
        second_request = self._request("maximize")
        second_request.model_book_id = second_book.book_id
        second = submit_inverse_design_request(
            second_request,
            project_path=self.project.path,
        )

        first_history = load_inverse_design_runs(
            self.project.path,
            model_book_id=first_book.book_id,
        )
        second_history = load_inverse_design_runs(
            self.project.path,
            model_book_id=second_book.book_id,
        )

        self.assertIn(first.run_id, [item["run_id"] for item in first_history.runs])
        self.assertNotIn(second.run_id, [item["run_id"] for item in first_history.runs])
        self.assertEqual(
            [item["run_id"] for item in second_history.runs],
            [second.run_id],
        )
        set_active_model_book(self.project.path, first_book.book_id)


if __name__ == "__main__":
    unittest.main()
