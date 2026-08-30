import csv
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from studio.dataset_registry import register_dataset
from studio.dataset_validation import validate_dataset
from studio.inference import (
    INFERENCE_COMPLETED,
    INFERENCE_FAILED,
    InferenceRequest,
    load_inference_runs,
    submit_inference_request,
)
from studio.model_book import save_model_book, set_active_model_book
from studio.model_training import (
    ModelTrainingRequest,
    submit_model_training_request,
)
from studio.parser_engine import TrainingRequest
from studio.project_store import ProjectStore, atomic_write_json


class BasicInferenceBackendTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        self.store = ProjectStore(Path(self.temp_dir.name) / "library")
        self.project = self.store.create_project("Inference Backend")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _completed_book(
        self,
        *,
        name: str = "Inference Model",
        target_columns: tuple[str, ...] = ("gain",),
    ):
        input_path = self.project.path / "data" / "prepared" / "inputs.csv"
        output_path = self.project.path / "data" / "prepared" / "outputs.csv"
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
            writer.writerow(["Sample ID", *target_columns])
            for index in range(1, 31):
                p2 = float(index)
                p3 = float((index * 2) % 7)
                p4 = float((index * 3) % 5)
                candidates = (
                    2.0 * p2 + 3.0 * p3 - 0.5 * p4 + 1.0,
                    -1.0 * p2 + 0.25 * p3 + 4.0 * p4 - 2.0,
                    0.5 * p2 - 2.0 * p3 + 1.5 * p4 + 8.0,
                )
                writer.writerow(
                    [
                        f"Design_{index:03d}",
                        *candidates[: len(target_columns)],
                    ]
                )

        validation = validate_dataset(
            TrainingRequest(
                input_csv_path=input_path,
                output_csv_path=output_path,
                feature_columns=["P2", "P3", "P4"],
                target_columns=list(target_columns),
                sample_id_column="Sample ID",
            )
        )
        register_dataset(self.project.path, validation)
        run = submit_model_training_request(
            ModelTrainingRequest(
                model_name="linear_regression",
                training_mode="auto",
                search_level="medium",
            ),
            project_path=self.project.path,
        )
        self.assertTrue(run.success, run.error_message)
        book = save_model_book(self.project.path, run.run_id, name)
        set_active_model_book(self.project.path, book.book_id)
        return book

    def _infer(self, inputs, *, model_book_id=None):
        return submit_inference_request(
            InferenceRequest(inputs=inputs, model_book_id=model_book_id),
            project_path=self.project.path,
        )

    def _tree_hashes(self, root: Path):
        return {
            path.relative_to(root).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    def test_active_book_predicts_one_sample(self):
        book = self._completed_book()

        result = self._infer(
            {"P2": 4.0, "P3": 2.0, "P4": 3.0},
            model_book_id=book.book_id,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.status, INFERENCE_COMPLETED)
        self.assertEqual(result.model_book_id, book.book_id)
        self.assertEqual(result.model_book_name, book.name)
        self.assertEqual(result.model_name, "linear_regression")
        self.assertEqual(result.target_order, ["gain"])
        self.assertTrue(math.isclose(result.predictions["gain"], 13.5, abs_tol=1e-9))
        self.assertIsNone(result.error_message)

    def test_feature_order_is_preserved_independent_of_mapping_order(self):
        self._completed_book()

        result = self._infer({"P4": 3, "P2": 4, "P3": 2})

        self.assertTrue(result.success)
        self.assertEqual(result.feature_order, ["P2", "P3", "P4"])
        self.assertEqual(list(result.input_values), ["P2", "P3", "P4"])
        self.assertEqual(list(result.input_values.values()), [4.0, 2.0, 3.0])
        self.assertTrue(math.isclose(result.predictions["gain"], 13.5, abs_tol=1e-9))

    def test_inference_preserves_book_and_saves_separate_history_runs(self):
        book = self._completed_book()
        before = self._tree_hashes(book.directory)

        first = self._infer(
            {"P2": 4.0, "P3": 2.0, "P4": 3.0},
            model_book_id=book.book_id,
        )
        second = self._infer(
            {"P2": 7.0, "P3": 1.0, "P4": 2.0},
            model_book_id=book.book_id,
        )

        self.assertTrue(first.success)
        self.assertTrue(second.success)
        self.assertEqual(self._tree_hashes(book.directory), before)
        self.assertEqual(first.run_id, "inference-0001")
        self.assertEqual(second.run_id, "inference-0002")
        self.assertNotEqual(first.artifact_directory, second.artifact_directory)
        for result in (first, second):
            self.assertTrue((result.artifact_directory / "request.json").exists())
            self.assertTrue((result.artifact_directory / "result.json").exists())
            self.assertTrue((result.artifact_directory / "prediction.csv").exists())
        saved_request = json.loads(
            (first.artifact_directory / "request.json").read_text(encoding="utf-8")
        )
        self.assertEqual(saved_request["model_book_id"], book.book_id)
        self.assertEqual(list(saved_request["inputs"]), ["P2", "P3", "P4"])
        history = load_inference_runs(
            self.project.path,
            model_book_id=book.book_id,
        )
        self.assertEqual([result.run_id for result in history.runs], [
            "inference-0001",
            "inference-0002",
        ])
        self.assertEqual(history.errors, [])
        manifest = json.loads(
            (self.project.path / "project.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["inference"]["run_count"], 2)
        self.assertEqual(manifest["inference"]["latest_run_id"], "inference-0002")

    def test_inference_history_skips_one_corrupt_run(self):
        book = self._completed_book()
        first = self._infer({"P2": 4.0, "P3": 2.0, "P4": 3.0})
        second = self._infer({"P2": 7.0, "P3": 1.0, "P4": 2.0})
        (second.artifact_directory / "result.json").write_text(
            "{invalid json", encoding="utf-8"
        )

        history = load_inference_runs(
            self.project.path,
            model_book_id=book.book_id,
        )

        self.assertEqual([result.run_id for result in history.runs], [first.run_id])
        self.assertEqual(len(history.errors), 1)
        self.assertIn(second.run_id, history.errors[0])

    def test_inference_history_is_filtered_by_model_book(self):
        first_book = self._completed_book(name="First History Book")
        first = self._infer({"P2": 4.0, "P3": 2.0, "P4": 3.0})
        second_book = save_model_book(
            self.project.path,
            first_book.source_run_id,
            "Second History Book",
        )
        set_active_model_book(self.project.path, second_book.book_id)
        second = self._infer({"P2": 7.0, "P3": 1.0, "P4": 2.0})

        first_history = load_inference_runs(
            self.project.path,
            model_book_id=first_book.book_id,
        )
        second_history = load_inference_runs(
            self.project.path,
            model_book_id=second_book.book_id,
        )

        self.assertEqual([item.run_id for item in first_history.runs], [first.run_id])
        self.assertEqual([item.run_id for item in second_history.runs], [second.run_id])

    def test_multi_output_linear_regression_prediction(self):
        book = self._completed_book(
            target_columns=("gain_phi_0", "gain_phi_90", "efficiency"),
        )

        result = self._infer(
            {"P2": 4.0, "P3": 2.0, "P4": 3.0},
            model_book_id=book.book_id,
        )

        self.assertTrue(result.success)
        self.assertEqual(
            result.target_order,
            ["gain_phi_0", "gain_phi_90", "efficiency"],
        )
        self.assertEqual(list(result.predictions), result.target_order)
        self.assertTrue(
            math.isclose(result.predictions["gain_phi_0"], 13.5, abs_tol=1e-9)
        )
        self.assertTrue(
            math.isclose(result.predictions["gain_phi_90"], 6.5, abs_tol=1e-9)
        )
        self.assertTrue(
            math.isclose(result.predictions["efficiency"], 10.5, abs_tol=1e-9)
        )

    def test_missing_required_input_fails_without_prediction(self):
        self._completed_book()

        result = self._infer({"P2": 1.0, "P4": 3.0})

        self.assertFalse(result.success)
        self.assertEqual(result.status, INFERENCE_FAILED)
        self.assertIn("Missing required input: P3", result.error_message)
        self.assertEqual(result.predictions, {})

    def test_unexpected_input_fails_without_prediction(self):
        self._completed_book()

        result = self._infer(
            {"P2": 1.0, "P3": 2.0, "P4": 3.0, "P5": 4.0}
        )

        self.assertFalse(result.success)
        self.assertIn("Unexpected input: P5", result.error_message)
        self.assertEqual(result.predictions, {})

    def test_non_numeric_boolean_and_non_finite_inputs_fail_clearly(self):
        self._completed_book()
        invalid_values = (
            ("not-a-number", "must be a numeric value"),
            (True, "must be a numeric value"),
            (float("nan"), "must be a finite numeric value"),
        )

        for value, message in invalid_values:
            with self.subTest(value=value):
                result = self._infer({"P2": 1.0, "P3": value, "P4": 3.0})
                self.assertFalse(result.success)
                self.assertIn(message, result.error_message)
                self.assertNotIn("Traceback", result.error_message)

    def test_inference_requires_an_active_model_book(self):
        result = self._infer({"P2": 1.0, "P3": 2.0, "P4": 3.0})

        self.assertFalse(result.success)
        self.assertIn("No active Model Book is selected", result.error_message)

    def test_request_id_must_match_the_active_book(self):
        first = self._completed_book(name="First Model")
        second = save_model_book(
            self.project.path,
            first.source_run_id,
            "Second Model",
        )
        self.assertNotEqual(first.book_id, second.book_id)
        set_active_model_book(self.project.path, second.book_id)

        result = self._infer(
            {"P2": 1.0, "P3": 2.0, "P4": 3.0},
            model_book_id=first.book_id,
        )

        self.assertFalse(result.success)
        self.assertIn("is not the active Model Book", result.error_message)

    def test_missing_model_artifact_fails_clearly(self):
        book = self._completed_book()
        book.model_artifact_path.unlink()

        result = self._infer(
            {"P2": 1.0, "P3": 2.0, "P4": 3.0},
            model_book_id=book.book_id,
        )

        self.assertFalse(result.success)
        self.assertIn("trained model artifact is missing", result.error_message)
        self.assertNotIn("Traceback", result.error_message)

    def test_corrupt_model_book_manifest_fails_clearly(self):
        book = self._completed_book()
        book.manifest_path.write_text("{not valid json", encoding="utf-8")

        result = self._infer(
            {"P2": 1.0, "P3": 2.0, "P4": 3.0},
            model_book_id=book.book_id,
        )

        self.assertFalse(result.success)
        self.assertIn("malformed or unreadable", result.error_message)
        self.assertNotIn("Traceback", result.error_message)

    def test_model_integrity_failure_stops_inference(self):
        book = self._completed_book()
        with book.model_artifact_path.open("ab") as handle:
            handle.write(b"tampered")

        result = self._infer(
            {"P2": 1.0, "P3": 2.0, "P4": 3.0},
            model_book_id=book.book_id,
        )

        self.assertFalse(result.success)
        self.assertIn("failed its integrity check", result.error_message)
        self.assertEqual(result.predictions, {})

    def test_corrupt_serialized_model_with_matching_hash_fails_safely(self):
        book = self._completed_book()
        book.model_artifact_path.write_bytes(b"not a joblib model")
        manifest = json.loads(book.manifest_path.read_text(encoding="utf-8"))
        manifest["model"]["artifact"]["sha256"] = hashlib.sha256(
            book.model_artifact_path.read_bytes()
        ).hexdigest()
        atomic_write_json(book.manifest_path, manifest)

        result = self._infer(
            {"P2": 1.0, "P3": 2.0, "P4": 3.0},
            model_book_id=book.book_id,
        )

        self.assertFalse(result.success)
        self.assertIn("model artifact could not be loaded", result.error_message)
        self.assertNotIn("Traceback", result.error_message)

    def test_explicit_active_selection_is_honored(self):
        first = self._completed_book(name="First Model")
        second = save_model_book(
            self.project.path,
            first.source_run_id,
            "Second Model",
        )
        set_active_model_book(self.project.path, first.book_id)

        result = self._infer(
            {"P2": 4.0, "P3": 2.0, "P4": 3.0},
            model_book_id=first.book_id,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.model_book_id, first.book_id)
        self.assertNotEqual(result.model_book_id, second.book_id)


class InferenceRequestContractTests(unittest.TestCase):
    def test_request_copies_inputs(self):
        values = {"P2": 1.0}
        request = InferenceRequest(inputs=values)
        values["P2"] = 8.0

        self.assertEqual(request.inputs, {"P2": 1.0})
        self.assertIsNone(request.model_book_id)

    def test_request_rejects_invalid_mapping_and_names(self):
        with self.assertRaisesRegex(ValueError, "dictionary"):
            InferenceRequest(inputs=[("P2", 1.0)])  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            InferenceRequest(inputs={"": 1.0})
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            InferenceRequest(inputs={"P2": 1.0}, model_book_id="  ")


if __name__ == "__main__":
    unittest.main()
