import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from studio.dataset_registry import register_dataset
from studio.dataset_validation import validate_dataset
from studio.model_book import (
    MODEL_BOOK_VERSION,
    ModelBookError,
    list_model_books,
    load_model_book,
    save_model_book,
    set_active_model_book,
)
from studio.model_training import (
    ModelTrainingRequest,
    submit_model_training_request,
)
from studio.parser_engine import TrainingRequest
from studio.project_store import ProjectStore, atomic_write_json


class ModelBookTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        self.store = ProjectStore(Path(self.temp_dir.name) / "library")
        self.project = self.store.create_project("Model Book Test")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _completed_run(self):
        input_path = self.project.path / "data" / "prepared" / "inputs.csv"
        output_path = self.project.path / "data" / "prepared" / "outputs.csv"
        with input_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Sample ID", "patch_length", "patch_width"])
            for index in range(1, 21):
                writer.writerow(
                    [
                        f"Design_{index:03d}",
                        float(index),
                        float((index * 3) % 7),
                    ]
                )
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Sample ID", "resonant_frequency"])
            for index in range(1, 21):
                width = float((index * 3) % 7)
                writer.writerow(
                    [f"Design_{index:03d}", 2.0 * index - 0.5 * width + 3.0]
                )

        validation = validate_dataset(
            TrainingRequest(
                input_csv_path=input_path,
                output_csv_path=output_path,
                feature_columns=["patch_length", "patch_width"],
                target_columns=["resonant_frequency"],
                sample_id_column="Sample ID",
            )
        )
        register_dataset(self.project.path, validation)
        return submit_model_training_request(
            ModelTrainingRequest(
                model_name="linear_regression",
                training_mode="auto",
                search_level="medium",
            ),
            project_path=self.project.path,
        )

    @staticmethod
    def _directory_hashes(directory: Path) -> dict[str, str]:
        return {
            path.relative_to(directory).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        }

    def test_completed_run_can_be_saved_and_loaded_as_model_book(self):
        run = self._completed_run()

        saved = save_model_book(
            self.project.path,
            run.run_id,
            "Patch Frequency Surrogate",
        )
        loaded = load_model_book(self.project.path, saved.book_id)
        loaded_by_name = load_model_book(self.project.path, saved.name)

        self.assertEqual(saved.book_id, "book-0001")
        self.assertEqual(saved.version, MODEL_BOOK_VERSION)
        self.assertEqual(loaded.to_dict(), saved.to_dict())
        self.assertEqual(loaded_by_name.book_id, saved.book_id)
        self.assertTrue(saved.model_artifact_path.is_file())
        self.assertTrue(saved.manifest_path.is_file())
        self.assertGreater(saved.model_artifact_path.stat().st_size, 0)

    def test_required_model_training_dataset_and_performance_metadata_are_preserved(self):
        run = self._completed_run()

        book = save_model_book(self.project.path, run.run_id, "Traceable Model")
        manifest = json.loads(book.manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(book.model_name, "linear_regression")
        self.assertEqual(
            book.model_type,
            "sklearn.linear_model.LinearRegression",
        )
        self.assertEqual(book.feature_columns, ["patch_length", "patch_width"])
        self.assertEqual(book.target_columns, ["resonant_frequency"])
        self.assertEqual(book.sample_id_column, "Sample ID")
        self.assertEqual(book.training_mode, "auto")
        self.assertEqual(book.search_level, "medium")
        self.assertEqual(
            set(book.parameters_used),
            {"fit_intercept", "positive"},
        )
        self.assertEqual(book.source_run_id, run.run_id)
        self.assertEqual(book.dataset_id, run.dataset_id)
        self.assertEqual(set(book.test_metrics), {"MAE", "RMSE", "R²"})
        self.assertIn("RMSE", book.validation_metrics)
        self.assertTrue(book.dataset_fingerprint)
        self.assertTrue(book.created_at)
        self.assertEqual(manifest["model_book_version"], MODEL_BOOK_VERSION)
        self.assertEqual(manifest["training"]["split"]["test_size"], 0.20)
        self.assertEqual(manifest["training"]["split"]["random_state"], 42)
        self.assertEqual(
            manifest["interface"]["output_axis"],
            {
                "label": "Frequency",
                "unit": None,
                "values": [1.0],
                "source": "output_index",
            },
        )
        self.assertEqual(book.output_axis.label, "Frequency")
        self.assertEqual(book.output_axis.values, (1.0,))
        project_manifest = json.loads(
            (self.project.path / "project.json").read_text(encoding="utf-8")
        )
        self.assertEqual(project_manifest["workflow"]["stage"], "model_saved")
        self.assertEqual(project_manifest["workflow"]["completed_steps"], 5)
        self.assertEqual(project_manifest["workflow"]["total_steps"], 5)
        self.assertIn("Open Model Library", project_manifest["workflow"]["next_action"])
        self.assertIsNone(project_manifest["model_library"]["active_book_id"])

    def test_legacy_model_book_without_axis_metadata_derives_a_safe_axis(self):
        run = self._completed_run()
        book = save_model_book(self.project.path, run.run_id, "Legacy Axis")
        manifest = json.loads(book.manifest_path.read_text(encoding="utf-8"))
        manifest["interface"].pop("output_axis")
        atomic_write_json(book.manifest_path, manifest)

        loaded = load_model_book(self.project.path, book.book_id)

        self.assertEqual(loaded.output_axis.label, "Frequency")
        self.assertEqual(loaded.output_axis.values, (1.0,))
        self.assertEqual(loaded.output_axis.source, "output_index")

    def test_source_training_run_remains_unchanged(self):
        run = self._completed_run()
        before = self._directory_hashes(run.run_directory)

        book = save_model_book(self.project.path, run.run_id, "Immutable Source")

        self.assertEqual(self._directory_hashes(run.run_directory), before)
        self.assertNotEqual(book.directory, run.run_directory)
        self.assertEqual(
            book.model_artifact_path.read_bytes(),
            run.model_artifact_path.read_bytes(),
        )

    def test_different_model_books_remain_separate(self):
        run = self._completed_run()

        first = save_model_book(self.project.path, run.run_id, "Candidate A")
        second = save_model_book(self.project.path, run.run_id, "Candidate B")
        books = list_model_books(self.project.path)

        self.assertEqual(first.book_id, "book-0001")
        self.assertEqual(second.book_id, "book-0002")
        self.assertNotEqual(first.directory, second.directory)
        self.assertEqual([book.name for book in books], ["Candidate A", "Candidate B"])
        project_manifest = json.loads(
            (self.project.path / "project.json").read_text(encoding="utf-8")
        )
        self.assertEqual(project_manifest["model_library"]["book_count"], 2)
        self.assertIsNone(project_manifest["model_library"]["active_book_id"])

    def test_saving_another_book_preserves_the_explicit_active_selection(self):
        run = self._completed_run()
        first = save_model_book(self.project.path, run.run_id, "Active Candidate")
        set_active_model_book(self.project.path, first.book_id)

        save_model_book(self.project.path, run.run_id, "New Candidate")
        project_manifest = json.loads(
            (self.project.path / "project.json").read_text(encoding="utf-8")
        )

        self.assertEqual(
            project_manifest["model_library"]["active_book_id"],
            first.book_id,
        )
        self.assertIn(
            "Run a prediction",
            project_manifest["workflow"]["next_action"],
        )

    def test_duplicate_names_are_rejected_without_overwriting(self):
        run = self._completed_run()
        first = save_model_book(self.project.path, run.run_id, "Array Surrogate")
        original_manifest = first.manifest_path.read_bytes()

        with self.assertRaisesRegex(ModelBookError, "already exists"):
            save_model_book(self.project.path, run.run_id, "array surrogate")

        self.assertEqual(first.manifest_path.read_bytes(), original_manifest)
        self.assertEqual(len(list_model_books(self.project.path)), 1)

    def test_incomplete_and_failed_runs_cannot_be_promoted(self):
        with self.assertRaisesRegex(ModelBookError, "No completed training run"):
            save_model_book(self.project.path, "run-0001", "Incomplete")

        project_manifest_path = self.project.path / "project.json"
        project_manifest = json.loads(project_manifest_path.read_text(encoding="utf-8"))
        project_manifest["model_training"] = {
            "runs": [
                {
                    "run_number": 1,
                    "run_id": "run-0001",
                    "status": "TRAINING_FAILED",
                }
            ]
        }
        atomic_write_json(project_manifest_path, project_manifest)

        with self.assertRaisesRegex(ModelBookError, "successfully completed"):
            save_model_book(self.project.path, "run-0001", "Failed")
        self.assertFalse((self.project.path / "books" / "book-0001").exists())

    def test_missing_required_artifact_prevents_promotion(self):
        run = self._completed_run()
        run.model_artifact_path.unlink()

        with self.assertRaisesRegex(ModelBookError, "trained model artifact is missing"):
            save_model_book(self.project.path, run.run_id, "Missing Model")
        self.assertFalse((self.project.path / "books" / "book-0001").exists())

    def test_missing_required_metadata_prevents_promotion(self):
        run = self._completed_run()
        run.training_config_artifact_path.unlink()

        with self.assertRaisesRegex(ModelBookError, "training_config.json artifact is missing"):
            save_model_book(self.project.path, run.run_id, "Missing Metadata")
        self.assertFalse((self.project.path / "books" / "book-0001").exists())


if __name__ == "__main__":
    unittest.main()
