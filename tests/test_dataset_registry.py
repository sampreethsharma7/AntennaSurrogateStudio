import json
import tempfile
import unittest
from pathlib import Path

from studio.dataset_registry import (
    DatasetRegistrationError,
    get_registered_dataset,
    list_registered_datasets,
    register_dataset,
)
from studio.dataset_validation import DatasetValidationError, validate_dataset
from studio.parser_engine import TrainingRequest
from studio.project_store import ProjectStore


class DatasetRegistryTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        self.store = ProjectStore(Path(self.temp_dir.name) / "library")
        self.project = self.store.create_project("Registry Test")
        self.inputs = self.project.path / "data" / "prepared" / "inputs.csv"
        self.outputs = self.project.path / "data" / "prepared" / "outputs.csv"
        self.inputs.write_text(
            "length,width\n10,12\n11,13\n",
            encoding="utf-8",
        )
        self.outputs.write_text(
            "frequency,efficiency\n2.4,0.82\n2.5,0.85\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def validation(self):
        return validate_dataset(
            TrainingRequest(
                input_csv_path=self.inputs,
                output_csv_path=self.outputs,
                feature_columns=["length", "width"],
                target_columns=["frequency", "efficiency"],
            )
        )

    def test_registers_project_local_immutable_snapshot(self):
        registered = register_dataset(
            self.project.path,
            self.validation(),
            name="Baseline antenna sweep",
        )

        self.assertTrue(registered.dataset_id.startswith("dataset-"))
        self.assertEqual(registered.name, "Baseline antenna sweep")
        self.assertEqual(registered.sample_count, 2)
        self.assertEqual(registered.feature_count, 2)
        self.assertEqual(registered.target_count, 2)
        self.assertTrue(registered.input_csv_path.is_file())
        self.assertTrue(registered.output_csv_path.is_file())
        self.assertTrue(registered.manifest_path.is_file())
        self.assertEqual(
            registered.input_csv_path.parent,
            self.project.path / "data" / "registered" / registered.dataset_id,
        )

        original_snapshot = registered.input_csv_path.read_text(encoding="utf-8")
        self.inputs.write_text("length,width\n99,100\n", encoding="utf-8")
        self.assertEqual(
            registered.input_csv_path.read_text(encoding="utf-8"),
            original_snapshot,
        )

        manifest = json.loads(
            (self.project.path / "project.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["dataset_registry"]["dataset_count"], 1)
        self.assertEqual(
            manifest["dataset_registry"]["active_dataset_id"],
            registered.dataset_id,
        )

    def test_identical_registration_is_idempotent(self):
        validation = self.validation()
        first = register_dataset(self.project.path, validation, name="First")
        second = register_dataset(self.project.path, validation, name="Second")

        self.assertEqual(first.dataset_id, second.dataset_id)
        self.assertEqual(second.name, "First")
        registered = list_registered_datasets(self.project.path)
        self.assertEqual(len(registered), 1)
        self.assertEqual(registered[0].dataset_id, first.dataset_id)

    def test_changed_data_creates_new_registration(self):
        first = register_dataset(self.project.path, self.validation())
        self.inputs.write_text(
            "length,width\n10,12\n11,14\n",
            encoding="utf-8",
        )
        second = register_dataset(self.project.path, self.validation())

        self.assertNotEqual(first.dataset_id, second.dataset_id)
        self.assertEqual(len(list_registered_datasets(self.project.path)), 2)
        loaded = get_registered_dataset(self.project.path, second.dataset_id)
        self.assertEqual(loaded.input_sha256, second.input_sha256)

        index = json.loads(
            (self.project.path / "data" / "registered" / "index.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(index["active_dataset_id"], second.dataset_id)

    def test_registration_revalidates_changed_source(self):
        validation = self.validation()
        self.inputs.write_text(
            "length,width\n10,not-numeric\n11,13\n",
            encoding="utf-8",
        )

        with self.assertRaises(DatasetValidationError):
            register_dataset(self.project.path, validation)
        self.assertEqual(list_registered_datasets(self.project.path), [])

    def test_rejects_registration_outside_a_project(self):
        outside = Path(self.temp_dir.name) / "not-a-project"
        outside.mkdir()

        with self.assertRaisesRegex(
            DatasetRegistrationError,
            "only be registered inside",
        ):
            register_dataset(outside, self.validation())

    def test_registered_snapshot_integrity_is_checked(self):
        registered = register_dataset(self.project.path, self.validation())
        registered.input_csv_path.write_text(
            "length,width\n999,999\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            DatasetRegistrationError,
            "failed its integrity check",
        ):
            get_registered_dataset(self.project.path, registered.dataset_id)


if __name__ == "__main__":
    unittest.main()
