import tempfile
import unittest
from pathlib import Path

from studio.dataset_validation import (
    DatasetValidationError,
    validate_dataset,
)
from studio.parser_engine import TrainingRequest


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        self.root = Path(self.temp_dir.name)
        self.input_path = self.root / "inputs.csv"
        self.output_path = self.root / "outputs.csv"

    def tearDown(self):
        self.temp_dir.cleanup()

    def request(self, **changes) -> TrainingRequest:
        values = {
            "input_csv_path": self.input_path,
            "output_csv_path": self.output_path,
            "feature_columns": ["patch_length", "patch_width"],
            "target_columns": ["resonant_frequency", "efficiency"],
            "sample_id_column": None,
        }
        values.update(changes)
        return TrainingRequest(**values)

    def write_valid_pair(self, *, with_sample_ids: bool = False) -> None:
        if with_sample_ids:
            self.input_path.write_text(
                "sample_id,patch_length,patch_width,notes\n"
                "Design_001,10,12,baseline\n"
                "Design_002,11,13,candidate\n",
                encoding="utf-8",
            )
            self.output_path.write_text(
                "sample_id,resonant_frequency,efficiency\n"
                "Design_001,2.4,0.82\n"
                "Design_002,2.5,0.85\n",
                encoding="utf-8",
            )
            return
        self.input_path.write_text(
            "patch_length,patch_width,notes\n"
            "10,12,baseline\n"
            "11,13,candidate\n",
            encoding="utf-8",
        )
        self.output_path.write_text(
            "resonant_frequency,efficiency\n"
            "2.4,0.82\n"
            "2.5,0.85\n",
            encoding="utf-8",
        )

    def test_valid_multi_output_dataset(self):
        self.write_valid_pair()

        result = validate_dataset(self.request())

        self.assertEqual(result.input_csv_path, self.input_path)
        self.assertEqual(result.output_csv_path, self.output_path)
        self.assertEqual(result.sample_count, 2)
        self.assertEqual(result.feature_count, 2)
        self.assertEqual(result.target_count, 2)
        self.assertEqual(
            result.target_columns,
            ["resonant_frequency", "efficiency"],
        )
        self.assertEqual(result.to_dict()["input_csv_path"], str(self.input_path))

    def test_valid_dataset_with_aligned_sample_ids(self):
        self.write_valid_pair(with_sample_ids=True)

        result = validate_dataset(
            self.request(sample_id_column="sample_id")
        )

        self.assertEqual(result.sample_id_column, "sample_id")
        self.assertEqual(result.sample_count, 2)

    def test_missing_input_csv(self):
        self.output_path.write_text(
            "resonant_frequency,efficiency\n2.4,0.82\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            DatasetValidationError,
            "training input CSV does not exist",
        ):
            validate_dataset(self.request())

    def test_missing_output_csv(self):
        self.input_path.write_text(
            "patch_length,patch_width\n10,12\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            DatasetValidationError,
            "training output CSV does not exist",
        ):
            validate_dataset(self.request())

    def test_missing_feature_column(self):
        self.write_valid_pair()

        with self.assertRaisesRegex(
            DatasetValidationError,
            "Selected input feature columns were not found: substrate_height",
        ):
            validate_dataset(
                self.request(
                    feature_columns=["patch_length", "substrate_height"]
                )
            )

    def test_missing_target_column(self):
        self.write_valid_pair()

        with self.assertRaisesRegex(
            DatasetValidationError,
            "Selected output target columns were not found: gain",
        ):
            validate_dataset(
                self.request(
                    target_columns=["resonant_frequency", "gain"]
                )
            )

    def test_mismatched_sample_counts(self):
        self.write_valid_pair()
        self.output_path.write_text(
            "resonant_frequency,efficiency\n2.4,0.82\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            DatasetValidationError,
            "same number of sample rows",
        ):
            validate_dataset(self.request())

    def test_non_numeric_feature_value(self):
        self.write_valid_pair()
        self.input_path.write_text(
            "patch_length,patch_width\n10,not-a-number\n",
            encoding="utf-8",
        )
        self.output_path.write_text(
            "resonant_frequency,efficiency\n2.4,0.82\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            DatasetValidationError,
            "non-numeric value in feature column 'patch_width'",
        ):
            validate_dataset(self.request())

    def test_non_finite_target_value(self):
        self.write_valid_pair()
        self.output_path.write_text(
            "resonant_frequency,efficiency\n2.4,nan\n2.5,0.85\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            DatasetValidationError,
            "non-finite value in target column 'efficiency'",
        ):
            validate_dataset(self.request())

    def test_sample_id_must_exist_in_both_files(self):
        self.write_valid_pair(with_sample_ids=True)
        self.output_path.write_text(
            "resonant_frequency,efficiency\n2.4,0.82\n2.5,0.85\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            DatasetValidationError,
            "must be present in both",
        ):
            validate_dataset(self.request(sample_id_column="sample_id"))

    def test_duplicate_sample_ids_are_rejected(self):
        self.write_valid_pair(with_sample_ids=True)
        self.input_path.write_text(
            "sample_id,patch_length,patch_width\n"
            "Design_001,10,12\n"
            "Design_001,11,13\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            DatasetValidationError,
            "input CSV contains duplicate 'sample_id' values",
        ):
            validate_dataset(self.request(sample_id_column="sample_id"))

    def test_misaligned_sample_ids_are_rejected(self):
        self.write_valid_pair(with_sample_ids=True)
        self.output_path.write_text(
            "sample_id,resonant_frequency,efficiency\n"
            "Design_001,2.4,0.82\n"
            "Design_003,2.5,0.85\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(DatasetValidationError, "Sample ID mismatch"):
            validate_dataset(self.request(sample_id_column="sample_id"))


if __name__ == "__main__":
    unittest.main()
