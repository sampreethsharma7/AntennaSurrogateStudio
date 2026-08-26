import unittest
from pathlib import Path

from studio.parser_engine import TrainingRequest


class TrainingRequestTests(unittest.TestCase):
    def test_valid_request(self):
        request = TrainingRequest(
            input_csv_path=Path("inputs.csv"),
            output_csv_path=Path("outputs.csv"),
            feature_columns=[
                "patch_length",
                "patch_width",
                "substrate_height",
            ],
            target_columns=[
                "resonant_frequency",
                "radiation_efficiency",
            ],
            sample_id_column="sample_id",
        )

        self.assertEqual(request.input_csv_path, Path("inputs.csv"))
        self.assertEqual(request.output_csv_path, Path("outputs.csv"))
        self.assertEqual(
            request.feature_columns,
            ["patch_length", "patch_width", "substrate_height"],
        )
        self.assertEqual(
            request.target_columns,
            ["resonant_frequency", "radiation_efficiency"],
        )
        self.assertEqual(request.sample_id_column, "sample_id")

    def test_missing_input_csv_path(self):
        with self.assertRaisesRegex(
            ValueError,
            "A training input CSV path is required",
        ):
            TrainingRequest(
                input_csv_path=None,  # type: ignore[arg-type]
                output_csv_path=Path("outputs.csv"),
                feature_columns=["patch_length"],
                target_columns=["resonant_frequency"],
            )

    def test_missing_output_csv_path(self):
        with self.assertRaisesRegex(
            ValueError,
            "A training output CSV path is required",
        ):
            TrainingRequest(
                input_csv_path=Path("inputs.csv"),
                output_csv_path=None,  # type: ignore[arg-type]
                feature_columns=["patch_length"],
                target_columns=["resonant_frequency"],
            )

    def test_empty_feature_list(self):
        with self.assertRaisesRegex(
            ValueError,
            "At least one feature column must be selected",
        ):
            TrainingRequest(
                input_csv_path=Path("inputs.csv"),
                output_csv_path=Path("outputs.csv"),
                feature_columns=[],
                target_columns=["resonant_frequency"],
            )

    def test_duplicate_feature_names(self):
        with self.assertRaisesRegex(
            ValueError,
            "Duplicate feature columns were provided: patch_length",
        ):
            TrainingRequest(
                input_csv_path=Path("inputs.csv"),
                output_csv_path=Path("outputs.csv"),
                feature_columns=["patch_length", "patch_width", "patch_length"],
                target_columns=["resonant_frequency"],
            )

    def test_empty_feature_name(self):
        with self.assertRaisesRegex(
            ValueError,
            "Feature column names cannot be empty",
        ):
            TrainingRequest(
                input_csv_path=Path("inputs.csv"),
                output_csv_path=Path("outputs.csv"),
                feature_columns=["patch_length", "   "],
                target_columns=["resonant_frequency"],
            )

    def test_empty_target_list(self):
        with self.assertRaisesRegex(
            ValueError,
            "At least one target column must be selected",
        ):
            TrainingRequest(
                input_csv_path=Path("inputs.csv"),
                output_csv_path=Path("outputs.csv"),
                feature_columns=["patch_length"],
                target_columns=[],
            )

    def test_empty_target_name(self):
        with self.assertRaisesRegex(
            ValueError,
            "Target column names cannot be empty",
        ):
            TrainingRequest(
                input_csv_path=Path("inputs.csv"),
                output_csv_path=Path("outputs.csv"),
                feature_columns=["patch_length"],
                target_columns=["resonant_frequency", "   "],
            )

    def test_duplicate_target_names(self):
        with self.assertRaisesRegex(
            ValueError,
            "Duplicate target columns were provided: resonant_frequency",
        ):
            TrainingRequest(
                input_csv_path=Path("inputs.csv"),
                output_csv_path=Path("outputs.csv"),
                feature_columns=["patch_length"],
                target_columns=[
                    "resonant_frequency",
                    "radiation_efficiency",
                    "resonant_frequency",
                ],
            )

    def test_target_included_in_features(self):
        with self.assertRaisesRegex(
            ValueError,
            "Target columns cannot also be used as input features: resonant_frequency",
        ):
            TrainingRequest(
                input_csv_path=Path("inputs.csv"),
                output_csv_path=Path("outputs.csv"),
                feature_columns=["patch_length", "resonant_frequency"],
                target_columns=["resonant_frequency", "radiation_efficiency"],
            )

    def test_sample_id_included_in_features(self):
        with self.assertRaisesRegex(
            ValueError,
            "The sample ID column cannot also be used as an input feature",
        ):
            TrainingRequest(
                input_csv_path=Path("inputs.csv"),
                output_csv_path=Path("outputs.csv"),
                feature_columns=["sample_id", "patch_length"],
                target_columns=["resonant_frequency"],
                sample_id_column="sample_id",
            )

    def test_sample_id_equal_to_target(self):
        with self.assertRaisesRegex(
            ValueError,
            "The sample ID column cannot also be used as an output target",
        ):
            TrainingRequest(
                input_csv_path=Path("inputs.csv"),
                output_csv_path=Path("outputs.csv"),
                feature_columns=["patch_length"],
                target_columns=["resonant_frequency", "radiation_efficiency"],
                sample_id_column="resonant_frequency",
            )


if __name__ == "__main__":
    unittest.main()
