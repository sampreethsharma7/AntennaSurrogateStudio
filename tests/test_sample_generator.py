import csv
import math
import tempfile
import unittest
from pathlib import Path

from studio.parser_engine import (
    IMPORTED_OUTPUT_LABEL,
    discover_input_output_files,
    prepare,
)
from studio.sample_generator import (
    MAX_LHS_SAMPLES,
    MAX_RANDOM_SEED,
    LHSSampleGenerationRequest,
    LHSVariable,
    generate_lhs_samples,
    write_lhs_inputs_csv,
)


class LHSSampleGeneratorTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        self.root = Path(self.temp_dir.name)
        self.request = LHSSampleGenerationRequest(
            variables=[
                LHSVariable("patch_length", 20, 40),
                LHSVariable("patch_width", 15, 30),
                LHSVariable("feed_offset", 1, 8),
            ],
            sample_count=12,
            random_seed=42,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_generation_is_deterministic_for_the_same_seed(self):
        first = generate_lhs_samples(self.request)
        second = generate_lhs_samples(self.request)

        self.assertEqual(first.variable_names, second.variable_names)
        self.assertEqual(first.rows, second.rows)

        changed_seed = generate_lhs_samples(
            LHSSampleGenerationRequest(
                variables=self.request.variables,
                sample_count=12,
                random_seed=43,
            )
        )
        self.assertNotEqual(first.rows, changed_seed.rows)

    def test_values_stay_in_bounds_and_cover_every_lhs_stratum(self):
        generated = generate_lhs_samples(self.request)
        for column, variable in enumerate(self.request.variables):
            values = [row[column] for row in generated.rows]
            self.assertTrue(
                all(variable.minimum <= value <= variable.maximum for value in values)
            )
            normalized_strata = {
                min(
                    self.request.sample_count - 1,
                    math.floor(
                        (value - variable.minimum)
                        / (variable.maximum - variable.minimum)
                        * self.request.sample_count
                    ),
                )
                for value in values
            }
            self.assertEqual(
                normalized_strata,
                set(range(self.request.sample_count)),
            )

    def test_sample_count_and_column_order_follow_the_csv_contract(self):
        generated = generate_lhs_samples(self.request)

        self.assertEqual(generated.sample_count, 12)
        self.assertEqual(
            generated.variable_names,
            ["patch_length", "patch_width", "feed_offset"],
        )
        self.assertTrue(all(len(row) == 3 for row in generated.rows))

    def test_variable_validation_rejects_invalid_names_and_limits(self):
        invalid_cases = (
            (("", 0, 1), "non-empty name"),
            (("sample_id", 0, 1), "reserved"),
            (("Sample ID", 0, 1), "reserved"),
            (("line\nbreak", 0, 1), "line breaks"),
            (("x", True, 1), "numeric value"),
            (("x", 0, float("inf")), "finite"),
            (("x", 2, 2), "less than"),
            (("x", 3, 2), "less than"),
        )
        for values, message in invalid_cases:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    LHSVariable(*values)

    def test_request_rejects_empty_and_duplicate_variables(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            LHSSampleGenerationRequest([], 10, 42)
        with self.assertRaisesRegex(ValueError, "Duplicate.*width"):
            LHSSampleGenerationRequest(
                [LHSVariable("Width", 0, 1), LHSVariable("width", 1, 2)],
                10,
                42,
            )

    def test_request_rejects_invalid_sample_counts(self):
        for value in (0, -1, True, 1.5, MAX_LHS_SAMPLES + 1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Sample count"):
                    LHSSampleGenerationRequest(
                        [LHSVariable("x", 0, 1)],
                        value,
                        42,
                    )

    def test_request_rejects_invalid_seeds(self):
        for value in (-1, True, 1.5, MAX_RANDOM_SEED + 1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Random seed"):
                    LHSSampleGenerationRequest(
                        [LHSVariable("x", 0, 1)],
                        10,
                        value,
                    )

    def test_csv_preserves_header_order_and_contains_only_numeric_rows(self):
        generated = generate_lhs_samples(self.request)
        destination = write_lhs_inputs_csv(self.root / "inputs.csv", generated)

        with destination.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(
            rows[0],
            ["patch_length", "patch_width", "feed_offset"],
        )
        self.assertNotIn("sample_id", rows[0])
        self.assertEqual(len(rows), 13)
        for row in rows[1:]:
            self.assertEqual(len(row), 3)
            for value in row:
                self.assertTrue(math.isfinite(float(value)))

    def test_generated_csv_is_compatible_with_data_prep_pair_workflow(self):
        generated = generate_lhs_samples(self.request)
        input_path = write_lhs_inputs_csv(self.root / "inputs.csv", generated)
        output_path = self.root / "outputs.csv"
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["response_1", "response_2"])
            for values in generated.rows:
                writer.writerow([values[0] + values[1], values[2] * 2])

        discovery = discover_input_output_files(input_path, output_path)
        self.assertEqual(discovery.sample_count, 12)
        self.assertEqual(discovery.input_variables, generated.variable_names)
        self.assertEqual(discovery.output_variables, ["response_1", "response_2"])

        prepared_inputs = self.root / "prepared" / "inputs.csv"
        prepared_outputs = self.root / "prepared" / "outputs.csv"
        prepared = prepare(
            "pair",
            input_path,
            generated.variable_names,
            IMPORTED_OUTPUT_LABEL,
            prepared_inputs,
            prepared_outputs,
            source_output_path=output_path,
        )
        self.assertEqual(prepared.rows, 12)
        self.assertEqual(prepared.input_columns, 3)
        self.assertEqual(prepared.output_columns, 2)
        self.assertIsNone(prepared.sample_id_column)
        with prepared_inputs.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(
            rows[0],
            ["patch_length", "patch_width", "feed_offset"],
        )
        self.assertEqual(len(rows[1]), 3)

    def test_export_requires_csv_extension(self):
        with self.assertRaisesRegex(ValueError, r"\.csv"):
            write_lhs_inputs_csv(
                self.root / "inputs.txt",
                generate_lhs_samples(self.request),
            )


if __name__ == "__main__":
    unittest.main()
