import csv
import tempfile
import unittest
from pathlib import Path

from studio.output_axis import infer_output_axis
from studio.parser_engine import (
    IMPORTED_OUTPUT_LABEL,
    LEGACY_GENERIC_TEMPLATE_INSTRUCTIONS,
    ParseError,
    discover,
    discover_filename_format,
    discover_input_output_files,
    discover_parameter_format,
    prepare,
    prepare_from_filename_format,
    prepare_from_parameter_format,
    write_input_output_templates,
)


FAR_FIELD_HEADER = (
    '# "Theta" "Phi" "Abs(Grlz)" "Abs(Theta)" '
    '"Phase(Theta)" "Abs(Phi)" "Phase(Phi)" "Ax.Ratio"\n'
)


class ParserEngineTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_filename_sweep_discovery_and_prepare(self):
        for name, phase, gain in (
            ("FF_A1_phi0__A1_phi90.txt", 90, 4.0),
            ("FF_A1_phi0__A1_phi-90.txt", -90, 2.0),
        ):
            (self.root / name).write_text(
                FAR_FIELD_HEADER
                + f"0 89.9999999 {gain} 0 0 0 0 0\n"
                + f"1 89.9999999 {gain + 1} 0 0 0 0 0\n",
                encoding="utf-8",
            )

        discovery = discover_filename_format(self.root)
        self.assertEqual(discovery.sample_count, 2)
        self.assertEqual(discovery.input_variables, ["A1", "P1", "A2", "P2"])
        self.assertIn("Abs(Grlz)", discovery.output_variables)

        input_destination = self.root / "out" / "inputs.csv"
        output_destination = self.root / "out" / "outputs.csv"
        result = prepare_from_filename_format(
            self.root,
            ["P1", "P2"],
            "Abs(Grlz)",
            input_destination,
            output_destination,
            phi_filter=90,
        )
        self.assertEqual(result.rows, 2)
        self.assertEqual(result.theta_points, 2)
        self.assertEqual(result.input_columns, 2)
        self.assertEqual(result.output_columns, 2)
        self.assertEqual(result.target_columns, ["theta_0", "theta_1"])
        with input_destination.open(newline="", encoding="utf-8") as handle:
            input_rows = list(csv.reader(handle))
        with output_destination.open(newline="", encoding="utf-8") as handle:
            output_rows = list(csv.reader(handle))
        self.assertEqual(input_rows[0], ["P1", "P2"])
        self.assertEqual(output_rows[0], ["theta_0", "theta_1"])
        self.assertEqual(len(input_rows), 3)
        self.assertEqual(len(output_rows), 3)
        self.assertEqual(len(input_rows), len(output_rows))

    def test_input_output_pair_discovery_and_prepare(self):
        source_inputs = self.root / "source_inputs.csv"
        source_outputs = self.root / "source_outputs.csv"
        source_inputs.write_text(
            "Sample ID,P1,P2,Er\n"
            "Design_001,0,10,2.2\n"
            "Design_002,20,30,3.1\n",
            encoding="utf-8",
        )
        source_outputs.write_text(
            "Sample ID,theta_-90,theta_0,theta_90\n"
            "Design_001,-10,8,-11\n"
            "Design_002,-12,9,-10\n",
            encoding="utf-8",
        )

        discovery = discover(
            "pair",
            source_inputs,
            output_path=source_outputs,
        )
        self.assertEqual(discovery.mode, "pair")
        self.assertEqual(discovery.sample_count, 2)
        self.assertEqual(discovery.input_variables, ["P1", "P2", "Er"])
        self.assertEqual(
            discovery.output_variables,
            ["theta_-90", "theta_0", "theta_90"],
        )

        destination_inputs = self.root / "prepared" / "inputs.csv"
        destination_outputs = self.root / "prepared" / "outputs.csv"
        result = prepare(
            "pair",
            source_inputs,
            ["P2", "Er"],
            IMPORTED_OUTPUT_LABEL,
            destination_inputs,
            destination_outputs,
            source_output_path=source_outputs,
        )
        self.assertEqual(result.output, IMPORTED_OUTPUT_LABEL)
        self.assertEqual(result.rows, 2)
        self.assertEqual(result.input_columns, 2)
        self.assertEqual(result.output_columns, 3)
        self.assertEqual(
            result.target_columns,
            ["theta_-90", "theta_0", "theta_90"],
        )
        self.assertEqual(result.sample_id_column, "Sample ID")
        with destination_inputs.open(newline="", encoding="utf-8") as handle:
            prepared_inputs = list(csv.reader(handle))
        with destination_outputs.open(newline="", encoding="utf-8") as handle:
            prepared_outputs = list(csv.reader(handle))
        self.assertEqual(prepared_inputs[0], ["Sample ID", "P2", "Er"])
        self.assertEqual(prepared_inputs[1], ["Design_001", "10.0", "2.2"])
        self.assertEqual(
            prepared_outputs[0],
            ["Sample ID", "theta_-90", "theta_0", "theta_90"],
        )
        self.assertEqual(prepared_outputs[1][0], "Design_001")
        self.assertEqual(prepared_inputs[2][0], "Design_002")
        self.assertEqual(prepared_outputs[2][0], "Design_002")
        self.assertEqual(len(prepared_inputs), len(prepared_outputs))

    def test_input_output_pair_rejects_mismatched_sample_ids(self):
        source_inputs = self.root / "source_inputs.csv"
        source_outputs = self.root / "source_outputs.csv"
        source_inputs.write_text(
            "Sample ID,Input Parameter 1\n"
            "Design_001,0\n"
            "Design_002,1\n",
            encoding="utf-8",
        )
        source_outputs.write_text(
            "Sample ID,Output 1\n"
            "Design_001,10\n"
            "Design_003,11\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ParseError, "Sample ID mismatch"):
            discover_input_output_files(source_inputs, source_outputs)

    def test_sample_id_must_be_present_in_both_files_and_unique(self):
        source_inputs = self.root / "source_inputs.csv"
        source_outputs = self.root / "source_outputs.csv"
        source_inputs.write_text(
            "Sample ID,Input Parameter 1\nDesign_001,0\nDesign_002,1\n",
            encoding="utf-8",
        )
        source_outputs.write_text(
            "Output 1\n10\n11\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ParseError, "in both CSVs"):
            discover_input_output_files(source_inputs, source_outputs)

        source_inputs.write_text(
            "Sample ID,Input Parameter 1\nDesign_001,0\nDesign_001,1\n",
            encoding="utf-8",
        )
        source_outputs.write_text(
            "Sample ID,Output 1\nDesign_001,10\nDesign_001,11\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ParseError, "duplicate Sample ID"):
            discover_input_output_files(source_inputs, source_outputs)

    def test_input_output_pair_requires_matching_rows(self):
        source_inputs = self.root / "source_inputs.csv"
        source_outputs = self.root / "source_outputs.csv"
        source_inputs.write_text("P1\n0\n1\n", encoding="utf-8")
        source_outputs.write_text("theta_0\n8\n", encoding="utf-8")

        with self.assertRaisesRegex(ParseError, "same number of sample rows"):
            discover_input_output_files(source_inputs, source_outputs)

    def test_input_output_templates_are_created_without_overwriting(self):
        template_folder = self.root / "templates"
        input_template, output_template, instructions = (
            write_input_output_templates(template_folder)
        )
        self.assertTrue(input_template.exists())
        self.assertTrue(output_template.exists())
        self.assertTrue(instructions.exists())
        discovery = discover_input_output_files(input_template, output_template)
        self.assertEqual(discovery.sample_count, 3)
        self.assertEqual(
            discovery.input_variables,
            [
                "Input Parameter 1",
                "Input Parameter 2",
                "Input Parameter 3",
            ],
        )
        self.assertEqual(
            discovery.output_variables,
            ["Output 1", "Output 2", "Output 3"],
        )
        with input_template.open(newline="", encoding="utf-8") as handle:
            input_rows = list(csv.reader(handle))
        with output_template.open(newline="", encoding="utf-8") as handle:
            output_rows = list(csv.reader(handle))
        self.assertEqual(input_rows[0][0], "Sample ID")
        self.assertEqual(output_rows[0][0], "Sample ID")
        self.assertEqual(input_rows[1][0], "Design_001")
        self.assertEqual(output_rows[1][0], "Design_001")
        guide = instructions.read_text(encoding="utf-8")
        self.assertIn("Keep Sample ID as the first column", guide)
        self.assertIn("S11 at 1 GHz", guide)
        self.assertIn("Gain at theta -90 deg", guide)
        self.assertIn("loads the pair automatically", guide)
        self.assertNotIn("Analyze pair", guide)

        input_template.write_text("Custom\n42\n", encoding="utf-8")
        output_template.write_text("My response\n-3\n", encoding="utf-8")
        instructions.write_text("My instructions\n", encoding="utf-8")
        write_input_output_templates(template_folder)
        self.assertEqual(
            input_template.read_text(encoding="utf-8"),
            "Custom\n42\n",
        )
        self.assertEqual(
            output_template.read_text(encoding="utf-8"),
            "My response\n-3\n",
        )
        self.assertEqual(
            instructions.read_text(encoding="utf-8"),
            "My instructions\n",
        )

    def test_untouched_legacy_templates_upgrade_to_sample_id_contract(self):
        template_folder = self.root / "legacy-templates"
        template_folder.mkdir()
        (template_folder / "inputs_template.csv").write_text(
            "Variable 1,Variable 2\n"
            "0,0\n"
            "0.5,1\n"
            "1,0.5\n",
            encoding="utf-8",
        )
        (template_folder / "outputs_template.csv").write_text(
            "Response at coordinate min,Response at coordinate midpoint,"
            "Response at coordinate max\n"
            "-12.5,7.8,-12.1\n"
            "-10.2,8.4,-13\n"
            "-14.1,6.9,-9.8\n",
            encoding="utf-8",
        )
        (template_folder / "README.txt").write_text(
            LEGACY_GENERIC_TEMPLATE_INSTRUCTIONS,
            encoding="utf-8",
        )

        input_template, output_template, instructions = (
            write_input_output_templates(template_folder)
        )
        discovery = discover_input_output_files(input_template, output_template)

        self.assertEqual(
            discovery.input_variables,
            [
                "Input Parameter 1",
                "Input Parameter 2",
                "Input Parameter 3",
            ],
        )
        self.assertEqual(
            discovery.output_variables[0],
            "Output 1",
        )
        self.assertIn(
            "Keep Sample ID as the first column",
            instructions.read_text(encoding="utf-8"),
        )

    def test_parameter_sweep_discovery_and_prepare(self):
        source = self.root / "parameter_export.txt"
        source.write_text(
            "#Parameters = {P1=0; P2=90}\n"
            '# "Theta" "Abs(Grlz)" "Phase(Theta)"\n'
            "0 3.0 10\n"
            "1 4.0 11\n"
            "#Parameters = {P1=45; P2=-45}\n"
            '# "Theta" "Abs(Grlz)" "Phase(Theta)"\n'
            "0 5.0 12\n"
            "1 6.0 13\n",
            encoding="utf-8",
        )

        discovery = discover_parameter_format(source)
        self.assertEqual(discovery.sample_count, 2)
        self.assertEqual(discovery.input_variables, ["P1", "P2"])
        self.assertEqual(discovery.output_variables, ["Abs(Grlz)", "Phase(Theta)"])

        input_destination = self.root / "inputs.csv"
        output_destination = self.root / "outputs.csv"
        result = prepare_from_parameter_format(
            source,
            ["P1", "P2"],
            "Abs(Grlz)",
            input_destination,
            output_destination,
        )
        self.assertEqual(result.rows, 2)
        self.assertEqual(result.columns, 4)
        self.assertEqual(result.input_csv, str(input_destination))
        self.assertEqual(result.output_csv, str(output_destination))
        self.assertEqual(
            result.target_columns,
            ["Abs(Grlz) at Theta 0", "Abs(Grlz) at Theta 1"],
        )
        with input_destination.open(newline="", encoding="utf-8") as handle:
            input_rows = list(csv.reader(handle))
        with output_destination.open(newline="", encoding="utf-8") as handle:
            output_rows = list(csv.reader(handle))
        self.assertEqual(input_rows[0], ["P1", "P2"])
        self.assertEqual(
            output_rows[0],
            ["Abs(Grlz) at Theta 0", "Abs(Grlz) at Theta 1"],
        )
        self.assertEqual(input_rows[1], ["0.0", "90.0"])
        self.assertEqual(output_rows[1], ["3.0", "4.0"])
        self.assertEqual(len(input_rows), len(output_rows))

    def test_parameter_sweep_preserves_frequency_axis_and_response_name(self):
        source = self.root / "cst_frequency_export.txt"
        source.write_text(
            "#Parameters = {patch_length=20; patch_width=15}\n"
            '# "Frequency / GHz" "S11" "Efficiency"\n'
            "1.0 -10.0 0.80\n"
            "2.5 -18.5 0.85\n"
            "#Parameters = {patch_length=22; patch_width=16}\n"
            '# "Frequency / GHz" "S11" "Efficiency"\n'
            "1.0 -12.0 0.81\n"
            "2.5 -21.0 0.86\n",
            encoding="utf-8",
        )

        discovery = discover_parameter_format(source)
        self.assertEqual(discovery.input_variables, ["patch_length", "patch_width"])
        self.assertEqual(discovery.output_variables, ["S11", "Efficiency"])

        input_destination = self.root / "frequency_inputs.csv"
        output_destination = self.root / "frequency_outputs.csv"
        result = prepare_from_parameter_format(
            source,
            ["patch_length", "patch_width"],
            "S11",
            input_destination,
            output_destination,
        )

        expected_targets = [
            "S11 at Frequency 1 GHz",
            "S11 at Frequency 2.5 GHz",
        ]
        self.assertEqual(result.target_columns, expected_targets)
        with output_destination.open(newline="", encoding="utf-8") as handle:
            output_rows = list(csv.reader(handle))
        self.assertEqual(output_rows[0], expected_targets)
        self.assertEqual(output_rows[1], ["-10.0", "-18.5"])

        axis = infer_output_axis(result.target_columns)
        self.assertEqual(axis.label, "Frequency")
        self.assertEqual(axis.unit, "GHz")
        self.assertEqual(axis.values, (1.0, 2.5))

    def test_parameter_sweep_reports_actual_coordinate_grid_mismatch(self):
        source = self.root / "mismatched_frequency_export.txt"
        source.write_text(
            "#Parameters = {P1=1}\n"
            '# "Frequency [GHz]" "S11"\n'
            "1 -10\n2 -12\n"
            "#Parameters = {P1=2}\n"
            '# "Frequency [GHz]" "S11"\n'
            "1 -11\n3 -13\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ParseError, "Frequency grid mismatch"):
            prepare_from_parameter_format(
                source,
                ["P1"],
                "S11",
                self.root / "bad_inputs.csv",
                self.root / "bad_outputs.csv",
            )

    def test_parameter_sweep_preserves_a_generic_solver_coordinate(self):
        source = self.root / "generic_coordinate_export.txt"
        source.write_text(
            "#Parameters = {P1=1}\n"
            '# "Sweep coordinate" "Response"\n'
            "10 2.0\n20 3.0\n"
            "#Parameters = {P1=2}\n"
            '# "Sweep coordinate" "Response"\n'
            "10 4.0\n20 5.0\n",
            encoding="utf-8",
        )

        result = prepare_from_parameter_format(
            source,
            ["P1"],
            "Response",
            self.root / "generic_inputs.csv",
            self.root / "generic_outputs.csv",
        )

        self.assertEqual(
            result.target_columns,
            [
                "Response at Sweep coordinate 10",
                "Response at Sweep coordinate 20",
            ],
        )
        axis = infer_output_axis(result.target_columns)
        self.assertEqual(axis.label, "Sweep coordinate")
        self.assertIsNone(axis.unit)
        self.assertEqual(axis.values, (10.0, 20.0))

    def test_inconsistent_filename_element_count_is_rejected(self):
        (self.root / "A1_phi0.txt").write_text(FAR_FIELD_HEADER, encoding="utf-8")
        (self.root / "A1_phi0__A1_phi90.txt").write_text(
            FAR_FIELD_HEADER, encoding="utf-8"
        )
        with self.assertRaises(ParseError):
            discover_filename_format(self.root)


if __name__ == "__main__":
    unittest.main()
