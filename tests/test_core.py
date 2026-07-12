import platform
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.core.compatibility import assess_project_compatibility
from app.core.cst_automation import cst_automation_available, extract_named_outputs
from app.core.data_validator import validate_dataset
from app.core.lhs_sampling import generate_lhs_samples
from app.core.project_manifest import ProjectManifest
from app.core.schema_manager import infer_output_axis
from app.utils.versioning import next_model_version


class CoreTests(unittest.TestCase):
    def test_manifest_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            manifest = ProjectManifest(project_name="Demo", project_type="Custom surrogate model")
            manifest.selected_input_columns = ["Length"]
            manifest.save(path)
            loaded = ProjectManifest.from_file(path / "project.json")
            self.assertEqual(loaded.project_name, "Demo")
            self.assertEqual(loaded.selected_input_columns, ["Length"])

    def test_validator_findings(self):
        df = pd.DataFrame({"x": [1.0, 1.0, 1.0], "y": [2.0, None, 4.0]})
        findings = validate_dataset(df, ["x"], ["y"])
        text = " ".join(item["message"] for item in findings)
        self.assertIn("missing", text)
        self.assertIn("constant", text)
        self.assertIn("Duplicate", text)

    def test_axis_inference(self):
        freq = infer_output_axis(["S11_2.40GHz", "S11_2.41GHz"])
        angle = infer_output_axis(["Gain_theta_-90deg", "Gain_theta_0deg"])
        self.assertEqual(freq["kind"], "frequency")
        self.assertEqual(angle["kind"], "angle")

    def test_compatibility(self):
        manifest = ProjectManifest(project_name="Demo", project_type="Custom surrogate model")
        status, _ = assess_project_compatibility(manifest)
        self.assertEqual(status, "compatible")

    def test_next_model_version(self):
        self.assertEqual(next_model_version([]), 1)
        self.assertEqual(next_model_version([{"version": 1}, {"version": 3}]), 4)

    def test_lhs_samples_within_bounds_and_stratified(self):
        bounds = {"Length": (20.0, 30.0), "Width": (5.0, 15.0)}
        samples = generate_lhs_samples(bounds, n_samples=10, seed=42)
        self.assertEqual(len(samples), 10)
        for row in samples:
            self.assertTrue(20.0 <= row["Length"] <= 30.0)
            self.assertTrue(5.0 <= row["Width"] <= 15.0)
        lengths = sorted(row["Length"] for row in samples)
        bins_hit = {int((v - 20.0) / 1.0) for v in lengths}
        self.assertEqual(len(bins_hit), 10)

    def test_lhs_samples_reproducible_with_seed(self):
        bounds = {"X": (0.0, 1.0)}
        first = generate_lhs_samples(bounds, n_samples=5, seed=7)
        second = generate_lhs_samples(bounds, n_samples=5, seed=7)
        self.assertEqual(first, second)

    def test_lhs_samples_rejects_invalid_bounds(self):
        with self.assertRaises(ValueError):
            generate_lhs_samples({"X": (5.0, 1.0)}, n_samples=3)
        with self.assertRaises(ValueError):
            generate_lhs_samples({}, n_samples=3)
        with self.assertRaises(ValueError):
            generate_lhs_samples({"X": (0.0, 1.0)}, n_samples=0)

    def test_extract_named_outputs_interpolates_sweep(self):
        x_values = [2.30, 2.35, 2.40, 2.45, 2.50]
        y_values = [-5.0, -8.0, -20.0, -9.0, -6.0]
        axis_metadata = infer_output_axis(["S11_2.40GHz", "S11_2.475GHz"])
        result = extract_named_outputs(x_values, y_values, ["S11_2.40GHz", "S11_2.475GHz"], axis_metadata)
        self.assertAlmostEqual(result["S11_2.40GHz"], -20.0)
        self.assertAlmostEqual(result["S11_2.475GHz"], -7.5)

    def test_extract_named_outputs_handles_empty_sweep(self):
        axis_metadata = infer_output_axis(["S11_2.40GHz"])
        result = extract_named_outputs([], [], ["S11_2.40GHz"], axis_metadata)
        self.assertIsNone(result["S11_2.40GHz"])

    def test_cst_automation_unavailable_without_windows(self):
        if platform.system() != "Windows":
            self.assertFalse(cst_automation_available())


if __name__ == "__main__":
    unittest.main()
