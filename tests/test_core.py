import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.core.compatibility import assess_project_compatibility
from app.core.data_validator import validate_dataset
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


if __name__ == "__main__":
    unittest.main()
