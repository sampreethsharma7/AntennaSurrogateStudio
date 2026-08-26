import hashlib
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLIND_GUI = ROOT / "snowbuddy" / "BLIND_GUI_READ.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SnowBuddyContractTests(unittest.TestCase):
    def test_character_and_blind_gui_files_are_packaged(self):
        character = (
            ROOT / "snowbuddy" / "SNOWBUDDY_CHARACTER.md"
        ).read_text(encoding="utf-8")
        blind_gui = BLIND_GUI.read_text(encoding="utf-8")

        self.assertIn("Grounding hierarchy", character)
        self.assertIn("Start page", blind_gui)
        self.assertIn("Data Prep page", blind_gui)
        self.assertIn("Model Training page", blind_gui)
        self.assertIn("Training Results page", blind_gui)
        self.assertIn("Model Library page", blind_gui)
        self.assertIn("Inverse Design page", blind_gui)
        self.assertIn("SnowBuddy local model", blind_gui)
        self.assertIn("Mean over range", character)
        self.assertIn("one scalar predicted", character)
        self.assertIn("draggable vertical divider", blind_gui)
        self.assertIn("known obsolete assistant claims", blind_gui)

    def test_blind_gui_hashes_match_gui_sources(self):
        content = BLIND_GUI.read_text(encoding="utf-8")
        ui_match = re.search(r"UI source SHA-256: ([0-9a-f]{64})", content)
        sample_generator_ui_match = re.search(
            r"Sample Generator UI source SHA-256: ([0-9a-f]{64})",
            content,
        )
        results_ui_match = re.search(
            r"Results UI source SHA-256: ([0-9a-f]{64})", content
        )
        library_ui_match = re.search(
            r"Library UI source SHA-256: ([0-9a-f]{64})", content
        )
        inference_ui_match = re.search(
            r"Inference UI source SHA-256: ([0-9a-f]{64})", content
        )
        inverse_design_ui_match = re.search(
            r"Inverse Design UI source SHA-256: ([0-9a-f]{64})", content
        )
        scientific_plot_match = re.search(
            r"Scientific Plot UI source SHA-256: ([0-9a-f]{64})", content
        )
        theme_match = re.search(r"Theme source SHA-256: ([0-9a-f]{64})", content)

        self.assertIsNotNone(
            ui_match,
            "BLIND_GUI_READ.md must record studio/ui.py SHA-256.",
        )
        self.assertIsNotNone(
            sample_generator_ui_match,
            "BLIND_GUI_READ.md must record studio/sample_generator_ui.py SHA-256.",
        )
        self.assertIsNotNone(
            results_ui_match,
            "BLIND_GUI_READ.md must record studio/results_ui.py SHA-256.",
        )
        self.assertIsNotNone(
            library_ui_match,
            "BLIND_GUI_READ.md must record studio/library_ui.py SHA-256.",
        )
        self.assertIsNotNone(
            inference_ui_match,
            "BLIND_GUI_READ.md must record studio/inference_ui.py SHA-256.",
        )
        self.assertIsNotNone(
            inverse_design_ui_match,
            "BLIND_GUI_READ.md must record studio/inverse_design_ui.py SHA-256.",
        )
        self.assertIsNotNone(
            scientific_plot_match,
            "BLIND_GUI_READ.md must record studio/scientific_plot.py SHA-256.",
        )
        self.assertIsNotNone(
            theme_match,
            "BLIND_GUI_READ.md must record studio/theme.py SHA-256.",
        )
        self.assertEqual(
            ui_match.group(1),
            sha256(ROOT / "studio" / "ui.py"),
            "studio/ui.py changed without updating BLIND_GUI_READ.md.",
        )
        self.assertEqual(
            sample_generator_ui_match.group(1),
            sha256(ROOT / "studio" / "sample_generator_ui.py"),
            "studio/sample_generator_ui.py changed without updating BLIND_GUI_READ.md.",
        )
        self.assertEqual(
            results_ui_match.group(1),
            sha256(ROOT / "studio" / "results_ui.py"),
            "studio/results_ui.py changed without updating BLIND_GUI_READ.md.",
        )
        self.assertEqual(
            library_ui_match.group(1),
            sha256(ROOT / "studio" / "library_ui.py"),
            "studio/library_ui.py changed without updating BLIND_GUI_READ.md.",
        )
        self.assertEqual(
            inference_ui_match.group(1),
            sha256(ROOT / "studio" / "inference_ui.py"),
            "studio/inference_ui.py changed without updating BLIND_GUI_READ.md.",
        )
        self.assertEqual(
            inverse_design_ui_match.group(1),
            sha256(ROOT / "studio" / "inverse_design_ui.py"),
            "studio/inverse_design_ui.py changed without updating BLIND_GUI_READ.md.",
        )
        self.assertEqual(
            scientific_plot_match.group(1),
            sha256(ROOT / "studio" / "scientific_plot.py"),
            "studio/scientific_plot.py changed without updating BLIND_GUI_READ.md.",
        )
        self.assertEqual(
            theme_match.group(1),
            sha256(ROOT / "studio" / "theme.py"),
            "studio/theme.py changed without updating BLIND_GUI_READ.md.",
        )


if __name__ == "__main__":
    unittest.main()
