import csv
import os
import sys
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace

from studio.dataset_registry import register_dataset
from studio.dataset_validation import validate_dataset
from studio.model_book import (
    load_model_library,
    save_model_book,
    set_active_model_book,
)
from studio.library_ui import (
    _book_output_summary,
    _compact_input_summary,
    _output_count_summary,
)
from studio.output_axis import OutputAxisMetadata
from studio.model_training import (
    ModelTrainingRequest,
    submit_model_training_request,
)
from studio.parser_engine import TrainingRequest
from studio.project_store import ProjectStore
from studio.ui import StudioApp


def create_completed_run(project):
    input_path = project.path / "data" / "prepared" / "inputs.csv"
    output_path = project.path / "data" / "prepared" / "outputs.csv"
    with input_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Sample ID", "length", "width"])
        for index in range(1, 21):
            writer.writerow(
                [f"Design_{index:03d}", float(index), float(index % 5)]
            )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Sample ID", "gain_peak"])
        for index in range(1, 21):
            writer.writerow(
                [f"Design_{index:03d}", 1.5 * index - float(index % 5)]
            )
    validation = validate_dataset(
        TrainingRequest(
            input_csv_path=input_path,
            output_csv_path=output_path,
            feature_columns=["length", "width"],
            target_columns=["gain_peak"],
            sample_id_column="Sample ID",
        )
    )
    register_dataset(project.path, validation)
    return submit_model_training_request(
        ModelTrainingRequest(
            model_name="linear_regression",
            training_mode="auto",
            search_level="medium",
        ),
        project_path=project.path,
    )


class ModelLibraryContractTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        self.store = ProjectStore(Path(self.temp_dir.name) / "library")
        self.project = self.store.create_project("Library Contract")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_empty_library_loads_without_error(self):
        library = load_model_library(self.project.path)

        self.assertEqual(library.entries, [])
        self.assertIsNone(library.active_book_id)
        self.assertEqual(library.valid_book_count, 0)

    def test_library_loads_saved_books_and_metadata(self):
        run = create_completed_run(self.project)
        first = save_model_book(self.project.path, run.run_id, "Gain Model A")
        second = save_model_book(self.project.path, run.run_id, "Gain Model B")

        library = load_model_library(self.project.path)

        self.assertEqual(
            [entry.name for entry in library.entries],
            ["Gain Model A", "Gain Model B"],
        )
        self.assertEqual(library.valid_book_count, 2)
        self.assertIsNone(library.active_book_id)
        self.assertFalse(library.entries[0].is_active)
        self.assertFalse(library.entries[1].is_active)
        self.assertEqual(library.entries[0].book.source_run_id, first.source_run_id)

    def test_active_selection_survives_project_reopen(self):
        run = create_completed_run(self.project)
        first = save_model_book(self.project.path, run.run_id, "First Model")
        save_model_book(self.project.path, run.run_id, "Second Model")

        selected = set_active_model_book(self.project.path, first.book_id)
        reopened = self.store.open_project(self.project.path, touch=False)
        reopened_library = load_model_library(reopened.path)

        self.assertEqual(selected.book_id, first.book_id)
        self.assertEqual(reopened_library.active_book_id, first.book_id)
        self.assertEqual(
            reopened.manifest["model_library"]["active_book_id"],
            first.book_id,
        )
        self.assertTrue(
            next(
                entry
                for entry in reopened_library.entries
                if entry.book_id == first.book_id
            ).is_active
        )

    def test_corrupted_book_is_reported_without_hiding_valid_books(self):
        run = create_completed_run(self.project)
        broken = save_model_book(self.project.path, run.run_id, "Broken Model")
        valid = save_model_book(self.project.path, run.run_id, "Valid Model")
        broken.manifest_path.write_text("{not valid json", encoding="utf-8")

        library = load_model_library(self.project.path)

        self.assertEqual(len(library.entries), 2)
        broken_entry = next(
            entry for entry in library.entries if entry.book_id == broken.book_id
        )
        valid_entry = next(
            entry for entry in library.entries if entry.book_id == valid.book_id
        )
        self.assertFalse(broken_entry.is_valid)
        self.assertIn("malformed or unreadable", broken_entry.error_message)
        self.assertTrue(valid_entry.is_valid)
        self.assertEqual(library.valid_book_count, 1)

    def test_missing_model_artifact_is_reported_as_invalid(self):
        run = create_completed_run(self.project)
        book = save_model_book(self.project.path, run.run_id, "Missing Artifact")
        book.model_artifact_path.unlink()

        library = load_model_library(self.project.path)

        self.assertEqual(len(library.entries), 1)
        self.assertFalse(library.entries[0].is_valid)
        self.assertIn("artifact is missing", library.entries[0].error_message)


class ModelLibraryPresentationTests(unittest.TestCase):
    def test_multi_output_summary_is_neutral_without_range_metadata(self):
        outputs = [f"output_{index}" for index in range(361)]

        self.assertEqual(
            _output_count_summary(outputs),
            "361 output variables",
        )

    def test_many_inputs_are_compact_but_keep_an_explicit_remainder(self):
        inputs = [f"P{index}" for index in range(1, 10)]

        self.assertEqual(
            _compact_input_summary(inputs),
            "P1, P2, P3, P4, P5, P6 · +3 more",
        )

    def test_structured_axis_metadata_adds_a_reliable_engineering_range(self):
        book = SimpleNamespace(
            target_columns=["theta_-90_deg", "theta_0_deg", "theta_90_deg"],
            output_axis=OutputAxisMetadata(
                label="Theta",
                unit="deg",
                values=(-90.0, 0.0, 90.0),
                source="target_columns",
            ),
        )

        self.assertEqual(
            _book_output_summary(book),
            "3 outputs\nTheta -90 to 90 deg",
        )


GUI_MAY_BE_AVAILABLE = (
    os.name == "nt"
    or sys.platform == "darwin"
    or bool(os.environ.get("DISPLAY"))
)


@unittest.skipUnless(GUI_MAY_BE_AVAILABLE, "A desktop display is required.")
class ModelLibraryPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        cls.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        cls.store = ProjectStore(Path(cls.temp_dir.name) / "library")

        cls.project = cls.store.create_project("Library UI")
        run = create_completed_run(cls.project)
        cls.first = save_model_book(cls.project.path, run.run_id, "Gain Model A")
        cls.second = save_model_book(cls.project.path, run.run_id, "Gain Model B")

        cls.empty_project = cls.store.create_project("Empty Library UI")

        cls.invalid_project = cls.store.create_project("Invalid Library UI")
        invalid_run = create_completed_run(cls.invalid_project)
        cls.broken = save_model_book(
            cls.invalid_project.path,
            invalid_run.run_id,
            "Broken Model",
        )
        cls.valid = save_model_book(
            cls.invalid_project.path,
            invalid_run.run_id,
            "Valid Model",
        )
        cls.broken.manifest_path.write_text("{not valid json", encoding="utf-8")

        try:
            cls.app = StudioApp(project_store=cls.store)
        except tk.TclError as exc:
            cls.temp_dir.cleanup()
            raise unittest.SkipTest(
                f"A desktop display is not available: {exc}"
            ) from exc
        cls.app.withdraw()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "app"):
            cls.app.destroy()
        if hasattr(cls, "temp_dir"):
            cls.temp_dir.cleanup()

    def setUp(self):
        set_active_model_book(self.project.path, self.second.book_id)
        project = self.store.open_project(self.project.path, touch=False)
        self.app.set_project(project, target_page="library")
        self.app.update_idletasks()
        self.page = self.app.library_page

    def test_library_page_displays_saved_books_and_active_state(self):
        self.assertEqual(self.app.active_page, "library")
        self.assertEqual(
            self.app.nav_buttons["library"].cget("state"),
            "normal",
        )
        self.assertEqual(len(self.page.library.entries), 2)
        self.assertIn(self.first.book_id, self.page.book_rows)
        self.assertIn(self.second.book_id, self.page.book_rows)
        self.assertEqual(
            self.page.book_rows[self.second.book_id]["status"].cget("text"),
            "ACTIVE",
        )
        row = self.page.book_rows[self.first.book_id]
        self.assertEqual(row["model_type"].cget("text"), "Linear Regression")
        self.assertIn("RMSE", row["metrics"].cget("text"))
        self.assertIn("R²", row["metrics"].cget("text"))
        self.assertEqual(row["interface"].cget("text"), "2 inputs → 1 output")
        self.assertNotIn("open", row)

    def test_saved_model_card_is_selectable_without_an_open_button(self):
        frame = self.page.book_rows[self.first.book_id]["frame"]
        self.assertTrue(frame._canvas.bind("<Button-1>"))
        self.assertNotIn("open", self.page.book_rows[self.first.book_id])

        self.page._open_entry(self.first.book_id)

        self.assertEqual(self.page.selected_book_id, self.first.book_id)
        self.assertEqual(self.page.details_title.cget("text"), "Gain Model A")

    def test_opening_book_displays_stored_metadata(self):
        self.page._open_entry(self.first.book_id)

        self.assertEqual(self.page.selected_entry.book_id, self.first.book_id)
        self.assertEqual(self.page.details_title.cget("text"), "Gain Model A")
        self.assertEqual(
            self.page.summary_values["model_type"].cget("text"),
            "Linear Regression",
        )
        self.assertEqual(
            self.page.summary_values["inputs"].cget("text"),
            "2 inputs",
        )
        self.assertEqual(
            self.page.summary_values["outputs"].cget("text"),
            "1 output",
        )
        self.assertIn("gain_peak", self.page.details_subtitle.cget("text"))
        self.assertEqual(
            self.page.required_inputs_value.cget("text"),
            "length, width",
        )
        self.assertEqual(
            self.page.metric_values["RMSE"].cget("text"),
            f"{self.first.test_metrics['RMSE']:.6g}",
        )
        self.assertEqual(
            self.page.metric_values["MAE"].cget("text"),
            f"{self.first.test_metrics['MAE']:.6g}",
        )
        self.assertEqual(
            self.page.metric_values["R²"].cget("text"),
            f"{self.first.test_metrics['R²']:.6g}",
        )
        self.assertEqual(self.page.details_badge.cget("text"), "Selected Model Book")
        self.assertFalse(self.page.provenance_expanded)
        self.assertEqual(self.page.provenance_body.winfo_manager(), "")

        self.page._toggle_provenance()

        self.assertTrue(self.page.provenance_expanded)
        self.assertEqual(self.page.provenance_body.winfo_manager(), "grid")
        self.assertEqual(
            self.page.provenance_values["source_run"].cget("text"),
            self.first.source_run_id,
        )
        self.assertEqual(
            self.page.provenance_values["dataset"].cget("text"),
            self.first.dataset_fingerprint,
        )

    def test_multi_output_and_many_input_summary_stays_compact(self):
        self.page._open_entry(self.first.book_id)
        book = self.page.selected_entry.book
        book.target_columns = [f"theta_{index}" for index in range(361)]
        book.feature_columns = [f"P{index}" for index in range(1, 10)]

        self.page._render_list()
        self.page._render_details()

        self.assertEqual(
            self.page.summary_values["outputs"].cget("text"),
            "361 output variables",
        )
        self.assertEqual(
            self.page.details_subtitle.cget("text"),
            "Predicts 361 output variables",
        )
        self.assertNotIn("theta_0", self.page.details_subtitle.cget("text"))
        self.assertIn("+3 more", self.page.required_inputs_value.cget("text"))
        self.assertEqual(self.page.view_inputs_button.winfo_manager(), "grid")
        self.assertEqual(
            self.page.book_rows[self.first.book_id]["interface"].cget("text"),
            "9 inputs → 361 outputs",
        )

    def test_selecting_active_book_persists_after_project_reopen(self):
        self.page._open_entry(self.first.book_id)
        self.page._set_active()

        self.assertEqual(self.page.library.active_book_id, self.first.book_id)
        self.assertEqual(self.page.details_badge.cget("text"), "✓ Active Model Book")
        self.assertEqual(self.page.set_active_button.winfo_manager(), "")
        self.assertNotIn("ACTIVE", self.page.library_badge.cget("text"))
        self.assertEqual(self.page.footer_status.cget("text"), "Ready for inference")
        reopened = self.store.open_project(self.project.path, touch=False)
        self.app.set_project(reopened, target_page="library")
        self.app.update_idletasks()
        self.assertEqual(self.app.library_page.selected_book_id, self.first.book_id)
        self.assertEqual(
            self.app.library_page.library.active_book_id,
            self.first.book_id,
        )

    def test_empty_library_has_clear_guidance(self):
        empty = self.store.open_project(self.empty_project.path, touch=False)
        self.app.set_project(empty, target_page="library")
        self.app.update_idletasks()

        self.assertIn("No Model Books are saved", self.app.library_page.empty_label.cget("text"))
        self.assertEqual(self.app.library_page.library.entries, [])

    def test_invalid_book_remains_visible_with_friendly_error(self):
        project = self.store.open_project(self.invalid_project.path, touch=False)
        self.app.set_project(project, target_page="library")
        self.app.update_idletasks()
        page = self.app.library_page

        self.assertEqual(len(page.library.entries), 2)
        self.assertIn(self.broken.book_id, page.book_rows)
        self.assertIn(self.valid.book_id, page.book_rows)
        self.assertEqual(
            page.book_rows[self.broken.book_id]["status"].cget("text"),
            "INVALID",
        )
        page._open_entry(self.broken.book_id)
        error = page.details_error.cget("text")
        self.assertIn("cannot be opened", error)
        self.assertIn("malformed or unreadable", error)
        self.assertNotIn("Traceback", error)
        self.assertEqual(page.set_active_button.cget("state"), "disabled")

    def test_footer_has_no_development_state_message(self):
        texts = []
        pending = [self.page]
        while pending:
            widget = pending.pop()
            pending.extend(widget.winfo_children())
            try:
                value = widget.cget("text")
            except (AttributeError, tk.TclError, ValueError):
                continue
            if value:
                texts.append(str(value))

        self.assertNotIn("Inference is not available yet", texts)


if __name__ == "__main__":
    unittest.main()
