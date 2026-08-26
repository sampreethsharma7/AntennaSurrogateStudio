import os
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

import customtkinter as ctk

from studio.parser_engine import DiscoveryResult
from studio.project_store import ProjectStore
from studio.ui import StudioApp


GUI_MAY_BE_AVAILABLE = (
    os.name == "nt"
    or os.sys.platform == "darwin"
    or bool(os.environ.get("DISPLAY"))
)


@unittest.skipUnless(GUI_MAY_BE_AVAILABLE, "A desktop display is required.")
class DataPrepPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        cls.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        cls.store = ProjectStore(Path(cls.temp_dir.name) / "library")
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
        self.project = self.store.create_project(self.id().split(".")[-1])
        self.app.set_project(self.project, target_page="data")
        self.page = self.app.data_page
        self.page.mode_var.set("parameters")
        self.page.mode_control.set("#Parameters sweep")
        self.page._mode_changed("#Parameters sweep")
        self.page._analysis_complete(
            DiscoveryResult(
                mode="parameters",
                files=["raw.txt"],
                input_variables=["P1", "P2", "P3"],
                output_variables=["Gain", "Phase"],
                sample_count=12,
            )
        )
        self.app.update_idletasks()

    def tearDown(self):
        dialog = self.page.sample_generator_dialog
        if dialog is not None and dialog.winfo_exists():
            dialog.destroy()
        self.page.sample_generator_dialog = None

    def test_raw_variable_contract_requires_explicit_save(self):
        self.assertEqual(self.page.active_subtask, "variables")
        self.assertEqual(self.page.prepare_button.cget("state"), "disabled")
        self.assertEqual(
            self.page.confirm_variables_button.cget("state"),
            "normal",
        )
        self.assertEqual(
            self.page.confirm_variables_button.cget("text"),
            "Save selection  →",
        )

        self.page.input_checks["P1"].set(True)
        self.page.input_checks["P3"].set(True)
        self.page.output_var.set("Phase")
        self.page.confirm_variable_selection()

        self.assertEqual(self.page.active_subtask, "prepare")
        self.assertEqual(self.page.prepare_button.cget("state"), "normal")
        self.assertEqual(
            self.page.confirm_variables_button.cget("text"),
            "Selection saved  ✓",
        )
        prep = self.app.current_project.manifest["data_prep"]
        self.assertEqual(prep["selected_inputs"], ["P1", "P3"])
        self.assertEqual(prep["selected_output"], "Phase")
        self.assertTrue(prep["variable_contract_confirmed"])

    def test_raw_response_copy_is_coordinate_neutral(self):
        help_text = self.page.output_help_label.cget("text")
        note_text = self.page.output_note_label.cget("text")

        self.assertIn("coordinate grid", help_text)
        self.assertIn("source coordinate grid", note_text)
        self.assertNotIn("theta", f"{help_text} {note_text}".lower())

    def test_changing_a_saved_selection_requires_saving_again(self):
        self.page.input_checks["P1"].set(True)
        self.page.confirm_variable_selection()

        self.page._expand_subtask("variables")
        self.page.input_checks["P2"].set(True)
        self.page._variable_selection_changed()

        self.assertEqual(self.page.active_subtask, "variables")
        self.assertEqual(self.page.prepare_button.cget("state"), "disabled")
        self.assertEqual(
            self.page.confirm_variables_button.cget("state"),
            "normal",
        )
        self.assertIn("Selection changed", self.page.variable_action_note.cget("text"))

    def test_saved_contract_is_restored_after_project_reopen(self):
        self.page.input_checks["P2"].set(True)
        self.page.output_var.set("Phase")
        self.page.confirm_variable_selection()
        reopened = self.store.open_project(self.project.path, touch=False)

        self.app.set_project(reopened, target_page="data")
        self.app.update_idletasks()

        self.assertEqual(self.page.active_subtask, "prepare")
        self.assertTrue(self.page.input_checks["P2"].get())
        self.assertEqual(self.page.output_var.get(), "Phase")
        self.assertEqual(self.page.prepare_button.cget("state"), "normal")
        self.assertEqual(
            self.page.confirm_variables_button.cget("text"),
            "Selection saved  ✓",
        )

    def test_lhs_generator_is_available_from_the_source_subtask(self):
        self.assertEqual(
            self.page.sample_generator_button.cget("text"),
            "LHS sample generator",
        )

        self.page.open_lhs_sample_generator()
        self.app.update_idletasks()
        dialog = self.page.sample_generator_dialog

        self.assertIsNotNone(dialog)
        self.assertTrue(dialog.winfo_exists())
        self.assertEqual(len(dialog.variable_editors), 3)
        self.assertEqual(dialog.export_button.cget("state"), "disabled")
        self.assertEqual(dialog.sample_count_var.get(), "100")
        self.assertEqual(dialog.seed_var.get(), "42")

    def test_lhs_variable_rows_add_remove_and_page_without_scrolling(self):
        self.page.open_lhs_sample_generator()
        self.app.update_idletasks()
        dialog = self.page.sample_generator_dialog

        for _index in range(3):
            dialog.add_variable()

        self.assertEqual(len(dialog.variable_editors), 6)
        self.assertEqual(dialog.variable_page, 1)
        self.assertEqual(dialog.variable_page_label.cget("text"), "Variables 6–6 of 6")
        self.assertEqual(dialog.previous_page_button.cget("state"), "normal")
        self.assertFalse(
            any(
                isinstance(widget, ctk.CTkScrollableFrame)
                for widget in dialog.winfo_children()
            )
        )

        dialog.remove_variable(5)

        self.assertEqual(len(dialog.variable_editors), 5)
        self.assertEqual(dialog.variable_page, 0)
        self.assertEqual(dialog.variable_page_label.cget("text"), "Variables 1–5 of 5")

    def test_lhs_generate_preview_and_export_loads_only_the_input_path(self):
        self.page.open_lhs_sample_generator()
        self.app.update_idletasks()
        dialog = self.page.sample_generator_dialog
        for editor, values in zip(
            dialog.variable_editors,
            (
                ("patch_length", "20", "40"),
                ("patch_width", "15", "30"),
                ("feed_offset", "1", "8"),
            ),
            strict=True,
        ):
            editor.name.set(values[0])
            editor.minimum.set(values[1])
            editor.maximum.set(values[2])
        dialog.sample_count_var.set("8")
        dialog.seed_var.set("7")

        dialog.generate_samples()
        self.app.update_idletasks()

        self.assertEqual(dialog.generated_samples.sample_count, 8)
        self.assertEqual(dialog.export_button.cget("state"), "normal")
        self.assertGreater(len(dialog.preview_frame.winfo_children()), 5)
        preview_text = {
            str(widget.cget("text"))
            for widget in dialog.preview_frame.winfo_children()
            if hasattr(widget, "cget")
        }
        self.assertNotIn("sample_id", preview_text)
        self.assertGreater(len(dialog.coverage_canvas.find_all()), 8)
        y_values = [row[1] for row in dialog.generated_samples.rows]
        canvas_text = {
            dialog.coverage_canvas.itemcget(item, "text")
            for item in dialog.coverage_canvas.find_all()
            if dialog.coverage_canvas.type(item) == "text"
        }
        self.assertIn(format(min(y_values), ".4g"), canvas_text)
        self.assertIn(format(max(y_values), ".4g"), canvas_text)

        dialog.seed_var.set("8")
        self.assertIsNone(dialog.generated_samples)
        self.assertEqual(dialog.export_button.cget("state"), "disabled")
        dialog.seed_var.set("7")
        dialog.generate_samples()

        destination = self.project.path / "data" / "generated" / "inputs.csv"
        with (
            patch(
                "studio.sample_generator_ui.filedialog.asksaveasfilename",
                return_value=str(destination),
            ),
            patch("studio.sample_generator_ui.messagebox.showinfo"),
        ):
            dialog.export_samples()

        self.assertTrue(destination.exists())
        self.assertEqual(self.page.mode_var.get(), "pair")
        self.assertEqual(self.page.input_path_var.get(), str(destination.resolve()))
        self.assertEqual(self.page.output_path_var.get(), "")
        self.assertEqual(self.page.active_subtask, "source")
        self.assertEqual(self.page.prepare_button.cget("state"), "disabled")
        self.assertIn("outputs are still required", self.page.status_var.get())
        exported_text = destination.read_text(encoding="utf-8")
        self.assertTrue(exported_text.startswith("patch_length,patch_width,feed_offset"))
        self.assertNotIn("sample_id", exported_text.splitlines()[0].lower())

    def test_lhs_invalid_form_is_reported_without_exportable_output(self):
        self.page.open_lhs_sample_generator()
        self.app.update_idletasks()
        dialog = self.page.sample_generator_dialog

        with patch("studio.sample_generator_ui.messagebox.showerror") as error:
            dialog.generate_samples()
            self.assertIn("Variable name in row 1", dialog.status_var.get())
            for editor in dialog.variable_editors:
                editor.name.set("duplicate")
                editor.minimum.set("2")
                editor.maximum.set("1")
            dialog.generate_samples()

        self.assertIsNone(dialog.generated_samples)
        self.assertEqual(dialog.export_button.cget("state"), "disabled")
        self.assertIn("less than", dialog.status_var.get())
        self.assertEqual(error.call_count, 2)


if __name__ == "__main__":
    unittest.main()
