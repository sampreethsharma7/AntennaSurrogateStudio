import json
import tempfile
import unittest
from pathlib import Path

from studio.project_store import ProjectError, ProjectStore
from studio.ui import project_resume_destination


class ProjectStoreTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        self.store = ProjectStore(Path(self.temp_dir.name) / "library")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_project_builds_portable_layout_and_recent_entry(self):
        project = self.store.create_project("8 Element Array", "A project")

        self.assertEqual(project.name, "8 Element Array")
        self.assertTrue((project.path / "project.json").exists())
        self.assertTrue((project.path / "data" / "prepared").is_dir())
        self.assertTrue((project.path / "data" / "registered").is_dir())
        self.assertTrue((project.path / "data" / "templates").is_dir())
        self.assertTrue((project.path / "data" / "generated").is_dir())
        self.assertTrue((project.path / "books").is_dir())
        self.assertTrue((project.path / "inverse_design").is_dir())
        self.assertTrue((project.path / "inverse_design").is_dir())
        self.assertTrue((project.path / "assistant" / "chat_history.json").exists())
        self.assertEqual(project.manifest["ui"]["last_page"], "data")
        self.assertEqual(project.manifest["dataset_registry"]["dataset_count"], 0)
        self.assertEqual(project.manifest["inference"]["run_count"], 0)
        self.assertEqual(project.manifest["inverse_design"]["run_count"], 0)
        self.assertIsNone(project.manifest["inverse_design"]["latest_run_id"])
        self.assertEqual(self.store.recent_projects()[0].path, project.path)

    def test_duplicate_names_get_stable_suffix(self):
        first = self.store.create_project("Array")
        second = self.store.create_project("Array")

        self.assertEqual(first.path.name, "Array")
        self.assertEqual(second.path.name, "Array-2")

    def test_chat_persists_per_project(self):
        project = self.store.create_project("Chat Project")
        self.store.append_chat(project, "user", "What is next?")
        self.store.append_chat(project, "assistant", "Prepare data.")

        history = self.store.load_chat(project)
        self.assertEqual([item["role"] for item in history], ["user", "assistant"])
        self.assertEqual(history[0]["content"], "What is next?")

    def test_welcome_chat_persists_outside_projects(self):
        project = self.store.create_project("Separate Project")
        self.store.append_welcome_chat("user", "How do I begin?")
        self.store.append_welcome_chat("assistant", "Create or open a project.")
        self.store.append_chat(project, "user", "Project question")

        welcome = self.store.load_welcome_chat()
        project_history = self.store.load_chat(project)

        self.assertEqual(len(welcome), 2)
        self.assertEqual(welcome[0]["content"], "How do I begin?")
        self.assertEqual(len(project_history), 1)
        self.assertTrue(self.store.welcome_chat_path.exists())

    def test_welcome_sessions_do_not_mix_between_launch_contexts(self):
        first_session = self.store.welcome_session_id
        first_path = self.store.welcome_chat_path
        self.store.append_welcome_chat("user", "Question from the first session")

        second_session = self.store.start_welcome_session()

        self.assertNotEqual(first_session, second_session)
        self.assertEqual(self.store.load_welcome_chat(), [])
        first_payload = json.loads(first_path.read_text(encoding="utf-8"))
        self.assertEqual(
            first_payload["messages"][0]["content"],
            "Question from the first session",
        )
        self.assertEqual(self.store.welcome_session_count, 2)

    def test_legacy_welcome_history_is_archived_during_migration(self):
        migration_root = Path(self.temp_dir.name) / "migration-library"
        legacy_path = migration_root / "assistant" / "welcome_chat_history.json"
        legacy_path.parent.mkdir(parents=True)
        legacy_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Legacy welcome question",
                            "created_at": "2026-01-01T00:00:00+00:00",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        migrated_store = ProjectStore(migration_root)
        session_index = json.loads(
            migrated_store.welcome_sessions_index_path.read_text(encoding="utf-8")
        )
        imported = next(
            entry
            for entry in session_index["sessions"]
            if entry["label"] == "Imported welcome history"
        )
        imported_path = migrated_store.welcome_sessions_root / imported["path"]
        imported_payload = json.loads(imported_path.read_text(encoding="utf-8"))

        self.assertEqual(migrated_store.load_welcome_chat(), [])
        self.assertEqual(
            imported_payload["messages"][0]["content"],
            "Legacy welcome question",
        )
        self.assertEqual(migrated_store.welcome_session_count, 2)

    def test_open_rejects_arbitrary_folder(self):
        random_folder = Path(self.temp_dir.name) / "not-a-project"
        random_folder.mkdir()
        with self.assertRaises(ProjectError):
            self.store.open_project(random_folder)

    def test_update_deep_merges_workflow(self):
        project = self.store.create_project("Merge")
        updated = self.store.update_project(
            project,
            {"workflow": {"stage": "data_prepared", "completed_steps": 2}},
        )

        self.assertEqual(updated.workflow_stage, "data_prepared")
        self.assertEqual(updated.manifest["workflow"]["total_steps"], 5)

    def test_registered_dataset_has_project_status_label(self):
        project = self.store.create_project("Registered")
        updated = self.store.update_project(
            project,
            {"workflow": {"stage": "dataset_registered"}},
        )

        self.assertEqual(updated.status_label, "Dataset registered")

    def test_trained_model_has_project_status_label(self):
        project = self.store.create_project("Trained")
        updated = self.store.update_project(
            project,
            {"workflow": {"stage": "model_trained"}},
        )

        self.assertEqual(updated.status_label, "Model trained")

    def test_legacy_model_saved_workflow_is_completed_when_project_opens(self):
        project = self.store.create_project("Legacy Saved Workflow")
        manifest = project.manifest
        manifest["workflow"] = {
            "stage": "model_saved",
            "completed_steps": 4,
            "total_steps": 5,
            "next_action": "Load the saved Model Book when inference is available.",
        }
        manifest["model_library"] = {
            "schema_version": 1,
            "book_count": 1,
            "active_book_id": "book-0001",
            "index": "books/index.json",
        }
        (project.path / "project.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        reopened = self.store.open_project(project.path)

        self.assertEqual(reopened.manifest["workflow"]["completed_steps"], 5)
        self.assertEqual(reopened.manifest["workflow"]["total_steps"], 5)
        self.assertIn("Run a prediction", reopened.manifest["workflow"]["next_action"])
        persisted = json.loads(
            (project.path / "project.json").read_text(encoding="utf-8")
        )
        self.assertEqual(persisted["workflow"]["completed_steps"], 5)

    def test_resume_destination_follows_the_completed_workflow_stage(self):
        project = self.store.create_project("Stage Aware Resume")
        expected = {
            "project_created": ("data", "Continue Data Prep"),
            "data_prepared": ("data", "Validate & Register Data"),
            "dataset_registered": ("training", "Continue Model Training"),
            "model_trained": ("results", "Review Training Results"),
            "model_saved": ("library", "Open Model Library"),
        }

        for stage, (page, label) in expected.items():
            project.manifest["workflow"]["stage"] = stage
            destination, action = project_resume_destination(project)
            self.assertEqual(destination, page)
            self.assertIn(label, action)

        project.manifest["workflow"]["stage"] = "model_saved"
        project.manifest["model_library"]["active_book_id"] = "book-0001"
        destination, action = project_resume_destination(project)
        self.assertEqual(destination, "inference")
        self.assertIn("Run Inference", action)

    def test_last_page_is_persisted_without_losing_project_state(self):
        project = self.store.create_project("Continuity")
        updated = self.store.update_project(
            project,
            {"ui": {"last_page": "start"}},
        )
        reopened = self.store.open_project(updated.path, touch=False)

        self.assertEqual(reopened.manifest["ui"]["last_page"], "start")
        self.assertEqual(reopened.workflow_stage, "project_created")

    def test_prepared_data_state_survives_reopen(self):
        project = self.store.create_project("Prepared")
        prepared_inputs = project.path / "data" / "prepared" / "inputs.csv"
        prepared_outputs = project.path / "data" / "prepared" / "outputs.csv"
        prepared_inputs.write_text("P1\n0\n", encoding="utf-8")
        prepared_outputs.write_text("theta_0\n1\n", encoding="utf-8")
        self.store.update_project(
            project,
            {
                "workflow": {"stage": "data_prepared", "completed_steps": 2},
                "data_prep": {
                    "prepared_inputs_csv": "data/prepared/inputs.csv",
                    "prepared_outputs_csv": "data/prepared/outputs.csv",
                    "prepared_rows": 1,
                    "theta_points": 1,
                    "selected_inputs": ["P1"],
                    "selected_output": "Gain",
                },
            },
        )

        reopened = self.store.open_project(project.path, touch=False)
        prep_state = reopened.manifest["data_prep"]
        self.assertEqual(reopened.workflow_stage, "data_prepared")
        self.assertEqual(prep_state["prepared_rows"], 1)
        self.assertTrue(
            (reopened.path / prep_state["prepared_inputs_csv"]).exists()
        )
        self.assertTrue(
            (reopened.path / prep_state["prepared_outputs_csv"]).exists()
        )


if __name__ == "__main__":
    unittest.main()
