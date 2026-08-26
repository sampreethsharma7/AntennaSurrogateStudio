import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from studio.assistant import (
    AssistantError,
    OllamaClient,
    SnowBuddyService,
    build_project_context,
    classify_project_question,
    retrieve_guidance,
)
from studio.project_store import ProjectStore


class SnowBuddyWorkflowNavigationTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        self.store = ProjectStore(Path(self.temp_dir.name) / "library")

    def tearDown(self):
        self.temp_dir.cleanup()

    def project(
        self,
        name,
        stage,
        *,
        prep=None,
        training=None,
        library=None,
        last_page="data",
    ):
        project = self.store.create_project(name)
        changes = {
            "workflow": {
                "stage": stage,
                "next_action": "Configure and train the first surrogate-model book.",
            },
            "ui": {"last_page": last_page},
        }
        if prep is not None:
            changes["data_prep"] = prep
        if training is not None:
            changes["model_training"] = training
        if library is not None:
            changes["model_library"] = library
        return self.store.update_project(project, changes)

    def ask_offline(self, project, question, live_ui_state):
        with patch.object(
            OllamaClient,
            "create_response",
            side_effect=AssistantError("offline for navigation test"),
        ):
            return SnowBuddyService(self.store, model="qwen3:1.7b").ask(
                project,
                question,
                live_ui_state=live_ui_state,
            )[0]

    def ask_with_bad_local_reply(self, project, question, live_ui_state, bad_reply):
        with patch.object(OllamaClient, "create_response", return_value=bad_reply):
            return SnowBuddyService(self.store, model="qwen3:8b").ask(
                project,
                question,
                live_ui_state=live_ui_state,
            )

    def test_workflow_knowledge_places_registration_before_training(self):
        workflow = next(
            chunk for chunk in retrieve_guidance("project workflow next steps")
            if chunk.title == "Project workflow"
        ).text
        self.assertLess(workflow.index("validate and register"), workflow.index("train"))
        self.assertIn("inference or inverse design", workflow)

    def test_navigation_questions_are_routed_by_product_area(self):
        cases = {
            "What should I do next from here?": "workflow_next",
            "How do I run a prediction?": "inference",
            "How does inverse design work?": "inverse_design",
            "How do I activate a Model Book?": "model_book",
            "What training options do I have?": "training_setup",
            "Why can't I continue?": "current_blocker",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(classify_project_question(question), expected)

    def test_next_step_follows_every_required_stage_gate(self):
        cases = [
            (
                self.project("new", "project_created"),
                "Visible page: Data Prep",
                ("Data Prep > Source", "Input + output files", "#Parameters sweep"),
            ),
            (
                self.project(
                    "parsed",
                    "data_discovered",
                    prep={
                        "mode": "parameters",
                        "variable_contract_confirmed": False,
                    },
                ),
                "Visible page: Data Prep",
                ("Save selection", "Prepare input + output"),
            ),
            (
                self.project("prepared", "data_prepared", prep={"mode": "pair"}),
                "Visible page: Data Prep",
                ("Validate and register", "Do not start Model Training"),
            ),
            (
                self.project("registered", "dataset_registered"),
                "Visible page: Model Training",
                ("Model Training", "Train Model", "Ensemble AI Engine"),
            ),
            (
                self.project("trained", "model_trained"),
                "Visible page: Training Results",
                ("Training Results", "Create Model Book"),
            ),
            (
                self.project(
                    "saved",
                    "model_saved",
                    library={"book_count": 1, "active_book_id": None},
                ),
                "Visible page: Model Library",
                ("Set as Active", "Inference and Inverse Design"),
            ),
        ]
        for project, live_state, markers in cases:
            with self.subTest(project=project.name):
                reply = self.ask_offline(
                    project,
                    "What should I do next from here?",
                    live_state,
                )
                for marker in markers:
                    self.assertIn(marker, reply)

    def test_bad_local_navigation_is_replaced_at_high_risk_gates(self):
        cases = [
            (
                self.project("new-local", "project_created"),
                "Visible page: Data Prep\nSource loaded or parsed: no",
                "Validate and register your dataset, then train a model.",
                "Data Prep > Source",
            ),
            (
                self.project(
                    "parsed-local",
                    "data_discovered",
                    prep={"mode": "parameters", "variable_contract_confirmed": False},
                ),
                "Visible page: Data Prep\nSave selection action required before Prepare",
                "Select Save selection and then validate the dataset.",
                "Prepare input + output",
            ),
            (
                self.project("prepared-local", "data_prepared"),
                "Visible page: Data Prep\nValidate and register action enabled: yes",
                "Open Model Training now and select Train Model.",
                "Validate and register",
            ),
        ]
        for project, live_state, bad_reply, expected in cases:
            with self.subTest(project=project.name):
                reply, used_local = self.ask_with_bad_local_reply(
                    project,
                    "What should I do next from here?",
                    live_state,
                    bad_reply,
                )
                self.assertFalse(used_local)
                self.assertIn(expected, reply)

    def test_next_step_and_blocker_answers_do_not_wait_for_ollama(self):
        project = self.project("immediate-next", "data_prepared")
        with patch.object(OllamaClient, "create_response") as local_response:
            reply, used_local = SnowBuddyService(
                self.store,
                model="qwen3:8b",
            ).ask(
                project,
                "What should I do next from here?",
                live_ui_state="Visible page: Data Prep",
            )

        local_response.assert_not_called()
        self.assertFalse(used_local)
        self.assertIn("Validate and register", reply)

    def test_active_inference_guidance_uses_current_page_without_reactivation(self):
        project = self.project(
            "active-inference",
            "model_saved",
            library={"book_count": 1, "active_book_id": "book-0001"},
            last_page="inference",
        )
        reply, used_local = self.ask_with_bad_local_reply(
            project,
            "How do I run a prediction and what can I do with the result?",
            "Visible page: Inference\nActive inference Model Book: RF Book",
            (
                "First ensure a Model Book is active. Enter inputs and Predict. "
                "You can also use Adjust & Train Again."
            ),
        )
        self.assertFalse(used_local)
        self.assertIn("every required input", reply)
        self.assertIn("Predict", reply)
        self.assertIn("View Raw Values", reply)
        self.assertIn("ordered curve CSV", reply)
        self.assertNotIn("Set as Active", reply)
        self.assertNotIn("Adjust & Train Again", reply)

    def test_existing_but_inactive_book_is_the_inverse_design_blocker(self):
        project = self.project(
            "inactive-inverse",
            "model_saved",
            library={"book_count": 1, "active_book_id": None},
            last_page="inverse_design",
        )
        reply, used_local = self.ask_with_bad_local_reply(
            project,
            "Why can't I run inverse design?",
            "Visible page: Inverse Design\nInverse Design unavailable: no active Model Book",
            (
                "Complete Model Training and choose Create Model Book before "
                "using Inverse Design."
            ),
        )
        self.assertFalse(used_local)
        self.assertIn("no Model Book is active", reply)
        self.assertIn("Model Library", reply)
        self.assertIn("Set as Active", reply)
        self.assertIn("do not need to retrain", reply)
        self.assertNotIn("Create Model Book", reply)

    def test_active_inverse_design_guidance_covers_configuration_and_run(self):
        project = self.project(
            "active-inverse",
            "model_saved",
            library={"book_count": 1, "active_book_id": "book-0001"},
            last_page="inverse_design",
        )
        reply = self.ask_offline(
            project,
            "How do I perform inverse design from here?",
            "Visible page: Inverse Design\nLatest inverse-design result: not run",
        )
        for marker in (
            "Variable or Fixed",
            "bounds or fixed values",
            "Mean over range",
            "Minimize, Maximize, or Target value",
            "Optional constraints",
            "Run Inverse Design",
        ):
            self.assertIn(marker, reply)

    def test_partial_csv_pair_identifies_the_missing_output(self):
        project = self.project(
            "partial-pair",
            "project_created",
            prep={
                "mode": "pair",
                "source_input_path": "inputs.csv",
                "source_output_path": "",
            },
        )
        reply = self.ask_offline(
            project,
            "I selected inputs.csv. Why can't I continue?",
            "Visible page: Data Prep\nInput CSV: inputs.csv\nOutput CSV: not selected",
        )
        self.assertIn("Output CSV", reply)
        self.assertIn("after both paths are present", reply)

    def test_live_validation_error_is_repeated_with_the_valid_recovery_gate(self):
        project = self.project("invalid-data", "data_prepared", prep={"mode": "pair"})
        reply = self.ask_offline(
            project,
            "Validation failed. What should I do?",
            (
                "Visible page: Data Prep\n"
                "Validation details: Output CSV row 4 contains a non-numeric value\n"
                "Validate and register action enabled: yes"
            ),
        )
        self.assertIn("Output CSV row 4 contains a non-numeric value", reply)
        self.assertIn("Validate and register", reply)
        self.assertIn("Do not bypass", reply)

    def test_status_uses_the_actual_trained_family_and_active_book_state(self):
        project = self.project(
            "xgb-status",
            "model_saved",
            training={
                "status": "TRAINING_COMPLETED",
                "model_name": "xgboost",
                "latest_run_number": 3,
                "metrics": {"MAE": 0.2, "RMSE": 0.3, "R²": 0.8},
            },
            library={"book_count": 1, "active_book_id": "book-0001"},
        )
        reply = self.ask_offline(
            project,
            "What is the current status?",
            "Visible page: Model Library",
        )
        self.assertIn("Run 3 completed XGBoost", reply)
        self.assertIn("Inference", reply)
        self.assertNotIn("set it active", reply.lower())

    def test_project_context_has_no_prepared_data_training_contradiction(self):
        project = self.project("prepared-context", "data_prepared")
        context = build_project_context(project)
        self.assertIn("Available next action: Validate and register", context)
        self.assertIn("Inverse Design", context)
        self.assertNotIn(
            "Available next action: Configure and train",
            context,
        )


if __name__ == "__main__":
    unittest.main()
