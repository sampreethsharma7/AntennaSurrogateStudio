import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from studio.assistant import (
    LIGHTWEIGHT_MODEL,
    STANDARD_MODEL,
    AssistantError,
    OllamaClient,
    SnowBuddyService,
    build_project_context,
    build_response_directive,
    classify_project_question,
    extract_ollama_text,
    _history_for_local_model,
    local_ollama_base_url,
    load_snowbuddy_artifacts,
    recommended_model,
    retrieve_guidance,
)
from studio.development_log import (
    DevelopmentConversationLog,
    development_logging_enabled,
)
from studio.project_store import ProjectStore
from studio.training_results import TrainingResultsError


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def completed_auto_result():
    return SimpleNamespace(
        run_id="run-0003",
        model_name="linear_regression",
        training_mode="auto",
        parameters_used={"fit_intercept": True, "positive": False},
        metrics={"MAE": 0.1, "RMSE": 0.2, "R²": 0.9},
        validation_rmse=0.18,
        training_rows=16,
        test_rows=4,
        median_absolute_error=0.12,
        largest_error_prediction=SimpleNamespace(
            sample_id="Design_014",
            actual_value=2.45,
            predicted_value=2.31,
            absolute_error=0.14,
        ),
        residual_interpretation=(
            "Residuals are centered near zero in the held-out samples."
        ),
        search_level="high",
        configurations_evaluated=4,
        cross_validation_folds=5,
        custom_recommendation=None,
        custom_guidance="",
        insights=[
            "Only 4 held-out samples support these test metrics; treat the "
            "result as preliminary."
        ],
    )


class AssistantTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        self.store = ProjectStore(Path(self.temp_dir.name) / "library")
        self.project = self.store.create_project("Snow Array")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_retrieval_prioritizes_input_output_template_guidance(self):
        chunks = retrieve_guidance(
            "How do I create templates for input and output CSV files?"
        )
        self.assertEqual(chunks[0].title, "Input and output CSV files")

    def test_retrieval_explains_sample_id_pairing(self):
        chunks = retrieve_guidance(
            "How should Sample ID match between my input and output files?"
        )
        self.assertEqual(chunks[0].title, "Input and output CSV files")
        self.assertIn("Sample ID verifies pairing", chunks[0].text)

    def test_context_reflects_project_state(self):
        self.project.manifest["data_prep"] = {
            "source_input_path": "source/inputs.csv",
            "source_output_path": "source/outputs.csv",
            "prepared_inputs_csv": "data/prepared/inputs.csv",
            "prepared_outputs_csv": "data/prepared/outputs.csv",
        }
        context = build_project_context(self.project)
        self.assertIn("Project: Snow Array", context)
        self.assertIn("Workflow stage: project_created", context)
        self.assertIn(
            "Prepared input CSV: data/prepared/inputs.csv",
            context,
        )
        self.assertIn(
            "Prepared output CSV: data/prepared/outputs.csv",
            context,
        )
        self.assertIn("Input CSV source: source/inputs.csv", context)
        self.assertIn("Output CSV source: source/outputs.csv", context)

    def test_context_reflects_registered_dataset(self):
        self.project.manifest["workflow"]["stage"] = "dataset_registered"
        self.project.manifest["dataset_registry"] = {
            "dataset_count": 1,
            "active_dataset_id": "dataset-abc123",
        }
        self.project.manifest["data_prep"] = {
            "prepared_inputs_csv": "data/prepared/inputs.csv",
            "prepared_outputs_csv": "data/prepared/outputs.csv",
            "validation": {"status": "passed"},
            "registration": {"dataset_id": "dataset-abc123"},
        }

        context = build_project_context(self.project)

        self.assertIn("Workflow stage: dataset_registered", context)
        self.assertIn("Registered datasets: 1", context)
        self.assertIn("Active dataset: dataset-abc123", context)
        self.assertIn("Dataset validation: passed", context)
        self.assertIn("Registered dataset: dataset-abc123", context)
        self.assertIn("can run Linear Regression", context)
        self.assertIn("selected Custom parameters", context)

    def test_context_reflects_completed_basic_training(self):
        self.project.manifest["workflow"]["stage"] = "model_trained"
        self.project.manifest["model_training"] = {
            "status": "TRAINING_COMPLETED",
            "model_name": "linear_regression",
            "latest_run_number": 3,
            "metrics": {"MAE": 0.1, "RMSE": 0.2, "R²": 0.9},
        }

        context = build_project_context(self.project)

        self.assertIn("Workflow stage: model_trained", context)
        self.assertIn("Last training status: TRAINING_COMPLETED", context)
        self.assertIn("Last trained model: linear_regression", context)
        self.assertIn("Latest training run: Run 3", context)
        self.assertIn("Last training MAE: 0.1", context)
        self.assertIn("Last training RMSE: 0.2", context)
        self.assertIn("Last training R²: 0.9", context)

    def test_context_includes_artifact_grounded_training_findings(self):
        self.project.manifest["workflow"]["stage"] = "model_trained"
        self.project.manifest["model_training"] = {
            "status": "TRAINING_COMPLETED",
            "model_name": "linear_regression",
            "latest_run_number": 3,
            "metrics": {"MAE": 0.1, "RMSE": 0.2, "R²": 0.9},
        }
        result = SimpleNamespace(
            run_id="run-0003",
            training_mode="auto",
            parameters_used={"fit_intercept": True, "positive": False},
            validation_rmse=0.18,
            training_rows=16,
            test_rows=4,
            median_absolute_error=0.12,
            largest_error_prediction=SimpleNamespace(
                sample_id="Design_014",
                actual_value=2.45,
                predicted_value=2.31,
                absolute_error=0.14,
            ),
            residual_interpretation=(
                "Residuals are centered near zero in the held-out samples."
            ),
            search_level="high",
            configurations_evaluated=4,
            cross_validation_folds=5,
            custom_recommendation=None,
            custom_guidance="",
            insights=[
                "Only 4 held-out samples support these test metrics; treat the "
                "result as preliminary."
            ],
        )

        with patch(
            "studio.assistant.load_latest_training_results",
            return_value=result,
        ):
            context = build_project_context(self.project)

        self.assertIn("calculated from the latest saved run artifacts", context)
        self.assertIn("Latest validation RMSE: 0.18", context)
        self.assertIn("Latest median absolute error: 0.12", context)
        self.assertIn("sample=Design_014", context)
        self.assertIn("Latest Auto search scope: 4 configurations, 5 folds", context)
        self.assertIn("Only 4 held-out samples", context)

    def test_context_handles_unreadable_training_findings_safely(self):
        self.project.manifest["workflow"]["stage"] = "model_trained"
        with patch(
            "studio.assistant.load_latest_training_results",
            side_effect=TrainingResultsError("private artifact detail"),
        ):
            context = build_project_context(self.project)

        self.assertIn("saved result artifacts could not be loaded safely", context)
        self.assertNotIn("private artifact detail", context)

    def test_test1_style_questions_receive_specific_directives(self):
        self.project.manifest["workflow"]["stage"] = "data_prepared"
        self.project.manifest["data_prep"] = {
            "mode": "parameters",
            "source_path": "exports/batch",
            "sample_count": 1000,
            "selected_inputs": ["P1", "P2", "P3", "P4"],
            "selected_output": "Gain,Phi=0.0 []",
            "prepared_rows": 1000,
            "prepared_output_columns": 361,
        }

        self.assertEqual(
            classify_project_question("Whats the current status?"),
            "status",
        )
        status = build_response_directive(
            self.project,
            "Whats the current status?",
            "Visible page: Data Prep",
        )
        loaded = build_response_directive(
            self.project,
            "What are loading here??",
            "Visible page: Data Prep",
        )
        comparison = build_response_directive(
            self.project,
            "whats the difference between two data source methods?",
            "Visible page: Data Prep",
        )

        self.assertIn("Prepared output columns: 361", status)
        self.assertIn("Selected inputs: P1, P2, P3, P4", loaded)
        self.assertIn("Input + output files", comparison)
        self.assertIn("Do not mention the retired", comparison)

    def test_lhs_questions_receive_current_generator_guidance(self):
        question = "How do I use the LHS sample generator?"
        self.assertEqual(
            classify_project_question(question),
            "lhs_sampling",
        )
        directive = build_response_directive(
            self.project,
            question,
            "Visible page: Data Prep",
        )
        self.assertIn("Data Prep > Source > LHS sample generator", directive)
        self.assertIn("does not generate physical responses", directive)
        self.assertTrue(
            any(
                chunk.title == "Latin Hypercube simulation samples"
                for chunk in retrieve_guidance(question)
            )
        )

        with patch.object(
            OllamaClient,
            "create_response",
            side_effect=AssistantError("offline"),
        ):
            reply, used_local_model = SnowBuddyService(
                self.store,
                model=LIGHTWEIGHT_MODEL,
            ).ask(self.project, question)

        self.assertFalse(used_local_model)
        self.assertIn("Generate Samples", reply)
        self.assertIn("Export inputs.csv", reply)
        self.assertIn("leaves Output CSV empty", reply)
        self.assertIn("does not simulate responses", reply)

    def test_parameter_sweep_questions_are_distinct_from_lhs(self):
        questions = (
            "What file or folder does the tool accept in parasweep?",
            "There is also #Parameters sweep option, how does it work?",
            "Can I use #parameter sweep to parse a CST .txt frequency and S11 export?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(
                    classify_project_question(question),
                    "parameter_sweep",
                )

        directive = build_response_directive(
            self.project,
            questions[-1],
            "Visible page: Data Prep",
        )
        self.assertIn("existing .txt export", directive)
        self.assertIn("Browse file or Browse folder", directive)
        self.assertIn("Save selection", directive)
        self.assertIn("first table column is preserved", directive)
        self.assertNotIn("Latin Hypercube", directive)

    def test_lhs_confused_parameter_sweep_answer_is_replaced(self):
        service = SnowBuddyService(self.store, model=LIGHTWEIGHT_MODEL)
        with patch.object(
            OllamaClient,
            "create_response",
            return_value=(
                "#Parameters sweep is the LHS sample generator. Use Generate "
                "Samples to create inputs.csv; input and output CSVs need the "
                "same row and column counts."
            ),
        ):
            reply, used_local_model = service.ask(
                self.project,
                "I have a CST .txt with #Parameters, frequency and S11. Can I parse it?",
            )

        self.assertFalse(used_local_model)
        self.assertIn("existing `.txt` solver export", reply)
        self.assertIn("Browse file", reply)
        self.assertIn("Parse", reply)
        self.assertIn("Save selection", reply)
        self.assertIn("Prepare input + output", reply)
        self.assertIn("Frequency or Theta", reply)
        self.assertIn("do not need the same number of columns", reply)
        self.assertNotIn("LHS sample generator. Use Generate", reply)

    def test_run_question_receives_compact_authoritative_evidence(self):
        self.project = self.store.update_project(
            self.project,
            {
                "workflow": {"stage": "model_trained"},
                "model_training": {
                    "status": "TRAINING_COMPLETED",
                    "model_name": "linear_regression",
                    "latest_run_number": 3,
                    "metrics": {"MAE": 0.1, "RMSE": 0.2, "R²": 0.9},
                },
            },
        )
        captured = {}

        def grounded_response(**kwargs):
            captured.update(kwargs)
            return (
                "run-0003 completed Linear Regression in Auto High mode. It "
                "evaluated 4 configurations with 5 folds and selected "
                "fit_intercept=True, positive=False. Validation RMSE: 0.18. "
                "Test MAE: 0.1, Test RMSE: 0.2, Test R²: 0.9. The split used "
                "16 training samples and 4 test samples. Median absolute error "
                "was 0.12. Use Open Test Data CSV for the saved rows."
            )

        self.assertEqual(
            classify_project_question("What should I know about this run?"),
            "training_result",
        )
        with patch(
            "studio.assistant.load_latest_training_results",
            return_value=completed_auto_result(),
        ), patch.object(
            OllamaClient,
            "create_response",
            side_effect=grounded_response,
        ):
            reply, used_local_model = SnowBuddyService(
                self.store,
                model=LIGHTWEIGHT_MODEL,
            ).ask(self.project, "What should I know about this run?")

        self.assertTrue(used_local_model)
        self.assertIn("run-0003", reply)
        self.assertIn("Latest run evidence status: available", captured["priority_evidence"])
        self.assertIn("Configurations evaluated: 4", captured["priority_evidence"])
        self.assertIn("Use the authoritative Latest Run Evidence", captured["response_directive"])
        self.assertIn("## Training Results page", captured["blind_gui_reference"])
        self.assertNotIn("## Data Prep page", captured["blind_gui_reference"])

    def test_generic_run_tour_is_replaced_by_grounded_summary(self):
        self.project = self.store.update_project(
            self.project,
            {
                "workflow": {"stage": "model_trained"},
                "model_training": {
                    "status": "TRAINING_COMPLETED",
                    "model_name": "linear_regression",
                    "latest_run_number": 3,
                    "metrics": {"MAE": 0.1, "RMSE": 0.2, "R²": 0.9},
                },
            },
        )
        generic_tour = (
            "This run is a completed model training session. It includes final "
            "metrics like MAE, RMSE, and R², along with test predictions and "
            "residual analysis. Explore the Training Results page for details."
        )
        with patch(
            "studio.assistant.load_latest_training_results",
            return_value=completed_auto_result(),
        ), patch.object(
            OllamaClient,
            "create_response",
            return_value=generic_tour,
        ):
            reply, used_local_model = SnowBuddyService(
                self.store,
                model=LIGHTWEIGHT_MODEL,
            ).ask(self.project, "What should I know about this run?")

        self.assertFalse(used_local_model)
        self.assertIn("run-0003 completed", reply)
        self.assertIn("Auto High", reply)
        self.assertIn("4 configurations", reply)
        self.assertIn("5 folds", reply)
        self.assertIn("fit_intercept=True", reply)
        self.assertIn("Validation RMSE", reply)
        self.assertIn("MAE=0.1", reply)
        self.assertIn("16 training samples", reply)
        self.assertIn("4 test samples", reply)
        self.assertIn("Design_014", reply)
        self.assertIn("Open Test Data CSV", reply)

    def test_stale_download_action_is_replaced_by_current_results_action(self):
        self.project = self.store.update_project(
            self.project,
            {
                "workflow": {"stage": "model_trained"},
                "model_training": {
                    "status": "TRAINING_COMPLETED",
                    "model_name": "linear_regression",
                    "latest_run_number": 3,
                    "metrics": {"MAE": 0.1, "RMSE": 0.2, "R²": 0.9},
                },
            },
        )
        stale_reply = (
            "run-0003 used Auto High with 4 configurations and 5 folds. "
            "fit_intercept=True, positive=False. Validation RMSE: 0.18. "
            "Test MAE: 0.1, Test RMSE: 0.2, Test R²: 0.9. It used 16 "
            "training samples and 4 test samples. Download predictions from the "
            "compact action strip."
        )
        with patch(
            "studio.assistant.load_latest_training_results",
            return_value=completed_auto_result(),
        ), patch.object(
            OllamaClient,
            "create_response",
            return_value=stale_reply,
        ):
            reply, used_local_model = SnowBuddyService(
                self.store,
                model=LIGHTWEIGHT_MODEL,
            ).ask(self.project, "Explain the latest run.")

        self.assertFalse(used_local_model)
        self.assertIn("Open Test Data CSV", reply)
        self.assertNotIn("Download predictions", reply)
        self.assertNotIn("compact action strip", reply)

    def test_offline_answer_is_persisted(self):
        service = SnowBuddyService(self.store, model=LIGHTWEIGHT_MODEL)
        with patch.object(
            OllamaClient,
            "create_response",
            side_effect=AssistantError("Ollama is not running"),
        ):
            reply, used_local_model = service.ask(
                self.project, "What should I do next?"
            )
        history = self.store.load_chat(self.project)

        self.assertFalse(used_local_model)
        self.assertIn("Data Prep", reply)
        self.assertEqual(len(history), 2)

    def test_welcome_mode_works_without_project(self):
        service = SnowBuddyService(self.store, model=LIGHTWEIGHT_MODEL)
        with patch.object(
            OllamaClient,
            "create_response",
            side_effect=AssistantError("Ollama is not running"),
        ):
            reply, used_local_model = service.ask(
                None,
                "How should I begin?",
                live_ui_state=(
                    "Visible page: Start\n"
                    "Active project: None\n"
                    "SnowBuddy mode: Welcome"
                ),
            )

        history = self.store.load_welcome_chat()
        self.assertFalse(used_local_model)
        self.assertIn("Create project", reply)
        self.assertIn("Welcome mode", reply)
        self.assertEqual(len(history), 2)

    def test_welcome_context_reaches_local_model(self):
        captured = {}

        def fake_response(**kwargs):
            captured.update(kwargs)
            return "Choose + Create project."

        service = SnowBuddyService(self.store, model=LIGHTWEIGHT_MODEL)
        with patch.object(
            OllamaClient,
            "create_response",
            side_effect=fake_response,
        ):
            reply, used_local_model = service.ask(
                None,
                "What can I do here?",
                live_ui_state="SnowBuddy mode: Welcome",
            )

        self.assertTrue(used_local_model)
        self.assertEqual(reply, "Choose + Create project.")
        self.assertIn("Studio mode: Welcome", captured["project_context"])
        self.assertIn("Active project: None", captured["project_context"])

    def test_extracts_ollama_message_text(self):
        payload = {"message": {"role": "assistant", "content": "Local answer"}}
        self.assertEqual(extract_ollama_text(payload), "Local answer")

    def test_extracts_ollama_text_without_thinking_tags(self):
        payload = {
            "message": {
                "role": "assistant",
                "content": "<think>private reasoning</think>\nGrounded answer</think>",
            }
        }
        self.assertEqual(extract_ollama_text(payload), "Grounded answer")

    def test_resource_recommendation(self):
        self.assertEqual(recommended_model(8), LIGHTWEIGHT_MODEL)
        self.assertEqual(recommended_model(16), STANDARD_MODEL)

    def test_model_selection_persists_per_machine_library(self):
        service = SnowBuddyService(self.store, model=STANDARD_MODEL)
        service.set_model(LIGHTWEIGHT_MODEL)
        restored = SnowBuddyService(self.store)
        self.assertEqual(restored.model, LIGHTWEIGHT_MODEL)

    def test_ollama_client_uses_local_chat_api(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(
                {"message": {"role": "assistant", "content": "Prepared locally"}}
            )

        client = OllamaClient(
            LIGHTWEIGHT_MODEL,
            base_url="http://127.0.0.1:11434",
            timeout=9,
        )
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            answer = client.create_response(
                project_context="Project: Snow Array",
                retrieved_guidance=retrieve_guidance("next"),
                history=[
                    {
                        "role": "assistant",
                        "content": "Choose Filename phase sweep and set Phi cut.",
                    },
                    {"role": "user", "content": "What is next?"},
                ],
                response_directive="Answer the current question before suggesting steps.",
            )

        self.assertEqual(answer, "Prepared locally")
        self.assertEqual(captured["url"], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(captured["payload"]["model"], LIGHTWEIGHT_MODEL)
        self.assertFalse(captured["payload"]["stream"])
        self.assertEqual(captured["payload"]["options"]["num_ctx"], 8192)
        serialized_messages = json.dumps(captured["payload"]["messages"])
        self.assertNotIn("Filename phase sweep", serialized_messages)
        self.assertIn("trusted turn-specific response directive", serialized_messages)
        self.assertIn("Answer the current question", serialized_messages)

    def test_stale_project_assistant_replies_do_not_reenter_local_model_prompt(self):
        history = [
            {"role": "user", "content": "Which models exist?"},
            {
                "role": "assistant",
                "content": "Linear Regression is the only model currently available.",
            },
            {"role": "user", "content": "What is an inverse-design objective?"},
        ]

        prepared = _history_for_local_model(history)
        serialized = json.dumps(prepared)

        self.assertNotIn("only model currently available", serialized)
        self.assertNotIn("Which models exist?", serialized)
        self.assertIn("inverse-design objective", serialized)

    def test_offline_snowbuddy_explains_inverse_design_objective_from_current_ui(self):
        service = SnowBuddyService(self.store, model=LIGHTWEIGHT_MODEL)
        with patch.object(
            OllamaClient,
            "create_response",
            side_effect=AssistantError("Ollama unavailable"),
        ):
            reply, used_local_model = service.ask(
                self.project,
                "What is the objective in inverse design?",
                live_ui_state="Visible page: Inverse Design",
            )

        self.assertFalse(used_local_model)
        self.assertIn("one scalar predicted score", reply)
        self.assertIn("Mean over range", reply)

    def test_retired_local_model_answer_is_replaced_by_current_guide(self):
        service = SnowBuddyService(self.store, model=LIGHTWEIGHT_MODEL)
        with patch.object(
            OllamaClient,
            "create_response",
            return_value=(
                "Choose Filename phase sweep and adjust the Phi cut."
            ),
        ):
            reply, used_local_model = service.ask(
                self.project,
                "What is the difference between the two data source methods?",
            )

        self.assertFalse(used_local_model)
        self.assertIn("Input + output files", reply)
        self.assertIn("#Parameters sweep", reply)
        self.assertNotIn("Filename phase sweep", reply)

    def test_explanation_does_not_force_a_next_action_heading(self):
        service = SnowBuddyService(self.store, model=LIGHTWEIGHT_MODEL)
        with patch.object(
            OllamaClient,
            "create_response",
            return_value=(
                "**Next action:** Review the methods. Input + output files uses "
                "CSVs and #Parameters sweep uses text blocks."
            ),
        ):
            reply, used_local_model = service.ask(
                self.project,
                "What is the difference between the two data source methods?",
            )

        self.assertFalse(used_local_model)
        self.assertTrue(reply.startswith("**Input + output files**"))

    def test_prepared_project_does_not_repeat_data_prep_as_next_step(self):
        self.project = self.store.update_project(
            self.project,
            {
                "workflow": {"stage": "data_prepared", "completed_steps": 2},
                "data_prep": {
                    "mode": "parameters",
                    "source_path": "exports/batch",
                    "sample_count": 1000,
                    "selected_inputs": ["P1", "P2", "P3", "P4"],
                    "selected_output": "Gain,Phi=0.0 []",
                    "prepared_rows": 1000,
                    "prepared_input_columns": 4,
                    "prepared_output_columns": 361,
                },
            },
        )
        service = SnowBuddyService(self.store, model=LIGHTWEIGHT_MODEL)
        with patch.object(
            OllamaClient,
            "create_response",
            return_value=(
                "The data is loaded. The next step is to define the surrogate "
                "model contract and prepare the data for training."
            ),
        ):
            reply, used_local_model = service.ask(
                self.project,
                "What are loading here??",
            )

        self.assertFalse(used_local_model)
        self.assertIn("1,000 samples", reply)
        self.assertIn("P1, P2, P3, P4", reply)
        self.assertNotIn("next step is to define", reply.lower())

    def test_ollama_connection_is_restricted_to_loopback(self):
        self.assertEqual(
            local_ollama_base_url("http://localhost:11434"),
            "http://localhost:11434",
        )
        self.assertEqual(
            local_ollama_base_url("http://[::1]:11434"),
            "http://[::1]:11434",
        )
        with self.assertRaises(AssistantError):
            local_ollama_base_url("https://ollama.example.com")
        with self.assertRaises(AssistantError):
            local_ollama_base_url("http://192.168.1.25:11434")

    def test_development_logging_requires_both_safety_switches(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(development_logging_enabled())
        with patch.dict(
            os.environ,
            {"ANTENNA_STUDIO_BUILD_CHANNEL": "development"},
            clear=True,
        ):
            self.assertFalse(development_logging_enabled())
        with patch.dict(
            os.environ,
            {
                "ANTENNA_STUDIO_BUILD_CHANNEL": "development",
                "SNOWBUDDY_DEVELOPMENT_LOG": "1",
            },
            clear=True,
        ):
            self.assertTrue(development_logging_enabled())

    def test_development_log_captures_local_question_and_response(self):
        log_path = Path(self.temp_dir.name) / "snowbuddy_sessions.jsonl"
        development_log = DevelopmentConversationLog(enabled=True, path=log_path)
        service = SnowBuddyService(
            self.store,
            model=LIGHTWEIGHT_MODEL,
            development_log=development_log,
        )
        with patch.object(
            OllamaClient,
            "create_response",
            return_value="Use Data Prep next.",
        ):
            service.ask(
                self.project,
                "What should I do?",
                live_ui_state="Visible page: Start",
            )

        entries = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["question"], "What should I do?")
        self.assertEqual(entries[0]["response"], "Use Data Prep next.")
        self.assertEqual(entries[0]["response_source"], "ollama")
        self.assertEqual(entries[0]["mode"], "focus")
        self.assertEqual(entries[0]["project_name"], "Snow Array")

    def test_disabled_development_log_creates_no_file(self):
        log_path = Path(self.temp_dir.name) / "disabled.jsonl"
        development_log = DevelopmentConversationLog(enabled=False, path=log_path)
        recorded = development_log.record(
            project=None,
            question="Hello",
            response="Hi",
            model=LIGHTWEIGHT_MODEL,
            used_local_model=True,
            live_ui_state="Visible page: Start",
        )
        self.assertFalse(recorded)
        self.assertFalse(log_path.exists())

    def test_runtime_contract_and_live_gui_state_reach_local_model(self):
        captured = {}

        def fake_response(**kwargs):
            captured.update(kwargs)
            return "Choose both CSV files; the pair loads automatically."

        service = SnowBuddyService(self.store, model=LIGHTWEIGHT_MODEL)
        with patch.object(
            OllamaClient,
            "create_response",
            side_effect=fake_response,
        ):
            reply, used_local_model = service.ask(
                self.project,
                "Where am I?",
                live_ui_state="Visible page: Data Prep\nSource loaded or parsed: no",
            )

        self.assertTrue(used_local_model)
        self.assertEqual(
            reply,
            "Choose both CSV files; the pair loads automatically.",
        )
        self.assertIn(
            "SnowBuddy Character Contract", captured["character_contract"]
        )
        self.assertIn("SnowBuddy Blind GUI Read", captured["blind_gui_reference"])
        self.assertIn("Visible page: Data Prep", captured["live_ui_state"])
        self.assertIn(
            "Answer the user's actual question directly",
            captured["response_directive"],
        )

    def test_contract_loader_has_packaged_fallbacks(self):
        with tempfile.TemporaryDirectory(dir=Path(self.temp_dir.name)) as empty:
            artifacts = load_snowbuddy_artifacts(empty)
        self.assertIn("project-aware", artifacts.character)
        self.assertIn(
            "Start, Data Prep, Model Training, Training Results, Model Library, Inference, and Inverse Design",
            artifacts.blind_gui,
        )


if __name__ == "__main__":
    unittest.main()
