import os
import tempfile
import tkinter as tk
import unittest
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from studio.antenna_builder_ui import (
    GEMINI_LABEL,
    GROQ_LABEL,
    INSTRUCTION_COMPOSER_MAX_HEIGHT,
    INSTRUCTION_COMPOSER_MIN_HEIGHT,
    LOCAL_QWEN_LABEL,
    OPENROUTER_NEMOTRON_LABEL,
    TRANSCRIPT_BODY_FONT,
)
from studio.antenna_builder import ProjectMemoryItem, load_builder_session, save_project_design
from studio.antenna_design import AntennaDesign
from studio.antenna_llm_planner import AgentStep, EngineeringDisposition, LLMToolPlan, PlannedToolCall
from studio.planner_credentials import KEYRING_SERVICE, LOCAL_OLLAMA
from studio.planner_model_discovery import GEMINI, ModelDiscoveryError, PlannerModel
from studio.planner_onboarding_ui import SetupOutcome
from studio.project_store import ProjectStore
from studio.theme import FONTS
from studio.ui import StudioApp


class _EmptyCredentialStore:
    """A credential store with nothing in it, and no connection to the machine."""

    def __init__(self, values=None):
        self.values = dict(values or {})

    def get_password(self, service, account):
        return self.values.get((service, account))

    def set_password(self, service, account, password):
        self.values[(service, account)] = password

    def delete_password(self, service, account):
        del self.values[(service, account)]


GUI_MAY_BE_AVAILABLE = (
    os.name == "nt"
    or os.sys.platform == "darwin"
    or bool(os.environ.get("DISPLAY"))
)


class _ImmediateThread:
    def __init__(self, *, target, daemon):
        self.target = target
        self.daemon = daemon

    def start(self):
        self.target()


class _DeferredThread:
    pending = []

    def __init__(self, *, target, daemon):
        self.target = target
        self.daemon = daemon
        self.__class__.pending.append(self)

    def start(self):
        return None


class _PlannerReturning:
    def __init__(self, plan):
        self.result = plan
        self.requests = []

    def plan(self, **request):
        self.requests.append(request)
        return self.result

    def plan_agent_step(self, **request):
        # UI fixtures supply a complete, valid agent turn. Keep requests as the
        # design-proposal captures used by these tests, separate from its finish.
        observation = request.get("agent_observation")
        if observation is not None and observation.outcome == "accepted":
            report = request["engineering_report"]
            return AgentStep("finish", "UI fixture completed with findings acknowledged.",
                engineering_disposition=EngineeringDisposition(report.semantic_design_hash,
                    tuple(f.observation_id for f in report.findings if f.severity == "warning")))
        plan = self.plan(**request)
        return AgentStep(plan.status, plan.message, plan.calls)


def _tool_plan(*calls, status="execute", message="UI test tool plan"):
    return LLMToolPlan(
        status,
        message,
        tuple(PlannedToolCall(name, arguments) for name, arguments in calls),
    )


@unittest.skipUnless(GUI_MAY_BE_AVAILABLE, "A desktop display is required.")
class AntennaBuilderPageTests(unittest.TestCase):
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
            raise unittest.SkipTest(f"A desktop display is not available: {exc}") from exc
        cls.app.withdraw()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "app"):
            cls.app.destroy()
        if hasattr(cls, "temp_dir"):
            cls.temp_dir.cleanup()

    def setUp(self):
        self.app.withdraw()
        self.project = self.store.create_project(self._testMethodName)
        project = self.store.open_project(self.project.path, touch=False)
        self.app.set_project(project, target_page="design_start")
        self.app.update()
        self.page = self.app.antenna_builder_page
        # Entering the builder now settles how the planner will be reached. A
        # test machine has no cloud credential, so without a stand-in these
        # tests would block on a modal dialog nothing closes. Local Ollama is
        # what they already exercise, so that is what the stand-in chooses.
        self.setup_dialog_calls = []
        self.app.design_start_page.credential_backend = _EmptyCredentialStore()
        self.app.design_start_page.setup_dialog_runner = self._offline_setup_choice
        self.page.credential_backend = _EmptyCredentialStore()

    def _offline_setup_choice(self, *args, **kwargs):
        self.setup_dialog_calls.append(kwargs.get("provider"))
        return SetupOutcome(provider=LOCAL_OLLAMA, verified=False)

    def tearDown(self):
        dialog = self.app.data_page.sample_generator_dialog
        if dialog is not None and dialog.winfo_exists():
            dialog.destroy()
        self.app.data_page.sample_generator_dialog = None
        self.app.withdraw()

    def _apply_llm_plan(self, instruction, *calls, status="execute", message="UI test tool plan"):
        planner = _PlannerReturning(_tool_plan(*calls, status=status, message=message))
        with (
            patch("studio.antenna_builder_ui.SchemaConstrainedLLMPlanner", return_value=planner),
            patch("studio.antenna_builder_ui.threading.Thread", _ImmediateThread),
            patch("studio.antenna_builder_ui.messagebox.showwarning"),
        ):
            self.page.instruction_var.set(instruction)
            self.page.apply_instruction()
            self.app.update()
        return planner

    def _create_inset_design(self):
        self.app.design_start_page.choose_template()
        return self._apply_llm_plan(
            "Design an inset-fed rectangular patch antenna at 2.45 GHz on FR4.",
            ("recipe.select", {"recipe_id": "inset_patch_v2"}),
            ("parameter.set", {"key": "frequency_ghz", "value": 2.45}),
            ("parameter.set", {"key": "material", "value": "FR4"}),
        )

    def _drain_ui_until(self, predicate):
        for _ in range(10):
            self.app.update_idletasks()
            self.app.update()
            if predicate():
                break

    def test_new_project_opens_design_start_and_existing_path_continues(self):
        self.assertEqual(self.app.active_page, "design_start")
        self.assertIn("design_start", self.app.nav_buttons)

        self.app.design_start_page.choose_existing()

        self.assertEqual(self.app.active_page, "data")
        reopened = self.store.open_project(self.project.path, touch=False)
        self.assertEqual(reopened.manifest["design_start"]["choice"], "existing_design")

    def test_template_path_opens_experimental_builder(self):
        self.app.design_start_page.choose_template()

        self.assertEqual(self.app.active_page, "antenna_builder")
        self.assertIsNone(self.page.state)
        self.assertIsNone(self.page.preview.scene)
        self.assertEqual(self.page.preview.mesh_actor_count, 0)
        self.assertIn("No antenna design yet", self.page.empty_preview.cget("text"))
        self.assertEqual(self.page.export_button.cget("state"), "disabled")
        self.assertEqual(self.page.native_cst_button.cget("state"), "disabled")
        self.assertEqual(self.page.lhs_button.cget("state"), "disabled")
        self.assertEqual(len(self.page.starter_example_buttons), 4)
        starter_labels = [button.cget("text") for button in self.page.starter_example_buttons]
        self.assertEqual(
            [label.split()[0] for label in starter_labels],
            ["Inset", "Circular", "Dipole", "1\u00d74"],
        )
        self.assertTrue(all(len(label) <= 38 for label in starter_labels))
        self.assertIn("Not supported: horns", self.page.conversation_frame.winfo_children()[-1].cget("text"))
        self.page.starter_example_buttons[0].invoke()
        self.assertIn("inset-fed rectangular patch", self.page.instruction_var.get())
        self.assertFalse((self.project.path / "design" / "antenna_state.json").exists())
        with (
            patch("studio.antenna_builder_ui.filedialog.asksaveasfilename") as save_dialog,
            patch.object(self.app, "show_page") as show_page,
        ):
            self.page.export_cst()
            self.page.create_native_cst()
            self.page.send_to_lhs()
        save_dialog.assert_not_called()
        show_page.assert_not_called()
        snapshot = "\n".join(self.page.describe_ui_state())
        self.assertIn("Builder status: awaiting_design", snapshot)
        self.assertIn("Design: null", snapshot)

    def test_welcome_and_sidebar_advertise_text_to_geometry_workflow(self):
        self.assertIn("Describe a supported antenna", self.app.start_page.workspace_subtitle.cget("text"))
        self.assertEqual(
            self.app.nav_buttons["design_start"].accessible_name,
            "Antenna Design",
        )
        current_project = self.app.current_project
        try:
            self.app.current_project = None
            self.app.start_page.refresh()
            self.assertIn("plain language", self.app.start_page.hero_subtitle.cget("text"))
        finally:
            self.app.current_project = current_project
            self.app.start_page.refresh()

    def test_first_inset_design_publishes_revision_zero_and_enables_controls(self):
        planner = self._create_inset_design()

        self.assertIsNotNone(self.page.state)
        self.assertEqual(self.page.state.recipe_id, "inset_patch_v2")
        self.assertEqual(self.page.state.revision, 0)
        self.assertEqual(self.page.export_button.cget("state"), "normal")
        self.assertEqual(self.page.native_cst_button.cget("state"), "normal")
        self.assertEqual(self.page.lhs_button.cget("state"), "normal")
        self.assertEqual(self.page.material_menu.cget("state"), "normal")
        self.assertIn("patch_width_mm", self.page.parameter_vars)
        self.assertTrue((self.project.path / "design" / "antenna_state.json").exists())
        restored = load_builder_session(self.project.path)
        self.assertEqual(restored.design, self.page.state)
        self.assertEqual(restored.memory.canonical_ref.revision, 0)
        self.assertEqual(len(restored.conversation), 2)
        self.assertIsNone(planner.requests[0]["current_design"])
        self.assertEqual(
            [tool["name"] for tool in planner.requests[0]["capability_manifest"]["callable_tools"]],
            ["recipe.select", "parameter.set"],
        )

    def test_project_context_persistently_marks_active_constraint_violation(self):
        self._create_inset_design()
        constraint = ProjectMemoryItem(
            item_id="semantic:board_width_limit:test:value",
            key="board_width_limit",
            value=10.0,
            unit="mm",
            status="active",
            source_turn_id="test",
            source="user_semantic",
            semantic_kind="constraint",
            evidence_quote="Keep the board under 10 mm.",
            constraint_operator="max",
        )
        self.page.session.memory = replace(
            self.page.session.memory,
            requirements=(*self.page.session.memory.requirements, constraint),
        )

        self.page._render_state()
        self.app.update_idletasks()

        self.assertIn("CONSTRAINT VIOLATION", self.page.project_context_title.cget("text"))
        self.assertIn("active maximum of 10 mm", self.page.project_context_label.cget("text"))

    def test_capability_question_persists_conversation_and_keeps_workspace_blank(self):
        self.app.design_start_page.choose_template()

        self._apply_llm_plan(
            "What antenna types can you currently build?",
            status="clarify",
            message="I can build an inset-fed patch, circular patch, or simple dipole. Which should I create?",
        )

        self.assertIsNone(self.page.state)
        self.assertEqual(len(self.page.conversation), 2)
        self.assertIn("Which should I create", self.page.conversation[-1]["content"])
        self.assertEqual(self.page.transcript_rows[-1]["marker"], "?")
        self.assertEqual(
            self.page.instruction_var.get(),
            "What antenna types can you currently build?",
        )
        self.assertFalse((self.project.path / "design" / "antenna_state.json").exists())
        restored = load_builder_session(self.project.path)
        self.assertIsNone(restored.design)
        self.assertEqual(len(restored.memory.open_questions), 1)

    def test_unsupported_request_persists_limitation_and_keeps_workspace_blank(self):
        self.app.design_start_page.choose_template()

        self._apply_llm_plan(
            "Design a helical antenna.",
            status="refuse",
            message="A helical antenna capability is not installed.",
        )

        self.assertIsNone(self.page.state)
        restored = load_builder_session(self.project.path)
        self.assertIsNone(restored.design)
        self.assertEqual(len(restored.memory.limitations), 1)
        self.assertEqual(restored.conversation[-1]["outcome"], "refusal")
        self.assertEqual(self.page.transcript_rows[-1]["marker"], "✕")
        self.assertEqual(self.page.instruction_var.get(), "Design a helical antenna.")

    def test_transcript_autoscrolls_and_finished_turn_has_success_style(self):
        self._create_inset_design()
        self.assertEqual(self.page.transcript_rows[-1]["marker"], "✓")
        self.page.conversation = [
            {
                "role": "user" if index % 2 == 0 else "builder",
                "content": f"Transcript line {index} " * 4,
                "outcome": "request" if index % 2 == 0 else "completed",
            }
            for index in range(30)
        ]
        self.page._render_conversation()
        self.app.update_idletasks()
        canvas = self.page.conversation_frame._parent_canvas
        self.assertAlmostEqual(canvas.yview()[1], 1.0, places=3)
        label = self.page.transcript_rows[-1]["label"]
        native_label = label._label
        self.assertTrue(native_label.bind("<MouseWheel>"))
        self.assertTrue(native_label.bind("<Button-4>"))
        self.assertTrue(native_label.bind("<Button-5>"))
        self.assertEqual(tuple(map(int, canvas.cget("scrollregion").split())), canvas.bbox("all"))

    def test_transcript_text_is_thirty_percent_larger_and_rewraps_with_width(self):
        self.app.design_start_page.choose_template()
        self.page.conversation = [{
            "role": "builder",
            "content": "A long engineering response that must reflow as the conversation pane changes width. " * 3,
            "outcome": "clarification",
        }]
        self.page._render_conversation()
        label = self.page.transcript_rows[0]["label"]

        self.page._conversation_resized(SimpleNamespace(width=360))
        narrow_wrap = int(label.cget("wraplength"))
        self.page._conversation_resized(SimpleNamespace(width=700))
        wide_wrap = int(label.cget("wraplength"))

        self.assertEqual(
            TRANSCRIPT_BODY_FONT[1],
            round(FONTS["body_small"][1] * 1.30),
        )
        self.assertGreater(wide_wrap, narrow_wrap)
        self.assertEqual((narrow_wrap, wide_wrap), (268, 608))

    def test_planning_disables_composer_and_apply_becomes_cancel(self):
        self.app.design_start_page.choose_template()
        _DeferredThread.pending.clear()
        instruction = "Create an inset-fed rectangular patch at 2.45 GHz on FR4."
        with patch("studio.antenna_builder_ui.threading.Thread", _DeferredThread):
            self.page.instruction_var.set(instruction)
            self.page.apply_instruction()
            self.assertEqual(self.page.instruction_entry.cget("state"), "disabled")
            self.assertEqual(self.page.apply_button.cget("text"), "Cancel")
            self.page.apply_instruction()
            self.assertEqual(self.page.apply_button.cget("text"), "Cancelling…")
            _DeferredThread.pending.pop().target()
            self.app.update()

        self.assertEqual(self.page.instruction_entry.cget("state"), "normal")
        self.assertEqual(self.page.apply_button.cget("text"), "Apply")
        self.assertEqual(self.page.instruction_var.get(), instruction)
        self.assertEqual(self.page.conversation, [])

    def test_blank_and_active_sessions_reopen_without_synthesizing_design(self):
        self.app.design_start_page.choose_template()
        blank = self.store.open_project(self.project.path, touch=False)
        self.app.set_project(blank, target_page="antenna_builder")
        self.assertIsNone(self.page.state)
        self.assertFalse((self.project.path / "design" / "antenna_state.json").exists())

        self._create_inset_design()
        active_design = self.page.state
        active_memory = self.page.session.memory
        active_conversation = list(self.page.conversation)
        reopened = self.store.open_project(self.project.path, touch=False)
        self.app.set_project(reopened, target_page="antenna_builder")

        self.assertEqual(self.page.state, active_design)
        self.assertEqual(self.page.session.memory, active_memory)
        self.assertEqual(self.page.conversation, active_conversation)

    def test_first_circular_design_can_be_created_from_blank_project(self):
        self.app.design_start_page.choose_template()

        self._apply_llm_plan(
            "Design a circular patch at 5.8 GHz on Rogers RT5880.",
            ("recipe.select", {"recipe_id": "circular_patch_v1"}),
            ("parameter.set", {"key": "frequency_ghz", "value": 5.8}),
            ("parameter.set", {"key": "material", "value": "Rogers RT5880"}),
        )

        self.assertEqual(self.page.state.family, "circular_patch")
        self.assertEqual(self.page.state.revision, 0)
        self.assertIn("patch_radius_mm", self.page.parameter_vars)

    def test_blank_workspace_controls_fit_maximized_1366_by_768_layout(self):
        self.app.design_start_page.choose_template()
        self.app.geometry("1366x768+0+0")
        self.app.deiconify()
        self.app.set_sidebar_collapsed(True)
        self.app.set_snowbuddy_collapsed(True)
        self.app.update()

        self.assertIsNone(self.page.state)
        self.assertGreater(self.page.instruction_entry.winfo_height(), 60)
        self.assertGreater(self.page.empty_preview.winfo_height(), 100)
        app_right = self.app.winfo_rootx() + self.app.winfo_width()
        app_bottom = self.app.winfo_rooty() + self.app.winfo_height()
        for control in (
            self.page.planner_menu,
            self.page.model_menu,
            self.page.apply_button,
            self.page.export_button,
            self.page.native_cst_button,
            self.page.lhs_button,
        ):
            self.assertLessEqual(control.winfo_rootx() + control.winfo_width(), app_right)
            self.assertLessEqual(control.winfo_rooty() + control.winfo_height(), app_bottom)

    def test_legacy_saved_design_loads_and_transient_load_failure_never_erases_it(self):
        legacy = AntennaDesign.starting_design()
        save_project_design(self.project.path, legacy, [{"role": "user", "content": "legacy"}])
        reopened = self.store.open_project(self.project.path, touch=False)
        self.app.set_project(reopened, target_page="antenna_builder")
        self.assertEqual(self.page.state, legacy)

        state_path = self.project.path / "design" / "antenna_state.json"
        with patch(
            "studio.antenna_builder_ui.load_builder_session",
            side_effect=ValueError("simulated load failure"),
        ):
            self.page.set_project(reopened)

        self.assertTrue(state_path.exists())
        self.assertIsNone(self.page.session)
        self.assertEqual(self.page.apply_button.cget("state"), "disabled")

    def test_the_planner_controls_are_all_wide_enough_to_use(self):
        """Every control in the planner row has to be visible, not just present.

        These share one grid, and the provider menu holds the only stretchy
        column, so anything added beside them comes out of its width. Adding
        the API keys button next to the model menu squeezed the provider menu
        to 2px: still there, still selectable from code, and invisible to the
        person who has to change provider with it.
        """

        self.app.design_start_page.choose_template()
        self.app.update()
        self.app.update_idletasks()
        for name, widget in (
            ("provider menu", self.page.planner_menu),
            ("model menu", self.page.model_menu),
            ("refresh button", self.page.refresh_models_button),
            ("API keys button", self.page.api_keys_button),
            ("free-only checkbox", self.page.free_only_checkbox),
        ):
            with self.subTest(control=name):
                self.assertEqual(widget.winfo_manager(), "grid", f"{name} is not laid out")
                self.assertGreaterEqual(
                    widget.winfo_width(),
                    60,
                    f"{name} is {widget.winfo_width()}px wide, too narrow to use",
                )

    def test_planner_selector_marks_cloud_context_and_uses_gemini_backend(self):
        self.app.design_start_page.choose_template()
        self.assertEqual(self.page.planner_var.get(), LOCAL_QWEN_LABEL)
        self.assertIn("stays on this computer", self.page.planner_notice_var.get())
        self.assertEqual(
            list(self.page.planner_menu.cget("values")),
            [LOCAL_QWEN_LABEL, GEMINI_LABEL, GROQ_LABEL, OPENROUTER_NEMOTRON_LABEL],
        )
        self.assertIn("qwen3:8b", self.page.model_var.get())
        self.assertIn("Tested", self.page.model_var.get())

        planner = _PlannerReturning(_tool_plan(
            ("recipe.select", {"recipe_id": "inset_patch_v2"}),
            ("parameter.set", {"key": "patch_width_mm", "value": 40}),
        ))
        self.page.planner_var.set(GEMINI_LABEL)
        with patch.object(self.page, "_refresh_models_async"):
            self.page._planner_changed(GEMINI_LABEL)
        with (
            patch("studio.antenna_builder_ui.GeminiSchemaConstrainedPlanner", return_value=planner) as factory,
            patch("studio.antenna_builder_ui.threading.Thread", _ImmediateThread),
        ):
            self.page.instruction_var.set("Change the patch width to 40 mm")
            self.page.apply_instruction()
            self._drain_ui_until(lambda: self.page.state.patch_width_mm == 40)

        factory.assert_called_once_with(model="gemini-3.8-flash")
        self.assertEqual(self.page.state.patch_width_mm, 40)
        self.assertIn("sent to Google Gemini", self.page.planner_notice_var.get())
        self.assertTrue((self.project.path / "design" / "planner_ab.jsonl").exists())
        reopened = self.store.open_project(self.project.path, touch=False)
        self.assertEqual(reopened.manifest["antenna_builder"]["planner_backend"], "gemini_cloud")
        self.assertEqual(reopened.manifest["antenna_builder"]["planner_model"], "gemini-3.8-flash")

    def test_planner_selector_marks_cloud_context_and_uses_groq_backend(self):
        self.app.design_start_page.choose_template()
        planner = _PlannerReturning(_tool_plan(
            ("recipe.select", {"recipe_id": "inset_patch_v2"}),
            ("parameter.set", {"key": "patch_width_mm", "value": 40}),
        ))
        self.page.planner_var.set(GROQ_LABEL)
        with patch.object(self.page, "_refresh_models_async"):
            self.page._planner_changed(GROQ_LABEL)
        with (
            patch("studio.antenna_builder_ui.GroqSchemaConstrainedPlanner", return_value=planner) as factory,
            patch("studio.antenna_builder_ui.threading.Thread", _ImmediateThread),
        ):
            self.page.instruction_var.set("Change the patch width to 40 mm")
            self.page.apply_instruction()
            self._drain_ui_until(lambda: self.page.state.patch_width_mm == 40)

        factory.assert_called_once_with(model="openai/gpt-oss-120b")
        self.assertEqual(self.page.state.patch_width_mm, 40)
        self.assertIn("sent to Groq", self.page.planner_notice_var.get())
        reopened = self.store.open_project(self.project.path, touch=False)
        self.assertEqual(
            reopened.manifest["antenna_builder"]["planner_backend"],
            "groq_cloud",
        )

    def test_planner_selector_marks_cloud_context_and_uses_openrouter_nemotron(self):
        self.app.design_start_page.choose_template()
        planner = _PlannerReturning(_tool_plan(
            ("recipe.select", {"recipe_id": "inset_patch_v2"}),
            ("parameter.set", {"key": "patch_width_mm", "value": 40}),
        ))
        self.page.planner_var.set(OPENROUTER_NEMOTRON_LABEL)
        with patch.object(self.page, "_refresh_models_async"):
            self.page._planner_changed(OPENROUTER_NEMOTRON_LABEL)
        with (
            patch("studio.antenna_builder_ui.OpenRouterNemotronPlanner", return_value=planner) as factory,
            patch("studio.antenna_builder_ui.threading.Thread", _ImmediateThread),
        ):
            self.page.instruction_var.set("Change the patch width to 40 mm")
            self.page.apply_instruction()
            self._drain_ui_until(lambda: self.page.state.patch_width_mm == 40)

        factory.assert_called_once_with(model="nvidia/nemotron-3-ultra-550b-a55b:free")
        self.assertEqual(self.page.state.patch_width_mm, 40)
        self.assertIn("sent to OpenRouter", self.page.planner_notice_var.get())
        reopened = self.store.open_project(self.project.path, touch=False)
        self.assertEqual(
            reopened.manifest["antenna_builder"]["planner_backend"],
            "openrouter_cloud",
        )

    def test_provider_discovery_labels_tested_and_untested_models_and_persists_choice(self):
        models = (
            PlannerModel(GEMINI, "gemini-3.8-flash", "Gemini 3.8 Flash", True),
            PlannerModel(GEMINI, "gemini-new-preview", "Gemini New Preview", False),
        )
        self.page.planner_var.set(GEMINI_LABEL)
        with (
            patch.object(self.page.model_discovery, "discover", return_value=models),
            patch("studio.antenna_builder_ui.threading.Thread", _ImmediateThread),
        ):
            self.page._planner_changed(GEMINI_LABEL)
            self._drain_ui_until(lambda: len(self.page.model_menu.cget("values")) == 2)

        values = list(self.page.model_menu.cget("values"))
        self.assertTrue(any("Tested" in value and "3.8" in value for value in values))
        untested = next(value for value in values if "Untested" in value)
        self.page.model_var.set(untested)
        self.page._model_changed(untested)
        reopened = self.store.open_project(self.project.path, touch=False)
        settings = reopened.manifest["antenna_builder"]
        self.assertEqual(settings["planner_provider"], GEMINI)
        self.assertEqual(settings["planner_model"], "gemini-new-preview")

    def test_discovery_failure_keeps_last_valid_model(self):
        retained = self.page._selected_model_id()
        with (
            patch.object(self.page.model_discovery, "discover", side_effect=ModelDiscoveryError("offline")),
            patch("studio.antenna_builder_ui.threading.Thread", _ImmediateThread),
        ):
            self.page._refresh_models_async()
            self._drain_ui_until(lambda: "failed" in self.page.planner_notice_var.get().lower())

        self.assertEqual(self.page._selected_model_id(), retained)
        self.assertIn("retained", self.page.planner_notice_var.get())

    def test_instruction_updates_parameters_preview_and_persistence(self):
        self.app.design_start_page.choose_template()
        self._apply_llm_plan(
            "Turn it into a 1x4 linear array with 0.55 lambda spacing",
            ("recipe.select", {"recipe_id": "inset_patch_v2"}),
            ("parameter.set", {"key": "array_rows", "value": 1}),
            ("parameter.set", {"key": "array_columns", "value": 4}),
            ("parameter.set", {"key": "element_spacing_lambda", "value": 0.55}),
        )

        self.assertEqual(self.page.state.array_columns, 4)
        self.assertEqual(self.page.parameter_vars["array_columns"].get(), "4")
        self.assertEqual(len(self.page.preview.scene.element_centers), 4)
        self.assertIn("1 × 4", self.page.preview_summary.cget("text") if "1 × 4" in self.page.preview_summary.cget("text") else self.page.state.topology_label)
        state_path = self.project.path / "design" / "antenna_state.json"
        self.assertTrue(state_path.exists())

        reopened = self.store.open_project(self.project.path, touch=False)
        self.app.set_project(reopened, target_page="antenna_builder")
        self.assertEqual(self.app.antenna_builder_page.state.array_columns, 4)

    def test_multiline_instruction_composer_expands_and_ctrl_enter_applies(self):
        self.app.design_start_page.choose_template()
        instruction = (
            "Create a circular patch at 5.8 GHz on Rogers RT5880.\n"
            "Turn it into a 1x4 linear array with 0.6 lambda spacing."
        )

        self.page.instruction_var.set(instruction)
        self.app.update_idletasks()

        self.assertEqual(
            self.page.instruction_entry.get("1.0", "end-1c"),
            instruction,
        )
        height = int(float(self.page.instruction_entry.cget("height")))
        self.assertGreater(height, INSTRUCTION_COMPOSER_MIN_HEIGHT)
        self.assertLessEqual(height, INSTRUCTION_COMPOSER_MAX_HEIGHT)
        self.assertRegex(
            self.page.instruction_size_label.cget("text"),
            r"· [2-9]\d* lines$",
        )

        planner = _PlannerReturning(_tool_plan(
            ("recipe.select", {"recipe_id": "circular_patch_v1"}),
            ("parameter.set", {"key": "frequency_ghz", "value": 5.8}),
            ("parameter.set", {"key": "material", "value": "Rogers RT5880"}),
            ("parameter.set", {"key": "array_rows", "value": 1}),
            ("parameter.set", {"key": "array_columns", "value": 4}),
            ("parameter.set", {"key": "element_spacing_lambda", "value": 0.6}),
        ))
        with (
            patch("studio.antenna_builder_ui.SchemaConstrainedLLMPlanner", return_value=planner),
            patch("studio.antenna_builder_ui.threading.Thread", _ImmediateThread),
        ):
            self.assertEqual(self.page._apply_instruction_shortcut(), "break")
            self.app.update()

        self.assertEqual(self.page.state.family, "circular_patch")
        self.assertEqual(self.page.state.array.element_count, 4)
        self.assertEqual(self.page.instruction_entry.get("1.0", "end-1c"), "")

    def test_parameter_edit_redraws_same_deterministic_scene(self):
        self._create_inset_design()
        self.page.parameter_vars["patch_width_mm"].set("38")
        self.page._apply_parameter_fields()
        self.app.update()

        self.assertEqual(self.page.state.patch_width_mm, 38.0)
        preview_patch = next(
            solid
            for solid in self.page.preview.scene.solids
            if solid.name.endswith("_patch")
        )
        self.assertEqual(preview_patch.bounds[1] - preview_patch.bounds[0], 38.0)
        self.assertIn("geometry updated", self.page.status_var.get())

    def test_substrate_table_edit_rederives_feed_and_reports_only_changed_field(self):
        self._create_inset_design()
        self.page.parameter_vars["substrate_thickness_mm"].set("0.8")
        self.page._apply_parameter_fields()
        self.app.update()

        self.assertAlmostEqual(self.page.state.substrate_thickness_mm, 0.8)
        self.assertAlmostEqual(self.page.state.feed_width_mm, 1.5295, places=4)
        response = self.page.conversation[-1]["content"]
        self.assertIn("Substrate thickness to 0.8", response)
        self.assertNotIn("Patch length", response)
        self.assertNotIn("Feed width", response)

    def test_composed_parameters_appear_persist_reopen_and_disappear_on_delete(self):
        self._create_inset_design()
        self._apply_llm_plan(
            "Add a 3 mm circular slot at the patch center.",
            ("parameter.create", {
                "key": "slot_radius_mm", "label": "Slot radius", "value": 3.0,
                "unit": "mm", "sweepable": True,
            }),
            ("geometry.cylinder", {
                "object_id": "center_slot_tool", "material_id": "copper", "axis": "z",
                "tags": ["planner_created", "boolean_tool", "slot"],
                "dimensions": {
                    "center_1": 0, "center_2": 0, "radius": "slot_radius_mm",
                    "start": "substrate_thickness_mm",
                    "end": "substrate_thickness_mm+copper_thickness_mm",
                },
            }),
            ("boolean.subtract", {
                "operation_id": "center_slot_subtract",
                "target_id": "element_1_1_patch",
                "tool_ids": ["center_slot_tool"],
            }),
        )
        group_id = self.page.state.composed_operations[0].group_id
        self.assertEqual(self.page.parameter_vars["slot_radius_mm"].get(), "3")
        self.assertIn("SlotRadius", self.page.sweep_vars)
        self.page.parameter_vars["slot_radius_mm"].set("4")
        self.page._apply_parameter_fields()
        self.assertEqual(self.page.state.value("slot_radius_mm"), 4.0)

        reopened = self.store.open_project(self.project.path, touch=False)
        self.app.set_project(reopened, target_page="antenna_builder")
        self.assertIn("slot_radius_mm", self.page.parameter_vars)
        self.assertEqual(self.page.parameter_vars["slot_radius_mm"].get(), "4")

        self._apply_llm_plan(
            "Remove the circular slot.",
            ("composition.delete", {"group_id": group_id}),
        )
        self.assertNotIn("slot_radius_mm", self.page.parameter_vars)
        self.assertNotIn("SlotRadius", self.page.sweep_vars)

        reopened = self.store.open_project(self.project.path, touch=False)
        self.app.set_project(reopened, target_page="antenna_builder")
        self.assertNotIn("slot_radius_mm", self.page.parameter_vars)

    def test_vary_selection_persists_across_project_reopen(self):
        self._create_inset_design()
        selected = {"PatchL", "PatchW", "Inset", "FeedW"}
        for name, variable in self.page.sweep_vars.items():
            variable.set(name in selected)
        self.page._sweep_selection_changed()

        reopened = self.store.open_project(self.project.path, touch=False)
        self.app.set_project(reopened, target_page="antenna_builder")

        self.assertEqual(
            {name for name, variable in self.page.sweep_vars.items() if variable.get()},
            selected,
        )

    def test_corner_cutout_prompt_adds_live_parameter_and_curved_preview(self):
        self._create_inset_design()
        self._apply_llm_plan(
            "Clear this and make a rectangular patch with circular fractals at each corner.\n"
            "Center each circle on the rectangular corner. Use radius = 1/4 of patch width.",
            ("design.reset", {}),
            ("modifier.apply", {"modifier_id": "corner_circle_cutouts_v1"}),
            ("parameter.set", {"key": "corner_radius_ratio", "value": 0.25}),
        )

        self.assertEqual(
            self.page.state.metadata_map()["active_modifiers"],
            "corner_circle_cutouts_v1",
        )
        self.assertEqual(self.page.parameter_vars["corner_radius_ratio"].get(), "0.25")
        self.assertTrue(self.page.sweep_vars["CornerRadiusRatio"].get())
        main_patch = next(
            solid
            for solid in self.page.preview.scene.solids
            if solid.name == "element_1_1_patch"
        )
        self.assertGreater(len(main_patch.vertices), 8)
        self.assertIn("centers sit on patch corners", self.page.feed_scope_label.cget("text"))

    def test_circular_patch_and_dipole_rebuild_dynamic_controls_and_preview(self):
        self.app.design_start_page.choose_template()

        self._apply_llm_plan(
            "Create a circular patch at 5.8 GHz on Rogers RT5880",
            ("recipe.select", {"recipe_id": "circular_patch_v1"}),
            ("parameter.set", {"key": "frequency_ghz", "value": 5.8}),
            ("parameter.set", {"key": "material", "value": "Rogers RT5880"}),
        )

        self.assertEqual(self.page.state.family, "circular_patch")
        original_id = self.page.state.design_id
        self.assertEqual(self.page.state.design_id, original_id)
        self.assertIn("patch_radius_mm", self.page.parameter_vars)
        self.assertNotIn("patch_length_mm", self.page.parameter_vars)
        self.assertEqual(
            len([solid for solid in self.page.preview.scene.solids if "circular_patch_element" in solid.tags]),
            1,
        )
        self.assertIn("PatchRadius", self.page.sweep_vars)

        self._apply_llm_plan(
            "Create a simple dipole at 915 MHz",
            ("recipe.select", {"recipe_id": "dipole_v1"}),
            ("parameter.set", {"key": "frequency_ghz", "value": 0.915}),
        )

        self.assertEqual(self.page.state.family, "dipole")
        self.assertEqual(self.page.state.design_id, original_id)
        self.assertIn("arm_length_mm", self.page.parameter_vars)
        self.assertNotIn("patch_radius_mm", self.page.parameter_vars)
        self.assertEqual(self.page.material_menu.cget("state"), "disabled")
        self.assertEqual(
            len([solid for solid in self.page.preview.scene.solids if "dipole_arm" in solid.tags]),
            2,
        )
        self.assertEqual(self.page.preview.mesh_actor_count, 2)

    def test_export_writes_one_parameterized_macro_and_design_record(self):
        self._create_inset_design()
        destination = self.project.path / "design" / "My_Patch.bas"
        with (
            patch(
                "studio.antenna_builder_ui.filedialog.asksaveasfilename",
                return_value=str(destination),
            ),
            patch("studio.antenna_builder_ui.messagebox.showinfo"),
        ):
            self.page.export_cst()

        self.assertTrue(destination.exists())
        self.assertTrue(destination.with_name("My_Patch_design.json").exists())
        script = destination.read_text(encoding="utf-8")
        self.assertIn('StoreParameter "PatchL"', script)
        self.assertNotIn("Solver.Start", script)
        reopened = self.store.open_project(self.project.path, touch=False)
        self.assertTrue(reopened.manifest["antenna_builder"]["accepted"])

    def test_selected_parameters_transfer_into_existing_lhs_dialog(self):
        self._create_inset_design()
        for name, selected in self.page.sweep_vars.items():
            selected.set(name in {"PatchL", "PatchW", "Inset"})

        self.page.send_to_lhs()
        self.app.update()

        self.assertEqual(self.app.active_page, "data")
        dialog = self.app.data_page.sample_generator_dialog
        self.assertIsNotNone(dialog)
        self.assertEqual(
            [editor.name.get() for editor in dialog.variable_editors],
            ["PatchL", "PatchW", "Inset"],
        )
        self.assertTrue(all(editor.minimum.get() for editor in dialog.variable_editors))
        self.assertTrue(all(editor.maximum.get() for editor in dialog.variable_editors))
        self.assertIn("Template parameters loaded", dialog.status_var.get())
        self.assertEqual(
            self.app.current_project.manifest["antenna_builder"][
                "selected_sweep_parameters"
            ],
            ["PatchL", "PatchW", "Inset"],
        )

    def test_native_cst_creation_runs_off_ui_thread_and_preserves_macro(self):
        self._create_inset_design()
        destination = self.project.path / "design" / "Native_Patch.cst"
        with (
            patch(
                "studio.antenna_builder_ui.filedialog.asksaveasfilename",
                return_value=str(destination),
            ),
            patch(
                "studio.antenna_builder_ui.create_native_cst_project",
                return_value=destination,
            ) as create,
            patch("studio.antenna_builder_ui.threading.Thread", _ImmediateThread),
            patch("studio.antenna_builder_ui.messagebox.showinfo"),
        ):
            self.page.create_native_cst()
            self.app.update()

        create.assert_called_once_with(destination, self.page.state)
        self.assertTrue(destination.with_suffix(".bas").exists())
        self.assertTrue(destination.with_name("Native_Patch_design.json").exists())
        self.assertEqual(self.page.native_cst_button.cget("state"), "normal")
        self.assertIn("Native CST project created", self.page.status_var.get())

    def test_every_text_request_uses_schema_constrained_llm_tool_planner(self):
        self._create_inset_design()
        original_id = self.page.state.design_id
        planner = self._apply_llm_plan(
            "Please use four elements along one line",
            ("parameter.set", {"key": "array_rows", "value": 1}),
            ("parameter.set", {"key": "array_columns", "value": 4}),
        )

        self.assertEqual(len(planner.requests), 1)
        request = planner.requests[0]
        self.assertEqual(request["current_design"]["design_id"], original_id)
        self.assertEqual(
            [item["name"] for item in request["capability_manifest"]["callable_tools"]],
            [
                "design.reset", "recipe.select", "parameter.set", "excitation.set_strategy",
                "modifier.apply", "modifier.remove",
                "parameter.create", "geometry.rectangle_sheet", "geometry.cylinder",
                "geometry.circle_sheet", "geometry.translate", "geometry.rotate", "geometry.duplicate",
                "boolean.subtract", "boolean.union", "engineering.design_summary", "engineering.array_spacing",
                "engineering.rectangular_patch_baseline", "engineering.dipole_baseline",
            ],
        )
        self.assertEqual(self.page.state.array_columns, 4)
        self.assertEqual(self.page.apply_button.cget("state"), "normal")
        self.assertIn("Array columns to 4", self.page.status_var.get())

    def test_preview_rotation_zoom_and_laptop_footer_remain_reachable(self):
        self._create_inset_design()
        self.app.geometry("1366x768+0+0")
        self.app.deiconify()
        self.app.set_sidebar_collapsed(True)
        self.app.set_snowbuddy_collapsed(True)
        self.page.instruction_var.set(
            "Create an inset-fed rectangular patch at 2.45 GHz on FR4.\n"
            "Make the substrate 1.6 mm thick.\n"
            "Change the patch width to 38 mm.\n"
            "Turn it into a 1x4 array with 0.55 lambda spacing."
        )
        self.app.update()
        preview = self.page.preview
        self.assertTrue(preview.available)
        camera = preview._renderer.GetActiveCamera()
        original_position = camera.GetPosition()
        original_focal = camera.GetFocalPoint()
        original_scale = camera.GetParallelScale()

        preview._start_orbit(SimpleNamespace(x=200, y=180))
        preview._orbit(SimpleNamespace(x=260, y=420))
        preview._start_pan(SimpleNamespace(x=300, y=200))
        preview._pan(SimpleNamespace(x=325, y=230))
        preview._zoom(1.1)
        self.app.update()

        self.assertNotEqual(camera.GetPosition(), original_position)
        self.assertNotEqual(camera.GetFocalPoint(), original_focal)
        self.assertLess(camera.GetParallelScale(), original_scale)
        self.assertEqual(preview.mesh_actor_count, len(preview.scene.solids))
        self.assertEqual(
            preview.vtk_mesh_counts,
            tuple(
                (len(solid.vertices), len(solid.faces))
                for solid in preview.scene.solids
            ),
        )
        self.assertGreaterEqual(self.page.editor_panel.winfo_width(), 420)
        self.assertGreaterEqual(self.page.preview_panel.winfo_width(), 660)
        self.assertIs(self.page.parameter_panel.master, self.page.preview_panel)
        self.assertGreaterEqual(self.page.parameter_table.winfo_height(), 190)
        self.assertIn("Operating frequency is fixed", self.page.sampling_policy_label.cget("text"))
        self.assertFalse(self.page.sweep_vars["FreqGHz"].get())
        app_right = self.app.winfo_rootx() + self.app.winfo_width()
        app_bottom = self.app.winfo_rooty() + self.app.winfo_height()
        for button in (
            self.page.export_button,
            self.page.native_cst_button,
            self.page.lhs_button,
        ):
            self.assertLessEqual(button.winfo_rootx() + button.winfo_width(), app_right)
            self.assertLessEqual(button.winfo_rooty() + button.winfo_height(), app_bottom)

    def test_preview_renders_physical_pixels_but_displays_logical_size(self):
        self._create_inset_design()
        self.app.geometry("1366x768+0+0")
        self.app.deiconify()
        self.app.set_sidebar_collapsed(True)
        self.app.set_snowbuddy_collapsed(True)
        self.app.update()
        preview = self.page.preview
        logical_width = max(preview._canvas.winfo_width(), 320)
        logical_height = max(preview._canvas.winfo_height(), 260)

        with patch("studio.antenna_vtk_preview._window_dpi_scale", return_value=1.5):
            preview.render()

        self.assertEqual(
            preview._render_window.GetSize(),
            (round(logical_width * 1.5), round(logical_height * 1.5)),
        )
        self.assertEqual(
            (preview._photo.width(), preview._photo.height()),
            (logical_width, logical_height),
        )
        self.app.withdraw()


if __name__ == "__main__":
    unittest.main()
