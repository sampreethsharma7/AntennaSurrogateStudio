"""Design-start choice and experimental parametric antenna-builder UI."""

from __future__ import annotations

import math
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import TYPE_CHECKING, Callable

import customtkinter as ctk

from studio.antenna_builder import (
    PLANNER_AUDIT_RELATIVE_PATH,
    MATERIALS,
    AntennaBuilderError,
    AntennaState,
    BuilderTurnCancelled,
    BuilderProjectSession,
    StateUpdate,
    build_geometry_scene,
    create_native_cst_project,
    execute_builder_turn,
    evaluate_project_constraints,
    lhs_variables_for_state,
    load_builder_session,
    publish_builder_state_update,
    recipe_parameter_definitions,
    save_cst_package,
)
from studio.antenna_agent import create_default_agent
from studio.antenna_vtk_preview import VtkAntennaPreview
from studio.antenna_llm_planner import (
    GeminiSchemaConstrainedPlanner,
    GroqSchemaConstrainedPlanner,
    OpenRouterNemotronPlanner,
    SchemaConstrainedLLMPlanner,
)
from studio.planner_model_discovery import (
    DEFAULT_MODELS,
    GEMINI,
    GROQ,
    LABEL_TO_PROVIDER,
    LOCAL_OLLAMA,
    OPENROUTER,
    PROVIDER_LABELS,
    ModelDiscoveryError,
    PlannerModel,
    PlannerModelDiscovery,
)
from studio.planner_credentials import (
    CLOUD_PROVIDERS,
    RECOMMENDED_CLOUD_PROVIDER,
    has_credential,
)
from studio.planner_onboarding_ui import run_setup_dialog
from studio.project_store import Project
from studio.theme import COLORS, FONTS

if TYPE_CHECKING:
    from studio.ui import StudioApp


INSTRUCTION_COMPOSER_MIN_HEIGHT = 92
INSTRUCTION_COMPOSER_MAX_HEIGHT = 154
LOCAL_QWEN_LABEL = PROVIDER_LABELS[LOCAL_OLLAMA]
GEMINI_LABEL = PROVIDER_LABELS[GEMINI]
GROQ_LABEL = PROVIDER_LABELS[GROQ]
OPENROUTER_NEMOTRON_LABEL = PROVIDER_LABELS[OPENROUTER]
STARTER_EXAMPLES = (
    "Create an inset-fed rectangular patch at 2.45 GHz on FR4.",
    "Create a probe-fed circular patch at 5.8 GHz on Rogers RT5880.",
    "Create a center-fed dipole at 915 MHz.",
    "Create a patch at 2.45 GHz as a 1×4 array with 0.55 lambda spacing.",
)
TRANSCRIPT_BODY_FONT = (FONTS["body_small"][0], round(FONTS["body_small"][1] * 1.30))
TRANSCRIPT_CAPTION_FONT = (FONTS["caption"][0], round(FONTS["caption"][1] * 1.30))
TRANSCRIPT_ROLE_FONT = (FONTS["mono"][0], round(FONTS["mono"][1] * 1.30))


def _active_color(value: str | tuple[str, str]) -> str:
    if isinstance(value, str):
        return value
    return value[1] if ctk.get_appearance_mode().lower() == "dark" else value[0]


def cloud_onboarding_required(
    project: Project | None,
    *,
    env_file: str | None = None,
    credential_backend: object | None = None,
) -> bool:
    """Whether credential setup should run before the builder opens.

    Three cases must not be interrupted, which is what the two early exits
    cover:

    * a project that already records a planner choice, including a deliberate
      Local Ollama one, because reopening an existing design is not a first
      run;
    * a machine that already resolves a cloud key from the environment, a
      `.env` file or the credential store, which is every developer and every
      user who set one up before this dialog existed.

    Everything else is a genuine first run with no way to reach a cloud
    planner, where proceeding would fail later inside the planner with a
    message about an environment variable.
    """

    if project is None:
        return False
    settings = project.manifest.get("antenna_builder")
    if isinstance(settings, dict) and (
        settings.get("planner_provider") or settings.get("planner_backend")
    ):
        return False
    return not any(
        has_credential(provider, env_file=env_file, backend=credential_backend)
        for provider in CLOUD_PROVIDERS
    )


class DesignStartPage(ctk.CTkFrame):
    """Ask whether a new project starts from existing data or a template."""

    def __init__(self, parent: ctk.CTkFrame, app: "StudioApp") -> None:
        super().__init__(parent, fg_color=COLORS["app_bg"], corner_radius=0)
        self.app = app
        self.project: Project | None = None
        # Injected by tests so the onboarding path can be exercised without a
        # real credential store, a browser, network access, or a modal dialog
        # that nothing would ever close.
        self.setup_discovery: PlannerModelDiscovery | None = None
        self.credential_backend: object | None = None
        self.setup_browser: Callable[[str], object] | None = None
        self.setup_dialog_runner: Callable[..., object] = run_setup_dialog
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_header()
        self._build_choices()

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, padx=34, pady=(26, 10), sticky="ew")
        ctk.CTkLabel(
            header,
            text="How do you want to start this design?",
            text_color=COLORS["ink"],
            font=FONTS["display"],
            anchor="w",
        ).pack(fill="x")
        ctk.CTkLabel(
            header,
            text="Choose the entry point that matches what you already have.",
            text_color=COLORS["muted"],
            font=FONTS["body"],
            anchor="w",
        ).pack(fill="x", pady=(5, 0))

    def _build_choices(self) -> None:
        host = ctk.CTkFrame(self, fg_color="transparent")
        host.grid(row=1, column=0, padx=34, pady=(8, 30), sticky="nsew")
        host.grid_columnconfigure((0, 1), weight=1, uniform="start")
        host.grid_rowconfigure(0, weight=1)
        self.existing_card = self._choice_card(
            host,
            column=0,
            eyebrow="EXISTING WORKFLOW",
            title="I already have a design",
            body=(
                "Continue with simulation inputs or exported CST/HFSS data. "
                "Nothing about the existing surrogate workflow changes."
            ),
            details=("Load paired CSVs", "Parse #Parameters exports", "Train from existing simulations"),
            action="Continue to Data Prep  →",
            command=self.choose_existing,
            accent=False,
        )
        self.generate_card = self._choice_card(
            host,
            column=1,
            eyebrow="EXPERIMENTAL",
            title="Design with antenna agent",
            body=(
                "Start with a blank workspace. Describe a supported antenna, then "
                "edit its parameters, inspect the live 3D geometry, and create CST output."
            ),
            details=("Conversational first design", "Three composable antenna recipes", "Solver-neutral state and CST adapter"),
            action="Open Experimental Builder  →",
            command=self.choose_template,
            accent=True,
        )

    def _choice_card(
        self,
        parent: ctk.CTkFrame,
        *,
        column: int,
        eyebrow: str,
        title: str,
        body: str,
        details: tuple[str, ...],
        action: str,
        command: Callable[[], None],
        accent: bool,
    ) -> ctk.CTkFrame:
        card = ctk.CTkFrame(
            parent,
            fg_color=COLORS["surface"],
            corner_radius=18,
            border_width=2 if accent else 1,
            border_color=COLORS["cyan"] if accent else COLORS["border"],
        )
        card.grid(
            row=0,
            column=column,
            padx=(0, 9) if column == 0 else (9, 0),
            pady=14,
            sticky="nsew",
        )
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(3, weight=1)
        ctk.CTkLabel(
            card,
            text=eyebrow,
            text_color=COLORS["cyan"] if accent else COLORS["muted"],
            font=FONTS["mono"],
            anchor="w",
        ).grid(row=0, column=0, padx=24, pady=(24, 8), sticky="ew")
        ctk.CTkLabel(
            card,
            text=title,
            text_color=COLORS["ink"],
            font=FONTS["section"],
            anchor="w",
        ).grid(row=1, column=0, padx=24, sticky="ew")
        ctk.CTkLabel(
            card,
            text=body,
            text_color=COLORS["muted"],
            font=FONTS["body"],
            justify="left",
            anchor="nw",
            wraplength=460,
        ).grid(row=2, column=0, padx=24, pady=(12, 18), sticky="ew")
        list_frame = ctk.CTkFrame(card, fg_color=COLORS["surface_alt"], corner_radius=12)
        list_frame.grid(row=3, column=0, padx=24, pady=(0, 22), sticky="new")
        for row, detail in enumerate(details):
            ctk.CTkLabel(
                list_frame,
                text=f"✓  {detail}",
                text_color=COLORS["ink"],
                font=FONTS["body_small"],
                anchor="w",
            ).grid(row=row, column=0, padx=15, pady=(10 if row == 0 else 4, 10 if row == len(details) - 1 else 4), sticky="ew")
        ctk.CTkButton(
            card,
            text=action,
            height=44,
            corner_radius=12,
            fg_color=COLORS["primary"] if accent else COLORS["surface_alt"],
            hover_color=COLORS["primary_hover"] if accent else COLORS["control_hover"],
            border_width=0 if accent else 1,
            border_color=COLORS["border"],
            text_color=COLORS["on_primary"] if accent else COLORS["ink"],
            font=FONTS["button"],
            command=command,
        ).grid(row=4, column=0, padx=24, pady=(0, 24), sticky="ew")
        return card

    def set_project(self, project: Project | None) -> None:
        self.project = project

    def choose_existing(self) -> None:
        if not self.project:
            return
        self.app.update_current_project(
            {"design_start": {"choice": "existing_design"}, "ui": {"last_page": "data"}}
        )
        self.app.show_page("data")

    def choose_template(self) -> None:
        if not self.project:
            return
        if not self.ensure_planner_credential():
            return
        self.app.update_current_project(
            {
                "design_start": {"choice": "generated_template"},
                "ui": {"last_page": "antenna_builder"},
            }
        )
        self.app.show_page("antenna_builder")

    def ensure_planner_credential(self) -> bool:
        """Settle how the planner will be reached before the builder opens.

        Returns False when the user closed setup without choosing anything, in
        which case this page stays put instead of handing them a workspace
        whose first request is going to fail.
        """

        if not cloud_onboarding_required(
            self.project, credential_backend=self.credential_backend
        ):
            return True
        outcome = self.setup_dialog_runner(
            self,
            provider=RECOMMENDED_CLOUD_PROVIDER,
            discovery=self.setup_discovery,
            credential_backend=self.credential_backend,
            browser=self.setup_browser,
        )
        if not outcome.completed:
            return False
        # Record the provider the user actually settled on, so reopening the
        # project never asks again. The key itself is not part of this.
        self.app.update_current_project(
            {
                "antenna_builder": {
                    "planner_provider": outcome.provider,
                    "planner_model": DEFAULT_MODELS[outcome.provider],
                }
            }
        )
        # The builder read its planner choice when the project was opened,
        # which was before any of this, so it has to be told.
        builder = getattr(self.app, "antenna_builder_page", None)
        if builder is not None:
            builder.adopt_planner_provider(outcome.provider)
        return True

    def describe_ui_state(self) -> list[str]:
        choice = (
            self.project.manifest.get("design_start", {}).get("choice")
            if self.project
            else None
        )
        return [
            f"Design-start choice: {choice or 'not selected'}",
            "Available paths: existing design; blank conversational antenna builder",
            "Installed experimental recipes: inset-fed rectangular patch, circular patch, and simple dipole",
        ]


class AntennaBuilderPage(ctk.CTkFrame):
    """Stateful constrained text editor with deterministic 3D preview and export."""

    def __init__(self, parent: ctk.CTkFrame, app: "StudioApp") -> None:
        super().__init__(parent, fg_color=COLORS["app_bg"], corner_radius=0)
        self.app = app
        self.agent = create_default_agent()
        self.project: Project | None = None
        self.session: BuilderProjectSession | None = None
        self.state: AntennaState | None = None
        self.conversation: list[dict[str, object]] = []
        self._session_load_error: str | None = None
        self.parameter_vars: dict[str, ctk.StringVar] = {}
        self.sweep_vars: dict[str, ctk.BooleanVar] = {}
        self._persisted_sweep_parameters: set[str] | None = None
        self._field_update_job: str | None = None
        self._active_plan_cancel: threading.Event | None = None
        self._syncing_fields = False
        self._syncing_instruction = False
        self.instruction_var = ctk.StringVar()
        self.provider_var = ctk.StringVar(value=LOCAL_QWEN_LABEL)
        # Kept as a compatibility alias for callers that used the former one-menu UI.
        self.planner_var = self.provider_var
        self.model_var = ctk.StringVar()
        self.openrouter_free_only_var = ctk.BooleanVar(value=True)
        self.model_discovery = PlannerModelDiscovery()
        # Injected by tests so the API-key dialog can be exercised without a
        # real credential store, a browser, or network access.
        self.credential_backend: object | None = None
        self.setup_browser: Callable[[str], object] | None = None
        self.setup_dialog_runner: Callable[..., object] = run_setup_dialog
        self._model_catalogs: dict[str, tuple[PlannerModel, ...]] = {}
        self._selected_model_ids = dict(DEFAULT_MODELS)
        self._set_model_choices(LOCAL_OLLAMA, self._fallback_models(LOCAL_OLLAMA))
        self.planner_notice_var = ctk.StringVar(
            value="Private/offline: design context stays on this computer."
        )
        self.material_var = ctk.StringVar(value="No design")
        self.status_var = ctk.StringVar(value="No antenna design yet. Describe the antenna you want to create.")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_header()
        self._build_workspace()
        self._build_footer()
        self.instruction_var.trace_add("write", self._instruction_variable_changed)
        self._sync_from_state()

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, padx=24, pady=(12, 8), sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="Parametric Antenna Builder",
            text_color=COLORS["ink"],
            font=FONTS["display"],
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header,
            text="EXPERIMENTAL · 3 VALIDATED RECIPES · CST ADAPTER",
            text_color=COLORS["warning"],
            font=FONTS["mono"],
            anchor="e",
        ).grid(row=0, column=1, padx=(16, 0), sticky="e")
        ctk.CTkLabel(
            header,
            text=(
                "Text edits one validated live state. Geometry and CST output are deterministic; "
                "electromagnetic performance still requires simulation."
            ),
            text_color=COLORS["muted"],
            font=FONTS["body_small"],
            anchor="w",
        ).grid(row=1, column=0, columnspan=2, pady=(3, 0), sticky="ew")

    def _build_workspace(self) -> None:
        split = tk.PanedWindow(
            self,
            orient=tk.HORIZONTAL,
            sashwidth=8,
            sashrelief=tk.FLAT,
            bg=_active_color(COLORS["border"]),
            bd=0,
        )
        split.grid(row=1, column=0, padx=24, pady=(0, 8), sticky="nsew")
        self.workspace_split = split
        editor = ctk.CTkFrame(split, fg_color=COLORS["surface"], corner_radius=14, border_width=1, border_color=COLORS["border"])
        preview = ctk.CTkFrame(split, fg_color=COLORS["surface"], corner_radius=14, border_width=1, border_color=COLORS["border"])
        self.editor_panel = editor
        self.preview_panel = preview
        split.add(editor, minsize=420, width=480, stretch="never")
        split.add(preview, minsize=660, stretch="always")
        self._build_editor(editor)
        self._build_preview(preview)

    def _build_editor(self, parent: ctk.CTkFrame) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(2, weight=1, minsize=160)
        ctk.CTkLabel(parent, text="Describe the design", text_color=COLORS["ink"], font=FONTS["card_title"], anchor="w").grid(row=0, column=0, padx=16, pady=(14, 5), sticky="ew")

        self.project_context_frame = ctk.CTkFrame(
            parent,
            fg_color=COLORS["surface_alt"],
            corner_radius=8,
            border_width=1,
            border_color=COLORS["border"],
        )
        self.project_context_frame.grid(row=1, column=0, padx=16, pady=(0, 8), sticky="ew")
        self.project_context_frame.grid_columnconfigure(0, weight=1)
        self.project_context_title = ctk.CTkLabel(
            self.project_context_frame,
            text="PROJECT CONTEXT",
            text_color=COLORS["subtle"],
            font=FONTS["mono"],
            anchor="w",
        )
        self.project_context_title.grid(row=0, column=0, padx=10, pady=(7, 1), sticky="ew")
        self.project_context_label = ctk.CTkLabel(
            self.project_context_frame,
            text="No active project memory.",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            justify="left",
            anchor="w",
            wraplength=390,
        )
        self.project_context_label.grid(row=1, column=0, padx=10, pady=(0, 7), sticky="ew")

        self.conversation_frame = ctk.CTkScrollableFrame(
            parent,
            fg_color=COLORS["surface_alt"],
            corner_radius=10,
            border_width=1,
            border_color=COLORS["border"],
        )
        self.conversation_frame.grid(row=2, column=0, padx=16, pady=(0, 8), sticky="nsew")
        self.conversation_frame.grid_columnconfigure(0, weight=1)
        self.conversation_frame.bind("<Configure>", self._conversation_inner_configured)
        conversation_canvas = getattr(self.conversation_frame, "_parent_canvas", None)
        if conversation_canvas is not None:
            conversation_canvas.bind("<Configure>", self._conversation_resized, add="+")
        self.transcript_rows: list[dict[str, object]] = []
        self.starter_example_buttons: list[ctk.CTkButton] = []
        self._conversation_text_labels: list[ctk.CTkLabel] = []
        self._conversation_scroll_pending = False

        instruction = ctk.CTkFrame(
            parent,
            fg_color=COLORS["surface_alt"],
            corner_radius=10,
            border_width=1,
            border_color=COLORS["border"],
        )
        instruction.grid(row=3, column=0, padx=16, pady=(0, 12), sticky="ew")
        instruction.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            instruction,
            text="DESIGN REQUEST",
            text_color=COLORS["subtle"],
            font=FONTS["mono"],
            anchor="w",
        ).grid(row=0, column=0, padx=(12, 6), pady=(9, 4), sticky="w")
        self.instruction_size_label = ctk.CTkLabel(
            instruction,
            text="0 chars · 1 line",
            text_color=COLORS["subtle"],
            font=FONTS["caption"],
            anchor="e",
        )
        self.instruction_size_label.grid(row=0, column=3, padx=(6, 12), pady=(9, 4), sticky="e")
        planner_row = ctk.CTkFrame(instruction, fg_color="transparent")
        planner_row.grid(row=1, column=0, columnspan=4, padx=10, pady=(1, 4), sticky="ew")
        planner_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(planner_row, text="Provider", text_color=COLORS["muted"], font=FONTS["caption"]).grid(row=0, column=0, padx=(2, 4))
        self.planner_menu = ctk.CTkOptionMenu(
            planner_row,
            variable=self.provider_var,
            values=list(PROVIDER_LABELS.values()),
            width=150,
            height=28,
            font=FONTS["caption"],
            command=self._planner_changed,
        )
        self.planner_menu.grid(row=0, column=1, padx=(0, 10), sticky="w")
        ctk.CTkLabel(planner_row, text="Model", text_color=COLORS["muted"], font=FONTS["caption"]).grid(row=1, column=0, padx=(2, 4), pady=(5, 0))
        self.model_menu = ctk.CTkOptionMenu(
            planner_row,
            variable=self.model_var,
            values=[self.model_var.get()],
            width=320,
            height=28,
            font=FONTS["caption"],
            command=self._model_changed,
        )
        self.model_menu.grid(row=1, column=1, columnspan=3, pady=(5, 0), sticky="ew")
        self.free_only_checkbox = ctk.CTkCheckBox(
            planner_row,
            text="Free only",
            variable=self.openrouter_free_only_var,
            width=82,
            font=FONTS["caption"],
            command=self._free_only_changed,
        )
        self.free_only_checkbox.grid(row=0, column=2, padx=(8, 4))
        self.refresh_models_button = ctk.CTkButton(
            planner_row,
            text="Refresh",
            width=70,
            height=28,
            font=FONTS["caption"],
            fg_color=COLORS["surface"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            command=self._refresh_models_async,
        )
        self.refresh_models_button.grid(row=0, column=3, padx=(4, 0))
        self.api_keys_button = ctk.CTkButton(
            planner_row,
            text="API keys",
            width=84,
            height=28,
            font=FONTS["caption"],
            fg_color=COLORS["surface"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            command=self.open_api_key_settings,
        )
        self.api_keys_button.grid(row=1, column=4, padx=(4, 0), pady=(5, 0))
        self.planner_notice_label = ctk.CTkLabel(
            instruction,
            textvariable=self.planner_notice_var,
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            anchor="w",
        )
        self.planner_notice_label.grid(row=2, column=0, columnspan=4, padx=12, pady=(0, 5), sticky="ew")
        self.instruction_entry = ctk.CTkTextbox(
            instruction,
            height=INSTRUCTION_COMPOSER_MIN_HEIGHT,
            corner_radius=8,
            border_width=1,
            border_color=COLORS["border"],
            fg_color=COLORS["surface"],
            text_color=COLORS["ink"],
            font=FONTS["body_small"],
            wrap="word",
            activate_scrollbars=True,
        )
        self.instruction_entry.grid(row=3, column=0, columnspan=4, padx=10, sticky="ew")
        self.instruction_entry.bind("<KeyRelease>", self._instruction_edited)
        self.instruction_entry.bind("<<Paste>>", lambda _event: self.after_idle(self._instruction_edited))
        self.instruction_entry.bind("<Configure>", lambda _event: self.after_idle(self._resize_instruction_composer))
        self.instruction_entry.bind("<Control-Return>", self._apply_instruction_shortcut)
        self.instruction_entry.bind("<Control-KP_Enter>", self._apply_instruction_shortcut)
        ctk.CTkLabel(
            instruction,
            text="Enter adds a new line · Ctrl+Enter applies",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            anchor="w",
        ).grid(row=4, column=0, columnspan=3, padx=(12, 8), pady=(7, 9), sticky="w")
        self.apply_button = ctk.CTkButton(
            instruction,
            text="Apply",
            width=112,
            height=36,
            corner_radius=10,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            font=FONTS["button"],
            command=self.apply_instruction,
        )
        self.apply_button.grid(row=4, column=3, padx=(8, 10), pady=(7, 9), sticky="e")

    def _build_parameter_table(self, parent: ctk.CTkFrame) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(2, weight=1)
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.grid(row=0, column=0, padx=(12, 24), pady=(2, 0), sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text="PARAMETER", text_color=COLORS["subtle"], font=FONTS["mono"], anchor="w").grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(header, text="VALUE", text_color=COLORS["subtle"], font=FONTS["mono"], width=128).grid(row=0, column=1)
        ctk.CTkLabel(header, text="VARY", text_color=COLORS["subtle"], font=FONTS["mono"], width=44).grid(row=0, column=2)
        self.sampling_policy_label = ctk.CTkLabel(
            parent,
            text=(
                "Operating frequency is fixed during LHS sampling because it also drives "
                "antenna dimensions, array spacing, and the solver range."
            ),
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            justify="left",
            anchor="w",
            wraplength=520,
        )
        self.sampling_policy_label.grid(row=1, column=0, padx=13, pady=(3, 4), sticky="ew")
        self.parameter_table = ctk.CTkScrollableFrame(parent, fg_color="transparent", corner_radius=0)
        self.parameter_table.grid(row=2, column=0, padx=8, pady=(0, 6), sticky="nsew")
        self._rebuild_parameter_table()

    def _use_starter_example(self, instruction: str) -> None:
        self.instruction_var.set(instruction)
        self.instruction_entry.focus_set()
        self.instruction_entry.mark_set("insert", "end-1c")

    def _render_empty_conversation(self) -> None:
        introduction = ctk.CTkLabel(
            self.conversation_frame,
            text="Start from one of these, or describe your own:",
            text_color=COLORS["ink"],
            font=TRANSCRIPT_BODY_FONT,
            anchor="w",
        )
        introduction.grid(row=0, column=0, padx=8, pady=(8, 6), sticky="ew")
        self._conversation_text_labels.append(introduction)
        self.starter_example_buttons = []
        labels = (
            "Inset patch · 2.45 GHz · FR4",
            "Circular patch · 5.8 GHz · RT5880",
            "Dipole · 915 MHz",
            "1×4 patch array · 2.45 GHz · 0.55λ",
        )
        for row, (label, instruction) in enumerate(zip(labels, STARTER_EXAMPLES), start=1):
            button = ctk.CTkButton(
                self.conversation_frame,
                text=label,
                height=40,
                fg_color=COLORS["surface"],
                hover_color=COLORS["control_hover"],
                border_width=1,
                border_color=COLORS["border"],
                text_color=COLORS["ink"],
                font=TRANSCRIPT_BODY_FONT,
                anchor="w",
                command=lambda value=instruction: self._use_starter_example(value),
            )
            button.grid(row=row, column=0, padx=8, pady=(0, 4), sticky="ew")
            self.starter_example_buttons.append(button)
        scope = ctk.CTkLabel(
            self.conversation_frame,
            text=(
                "Supported: 3 antenna families · arrays to 16×16 · circular and rectangular slots · circular corner notches\n"
                "Not supported: horns, Vivaldi, spirals, feed networks, solver runs"
            ),
            text_color=COLORS["muted"],
            font=TRANSCRIPT_CAPTION_FONT,
            justify="left",
            anchor="w",
            wraplength=self._conversation_wrap_width(),
        )
        scope.grid(row=len(labels) + 1, column=0, padx=8, pady=(6, 10), sticky="ew")
        self._conversation_text_labels.append(scope)

    @staticmethod
    def _transcript_style(message: dict[str, object]) -> tuple[str, object]:
        if message.get("role") == "user":
            return "you", COLORS["muted"]
        outcome = str(message.get("outcome") or "completed")
        if outcome in {"executed", "finished"}:
            return "✓", COLORS["success"]
        if outcome in {"clarification", "clarify"}:
            return "?", COLORS["warning"]
        if outcome in {"refusal", "refuse", "validation_rejection", "duplicate_rejection"}:
            return "✕", COLORS["danger"]
        return "·", COLORS["muted"]

    def _render_conversation(self) -> None:
        for child in self.conversation_frame.winfo_children():
            child.destroy()
        self.transcript_rows = []
        self.starter_example_buttons = []
        self._conversation_text_labels = []
        if not self.conversation:
            if self.state is None:
                self._render_empty_conversation()
            else:
                ready = ctk.CTkLabel(
                    self.conversation_frame,
                    text="The design is ready. Describe the next change.",
                    text_color=COLORS["muted"],
                    font=TRANSCRIPT_BODY_FONT,
                    anchor="w",
                )
                ready.grid(row=0, column=0, padx=8, pady=10, sticky="ew")
                self._conversation_text_labels.append(ready)
            self._conversation_scroll_pending = True
            self.after_idle(self._conversation_resized)
            self.after_idle(self._bind_conversation_wheel_tree)
            return

        for row, message in enumerate(self.conversation):
            marker, color = self._transcript_style(message)
            block = ctk.CTkFrame(self.conversation_frame, fg_color="transparent")
            block.grid(row=row, column=0, padx=6, pady=(4, 6), sticky="ew")
            block.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(
                block,
                text=marker,
                width=44,
                text_color=color,
                font=TRANSCRIPT_ROLE_FONT,
                anchor="nw",
            ).grid(row=0, column=0, padx=(0, 6), sticky="nw")
            message_label = ctk.CTkLabel(
                block,
                text=str(message.get("content", "")),
                text_color=COLORS["ink"],
                font=TRANSCRIPT_BODY_FONT,
                justify="left",
                anchor="nw",
                wraplength=self._conversation_wrap_width(),
            )
            message_label.grid(row=0, column=1, sticky="ew")
            self._conversation_text_labels.append(message_label)
            self.transcript_rows.append(
                {
                    "marker": marker,
                    "outcome": message.get("outcome"),
                    "frame": block,
                    "label": message_label,
                }
            )
        self._conversation_scroll_pending = True
        self.after_idle(self._conversation_resized)
        self.after_idle(self._bind_conversation_wheel_tree)

    def _conversation_wrap_width(self, width: int | None = None) -> int:
        available = width if width is not None else self.conversation_frame.winfo_width()
        return max(180, int(available) - 92)

    def _conversation_resized(self, event: object | None = None) -> None:
        width = getattr(event, "width", None)
        wraplength = self._conversation_wrap_width(width)
        for label in self._conversation_text_labels:
            if label.winfo_exists():
                label.configure(wraplength=wraplength)
        self.after_idle(self._refresh_conversation_scrollregion)

    def _conversation_inner_configured(self, event: object | None = None) -> None:
        self._conversation_resized(event)

    def _refresh_conversation_scrollregion(self) -> None:
        canvas = getattr(self.conversation_frame, "_parent_canvas", None)
        if canvas is None or not canvas.winfo_exists():
            return
        self.conversation_frame.update_idletasks()
        bounds = canvas.bbox("all")
        if bounds is not None:
            canvas.configure(scrollregion=bounds)
        if self._conversation_scroll_pending:
            self._conversation_scroll_pending = False
            self.after_idle(self._scroll_conversation_to_bottom)

    def _conversation_mousewheel(self, event: object) -> str:
        canvas = getattr(self.conversation_frame, "_parent_canvas", None)
        if canvas is None:
            return "break"
        number = getattr(event, "num", None)
        if number == 4:
            units = -1
        elif number == 5:
            units = 1
        else:
            delta = int(getattr(event, "delta", 0) or 0)
            units = -int(delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)
        canvas.yview_scroll(units, "units")
        return "break"

    def _bind_conversation_wheel_tree(self, widget: object | None = None) -> None:
        root = self.conversation_frame if widget is None else widget
        bind = getattr(root, "bind", None)
        if callable(bind):
            bind("<MouseWheel>", self._conversation_mousewheel, add="+")
            bind("<Button-4>", self._conversation_mousewheel, add="+")
            bind("<Button-5>", self._conversation_mousewheel, add="+")
        for child in getattr(root, "winfo_children", lambda: ())():
            self._bind_conversation_wheel_tree(child)

    def _render_project_context(self) -> None:
        if self.session is None:
            self.project_context_title.configure(text="PROJECT CONTEXT")
            self.project_context_label.configure(
                text="No project memory is loaded.",
                text_color=COLORS["muted"],
            )
            return
        memory = self.session.memory
        evaluations = evaluate_project_constraints(memory, self.state)
        lines: list[str] = []
        violations = [item for item in evaluations if item.status == "violating"]
        for item in evaluations:
            if item.status in {"satisfied", "violating"}:
                marker = "VIOLATING" if item.status == "violating" else "satisfied"
                lines.append(f"Constraint · {item.message} [{marker}]")
            elif item.status == "unevaluated":
                lines.append(f"Constraint · {item.label}: {item.limit} {item.unit or ''} [not evaluated]")
        active_intent = [
            item
            for collection in (memory.requirements, memory.decisions)
            for item in collection
            if item.source == "user_semantic"
            and item.status == "active"
            and item.semantic_kind != "constraint"
        ]
        if active_intent:
            rendered = "; ".join(
                f"{item.key.replace('_', ' ')}: {item.value}{(' ' + item.unit) if item.unit else ''}"
                for item in active_intent
            )
            lines.append(f"Intent · {rendered}")
        unsupported = [
            item for collection in (memory.requirements, memory.decisions)
            for item in collection if item.status == "requested_unsupported"
        ]
        if unsupported:
            lines.append(f"Unsupported requests retained for traceability: {len(unsupported)}")
        if memory.open_questions:
            lines.append(f"Open questions: {len([item for item in memory.open_questions if item.status == 'active'])}")
        if memory.limitations:
            lines.append(f"Recorded limitations: {len([item for item in memory.limitations if item.status == 'active'])}")
        self.project_context_title.configure(
            text="PROJECT CONTEXT · CONSTRAINT VIOLATION" if violations else "PROJECT CONTEXT",
            text_color=COLORS["danger"] if violations else COLORS["subtle"],
        )
        self.project_context_label.configure(
            text="\n".join(lines) if lines else "No active project memory.",
            text_color=COLORS["danger"] if violations else COLORS["muted"],
            wraplength=max(220, self.editor_panel.winfo_width() - 54),
        )

    def _scroll_conversation_to_bottom(self) -> None:
        canvas = getattr(self.conversation_frame, "_parent_canvas", None)
        if canvas is None:
            return
        self.conversation_frame.update_idletasks()
        bounds = canvas.bbox("all")
        if bounds is not None:
            canvas.configure(scrollregion=bounds)
        canvas.yview_moveto(1.0)

    def _rebuild_parameter_table(self) -> None:
        selected = {name for name, variable in self.sweep_vars.items() if variable.get()}
        if not self.sweep_vars and self._persisted_sweep_parameters is not None:
            selected = set(self._persisted_sweep_parameters)
        for child in self.parameter_table.winfo_children():
            child.destroy()
        self.parameter_vars.clear()
        self.sweep_vars.clear()
        table = self.parameter_table
        table.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            table,
            text="Substrate material",
            text_color=COLORS["ink"],
            font=FONTS["body_small"],
            anchor="w",
        ).grid(row=0, column=0, padx=(5, 6), pady=2, sticky="ew")
        material_values = list(MATERIALS) if self.state is not None else ["No design"]
        self.material_menu = ctk.CTkOptionMenu(
            table,
            variable=self.material_var,
            values=material_values,
            width=124,
            height=26,
            font=FONTS["body_small"],
            command=self._material_changed,
            state="normal" if self.state is not None else "disabled",
        )
        self.material_menu.grid(row=0, column=1, columnspan=2, pady=2, sticky="w")
        ctk.CTkLabel(
            table,
            text="—",
            width=24,
            text_color=COLORS["muted"],
            font=FONTS["caption"],
        ).grid(row=0, column=3, padx=(6, 2))
        if self.state is None:
            ctk.CTkLabel(
                table,
                text=(
                    "No antenna parameters yet.\n"
                    "Describe a supported antenna above to create the first validated design."
                ),
                text_color=COLORS["muted"],
                font=FONTS["body_small"],
                justify="left",
                anchor="nw",
            ).grid(row=1, column=0, columnspan=4, padx=8, pady=12, sticky="ew")
            return
        definitions = recipe_parameter_definitions(self.state)
        available = {item.name for item in definitions if item.sweepable}
        if self._persisted_sweep_parameters is None and not (selected & available):
            preferred = {
                "rectangular_inset_patch": {"PatchL", "PatchW", "Inset"},
                "circular_patch": {"PatchRadius", "FeedOffset"},
                "dipole": {"ArmLength", "WireRadius", "FeedGap"},
            }
            selected = preferred.get(self.state.family, set())
        if "corner_circle_cutouts_v1" in self.state.metadata_map().get("active_modifiers", "").split(","):
            selected = {*selected, "CornerRadiusRatio"}
        row = 1
        for definition in definitions:
            if definition.kind == "choice":
                continue
            ctk.CTkLabel(table, text=definition.label, text_color=COLORS["ink"], font=FONTS["body_small"], anchor="w").grid(row=row, column=0, padx=(5, 6), pady=2, sticky="ew")
            variable = ctk.StringVar()
            self.parameter_vars[definition.key] = variable
            ctk.CTkEntry(table, textvariable=variable, width=88, height=26, font=FONTS["body_small"]).grid(row=row, column=1, pady=2)
            ctk.CTkLabel(table, text=definition.unit, width=35, text_color=COLORS["muted"], font=FONTS["caption"], anchor="w").grid(row=row, column=2, padx=(5, 1), sticky="w")
            sweep = ctk.BooleanVar(value=definition.sweepable and definition.name in selected)
            self.sweep_vars[definition.name] = sweep
            ctk.CTkCheckBox(
                table,
                text="",
                variable=sweep,
                width=24,
                state="normal" if definition.sweepable else "disabled",
                command=self._sweep_selection_changed,
            ).grid(row=row, column=3, padx=(6, 2))
            variable.trace_add("write", lambda *_args: self._schedule_field_update())
            row += 1

    def _build_preview(self, parent: ctk.CTkFrame) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=3, minsize=240)
        parent.grid_rowconfigure(5, weight=2, minsize=200)
        toolbar = ctk.CTkFrame(parent, fg_color="transparent")
        toolbar.grid(row=0, column=0, padx=14, pady=(12, 7), sticky="ew")
        toolbar.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(toolbar, text="Live deterministic geometry", text_color=COLORS["ink"], font=FONTS["card_title"], anchor="w").grid(row=0, column=0, sticky="w")
        self.reset_view_button = ctk.CTkButton(
            toolbar,
            text="Reset view",
            width=106,
            height=32,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            command=lambda: self.preview.reset_view(),
        )
        self.reset_view_button.grid(row=0, column=1, sticky="e")
        self.preview = VtkAntennaPreview(
            parent,
            background=_active_color(COLORS["surface_alt"]),
            ink=_active_color(COLORS["ink"]),
            muted=_active_color(COLORS["muted"]),
            danger=_active_color(COLORS["danger"]),
        )
        self.preview.grid(row=1, column=0, padx=12, pady=(0, 8), sticky="nsew")
        self.empty_preview = ctk.CTkLabel(
            parent,
            text="No antenna design yet.\nDescribe the antenna you want to create.",
            text_color=COLORS["muted"],
            fg_color=COLORS["surface_alt"],
            font=FONTS["section"],
            justify="center",
            corner_radius=10,
        )
        self.preview_summary = ctk.CTkLabel(
            parent,
            text="",
            text_color=COLORS["muted"],
            font=FONTS["body_small"],
            anchor="w",
        )
        self.preview_summary.grid(row=2, column=0, padx=14, pady=(0, 4), sticky="ew")
        self.feed_scope_label = ctk.CTkLabel(
            parent,
            text="Each replicated element has an independent port. Inspect every excitation before simulation.",
            text_color=COLORS["warning"],
            font=FONTS["caption"],
            anchor="w",
        )
        self.feed_scope_label.grid(row=3, column=0, padx=14, pady=(0, 5), sticky="ew")
        ctk.CTkFrame(parent, height=1, fg_color=COLORS["border"], corner_radius=0).grid(
            row=4, column=0, padx=12, pady=(0, 3), sticky="ew"
        )
        self.parameter_panel = ctk.CTkFrame(parent, fg_color="transparent")
        self.parameter_panel.grid(row=5, column=0, padx=4, pady=(0, 4), sticky="nsew")
        self._build_parameter_table(self.parameter_panel)

    def _build_footer(self) -> None:
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, padx=24, pady=(0, 10), sticky="ew")
        footer.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(
            footer,
            text="←  Antenna Design",
            width=150,
            height=38,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            command=lambda: self.app.show_page("design_start"),
        ).grid(row=0, column=0, sticky="w")
        self.export_button = ctk.CTkButton(
            footer,
            text="Export CST script",
            width=162,
            height=40,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            command=self.export_cst,
        )
        self.export_button.grid(row=0, column=2, padx=(0, 8), sticky="e")
        self.native_cst_button = ctk.CTkButton(
            footer,
            text="Create CST project",
            width=174,
            height=40,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            command=self.create_native_cst,
        )
        self.native_cst_button.grid(row=0, column=3, padx=(0, 8), sticky="e")
        self.sampling_summary = ctk.CTkLabel(
            footer,
            text="0 selected",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            anchor="e",
        )
        self.sampling_summary.grid(row=0, column=4, padx=(16, 8), sticky="e")
        self.lhs_button = ctk.CTkButton(
            footer,
            text="Send selected to LHS  →",
            width=204,
            height=40,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            font=FONTS["button"],
            command=self.send_to_lhs,
        )
        self.lhs_button.grid(row=0, column=5, sticky="e")

    def set_project(self, project: Project | None) -> None:
        if self._active_plan_cancel is not None:
            self._active_plan_cancel.set()
            self._active_plan_cancel = None
        self.project = project
        self.session = None
        self.state = None
        self.conversation = []
        self._persisted_sweep_parameters = None
        self._session_load_error = None
        provider = LOCAL_OLLAMA
        model = DEFAULT_MODELS[provider]
        if project is not None:
            settings = project.manifest.get("antenna_builder", {})
            stored_sweep_parameters = settings.get("selected_sweep_parameters")
            if isinstance(stored_sweep_parameters, list) and all(
                isinstance(name, str) for name in stored_sweep_parameters
            ):
                self._persisted_sweep_parameters = set(stored_sweep_parameters)
            legacy_backend = str(settings.get("planner_backend", "local_qwen"))
            provider = str(settings.get("planner_provider") or {
                "local_qwen": LOCAL_OLLAMA,
                "gemini_cloud": GEMINI,
                "groq_gpt_oss_120b_cloud": GROQ,
                "groq_cloud": GROQ,
                "openrouter_nemotron_3_ultra_free_cloud": OPENROUTER,
                "openrouter_cloud": OPENROUTER,
            }.get(legacy_backend, LOCAL_OLLAMA))
            if provider not in PROVIDER_LABELS:
                provider = LOCAL_OLLAMA
            model = str(settings.get("planner_model") or DEFAULT_MODELS[provider])
            self.openrouter_free_only_var.set(bool(settings.get("openrouter_free_only", True)))
            try:
                self.session = load_builder_session(project.path)
            except (OSError, ValueError, AntennaBuilderError) as exc:
                self._session_load_error = str(exc)
                self.status_var.set(f"Saved builder session could not be restored: {exc}")
            else:
                self.state = self.session.design
                self.conversation = self.session.conversation
                if self.state is None:
                    self.status_var.set(
                        "No antenna design yet. Describe the antenna you want to create."
                    )
                else:
                    self.status_var.set(
                        f"Restored {self.state.topology_label} revision {self.state.revision}."
                    )
        else:
            self.status_var.set("Open a project to use the antenna builder.")
        self._selected_model_ids[provider] = model
        self.provider_var.set(PROVIDER_LABELS[provider])
        catalog = self._model_catalogs.get(provider) or self._fallback_models(provider)
        if model not in {item.model_id for item in catalog}:
            catalog = (*catalog, PlannerModel(provider, model, model, False, model.endswith(":free") if provider == OPENROUTER else None))
        self._set_model_choices(provider, tuple(catalog), preferred=model)
        self._update_planner_notice()
        self._sync_from_state()

    @staticmethod
    def _fallback_models(provider: str) -> tuple[PlannerModel, ...]:
        model_id = DEFAULT_MODELS[provider]
        return (PlannerModel(provider, model_id, model_id, True, model_id.endswith(":free") if provider == OPENROUTER else None),)

    def _provider_id(self) -> str:
        return LABEL_TO_PROVIDER.get(self.provider_var.get(), LOCAL_OLLAMA)

    def _set_model_choices(
        self,
        provider: str,
        models: tuple[PlannerModel, ...],
        *,
        preferred: str | None = None,
    ) -> None:
        catalog = tuple(models) or self._fallback_models(provider)
        self._model_catalogs[provider] = catalog
        selected_id = preferred or self._selected_model_ids.get(provider) or DEFAULT_MODELS[provider]
        by_id = {model.model_id: model for model in catalog}
        selected = by_id.get(selected_id)
        if selected is None:
            selected = next((model for model in catalog if model.tested), catalog[0])
        self._selected_model_ids[provider] = selected.model_id
        labels = [model.menu_label for model in catalog]
        self.model_var.set(selected.menu_label)
        if hasattr(self, "model_menu"):
            self.model_menu.configure(values=labels)

    def _selected_model_id(self) -> str:
        provider = self._provider_id()
        label = self.model_var.get()
        for model in self._model_catalogs.get(provider, ()):
            if model.menu_label == label:
                return model.model_id
        return self._selected_model_ids.get(provider, DEFAULT_MODELS[provider])

    def _update_planner_notice(self) -> None:
        provider = self._provider_id()
        if hasattr(self, "free_only_checkbox"):
            self.free_only_checkbox.configure(state="normal" if provider == OPENROUTER else "disabled")
        if provider == GEMINI:
            self.planner_notice_var.set(
                "Cloud provider: design state and request are sent to Google Gemini. Model labels show validation status."
            )
        elif provider == GROQ:
            self.planner_notice_var.set(
                "Cloud provider: design state and request are sent to Groq. Model labels show validation status."
            )
        elif provider == OPENROUTER:
            self.planner_notice_var.set(
                "Cloud provider: design state and request are sent to OpenRouter. Free-only uses catalog pricing metadata."
            )
        else:
            self.planner_notice_var.set(
                "Private/offline: models are discovered from local Ollama; design context stays on this computer."
            )

    def _planner_changed(self, _selection: str) -> None:
        provider = self._provider_id()
        self._set_model_choices(provider, self._model_catalogs.get(provider) or self._fallback_models(provider))
        self._update_planner_notice()
        self._persist_planner_choice()
        self._refresh_models_async()

    def _model_changed(self, selection: str) -> None:
        provider = self._provider_id()
        for model in self._model_catalogs.get(provider, ()):
            if model.menu_label == selection:
                self._selected_model_ids[provider] = model.model_id
                break
        self._persist_planner_choice()

    def _free_only_changed(self) -> None:
        if self._provider_id() == OPENROUTER:
            self._persist_planner_choice()
            self._refresh_models_async()

    def _refresh_models_async(self) -> None:
        provider = self._provider_id()
        retained = self._selected_model_id()
        free_only = bool(self.openrouter_free_only_var.get())
        self.refresh_models_button.configure(state="disabled", text="Loading…")
        self.planner_notice_var.set(f"Discovering models from {PROVIDER_LABELS[provider]}…")

        def worker() -> None:
            try:
                models = self.model_discovery.discover(
                    provider,
                    ollama_base_url=self.app.snowbuddy.base_url,
                    free_only=free_only,
                )
            except Exception as exc:
                self.after(0, lambda error=exc: self._model_discovery_failed(provider, retained, error))
                return
            self.after(0, lambda: self._model_discovery_complete(provider, retained, models))

        threading.Thread(target=worker, daemon=True).start()

    def _model_discovery_complete(
        self,
        provider: str,
        retained: str,
        models: tuple[PlannerModel, ...],
    ) -> None:
        if self._provider_id() != provider:
            return
        self._set_model_choices(provider, models, preferred=retained)
        self.refresh_models_button.configure(state="normal", text="Refresh")
        self._update_planner_notice()
        self.planner_notice_var.set(
            f"{len(models)} compatible model{'s' if len(models) != 1 else ''} discovered from {PROVIDER_LABELS[provider]}. "
            + self.planner_notice_var.get()
        )
        self._persist_planner_choice()

    def _model_discovery_failed(self, provider: str, retained: str, exc: Exception) -> None:
        if self._provider_id() != provider:
            return
        self._selected_model_ids[provider] = retained
        self._set_model_choices(provider, self._model_catalogs.get(provider) or self._fallback_models(provider), preferred=retained)
        self.refresh_models_button.configure(state="normal", text="Refresh")
        reason = str(exc) if isinstance(exc, ModelDiscoveryError) else "unexpected discovery error"
        self.planner_notice_var.set(
            f"Model discovery failed ({reason}); retained {retained}."
        )

    def open_api_key_settings(self) -> None:
        """Let the user replace or remove a stored cloud key at any time.

        Without this the first key entered during setup would be the only one
        the application ever used, with no way to rotate it from the GUI.
        """

        provider = self._provider_id()
        outcome = self.setup_dialog_runner(
            self,
            provider=provider if provider in CLOUD_PROVIDERS else RECOMMENDED_CLOUD_PROVIDER,
            manage=True,
            discovery=self.model_discovery,
            credential_backend=self.credential_backend,
            browser=self.setup_browser,
        )
        if not outcome.completed or outcome.provider == self._provider_id():
            self._update_planner_notice()
            return
        # A key was verified for a different provider; follow the user there
        # rather than leaving the menu pointing at the one they left behind.
        self.adopt_planner_provider(outcome.provider)

    def adopt_planner_provider(self, provider: str) -> None:
        """Point the selector at a provider that was settled somewhere else.

        ``set_project`` reads the stored choice, but it runs when the project
        is opened, which is before first-run setup has had a chance to happen.
        Without this the page keeps the default it was built with, and someone
        who has just verified a cloud key is handed a workspace pointed at the
        local planner they never chose.
        """

        if provider not in PROVIDER_LABELS or provider == self._provider_id():
            return
        self.provider_var.set(PROVIDER_LABELS[provider])
        self._planner_changed(PROVIDER_LABELS[provider])

    def _persist_planner_choice(self) -> None:
        if self.project is not None:
            self.app.update_current_project({"antenna_builder": self._planner_settings()})

    def _planner_settings(self) -> dict[str, object]:
        return {
            "planner_backend": self._planner_backend_id(),
            "planner_provider": self._provider_id(),
            "planner_model": self._selected_model_id(),
            "openrouter_free_only": bool(self.openrouter_free_only_var.get()),
        }

    def _planner_backend_id(self) -> str:
        return {
            GEMINI: "gemini_cloud",
            GROQ: "groq_cloud",
            OPENROUTER: "openrouter_cloud",
        }.get(self._provider_id(), "local_qwen")

    def _create_planner(self):
        provider = self._provider_id()
        model = self._selected_model_id()
        if provider == GEMINI:
            return GeminiSchemaConstrainedPlanner(model=model)
        if provider == GROQ:
            return GroqSchemaConstrainedPlanner(model=model)
        if provider == OPENROUTER:
            return OpenRouterNemotronPlanner(model=model)
        return SchemaConstrainedLLMPlanner(
            model,
            base_url=self.app.snowbuddy.base_url,
        )

    def refresh_theme(self) -> None:
        self.workspace_split.configure(bg=_active_color(COLORS["border"]))
        self.preview.refresh_theme(
            background=_active_color(COLORS["surface_alt"]),
            ink=_active_color(COLORS["ink"]),
            muted=_active_color(COLORS["muted"]),
            danger=_active_color(COLORS["danger"]),
        )

    def _instruction_variable_changed(self, *_args: object) -> None:
        if self._syncing_instruction or not hasattr(self, "instruction_entry"):
            return
        value = self.instruction_var.get()
        current = self.instruction_entry.get("1.0", "end-1c")
        if current != value:
            self._syncing_instruction = True
            try:
                self.instruction_entry.delete("1.0", "end")
                if value:
                    self.instruction_entry.insert("1.0", value)
            finally:
                self._syncing_instruction = False
        self.after_idle(self._resize_instruction_composer)

    def _instruction_edited(self, _event: object | None = None) -> None:
        if self._syncing_instruction:
            return
        value = self.instruction_entry.get("1.0", "end-1c")
        if self.instruction_var.get() != value:
            self._syncing_instruction = True
            try:
                self.instruction_var.set(value)
            finally:
                self._syncing_instruction = False
        self._resize_instruction_composer()

    def _resize_instruction_composer(self) -> None:
        if not hasattr(self, "instruction_entry"):
            return
        value = self.instruction_entry.get("1.0", "end-1c")
        width = max(self.instruction_entry.winfo_width(), 360)
        characters_per_line = max(28, int((width - 42) / 8.5))
        visual_lines = sum(
            max(1, math.ceil(max(len(line), 1) / characters_per_line))
            for line in (value.splitlines() or [""])
        )
        desired_height = min(
            INSTRUCTION_COMPOSER_MAX_HEIGHT,
            max(INSTRUCTION_COMPOSER_MIN_HEIGHT, 58 + visual_lines * 22),
        )
        if int(float(self.instruction_entry.cget("height"))) != desired_height:
            self.instruction_entry.configure(height=desired_height)
        suffix = "line" if visual_lines == 1 else "lines"
        self.instruction_size_label.configure(
            text=f"{len(value)} chars · {visual_lines} {suffix}"
        )

    def _apply_instruction_shortcut(self, _event: object | None = None) -> str:
        self._instruction_edited()
        self.apply_instruction()
        return "break"

    def _set_planning_state(self, active: bool) -> None:
        if active:
            self.instruction_entry.configure(state="disabled")
            self.apply_button.configure(state="normal", text="Cancel")
        else:
            self.instruction_entry.configure(state="normal")
            available = self.project is not None and self.session is not None and not self._session_load_error
            self.apply_button.configure(state="normal" if available else "disabled", text="Apply")

    def _cancel_active_turn(self) -> None:
        if self._active_plan_cancel is None:
            return
        self._active_plan_cancel.set()
        self.apply_button.configure(state="disabled", text="Cancelling…")
        self.status_var.set("Cancelling the current planning turn; no result will be published.")

    def _schedule_field_update(self) -> None:
        if self._syncing_fields:
            return
        if self._field_update_job is not None:
            try:
                self.after_cancel(self._field_update_job)
            except tk.TclError:
                pass
        self._field_update_job = self.after(350, self._apply_parameter_fields)

    def _material_changed(self, material: str) -> None:
        if self.state is None or self.session is None or self.project is None:
            return
        if self.state.material == "No substrate":
            return
        try:
            result = self.agent.update_parameters(
                self.state,
                {"material": material},
                intent="material edit",
            )
            turn = publish_builder_state_update(
                self.project.path,
                self.session,
                StateUpdate(result.design, result.changes, result.plan, result.executed_tools),
                description=f"Changed substrate material to {material} in the parameter table.",
                interaction_kind="parameter_table",
            )
        except AntennaBuilderError as exc:
            self.status_var.set(str(exc))
            return
        self._accept_published_session(
            turn.session,
            f"Material changed to {material}.",
        )

    def _apply_parameter_fields(self) -> None:
        if self._field_update_job is not None:
            try:
                self.after_cancel(self._field_update_job)
            except tk.TclError:
                pass
        self._field_update_job = None
        if self.state is None or self.session is None or self.project is None:
            return
        updates: dict[str, object] = {}
        try:
            current_values = self.agent.values_for_design(self.state)
            for definition in recipe_parameter_definitions(self.state):
                if definition.kind == "choice":
                    continue
                raw = self.parameter_vars[definition.key].get().strip()
                if definition.kind == "integer":
                    if not raw or not raw.isdigit():
                        raise AntennaBuilderError(f"{definition.label} must be a whole number.")
                    value: object = int(raw)
                else:
                    value = float(raw)
                current = current_values.get(definition.key)
                if (
                    isinstance(value, float)
                    and isinstance(current, (int, float))
                    and not isinstance(current, bool)
                ):
                    changed = not math.isclose(value, float(current), rel_tol=0.0, abs_tol=1e-12)
                else:
                    changed = value != current
                if changed:
                    updates[definition.key] = value
            if not updates:
                return
            result = self.agent.update_parameters(
                self.state,
                updates,
                intent="parameter table edit",
            )
            turn = publish_builder_state_update(
                self.project.path,
                self.session,
                StateUpdate(result.design, result.changes, result.plan, result.executed_tools),
                description="Edited validated antenna parameters in the parameter table.",
                interaction_kind="parameter_table",
            )
        except (ValueError, AntennaBuilderError) as exc:
            self.status_var.set(f"Parameter edit not applied: {exc}")
            return
        self._accept_published_session(
            turn.session,
            "Parameters validated; geometry updated.",
            sync_fields=False,
        )

    def apply_instruction(self) -> None:
        if self._active_plan_cancel is not None:
            self._cancel_active_turn()
            return
        self._instruction_edited()
        instruction = self.instruction_var.get().strip()
        if not instruction:
            self.status_var.set("Enter an antenna instruction first.")
            messagebox.showwarning("Instruction not applied", self.status_var.get(), parent=self)
            return
        if self.project is None or self.session is None:
            if self._session_load_error:
                message = "The saved builder session could not be loaded. Reopen the project before making changes."
            else:
                message = "Open a project before using the antenna builder."
            self.status_var.set(message)
            messagebox.showwarning("Builder unavailable", message, parent=self)
            return
        base_state = self.state
        base_session = self.session
        project_path = self.project.path
        backend_label = f"{self.provider_var.get()} / {self._selected_model_id()}"
        cancel_event = threading.Event()
        self._active_plan_cancel = cancel_event
        self._set_planning_state(True)
        self.status_var.set(
            f"Asking {backend_label} for a schema-constrained registered-tool plan..."
        )

        def worker() -> None:
            try:
                planner = self._create_planner()
                turn = execute_builder_turn(
                    project_path,
                    base_session,
                    instruction,
                    planner=planner,
                    cancel_requested=cancel_event.is_set,
                )
            except BuilderTurnCancelled:
                self.after(
                    0,
                    lambda planned_session=base_session, token=cancel_event: self._planning_cancelled(
                        planned_session,
                        token,
                    ),
                )
                return
            except Exception as exc:
                self.after(
                    0,
                    lambda error=exc, planned_session=base_session, token=cancel_event: self._llm_plan_failed(
                        error,
                        planned_session,
                        token,
                    ),
                )
                return
            self.after(
                0,
                lambda: self._llm_plan_complete(base_state, base_session, turn, cancel_event),
            )

        threading.Thread(target=worker, daemon=True).start()

    def _planning_cancelled(
        self,
        planned_session: BuilderProjectSession,
        token: threading.Event,
    ) -> None:
        if self._active_plan_cancel is not token:
            return
        self._active_plan_cancel = None
        self._set_planning_state(False)
        if self.session is planned_session:
            self.status_var.set("Planning cancelled. The design and conversation were not changed.")
            self.instruction_entry.focus_set()

    def _llm_plan_failed(
        self,
        exc: Exception,
        planned_session: BuilderProjectSession,
        token: threading.Event,
    ) -> None:
        if self._active_plan_cancel is not token:
            return
        self._active_plan_cancel = None
        self._set_planning_state(False)
        if self.session is not planned_session:
            self.status_var.set("The project changed while planning; the completed turn was not displayed here.")
            return
        self.state = planned_session.design
        self.conversation = planned_session.conversation
        self._render_state()
        self._update_builder_manifest()
        self.status_var.set(str(exc))
        messagebox.showerror("LLM planning fault", str(exc), parent=self)

    def _llm_plan_complete(
        self,
        base_state: AntennaState | None,
        planned_session: BuilderProjectSession,
        turn,
        token: threading.Event,
    ) -> None:
        if self._active_plan_cancel is not token:
            return
        self._active_plan_cancel = None
        self._set_planning_state(False)
        if self.session is not planned_session or self.state != base_state:
            self.status_var.set("The design changed while LLM planning was running; the stale plan was discarded.")
            return
        terminal = turn.terminal_result
        outcome = terminal.outcome if terminal is not None else "finished"
        if outcome == "finished":
            if terminal is None or terminal.has_publishable_change:
                self.instruction_var.set("")
            status = turn.update.summary
        elif outcome == "clarify":
            status = f"Clarification requested: {turn.update.message or terminal.message}"
        elif outcome == "refuse":
            status = f"Capability unavailable: {turn.update.message or terminal.message}"
        else:
            raise AntennaBuilderError(f"Unexpected terminal builder outcome: {outcome}")
        self._accept_published_session(turn.session, status)
        if outcome in {"clarify", "refuse"}:
            self.instruction_entry.focus_set()

    def _accept_published_session(
        self,
        session: BuilderProjectSession,
        status: str,
        *,
        sync_fields: bool = True,
    ) -> None:
        self.session = session
        self.state = session.design
        self.conversation = session.conversation
        if sync_fields:
            self._sync_from_state()
        else:
            self._render_state()
        self.status_var.set(status)
        self._update_builder_manifest()

    def _update_builder_manifest(self) -> None:
        if self.project is None or self.session is None:
            return
        design = self.session.design
        self.app.update_current_project(
            {
                "antenna_builder": {
                    "schema_version": 2,
                    "status": "experimental" if design is not None else "awaiting_design",
                    "template_id": design.template_id if design is not None else None,
                    "state": "design/antenna_state.json" if design is not None else None,
                    "conversation": "design/builder_conversation.json",
                    "memory": "design/project_memory.json",
                    **self._planner_settings(),
                    "planner_audit": str(PLANNER_AUDIT_RELATIVE_PATH).replace("\\", "/"),
                }
            }
        )

    def _sync_from_state(self) -> None:
        self._syncing_fields = True
        try:
            if self.state is None:
                self.material_var.set("No design")
                self.material_menu.configure(values=["No design"], state="disabled")
                if self.parameter_vars or self.sweep_vars:
                    self._rebuild_parameter_table()
                self._render_state()
                return
            definitions = recipe_parameter_definitions(self.state)
            expected = {item.key for item in definitions if item.kind != "choice"}
            if set(self.parameter_vars) != expected:
                self._rebuild_parameter_table()
            if self.state.material == "No substrate":
                self.material_var.set("No substrate")
                self.material_menu.configure(values=["No substrate"], state="disabled")
            else:
                self.material_menu.configure(values=list(MATERIALS), state="normal")
                self.material_var.set(self.state.material)
            values = self.agent.values_for_design(self.state)
            for definition in definitions:
                if definition.kind == "choice":
                    continue
                value = values[definition.key]
                rendered = str(int(value)) if definition.kind == "integer" else f"{float(value):.6g}"
                self.parameter_vars[definition.key].set(rendered)
        finally:
            self._syncing_fields = False
        self._render_state()

    def _render_state(self) -> None:
        self._render_project_context()
        if self.state is None:
            self.preview.clear_scene()
            self.preview.grid_remove()
            self.empty_preview.grid(
                row=1,
                column=0,
                padx=12,
                pady=(0, 8),
                sticky="nsew",
            )
            self.preview_summary.configure(text="Awaiting first validated antenna design")
            self.feed_scope_label.configure(
                text="CST export and sampling become available after a design is validated."
            )
            self._set_design_controls_enabled(False)
            self._render_conversation()
            self._update_sampling_summary()
            if self._session_load_error:
                self.apply_button.configure(state="disabled")
            elif self.project is not None and self.session is not None and self._active_plan_cancel is None:
                self.apply_button.configure(state="normal", text="Apply")
            else:
                self.apply_button.configure(state="disabled")
            return
        self.empty_preview.grid_remove()
        self.preview.grid()
        if self._active_plan_cancel is None:
            self.apply_button.configure(state="normal", text="Apply")
        scene = build_geometry_scene(self.state)
        self.preview.set_scene(scene)
        self._set_design_controls_enabled(True)
        self.preview_summary.configure(
            text=(
                f"{self.state.material} · {self.state.frequency_ghz:.4g} GHz · "
                f"{self.state.array.element_count} element{'s' if self.state.array.element_count != 1 else ''} · "
                f"{len(self.state.ports)} validated port{'s' if len(self.state.ports) != 1 else ''}"
            )
        )
        if self.state.family == "dipole":
            scope = "Center-fed discrete port per dipole. Arrays use independent excitations."
        elif self.state.family == "circular_patch":
            scope = "Probe-fed circular elements use independent ports. Validate the probe transition in CST."
        else:
            if "corner_circle_cutouts_v1" in self.state.metadata_map().get("active_modifiers", "").split(","):
                scope = "Corner-circle centers sit on patch corners. No array feed network is generated."
            else:
                scope = "Inset-fed elements use independent ports. No corporate feed network is generated."
        self.feed_scope_label.configure(text=scope)
        self._render_conversation()
        self._update_sampling_summary()

    def _set_design_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.reset_view_button.configure(state=state)
        self.export_button.configure(state=state)
        self.native_cst_button.configure(state=state)
        self.lhs_button.configure(state=state)
        if not enabled:
            self.material_menu.configure(values=["No design"], state="disabled")

    def _update_sampling_summary(self) -> None:
        if self.state is None:
            self.sampling_summary.configure(
                text="0 selected"
            )
            return
        selected = [name for name, variable in self.sweep_vars.items() if variable.get()]
        self.sampling_summary.configure(
            text=f"{len(selected)} selected for sweep"
        )

    def _sweep_selection_changed(self) -> None:
        self._update_sampling_summary()
        selected = [
            name for name, variable in self.sweep_vars.items() if variable.get()
        ]
        self._persisted_sweep_parameters = set(selected)
        if self.project is not None:
            self.project = self.app.update_current_project(
                {"antenna_builder": {"selected_sweep_parameters": selected}}
            )

    def export_cst(self) -> None:
        if self.project is None or self.state is None:
            return
        destination = filedialog.asksaveasfilename(
            title="Save parameterized CST construction macro",
            initialdir=str(self.project.path / "design"),
            initialfile=f"Create_{self.state.recipe_id}_in_CST.bas",
            defaultextension=".bas",
            filetypes=[("CST VBA macro", "*.bas")],
            parent=self,
        )
        if not destination:
            return
        try:
            macro_path, manifest_path = save_cst_package(
                destination,
                self.state,
                conversation=self.conversation,
            )
        except OSError as exc:
            messagebox.showerror("Could not export CST script", str(exc), parent=self)
            return
        self.app.update_current_project(
            {
                "antenna_builder": {
                    "accepted": True,
                    "last_cst_macro": str(macro_path),
                    "last_export_manifest": str(manifest_path),
                }
            }
        )
        self.status_var.set(f"CST script exported: {macro_path.name}")
        messagebox.showinfo(
            "CST script ready",
            (
                f"Saved:\n{macro_path}\n\nDesign record:\n{manifest_path}\n\n"
                "Run the macro in a new CST Microwave Studio project. Review ports, mesh, and materials before solving."
            ),
            parent=self,
        )

    def send_to_lhs(self) -> None:
        if self.state is None:
            return
        selected = [name for name, variable in self.sweep_vars.items() if variable.get()]
        try:
            variables = lhs_variables_for_state(self.state, selected)
        except AntennaBuilderError as exc:
            messagebox.showwarning("Select sampling variables", str(exc), parent=self)
            return
        self.project = self.app.update_current_project(
            {
                "antenna_builder": {
                    "selected_sweep_parameters": selected,
                }
            }
        )
        self._persisted_sweep_parameters = set(selected)
        self.status_var.set("Selected CST parameters transferred to the LHS generator.")
        self.app.show_page("data")
        self.app.after_idle(
            lambda: self.app.data_page.open_lhs_sample_generator(
                initial_variables=variables
            )
        )

    def create_native_cst(self) -> None:
        if self.project is None or self.state is None:
            return
        destination = filedialog.asksaveasfilename(
            title="Create native CST antenna project",
            initialdir=str(self.project.path / "design"),
            initialfile=f"Generated_{self.state.recipe_id}.cst",
            defaultextension=".cst",
            filetypes=[("CST Microwave Studio project", "*.cst")],
            parent=self,
        )
        if not destination:
            return
        path = Path(destination)
        if path.exists():
            messagebox.showwarning(
                "Choose a new CST filename",
                "The experimental builder does not overwrite existing CST projects.",
                parent=self,
            )
            return
        try:
            macro_path, manifest_path = save_cst_package(
                path.with_suffix(".bas"),
                self.state,
                conversation=self.conversation,
            )
        except OSError as exc:
            messagebox.showerror("Could not save CST package", str(exc), parent=self)
            return
        self.native_cst_button.configure(state="disabled", text="Creating in CST…")
        self.export_button.configure(state="disabled")
        self.status_var.set("Creating an unsolved native CST project…")

        def worker() -> None:
            try:
                created = create_native_cst_project(path, self.state)
            except Exception as exc:
                self.after(
                    0,
                    lambda error=exc: self._native_cst_failed(
                        error,
                        macro_path,
                    ),
                )
                return
            self.after(
                0,
                lambda: self._native_cst_complete(
                    created,
                    macro_path,
                    manifest_path,
                ),
            )

        threading.Thread(target=worker, daemon=True).start()

    def _native_cst_failed(self, exc: Exception, macro_path: Path) -> None:
        self.native_cst_button.configure(state="normal", text="Create CST project")
        self.export_button.configure(state="normal")
        self.status_var.set("Native CST creation failed; the construction script was preserved.")
        messagebox.showerror(
            "Could not create native CST project",
            f"{exc}\n\nThe reliable construction script is still available at:\n{macro_path}",
            parent=self,
        )

    def _native_cst_complete(
        self,
        cst_path: Path,
        macro_path: Path,
        manifest_path: Path,
    ) -> None:
        self.native_cst_button.configure(state="normal", text="Create CST project")
        self.export_button.configure(state="normal")
        self.app.update_current_project(
            {
                "antenna_builder": {
                    "accepted": True,
                    "last_cst_project": str(cst_path),
                    "last_cst_macro": str(macro_path),
                    "last_export_manifest": str(manifest_path),
                }
            }
        )
        self.status_var.set(f"Native CST project created: {cst_path.name}")
        messagebox.showinfo(
            "CST project ready",
            (
                f"Native project:\n{cst_path}\n\nConstruction script:\n{macro_path}\n\n"
                "The solver was not started. Review the ports, boundaries, mesh, and material model in CST."
            ),
            parent=self,
        )

    def describe_ui_state(self) -> list[str]:
        selected = [name for name, variable in self.sweep_vars.items() if variable.get()]
        if self.state is None:
            session_status = (
                "load_error"
                if self._session_load_error
                else "awaiting_design"
            )
            return [
                f"Builder status: {session_status}",
                "Design: null",
                "Canonical antenna geometry: none",
                "Parameters, material editing, CST export, native CST creation, and LHS transfer: disabled",
                f"Conversation turns: {len(self.conversation)} persisted messages",
                "3D preview: empty state; no antenna geometry is rendered",
                f"Language planner provider: {self.provider_var.get()}; model: {self._selected_model_id()}; nullable capability manifest is active",
                self.planner_notice_var.get(),
                "Planner audit: design/planner_ab.jsonl records request, returned plan, validation, repair, and executed tool sequence",
                "Agent scope: installed validated recipes only; capability questions do not create a design",
            ]
        return [
            "Builder status: validated parametric design",
            f"Recipe: {self.state.recipe_id}",
            f"Solver-neutral design ID: {self.state.design_id}; revision {self.state.revision}",
            f"Topology: {self.state.topology_label}",
            f"Frequency: {self.state.frequency_ghz:.7g} GHz",
            f"Material: {self.state.material}; epsilon_r={self.state.epsilon_r:.7g}; tan_delta={self.state.loss_tangent:.7g}",
            f"Array spacing: {self.state.element_spacing_lambda:.7g} lambda = {self.state.element_spacing_mm:.7g} mm",
            f"Canonical graph: {len(self.state.geometry)} geometry objects, {len(self.state.booleans)} Boolean operations, {len(self.state.ports)} ports",
            "Validation stages: " + ", ".join(
                f"{record.stage}={'passed' if record.passed else 'failed'}"
                for record in self.state.validation
            ),
            f"Selected LHS parameters: {', '.join(selected) or 'none'}",
            "3D preview: depth-buffered VTK rendering of the Boolean-evaluated canonical mesh with orbit, pan, wheel zoom, true Z scale, and canonical port arrows",
            "Solver adapter: CST parameterized construction macro or native project; solver is not started",
            f"Language planner provider: {self.provider_var.get()}; model: {self._selected_model_id()}; only registered planning tools can execute",
            self.planner_notice_var.get(),
            "Planner audit: design/planner_ab.jsonl records request, returned plan, validation, repair, and executed tool sequence",
            "Agent scope: installed validated recipes only; unsupported families are rejected",
        ]
