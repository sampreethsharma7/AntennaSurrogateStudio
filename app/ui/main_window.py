import json
import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk

from app import APP_NAME, APP_VERSION
from app.assistant.context_builder import build_app_context
from app.assistant.documentation_search import DocumentationSearch
from app.assistant.local_llm_backend import LocalLLMBackend
from app.assistant.offline_help import OfflineGuideBackend
from app.core.compatibility import assess_project_compatibility
from app.core.logging_manager import ProjectLogger
from app.core.project_manager import ProjectManager
from app.core.schema_manager import infer_output_axis, save_schema
from app.ui import theme
from app.utils.paths import APP_DIR, PROJECTS_DIR, ensure_app_dirs
from app.utils.threading_utils import BackgroundRunner

PAGES = [
    ("Library", "Open a recent project or start something new."),
    ("Project Setup", "Describe the antenna project you're building a surrogate model for."),
    ("Generate & Run in CST", "Optional: generate LHS design samples and run them in CST Studio Suite."),
    ("Import & Configure Data", "Load your CSV data and choose which columns are inputs and outputs."),
    ("Diagnostics", "Review data quality before training."),
    ("Train Model", "Configure XGBoost hyperparameters and train a new model version."),
    ("Validate Model", "Review accuracy metrics for the trained model."),
    ("Predict", "Run predictions for new antenna designs."),
    ("Model History", "Compare and activate previous model versions."),
    ("Assistant", "Ask questions about using Antenna Surrogate Studio."),
]
STEP_NUMBERS = {1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤", 6: "⑥", 7: "⑦", 8: "⑧"}
STEP_DONE_KEY = {1: "project", 2: "cst_run", 3: "prepared", 4: "diagnostics_reviewed", 5: "trained", 6: "trained", 7: "trained", 8: "trained"}


class ColumnPicker(ctk.CTkScrollableFrame):
    def __init__(self, parent, height=260):
        palette = theme.PALETTE
        super().__init__(parent, fg_color=palette["surface_alt"], corner_radius=10, border_width=1, border_color=palette["border"], height=height)
        self._vars = {}
        self._order = []

    def set_items(self, columns):
        for child in self.winfo_children():
            child.destroy()
        self._vars = {}
        self._order = list(columns)
        palette = theme.PALETTE
        for col in self._order:
            var = tk.BooleanVar(value=False)
            ctk.CTkCheckBox(
                self, text=col, variable=var, font=theme.ui_font(12),
                text_color=palette["text"], fg_color=palette["accent"], hover_color=palette["accent_hover"],
                checkmark_color="#ffffff", border_color=palette["border_strong"],
            ).pack(anchor="w", padx=10, pady=5)
            self._vars[col] = var

    def get_selected(self):
        return [col for col in self._order if self._vars[col].get()]

    def clear(self):
        self.set_items([])


class AntennaSurrogateStudio(ctk.CTk):
    def __init__(self):
        theme.setup()
        super().__init__(fg_color=theme.PALETTE["app_bg"])
        self.title(APP_NAME)
        self.geometry("1440x920")
        self.minsize(1180, 780)
        ensure_app_dirs()
        self.app_version = APP_VERSION
        self.manager = ProjectManager(PROJECTS_DIR)
        self.project_dir = None
        self.manifest = None
        self.imported_df = None
        self.sample_count = 0
        self.visible_warnings = []
        self.visible_metrics = {}
        self.workflow = {"project": False, "cst_run": False, "imported": False, "prepared": False, "diagnostics_reviewed": False, "trained": False}
        self.tab_titles = [title for title, _subtitle in PAGES]
        self.current_page_name = "Library"
        self.worker_queue = queue.Queue()
        self.runner = BackgroundRunner()
        self.logger = ProjectLogger()
        knowledge_base_dir = APP_DIR / "app" / "assistant" / "knowledge_base"
        self.assistant = OfflineGuideBackend(knowledge_base_dir)
        manual_text = DocumentationSearch(knowledge_base_dir).combined_text(APP_DIR / "USER_MANUAL.txt")
        self.local_llm = LocalLLMBackend(manual_text)
        self._vars()
        self._layout()
        self._load_examples()
        self._refresh_library()
        self._poll_queue()
        self.log("Antenna Surrogate Studio is ready.")

    def _vars(self):
        self.project_name = tk.StringVar(value="Untitled Antenna Project")
        self.project_type = tk.StringVar(value="Custom surrogate model")
        self.project_description = tk.StringVar()
        self.antenna_category = tk.StringVar()
        self.units_preference = tk.StringVar(value="mm/GHz/dB")
        self.import_mode = tk.StringVar(value="wide")
        self.wide_csv_path = tk.StringVar()
        self.inputs_csv_path = tk.StringVar()
        self.outputs_csv_path = tk.StringVar()
        self.sample_id_column = tk.StringVar(value="sample_id")
        self.input_list = None
        self.output_list = None
        self.n_estimators = tk.StringVar(value="100")
        self.max_depth = tk.StringVar(value="6")
        self.learning_rate = tk.StringVar(value="0.1")
        self.subsample = tk.StringVar(value="1.0")
        self.colsample = tk.StringVar(value="1.0")
        self.min_child_weight = tk.StringVar(value="1")
        self.test_split = tk.StringVar(value="0.2")
        self.random_seed = tk.StringVar(value="42")
        self.cv_folds = tk.StringVar(value="0")
        self.prediction_entries = {}
        self.assistant_question = tk.StringVar()
        self.cst_project_path = tk.StringVar()
        self.cst_output_columns = tk.StringVar()
        self.cst_result_tree_path = tk.StringVar(value="1D Results\\S-Parameters\\S1,1")
        self.cst_solver_type = tk.StringVar(value="Time")
        self.cst_sample_count = tk.StringVar(value="20")
        self.cst_seed = tk.StringVar(value="42")
        self.cst_param_rows = []
        self.cst_generated_samples = None
        self.cst_result_csv_path = None

    # ---------- shell layout ----------

    def _layout(self):
        palette = theme.PALETTE
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self, width=236, corner_radius=0, fg_color=palette["sidebar_bg"])
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)
        self._build_sidebar(sidebar)

        content = ctk.CTkFrame(self, fg_color=palette["app_bg"], corner_radius=0)
        content.grid(row=0, column=1, sticky="nsew")
        content.grid_rowconfigure(1, weight=1)
        content.grid_columnconfigure(0, weight=1)

        topbar = ctk.CTkFrame(content, fg_color="transparent")
        topbar.grid(row=0, column=0, sticky="ew", padx=32, pady=(26, 8))
        self.page_title_label = ctk.CTkLabel(topbar, text="Library", font=theme.ui_font(24, bold=True), text_color=palette["text"], anchor="w")
        self.page_title_label.pack(anchor="w")
        self.page_subtitle_label = ctk.CTkLabel(topbar, text="", font=theme.ui_font(12), text_color=palette["text_secondary"], anchor="w")
        self.page_subtitle_label.pack(anchor="w", pady=(2, 0))

        page_container = ctk.CTkFrame(content, fg_color="transparent")
        page_container.grid(row=1, column=0, sticky="nsew", padx=32, pady=(0, 8))
        page_container.grid_rowconfigure(0, weight=1)
        page_container.grid_columnconfigure(0, weight=1)

        builders = [
            self._build_library_page, self._build_project_page, self._build_cst_automation_page, self._build_import_page,
            self._build_diagnostics_page, self._build_training_page, self._build_validation_page,
            self._build_prediction_page, self._build_model_history_page, self._build_assistant_page,
        ]
        self.pages = []
        for builder in builders:
            page_frame = ctk.CTkFrame(page_container, fg_color="transparent")
            page_frame.grid(row=0, column=0, sticky="nsew")
            page_frame.grid_rowconfigure(0, weight=1)
            page_frame.grid_columnconfigure(0, weight=1)
            builder(page_frame)
            self.pages.append(page_frame)

        self._build_log(content)
        self.pages[0].tkraise()
        self._update_tab_states()

    def _build_sidebar(self, sidebar):
        palette = theme.PALETTE
        brand = ctk.CTkFrame(sidebar, fg_color="transparent")
        brand.pack(fill="x", padx=22, pady=(26, 20))
        ctk.CTkLabel(brand, text="Antenna Surrogate\nStudio", font=theme.ui_font(16, bold=True), text_color="#ffffff", justify="left", anchor="w").pack(anchor="w")
        ctk.CTkLabel(brand, text=f"v{self.app_version}", font=theme.ui_font(10), text_color=palette["sidebar_section_label"]).pack(anchor="w", pady=(3, 0))

        self.nav_buttons = {}
        self._add_nav_item(sidebar, 0)

        section = ctk.CTkLabel(sidebar, text="WORKFLOW", font=theme.ui_font(10, bold=True), text_color=palette["sidebar_section_label"], anchor="w")
        section.pack(fill="x", padx=22, pady=(18, 6))
        for index in range(1, 9):
            self._add_nav_item(sidebar, index)

        spacer = ctk.CTkFrame(sidebar, fg_color="transparent")
        spacer.pack(fill="both", expand=True)

        divider = ctk.CTkFrame(sidebar, fg_color=palette["sidebar_border"], height=1)
        divider.pack(fill="x", padx=22, pady=(6, 10))
        self._add_nav_item(sidebar, 9)
        ctk.CTkFrame(sidebar, fg_color="transparent", height=16).pack()

    def _add_nav_item(self, sidebar, index):
        palette = theme.PALETTE
        title = self.tab_titles[index]
        button = ctk.CTkButton(
            sidebar, text=title, anchor="w", font=theme.ui_font(13), corner_radius=8, height=38,
            fg_color="transparent", text_color=palette["sidebar_text"], hover_color=palette["sidebar_bg_hover"],
            command=lambda i=index: self._select_tab(i),
        )
        button.pack(fill="x", padx=14, pady=3)
        self.nav_buttons[index] = button

    def _friendly_error(self, exc):
        text = str(exc)
        lowered = text.lower()
        if "libxgboost" in lowered or "openmp" in lowered or "kmpc" in lowered:
            return (
                "xgboost could not load its native library. This almost always means the OpenMP "
                "runtime is missing.\n\nOn macOS, install it once with Terminal, then restart the app:\n"
                "  brew install libomp\n\n"
                f"Original error: {text}"
            )
        return text

    def _card(self, parent, **pack_kwargs):
        palette = theme.PALETTE
        card = ctk.CTkFrame(parent, fg_color=palette["surface"], corner_radius=14, border_width=1, border_color=palette["border"])
        pack_kwargs.setdefault("fill", "x")
        pack_kwargs.setdefault("pady", (0, 16))
        card.pack(**pack_kwargs)
        return card

    def _card_title(self, card, text, subtitle=None):
        palette = theme.PALETTE
        ctk.CTkLabel(card, text=text, font=theme.ui_font(15, bold=True), text_color=palette["text"], anchor="w").pack(anchor="w", padx=20, pady=(18, 2 if subtitle else 0))
        if subtitle:
            ctk.CTkLabel(card, text=subtitle, font=theme.ui_font(11), text_color=palette["text_secondary"], anchor="w").pack(anchor="w", padx=20, pady=(0, 2))
        ctk.CTkFrame(card, fg_color=palette["border"], height=1).pack(fill="x", padx=20, pady=(12, 0))

    def _primary_button(self, parent, text, command, **kwargs):
        palette = theme.PALETTE
        kwargs.setdefault("height", 38)
        return ctk.CTkButton(parent, text=text, command=command, fg_color=palette["accent"], hover_color=palette["accent_hover"], text_color="#ffffff", corner_radius=9, font=theme.ui_font(13, bold=True), **kwargs)

    def _secondary_button(self, parent, text, command, **kwargs):
        palette = theme.PALETTE
        kwargs.setdefault("height", 38)
        return ctk.CTkButton(parent, text=text, command=command, fg_color=palette["surface_alt"], hover_color=palette["border"], text_color=palette["text"], border_width=1, border_color=palette["border_strong"], corner_radius=9, font=theme.ui_font(13, bold=True), **kwargs)

    def _entry(self, parent, textvariable, **kwargs):
        palette = theme.PALETTE
        kwargs.setdefault("height", 34)
        return ctk.CTkEntry(parent, textvariable=textvariable, corner_radius=8, fg_color=palette["surface"], border_color=palette["border_strong"], text_color=palette["text"], font=theme.ui_font(12), **kwargs)

    def _output_text(self, parent, height=300, wrap="word"):
        palette = theme.PALETTE
        box = ctk.CTkTextbox(parent, height=height, wrap=wrap, font=theme.mono_font(12), fg_color=palette["surface_alt"], text_color=palette["text"], corner_radius=10, border_width=1, border_color=palette["border"])
        box.tag_config("error", foreground=palette["danger"])
        box.tag_config("warning", foreground=palette["warning"])
        box.tag_config("info", foreground=palette["text_secondary"])
        return box

    def _listbox(self, parent, height=260):
        return ColumnPicker(parent, height=height)

    # ---------- workflow / navigation ----------

    def _allowed_tab_indices(self):
        allowed = {0, 1, 9}
        if self.workflow.get("project"):
            allowed.update({2, 3})
        if self.workflow.get("prepared"):
            allowed.add(4)
        if self.workflow.get("prepared") and self.workflow.get("diagnostics_reviewed"):
            allowed.add(5)
        if self.workflow.get("trained"):
            allowed.update({6, 7, 8})
        return allowed

    def _update_tab_states(self):
        if not hasattr(self, "nav_buttons"):
            return
        palette = theme.PALETTE
        allowed = self._allowed_tab_indices()
        current_index = self.tab_titles.index(self.current_page_name)
        for index, button in self.nav_buttons.items():
            title = self.tab_titles[index]
            locked = index not in allowed
            is_current = index == current_index
            if index in STEP_NUMBERS:
                done = bool(self.workflow.get(STEP_DONE_KEY[index]))
                glyph = "✓" if done and not is_current else STEP_NUMBERS[index]
                label = f"{glyph}   {title}"
            else:
                label = title
            button.configure(text=label)
            if locked:
                button.configure(state="disabled", fg_color="transparent", text_color=palette["sidebar_text_locked"], hover_color=palette["sidebar_bg"])
            elif is_current:
                button.configure(state="normal", fg_color=palette["sidebar_active"], text_color=palette["sidebar_text_active"], hover_color=palette["sidebar_active"])
            else:
                button.configure(state="normal", fg_color="transparent", text_color=palette["sidebar_text"], hover_color=palette["sidebar_bg_hover"])
        if current_index not in allowed:
            fallback = max((i for i in allowed if i <= current_index), default=0)
            self._select_tab(fallback)

    def _select_tab(self, title_or_index):
        index = self.tab_titles.index(title_or_index) if isinstance(title_or_index, str) else int(title_or_index)
        if index not in self._allowed_tab_indices():
            self._show_step_lock_message(index)
            return False
        self.current_page_name = self.tab_titles[index]
        self.pages[index].tkraise()
        self.page_title_label.configure(text=self.tab_titles[index])
        self.page_subtitle_label.configure(text=PAGES[index][1])
        self._update_tab_states()
        return True

    def _show_step_lock_message(self, index):
        title = self.tab_titles[index] if 0 <= index < len(self.tab_titles) else "that step"
        message = self._step_lock_reason(index)
        self.log(f"{title} is locked: {message}", "warning")
        messagebox.showinfo(APP_NAME, message)

    def _step_lock_reason(self, index):
        if index in {2, 3}:
            return "Create or open a project before generating CST samples or importing data."
        if index == 4:
            return "Import data, select valid input/output columns, and click Prepare Dataset before reviewing diagnostics."
        if index == 5:
            return "Review the Data Diagnostics page and click Continue to Train before training."
        if index in {6, 7, 8}:
            return "Train a model successfully before validating, predicting, or managing model history."
        return "Complete the previous workflow step first."

    def _has_prepared_dataset(self):
        return bool(
            self.project_dir
            and self.manifest
            and self.manifest.selected_input_columns
            and self.manifest.selected_output_columns
            and (self.project_dir / "data" / "prepared_training_data.csv").exists()
        )

    def _has_active_model(self):
        if not (self.project_dir and self.manifest and self.manifest.active_model_version):
            return False
        version = self.manifest.active_model_version
        return (self.project_dir / "models" / f"model_v{version}.joblib").exists()

    def _reset_downstream_workflow(self, from_step):
        if from_step in {"project", "import"}:
            self.workflow.update({"imported": False, "prepared": False, "diagnostics_reviewed": False, "trained": False})
        elif from_step == "loaded_data":
            self.workflow.update({"prepared": False, "diagnostics_reviewed": False, "trained": False})
        elif from_step == "prepared":
            self.workflow.update({"diagnostics_reviewed": False, "trained": False})
        self._update_tab_states()

    def accept_diagnostics(self):
        if not self.workflow.get("prepared"):
            messagebox.showwarning(APP_NAME, "Prepare a valid dataset before continuing to training.")
            return
        self.workflow["diagnostics_reviewed"] = True
        self._update_tab_states()
        self.log("Diagnostics reviewed. Train Model is now available.")
        self._select_tab("Train Model")

    def _build_log(self, parent):
        palette = theme.PALETTE
        frame = ctk.CTkFrame(parent, fg_color=palette["surface"], corner_radius=14, border_width=1, border_color=palette["border"], height=118)
        frame.grid(row=2, column=0, sticky="ew", padx=32, pady=(0, 20))
        frame.grid_propagate(False)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=16, pady=(10, 4))
        ctk.CTkLabel(header, text="Messages", font=theme.ui_font(12, bold=True), text_color=palette["text"]).pack(side="left")
        self.log_text = self._output_text(frame, height=70)
        self.log_text.grid(row=1, column=0, sticky="nsew", padx=(16, 8), pady=(0, 12))
        ctk.CTkButton(
            frame, text="Copy Error Details", width=140, height=30, corner_radius=8, font=theme.ui_font(11),
            fg_color=palette["surface_alt"], hover_color=palette["border"], text_color=palette["text"],
            border_width=1, border_color=palette["border_strong"],
            command=lambda: self.clipboard_append(self.log_text.get("1.0", "end")),
        ).grid(row=1, column=1, sticky="n", padx=(0, 16), pady=(0, 12))

    # ---------- pages ----------

    def _build_library_page(self, parent):
        actions = self._card(parent)
        self._card_title(actions, "Project Library")
        row = ctk.CTkFrame(actions, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=(14, 20))
        self._primary_button(row, "New Project", lambda: self._select_tab("Project Setup")).pack(side="left")
        self._secondary_button(row, "Open Existing Project", self.open_project_dialog).pack(side="left", padx=(10, 0))
        self._secondary_button(row, "Import Project ZIP", self.import_project_zip).pack(side="left", padx=(10, 0))

        library_card = self._card(parent, fill="both", expand=True)
        self._card_title(library_card, "Recent and Example Projects")
        self.library_frame = ctk.CTkScrollableFrame(library_card, fg_color="transparent")
        self.library_frame.pack(fill="both", expand=True, padx=20, pady=(14, 20))

    def _build_project_page(self, parent):
        palette = theme.PALETTE
        card = self._card(parent)
        self._card_title(card, "Create Project")
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="x", padx=20, pady=(16, 20))
        body.columnconfigure(1, weight=1)
        fields = [("Project name", self.project_name), ("Description", self.project_description), ("Antenna category", self.antenna_category), ("Units preference", self.units_preference)]
        for i, (label, var) in enumerate(fields):
            ctk.CTkLabel(body, text=label, font=theme.ui_font(12), text_color=palette["text_secondary"], anchor="w").grid(row=i, column=0, sticky="w", pady=8, padx=(0, 14))
            self._entry(body, var, width=420).grid(row=i, column=1, sticky="ew", pady=8)
        ctk.CTkLabel(body, text="Project type", font=theme.ui_font(12), text_color=palette["text_secondary"], anchor="w").grid(row=len(fields), column=0, sticky="w", pady=8, padx=(0, 14))
        ctk.CTkOptionMenu(
            body, variable=self.project_type,
            values=["S-parameter prediction", "Radiation-pattern prediction", "Gain / efficiency prediction", "Multi-metric antenna prediction", "Custom surrogate model"],
            corner_radius=8, fg_color=palette["surface_alt"], button_color=palette["accent"], button_hover_color=palette["accent_hover"],
            text_color=palette["text"], font=theme.ui_font(12), dropdown_font=theme.ui_font(12), height=34,
        ).grid(row=len(fields), column=1, sticky="w", pady=8)
        self._primary_button(body, "Create Project", self.create_project).grid(row=len(fields) + 1, column=1, sticky="e", pady=(14, 0))

    def _build_cst_automation_page(self, parent):
        palette = theme.PALETTE
        from app.core.cst_automation import cst_automation_available

        bounds_card = self._card(parent)
        self._card_title(bounds_card, "Design Parameters (Latin Hypercube Sampling)", subtitle="Define design variables and ranges for CST to sweep. Use the same names you plan to use as input columns after import.")
        header_row = ctk.CTkFrame(bounds_card, fg_color="transparent")
        header_row.pack(fill="x", padx=20, pady=(14, 4))
        for text, width in [("Parameter name", 200), ("Min", 120), ("Max", 120)]:
            ctk.CTkLabel(header_row, text=text, font=theme.ui_font(10, bold=True), text_color=palette["text_muted"], width=width, anchor="w").pack(side="left", padx=(0, 10))
        self.cst_param_rows_frame = ctk.CTkFrame(bounds_card, fg_color="transparent")
        self.cst_param_rows_frame.pack(fill="x", padx=20)
        self.cst_param_rows = []
        for _ in range(2):
            self._add_cst_param_row()
        self._secondary_button(bounds_card, "+ Add Parameter", self._add_cst_param_row, width=160).pack(anchor="w", padx=20, pady=(8, 16))

        sampling_row = ctk.CTkFrame(bounds_card, fg_color="transparent")
        sampling_row.pack(fill="x", padx=20, pady=(0, 14))
        ctk.CTkLabel(sampling_row, text="Number of samples", font=theme.ui_font(11), text_color=palette["text_secondary"]).pack(side="left")
        self._entry(sampling_row, self.cst_sample_count, width=80).pack(side="left", padx=(8, 20))
        ctk.CTkLabel(sampling_row, text="Random seed", font=theme.ui_font(11), text_color=palette["text_secondary"]).pack(side="left")
        self._entry(sampling_row, self.cst_seed, width=80).pack(side="left", padx=(8, 20))
        self._primary_button(sampling_row, "Generate Samples", self.generate_cst_samples).pack(side="left")

        self.cst_preview_text = self._output_text(bounds_card, height=140)
        self.cst_preview_text.pack(fill="x", padx=20, pady=(0, 20))

        run_card = self._card(parent, fill="both", expand=True)
        self._card_title(run_card, "CST Project & Run")
        form = ctk.CTkFrame(run_card, fg_color="transparent")
        form.pack(fill="x", padx=20, pady=(16, 0))
        form.columnconfigure(1, weight=1)
        ctk.CTkLabel(form, text="CST project template (.cst)", font=theme.ui_font(11), text_color=palette["text_secondary"], anchor="w").grid(row=0, column=0, sticky="w", pady=6, padx=(0, 10))
        self._entry(form, self.cst_project_path).grid(row=0, column=1, sticky="ew", pady=6)
        self._secondary_button(form, "Browse", self._browse_cst_project, width=90).grid(row=0, column=2, padx=(10, 0))
        ctk.CTkLabel(form, text="Output columns (comma-separated)", font=theme.ui_font(11), text_color=palette["text_secondary"], anchor="w").grid(row=1, column=0, sticky="w", pady=6, padx=(0, 10))
        self._entry(form, self.cst_output_columns).grid(row=1, column=1, columnspan=2, sticky="ew", pady=6)
        ctk.CTkLabel(form, text="CST result tree path", font=theme.ui_font(11), text_color=palette["text_secondary"], anchor="w").grid(row=2, column=0, sticky="w", pady=6, padx=(0, 10))
        self._entry(form, self.cst_result_tree_path).grid(row=2, column=1, columnspan=2, sticky="ew", pady=6)
        ctk.CTkLabel(form, text="Solver", font=theme.ui_font(11), text_color=palette["text_secondary"], anchor="w").grid(row=3, column=0, sticky="w", pady=6, padx=(0, 10))
        ctk.CTkOptionMenu(
            form, variable=self.cst_solver_type, values=["Time", "Frequency"], corner_radius=8,
            fg_color=palette["surface_alt"], button_color=palette["accent"], button_hover_color=palette["accent_hover"],
            text_color=palette["text"], font=theme.ui_font(12), height=34, width=140,
        ).grid(row=3, column=1, sticky="w", pady=6)

        available = cst_automation_available()
        if available:
            status_text = "CST Studio Suite automation is available on this machine."
        else:
            status_text = (
                "CST automation requires Windows with CST Studio Suite and the pywin32 package installed. "
                "Not available on this machine - use manual CSV import on the Import & Configure Data page instead. "
                "This integration is best-effort against CST's documented COM automation API and has not been "
                "verified against a live CST installation; confirm method names in CST's Macro Editor before relying on it."
            )
        ctk.CTkLabel(run_card, text=status_text, font=theme.ui_font(10), text_color=palette["text_muted"] if available else palette["warning"], wraplength=900, anchor="w", justify="left").pack(anchor="w", padx=20, pady=(12, 8))

        actions = ctk.CTkFrame(run_card, fg_color="transparent")
        actions.pack(fill="x", padx=20)
        run_button = self._primary_button(actions, "Run in CST", self.run_cst_batch)
        run_button.pack(side="left")
        if not available:
            run_button.configure(state="disabled")
        self.cst_load_button = self._secondary_button(actions, "Load Results into Import", self._load_cst_results_into_import, width=200)
        self.cst_load_button.pack(side="left", padx=(10, 0))
        self.cst_load_button.configure(state="disabled")

        self.cst_output_text = self._output_text(run_card, height=160)
        self.cst_output_text.pack(fill="both", expand=True, padx=20, pady=(12, 20))

    def _add_cst_param_row(self):
        row = ctk.CTkFrame(self.cst_param_rows_frame, fg_color="transparent")
        row.pack(fill="x", pady=3)
        name_var = tk.StringVar()
        min_var = tk.StringVar()
        max_var = tk.StringVar()
        self._entry(row, name_var, width=200).pack(side="left", padx=(0, 10))
        self._entry(row, min_var, width=120).pack(side="left", padx=(0, 10))
        self._entry(row, max_var, width=120).pack(side="left", padx=(0, 10))
        entry_record = {"name": name_var, "min": min_var, "max": max_var, "row": row}
        self._secondary_button(row, "Remove", lambda: self._remove_cst_param_row(entry_record), width=80, height=30).pack(side="left")
        self.cst_param_rows.append(entry_record)

    def _remove_cst_param_row(self, entry_record):
        entry_record["row"].destroy()
        self.cst_param_rows.remove(entry_record)

    def _browse_cst_project(self):
        path = filedialog.askopenfilename(filetypes=[("CST project files", "*.cst"), ("All files", "*.*")])
        if path:
            self.cst_project_path.set(path)

    def _collect_cst_bounds(self):
        bounds = {}
        for record in self.cst_param_rows:
            name = record["name"].get().strip()
            if not name:
                continue
            try:
                low = float(record["min"].get())
                high = float(record["max"].get())
            except ValueError as exc:
                raise ValueError(f"Parameter '{name}' needs numeric min/max bounds.") from exc
            bounds[name] = (low, high)
        if not bounds:
            raise ValueError("Add at least one parameter with a name and numeric bounds.")
        return bounds

    def generate_cst_samples(self):
        try:
            bounds = self._collect_cst_bounds()
            n_samples = int(self.cst_sample_count.get())
            seed = int(self.cst_seed.get()) if self.cst_seed.get().strip() else None
            from app.core.lhs_sampling import generate_lhs_samples
            self.cst_generated_samples = generate_lhs_samples(bounds, n_samples, seed)
            preview = json.dumps(self.cst_generated_samples[:10], indent=2)
            suffix = f"\n\n... and {n_samples - 10} more samples." if n_samples > 10 else ""
            self.cst_preview_text.delete("1.0", "end")
            self.cst_preview_text.insert("end", preview + suffix)
            self.log(f"Generated {n_samples} LHS design samples.")
        except Exception as exc:
            messagebox.showerror(APP_NAME, str(exc))

    def run_cst_batch(self):
        if not self.project_dir:
            messagebox.showwarning(APP_NAME, "Create or open a project first.")
            return
        try:
            bounds = self._collect_cst_bounds()
            n_samples = int(self.cst_sample_count.get())
            seed = int(self.cst_seed.get()) if self.cst_seed.get().strip() else None
            output_columns = [c.strip() for c in self.cst_output_columns.get().split(",") if c.strip()]
            if not output_columns:
                raise ValueError("List at least one output column name (comma-separated).")
            project_template = self.cst_project_path.get().strip()
            if not project_template:
                raise ValueError("Select a CST project template (.cst file).")
            result_tree_path = self.cst_result_tree_path.get().strip()
            if not result_tree_path:
                raise ValueError("Enter the CST result tree path to read for each run.")
        except Exception as exc:
            messagebox.showerror(APP_NAME, str(exc))
            return

        axis_metadata = infer_output_axis(output_columns)
        solver = self.cst_solver_type.get()
        self.cst_output_text.delete("1.0", "end")
        self.cst_output_text.insert("end", "Starting CST batch run...\n")
        self.log("CST batch run started in background.")

        def work():
            from app.core.cst_automation import run_lhs_batch
            return run_lhs_batch(
                Path(project_template), bounds, n_samples, output_columns, axis_metadata, result_tree_path,
                seed=seed, solver=solver, progress=lambda m: self.worker_queue.put(("log", m, "info")),
            )
        future = self.runner.submit(work)
        future.add_done_callback(lambda f: self.worker_queue.put(("cst_batch_done", f, "info")))

    def _load_cst_results_into_import(self):
        if not self.cst_result_csv_path:
            return
        self.wide_csv_path.set(str(self.cst_result_csv_path))
        self.import_mode.set("wide")
        self._select_tab("Import & Configure Data")
        self.log("Loaded CST-generated dataset into Import & Configure Data. Click Load Columns to continue.")

    def _build_import_page(self, parent):
        palette = theme.PALETTE
        source = self._card(parent)
        self._card_title(source, "Import Dataset")
        body = ctk.CTkFrame(source, fg_color="transparent")
        body.pack(fill="x", padx=20, pady=(16, 20))
        body.columnconfigure(1, weight=1)
        ctk.CTkRadioButton(body, text="Single wide CSV", variable=self.import_mode, value="wide", font=theme.ui_font(12), text_color=palette["text"], fg_color=palette["accent"]).grid(row=0, column=0, sticky="w", pady=8)
        self._entry(body, self.wide_csv_path).grid(row=0, column=1, sticky="ew", padx=10)
        self._secondary_button(body, "Browse", lambda: self._browse(self.wide_csv_path), width=90).grid(row=0, column=2)
        ctk.CTkRadioButton(body, text="Split input/output CSV", variable=self.import_mode, value="split", font=theme.ui_font(12), text_color=palette["text"], fg_color=palette["accent"]).grid(row=1, column=0, sticky="w", pady=8)
        self._entry(body, self.inputs_csv_path).grid(row=1, column=1, sticky="ew", padx=10)
        self._secondary_button(body, "Inputs", lambda: self._browse(self.inputs_csv_path), width=90).grid(row=1, column=2)
        self._entry(body, self.outputs_csv_path).grid(row=2, column=1, sticky="ew", padx=10, pady=6)
        self._secondary_button(body, "Outputs", lambda: self._browse(self.outputs_csv_path), width=90).grid(row=2, column=2)
        id_row = ctk.CTkFrame(body, fg_color="transparent")
        id_row.grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ctk.CTkLabel(id_row, text="Sample ID column", font=theme.ui_font(11), text_color=palette["text_secondary"]).pack(side="left", padx=(0, 8))
        self._entry(id_row, self.sample_id_column, width=160).pack(side="left")
        self._primary_button(body, "Load Columns", self.load_dataset).grid(row=4, column=2, sticky="e", pady=(14, 0))

        lists_card = self._card(parent)
        lists_body = ctk.CTkFrame(lists_card, fg_color="transparent")
        lists_body.pack(fill="both", expand=True, padx=20, pady=(16, 20))
        lists_body.columnconfigure(0, weight=1)
        lists_body.columnconfigure(1, weight=1)
        ctk.CTkLabel(lists_body, text="Input design variables", font=theme.ui_font(13, bold=True), text_color=palette["text"], anchor="w").grid(row=0, column=0, sticky="w", pady=(0, 8))
        ctk.CTkLabel(lists_body, text="Output response columns", font=theme.ui_font(13, bold=True), text_color=palette["text"], anchor="w").grid(row=0, column=1, sticky="w", pady=(0, 8), padx=(14, 0))
        self.input_list = self._listbox(lists_body)
        self.output_list = self._listbox(lists_body)
        self.input_list.grid(row=1, column=0, sticky="nsew")
        self.output_list.grid(row=1, column=1, sticky="nsew", padx=(14, 0))
        self._primary_button(parent, "Prepare Dataset", self.prepare_dataset).pack(anchor="e", pady=(0, 20))

    def _build_diagnostics_page(self, parent):
        card = self._card(parent, fill="both", expand=True)
        header = ctk.CTkFrame(card, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(18, 0))
        ctk.CTkLabel(header, text="Data Diagnostics", font=theme.ui_font(15, bold=True), text_color=theme.PALETTE["text"]).pack(side="left")
        self._primary_button(header, "Continue to Train", self.accept_diagnostics).pack(side="right")
        ctk.CTkFrame(card, fg_color=theme.PALETTE["border"], height=1).pack(fill="x", padx=20, pady=(12, 14))
        self.diagnostics_text = self._output_text(card, height=440)
        self.diagnostics_text.pack(fill="both", expand=True, padx=20, pady=(0, 20))

    def _build_training_page(self, parent):
        palette = theme.PALETTE
        card = self._card(parent)
        self._card_title(card, "Train XGBoost Model")
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="x", padx=20, pady=(16, 20))
        settings = [("n_estimators", self.n_estimators), ("max_depth", self.max_depth), ("learning_rate", self.learning_rate), ("subsample", self.subsample), ("colsample_bytree", self.colsample), ("min_child_weight", self.min_child_weight), ("test split", self.test_split), ("random seed", self.random_seed), ("CV folds", self.cv_folds)]
        for i, (label, var) in enumerate(settings):
            r, c = divmod(i, 3)
            cell = ctk.CTkFrame(body, fg_color="transparent")
            cell.grid(row=r, column=c, sticky="w", padx=(0, 24), pady=8)
            ctk.CTkLabel(cell, text=label, font=theme.ui_font(11), text_color=palette["text_secondary"], anchor="w").pack(anchor="w")
            self._entry(cell, var, width=140).pack(anchor="w", pady=(4, 0))
        self._primary_button(body, "Train New Model Version", self.train_model).grid(row=(len(settings) // 3) + 1, column=0, sticky="w", pady=(16, 0))

    def _build_validation_page(self, parent):
        card = self._card(parent, fill="both", expand=True)
        self._card_title(card, "Validation Metrics")
        self.validation_text = self._output_text(card, height=460)
        self.validation_text.pack(fill="both", expand=True, padx=20, pady=(14, 20))

    def _build_prediction_page(self, parent):
        palette = theme.PALETTE
        form_card = self._card(parent)
        header = ctk.CTkFrame(form_card, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(18, 0))
        ctk.CTkLabel(header, text="Predict New Design", font=theme.ui_font(15, bold=True), text_color=palette["text"]).pack(side="left")
        self._secondary_button(header, "Refresh Fields", self.refresh_prediction_fields, width=130).pack(side="right")
        ctk.CTkFrame(form_card, fg_color=palette["border"], height=1).pack(fill="x", padx=20, pady=(12, 0))
        self.prediction_fields_frame = ctk.CTkFrame(form_card, fg_color="transparent")
        self.prediction_fields_frame.pack(fill="x", padx=20, pady=(14, 0))
        actions = ctk.CTkFrame(form_card, fg_color="transparent")
        actions.pack(fill="x", padx=20, pady=(10, 20))
        self._primary_button(actions, "Predict", self.predict_one).pack(side="left")
        self._secondary_button(actions, "Batch CSV", self.predict_batch).pack(side="left", padx=(10, 0))

        output_card = self._card(parent, fill="both", expand=True)
        self._card_title(output_card, "Prediction Output")
        self.prediction_output = self._output_text(output_card, height=260, wrap="none")
        self.prediction_output.pack(fill="both", expand=True, padx=20, pady=(14, 20))

    def _build_model_history_page(self, parent):
        palette = theme.PALETTE
        card = self._card(parent, fill="both", expand=True)
        self._card_title(card, "Model History")
        tree_wrap = ctk.CTkFrame(card, fg_color=palette["surface"])
        tree_wrap.pack(fill="both", expand=True, padx=20, pady=(14, 10))
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Treeview", background=palette["surface"], fieldbackground=palette["surface"], foreground=palette["text"], rowheight=28, font=theme.ui_font(11), borderwidth=0)
        style.configure("Treeview.Heading", background=palette["surface_alt"], foreground=palette["text_secondary"], font=theme.ui_font(10, bold=True), relief="flat")
        style.map("Treeview", background=[("selected", palette["accent_soft"])], foreground=[("selected", palette["accent"])])
        self.model_tree = ttk.Treeview(tree_wrap, columns=("version", "type", "rmse", "r2", "active"), show="headings", height=14)
        for col in ("version", "type", "rmse", "r2", "active"):
            self.model_tree.heading(col, text=col.title())
        self.model_tree.tag_configure("odd", background=palette["surface_alt"])
        self.model_tree.tag_configure("even", background=palette["surface"])
        self.model_tree.pack(fill="both", expand=True)
        self._primary_button(card, "Activate Selected Version", self.activate_selected_model).pack(anchor="e", padx=20, pady=(0, 20))

    def _build_assistant_page(self, parent):
        palette = theme.PALETTE
        card = self._card(parent, fill="both", expand=True)
        self._card_title(card, "Assistant", subtitle="Assistant responses are generated from local product documentation and optional local models. Your project data is not sent to cloud services by this application.")
        status_text = self.local_llm.get_status() if self.local_llm.is_available() else f"{self.local_llm.get_status()} Using Basic Offline Guide for now."
        ctk.CTkLabel(card, text=status_text, font=theme.ui_font(10), text_color=palette["text_muted"], anchor="w").pack(anchor="w", padx=20, pady=(6, 0))
        self.assistant_chat = self._output_text(card, height=300)
        self.assistant_chat.pack(fill="both", expand=True, padx=20, pady=(14, 12))
        entry = ctk.CTkFrame(card, fg_color="transparent")
        entry.pack(fill="x", padx=20)
        self._entry(entry, self.assistant_question).pack(side="left", fill="x", expand=True)
        self._primary_button(entry, "Ask", self.ask_assistant, width=90).pack(side="left", padx=(8, 0))
        self._secondary_button(entry, "Clear", lambda: self.assistant_chat.delete("1.0", "end"), width=90).pack(side="left", padx=(8, 0))
        topics = ctk.CTkFrame(card, fg_color="transparent")
        topics.pack(fill="x", padx=20, pady=(12, 20))
        for topic in ["How do I create a new project?", "Which CSV format should I use?", "What is RMSE?", "How do I make a prediction?"]:
            self._secondary_button(topics, topic, lambda t=topic: self._ask_topic(t), height=30).pack(side="left", padx=(0, 8))

    def _load_examples(self):
        try:
            from app.core.example_data import ensure_example_projects
            ensure_example_projects(PROJECTS_DIR)
        except Exception as exc:
            self.log(f"Example setup skipped: {exc}", "warning")

    def _refresh_library(self):
        palette = theme.PALETTE
        for child in self.library_frame.winfo_children():
            child.destroy()
        projects = self.manager.recent_projects()
        if not projects:
            ctk.CTkLabel(self.library_frame, text="No projects yet. Create one to get started.", font=theme.ui_font(12), text_color=palette["text_secondary"]).pack(anchor="w")
        for project in projects:
            try:
                manifest = self.manager.load_manifest(project)
                models = manifest.model_versions
                active = next((m for m in models if m.get("version") == manifest.active_model_version), {}) if models else {}
                row_frame = ctk.CTkFrame(self.library_frame, fg_color=palette["surface_alt"], corner_radius=10)
                row_frame.pack(fill="x", pady=5)
                accent_bar = ctk.CTkFrame(row_frame, fg_color=palette["accent"], width=4, corner_radius=0)
                accent_bar.pack(side="left", fill="y")
                body = ctk.CTkFrame(row_frame, fg_color="transparent")
                body.pack(side="left", fill="both", expand=True, padx=16, pady=12)
                ctk.CTkLabel(body, text=manifest.project_name, font=theme.ui_font(14, bold=True), text_color=palette["text"], anchor="w").pack(anchor="w")
                ctk.CTkLabel(
                    body,
                    text=f"{manifest.project_type}  •  {len(manifest.selected_input_columns)} inputs  •  {len(manifest.selected_output_columns)} outputs  •  {self._project_samples(project)} samples",
                    font=theme.ui_font(11), text_color=palette["text_secondary"], anchor="w",
                ).pack(anchor="w", pady=(3, 8))
                badges = ctk.CTkFrame(body, fg_color="transparent")
                badges.pack(anchor="w")
                if active:
                    self._badge(badges, f"Model: {active.get('model_type', 'None')}", palette["accent_soft"], palette["accent"])
                    self._badge(badges, f"RMSE {active.get('overall_rmse', 0):.4f}", palette["success_soft"], palette["success"])
                    self._badge(badges, f"R² {active.get('overall_r2', 0):.4f}", palette["success_soft"], palette["success"])
                else:
                    self._badge(badges, "No trained model yet", palette["app_bg"], palette["text_muted"])
                self._primary_button(row_frame, "Open", lambda p=project: self.open_project(p), width=90).pack(side="right", padx=14)
            except Exception as exc:
                ctk.CTkLabel(self.library_frame, text=f"{project.name}: {exc}", font=theme.ui_font(11), text_color=palette["text_secondary"]).pack(anchor="w")

    def _badge(self, parent, text, bg, fg):
        ctk.CTkLabel(parent, text=text, font=theme.ui_font(10, bold=True), fg_color=bg, text_color=fg, corner_radius=6, height=22).pack(side="left", padx=(0, 6), ipadx=6)

    def _project_samples(self, project):
        path = project / "data" / "prepared_training_data.csv"
        if not path.exists():
            path = project / "data" / "imported_dataset.csv"
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return max(sum(1 for _ in handle) - 1, 0)

    def _browse(self, var):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            var.set(path)

    def create_project(self):
        self.project_dir = self.manager.create_project(self.project_name.get(), self.project_type.get(), self.project_description.get(), self.antenna_category.get(), self.units_preference.get())
        self.open_project(self.project_dir)
        self.log(f"Created project: {self.project_dir}")
        self._select_tab("Import & Configure Data")

    def open_project_dialog(self):
        path = filedialog.askdirectory(initialdir=PROJECTS_DIR)
        if path:
            self.open_project(Path(path))

    def open_project(self, project_dir):
        self.project_dir = Path(project_dir)
        self.manifest = self.manager.load_manifest(self.project_dir)
        status, note = assess_project_compatibility(self.manifest)
        self.logger.bind_project(self.project_dir)
        prepared = self._has_prepared_dataset()
        trained = self._has_active_model()
        self.workflow.update({
            "project": True,
            "imported": (self.project_dir / "data" / "imported_dataset.csv").exists(),
            "prepared": prepared,
            "diagnostics_reviewed": trained,
            "trained": trained,
        })
        self.sample_count = self._project_samples(self.project_dir)
        self.log(f"Opened project {self.manifest.project_name}. Compatibility: {status}. {note}")
        self.refresh_prediction_fields()
        self.refresh_model_history()
        self._refresh_library()
        self._update_tab_states()

    def import_project_zip(self):
        path = filedialog.askopenfilename(filetypes=[("ZIP archives", "*.zip")])
        if path:
            self.open_project(self.manager.import_project_zip(Path(path)))

    def load_dataset(self):
        if not self.project_dir:
            messagebox.showwarning(APP_NAME, "Create or open a project first.")
            return
        try:
            from app.core.data_importer import DataImporter
            importer = DataImporter()
            if self.import_mode.get() == "wide":
                self.imported_df = importer.import_wide_csv(Path(self.wide_csv_path.get()), self.project_dir)
            else:
                self.imported_df = importer.import_split_csv(Path(self.inputs_csv_path.get()), Path(self.outputs_csv_path.get()), self.sample_id_column.get(), self.project_dir)
            self.input_list.set_items(self.imported_df.columns)
            self.output_list.set_items(self.imported_df.columns)
            self.workflow["imported"] = True
            self._reset_downstream_workflow("loaded_data")
            self.log(f"Loaded dataset with {len(self.imported_df)} rows and {len(self.imported_df.columns)} columns. Select inputs/outputs and prepare the dataset to continue.")
        except Exception as exc:
            self.log(f"Dataset import failed: {exc}", "error")
            messagebox.showerror(APP_NAME, str(exc))

    def prepare_dataset(self):
        if self.imported_df is None:
            messagebox.showwarning(APP_NAME, "Load a dataset first.")
            return
        inputs = self.input_list.get_selected()
        outputs = self.output_list.get_selected()
        if not inputs or not outputs:
            messagebox.showwarning(APP_NAME, "Select at least one input and one output column.")
            return
        try:
            from app.core.data_validator import validate_dataset
            from app.core.diagnostics import compute_diagnostics
            findings = validate_dataset(self.imported_df, inputs, outputs, self.sample_id_column.get() or None)
            errors = [f for f in findings if f["level"] == "error"]
            self.visible_warnings = [f["message"] for f in findings if f["level"] != "info"]
            if errors:
                self.diagnostics_text.delete("1.0", "end")
                self.diagnostics_text.insert("end", json.dumps({"findings": findings}, indent=2))
                self.log("Prepare dataset blocked because validation found errors. Fix the selected data before continuing.", "error")
                messagebox.showerror(APP_NAME, "Validation found blocking errors. Fix missing values, nonnumeric selected columns, duplicate IDs, or missing columns before continuing.")
                return
            prepared = self.imported_df[inputs + outputs].copy()
            prepared.to_csv(self.project_dir / "data" / "prepared_training_data.csv", index=False)
            self.manifest.selected_input_columns = inputs
            self.manifest.selected_output_columns = outputs
            self.manifest.input_units = {c: "" for c in inputs}
            self.manifest.output_units = {c: "" for c in outputs}
            self.manifest.input_bounds = {c: {"min": float(prepared[c].min()), "max": float(prepared[c].max())} for c in inputs}
            self.manifest.output_axis_metadata = infer_output_axis(outputs)
            self.manifest.data_paths = {"prepared_training_data": "data/prepared_training_data.csv", "imported_dataset": "data/imported_dataset.csv"}
            self.manifest.active_model_version = None
            active_pointer = self.project_dir / "models" / "active_model.json"
            active_pointer.write_text(json.dumps({"active_model_version": None}, indent=2), encoding="utf-8")
            self.manifest.save(self.project_dir)
            save_schema(self.project_dir, {"input_columns": inputs, "output_columns": outputs, "output_axis_metadata": self.manifest.output_axis_metadata})
            diagnostics = compute_diagnostics(prepared, inputs, outputs, float(self.test_split.get()))
            (self.project_dir / "analysis" / "data_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            self.diagnostics_text.delete("1.0", "end")
            self.diagnostics_text.insert("end", json.dumps({"findings": findings, "diagnostics": diagnostics}, indent=2))
            self.workflow["prepared"] = True
            self.workflow["diagnostics_reviewed"] = False
            self.workflow["trained"] = False
            self.sample_count = len(prepared)
            self.refresh_prediction_fields()
            self._update_tab_states()
            self.log("Prepared training dataset and saved schema. Review diagnostics before training.")
            self._select_tab("Diagnostics")
        except Exception as exc:
            self.log(f"Prepare dataset failed: {exc}", "error")
            messagebox.showerror(APP_NAME, str(exc))

    def train_model(self):
        if not self.workflow.get("prepared"):
            messagebox.showwarning(APP_NAME, "Prepare a dataset before training.")
            return
        if not self.workflow.get("diagnostics_reviewed"):
            messagebox.showwarning(APP_NAME, "Review Data Diagnostics and click Continue to Train before training.")
            return
        settings = {
            "test_split": float(self.test_split.get()),
            "random_seed": int(self.random_seed.get()),
            "cross_validation_folds": int(self.cv_folds.get()),
            "hyperparameters": {
                "n_estimators": int(self.n_estimators.get()),
                "max_depth": int(self.max_depth.get()),
                "learning_rate": float(self.learning_rate.get()),
                "subsample": float(self.subsample.get()),
                "colsample_bytree": float(self.colsample.get()),
                "min_child_weight": float(self.min_child_weight.get()),
            },
        }
        self.log("Training started in background.")
        def work():
            from app.core.model_trainer import train_xgboost_model
            return train_xgboost_model(self.project_dir, self.manifest, settings, progress=lambda m: self.worker_queue.put(("log", m, "info")))
        future = self.runner.submit(work)
        future.add_done_callback(lambda f: self.worker_queue.put(("trained", f, "info")))

    def refresh_model_history(self):
        if not hasattr(self, "model_tree"):
            return
        for item in self.model_tree.get_children():
            self.model_tree.delete(item)
        if not self.manifest:
            return
        for i, model in enumerate(self.manifest.model_versions):
            active = "yes" if model.get("version") == self.manifest.active_model_version else ""
            self.model_tree.insert("", "end", values=(model.get("version"), model.get("model_type"), round(model.get("overall_rmse", 0), 6), round(model.get("overall_r2", 0), 6), active), tags=("even" if i % 2 == 0 else "odd",))

    def activate_selected_model(self):
        selected = self.model_tree.selection()
        if not selected:
            return
        version = int(self.model_tree.item(selected[0], "values")[0])
        from app.core.model_registry import set_active_model
        set_active_model(self.project_dir, self.manifest, version)
        self.workflow["trained"] = self._has_active_model()
        self._update_tab_states()
        self.refresh_model_history()
        self.log(f"Activated model version {version}.")

    def refresh_prediction_fields(self):
        if not hasattr(self, "prediction_fields_frame"):
            return
        for child in self.prediction_fields_frame.winfo_children():
            child.destroy()
        self.prediction_entries = {}
        if not self.manifest:
            return
        palette = theme.PALETTE
        self.prediction_fields_frame.columnconfigure(1, weight=1)
        for row, col in enumerate(self.manifest.selected_input_columns):
            ctk.CTkLabel(self.prediction_fields_frame, text=col, font=theme.ui_font(12), text_color=palette["text_secondary"], anchor="w").grid(row=row, column=0, sticky="w", pady=5, padx=(0, 14))
            var = tk.StringVar()
            bounds = self.manifest.input_bounds.get(col, {})
            if bounds:
                var.set(str(round((bounds["min"] + bounds["max"]) / 2, 6)))
            self._entry(self.prediction_fields_frame, var, width=180).grid(row=row, column=1, sticky="w", pady=5)
            ctk.CTkLabel(self.prediction_fields_frame, text=f"range {bounds.get('min', '--')} to {bounds.get('max', '--')}", font=theme.ui_font(10), text_color=palette["text_muted"], anchor="w").grid(row=row, column=2, sticky="w", padx=10)
            self.prediction_entries[col] = var

    def predict_one(self):
        try:
            import pandas as pd
            from app.core.predictor import extrapolation_warnings, predict_dataframe
            values = {col: float(var.get()) for col, var in self.prediction_entries.items()}
            warnings = extrapolation_warnings(values, self.manifest)
            result = predict_dataframe(self.project_dir, self.manifest, pd.DataFrame([values]))
            self.prediction_output.delete("1.0", "end")
            if warnings:
                self.prediction_output.insert("end", "\n".join(warnings) + "\n\n")
            self.prediction_output.insert("end", result.to_string(index=False))
            self.log("Prediction complete.")
        except Exception as exc:
            self.log(f"Prediction failed: {exc}", "error")
            messagebox.showerror(APP_NAME, self._friendly_error(exc))

    def predict_batch(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        try:
            import pandas as pd
            from app.core.predictor import predict_dataframe
            result = predict_dataframe(self.project_dir, self.manifest, pd.read_csv(path))
            export_path = self.project_dir / "predictions" / "exported_predictions" / f"{Path(path).stem}_predicted.csv"
            result.to_csv(export_path, index=False)
            self.prediction_output.delete("1.0", "end")
            self.prediction_output.insert("end", result.head(50).to_string(index=False))
            self.log(f"Batch prediction complete. Exported {export_path}.")
        except Exception as exc:
            self.log(f"Batch prediction failed: {exc}", "error")
            messagebox.showerror(APP_NAME, self._friendly_error(exc))

    def ask_assistant(self):
        question = self.assistant_question.get().strip()
        if not question:
            return
        self.assistant_chat.insert("end", f"You: {question}\n")
        self.assistant_question.set("")
        if self.local_llm.is_available():
            placeholder_start = self.assistant_chat.index("end-1c")
            self.assistant_chat.insert("end", f"Assistant ({self.local_llm.model_name}) is thinking, this can take up to a minute on this machine...\n\n")
            self.assistant_chat.see("end")
            context = build_app_context(self)
            def work():
                return self.local_llm.answer(question, context)
            future = self.runner.submit(work)
            future.add_done_callback(lambda f: self.worker_queue.put(("assistant_answer", (f, placeholder_start), "info")))
        else:
            answer = self.assistant.answer(question, build_app_context(self))
            self.assistant_chat.insert("end", f"Assistant: {answer}\n\n")
            self.assistant_chat.see("end")

    def _ask_topic(self, topic):
        self.assistant_question.set(topic)
        self.ask_assistant()

    def _poll_queue(self):
        while True:
            try:
                event, payload, level = self.worker_queue.get_nowait()
            except queue.Empty:
                break
            if event == "log":
                self.log(payload, level)
            elif event == "trained":
                try:
                    metadata = payload.result()
                    self.manifest = self.manager.load_manifest(self.project_dir)
                    self.visible_metrics = metadata["metrics"]
                    self.validation_text.delete("1.0", "end")
                    self.validation_text.insert("end", json.dumps(metadata["metrics"], indent=2))
                    self.workflow["trained"] = True
                    self._update_tab_states()
                    self.refresh_model_history()
                    self.log("Training artifacts saved and active model updated. Validation, Predict, and Model History are now available.")
                    self._select_tab("Validate Model")
                except Exception as exc:
                    self.log(f"Training failed: {exc}", "error")
                    messagebox.showerror(APP_NAME, self._friendly_error(exc))
            elif event == "assistant_answer":
                future, placeholder_start = payload
                try:
                    answer = future.result()
                except Exception as exc:
                    answer = f"Local LLM request failed: {exc}"
                end_index = self.assistant_chat.index("end-1c")
                self.assistant_chat.delete(placeholder_start, end_index)
                self.assistant_chat.insert(placeholder_start, f"Assistant: {answer}\n\n")
                self.assistant_chat.see("end")
            elif event == "cst_batch_done":
                try:
                    result_df = payload.result()
                    csv_path = self.project_dir / "data" / "cst_generated_dataset.csv"
                    result_df.to_csv(csv_path, index=False)
                    self.cst_result_csv_path = csv_path
                    self.workflow["cst_run"] = True
                    self._update_tab_states()
                    self.cst_output_text.insert("end", f"\nCompleted {len(result_df)} CST runs.\nSaved to {csv_path}\n")
                    self.cst_load_button.configure(state="normal")
                    self.log(f"CST batch complete. {len(result_df)} samples saved to {csv_path}.")
                except Exception as exc:
                    self.cst_output_text.insert("end", f"\nCST batch failed: {exc}\n")
                    self.log(f"CST batch failed: {exc}", "error")
                    messagebox.showerror(APP_NAME, self._friendly_error(exc))
        self.after(200, self._poll_queue)

    def log(self, message, level="info"):
        line = self.logger.write(level, message)
        if hasattr(self, "log_text"):
            tag = level if level in {"error", "warning"} else "info"
            self.log_text.insert("end", line + "\n", tag)
            self.log_text.see("end")


def main():
    app = AntennaSurrogateStudio()
    app.mainloop()
