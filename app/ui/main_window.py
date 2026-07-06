import json
import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from app import APP_NAME, APP_VERSION
from app.assistant.context_builder import build_app_context
from app.assistant.offline_help import OfflineGuideBackend
from app.core.compatibility import assess_project_compatibility
from app.core.logging_manager import ProjectLogger
from app.core.project_manager import ProjectManager
from app.core.schema_manager import infer_output_axis, save_schema
from app.utils.paths import APP_DIR, PROJECTS_DIR, ensure_app_dirs
from app.utils.threading_utils import BackgroundRunner


class AntennaSurrogateStudio(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1280x860")
        self.minsize(1080, 760)
        ensure_app_dirs()
        self.app_version = APP_VERSION
        self.manager = ProjectManager(PROJECTS_DIR)
        self.project_dir = None
        self.manifest = None
        self.imported_df = None
        self.sample_count = 0
        self.visible_warnings = []
        self.visible_metrics = {}
        self.workflow = {"project": False, "imported": False, "prepared": False, "diagnostics_reviewed": False, "trained": False}
        self.tab_titles = []
        self.current_page_name = "Library"
        self._internal_tab_change = False
        self.worker_queue = queue.Queue()
        self.runner = BackgroundRunner()
        self.logger = ProjectLogger()
        self.assistant = OfflineGuideBackend(APP_DIR / "app" / "assistant" / "knowledge_base")
        self._vars()
        self._style()
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

    def _style(self):
        self.configure(bg="#f4f6f8")
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#f4f6f8")
        style.configure("Card.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        style.configure("TLabel", background="#f4f6f8", foreground="#1f2937", font=("Segoe UI", 10))
        style.configure("Card.TLabel", background="#ffffff", foreground="#1f2937", font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#f4f6f8", foreground="#111827", font=("Segoe UI Semibold", 20))
        style.configure("Subtle.TLabel", background="#f4f6f8", foreground="#64748b", font=("Segoe UI", 9))
        style.configure("CardTitle.TLabel", background="#ffffff", foreground="#111827", font=("Segoe UI Semibold", 13))
        style.configure("TButton", font=("Segoe UI Semibold", 10), padding=(12, 7))
        style.configure("Accent.TButton", background="#0f766e", foreground="#ffffff")
        style.map("Accent.TButton", background=[("active", "#115e59"), ("disabled", "#99f6e4")])
        style.configure("TNotebook.Tab", font=("Segoe UI Semibold", 10), padding=(14, 9))
        style.map("TNotebook.Tab", foreground=[("disabled", "#94a3b8")])

    def _layout(self):
        shell = ttk.Frame(self, padding=(16, 12))
        shell.pack(fill="both", expand=True)
        shell.rowconfigure(1, weight=1)
        shell.columnconfigure(0, weight=1)
        header = ttk.Frame(shell)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(header, text=APP_NAME, style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Build, validate, and use local surrogate models from antenna simulation data.", style="Subtle.TLabel").pack(anchor="w")
        self.notebook = ttk.Notebook(shell)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        self.notebook.bind("<<NotebookTabChanged>>", self._tab_changed)
        pages = [
            ("Library", self._build_library_page),
            ("Project Setup", self._build_project_page),
            ("Import & Configure Data", self._build_import_page),
            ("Diagnostics", self._build_diagnostics_page),
            ("Train Model", self._build_training_page),
            ("Validate Model", self._build_validation_page),
            ("Predict", self._build_prediction_page),
            ("Model History", self._build_model_history_page),
            ("Assistant", self._build_assistant_page),
        ]
        self.tab_titles = [title for title, _builder in pages]
        for title, builder in pages:
            frame = ttk.Frame(self.notebook, padding=14)
            self.notebook.add(frame, text=title)
            builder(frame)
        self._build_log(shell)
        self._update_tab_states()

    def _card(self, parent, row, column=0, **grid):
        card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        card.grid(row=row, column=column, sticky=grid.pop("sticky", "nsew"), **grid)
        return card

    def _allowed_tab_indices(self):
        allowed = {0, 1, 8}
        if self.workflow.get("project"):
            allowed.add(2)
        if self.workflow.get("prepared"):
            allowed.add(3)
        if self.workflow.get("prepared") and self.workflow.get("diagnostics_reviewed"):
            allowed.add(4)
        if self.workflow.get("trained"):
            allowed.update({5, 6, 7})
        return allowed

    def _update_tab_states(self):
        if not hasattr(self, "notebook") or not self.tab_titles:
            return
        allowed = self._allowed_tab_indices()
        current = self.notebook.index(self.notebook.select()) if self.notebook.tabs() else 0
        for index, _title in enumerate(self.tab_titles):
            self.notebook.tab(index, state="normal" if index in allowed else "disabled")
        if current not in allowed:
            self._select_tab(max(i for i in allowed if i <= current) if any(i <= current for i in allowed) else 0)

    def _select_tab(self, title_or_index):
        index = self.tab_titles.index(title_or_index) if isinstance(title_or_index, str) else int(title_or_index)
        if index not in self._allowed_tab_indices():
            self._show_step_lock_message(index)
            return False
        self._internal_tab_change = True
        self.notebook.select(index)
        self._internal_tab_change = False
        self.current_page_name = self.tab_titles[index]
        return True

    def _show_step_lock_message(self, index):
        title = self.tab_titles[index] if 0 <= index < len(self.tab_titles) else "that step"
        message = self._step_lock_reason(index)
        self.log(f"{title} is locked: {message}", "warning")
        messagebox.showinfo(APP_NAME, message)

    def _step_lock_reason(self, index):
        if index == 2:
            return "Create or open a project before importing data."
        if index == 3:
            return "Import data, select valid input/output columns, and click Prepare Dataset before reviewing diagnostics."
        if index == 4:
            return "Review the Data Diagnostics page and click Continue to Train before training."
        if index in {5, 6, 7}:
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
        frame = ttk.Frame(parent, style="Card.TFrame", padding=8)
        frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text="Messages", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.log_text = tk.Text(frame, height=5, wrap="word", font=("Consolas", 9), bg="#ffffff", relief="flat")
        self.log_text.grid(row=1, column=0, sticky="ew")
        ttk.Button(frame, text="Copy Error Details", command=lambda: self.clipboard_append(self.log_text.get("1.0", "end"))).grid(row=1, column=1, padx=(8, 0))

    def _build_library_page(self, parent):
        parent.columnconfigure(0, weight=1)
        actions = self._card(parent, 0, sticky="ew")
        ttk.Label(actions, text="Project Library", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(actions, text="New Project", style="Accent.TButton", command=lambda: self._select_tab("Project Setup")).grid(row=1, column=0, pady=(12, 0), sticky="w")
        ttk.Button(actions, text="Open Existing Project", command=self.open_project_dialog).grid(row=1, column=1, padx=8, pady=(12, 0))
        ttk.Button(actions, text="Import Project ZIP", command=self.import_project_zip).grid(row=1, column=2, pady=(12, 0))
        self.library_frame = self._card(parent, 1, sticky="nsew", pady=(12, 0))
        parent.rowconfigure(1, weight=1)

    def _build_project_page(self, parent):
        parent.columnconfigure(0, weight=1)
        card = self._card(parent, 0, sticky="ew")
        ttk.Label(card, text="Create Project", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", columnspan=2)
        fields = [("Project name", self.project_name), ("Description", self.project_description), ("Antenna category", self.antenna_category), ("Units preference", self.units_preference)]
        for i, (label, var) in enumerate(fields, start=1):
            ttk.Label(card, text=label, style="Card.TLabel").grid(row=i, column=0, sticky="w", pady=6)
            ttk.Entry(card, textvariable=var, width=58).grid(row=i, column=1, sticky="ew", pady=6)
        ttk.Label(card, text="Project type", style="Card.TLabel").grid(row=5, column=0, sticky="w", pady=6)
        ttk.Combobox(card, textvariable=self.project_type, values=["S-parameter prediction", "Radiation-pattern prediction", "Gain / efficiency prediction", "Multi-metric antenna prediction", "Custom surrogate model"], state="readonly").grid(row=5, column=1, sticky="ew", pady=6)
        card.columnconfigure(1, weight=1)
        ttk.Button(card, text="Create Project", style="Accent.TButton", command=self.create_project).grid(row=6, column=1, sticky="e", pady=(10, 0))

    def _build_import_page(self, parent):
        parent.columnconfigure(0, weight=1)
        source = self._card(parent, 0, sticky="ew")
        ttk.Label(source, text="Import Dataset", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", columnspan=4)
        ttk.Radiobutton(source, text="Single wide CSV", variable=self.import_mode, value="wide").grid(row=1, column=0, sticky="w", pady=8)
        ttk.Entry(source, textvariable=self.wide_csv_path).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(source, text="Browse", command=lambda: self._browse(self.wide_csv_path)).grid(row=1, column=2)
        ttk.Radiobutton(source, text="Split input/output CSV", variable=self.import_mode, value="split").grid(row=2, column=0, sticky="w")
        ttk.Entry(source, textvariable=self.inputs_csv_path).grid(row=2, column=1, sticky="ew", padx=8)
        ttk.Button(source, text="Inputs", command=lambda: self._browse(self.inputs_csv_path)).grid(row=2, column=2)
        ttk.Entry(source, textvariable=self.outputs_csv_path).grid(row=3, column=1, sticky="ew", padx=8, pady=5)
        ttk.Button(source, text="Outputs", command=lambda: self._browse(self.outputs_csv_path)).grid(row=3, column=2)
        ttk.Entry(source, textvariable=self.sample_id_column, width=16).grid(row=3, column=3, padx=8)
        ttk.Button(source, text="Load Columns", style="Accent.TButton", command=self.load_dataset).grid(row=4, column=2, sticky="e", pady=8)
        source.columnconfigure(1, weight=1)
        lists = ttk.Frame(parent)
        lists.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        lists.columnconfigure(0, weight=1)
        lists.columnconfigure(1, weight=1)
        self.input_list = tk.Listbox(lists, selectmode="extended", exportselection=False, height=14)
        self.output_list = tk.Listbox(lists, selectmode="extended", exportselection=False, height=14)
        ttk.Label(lists, text="Input design variables").grid(row=0, column=0, sticky="w")
        ttk.Label(lists, text="Output response columns").grid(row=0, column=1, sticky="w")
        self.input_list.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.output_list.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        ttk.Button(parent, text="Prepare Dataset", style="Accent.TButton", command=self.prepare_dataset).grid(row=2, column=0, sticky="e", pady=10)

    def _build_diagnostics_page(self, parent):
        parent.columnconfigure(0, weight=1)
        card = self._card(parent, 0, sticky="nsew")
        parent.rowconfigure(0, weight=1)
        header = ttk.Frame(card, style="Card.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="Data Diagnostics", style="CardTitle.TLabel").pack(side="left", anchor="w")
        ttk.Button(header, text="Continue to Train", style="Accent.TButton", command=self.accept_diagnostics).pack(side="right")
        self.diagnostics_text = tk.Text(card, wrap="word", height=26, bg="#ffffff", relief="flat")
        self.diagnostics_text.pack(fill="both", expand=True, pady=(10, 0))

    def _build_training_page(self, parent):
        parent.columnconfigure(0, weight=1)
        card = self._card(parent, 0, sticky="ew")
        ttk.Label(card, text="Train XGBoost Model", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", columnspan=4)
        settings = [("n_estimators", self.n_estimators), ("max_depth", self.max_depth), ("learning_rate", self.learning_rate), ("subsample", self.subsample), ("colsample_bytree", self.colsample), ("min_child_weight", self.min_child_weight), ("test split", self.test_split), ("random seed", self.random_seed), ("CV folds", self.cv_folds)]
        for i, (label, var) in enumerate(settings, start=1):
            ttk.Label(card, text=label, style="Card.TLabel").grid(row=i, column=0, sticky="w", pady=4)
            ttk.Entry(card, textvariable=var, width=14).grid(row=i, column=1, sticky="w", pady=4)
        ttk.Button(card, text="Train New Model Version", style="Accent.TButton", command=self.train_model).grid(row=len(settings) + 1, column=1, sticky="w", pady=10)

    def _build_validation_page(self, parent):
        parent.columnconfigure(0, weight=1)
        self.validation_text = tk.Text(self._card(parent, 0, sticky="nsew"), wrap="word", height=28, bg="#ffffff", relief="flat")
        self.validation_text.pack(fill="both", expand=True)
        parent.rowconfigure(0, weight=1)

    def _build_prediction_page(self, parent):
        parent.columnconfigure(0, weight=1)
        self.prediction_form = self._card(parent, 0, sticky="ew")
        ttk.Label(self.prediction_form, text="Predict New Design", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", columnspan=3)
        ttk.Button(self.prediction_form, text="Refresh Fields", command=self.refresh_prediction_fields).grid(row=0, column=2, sticky="e")
        ttk.Button(self.prediction_form, text="Predict", style="Accent.TButton", command=self.predict_one).grid(row=99, column=1, sticky="w", pady=10)
        ttk.Button(self.prediction_form, text="Batch CSV", command=self.predict_batch).grid(row=99, column=2, sticky="w", pady=10)
        self.prediction_output = tk.Text(self._card(parent, 1, sticky="nsew", pady=(12, 0)), wrap="none", height=18, bg="#ffffff", relief="flat")
        self.prediction_output.pack(fill="both", expand=True)
        parent.rowconfigure(1, weight=1)

    def _build_model_history_page(self, parent):
        parent.columnconfigure(0, weight=1)
        card = self._card(parent, 0, sticky="nsew")
        ttk.Label(card, text="Model History", style="CardTitle.TLabel").pack(anchor="w")
        self.model_tree = ttk.Treeview(card, columns=("version", "type", "rmse", "r2", "active"), show="headings", height=14)
        for col in ("version", "type", "rmse", "r2", "active"):
            self.model_tree.heading(col, text=col.title())
        self.model_tree.pack(fill="both", expand=True, pady=8)
        ttk.Button(card, text="Activate Selected Version", command=self.activate_selected_model).pack(anchor="e")

    def _build_assistant_page(self, parent):
        parent.columnconfigure(0, weight=1)
        card = self._card(parent, 0, sticky="nsew")
        ttk.Label(card, text="Assistant", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(card, text="Assistant responses are generated from local product documentation and optional local models. Your project data is not sent to cloud services by this application.", style="Card.TLabel", wraplength=960).pack(anchor="w", pady=6)
        self.assistant_chat = tk.Text(card, wrap="word", height=20, bg="#ffffff", relief="flat")
        self.assistant_chat.pack(fill="both", expand=True, pady=8)
        entry = ttk.Frame(card, style="Card.TFrame")
        entry.pack(fill="x")
        ttk.Entry(entry, textvariable=self.assistant_question).pack(side="left", fill="x", expand=True)
        ttk.Button(entry, text="Ask", style="Accent.TButton", command=self.ask_assistant).pack(side="left", padx=(8, 0))
        ttk.Button(entry, text="Clear", command=lambda: self.assistant_chat.delete("1.0", "end")).pack(side="left", padx=(8, 0))
        for topic in ["How do I create a new project?", "Which CSV format should I use?", "What is RMSE?", "How do I make a prediction?"]:
            ttk.Button(card, text=topic, command=lambda t=topic: self._ask_topic(t)).pack(anchor="w", pady=2)
        parent.rowconfigure(0, weight=1)

    def _load_examples(self):
        try:
            from app.core.example_data import ensure_example_projects
            ensure_example_projects(PROJECTS_DIR)
        except Exception as exc:
            self.log(f"Example setup skipped: {exc}", "warning")

    def _refresh_library(self):
        for child in self.library_frame.winfo_children():
            child.destroy()
        ttk.Label(self.library_frame, text="Recent and Example Projects", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        for i, project in enumerate(self.manager.recent_projects(), start=1):
            try:
                manifest = self.manager.load_manifest(project)
                models = manifest.model_versions
                active = next((m for m in models if m.get("version") == manifest.active_model_version), {}) if models else {}
                text = f"{manifest.project_name}\nType: {manifest.project_type}\nInputs: {len(manifest.selected_input_columns)}  Outputs: {len(manifest.selected_output_columns)}  Samples: {self._project_samples(project)}\nModel: {active.get('model_type', 'None')}  RMSE: {active.get('overall_rmse', '--')}  R2: {active.get('overall_r2', '--')}"
                card = ttk.Frame(self.library_frame, style="Card.TFrame", padding=10)
                card.grid(row=i, column=0, sticky="ew", pady=6)
                ttk.Label(card, text=text, style="Card.TLabel", justify="left").pack(side="left", fill="x", expand=True)
                ttk.Button(card, text="Open", command=lambda p=project: self.open_project(p)).pack(side="right")
            except Exception as exc:
                ttk.Label(self.library_frame, text=f"{project.name}: {exc}", style="Card.TLabel").grid(row=i, column=0, sticky="w")
        self.library_frame.columnconfigure(0, weight=1)

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
            self.input_list.delete(0, "end")
            self.output_list.delete(0, "end")
            for col in self.imported_df.columns:
                self.input_list.insert("end", col)
                self.output_list.insert("end", col)
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
        inputs = [self.input_list.get(i) for i in self.input_list.curselection()]
        outputs = [self.output_list.get(i) for i in self.output_list.curselection()]
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
        for model in self.manifest.model_versions:
            active = "yes" if model.get("version") == self.manifest.active_model_version else ""
            self.model_tree.insert("", "end", values=(model.get("version"), model.get("model_type"), round(model.get("overall_rmse", 0), 6), round(model.get("overall_r2", 0), 6), active))

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
        if not hasattr(self, "prediction_form"):
            return
        for child in list(self.prediction_form.grid_slaves()):
            info = child.grid_info()
            if int(info.get("row", 0)) not in {0, 99}:
                child.destroy()
        self.prediction_entries = {}
        if not self.manifest:
            return
        for row, col in enumerate(self.manifest.selected_input_columns, start=1):
            ttk.Label(self.prediction_form, text=col, style="Card.TLabel").grid(row=row, column=0, sticky="w", pady=4)
            var = tk.StringVar()
            bounds = self.manifest.input_bounds.get(col, {})
            if bounds:
                var.set(str(round((bounds["min"] + bounds["max"]) / 2, 6)))
            ttk.Entry(self.prediction_form, textvariable=var, width=20).grid(row=row, column=1, sticky="w", pady=4)
            ttk.Label(self.prediction_form, text=f"range {bounds.get('min', '--')} to {bounds.get('max', '--')}", style="Card.TLabel").grid(row=row, column=2, sticky="w", padx=8)
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
            messagebox.showerror(APP_NAME, str(exc))

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
            messagebox.showerror(APP_NAME, str(exc))

    def ask_assistant(self):
        question = self.assistant_question.get().strip()
        if not question:
            return
        answer = self.assistant.answer(question, build_app_context(self))
        self.assistant_chat.insert("end", f"You: {question}\nAssistant: {answer}\n\n")
        self.assistant_question.set("")

    def _ask_topic(self, topic):
        self.assistant_question.set(topic)
        self.ask_assistant()

    def _tab_changed(self, _event):
        selected = self.notebook.index(self.notebook.select())
        if selected not in self._allowed_tab_indices() and not self._internal_tab_change:
            self._show_step_lock_message(selected)
            self._update_tab_states()
            return
        self.current_page_name = self.notebook.tab(self.notebook.select(), "text")

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
                    messagebox.showerror(APP_NAME, str(exc))
        self.after(200, self._poll_queue)

    def log(self, message, level="info"):
        line = self.logger.write(level, message)
        if hasattr(self, "log_text"):
            self.log_text.insert("end", line + "\n")
            self.log_text.see("end")


def main():
    app = AntennaSurrogateStudio()
    app.mainloop()
