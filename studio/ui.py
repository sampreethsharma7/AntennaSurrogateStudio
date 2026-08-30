"""Premium CustomTkinter shell for the first Studio product slice."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
import webbrowser
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Callable

import customtkinter as ctk

from studio import __version__
from studio.assistant import (
    MODEL_PROFILES,
    AssistantError,
    RuntimeStatus,
    SnowBuddyService,
    model_profile,
    total_memory_gb,
)
from studio.dataset_registry import (
    RegisteredDataset,
    get_registered_dataset,
    register_dataset,
)
from studio.dataset_validation import (
    DatasetValidationResult,
    read_dataset_columns,
    validate_dataset,
)
from studio.library_ui import ModelLibraryPage
from studio.inference_ui import InferencePage
from studio.inverse_design_ui import (
    INVERSE_DESIGN_MIN_WORKSPACE_WIDTH,
    InverseDesignPage,
)
from studio.model_training import (
    NEURAL_NETWORK_CUSTOM_DEFAULTS,
    NEURAL_NETWORK_CUSTOM_PARAMETER_NAMES,
    XGBOOST_CUSTOM_DEFAULTS,
    XGBOOST_CUSTOM_PARAMETER_NAMES,
    ModelTrainingRequest,
    ModelTrainingResult,
    submit_model_training_request,
)
from studio.parser_engine import (
    IMPORTED_OUTPUT_LABEL,
    DiscoveryResult,
    ParseError,
    PreparedResult,
    TrainingRequest,
    discover,
    prepare,
    write_input_output_templates,
)
from studio.project_store import (
    Project,
    ProjectError,
    ProjectStore,
    atomic_write_json,
)
from studio.results_ui import TrainingResultsPage
from studio.sample_generator_ui import LHSSampleGeneratorDialog
from studio.settings import load_appearance_mode, save_appearance_mode
from studio.theme import COLORS, FONTS, status_palette
from studio.training_ui import (
    AUTO_SEARCH_DESCRIPTIONS,
    AUTO_SEARCH_LEVELS,
    MODEL_REQUEST_NAMES,
    SEARCH_LEVEL_REQUEST_NAMES,
    SUPPORTED_MODELS,
    TRAIN_BUTTON_LABEL,
    TRAINING_MODES,
    TRAINING_MODE_REQUEST_NAMES,
    TrainingPageState,
)


ctk.set_default_color_theme("blue")


DESIGN_MIN_WIDTH = 1260
DESIGN_MIN_HEIGHT = 700
DEFAULT_WINDOW_WIDTH = 1440
DEFAULT_WINDOW_HEIGHT = 900
MAX_EFFECTIVE_UI_SCALE = 1.08
SNOWBUDDY_PANEL_WIDTH = 390
MIN_DOCKED_PAGE_WIDTH = 980


@dataclass(frozen=True, slots=True)
class ResponsiveWindowLayout:
    """Physical window geometry plus bounded CustomTkinter scaling factors."""

    width: int
    height: int
    min_width: int
    min_height: int
    x: int
    y: int
    dpi_scaling: float
    ui_scaling: float
    widget_scaling_factor: float
    window_scaling_factor: float
    compact: bool


def responsive_window_layout(
    screen_width: int,
    screen_height: int,
    dpi_scaling: float,
) -> ResponsiveWindowLayout:
    """Return DPI-safe physical geometry for the current monitor.

    Tk reports logical monitor dimensions on DPI-aware Windows. CustomTkinter
    then scales window geometry a second time unless its window factor is
    normalized. The bounded UI scale keeps the full no-scroll workflow usable
    in a 1366x768 physical viewport at 100%, 125%, and 150% Windows scaling.
    """

    dpi = max(1.0, float(dpi_scaling or 1.0))
    physical_width = max(1, int(round(screen_width * dpi)))
    physical_height = max(1, int(round(screen_height * dpi)))
    width = min(DEFAULT_WINDOW_WIDTH, physical_width)
    height = min(DEFAULT_WINDOW_HEIGHT, physical_height)
    ui_scaling = min(
        dpi,
        MAX_EFFECTIVE_UI_SCALE,
        max(0.8, width / DESIGN_MIN_WIDTH),
        max(0.8, height / DESIGN_MIN_HEIGHT),
    )
    min_width = min(width, int(round(DESIGN_MIN_WIDTH * ui_scaling)))
    min_height = min(height, int(round(DESIGN_MIN_HEIGHT * ui_scaling)))
    return ResponsiveWindowLayout(
        width=width,
        height=height,
        min_width=min_width,
        min_height=min_height,
        x=max(0, (physical_width - width) // 2),
        y=max(0, (physical_height - height) // 2),
        dpi_scaling=dpi,
        ui_scaling=ui_scaling,
        widget_scaling_factor=ui_scaling / dpi,
        window_scaling_factor=1.0 / dpi,
        compact=width <= 1366 or height <= 768,
    )


def _window_dpi_scaling(window: tk.Misc) -> float:
    if os.name == "nt":
        try:
            import ctypes

            return max(
                1.0,
                ctypes.windll.user32.GetDpiForWindow(window.winfo_id()) / 96.0,
            )
        except (AttributeError, OSError, ValueError):
            pass
    return max(1.0, float(getattr(window, "_window_scaling", 1.0)))


class HoverTooltip:
    """Themed hover/focus text for compact navigation controls."""

    def __init__(
        self,
        widget: ctk.CTkBaseClass,
        text: str,
        *,
        enabled: Callable[[], bool] | None = None,
    ):
        self.widget = widget
        self.text = text
        self.enabled = enabled or (lambda: True)
        self.window: tk.Toplevel | None = None
        setattr(widget, "accessible_name", text)
        widget.bind("<Enter>", self.show, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<FocusIn>", self.show, add="+")
        widget.bind("<FocusOut>", self.hide, add="+")
        widget.bind("<Unmap>", self.hide, add="+")

    def show(self, _event: tk.Event | None = None) -> None:
        if not self.enabled() or self.window is not None:
            return
        tooltip = tk.Toplevel(self.widget)
        tooltip.overrideredirect(True)
        try:
            tooltip.attributes("-topmost", True)
        except tk.TclError:
            pass
        index = 0 if ctk.get_appearance_mode() == "Light" else 1
        tk.Label(
            tooltip,
            text=self.text,
            bg=COLORS["ink"][index],
            fg=COLORS["surface"][index],
            padx=9,
            pady=5,
            font=("Segoe UI", 12),
        ).pack()
        tooltip.update_idletasks()
        tooltip.geometry(
            f"+{self.widget.winfo_rootx() + self.widget.winfo_width() + 8}"
            f"+{self.widget.winfo_rooty()}"
        )
        self.window = tooltip

    def hide(self, _event: tk.Event | None = None) -> None:
        if self.window is not None:
            try:
                self.window.destroy()
            except tk.TclError:
                pass
        self.window = None


def project_resume_destination(project: Project) -> tuple[str, str]:
    """Return the next page and action label for the persisted workflow stage."""

    if project.workflow_stage == "model_saved":
        active_book_id = (
            project.manifest.get("model_library", {}).get("active_book_id")
        )
        if active_book_id:
            return "inference", "Run Inference  →"
        return "library", "Open Model Library  →"
    return {
        "project_created": ("data", "Continue Data Prep  →"),
        "data_discovered": ("data", "Continue Data Prep  →"),
        "data_prepared": ("data", "Validate & Register Data  →"),
        "dataset_registered": ("training", "Continue Model Training  →"),
        "model_trained": ("results", "Review Training Results  →"),
    }.get(project.workflow_stage, ("data", "Resume Project  →"))


class StudioApp(ctk.CTk):
    def __init__(self, project_store: ProjectStore | None = None):
        resolved_store = project_store or ProjectStore()
        appearance_mode = load_appearance_mode(resolved_store.library_root)
        ctk.set_appearance_mode(appearance_mode)
        super().__init__(fg_color=COLORS["app_bg"])
        self.store = resolved_store
        self.appearance_mode = appearance_mode
        self.title("Antenna Surrogate Studio")
        self.window_layout = responsive_window_layout(
            self.winfo_screenwidth(),
            self.winfo_screenheight(),
            _window_dpi_scaling(self),
        )
        ctk.set_window_scaling(self.window_layout.window_scaling_factor)
        ctk.set_widget_scaling(self.window_layout.widget_scaling_factor)
        self.ui_scaling = self.window_layout.ui_scaling
        self.minsize(
            self.window_layout.min_width,
            self.window_layout.min_height,
        )
        self.geometry(
            f"{self.window_layout.width}x{self.window_layout.height}"
            f"+{self.window_layout.x}+{self.window_layout.y}"
        )
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self._destroying = False

        self.current_project: Project | None = None
        self.snowbuddy = SnowBuddyService(self.store)
        self.nav_buttons: dict[str, ctk.CTkButton] = {}
        self.pages: dict[str, ctk.CTkFrame] = {}
        self.active_page = "start"
        self.sidebar_collapsed = False
        self.snowbuddy_collapsed = True
        self.snowbuddy_display_mode = "hidden"
        self.nav_tooltips: dict[str, HoverTooltip] = {}

        self._set_windows_identity()
        self._build_shell()
        if self.window_layout.compact:
            self.set_sidebar_collapsed(True)
        self.bind("<Configure>", self._window_resized, add="+")
        self.start_page.refresh()
        self.snowbuddy_panel.load_project(None)
        self.show_page("start", persist=False)

    def _set_windows_identity(self) -> None:
        if os.name != "nt":
            return
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "AntennaSurrogateStudio.Desktop.0.1"
            )
        except Exception:
            pass

    def destroy(self) -> None:
        """Cancel deferred Tk callbacks before destroying the widget tree."""

        if self._destroying:
            return
        self._destroying = True
        if hasattr(self, "snowbuddy_panel"):
            self.snowbuddy_panel.cancel_pending_callbacks()
        # ``after`` callbacks belong to the widget that registered their Tcl
        # command, but all of them share the root interpreter's scheduler.
        # Cancel the scheduled jobs at Tcl level and let normal widget
        # destruction delete each command from its real owner.  Calling
        # ``self.after_cancel`` here would remove child-owned commands through
        # the root and leave stale entries in those children's command lists.
        try:
            pending_callbacks = self.tk.splitlist(self.tk.call("after", "info"))
        except tk.TclError:
            pending_callbacks = ()
        for callback_id in pending_callbacks:
            try:
                self.tk.call("after", "cancel", callback_id)
            except tk.TclError:
                pass
        super().destroy()

    def _build_shell(self) -> None:
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_menu_bar()
        self._build_sidebar()

        self.workflow_divider = ctk.CTkFrame(
            self,
            width=2,
            corner_radius=0,
            fg_color=COLORS["border_strong"],
        )
        self.workflow_divider.grid(row=1, column=1, sticky="ns")

        self.workspace = ctk.CTkFrame(
            self,
            fg_color=COLORS["app_bg"],
            corner_radius=0,
        )
        self.workspace.grid(row=1, column=2, sticky="nsew")
        self.workspace.grid_columnconfigure(0, weight=1)
        self.workspace.grid_columnconfigure(1, weight=0, minsize=0)
        self.workspace.grid_rowconfigure(0, weight=1)

        self.page_host = ctk.CTkFrame(
            self.workspace,
            fg_color=COLORS["app_bg"],
            corner_radius=0,
        )
        self.page_host.grid(row=0, column=0, sticky="nsew")
        self.page_host.grid_columnconfigure(0, weight=1)
        self.page_host.grid_rowconfigure(0, weight=1)

        self.pages["start"] = StartPage(self.page_host, self)
        self.pages["data"] = DataPrepPage(self.page_host, self)
        self.pages["training"] = ModelTrainingPage(self.page_host, self)
        self.pages["results"] = TrainingResultsPage(self.page_host, self)
        self.pages["library"] = ModelLibraryPage(self.page_host, self)
        self.pages["inference"] = InferencePage(self.page_host, self)
        self.pages["inverse_design"] = InverseDesignPage(self.page_host, self)
        for page in self.pages.values():
            page.grid(row=0, column=0, sticky="nsew")

        self.snowbuddy_panel = SnowBuddyPanel(self.workspace, self)
        self.snowbuddy_panel.grid(
            row=0,
            column=1,
            padx=(0, 18),
            pady=18,
            sticky="nsew",
        )
        self.snowbuddy_restore_button = self.snowbuddy_menu_button
        self.set_snowbuddy_collapsed(True)

    def set_snowbuddy_collapsed(self, collapsed: bool) -> None:
        """Minimize or restore the companion without changing its chat state."""

        self.snowbuddy_collapsed = bool(collapsed)
        if self.snowbuddy_collapsed:
            self.snowbuddy_panel.grid_remove()
            self.workspace.grid_columnconfigure(1, minsize=0)
            self.page_host.grid()
            self.snowbuddy_display_mode = "hidden"
            self.snowbuddy_restore_button.configure(
                text="✦  SnowBuddy",
                command=lambda: self.set_snowbuddy_collapsed(False),
            )
            self.snowbuddy_restore_button.accessible_name = "Open SnowBuddy"
        else:
            self._place_snowbuddy_panel(force=True)
            self.snowbuddy_restore_button.configure(
                text="Close SnowBuddy",
                command=lambda: self.set_snowbuddy_collapsed(True),
            )
            self.snowbuddy_restore_button.accessible_name = "Close SnowBuddy"

    def _snowbuddy_can_dock(self) -> bool:
        workspace_width = self.workspace.winfo_width()
        if workspace_width <= 1:
            workspace_width = max(1, self.winfo_width() - self.sidebar.winfo_width())
        minimum_page_width = (
            INVERSE_DESIGN_MIN_WORKSPACE_WIDTH
            if getattr(self, "active_page", "") == "inverse_design"
            else MIN_DOCKED_PAGE_WIDTH
        )
        required = int(
            round(
                (SNOWBUDDY_PANEL_WIDTH + minimum_page_width + 36)
                * self.ui_scaling
            )
        )
        return workspace_width >= required

    def _place_snowbuddy_panel(self, *, force: bool = False) -> None:
        if self.snowbuddy_collapsed:
            return
        mode = "docked" if self._snowbuddy_can_dock() else "focus"
        if mode == self.snowbuddy_display_mode and not force:
            return
        if mode == "docked":
            self.page_host.grid()
            self.workspace.grid_columnconfigure(1, minsize=SNOWBUDDY_PANEL_WIDTH)
            self.snowbuddy_panel.grid(
                row=0,
                column=1,
                padx=(0, 18),
                pady=18,
                sticky="nsew",
            )
        else:
            # A narrow technical workspace cannot show a useful plot and the
            # assistant simultaneously. Give SnowBuddy a temporary focus view;
            # closing it restores the unchanged page at its full width.
            self.page_host.grid_remove()
            self.workspace.grid_columnconfigure(1, minsize=0)
            self.snowbuddy_panel.grid(
                row=0,
                column=0,
                padx=18,
                pady=18,
                sticky="ns",
            )
        self.snowbuddy_display_mode = mode
        self.snowbuddy_panel.tkraise()

    def _window_resized(self, event: tk.Event) -> None:
        if event.widget is self and not self.snowbuddy_collapsed:
            self._place_snowbuddy_panel()

    def _build_menu_bar(self) -> None:
        menu_bar = ctk.CTkFrame(
            self,
            height=42,
            corner_radius=0,
            fg_color=COLORS["surface"],
            border_width=1,
            border_color=COLORS["border"],
        )
        menu_bar.grid(row=0, column=0, columnspan=3, sticky="ew")
        menu_bar.grid_propagate(False)
        menu_bar.grid_columnconfigure(3, weight=1)

        self.top_menu_buttons: dict[str, ctk.CTkButton] = {}
        for column, label in enumerate(("File", "Edit", "Help")):
            button = ctk.CTkButton(
                menu_bar,
                text=label,
                width=58,
                height=30,
                corner_radius=8,
                fg_color="transparent",
                hover_color=COLORS["control_hover"],
                text_color=COLORS["ink"],
                font=FONTS["body_small"],
            )
            button.configure(
                command=lambda name=label.lower(), source=button: (
                    self._show_top_menu(name, source)
                )
            )
            button.grid(row=0, column=column, padx=(8 if column == 0 else 0, 0))
            self.top_menu_buttons[label.lower()] = button

        self.snowbuddy_menu_button = ctk.CTkButton(
            menu_bar,
            text="✦  SnowBuddy",
            width=116,
            height=30,
            corner_radius=9,
            fg_color=COLORS["violet_soft"],
            hover_color=COLORS["violet_hover"],
            text_color=COLORS["on_violet_soft"],
            font=FONTS["caption"],
            command=lambda: self.set_snowbuddy_collapsed(False),
        )
        self.snowbuddy_menu_button.grid(row=0, column=4, padx=(8, 0), sticky="e")
        self.snowbuddy_menu_button.accessible_name = "Open SnowBuddy"

        appearance = ctk.CTkFrame(menu_bar, fg_color="transparent")
        appearance.grid(row=0, column=5, padx=12, sticky="e")
        ctk.CTkLabel(
            appearance,
            text="Appearance",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
        ).pack(side="left", padx=(0, 8))
        self.appearance_control = ctk.CTkSegmentedButton(
            appearance,
            values=["Light", "Dark"],
            width=142,
            height=30,
            corner_radius=9,
            selected_color=COLORS["primary"],
            selected_hover_color=COLORS["primary_hover"],
            unselected_color=COLORS["surface_alt"],
            unselected_hover_color=COLORS["control_hover"],
            text_color=COLORS["ink"],
            font=FONTS["caption"],
            command=self._appearance_changed,
        )
        self.appearance_control.pack(side="left")
        self.appearance_control.set(self.appearance_mode.title())

    def _show_top_menu(
        self,
        name: str,
        source: ctk.CTkButton,
    ) -> None:
        color_index = 0 if self.appearance_mode == "light" else 1
        menu = tk.Menu(
            self,
            tearoff=False,
            background=COLORS["surface"][color_index],
            foreground=COLORS["ink"][color_index],
            activebackground=COLORS["nav_active"][color_index],
            activeforeground=COLORS["ink"][color_index],
            disabledforeground=COLORS["disabled_text"][color_index],
            borderwidth=1,
            relief="solid",
            font=("Segoe UI", 12),
        )
        if name == "file":
            menu.add_command(
                label="New project…",
                command=self.create_project_dialog,
            )
            menu.add_command(
                label="Open project…",
                command=self.open_project_dialog,
            )
            menu.add_separator()
            menu.add_command(
                label="Return to Welcome",
                command=self.return_to_welcome,
                state="normal" if self.current_project else "disabled",
            )
            menu.add_separator()
            menu.add_command(label="Exit", command=self.destroy)
        elif name == "edit":
            menu.add_command(
                label="Editing tools will be added here",
                state="disabled",
            )
        else:
            menu.add_command(
                label="SnowBuddy local model…",
                command=self.snowbuddy_panel._show_model_dialog,
            )
            menu.add_separator()
            menu.add_command(label="About Antenna Surrogate Studio", command=self._show_about)
        try:
            menu.tk_popup(
                source.winfo_rootx(),
                source.winfo_rooty() + source.winfo_height(),
            )
        finally:
            menu.grab_release()

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About Antenna Surrogate Studio",
            (
                f"Antenna Surrogate Studio v{__version__}\n\n"
                "Local compute · Private projects · Reusable surrogate books\n\n"
                "Created by Sai Sampreeth Indharapu\n"
                "sampreethsharma@gmail.com\n"
                "linkedin.com/in/sai-sampreeth-indharapu-ph-d-a98802110/"
            ),
            parent=self,
        )

    def _build_sidebar(self) -> None:
        self.sidebar = ctk.CTkFrame(
            self,
            width=226,
            corner_radius=0,
            fg_color=COLORS["sidebar"],
        )
        self.sidebar.grid(row=1, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_rowconfigure(9, weight=1)

        self.sidebar_brand = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.sidebar_brand.grid(row=0, column=0, padx=16, pady=(22, 30), sticky="ew")
        self.sidebar_brand_badge = ctk.CTkLabel(
            self.sidebar_brand,
            width=42,
            height=42,
            corner_radius=13,
            fg_color=COLORS["primary"],
            text="AS",
            text_color=COLORS["on_accent"],
            font=("Segoe UI Semibold", 21),
        )
        self.sidebar_brand_badge.pack(side="left")
        self.sidebar_brand_text = ctk.CTkFrame(
            self.sidebar_brand,
            fg_color="transparent",
        )
        self.sidebar_brand_text.pack(side="left", padx=(9, 0))
        ctk.CTkLabel(
            self.sidebar_brand_text,
            text="ANTENNA",
            text_color=COLORS["sidebar_ink"],
            font=("Segoe UI Semibold", 17),
            anchor="w",
        ).pack(anchor="w")
        ctk.CTkLabel(
            self.sidebar_brand_text,
            text="RF SURROGATE LAB",
            text_color=COLORS["subtle"],
            font=("Segoe UI", 14),
            anchor="w",
        ).pack(anchor="w")
        self.sidebar_toggle_button = ctk.CTkButton(
            self.sidebar_brand,
            text="‹",
            width=28,
            height=32,
            corner_radius=9,
            fg_color="transparent",
            hover_color=COLORS["sidebar_hover"],
            border_width=1,
            border_color=COLORS["border_strong"],
            text_color=COLORS["ink"],
            font=("Segoe UI Semibold", 22),
            command=lambda: self.set_sidebar_collapsed(not self.sidebar_collapsed),
        )
        self.sidebar_toggle_button.pack(side="right")
        self.sidebar_toggle_button.accessible_name = "Collapse workflow navigation"

        self.sidebar_workflow_label = ctk.CTkLabel(
            self.sidebar,
            text="LAB WORKFLOW",
            text_color=COLORS["cyan"],
            font=FONTS["mono"],
            anchor="w",
        )
        self.sidebar_workflow_label.grid(
            row=1,
            column=0,
            padx=24,
            pady=(0, 8),
            sticky="ew",
        )

        self.nav_specs = {
            "start": ("⌂", "Start"),
            "data": ("≋", "Data Prep"),
            "training": ("◇", "Model Training"),
            "results": ("◎", "Training Results"),
            "library": ("▤", "Model Library"),
            "inference": ("∿", "Inference"),
            "inverse_design": ("⌾", "Inverse Design"),
        }

        self.nav_buttons["start"] = self._nav_button(
            self.sidebar, 2, "⌂", "Start", lambda: self.show_page("start")
        )
        self.nav_buttons["data"] = self._nav_button(
            self.sidebar, 3, "≋", "Data Prep", lambda: self.show_page("data")
        )
        self.nav_buttons["training"] = self._nav_button(
            self.sidebar,
            4,
            "◇",
            "Model Training",
            lambda: self.show_page("training"),
        )
        self.nav_buttons["results"] = self._nav_button(
            self.sidebar,
            5,
            "◎",
            "Training Results",
            lambda: self.show_page("results"),
        )

        self.nav_buttons["library"] = self._nav_button(
            self.sidebar,
            6,
            "▤",
            "Model Library",
            lambda: self.show_page("library"),
        )
        self.nav_buttons["inference"] = self._nav_button(
            self.sidebar,
            7,
            "∿",
            "Inference",
            lambda: self.show_page("inference"),
        )
        self.nav_buttons["inverse_design"] = self._nav_button(
            self.sidebar,
            8,
            "⌾",
            "Inverse Design",
            lambda: self.show_page("inverse_design"),
        )

        self.sidebar_project_shell = ctk.CTkFrame(
            self.sidebar,
            fg_color=COLORS["surface_alt"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["border"],
        )
        self.sidebar_project_shell.grid(
            row=9,
            column=0,
            padx=16,
            pady=(18, 0),
            sticky="ew",
        )
        ctk.CTkLabel(
            self.sidebar_project_shell,
            text="ACTIVE PROJECT",
            text_color=COLORS["cyan"],
            font=FONTS["mono"],
            anchor="w",
        ).pack(fill="x", padx=14, pady=(12, 4))
        self.sidebar_project_name = ctk.CTkLabel(
            self.sidebar_project_shell,
            text="No project open",
            text_color=COLORS["ink"],
            font=("Segoe UI Semibold", 16),
            anchor="w",
            wraplength=160,
        )
        self.sidebar_project_name.pack(fill="x", padx=14)
        self.sidebar_project_status = ctk.CTkLabel(
            self.sidebar_project_shell,
            text="Create or open a project",
            text_color=COLORS["muted"],
            font=("Segoe UI", 14),
            anchor="w",
        )
        self.sidebar_project_status.pack(fill="x", padx=14, pady=(3, 8))
        self.sidebar_return_button = ctk.CTkButton(
            self.sidebar_project_shell,
            text="←  Return to Welcome",
            height=32,
            corner_radius=9,
            fg_color="transparent",
            hover_color=COLORS["sidebar_hover"],
            border_width=1,
            border_color=COLORS["border_strong"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            command=self.return_to_welcome,
        )

        self.sidebar_footer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.sidebar_footer.grid(row=10, column=0, padx=22, pady=20, sticky="ew")
        self.sidebar_version_label = ctk.CTkLabel(
            self.sidebar_footer,
            text=f"Studio Preview  ·  v{__version__}",
            text_color=COLORS["muted"],
            font=("Segoe UI", 14),
            anchor="w",
        )
        self.sidebar_version_label.pack(anchor="w")
        ctk.CTkLabel(
            self.sidebar_footer,
            text="LOCAL COMPUTE · PRIVATE",
            text_color=COLORS["subtle"],
            font=FONTS["mono"],
            anchor="w",
        ).pack(anchor="w", pady=(3, 0))

    def set_sidebar_collapsed(self, collapsed: bool) -> None:
        """Switch between full workflow navigation and compact icon navigation."""

        self.sidebar_collapsed = bool(collapsed)
        self.sidebar_brand_badge.pack_forget()
        self.sidebar_brand_text.pack_forget()
        self.sidebar_toggle_button.pack_forget()
        if self.sidebar_collapsed:
            self.sidebar.configure(width=76)
            self.sidebar_brand.grid_configure(padx=18, pady=(20, 24))
            self.sidebar_toggle_button.configure(text="›")
            self.sidebar_toggle_button.accessible_name = "Expand workflow navigation"
            self.sidebar_toggle_button.pack(side="top")
            self.sidebar_workflow_label.grid_remove()
            self.sidebar_project_shell.grid_remove()
            self.sidebar_footer.grid_remove()
        else:
            self.sidebar.configure(width=226)
            self.sidebar_brand.grid_configure(padx=16, pady=(22, 30))
            self.sidebar_brand_badge.pack(side="left")
            self.sidebar_brand_text.pack(side="left", padx=(9, 0))
            self.sidebar_toggle_button.configure(text="‹")
            self.sidebar_toggle_button.accessible_name = "Collapse workflow navigation"
            self.sidebar_toggle_button.pack(side="right")
            self.sidebar_workflow_label.grid()
            self.sidebar_project_shell.grid()
            self.sidebar_footer.grid()
        for name, button in self.nav_buttons.items():
            icon, label = self.nav_specs[name]
            button.configure(
                text=icon if self.sidebar_collapsed else f"{icon}    {label}",
                anchor="center" if self.sidebar_collapsed else "w",
                width=48 if self.sidebar_collapsed else 200,
            )
            button.grid_configure(
                padx=12 if self.sidebar_collapsed else 12,
            )
        if not self.snowbuddy_collapsed:
            self._place_snowbuddy_panel(force=True)

    def _appearance_changed(self, value: str) -> None:
        mode = value.strip().lower()
        if mode == self.appearance_mode:
            return
        try:
            save_appearance_mode(self.store.library_root, mode)
        except (OSError, ValueError) as exc:
            self.appearance_control.set(self.appearance_mode.title())
            messagebox.showerror(
                "Could not save appearance",
                str(exc),
                parent=self,
            )
            return
        self.appearance_mode = mode
        ctk.set_appearance_mode(mode)
        self.appearance_control.set(mode.title())
        self.results_page.refresh_theme()
        self.inference_page.refresh_theme()
        self.inverse_design_page.refresh_theme()

    def _nav_button(
        self,
        parent: ctk.CTkFrame,
        row: int,
        icon: str,
        label: str,
        command: Callable[[], None],
    ) -> ctk.CTkButton:
        button = ctk.CTkButton(
            parent,
            text=f"{icon}    {label}",
            height=44,
            corner_radius=10,
            anchor="w",
            font=FONTS["button"],
            fg_color="transparent",
            hover_color=COLORS["sidebar_hover"],
            text_color=COLORS["muted"],
            command=command,
        )
        button.grid(row=row, column=0, padx=12, pady=3, sticky="ew")
        button.accessible_name = label
        try:
            button._canvas.configure(takefocus=1)  # type: ignore[attr-defined]
        except (AttributeError, tk.TclError):
            pass
        button.bind("<Return>", lambda _event: command(), add="+")
        button.bind("<space>", lambda _event: command(), add="+")
        self.nav_tooltips[label] = HoverTooltip(
            button,
            label,
            enabled=lambda: self.sidebar_collapsed,
        )
        return button

    def show_page(self, name: str, *, persist: bool = True) -> None:
        if name in {"data", "training", "results", "library", "inference", "inverse_design"} and not self.current_project:
            messagebox.showinfo(
                "Open a project",
                "Create or open a project before continuing the workflow.",
                parent=self,
            )
            name = "start"
        if name == "library" and self.current_project:
            self.library_page.project = self.current_project
            self.library_page.reload()
        if name == "inference" and self.current_project:
            self.inference_page.project = self.current_project
            self.inference_page.reload()
        if name == "inverse_design" and self.current_project:
            self.inverse_design_page.project = self.current_project
            self.inverse_design_page.reload()
        self.active_page = name
        self.pages[name].tkraise()
        if not self.snowbuddy_collapsed:
            self._place_snowbuddy_panel(force=True)
        for key, button in self.nav_buttons.items():
            active = key == name
            button.configure(
                fg_color=COLORS["nav_active"] if active else "transparent",
                text_color=COLORS["cyan"] if active else COLORS["muted"],
            )
        if (
            persist
            and self.current_project
            and name in {"start", "data", "training", "results", "library", "inference", "inverse_design"}
            and self.current_project.manifest.get("ui", {}).get("last_page") != name
        ):
            self.current_project = self.store.update_project(
                self.current_project,
                {"ui": {"last_page": name}},
            )
            self.start_page.refresh()

    def set_project(
        self,
        project: Project,
        *,
        target_page: str | None = None,
    ) -> None:
        self.current_project = self.store.open_project(project.path)
        self.sidebar_project_name.configure(text=self.current_project.name)
        self.sidebar_project_status.configure(text=self.current_project.status_label)
        self.sidebar_return_button.pack(fill="x", padx=12, pady=(0, 12))
        self.start_page.refresh()
        self.snowbuddy_panel.load_project(self.current_project)
        self.data_page.set_project(self.current_project)
        self.training_page.set_project(self.current_project)
        self.results_page.set_project(self.current_project)
        self.library_page.set_project(self.current_project)
        self.inference_page.set_project(self.current_project)
        self.inverse_design_page.set_project(self.current_project)
        remembered_page = str(
            self.current_project.manifest.get("ui", {}).get("last_page") or "data"
        )
        destination = target_page or remembered_page
        if destination not in self.pages:
            destination = "data"
        self.show_page(destination)

    def return_to_welcome(self) -> None:
        self.current_project = None
        self.sidebar_project_name.configure(text="No project open")
        self.sidebar_project_status.configure(text="Create or open a project")
        self.sidebar_return_button.pack_forget()
        self.data_page.set_project(None)
        self.training_page.set_project(None)
        self.results_page.set_project(None)
        self.library_page.set_project(None)
        self.inference_page.set_project(None)
        self.inverse_design_page.set_project(None)
        self.start_page.refresh()
        self.snowbuddy_panel.load_project(None)
        self.show_page("start", persist=False)

    def update_current_project(self, changes: dict) -> Project:
        if not self.current_project:
            raise ProjectError("No project is open.")
        self.current_project = self.store.update_project(self.current_project, changes)
        self.sidebar_project_name.configure(text=self.current_project.name)
        self.sidebar_project_status.configure(text=self.current_project.status_label)
        self.start_page.refresh()
        if (
            hasattr(self, "snowbuddy_panel")
            and self.snowbuddy_panel.current_project is not None
            and self.snowbuddy_panel.current_project.path == self.current_project.path
        ):
            # Keep the next assistant request grounded in newly persisted state
            # without clearing or re-rendering the active project conversation.
            self.snowbuddy_panel.current_project = self.current_project
        if (
            hasattr(self, "library_page")
            and self.library_page.project is not None
            and self.library_page.project.path == self.current_project.path
        ):
            self.library_page.project = self.current_project
        if (
            hasattr(self, "inference_page")
            and self.inference_page.project is not None
            and self.inference_page.project.path == self.current_project.path
        ):
            self.inference_page.project = self.current_project
        if (
            hasattr(self, "inverse_design_page")
            and self.inverse_design_page.project is not None
            and self.inverse_design_page.project.path == self.current_project.path
        ):
            self.inverse_design_page.project = self.current_project
        return self.current_project

    @property
    def start_page(self) -> "StartPage":
        return self.pages["start"]  # type: ignore[return-value]

    @property
    def data_page(self) -> "DataPrepPage":
        return self.pages["data"]  # type: ignore[return-value]

    @property
    def training_page(self) -> "ModelTrainingPage":
        return self.pages["training"]  # type: ignore[return-value]

    @property
    def results_page(self) -> TrainingResultsPage:
        return self.pages["results"]  # type: ignore[return-value]

    @property
    def library_page(self) -> ModelLibraryPage:
        return self.pages["library"]  # type: ignore[return-value]

    @property
    def inference_page(self) -> InferencePage:
        return self.pages["inference"]  # type: ignore[return-value]

    @property
    def inverse_design_page(self) -> InverseDesignPage:
        return self.pages["inverse_design"]  # type: ignore[return-value]

    def snowbuddy_ui_state(self) -> str:
        page_label = {
            "data": "Data Prep",
            "training": "Model Training",
            "results": "Training Results",
            "library": "Model Library",
            "inference": "Inference",
            "inverse_design": "Inverse Design",
        }.get(self.active_page, "Start")
        lines = [
            f"Visible page: {page_label}",
            f"Appearance mode: {self.appearance_mode.title()}",
            "Top application bar: File, Edit, Help, SnowBuddy, Appearance",
            f"Active project: {self.current_project.name if self.current_project else 'None'}",
            (
                "Workflow sidebar: collapsed icon-only navigation"
                if self.sidebar_collapsed
                else "Workflow sidebar: expanded navigation"
            ),
            (
                "SnowBuddy companion: closed; launcher in the top application bar"
                if self.snowbuddy_collapsed
                else (
                    "SnowBuddy companion: open docked panel"
                    if self.snowbuddy_display_mode == "docked"
                    else "SnowBuddy companion: open focused assistant view"
                )
            ),
            (
                "SnowBuddy composer enabled: restore companion to type"
                if self.snowbuddy_collapsed
                else "SnowBuddy composer enabled: yes"
            ),
            (
                "SnowBuddy mode: Focus"
                if self.current_project
                else "SnowBuddy mode: Welcome"
            ),
        ]
        if self.active_page == "start":
            lines.append(
                f"Recent project cards: {len(self.store.recent_projects(limit=5))}"
            )
        if self.current_project:
            data_visibility = (
                "visible now"
                if self.active_page == "data"
                else "retained state; Data Prep is not currently visible"
            )
            lines.append(f"Data Prep UI ({data_visibility}):")
            lines.extend(f"- {item}" for item in self.data_page.describe_ui_state())
            training_visibility = (
                "visible now"
                if self.active_page == "training"
                else "retained state; Model Training is not currently visible"
            )
            lines.append(f"Model Training UI ({training_visibility}):")
            lines.extend(
                f"- {item}"
                for item in self.training_page.describe_ui_state()
            )
            results_visibility = (
                "visible now"
                if self.active_page == "results"
                else "retained state; Training Results is not currently visible"
            )
            lines.append(f"Training Results UI ({results_visibility}):")
            lines.extend(
                f"- {item}"
                for item in self.results_page.describe_ui_state()
            )
            library_visibility = (
                "visible now"
                if self.active_page == "library"
                else "retained state; Model Library reloads when opened"
            )
            lines.append(f"Model Library UI ({library_visibility}):")
            if self.active_page == "library":
                lines.extend(
                    f"- {item}"
                    for item in self.library_page.describe_ui_state()
                )
            inference_visibility = (
                "visible now"
                if self.active_page == "inference"
                else "retained state; Inference reloads when opened"
            )
            lines.append(f"Inference UI ({inference_visibility}):")
            if self.active_page == "inference":
                lines.extend(
                    f"- {item}"
                    for item in self.inference_page.describe_ui_state()
                )
            inverse_visibility = (
                "visible now"
                if self.active_page == "inverse_design"
                else "retained state; Inverse Design reloads when opened"
            )
            lines.append(f"Inverse Design UI ({inverse_visibility}):")
            if self.active_page == "inverse_design":
                lines.extend(
                    f"- {item}"
                    for item in self.inverse_design_page.describe_ui_state()
                )
        return "\n".join(lines)

    def create_project_dialog(self) -> None:
        CreateProjectDialog(self, self._create_project)

    def _create_project(self, name: str, description: str) -> None:
        try:
            project = self.store.create_project(name, description)
        except ProjectError as exc:
            messagebox.showerror("Could not create project", str(exc), parent=self)
            return
        self.set_project(project, target_page="data")

    def open_project_dialog(self) -> None:
        path = filedialog.askdirectory(
            title="Open Antenna Surrogate Studio project",
            initialdir=str(self.store.projects_root),
            parent=self,
        )
        if path:
            self.open_project(path)

    def open_project(self, path: str | Path) -> None:
        try:
            project = self.store.open_project(path)
        except ProjectError as exc:
            messagebox.showerror("Could not open project", str(exc), parent=self)
            return
        self.set_project(project)


class StartPage(ctk.CTkFrame):
    def __init__(self, parent: ctk.CTkFrame, app: StudioApp):
        super().__init__(parent, fg_color=COLORS["app_bg"], corner_radius=0)
        self.app = app
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_workspace()
        self._build_footer()

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, padx=28, pady=(24, 18), sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        heading = ctk.CTkFrame(header, fg_color="transparent")
        heading.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            heading,
            text="Your surrogate workspace",
            text_color=COLORS["ink"],
            font=FONTS["display"],
            anchor="w",
        ).pack(anchor="w")
        ctk.CTkLabel(
            heading,
            text="Build trusted antenna models. Save them as books. Reuse them anytime.",
            text_color=COLORS["muted"],
            font=FONTS["body"],
            anchor="w",
        ).pack(anchor="w", pady=(4, 0))

        date_text = datetime.now().strftime("%A  ·  %B %d")
        ctk.CTkLabel(
            header,
            text=date_text.upper(),
            height=30,
            corner_radius=15,
            fg_color=COLORS["surface_alt"],
            text_color=COLORS["cyan"],
            font=FONTS["mono"],
        ).grid(row=0, column=1, sticky="e")

    def _build_workspace(self) -> None:
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, padx=(28, 24), pady=(0, 12), sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(1, weight=1)

        self.hero = ctk.CTkFrame(
            body,
            fg_color=COLORS["hero"],
            corner_radius=16,
            border_width=1,
            border_color=COLORS["primary"],
        )
        self.hero.grid(row=0, column=0, sticky="ew")
        self.hero.grid_columnconfigure(0, weight=1)
        self.hero.grid_columnconfigure(1, weight=0)
        self._build_hero_contents()

        recent_shell = ctk.CTkFrame(
            body,
            fg_color=COLORS["surface"],
            corner_radius=20,
            border_width=1,
            border_color=COLORS["border"],
        )
        recent_shell.grid(row=1, column=0, pady=(18, 0), sticky="nsew")
        recent_shell.grid_columnconfigure(0, weight=1)
        recent_shell.grid_rowconfigure(1, weight=1)

        title_row = ctk.CTkFrame(recent_shell, fg_color="transparent")
        title_row.grid(row=0, column=0, padx=20, pady=(18, 8), sticky="ew")
        title_row.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            title_row,
            text="Recent projects",
            text_color=COLORS["ink"],
            font=FONTS["section"],
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            title_row,
            text="Your five latest workspaces",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
        ).grid(row=0, column=1, sticky="e")

        self.recent_frame = ctk.CTkFrame(
            recent_shell,
            fg_color="transparent",
        )
        self.recent_frame.grid(row=1, column=0, padx=12, pady=(2, 14), sticky="nsew")
        for column in range(5):
            self.recent_frame.grid_columnconfigure(column, weight=1, uniform="recent")

    def _build_footer(self) -> None:
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, padx=(28, 24), pady=(0, 18), sticky="ew")
        footer.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            footer,
            text="MAIN WORKFLOW",
            text_color=COLORS["subtle"],
            font=FONTS["mono"],
        ).grid(row=0, column=0, sticky="w")
        self.next_page_button = ctk.CTkButton(
            footer,
            text="Next workflow step  →",
            width=176,
            height=42,
            corner_radius=11,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            font=FONTS["button"],
            state="disabled",
            command=self._continue_project,
        )
        self.next_page_button.grid(row=0, column=1, sticky="e")

    def _build_hero_contents(self) -> None:
        left = ctk.CTkFrame(self.hero, fg_color="transparent")
        left.grid(row=0, column=0, padx=26, pady=24, sticky="nsew")

        ctk.CTkLabel(
            left,
            text="ACTIVE WORKSPACE",
            text_color=COLORS["cyan"],
            font=FONTS["mono"],
            anchor="w",
        ).pack(anchor="w")
        self.hero_title = ctk.CTkLabel(
            left,
            text="Start something precise",
            text_color=COLORS["hero_ink"],
            font=("Segoe UI Semibold", 33),
            anchor="w",
        )
        self.hero_title.pack(anchor="w", pady=(6, 3))
        self.hero_subtitle = ctk.CTkLabel(
            left,
            text="Create a project or reopen an existing antenna workspace.",
            text_color=COLORS["muted"],
            font=FONTS["body_small"],
            anchor="w",
            wraplength=315,
            justify="left",
        )
        self.hero_subtitle.pack(anchor="w")

        self.progress_row = ctk.CTkFrame(left, fg_color="transparent")
        self.progress_row.pack(fill="x", pady=(18, 0))
        self.progress_bar = ctk.CTkProgressBar(
            self.progress_row,
            width=190,
            height=7,
            corner_radius=4,
            progress_color=COLORS["cyan"],
            fg_color=COLORS["disabled"],
        )
        self.progress_bar.pack(side="left")
        self.progress_text = ctk.CTkLabel(
            self.progress_row,
            text="1 of 5",
            text_color=COLORS["subtle"],
            font=("Segoe UI Semibold", 14),
        )
        self.progress_text.pack(side="left", padx=(10, 0))

        self.hero_actions = ctk.CTkFrame(self.hero, fg_color="transparent")
        self.hero_actions.grid(
            row=0,
            column=1,
            padx=(10, 26),
            pady=24,
            sticky="e",
        )
        self.create_project_button = ctk.CTkButton(
            self.hero_actions,
            text="+  Create project",
            width=164,
            height=43,
            corner_radius=12,
            font=FONTS["button"],
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            command=self.app.create_project_dialog,
        )
        self.create_project_button.pack()
        self.open_project_button = ctk.CTkButton(
            self.hero_actions,
            text="Open project",
            width=164,
            height=41,
            corner_radius=12,
            font=FONTS["button"],
            fg_color="transparent",
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border_strong"],
            text_color=COLORS["ink"],
            command=self.app.open_project_dialog,
        )
        self.open_project_button.pack(pady=(10, 0))
        self.continue_project_button = ctk.CTkButton(
            self.hero_actions,
            text="Continue Data Prep  →",
            width=182,
            height=43,
            corner_radius=12,
            font=FONTS["button"],
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            text_color=COLORS["on_primary"],
            command=self._continue_project,
        )

    def _continue_project(self) -> None:
        if self.app.current_project is None:
            return
        destination, _label = project_resume_destination(self.app.current_project)
        self.app.show_page(destination)

    def refresh(self) -> None:
        project = self.app.current_project
        if project:
            workflow = project.manifest.get("workflow", {})
            completed = int(workflow.get("completed_steps", 1))
            total = max(1, int(workflow.get("total_steps", 5)))
            self.hero_title.configure(text=project.name)
            self.hero_subtitle.configure(
                text=workflow.get("next_action", "Continue building this surrogate project.")
            )
            _destination, action_label = project_resume_destination(project)
            self.continue_project_button.configure(text=action_label)
            self.progress_bar.set(min(1.0, completed / total))
            self.progress_text.configure(text=f"{completed} of {total} steps")
            self.create_project_button.pack_forget()
            self.open_project_button.pack_forget()
            self.continue_project_button.pack()
        else:
            self.hero_title.configure(text="Start something precise")
            self.hero_subtitle.configure(
                text="Create a project or reopen an existing antenna workspace."
            )
            self.progress_bar.set(0.0)
            self.progress_text.configure(text="No active project")
            self.continue_project_button.pack_forget()
            self.create_project_button.pack()
            self.open_project_button.pack(pady=(10, 0))
        self.next_page_button.configure(
            state="normal" if project else "disabled",
            text=(action_label if project else "Next workflow step  →"),
            fg_color=(COLORS["primary"] if project else COLORS["disabled"]),
            text_color=(
                COLORS["on_primary"] if project else COLORS["disabled_text"]
            ),
        )

        for child in self.recent_frame.winfo_children():
            child.destroy()
        recent = self.app.store.recent_projects(limit=5)
        if not recent:
            EmptyRecentCard(self.recent_frame, self.app).grid(
                row=0,
                column=0,
                columnspan=2,
                padx=4,
                pady=6,
                sticky="nsew",
            )
        else:
            for index, item in enumerate(recent):
                ProjectCard(
                    self.recent_frame,
                    item,
                    command=lambda value=item: self.app.open_project(value.path),
                    accent_index=index,
                ).grid(
                    row=0,
                    column=index,
                    padx=4,
                    pady=6,
                    sticky="nsew",
                )


class ProjectCard(ctk.CTkFrame):
    ACCENTS = ("#0C8091", "#6952D4", "#138159", "#A66A00", "#2D6FD2")

    def __init__(
        self,
        parent: ctk.CTkFrame,
        project: Project,
        command: Callable[[], None],
        accent_index: int = 0,
    ):
        super().__init__(
            parent,
            width=100,
            height=196,
            corner_radius=16,
            fg_color=COLORS["surface_alt"],
            border_width=1,
            border_color=COLORS["border"],
        )
        self.pack_propagate(False)
        self.command = command
        accent = self.ACCENTS[accent_index % len(self.ACCENTS)]

        icon_shell = ctk.CTkFrame(
            self, width=38, height=38, corner_radius=12, fg_color=accent
        )
        icon_shell.pack(anchor="w", padx=9, pady=(11, 8))
        icon_shell.pack_propagate(False)
        ctk.CTkLabel(
            icon_shell,
            text="▤",
            text_color=COLORS["on_accent"],
            font=("Segoe UI Symbol", 24),
        ).place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            self,
            text=project.name,
            text_color=COLORS["ink"],
            font=FONTS["card_title"],
            justify="left",
            anchor="w",
            wraplength=78,
        ).pack(fill="x", padx=9)

        date = _friendly_date(project.last_opened_at)
        ctk.CTkLabel(
            self,
            text=f"Opened {date}",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            anchor="w",
        ).pack(fill="x", padx=9, pady=(4, 0))

        background, foreground = status_palette(project.status_label)
        ctk.CTkLabel(
            self,
            text=project.status_label,
            height=24,
            corner_radius=12,
            fg_color=background,
            text_color=foreground,
            font=("Segoe UI Semibold", 12),
        ).pack(anchor="w", padx=9, pady=(10, 0))

        self._bind_clicks(self)

    def _bind_clicks(self, widget) -> None:
        widget.bind("<Button-1>", lambda _event=None: self.command(), add="+")
        widget.bind(
            "<Enter>",
            lambda _event=None: self.configure(border_color=COLORS["cyan"]),
            add="+",
        )
        widget.bind(
            "<Leave>",
            lambda _event=None: self.configure(border_color=COLORS["border"]),
            add="+",
        )
        for child in widget.winfo_children():
            self._bind_clicks(child)


class EmptyRecentCard(ctk.CTkFrame):
    def __init__(self, parent: ctk.CTkFrame, app: StudioApp):
        super().__init__(
            parent,
            width=300,
            height=210,
            corner_radius=16,
            fg_color=COLORS["surface_alt"],
            border_width=1,
            border_color=COLORS["border"],
        )
        self.pack_propagate(False)
        ctk.CTkLabel(
            self,
            text="＋",
            text_color=COLORS["primary"],
            font=("Segoe UI Light", 39),
        ).pack(pady=(28, 8))
        ctk.CTkLabel(
            self,
            text="Your project shelf is empty",
            text_color=COLORS["ink"],
            font=FONTS["card_title"],
        ).pack()
        ctk.CTkLabel(
            self,
            text="Create the first portable antenna workspace.",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
        ).pack(pady=(5, 12))
        ctk.CTkButton(
            self,
            text="Create project",
            height=34,
            corner_radius=10,
            fg_color=COLORS["primary_soft"],
            hover_color=COLORS["control_hover"],
            text_color=COLORS["cyan"],
            font=FONTS["button"],
            command=app.create_project_dialog,
        ).pack()


class SnowBuddyPanel(ctk.CTkFrame):
    def __init__(self, parent: ctk.CTkFrame, app: StudioApp):
        super().__init__(
            parent,
            width=390,
            corner_radius=20,
            fg_color=COLORS["surface"],
            border_width=1,
            border_color=COLORS["border"],
        )
        self.app = app
        self.current_project: Project | None = None
        self.mode = "welcome"
        self.context_generation = 0
        self._scroll_after_id: str | None = None
        self.context_hint = ctk.StringVar(
            value="Welcome session · Local history · Ctrl+Enter to send"
        )
        self.grid_propagate(False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_header()
        self._build_chat()
        self._refresh_status()

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, padx=18, pady=(18, 10), sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        avatar = ctk.CTkLabel(
            header,
            text="✦",
            width=43,
            height=43,
            corner_radius=14,
            fg_color=COLORS["violet"],
            text_color=COLORS["on_accent"],
            font=("Segoe UI Symbol", 24),
        )
        avatar.grid(row=0, column=0, rowspan=2, sticky="w")
        ctk.CTkLabel(
            header,
            text="SnowBuddy",
            text_color=COLORS["ink"],
            font=FONTS["section"],
            anchor="w",
        ).grid(row=0, column=1, padx=(11, 0), sticky="sw")
        self.status_label = ctk.CTkLabel(
            header,
            text="Welcome mode",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            anchor="w",
        )
        self.status_label.grid(row=1, column=1, padx=(11, 0), sticky="nw")
        self.connect_button = ctk.CTkButton(
            header,
            text="Local model",
            width=104,
            height=30,
            corner_radius=10,
            fg_color=COLORS["violet_soft"],
            hover_color=COLORS["violet_hover"],
            text_color=COLORS["on_violet_soft"],
            font=("Segoe UI Semibold", 14),
            command=self._show_model_dialog,
        )
        self.connect_button.grid(row=0, column=2, rowspan=2, sticky="e")
        self.collapse_button = ctk.CTkButton(
            header,
            text="—",
            width=30,
            height=30,
            corner_radius=9,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=("Segoe UI Semibold", 16),
            command=lambda: self.app.set_snowbuddy_collapsed(True),
        )
        self.collapse_button.grid(
            row=0,
            column=3,
            rowspan=2,
            padx=(7, 0),
            sticky="e",
        )

    def _build_chat(self) -> None:
        self.messages_frame = ctk.CTkScrollableFrame(
            self,
            fg_color=COLORS["surface_alt"],
            corner_radius=14,
            scrollbar_button_color=COLORS["scrollbar"],
            scrollbar_button_hover_color=COLORS["scrollbar_hover"],
        )
        self.messages_frame.grid(
            row=1, column=0, padx=14, pady=(0, 12), sticky="nsew"
        )
        self.messages_frame.grid_columnconfigure(0, weight=1)

        composer = ctk.CTkFrame(self, fg_color="transparent")
        composer.grid(row=2, column=0, padx=14, pady=(0, 15), sticky="ew")
        composer.grid_columnconfigure(0, weight=1)
        self.input_box = ctk.CTkTextbox(
            composer,
            height=74,
            corner_radius=13,
            border_width=1,
            border_color=COLORS["border"],
            fg_color=COLORS["surface"],
            text_color=COLORS["ink"],
            font=FONTS["body_small"],
            wrap="word",
        )
        self.input_box.grid(row=0, column=0, sticky="ew")
        self.input_box.bind("<Control-Return>", lambda _event: self.send())
        self.send_button = ctk.CTkButton(
            composer,
            text="↑",
            width=44,
            height=44,
            corner_radius=14,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            font=("Segoe UI Semibold", 23),
            command=self.send,
        )
        self.send_button.grid(row=0, column=1, padx=(8, 0), sticky="se")
        ctk.CTkLabel(
            composer,
            textvariable=self.context_hint,
            text_color=COLORS["subtle"],
            font=("Segoe UI", 12),
            anchor="w",
        ).grid(row=1, column=0, columnspan=2, pady=(5, 0), sticky="w")

    def load_project(self, project: Project | None) -> None:
        self.context_generation += 1
        self.current_project = project
        self.mode = "focus" if project else "welcome"
        self._clear_messages()
        if not project:
            self.context_hint.set(
                "Welcome session · Local history · Ctrl+Enter to send"
            )
            history = self.app.store.load_welcome_chat()
            if history:
                for item in history[-30:]:
                    self._add_message(item["role"], item["content"])
            else:
                self._add_message(
                    "assistant",
                    "Hi—I’m SnowBuddy. Welcome mode has started a fresh local "
                    "session. I can explain the workflow, help you choose Create or "
                    "Open, and check the local-model setup.",
                )
            self._set_composer_enabled(True)
            self._refresh_status()
            return

        history = self.app.store.load_chat(project)
        self.context_hint.set(
            f"Focus: {project.name} · Local history · Ctrl+Enter to send"
        )
        if history:
            for item in history[-30:]:
                self._add_message(item["role"], item["content"])
        else:
                self._add_message(
                    "assistant",
                    f"Hi—I’m SnowBuddy. Focus mode is active for **{project.name}**. "
                    "I’ll stay with this project as you move through each step.",
                )
        self._set_composer_enabled(True)
        self._refresh_status()

    def send(self) -> None:
        question = self.input_box.get("1.0", "end").strip()
        if not question:
            return
        live_ui_state = self.app.snowbuddy_ui_state()
        project_for_question = self.current_project
        context_generation = self.context_generation
        self.input_box.delete("1.0", "end")
        self._add_message("user", question)
        self._add_message("assistant", "Thinking with your project context…", temporary=True)
        self._set_composer_enabled(False)

        def worker() -> None:
            try:
                reply, used_local_model = self.app.snowbuddy.ask(
                    project_for_question,
                    question,
                    live_ui_state=live_ui_state,
                )
            except Exception as exc:
                self._run_on_ui(
                    lambda error=exc, generation=context_generation: (
                        self._finish_error(error, generation)
                    ),
                )
                return
            self._run_on_ui(
                lambda response=reply, used=used_local_model, generation=context_generation: (
                    self._finish_reply(
                        response,
                        used,
                        generation,
                    )
                ),
            )

        threading.Thread(target=worker, daemon=True).start()

    def _finish_reply(
        self,
        reply: str,
        used_local_model: bool,
        context_generation: int,
    ) -> None:
        if context_generation != self.context_generation:
            return
        self._remove_temporary()
        self._add_message("assistant", reply)
        self._set_composer_enabled(True)
        self._refresh_status(used_local_model=used_local_model)
        self.input_box.focus_set()

    def _finish_error(self, exc: Exception, context_generation: int) -> None:
        if context_generation != self.context_generation:
            return
        self._remove_temporary()
        self._add_message("assistant", f"I couldn’t answer that yet: {exc}")
        self._set_composer_enabled(True)

    def _add_message(self, role: str, content: str, temporary: bool = False) -> None:
        row = ctk.CTkFrame(self.messages_frame, fg_color="transparent")
        row.grid(
            row=len(self.messages_frame.winfo_children()),
            column=0,
            padx=4,
            pady=5,
            sticky="ew",
        )
        row.grid_columnconfigure(0, weight=1)
        is_user = role == "user"
        label = ctk.CTkLabel(
            row,
            text=_display_markdown(content),
            justify="left",
            anchor="w",
            wraplength=292,
            corner_radius=13,
            fg_color=COLORS["chat_user"] if is_user else COLORS["chat_assistant"],
            text_color=COLORS["ink"],
            font=FONTS["body_small"],
            padx=13,
            pady=10,
        )
        label.grid(row=0, column=0, sticky="e" if is_user else "w")
        row._temporary = temporary  # type: ignore[attr-defined]
        if self._scroll_after_id is not None:
            try:
                self.after_cancel(self._scroll_after_id)
            except (tk.TclError, ValueError):
                pass
        self._scroll_after_id = self.after(30, self._scroll_to_bottom)

    def _clear_messages(self) -> None:
        for child in self.messages_frame.winfo_children():
            child.destroy()

    def _remove_temporary(self) -> None:
        for child in self.messages_frame.winfo_children():
            if getattr(child, "_temporary", False):
                child.destroy()

    def _scroll_to_bottom(self) -> None:
        self._scroll_after_id = None
        try:
            canvas = self.messages_frame._parent_canvas
            self.messages_frame.update_idletasks()
            bounds = canvas.bbox("all")
            if bounds:
                canvas.configure(scrollregion=bounds)
            canvas.yview_moveto(1.0)
        except Exception:
            pass

    def _run_on_ui(self, callback: Callable[[], None]) -> None:
        if self.app._destroying:
            return
        try:
            self.after(0, callback)
        except (RuntimeError, tk.TclError):
            pass

    def cancel_pending_callbacks(self) -> None:
        self.context_generation += 1
        if self._scroll_after_id is not None:
            try:
                self.after_cancel(self._scroll_after_id)
            except (tk.TclError, ValueError):
                pass
            self._scroll_after_id = None

    def _set_composer_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.input_box.configure(state=state)
        self.send_button.configure(state=state)

    def _show_model_dialog(self) -> None:
        LocalModelDialog(self, self.app.snowbuddy, self._model_settings_changed)

    def _model_settings_changed(self) -> None:
        self._refresh_status()

    def _refresh_status(self, used_local_model: bool | None = None) -> None:
        mode_label = "Focus mode" if self.mode == "focus" else "Welcome mode"
        if used_local_model:
            self.status_label.configure(
                text=f"{mode_label} · Local {self.app.snowbuddy.model}",
                text_color=COLORS["success"],
            )
            self.connect_button.configure(text="Model settings")
        else:
            profile = model_profile(self.app.snowbuddy.model)
            label = profile.label if profile else self.app.snowbuddy.model
            self.status_label.configure(
                text=f"{mode_label} · {label}",
                text_color=COLORS["muted"],
            )
            self.connect_button.configure(text="Local model")


class DataPrepPage(ctk.CTkFrame):
    def __init__(self, parent: ctk.CTkFrame, app: StudioApp):
        super().__init__(parent, fg_color=COLORS["app_bg"], corner_radius=0)
        self.app = app
        self.project: Project | None = None
        self.discovery: DiscoveryResult | None = None
        self.input_checks: dict[str, ctk.BooleanVar] = {}
        self.mode_var = ctk.StringVar(value="pair")
        self.path_var = ctk.StringVar()
        self.input_path_var = ctk.StringVar()
        self.output_path_var = ctk.StringVar()
        self.output_var = ctk.StringVar(value="")
        self.status_var = ctk.StringVar(value="Choose a source to begin.")
        self.registration_status_var = ctk.StringVar(
            value="Prepare data to enable validation."
        )
        self.registration_detail_var = ctk.StringVar(
            value="No validation result yet."
        )
        self.registration_id_var = ctk.StringVar(value="Dataset ID: not registered")
        self.registered_dataset_id: str | None = None
        self.pair_load_in_progress = False
        self.last_pair_signature: tuple[str, str] | None = None
        self.sample_generator_dialog: LHSSampleGeneratorDialog | None = None
        self.generated_lhs_input_path: str | None = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_header()
        self._build_content()

    def describe_ui_state(self) -> list[str]:
        mode_label = (
            "Input + output files"
            if self.mode_var.get() == "pair"
            else "#Parameters sweep"
        )
        selected_inputs = [
            name for name, state in self.input_checks.items() if state.get()
        ]
        lines = [
            f"Source mode: {mode_label}",
            f"Expanded Data Prep subtask: {self.active_subtask}",
            "Data Prep page scrolling: none",
            f"Analysis status: {self.status_var.get()}",
            f"Selected inputs: {', '.join(selected_inputs) or 'none'}",
            f"Selected output: {self.output_var.get().strip() or 'none'}",
        ]
        if self.mode_var.get() == "pair":
            lines.extend(
                [
                    (
                        "Input CSV: "
                        f"{self.input_path_var.get().strip() or 'not selected'}"
                    ),
                    (
                        "Output CSV: "
                        f"{self.output_path_var.get().strip() or 'not selected'}"
                    ),
                    "Template action: Create templates",
                    "Sample design action: LHS sample generator",
                    (
                        "Generated LHS input CSV: loaded; solver output CSV still required"
                        if self.generated_lhs_input_path
                        and self.input_path_var.get().strip()
                        == self.generated_lhs_input_path
                        and not self.output_path_var.get().strip()
                        else "Generated LHS input CSV: not pending"
                    ),
                    "CSV pair behavior: automatic load, all columns retained",
                    "Parse action visible: no",
                ]
            )
        else:
            lines.append(
                f"Source path: {self.path_var.get().strip() or 'not selected'}"
            )
            lines.append("Raw extract action: Parse")
        prep_state = (
            self.project.manifest.get("data_prep", {}) if self.project else {}
        )
        variable_contract_confirmed = bool(
            prep_state.get("variable_contract_confirmed")
        )
        if self.mode_var.get() == "parameters" and self.discovery is not None:
            variable_contract_confirmed = (
                self.confirm_variables_button.cget("state") == "disabled"
                and self.prepare_button.cget("state") == "normal"
            )
        prepared_inputs_csv = str(
            prep_state.get("prepared_inputs_csv") or ""
        )
        prepared_outputs_csv = str(
            prep_state.get("prepared_outputs_csv") or ""
        )
        prepared_ready = bool(
            self.project
            and prepared_inputs_csv
            and prepared_outputs_csv
            and (self.project.path / prepared_inputs_csv).exists()
            and (self.project.path / prepared_outputs_csv).exists()
        )
        lines.append(
            f"Prepared data: {'ready' if prepared_ready else 'not ready'}"
        )
        if prepared_ready:
            lines.append(
                "Prepared files: separate input and output CSV tables"
            )
        lines.extend(
            [
                f"Dataset validation: {self.registration_status_var.get()}",
                f"Validation details: {self.registration_detail_var.get()}",
                f"Registered dataset: {self.registered_dataset_id or 'none'}",
                (
                    "Validate and register action enabled: yes"
                    if prepared_ready
                    else "Validate and register action enabled: no"
                ),
                "Training action triggered here: no",
            ]
        )
        if self.discovery:
            lines.extend(
                [
                    f"Discovered samples: {self.discovery.sample_count}",
                    (
                        "Available inputs: "
                        f"{', '.join(self.discovery.input_variables) or 'none'}"
                    ),
                    (
                        "Available outputs: "
                        f"{', '.join(self.discovery.output_variables) or 'none'}"
                    ),
                    (
                    "Prepare action: re-prepare available"
                    if prepared_ready
                    else (
                        "Prepare action: automatic for CSV pair"
                        if self.discovery.mode == "pair"
                        else (
                            "Prepare action enabled: yes"
                            if variable_contract_confirmed
                            else "Save selection action required before Prepare"
                        )
                    )
                    ),
                ]
            )
        else:
            lines.extend(
                [
                    "Source loaded or parsed: no",
                    "Prepare action enabled: no",
                ]
            )
        lines.append("Bottom navigation: Back to Start; Next Model Training enabled")
        return lines

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, padx=28, pady=(14, 8), sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="Data Prep",
            text_color=COLORS["ink"],
            font=FONTS["display"],
            height=34,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header,
            text=(
                "Generate simulation inputs, validate paired CSVs, or convert "
                "#Parameters exports into model-ready tables."
            ),
            text_color=COLORS["muted"],
            font=FONTS["body"],
            height=18,
            anchor="w",
        ).grid(row=1, column=0, pady=(3, 0), sticky="w")
    def _build_content(self) -> None:
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.grid(row=1, column=0, padx=22, pady=(0, 8), sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        self.subtask_bodies: dict[str, ctk.CTkFrame] = {}
        self.subtask_chevrons: dict[str, ctk.CTkButton] = {}
        self.subtask_cards: dict[str, ctk.CTkFrame] = {}
        self.step_labels: list[ctk.CTkLabel] = []
        self.active_subtask = "source"

        self._build_source_card(content)
        self._build_variables_card(content)
        self._build_prepare_card(content)
        self._build_registration_card(content)
        self._expand_subtask("source")
        self._mode_changed("Input + output files")
        self._build_page_footer()

    def _build_page_footer(self) -> None:
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, padx=28, pady=(0, 10), sticky="ew")
        footer.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(
            footer,
            text="←  Back to Start",
            width=150,
            height=36,
            corner_radius=11,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            command=lambda: self.app.show_page("start"),
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            footer,
            text="Configure the model-training behavior next.",
            text_color=COLORS["subtle"],
            font=FONTS["caption"],
        ).grid(row=0, column=1)
        self.next_main_page_button = ctk.CTkButton(
            footer,
            text="Next: Model Training  →",
            width=194,
            height=36,
            corner_radius=11,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            text_color=COLORS["on_primary"],
            font=FONTS["button"],
            command=lambda: self.app.show_page("training"),
        )
        self.next_main_page_button.grid(row=0, column=2, sticky="e")

    def _card(self, parent: ctk.CTkFrame, row: int) -> ctk.CTkFrame:
        definitions = (
            ("source", "Select data source", "CSV pair or raw extract"),
            ("variables", "Define model contract", "Automatic for CSV pairs"),
            ("prepare", "Prepare project tables", "Create the model-ready pair"),
            ("register", "Validate and register", "Lock a trusted local snapshot"),
        )
        key, title, subtitle = definitions[row - 1]
        frame = ctk.CTkFrame(
            parent,
            fg_color=COLORS["surface"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["border"],
        )
        frame.grid(row=row - 1, column=0, padx=6, pady=(0, 5), sticky="ew")
        frame.grid_columnconfigure(0, weight=1)
        self.subtask_cards[key] = frame

        header = ctk.CTkFrame(frame, fg_color="transparent", height=44)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(2, weight=1)
        badge = ctk.CTkLabel(
            header,
            text="●" if row == 1 else "○",
            width=30,
            height=28,
            corner_radius=14,
            fg_color=COLORS["primary"] if row == 1 else COLORS["disabled"],
            text_color=COLORS["on_accent"] if row == 1 else COLORS["muted"],
            font=("Segoe UI Symbol", 16),
        )
        badge.grid(row=0, column=0, padx=(12, 8), pady=6)
        self.step_labels.append(badge)
        ctk.CTkLabel(
            header,
            text=f"SUBTASK {row}",
            text_color=COLORS["cyan"],
            font=FONTS["mono"],
            anchor="w",
        ).grid(row=0, column=1, padx=(0, 8), sticky="w")
        title_label = ctk.CTkLabel(
            header,
            text=title,
            text_color=COLORS["ink"],
            font=FONTS["card_title"],
            anchor="w",
        )
        title_label.grid(row=0, column=2, sticky="w")
        subtitle_label = ctk.CTkLabel(
            header,
            text=subtitle,
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            anchor="w",
        )
        subtitle_label.grid(row=0, column=3, padx=(12, 0), sticky="w")
        chevron = ctk.CTkButton(
            header,
            text="⌄" if row == 1 else "›",
            width=36,
            height=34,
            corner_radius=9,
            fg_color="transparent",
            hover_color=COLORS["control_hover"],
            text_color=COLORS["muted"],
            font=("Segoe UI Symbol", 22),
            command=lambda value=key: self._expand_subtask(value),
        )
        chevron.grid(row=0, column=4, padx=8)
        self.subtask_chevrons[key] = chevron
        for widget in (header, badge, title_label, subtitle_label):
            widget.bind(
                "<Button-1>",
                lambda _event, value=key: self._expand_subtask(value),
                add="+",
            )

        body = ctk.CTkFrame(frame, fg_color="transparent")
        body.grid(row=1, column=0, sticky="ew")
        body.grid_columnconfigure(0, weight=1)
        self.subtask_bodies[key] = body
        if key != self.active_subtask:
            body.grid_remove()
        return body

    def _expand_subtask(self, key: str) -> None:
        if key not in self.subtask_bodies:
            return
        self.active_subtask = key
        for name, body in self.subtask_bodies.items():
            if name == key:
                body.grid()
                self.subtask_chevrons[name].configure(text="⌃")
                self.subtask_cards[name].configure(
                    border_color=COLORS["border_strong"]
                )
            else:
                body.grid_remove()
                self.subtask_chevrons[name].configure(text="›")
                self.subtask_cards[name].configure(border_color=COLORS["border"])

    def _set_subtask_status(self, index: int, status: str) -> None:
        styles = {
            "pending": ("○", COLORS["disabled"], COLORS["muted"]),
            "active": ("●", COLORS["primary"], COLORS["on_accent"]),
            "complete": ("✓", COLORS["success"], COLORS["on_accent"]),
            "error": ("!", COLORS["danger"], COLORS["on_accent"]),
        }
        symbol, background, foreground = styles[status]
        self.step_labels[index].configure(
            text=symbol,
            fg_color=background,
            text_color=foreground,
        )

    def _build_source_card(self, parent: ctk.CTkFrame) -> None:
        card = self._card(parent, 1)
        ctk.CTkLabel(
            card,
            text=(
                "Generate simulation inputs, load a ready input/output pair, or "
                "parse a #Parameters export."
            ),
            text_color=COLORS["muted"],
            font=FONTS["body_small"],
            height=18,
            anchor="w",
        ).grid(row=0, column=0, padx=16, pady=(7, 6), sticky="w")

        mode_row = ctk.CTkFrame(card, fg_color="transparent")
        mode_row.grid(row=1, column=0, padx=16, pady=(0, 8), sticky="ew")
        mode_row.grid_columnconfigure(0, weight=1)
        self.mode_control = ctk.CTkSegmentedButton(
            mode_row,
            values=["Input + output files", "#Parameters sweep"],
            height=34,
            corner_radius=10,
            selected_color=COLORS["primary"],
            selected_hover_color=COLORS["primary_hover"],
            unselected_color=COLORS["surface_alt"],
            unselected_hover_color=COLORS["control_hover"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            command=self._mode_changed,
        )
        self.mode_control.grid(row=0, column=0, sticky="w")
        self.mode_control.set("Input + output files")
        self.sample_generator_button = ctk.CTkButton(
            mode_row,
            text="LHS sample generator",
            width=174,
            height=34,
            corner_radius=10,
            fg_color=COLORS["primary_soft"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["primary"],
            text_color=COLORS["cyan"],
            font=FONTS["button"],
            command=self.open_lhs_sample_generator,
        )
        self.sample_generator_button.grid(row=0, column=1, padx=(8, 0), sticky="e")
        self.template_button = ctk.CTkButton(
            mode_row,
            text="Create templates",
            width=132,
            height=34,
            corner_radius=10,
            fg_color=COLORS["violet_soft"],
            hover_color=COLORS["violet_hover"],
            border_width=1,
            border_color=COLORS["violet"],
            text_color=COLORS["on_violet_soft"],
            font=FONTS["button"],
            command=self.create_input_output_templates,
        )
        self.template_button.grid(row=0, column=2, padx=(8, 0), sticky="e")

        self.pair_paths_frame = ctk.CTkFrame(card, fg_color="transparent")
        self.pair_paths_frame.grid(
            row=2, column=0, padx=16, pady=(0, 6), sticky="ew"
        )
        self.pair_paths_frame.grid_columnconfigure(1, weight=1)
        for row, (label, variable, command) in enumerate(
            (
                ("Input CSV", self.input_path_var, self._browse_input_file),
                ("Output CSV", self.output_path_var, self._browse_output_file),
            )
        ):
            ctk.CTkLabel(
                self.pair_paths_frame,
                text=label,
                width=82,
                text_color=COLORS["muted"],
                font=FONTS["caption"],
                height=18,
                anchor="w",
            ).grid(row=row, column=0, padx=(0, 8), pady=2, sticky="w")
            entry = ctk.CTkEntry(
                self.pair_paths_frame,
                textvariable=variable,
                height=34,
                corner_radius=10,
                border_color=COLORS["border"],
                placeholder_text=f"Choose the {label.lower()} file",
                font=FONTS["mono"],
            )
            entry.grid(row=row, column=1, pady=3, sticky="ew")
            entry.bind("<Return>", lambda _event: self._maybe_load_pair())
            entry.bind("<FocusOut>", lambda _event: self._maybe_load_pair())
            ctk.CTkButton(
                self.pair_paths_frame,
                text=f"Browse {label.split()[0].lower()}",
                width=116,
                height=34,
                corner_radius=10,
                fg_color=COLORS["surface_alt"],
                hover_color=COLORS["control_hover"],
                border_width=1,
                border_color=COLORS["border"],
                text_color=COLORS["ink"],
                font=FONTS["button"],
                command=command,
            ).grid(row=row, column=2, padx=(8, 0), pady=2)

        self.parameter_path_frame = ctk.CTkFrame(card, fg_color="transparent")
        self.parameter_path_frame.grid_columnconfigure(0, weight=1)
        self.path_entry = ctk.CTkEntry(
            self.parameter_path_frame,
            textvariable=self.path_var,
            height=34,
            corner_radius=11,
            border_color=COLORS["border"],
            placeholder_text="Select a .txt export or a folder of exports",
            font=FONTS["mono"],
        )
        self.path_entry.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(
            self.parameter_path_frame,
            text="Browse file",
            width=102,
            height=34,
            corner_radius=11,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            command=self._browse_file,
        ).grid(row=0, column=1, padx=(8, 0))
        ctk.CTkButton(
            self.parameter_path_frame,
            text="Browse folder",
            width=112,
            height=34,
            corner_radius=11,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            command=self._browse_folder,
        ).grid(row=0, column=2, padx=(8, 0))

        action_row = ctk.CTkFrame(card, fg_color="transparent")
        action_row.grid(row=3, column=0, padx=16, pady=(0, 6), sticky="ew")
        action_row.grid_columnconfigure(0, weight=1)
        self.source_hint_label = ctk.CTkLabel(
            action_row,
            text="Use matching rows/IDs and numeric parameter and output values.",
            text_color=COLORS["subtle"],
            font=FONTS["caption"],
            height=32,
            wraplength=520,
            justify="left",
            anchor="w",
        )
        self.source_hint_label.grid(row=0, column=0, sticky="w")
        self.analyze_button = ctk.CTkButton(
            action_row,
            text="Parse",
            width=130,
            height=34,
            corner_radius=11,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            font=FONTS["button"],
            command=self.analyze_source,
        )
        self.analyze_button.grid(row=0, column=1, padx=(8, 0))

    def _build_variables_card(self, parent: ctk.CTkFrame) -> None:
        self.variables_card = self._card(parent, 2)
        self.variables_card.grid_columnconfigure(0, weight=1)
        self.variables_card.grid_columnconfigure(1, weight=1)
        self.discovery_summary = ctk.CTkLabel(
            self.variables_card,
            text="Parse a raw extract to discover inputs and outputs.",
            text_color=COLORS["muted"],
            font=FONTS["body_small"],
            height=18,
            anchor="w",
        )
        self.discovery_summary.grid(
            row=0,
            column=0,
            columnspan=2,
            padx=16,
            pady=(7, 3),
            sticky="w",
        )

        inputs_shell = ctk.CTkFrame(
            self.variables_card,
            fg_color=COLORS["surface_alt"],
            corner_radius=13,
        )
        inputs_shell.grid(
            row=1,
            column=0,
            padx=(16, 6),
            pady=(6, 10),
            sticky="nsew",
        )
        inputs_shell.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            inputs_shell,
            text="Model inputs",
            text_color=COLORS["ink"],
            font=FONTS["card_title"],
            height=18,
            anchor="w",
        ).grid(row=0, column=0, padx=12, pady=(7, 2), sticky="w")
        ctk.CTkLabel(
            inputs_shell,
            text="Select one or more design variables",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            height=18,
            anchor="w",
        ).grid(row=1, column=0, padx=12, sticky="w")
        self.inputs_frame = ctk.CTkFrame(
            inputs_shell,
            height=50,
            fg_color="transparent",
        )
        self.inputs_frame.grid(row=2, column=0, padx=6, pady=(3, 5), sticky="ew")

        output_shell = ctk.CTkFrame(
            self.variables_card,
            fg_color=COLORS["surface_alt"],
            corner_radius=13,
        )
        output_shell.grid(
            row=1,
            column=1,
            padx=(6, 16),
            pady=(6, 10),
            sticky="nsew",
        )
        output_shell.grid_columnconfigure(0, weight=1)
        self.output_title_label = ctk.CTkLabel(
            output_shell,
            text="Pattern output",
            text_color=COLORS["ink"],
            font=FONTS["card_title"],
            height=18,
            anchor="w",
        )
        self.output_title_label.grid(
            row=0, column=0, padx=12, pady=(7, 2), sticky="w"
        )
        self.output_help_label = ctk.CTkLabel(
            output_shell,
            text="Select one response to expand across its coordinate grid",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            height=18,
            anchor="w",
        )
        self.output_help_label.grid(row=1, column=0, padx=12, sticky="w")
        self.output_menu = ctk.CTkOptionMenu(
            output_shell,
            variable=self.output_var,
            values=["Parse a raw extract first"],
            height=34,
            corner_radius=10,
            fg_color=COLORS["surface"],
            button_color=COLORS["primary"],
            button_hover_color=COLORS["primary_hover"],
            text_color=COLORS["ink"],
            font=FONTS["body_small"],
            dropdown_font=FONTS["body_small"],
            state="disabled",
            command=lambda _value: self._variable_selection_changed(),
        )
        self.output_menu.grid(row=2, column=0, padx=12, pady=(6, 0), sticky="ew")
        self.output_note_label = ctk.CTkLabel(
            output_shell,
            text=(
                "The response and source coordinate grid will be saved in the "
                "project schema."
            ),
            text_color=COLORS["subtle"],
            font=("Segoe UI", 14),
            height=28,
            wraplength=370,
            justify="left",
            anchor="w",
        )
        self.output_note_label.grid(
            row=3, column=0, padx=12, pady=(5, 6), sticky="w"
        )

        self.variable_action_row = ctk.CTkFrame(
            self.variables_card,
            fg_color="transparent",
        )
        self.variable_action_row.grid(
            row=2,
            column=0,
            columnspan=2,
            padx=16,
            pady=(0, 10),
            sticky="ew",
        )
        self.variable_action_row.grid_columnconfigure(0, weight=1)
        self.variable_action_note = ctk.CTkLabel(
            self.variable_action_row,
            text="Save this input/output contract before preparing project tables.",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            anchor="w",
        )
        self.variable_action_note.grid(row=0, column=0, sticky="w")
        self.confirm_variables_button = ctk.CTkButton(
            self.variable_action_row,
            text="Save selection  →",
            width=148,
            height=34,
            corner_radius=10,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            text_color=COLORS["on_primary"],
            font=FONTS["button"],
            state="disabled",
            command=self.confirm_variable_selection,
        )
        self.confirm_variables_button.grid(row=0, column=1, sticky="e")

    def _build_prepare_card(self, parent: ctk.CTkFrame) -> None:
        card = self._card(parent, 3)
        card.grid_columnconfigure(0, weight=1)
        status = ctk.CTkFrame(card, fg_color="transparent")
        status.grid(row=0, column=0, padx=16, pady=12, sticky="ew")
        status.grid_columnconfigure(1, weight=1)
        self.status_dot = ctk.CTkLabel(
            status,
            text="",
            width=12,
            height=12,
            corner_radius=6,
            fg_color=COLORS["warning"],
        )
        self.status_dot.grid(row=0, column=0, padx=(0, 10))
        ctk.CTkLabel(
            status,
            textvariable=self.status_var,
            text_color=COLORS["ink"],
            font=FONTS["body_small"],
            anchor="w",
        ).grid(row=0, column=1, sticky="w")
        self.prepare_button = ctk.CTkButton(
            status,
            text="Prepare input + output  →",
            width=218,
            height=43,
            corner_radius=12,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            font=FONTS["button"],
            state="disabled",
            command=self.prepare_data,
        )
        self.prepare_button.grid(row=0, column=3, sticky="e")
        self.open_prepared_button = ctk.CTkButton(
            status,
            text="View prepared files",
            width=142,
            height=43,
            corner_radius=12,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            state="disabled",
            command=self.open_prepared_folder,
        )
        self.open_prepared_button.grid(
            row=0,
            column=2,
            padx=(8, 8),
            sticky="e",
        )

    def _build_registration_card(self, parent: ctk.CTkFrame) -> None:
        card = self._card(parent, 4)
        card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            card,
            text=(
                "One action validates the prepared tables, then creates an "
                "integrity-protected local dataset snapshot."
            ),
            text_color=COLORS["muted"],
            font=FONTS["body_small"],
            height=18,
            anchor="w",
        ).grid(row=0, column=0, padx=16, pady=(7, 3), sticky="w")

        result_shell = ctk.CTkFrame(
            card,
            fg_color=COLORS["surface_alt"],
            corner_radius=13,
        )
        result_shell.grid(
            row=1,
            column=0,
            padx=16,
            pady=(6, 8),
            sticky="ew",
        )
        result_shell.grid_columnconfigure(1, weight=1)
        self.registration_dot = ctk.CTkLabel(
            result_shell,
            text="",
            width=12,
            height=12,
            corner_radius=6,
            fg_color=COLORS["disabled"],
        )
        self.registration_dot.grid(
            row=0,
            column=0,
            rowspan=3,
            padx=(14, 11),
            pady=8,
            sticky="n",
        )
        ctk.CTkLabel(
            result_shell,
            textvariable=self.registration_status_var,
            text_color=COLORS["ink"],
            font=FONTS["card_title"],
            height=18,
            anchor="w",
        ).grid(row=0, column=1, padx=(0, 14), pady=(6, 1), sticky="w")
        ctk.CTkLabel(
            result_shell,
            textvariable=self.registration_detail_var,
            text_color=COLORS["muted"],
            font=FONTS["body_small"],
            height=18,
            anchor="w",
            justify="left",
            wraplength=720,
        ).grid(row=1, column=1, padx=(0, 14), sticky="w")
        ctk.CTkLabel(
            result_shell,
            textvariable=self.registration_id_var,
            text_color=COLORS["subtle"],
            font=FONTS["mono"],
            height=18,
            anchor="w",
        ).grid(row=2, column=1, padx=(0, 14), pady=(2, 6), sticky="w")

        actions = ctk.CTkFrame(card, fg_color="transparent")
        actions.grid(row=2, column=0, padx=16, pady=(0, 10), sticky="ew")
        actions.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            actions,
            text="Training remains a separate step and is never started here.",
            text_color=COLORS["subtle"],
            font=FONTS["caption"],
            height=18,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        self.open_registered_button = ctk.CTkButton(
            actions,
            text="View registered files",
            width=156,
            height=36,
            corner_radius=12,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            state="disabled",
            command=self.open_registered_folder,
        )
        self.open_registered_button.grid(row=0, column=1, padx=(8, 8), sticky="e")
        self.register_button = ctk.CTkButton(
            actions,
            text="Validate and register  →",
            width=208,
            height=36,
            corner_radius=12,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            font=FONTS["button"],
            state="disabled",
            command=self.validate_and_register_dataset,
        )
        self.register_button.grid(row=0, column=2, sticky="e")

    def set_project(self, project: Project | None) -> None:
        if (
            self.sample_generator_dialog is not None
            and self.sample_generator_dialog.winfo_exists()
        ):
            self.sample_generator_dialog.destroy()
        self.sample_generator_dialog = None
        self.generated_lhs_input_path = None
        self.project = project
        self.discovery = None
        self._clear_inputs()
        self.status_dot.configure(fg_color=COLORS["warning"])
        self.status_var.set("Choose a source to begin.")
        self.prepare_button.configure(
            state="disabled",
            text="Prepare input + output  →",
        )
        self.open_prepared_button.configure(state="disabled")
        self._reset_registration_panel()
        self.pair_load_in_progress = False
        self.last_pair_signature = None
        for index in range(4):
            self._set_subtask_status(
                index,
                "active" if index == 0 else "pending",
            )
        self._expand_subtask("source")
        if not project:
            self.path_var.set("")
            self.input_path_var.set("")
            self.output_path_var.set("")
            self.status_var.set("Open a project to begin.")
            return

        prep_state = project.manifest.get("data_prep", {})
        stored_mode = str(prep_state.get("mode") or "pair")
        mode = stored_mode if stored_mode in {"pair", "parameters"} else "pair"
        self.mode_var.set(mode)
        self.mode_control.set(
            "Input + output files" if mode == "pair" else "#Parameters sweep"
        )
        self._mode_changed(self.mode_control.get())
        self.path_var.set(prep_state.get("source_path", ""))
        self.input_path_var.set(prep_state.get("source_input_path", ""))
        self.output_path_var.set(prep_state.get("source_output_path", ""))
        if (
            stored_mode in {"pair", "parameters"}
            and prep_state.get("available_inputs")
            and prep_state.get("available_outputs")
        ):
            self.discovery = DiscoveryResult(
                mode=mode,
                files=list(prep_state.get("files", [])),
                input_variables=list(prep_state.get("available_inputs", [])),
                output_variables=list(prep_state.get("available_outputs", [])),
                sample_count=int(prep_state.get("sample_count", 0)),
            )
            self._render_discovery(
                self.discovery,
                selected_inputs=list(prep_state.get("selected_inputs", [])),
                selected_output=str(prep_state.get("selected_output", "")),
            )
            self.status_var.set(
                "CSV pair accepted · all columns selected automatically."
                if mode == "pair"
                else "Source discovered. Select inputs and one output."
            )
            self.status_dot.configure(fg_color=COLORS["warning"])
            self._set_subtask_status(0, "complete")
            contract_confirmed = bool(
                prep_state.get("variable_contract_confirmed")
            )
            if mode == "pair" or contract_confirmed:
                self._set_subtask_status(1, "complete")
                self._set_subtask_status(2, "active")
                self.prepare_button.configure(state="normal")
                if mode == "parameters":
                    self.confirm_variables_button.configure(
                        state="disabled",
                        text="Selection saved  ✓",
                    )
                    self.variable_action_note.configure(
                        text="Variable contract saved in this project."
                    )
                self._expand_subtask("prepare")
            else:
                self._set_subtask_status(1, "active")
                self._set_subtask_status(2, "pending")
                self._expand_subtask("variables")
        elif stored_mode == "filename":
            self.status_var.set(
                "Filename sweeps are retired · choose a CSV pair or #Parameters."
            )
            self.status_dot.configure(fg_color=COLORS["warning"])
        prepared_inputs = str(prep_state.get("prepared_inputs_csv") or "")
        prepared_outputs = str(prep_state.get("prepared_outputs_csv") or "")
        if prepared_inputs or prepared_outputs:
            input_path = project.path / prepared_inputs
            output_path = project.path / prepared_outputs
            if (
                prepared_inputs
                and prepared_outputs
                and input_path.exists()
                and output_path.exists()
            ):
                output_columns = prep_state.get(
                    "prepared_output_columns",
                    prep_state.get("theta_points", 0),
                )
                self.status_var.set(
                    f"Data ready · {prep_state.get('prepared_rows', 0)} samples · "
                    f"{output_columns} output columns"
                )
                self.status_dot.configure(fg_color=COLORS["success"])
                self.prepare_button.configure(text="Regenerate both files")
                self.open_prepared_button.configure(state="normal")
                self.register_button.configure(state="normal")
                for index in range(3):
                    self._set_subtask_status(index, "complete")
                self._set_subtask_status(3, "active")
                self._expand_subtask("register")
                registration = prep_state.get("registration", {})
                registered_id = str(
                    registration.get("dataset_id")
                    if isinstance(registration, dict)
                    else ""
                )
                if registered_id:
                    try:
                        registered = get_registered_dataset(
                            project.path,
                            registered_id,
                        )
                    except Exception as exc:
                        self.registration_status_var.set(
                            "Registered dataset needs attention."
                        )
                        self.registration_detail_var.set(str(exc))
                        self.registration_dot.configure(
                            fg_color=COLORS["danger"]
                        )
                    else:
                        self._show_registered_dataset(registered)
            else:
                self.status_var.set(
                    "Input/output pair incomplete · regenerate both."
                )
                self.status_dot.configure(fg_color=COLORS["warning"])
                self.prepare_button.configure(text="Prepare input + output  →")
                self.open_prepared_button.configure(state="disabled")
                self._set_subtask_status(2, "error")
                self._expand_subtask("prepare")
        elif prep_state.get("prepared_csv"):
            legacy_path = project.path / str(prep_state["prepared_csv"])
            if legacy_path.exists():
                self.status_var.set(
                    "Legacy table found · create the new file pair."
                )
                self.status_dot.configure(fg_color=COLORS["warning"])
                self.prepare_button.configure(text="Generate separate files")
                self.open_prepared_button.configure(state="normal")

    def _mode_changed(self, value: str) -> None:
        mode = "pair" if value.startswith("Input") else "parameters"
        self.mode_var.set(mode)
        if mode == "pair":
            self.parameter_path_frame.grid_remove()
            self.pair_paths_frame.grid(
                row=2, column=0, padx=16, pady=(0, 6), sticky="ew"
            )
            self.source_hint_label.configure(
                text=(
                    "Choose both CSVs. All columns are adopted and project copies "
                    "are prepared automatically."
                )
            )
            self.analyze_button.grid_remove()
        else:
            self.pair_paths_frame.grid_remove()
            self.parameter_path_frame.grid(
                row=2, column=0, padx=16, pady=(0, 6), sticky="ew"
            )
            self.source_hint_label.configure(
                text="Choose a .txt export or folder containing #Parameters blocks."
            )
            self.analyze_button.configure(text="Parse")
            self.analyze_button.grid()
        if self.discovery and self.discovery.mode != mode:
            self.discovery = None
            self._clear_inputs()
            self.discovery_summary.configure(
                text="Source type changed. Parse or load the source again."
            )
            self.prepare_button.configure(state="disabled")

    def _browse_input_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose model input CSV",
            filetypes=[("CSV data tables", "*.csv"), ("All files", "*.*")],
            parent=self,
        )
        if path:
            self.generated_lhs_input_path = None
            self.input_path_var.set(path)
            self._maybe_load_pair()

    def _browse_output_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose model output CSV",
            filetypes=[("CSV data tables", "*.csv"), ("All files", "*.*")],
            parent=self,
        )
        if path:
            self.output_path_var.set(path)
            self._maybe_load_pair()

    def _browse_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose antenna simulation export",
            filetypes=[("Text simulation exports", "*.txt"), ("All files", "*.*")],
            parent=self,
        )
        if path:
            self.path_var.set(path)

    def _browse_folder(self) -> None:
        path = filedialog.askdirectory(
            title="Choose antenna simulation export folder",
            parent=self,
        )
        if path:
            self.path_var.set(path)

    def create_input_output_templates(self) -> None:
        if not self.project:
            messagebox.showwarning(
                "Open a project",
                "Create or open a project before creating data templates.",
                parent=self,
            )
            return
        folder = self.project.path / "data" / "templates" / "input_output"
        try:
            input_template, output_template, instructions = (
                write_input_output_templates(folder)
            )
        except OSError as exc:
            messagebox.showerror(
                "Could not create templates",
                str(exc),
                parent=self,
            )
            return

        self.mode_control.set("Input + output files")
        self._mode_changed("Input + output files")
        self.input_path_var.set(str(input_template))
        self.output_path_var.set(str(output_template))
        messagebox.showinfo(
            "Input/output templates ready",
            (
                "The template paths are now loaded into Data Prep.\n\n"
                f"Input template:\n{input_template}\n\n"
                f"Output template:\n{output_template}\n\n"
                f"Instructions:\n{instructions}"
            ),
            parent=self,
        )
        self._maybe_load_pair()
        self._open_local_folder(folder, "Could not open template folder")

    def open_lhs_sample_generator(self) -> None:
        if not self.project:
            messagebox.showwarning(
                "Open a project",
                "Create or open a project before generating simulation samples.",
                parent=self,
            )
            return
        if (
            self.sample_generator_dialog is not None
            and self.sample_generator_dialog.winfo_exists()
        ):
            self.sample_generator_dialog.lift()
            self.sample_generator_dialog.focus_force()
            return
        self.sample_generator_dialog = LHSSampleGeneratorDialog(
            self,
            project_path=self.project.path,
            on_export=self._lhs_samples_exported,
        )

    def _lhs_samples_exported(self, path: Path) -> None:
        """Load generated inputs without pretending solver outputs already exist."""

        self.mode_control.set("Input + output files")
        self._mode_changed("Input + output files")
        self.input_path_var.set(str(path))
        self.generated_lhs_input_path = str(path)
        self.output_path_var.set("")
        self.discovery = None
        self.last_pair_signature = None
        self._clear_inputs()
        self.discovery_summary.configure(
            text=(
                "LHS input samples are ready. Run them in your simulator, then "
                "choose the matching output CSV."
            )
        )
        self.status_var.set(
            "LHS inputs loaded · matching simulation outputs are still required."
        )
        self.status_dot.configure(fg_color=COLORS["warning"])
        self.prepare_button.configure(state="disabled")
        self._set_subtask_status(0, "active")
        for index in range(1, 4):
            self._set_subtask_status(index, "pending")
        self._expand_subtask("source")

    def _maybe_load_pair(self) -> None:
        if not self.project or self.mode_var.get() != "pair":
            return
        input_path = self.input_path_var.get().strip()
        output_path = self.output_path_var.get().strip()
        if not input_path or not output_path:
            return
        signature = (input_path, output_path)
        if self.pair_load_in_progress:
            return
        if self.last_pair_signature == signature and self.discovery:
            return

        self.pair_load_in_progress = True
        self._set_busy(True, "Loading and validating the CSV pair…")
        self._set_subtask_status(0, "active")

        def worker() -> None:
            try:
                result = discover(
                    "pair",
                    input_path,
                    output_path=output_path,
                )
            except Exception as exc:
                self.after(
                    0,
                    lambda error=exc: self._pair_load_failed(error),
                )
                return
            self.after(0, lambda: self._analysis_complete(result))

        threading.Thread(target=worker, daemon=True).start()

    def _pair_load_failed(self, exc: Exception) -> None:
        self.pair_load_in_progress = False
        self.last_pair_signature = None
        self._operation_failed(exc, "source")

    def analyze_source(self) -> None:
        if not self.project:
            return
        if self.mode_var.get() == "pair":
            self._maybe_load_pair()
            return
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning(
                "Choose a raw extract",
                "Select a #Parameters export file or folder first.",
                parent=self,
            )
            return
        self._set_busy(True, "Parsing raw #Parameters extract…")
        self._set_subtask_status(0, "active")

        def worker() -> None:
            try:
                result = discover("parameters", path)
            except Exception as exc:
                self.after(
                    0,
                    lambda error=exc: self._operation_failed(error, "source"),
                )
                return
            self.after(0, lambda: self._analysis_complete(result))

        threading.Thread(target=worker, daemon=True).start()

    def _analysis_complete(self, result: DiscoveryResult) -> None:
        self.pair_load_in_progress = False
        if result.mode == "pair":
            self.last_pair_signature = (
                self.input_path_var.get().strip(),
                self.output_path_var.get().strip(),
            )
        self.discovery = result
        self._render_discovery(result)
        self.status_var.set(
            "CSV pair accepted · all columns selected automatically."
            if result.mode == "pair"
            else "Source discovered. Select inputs and one output."
        )
        self.status_dot.configure(fg_color=COLORS["warning"])
        self.prepare_button.configure(text="Prepare input + output  →")
        self.open_prepared_button.configure(state="disabled")
        self._set_busy(False)
        source_path = (
            self.path_var.get().strip() if result.mode == "parameters" else ""
        )
        source_input_path = (
            self.input_path_var.get().strip() if result.mode == "pair" else ""
        )
        source_output_path = (
            self.output_path_var.get().strip() if result.mode == "pair" else ""
        )
        changes = {
            "workflow": {
                "stage": "data_discovered",
                "completed_steps": 1,
                "next_action": (
                    "Prepare the accepted input/output CSV pair."
                    if result.mode == "pair"
                    else "Select variables and prepare the input/output tables."
                ),
            },
            "data_prep": {
                "mode": result.mode,
                "source_path": source_path,
                "source_input_path": source_input_path,
                "source_output_path": source_output_path,
                "files": result.files,
                "sample_count": result.sample_count,
                "available_inputs": result.input_variables,
                "available_outputs": result.output_variables,
                "selected_inputs": (
                    result.input_variables if result.mode == "pair" else []
                ),
                "selected_output": (
                    IMPORTED_OUTPUT_LABEL if result.mode == "pair" else None
                ),
                "variable_contract_confirmed": result.mode == "pair",
                "prepared_csv": None,
                "prepared_inputs_csv": None,
                "prepared_outputs_csv": None,
                "target_columns": None,
                "sample_id_column": None,
                "validation": None,
                "registration": None,
            },
        }
        self.project = self.app.update_current_project(changes)
        self._reset_registration_panel()
        if result.mode == "pair":
            self._set_subtask_status(0, "complete")
            self._set_subtask_status(1, "complete")
            self._set_subtask_status(2, "active")
            self._expand_subtask("prepare")
            self.prepare_data()
        else:
            self._set_subtask_status(0, "complete")
            self._set_subtask_status(1, "active")
            self._set_subtask_status(2, "pending")
            self._set_subtask_status(3, "pending")
            self._expand_subtask("variables")

    def _render_discovery(
        self,
        result: DiscoveryResult,
        *,
        selected_inputs: list[str] | None = None,
        selected_output: str = "",
    ) -> None:
        self._clear_inputs()
        if selected_inputs is None and result.mode == "pair":
            selected = set(result.input_variables)
        else:
            selected = set(selected_inputs or [])
        for index, variable in enumerate(result.input_variables):
            state = ctk.BooleanVar(value=variable in selected)
            self.input_checks[variable] = state
            if result.mode != "pair":
                column = index % 4
                grid_row = index // 4
                self.inputs_frame.grid_columnconfigure(column, weight=1)
                ctk.CTkCheckBox(
                    self.inputs_frame,
                    text=variable,
                    variable=state,
                    height=26,
                    corner_radius=5,
                    checkbox_width=18,
                    checkbox_height=18,
                    border_color=COLORS["border_strong"],
                    fg_color=COLORS["primary"],
                    hover_color=COLORS["primary_hover"],
                    text_color=COLORS["ink"],
                    font=FONTS["body_small"],
                    command=self._variable_selection_changed,
                ).grid(
                    row=grid_row,
                    column=column,
                    padx=6,
                    pady=2,
                    sticky="w",
                )
        if result.mode == "pair":
            self.variable_action_row.grid_remove()
            ctk.CTkLabel(
                self.inputs_frame,
                text=(
                    f"All {len(result.input_variables):,} input columns are "
                    "used automatically."
                ),
                text_color=COLORS["success"],
                font=FONTS["body_small"],
                anchor="w",
            ).grid(row=0, column=0, padx=6, pady=6, sticky="w")
            output_summary = (
                f"All {len(result.output_variables):,} output columns"
            )
            self.output_menu.configure(
                values=[output_summary],
                state="disabled",
            )
            self.output_var.set(output_summary)
            self.output_title_label.configure(text="Output table")
            self.output_help_label.configure(
                text="Every response column will be preserved"
            )
            self.output_note_label.configure(
                text=(
                    "Rows remain aligned with the input table; output headers "
                    "are saved in the schema."
                )
            )
            summary = (
                f"CSV contract adopted automatically · {result.sample_count:,} "
                f"samples · {len(result.input_variables)} inputs · "
                f"{len(result.output_variables)} outputs"
            )
        else:
            self.variable_action_row.grid()
            self.confirm_variables_button.configure(state="normal")
            outputs = result.output_variables or ["No outputs found"]
            self.output_menu.configure(values=outputs, state="normal")
            chosen = (
                selected_output if selected_output in outputs else outputs[0]
            )
            self.output_var.set(chosen)
            self.output_title_label.configure(text="Pattern output")
            self.output_help_label.configure(
                text="Select one response to expand across its coordinate grid"
            )
            self.output_note_label.configure(
                text=(
                    "The response and source coordinate grid will be saved in "
                    "the project schema."
                )
            )
            summary = (
                f"Found {result.sample_count:,} samples  ·  "
                f"{len(result.input_variables)} inputs  ·  "
                f"{len(result.output_variables)} outputs"
            )
        self.discovery_summary.configure(text=summary)
        self.prepare_button.configure(
            state="normal" if result.mode == "pair" else "disabled"
        )

    def _clear_inputs(self) -> None:
        self.input_checks.clear()
        for child in self.inputs_frame.winfo_children():
            child.destroy()
        self.output_var.set("")
        self.output_menu.configure(
            values=["Parse a raw extract first"],
            state="disabled",
        )
        self.output_title_label.configure(text="Pattern output")
        self.output_help_label.configure(
            text="Parse a raw extract to inspect output columns"
        )
        self.output_note_label.configure(
            text=(
                "The selected response contract will be saved in the "
                "project schema."
            )
        )
        if hasattr(self, "variable_action_row"):
            self.variable_action_row.grid_remove()
            self.confirm_variables_button.configure(
                state="disabled",
                text="Save selection  →",
            )
            self.variable_action_note.configure(
                text="Save this input/output contract before preparing project tables."
            )

    def _variable_selection_changed(self) -> None:
        if self.mode_var.get() != "parameters" or self.discovery is None:
            return
        self.confirm_variables_button.configure(
            state="normal",
            text="Save selection  →",
        )
        self.prepare_button.configure(state="disabled")
        self.variable_action_note.configure(
            text="Selection changed · save it before preparing project tables."
        )
        self._set_subtask_status(0, "complete")
        self._set_subtask_status(1, "active")
        self._set_subtask_status(2, "pending")

    def _selected_variable_contract(self) -> tuple[list[str], str] | None:
        selected_inputs = [
            name for name, state in self.input_checks.items() if state.get()
        ]
        output = self.output_var.get().strip()
        if not selected_inputs:
            messagebox.showwarning(
                "Select inputs",
                "Select at least one model input.",
                parent=self,
            )
            return None
        if not output:
            messagebox.showwarning(
                "Select an output",
                "Select one pattern output.",
                parent=self,
            )
            return None
        return selected_inputs, output

    def confirm_variable_selection(self) -> None:
        """Persist the raw-extract model contract before preparation."""

        if (
            not self.project
            or not self.discovery
            or self.mode_var.get() != "parameters"
        ):
            return
        contract = self._selected_variable_contract()
        if contract is None:
            return
        selected_inputs, output = contract
        self.project = self.app.update_current_project(
            {
                "workflow": {
                    "stage": "data_discovered",
                    "completed_steps": 1,
                    "next_action": "Prepare the selected input/output contract.",
                },
                "data_prep": {
                    "selected_inputs": selected_inputs,
                    "selected_output": output,
                    "variable_contract_confirmed": True,
                },
            }
        )
        self.confirm_variables_button.configure(
            state="disabled",
            text="Selection saved  ✓",
        )
        self.variable_action_note.configure(
            text="Variable contract saved in this project."
        )
        self.prepare_button.configure(state="normal")
        self.status_var.set("Variable contract saved · prepare project tables.")
        self._set_subtask_status(0, "complete")
        self._set_subtask_status(1, "complete")
        self._set_subtask_status(2, "active")
        self._expand_subtask("prepare")

    def prepare_data(self) -> None:
        if not self.project or not self.discovery:
            return
        mode = self.mode_var.get()
        if mode == "parameters":
            contract = self._selected_variable_contract()
            if contract is None:
                return
            selected_inputs, output = contract
        else:
            selected_inputs = [
                name for name, state in self.input_checks.items() if state.get()
            ]
            output = IMPORTED_OUTPUT_LABEL

        self._set_subtask_status(0, "complete")
        self._set_subtask_status(1, "complete")
        self._set_subtask_status(2, "active")
        self._expand_subtask("prepare")

        prepared_root = self.project.path / "data" / "prepared"
        input_destination = prepared_root / "inputs.csv"
        output_destination = prepared_root / "outputs.csv"
        self._set_busy(True, "Preparing separate input and output tables…")
        if mode == "pair":
            source = self.input_path_var.get().strip()
            source_output = self.output_path_var.get().strip()
        else:
            source = self.path_var.get().strip()
            source_output = None

        def worker() -> None:
            try:
                result = prepare(
                    mode,
                    source,
                    selected_inputs,
                    output,
                    input_destination,
                    output_destination,
                    source_output_path=source_output,
                )
            except Exception as exc:
                self.after(
                    0,
                    lambda error=exc: self._operation_failed(error, "prepare"),
                )
                return
            self.after(0, lambda: self._prepare_complete(result))

        threading.Thread(target=worker, daemon=True).start()

    def _prepare_complete(self, result: PreparedResult) -> None:
        if not self.project:
            return
        schema_path = self.project.path / "data" / "prepared" / "schema.json"
        atomic_write_json(
            schema_path,
            {
                "schema_version": 3,
                "source_mode": result.mode,
                "inputs": result.inputs,
                "target_columns": result.target_columns,
                "sample_id_column": result.sample_id_column,
                "output": result.output,
                "output_columns": (
                    list(self.discovery.output_variables)
                    if self.discovery
                    else []
                ),
                "theta_points": result.theta_points,
                "rows": result.rows,
                "inputs_csv": "inputs.csv",
                "outputs_csv": "outputs.csv",
            },
        )
        relative_inputs_csv = str(
            Path(result.input_csv).resolve().relative_to(self.project.path.resolve())
        )
        relative_outputs_csv = str(
            Path(result.output_csv).resolve().relative_to(self.project.path.resolve())
        )
        legacy_path = (
            self.project.path / "data" / "prepared" / "training_data.csv"
        )
        try:
            legacy_path.unlink(missing_ok=True)
        except OSError:
            pass
        selected_inputs = result.inputs
        changes = {
            "workflow": {
                "stage": "data_prepared",
                "completed_steps": 2,
                "next_action": "Configure and train the first surrogate-model book.",
            },
            "data_prep": {
                "selected_inputs": selected_inputs,
                "selected_output": result.output,
                "variable_contract_confirmed": True,
                "prepared_csv": None,
                "prepared_inputs_csv": relative_inputs_csv,
                "prepared_outputs_csv": relative_outputs_csv,
                "prepared_rows": result.rows,
                "prepared_columns": result.columns,
                "prepared_input_columns": result.input_columns,
                "prepared_output_columns": result.output_columns,
                "target_columns": result.target_columns,
                "sample_id_column": result.sample_id_column,
                "theta_points": result.theta_points,
                "schema": "data/prepared/schema.json",
                "validation": None,
                "registration": None,
            },
        }
        self.project = self.app.update_current_project(changes)
        self._set_busy(False)
        self._reset_registration_panel(prepared_ready=True)
        for index in range(3):
            self._set_subtask_status(index, "complete")
        self._set_subtask_status(3, "active")
        self._expand_subtask("register")
        self.status_var.set(
            f"Data ready · {result.rows:,} samples · "
            f"{result.output_columns:,} output columns"
        )
        self.status_dot.configure(fg_color=COLORS["success"])
        self.prepare_button.configure(text="Regenerate both files")
        self.open_prepared_button.configure(state="normal")
        messagebox.showinfo(
            "Input and output files ready",
            (
                f"Prepared {result.rows:,} matched samples.\n\n"
                f"Input table:\n{result.input_csv}\n\n"
                f"Output table:\n{result.output_csv}"
            ),
            parent=self,
        )

    def _prepared_pair_paths(self) -> tuple[Path, Path] | None:
        if not self.project:
            return None
        prep_state = self.project.manifest.get("data_prep", {})
        input_value = str(prep_state.get("prepared_inputs_csv") or "")
        output_value = str(prep_state.get("prepared_outputs_csv") or "")
        if not input_value or not output_value:
            return None
        input_path = self.project.path / input_value
        output_path = self.project.path / output_value
        if not input_path.is_file() or not output_path.is_file():
            return None
        return input_path, output_path

    def _reset_registration_panel(self, *, prepared_ready: bool = False) -> None:
        self.registered_dataset_id = None
        self.registration_status_var.set(
            "Ready to validate."
            if prepared_ready
            else "Prepare data to enable validation."
        )
        self.registration_detail_var.set("No validation result yet.")
        self.registration_id_var.set("Dataset ID: not registered")
        self.registration_dot.configure(
            fg_color=(
                COLORS["warning"] if prepared_ready else COLORS["disabled"]
            )
        )
        self.register_button.configure(
            state="normal" if prepared_ready else "disabled",
            text="Validate and register  →",
        )
        self.open_registered_button.configure(state="disabled")

    def _show_registered_dataset(self, dataset: RegisteredDataset) -> None:
        self.registered_dataset_id = dataset.dataset_id
        self.registration_status_var.set(
            "Validation passed · Dataset registered"
        )
        self.registration_detail_var.set(
            f"{dataset.sample_count:,} samples · "
            f"{dataset.feature_count:,} inputs · "
            f"{dataset.target_count:,} outputs"
        )
        self.registration_id_var.set(
            f"Dataset ID: {dataset.dataset_id} · "
            f"Fingerprint: {dataset.fingerprint_sha256[:16]}…"
        )
        self.registration_dot.configure(fg_color=COLORS["success"])
        self.register_button.configure(
            state="normal",
            text="Revalidate and register",
        )
        self.open_registered_button.configure(state="normal")
        self._set_subtask_status(3, "complete")

    def validate_and_register_dataset(self) -> None:
        if not self.project:
            return
        prepared_paths = self._prepared_pair_paths()
        if not prepared_paths:
            messagebox.showwarning(
                "Prepare data first",
                "Create the prepared input and output tables before validation.",
                parent=self,
            )
            return

        input_path, output_path = prepared_paths
        prep_state = self.project.manifest.get("data_prep", {})
        feature_columns = list(prep_state.get("selected_inputs") or [])
        target_columns = list(prep_state.get("target_columns") or [])
        sample_id_column = (
            str(prep_state.get("sample_id_column") or "").strip() or None
        )
        try:
            discovered_inputs, discovered_targets = read_dataset_columns(
                input_path,
                output_path,
            )
            if not feature_columns:
                feature_columns = [
                    column
                    for column in discovered_inputs
                    if column != sample_id_column
                ]
            if not target_columns:
                target_columns = [
                    column
                    for column in discovered_targets
                    if column != sample_id_column
                ]
            request = TrainingRequest(
                input_csv_path=input_path,
                output_csv_path=output_path,
                feature_columns=feature_columns,
                target_columns=target_columns,
                sample_id_column=sample_id_column,
            )
        except Exception as exc:
            self._registration_failed(exc)
            return

        project_path = self.project.path
        self._set_busy(True)
        self.register_button.configure(state="disabled")
        self.open_registered_button.configure(state="disabled")
        self.registration_status_var.set("Validating prepared dataset…")
        self.registration_detail_var.set(
            "Checking columns, row alignment, and finite numeric values."
        )
        self.registration_id_var.set("Dataset ID: pending validation")
        self.registration_dot.configure(fg_color=COLORS["primary"])

        def worker() -> None:
            try:
                validation = validate_dataset(request)
                registered = register_dataset(project_path, validation)
            except Exception as exc:
                self.after(
                    0,
                    lambda error=exc: self._registration_failed(error),
                )
                return
            self.after(
                0,
                lambda: self._registration_complete(validation, registered),
            )

        threading.Thread(target=worker, daemon=True).start()

    def _registration_complete(
        self,
        validation: DatasetValidationResult,
        registered: RegisteredDataset,
    ) -> None:
        if not self.project:
            return
        relative_manifest = registered.manifest_path.resolve().relative_to(
            self.project.path.resolve()
        )
        changes = {
            "workflow": {
                "stage": "dataset_registered",
                "completed_steps": 2,
                "next_action": "Configure and train the first surrogate-model book.",
            },
            "data_prep": {
                "target_columns": list(validation.target_columns),
                "sample_id_column": validation.sample_id_column,
                "validation": {
                    "status": "passed",
                    "sample_count": validation.sample_count,
                    "feature_count": validation.feature_count,
                    "target_count": validation.target_count,
                },
                "registration": {
                    "dataset_id": registered.dataset_id,
                    "name": registered.name,
                    "fingerprint_sha256": registered.fingerprint_sha256,
                    "manifest": relative_manifest.as_posix(),
                },
            },
        }
        self.project = self.app.update_current_project(changes)
        self._set_busy(False)
        self._show_registered_dataset(registered)
        self.status_var.set(
            "Data ready · validation passed · dataset registered"
        )
        self.status_dot.configure(fg_color=COLORS["success"])
        messagebox.showinfo(
            "Dataset registered",
            (
                f"Validation passed for {validation.sample_count:,} samples.\n\n"
                f"Dataset ID:\n{registered.dataset_id}\n\n"
                f"Registered files:\n{registered.input_csv_path.parent}"
            ),
            parent=self,
        )

    def _registration_failed(self, exc: Exception) -> None:
        self._set_busy(False)
        self._set_subtask_status(3, "error")
        self._expand_subtask("register")
        self.registration_status_var.set("Validation or registration failed")
        self.registration_detail_var.set(str(exc))
        self.registration_id_var.set("Dataset ID: not registered")
        self.registration_dot.configure(fg_color=COLORS["danger"])
        self.register_button.configure(
            state=("normal" if self._prepared_pair_paths() else "disabled")
        )
        messagebox.showerror(
            "Dataset could not be registered",
            str(exc),
            parent=self,
        )

    def open_registered_folder(self) -> None:
        if not self.project or not self.registered_dataset_id:
            return
        folder = (
            self.project.path
            / "data"
            / "registered"
            / self.registered_dataset_id
        )
        if not folder.is_dir():
            messagebox.showwarning(
                "Registered dataset unavailable",
                "The registered dataset folder is missing.",
                parent=self,
            )
            return
        self._open_local_folder(folder, "Could not open registered files")

    def open_prepared_folder(self) -> None:
        if not self.project:
            return
        folder = self.project.path / "data" / "prepared"
        if not folder.exists():
            messagebox.showwarning(
                "Prepared files unavailable",
                "The prepared-data folder is missing. Prepare the data again.",
                parent=self,
            )
            return
        self._open_local_folder(folder, "Could not open prepared files")

    def _open_local_folder(self, folder: Path, error_title: str) -> None:
        try:
            if os.name == "nt":
                os.startfile(str(folder))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except OSError as exc:
            messagebox.showerror(
                error_title,
                str(exc),
                parent=self,
            )

    def _operation_failed(
        self,
        exc: Exception,
        subtask: str | None = None,
    ) -> None:
        self._set_busy(False)
        index_by_subtask = {
            "source": 0,
            "variables": 1,
            "prepare": 2,
            "register": 3,
        }
        self._set_subtask_status(
            index_by_subtask.get(subtask or self.active_subtask, 0),
            "error",
        )
        self.status_var.set("The operation stopped. Review the source and try again.")
        self.status_dot.configure(fg_color=COLORS["danger"])
        title = "Data source could not be processed"
        messagebox.showerror(title, str(exc), parent=self)

    def _set_busy(self, busy: bool, status: str | None = None) -> None:
        state = "disabled" if busy else "normal"
        self.analyze_button.configure(state=state)
        self.sample_generator_button.configure(state=state)
        self.template_button.configure(state=state)
        raw_contract_ready = (
            self.mode_var.get() != "parameters"
            or self.confirm_variables_button.cget("text").startswith(
                "Selection saved"
            )
        )
        self.prepare_button.configure(
            state=(
                "disabled"
                if busy or not self.discovery or not raw_contract_ready
                else "normal"
            )
        )
        if busy:
            self.confirm_variables_button.configure(state="disabled")
        elif self.mode_var.get() == "parameters" and self.discovery is not None:
            self.confirm_variables_button.configure(
                state="disabled" if raw_contract_ready else "normal"
            )
        self.register_button.configure(
            state=(
                "disabled"
                if busy or not self._prepared_pair_paths()
                else "normal"
            )
        )
        if status:
            self.status_var.set(status)
            self.status_dot.configure(fg_color=COLORS["primary"])


class ModelTrainingPage(ctk.CTkFrame):
    """Configuration and training page for supported surrogate models."""

    def __init__(self, parent: ctk.CTkFrame, app: StudioApp):
        super().__init__(parent, fg_color=COLORS["app_bg"], corner_radius=0)
        self.app = app
        self.project: Project | None = None
        self.state = TrainingPageState()
        self.selected_model = ctk.StringVar(value=self.state.selected_model)
        self.training_mode = ctk.StringVar(value=self.state.training_mode)
        self.search_level = ctk.StringVar(value=self.state.search_level)
        self.fit_intercept = ctk.BooleanVar(
            value=self.state.custom_hyperparameters["fit_intercept"]
        )
        self.positive = ctk.BooleanVar(
            value=self.state.custom_hyperparameters["positive"]
        )
        self.xgboost_parameter_vars = {
            name: ctk.StringVar(value=str(XGBOOST_CUSTOM_DEFAULTS[name]))
            for name in XGBOOST_CUSTOM_PARAMETER_NAMES
        }
        self.neural_network_parameter_vars = {
            "hidden_layer_sizes": ctk.StringVar(
                value=", ".join(
                    str(width)
                    for width in NEURAL_NETWORK_CUSTOM_DEFAULTS[
                        "hidden_layer_sizes"
                    ]
                )
            ),
            "activation": ctk.StringVar(
                value=str(NEURAL_NETWORK_CUSTOM_DEFAULTS["activation"])
            ),
            "learning_rate_init": ctk.StringVar(
                value=str(NEURAL_NETWORK_CUSTOM_DEFAULTS["learning_rate_init"])
            ),
            "batch_size": ctk.StringVar(
                value=str(NEURAL_NETWORK_CUSTOM_DEFAULTS["batch_size"])
            ),
            "max_iter": ctk.StringVar(
                value=str(NEURAL_NETWORK_CUSTOM_DEFAULTS["max_iter"])
            ),
        }
        self.last_training_request: ModelTrainingRequest | None = None
        self.last_training_result: ModelTrainingResult | None = None
        self.training_in_progress = False
        self._training_job_id = 0
        self._training_events: queue.SimpleQueue[
            tuple[int, ModelTrainingResult | None, str | None]
        ] = queue.SimpleQueue()
        self._training_poll_after_id: str | None = None
        self._training_elapsed_after_id: str | None = None
        self._training_started_at = 0.0
        self.latest_run_number: int | None = None
        self.latest_run_var = ctk.StringVar(value="Latest Run: None")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_header()
        self._build_content()
        self._build_footer()
        self._apply_training_mode()

    @property
    def custom_hyperparameters(self) -> dict[str, bool]:
        return dict(self.state.custom_hyperparameters)

    @property
    def xgboost_custom_hyperparameters(self) -> dict[str, int | float]:
        return dict(self.state.xgboost_custom_hyperparameters)

    @property
    def neural_network_custom_hyperparameters(self) -> dict[str, object]:
        return dict(self.state.neural_network_custom_hyperparameters)

    def set_project(self, project: Project | None) -> None:
        previous_path = self.project.path if self.project else None
        self.project = project
        next_path = project.path if project else None
        if previous_path != next_path:
            self._reset_ui_state()
        elif project is None:
            self.last_training_request = None
            self.last_training_result = None
        self._load_latest_run(project)
        self.train_button.configure(
            state="normal" if project else "disabled"
        )

    def describe_ui_state(self) -> list[str]:
        return [
            f"Selected model: {self.state.selected_model}",
            f"Training mode: {self.state.training_mode}",
            f"Auto search level: {self.state.search_level}",
            (
                "Auto Search Level available: yes"
                if self.state.auto_search_enabled
                else "Auto Search Level available: no"
            ),
            (
                "Advanced Settings enabled: yes"
                if self.state.advanced_settings_enabled
                else "Advanced Settings enabled: no"
            ),
            (
                "Advanced Settings visible: yes"
                if self.state.advanced_settings_enabled
                else "Advanced Settings visible: no"
            ),
            self._custom_hyperparameter_snapshot(),
            (
                "Last validated training request: available"
                if self.last_training_request is not None
                else "Last validated training request: none"
            ),
            (
                f"Last training result: {self.last_training_result.status}"
                if self.last_training_result is not None
                else "Last training result: none"
            ),
            (
                "Training in progress: yes"
                if self.training_in_progress
                else "Training in progress: no"
            ),
            self.latest_run_var.get(),
            (
                "Train Model action: run the selected supported model family "
                "with its validated configuration"
            ),
            "Model Training page scrolling: none",
        ]

    def _custom_hyperparameter_snapshot(self) -> str:
        if self.state.selected_model == "Ensemble AI Engine":
            return (
                "Ensemble components: Linear Regression, XGBoost, and Neural "
                "Network · Auto High"
            )
        if self.state.selected_model == "Neural Network":
            values = ", ".join(
                f"{name}={self.neural_network_parameter_vars[name].get()}"
                for name in NEURAL_NETWORK_CUSTOM_PARAMETER_NAMES
            )
            return f"Custom Neural Network hyperparameters: {values}"
        if self.state.selected_model == "XGBoost":
            values = ", ".join(
                f"{name}={self.xgboost_parameter_vars[name].get()}"
                for name in XGBOOST_CUSTOM_PARAMETER_NAMES
            )
            return f"Custom XGBoost hyperparameters: {values}"
        return (
            "Custom Linear Regression hyperparameters: "
            f"fit_intercept={self.fit_intercept.get()}, "
            f"positive={self.positive.get()}"
        )

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, padx=28, pady=(14, 8), sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="Model Training",
            text_color=COLORS["ink"],
            font=FONTS["display"],
            height=34,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header,
            text="Configure how the surrogate model will be trained.",
            text_color=COLORS["muted"],
            font=FONTS["body"],
            height=18,
            anchor="w",
        ).grid(row=1, column=0, pady=(3, 0), sticky="w")
        ctk.CTkLabel(
            header,
            text="LOCAL  ·  MODEL TRAINING",
            height=28,
            corner_radius=14,
            fg_color=COLORS["primary_soft"],
            text_color=COLORS["cyan"],
            font=FONTS["mono"],
        ).grid(row=0, column=1, rowspan=2, sticky="e")

    def _build_content(self) -> None:
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.grid(row=1, column=0, padx=28, pady=(0, 8), sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(3, weight=1)

        model_card = self._section_card(content, row=0)
        model_card.grid_columnconfigure(1, weight=1)
        self._section_heading(
            model_card,
            "1",
            "Model selection",
            "Choose the estimator family for this training run.",
        )
        ctk.CTkLabel(
            model_card,
            text="Model",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            anchor="w",
        ).grid(row=1, column=0, padx=(18, 12), pady=(5, 14), sticky="w")
        self.model_dropdown_shell = ctk.CTkFrame(
            model_card,
            height=40,
            corner_radius=11,
            fg_color=COLORS["control"],
            border_width=1,
            border_color=COLORS["border"],
        )
        self.model_dropdown_shell.grid(
            row=1,
            column=1,
            padx=(0, 18),
            pady=(5, 14),
            sticky="ew",
        )
        self.model_dropdown_shell.grid_columnconfigure(0, weight=1)
        self.model_dropdown = ctk.CTkOptionMenu(
            self.model_dropdown_shell,
            variable=self.selected_model,
            values=list(SUPPORTED_MODELS),
            height=38,
            corner_radius=10,
            fg_color=COLORS["control"],
            button_color=COLORS["surface_elevated"],
            button_hover_color=COLORS["control_hover"],
            text_color=COLORS["ink"],
            text_color_disabled=COLORS["disabled_text"],
            dropdown_fg_color=COLORS["surface"],
            dropdown_hover_color=COLORS["nav_active"],
            dropdown_text_color=COLORS["ink"],
            font=FONTS["body_small"],
            dropdown_font=FONTS["body_small"],
            anchor="w",
            command=self._model_changed,
        )
        self.model_dropdown.grid(
            row=0,
            column=0,
            padx=1,
            pady=1,
            sticky="ew",
        )

        self.training_mode_card = self._section_card(content, row=1)
        self.training_mode_card.grid_columnconfigure(0, weight=1)
        self._section_heading(
            self.training_mode_card,
            "2",
            "Training mode",
            "Available configuration paths depend on the selected model.",
        )
        self.training_mode_control = ctk.CTkSegmentedButton(
            self.training_mode_card,
            variable=self.training_mode,
            values=list(TRAINING_MODES),
            height=36,
            corner_radius=10,
            selected_color=COLORS["primary"],
            selected_hover_color=COLORS["primary_hover"],
            unselected_color=COLORS["surface_alt"],
            unselected_hover_color=COLORS["control_hover"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            command=self._training_mode_changed,
        )
        self.training_mode_control.grid(
            row=1,
            column=0,
            padx=18,
            pady=(5, 8),
            sticky="w",
        )
        self.training_mode_control.set(self.state.training_mode)

        self.auto_search_frame = ctk.CTkFrame(
            self.training_mode_card,
            fg_color=COLORS["surface_alt"],
            corner_radius=12,
        )
        self.auto_search_frame.grid(
            row=2,
            column=0,
            padx=18,
            pady=(0, 14),
            sticky="ew",
        )
        self.auto_search_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            self.auto_search_frame,
            text="Auto Search Level",
            text_color=COLORS["ink"],
            font=FONTS["card_title"],
            anchor="w",
        ).grid(row=0, column=0, padx=14, pady=(10, 2), sticky="w")
        self.search_level_dropdown_shell = ctk.CTkFrame(
            self.auto_search_frame,
            width=150,
            height=38,
            corner_radius=11,
            fg_color=COLORS["control"],
            border_width=1,
            border_color=COLORS["border"],
        )
        self.search_level_dropdown_shell.grid(
            row=0,
            column=1,
            padx=14,
            pady=(8, 2),
            sticky="e",
        )
        self.search_level_dropdown_shell.grid_propagate(False)
        self.search_level_dropdown_shell.grid_columnconfigure(0, weight=1)
        self.search_level_dropdown_shell.grid_rowconfigure(0, weight=1)
        self.search_level_dropdown = ctk.CTkOptionMenu(
            self.search_level_dropdown_shell,
            variable=self.search_level,
            values=list(AUTO_SEARCH_LEVELS),
            width=148,
            height=36,
            corner_radius=10,
            fg_color=COLORS["control"],
            button_color=COLORS["surface_elevated"],
            button_hover_color=COLORS["control_hover"],
            text_color=COLORS["ink"],
            text_color_disabled=COLORS["disabled_text"],
            dropdown_fg_color=COLORS["surface"],
            dropdown_hover_color=COLORS["nav_active"],
            dropdown_text_color=COLORS["ink"],
            font=FONTS["body_small"],
            dropdown_font=FONTS["body_small"],
            anchor="w",
            command=self._search_level_changed,
        )
        self.search_level_dropdown.grid(
            row=0,
            column=0,
            padx=1,
            pady=1,
            sticky="nsew",
        )
        for row, level in enumerate(AUTO_SEARCH_LEVELS, start=1):
            ctk.CTkLabel(
                self.auto_search_frame,
                text=f"{level} — {AUTO_SEARCH_DESCRIPTIONS[level]}",
                text_color=COLORS["muted"],
                font=FONTS["caption"],
                anchor="w",
            ).grid(
                row=row,
                column=0,
                columnspan=2,
                padx=14,
                pady=(2, 7 if row == len(AUTO_SEARCH_LEVELS) else 1),
                sticky="w",
            )

        self.custom_mode_note = ctk.CTkFrame(
            self.training_mode_card,
            fg_color=COLORS["primary_soft"],
            corner_radius=12,
        )
        ctk.CTkLabel(
            self.custom_mode_note,
            text="Custom mode is active · configure Advanced Settings below.",
            text_color=COLORS["cyan"],
            font=FONTS["body_small"],
            anchor="w",
        ).pack(fill="x", padx=14, pady=11)

        self.fixed_baseline_note = ctk.CTkFrame(
            self.training_mode_card,
            fg_color=COLORS["primary_soft"],
            corner_radius=12,
        )
        self.fixed_baseline_label = ctk.CTkLabel(
            self.fixed_baseline_note,
            text=(
                "Fixed XGBoost baseline · select Custom to set the five "
                "supported parameters."
            ),
            text_color=COLORS["cyan"],
            font=FONTS["body_small"],
            anchor="w",
        )
        self.fixed_baseline_label.pack(fill="x", padx=14, pady=11)

        self.ensemble_mode_note = ctk.CTkFrame(
            self.training_mode_card,
            fg_color=COLORS["primary_soft"],
            corner_radius=12,
        )
        ctk.CTkLabel(
            self.ensemble_mode_note,
            text=(
                "Auto High · trains Linear Regression, XGBoost, and Neural "
                "Network, then weights valid models by validation RMSE."
            ),
            text_color=COLORS["cyan"],
            font=FONTS["body_small"],
            anchor="w",
        ).pack(fill="x", padx=14, pady=11)

        self.advanced_card = self._section_card(content, row=2)
        self.advanced_card.grid_columnconfigure(0, weight=1)
        self._section_heading(
            self.advanced_card,
            "3",
            "Advanced Settings",
            "Parameters for the selected model, available only in Custom mode.",
        )
        self.advanced_status = ctk.CTkLabel(
            self.advanced_card,
            text="Disabled in Auto mode",
            height=24,
            corner_radius=12,
            fg_color=COLORS["disabled"],
            text_color=COLORS["disabled_text"],
            font=FONTS["caption"],
        )
        self.advanced_status.grid(
            row=1,
            column=0,
            padx=18,
            pady=(5, 8),
            sticky="w",
        )

        self.linear_settings_row = ctk.CTkFrame(
            self.advanced_card,
            fg_color="transparent",
        )
        self.linear_settings_row.grid(
            row=2, column=0, padx=18, pady=(0, 8), sticky="ew"
        )
        self.fit_intercept_switch = ctk.CTkSwitch(
            self.linear_settings_row,
            text="fit_intercept",
            variable=self.fit_intercept,
            onvalue=True,
            offvalue=False,
            width=178,
            text_color=COLORS["ink"],
            progress_color=COLORS["primary"],
            button_color=COLORS["surface"],
            button_hover_color=COLORS["control_hover"],
            font=FONTS["body_small"],
            state="disabled",
            command=self._custom_hyperparameters_changed,
        )
        self.fit_intercept_switch.pack(side="left")
        self.positive_switch = ctk.CTkSwitch(
            self.linear_settings_row,
            text="positive",
            variable=self.positive,
            onvalue=True,
            offvalue=False,
            width=150,
            text_color=COLORS["ink"],
            progress_color=COLORS["primary"],
            button_color=COLORS["surface"],
            button_hover_color=COLORS["control_hover"],
            font=FONTS["body_small"],
            state="disabled",
            command=self._custom_hyperparameters_changed,
        )
        self.positive_switch.pack(side="left", padx=(18, 0))
        self.linear_settings_help = ctk.CTkLabel(
            self.advanced_card,
            text="Applied directly to Linear Regression for Custom runs.",
            text_color=COLORS["subtle"],
            font=FONTS["caption"],
            anchor="w",
        )
        self.linear_settings_help.grid(
            row=3, column=0, padx=18, pady=(0, 12), sticky="w"
        )

        self.xgboost_settings_grid = ctk.CTkFrame(
            self.advanced_card,
            fg_color="transparent",
        )
        self.xgboost_settings_grid.grid(
            row=2, column=0, padx=14, pady=(0, 8), sticky="ew"
        )
        self.xgboost_parameter_entries: dict[str, ctk.CTkEntry] = {}
        range_labels = {
            "n_estimators": "1–5000",
            "max_depth": "1–64",
            "learning_rate": ">0 to 1",
            "subsample": ">0 to 1",
            "colsample_bytree": ">0 to 1",
        }
        for column, name in enumerate(XGBOOST_CUSTOM_PARAMETER_NAMES):
            self.xgboost_settings_grid.grid_columnconfigure(column, weight=1)
            field = ctk.CTkFrame(
                self.xgboost_settings_grid,
                fg_color="transparent",
            )
            field.grid(row=0, column=column, padx=4, sticky="ew")
            ctk.CTkLabel(
                field,
                text=name,
                text_color=COLORS["ink"],
                font=FONTS["caption"],
                anchor="w",
            ).pack(fill="x", pady=(0, 3))
            entry = ctk.CTkEntry(
                field,
                textvariable=self.xgboost_parameter_vars[name],
                height=36,
                corner_radius=10,
                fg_color=COLORS["control"],
                border_color=COLORS["border"],
                text_color=COLORS["ink"],
                font=FONTS["body_small"],
                state="disabled",
            )
            entry.pack(fill="x")
            self.xgboost_parameter_entries[name] = entry
            ctk.CTkLabel(
                field,
                text=range_labels[name],
                text_color=COLORS["subtle"],
                font=FONTS["caption"],
                anchor="w",
            ).pack(fill="x", pady=(3, 0))
        self.xgboost_settings_help = ctk.CTkLabel(
            self.advanced_card,
            text=(
                "Applied to XGBoost with deterministic runtime settings "
                "preserved."
            ),
            text_color=COLORS["subtle"],
            font=FONTS["caption"],
            anchor="w",
        )
        self.xgboost_settings_help.grid(
            row=3, column=0, padx=18, pady=(0, 12), sticky="w"
        )

        self.neural_network_settings_grid = ctk.CTkFrame(
            self.advanced_card,
            fg_color="transparent",
        )
        self.neural_network_settings_grid.grid(
            row=2, column=0, padx=14, pady=(0, 8), sticky="ew"
        )
        self.neural_network_parameter_controls: dict[str, ctk.CTkBaseClass] = {}
        neural_fields = (
            ("hidden_layer_sizes", "Hidden layers", "e.g. 64, 32"),
            ("activation", "Activation", "relu / tanh"),
            ("learning_rate_init", "Learning rate", ">0 to 1"),
            ("batch_size", "Batch size", "1–65536"),
            ("max_iter", "Epochs", "1–100000"),
        )
        for column, (name, label, hint) in enumerate(neural_fields):
            self.neural_network_settings_grid.grid_columnconfigure(column, weight=1)
            field = ctk.CTkFrame(
                self.neural_network_settings_grid,
                fg_color="transparent",
            )
            field.grid(row=0, column=column, padx=4, sticky="ew")
            ctk.CTkLabel(
                field,
                text=label,
                text_color=COLORS["ink"],
                font=FONTS["caption"],
                anchor="w",
            ).pack(fill="x", pady=(0, 3))
            if name == "activation":
                control: ctk.CTkBaseClass = ctk.CTkOptionMenu(
                    field,
                    variable=self.neural_network_parameter_vars[name],
                    values=["relu", "tanh", "logistic", "identity"],
                    height=36,
                    corner_radius=10,
                    fg_color=COLORS["control"],
                    button_color=COLORS["surface_elevated"],
                    button_hover_color=COLORS["control_hover"],
                    dropdown_fg_color=COLORS["surface"],
                    dropdown_hover_color=COLORS["nav_active"],
                    text_color=COLORS["ink"],
                    dropdown_text_color=COLORS["ink"],
                    font=FONTS["body_small"],
                    dropdown_font=FONTS["body_small"],
                    state="disabled",
                )
            else:
                control = ctk.CTkEntry(
                    field,
                    textvariable=self.neural_network_parameter_vars[name],
                    height=36,
                    corner_radius=10,
                    fg_color=COLORS["control"],
                    border_color=COLORS["border"],
                    text_color=COLORS["ink"],
                    font=FONTS["body_small"],
                    state="disabled",
                )
            control.pack(fill="x")
            self.neural_network_parameter_controls[name] = control
            ctk.CTkLabel(
                field,
                text=hint,
                text_color=COLORS["subtle"],
                font=FONTS["caption"],
                anchor="w",
            ).pack(fill="x", pady=(3, 0))
        self.neural_network_settings_help = ctk.CTkLabel(
            self.advanced_card,
            text=(
                "Inputs are standardized automatically; deterministic runtime "
                "settings remain fixed."
            ),
            text_color=COLORS["subtle"],
            font=FONTS["caption"],
            anchor="w",
        )
        self.neural_network_settings_help.grid(
            row=3, column=0, padx=18, pady=(0, 12), sticky="w"
        )

        self.action_bar = ctk.CTkFrame(
            content,
            fg_color=COLORS["surface"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["border"],
        )
        self.action_bar.grid(
            row=4,
            column=0,
            padx=6,
            pady=(0, 5),
            sticky="ew",
        )
        self.action_bar.grid_columnconfigure(0, weight=1)
        self.latest_run_label = ctk.CTkLabel(
            self.action_bar,
            textvariable=self.latest_run_var,
            text_color=COLORS["ink"],
            font=FONTS["card_title"],
            anchor="w",
        )
        self.latest_run_label.grid(
            row=0,
            column=0,
            padx=18,
            pady=12,
            sticky="w",
        )
        self.training_status_label = ctk.CTkLabel(
            self.action_bar,
            text="Local training running · 0:00 elapsed",
            text_color=COLORS["cyan"],
            font=FONTS["caption"],
        )
        self.training_status_label.grid(
            row=0,
            column=1,
            padx=(8, 4),
            sticky="e",
        )
        self.training_status_label.grid_remove()
        self.train_button = ctk.CTkButton(
            self.action_bar,
            text=TRAIN_BUTTON_LABEL,
            width=154,
            height=40,
            corner_radius=11,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            text_color=COLORS["on_primary"],
            font=FONTS["button"],
            state="disabled",
            command=self._train_model,
        )
        self.train_button.grid(row=0, column=2, padx=12, pady=8, sticky="e")

    def _build_footer(self) -> None:
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, padx=28, pady=(0, 10), sticky="ew")
        footer.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(
            footer,
            text="←  Back to Data Prep",
            width=166,
            height=36,
            corner_radius=11,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            command=lambda: self.app.show_page("data"),
        ).grid(row=0, column=0, sticky="w")
        self.training_footer_model_label = ctk.CTkLabel(
            footer,
            text="Linear Regression · Local",
            text_color=COLORS["subtle"],
            font=FONTS["caption"],
        )
        self.training_footer_model_label.grid(row=0, column=1)
        self.results_button = ctk.CTkButton(
            footer,
            text="View Training Results  →",
            width=194,
            height=36,
            corner_radius=11,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            command=lambda: self.app.show_page("results"),
        )
        self.results_button.grid(row=0, column=2, sticky="e")

    def _section_card(self, parent: ctk.CTkFrame, *, row: int) -> ctk.CTkFrame:
        card = ctk.CTkFrame(
            parent,
            fg_color=COLORS["surface"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["border"],
        )
        card.grid(row=row, column=0, padx=6, pady=(0, 6), sticky="ew")
        return card

    def _section_heading(
        self,
        parent: ctk.CTkFrame,
        number: str,
        title: str,
        description: str,
    ) -> None:
        heading = ctk.CTkFrame(parent, fg_color="transparent")
        heading.grid(row=0, column=0, columnspan=2, padx=18, pady=(10, 2), sticky="ew")
        ctk.CTkLabel(
            heading,
            text=number,
            width=26,
            height=26,
            corner_radius=13,
            fg_color=COLORS["primary_soft"],
            text_color=COLORS["cyan"],
            font=FONTS["mono"],
        ).pack(side="left")
        ctk.CTkLabel(
            heading,
            text=title,
            text_color=COLORS["ink"],
            font=FONTS["card_title"],
        ).pack(side="left", padx=(10, 12))
        ctk.CTkLabel(
            heading,
            text=description,
            text_color=COLORS["muted"],
            font=FONTS["caption"],
        ).pack(side="left")

    def _model_changed(self, value: str) -> None:
        self.selected_model.set(value)
        self.state.set_model(value)
        self.training_mode.set(self.state.training_mode)
        self.search_level.set(self.state.search_level)
        self.training_mode_control.set(self.state.training_mode)
        self.training_footer_model_label.configure(text=f"{value} · Local")
        self._apply_training_mode()

    def _training_mode_changed(self, value: str) -> None:
        if self.state.ensemble_mode_enabled:
            self.training_mode.set("Auto")
            self.state.training_mode = "Auto"
            self._apply_training_mode()
            return
        self.training_mode.set(value)
        self.state.set_training_mode(value)
        self._apply_training_mode()

    def _search_level_changed(self, value: str) -> None:
        if self.state.ensemble_mode_enabled:
            self.search_level.set("High")
            self.state.search_level = "High"
            return
        self.search_level.set(value)
        self.state.set_search_level(value)

    def _custom_hyperparameters_changed(self) -> None:
        self.state.set_custom_hyperparameter(
            "fit_intercept",
            self.fit_intercept.get(),
        )
        self.state.set_custom_hyperparameter(
            "positive",
            self.positive.get(),
        )

    def _apply_training_mode(self) -> None:
        if self.state.ensemble_mode_enabled:
            self.training_mode_control.configure(state="disabled")
            self.training_mode_control.set("Auto")
            self.training_mode.set("Auto")
            self.search_level.set("High")
            self.state.training_mode = "Auto"
            self.state.search_level = "High"
            self.auto_search_frame.grid_remove()
            self.custom_mode_note.grid_remove()
            self.fixed_baseline_note.grid_remove()
            self.ensemble_mode_note.grid(
                row=2,
                column=0,
                padx=18,
                pady=(0, 14),
                sticky="ew",
            )
            self.advanced_card.grid_remove()
        elif self.state.fixed_baseline_enabled:
            self.training_mode_control.configure(state="normal")
            self.auto_search_frame.grid_remove()
            self.custom_mode_note.grid_remove()
            self.fixed_baseline_note.grid(
                row=2,
                column=0,
                padx=18,
                pady=(0, 14),
                sticky="ew",
            )
            self.ensemble_mode_note.grid_remove()
            self.advanced_card.grid_remove()
        elif self.state.auto_search_enabled:
            self.training_mode_control.configure(state="normal")
            self.auto_search_frame.grid(
                row=2,
                column=0,
                padx=18,
                pady=(0, 14),
                sticky="ew",
            )
            self.custom_mode_note.grid_remove()
            self.fixed_baseline_note.grid_remove()
            self.ensemble_mode_note.grid_remove()
            self.advanced_card.grid_remove()
            self.advanced_status.configure(
                text="Disabled in Auto mode",
                fg_color=COLORS["disabled"],
                text_color=COLORS["disabled_text"],
            )
        else:
            self.training_mode_control.configure(state="normal")
            self.auto_search_frame.grid_remove()
            self.fixed_baseline_note.grid_remove()
            self.ensemble_mode_note.grid_remove()
            self.custom_mode_note.grid(
                row=2,
                column=0,
                padx=18,
                pady=(0, 14),
                sticky="ew",
            )
            self.advanced_card.grid()
            self.advanced_status.configure(
                text=f"{self.state.selected_model} · Enabled in Custom mode",
                fg_color=COLORS["success_soft"],
                text_color=COLORS["success"],
            )
        linear_custom = (
            self.state.selected_model == "Linear Regression"
            and self.state.advanced_settings_enabled
        )
        xgboost_custom = (
            self.state.selected_model == "XGBoost"
            and self.state.advanced_settings_enabled
        )
        neural_network_custom = (
            self.state.selected_model == "Neural Network"
            and self.state.advanced_settings_enabled
        )
        if linear_custom:
            self.linear_settings_row.grid()
            self.linear_settings_help.grid()
        else:
            self.linear_settings_row.grid_remove()
            self.linear_settings_help.grid_remove()
        if xgboost_custom:
            self.xgboost_settings_grid.grid()
            self.xgboost_settings_help.grid()
        else:
            self.xgboost_settings_grid.grid_remove()
            self.xgboost_settings_help.grid_remove()
        if neural_network_custom:
            self.neural_network_settings_grid.grid()
            self.neural_network_settings_help.grid()
        else:
            self.neural_network_settings_grid.grid_remove()
            self.neural_network_settings_help.grid_remove()
        linear_state = "normal" if linear_custom else "disabled"
        self.fit_intercept_switch.configure(state=linear_state)
        self.positive_switch.configure(state=linear_state)
        xgboost_state = "normal" if xgboost_custom else "disabled"
        for entry in self.xgboost_parameter_entries.values():
            entry.configure(state=xgboost_state)
        neural_state = "normal" if neural_network_custom else "disabled"
        for control in self.neural_network_parameter_controls.values():
            control.configure(state=neural_state)

    def _reset_ui_state(self) -> None:
        if self.training_in_progress:
            self._training_job_id += 1
        self._set_training_busy(False)
        self.state.reset()
        self.last_training_request = None
        self.last_training_result = None
        self.selected_model.set(self.state.selected_model)
        self.training_mode.set(self.state.training_mode)
        self.search_level.set(self.state.search_level)
        self.fit_intercept.set(
            self.state.custom_hyperparameters["fit_intercept"]
        )
        self.positive.set(self.state.custom_hyperparameters["positive"])
        for name in XGBOOST_CUSTOM_PARAMETER_NAMES:
            self.xgboost_parameter_vars[name].set(
                str(self.state.xgboost_custom_hyperparameters[name])
            )
        for name in NEURAL_NETWORK_CUSTOM_PARAMETER_NAMES:
            value = self.state.neural_network_custom_hyperparameters[name]
            if name == "hidden_layer_sizes":
                rendered = ", ".join(str(width) for width in value)
            else:
                rendered = str(value)
            self.neural_network_parameter_vars[name].set(rendered)
        self.training_mode_control.set(self.state.training_mode)
        self.training_footer_model_label.configure(
            text=f"{self.state.selected_model} · Local"
        )
        self._load_latest_run(self.project)
        self._apply_training_mode()

    def build_model_training_request(self) -> ModelTrainingRequest:
        """Map the current display state to the backend request contract."""

        self.state.selected_model = self.selected_model.get()
        self.state.training_mode = self.training_mode.get()
        self.state.search_level = self.search_level.get()
        self.state.custom_hyperparameters = {
            "fit_intercept": self.fit_intercept.get(),
            "positive": self.positive.get(),
        }

        model_name = MODEL_REQUEST_NAMES.get(
            self.state.selected_model,
            self.state.selected_model.strip(),
        )
        training_mode = TRAINING_MODE_REQUEST_NAMES.get(
            self.state.training_mode,
            self.state.training_mode.strip(),
        )
        if model_name == "ensemble_ai_engine":
            training_mode = "auto"
            search_level = "high"
            custom_hyperparameters = None
        elif model_name == "xgboost" and training_mode == "auto":
            search_level = SEARCH_LEVEL_REQUEST_NAMES.get(
                self.state.search_level,
                self.state.search_level.strip(),
            )
            custom_hyperparameters = None
        elif model_name == "xgboost" and training_mode == "custom":
            search_level = None
            custom_hyperparameters = self._parse_xgboost_custom_parameters()
        elif model_name == "neural_network" and training_mode == "custom":
            search_level = None
            custom_hyperparameters = self._parse_neural_network_custom_parameters()
        elif training_mode == "auto":
            search_level = SEARCH_LEVEL_REQUEST_NAMES.get(
                self.state.search_level,
                self.state.search_level.strip(),
            )
            custom_hyperparameters = None
        elif training_mode == "custom":
            search_level = None
            custom_hyperparameters = dict(self.state.custom_hyperparameters)
        else:
            search_level = None
            custom_hyperparameters = None

        return ModelTrainingRequest(
            model_name=model_name,
            training_mode=training_mode,
            search_level=search_level,
            custom_hyperparameters=custom_hyperparameters,
        )

    def _parse_xgboost_custom_parameters(self) -> dict[str, int | float]:
        """Convert entry text to numeric values; the request owns range validation."""

        parameters: dict[str, int | float] = {}
        for name in XGBOOST_CUSTOM_PARAMETER_NAMES:
            raw = self.xgboost_parameter_vars[name].get().strip()
            if not raw:
                raise ValueError(f"XGBoost parameter '{name}' is required.")
            try:
                if name in {"n_estimators", "max_depth"}:
                    value: int | float = int(raw)
                else:
                    value = float(raw)
            except ValueError as exc:
                kind = "an integer" if name in {"n_estimators", "max_depth"} else "numeric"
                raise ValueError(
                    f"XGBoost parameter '{name}' must be {kind}."
                ) from exc
            parameters[name] = value
            self.state.set_xgboost_custom_hyperparameter(name, value)
        return parameters

    def _parse_neural_network_custom_parameters(self) -> dict[str, object]:
        """Parse user text; the backend request remains the validation authority."""

        raw_layers = self.neural_network_parameter_vars[
            "hidden_layer_sizes"
        ].get().strip()
        if not raw_layers:
            raise ValueError("Neural Network hidden layers are required.")
        try:
            layers = [
                int(value.strip())
                for value in raw_layers.split(",")
                if value.strip()
            ]
        except ValueError as exc:
            raise ValueError(
                "Neural Network hidden layers must be comma-separated integers."
            ) from exc
        if not layers:
            raise ValueError(
                "Neural Network hidden layers must include at least one width."
            )

        parameters: dict[str, object] = {
            "hidden_layer_sizes": layers,
            "activation": self.neural_network_parameter_vars[
                "activation"
            ].get().strip(),
        }
        for name, kind in (
            ("learning_rate_init", "number"),
            ("batch_size", "integer"),
            ("max_iter", "integer"),
        ):
            raw = self.neural_network_parameter_vars[name].get().strip()
            if not raw:
                raise ValueError(f"Neural Network parameter '{name}' is required.")
            try:
                value: object = float(raw) if kind == "number" else int(raw)
            except ValueError as exc:
                raise ValueError(
                    f"Neural Network parameter '{name}' must be a {kind}."
                ) from exc
            parameters[name] = value
        for name, value in parameters.items():
            self.state.set_neural_network_custom_hyperparameter(name, value)
        return parameters

    @staticmethod
    def _display_model_name(model_name: str) -> str:
        return {
            "linear_regression": "Linear Regression",
            "xgboost": "XGBoost",
            "neural_network": "Neural Network",
            "ensemble_ai_engine": "Ensemble AI Engine",
        }.get(model_name, model_name)

    @staticmethod
    def _format_auto_parameters(
        model_name: str,
        parameters: dict[str, object],
    ) -> str:
        if model_name == "ensemble_ai_engine":
            weights = parameters.get("weights", {})
            return "\n".join(
                f"{ModelTrainingPage._display_model_name(name)}: {float(weight):.2%}"
                for name, weight in weights.items()
            )
        names = (
            NEURAL_NETWORK_CUSTOM_PARAMETER_NAMES
            if model_name == "neural_network"
            else (
                XGBOOST_CUSTOM_PARAMETER_NAMES
                if model_name == "xgboost"
                else ("fit_intercept", "positive")
            )
        )
        return "\n".join(f"{name}: {parameters[name]}" for name in names)

    def _train_model(self) -> None:
        self.last_training_request = None
        self.last_training_result = None
        try:
            request = self.build_model_training_request()
        except (TypeError, ValueError) as exc:
            messagebox.showerror(
                "Invalid training configuration",
                str(exc),
                parent=self,
            )
            return

        self.last_training_request = request
        if not self.project:
            messagebox.showerror(
                "Training failed",
                "Open a project before training a model.",
                parent=self,
            )
            return

        self._set_training_busy(True)
        self._training_job_id += 1
        job_id = self._training_job_id
        backend = submit_model_training_request
        project_path = self.project.path
        threading.Thread(
            target=self._training_worker,
            args=(job_id, backend, request, project_path),
            daemon=True,
            name=f"studio-training-{job_id}",
        ).start()
        self._schedule_training_poll()

    def _training_worker(
        self,
        job_id: int,
        backend: Callable[..., ModelTrainingResult],
        request: ModelTrainingRequest,
        project_path: Path,
    ) -> None:
        """Run compute without touching Tk from the worker thread."""

        try:
            result = backend(request, project_path=project_path)
        except Exception:
            self._training_events.put(
                (
                    job_id,
                    None,
                    "Training failed because an unexpected local error occurred.",
                )
            )
            return
        self._training_events.put((job_id, result, None))

    def _schedule_training_poll(self) -> None:
        if self._training_poll_after_id is None:
            self._training_poll_after_id = self.after(
                80,
                self._poll_training_events,
            )

    def _poll_training_events(self) -> None:
        self._training_poll_after_id = None
        try:
            job_id, result, error_message = self._training_events.get_nowait()
        except queue.Empty:
            if self.training_in_progress:
                self._schedule_training_poll()
            return
        if job_id != self._training_job_id:
            if self.training_in_progress:
                self._schedule_training_poll()
            return
        if error_message is not None or result is None:
            self._set_training_busy(False)
            self.last_training_result = None
            self.app.results_page.show_training_failure()
            messagebox.showerror(
                "Training failed",
                error_message or "Training could not be completed.",
                parent=self,
            )
            return
        self._training_completed(result)

    def _training_completed(self, result: ModelTrainingResult) -> None:
        self._set_training_busy(False)
        self.last_training_result = result
        request = self.last_training_request
        if not result.success:
            self.app.results_page.show_training_failure()
            messagebox.showerror(
                "Training failed",
                result.error_message or "Training could not be completed.",
                parent=self,
            )
            return

        self.project = self.app.update_current_project({})
        self._set_latest_run(result.run_number)
        metrics = result.metrics
        parameters = result.parameters_used
        training_mode_label = (
            result.training_mode or (request.training_mode if request else "")
        ).title()
        if result.model_name == "ensemble_ai_engine":
            best_model = self._display_model_name(
                result.best_individual_model or "Unknown"
            )
            decision = (
                "Ensemble recommended"
                if result.ensemble_improved_on_best
                else f"{best_model} remains recommended"
            )
            messagebox.showinfo(
                "Ensemble Training Completed",
                (
                    "Ensemble AI Engine Completed\n\n"
                    "Components trained in Auto High: "
                    f"{len(result.component_results)}\n"
                    f"Component failures: {len(result.component_failures)}\n\n"
                    "Normalized weights:\n"
                    f"{self._format_auto_parameters(result.model_name, parameters)}\n\n"
                    f"Ensemble Validation RMSE: {result.ensemble_validation_rmse:.6g}\n"
                    f"Best Individual Validation RMSE: "
                    f"{result.best_individual_validation_rmse:.6g}\n"
                    f"Recommendation: {decision}\n\n"
                    f"Test MAE: {metrics['MAE']:.6g}\n"
                    f"Test RMSE: {metrics['RMSE']:.6g}\n"
                    f"Test R²: {metrics['R²']:.6g}"
                ),
                parent=self,
            )
            self._open_latest_training_results(result)
            return
        if result.model_name in {"xgboost", "neural_network"} and (
            result.training_mode or request.training_mode
        ) == "custom":
            parameter_names = (
                NEURAL_NETWORK_CUSTOM_PARAMETER_NAMES
                if result.model_name == "neural_network"
                else XGBOOST_CUSTOM_PARAMETER_NAMES
            )
            configuration = (
                "Training Mode: Custom\n\n"
                "Parameters Used:\n"
                + "\n".join(
                    f"{name}: {parameters[name]}"
                    for name in parameter_names
                )
            )
            messagebox.showinfo(
                "Training Completed",
                (
                    "Training Completed\n\n"
                    f"Model: {self._display_model_name(result.model_name)}\n"
                    f"{configuration}\n\n"
                    f"MAE: {metrics['MAE']:.6g}\n"
                    f"RMSE: {metrics['RMSE']:.6g}\n"
                    f"R²: {metrics['R²']:.6g}"
                ),
                parent=self,
            )
            self._open_latest_training_results(result)
            return
        if (
            result.training_mode
            or (request.training_mode if request else "")
        ) == "auto":
            search_level_label = (
                result.search_level
                or (request.search_level if request else None)
                or ""
            ).title()
            validation_rmse = result.best_validation_rmse
            validation_rmse_text = (
                f"{validation_rmse:.6g}"
                if validation_rmse is not None
                else "Unavailable"
            )
            test_metrics = result.test_metrics or metrics
            messagebox.showinfo(
                "Auto Search Completed",
                (
                    "Auto Search Completed\n\n"
                    f"Model: {self._display_model_name(result.model_name)}\n"
                    f"Search Level: {search_level_label}\n"
                    "Configurations Evaluated: "
                    f"{result.configurations_evaluated}\n"
                    "Cross-Validation Folds: "
                    f"{result.cross_validation_folds}\n\n"
                    "Best Parameters:\n"
                    f"{self._format_auto_parameters(result.model_name, parameters)}\n\n"
                    f"Validation RMSE: {validation_rmse_text}\n"
                    f"Test MAE: {test_metrics['MAE']:.6g}\n"
                    f"Test RMSE: {test_metrics['RMSE']:.6g}\n"
                    f"Test R²: {test_metrics['R²']:.6g}"
                ),
                parent=self,
            )
            self._open_latest_training_results(result)
            return
        messagebox.showinfo(
            "Training Completed",
            (
                "Training Completed\n\n"
                "Model: Linear Regression\n"
                f"Training Mode: {training_mode_label}\n\n"
                "Parameters Used:\n"
                f"fit_intercept: {parameters['fit_intercept']}\n"
                f"positive: {parameters['positive']}\n\n"
                f"MAE: {metrics['MAE']:.6g}\n"
                f"RMSE: {metrics['RMSE']:.6g}\n"
                f"R²: {metrics['R²']:.6g}"
            ),
            parent=self,
        )
        self._open_latest_training_results(result)

    def _open_latest_training_results(
        self,
        result: ModelTrainingResult,
    ) -> None:
        if (
            self.project is None
            or result.run_number is None
            or result.run_directory is None
        ):
            return
        self.app.results_page.set_project(self.project)
        self.app.show_page("results")

    def _set_training_busy(self, busy: bool) -> None:
        self.training_in_progress = busy
        self.train_button.configure(
            state="disabled" if busy or not self.project else "normal",
            text="Training…" if busy else TRAIN_BUTTON_LABEL,
        )
        if busy:
            self._training_started_at = time.monotonic()
            self.training_status_label.grid()
            self._update_training_elapsed()
            self.update_idletasks()
        else:
            if self._training_elapsed_after_id is not None:
                try:
                    self.after_cancel(self._training_elapsed_after_id)
                except tk.TclError:
                    pass
                self._training_elapsed_after_id = None
            self.training_status_label.grid_remove()

    def _update_training_elapsed(self) -> None:
        self._training_elapsed_after_id = None
        if not self.training_in_progress:
            return
        elapsed = max(0, int(time.monotonic() - self._training_started_at))
        minutes, seconds = divmod(elapsed, 60)
        self.training_status_label.configure(
            text=f"Local training running · {minutes}:{seconds:02d} elapsed"
        )
        self._training_elapsed_after_id = self.after(
            1000,
            self._update_training_elapsed,
        )

    def _load_latest_run(self, project: Project | None) -> None:
        if project is None:
            self._set_latest_run(None)
            return
        training_state = project.manifest.get("model_training")
        if not isinstance(training_state, dict):
            self._set_latest_run(None)
            return
        raw_number = training_state.get("latest_run_number")
        if raw_number is None and training_state.get("status") == "TRAINING_COMPLETED":
            raw_number = 1
        try:
            run_number = int(raw_number)
        except (TypeError, ValueError):
            run_number = 0
        self._set_latest_run(run_number if run_number > 0 else None)

    def _set_latest_run(self, run_number: int | None) -> None:
        self.latest_run_number = run_number
        self.latest_run_var.set(
            f"Latest Run: Run {run_number}"
            if run_number is not None
            else "Latest Run: None"
        )


class CreateProjectDialog(ctk.CTkToplevel):
    def __init__(
        self,
        parent: StudioApp,
        callback: Callable[[str, str], None],
    ):
        super().__init__(parent, fg_color=COLORS["surface"])
        self.callback = callback
        self.title("Create antenna project")
        self.geometry("540x430")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.after(20, self._center)

        ctk.CTkLabel(
            self,
            text="Create a portable project",
            text_color=COLORS["ink"],
            font=FONTS["title"],
            anchor="w",
        ).pack(fill="x", padx=28, pady=(28, 5))
        ctk.CTkLabel(
            self,
            text="The Studio creates a complete workspace for data, model books, inference, and SnowBuddy history.",
            text_color=COLORS["muted"],
            font=FONTS["body_small"],
            wraplength=470,
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=28)

        ctk.CTkLabel(
            self,
            text="PROJECT NAME",
            text_color=COLORS["muted"],
            font=("Segoe UI Semibold", 14),
            anchor="w",
        ).pack(fill="x", padx=28, pady=(24, 7))
        self.name_entry = ctk.CTkEntry(
            self,
            height=43,
            corner_radius=11,
            border_color=COLORS["border"],
            placeholder_text="e.g. 8-element mmWave array",
            font=FONTS["body_small"],
        )
        self.name_entry.pack(fill="x", padx=28)

        ctk.CTkLabel(
            self,
            text="DESCRIPTION  ·  OPTIONAL",
            text_color=COLORS["muted"],
            font=("Segoe UI Semibold", 14),
            anchor="w",
        ).pack(fill="x", padx=28, pady=(18, 7))
        self.description = ctk.CTkTextbox(
            self,
            height=84,
            corner_radius=11,
            border_width=1,
            border_color=COLORS["border"],
            font=FONTS["body_small"],
        )
        self.description.pack(fill="x", padx=28)

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.pack(fill="x", padx=28, pady=(22, 24))
        ctk.CTkButton(
            actions,
            text="Cancel",
            width=100,
            height=40,
            corner_radius=11,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            text_color=COLORS["ink"],
            command=self.destroy,
        ).pack(side="right")
        ctk.CTkButton(
            actions,
            text="Create project  →",
            width=150,
            height=40,
            corner_radius=11,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            font=FONTS["button"],
            command=self._submit,
        ).pack(side="right", padx=(0, 8))
        self.name_entry.focus_set()
        self.bind("<Return>", lambda _event: self._submit())

    def _submit(self) -> None:
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning(
                "Project name required",
                "Enter a clear project name.",
                parent=self,
            )
            return
        description = self.description.get("1.0", "end").strip()
        self.destroy()
        self.callback(name, description)

    def _center(self) -> None:
        parent = self.master
        x = parent.winfo_rootx() + (parent.winfo_width() - 540) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - 430) // 2
        self.geometry(f"540x430+{max(0, x)}+{max(0, y)}")


class LocalModelDialog(ctk.CTkToplevel):
    def __init__(
        self,
        parent: SnowBuddyPanel,
        service: SnowBuddyService,
        callback: Callable[[], None],
    ):
        super().__init__(parent, fg_color=COLORS["surface"])
        self.service = service
        self.callback = callback
        self.selected_model = ctk.StringVar(value=service.model)
        self.title("SnowBuddy local model")
        self.geometry("580x540")
        self.resizable(False, False)
        self.transient(parent.winfo_toplevel())
        self.grab_set()

        ctk.CTkLabel(
            self,
            text="Run SnowBuddy locally",
            text_color=COLORS["ink"],
            font=FONTS["title"],
            anchor="w",
        ).pack(fill="x", padx=26, pady=(26, 5))
        ctk.CTkLabel(
            self,
            text=(
                "Ollama keeps project context and inference on this computer. "
                "Choose a profile that matches the machine; you can change it later."
            ),
            text_color=COLORS["muted"],
            font=FONTS["body_small"],
            wraplength=520,
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=26)

        memory = total_memory_gb()
        recommendation = model_profile(service.recommendation)
        memory_text = f"{memory:g} GB system RAM detected. " if memory else ""
        recommendation_text = (
            f"{memory_text}Recommended: "
            f"{recommendation.label if recommendation else service.recommendation}."
        )
        ctk.CTkLabel(
            self,
            text=recommendation_text,
            text_color=COLORS["primary"],
            font=("Segoe UI Semibold", 15),
            anchor="w",
        ).pack(fill="x", padx=26, pady=(14, 10))

        profiles = ctk.CTkFrame(self, fg_color="transparent")
        profiles.pack(fill="x", padx=26)
        for profile in MODEL_PROFILES:
            card = ctk.CTkFrame(
                profiles,
                height=88,
                corner_radius=13,
                fg_color=COLORS["surface_alt"],
                border_width=1,
                border_color=COLORS["border"],
            )
            card.pack(fill="x", pady=4)
            card.pack_propagate(False)
            radio = ctk.CTkRadioButton(
                card,
                text=f"{profile.label}  ·  {profile.model}",
                variable=self.selected_model,
                value=profile.model,
                command=self._selection_changed,
                text_color=COLORS["ink"],
                fg_color=COLORS["primary"],
                hover_color=COLORS["primary_hover"],
                font=FONTS["button"],
            )
            radio.pack(anchor="w", padx=16, pady=(13, 3))
            ctk.CTkLabel(
                card,
                text=f"{profile.download_gb:g} GB download  ·  {profile.description}",
                text_color=COLORS["muted"],
                font=FONTS["caption"],
                anchor="w",
            ).pack(fill="x", padx=43)

        status_shell = ctk.CTkFrame(
            self,
            corner_radius=12,
            fg_color=COLORS["surface_alt"],
            border_width=1,
            border_color=COLORS["border"],
        )
        status_shell.pack(fill="x", padx=26, pady=(14, 0))
        self.runtime_status = ctk.CTkLabel(
            status_shell,
            text="Checking the local Ollama runtime…",
            text_color=COLORS["muted"],
            font=FONTS["body_small"],
            anchor="w",
            wraplength=495,
            justify="left",
        )
        self.runtime_status.pack(fill="x", padx=14, pady=11)

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.pack(fill="x", padx=26, pady=(16, 22))
        self.ollama_button = ctk.CTkButton(
            actions,
            text="Get Ollama",
            width=100,
            height=40,
            corner_radius=11,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            text_color=COLORS["ink"],
            command=lambda: webbrowser.open("https://ollama.com/download"),
        )
        self.ollama_button.pack(side="left")
        self.download_button = ctk.CTkButton(
            actions,
            text="Download selected",
            width=145,
            height=40,
            corner_radius=11,
            fg_color=COLORS["violet_soft"],
            hover_color=COLORS["violet_hover"],
            text_color=COLORS["on_violet_soft"],
            command=self._download,
        )
        self.download_button.pack(side="left", padx=8)
        self.use_button = ctk.CTkButton(
            actions,
            text="Use selected",
            width=130,
            height=40,
            corner_radius=11,
            fg_color=COLORS["violet"],
            hover_color=COLORS["violet_hover"],
            font=FONTS["button"],
            command=self._use_selected,
        )
        self.use_button.pack(side="right")
        self.after(10, self._center)
        self.after(80, self._check_runtime)

    def _selection_changed(self) -> None:
        self._check_runtime()

    def _check_runtime(self) -> None:
        selected = self.selected_model.get()
        self.download_button.configure(state="disabled", text="Checking…")
        self.runtime_status.configure(
            text=f"Checking Ollama and {selected}…",
            text_color=COLORS["muted"],
        )

        def worker() -> None:
            status = self.service.runtime_status(selected)
            self.after(0, lambda: self._show_runtime_status(status))

        threading.Thread(target=worker, daemon=True).start()

    def _show_runtime_status(self, status: RuntimeStatus) -> None:
        if not self.winfo_exists():
            return
        if status.ready:
            text = f"Ready locally · {status.model} is installed."
            color = COLORS["success"]
            self.download_button.configure(state="disabled", text="Installed")
        elif status.available:
            text = (
                f"Ollama is running. {status.model} is not downloaded yet; "
                "choose Download selected."
            )
            color = COLORS["warning"]
            self.download_button.configure(
                state="normal", text="Download selected"
            )
        else:
            text = (
                "Ollama is not detected. Choose Get Ollama, install the native "
                "runtime for this operating system, then return here."
            )
            color = COLORS["muted"]
            self.download_button.configure(
                state="disabled", text="Ollama required"
            )
        self.runtime_status.configure(text=text, text_color=color)

    def _download(self) -> None:
        selected = self.selected_model.get()
        self.download_button.configure(state="disabled", text="Downloading…")
        self.use_button.configure(state="disabled")
        self.runtime_status.configure(
            text=(
                f"Downloading {selected}. Keep Ollama running; the first download "
                "can take several minutes."
            ),
            text_color=COLORS["primary"],
        )

        def worker() -> None:
            try:
                self.service.pull_model(selected)
            except Exception as exc:
                self.after(0, lambda: self._download_finished(exc))
                return
            self.after(0, lambda: self._download_finished(None))

        threading.Thread(target=worker, daemon=True).start()

    def _download_finished(self, error: Exception | None) -> None:
        if not self.winfo_exists():
            return
        self.download_button.configure(state="normal", text="Download selected")
        self.use_button.configure(state="normal")
        if error:
            self.runtime_status.configure(
                text=f"Download could not finish: {error}",
                text_color=COLORS["danger"],
            )
            return
        self.callback()
        self._check_runtime()

    def _use_selected(self) -> None:
        try:
            self.service.set_model(self.selected_model.get())
        except AssistantError as exc:
            messagebox.showwarning("Select a model", str(exc), parent=self)
            return
        self.callback()
        self.destroy()

    def _center(self) -> None:
        parent = self.master.winfo_toplevel()
        x = parent.winfo_rootx() + (parent.winfo_width() - 580) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - 540) // 2
        self.geometry(f"580x540+{max(0, x)}+{max(0, y)}")


def _friendly_date(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return "recently"
    now = datetime.now(parsed.tzinfo)
    delta = now - parsed
    if delta.days <= 0:
        return "today"
    if delta.days == 1:
        return "yesterday"
    if delta.days < 7:
        return f"{delta.days} days ago"
    return parsed.strftime("%b %d")


def _display_markdown(value: str) -> str:
    return value.replace("**", "")


def run() -> None:
    app = StudioApp()
    app.mainloop()
