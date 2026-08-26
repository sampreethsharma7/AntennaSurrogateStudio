"""Persistent, non-scrolling inverse-design and scientific-plot workspace."""

from __future__ import annotations

import math
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import customtkinter as ctk

from studio.inverse_design import (
    INVERSE_DESIGN_COMPLETED,
    InverseDesignError,
    InverseDesignObjective,
    InverseDesignRequest,
    InverseDesignResult,
    OutputConstraint,
    load_inverse_design_run,
    submit_inverse_design_request,
)
from studio.model_book import ModelBook, ModelBookError, load_model_library
from studio.project_store import Project
from studio.scientific_plot import (
    CURVE_MANAGER_MIN_WIDTH,
    PLOT_PANE_MIN_WIDTH,
    ScientificPlotWorkbench,
)
from studio.theme import COLORS, FONTS

if TYPE_CHECKING:
    from studio.ui import StudioApp


INPUTS_PER_PAGE = 5
MAX_CONSTRAINTS = 4
INPUT_COLUMN_MIN_WIDTHS = (105, 170, 56, 56, 56)
CONFIGURATION_MIN_WIDTH = 520
CONFIGURATION_DEFAULT_WIDTH = 520
# Include the workbench's plot/manager minima, its sash, and the result-card
# padding.  Tk can otherwise satisfy the outer pane while silently compressing
# both inner panes below their own readable widths.
RESULT_MIN_WIDTH = PLOT_PANE_MIN_WIDTH + CURVE_MANAGER_MIN_WIDTH + 68
INVERSE_DESIGN_MIN_WORKSPACE_WIDTH = (
    48 + CONFIGURATION_MIN_WIDTH + 10 + RESULT_MIN_WIDTH
)
DUPLICATE_RESPONSE_REL_TOLERANCE = 1.0e-4
DUPLICATE_RESPONSE_ABS_TOLERANCE = 1.0e-8
DUPLICATE_INPUT_REL_TOLERANCE = 1.0e-4
DUPLICATE_INPUT_ABS_TOLERANCE = 1.0e-8
GOAL_LABELS = {
    "Minimize": "minimize",
    "Maximize": "maximize",
    "Target value": "target",
}
CONSTRAINT_LABELS = {
    "At least (≥)": "greater_than_or_equal",
    "At most (≤)": "less_than_or_equal",
    "Within range": "within_range",
}
MODEL_LABELS = {
    "linear_regression": "Linear Regression",
    "xgboost": "XGBoost",
    "neural_network": "Neural Network",
    "ensemble_ai_engine": "Ensemble AI Engine",
}


@dataclass(slots=True)
class InputWidgets:
    frame: ctk.CTkFrame
    mode: ctk.StringVar
    mode_control: ctk.CTkSegmentedButton
    lower: ctk.CTkEntry
    upper: ctk.CTkEntry
    fixed: ctk.CTkEntry


@dataclass(slots=True)
class ConstraintWidgets:
    frame: ctk.CTkFrame
    scope: ctk.CTkOptionMenu
    coordinate_start: ctk.CTkEntry
    coordinate_end: ctk.CTkEntry
    operator: ctk.CTkOptionMenu
    first_value: ctk.CTkEntry
    second_value: ctk.CTkEntry
    remove_button: ctk.CTkButton


class InverseDesignPage(ctk.CTkFrame):
    """Keep configuration and accumulated inverse-search curves together."""

    def __init__(self, parent: ctk.CTkFrame, app: "StudioApp"):
        super().__init__(parent, fg_color=COLORS["app_bg"], corner_radius=0)
        self.app = app
        self.project: Project | None = None
        self.active_book: ModelBook | None = None
        self.load_error: str | None = None
        self.optimization_in_progress = False
        self.last_result: InverseDesignResult | None = None
        self._workspace_key: tuple[Path, str] | None = None
        self.input_page = 0
        self.input_widgets: dict[str, InputWidgets] = {}
        self.constraint_widgets: list[ConstraintWidgets] = []
        self._clamping_workspace_sash = False

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_header()
        self._build_workspace()
        self._build_footer()
        self._show_config_section("Inputs")
        self._refresh()

    def set_project(self, project: Project | None) -> None:
        self._workspace_key = None
        self.project = project
        self.reload()

    def reload(self) -> None:
        self.active_book = None
        self.load_error = None
        if self.project is not None:
            try:
                library = load_model_library(self.project.path)
            except ModelBookError as exc:
                self.load_error = str(exc)
            else:
                entry = next(
                    (
                        item
                        for item in library.entries
                        if item.book_id == library.active_book_id
                    ),
                    None,
                )
                if not library.active_book_id:
                    self.load_error = (
                        "No active Model Book is selected. Open Model Library and "
                        "set a valid model as active."
                    )
                elif entry is None:
                    self.load_error = (
                        "The active Model Book is not present in this project's library."
                    )
                elif not entry.is_valid or entry.book is None:
                    self.load_error = (
                        "The active Model Book is unavailable. "
                        f"{entry.error_message or 'Its saved files are invalid.'}"
                    )
                else:
                    self.active_book = entry.book
        new_key = (
            (self.project.path.resolve(), self.active_book.book_id)
            if self.project is not None and self.active_book is not None
            else None
        )
        if new_key is not None and new_key == self._workspace_key:
            self._refresh_model_header()
            return
        self._workspace_key = new_key
        self.last_result = None
        self.input_page = 0
        self._refresh()
        self._restore_latest_result()

    def refresh_theme(self) -> None:
        self.workspace_split.configure(bg=self._palette(COLORS["border"]))
        self.response_plot.refresh_theme()

    def describe_ui_state(self) -> list[str]:
        if self.project is None:
            return ["Inverse Design state: no project"]
        if self.active_book is None:
            return [
                "Inverse Design unavailable: "
                + (self.load_error or "no active Model Book")
            ]
        variables = [
            name
            for name, widgets in self.input_widgets.items()
            if widgets.mode.get() == "Variable"
        ]
        fixed = [name for name in self.input_widgets if name not in variables]
        constraint_scopes = [
            widgets.scope.get() for widgets in self.constraint_widgets
        ]
        return [
            f"Active inverse-design Model Book: {self.active_book.name} ({self.active_book.book_id})",
            "Optimizer: deterministic Differential Evolution",
            f"Variable inputs: {', '.join(variables) or 'none'}",
            f"Fixed inputs: {', '.join(fixed) or 'none'}",
            f"Objective scope: {self.objective_scope.get()}",
            f"Objective goal: {self.objective_goal.get()}",
            f"Output constraints: {len(self.constraint_widgets)}",
            (
                "Constraint scopes: " + ", ".join(constraint_scopes)
                if constraint_scopes
                else "Constraint scopes: none"
            ),
            f"Configuration section: {self.config_section_control.get()}",
            f"Plot action for next result: {self.plot_action.get()}",
            f"Inverse-design plot curves: {len(self.response_plot.state.curves)}",
            (
                "Adjustable panel widths: Search Configuration "
                f"{self.configuration_card.winfo_width()} px; Curves manager "
                f"{self.response_plot.curve_manager.winfo_width()} px"
            ),
            (
                f"Latest inverse-design result: {self.last_result.status}"
                if self.last_result is not None
                else "Latest inverse-design result: not run"
            ),
            (
                f"Latest inverse-design outcome: {self.result_badge.cget('text')}"
                if self.last_result is not None
                else "Latest inverse-design outcome: not run"
            ),
            (
                f"Latest target gap: {self.last_result.target_gap:.7g}"
                if (
                    self.last_result is not None
                    and self.last_result.objective.get("goal") == "target"
                    and self.last_result.target_gap is not None
                )
                else "Latest target gap: not applicable"
            ),
            "Successful searches use separate project-local run folders",
        ]

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, padx=24, pady=(12, 8), sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="Inverse Design",
            text_color=COLORS["ink"],
            font=FONTS["display"],
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        self.model_header = ctk.CTkLabel(
            header,
            text="Select an active Model Book to begin",
            text_color=COLORS["muted"],
            font=FONTS["body_small"],
            anchor="e",
        )
        self.model_header.grid(row=0, column=1, sticky="e")

    def _build_workspace(self) -> None:
        workspace = ctk.CTkFrame(self, fg_color="transparent")
        workspace.grid(row=1, column=0, padx=24, pady=(0, 8), sticky="nsew")
        workspace.grid_columnconfigure(0, weight=1)
        workspace.grid_rowconfigure(0, weight=1)
        self.workspace = workspace
        split = tk.PanedWindow(
            workspace,
            orient=tk.HORIZONTAL,
            bd=0,
            relief=tk.FLAT,
            bg=self._palette(COLORS["border"]),
            sashwidth=10,
            sashpad=1,
            showhandle=True,
            handlesize=8,
            handlepad=1,
            opaqueresize=True,
            cursor="sb_h_double_arrow",
        )
        split.grid(row=0, column=0, sticky="nsew")
        self.workspace_split = split
        split.bind("<Configure>", self._workspace_resized, add="+")
        self._build_configuration(split)
        self._build_result(split)
        split.add(
            self.configuration_card,
            minsize=CONFIGURATION_MIN_WIDTH,
            width=CONFIGURATION_DEFAULT_WIDTH,
            stretch="never",
        )
        split.add(self.result_card, minsize=RESULT_MIN_WIDTH, stretch="always")

    @staticmethod
    def _palette(color: tuple[str, str]) -> str:
        return color[1] if ctk.get_appearance_mode() == "Dark" else color[0]

    def _card(self, parent: ctk.CTkFrame) -> ctk.CTkFrame:
        return ctk.CTkFrame(
            parent,
            fg_color=COLORS["surface"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["border"],
        )

    def _build_configuration(self, parent: ctk.CTkFrame) -> None:
        card = self._card(parent)
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(3, weight=1)
        self.configuration_card = card
        card.bind("<Configure>", self._configuration_resized, add="+")
        ctk.CTkLabel(
            card,
            text="SEARCH CONFIGURATION",
            text_color=COLORS["cyan"],
            font=FONTS["mono"],
            anchor="w",
        ).grid(row=0, column=0, padx=14, pady=(12, 3), sticky="ew")
        self.configuration_intro = ctk.CTkLabel(
            card,
            text="Define inputs, one scalar objective, and optional output limits.",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            anchor="w",
            wraplength=350,
        )
        self.configuration_intro.grid(
            row=1, column=0, padx=14, pady=(0, 7), sticky="ew"
        )
        self.config_section_control = ctk.CTkSegmentedButton(
            card,
            values=["Inputs", "Objective", "Constraints"],
            height=34,
            corner_radius=9,
            selected_color=COLORS["primary"],
            selected_hover_color=COLORS["primary_hover"],
            unselected_color=COLORS["surface_alt"],
            unselected_hover_color=COLORS["control_hover"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            command=self._show_config_section,
        )
        self.config_section_control.grid(row=2, column=0, padx=12, pady=(0, 7), sticky="ew")
        self.config_section_control.set("Inputs")

        self.section_host = ctk.CTkFrame(card, fg_color="transparent")
        self.section_host.grid(row=3, column=0, padx=12, sticky="nsew")
        self.section_host.grid_columnconfigure(0, weight=1)
        self.section_host.grid_rowconfigure(0, weight=1)
        self._build_inputs_section()
        self._build_objective_section()
        self._build_constraints_section()

        action_shell = ctk.CTkFrame(card, fg_color=COLORS["surface_alt"], corner_radius=9)
        action_shell.grid(row=4, column=0, padx=12, pady=(7, 11), sticky="ew")
        action_shell.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            action_shell,
            text="NEXT RESULT",
            text_color=COLORS["muted"],
            font=FONTS["mono"],
        ).grid(row=0, column=0, padx=8, pady=(6, 1), sticky="w")
        self.plot_action = ctk.CTkSegmentedButton(
            action_shell,
            values=["Replace selected curve", "Add to plot"],
            height=32,
            corner_radius=8,
            selected_color=COLORS["primary"],
            selected_hover_color=COLORS["primary_hover"],
            unselected_color=COLORS["control"],
            unselected_hover_color=COLORS["control_hover"],
            text_color=COLORS["ink"],
            font=FONTS["caption"],
        )
        self.plot_action.grid(row=1, column=0, padx=7, pady=(1, 7), sticky="ew")
        self.plot_action.set("Add to plot")

    def _build_inputs_section(self) -> None:
        section = ctk.CTkFrame(self.section_host, fg_color="transparent")
        section.grid(row=0, column=0, sticky="nsew")
        section.grid_columnconfigure(0, weight=1)
        section.grid_rowconfigure(2, weight=1)
        self.config_sections = {"Inputs": section}
        self.input_explanation = ctk.CTkLabel(
            section,
            text="Variable uses bounds · Fixed uses one retained value",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            anchor="w",
        )
        self.input_explanation.grid(
            row=0, column=0, padx=2, pady=(2, 5), sticky="ew"
        )
        headings = ctk.CTkFrame(section, fg_color=COLORS["surface_alt"], corner_radius=8)
        headings.grid(row=1, column=0, pady=(0, 4), sticky="ew")
        for column, (label, weight) in enumerate(
            (("INPUT", 2), ("ROLE", 2), ("LOW", 1), ("HIGH", 1), ("FIXED", 1))
        ):
            headings.grid_columnconfigure(
                column,
                weight=weight,
                minsize=INPUT_COLUMN_MIN_WIDTHS[column],
            )
            ctk.CTkLabel(
                headings,
                text=label,
                text_color=COLORS["muted"],
                font=FONTS["mono"],
            ).grid(row=0, column=column, padx=4, pady=5, sticky="w")
        self.input_rows_host = ctk.CTkFrame(section, fg_color="transparent")
        self.input_rows_host.grid(row=2, column=0, sticky="nsew")
        self.input_rows_host.grid_columnconfigure(0, weight=1)
        self.input_pager = ctk.CTkFrame(section, fg_color="transparent")
        self.input_pager.grid(row=3, column=0, pady=(4, 0), sticky="ew")
        self.input_pager.grid_columnconfigure(1, weight=1)
        self.input_previous = self._small_button(
            self.input_pager, "‹", lambda: self._change_input_page(-1)
        )
        self.input_previous.grid(row=0, column=0)
        self.input_page_label = ctk.CTkLabel(
            self.input_pager,
            text="Inputs 1–1 of 1",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
        )
        self.input_page_label.grid(row=0, column=1)
        self.input_next = self._small_button(
            self.input_pager, "›", lambda: self._change_input_page(1)
        )
        self.input_next.grid(row=0, column=2)

    def _build_objective_section(self) -> None:
        section = ctk.CTkFrame(self.section_host, fg_color="transparent")
        section.grid(row=0, column=0, sticky="nsew")
        section.grid_columnconfigure(0, weight=1)
        self.config_sections["Objective"] = section
        self.objective_scope_heading = ctk.CTkLabel(
            section,
            text="OUTPUT SCOPE",
            text_color=COLORS["muted"],
            font=FONTS["mono"],
            anchor="w",
        )
        self.objective_scope_heading.grid(
            row=0, column=0, pady=(3, 3), sticky="ew"
        )
        self.objective_scope = ctk.CTkSegmentedButton(
            section,
            values=["Single point", "Mean over range"],
            height=34,
            corner_radius=8,
            selected_color=COLORS["primary"],
            selected_hover_color=COLORS["primary_hover"],
            unselected_color=COLORS["control"],
            unselected_hover_color=COLORS["control_hover"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            command=self._objective_scope_changed,
        )
        self.objective_scope.grid(row=1, column=0, sticky="ew")
        self.objective_scope.set("Single point")
        self.axis_help = ctk.CTkLabel(
            section,
            text="Saved output coordinates are shown after a Model Book is loaded.",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            anchor="w",
            justify="left",
            wraplength=350,
        )
        self.axis_help.grid(row=2, column=0, pady=(6, 8), sticky="ew")
        self.single_coordinate_shell = self._field_shell(section, "OUTPUT COORDINATE")
        self.single_coordinate_shell.grid(row=3, column=0, sticky="ew")
        self.single_coordinate = self._numeric_entry(
            self.single_coordinate_shell,
            "Exact saved coordinate",
        )
        self.single_coordinate.grid(row=1, column=0, sticky="ew")
        self.range_coordinate_shell = ctk.CTkFrame(section, fg_color="transparent")
        self.range_coordinate_shell.grid(row=3, column=0, sticky="ew")
        self.range_coordinate_shell.grid_columnconfigure((0, 1), weight=1)
        start_shell = self._field_shell(self.range_coordinate_shell, "RANGE START")
        start_shell.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self.range_start = self._numeric_entry(start_shell, "Inclusive start")
        self.range_start.grid(row=1, column=0, sticky="ew")
        end_shell = self._field_shell(self.range_coordinate_shell, "RANGE END")
        end_shell.grid(row=0, column=1, padx=(4, 0), sticky="ew")
        self.range_end = self._numeric_entry(end_shell, "Inclusive end")
        self.range_end.grid(row=1, column=0, sticky="ew")
        self.range_coordinate_shell.grid_remove()

        ctk.CTkLabel(
            section,
            text="GOAL",
            text_color=COLORS["muted"],
            font=FONTS["mono"],
            anchor="w",
        ).grid(row=4, column=0, pady=(12, 3), sticky="ew")
        self.objective_goal = ctk.CTkSegmentedButton(
            section,
            values=list(GOAL_LABELS),
            height=34,
            corner_radius=8,
            selected_color=COLORS["primary"],
            selected_hover_color=COLORS["primary_hover"],
            unselected_color=COLORS["control"],
            unselected_hover_color=COLORS["control_hover"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            command=self._goal_changed,
        )
        self.objective_goal.grid(row=5, column=0, sticky="ew")
        self.objective_goal.set("Minimize")
        self.target_shell = self._field_shell(section, "TARGET VALUE")
        self.target_shell.grid(row=6, column=0, pady=(10, 0), sticky="ew")
        self.target_value = self._numeric_entry(self.target_shell, "Desired scalar value")
        self.target_value.grid(row=1, column=0, sticky="ew")
        self.target_shell.grid_remove()
        self.objective_explanation = ctk.CTkLabel(
            section,
            text=(
                "Objective = the one predicted score Differential Evolution improves. "
                "Minimize finds the lowest value, Maximize the highest, and Target "
                "the value closest to your requested number. Constraints are separate "
                "pass/fail limits and do not define this goal."
            ),
            text_color=COLORS["muted"],
            font=FONTS["body_small"],
            anchor="nw",
            justify="left",
            wraplength=350,
        )
        self.objective_explanation.grid(row=7, column=0, pady=(12, 0), sticky="ew")

    def _build_constraints_section(self) -> None:
        section = ctk.CTkFrame(self.section_host, fg_color="transparent")
        section.grid(row=0, column=0, sticky="nsew")
        section.grid_columnconfigure(0, weight=1)
        section.grid_rowconfigure(2, weight=1)
        self.config_sections["Constraints"] = section
        header = ctk.CTkFrame(section, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        self.constraint_intro = ctk.CTkLabel(
            header,
            text=(
                "Optional pass/fail limits; they filter allowed designs and do not "
                "replace the objective"
            ),
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            anchor="w",
        )
        self.constraint_intro.grid(row=0, column=0, sticky="w")
        self.add_constraint_button = ctk.CTkButton(
            header,
            text="+ Add",
            width=74,
            height=30,
            corner_radius=8,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            command=self._add_constraint,
        )
        self.add_constraint_button.grid(row=0, column=1, sticky="e")
        self.constraint_empty = ctk.CTkLabel(
            section,
            text=(
                "No constraints: every predicted design inside the input bounds is "
                "eligible. Add one only for a required output limit."
            ),
            text_color=COLORS["muted"],
            font=FONTS["body_small"],
            anchor="nw",
            justify="left",
            wraplength=345,
        )
        self.constraint_empty.grid(row=1, column=0, pady=(12, 0), sticky="ew")
        self.constraint_rows_host = ctk.CTkFrame(section, fg_color="transparent")
        self.constraint_rows_host.grid(row=2, column=0, pady=(5, 0), sticky="nsew")
        self.constraint_rows_host.grid_columnconfigure(0, weight=1)

    def _build_result(self, parent: ctk.CTkFrame) -> None:
        card = self._card(parent)
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(4, weight=1)
        self.result_card = card
        card.bind("<Configure>", self._result_resized, add="+")
        header = ctk.CTkFrame(card, fg_color="transparent")
        header.grid(row=0, column=0, padx=14, pady=(10, 4), sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        self.result_title = ctk.CTkLabel(
            header,
            text="Inverse-search plot",
            text_color=COLORS["ink"],
            font=FONTS["section"],
            anchor="w",
        )
        self.result_title.grid(row=0, column=0, sticky="w")
        self.result_badge = ctk.CTkLabel(
            header,
            text="READY",
            height=26,
            corner_radius=13,
            fg_color=COLORS["surface_alt"],
            text_color=COLORS["muted"],
            font=FONTS["mono"],
        )
        self.result_badge.grid(row=0, column=1, sticky="e")
        self.result_summary = ctk.CTkLabel(
            card,
            text="Configure and run a search. Completed designs can be added as curves.",
            text_color=COLORS["muted"],
            font=FONTS["body_small"],
            anchor="w",
        )
        self.result_summary.grid(row=1, column=0, padx=14, sticky="ew")
        metrics = ctk.CTkFrame(card, fg_color="transparent")
        metrics.grid(row=2, column=0, padx=11, pady=(5, 3), sticky="ew")
        metrics.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.objective_metric_title, self.objective_metric = self._metric(
            metrics, 0, "OBJECTIVE"
        )
        self.feasible_metric_title, self.feasible_metric = self._metric(
            metrics, 1, "CONSTRAINTS"
        )
        self.evaluation_metric_title, self.evaluation_metric = self._metric(
            metrics, 2, "EVALUATIONS"
        )
        self.iteration_metric_title, self.iteration_metric = self._metric(
            metrics, 3, "ITERATIONS"
        )
        self.latest_inputs = ctk.CTkLabel(
            card,
            text="Best inputs will appear here; each curve also retains its own inputs.",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            anchor="w",
            justify="left",
            wraplength=760,
        )
        self.latest_inputs.grid(row=3, column=0, padx=14, pady=(2, 4), sticky="ew")
        self.response_plot = ScientificPlotWorkbench(card)
        self.response_plot.grid(row=4, column=0, padx=12, pady=(2, 12), sticky="nsew")

    def _build_footer(self) -> None:
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, padx=24, pady=(0, 10), sticky="ew")
        footer.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(
            footer,
            text="←  Back to Inference",
            width=166,
            height=38,
            corner_radius=11,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            command=lambda: self.app.show_page("inference"),
        ).grid(row=0, column=0, sticky="w")
        self.footer_status = ctk.CTkLabel(
            footer,
            text="Differential Evolution · single objective · fixed seed 42",
            text_color=COLORS["subtle"],
            font=FONTS["caption"],
        )
        self.footer_status.grid(row=0, column=1, padx=12, sticky="e")
        self.run_button = ctk.CTkButton(
            footer,
            text="Run Inverse Design",
            width=184,
            height=40,
            corner_radius=11,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            text_color=COLORS["on_primary"],
            font=FONTS["button"],
            state="disabled",
            command=self._run_optimization,
        )
        self.run_button.grid(row=0, column=2, sticky="e")

    def _field_shell(self, parent: ctk.CTkFrame, label: str) -> ctk.CTkFrame:
        shell = ctk.CTkFrame(parent, fg_color="transparent")
        shell.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            shell,
            text=label,
            text_color=COLORS["muted"],
            font=FONTS["mono"],
            anchor="w",
        ).grid(row=0, column=0, pady=(0, 3), sticky="ew")
        return shell

    def _configuration_resized(self, event: tk.Event) -> None:
        """Resize wrapped content with the configuration sash."""

        self._clamp_workspace_sash()
        available = max(220, int(event.width) - 28)
        for label in (
            self.configuration_intro,
            self.input_explanation,
            self.axis_help,
            self.objective_explanation,
            self.constraint_intro,
            self.constraint_empty,
        ):
            label.configure(wraplength=available)

    def _result_resized(self, event: tk.Event) -> None:
        """Keep result summaries inside the current result-pane width."""

        self._clamp_workspace_sash()
        available = max(240, int(event.width) - 28)
        self.result_summary.configure(wraplength=available)
        self.latest_inputs.configure(wraplength=available)

    def _workspace_resized(self, _event: tk.Event) -> None:
        """Keep both adjustable panes above their readable minimum widths."""

        self._clamp_workspace_sash()

    def _clamp_workspace_sash(self) -> None:
        if self._clamping_workspace_sash or len(self.workspace_split.panes()) < 2:
            return
        total_width = int(self.workspace_split.winfo_width())
        if total_width <= 1:
            return
        sash_width = int(float(self.workspace_split.cget("sashwidth")))
        maximum_configuration = total_width - RESULT_MIN_WIDTH - sash_width
        if maximum_configuration < CONFIGURATION_MIN_WIDTH:
            target = CONFIGURATION_MIN_WIDTH
        else:
            current = int(self.workspace_split.sash_coord(0)[0])
            target = max(
                CONFIGURATION_MIN_WIDTH,
                min(current, maximum_configuration),
            )
        current = int(self.workspace_split.sash_coord(0)[0])
        if abs(current - target) <= 1:
            return
        self._clamping_workspace_sash = True
        try:
            self.workspace_split.sash_place(0, target, 1)
        finally:
            self._clamping_workspace_sash = False

    def _small_button(self, parent: ctk.CTkFrame, text: str, command: Any) -> ctk.CTkButton:
        return ctk.CTkButton(
            parent,
            text=text,
            width=32,
            height=28,
            corner_radius=8,
            fg_color=COLORS["control"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            command=command,
        )

    def _numeric_entry(self, parent: ctk.CTkFrame, placeholder: str) -> ctk.CTkEntry:
        return ctk.CTkEntry(
            parent,
            height=34,
            corner_radius=8,
            placeholder_text=placeholder,
            fg_color=COLORS["control"],
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=FONTS["body_small"],
        )

    def _metric(
        self,
        parent: ctk.CTkFrame,
        column: int,
        title: str,
    ) -> tuple[ctk.CTkLabel, ctk.CTkLabel]:
        card = ctk.CTkFrame(parent, fg_color=COLORS["surface_alt"], corner_radius=9)
        card.grid(row=0, column=column, padx=3, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        heading = ctk.CTkLabel(
            card,
            text=title,
            text_color=COLORS["muted"],
            font=FONTS["mono"],
        )
        heading.grid(row=0, column=0, padx=5, pady=(4, 0))
        value = ctk.CTkLabel(
            card,
            text="—",
            text_color=COLORS["ink"],
            font=FONTS["card_title"],
        )
        value.grid(row=1, column=0, padx=5, pady=(0, 4))
        return heading, value

    def _refresh(self) -> None:
        self._clear_form()
        self.response_plot.clear()
        self._clear_result()
        self._refresh_model_header()
        if self.active_book is None:
            message = self.load_error or "Open a project to configure inverse design."
            self.footer_status.configure(text=message, text_color=COLORS["danger"])
            self.run_button.configure(state="disabled")
            return
        self._create_input_rows(self.active_book.feature_columns)
        self._configure_output_axis()
        self.run_button.configure(state="normal")
        self.footer_status.configure(
            text="Differential Evolution · single objective · fixed seed 42",
            text_color=COLORS["subtle"],
        )

    def _refresh_model_header(self) -> None:
        if self.active_book is None:
            self.model_header.configure(
                text=self.load_error or "Select an active Model Book to begin"
            )
            return
        self.model_header.configure(
            text=(
                f"{self.active_book.name} · "
                f"{MODEL_LABELS.get(self.active_book.model_name, self.active_book.model_name)} · "
                f"{len(self.active_book.feature_columns)} inputs → "
                f"{len(self.active_book.target_columns)} outputs"
            )
        )

    def _clear_form(self) -> None:
        for widgets in self.input_widgets.values():
            widgets.frame.destroy()
        self.input_widgets.clear()
        for widgets in self.constraint_widgets:
            widgets.frame.destroy()
        self.constraint_widgets.clear()
        self.input_pager.grid_remove()
        self.constraint_empty.grid()
        self.add_constraint_button.configure(state="normal")
        for entry in (
            self.single_coordinate,
            self.range_start,
            self.range_end,
            self.target_value,
        ):
            entry.configure(state="normal")
            entry.delete(0, "end")
        self.objective_scope.set("Single point")
        self.objective_goal.set("Minimize")
        self._objective_scope_changed("Single point")
        self._goal_changed("Minimize")
        self.config_section_control.set("Inputs")
        self._show_config_section("Inputs")

    def _configure_output_axis(self) -> None:
        if self.active_book is None or self.active_book.output_axis is None:
            return
        axis = self.active_book.output_axis
        values = list(axis.values)
        if not values:
            return
        label = "Output index" if axis.source == "output_index" else axis.display_label
        self.axis_help.configure(
            text=(
                f"{label}: {min(values):.8g} to {max(values):.8g} · "
                f"{len(values)} saved points · enter coordinates directly"
            )
        )
        for entry, value in (
            (self.single_coordinate, values[0]),
            (self.range_start, min(values)),
            (self.range_end, max(values)),
        ):
            entry.delete(0, "end")
            entry.insert(0, f"{value:.12g}")

    def _create_input_rows(self, features: list[str]) -> None:
        for index, name in enumerate(features):
            frame = ctk.CTkFrame(
                self.input_rows_host,
                fg_color="transparent" if index % 2 == 0 else COLORS["surface_alt"],
                corner_radius=8,
            )
            for column, weight in enumerate((2, 2, 1, 1, 1)):
                frame.grid_columnconfigure(
                    column,
                    weight=weight,
                    minsize=INPUT_COLUMN_MIN_WIDTHS[column],
                )
            ctk.CTkLabel(
                frame,
                text=name,
                text_color=COLORS["ink"],
                font=FONTS["body_small"],
                anchor="w",
                justify="left",
                wraplength=95,
            ).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
            mode = ctk.StringVar(value="Variable" if index == 0 else "Fixed")
            control = ctk.CTkSegmentedButton(
                frame,
                values=["Variable", "Fixed"],
                height=30,
                corner_radius=7,
                selected_color=COLORS["primary"],
                selected_hover_color=COLORS["primary_hover"],
                unselected_color=COLORS["control"],
                unselected_hover_color=COLORS["control_hover"],
                text_color=COLORS["ink"],
                font=FONTS["caption"],
                width=INPUT_COLUMN_MIN_WIDTHS[1],
                variable=mode,
                command=lambda _value, feature=name: self._input_mode_changed(feature),
            )
            control.grid(row=0, column=1, padx=3, pady=3, sticky="ew")
            lower = self._numeric_entry(frame, "Min")
            upper = self._numeric_entry(frame, "Max")
            fixed = self._numeric_entry(frame, "Value")
            for entry in (lower, upper, fixed):
                entry.configure(width=INPUT_COLUMN_MIN_WIDTHS[2])
            lower.grid(row=0, column=2, padx=2, pady=3, sticky="ew")
            upper.grid(row=0, column=3, padx=2, pady=3, sticky="ew")
            fixed.grid(row=0, column=4, padx=2, pady=3, sticky="ew")
            self.input_widgets[name] = InputWidgets(
                frame, mode, control, lower, upper, fixed
            )
            self._input_mode_changed(name)
        self._render_input_page()

    def _input_mode_changed(self, name: str) -> None:
        widgets = self.input_widgets[name]
        variable = widgets.mode.get() == "Variable"
        widgets.lower.configure(state="normal" if variable else "disabled")
        widgets.upper.configure(state="normal" if variable else "disabled")
        widgets.fixed.configure(state="disabled" if variable else "normal")

    def _render_input_page(self) -> None:
        names = list(self.input_widgets)
        page_count = max(1, (len(names) + INPUTS_PER_PAGE - 1) // INPUTS_PER_PAGE)
        self.input_page = min(self.input_page, page_count - 1)
        for widgets in self.input_widgets.values():
            widgets.frame.grid_remove()
        start = self.input_page * INPUTS_PER_PAGE
        visible = names[start : start + INPUTS_PER_PAGE]
        for row, name in enumerate(visible):
            self.input_widgets[name].frame.grid(row=row, column=0, pady=2, sticky="ew")
        if len(names) > INPUTS_PER_PAGE:
            self.input_page_label.configure(
                text=f"Inputs {start + 1}–{start + len(visible)} of {len(names)}"
            )
            self.input_previous.configure(
                state="normal" if self.input_page > 0 else "disabled"
            )
            self.input_next.configure(
                state="normal" if self.input_page + 1 < page_count else "disabled"
            )
            self.input_pager.grid()
        else:
            self.input_pager.grid_remove()

    def _change_input_page(self, offset: int) -> None:
        pages = max(1, (len(self.input_widgets) + INPUTS_PER_PAGE - 1) // INPUTS_PER_PAGE)
        self.input_page = max(0, min(self.input_page + offset, pages - 1))
        self._render_input_page()

    def _show_config_section(self, name: str) -> None:
        for section_name, section in self.config_sections.items():
            if section_name == name:
                section.tkraise()
                section.grid()
            else:
                section.grid_remove()

    def _objective_scope_changed(self, value: str) -> None:
        if value == "Mean over range":
            self.single_coordinate_shell.grid_remove()
            self.range_coordinate_shell.grid()
        else:
            self.range_coordinate_shell.grid_remove()
            self.single_coordinate_shell.grid()

    def _goal_changed(self, value: str) -> None:
        if value == "Target value":
            self.target_shell.grid()
        else:
            self.target_shell.grid_remove()

    def _add_constraint(self) -> None:
        if self.active_book is None or len(self.constraint_widgets) >= MAX_CONSTRAINTS:
            return
        index = len(self.constraint_widgets)
        frame = ctk.CTkFrame(
            self.constraint_rows_host,
            fg_color=COLORS["surface_alt"],
            corner_radius=9,
        )
        frame.grid(row=index, column=0, pady=3, sticky="ew")
        frame.grid_columnconfigure(0, weight=5)
        frame.grid_columnconfigure((1, 2), weight=4)
        scope = ctk.CTkOptionMenu(
            frame,
            values=["Single point", "Mean over range"],
            height=32,
            corner_radius=8,
            fg_color=COLORS["primary"],
            button_color=COLORS["primary_hover"],
            button_hover_color=COLORS["primary_hover"],
            dropdown_fg_color=COLORS["surface"],
            dropdown_hover_color=COLORS["nav_active"],
            text_color=COLORS["on_primary"],
            dropdown_text_color=COLORS["ink"],
            font=FONTS["caption"],
            dropdown_font=FONTS["body_small"],
        )
        scope.grid(row=0, column=0, padx=(6, 3), pady=(6, 3), sticky="ew")
        coordinate_start = self._numeric_entry(frame, "Coordinate")
        coordinate_end = self._numeric_entry(frame, "Range end")
        coordinate_start.grid(row=0, column=1, padx=3, pady=(6, 3), sticky="ew")
        coordinate_end.grid(row=0, column=2, padx=3, pady=(6, 3), sticky="ew")
        axis_values = list(self.active_book.output_axis.values)
        coordinate_start.insert(0, f"{axis_values[0]:.12g}")
        coordinate_end.insert(0, f"{axis_values[-1]:.12g}")
        operator = ctk.CTkOptionMenu(
            frame,
            values=list(CONSTRAINT_LABELS),
            height=32,
            corner_radius=8,
            fg_color=COLORS["primary"],
            button_color=COLORS["primary_hover"],
            button_hover_color=COLORS["primary_hover"],
            dropdown_fg_color=COLORS["surface"],
            dropdown_hover_color=COLORS["nav_active"],
            text_color=COLORS["on_primary"],
            dropdown_text_color=COLORS["ink"],
            font=FONTS["caption"],
            dropdown_font=FONTS["body_small"],
        )
        operator.grid(row=1, column=0, padx=(6, 3), pady=(3, 6), sticky="ew")
        first = self._numeric_entry(frame, "Value")
        second = self._numeric_entry(frame, "Upper")
        first.grid(row=1, column=1, padx=3, pady=(3, 6), sticky="ew")
        second.grid(row=1, column=2, padx=3, pady=(3, 6), sticky="ew")
        remove = ctk.CTkButton(
            frame,
            text="×",
            width=30,
            height=30,
            corner_radius=8,
            fg_color="transparent",
            hover_color=COLORS["control_hover"],
            text_color=COLORS["danger"],
            font=FONTS["section"],
            command=lambda row=index: self._remove_constraint(row),
        )
        remove.grid(row=0, column=3, rowspan=2, padx=(3, 6))
        self.constraint_widgets.append(
            ConstraintWidgets(
                frame,
                scope,
                coordinate_start,
                coordinate_end,
                operator,
                first,
                second,
                remove,
            )
        )
        self.constraint_empty.grid_remove()
        self._rebind_constraint_commands()
        self._constraint_scope_changed(index)
        self._constraint_operator_changed(index)
        if len(self.constraint_widgets) >= MAX_CONSTRAINTS:
            self.add_constraint_button.configure(state="disabled")

    def _remove_constraint(self, index: int) -> None:
        if not 0 <= index < len(self.constraint_widgets):
            return
        self.constraint_widgets.pop(index).frame.destroy()
        for row, widgets in enumerate(self.constraint_widgets):
            widgets.frame.grid_configure(row=row)
        self._rebind_constraint_commands()
        self.add_constraint_button.configure(state="normal")
        if not self.constraint_widgets:
            self.constraint_empty.grid()

    def _rebind_constraint_commands(self) -> None:
        for index, widgets in enumerate(self.constraint_widgets):
            widgets.scope.configure(
                command=lambda _value, row=index: self._constraint_scope_changed(row)
            )
            widgets.operator.configure(
                command=lambda _value, row=index: self._constraint_operator_changed(row)
            )
            widgets.remove_button.configure(
                command=lambda row=index: self._remove_constraint(row)
            )

    def _constraint_scope_changed(self, index: int) -> None:
        if not 0 <= index < len(self.constraint_widgets):
            return
        widgets = self.constraint_widgets[index]
        mean = widgets.scope.get() == "Mean over range"
        widgets.coordinate_start.configure(
            placeholder_text="Range start" if mean else "Coordinate"
        )
        widgets.coordinate_end.configure(state="normal" if mean else "disabled")

    def _constraint_operator_changed(self, index: int) -> None:
        if not 0 <= index < len(self.constraint_widgets):
            return
        widgets = self.constraint_widgets[index]
        within = widgets.operator.get() == "Within range"
        widgets.first_value.configure(
            placeholder_text="Lower limit" if within else "Threshold"
        )
        widgets.second_value.configure(
            placeholder_text="Upper limit",
            state="normal" if within else "disabled",
        )

    def _float_entry(self, entry: ctk.CTkEntry, label: str) -> float:
        raw = entry.get().strip()
        if not raw:
            raise ValueError(f"Enter {label}.")
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"{label.capitalize()} must be numeric.") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label.capitalize()} must be finite.")
        return value

    def _axis_pairs(self) -> list[tuple[float, str]]:
        if self.active_book is None or self.active_book.output_axis is None:
            raise ValueError("The active Model Book does not have a usable output axis.")
        return list(
            zip(
                self.active_book.output_axis.values,
                self.active_book.target_columns,
                strict=True,
            )
        )

    def _objective_outputs(self) -> tuple[str, list[str]]:
        pairs = self._axis_pairs()
        if self.objective_scope.get() == "Single point":
            requested = self._float_entry(
                self.single_coordinate,
                "an exact saved output coordinate",
            )
            match = next(
                (
                    name
                    for coordinate, name in pairs
                    if math.isclose(coordinate, requested, rel_tol=1e-10, abs_tol=1e-12)
                ),
                None,
            )
            if match is None:
                raise ValueError(
                    f"No saved output exists at coordinate {requested:.12g}."
                )
            return "single", [match]
        start = self._float_entry(self.range_start, "an output-range start")
        end = self._float_entry(self.range_end, "an output-range end")
        if start > end:
            raise ValueError("Output-range start cannot exceed output-range end.")
        names = [name for coordinate, name in pairs if start <= coordinate <= end]
        if len(names) < 2:
            raise ValueError(
                "Mean over range requires at least two saved output coordinates."
            )
        return "mean", names

    def _constraint_outputs(
        self,
        widgets: ConstraintWidgets,
        index: int,
    ) -> tuple[str, list[str]]:
        pairs = self._axis_pairs()
        if widgets.scope.get() == "Single point":
            requested = self._float_entry(
                widgets.coordinate_start,
                f"constraint {index} output coordinate",
            )
            match = next(
                (
                    name
                    for coordinate, name in pairs
                    if math.isclose(
                        coordinate,
                        requested,
                        rel_tol=1e-10,
                        abs_tol=1e-12,
                    )
                ),
                None,
            )
            if match is None:
                raise ValueError(
                    f"Constraint {index} has no saved output at coordinate "
                    f"{requested:.12g}."
                )
            return "single", [match]
        start = self._float_entry(
            widgets.coordinate_start,
            f"constraint {index} range start",
        )
        end = self._float_entry(
            widgets.coordinate_end,
            f"constraint {index} range end",
        )
        if start > end:
            raise ValueError(
                f"Constraint {index} range start cannot exceed its range end."
            )
        names = [name for coordinate, name in pairs if start <= coordinate <= end]
        if len(names) < 2:
            raise ValueError(
                f"Constraint {index} mean requires at least two saved output coordinates."
            )
        return "mean", names

    def build_request(self) -> InverseDesignRequest:
        if self.active_book is None:
            raise ValueError(self.load_error or "Select an active Model Book first.")
        variable_bounds: dict[str, tuple[float, float]] = {}
        fixed_inputs: dict[str, float] = {}
        for name, widgets in self.input_widgets.items():
            if widgets.mode.get() == "Variable":
                variable_bounds[name] = (
                    self._float_entry(widgets.lower, f"a lower bound for {name}"),
                    self._float_entry(widgets.upper, f"an upper bound for {name}"),
                )
            else:
                fixed_inputs[name] = self._float_entry(
                    widgets.fixed, f"a fixed value for {name}"
                )
        aggregation, output_names = self._objective_outputs()
        goal = GOAL_LABELS[self.objective_goal.get()]
        target = (
            self._float_entry(self.target_value, "an objective target value")
            if goal == "target"
            else None
        )
        objective = InverseDesignObjective(
            output_names[0] if aggregation == "single" else None,
            goal,
            target,
            aggregation=aggregation,
            output_names=output_names,
        )
        constraints: list[OutputConstraint] = []
        for index, widgets in enumerate(self.constraint_widgets, start=1):
            constraint_aggregation, constraint_outputs = self._constraint_outputs(
                widgets,
                index,
            )
            operator = CONSTRAINT_LABELS[widgets.operator.get()]
            first = self._float_entry(
                widgets.first_value, f"constraint {index} value"
            )
            if operator == "within_range":
                constraints.append(
                    OutputConstraint(
                        (
                            constraint_outputs[0]
                            if constraint_aggregation == "single"
                            else None
                        ),
                        operator,
                        lower_bound=first,
                        upper_bound=self._float_entry(
                            widgets.second_value,
                            f"constraint {index} upper value",
                        ),
                        aggregation=constraint_aggregation,
                        output_names=constraint_outputs,
                    )
                )
            else:
                constraints.append(
                    OutputConstraint(
                        (
                            constraint_outputs[0]
                            if constraint_aggregation == "single"
                            else None
                        ),
                        operator,
                        value=first,
                        aggregation=constraint_aggregation,
                        output_names=constraint_outputs,
                    )
                )
        return InverseDesignRequest(
            model_book_id=self.active_book.book_id,
            variable_bounds=variable_bounds,
            fixed_inputs=fixed_inputs,
            objective=objective,
            constraints=constraints,
        )

    def _run_optimization(self) -> None:
        if self.optimization_in_progress or self.project is None:
            return
        try:
            request = self.build_request()
        except (KeyError, TypeError, ValueError) as exc:
            self.footer_status.configure(text=str(exc), text_color=COLORS["danger"])
            return
        self._set_busy(True)
        project_path = self.project.path

        def worker() -> None:
            result = submit_inverse_design_request(request, project_path=project_path)
            try:
                self.after(0, lambda: self._finish_optimization(result))
            except tk.TclError:
                pass

        threading.Thread(target=worker, daemon=True).start()

    def _set_busy(self, busy: bool) -> None:
        self.optimization_in_progress = busy
        self.run_button.configure(
            text="Optimizing…" if busy else "Run Inverse Design",
            state="disabled" if busy or self.active_book is None else "normal",
        )
        self.footer_status.configure(
            text=(
                "Differential Evolution is evaluating the local surrogate…"
                if busy
                else "Differential Evolution · single objective · fixed seed 42"
            ),
            text_color=COLORS["subtle"],
        )
        self.update_idletasks()

    def _finish_optimization(self, result: InverseDesignResult) -> None:
        self.last_result = result
        self._set_busy(False)
        if result.success:
            self._show_success(result)
            if self.project is not None:
                try:
                    self.app.current_project = self.app.store.open_project(
                        self.project.path, touch=False
                    )
                    self.project = self.app.current_project
                except Exception:
                    pass
            return
        failure_message = result.error_message or "No completed result is available."
        self.result_title.configure(text="Inverse design did not complete")
        self.result_summary.configure(
            text=failure_message,
            text_color=COLORS["danger"],
        )
        self.result_badge.configure(
            text=(
                "NO CONSTRAINT MATCH"
                if "output constraints" in failure_message.lower()
                else "FAILED"
            ),
            fg_color=COLORS["warning_soft"],
            text_color=COLORS["danger"],
        )
        for heading, text in (
            (self.objective_metric_title, "OBJECTIVE"),
            (self.feasible_metric_title, "CONSTRAINTS"),
            (self.evaluation_metric_title, "EVALUATIONS"),
            (self.iteration_metric_title, "ITERATIONS"),
        ):
            heading.configure(text=text)
        for metric in (
            self.objective_metric,
            self.feasible_metric,
            self.evaluation_metric,
            self.iteration_metric,
        ):
            metric.configure(text="—")
        self.latest_inputs.configure(
            text="No new design was added; previously plotted curves remain available."
        )
        self.footer_status.configure(
            text=failure_message,
            text_color=COLORS["danger"],
        )

    def _show_success(self, result: InverseDesignResult) -> None:
        self._show_result_values(
            run_id=result.run_id,
            objective=result.objective,
            objective_value=result.objective_value,
            target_gap=result.target_gap,
            best_inputs=result.best_inputs,
            predicted_outputs=result.predicted_outputs,
            constraint_evaluations=result.constraint_evaluations,
            evaluations=result.evaluations,
            iterations=result.iterations,
            replace_selected=self.plot_action.get() == "Replace selected curve",
        )

    @staticmethod
    def _mappings_effectively_match(
        first: dict[str, float],
        second: dict[str, float],
        *,
        relative_tolerance: float,
        absolute_tolerance: float,
    ) -> bool:
        if list(first) != list(second):
            return False
        if not first:
            return True
        scale = max(
            1.0,
            *(abs(float(value)) for value in first.values()),
            *(abs(float(value)) for value in second.values()),
        )
        threshold = absolute_tolerance + relative_tolerance * scale
        return all(
            abs(float(first[name]) - float(second[name])) <= threshold
            for name in first
        )

    def _matching_plotted_curve(
        self,
        predicted_outputs: dict[str, float],
        best_inputs: dict[str, float],
        *,
        replace_selected: bool,
    ) -> tuple[str, bool] | None:
        selected_id = (
            self.response_plot.state.selected_curve_id if replace_selected else None
        )
        for curve in self.response_plot.state.curves:
            if curve.curve_id == selected_id:
                continue
            existing_outputs = dict(zip(curve.target_names, curve.y_values, strict=True))
            if not self._mappings_effectively_match(
                existing_outputs,
                predicted_outputs,
                relative_tolerance=DUPLICATE_RESPONSE_REL_TOLERANCE,
                absolute_tolerance=DUPLICATE_RESPONSE_ABS_TOLERANCE,
            ):
                continue
            same_inputs = self._mappings_effectively_match(
                curve.inputs,
                best_inputs,
                relative_tolerance=DUPLICATE_INPUT_REL_TOLERANCE,
                absolute_tolerance=DUPLICATE_INPUT_ABS_TOLERANCE,
            )
            return curve.name, same_inputs
        return None

    def _show_result_values(
        self,
        *,
        run_id: str | None,
        objective: dict[str, Any],
        objective_value: float | None,
        target_gap: float | None,
        best_inputs: dict[str, float],
        predicted_outputs: dict[str, float],
        constraint_evaluations: list[dict[str, Any]],
        evaluations: int,
        iterations: int,
        replace_selected: bool,
    ) -> None:
        aggregation = str(objective.get("aggregation") or "single")
        output_names = list(objective.get("output_names") or [])
        objective_label = (
            str(objective.get("output_name") or "output")
            if aggregation == "single"
            else (
                f"Mean · {output_names[0]} to {output_names[-1]}"
                if output_names
                else "Mean output range"
            )
        )
        goal = str(objective.get("goal") or "objective")
        has_constraints = bool(constraint_evaluations)
        is_target = goal == "target"
        if is_target and target_gap is None:
            target_value = objective.get("target_value")
            if objective_value is not None and target_value is not None:
                target_gap = abs(objective_value - float(target_value))
        duplicate = self._matching_plotted_curve(
            predicted_outputs,
            best_inputs,
            replace_selected=replace_selected,
        )
        self.result_title.configure(
            text=(
                "Closest predicted design"
                if is_target
                else (
                    "Best constraint-satisfying design"
                    if has_constraints
                    else "Optimized design"
                )
            )
        )
        summary_parts = [
            run_id or "Saved run",
            f"{goal.replace('_', ' ').title()} {objective_label}",
        ]
        if is_target and objective.get("target_value") is not None:
            summary_parts.append(f"requested {float(objective['target_value']):.7g}")
        if duplicate is not None:
            matching_name, same_inputs = duplicate
            summary_parts.append(
                (
                    "same design and response as "
                    if same_inputs
                    else "response matches "
                )
                + matching_name
            )
        self.result_summary.configure(
            text=" · ".join(summary_parts),
            text_color=COLORS["muted"],
        )
        self.result_badge.configure(
            text=(
                "CONSTRAINTS MET"
                if has_constraints
                else ("CLOSEST FOUND" if is_target else "OPTIMIZED")
            ),
            fg_color=COLORS["success_soft"],
            text_color=COLORS["success"],
        )
        self.objective_metric_title.configure(
            text="ACHIEVED" if is_target else "OBJECTIVE"
        )
        self.objective_metric.configure(
            text="—" if objective_value is None else f"{objective_value:.7g}"
        )
        self.feasible_metric_title.configure(text="CONSTRAINTS")
        self.feasible_metric.configure(
            text=(
                f"{len(constraint_evaluations)}/{len(constraint_evaluations)} met"
                if has_constraints
                else "Not used"
            )
        )
        self.evaluation_metric_title.configure(text="EVALUATIONS")
        self.evaluation_metric.configure(text=str(evaluations))
        self.iteration_metric_title.configure(
            text="TARGET GAP" if is_target else "ITERATIONS"
        )
        self.iteration_metric.configure(
            text=("—" if target_gap is None else f"{target_gap:.7g}")
            if is_target
            else str(iterations)
        )
        input_text = " · ".join(
            f"{name} = {value:.7g}" for name, value in best_inputs.items()
        )
        constraint_text = (
            f" · {len(constraint_evaluations)}/{len(constraint_evaluations)} "
            "constraints met"
            if constraint_evaluations
            else " · constraints not used"
        )
        iteration_text = f" · {iterations} optimizer iterations" if is_target else ""
        self.latest_inputs.configure(
            text=f"Latest best inputs: {input_text}{constraint_text}{iteration_text}"
        )
        targets = tuple(predicted_outputs)
        values = tuple(float(predicted_outputs[name]) for name in targets)
        axis = self.active_book.output_axis if self.active_book is not None else None
        x_values = (
            axis.values
            if axis is not None and len(axis.values) == len(targets)
            else tuple(float(index) for index in range(1, len(targets) + 1))
        )
        curve_name = f"{run_id or 'Inverse design'} · {objective_label}"
        self.response_plot.add_curve(
            x_values=x_values,
            y_values=values,
            target_names=targets,
            inputs=best_inputs,
            replace_selected=replace_selected,
            x_label=axis.display_label if axis is not None else "Output coordinate",
            name=curve_name,
        )
        self.response_plot.state.plot_title = "Inverse Design Responses"
        self.response_plot.state.y_label = "Predicted value"
        self.response_plot.redraw()
        self.footer_status.configure(
            text=(
                f"Result matches plotted curve {duplicate[0]} within the "
                "0.01% response tolerance"
                if duplicate is not None
                else (
                    f"Inverse design completed · {run_id or 'saved run'} · "
                    f"{len(self.response_plot.state.curves)} plotted curve"
                    f"{'s' if len(self.response_plot.state.curves) != 1 else ''}"
                )
            ),
            text_color=(
                COLORS["warning"] if duplicate is not None else COLORS["success"]
            ),
        )

    def _clear_result(self) -> None:
        self.result_title.configure(text="Inverse-search plot")
        self.result_summary.configure(
            text="Configure and run a search. Completed designs can be added as curves.",
            text_color=COLORS["muted"],
        )
        self.result_badge.configure(
            text="READY",
            fg_color=COLORS["surface_alt"],
            text_color=COLORS["muted"],
        )
        for heading, text in (
            (self.objective_metric_title, "OBJECTIVE"),
            (self.feasible_metric_title, "CONSTRAINTS"),
            (self.evaluation_metric_title, "EVALUATIONS"),
            (self.iteration_metric_title, "ITERATIONS"),
        ):
            heading.configure(text=text)
        for label in (
            self.objective_metric,
            self.feasible_metric,
            self.evaluation_metric,
            self.iteration_metric,
        ):
            label.configure(text="—")
        self.latest_inputs.configure(
            text="Best inputs will appear here; each curve also retains its own inputs."
        )

    def _restore_latest_result(self) -> None:
        if self.project is None or self.active_book is None:
            return
        try:
            payload = load_inverse_design_run(self.project.path)
        except InverseDesignError as exc:
            self.footer_status.configure(text=str(exc), text_color=COLORS["danger"])
            return
        if (
            not payload
            or not payload.get("success")
            or payload.get("model_book_id") != self.active_book.book_id
        ):
            return
        try:
            result = self._restored_result(payload)
        except (TypeError, ValueError):
            self.footer_status.configure(
                text="The latest inverse-design result contains invalid saved values.",
                text_color=COLORS["danger"],
            )
            return
        self.last_result = result
        self._show_success(result)

    def _restored_result(self, payload: dict[str, Any]) -> InverseDesignResult:
        """Rehydrate the same canonical result object used by a live search."""

        artifact_directory: Path | None = None
        saved_artifact = str(payload.get("artifact_directory") or "").strip()
        if saved_artifact:
            artifact_directory = Path(saved_artifact)
            if not artifact_directory.is_absolute() and self.project is not None:
                artifact_directory = self.project.path / artifact_directory
            artifact_directory = artifact_directory.resolve()

        def optional_float(name: str) -> float | None:
            value = payload.get(name)
            return float(value) if value is not None else None

        objective = dict(payload.get("objective") or {})
        objective_value = optional_float("objective_value")
        target_gap = optional_float("target_gap")
        if (
            target_gap is None
            and objective.get("goal") == "target"
            and objective_value is not None
            and objective.get("target_value") is not None
        ):
            target_gap = abs(objective_value - float(objective["target_value"]))

        return InverseDesignResult(
            success=bool(payload.get("success")),
            status=str(payload.get("status") or INVERSE_DESIGN_COMPLETED),
            model_book_id=str(payload.get("model_book_id") or "") or None,
            model_book_name=str(payload.get("model_book_name") or "") or None,
            model_name=str(payload.get("model_name") or "") or None,
            run_id=str(payload.get("run_id") or "") or None,
            best_inputs={
                str(name): float(value)
                for name, value in dict(payload.get("best_inputs") or {}).items()
            },
            predicted_outputs={
                str(name): float(value)
                for name, value in dict(payload.get("predicted_outputs") or {}).items()
            },
            objective=objective,
            objective_value=objective_value,
            objective_score=optional_float("objective_score"),
            target_gap=target_gap,
            constraint_evaluations=list(payload.get("constraint_evaluations") or []),
            feasible=bool(payload.get("feasible")),
            evaluations=int(payload.get("evaluations") or 0),
            iterations=int(payload.get("iterations") or 0),
            optimizer_message=str(payload.get("optimizer_message") or "") or None,
            artifact_directory=artifact_directory,
            error_message=str(payload.get("error_message") or "") or None,
        )
