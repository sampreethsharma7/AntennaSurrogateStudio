"""Compact project-local Latin Hypercube sample-generator dialog."""

from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Callable

import customtkinter as ctk

from studio.sample_generator import (
    LHSSampleGenerationRequest,
    LHSSampleSet,
    LHSVariable,
    generate_lhs_samples,
    write_lhs_inputs_csv,
)
from studio.theme import COLORS, FONTS


VISIBLE_VARIABLE_ROWS = 5
MAX_UI_VARIABLES = 20
PREVIEW_ROWS = 5
PREVIEW_VARIABLES = 2


def _active_color(value: str | tuple[str, str]) -> str:
    if isinstance(value, str):
        return value
    return value[1] if ctk.get_appearance_mode().lower() == "dark" else value[0]


@dataclass(slots=True)
class LHSVariableEditor:
    name: ctk.StringVar
    minimum: ctk.StringVar
    maximum: ctk.StringVar


class LHSSampleGeneratorDialog(ctk.CTkToplevel):
    """Non-scrolling LHS editor, coverage preview, and CSV export workflow."""

    def __init__(
        self,
        parent: ctk.CTkFrame,
        *,
        project_path: Path,
        on_export: Callable[[Path], None],
    ) -> None:
        super().__init__(parent)
        self.project_path = Path(project_path)
        self.on_export = on_export
        self.variable_editors: list[LHSVariableEditor] = []
        self.variable_page = 0
        self.generated_samples: LHSSampleSet | None = None
        self.sample_count_var = ctk.StringVar(value="100")
        self.seed_var = ctk.StringVar(value="42")
        self.status_var = ctk.StringVar(
            value="Define numeric ranges, then generate a solver-ready input design."
        )

        self.title("LHS Sample Generator")
        self.configure(fg_color=COLORS["app_bg"])
        self.resizable(True, True)
        self.minsize(980, 620)
        self._place_for_screen()
        self.transient(parent.winfo_toplevel())
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_workspace()
        self._build_footer()
        self.sample_count_var.trace_add(
            "write",
            lambda *_args: self._invalidate_generated_samples(),
        )
        self.seed_var.trace_add(
            "write",
            lambda *_args: self._invalidate_generated_samples(),
        )
        for _index in range(3):
            self._append_editor()
        self._render_variable_page()
        self.after_idle(self._draw_empty_coverage)

    def _place_for_screen(self) -> None:
        screen_width = max(1024, int(self.winfo_screenwidth()))
        screen_height = max(700, int(self.winfo_screenheight()))
        width = min(1160, screen_width - 72)
        height = min(720, screen_height - 76)
        x = max(12, (screen_width - width) // 2)
        y = max(12, (screen_height - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, padx=24, pady=(18, 10), sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="Latin Hypercube Sample Generator",
            text_color=COLORS["ink"],
            font=FONTS["section"],
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header,
            text=(
                "Create well-distributed, generic input settings for CST, HFSS, "
                "or another simulator."
            ),
            text_color=COLORS["muted"],
            font=FONTS["body_small"],
            anchor="w",
        ).grid(row=1, column=0, pady=(3, 0), sticky="w")
        ctk.CTkButton(
            header,
            text="Close",
            width=88,
            height=34,
            corner_radius=10,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            command=self.close,
        ).grid(row=0, column=1, rowspan=2, padx=(12, 0), sticky="e")

    def _build_workspace(self) -> None:
        workspace = ctk.CTkFrame(self, fg_color="transparent")
        workspace.grid(row=1, column=0, padx=24, pady=(0, 8), sticky="nsew")
        workspace.grid_columnconfigure(0, weight=6, uniform="lhs")
        workspace.grid_columnconfigure(1, weight=5, uniform="lhs")
        workspace.grid_rowconfigure(0, weight=1)

        self._build_configuration_panel(workspace)
        self._build_preview_panel(workspace)

    def _build_configuration_panel(self, parent: ctk.CTkFrame) -> None:
        panel = ctk.CTkFrame(
            parent,
            fg_color=COLORS["surface"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["border"],
        )
        panel.grid(row=0, column=0, padx=(0, 6), sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)

        title_row = ctk.CTkFrame(panel, fg_color="transparent")
        title_row.grid(row=0, column=0, padx=16, pady=(12, 7), sticky="ew")
        title_row.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            title_row,
            text="Simulation variables",
            text_color=COLORS["ink"],
            font=FONTS["card_title"],
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        self.add_variable_button = ctk.CTkButton(
            title_row,
            text="+ Add variable",
            width=122,
            height=32,
            corner_radius=9,
            fg_color=COLORS["primary_soft"],
            hover_color=COLORS["control_hover"],
            text_color=COLORS["cyan"],
            font=FONTS["button"],
            command=self.add_variable,
        )
        self.add_variable_button.grid(row=0, column=1, sticky="e")

        headings = ctk.CTkFrame(panel, fg_color="transparent")
        headings.grid(row=1, column=0, padx=16, sticky="ew")
        headings.grid_columnconfigure(0, weight=1, minsize=180)
        headings.grid_columnconfigure(1, weight=0, minsize=88)
        headings.grid_columnconfigure(2, weight=0, minsize=88)
        for column, (label, anchor) in enumerate(
            (("Variable name", "w"), ("Min", "w"), ("Max", "w"))
        ):
            ctk.CTkLabel(
                headings,
                text=label,
                text_color=COLORS["muted"],
                font=FONTS["caption"],
                anchor=anchor,
            ).grid(row=0, column=column, padx=(0, 7), sticky="ew")

        self.variable_rows_frame = ctk.CTkFrame(panel, fg_color="transparent")
        self.variable_rows_frame.grid(
            row=2,
            column=0,
            padx=16,
            pady=(2, 4),
            sticky="ew",
        )
        self.variable_rows_frame.grid_columnconfigure(0, weight=1, minsize=180)
        self.variable_rows_frame.grid_columnconfigure(1, weight=0, minsize=88)
        self.variable_rows_frame.grid_columnconfigure(2, weight=0, minsize=88)

        pager = ctk.CTkFrame(panel, fg_color="transparent")
        pager.grid(row=3, column=0, padx=16, pady=(1, 9), sticky="ew")
        pager.grid_columnconfigure(1, weight=1)
        self.previous_page_button = ctk.CTkButton(
            pager,
            text="‹",
            width=36,
            height=30,
            corner_radius=8,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            command=lambda: self._change_page(-1),
        )
        self.previous_page_button.grid(row=0, column=0, sticky="w")
        self.variable_page_label = ctk.CTkLabel(
            pager,
            text="Variables 1–3 of 3",
            text_color=COLORS["subtle"],
            font=FONTS["caption"],
        )
        self.variable_page_label.grid(row=0, column=1)
        self.next_page_button = ctk.CTkButton(
            pager,
            text="›",
            width=36,
            height=30,
            corner_radius=8,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            command=lambda: self._change_page(1),
        )
        self.next_page_button.grid(row=0, column=2, sticky="e")

        settings = ctk.CTkFrame(
            panel,
            fg_color=COLORS["surface_alt"],
            corner_radius=11,
        )
        settings.grid(row=4, column=0, padx=16, pady=(0, 10), sticky="ew")
        settings.grid_columnconfigure(1, weight=1)
        settings.grid_columnconfigure(3, weight=1)
        ctk.CTkLabel(
            settings,
            text="Samples",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
        ).grid(row=0, column=0, padx=(12, 7), pady=10)
        self.sample_count_entry = ctk.CTkEntry(
            settings,
            textvariable=self.sample_count_var,
            width=112,
            height=34,
            border_color=COLORS["border"],
            font=FONTS["mono"],
        )
        self.sample_count_entry.grid(row=0, column=1, sticky="ew")
        ctk.CTkLabel(
            settings,
            text="Seed (optional)",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
        ).grid(row=0, column=2, padx=(14, 7), pady=10)
        self.seed_entry = ctk.CTkEntry(
            settings,
            textvariable=self.seed_var,
            width=112,
            height=34,
            border_color=COLORS["border"],
            placeholder_text="blank = random",
            font=FONTS["mono"],
        )
        self.seed_entry.grid(row=0, column=3, padx=(0, 12), sticky="ew")

        self.generate_button = ctk.CTkButton(
            panel,
            text="Generate Samples",
            height=38,
            corner_radius=11,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            text_color=COLORS["on_primary"],
            font=FONTS["button"],
            command=self.generate_samples,
        )
        self.generate_button.grid(row=5, column=0, padx=16, pady=(0, 12), sticky="ew")

    def _build_preview_panel(self, parent: ctk.CTkFrame) -> None:
        panel = ctk.CTkFrame(
            parent,
            fg_color=COLORS["surface"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["border"],
        )
        panel.grid(row=0, column=1, padx=(6, 0), sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(
            panel,
            text="Sampling coverage",
            text_color=COLORS["ink"],
            font=FONTS["card_title"],
            anchor="w",
        ).grid(row=0, column=0, padx=16, pady=(12, 5), sticky="w")
        self.coverage_canvas = tk.Canvas(
            panel,
            height=245,
            highlightthickness=1,
            highlightbackground=_active_color(COLORS["border"]),
            background=_active_color(COLORS["control"]),
        )
        self.coverage_canvas.grid(row=1, column=0, padx=16, sticky="nsew")
        self.coverage_canvas.bind(
            "<Configure>",
            lambda _event: self._draw_coverage(),
        )

        self.preview_title_label = ctk.CTkLabel(
            panel,
            text="Sample preview",
            text_color=COLORS["ink"],
            font=FONTS["body"],
            anchor="w",
        )
        self.preview_title_label.grid(
            row=2,
            column=0,
            padx=16,
            pady=(9, 2),
            sticky="w",
        )
        self.preview_frame = ctk.CTkFrame(
            panel,
            fg_color=COLORS["surface_alt"],
            corner_radius=10,
        )
        self.preview_frame.grid(row=3, column=0, padx=16, pady=(0, 12), sticky="ew")
        self._render_table_preview()

    def _build_footer(self) -> None:
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, padx=24, pady=(0, 16), sticky="ew")
        footer.grid_columnconfigure(0, weight=1)
        self.status_label = ctk.CTkLabel(
            footer,
            textvariable=self.status_var,
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            anchor="w",
            justify="left",
            wraplength=720,
        )
        self.status_label.grid(row=0, column=0, sticky="w")
        self.export_button = ctk.CTkButton(
            footer,
            text="Export inputs.csv",
            width=174,
            height=38,
            corner_radius=11,
            fg_color=COLORS["violet_soft"],
            hover_color=COLORS["violet_hover"],
            border_width=1,
            border_color=COLORS["violet"],
            text_color=COLORS["on_violet_soft"],
            font=FONTS["button"],
            state="disabled",
            command=self.export_samples,
        )
        self.export_button.grid(row=0, column=1, padx=(12, 0), sticky="e")

    def _append_editor(self) -> None:
        editor = LHSVariableEditor(
            name=ctk.StringVar(value=""),
            minimum=ctk.StringVar(value=""),
            maximum=ctk.StringVar(value=""),
        )
        for variable in (editor.name, editor.minimum, editor.maximum):
            variable.trace_add(
                "write",
                lambda *_args: self._invalidate_generated_samples(),
            )
        self.variable_editors.append(editor)

    def add_variable(self) -> None:
        if len(self.variable_editors) >= MAX_UI_VARIABLES:
            self._show_error(
                f"The compact generator supports up to {MAX_UI_VARIABLES} variables."
            )
            return
        self._append_editor()
        self.variable_page = (len(self.variable_editors) - 1) // VISIBLE_VARIABLE_ROWS
        self._render_variable_page()
        self._invalidate_generated_samples()

    def remove_variable(self, absolute_index: int) -> None:
        if len(self.variable_editors) == 1:
            self._show_error("At least one variable row is required.")
            return
        if 0 <= absolute_index < len(self.variable_editors):
            self.variable_editors.pop(absolute_index)
            maximum_page = (len(self.variable_editors) - 1) // VISIBLE_VARIABLE_ROWS
            self.variable_page = min(self.variable_page, maximum_page)
            self._render_variable_page()
            self._invalidate_generated_samples()

    def _change_page(self, change: int) -> None:
        maximum_page = (len(self.variable_editors) - 1) // VISIBLE_VARIABLE_ROWS
        self.variable_page = max(0, min(maximum_page, self.variable_page + change))
        self._render_variable_page()

    def _render_variable_page(self) -> None:
        for child in self.variable_rows_frame.winfo_children():
            child.destroy()
        start = self.variable_page * VISIBLE_VARIABLE_ROWS
        stop = min(start + VISIBLE_VARIABLE_ROWS, len(self.variable_editors))
        for row_index, absolute_index in enumerate(range(start, stop)):
            editor = self.variable_editors[absolute_index]
            ctk.CTkEntry(
                self.variable_rows_frame,
                textvariable=editor.name,
                width=180,
                height=34,
                border_color=COLORS["border"],
                placeholder_text=f"variable_{absolute_index + 1}",
                font=FONTS["mono"],
            ).grid(row=row_index, column=0, padx=(0, 7), pady=3, sticky="ew")
            ctk.CTkEntry(
                self.variable_rows_frame,
                textvariable=editor.minimum,
                width=88,
                height=34,
                border_color=COLORS["border"],
                placeholder_text="min",
                font=FONTS["mono"],
            ).grid(row=row_index, column=1, padx=(0, 7), pady=3, sticky="ew")
            ctk.CTkEntry(
                self.variable_rows_frame,
                textvariable=editor.maximum,
                width=88,
                height=34,
                border_color=COLORS["border"],
                placeholder_text="max",
                font=FONTS["mono"],
            ).grid(row=row_index, column=2, padx=(0, 7), pady=3, sticky="ew")
            ctk.CTkButton(
                self.variable_rows_frame,
                text="×",
                width=34,
                height=34,
                corner_radius=9,
                fg_color=COLORS["surface_alt"],
                hover_color=COLORS["control_hover"],
                text_color=COLORS["danger"],
                font=FONTS["button"],
                command=lambda index=absolute_index: self.remove_variable(index),
            ).grid(row=row_index, column=3, pady=3)
        self.variable_page_label.configure(
            text=f"Variables {start + 1}–{stop} of {len(self.variable_editors)}"
        )
        self.previous_page_button.configure(
            state="normal" if self.variable_page > 0 else "disabled"
        )
        self.next_page_button.configure(
            state=(
                "normal"
                if stop < len(self.variable_editors)
                else "disabled"
            )
        )
        self.add_variable_button.configure(
            state=(
                "normal"
                if len(self.variable_editors) < MAX_UI_VARIABLES
                else "disabled"
            )
        )

    @staticmethod
    def _parse_integer(value: str, label: str, *, optional: bool = False) -> int | None:
        cleaned = value.strip()
        if optional and not cleaned:
            return None
        if not cleaned:
            raise ValueError(f"{label} is required.")
        try:
            return int(cleaned)
        except ValueError as exc:
            raise ValueError(f"{label} must be a whole number.") from exc

    @staticmethod
    def _parse_number(value: str, label: str) -> float:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{label} is required.")
        try:
            number = float(cleaned)
        except ValueError as exc:
            raise ValueError(f"{label} must be numeric.") from exc
        if not math.isfinite(number):
            raise ValueError(f"{label} must be a finite numeric value.")
        return number

    def _request_from_form(self) -> LHSSampleGenerationRequest:
        variables: list[LHSVariable] = []
        for index, editor in enumerate(self.variable_editors, start=1):
            name = editor.name.get().strip()
            if not name:
                raise ValueError(f"Variable name in row {index} is required.")
            minimum = self._parse_number(
                editor.minimum.get(),
                f"Minimum in variable row {index}",
            )
            maximum = self._parse_number(
                editor.maximum.get(),
                f"Maximum in variable row {index}",
            )
            variables.append(LHSVariable(name, minimum, maximum))
        return LHSSampleGenerationRequest(
            variables=variables,
            sample_count=self._parse_integer(
                self.sample_count_var.get(),
                "Sample count",
            ),
            random_seed=self._parse_integer(
                self.seed_var.get(),
                "Random seed",
                optional=True,
            ),
        )

    def generate_samples(self) -> None:
        self.generate_button.configure(state="disabled", text="Generating…")
        self.export_button.configure(state="disabled")
        try:
            request = self._request_from_form()
            generated = generate_lhs_samples(request)
        except (TypeError, ValueError, ImportError) as exc:
            self.generated_samples = None
            self._show_error(str(exc))
        else:
            self.generated_samples = generated
            seed_label = (
                str(generated.random_seed)
                if generated.random_seed is not None
                else "random"
            )
            self.status_var.set(
                f"Generated {generated.sample_count:,} samples across "
                f"{len(generated.variable_names)} variables · seed {seed_label}."
            )
            self.status_label.configure(text_color=COLORS["success"])
            self.export_button.configure(state="normal")
            self._render_table_preview()
            self._draw_coverage()
        finally:
            self.generate_button.configure(state="normal", text="Generate Samples")

    def _invalidate_generated_samples(self) -> None:
        if self.generated_samples is None:
            return
        self.generated_samples = None
        self.export_button.configure(state="disabled")
        self.status_var.set("Settings changed · generate again before exporting.")
        self.status_label.configure(text_color=COLORS["warning"])
        self._render_table_preview()
        self._draw_empty_coverage()

    def _show_error(self, message: str) -> None:
        self.status_var.set(message)
        self.status_label.configure(text_color=COLORS["danger"])
        messagebox.showerror("LHS sample generator", message, parent=self)

    def _render_table_preview(self) -> None:
        for child in self.preview_frame.winfo_children():
            child.destroy()
        if self.generated_samples is None:
            self.preview_title_label.configure(text="Sample preview")
            ctk.CTkLabel(
                self.preview_frame,
                text="Generate samples to preview the first five rows.",
                text_color=COLORS["muted"],
                font=FONTS["caption"],
            ).grid(row=0, column=0, padx=12, pady=13, sticky="w")
            return
        shown_variables = self.generated_samples.variable_names[:PREVIEW_VARIABLES]
        headers = list(shown_variables)
        if len(self.generated_samples.variable_names) > PREVIEW_VARIABLES:
            self.preview_title_label.configure(
                text=(
                    f"Sample preview · first {PREVIEW_VARIABLES} of "
                    f"{len(self.generated_samples.variable_names)} variables"
                )
            )
        else:
            self.preview_title_label.configure(text="Sample preview")
        for column, header in enumerate(headers):
            self.preview_frame.grid_columnconfigure(column, weight=1)
            ctk.CTkLabel(
                self.preview_frame,
                text=header,
                text_color=COLORS["ink"],
                font=FONTS["caption"],
                anchor="w",
            ).grid(row=0, column=column, padx=7, pady=(6, 2), sticky="ew")
        for row_index, row in enumerate(
            self.generated_samples.rows[:PREVIEW_ROWS],
            start=1,
        ):
            values = [
                format(value, ".6g")
                for value in row[:PREVIEW_VARIABLES]
            ]
            for column, value in enumerate(values):
                ctk.CTkLabel(
                    self.preview_frame,
                    text=value,
                    text_color=COLORS["muted"],
                    font=FONTS["mono"],
                    anchor="w",
                ).grid(
                    row=row_index,
                    column=column,
                    padx=7,
                    pady=1,
                    sticky="ew",
                )

    def _draw_empty_coverage(self) -> None:
        canvas = self.coverage_canvas
        canvas.delete("all")
        width = max(320, canvas.winfo_width())
        height = max(180, canvas.winfo_height())
        canvas.create_text(
            width / 2,
            height / 2,
            text="Coverage preview appears after generation",
            fill=_active_color(COLORS["muted"]),
            font=("Segoe UI", 14),
        )

    def _draw_coverage(self) -> None:
        generated = self.generated_samples
        if generated is None or not generated.rows:
            self._draw_empty_coverage()
            return
        canvas = self.coverage_canvas
        canvas.delete("all")
        width = max(320, canvas.winfo_width())
        height = max(180, canvas.winfo_height())
        left, top, right, bottom = 82, 18, width - 20, height - 42
        ink = _active_color(COLORS["ink"])
        muted = _active_color(COLORS["muted"])
        grid = _active_color(COLORS["border"])
        point = _active_color(COLORS["primary"])
        canvas.create_rectangle(left, top, right, bottom, outline=grid)
        for step in range(1, 5):
            x = left + (right - left) * step / 5
            y = top + (bottom - top) * step / 5
            canvas.create_line(x, top, x, bottom, fill=grid, dash=(2, 3))
            canvas.create_line(left, y, right, y, fill=grid, dash=(2, 3))

        x_values = [row[0] for row in generated.rows]
        if len(generated.variable_names) >= 2:
            y_values = [row[1] for row in generated.rows]
            x_label = generated.variable_names[0]
            y_label = generated.variable_names[1]
        else:
            y_values = list(range(1, len(generated.rows) + 1))
            x_label = generated.variable_names[0]
            y_label = "Sample index"
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        x_span = x_max - x_min or 1.0
        y_span = y_max - y_min or 1.0
        stride = max(1, len(x_values) // 800)
        for x_value, y_value in zip(x_values[::stride], y_values[::stride], strict=True):
            x = left + (x_value - x_min) / x_span * (right - left)
            y = bottom - (y_value - y_min) / y_span * (bottom - top)
            canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=point, outline="")
        canvas.create_text(
            (left + right) / 2,
            height - 16,
            text=x_label,
            fill=ink,
            font=("Segoe UI Semibold", 12),
        )
        canvas.create_text(
            17,
            (top + bottom) / 2,
            text=y_label,
            angle=90,
            fill=ink,
            font=("Segoe UI Semibold", 12),
        )
        canvas.create_text(
            left - 8,
            top,
            text=format(y_max, ".4g"),
            fill=muted,
            anchor="ne",
            font=("Segoe UI", 10),
        )
        canvas.create_text(
            left - 8,
            bottom,
            text=format(y_min, ".4g"),
            fill=muted,
            anchor="se",
            font=("Segoe UI", 10),
        )
        canvas.create_line(left - 4, top, left, top, fill=grid)
        canvas.create_line(left - 4, bottom, left, bottom, fill=grid)
        canvas.create_text(left, bottom + 12, text=format(x_min, ".4g"), fill=muted, anchor="w")
        canvas.create_text(right, bottom + 12, text=format(x_max, ".4g"), fill=muted, anchor="e")

    def export_samples(self) -> None:
        if self.generated_samples is None:
            self._show_error("Generate samples before exporting inputs.csv.")
            return
        export_folder = self.project_path / "data" / "generated" / "lhs"
        export_folder.mkdir(parents=True, exist_ok=True)
        destination = filedialog.asksaveasfilename(
            title="Export LHS input samples",
            initialdir=export_folder,
            initialfile="inputs.csv",
            defaultextension=".csv",
            filetypes=[("CSV input table", "*.csv")],
            parent=self,
        )
        if not destination:
            return
        try:
            exported = write_lhs_inputs_csv(destination, self.generated_samples)
        except (OSError, ValueError) as exc:
            self._show_error(str(exc))
            return
        self.status_var.set(f"Exported and loaded into Data Prep: {exported.name}")
        self.status_label.configure(text_color=COLORS["success"])
        self.on_export(exported)
        messagebox.showinfo(
            "LHS inputs ready",
            (
                f"Saved {self.generated_samples.sample_count:,} simulation inputs to:\n"
                f"{exported}\n\n"
                "Run these rows in your simulator without reordering them, then "
                "return with an output CSV containing the same row count and order."
            ),
            parent=self,
        )

    def close(self) -> None:
        self.destroy()
