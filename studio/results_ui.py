"""Artifact-backed, non-scrolling Training Results page."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import TYPE_CHECKING, Any, Callable

import customtkinter as ctk

from studio.model_book import (
    ModelBook,
    ModelBookError,
    load_model_library,
    save_model_book,
)
from studio.model_comparison import (
    MODEL_FAMILY_ORDER,
    ModelComparisonError,
    ModelComparisonResult,
    compare_compatible_model_runs,
)
from studio.output_axis import infer_output_axis
from studio.project_store import Project
from studio.scientific_plot import (
    MAX_SCATTER_MARKERS,
    ScientificPlotState,
    ScientificPlotWorkbench,
)
from studio.theme import COLORS, FONTS
from studio.training_results import (
    TrainingResultsError,
    TrainingResultsView,
    load_latest_training_results,
    metric_card_data,
)

if TYPE_CHECKING:
    from studio.ui import StudioApp


RESULT_SECTIONS = (
    ("fit", "Predictions"),
    ("residuals", "Residuals"),
    ("errors", "Error Distribution"),
    ("configuration", "Configuration"),
    ("comparison", "Model Comparison"),
    ("run", "Run Info"),
)
DEFAULT_RESULTS_SECTION = "fit"


def _model_display_name(model_name: str) -> str:
    return {
        "linear_regression": "Linear Regression",
        "xgboost": "XGBoost",
        "neural_network": "Neural Network",
        "ensemble_ai_engine": "Ensemble AI Engine",
    }.get(model_name, model_name)


def _format_parameters(parameters: dict[str, Any]) -> str:
    if "weights" in parameters and "components" in parameters:
        return "weights: " + ", ".join(
            f"{_model_display_name(name)}={float(weight):.3f}"
            for name, weight in parameters["weights"].items()
        )
    if "hidden_layer_sizes" in parameters:
        return (
            f"layers={parameters['hidden_layer_sizes']}, "
            f"activation={parameters.get('activation')}, "
            f"learning_rate={parameters.get('learning_rate_init')} "
            f"(+{max(0, len(parameters) - 3)} training settings)"
        )
    if "n_estimators" in parameters:
        return (
            f"n_estimators={parameters['n_estimators']}, "
            f"max_depth={parameters.get('max_depth')}, "
            f"learning_rate={parameters.get('learning_rate')} "
            f"(+{max(0, len(parameters) - 3)} fixed settings)"
        )
    return ", ".join(f"{name}={value}" for name, value in parameters.items())


def _comparison_parameter_lines(parameters: dict[str, Any]) -> str:
    """Format only the user-controlled family parameters for a compact card."""

    if "weights" in parameters:
        return "weights: " + ", ".join(
            f"{_model_display_name(name)} {float(weight):.2f}"
            for name, weight in parameters["weights"].items()
        )
    if "hidden_layer_sizes" in parameters:
        return (
            f"layers={parameters.get('hidden_layer_sizes')}, "
            f"activation={parameters.get('activation')}\n"
            f"lr={parameters.get('learning_rate_init')}, "
            f"batch={parameters.get('batch_size')}, "
            f"epochs={parameters.get('max_iter')}"
        )
    if "n_estimators" in parameters:
        names = (
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
        )
        return "\n".join(
            ", ".join(
                f"{name}={parameters.get(name)}"
                for name in names[offset : offset + 2]
            )
            for offset in range(0, len(names), 2)
        )
    return "  ·  ".join(
        f"{name}={parameters.get(name)}"
        for name in ("fit_intercept", "positive")
    )


def _palette(color: tuple[str, str]) -> str:
    return color[1] if ctk.get_appearance_mode() == "Dark" else color[0]


def infer_curve_axis(target_names: tuple[str, ...]) -> tuple[str, tuple[float, ...]]:
    """Infer an editable axis label and coordinates from output-column names."""

    metadata = infer_output_axis(target_names)
    return metadata.display_label, metadata.values


class MetricHelpButton(ctk.CTkButton):
    """Compact, hover-and-focus help for a metric card."""

    def __init__(self, parent: ctk.CTkFrame):
        super().__init__(
            parent,
            text="?",
            width=22,
            height=22,
            corner_radius=11,
            fg_color="transparent",
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border_strong"],
            text_color=COLORS["muted"],
            font=("Segoe UI Semibold", 14),
        )
        self.help_title = ""
        self.help_text = ""
        self.tooltip_window: tk.Toplevel | None = None
        self.bind("<Enter>", self._show_tooltip, add="+")
        self.bind("<Leave>", self._hide_tooltip, add="+")
        self.bind("<FocusIn>", self._show_tooltip, add="+")
        self.bind("<FocusOut>", self._hide_tooltip, add="+")
        self.bind("<Button-1>", self._show_tooltip, add="+")
        self.bind("<Unmap>", self._hide_tooltip, add="+")
        self.bind("<Destroy>", self._hide_tooltip, add="+")

    def set_content(self, title: str, meaning: str, direction: str) -> None:
        self._hide_tooltip()
        self.help_title = title
        self.help_text = f"{meaning}\n{direction}"

    def _show_tooltip(self, _event: tk.Event | None = None) -> None:
        if not self.help_text or self.tooltip_window is not None:
            return
        tooltip = tk.Toplevel(self)
        tooltip.withdraw()
        tooltip.overrideredirect(True)
        try:
            tooltip.attributes("-topmost", True)
        except tk.TclError:
            pass
        tooltip.configure(bg=_palette(COLORS["border_strong"]))
        frame = ctk.CTkFrame(
            tooltip,
            fg_color=COLORS["surface_elevated"],
            corner_radius=9,
            border_width=1,
            border_color=COLORS["border_strong"],
        )
        frame.pack(fill="both", expand=True)
        ctk.CTkLabel(
            frame,
            text=self.help_title,
            text_color=COLORS["cyan"],
            font=FONTS["mono"],
            anchor="w",
        ).pack(fill="x", padx=12, pady=(9, 2))
        ctk.CTkLabel(
            frame,
            text=self.help_text,
            text_color=COLORS["ink"],
            font=FONTS["caption"],
            anchor="w",
            justify="left",
            wraplength=260,
        ).pack(fill="x", padx=12, pady=(0, 10))
        tooltip.update_idletasks()
        width = tooltip.winfo_reqwidth()
        height = tooltip.winfo_reqheight()
        x_position = self.winfo_rootx() + self.winfo_width() - width
        y_position = self.winfo_rooty() + self.winfo_height() + 6
        x_position = max(
            8,
            min(x_position, self.winfo_screenwidth() - width - 8),
        )
        if y_position + height > self.winfo_screenheight() - 8:
            y_position = max(8, self.winfo_rooty() - height - 6)
        tooltip.geometry(f"+{x_position}+{y_position}")
        tooltip.deiconify()
        self.tooltip_window = tooltip

    def _hide_tooltip(self, _event: tk.Event | None = None) -> None:
        if self.tooltip_window is None:
            return
        try:
            self.tooltip_window.destroy()
        except tk.TclError:
            pass
        self.tooltip_window = None


class CurveComparisonChart(ctk.CTkFrame):
    """Actual and predicted output curves for one selected test sample."""

    mode = "actual_predicted_curves"

    def __init__(
        self,
        parent: ctk.CTkFrame,
        view: TrainingResultsView,
        sample_id: str,
        x_label: str,
        y_label: str,
        x_values: tuple[float, ...],
    ):
        super().__init__(
            parent,
            fg_color=COLORS["surface"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"],
        )
        self.view = view
        self.sample_id = sample_id
        self.x_label = x_label
        self.y_label = y_label
        records = view.predictions_for_sample(sample_id)
        if len(records) != len(x_values):
            raise ValueError(
                "The X-axis value count does not match this sample's output count."
            )
        ordered = sorted(
            zip(x_values, records),
            key=lambda pair: pair[0],
        )
        self.x_values = tuple(pair[0] for pair in ordered)
        self.curve_predictions = tuple(pair[1] for pair in ordered)
        self.point_locations: list[tuple[float, float, str]] = []
        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas.bind("<Configure>", self._redraw)
        self.canvas.bind("<Motion>", self._show_hover)
        self.canvas.bind("<Leave>", lambda _event: self._clear_tooltip())

    def refresh_theme(self) -> None:
        self._redraw()

    def _redraw(self, _event: tk.Event | None = None) -> None:
        canvas = self.canvas
        canvas.delete("all")
        canvas.configure(bg=_palette(COLORS["surface"]))
        self.point_locations = []
        width = max(canvas.winfo_width(), 500)
        height = max(canvas.winfo_height(), 245)
        left, right, top, bottom = 72, width - 24, 54, height - 54
        actual_values = [row.actual_value for row in self.curve_predictions]
        predicted_values = [row.predicted_value for row in self.curve_predictions]
        x_min, x_max = _numeric_range(list(self.x_values))
        y_min, y_max = _numeric_range([*actual_values, *predicted_values])

        def map_x(value: float) -> float:
            return left + (value - x_min) / (x_max - x_min) * (right - left)

        def map_y(value: float) -> float:
            return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

        ink = _palette(COLORS["ink"])
        muted = _palette(COLORS["muted"])
        border = _palette(COLORS["border_strong"])
        grid = _palette(COLORS["border"])
        actual_color = _palette(COLORS["primary"])
        predicted_color = _palette(COLORS["violet"])

        canvas.create_text(
            left,
            20,
            text=f"Actual vs Predicted — {self.sample_id}",
            anchor="w",
            fill=ink,
            font=("Segoe UI Semibold", 16),
            tags="plot_title",
        )
        legend_x = max(left + 225, right - 205)
        canvas.create_line(
            legend_x,
            18,
            legend_x + 28,
            18,
            fill=actual_color,
            width=3,
            tags="actual_legend",
        )
        canvas.create_oval(
            legend_x + 10,
            14,
            legend_x + 18,
            22,
            fill=actual_color,
            outline=_palette(COLORS["on_primary"]),
            tags="actual_legend",
        )
        canvas.create_text(
            legend_x + 34,
            18,
            text="Actual",
            anchor="w",
            fill=ink,
            font=("Segoe UI Semibold", 12),
            tags="actual_legend",
        )
        predicted_x = legend_x + 98
        canvas.create_line(
            predicted_x,
            18,
            predicted_x + 28,
            18,
            fill=predicted_color,
            width=3,
            dash=(7, 4),
            tags="predicted_legend",
        )
        canvas.create_polygon(
            predicted_x + 14,
            13,
            predicted_x + 19,
            18,
            predicted_x + 14,
            23,
            predicted_x + 9,
            18,
            fill=predicted_color,
            outline=_palette(COLORS["on_accent"]),
            tags="predicted_legend",
        )
        canvas.create_text(
            predicted_x + 34,
            18,
            text="Predicted",
            anchor="w",
            fill=ink,
            font=("Segoe UI Semibold", 12),
            tags="predicted_legend",
        )

        for tick_index in range(5):
            value = y_min + (y_max - y_min) * tick_index / 4
            y_coordinate = map_y(value)
            canvas.create_line(
                left,
                y_coordinate,
                right,
                y_coordinate,
                fill=grid,
                width=1,
                dash=(2, 4),
            )
            canvas.create_text(
                left - 8,
                y_coordinate,
                text=f"{value:.5g}",
                anchor="e",
                fill=muted,
                font=("Segoe UI", 11),
            )
        canvas.create_line(left, bottom, right, bottom, fill=border, width=2)
        canvas.create_line(left, top, left, bottom, fill=border, width=2)

        tick_count = min(8, len(self.x_values))
        if tick_count == 1:
            tick_indexes = (0,)
        else:
            tick_indexes = tuple(
                sorted(
                    {
                        round(index * (len(self.x_values) - 1) / (tick_count - 1))
                        for index in range(tick_count)
                    }
                )
            )
        for index in tick_indexes:
            value = self.x_values[index]
            x_coordinate = map_x(value)
            canvas.create_line(
                x_coordinate,
                bottom,
                x_coordinate,
                bottom + 5,
                fill=border,
                width=1,
            )
            canvas.create_text(
                x_coordinate,
                bottom + 16,
                text=f"{value:.5g}",
                fill=muted,
                font=("Segoe UI", 11),
            )
        canvas.create_text(
            (left + right) / 2,
            bottom + 38,
            text=self.x_label,
            fill=ink,
            font=("Segoe UI Semibold", 14),
            tags="x_axis_label",
        )
        canvas.create_text(
            18,
            (top + bottom) / 2,
            text=self.y_label,
            fill=ink,
            font=("Segoe UI Semibold", 14),
            angle=90,
            tags="y_axis_label",
        )

        actual_coordinates: list[float] = []
        predicted_coordinates: list[float] = []
        for x_value, actual, predicted in zip(
            self.x_values,
            actual_values,
            predicted_values,
        ):
            x_coordinate = map_x(x_value)
            actual_coordinates.extend((x_coordinate, map_y(actual)))
            predicted_coordinates.extend((x_coordinate, map_y(predicted)))
        if len(self.x_values) > 1:
            canvas.create_line(
                *actual_coordinates,
                fill=actual_color,
                width=3,
                tags="actual_curve",
            )
            canvas.create_line(
                *predicted_coordinates,
                fill=predicted_color,
                width=3,
                dash=(8, 5),
                tags="predicted_curve",
            )
        else:
            canvas.create_text(
                (left + right) / 2,
                top + 18,
                text=(
                    "This sample has one output point. Multiple output columns "
                    "are required to form curves."
                ),
                fill=muted,
                font=("Segoe UI", 12),
            )

        for index, (x_value, record) in enumerate(
            zip(self.x_values, self.curve_predictions)
        ):
            x_coordinate = map_x(x_value)
            actual_y = map_y(record.actual_value)
            predicted_y = map_y(record.predicted_value)
            canvas.create_oval(
                x_coordinate - 5,
                actual_y - 5,
                x_coordinate + 5,
                actual_y + 5,
                fill=actual_color,
                outline=_palette(COLORS["on_primary"]),
                width=1,
                tags="actual_point",
            )
            canvas.create_polygon(
                x_coordinate,
                predicted_y - 5,
                x_coordinate + 5,
                predicted_y,
                x_coordinate,
                predicted_y + 5,
                x_coordinate - 5,
                predicted_y,
                fill=predicted_color,
                outline=_palette(COLORS["on_accent"]),
                width=1,
                tags="predicted_point",
            )
            target_name = record.target_name or self.view.target_columns[index]
            detail = (
                f"{self.sample_id}\n"
                f"{self.x_label}: {x_value:.6g}\n"
                f"Output: {target_name}\n"
                f"Actual: {record.actual_value:.6g}\n"
                f"Predicted: {record.predicted_value:.6g}\n"
                f"Residual: {record.residual:.6g}"
            )
            self.point_locations.extend(
                (
                    (x_coordinate, actual_y, detail),
                    (x_coordinate, predicted_y, detail),
                )
            )

    def _show_hover(self, event: tk.Event) -> None:
        self._clear_tooltip()
        for x_coordinate, y_coordinate, detail in self.point_locations:
            if (event.x - x_coordinate) ** 2 + (event.y - y_coordinate) ** 2 <= 81:
                text_id = self.canvas.create_text(
                    event.x + 12,
                    event.y - 10,
                    text=detail,
                    anchor="sw",
                    fill=_palette(COLORS["ink"]),
                    font=("Segoe UI", 11),
                    tags="tooltip",
                )
                bounds = self.canvas.bbox(text_id)
                if bounds:
                    rectangle = self.canvas.create_rectangle(
                        bounds[0] - 6,
                        bounds[1] - 4,
                        bounds[2] + 6,
                        bounds[3] + 4,
                        fill=_palette(COLORS["surface_elevated"]),
                        outline=_palette(COLORS["border_strong"]),
                        tags="tooltip",
                    )
                    self.canvas.tag_lower(rectangle, text_id)
                return

    def _clear_tooltip(self) -> None:
        self.canvas.delete("tooltip")


class ResultsChart(ctk.CTkFrame):
    """Small native-canvas chart with sample details on pointer hover."""

    def __init__(
        self,
        parent: ctk.CTkFrame,
        view: TrainingResultsView,
        mode: str,
    ):
        super().__init__(
            parent,
            fg_color=COLORS["surface"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"],
        )
        self.view = view
        self.mode = mode
        self.point_locations: list[tuple[float, float, str]] = []
        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas.bind("<Configure>", self._redraw)
        self.canvas.bind("<Motion>", self._show_hover)
        self.canvas.bind("<Leave>", lambda _event: self._clear_tooltip())

    def refresh_theme(self) -> None:
        self._redraw()

    def _redraw(self, _event: tk.Event | None = None) -> None:
        canvas = self.canvas
        canvas.delete("all")
        canvas.configure(bg=_palette(COLORS["surface"]))
        self.point_locations = []
        width = max(canvas.winfo_width(), 420)
        height = max(canvas.winfo_height(), 210)
        if self.mode == "residuals":
            self._draw_residuals(width, height)
        else:
            self._draw_histogram(width, height)

    def _draw_residuals(self, width: int, height: int) -> None:
        left, right, top, bottom = 58, width - 18, 18, height - 42
        predictions = self.view.predictions
        x_values = [item.predicted_value for item in predictions]
        y_values = [item.residual for item in predictions]
        x_min, x_max = _numeric_range(x_values)
        y_min, y_max = _numeric_range([*y_values, 0.0])

        def map_x(value: float) -> float:
            return left + (value - x_min) / (x_max - x_min) * (right - left)

        def map_y(value: float) -> float:
            return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

        self._axes(
            left,
            right,
            top,
            bottom,
            "Predicted value",
            "Residual (Actual − Predicted)",
            x_min,
            x_max,
            y_min,
            y_max,
        )
        zero_y = map_y(0.0)
        self.canvas.create_line(
            left,
            zero_y,
            right,
            zero_y,
            fill=_palette(COLORS["warning"]),
            width=2,
            dash=(6, 4),
        )
        for item, x_value, y_value in zip(predictions, x_values, y_values):
            x_coordinate = map_x(x_value)
            y_coordinate = map_y(y_value)
            self.canvas.create_oval(
                x_coordinate - 4,
                y_coordinate - 4,
                x_coordinate + 4,
                y_coordinate + 4,
                fill=_palette(COLORS["primary"]),
                outline=_palette(COLORS["on_primary"]),
                width=1,
            )
            detail = (
                f"{item.sample_id}\n"
                f"Actual: {item.actual_value:.6g}\n"
                f"Predicted: {item.predicted_value:.6g}\n"
                f"Residual: {item.residual:.6g}"
            )
            self.point_locations.append((x_coordinate, y_coordinate, detail))

    def _draw_histogram(self, width: int, height: int) -> None:
        left, right, top, bottom = 58, width - 18, 18, height - 42
        values = [item.absolute_error for item in self.view.predictions]
        maximum = max(values)
        bin_count = min(8, max(3, int(math.sqrt(len(values))) + 1))
        bin_width = maximum / bin_count if maximum > 0 else 1.0
        counts = [0] * bin_count
        for value in values:
            index = min(int(value / bin_width), bin_count - 1)
            counts[index] += 1
        max_count = max(counts) or 1
        self._axes(
            left,
            right,
            top,
            bottom,
            "Absolute error",
            "Test samples",
            0.0,
            maximum if maximum > 0 else 1.0,
            0.0,
            float(max_count),
        )
        available = right - left
        bar_width = available / bin_count
        for index, count in enumerate(counts):
            x0 = left + index * bar_width + 3
            x1 = left + (index + 1) * bar_width - 3
            y0 = bottom - count / max_count * (bottom - top)
            self.canvas.create_rectangle(
                x0,
                y0,
                x1,
                bottom,
                fill=_palette(COLORS["violet"]),
                outline="",
            )

    def _axes(
        self,
        left: float,
        right: float,
        top: float,
        bottom: float,
        x_label: str,
        y_label: str,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> None:
        ink = _palette(COLORS["ink"])
        border = _palette(COLORS["border_strong"])
        self.canvas.create_line(left, bottom, right, bottom, fill=border, width=1)
        self.canvas.create_line(left, top, left, bottom, fill=border, width=1)
        self.canvas.create_text(
            (left + right) / 2,
            bottom + 28,
            text=x_label,
            fill=ink,
            font=("Segoe UI", 12),
        )
        self.canvas.create_text(
            14,
            (top + bottom) / 2,
            text=y_label,
            fill=ink,
            font=("Segoe UI", 12),
            angle=90,
        )
        for value, x in ((x_min, left), (x_max, right)):
            self.canvas.create_text(
                x,
                bottom + 12,
                text=f"{value:.4g}",
                fill=_palette(COLORS["muted"]),
                font=("Segoe UI", 11),
            )
        for value, y in ((y_min, bottom), (y_max, top)):
            self.canvas.create_text(
                left - 6,
                y,
                text=f"{value:.4g}",
                fill=_palette(COLORS["muted"]),
                font=("Segoe UI", 11),
                anchor="e",
            )

    def _show_hover(self, event: tk.Event) -> None:
        self._clear_tooltip()
        for x_coordinate, y_coordinate, detail in self.point_locations:
            if (event.x - x_coordinate) ** 2 + (event.y - y_coordinate) ** 2 <= 64:
                text_id = self.canvas.create_text(
                    event.x + 12,
                    event.y - 10,
                    text=detail,
                    anchor="sw",
                    fill=_palette(COLORS["ink"]),
                    font=("Segoe UI", 11),
                    tags="tooltip",
                )
                bounds = self.canvas.bbox(text_id)
                if bounds:
                    rectangle = self.canvas.create_rectangle(
                        bounds[0] - 6,
                        bounds[1] - 4,
                        bounds[2] + 6,
                        bounds[3] + 4,
                        fill=_palette(COLORS["surface_elevated"]),
                        outline=_palette(COLORS["border_strong"]),
                        tags="tooltip",
                    )
                    self.canvas.tag_lower(rectangle, text_id)
                return

    def _clear_tooltip(self) -> None:
        self.canvas.delete("tooltip")


class ModelComparisonMetricChart(ctk.CTkFrame):
    """Compact comparison where longer bars consistently mean better quality."""

    metric_names = ("Validation RMSE", "Test RMSE", "MAE", "R²")

    def __init__(
        self,
        parent: ctk.CTkFrame,
        comparison: ModelComparisonResult,
    ):
        super().__init__(
            parent,
            fg_color=COLORS["surface_alt"],
            corner_radius=10,
            border_width=1,
            border_color=COLORS["border"],
        )
        self.comparison = comparison
        self.metric_values: dict[str, dict[str, float]] = {}
        self.metric_bar_values: dict[str, dict[str, float]] = {}
        self.metric_bars: dict[str, dict[str, ctk.CTkProgressBar]] = {}
        pair_columns = tuple(2 + index * 2 for index in range(len(MODEL_FAMILY_ORDER)))
        for column in pair_columns:
            self.grid_columnconfigure(column, weight=1)
        self.quality_caption = ctk.CTkLabel(
            self,
            text="RELATIVE QUALITY · LONGER BAR IS BETTER",
            height=18,
            text_color=COLORS["muted"],
            font=FONTS["mono"],
            anchor="w",
        )
        self.quality_caption.grid(
            row=0,
            column=0,
            columnspan=1 + 2 * len(MODEL_FAMILY_ORDER),
            padx=12,
            pady=(4, 0),
            sticky="ew",
        )
        ctk.CTkLabel(
            self,
            text="Metric",
            height=18,
            text_color=COLORS["cyan"],
            font=FONTS["mono"],
            anchor="w",
        ).grid(row=1, column=0, padx=(12, 8), pady=(1, 2), sticky="w")
        for index, model_name in enumerate(MODEL_FAMILY_ORDER):
            column = 1 + index * 2
            family = comparison.family(model_name)
            ctk.CTkLabel(
                self,
                text=family.display_name,
                height=18,
                text_color=(
                    COLORS["success"]
                    if comparison.recommended_model == model_name
                    else COLORS["ink"]
                ),
                font=("Segoe UI Semibold", 12),
            ).grid(
                row=1,
                column=column,
                columnspan=2,
                padx=(4, 10),
                pady=(1, 2),
                sticky="w",
            )

        for row, metric_name in enumerate(self.metric_names, start=2):
            lower_is_better = metric_name != "R²"
            ctk.CTkLabel(
                self,
                text=f"{metric_name} {'↓' if lower_is_better else '↑'}",
                height=18,
                text_color=COLORS["muted"],
                font=("Segoe UI", 12),
                anchor="w",
            ).grid(row=row, column=0, padx=(12, 8), pady=1, sticky="w")
            values = self._metric_pair(metric_name)
            self.metric_values[metric_name] = values
            bar_values = self._quality_values(
                values,
                lower_is_better=lower_is_better,
            )
            self.metric_bar_values[metric_name] = bar_values
            self.metric_bars[metric_name] = {}
            colors = (
                COLORS["primary"],
                COLORS["violet"],
                COLORS["cyan"],
                COLORS["success"],
            )
            for index, model_name in enumerate(MODEL_FAMILY_ORDER):
                value_column = 1 + index * 2
                bar_column = 2 + index * 2
                color = colors[index]
                value = values.get(model_name)
                ctk.CTkLabel(
                    self,
                    text="—" if value is None else f"{value:.6g}",
                    height=18,
                    text_color=COLORS["ink"],
                    font=("Cascadia Mono", 11),
                    width=70,
                    anchor="e",
                ).grid(row=row, column=value_column, padx=(4, 5), pady=1, sticky="e")
                bar = ctk.CTkProgressBar(
                    self,
                    height=7,
                    corner_radius=4,
                    fg_color=COLORS["surface_elevated"],
                    progress_color=color,
                )
                bar.grid(row=row, column=bar_column, padx=(0, 10), pady=1, sticky="ew")
                bar.set(bar_values.get(model_name, 0.0))
                self.metric_bars[metric_name][model_name] = bar

    @staticmethod
    def _quality_values(
        values: dict[str, float],
        *,
        lower_is_better: bool,
    ) -> dict[str, float]:
        if not values:
            return {}
        if len(values) == 1:
            return {name: 1.0 for name in values}
        best = min(values.values()) if lower_is_better else max(values.values())
        worst = max(values.values()) if lower_is_better else min(values.values())
        span = abs(worst - best)
        if math.isclose(span, 0.0, rel_tol=1e-12, abs_tol=1e-15):
            return {name: 1.0 for name in values}
        quality: dict[str, float] = {}
        for name, value in values.items():
            relative = (
                (worst - value) / span
                if lower_is_better
                else (value - worst) / span
            )
            quality[name] = 0.25 + 0.75 * max(0.0, min(1.0, relative))
        return quality

    def _metric_pair(self, metric_name: str) -> dict[str, float]:
        values: dict[str, float] = {}
        for model_name in MODEL_FAMILY_ORDER:
            run = self.comparison.family(model_name).best_run
            if run is None:
                continue
            values[model_name] = {
                "Validation RMSE": float(run.validation_rmse),
                "Test RMSE": run.test_rmse,
                "MAE": run.mae,
                "R²": run.r_squared,
            }[metric_name]
        return values


class TrainingResultsPage(ctk.CTkFrame):
    """Latest-run results view driven entirely by saved artifacts."""

    def __init__(self, parent: ctk.CTkFrame, app: "StudioApp"):
        super().__init__(parent, fg_color=COLORS["app_bg"], corner_radius=0)
        self.app = app
        self.project: Project | None = None
        self.result: TrainingResultsView | None = None
        self.load_error: str | None = None
        self.failure_state: str | None = None
        self.requested_run_id: str | None = None
        self.active_section = DEFAULT_RESULTS_SECTION
        self.section_buttons: dict[str, ctk.CTkButton] = {}
        self.current_chart: ResultsChart | ScientificPlotWorkbench | None = None
        self.prediction_plot_state = ScientificPlotState()
        self.residual_plot_state = ScientificPlotState()
        self.curve_sample = ctk.StringVar(value="")
        self.curve_axis_values: tuple[float, ...] = (1.0,)
        self._curve_run_id: str | None = None
        self.saved_model_book_id: str | None = None
        self.saved_model_book_name: str | None = None
        self.model_comparison: ModelComparisonResult | None = None
        self.comparison_error: str | None = None
        self.comparison_metric_chart: ModelComparisonMetricChart | None = None
        self.comparison_run_buttons: dict[str, ctk.CTkButton] = {}

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)
        self._build_header()
        self._build_recommendation()
        self._build_metrics()
        self._build_section_navigation()
        self._build_content()
        self._build_footer()
        self._show_empty_state()

    def set_project(self, project: Project | None) -> None:
        # A loaded project or newly completed run should begin with the primary
        # engineering deliverable, never a subview selected for another result.
        self.active_section = DEFAULT_RESULTS_SECTION
        self.requested_run_id = None
        self.project = project
        self.failure_state = None
        self.reload()

    def reload(self) -> None:
        self.result = None
        self.load_error = None
        self.saved_model_book_id = None
        self.saved_model_book_name = None
        self.model_comparison = None
        self.comparison_error = None
        self.comparison_metric_chart = None
        self.comparison_run_buttons = {}
        if self.project is not None:
            try:
                self.result = load_latest_training_results(
                    self.project.path,
                    run_id=self.requested_run_id,
                )
            except TrainingResultsError as exc:
                self.load_error = str(exc)
        self._refresh()

    def show_training_failure(self) -> None:
        self.failure_state = (
            "Training did not complete.\n"
            "No performance results are available for this run."
        )
        self._show_empty_state()

    def describe_ui_state(self) -> list[str]:
        if self.failure_state:
            return ["Training Results state: latest training attempt failed"]
        if self.load_error:
            return [f"Training Results artifact error: {self.load_error}"]
        if self.result is None:
            return ["Training Results state: no completed run"]
        state = [
            f"Displayed results run: {self.result.run_id}",
            f"Results training mode: {self.result.training_mode.title()}",
            f"Results section: {self.active_section}",
            f"Dataset fingerprint: {self.result.dataset_fingerprint}",
            (
                "Displayed result scope: latest completed run"
                if self.requested_run_id is None
                else "Displayed result scope: selected comparison run"
            ),
            (
                f"Open Model Library action: available for {self.saved_model_book_name}"
                if self.saved_model_book_id
                else "Create Model Book action: available for this completed run"
            ),
            (
                "Comparable Auto recommendation: available"
                if self.result.custom_recommendation
                else "Comparable Auto recommendation: unavailable"
            ),
            "Results page scrolling: none; ordered section navigator",
        ]
        if self.active_section == "fit":
            state.extend(
                (
                    f"Prediction curve sample: {self.curve_sample.get()}",
                    f"Prediction curve X-axis: {self.prediction_plot_state.x_label}",
                    f"Prediction curve Y-axis: {self.prediction_plot_state.y_label}",
                    "Predictions display: overlaid Actual and Predicted curves",
                    "Predictions plot: shared Scientific Plot Workbench controls",
                )
            )
        elif self.active_section == "comparison":
            state.append(
                f"Model comparison recommendation: "
                f"{self.model_comparison.recommendation_title}"
                if self.model_comparison
                else f"Model comparison unavailable: {self.comparison_error or 'not loaded'}"
            )
        return state

    def refresh_theme(self) -> None:
        if self.current_chart is not None:
            self.current_chart.refresh_theme()

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, padx=28, pady=(14, 7), sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="Training Results",
            text_color=COLORS["ink"],
            font=FONTS["display"],
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        self.run_badge = ctk.CTkLabel(
            header,
            text="NO COMPLETED RUN",
            height=26,
            corner_radius=13,
            fg_color=COLORS["surface_alt"],
            text_color=COLORS["muted"],
            font=FONTS["mono"],
        )
        self.run_badge.grid(row=0, column=1, sticky="e")

    def _build_recommendation(self) -> None:
        self.recommendation_card = ctk.CTkFrame(
            self,
            height=44,
            fg_color=COLORS["primary_soft"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border_strong"],
        )
        self.recommendation_card.grid(
            row=1, column=0, padx=28, pady=(0, 7), sticky="ew"
        )
        self.recommendation_card.grid_propagate(False)
        self.recommendation_card.grid_columnconfigure(1, weight=1)
        self.recommendation_title = ctk.CTkLabel(
            self.recommendation_card,
            text="",
            height=24,
            corner_radius=8,
            fg_color=COLORS["primary"],
            text_color=COLORS["on_primary"],
            font=("Segoe UI Semibold", 12),
        )
        self.recommendation_title.grid(
            row=0, column=0, padx=(10, 8), pady=10, sticky="w"
        )
        self.recommendation_body = ctk.CTkLabel(
            self.recommendation_card,
            text="",
            text_color=COLORS["ink"],
            font=FONTS["body_small"],
            anchor="w",
        )
        self.recommendation_body.grid(
            row=0, column=1, padx=(0, 10), pady=8, sticky="ew"
        )

    def _build_metrics(self) -> None:
        self.metrics_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.metrics_frame.grid(
            row=2, column=0, padx=28, pady=(0, 7), sticky="ew"
        )
        for column in range(4):
            self.metrics_frame.grid_columnconfigure(column, weight=1, uniform="metric")
        self.metrics_context_label = ctk.CTkLabel(
            self.metrics_frame,
            text="LATEST SELECTED RUN METRICS",
            text_color=COLORS["muted"],
            font=FONTS["mono"],
            anchor="w",
        )
        self.metrics_context_label.grid(
            row=0,
            column=0,
            columnspan=4,
            padx=2,
            pady=(0, 4),
            sticky="ew",
        )
        self.metric_widgets: list[
            tuple[ctk.CTkLabel, ctk.CTkLabel, MetricHelpButton]
        ] = []
        for column in range(4):
            card = ctk.CTkFrame(
                self.metrics_frame,
                height=62,
                fg_color=COLORS["surface"],
                corner_radius=12,
                border_width=1,
                border_color=COLORS["border"],
            )
            card.grid(
                row=1,
                column=column,
                padx=(0 if column == 0 else 3, 0 if column == 3 else 3),
                sticky="nsew",
            )
            card.grid_propagate(False)
            card.grid_columnconfigure(0, weight=1)
            name = ctk.CTkLabel(
                card,
                text="",
                text_color=COLORS["cyan"],
                font=FONTS["mono"],
                anchor="w",
            )
            name.grid(row=0, column=0, padx=(12, 4), pady=(6, 0), sticky="ew")
            help_button = MetricHelpButton(card)
            help_button.grid(row=0, column=1, padx=(0, 8), pady=(5, 0), sticky="e")
            value = ctk.CTkLabel(
                card,
                text="",
                text_color=COLORS["ink"],
                font=FONTS["section"],
                anchor="w",
            )
            value.grid(
                row=1,
                column=0,
                columnspan=2,
                padx=12,
                pady=(0, 6),
                sticky="ew",
            )
            self.metric_widgets.append((name, value, help_button))

    def _build_section_navigation(self) -> None:
        self.section_nav = ctk.CTkFrame(self, fg_color="transparent")
        self.section_nav.grid(row=3, column=0, padx=28, pady=(0, 6), sticky="ew")
        for column, (key, label) in enumerate(RESULT_SECTIONS):
            self.section_nav.grid_columnconfigure(column, weight=1)
            button = ctk.CTkButton(
                self.section_nav,
                text=label,
                height=31,
                corner_radius=9,
                fg_color=(COLORS["nav_active"] if key == "fit" else "transparent"),
                hover_color=COLORS["control_hover"],
                border_width=1,
                border_color=COLORS["border"],
                text_color=COLORS["ink"],
                font=("Segoe UI Semibold", 12),
                command=lambda section=key: self.show_section(section),
            )
            button.grid(
                row=0,
                column=column,
                padx=(
                    0 if column == 0 else 2,
                    0 if column == len(RESULT_SECTIONS) - 1 else 2,
                ),
                sticky="ew",
            )
            self.section_buttons[key] = button

    def _build_content(self) -> None:
        self.content_card = ctk.CTkFrame(
            self,
            fg_color=COLORS["surface"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["border"],
        )
        self.content_card.grid(
            row=4, column=0, padx=28, pady=(0, 7), sticky="nsew"
        )
        self.content_card.grid_columnconfigure(0, weight=1)
        self.content_card.grid_rowconfigure(0, weight=1)
        self.empty_label = ctk.CTkLabel(
            self.content_card,
            text="",
            text_color=COLORS["muted"],
            font=FONTS["body"],
            justify="center",
        )

    def _build_footer(self) -> None:
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=5, column=0, padx=28, pady=(0, 10), sticky="ew")
        footer.grid_columnconfigure(1, weight=1)
        self.train_again_button = ctk.CTkButton(
            footer,
            text="Adjust & Train Again",
            width=176,
            height=36,
            corner_radius=11,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            command=lambda: self.app.show_page("training"),
        )
        self.train_again_button.grid(row=0, column=0, sticky="w")
        self.footer_status_label = ctk.CTkLabel(
            footer,
            text="Saved run artifacts · Read-only results",
            text_color=COLORS["subtle"],
            font=FONTS["caption"],
        )
        self.footer_status_label.grid(row=0, column=1, padx=10)
        self.save_model_button = ctk.CTkButton(
            footer,
            text="Create Model Book  →",
            width=176,
            height=36,
            corner_radius=11,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            text_color=COLORS["on_primary"],
            font=FONTS["button"],
            state="disabled",
            command=self._primary_model_action,
        )
        self.save_model_button.grid(row=0, column=2, sticky="e")

    def _refresh(self) -> None:
        if self.failure_state or self.load_error or self.result is None:
            self._show_empty_state()
            return
        self.run_badge.configure(
            text=(
                f"LATEST · {self.result.run_id.upper()}"
                if self.requested_run_id is None
                else f"SELECTED · {self.result.run_id.upper()}"
            ),
            fg_color=COLORS["success_soft"],
            text_color=COLORS["success"],
        )
        self._find_saved_model_book()
        self._configure_primary_model_action()
        self.recommendation_card.grid()
        self.metrics_frame.grid()
        self.section_nav.grid()
        self._render_recommendation()
        self._prepare_curve_state()
        scope = (
            "LATEST SELECTED RUN METRICS"
            if self.requested_run_id is None
            else "SELECTED RUN METRICS"
        )
        self.metrics_context_label.configure(
            text=(
                f"{scope}  ·  {self.result.run_id.upper()}  ·  "
                f"{_model_display_name(self.result.model_name).upper()}"
            )
        )
        for widgets, metric in zip(self.metric_widgets, metric_card_data(self.result)):
            name, value, help_button = widgets
            name.configure(text=metric["name"])
            value.configure(text=metric["display_value"])
            help_button.set_content(
                metric["name"],
                metric["meaning"],
                metric["direction"],
            )
        self.show_section(self.active_section)

    def _prepare_curve_state(self) -> None:
        assert self.result is not None
        sample_ids = self.result.prediction_sample_ids
        if not sample_ids:
            return
        curve_result_id = (
            f"{self.result.project_path}|{self.result.dataset_fingerprint}|"
            f"{self.result.run_id}"
        )
        if self._curve_run_id != curve_result_id:
            self._curve_run_id = curve_result_id
            self.prediction_plot_state = ScientificPlotState()
            self.residual_plot_state = ScientificPlotState()
            self.curve_sample.set(sample_ids[0])
            axis_label, axis_values = infer_curve_axis(self.result.target_columns)
            y_label = "Response value"
            if self.result.target_unit:
                y_label = f"{y_label} ({self.result.target_unit})"
            self.curve_axis_values = axis_values
            self.prediction_plot_state.x_label = axis_label
            self.prediction_plot_state.y_label = y_label
            self.prediction_plot_state.plot_title = "Actual vs Predicted Response"
            self.residual_plot_state.x_label = "Predicted value"
            self.residual_plot_state.y_label = "Residual (Actual − Predicted)"
            self.residual_plot_state.plot_title = "Residual Analysis"
        elif self.curve_sample.get() not in sample_ids:
            self.curve_sample.set(sample_ids[0])

    def _show_empty_state(self) -> None:
        self.recommendation_card.grid_remove()
        self.metrics_frame.grid_remove()
        self.section_nav.grid_remove()
        for child in self.content_card.winfo_children():
            child.grid_forget()
            if child is not self.empty_label:
                child.destroy()
        if self.failure_state:
            message = self.failure_state
        elif self.load_error:
            message = (
                "Training results could not be loaded.\n"
                f"{self.load_error}"
            )
        else:
            message = (
                "No completed training run is available yet.\n"
                "Train a model to view performance and prediction plots."
            )
        self.empty_label.configure(text=message)
        self.empty_label.grid(row=0, column=0, padx=30, pady=30, sticky="nsew")
        self.run_badge.configure(
            text="NO COMPLETED RUN",
            fg_color=COLORS["surface_alt"],
            text_color=COLORS["muted"],
        )
        self.saved_model_book_id = None
        self.saved_model_book_name = None
        self.save_model_button.configure(
            state="disabled",
            text="Create Model Book  →",
        )
        self.footer_status_label.configure(
            text="A completed run is required before saving a model"
        )

    def _ask_model_book_name(self) -> str | None:
        if self.result is None:
            return None
        dialog = ctk.CTkInputDialog(
            title="Create Model Book",
            text=(
                f"Name the reusable Model Book created from "
                f"{self.result.run_id}."
            ),
        )
        return dialog.get_input()

    def _find_saved_model_book(self) -> ModelBook | None:
        self.saved_model_book_id = None
        self.saved_model_book_name = None
        if self.project is None or self.result is None:
            return None
        try:
            library = load_model_library(self.project.path)
        except ModelBookError:
            return None
        matching = [
            entry.book
            for entry in library.entries
            if entry.is_valid
            and entry.book is not None
            and entry.book.source_run_id == self.result.run_id
        ]
        if not matching:
            return None
        book = matching[-1]
        self.saved_model_book_id = book.book_id
        self.saved_model_book_name = book.name
        return book

    def _configure_primary_model_action(self) -> None:
        if self.result is None:
            self.save_model_button.configure(
                state="disabled",
                text="Create Model Book  →",
            )
            return
        if self.saved_model_book_id:
            self.save_model_button.configure(
                state="normal",
                text="Open Model Library  →",
            )
            self.footer_status_label.configure(
                text=f"Saved Model Book · {self.saved_model_book_name}"
            )
            return
        self.save_model_button.configure(
            state="normal",
            text="Create Model Book  →",
        )
        self.footer_status_label.configure(
            text="Saved run artifacts · Read-only results"
        )

    def _primary_model_action(self) -> None:
        if self.saved_model_book_id:
            self._open_saved_model_book()
        else:
            self._save_as_model()

    def _open_saved_model_book(self) -> None:
        if self.project is None or not self.saved_model_book_id:
            return
        self.app.library_page.selected_book_id = self.saved_model_book_id
        self.app.show_page("library")

    def _save_as_model(self) -> None:
        if self.project is None or self.result is None:
            return
        name = self._ask_model_book_name()
        if name is None:
            return
        if not name.strip():
            messagebox.showwarning(
                "Model Book name required",
                "Enter a name before saving this model.",
                parent=self,
            )
            return

        self.save_model_button.configure(state="disabled", text="Saving…")
        self.footer_status_label.configure(text="Creating Model Book…")
        self.update_idletasks()
        try:
            book = save_model_book(
                self.project.path,
                self.result.run_id,
                name,
            )
        except ModelBookError as exc:
            self.footer_status_label.configure(text="Model Book was not saved")
            messagebox.showerror(
                "Could not save Model Book",
                str(exc),
                parent=self,
            )
        else:
            self.project = self.app.update_current_project({})
            self.saved_model_book_id = book.book_id
            self.saved_model_book_name = book.name
            self.footer_status_label.configure(
                text=f"Saved Model Book · {book.name}"
            )
            messagebox.showinfo(
                "Model Book saved",
                (
                    f"{book.name} was saved as {book.book_id}.\n\n"
                    f"The original {self.result.run_id} training run was not changed.\n"
                    "Open Model Library when you are ready to review and activate it."
                ),
                parent=self,
            )
        finally:
            self._configure_primary_model_action()

    def _render_recommendation(self) -> None:
        assert self.result is not None
        result = self.result
        parameters = result.parameters_used
        if result.model_name == "ensemble_ai_engine":
            weights = result.ensemble_weights
            configuration = "  ·  ".join(
                f"{_model_display_name(name)} {weight:.1%}"
                for name, weight in weights.items()
            )
            recommendation_label = (
                "ENSEMBLE RECOMMENDED"
                if result.ensemble_improved_on_best
                else "ENSEMBLE EVALUATED"
            )
        elif result.model_name == "neural_network":
            configuration = (
                f"layers {parameters['hidden_layer_sizes']}  ·  "
                f"{parameters['activation']}  ·  "
                f"learning rate {parameters['learning_rate_init']}"
            )
            recommendation_label = (
                "AUTO BEST" if result.training_mode == "auto" else "CUSTOM USED"
            )
        elif result.model_name == "xgboost":
            configuration = (
                f"{parameters['n_estimators']} trees · "
                f"depth {parameters['max_depth']} · "
                f"learning rate {parameters['learning_rate']}"
            )
            recommendation_label = (
                "AUTO BEST"
                if result.training_mode == "auto" and result.search_level
                else (
                    "FIXED BASELINE"
                    if result.training_mode == "auto"
                    else "CUSTOM USED"
                )
            )
        else:
            configuration = (
                f"fit_intercept={parameters['fit_intercept']}  ·  "
                f"positive={parameters['positive']}"
            )
            recommendation_label = (
                "AUTO BEST" if result.training_mode == "auto" else "CUSTOM USED"
            )
        self.recommendation_title.configure(text=recommendation_label)
        self.recommendation_body.configure(text=configuration)

    def show_section(self, section: str) -> None:
        if section not in dict(RESULT_SECTIONS):
            return
        self.active_section = section
        for key, button in self.section_buttons.items():
            button.configure(
                fg_color=COLORS["nav_active"] if key == section else "transparent"
            )
        if self.result is None:
            return
        for child in self.content_card.winfo_children():
            child.grid_forget()
            if child is not self.empty_label:
                child.destroy()
        self.current_chart = None
        renderers: dict[str, Callable[[], None]] = {
            "fit": self._render_predictions_plot,
            "residuals": self._render_residual_plot,
            "errors": self._render_error_distribution,
            "configuration": self._render_configuration,
            "comparison": self._render_model_comparison,
            "run": self._render_run_info,
        }
        renderers[section]()

    def _content_shell(self, title: str, description: str = "") -> ctk.CTkFrame:
        shell = ctk.CTkFrame(self.content_card, fg_color="transparent")
        shell.grid(row=0, column=0, padx=16, pady=12, sticky="nsew")
        shell.grid_columnconfigure(0, weight=1)
        shell.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(
            shell,
            text=title,
            text_color=COLORS["ink"],
            font=FONTS["card_title"],
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")
        if description:
            ctk.CTkLabel(
                shell,
                text=description,
                text_color=COLORS["muted"],
                font=FONTS["caption"],
                anchor="e",
                justify="right",
                wraplength=430,
            ).grid(row=0, column=1, padx=(12, 0), sticky="e")
        return shell

    def _render_chart_section(self, title: str, description: str, mode: str) -> None:
        assert self.result is not None
        shell = self._content_shell(title, description)
        shell.grid_columnconfigure(1, weight=0)
        self.current_chart = ResultsChart(shell, self.result, mode)
        self.current_chart.grid(row=1, column=0, columnspan=2, pady=(8, 0), sticky="nsew")

    def _render_predictions_plot(self) -> None:
        assert self.result is not None
        sample_ids = self.result.prediction_sample_ids
        selected_sample = self.curve_sample.get()
        if selected_sample not in sample_ids:
            selected_sample = sample_ids[0]
            self.curve_sample.set(selected_sample)
        selected_predictions = self.result.predictions_for_sample(selected_sample)
        if len(self.curve_axis_values) != len(selected_predictions):
            _, self.curve_axis_values = infer_curve_axis(
                tuple(
                    prediction.target_name
                    for prediction in selected_predictions
                )
            )

        shell = self._content_shell("Actual vs Predicted Response")
        shell.grid_rowconfigure(1, weight=0)
        shell.grid_rowconfigure(2, weight=1)
        self.open_predictions_button = ctk.CTkButton(
            shell,
            text="Open Test Data CSV",
            width=136,
            height=28,
            corner_radius=8,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=("Segoe UI Semibold", 12),
            command=self._open_prediction_file,
        )
        self.open_predictions_button.grid(
            row=0,
            column=1,
            padx=(12, 0),
            sticky="e",
        )

        input_values = self.result.sample_input_values.get(selected_sample, {})
        input_text = "INPUT VALUES"
        if input_values:
            input_text += "   " + "   ·   ".join(
                f"{name} = {value:.6g}"
                for name, value in input_values.items()
            )
        else:
            input_text += "   Unavailable for this saved sample"
        self.input_values_label = ctk.CTkLabel(
            shell,
            text=input_text,
            height=30,
            corner_radius=9,
            fg_color=COLORS["primary_soft"],
            text_color=COLORS["ink"],
            font=FONTS["caption"],
            anchor="w",
            justify="left",
            wraplength=680,
        )
        self.input_values_label.grid(
            row=1,
            column=0,
            columnspan=2,
            pady=(8, 6),
            ipadx=10,
            sticky="ew",
        )

        self.prediction_plot_state.clear_curves()
        actual_curve = self.prediction_plot_state.add_curve(
            x_values=self.curve_axis_values,
            y_values=(row.actual_value for row in selected_predictions),
            target_names=(row.target_name for row in selected_predictions),
            inputs=input_values,
            name="Actual",
        )
        actual_curve.color_index = 0
        actual_curve.line_width = 2.6
        actual_curve.marker_style = "Circle"
        predicted_curve = self.prediction_plot_state.add_curve(
            x_values=self.curve_axis_values,
            y_values=(row.predicted_value for row in selected_predictions),
            target_names=(row.target_name for row in selected_predictions),
            inputs=input_values,
            name="Predicted",
        )
        predicted_curve.color_index = 1
        predicted_curve.line_width = 2.6
        predicted_curve.line_style = "Dashed"
        predicted_curve.marker_style = "Diamond"
        if not self.prediction_plot_state.axis_limits_user_defined:
            self.prediction_plot_state.autoscale()
        self.current_chart = ScientificPlotWorkbench(
            shell,
            state=self.prediction_plot_state,
            manager_header_builder=self._build_prediction_sample_selector,
        )
        self.current_chart.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="nsew",
        )

    def _build_prediction_sample_selector(self, parent: ctk.CTkFrame) -> None:
        assert self.result is not None
        ctk.CTkLabel(
            parent,
            text="TEST SAMPLE",
            text_color=COLORS["cyan"],
            font=("Cascadia Mono", 11),
            anchor="w",
        ).grid(row=0, column=0, pady=(0, 3), sticky="ew")
        self.curve_sample_menu = ctk.CTkOptionMenu(
            parent,
            variable=self.curve_sample,
            values=list(self.result.prediction_sample_ids),
            height=28,
            corner_radius=8,
            fg_color=COLORS["control"],
            button_color=COLORS["surface_elevated"],
            button_hover_color=COLORS["control_hover"],
            dropdown_fg_color=COLORS["surface"],
            dropdown_hover_color=COLORS["nav_active"],
            text_color=COLORS["ink"],
            command=self._curve_sample_changed,
        )
        self.curve_sample_menu.grid(row=1, column=0, sticky="ew")

    def _curve_sample_changed(self, sample_id: str) -> None:
        self.curve_sample.set(sample_id)
        if self.active_section == "fit":
            self.show_section("fit")

    def _render_residual_plot(self) -> None:
        assert self.result is not None
        residual_summary = self.result.residual_interpretation
        if len(self.result.predictions) > MAX_SCATTER_MARKERS:
            residual_summary += (
                f" Displaying {MAX_SCATTER_MARKERS:,} representative markers from "
                f"{len(self.result.predictions):,}; metrics and artifacts retain all values."
            )
        shell = self._content_shell(
            "Residual analysis",
            residual_summary,
        )
        shell.grid_columnconfigure(1, weight=0)
        self.residual_plot_state.clear_curves()
        ordered = tuple(
            sorted(
                enumerate(self.result.predictions),
                key=lambda item: (item[1].predicted_value, item[0]),
            )
        )
        residual_curve = self.residual_plot_state.add_curve(
            x_values=(item.predicted_value for _, item in ordered),
            y_values=(item.residual for _, item in ordered),
            target_names=(
                f"{item.sample_id} · {item.target_name}" for _, item in ordered
            ),
            inputs={},
            name="Residual",
        )
        residual_curve.line_style = "None"
        residual_curve.marker_style = "Circle"
        residual_curve.marker_size = 4.0
        zero_reference = self.residual_plot_state.add_curve(
            x_values=(
                min(item.predicted_value for _, item in ordered),
                max(item.predicted_value for _, item in ordered),
            ),
            y_values=(0.0, 0.0),
            target_names=("Zero error", "Zero error"),
            inputs={},
            name="Zero error",
        )
        zero_reference.color_index = 2
        zero_reference.line_style = "Dashed"
        zero_reference.marker_style = "None"
        if not self.residual_plot_state.axis_limits_user_defined:
            self.residual_plot_state.autoscale()
        self.current_chart = ScientificPlotWorkbench(
            shell,
            state=self.residual_plot_state,
        )
        self.current_chart.grid(
            row=1,
            column=0,
            columnspan=2,
            pady=(8, 0),
            sticky="nsew",
        )

    def _render_error_distribution(self) -> None:
        assert self.result is not None
        largest = self.result.largest_error_prediction
        summary = (
            f"Median absolute error: {self.result.median_absolute_error:.6g}  ·  "
            f"Maximum: {largest.absolute_error:.6g}  ·  "
            f"Largest: {largest.sample_id} (actual {largest.actual_value:.6g}, "
            f"predicted {largest.predicted_value:.6g})"
        )
        shell = self._content_shell("Absolute-error distribution", summary)
        self.current_chart = ResultsChart(shell, self.result, "histogram")
        self.current_chart.grid(row=1, column=0, columnspan=2, pady=(8, 0), sticky="nsew")

    def _render_configuration(self) -> None:
        assert self.result is not None
        if self.result.model_name == "ensemble_ai_engine":
            self._render_ensemble_configuration()
        elif self.result.training_mode == "auto" and self.result.search_level:
            self._render_auto_comparison()
        elif self.result.model_name in {"xgboost", "neural_network"}:
            self._render_fixed_configuration()
        else:
            self._render_custom_comparison()

    def _render_ensemble_configuration(self) -> None:
        assert self.result is not None
        best_name = _model_display_name(self.result.best_individual_model or "")
        ensemble_score = self.result.validation_rmse
        best_score = self.result.best_individual_validation_rmse
        recommendation = (
            "Ensemble AI Engine has lower validation RMSE and is recommended."
            if self.result.ensemble_improved_on_best
            else f"{best_name} retains the lower validation RMSE and remains recommended."
        )
        shell = self._content_shell(
            "Ensemble composition",
            (
                "Weights use inverse component validation RMSE only. "
                f"{recommendation}"
            ),
        )
        table = ctk.CTkFrame(shell, fg_color="transparent")
        table.grid(row=1, column=0, columnspan=2, pady=(10, 0), sticky="nsew")
        headers = ("Component", "Validation RMSE", "Weight", "Source run", "Status")
        for column, header in enumerate(headers):
            table.grid_columnconfigure(column, weight=1)
            ctk.CTkLabel(
                table,
                text=header,
                text_color=COLORS["cyan"],
                font=("Segoe UI Semibold", 12),
            ).grid(row=0, column=column, padx=4, pady=4, sticky="ew")
        row_number = 1
        for component in self.result.ensemble_components:
            values = (
                _model_display_name(component["model_name"]),
                f"{component['validation_rmse']:.6g}",
                f"{component['weight']:.2%}",
                str(component["source_run_id"]),
                "Completed",
            )
            for column, value in enumerate(values):
                ctk.CTkLabel(
                    table,
                    text=value,
                    height=28,
                    corner_radius=7,
                    fg_color=COLORS["surface_alt"],
                    text_color=COLORS["ink"],
                    font=("Segoe UI", 12),
                ).grid(row=row_number, column=column, padx=2, pady=2, sticky="nsew")
            row_number += 1
        for failure in self.result.ensemble_failures:
            values = (
                _model_display_name(failure["model_name"]),
                "—",
                "—",
                "—",
                f"Failed: {failure['error_message']}",
            )
            for column, value in enumerate(values):
                ctk.CTkLabel(
                    table,
                    text=value,
                    height=28,
                    corner_radius=7,
                    fg_color=COLORS["surface_alt"],
                    text_color=COLORS["danger"] if column == 4 else COLORS["muted"],
                    font=("Segoe UI", 12),
                    wraplength=220 if column == 4 else 130,
                ).grid(row=row_number, column=column, padx=2, pady=2, sticky="nsew")
            row_number += 1
        ctk.CTkLabel(
            table,
            text=(
                f"Ensemble validation RMSE: {ensemble_score:.6g}  ·  "
                f"Best individual ({best_name}): {best_score:.6g}"
            ),
            text_color=(
                COLORS["success"]
                if self.result.ensemble_improved_on_best
                else COLORS["ink"]
            ),
            font=FONTS["body_small"],
        ).grid(row=row_number, column=0, columnspan=5, padx=4, pady=(10, 0), sticky="w")

    def _render_fixed_configuration(self) -> None:
        assert self.result is not None
        is_custom = self.result.training_mode == "custom"
        model_label = _model_display_name(self.result.model_name)
        shell = self._content_shell(
            (
                f"Custom {model_label} configuration"
                if is_custom
                else f"Fixed {model_label} baseline"
            ),
            (
                "These are the exact estimator parameters saved for this Custom run."
                if is_custom
                else (
                    "This run used the documented deterministic baseline; "
                    "no tuning was performed."
                )
            ),
        )
        grid = ctk.CTkFrame(shell, fg_color="transparent")
        grid.grid(row=1, column=0, columnspan=2, pady=(10, 0), sticky="nsew")
        grid.grid_columnconfigure(1, weight=1)
        for row, (name, value) in enumerate(self.result.parameters_used.items()):
            ctk.CTkLabel(
                grid,
                text=name,
                text_color=COLORS["cyan"],
                font=FONTS["mono"],
                anchor="w",
            ).grid(row=row, column=0, padx=(0, 18), pady=3, sticky="w")
            ctk.CTkLabel(
                grid,
                text=str(value),
                text_color=COLORS["ink"],
                font=FONTS["body_small"],
                anchor="w",
            ).grid(row=row, column=1, pady=3, sticky="w")

    def _render_auto_comparison(self) -> None:
        assert self.result is not None
        validation = (
            f"{self.result.validation_rmse:.6g}"
            if self.result.validation_rmse is not None
            else "unavailable"
        )
        shell = self._content_shell(
            "Auto-search comparison",
            (
                f"{self.result.search_level.title()} · "
                f"{self.result.configurations_evaluated} configurations · "
                f"{self.result.cross_validation_folds} folds · "
                f"lowest validation RMSE {validation}"
            ),
        )
        table = ctk.CTkFrame(shell, fg_color="transparent")
        table.grid(row=1, column=0, columnspan=2, pady=(10, 0), sticky="nsew")
        if self.result.model_name in {"xgboost", "neural_network"}:
            headers = (
                "Configuration",
                f"{_model_display_name(self.result.model_name)} parameters",
                "Mean validation RMSE",
                "Status",
                "Selected",
            )
        else:
            headers = (
                "Configuration",
                "fit_intercept",
                "positive",
                "Mean validation RMSE",
                "Status",
                "Selected",
            )
        for column, header in enumerate(headers):
            table.grid_columnconfigure(column, weight=1)
            ctk.CTkLabel(
                table,
                text=header,
                text_color=COLORS["cyan"],
                font=("Segoe UI Semibold", 12),
            ).grid(row=0, column=column, padx=4, pady=4, sticky="ew")
        for row, candidate in enumerate(self.result.auto_candidates, start=1):
            status = "Completed" if candidate.success else "Failed"
            score = (
                f"{candidate.mean_validation_rmse:.6g}"
                if candidate.mean_validation_rmse is not None
                else candidate.error_message or "Failed"
            )
            if self.result.model_name == "neural_network":
                parameters = candidate.parameters
                parameter_summary = (
                    f"layers={parameters['hidden_layer_sizes']} · "
                    f"{parameters['activation']} · "
                    f"lr={parameters['learning_rate_init']} · "
                    f"batch={parameters['batch_size']} · "
                    f"epochs={parameters['max_iter']}"
                )
                values = (
                    f"Config {row}",
                    parameter_summary,
                    score,
                    status,
                    "✓ Selected" if candidate.selected else "—",
                )
            elif self.result.model_name == "xgboost":
                parameters = candidate.parameters
                parameter_summary = (
                    f"n={parameters['n_estimators']} · "
                    f"depth={parameters['max_depth']} · "
                    f"lr={parameters['learning_rate']} · "
                    f"rows={parameters['subsample']} · "
                    f"columns={parameters['colsample_bytree']}"
                )
                values = (
                    f"Config {row}",
                    parameter_summary,
                    score,
                    status,
                    "✓ Selected" if candidate.selected else "—",
                )
            else:
                values = (
                    f"Config {row}",
                    str(candidate.parameters["fit_intercept"]),
                    str(candidate.parameters["positive"]),
                    score,
                    status,
                    "✓ Selected" if candidate.selected else "—",
                )
            for column, value in enumerate(values):
                ctk.CTkLabel(
                    table,
                    text=value,
                    height=28,
                    corner_radius=7,
                    fg_color=(
                        COLORS["success_soft"]
                        if candidate.selected
                        else COLORS["surface_alt"]
                    ),
                    text_color=(
                        COLORS["danger"]
                        if not candidate.success
                        else COLORS["ink"]
                    ),
                    font=("Segoe UI", 12),
                    wraplength=(
                        260
                        if self.result.model_name in {"xgboost", "neural_network"}
                        else 150
                    ),
                ).grid(row=row, column=column, padx=2, pady=2, sticky="nsew")

    def _render_custom_comparison(self) -> None:
        assert self.result is not None
        comparison = self.result.custom_recommendation
        shell = self._content_shell(
            "Custom recommendation comparison",
            "Recommendations use compatible validation evidence only.",
        )
        if comparison is None:
            ctk.CTkLabel(
                shell,
                text=self.result.custom_guidance,
                text_color=COLORS["muted"],
                font=FONTS["body"],
                justify="center",
                wraplength=600,
            ).grid(row=1, column=0, columnspan=2, padx=20, pady=40)
            return
        comparison_frame = ctk.CTkFrame(shell, fg_color="transparent")
        comparison_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0), sticky="nsew")
        comparison_frame.grid_columnconfigure((0, 1), weight=1)
        custom_text = self._comparison_text(
            "Your Custom Configuration",
            self.result.parameters_used,
            comparison.custom_validation_rmse,
            self.result.metrics,
        )
        auto_text = self._comparison_text(
            "Suggested Auto Configuration",
            comparison.suggested_parameters,
            comparison.suggested_validation_rmse,
            comparison.auto_test_metrics,
        )
        for column, text in enumerate((custom_text, auto_text)):
            ctk.CTkLabel(
                comparison_frame,
                text=text,
                fg_color=COLORS["surface_alt"],
                corner_radius=12,
                text_color=COLORS["ink"],
                font=FONTS["body_small"],
                justify="left",
                anchor="nw",
            ).grid(row=0, column=column, padx=(0, 5) if column == 0 else (5, 0), pady=4, ipadx=14, ipady=12, sticky="nsew")
        ctk.CTkLabel(
            comparison_frame,
            text=f"{comparison.recommendation}\n{comparison.explanation}",
            text_color=COLORS["success"],
            font=FONTS["body_small"],
            wraplength=650,
            justify="center",
        ).grid(row=1, column=0, columnspan=2, pady=(10, 0))

    @staticmethod
    def _comparison_text(
        title: str,
        parameters: dict[str, bool],
        validation_rmse: float | None,
        metrics: dict[str, float],
    ) -> str:
        validation = (
            f"{validation_rmse:.6g}" if validation_rmse is not None else "Not available"
        )
        return (
            f"{title}\n\n"
            f"fit_intercept: {parameters['fit_intercept']}\n"
            f"positive: {parameters['positive']}\n\n"
            f"Validation RMSE: {validation}\n"
            f"Test RMSE: {metrics['RMSE']:.6g}\n"
            f"Test MAE: {metrics['MAE']:.6g}\n"
            f"Test R²: {metrics['R²']:.6g}"
        )

    def _open_prediction_file(self) -> None:
        if self.result is None or self.result.predictions_path is None:
            return
        path = self.result.predictions_path
        if not path.is_file():
            messagebox.showwarning(
                "Prediction file unavailable",
                "test_predictions.csv is missing from this run.",
                parent=self,
            )
            return
        try:
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as exc:
            messagebox.showerror(
                "Could not open predictions",
                str(exc),
                parent=self,
            )

    def _render_model_comparison(self) -> None:
        assert self.result is not None
        shell = self._content_shell(
            "MODEL FAMILY COMPARISON",
            (
                "Validation-backed family recommendation; separate from the "
                "displayed-run metrics above."
            ),
        )
        shell.grid_rowconfigure(1, weight=0)
        shell.grid_rowconfigure(3, weight=1)
        self.model_comparison = None
        self.comparison_error = None
        self.comparison_metric_chart = None
        self.comparison_run_buttons = {}
        try:
            comparison = compare_compatible_model_runs(
                self.result.project_path,
                anchor_run_id=self.result.run_id,
            )
        except ModelComparisonError as exc:
            self.comparison_error = str(exc)
            ctk.CTkLabel(
                shell,
                text=(
                    "Model comparison could not be loaded.\n"
                    f"{self.comparison_error}"
                ),
                text_color=COLORS["danger"],
                font=FONTS["body"],
                justify="center",
                wraplength=680,
            ).grid(row=1, column=0, columnspan=2, padx=20, pady=48)
            return

        self.model_comparison = comparison
        recommendation = ctk.CTkFrame(
            shell,
            height=50,
            fg_color=(
                COLORS["success_soft"]
                if comparison.recommended_model
                else COLORS["warning_soft"]
            ),
            corner_radius=10,
            border_width=1,
            border_color=COLORS["border_strong"],
        )
        recommendation.grid(
            row=1,
            column=0,
            columnspan=2,
            pady=(8, 6),
            sticky="ew",
        )
        recommendation.grid_propagate(False)
        recommendation.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            recommendation,
            text=comparison.recommendation_title,
            text_color=(
                COLORS["success"]
                if comparison.recommended_model
                else COLORS["warning"]
            ),
            font=FONTS["body_small"],
            anchor="w",
        ).grid(row=0, column=0, padx=(12, 14), pady=6, sticky="w")
        ctk.CTkLabel(
            recommendation,
            text=comparison.recommendation_reason,
            text_color=COLORS["ink"],
            font=("Segoe UI", 11),
            anchor="w",
            justify="left",
            wraplength=650,
        ).grid(row=0, column=1, padx=(0, 12), pady=5, sticky="ew")

        cards = ctk.CTkFrame(shell, fg_color="transparent")
        cards.grid(row=2, column=0, columnspan=2, pady=(0, 4), sticky="ew")
        cards.grid_columnconfigure(
            tuple(range(len(MODEL_FAMILY_ORDER))),
            weight=1,
            uniform="comparison_family",
        )
        for column, model_name in enumerate(MODEL_FAMILY_ORDER):
            family = comparison.family(model_name)
            self._render_model_family_card(
                cards,
                column,
                family,
                selected=comparison.recommended_model == model_name,
            )

        self.comparison_metric_chart = ModelComparisonMetricChart(shell, comparison)
        self.comparison_metric_chart.grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="nsew",
        )

    def _render_model_family_card(
        self,
        parent: ctk.CTkFrame,
        column: int,
        family: Any,
        *,
        selected: bool,
    ) -> None:
        card = ctk.CTkFrame(
            parent,
            fg_color=COLORS["surface_alt"],
            corner_radius=10,
            border_width=2 if selected else 1,
            border_color=COLORS["success"] if selected else COLORS["border"],
        )
        card.grid(
            row=0,
            column=column,
            padx=(
                (0, 3)
                if column == 0
                else ((3, 3) if column < len(MODEL_FAMILY_ORDER) - 1 else (3, 0))
            ),
            sticky="nsew",
        )
        card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            card,
            text=(
                f"{family.display_name}  ·  RECOMMENDED"
                if selected
                else family.display_name
            ),
            text_color=COLORS["success"] if selected else COLORS["ink"],
            font=FONTS["card_title"],
            anchor="w",
        ).grid(row=0, column=0, padx=12, pady=(5, 0), sticky="ew")
        run = family.best_run
        if run is None:
            if family.compatible_run_count:
                message = (
                    f"{family.compatible_run_count} compatible run(s), but none "
                    "contains valid validation RMSE.\nTest metrics are not used "
                    "as a substitute."
                )
            else:
                message = "No compatible completed run is available."
            ctk.CTkLabel(
                card,
                text=message,
                text_color=COLORS["muted"],
                font=("Segoe UI", 12),
                justify="left",
                anchor="nw",
                wraplength=390,
            ).grid(row=1, column=0, padx=12, pady=(4, 12), sticky="nsew")
            return

        mode = run.training_mode.title()
        if run.search_level:
            mode = f"{mode} · {run.search_level.title()}"
        ctk.CTkLabel(
            card,
            text=(
                f"Run {run.run_number}  ·  {mode}\n"
                f"{_comparison_parameter_lines(run.parameters_used)}"
            ),
            text_color=COLORS["muted"],
            font=("Segoe UI", 11),
            justify="left",
            anchor="nw",
            wraplength=390,
        ).grid(row=1, column=0, padx=12, pady=(0, 1), sticky="ew")
        ctk.CTkLabel(
            card,
            text=(
                f"Validation RMSE {run.validation_rmse:.6g}   ·   "
                f"Test RMSE {run.test_rmse:.6g}\n"
                f"MAE {run.mae:.6g}   ·   R² {run.r_squared:.6g}"
            ),
            text_color=COLORS["ink"],
            font=("Cascadia Mono", 10),
            justify="left",
            anchor="w",
        ).grid(row=2, column=0, padx=12, pady=(0, 2), sticky="ew")
        button = ctk.CTkButton(
            card,
            text=f"Open Run {run.run_number} Results",
            height=24,
            corner_radius=8,
            fg_color="transparent",
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border_strong"],
            text_color=COLORS["primary"],
            font=("Segoe UI Semibold", 12),
            command=lambda selected_run=run.run_id: self._open_comparison_run(
                selected_run
            ),
        )
        button.grid(row=3, column=0, padx=12, pady=(0, 5), sticky="w")
        self.comparison_run_buttons[family.model_name] = button

    def _open_comparison_run(self, run_id: str) -> None:
        self.requested_run_id = run_id
        self.active_section = DEFAULT_RESULTS_SECTION
        self.failure_state = None
        self.reload()

    def _render_run_info(self) -> None:
        assert self.result is not None
        result = self.result
        shell = self._content_shell(
            "Run information",
            "Secondary provenance for this immutable result.",
        )
        details = (
            ("Run ID", result.run_id),
            ("Model", _model_display_name(result.model_name)),
            ("Training mode", result.training_mode.title()),
            ("Search level", result.search_level.title() if result.search_level else "Not applicable"),
            (
                "Parameters used",
                _format_parameters(result.parameters_used),
            ),
            ("Training samples", str(result.training_rows)),
            ("Test samples", str(result.test_rows)),
            ("Dataset fingerprint", result.dataset_fingerprint),
            ("Training timestamp", result.trained_at or "Unavailable"),
        )
        grid = ctk.CTkFrame(shell, fg_color="transparent")
        grid.grid(row=1, column=0, columnspan=2, pady=(10, 0), sticky="nsew")
        grid.grid_columnconfigure(1, weight=1)
        for row, (label, value) in enumerate(details):
            ctk.CTkLabel(
                grid,
                text=label,
                text_color=COLORS["cyan"],
                font=FONTS["mono"],
                anchor="w",
            ).grid(row=row, column=0, padx=(0, 14), pady=2, sticky="w")
            ctk.CTkLabel(
                grid,
                text=value,
                text_color=COLORS["ink"],
                font=FONTS["body_small"],
                anchor="w",
                wraplength=560,
            ).grid(row=row, column=1, pady=2, sticky="ew")


def _numeric_range(values: list[float]) -> tuple[float, float]:
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum
    padding = span * 0.08 if span > 0 else max(abs(maximum) * 0.08, 1.0)
    return minimum - padding, maximum + padding
