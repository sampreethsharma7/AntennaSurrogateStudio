"""Reusable, dependency-light scientific curve plotting workbench."""

from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass, field
from typing import Callable, Iterable

import customtkinter as ctk

from studio.theme import COLORS, FONTS


CURVES_PER_PAGE = 5
PLOT_PANE_MIN_WIDTH = 320
PLOT_CANVAS_HORIZONTAL_INSET = 8
CURVE_MANAGER_MIN_WIDTH = 220
CURVE_MANAGER_DEFAULT_WIDTH = 250
AXIS_SCALES = ("Linear", "Log")
LEGEND_LOCATIONS = (
    "Upper right",
    "Upper left",
    "Lower right",
    "Lower left",
)
LINE_STYLES = ("Solid", "Dashed", "Dotted", "None")
MARKER_STYLES = ("Circle", "Square", "Diamond", "None")
LEGEND_POSITIONS = {
    "Upper right": (0.98, 0.04),
    "Upper left": (0.28, 0.04),
    "Lower right": (0.98, 0.72),
    "Lower left": (0.28, 0.72),
}
LIGHT_CURVE_COLORS = (
    "#087D8E",
    "#6650C8",
    "#D2691E",
    "#16734A",
    "#C33F3F",
    "#1F62A5",
    "#A43A78",
    "#6C6F16",
)
DARK_CURVE_COLORS = (
    "#35D6E7",
    "#9B86F2",
    "#FF9D57",
    "#39D98A",
    "#FF7373",
    "#6CB6FF",
    "#E77DB3",
    "#D6DA59",
)


def _palette(color: tuple[str, str]) -> str:
    return color[1] if ctk.get_appearance_mode() == "Dark" else color[0]


def curve_color(index: int) -> str:
    palette = DARK_CURVE_COLORS if ctk.get_appearance_mode() == "Dark" else LIGHT_CURVE_COLORS
    return palette[index % len(palette)]


def engineering_tick(value: float) -> str:
    """Format a numeric tick using conventional engineering prefixes."""

    if not math.isfinite(value):
        return "—"
    if math.isclose(value, 0.0, abs_tol=1e-15):
        return "0"
    exponent = int(math.floor(math.log10(abs(value)) / 3) * 3)
    exponent = max(-12, min(12, exponent))
    prefixes = {
        -12: "p",
        -9: "n",
        -6: "µ",
        -3: "m",
        0: "",
        3: "k",
        6: "M",
        9: "G",
        12: "T",
    }
    scaled = value / (10.0**exponent)
    return f"{scaled:.4g}{prefixes[exponent]}"


def adaptive_major_interval_count(plot_span: float, font_size: float) -> int:
    """Return a compact-safe count of major X-axis intervals.

    Engineering labels can be several characters wide. Keep enough horizontal
    separation for those labels while retaining the full six-label grid on a
    normally sized scientific canvas.
    """

    minimum_spacing = max(64.0, float(font_size) * 5.5)
    return max(2, min(5, int(max(0.0, plot_span) // minimum_spacing)))


def _scale_value(value: float, scale: str) -> float:
    if scale == "Log":
        if value <= 0:
            raise ValueError("Log axes require positive values.")
        return math.log10(value)
    return value


def _unscale_value(value: float, scale: str) -> float:
    return 10.0**value if scale == "Log" else value


def _scale_pair(minimum: float, maximum: float, scale: str) -> tuple[float, float]:
    return _scale_value(minimum, scale), _scale_value(maximum, scale)


@dataclass(slots=True)
class ScientificCurve:
    """One plotted response plus the input sample that produced it."""

    curve_id: str
    name: str
    x_values: tuple[float, ...]
    y_values: tuple[float, ...]
    target_names: tuple[str, ...]
    inputs: dict[str, float] = field(default_factory=dict)
    visible: bool = True
    color_index: int = 0
    line_width: float = 2.0
    line_style: str = "Solid"
    marker_style: str = "Circle"
    marker_size: float = 3.0


@dataclass(slots=True)
class PlotAnnotation:
    annotation_id: str
    x: float
    y: float
    label: str
    curve_id: str | None = None


class ScientificPlotState:
    """Widget-independent state and deterministic plot interactions."""

    def __init__(self) -> None:
        self.curves: list[ScientificCurve] = []
        self.annotations: list[PlotAnnotation] = []
        self.selected_curve_id: str | None = None
        self.x_label = "Output coordinate"
        self.y_label = "Predicted value"
        self.plot_title = "Predicted Response"
        self.x_scale = "Linear"
        self.y_scale = "Linear"
        self.major_grid = True
        self.minor_grid = True
        self.legend_visible = True
        self.legend_location = "Upper right"
        self.legend_position = (0.98, 0.04)
        self.plot_title_font_size = 17.0
        self.x_label_font_size = 14.0
        self.y_label_font_size = 14.0
        self.x_value_font_size = 11.0
        self.y_value_font_size = 11.0
        self.legend_font_size = 11.0
        self.legend_line_width = 2.0
        self.view_limits = (0.0, 1.0, 0.0, 1.0)
        self.axis_labels_user_defined = False
        self.axis_limits_user_defined = False
        self._curve_number = 0
        self._annotation_number = 0

    @property
    def selected_curve(self) -> ScientificCurve | None:
        return self.curve(self.selected_curve_id)

    def curve(self, curve_id: str | None) -> ScientificCurve | None:
        return next(
            (candidate for candidate in self.curves if candidate.curve_id == curve_id),
            None,
        )

    def add_curve(
        self,
        *,
        x_values: Iterable[float],
        y_values: Iterable[float],
        target_names: Iterable[str],
        inputs: dict[str, float] | None = None,
        replace_selected: bool = False,
        name: str | None = None,
    ) -> ScientificCurve:
        xs = tuple(float(value) for value in x_values)
        ys = tuple(float(value) for value in y_values)
        targets = tuple(str(value) for value in target_names)
        if not xs or len(xs) != len(ys) or len(xs) != len(targets):
            raise ValueError("A curve requires matching non-empty X, Y, and target values.")
        if not all(math.isfinite(value) for value in (*xs, *ys)):
            raise ValueError("Plot curves require finite numeric values.")
        if self.x_scale == "Log" and any(value <= 0 for value in xs):
            raise ValueError(
                "This curve has nonpositive X values and cannot be added while X scale is Log."
            )
        if self.y_scale == "Log" and any(value <= 0 for value in ys):
            raise ValueError(
                "This curve has nonpositive Y values and cannot be added while Y scale is Log."
            )
        input_snapshot = {
            str(key): float(value) for key, value in (inputs or {}).items()
        }

        selected = self.selected_curve if replace_selected else None
        if selected is not None:
            selected.x_values = xs
            selected.y_values = ys
            selected.target_names = targets
            selected.inputs = input_snapshot
            selected.visible = True
            self.annotations = [
                marker
                for marker in self.annotations
                if marker.curve_id != selected.curve_id
            ]
            curve = selected
        else:
            self._curve_number += 1
            curve = ScientificCurve(
                curve_id=f"curve-{self._curve_number:04d}",
                name=(name or f"Prediction {self._curve_number}").strip(),
                x_values=xs,
                y_values=ys,
                target_names=targets,
                inputs=input_snapshot,
                color_index=self._curve_number - 1,
            )
            self.curves.append(curve)
            self.selected_curve_id = curve.curve_id
        if not self.axis_limits_user_defined:
            self.autoscale()
        return curve

    def select_curve(self, curve_id: str) -> ScientificCurve:
        curve = self.curve(curve_id)
        if curve is None:
            raise ValueError(f"Unknown plot curve: {curve_id}")
        self.selected_curve_id = curve_id
        return curve

    def rename_curve(self, curve_id: str, name: str) -> ScientificCurve:
        curve = self.select_curve(curve_id)
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("Curve name cannot be empty.")
        curve.name = clean_name
        return curve

    def set_curve_visible(self, curve_id: str, visible: bool) -> ScientificCurve:
        curve = self.select_curve(curve_id)
        curve.visible = bool(visible)
        if (
            any(candidate.visible for candidate in self.curves)
            and not self.axis_limits_user_defined
        ):
            self.autoscale()
        return curve

    def delete_curve(self, curve_id: str) -> None:
        index = next(
            (i for i, curve in enumerate(self.curves) if curve.curve_id == curve_id),
            None,
        )
        if index is None:
            raise ValueError(f"Unknown plot curve: {curve_id}")
        self.curves.pop(index)
        self.annotations = [
            marker for marker in self.annotations if marker.curve_id != curve_id
        ]
        if self.selected_curve_id == curve_id:
            if self.curves:
                self.selected_curve_id = self.curves[min(index, len(self.curves) - 1)].curve_id
            else:
                self.selected_curve_id = None
        if not self.axis_limits_user_defined:
            self.autoscale()

    def clear_curves(self) -> None:
        """Remove plotted data while preserving axes and visual settings."""

        self.curves.clear()
        self.annotations.clear()
        self.selected_curve_id = None
        self._curve_number = 0
        self._annotation_number = 0
        if not self.axis_limits_user_defined:
            self.autoscale()

    def add_annotation(
        self,
        x: float,
        y: float,
        *,
        label: str | None = None,
        curve_id: str | None = None,
    ) -> PlotAnnotation:
        x_value = float(x)
        y_value = float(y)
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            raise ValueError("Marker coordinates must be finite numbers.")
        if curve_id is not None and self.curve(curve_id) is None:
            raise ValueError(f"Unknown plot curve: {curve_id}")
        self._annotation_number += 1
        marker = PlotAnnotation(
            annotation_id=f"marker-{self._annotation_number:04d}",
            x=x_value,
            y=y_value,
            label=(label or f"M{self._annotation_number}").strip(),
            curve_id=curve_id,
        )
        self.annotations.append(marker)
        return marker

    def clear_annotations(self) -> None:
        self.annotations.clear()

    def data_bounds(self) -> tuple[float, float, float, float]:
        visible = [curve for curve in self.curves if curve.visible]
        if not visible:
            return (
                1.0 if self.x_scale == "Log" else 0.0,
                10.0 if self.x_scale == "Log" else 1.0,
                1.0 if self.y_scale == "Log" else 0.0,
                10.0 if self.y_scale == "Log" else 1.0,
            )
        xs = [value for curve in visible for value in curve.x_values]
        ys = [value for curve in visible for value in curve.y_values]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if self.x_scale == "Log":
            log_min, log_max = math.log10(x_min), math.log10(x_max)
            padding = max((log_max - log_min) * 0.04, 0.04)
            x_min, x_max = 10 ** (log_min - padding), 10 ** (log_max + padding)
        elif math.isclose(x_min, x_max):
            padding = max(abs(x_min) * 0.05, 1.0)
            x_min, x_max = x_min - padding, x_max + padding
        else:
            padding = (x_max - x_min) * 0.04
            x_min, x_max = x_min - padding, x_max + padding
        if self.y_scale == "Log":
            log_min, log_max = math.log10(y_min), math.log10(y_max)
            padding = max((log_max - log_min) * 0.08, 0.04)
            y_min, y_max = 10 ** (log_min - padding), 10 ** (log_max + padding)
        elif math.isclose(y_min, y_max):
            padding = max(abs(y_min) * 0.05, 1.0)
            y_min, y_max = y_min - padding, y_max + padding
        else:
            padding = (y_max - y_min) * 0.08
            y_min, y_max = y_min - padding, y_max + padding
        return (x_min, x_max, y_min, y_max)

    def autoscale(self) -> tuple[float, float, float, float]:
        self.view_limits = self.data_bounds()
        self.axis_limits_user_defined = False
        return self.view_limits

    def reset_view(self) -> tuple[float, float, float, float]:
        return self.autoscale()

    def set_limits(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        *,
        user_defined: bool = False,
    ) -> None:
        limits = tuple(float(value) for value in (x_min, x_max, y_min, y_max))
        if not all(math.isfinite(value) for value in limits):
            raise ValueError("Axis limits must be finite numeric values.")
        if limits[0] >= limits[1]:
            raise ValueError("X-axis minimum must be less than its maximum.")
        if limits[2] >= limits[3]:
            raise ValueError("Y-axis minimum must be less than its maximum.")
        if self.x_scale == "Log" and limits[0] <= 0:
            raise ValueError("Log X scale requires positive axis limits.")
        if self.y_scale == "Log" and limits[2] <= 0:
            raise ValueError("Log Y scale requires positive axis limits.")
        self.view_limits = limits
        if user_defined:
            self.axis_limits_user_defined = True

    def set_axis_labels(self, x_label: str, y_label: str) -> None:
        clean_x = str(x_label).strip()
        clean_y = str(y_label).strip()
        if not clean_x or not clean_y:
            raise ValueError("Both axis labels are required.")
        self.x_label = clean_x
        self.y_label = clean_y
        self.axis_labels_user_defined = True

    def configure_plot(
        self,
        *,
        plot_title: str,
        x_label: str,
        y_label: str,
        limits: tuple[float, float, float, float],
        x_scale: str,
        y_scale: str,
        major_grid: bool,
        minor_grid: bool,
        legend_visible: bool,
        legend_location: str,
        plot_title_font_size: float | None = None,
        x_label_font_size: float | None = None,
        y_label_font_size: float | None = None,
        x_value_font_size: float | None = None,
        y_value_font_size: float | None = None,
        legend_font_size: float | None = None,
        legend_line_width: float | None = None,
    ) -> None:
        """Validate and apply global plot presentation settings atomically."""

        clean_title = str(plot_title).strip()
        clean_x = str(x_label).strip()
        clean_y = str(y_label).strip()
        if not clean_title:
            raise ValueError("Plot title cannot be empty.")
        if not clean_x or not clean_y:
            raise ValueError("Both axis labels are required.")
        if x_scale not in AXIS_SCALES or y_scale not in AXIS_SCALES:
            raise ValueError("Axis scale must be Linear or Log.")
        if legend_location not in LEGEND_LOCATIONS:
            raise ValueError("Choose a supported legend location.")
        typography = {
            "Plot title font size": (
                self.plot_title_font_size
                if plot_title_font_size is None
                else float(plot_title_font_size),
                10.0,
                36.0,
            ),
            "X-label font size": (
                self.x_label_font_size
                if x_label_font_size is None
                else float(x_label_font_size),
                8.0,
                28.0,
            ),
            "Y-label font size": (
                self.y_label_font_size
                if y_label_font_size is None
                else float(y_label_font_size),
                8.0,
                28.0,
            ),
            "X-value font size": (
                self.x_value_font_size
                if x_value_font_size is None
                else float(x_value_font_size),
                7.0,
                24.0,
            ),
            "Y-value font size": (
                self.y_value_font_size
                if y_value_font_size is None
                else float(y_value_font_size),
                7.0,
                24.0,
            ),
            "Legend font size": (
                self.legend_font_size
                if legend_font_size is None
                else float(legend_font_size),
                7.0,
                24.0,
            ),
            "Legend line width": (
                self.legend_line_width
                if legend_line_width is None
                else float(legend_line_width),
                0.5,
                8.0,
            ),
        }
        for label, (value, minimum, maximum) in typography.items():
            if not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{label} must be between {minimum:g} and {maximum:g}.")
        resolved_limits = tuple(float(value) for value in limits)
        if not all(math.isfinite(value) for value in resolved_limits):
            raise ValueError("Axis limits must be finite numeric values.")
        if resolved_limits[0] >= resolved_limits[1]:
            raise ValueError("X-axis minimum must be less than its maximum.")
        if resolved_limits[2] >= resolved_limits[3]:
            raise ValueError("Y-axis minimum must be less than its maximum.")
        visible = [curve for curve in self.curves if curve.visible]
        if x_scale == "Log":
            if resolved_limits[0] <= 0 or any(
                value <= 0 for curve in visible for value in curve.x_values
            ):
                raise ValueError("Log X scale requires positive limits and curve values.")
        if y_scale == "Log":
            if resolved_limits[2] <= 0 or any(
                value <= 0 for curve in visible for value in curve.y_values
            ):
                raise ValueError("Log Y scale requires positive limits and curve values.")

        self.plot_title = clean_title
        self.x_label = clean_x
        self.y_label = clean_y
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.view_limits = resolved_limits
        self.major_grid = bool(major_grid)
        self.minor_grid = bool(minor_grid)
        self.legend_visible = bool(legend_visible)
        self.legend_location = legend_location
        self.legend_position = LEGEND_POSITIONS[legend_location]
        self.plot_title_font_size = typography["Plot title font size"][0]
        self.x_label_font_size = typography["X-label font size"][0]
        self.y_label_font_size = typography["Y-label font size"][0]
        self.x_value_font_size = typography["X-value font size"][0]
        self.y_value_font_size = typography["Y-value font size"][0]
        self.legend_font_size = typography["Legend font size"][0]
        self.legend_line_width = typography["Legend line width"][0]
        self.axis_labels_user_defined = True
        self.axis_limits_user_defined = True

    def configure_selected_curve(
        self,
        *,
        line_width: float,
        line_style: str,
        marker_style: str,
        marker_size: float,
    ) -> ScientificCurve:
        """Apply validated style settings to the selected curve."""

        curve = self.selected_curve
        if curve is None:
            raise ValueError("Select a curve before applying curve settings.")
        width = float(line_width)
        size = float(marker_size)
        if not math.isfinite(width) or not 0.5 <= width <= 8.0:
            raise ValueError("Curve line width must be between 0.5 and 8.")
        if line_style not in LINE_STYLES:
            raise ValueError("Choose a supported curve line style.")
        if marker_style not in MARKER_STYLES:
            raise ValueError("Choose a supported marker style.")
        if not math.isfinite(size) or not 1.0 <= size <= 12.0:
            raise ValueError("Marker size must be between 1 and 12.")
        curve.line_width = width
        curve.line_style = line_style
        curve.marker_style = marker_style
        curve.marker_size = size
        return curve

    def zoom(
        self,
        factor: float,
        *,
        center: tuple[float, float] | None = None,
    ) -> None:
        scale = float(factor)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Zoom factor must be a positive finite number.")
        x_min, x_max, y_min, y_max = self.view_limits
        tx_min, tx_max = _scale_pair(x_min, x_max, self.x_scale)
        ty_min, ty_max = _scale_pair(y_min, y_max, self.y_scale)
        if center is None:
            center_tx = (tx_min + tx_max) / 2
            center_ty = (ty_min + ty_max) / 2
        else:
            center_tx = _scale_value(center[0], self.x_scale)
            center_ty = _scale_value(center[1], self.y_scale)
        self.set_limits(
            _unscale_value(center_tx + (tx_min - center_tx) * scale, self.x_scale),
            _unscale_value(center_tx + (tx_max - center_tx) * scale, self.x_scale),
            _unscale_value(center_ty + (ty_min - center_ty) * scale, self.y_scale),
            _unscale_value(center_ty + (ty_max - center_ty) * scale, self.y_scale),
        )

    def pan(self, x_fraction: float, y_fraction: float) -> None:
        x_min, x_max, y_min, y_max = self.view_limits
        tx_min, tx_max = _scale_pair(x_min, x_max, self.x_scale)
        ty_min, ty_max = _scale_pair(y_min, y_max, self.y_scale)
        x_shift = (tx_max - tx_min) * float(x_fraction)
        y_shift = (ty_max - ty_min) * float(y_fraction)
        self.set_limits(
            _unscale_value(tx_min + x_shift, self.x_scale),
            _unscale_value(tx_max + x_shift, self.x_scale),
            _unscale_value(ty_min + y_shift, self.y_scale),
            _unscale_value(ty_max + y_shift, self.y_scale),
        )


class PlotSettingsDialog(ctk.CTkToplevel):
    """Global plot and selected-curve presentation settings."""

    def __init__(self, parent: "ScientificPlotWorkbench"):
        super().__init__(parent, fg_color=COLORS["app_bg"])
        self.workbench = parent
        self.title("Plot Settings")
        self.geometry("760x620")
        self.minsize(620, 440)
        self.transient(parent.winfo_toplevel())
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(
            self,
            text="Plot Settings",
            text_color=COLORS["ink"],
            font=FONTS["title"],
            anchor="w",
        ).grid(row=0, column=0, padx=24, pady=(18, 10), sticky="ew")

        self.settings_tabs = ctk.CTkTabview(
            self,
            fg_color=COLORS["surface_alt"],
            segmented_button_selected_color=COLORS["primary"],
            segmented_button_selected_hover_color=COLORS["primary_hover"],
            segmented_button_unselected_color=COLORS["control"],
            segmented_button_unselected_hover_color=COLORS["control_hover"],
            text_color=COLORS["ink"],
            corner_radius=14,
        )
        self.settings_tabs.grid(row=1, column=0, padx=24, sticky="nsew")
        axes_tab = self.settings_tabs.add("Axes & Grid")
        text_tab = self.settings_tabs.add("Text & Legend")
        curve_tab = self.settings_tabs.add("Selected Curve")
        for tab in (axes_tab, text_tab, curve_tab):
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(0, weight=1)
        self.settings_scrolls: dict[str, ctk.CTkScrollableFrame] = {}
        plot_card = self._scrollable_settings_card(
            axes_tab, "Axes & Grid", "PLOT & AXES"
        )
        text_card = self._scrollable_settings_card(
            text_tab, "Text & Legend", "TEXT & LEGEND"
        )
        curve_card = self._scrollable_settings_card(
            curve_tab, "Selected Curve", "SELECTED CURVE"
        )
        state = parent.state
        x_min, x_max, y_min, y_max = state.view_limits

        self.plot_title_entry = self._entry_row(
            plot_card, 1, "Plot title", state.plot_title
        )
        self.x_label_entry = self._entry_row(plot_card, 2, "X label", state.x_label)
        self.y_label_entry = self._entry_row(plot_card, 3, "Y label", state.y_label)
        self.x_min_entry = self._entry_row(plot_card, 4, "X minimum", f"{x_min:.12g}")
        self.x_max_entry = self._entry_row(plot_card, 5, "X maximum", f"{x_max:.12g}")
        self.y_min_entry = self._entry_row(plot_card, 6, "Y minimum", f"{y_min:.12g}")
        self.y_max_entry = self._entry_row(plot_card, 7, "Y maximum", f"{y_max:.12g}")
        self.x_scale_menu = self._option_row(
            plot_card, 8, "X scale", AXIS_SCALES, state.x_scale
        )
        self.y_scale_menu = self._option_row(
            plot_card, 9, "Y scale", AXIS_SCALES, state.y_scale
        )
        self.major_grid_var = tk.BooleanVar(value=state.major_grid)
        self.minor_grid_var = tk.BooleanVar(value=state.minor_grid)
        self.legend_visible_var = tk.BooleanVar(value=state.legend_visible)
        toggles = ctk.CTkFrame(plot_card, fg_color="transparent")
        toggles.grid(row=10, column=0, columnspan=2, padx=12, pady=(9, 8), sticky="ew")
        for label, variable in (
            ("Major grid", self.major_grid_var),
            ("Minor grid", self.minor_grid_var),
        ):
            ctk.CTkCheckBox(
                toggles,
                text=label,
                variable=variable,
                text_color=COLORS["ink"],
                fg_color=COLORS["primary"],
                hover_color=COLORS["primary_hover"],
                font=("Segoe UI", 14),
                width=92,
            ).pack(side="left", padx=(0, 7))

        self.plot_title_font_size_entry = self._entry_row(
            text_card, 1, "Title font size", f"{state.plot_title_font_size:g}"
        )
        self.x_label_font_size_entry = self._entry_row(
            text_card, 2, "X-label font size", f"{state.x_label_font_size:g}"
        )
        self.y_label_font_size_entry = self._entry_row(
            text_card, 3, "Y-label font size", f"{state.y_label_font_size:g}"
        )
        self.x_value_font_size_entry = self._entry_row(
            text_card, 4, "X-value font size", f"{state.x_value_font_size:g}"
        )
        self.y_value_font_size_entry = self._entry_row(
            text_card, 5, "Y-value font size", f"{state.y_value_font_size:g}"
        )
        self.legend_font_size_entry = self._entry_row(
            text_card, 6, "Legend font size", f"{state.legend_font_size:g}"
        )
        self.legend_line_width_entry = self._entry_row(
            text_card, 7, "Legend sample width", f"{state.legend_line_width:g}"
        )
        self.legend_location_menu = self._option_row(
            text_card,
            8,
            "Legend location",
            LEGEND_LOCATIONS,
            state.legend_location,
        )
        self.show_legend_checkbox = ctk.CTkCheckBox(
            text_card,
            text="Show legend",
            variable=self.legend_visible_var,
            text_color=COLORS["ink"],
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            font=("Segoe UI", 14),
        )
        self.show_legend_checkbox.grid(
            row=9,
            column=0,
            columnspan=2,
            padx=14,
            pady=(10, 12),
            sticky="w",
        )

        curve = state.selected_curve
        curve_name = curve.name if curve is not None else "No curve selected"
        self.curve_name_label = ctk.CTkLabel(
            curve_card,
            text=curve_name,
            text_color=COLORS["ink"],
            font=FONTS["card_title"],
            anchor="w",
        )
        self.curve_name_label.grid(
            row=1, column=0, columnspan=2, padx=14, pady=(8, 12), sticky="ew"
        )
        self.line_width_entry = self._entry_row(
            curve_card,
            2,
            "Line width",
            f"{curve.line_width:g}" if curve else "2",
        )
        self.line_style_menu = self._option_row(
            curve_card,
            3,
            "Line style",
            LINE_STYLES,
            curve.line_style if curve else "Solid",
        )
        self.marker_style_menu = self._option_row(
            curve_card,
            4,
            "Marker style",
            MARKER_STYLES,
            curve.marker_style if curve else "Circle",
        )
        self.marker_size_entry = self._entry_row(
            curve_card,
            5,
            "Marker size",
            f"{curve.marker_size:g}" if curve else "3",
        )
        self.help_label = ctk.CTkLabel(
            curve_card,
            text=(
                "Curve settings apply only to the currently selected curve."
                if curve
                else "Select a curve in the workbench to enable curve settings."
            ),
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            anchor="w",
            justify="left",
            wraplength=560,
        )
        self.help_label.grid(
            row=6, column=0, columnspan=2, padx=14, pady=(12, 10), sticky="ew"
        )
        if curve is None:
            for widget in (
                self.line_width_entry,
                self.line_style_menu,
                self.marker_style_menu,
                self.marker_size_entry,
            ):
                widget.configure(state="disabled")

        self.error_label = ctk.CTkLabel(
            self,
            text="",
            text_color=COLORS["danger"],
            font=FONTS["caption"],
            anchor="w",
        )
        self.error_label.grid(row=2, column=0, padx=26, pady=(5, 0), sticky="ew")
        self.actions = ctk.CTkFrame(self, fg_color="transparent")
        self.actions.grid(row=3, column=0, padx=24, pady=(7, 18), sticky="e")
        self.cancel_button = ctk.CTkButton(
            self.actions,
            text="Cancel",
            width=90,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            text_color=COLORS["ink"],
            command=self.destroy,
        )
        self.cancel_button.pack(side="left", padx=(0, 8))
        self.apply_button = ctk.CTkButton(
            self.actions,
            text="Apply",
            width=90,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            text_color=COLORS["on_primary"],
            command=self._apply,
        )
        self.apply_button.pack(side="left")
        self.grab_set()

    def _scrollable_settings_card(
        self,
        parent: ctk.CTkFrame,
        tab_name: str,
        title: str,
    ) -> ctk.CTkFrame:
        scroll = ctk.CTkScrollableFrame(
            parent,
            fg_color="transparent",
            corner_radius=0,
            scrollbar_button_color=COLORS["scrollbar"],
            scrollbar_button_hover_color=COLORS["scrollbar_hover"],
        )
        scroll.grid(row=0, column=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)
        self.settings_scrolls[tab_name] = scroll
        return self._settings_card(scroll, title, 0)

    @staticmethod
    def _settings_card(parent: ctk.CTkFrame, title: str, column: int) -> ctk.CTkFrame:
        card = ctk.CTkFrame(parent, fg_color=COLORS["surface"], corner_radius=14)
        card.grid(
            row=0,
            column=column,
            padx=(0, 6) if column == 0 else (6, 0),
            sticky="nsew",
        )
        card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            card,
            text=title,
            text_color=COLORS["cyan"],
            font=FONTS["mono"],
            anchor="w",
        ).grid(row=0, column=0, columnspan=2, padx=14, pady=(14, 8), sticky="ew")
        return card

    @staticmethod
    def _entry_row(
        parent: ctk.CTkFrame,
        row: int,
        label: str,
        value: str,
    ) -> ctk.CTkEntry:
        ctk.CTkLabel(
            parent,
            text=label,
            text_color=COLORS["muted"],
            font=("Segoe UI", 14),
            anchor="w",
        ).grid(row=row, column=0, padx=(14, 7), pady=4, sticky="w")
        entry = ctk.CTkEntry(
            parent,
            height=30,
            fg_color=COLORS["control"],
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=("Segoe UI", 14),
        )
        entry.grid(row=row, column=1, padx=(0, 14), pady=4, sticky="ew")
        entry.insert(0, value)
        return entry

    @staticmethod
    def _option_row(
        parent: ctk.CTkFrame,
        row: int,
        label: str,
        values: tuple[str, ...],
        selected: str,
    ) -> ctk.CTkOptionMenu:
        ctk.CTkLabel(
            parent,
            text=label,
            text_color=COLORS["muted"],
            font=("Segoe UI", 14),
            anchor="w",
        ).grid(row=row, column=0, padx=(14, 7), pady=4, sticky="w")
        menu = ctk.CTkOptionMenu(
            parent,
            values=list(values),
            height=30,
            corner_radius=8,
            fg_color=COLORS["control"],
            button_color=COLORS["primary"],
            button_hover_color=COLORS["primary_hover"],
            dropdown_fg_color=COLORS["surface"],
            dropdown_hover_color=COLORS["control_hover"],
            dropdown_text_color=COLORS["ink"],
            text_color=COLORS["ink"],
            font=("Segoe UI", 14),
            dropdown_font=("Segoe UI", 14),
        )
        menu.grid(row=row, column=1, padx=(0, 14), pady=4, sticky="ew")
        menu.set(selected)
        return menu

    def _apply(self) -> None:
        state = self.workbench.state
        try:
            limits = tuple(
                float(entry.get().strip())
                for entry in (
                    self.x_min_entry,
                    self.x_max_entry,
                    self.y_min_entry,
                    self.y_max_entry,
                )
            )
            curve_style: tuple[float, str, str, float] | None = None
            if state.selected_curve is not None:
                curve_style = (
                    float(self.line_width_entry.get().strip()),
                    self.line_style_menu.get(),
                    self.marker_style_menu.get(),
                    float(self.marker_size_entry.get().strip()),
                )
                # Validate curve settings before mutating global plot state.
                width, line_style, marker_style, size = curve_style
                if not math.isfinite(width) or not 0.5 <= width <= 8.0:
                    raise ValueError("Curve line width must be between 0.5 and 8.")
                if line_style not in LINE_STYLES:
                    raise ValueError("Choose a supported curve line style.")
                if marker_style not in MARKER_STYLES:
                    raise ValueError("Choose a supported marker style.")
                if not math.isfinite(size) or not 1.0 <= size <= 12.0:
                    raise ValueError("Marker size must be between 1 and 12.")
            state.configure_plot(
                plot_title=self.plot_title_entry.get(),
                x_label=self.x_label_entry.get(),
                y_label=self.y_label_entry.get(),
                limits=limits,
                x_scale=self.x_scale_menu.get(),
                y_scale=self.y_scale_menu.get(),
                major_grid=bool(self.major_grid_var.get()),
                minor_grid=bool(self.minor_grid_var.get()),
                legend_visible=bool(self.legend_visible_var.get()),
                legend_location=self.legend_location_menu.get(),
                plot_title_font_size=float(
                    self.plot_title_font_size_entry.get().strip()
                ),
                x_label_font_size=float(self.x_label_font_size_entry.get().strip()),
                y_label_font_size=float(self.y_label_font_size_entry.get().strip()),
                x_value_font_size=float(self.x_value_font_size_entry.get().strip()),
                y_value_font_size=float(self.y_value_font_size_entry.get().strip()),
                legend_font_size=float(self.legend_font_size_entry.get().strip()),
                legend_line_width=float(self.legend_line_width_entry.get().strip()),
            )
            if curve_style is not None:
                state.configure_selected_curve(
                    line_width=curve_style[0],
                    line_style=curve_style[1],
                    marker_style=curve_style[2],
                    marker_size=curve_style[3],
                )
        except ValueError as exc:
            self.error_label.configure(text=str(exc))
            return
        self.workbench.redraw()
        self.destroy()


# Compatibility name for extensions that used the earlier axis-only dialog.
AxisSettingsDialog = PlotSettingsDialog


class ScientificPlotWorkbench(ctk.CTkFrame):
    """CST/MATLAB-style reusable curve plot and compact curve manager."""

    def __init__(
        self,
        parent: ctk.CTkFrame,
        *,
        state: ScientificPlotState | None = None,
        selection_changed: Callable[[ScientificCurve | None], None] | None = None,
        manager_header_builder: Callable[[ctk.CTkFrame], None] | None = None,
    ):
        super().__init__(parent, fg_color="transparent")
        self.state = state or ScientificPlotState()
        self.selection_changed = selection_changed
        self.manager_header_builder = manager_header_builder
        self.navigation_mode = "Explore"
        self.curve_page = 0
        self.hover_details = "Move over a curve to inspect X and Y."
        self._hover_point: tuple[float, float, ScientificCurve] | None = None
        self._plot_bounds = (76.0, 30.0, 500.0, 300.0)
        self._legend_bounds: tuple[float, float, float, float] | None = None
        self._drag_origin: tuple[float, float] | None = None
        self._legend_drag_offset: tuple[float, float] | None = None
        self.plot_settings_dialog: PlotSettingsDialog | None = None
        self.axis_dialog: PlotSettingsDialog | None = None
        self.rename_dialog: ctk.CTkInputDialog | None = None
        self._clamping_plot_sash = False
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_toolbar()
        self._build_body()
        self._refresh_manager()
        self.redraw()

    @property
    def target_names(self) -> tuple[str, ...]:
        curve = self.state.selected_curve
        return curve.target_names if curve else ()

    @property
    def x_values(self) -> tuple[float, ...]:
        curve = self.state.selected_curve
        return curve.x_values if curve else ()

    @property
    def y_values(self) -> tuple[float, ...]:
        curve = self.state.selected_curve
        return curve.y_values if curve else ()

    @property
    def x_label(self) -> str:
        return self.state.x_label

    @property
    def y_label(self) -> str:
        return self.state.y_label

    def _build_toolbar(self) -> None:
        toolbar = ctk.CTkFrame(
            self,
            fg_color=COLORS["surface_alt"],
            corner_radius=10,
            border_width=1,
            border_color=COLORS["border"],
        )
        self.toolbar = toolbar
        self.toolbar_buttons: dict[str, ctk.CTkButton] = {}
        self.toolbar_help_visible = True
        toolbar.grid(row=0, column=0, pady=(0, 6), sticky="ew")
        toolbar.grid_columnconfigure(8, weight=1)
        self.mode_control = ctk.CTkSegmentedButton(
            toolbar,
            values=["Explore", "Pan", "Marker"],
            height=28,
            corner_radius=8,
            fg_color=COLORS["control"],
            selected_color=COLORS["primary"],
            selected_hover_color=COLORS["primary_hover"],
            unselected_color=COLORS["control"],
            unselected_hover_color=COLORS["control_hover"],
            text_color=COLORS["ink"],
            font=("Segoe UI Semibold", 14),
            command=self._set_mode,
        )
        self.mode_control.grid(row=0, column=0, padx=6, pady=5)
        self.mode_control.set("Explore")
        controls = (
            ("+", "Zoom in", self.zoom_in),
            ("-", "Zoom out", self.zoom_out),
            ("Reset", "Reset view", self.reset_view),
            ("Autoscale", "Fit visible curves", self.autoscale),
            ("Plot Settings", "Edit plot and selected curve", self.open_plot_settings),
        )
        for column, (label, _description, command) in enumerate(controls, start=1):
            button = ctk.CTkButton(
                toolbar,
                text=label,
                width=32 if label in {"+", "-"} else (96 if label == "Plot Settings" else 68),
                height=28,
                corner_radius=8,
                fg_color=COLORS["control"],
                hover_color=COLORS["control_hover"],
                border_width=1,
                border_color=COLORS["border"],
                text_color=COLORS["ink"],
                font=("Segoe UI Semibold", 14),
                command=command,
            )
            button.grid(row=0, column=column, padx=(0, 4), pady=5)
            self.toolbar_buttons[label] = button
            if label == "Reset":
                self.reset_button = button
            elif label == "Autoscale":
                self.autoscale_button = button
            elif label == "Plot Settings":
                self.plot_settings_button = button
                self.axes_button = button
        self.hover_label = ctk.CTkLabel(
            toolbar,
            text=self.hover_details,
            text_color=COLORS["muted"],
            font=("Cascadia Mono", 12),
            anchor="e",
        )
        self.hover_label.grid(row=0, column=8, padx=8, sticky="ew")
        toolbar.bind("<Configure>", self._toolbar_resized, add="+")

    def _toolbar_resized(self, event: tk.Event) -> None:
        show_help = event.width >= 820
        if show_help == self.toolbar_help_visible:
            return
        self.toolbar_help_visible = show_help
        if show_help:
            self.hover_label.grid()
        else:
            self.hover_label.grid_remove()

    def _build_body(self) -> None:
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)
        split = tk.PanedWindow(
            body,
            orient=tk.HORIZONTAL,
            bd=0,
            relief=tk.FLAT,
            bg=_palette(COLORS["border"]),
            sashwidth=10,
            sashpad=1,
            showhandle=True,
            handlesize=8,
            handlepad=1,
            opaqueresize=True,
            cursor="sb_h_double_arrow",
        )
        split.grid(row=0, column=0, sticky="nsew")
        self.plot_split = split
        split.bind("<Configure>", self._plot_split_resized, add="+")
        plot_shell = ctk.CTkFrame(
            split,
            fg_color=COLORS["surface_alt"],
            corner_radius=10,
            border_width=1,
            border_color=COLORS["border"],
        )
        self.plot_shell = plot_shell
        self.canvas = tk.Canvas(plot_shell, highlightthickness=0, bd=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas.bind("<Configure>", lambda _event: self.redraw())
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._clear_hover)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda event: self._wheel_factor(event, 0.8))
        self.canvas.bind("<Button-5>", lambda event: self._wheel_factor(event, 1.25))

        manager = ctk.CTkFrame(
            split,
            fg_color=COLORS["surface_alt"],
            corner_radius=10,
            border_width=1,
            border_color=COLORS["border"],
        )
        self.curve_manager = manager
        manager.bind("<Configure>", self._curve_manager_resized, add="+")
        manager.grid_columnconfigure(0, weight=1)
        manager.grid_rowconfigure(2, weight=1)
        header = ctk.CTkFrame(manager, fg_color="transparent")
        header.grid(row=0, column=0, padx=10, pady=(9, 4), sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        self.curve_count_label = ctk.CTkLabel(
            header,
            text="CURVES · 0",
            text_color=COLORS["cyan"],
            font=("Cascadia Mono", 12),
            anchor="w",
        )
        self.curve_count_label.grid(row=0, column=0, sticky="w")
        self.marker_count_label = ctk.CTkLabel(
            header,
            text="MARKERS · 0",
            text_color=COLORS["muted"],
            font=("Cascadia Mono", 11),
        )
        self.marker_count_label.grid(row=0, column=1, sticky="e")
        self.manager_context = ctk.CTkFrame(manager, fg_color="transparent")
        self.manager_context.grid(row=1, column=0, padx=8, pady=(2, 5), sticky="ew")
        self.manager_context.grid_columnconfigure(0, weight=1)
        if self.manager_header_builder is None:
            self.manager_context.grid_remove()
        else:
            self.manager_header_builder(self.manager_context)
        self.curve_rows = ctk.CTkFrame(manager, fg_color="transparent")
        self.curve_rows.grid(row=2, column=0, padx=8, sticky="nsew")
        self.curve_rows.grid_columnconfigure(0, weight=1)
        self.manager_pager = ctk.CTkFrame(manager, fg_color="transparent")
        self.manager_pager.grid(row=3, column=0, padx=10, sticky="ew")
        self.manager_pager.grid_columnconfigure(1, weight=1)
        self.curve_previous = ctk.CTkButton(
            self.manager_pager,
            text="<",
            width=28,
            height=24,
            fg_color=COLORS["control"],
            hover_color=COLORS["control_hover"],
            text_color=COLORS["ink"],
            command=lambda: self._change_curve_page(-1),
        )
        self.curve_previous.grid(row=0, column=0)
        self.curve_page_label = ctk.CTkLabel(
            self.manager_pager,
            text="1 / 1",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
        )
        self.curve_page_label.grid(row=0, column=1)
        self.curve_next = ctk.CTkButton(
            self.manager_pager,
            text=">",
            width=28,
            height=24,
            fg_color=COLORS["control"],
            hover_color=COLORS["control_hover"],
            text_color=COLORS["ink"],
            command=lambda: self._change_curve_page(1),
        )
        self.curve_next.grid(row=0, column=2)
        self.manager_pager.grid_remove()
        self.selected_inputs = ctk.CTkLabel(
            manager,
            text="Select a curve to inspect its inputs.",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            anchor="w",
            justify="left",
            wraplength=166,
        )
        self.selected_inputs.grid(row=4, column=0, padx=10, pady=(6, 4), sticky="ew")
        actions = ctk.CTkFrame(manager, fg_color="transparent")
        actions.grid(row=5, column=0, padx=8, pady=(4, 8), sticky="ew")
        actions.grid_columnconfigure((0, 1), weight=1)
        self.rename_button = ctk.CTkButton(
            actions,
            text="Rename",
            height=27,
            fg_color=COLORS["control"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=("Segoe UI Semibold", 12),
            state="disabled",
            command=self._rename_selected_dialog,
        )
        self.rename_button.grid(row=0, column=0, padx=(0, 3), sticky="ew")
        self.delete_button = ctk.CTkButton(
            actions,
            text="Delete",
            height=27,
            fg_color=COLORS["control"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["danger"],
            font=("Segoe UI Semibold", 12),
            state="disabled",
            command=self.delete_selected,
        )
        self.delete_button.grid(row=0, column=1, padx=(3, 0), sticky="ew")
        self.clear_markers_button = ctk.CTkButton(
            actions,
            text="Clear markers",
            height=27,
            fg_color="transparent",
            hover_color=COLORS["control_hover"],
            text_color=COLORS["muted"],
            font=("Segoe UI", 12),
            state="disabled",
            command=self.clear_annotations,
        )
        self.clear_markers_button.grid(row=1, column=0, columnspan=2, pady=(4, 0), sticky="ew")
        split.add(
            plot_shell,
            minsize=PLOT_PANE_MIN_WIDTH + PLOT_CANVAS_HORIZONTAL_INSET,
            stretch="always",
        )
        split.add(
            manager,
            minsize=CURVE_MANAGER_MIN_WIDTH,
            width=CURVE_MANAGER_DEFAULT_WIDTH,
            stretch="never",
        )

    def _curve_manager_resized(self, event: tk.Event) -> None:
        """Keep wrapped curve details aligned with the draggable manager pane."""

        self._clamp_plot_sash()
        available = max(120, int(event.width) - 24)
        self.selected_inputs.configure(wraplength=available)

    def _plot_split_resized(self, _event: tk.Event) -> None:
        """Keep plot and curve controls usable while preserving sash freedom."""

        self._clamp_plot_sash()

    def _clamp_plot_sash(self) -> None:
        if self._clamping_plot_sash or len(self.plot_split.panes()) < 2:
            return
        total_width = int(self.plot_split.winfo_width())
        if total_width <= 1:
            return
        sash_width = int(float(self.plot_split.cget("sashwidth")))
        minimum_plot_shell_width = (
            PLOT_PANE_MIN_WIDTH + PLOT_CANVAS_HORIZONTAL_INSET
        )
        maximum_plot_width = total_width - CURVE_MANAGER_MIN_WIDTH - sash_width
        if maximum_plot_width < minimum_plot_shell_width:
            return
        current = int(self.plot_split.sash_coord(0)[0])
        target = max(minimum_plot_shell_width, min(current, maximum_plot_width))
        if abs(current - target) <= 1:
            return
        self._clamping_plot_sash = True
        try:
            self.plot_split.sash_place(0, target, 1)
        finally:
            self._clamping_plot_sash = False

    def add_curve(
        self,
        *,
        x_values: Iterable[float],
        y_values: Iterable[float],
        target_names: Iterable[str],
        inputs: dict[str, float],
        replace_selected: bool,
        x_label: str | None = None,
        name: str | None = None,
    ) -> ScientificCurve:
        curve = self.state.add_curve(
            x_values=x_values,
            y_values=y_values,
            target_names=target_names,
            inputs=inputs,
            replace_selected=replace_selected,
            name=name,
        )
        if x_label and not self.state.axis_labels_user_defined:
            self.state.x_label = x_label
        self._refresh_manager()
        self.redraw()
        self._notify_selection()
        return curve

    def set_response(self, predictions: dict[str, float]) -> None:
        targets = tuple(predictions)
        self.add_curve(
            x_values=range(1, len(targets) + 1),
            y_values=predictions.values(),
            target_names=targets,
            inputs={},
            replace_selected=True,
        )

    def clear(self) -> None:
        self.state = ScientificPlotState()
        self.curve_page = 0
        self._hover_point = None
        self._refresh_manager()
        self.redraw()
        self._notify_selection()

    def refresh_theme(self) -> None:
        self.plot_split.configure(bg=_palette(COLORS["border"]))
        self.redraw()

    def select_curve(self, curve_id: str) -> None:
        self.state.select_curve(curve_id)
        self._refresh_manager()
        self.redraw()
        self._notify_selection()

    def set_curve_visible(self, curve_id: str, visible: bool) -> None:
        self.state.set_curve_visible(curve_id, visible)
        self._refresh_manager()
        self.redraw()
        self._notify_selection()

    def rename_selected(self, name: str) -> None:
        if self.state.selected_curve_id is None:
            return
        self.state.rename_curve(self.state.selected_curve_id, name)
        self._refresh_manager()
        self.redraw()
        self._notify_selection()

    def delete_selected(self) -> None:
        if self.state.selected_curve_id is None:
            return
        self.state.delete_curve(self.state.selected_curve_id)
        self._refresh_manager()
        self.redraw()
        self._notify_selection()

    def add_annotation(self, x: float, y: float, label: str | None = None) -> PlotAnnotation:
        marker = self.state.add_annotation(
            x,
            y,
            label=label,
            curve_id=self.state.selected_curve_id,
        )
        self._refresh_manager()
        self.redraw()
        return marker

    def clear_annotations(self) -> None:
        self.state.clear_annotations()
        self._refresh_manager()
        self.redraw()

    def zoom_in(self) -> None:
        self.state.zoom(0.8)
        self.redraw()

    def zoom_out(self) -> None:
        self.state.zoom(1.25)
        self.redraw()

    def reset_view(self) -> None:
        self.state.reset_view()
        self.redraw()

    def autoscale(self) -> None:
        self.state.autoscale()
        self.redraw()

    def open_plot_settings(self) -> None:
        if (
            self.plot_settings_dialog is not None
            and self.plot_settings_dialog.winfo_exists()
        ):
            self.plot_settings_dialog.focus_set()
            return
        self.plot_settings_dialog = PlotSettingsDialog(self)
        self.axis_dialog = self.plot_settings_dialog

    def open_axis_settings(self) -> None:
        """Compatibility alias for the former Axes toolbar action."""

        self.open_plot_settings()

    def _set_mode(self, mode: str) -> None:
        self.navigation_mode = mode
        cursor = "fleur" if mode == "Pan" else "crosshair"
        self.canvas.configure(cursor=cursor)
        if mode == "Marker":
            self._set_hover_text("Click the plot to place a marker.")

    def _rename_selected_dialog(self) -> None:
        curve = self.state.selected_curve
        if curve is None:
            return
        dialog = ctk.CTkInputDialog(
            text="Enter a clear curve name:",
            title="Rename Curve",
        )
        self.rename_dialog = dialog
        value = dialog.get_input()
        if value is None:
            return
        try:
            self.rename_selected(value)
        except ValueError:
            return

    def _refresh_manager(self) -> None:
        for child in self.curve_rows.winfo_children():
            child.destroy()
        count = len(self.state.curves)
        self.curve_count_label.configure(text=f"CURVES · {count}")
        self.marker_count_label.configure(text=f"MARKERS · {len(self.state.annotations)}")
        page_count = max(1, math.ceil(count / CURVES_PER_PAGE))
        self.curve_page = min(self.curve_page, page_count - 1)
        start = self.curve_page * CURVES_PER_PAGE
        for row, curve in enumerate(self.state.curves[start : start + CURVES_PER_PAGE]):
            shell = ctk.CTkFrame(
                self.curve_rows,
                fg_color=(
                    COLORS["primary_soft"]
                    if curve.curve_id == self.state.selected_curve_id
                    else COLORS["surface"]
                ),
                corner_radius=8,
                border_width=1,
                border_color=COLORS["border"],
            )
            shell.grid(row=row, column=0, pady=3, sticky="ew")
            shell.grid_columnconfigure(1, weight=1)
            visible_var = tk.BooleanVar(value=curve.visible)
            ctk.CTkCheckBox(
                shell,
                text="",
                width=18,
                checkbox_width=17,
                checkbox_height=17,
                variable=visible_var,
                fg_color=curve_color(curve.color_index),
                hover_color=curve_color(curve.color_index),
                command=lambda curve_id=curve.curve_id, variable=visible_var: (
                    self.set_curve_visible(curve_id, bool(variable.get()))
                ),
            ).grid(row=0, column=0, padx=(7, 2), pady=7)
            ctk.CTkButton(
                shell,
                text=curve.name,
                height=25,
                anchor="w",
                fg_color="transparent",
                hover_color=COLORS["control_hover"],
                text_color=COLORS["ink"],
                font=("Segoe UI Semibold", 14),
                command=lambda curve_id=curve.curve_id: self.select_curve(curve_id),
            ).grid(row=0, column=1, padx=(0, 4), pady=3, sticky="ew")
        if count > CURVES_PER_PAGE:
            self.curve_page_label.configure(text=f"{self.curve_page + 1} / {page_count}")
            self.curve_previous.configure(state="normal" if self.curve_page else "disabled")
            self.curve_next.configure(
                state="normal" if self.curve_page + 1 < page_count else "disabled"
            )
            self.manager_pager.grid()
        else:
            self.manager_pager.grid_remove()
        selected = self.state.selected_curve
        action_state = "normal" if selected else "disabled"
        self.rename_button.configure(state=action_state)
        self.delete_button.configure(state=action_state)
        self.clear_markers_button.configure(
            state="normal" if self.state.annotations else "disabled"
        )
        if selected is None:
            self.selected_inputs.configure(text="Select a curve to inspect its inputs.")
        else:
            values = " · ".join(
                f"{name}={value:.5g}" for name, value in selected.inputs.items()
            )
            self.selected_inputs.configure(
                text=f"{selected.name}\nInputs: {values or 'not recorded'}"
            )

    def _change_curve_page(self, offset: int) -> None:
        count = len(self.state.curves)
        page_count = max(1, math.ceil(count / CURVES_PER_PAGE))
        self.curve_page = max(0, min(self.curve_page + offset, page_count - 1))
        self._refresh_manager()

    def _notify_selection(self) -> None:
        if self.selection_changed:
            self.selection_changed(self.state.selected_curve)

    def redraw(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        canvas.configure(bg=_palette(COLORS["surface_alt"]))
        measured_width = canvas.winfo_width()
        measured_height = canvas.winfo_height()
        width = measured_width if measured_width > 1 else 560
        height = measured_height if measured_height > 1 else 320
        left_margin = max(76.0, 55.0 + self.state.y_label_font_size)
        top_margin = max(34.0, self.state.plot_title_font_size + 20.0)
        bottom_margin = max(
            62.0,
            34.0 + self.state.x_label_font_size + self.state.x_value_font_size,
        )
        left, right, top, bottom = (
            left_margin,
            max(left_margin + 12.0, width - 26.0),
            top_margin,
            max(top_margin + 12.0, height - bottom_margin),
        )
        self._plot_bounds = (left, top, right, bottom)
        x_min, x_max, y_min, y_max = self.state.view_limits
        tx_min, tx_max = _scale_pair(x_min, x_max, self.state.x_scale)
        ty_min, ty_max = _scale_pair(y_min, y_max, self.state.y_scale)
        ink = _palette(COLORS["ink"])
        muted = _palette(COLORS["muted"])
        border = _palette(COLORS["border_strong"])
        major = _palette(COLORS["border"])
        minor = _palette(COLORS["surface_elevated"])

        def map_x(value: float) -> float:
            transformed = _scale_value(value, self.state.x_scale)
            return left + (transformed - tx_min) / (tx_max - tx_min) * (right - left)

        def map_y(value: float) -> float:
            transformed = _scale_value(value, self.state.y_scale)
            return bottom - (transformed - ty_min) / (ty_max - ty_min) * (bottom - top)

        canvas.create_text(
            (left + right) / 2,
            max(16.0, self.state.plot_title_font_size * 0.72),
            text=self.state.plot_title,
            fill=ink,
            font=("Segoe UI Semibold", int(round(self.state.plot_title_font_size))),
        )

        x_major_count = adaptive_major_interval_count(
            right - left,
            self.state.x_value_font_size,
        )
        y_major_count = 5
        for index in range(x_major_count + 1):
            fraction = index / x_major_count
            x = left + fraction * (right - left)
            if self.state.major_grid:
                canvas.create_line(x, top, x, bottom, fill=major, width=1)
            x_value = _unscale_value(
                tx_min + fraction * (tx_max - tx_min),
                self.state.x_scale,
            )
            canvas.create_text(
                x,
                bottom + 16,
                text=engineering_tick(x_value),
                fill=muted,
                font=("Cascadia Mono", int(round(self.state.x_value_font_size))),
                tags=("x_tick_label",),
            )
            if index < x_major_count and self.state.minor_grid:
                for subdivision in range(1, 5):
                    minor_fraction = (
                        index + subdivision / 5
                    ) / x_major_count
                    minor_x = left + minor_fraction * (right - left)
                    canvas.create_line(minor_x, top, minor_x, bottom, fill=minor, width=1)

        for index in range(y_major_count + 1):
            fraction = index / y_major_count
            y = bottom - fraction * (bottom - top)
            if self.state.major_grid:
                canvas.create_line(left, y, right, y, fill=major, width=1)
            y_value = _unscale_value(
                ty_min + fraction * (ty_max - ty_min),
                self.state.y_scale,
            )
            canvas.create_text(
                left - 8,
                y,
                text=engineering_tick(y_value),
                fill=muted,
                font=("Cascadia Mono", int(round(self.state.y_value_font_size))),
                anchor="e",
                tags=("y_tick_label",),
            )
            if index < y_major_count and self.state.minor_grid:
                for subdivision in range(1, 5):
                    minor_fraction = (
                        index + subdivision / 5
                    ) / y_major_count
                    minor_y = bottom - minor_fraction * (bottom - top)
                    canvas.create_line(left, minor_y, right, minor_y, fill=minor, width=1)
        canvas.create_rectangle(left, top, right, bottom, outline=border, width=1)
        canvas.create_text(
            (left + right) / 2,
            height - max(18.0, self.state.x_label_font_size * 0.75),
            text=self.state.x_label,
            fill=ink,
            font=("Segoe UI Semibold", int(round(self.state.x_label_font_size))),
        )
        canvas.create_text(
            max(16.0, self.state.y_label_font_size * 0.75),
            (top + bottom) / 2,
            text=self.state.y_label,
            fill=ink,
            font=("Segoe UI Semibold", int(round(self.state.y_label_font_size))),
            angle=90,
        )

        visible_curves = [curve for curve in self.state.curves if curve.visible]
        if not visible_curves:
            canvas.create_text(
                (left + right) / 2,
                (top + bottom) / 2,
                text="Run a prediction to add a scientific response curve.",
                fill=muted,
                font=FONTS["body_small"],
            )
        for curve in visible_curves:
            points = [
                coordinate
                for x_value, y_value in zip(curve.x_values, curve.y_values, strict=True)
                for coordinate in (map_x(x_value), map_y(y_value))
            ]
            color = curve_color(curve.color_index)
            if len(points) >= 4 and curve.line_style != "None":
                line_dash = {
                    "Solid": None,
                    "Dashed": (8, 4),
                    "Dotted": (2, 4),
                }[curve.line_style]
                canvas.create_line(
                    *points,
                    fill=color,
                    width=(
                        curve.line_width + 1
                        if curve.curve_id == self.state.selected_curve_id
                        else curve.line_width
                    ),
                    dash=line_dash,
                )
            for point_index in range(0, len(points), 2):
                point_x, point_y = points[point_index : point_index + 2]
                if (
                    curve.marker_style != "None"
                    and (
                        curve.line_style == "None"
                        or len(points) == 2
                        or len(points) <= 48
                    )
                ):
                    self._draw_curve_marker(
                        point_x,
                        point_y,
                        curve.marker_style,
                        curve.marker_size,
                        color,
                    )

        for marker in self.state.annotations:
            if marker.curve_id is not None:
                curve = self.state.curve(marker.curve_id)
                if curve is None or not curve.visible:
                    continue
            try:
                marker_x, marker_y = map_x(marker.x), map_y(marker.y)
            except ValueError:
                continue
            if not (left <= marker_x <= right and top <= marker_y <= bottom):
                continue
            color = _palette(COLORS["warning"])
            canvas.create_line(marker_x - 6, marker_y, marker_x + 6, marker_y, fill=color, width=2)
            canvas.create_line(marker_x, marker_y - 6, marker_x, marker_y + 6, fill=color, width=2)
            canvas.create_text(
                marker_x + 8,
                marker_y - 8,
                text=f"{marker.label}  X={engineering_tick(marker.x)}  Y={engineering_tick(marker.y)}",
                fill=color,
                font=("Cascadia Mono", 11),
                anchor="sw",
            )

        if self._hover_point is not None:
            hover_x, hover_y, _curve = self._hover_point
            pixel_x, pixel_y = map_x(hover_x), map_y(hover_y)
            if left <= pixel_x <= right and top <= pixel_y <= bottom:
                canvas.create_line(pixel_x, top, pixel_x, bottom, fill=muted, dash=(4, 3))
                canvas.create_line(left, pixel_y, right, pixel_y, fill=muted, dash=(4, 3))
        self._draw_legend(visible_curves)

    def _draw_curve_marker(
        self,
        x: float,
        y: float,
        style: str,
        size: float,
        color: str,
    ) -> None:
        outline = _palette(COLORS["surface"])
        if style == "Circle":
            self.canvas.create_oval(
                x - size,
                y - size,
                x + size,
                y + size,
                fill=color,
                outline=outline,
            )
        elif style == "Square":
            self.canvas.create_rectangle(
                x - size,
                y - size,
                x + size,
                y + size,
                fill=color,
                outline=outline,
            )
        elif style == "Diamond":
            self.canvas.create_polygon(
                x,
                y - size,
                x + size,
                y,
                x,
                y + size,
                x - size,
                y,
                fill=color,
                outline=outline,
            )

    def _draw_legend(self, curves: list[ScientificCurve]) -> None:
        if not self.state.legend_visible or not curves:
            self._legend_bounds = None
            return
        left, top, right, bottom = self._plot_bounds
        font_size = int(round(self.state.legend_font_size))
        row_height = max(20.0, font_size + 11.0)
        character_width = max(6.0, font_size * 0.72)
        width = min(
            280.0,
            max(
                130.0,
                max(len(curve.name) for curve in curves) * character_width + 50.0,
            ),
        )
        height = min(24.0 + len(curves) * row_height, bottom - top - 8)
        anchor_x = left + self.state.legend_position[0] * (right - left)
        anchor_y = top + self.state.legend_position[1] * (bottom - top)
        x2 = max(left + width, min(anchor_x, right - 4))
        y1 = max(top + 4, min(anchor_y, bottom - height - 4))
        x1, y2 = x2 - width, y1 + height
        self._legend_bounds = (x1, y1, x2, y2)
        self.canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            fill=_palette(COLORS["surface"]),
            outline=_palette(COLORS["border_strong"]),
            width=1,
        )
        self.canvas.create_text(
            x1 + 9,
            y1 + 10,
            text="LEGEND · drag",
            fill=_palette(COLORS["muted"]),
            font=("Cascadia Mono", max(7, font_size - 1)),
            anchor="w",
        )
        available_rows = max(0, int((height - 22) // row_height))
        for row, curve in enumerate(curves[:available_rows]):
            y = y1 + 25 + row * row_height
            color = curve_color(curve.color_index)
            if curve.line_style == "None":
                self._draw_curve_marker(
                    x1 + 20,
                    y,
                    curve.marker_style if curve.marker_style != "None" else "Circle",
                    max(2.0, curve.marker_size),
                    color,
                )
            else:
                self.canvas.create_line(
                    x1 + 10,
                    y,
                    x1 + 30,
                    y,
                    fill=color,
                    width=self.state.legend_line_width,
                    dash={"Solid": None, "Dashed": (8, 4), "Dotted": (2, 4)}[
                        curve.line_style
                    ],
                )
            available_label_width = max(40.0, width - 47.0)
            maximum_characters = max(
                6,
                int(available_label_width / character_width),
            )
            label = (
                curve.name
                if len(curve.name) <= maximum_characters
                else f"{curve.name[: maximum_characters - 1]}…"
            )
            self.canvas.create_text(
                x1 + 37,
                y,
                text=label,
                fill=_palette(COLORS["ink"]),
                font=("Segoe UI", font_size),
                anchor="w",
            )

    def _map_pixel_to_data(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        left, top, right, bottom = self._plot_bounds
        x_min, x_max, y_min, y_max = self.state.view_limits
        tx_min, tx_max = _scale_pair(x_min, x_max, self.state.x_scale)
        ty_min, ty_max = _scale_pair(y_min, y_max, self.state.y_scale)
        x_value = _unscale_value(
            tx_min + (pixel_x - left) / (right - left) * (tx_max - tx_min),
            self.state.x_scale,
        )
        y_value = _unscale_value(
            ty_max - (pixel_y - top) / (bottom - top) * (ty_max - ty_min),
            self.state.y_scale,
        )
        return x_value, y_value

    def _nearest_point(
        self,
        pixel_x: float,
        pixel_y: float,
        *,
        maximum_distance: float = 16.0,
    ) -> tuple[float, float, ScientificCurve] | None:
        left, top, right, bottom = self._plot_bounds
        x_min, x_max, y_min, y_max = self.state.view_limits
        tx_min, tx_max = _scale_pair(x_min, x_max, self.state.x_scale)
        ty_min, ty_max = _scale_pair(y_min, y_max, self.state.y_scale)
        nearest: tuple[float, float, ScientificCurve] | None = None
        nearest_distance = maximum_distance
        for curve in self.state.curves:
            if not curve.visible:
                continue
            for x_value, y_value in zip(curve.x_values, curve.y_values, strict=True):
                x = left + (
                    _scale_value(x_value, self.state.x_scale) - tx_min
                ) / (tx_max - tx_min) * (right - left)
                y = bottom - (
                    _scale_value(y_value, self.state.y_scale) - ty_min
                ) / (ty_max - ty_min) * (bottom - top)
                distance = math.hypot(pixel_x - x, pixel_y - y)
                if distance <= nearest_distance:
                    nearest_distance = distance
                    nearest = (x_value, y_value, curve)
        return nearest

    def _on_motion(self, event: tk.Event) -> None:
        if self._drag_origin is not None or self._legend_drag_offset is not None:
            return
        nearest = self._nearest_point(float(event.x), float(event.y))
        self._hover_point = nearest
        if nearest is None:
            self._set_hover_text("Move over a curve to inspect X and Y.")
        else:
            x_value, y_value, curve = nearest
            self._set_hover_text(
                f"{curve.name} · X {engineering_tick(x_value)} · Y {engineering_tick(y_value)}"
            )
        self.redraw()

    def _clear_hover(self, _event: tk.Event | None = None) -> None:
        self._hover_point = None
        self._set_hover_text("Move over a curve to inspect X and Y.")
        self.redraw()

    def _set_hover_text(self, text: str) -> None:
        self.hover_details = text
        self.hover_label.configure(text=text)

    def _on_press(self, event: tk.Event) -> None:
        x, y = float(event.x), float(event.y)
        if self._legend_bounds is not None:
            x1, y1, x2, y2 = self._legend_bounds
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._legend_drag_offset = (x2 - x, y - y1)
                return
        left, top, right, bottom = self._plot_bounds
        if not (left <= x <= right and top <= y <= bottom):
            return
        if self.navigation_mode == "Marker":
            data_x, data_y = self._map_pixel_to_data(x, y)
            nearest = self._nearest_point(x, y, maximum_distance=24.0)
            if nearest is not None:
                data_x, data_y, curve = nearest
                self.state.selected_curve_id = curve.curve_id
            self.add_annotation(data_x, data_y)
            self.mode_control.set("Explore")
            self._set_mode("Explore")
            return
        if self.navigation_mode == "Pan":
            self._drag_origin = (x, y)

    def _on_drag(self, event: tk.Event) -> None:
        x, y = float(event.x), float(event.y)
        left, top, right, bottom = self._plot_bounds
        if self._legend_drag_offset is not None:
            right_offset, top_offset = self._legend_drag_offset
            anchor_x = min(right, max(left, x + right_offset))
            anchor_y = min(bottom, max(top, y - top_offset))
            self.state.legend_position = (
                (anchor_x - left) / (right - left),
                (anchor_y - top) / (bottom - top),
            )
            self.redraw()
            return
        if self._drag_origin is None:
            return
        old_x, old_y = self._drag_origin
        self.state.pan(
            -(x - old_x) / max(1.0, right - left),
            (y - old_y) / max(1.0, bottom - top),
        )
        self._drag_origin = (x, y)
        self.redraw()

    def _on_release(self, _event: tk.Event) -> None:
        self._drag_origin = None
        self._legend_drag_offset = None

    def _on_wheel(self, event: tk.Event) -> None:
        self._wheel_factor(event, 0.8 if event.delta > 0 else 1.25)

    def _wheel_factor(self, event: tk.Event, factor: float) -> None:
        left, top, right, bottom = self._plot_bounds
        if left <= event.x <= right and top <= event.y <= bottom:
            center = self._map_pixel_to_data(float(event.x), float(event.y))
            self.state.zoom(factor, center=center)
            self.redraw()
