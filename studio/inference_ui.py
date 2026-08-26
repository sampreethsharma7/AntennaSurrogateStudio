"""Fixed, non-scrolling single-sample Inference page."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import TYPE_CHECKING

import customtkinter as ctk

from studio.inference import (
    InferenceRequest,
    InferenceResult,
    submit_inference_request,
)
from studio.model_book import ModelBook, ModelBookError, load_model_library
from studio.project_store import Project, atomic_write_json, utc_now
from studio.scientific_plot import ScientificCurve, ScientificPlotWorkbench
from studio.theme import COLORS, FONTS

if TYPE_CHECKING:
    from studio.ui import StudioApp


INPUTS_PER_PAGE = 8
MIN_USABLE_PREDICTION_PLOT_WIDTH = 420
PREDICTION_EXPORT_SCHEMA_VERSION = 1


def _ordered_items(
    preferred_order: list[str],
    values: dict[str, float],
) -> list[tuple[str, float]]:
    """Return saved-order values while retaining any unexpected trailing keys."""

    ordered_names = [name for name in preferred_order if name in values]
    ordered_names.extend(name for name in values if name not in ordered_names)
    return [(name, float(values[name])) for name in ordered_names]


def prediction_export_payload(
    result: InferenceResult,
    book: ModelBook,
) -> dict[str, object]:
    """Build a portable, explicitly ordered prediction export."""

    if not result.success or not result.predictions:
        raise ValueError("A successful prediction is required before export.")
    return {
        "schema_version": PREDICTION_EXPORT_SCHEMA_VERSION,
        "export_type": "antenna_surrogate_studio_prediction",
        "exported_at": utc_now(),
        "model_book": {
            "book_id": book.book_id,
            "name": book.name,
            "version": book.version,
            "model_name": book.model_name,
            "model_type": book.model_type,
            "dataset_fingerprint": book.dataset_fingerprint,
        },
        "output_axis": (
            book.output_axis.to_dict() if book.output_axis is not None else None
        ),
        "inputs": [
            {"name": name, "value": value}
            for name, value in _ordered_items(result.feature_order, result.input_values)
        ],
        "output_count": len(result.predictions),
        "predicted_outputs": [
            {"target": target, "value": value}
            for target, value in _ordered_items(
                result.target_order,
                result.predictions,
            )
        ],
    }


def write_prediction_curve_csv(
    path: Path,
    result: InferenceResult,
    book: ModelBook,
) -> None:
    """Write one engineering-friendly ordered response curve."""

    if not result.success or not result.predictions:
        raise ValueError("A successful prediction is required before export.")
    axis = book.output_axis
    if axis is None or len(axis.values) != len(result.target_order):
        raise ValueError("The active Model Book does not have a valid output axis.")
    ordered_predictions = dict(
        _ordered_items(result.target_order, result.predictions)
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    axis.display_label,
                    "Predicted value",
                    "Output variable",
                ]
            )
            for coordinate, target in zip(
                axis.values,
                result.target_order,
                strict=True,
            ):
                writer.writerow([coordinate, ordered_predictions[target], target])
    except OSError as exc:
        raise OSError(f"The curve CSV could not be written: {exc}") from exc


def raw_prediction_text(result: InferenceResult, book: ModelBook) -> str:
    """Format complete saved-order inputs and outputs for the raw-value dialog."""

    input_lines = [
        f"{name} = {value:.12g}"
        for name, value in _ordered_items(result.feature_order, result.input_values)
    ]
    output_lines = [
        f"{target} = {value:.12g}"
        for target, value in _ordered_items(result.target_order, result.predictions)
    ]
    return "\n".join(
        [
            f"Model Book: {book.name}",
            f"Book ID: {book.book_id}",
            f"Model type: {_model_type_label(book)}",
            "",
            f"INPUTS USED ({len(input_lines)})",
            *input_lines,
            "",
            f"PREDICTED OUTPUTS ({len(output_lines)})",
            *output_lines,
        ]
    )


def _model_type_label(book: ModelBook) -> str:
    if book.model_name == "linear_regression":
        return "Linear Regression"
    if book.model_name == "xgboost":
        return "XGBoost"
    if book.model_name == "neural_network":
        return "Neural Network"
    if book.model_name == "ensemble_ai_engine":
        return "Ensemble AI Engine"
    return book.model_type


class RawPredictionDialog(ctk.CTkToplevel):
    """Scrollable complete-value view for one successful prediction."""

    def __init__(
        self,
        parent: ctk.CTkFrame,
        result: InferenceResult,
        book: ModelBook,
    ):
        super().__init__(parent, fg_color=COLORS["app_bg"])
        self.title("Raw Prediction Values")
        self.geometry("680x620")
        self.minsize(560, 460)
        self.transient(parent.winfo_toplevel())
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        ctk.CTkLabel(
            self,
            text="Raw Prediction Values",
            text_color=COLORS["ink"],
            font=FONTS["title"],
            anchor="w",
        ).grid(row=0, column=0, padx=24, pady=(20, 2), sticky="ew")
        ctk.CTkLabel(
            self,
            text=f"{book.name} · {_model_type_label(book)} · saved target order",
            text_color=COLORS["muted"],
            font=FONTS["body_small"],
            anchor="w",
        ).grid(row=1, column=0, padx=24, pady=(0, 10), sticky="ew")
        self.raw_text = raw_prediction_text(result, book)
        self.textbox = ctk.CTkTextbox(
            self,
            fg_color=COLORS["surface"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=("Cascadia Mono", 15),
            wrap="none",
        )
        self.textbox.grid(row=2, column=0, padx=24, pady=(0, 12), sticky="nsew")
        self.textbox.insert("1.0", self.raw_text)
        self.textbox.configure(state="disabled")
        ctk.CTkButton(
            self,
            text="Close",
            width=92,
            height=34,
            corner_radius=10,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            command=self.destroy,
        ).grid(row=3, column=0, padx=24, pady=(0, 18), sticky="e")
        self.grab_set()


class InferencePage(ctk.CTkFrame):
    """Enter one sample and predict with the active Model Book."""

    def __init__(self, parent: ctk.CTkFrame, app: "StudioApp"):
        super().__init__(parent, fg_color=COLORS["app_bg"], corner_radius=0)
        self.app = app
        self.project: Project | None = None
        self.active_book: ModelBook | None = None
        self.load_error: str | None = None
        self.input_entries: dict[str, ctk.CTkEntry] = {}
        self.input_shells: dict[str, ctk.CTkFrame] = {}
        self.input_page = 0
        self.prediction_in_progress = False
        self.last_result: InferenceResult | None = None
        self.raw_values_dialog: RawPredictionDialog | None = None
        self.result_summary_values: dict[str, ctk.CTkLabel] = {}
        self._workspace_key: tuple[Path, str] | None = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_header()
        self._build_content()
        self._build_footer()
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
                if not library.active_book_id:
                    self.load_error = (
                        "No active Model Book is selected. Open Model Library and "
                        "set a valid model as active."
                    )
                else:
                    entry = next(
                        (
                            candidate
                            for candidate in library.entries
                            if candidate.book_id == library.active_book_id
                        ),
                        None,
                    )
                    if entry is None:
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
        new_workspace_key = (
            (self.project.path.resolve(), self.active_book.book_id)
            if self.project is not None and self.active_book is not None
            else None
        )
        if new_workspace_key is not None and new_workspace_key == self._workspace_key:
            self.result_model_name.configure(text=self.active_book.name)
            self.result_model_type.configure(
                text=f"{_model_type_label(self.active_book)} · {self.active_book.book_id}"
            )
            return
        if (
            self.raw_values_dialog is not None
            and self.raw_values_dialog.winfo_exists()
        ):
            self.raw_values_dialog.destroy()
        self.raw_values_dialog = None
        self._workspace_key = new_workspace_key
        self.last_result = None
        self.input_page = 0
        self._refresh()

    def describe_ui_state(self) -> list[str]:
        if self.project is None:
            return ["Inference state: no project"]
        if self.active_book is None:
            return [f"Inference unavailable: {self.load_error or 'no active Model Book'}"]
        result_state = (
            self.last_result.status if self.last_result is not None else "not run"
        )
        state = [
            f"Active inference Model Book: {self.active_book.name} ({self.active_book.book_id})",
            f"Required numeric inputs: {', '.join(self.active_book.feature_columns)}",
            f"Saved outputs: {len(self.active_book.target_columns)}",
            f"Latest page prediction: {result_state}",
            f"Plot curves: {len(self.response_plot.state.curves)}",
            f"Prediction plot action: {self.prediction_plot_mode.get()}",
            "Plot tools: major/minor grid, engineering ticks, zoom, pan, reset, autoscale, hover crosshair, movable legend, markers, editable axes, and curve management",
            "Inference mode: one sample; no batch, CSV, or automatic history",
        ]
        if self.last_result is not None and self.last_result.success:
            state.append(
                "Prediction actions: View Raw Values or export JSON/curve CSV"
            )
        return state

    def refresh_theme(self) -> None:
        self.response_plot.refresh_theme()

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, padx=28, pady=(18, 12), sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="Inference",
            text_color=COLORS["ink"],
            font=FONTS["display"],
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        self.active_badge = ctk.CTkLabel(
            header,
            text="NO ACTIVE MODEL BOOK",
            height=26,
            corner_radius=13,
            fg_color=COLORS["surface_alt"],
            text_color=COLORS["muted"],
            font=FONTS["mono"],
        )
        self.active_badge.grid(row=0, column=1, sticky="e")

    def _build_content(self) -> None:
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.grid(row=1, column=0, padx=28, pady=(0, 10), sticky="nsew")
        content.grid_columnconfigure(0, weight=0, minsize=270)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        self.input_card = ctk.CTkFrame(
            content,
            fg_color=COLORS["surface"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["border"],
        )
        self.input_card.grid(row=0, column=0, padx=(0, 6), sticky="nsew")
        self.input_card.grid_columnconfigure(0, weight=1)
        self.input_card.grid_rowconfigure(2, weight=1)
        ctk.CTkLabel(
            self.input_card,
            text="NEW SAMPLE",
            text_color=COLORS["cyan"],
            font=FONTS["mono"],
            anchor="w",
        ).grid(row=0, column=0, padx=16, pady=(14, 3), sticky="ew")
        self.model_summary = ctk.CTkLabel(
            self.input_card,
            text="Select an active Model Book to begin.",
            text_color=COLORS["muted"],
            font=FONTS["body_small"],
            anchor="w",
            justify="left",
            wraplength=330,
        )
        self.model_summary.grid(row=1, column=0, padx=16, pady=(0, 8), sticky="ew")
        self.input_host = ctk.CTkFrame(self.input_card, fg_color="transparent")
        self.input_host.grid(row=2, column=0, padx=12, sticky="nsew")
        self.input_host.grid_columnconfigure(0, weight=1, uniform="input")
        self.input_host.grid_columnconfigure(1, weight=1, uniform="input")
        for row in range(4):
            self.input_host.grid_rowconfigure(row, weight=1, uniform="input-row")

        self.input_pager = ctk.CTkFrame(self.input_card, fg_color="transparent")
        self.input_pager.grid(row=3, column=0, padx=14, pady=(5, 0), sticky="ew")
        self.input_pager.grid_columnconfigure(1, weight=1)
        self.input_previous = ctk.CTkButton(
            self.input_pager,
            text="←",
            width=38,
            height=28,
            corner_radius=8,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            text_color=COLORS["ink"],
            command=lambda: self._change_input_page(-1),
        )
        self.input_previous.grid(row=0, column=0, sticky="w")
        self.input_page_label = ctk.CTkLabel(
            self.input_pager,
            text="Inputs 1–1 of 1",
            text_color=COLORS["subtle"],
            font=FONTS["caption"],
        )
        self.input_page_label.grid(row=0, column=1)
        self.input_next = ctk.CTkButton(
            self.input_pager,
            text="→",
            width=38,
            height=28,
            corner_radius=8,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            text_color=COLORS["ink"],
            command=lambda: self._change_input_page(1),
        )
        self.input_next.grid(row=0, column=2, sticky="e")
        self.input_pager.grid_remove()

        self.input_error = ctk.CTkLabel(
            self.input_card,
            text="",
            text_color=COLORS["danger"],
            font=FONTS["body_small"],
            anchor="w",
            justify="left",
            wraplength=330,
        )
        self.input_error.grid(row=4, column=0, padx=16, pady=(5, 0), sticky="ew")
        self.predict_button = ctk.CTkButton(
            self.input_card,
            text="Predict",
            height=38,
            corner_radius=11,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            text_color=COLORS["on_primary"],
            font=FONTS["button"],
            state="disabled",
            command=self._predict,
        )
        self.prediction_plot_mode = ctk.CTkSegmentedButton(
            self.input_card,
            values=["Replace current curve", "Add to plot"],
            height=30,
            corner_radius=8,
            fg_color=COLORS["control"],
            selected_color=COLORS["primary"],
            selected_hover_color=COLORS["primary_hover"],
            unselected_color=COLORS["control"],
            unselected_hover_color=COLORS["control_hover"],
            text_color=COLORS["ink"],
            font=("Segoe UI Semibold", 12),
        )
        self.prediction_plot_mode.grid(
            row=5,
            column=0,
            padx=16,
            pady=(7, 0),
            sticky="ew",
        )
        self.prediction_plot_mode.set("Replace current curve")
        self.predict_button.grid(
            row=6,
            column=0,
            padx=16,
            pady=(8, 14),
            sticky="ew",
        )

        self.result_card = ctk.CTkFrame(
            content,
            fg_color=COLORS["surface"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["border"],
        )
        self.result_card.grid(row=0, column=1, padx=(6, 0), sticky="nsew")
        self.result_card.grid_columnconfigure(0, weight=1)
        self.result_card.grid_rowconfigure(5, weight=1)
        ctk.CTkLabel(
            self.result_card,
            text="PREDICTION RESULT",
            text_color=COLORS["cyan"],
            font=FONTS["mono"],
            anchor="w",
        ).grid(row=0, column=0, padx=18, pady=(14, 3), sticky="ew")
        model_header = ctk.CTkFrame(
            self.result_card,
            fg_color=COLORS["primary_soft"],
            corner_radius=10,
        )
        model_header.grid(row=1, column=0, padx=16, pady=(2, 7), sticky="ew")
        model_header.grid_columnconfigure(0, weight=1)
        model_header.grid_columnconfigure(1, weight=0)
        self.result_model_name = ctk.CTkLabel(
            model_header,
            text="No active Model Book",
            text_color=COLORS["ink"],
            font=FONTS["card_title"],
            anchor="w",
        )
        self.result_model_name.grid(row=0, column=0, padx=(12, 6), pady=7, sticky="ew")
        self.result_model_type = ctk.CTkLabel(
            model_header,
            text="Select a Model Book from the library",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            anchor="w",
        )
        self.result_model_type.grid(row=0, column=1, padx=6, pady=7, sticky="e")
        self.model_info_button = ctk.CTkButton(
            model_header,
            text="Model Info",
            width=82,
            height=26,
            corner_radius=8,
            fg_color=COLORS["surface"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=("Segoe UI Semibold", 12),
            command=self._show_model_info,
        )
        self.model_info_button.grid(row=0, column=2, padx=(6, 8), pady=5)

        self.result_title = ctk.CTkLabel(
            self.result_card,
            text="Ready for one prediction",
            text_color=COLORS["ink"],
            font=FONTS["section"],
            anchor="w",
        )
        self.result_title.grid(row=2, column=0, padx=18, sticky="ew")
        self.result_summary = ctk.CTkLabel(
            self.result_card,
            text="Enter the required numeric inputs, then select Predict.",
            text_color=COLORS["muted"],
            font=FONTS["body_small"],
            anchor="w",
            justify="left",
            wraplength=480,
        )
        self.result_summary.grid(row=3, column=0, padx=18, pady=(2, 7), sticky="ew")

        self.summary_strip = ctk.CTkFrame(
            self.result_card,
            fg_color="transparent",
        )
        self.summary_strip.grid(row=4, column=0, padx=14, sticky="ew")
        for column in range(3):
            self.summary_strip.grid_columnconfigure(column, weight=1, uniform="result")
        for column, (key, label) in enumerate(
            (("count", "OUTPUT COUNT"), ("minimum", "MINIMUM"), ("maximum", "MAXIMUM"))
        ):
            self.result_summary_values[key] = self._result_metric(
                self.summary_strip,
                column,
                label,
            )

        inputs_used_card = ctk.CTkFrame(
            self.result_card,
            fg_color=COLORS["surface_alt"],
            corner_radius=10,
        )
        inputs_used_card.grid(row=5, column=0, padx=14, pady=(7, 7), sticky="ew")
        inputs_used_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            inputs_used_card,
            text="INPUTS USED",
            text_color=COLORS["cyan"],
            font=("Cascadia Mono", 11),
            anchor="w",
        ).grid(row=0, column=0, padx=10, pady=(6, 0), sticky="ew")
        self.inputs_used_value = ctk.CTkLabel(
            inputs_used_card,
            text="—",
            text_color=COLORS["ink"],
            font=FONTS["caption"],
            anchor="w",
            justify="left",
            wraplength=480,
        )
        self.inputs_used_value.grid(
            row=1,
            column=0,
            padx=10,
            pady=(1, 7),
            sticky="ew",
        )

        inputs_used_card.grid_remove()

        self.response_plot = ScientificPlotWorkbench(
            self.result_card,
            selection_changed=self._plot_selection_changed,
        )
        self.response_plot.grid(row=5, column=0, padx=14, pady=(7, 7), sticky="nsew")

        result_actions = ctk.CTkFrame(self.result_card, fg_color="transparent")
        result_actions.grid(row=6, column=0, padx=14, pady=(0, 14), sticky="ew")
        result_actions.grid_columnconfigure(0, weight=1)
        self.raw_values_button = ctk.CTkButton(
            result_actions,
            text="View Raw Values",
            width=132,
            height=34,
            corner_radius=10,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            state="disabled",
            command=self._view_raw_values,
        )
        self.raw_values_button.grid(row=0, column=1, padx=(0, 6), sticky="e")
        self.export_button = ctk.CTkButton(
            result_actions,
            text="Export Prediction",
            width=144,
            height=34,
            corner_radius=10,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            text_color=COLORS["on_primary"],
            font=FONTS["button"],
            state="disabled",
            command=self._export_prediction,
        )
        self.export_button.grid(row=0, column=2, sticky="e")

    def _result_metric(
        self,
        parent: ctk.CTkFrame,
        column: int,
        label: str,
    ) -> ctk.CTkLabel:
        card = ctk.CTkFrame(
            parent,
            fg_color=COLORS["surface_alt"],
            corner_radius=9,
        )
        card.grid(
            row=0,
            column=column,
            padx=(0 if column == 0 else 3, 0 if column == 2 else 3),
            sticky="nsew",
        )
        ctk.CTkLabel(
            card,
            text=label,
            text_color=COLORS["muted"],
            font=("Cascadia Mono", 10),
        ).grid(row=0, column=0, padx=8, pady=(5, 0))
        value = ctk.CTkLabel(
            card,
            text="—",
            text_color=COLORS["ink"],
            font=FONTS["card_title"],
        )
        value.grid(row=1, column=0, padx=8, pady=(0, 5))
        return value

    def _build_footer(self) -> None:
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, padx=28, pady=(0, 12), sticky="ew")
        footer.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(
            footer,
            text="←  Back to Model Library",
            width=184,
            height=36,
            corner_radius=11,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=FONTS["button"],
            command=lambda: self.app.show_page("library"),
        ).grid(row=0, column=0, sticky="w")
        self.footer_status = ctk.CTkLabel(
            footer,
            text="Single-sample prediction · no automatic history",
            text_color=COLORS["subtle"],
            font=FONTS["caption"],
        )
        self.footer_status.grid(row=0, column=1, sticky="e")
        self.inverse_design_button = ctk.CTkButton(
            footer,
            text="Inverse Design  →",
            width=148,
            height=36,
            corner_radius=11,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            text_color=COLORS["on_primary"],
            font=FONTS["button"],
            state="disabled",
            command=lambda: self.app.show_page("inverse_design"),
        )
        self.inverse_design_button.grid(row=0, column=2, padx=(12, 0), sticky="e")

    def _refresh(self) -> None:
        self._clear_input_fields()
        self.response_plot.clear()
        self.input_error.configure(text="")
        self.inputs_used_value.configure(text="—")
        for value_label in self.result_summary_values.values():
            value_label.configure(text="—")
        self.raw_values_button.configure(state="disabled")
        self.export_button.configure(state="disabled")
        self.result_title.configure(text="Ready for one prediction")
        self.result_summary.configure(
            text="Enter the required numeric inputs, then select Predict.",
            text_color=COLORS["muted"],
        )
        if self.active_book is None:
            self.active_badge.grid()
            self.result_model_name.configure(text="No active Model Book")
            self.result_model_type.configure(
                text="Select a Model Book from the library"
            )
            self.active_badge.configure(
                text="NO ACTIVE MODEL BOOK",
                fg_color=COLORS["warning_soft"],
                text_color=COLORS["warning"],
            )
            message = self.load_error or "Open a project to run inference."
            self.model_summary.configure(text=message)
            self.input_error.configure(text=message)
            self.predict_button.configure(text="Predict", state="disabled")
            self.prediction_plot_mode.configure(state="disabled")
            self.model_info_button.configure(state="disabled")
            self.inverse_design_button.configure(state="disabled")
            self.result_title.configure(text="Prediction unavailable")
            self.result_summary.configure(text=message, text_color=COLORS["danger"])
            return

        book = self.active_book
        self.prediction_plot_mode.configure(state="normal")
        self.model_info_button.configure(state="normal")
        self.inverse_design_button.configure(state="normal")
        self.result_model_name.configure(text=book.name)
        self.result_model_type.configure(
            text=f"{_model_type_label(book)} · {book.book_id}"
        )
        # The compact model strip already identifies the active book. Keep the
        # header badge only for the actionable no-active-book state.
        self.active_badge.grid_remove()
        self.model_summary.configure(
            text=(
                f"{len(book.feature_columns)} inputs → "
                f"{len(book.target_columns)} outputs"
            )
        )
        self._create_input_fields(book.feature_columns)
        self.predict_button.configure(text="Predict", state="normal")

    def _clear_input_fields(self) -> None:
        for child in self.input_host.winfo_children():
            child.destroy()
        self.input_entries.clear()
        self.input_shells.clear()
        self.input_pager.grid_remove()

    def _create_input_fields(self, feature_columns: list[str]) -> None:
        for name in feature_columns:
            shell = ctk.CTkFrame(
                self.input_host,
                fg_color=COLORS["surface_alt"],
                corner_radius=10,
            )
            shell.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(
                shell,
                text=name,
                text_color=COLORS["ink"],
                font=FONTS["body_small"],
                anchor="w",
            ).grid(row=0, column=0, padx=10, pady=(6, 2), sticky="ew")
            entry = ctk.CTkEntry(
                shell,
                placeholder_text="Numeric value",
                height=32,
                corner_radius=8,
                fg_color=COLORS["control"],
                border_color=COLORS["border"],
                text_color=COLORS["ink"],
                font=FONTS["body_small"],
            )
            entry.grid(row=1, column=0, padx=8, pady=(0, 7), sticky="ew")
            self.input_shells[name] = shell
            self.input_entries[name] = entry
        self._render_input_page()

    def _render_input_page(self) -> None:
        names = list(self.input_entries)
        page_count = max(1, (len(names) + INPUTS_PER_PAGE - 1) // INPUTS_PER_PAGE)
        self.input_page = min(self.input_page, page_count - 1)
        for shell in self.input_shells.values():
            shell.grid_remove()
        start = self.input_page * INPUTS_PER_PAGE
        visible = names[start : start + INPUTS_PER_PAGE]
        for index, name in enumerate(visible):
            self.input_shells[name].grid(
                row=index // 2,
                column=index % 2,
                padx=(0 if index % 2 == 0 else 4, 4 if index % 2 == 0 else 0),
                pady=4,
                sticky="nsew",
            )
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
        names = list(self.input_entries)
        page_count = max(1, (len(names) + INPUTS_PER_PAGE - 1) // INPUTS_PER_PAGE)
        self.input_page = max(0, min(self.input_page + offset, page_count - 1))
        self._render_input_page()

    def _input_values(self) -> dict[str, float] | None:
        values: dict[str, float] = {}
        for name, entry in self.input_entries.items():
            raw = entry.get().strip()
            if not raw:
                self.input_error.configure(text=f"Enter a value for {name}.")
                entry.focus_set()
                return None
            try:
                value = float(raw)
            except ValueError:
                self.input_error.configure(text=f"{name} must be a numeric value.")
                entry.focus_set()
                return None
            if not math.isfinite(value):
                self.input_error.configure(
                    text=f"{name} must be a finite numeric value."
                )
                entry.focus_set()
                return None
            values[name] = value
        return values

    def _predict(self) -> None:
        if self.prediction_in_progress or self.project is None or self.active_book is None:
            return
        values = self._input_values()
        if values is None:
            return
        self.input_error.configure(text="")
        self._set_prediction_busy(True)
        try:
            result = submit_inference_request(
                InferenceRequest(
                    model_book_id=self.active_book.book_id,
                    inputs=values,
                ),
                project_path=self.project.path,
            )
            self.last_result = result
            if result.success:
                self._show_success(result)
            else:
                self._show_failure(
                    result.error_message or "The active Model Book could not predict this sample."
                )
        except (TypeError, ValueError) as exc:
            self._show_failure(str(exc))
        except Exception:
            self._show_failure(
                "Prediction failed because an unexpected local error occurred."
            )
        finally:
            self._set_prediction_busy(False)

    def _set_prediction_busy(self, busy: bool) -> None:
        self.prediction_in_progress = busy
        if busy:
            self.raw_values_button.configure(state="disabled")
            self.export_button.configure(state="disabled")
        self.prediction_plot_mode.configure(
            state="disabled" if busy else "normal"
        )
        self.predict_button.configure(
            text="Predicting…" if busy else "Predict",
            state="disabled" if busy else "normal",
        )
        self.update_idletasks()

    def _show_success(self, result: InferenceResult) -> None:
        predictions = result.predictions
        values = tuple(float(value) for value in predictions.values())
        self.result_summary_values["count"].configure(text=str(len(values)))
        self.result_summary_values["minimum"].configure(text=f"{min(values):.6g}")
        self.result_summary_values["maximum"].configure(text=f"{max(values):.6g}")
        inputs_used = " · ".join(
            f"{name} = {value:.6g}"
            for name, value in _ordered_items(
                result.feature_order,
                result.input_values,
            )
        )
        self.inputs_used_value.configure(text=inputs_used or "No inputs recorded")
        self.result_summary.configure(text_color=COLORS["muted"])
        target_names = tuple(predictions)
        axis = self.active_book.output_axis if self.active_book is not None else None
        x_label = axis.display_label if axis is not None else "Output coordinate"
        x_values = (
            axis.values
            if axis is not None and len(axis.values) == len(target_names)
            else tuple(float(index) for index in range(1, len(target_names) + 1))
        )
        self.response_plot.add_curve(
            x_values=x_values,
            y_values=values,
            target_names=target_names,
            inputs=result.input_values,
            replace_selected=(
                self.prediction_plot_mode.get() == "Replace current curve"
            ),
            x_label=x_label,
        )
        if len(predictions) == 1:
            target, value = next(iter(predictions.items()))
            self.result_title.configure(text=target)
            self.result_summary.configure(
                text=f"Predicted value: {value:.10g}",
                font=FONTS["body_small"],
            )
        else:
            self.result_title.configure(
                text=f"Prediction completed · {len(predictions)} outputs"
            )
            self.result_summary.configure(
                text="Ordered response from the saved Model Book output interface.",
                font=FONTS["body_small"],
            )
        self.raw_values_button.configure(state="normal")
        self.export_button.configure(state="normal")
        assistant_closed = self._ensure_prediction_plot_visible()
        self.footer_status.configure(
            text=(
                f"Prediction complete · {result.model_book_name or result.model_book_id}"
                + (" · SnowBuddy closed to show plot" if assistant_closed else "")
            )
        )

    def _ensure_prediction_plot_visible(self) -> bool:
        """Recover plot width when the assistant leaves less than a usable canvas."""

        self.update_idletasks()
        canvas = self.response_plot.canvas
        if (
            canvas.winfo_ismapped()
            and canvas.winfo_width() >= MIN_USABLE_PREDICTION_PLOT_WIDTH
        ):
            return False
        if self.app.snowbuddy_collapsed:
            return False
        self.app.set_snowbuddy_collapsed(True)
        self.update_idletasks()
        self.response_plot.redraw()
        return True

    def _show_failure(self, message: str) -> None:
        for value_label in self.result_summary_values.values():
            value_label.configure(text="—")
        self.inputs_used_value.configure(text="—")
        self.raw_values_button.configure(state="disabled")
        self.export_button.configure(state="disabled")
        self.result_title.configure(text="Prediction failed")
        self.result_summary.configure(text=message, text_color=COLORS["danger"])
        self.footer_status.configure(text="Prediction did not complete")

    def _plot_selection_changed(self, curve: ScientificCurve | None) -> None:
        if curve is None:
            self.inputs_used_value.configure(text="—")
            return
        values = curve.y_values
        self.result_summary_values["count"].configure(text=str(len(values)))
        self.result_summary_values["minimum"].configure(text=f"{min(values):.6g}")
        self.result_summary_values["maximum"].configure(text=f"{max(values):.6g}")
        self.inputs_used_value.configure(
            text=" · ".join(
                f"{name} = {value:.6g}" for name, value in curve.inputs.items()
            )
            or "No inputs recorded"
        )

    def _show_model_info(self) -> None:
        book = self.active_book
        if book is None:
            return
        messagebox.showinfo(
            "Active Model Book",
            (
                f"Model Book: {book.name}\n"
                f"Book ID: {book.book_id}\n"
                f"Model type: {_model_type_label(book)}\n"
                f"Inputs: {len(book.feature_columns)}\n"
                f"Outputs: {len(book.target_columns)}"
            ),
            parent=self,
        )

    def _successful_result(self) -> InferenceResult | None:
        result = self.last_result
        if result is None or not result.success or not result.predictions:
            return None
        return result

    def _view_raw_values(self) -> None:
        result = self._successful_result()
        if result is None or self.active_book is None:
            return
        if (
            self.raw_values_dialog is not None
            and self.raw_values_dialog.winfo_exists()
        ):
            self.raw_values_dialog.focus_set()
            return
        self.raw_values_dialog = RawPredictionDialog(
            self,
            result,
            self.active_book,
        )

    def _export_prediction(self) -> None:
        result = self._successful_result()
        if result is None or self.active_book is None or self.project is None:
            return
        destination = filedialog.asksaveasfilename(
            parent=self,
            title="Export Prediction",
            initialdir=str(self.project.path / "inference"),
            initialfile=f"prediction_{self.active_book.book_id}.json",
            defaultextension=".json",
            filetypes=[
                ("JSON prediction", "*.json"),
                ("Curve CSV", "*.csv"),
                ("All files", "*.*"),
            ],
        )
        if not destination:
            return
        try:
            export_path = Path(destination)
            if export_path.suffix.lower() == ".csv":
                write_prediction_curve_csv(export_path, result, self.active_book)
            else:
                atomic_write_json(
                    export_path,
                    prediction_export_payload(result, self.active_book),
                )
        except (OSError, ValueError) as exc:
            messagebox.showerror(
                "Prediction export failed",
                f"The prediction could not be exported. {exc}",
                parent=self,
            )
            return
        self.footer_status.configure(
            text=f"Prediction exported · {Path(destination).name}"
        )
        messagebox.showinfo(
            "Prediction exported",
            f"Prediction saved to:\n{destination}",
            parent=self,
        )
