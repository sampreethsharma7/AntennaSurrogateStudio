"""Fixed, non-scrolling Model Library page for saved Model Books."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import customtkinter as ctk
from tkinter import messagebox

from studio.model_book import (
    ModelBook,
    ModelBookError,
    ModelBookLibrary,
    ModelBookLibraryEntry,
    load_model_library,
    set_active_model_book,
)
from studio.project_store import Project
from studio.theme import COLORS, FONTS

if TYPE_CHECKING:
    from studio.ui import StudioApp


LIBRARY_PAGE_SIZE = 5
COMPACT_INPUT_LIMIT = 6


class ModelLibraryPage(ctk.CTkFrame):
    """Browse saved Model Books, inspect metadata, and select the active book."""

    def __init__(self, parent: ctk.CTkFrame, app: "StudioApp"):
        super().__init__(parent, fg_color=COLORS["app_bg"], corner_radius=0)
        self.app = app
        self.project: Project | None = None
        self.library: ModelBookLibrary | None = None
        self.load_error: str | None = None
        self.selected_book_id: str | None = None
        self.current_page = 0
        self.book_rows: dict[str, dict[str, ctk.CTkBaseClass]] = {}
        self.summary_values: dict[str, ctk.CTkLabel] = {}
        self.metric_cards: dict[str, ctk.CTkFrame] = {}
        self.metric_values: dict[str, ctk.CTkLabel] = {}
        self.provenance_values: dict[str, ctk.CTkLabel] = {}
        self.provenance_expanded = False

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_header()
        self._build_content()
        self._build_footer()
        self._refresh()

    @property
    def selected_entry(self) -> ModelBookLibraryEntry | None:
        if self.library is None or self.selected_book_id is None:
            return None
        return next(
            (
                entry
                for entry in self.library.entries
                if entry.book_id == self.selected_book_id
            ),
            None,
        )

    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.selected_book_id = None
        self.current_page = 0
        self.reload()

    def reload(self) -> None:
        previous_selection = self.selected_book_id
        self.library = None
        self.load_error = None
        if self.project is not None:
            try:
                self.library = load_model_library(self.project.path)
            except ModelBookError as exc:
                self.load_error = str(exc)
        if self.library is not None and self.library.entries:
            available_ids = {entry.book_id for entry in self.library.entries}
            if previous_selection in available_ids:
                self.selected_book_id = previous_selection
            elif self.library.active_book_id in available_ids:
                self.selected_book_id = self.library.active_book_id
            else:
                self.selected_book_id = self.library.entries[-1].book_id
            display_entries = self._display_entries()
            selected_index = next(
                (
                    index
                    for index, entry in enumerate(display_entries)
                    if entry.book_id == self.selected_book_id
                ),
                0,
            )
            self.current_page = selected_index // LIBRARY_PAGE_SIZE
        else:
            self.selected_book_id = None
            self.current_page = 0
        self.provenance_expanded = False
        self._refresh()

    def describe_ui_state(self) -> list[str]:
        if self.load_error:
            return [f"Model Library error: {self.load_error}"]
        if self.library is None:
            return ["Model Library state: no project"]
        invalid_count = sum(not entry.is_valid for entry in self.library.entries)
        selected = self.selected_entry
        return [
            f"Model Books indexed: {len(self.library.entries)}",
            f"Valid Model Books: {self.library.valid_book_count}",
            f"Invalid Model Books: {invalid_count}",
            f"Active Model Book: {self.library.active_book_id or 'none'}",
            f"Selected Model Book: {selected.name if selected else 'none'}",
            "Model Library page scrolling: none; five books per page",
            "Model Book cards are selectable; the selected Model Book shows summary, "
            "performance, inputs, and optional provenance",
        ]

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, padx=28, pady=(18, 12), sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="Model Library",
            text_color=COLORS["ink"],
            font=FONTS["display"],
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        self.library_badge = ctk.CTkLabel(
            header,
            text="0 MODEL BOOKS",
            height=26,
            corner_radius=13,
            fg_color=COLORS["surface_alt"],
            text_color=COLORS["muted"],
            font=FONTS["mono"],
        )
        self.library_badge.grid(row=0, column=1, padx=(12, 8), sticky="e")
        self.refresh_button = ctk.CTkButton(
            header,
            text="Refresh",
            width=78,
            height=30,
            corner_radius=9,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            font=FONTS["caption"],
            command=self.reload,
        )
        self.refresh_button.grid(row=0, column=2, sticky="e")

    def _build_content(self) -> None:
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.grid(row=1, column=0, padx=28, pady=(0, 10), sticky="nsew")
        self.content.grid_columnconfigure(0, weight=2, uniform="library")
        self.content.grid_columnconfigure(1, weight=3, uniform="library")
        self.content.grid_rowconfigure(0, weight=1)
        self._build_saved_models_panel()
        self._build_selected_model_panel()
        self.empty_label = ctk.CTkLabel(
            self.content,
            text="",
            text_color=COLORS["muted"],
            font=FONTS["body"],
            justify="center",
        )

    def _build_saved_models_panel(self) -> None:
        self.list_card = ctk.CTkFrame(
            self.content,
            fg_color=COLORS["surface"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["border"],
        )
        self.list_card.grid(row=0, column=0, padx=(0, 6), sticky="nsew")
        self.list_card.grid_columnconfigure(0, weight=1)
        self.list_card.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(
            self.list_card,
            text="MODEL BOOKS",
            text_color=COLORS["cyan"],
            font=FONTS["mono"],
            anchor="w",
        ).grid(row=0, column=0, padx=16, pady=(14, 8), sticky="ew")
        self.list_host = ctk.CTkFrame(self.list_card, fg_color="transparent")
        self.list_host.grid(row=1, column=0, padx=10, sticky="nsew")
        self.list_host.grid_columnconfigure(0, weight=1)
        for row in range(LIBRARY_PAGE_SIZE):
            self.list_host.grid_rowconfigure(row, weight=1, uniform="book-row")

        pager = ctk.CTkFrame(self.list_card, fg_color="transparent")
        pager.grid(row=2, column=0, padx=12, pady=(8, 12), sticky="ew")
        pager.grid_columnconfigure(1, weight=1)
        self.previous_button = ctk.CTkButton(
            pager,
            text="←",
            width=38,
            height=28,
            corner_radius=8,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            text_color=COLORS["ink"],
            state="disabled",
            command=lambda: self._change_page(-1),
        )
        self.previous_button.grid(row=0, column=0, sticky="w")
        self.page_label = ctk.CTkLabel(
            pager,
            text="Page 1 of 1",
            text_color=COLORS["subtle"],
            font=FONTS["caption"],
        )
        self.page_label.grid(row=0, column=1)
        self.next_button = ctk.CTkButton(
            pager,
            text="→",
            width=38,
            height=28,
            corner_radius=8,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["control_hover"],
            text_color=COLORS["ink"],
            state="disabled",
            command=lambda: self._change_page(1),
        )
        self.next_button.grid(row=0, column=2, sticky="e")

    def _build_selected_model_panel(self) -> None:
        self.details_card = ctk.CTkFrame(
            self.content,
            fg_color=COLORS["surface"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["border"],
        )
        self.details_card.grid(row=0, column=1, padx=(6, 0), sticky="nsew")
        self.details_card.grid_columnconfigure(0, weight=1)
        self.details_card.grid_rowconfigure(2, weight=1)

        details_header = ctk.CTkFrame(self.details_card, fg_color="transparent")
        details_header.grid(row=0, column=0, padx=18, pady=(14, 2), sticky="ew")
        details_header.grid_columnconfigure(0, weight=1)
        self.details_title = ctk.CTkLabel(
            details_header,
            text="Selected Model Book",
            text_color=COLORS["ink"],
            font=FONTS["section"],
            anchor="w",
        )
        self.details_title.grid(row=0, column=0, sticky="ew")
        self.details_badge = ctk.CTkLabel(
            details_header,
            text="NO SELECTION",
            height=24,
            corner_radius=12,
            fg_color=COLORS["surface_alt"],
            text_color=COLORS["muted"],
            font=("Segoe UI Semibold", 12),
        )
        self.details_badge.grid(row=0, column=1, padx=(8, 0), sticky="e")
        self.details_subtitle = ctk.CTkLabel(
            self.details_card,
            text="Select a Model Book to inspect it.",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            anchor="w",
        )
        self.details_subtitle.grid(row=1, column=0, padx=18, sticky="ew")

        self.details_body = ctk.CTkFrame(self.details_card, fg_color="transparent")
        self.details_body.grid(row=2, column=0, padx=14, pady=(8, 6), sticky="nsew")
        self.details_body.grid_columnconfigure(0, weight=1)
        self._build_model_summary()
        self._build_performance_summary()
        self._build_required_inputs()
        self._build_provenance()

        self.details_error = ctk.CTkLabel(
            self.details_card,
            text="",
            text_color=COLORS["danger"],
            font=FONTS["body_small"],
            justify="left",
            wraplength=430,
        )
        self.set_active_button = ctk.CTkButton(
            self.details_card,
            text="Set as Active",
            height=36,
            corner_radius=11,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            text_color=COLORS["on_primary"],
            font=FONTS["button"],
            state="disabled",
            command=self._set_active,
        )
        self.set_active_button.grid(
            row=3,
            column=0,
            padx=18,
            pady=(0, 14),
            sticky="e",
        )

    def _build_model_summary(self) -> None:
        overview = ctk.CTkFrame(
            self.details_body,
            fg_color=COLORS["primary_soft"],
            corner_radius=12,
        )
        overview.grid(row=0, column=0, sticky="ew")
        for column in range(3):
            overview.grid_columnconfigure(column, weight=1, uniform="summary")
        fields = (
            ("model_type", "MODEL TYPE"),
            ("inputs", "INPUTS"),
            ("outputs", "OUTPUT"),
        )
        for column, (key, label) in enumerate(fields):
            self.summary_values[key] = self._summary_field(overview, column, label)

    def _summary_field(
        self,
        parent: ctk.CTkFrame,
        column: int,
        label: str,
    ) -> ctk.CTkLabel:
        card = ctk.CTkFrame(parent, fg_color="transparent")
        card.grid(row=0, column=column, padx=10, pady=8, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            card,
            text=label,
            text_color=COLORS["cyan"],
            font=("Cascadia Mono", 11),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")
        value = ctk.CTkLabel(
            card,
            text="—",
            text_color=COLORS["ink"],
            font=FONTS["card_title"],
            anchor="w",
            justify="left",
            wraplength=150,
        )
        value.grid(row=1, column=0, pady=(1, 0), sticky="ew")
        return value

    def _build_performance_summary(self) -> None:
        ctk.CTkLabel(
            self.details_body,
            text="PERFORMANCE",
            text_color=COLORS["cyan"],
            font=FONTS["mono"],
            anchor="w",
        ).grid(row=1, column=0, pady=(10, 4), sticky="ew")
        self.performance_panel = ctk.CTkFrame(
            self.details_body,
            fg_color="transparent",
        )
        self.performance_panel.grid(row=2, column=0, sticky="ew")
        for column in range(4):
            self.performance_panel.grid_columnconfigure(
                column,
                weight=1,
                uniform="metric",
            )
        metrics = (
            ("RMSE", "RMSE"),
            ("MAE", "MAE"),
            ("R²", "R²"),
            ("validation_rmse", "VALIDATION RMSE"),
        )
        for column, (key, label) in enumerate(metrics):
            card, value = self._metric_field(
                self.performance_panel,
                column,
                label,
            )
            self.metric_cards[key] = card
            self.metric_values[key] = value

    def _metric_field(
        self,
        parent: ctk.CTkFrame,
        column: int,
        label: str,
    ) -> tuple[ctk.CTkFrame, ctk.CTkLabel]:
        card = ctk.CTkFrame(
            parent,
            fg_color=COLORS["surface_alt"],
            corner_radius=10,
        )
        card.grid(
            row=0,
            column=column,
            padx=(0 if column == 0 else 3, 0 if column == 3 else 3),
            sticky="nsew",
        )
        ctk.CTkLabel(
            card,
            text=label,
            text_color=COLORS["muted"],
            font=("Cascadia Mono", 11),
        ).grid(row=0, column=0, padx=8, pady=(7, 0))
        value = ctk.CTkLabel(
            card,
            text="—",
            text_color=COLORS["ink"],
            font=FONTS["section"],
        )
        value.grid(row=1, column=0, padx=8, pady=(0, 7))
        return card, value

    def _build_required_inputs(self) -> None:
        inputs_card = ctk.CTkFrame(
            self.details_body,
            fg_color=COLORS["surface_alt"],
            corner_radius=10,
        )
        inputs_card.grid(row=3, column=0, pady=(10, 0), sticky="ew")
        inputs_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            inputs_card,
            text="REQUIRED INPUTS",
            text_color=COLORS["cyan"],
            font=("Cascadia Mono", 11),
            anchor="w",
        ).grid(row=0, column=0, padx=12, pady=(7, 0), sticky="ew")
        self.required_inputs_value = ctk.CTkLabel(
            inputs_card,
            text="—",
            text_color=COLORS["ink"],
            font=FONTS["body"],
            anchor="w",
            justify="left",
        )
        self.required_inputs_value.grid(
            row=1,
            column=0,
            padx=12,
            pady=(1, 8),
            sticky="ew",
        )
        self.view_inputs_button = ctk.CTkButton(
            inputs_card,
            text="View all inputs",
            width=102,
            height=26,
            corner_radius=8,
            fg_color="transparent",
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["cyan"],
            font=FONTS["caption"],
            command=self._show_all_inputs,
        )
        self.view_inputs_button.grid(
            row=0,
            column=1,
            rowspan=2,
            padx=(6, 10),
            pady=8,
            sticky="e",
        )
        self.view_inputs_button.grid_remove()

    def _build_provenance(self) -> None:
        self.provenance_card = ctk.CTkFrame(
            self.details_body,
            fg_color=COLORS["surface_alt"],
            corner_radius=10,
        )
        self.provenance_card.grid(row=4, column=0, pady=(8, 0), sticky="ew")
        self.provenance_card.grid_columnconfigure(0, weight=1)
        provenance_header = ctk.CTkFrame(
            self.provenance_card,
            fg_color="transparent",
        )
        provenance_header.grid(row=0, column=0, padx=12, pady=6, sticky="ew")
        provenance_header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            provenance_header,
            text="MODEL DETAILS",
            text_color=COLORS["cyan"],
            font=("Cascadia Mono", 11),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        self.provenance_button = ctk.CTkButton(
            provenance_header,
            text="View",
            width=58,
            height=24,
            corner_radius=8,
            fg_color="transparent",
            hover_color=COLORS["control_hover"],
            text_color=COLORS["cyan"],
            font=FONTS["caption"],
            command=self._toggle_provenance,
        )
        self.provenance_button.grid(row=0, column=1, sticky="e")
        self.provenance_body = ctk.CTkFrame(
            self.provenance_card,
            fg_color="transparent",
        )
        self.provenance_body.grid(
            row=1,
            column=0,
            padx=4,
            pady=(0, 8),
            sticky="ew",
        )
        self.provenance_body.grid_columnconfigure(0, weight=1, uniform="provenance")
        self.provenance_body.grid_columnconfigure(1, weight=1, uniform="provenance")
        fields = (
            ("source_run", "SOURCE RUN"),
            ("created", "CREATED"),
            ("training", "TRAINING"),
            ("parameters", "PARAMETERS USED"),
            ("dataset", "DATASET FINGERPRINT"),
            ("version", "MODEL BOOK VERSION"),
        )
        for index, (key, label) in enumerate(fields):
            self.provenance_values[key] = self._provenance_field(
                self.provenance_body,
                index // 2,
                index % 2,
                label,
            )
        self.provenance_body.grid_remove()

    def _provenance_field(
        self,
        parent: ctk.CTkFrame,
        row: int,
        column: int,
        label: str,
    ) -> ctk.CTkLabel:
        field = ctk.CTkFrame(parent, fg_color="transparent")
        field.grid(row=row, column=column, padx=8, pady=3, sticky="nsew")
        field.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            field,
            text=label,
            text_color=COLORS["subtle"],
            font=("Cascadia Mono", 10),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")
        value = ctk.CTkLabel(
            field,
            text="—",
            text_color=COLORS["ink"],
            font=FONTS["caption"],
            anchor="w",
            justify="left",
            wraplength=220,
        )
        value.grid(row=1, column=0, sticky="ew")
        return value

    def _build_footer(self) -> None:
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, padx=28, pady=(0, 12), sticky="ew")
        footer.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(
            footer,
            text="←  Back to Training Results",
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
        ).grid(row=0, column=0, sticky="w")
        self.footer_status = ctk.CTkLabel(
            footer,
            text="Model Books stay inside this project",
            text_color=COLORS["subtle"],
            font=FONTS["caption"],
        )
        self.footer_status.grid(row=0, column=1, sticky="e")

    def _refresh(self) -> None:
        self.empty_label.grid_forget()
        self.list_card.grid()
        self.details_card.grid()
        if self.project is None:
            self._show_empty("Open a project to view its Model Books.")
            return
        if self.load_error:
            self._show_empty(
                "The Model Library could not be loaded.\n" f"{self.load_error}"
            )
            return
        if self.library is None or not self.library.entries:
            self.library_badge.configure(text="0 MODEL BOOKS")
            self._show_empty(
                "No Model Books are saved in this project yet.\n"
                "Open a completed Training Result and select Create Model Book."
            )
            return

        total = len(self.library.entries)
        self.library_badge.configure(
            text=f"{total} MODEL BOOK{'S' if total != 1 else ''}",
            fg_color=COLORS["primary_soft"],
            text_color=COLORS["cyan"],
        )
        self._render_list()
        self._render_details()

    def _show_empty(self, message: str) -> None:
        self.list_card.grid_remove()
        self.details_card.grid_remove()
        self.empty_label.configure(text=message)
        self.empty_label.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.footer_status.configure(
            text="Model Library unavailable" if self.load_error else "No Model Books"
        )

    def _display_entries(self) -> list[ModelBookLibraryEntry]:
        if self.library is None:
            return []
        return list(reversed(self.library.entries))

    def _render_list(self) -> None:
        for child in self.list_host.winfo_children():
            child.destroy()
        self.book_rows.clear()
        entries = self._display_entries()
        page_count = max(1, (len(entries) + LIBRARY_PAGE_SIZE - 1) // LIBRARY_PAGE_SIZE)
        self.current_page = min(self.current_page, page_count - 1)
        start = self.current_page * LIBRARY_PAGE_SIZE
        visible = entries[start : start + LIBRARY_PAGE_SIZE]
        for row, entry in enumerate(visible):
            selected = entry.book_id == self.selected_book_id
            frame = ctk.CTkFrame(
                self.list_host,
                fg_color=(
                    COLORS["warning_soft"]
                    if not entry.is_valid
                    else COLORS["primary_soft"]
                    if selected
                    else COLORS["surface_alt"]
                ),
                corner_radius=10,
                border_width=1,
                border_color=(
                    COLORS["danger"]
                    if not entry.is_valid
                    else COLORS["primary"]
                    if selected
                    else COLORS["border"]
                ),
                cursor="hand2",
            )
            frame.grid(row=row, column=0, pady=3, sticky="nsew")
            frame.grid_columnconfigure(0, weight=1)
            name_label = ctk.CTkLabel(
                frame,
                text=entry.name,
                text_color=COLORS["ink"],
                font=FONTS["card_title"],
                anchor="w",
                cursor="hand2",
            )
            name_label.grid(row=0, column=0, padx=(11, 6), pady=(6, 0), sticky="ew")

            model_type = "Saved Model Book"
            metrics = "Metadata unavailable"
            interface = "Cannot inspect this Model Book"
            if entry.book is not None:
                book = entry.book
                model_type = _model_type_label(book)
                metrics = _card_performance_summary(book)
                interface = _interface_summary(book)
            model_label = ctk.CTkLabel(
                frame,
                text=model_type,
                text_color=COLORS["muted"],
                font=("Segoe UI", 12),
                anchor="w",
                cursor="hand2",
            )
            model_label.grid(row=1, column=0, padx=(11, 6), sticky="ew")
            metrics_label = ctk.CTkLabel(
                frame,
                text=metrics,
                text_color=COLORS["ink"],
                font=("Segoe UI Semibold", 12),
                anchor="w",
                cursor="hand2",
            )
            metrics_label.grid(row=2, column=0, padx=(11, 6), sticky="ew")
            interface_label = ctk.CTkLabel(
                frame,
                text=interface,
                text_color=COLORS["muted"],
                font=("Segoe UI", 12),
                anchor="w",
                cursor="hand2",
            )
            interface_label.grid(
                row=3,
                column=0,
                padx=(11, 6),
                pady=(0, 6),
                sticky="ew",
            )
            status_text = (
                "INVALID"
                if not entry.is_valid
                else "ACTIVE"
                if entry.is_active
                else "SELECTED"
                if selected
                else "SAVED"
            )
            status_label = ctk.CTkLabel(
                frame,
                text=status_text,
                width=62,
                height=20,
                corner_radius=10,
                fg_color=(
                    COLORS["warning_soft"]
                    if not entry.is_valid
                    else COLORS["success_soft"]
                    if entry.is_active
                    else COLORS["surface"]
                ),
                text_color=(
                    COLORS["danger"]
                    if not entry.is_valid
                    else COLORS["success"]
                    if entry.is_active
                    else COLORS["muted"]
                ),
                font=("Segoe UI Semibold", 11),
                cursor="hand2",
            )
            status_label.grid(row=0, column=1, padx=(0, 8), pady=(7, 2))
            widgets = (
                frame,
                name_label,
                model_label,
                metrics_label,
                interface_label,
                status_label,
            )
            self._bind_book_selection(widgets, entry.book_id)
            self.book_rows[entry.book_id] = {
                "frame": frame,
                "name": name_label,
                "model_type": model_label,
                "metrics": metrics_label,
                "interface": interface_label,
                "summary": metrics_label,
                "status": status_label,
            }

        self.page_label.configure(text=f"Page {self.current_page + 1} of {page_count}")
        self.previous_button.configure(
            state="normal" if self.current_page > 0 else "disabled"
        )
        self.next_button.configure(
            state="normal" if self.current_page + 1 < page_count else "disabled"
        )

    def _bind_book_selection(
        self,
        widgets: tuple[ctk.CTkBaseClass, ...],
        book_id: str,
    ) -> None:
        for widget in widgets:
            widget.bind(
                "<Button-1>",
                lambda _event, value=book_id: self._open_entry(value),
            )

    def _render_details(self) -> None:
        entry = self.selected_entry
        self.details_error.grid_forget()
        self.details_body.grid()
        if entry is None:
            self.details_title.configure(text="Selected Model Book")
            self.details_subtitle.configure(text="Select a Model Book to inspect it.")
            self.details_badge.configure(
                text="NO SELECTION",
                fg_color=COLORS["surface_alt"],
                text_color=COLORS["muted"],
            )
            for value in self.summary_values.values():
                value.configure(text="—")
            for value in self.metric_values.values():
                value.configure(text="—")
            self.required_inputs_value.configure(text="—")
            self.set_active_button.configure(text="Set as Active", state="disabled")
            return
        if not entry.is_valid or entry.book is None:
            self.details_title.configure(text=entry.name)
            self.details_subtitle.configure(text=entry.book_id)
            self.details_badge.configure(
                text="INVALID",
                fg_color=COLORS["warning_soft"],
                text_color=COLORS["danger"],
            )
            self.details_body.grid_remove()
            self.details_error.configure(
                text=(
                    "This Model Book cannot be opened.\n"
                    f"{entry.error_message or 'The saved book is invalid.'}"
                )
            )
            self.details_error.grid(row=2, column=0, padx=20, pady=24, sticky="nw")
            self.set_active_button.configure(
                text="Set as Active",
                state="disabled",
            )
            self.set_active_button.grid_remove()
            self.footer_status.configure(text="Invalid Model Book · review saved files")
            return

        book = entry.book
        self.details_title.configure(text=book.name)
        self.details_subtitle.configure(text=_prediction_summary(book))
        self.details_badge.configure(
            text="✓ Active Model Book" if entry.is_active else "Selected Model Book",
            fg_color=(
                COLORS["success_soft"] if entry.is_active else COLORS["primary_soft"]
            ),
            text_color=(COLORS["success"] if entry.is_active else COLORS["cyan"]),
        )
        self.summary_values["model_type"].configure(text=_model_type_label(book))
        self.summary_values["inputs"].configure(
            text=_count_label(len(book.feature_columns), "input")
        )
        self.summary_values["outputs"].configure(
            text=_book_output_summary(book)
        )
        for key in ("RMSE", "MAE", "R²"):
            self.metric_values[key].configure(text=_format_metric(book.test_metrics[key]))
            self.metric_cards[key].grid()
        validation_rmse = book.validation_metrics.get("RMSE")
        if validation_rmse is None:
            self.metric_cards["validation_rmse"].grid_remove()
            self.performance_panel.grid_columnconfigure(3, weight=0)
        else:
            self.metric_values["validation_rmse"].configure(
                text=_format_metric(validation_rmse)
            )
            self.performance_panel.grid_columnconfigure(3, weight=1)
            self.metric_cards["validation_rmse"].grid()

        self.required_inputs_value.configure(
            text=_compact_input_summary(book.feature_columns)
        )
        if len(book.feature_columns) > COMPACT_INPUT_LIMIT:
            self.view_inputs_button.grid()
        else:
            self.view_inputs_button.grid_remove()

        self.provenance_values["source_run"].configure(text=book.source_run_id)
        self.provenance_values["created"].configure(
            text=_format_timestamp(book.created_at)
        )
        search = f" · {book.search_level.title()}" if book.search_level else ""
        self.provenance_values["training"].configure(
            text=f"{book.training_mode.title()}{search}"
        )
        self.provenance_values["parameters"].configure(
            text=_format_parameters(book.parameters_used)
        )
        self.provenance_values["dataset"].configure(
            text=book.dataset_fingerprint
        )
        self.provenance_values["version"].configure(text=book.version)
        self._set_provenance_expanded(False)

        if entry.is_active:
            self.set_active_button.configure(
                text="Set as Active",
                state="disabled",
            )
            self.set_active_button.grid_remove()
        else:
            self.set_active_button.configure(text="Set as Active", state="normal")
            self.set_active_button.grid()
        self.footer_status.configure(
            text=(
                "Ready for inference"
                if entry.is_active
                else f"Selected · {book.name}"
            )
        )

    def _open_entry(self, book_id: str) -> None:
        self.selected_book_id = book_id
        self.provenance_expanded = False
        self._render_list()
        self._render_details()

    def _set_active(self) -> None:
        entry = self.selected_entry
        if self.project is None or entry is None or not entry.is_valid:
            return
        self.set_active_button.configure(state="disabled", text="Saving…")
        self.update_idletasks()
        try:
            book = set_active_model_book(self.project.path, entry.book_id)
        except ModelBookError as exc:
            messagebox.showerror(
                "Could not select Model Book",
                str(exc),
                parent=self,
            )
            self.set_active_button.configure(state="normal", text="Set as Active")
            return
        self.project = self.app.update_current_project({})
        self.selected_book_id = book.book_id
        self.reload()
        self.footer_status.configure(text="Ready for inference")

    def _toggle_provenance(self) -> None:
        self._set_provenance_expanded(not self.provenance_expanded)

    def _set_provenance_expanded(self, expanded: bool) -> None:
        self.provenance_expanded = expanded
        if expanded:
            self.provenance_body.grid()
            self.provenance_button.configure(text="Hide")
        else:
            self.provenance_body.grid_remove()
            self.provenance_button.configure(text="View")

    def _show_all_inputs(self) -> None:
        entry = self.selected_entry
        if entry is None or entry.book is None:
            return
        messagebox.showinfo(
            "Required model inputs",
            "\n".join(entry.book.feature_columns),
            parent=self,
        )

    def _change_page(self, offset: int) -> None:
        if self.library is None:
            return
        page_count = max(
            1,
            (len(self.library.entries) + LIBRARY_PAGE_SIZE - 1)
            // LIBRARY_PAGE_SIZE,
        )
        self.current_page = max(0, min(self.current_page + offset, page_count - 1))
        visible = self._display_entries()[
            self.current_page
            * LIBRARY_PAGE_SIZE : (self.current_page + 1)
            * LIBRARY_PAGE_SIZE
        ]
        if visible:
            self.selected_book_id = visible[0].book_id
        self.provenance_expanded = False
        self._render_list()
        self._render_details()


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


def _format_parameters(parameters: dict[str, object]) -> str:
    if "weights" in parameters and isinstance(parameters["weights"], dict):
        return "Weights · " + " · ".join(
            f"{name.replace('_', ' ').title()}={float(weight):.1%}"
            for name, weight in parameters["weights"].items()
        )
    if "hidden_layer_sizes" in parameters:
        return (
            f"layers={parameters['hidden_layer_sizes']} · "
            f"activation={parameters.get('activation')} · "
            f"learning_rate={parameters.get('learning_rate_init')} · "
            f"batch_size={parameters.get('batch_size')} · "
            f"epochs={parameters.get('max_iter')}"
        )
    if "n_estimators" in parameters:
        return (
            f"n_estimators={parameters['n_estimators']} · "
            f"max_depth={parameters.get('max_depth')} · "
            f"learning_rate={parameters.get('learning_rate')} · "
            f"{max(0, len(parameters) - 3)} more fixed settings"
        )
    return " · ".join(f"{name}={value}" for name, value in parameters.items())


def _count_label(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _output_count_summary(target_columns: list[str]) -> str:
    count = len(target_columns)
    if count == 1:
        return "1 output"
    return f"{count} output variables"


def _book_output_summary(book: ModelBook) -> str:
    base = _output_count_summary(book.target_columns)
    axis = book.output_axis
    if (
        axis is None
        or axis.source != "target_columns"
        or len(axis.values) < 2
    ):
        return base
    unit = f" {axis.unit}" if axis.unit else ""
    return (
        f"{len(book.target_columns)} outputs\n"
        f"{axis.label} {min(axis.values):g} to {max(axis.values):g}{unit}"
    )


def _prediction_summary(book: ModelBook) -> str:
    if len(book.target_columns) == 1:
        return f"Predicts {book.target_columns[0]} · 1 output"
    axis = book.output_axis
    if axis is not None and axis.source == "target_columns":
        return f"Predicts an ordered {axis.display_label} response · {len(book.target_columns)} outputs"
    return f"Predicts {len(book.target_columns)} output variables"


def _compact_input_summary(feature_columns: list[str]) -> str:
    if len(feature_columns) <= COMPACT_INPUT_LIMIT:
        return ", ".join(feature_columns)
    visible = ", ".join(feature_columns[:COMPACT_INPUT_LIMIT])
    return f"{visible} · +{len(feature_columns) - COMPACT_INPUT_LIMIT} more"


def _format_metric(value: float) -> str:
    return f"{value:.6g}"


def _card_performance_summary(book: ModelBook) -> str:
    return (
        f"RMSE {_format_metric(book.test_metrics['RMSE'])} · "
        f"R² {_format_metric(book.test_metrics['R²'])}"
    )


def _interface_summary(book: ModelBook) -> str:
    return (
        f"{_count_label(len(book.feature_columns), 'input')} → "
        f"{_count_label(len(book.target_columns), 'output')}"
    )


def _format_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value or "Unknown"
    return parsed.strftime("%Y-%m-%d %H:%M UTC")
