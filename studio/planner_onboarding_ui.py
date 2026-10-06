"""First-run setup for the Text-to-CAD cloud planner.

An RF engineer should be able to paste a key, have it checked against the
provider, and get on with designing. Nothing here asks the user to know what a
`.env` file is, what an endpoint is, or what an HTTP status means: the dialog
takes a key, verifies it for real, and either continues or says one plain
sentence about what to fix.

The dialog deliberately stays small. Pricing, privacy and the other providers
live behind a question-mark button that opens a separate help window, because
an onboarding screen that opens with configuration documentation is one the
user closes -- and because that material is far too tall to unfold inside a
dialog that still has to fit above a laptop's taskbar.

Every wrapped paragraph takes its wrap width from the width it is actually
given on screen rather than from a constant, so no message can be clipped by a
window that was sized before that message existed.
"""

from __future__ import annotations

import threading
import tkinter as tk
import webbrowser
from dataclasses import dataclass
from typing import Callable

import customtkinter as ctk

from studio.planner_credentials import (
    CLOUD_PROVIDERS,
    CREDENTIAL_STORE,
    LOCAL_OLLAMA,
    RECOMMENDED_CLOUD_PROVIDER,
    CredentialStoreUnavailable,
    credential_store_available,
    delete_credential,
    provider as provider_facts,
    resolve_credential,
    store_credential,
)
from studio.planner_model_discovery import (
    PROVIDER_LABELS,
    CredentialVerification,
    ModelDiscoveryError,
    PlannerModelDiscovery,
    friendly_discovery_message,
    verify_cloud_credential,
)
from studio.theme import COLORS, FONTS, column_safe_width


SETUP_TITLE = "Set up Text-to-CAD"
SETUP_INTRO = (
    "Text-to-CAD uses a cloud AI model to interpret your antenna design requests."
)
VERIFIED_HEADLINE = "✓ API key verified"
VERIFIED_DETAIL = "Text-to-CAD is ready."
HELP_TITLE = "About Text-to-CAD and your API key"

# The width a body paragraph is written to, and the margin either side of it.
# The window is sized to hold these, so wrapping normally needs no correction.
CONTENT_WRAP = 420
BODY_PADX = 26
# A wrapped line stops this many logical pixels short of the width it was
# given, so a rounded-up glyph never touches the border it wraps against.
WRAP_MARGIN = 8
# Below this the column is too narrow to read, so we would rather leave the
# declared wrap alone than shrink into a ladder of single words.
MIN_WRAPLENGTH = 180
# The help window asks for no more of the screen than this.
HELP_SCREEN_FRACTION = 0.72
# Width a help paragraph is written to, and the margin either side of it.
HELP_WRAP = 380
HELP_PADX = 22

# Kept out of the main dialog on purpose; shown only when the user asks.
HELP_SECTIONS: tuple[tuple[str, str], ...] = (
    (
        "Where the key is stored",
        "The key is saved on this computer using the operating system's own "
        "credential store, under the name Antenna Surrogate Studio. It is not "
        "written into your project, your logs, or any file in the repository, "
        "and the Studio never shows it again after you save it.",
    ),
    (
        "Where the key is sent",
        "Only to the provider you choose, to authenticate your own requests. "
        "The Studio does not send it anywhere else.",
    ),
    (
        "What the provider receives",
        "When you use a cloud planner, your design request, the solver-neutral "
        "design state and the tool manifest are sent to that provider, as the "
        "User Manual already describes. Your datasets, trained surrogate "
        "models, CST output files and other project files are not uploaded by "
        "this feature. The provider's own retention and training policies "
        "still apply to what is sent.",
    ),
    (
        "Cost",
        "Text-to-CAD uses your own cloud-provider account. Provider pricing can "
        "change. For initial testing, start with a small spending limit or "
        "prepaid balance where the provider supports it, then increase it only "
        "if needed.",
    ),
    (
        "Working without a cloud account",
        "Local Ollama runs entirely on this computer and needs no API key. "
        "Local models are smaller, so they interpret fewer design requests "
        "correctly than the cloud planners.",
    ),
)


def physical_width(widget: ctk.CTkBaseClass, logical: int) -> int:
    """Logical pixels as the pixels they occupy at the current scaling.

    ``column_safe_width`` converts the other way; geometry is set in real
    pixels while a wraplength is written in logical ones, and the two have to
    be compared somewhere.
    """

    from customtkinter import ScalingTracker

    try:
        scaling = float(ScalingTracker.get_widget_scaling(widget))
    except Exception:  # pragma: no cover - scaling is unavailable before Tk
        scaling = 1.0
    return int(logical * max(0.1, scaling))


def fit_wrapping(
    entries: list[tuple[ctk.CTkLabel, int]], *, available: int | None = None
) -> None:
    """Shrink each label's wrap to the width that label was actually given.

    A wraplength is a logical measurement that CustomTkinter multiplies by the
    widget scaling, while the width a label ends up with comes from the window
    around it.  Declaring both independently is how a message ends up wrapping
    at 453px inside 415px of window and losing its last character, so the
    declared value is treated as a maximum and the real one is measured.

    Callers that already know the interior width of the window pass it, since
    a label reports its own width only once the window has been mapped, and a
    wrap chosen before that would be computed against a placeholder.
    """

    for label, declared in entries:
        try:
            width = label.winfo_width() if available is None else available
        except tk.TclError:  # pragma: no cover - the dialog is going away
            continue
        if width <= 40:
            # Not laid out yet; the declared wrap still applies.
            continue
        logical = column_safe_width(label, width) - WRAP_MARGIN
        target = max(MIN_WRAPLENGTH, min(declared, logical))
        if int(label.cget("wraplength")) != target:
            label.configure(wraplength=target)


class TextToCadHelpDialog(ctk.CTkToplevel):
    """The storage, privacy, cost and provider notes, in a window of their own.

    These five sections run to roughly 750px of text.  Unfolding them inside
    the setup dialog pushed it past the bottom of a 1366x768 laptop screen, and
    capping the panel short enough to fit left a letterbox nobody could read.
    """

    def __init__(self, parent: ctk.CTkBaseClass) -> None:
        super().__init__(parent, fg_color=COLORS["surface"])
        self._wrapping: list[tuple[ctk.CTkLabel, int]] = []
        self.title(HELP_TITLE)
        self.transient(parent.winfo_toplevel())
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build()
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Configure>", self._reflow)
        self.after(0, self._fit_to_screen)

    def _build(self) -> None:
        # A scrollable frame reports neither its content's width nor its
        # height, so it is told both. Left to itself it opens as a 287x352
        # letterbox showing a third of one section.
        self.sections = ctk.CTkScrollableFrame(
            self,
            width=HELP_WRAP + 2 * WRAP_MARGIN,
            fg_color=COLORS["surface"],
            corner_radius=0,
        )
        self.sections.grid(
            row=0, column=0, padx=(HELP_PADX, 8), pady=(18, 0), sticky="nsew"
        )
        self.sections.grid_columnconfigure(0, weight=1)
        for index, (heading, detail) in enumerate(HELP_SECTIONS):
            ctk.CTkLabel(
                self.sections,
                text=heading,
                text_color=COLORS["ink"],
                font=FONTS["caption"],
                anchor="w",
            ).grid(row=index * 2, column=0, pady=(14 if index else 0, 2), sticky="ew")
            body = ctk.CTkLabel(
                self.sections,
                text=detail,
                text_color=COLORS["muted"],
                font=FONTS["caption"],
                wraplength=HELP_WRAP,
                justify="left",
                anchor="w",
            )
            body.grid(row=index * 2 + 1, column=0, sticky="ew")
            self._wrapping.append((body, HELP_WRAP))

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=1, column=0, padx=22, pady=(12, 16), sticky="ew")
        footer.grid_columnconfigure(0, weight=1)
        self.close_button = ctk.CTkButton(
            footer,
            text="Close",
            width=90,
            height=30,
            font=FONTS["button"],
            fg_color=COLORS["surface"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            command=self.close,
        )
        self.close_button.grid(row=0, column=1, sticky="e")

    def _reflow(self, _event: object = None) -> None:
        fit_wrapping(self._wrapping, available=physical_width(self, HELP_WRAP))

    def _fit_to_screen(self) -> None:
        if not self.winfo_exists():
            return
        self.update_idletasks()
        fit_wrapping(self._wrapping, available=physical_width(self, HELP_WRAP))
        self.update_idletasks()
        # Ask for the whole thing first, then take back whatever the screen
        # cannot spare. Measuring the window once it holds its real content is
        # more reliable than predicting how much of it the footer will use.
        content = sum(
            child.winfo_reqheight() for child in self.sections.winfo_children()
        ) + 6 * len(HELP_SECTIONS)
        self.sections.configure(
            height=column_safe_width(self.sections, content)
        )
        self.update_idletasks()
        overflow = self.winfo_reqheight() - int(
            self.winfo_screenheight() * HELP_SCREEN_FRACTION
        )
        if overflow > 0:
            self.sections.configure(
                height=column_safe_width(self.sections, max(200, content - overflow))
            )
            self.update_idletasks()
        width = self.winfo_reqwidth()
        height = self.winfo_reqheight()
        parent = self.master.winfo_toplevel()
        x = parent.winfo_rootx() + max(0, (parent.winfo_width() - width) // 2)
        y = parent.winfo_rooty() + max(0, (parent.winfo_height() - height) // 3)
        self.geometry(f"{width}x{height}+{max(0, x)}+{max(0, y)}")
        self.update_idletasks()
        fit_wrapping(self._wrapping, available=physical_width(self, HELP_WRAP))

    def close(self) -> None:
        if not self.winfo_exists():
            return
        opener = self.master
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()
        # Give the modal grab back to the setup dialog that opened this window,
        # otherwise the user is left with a dialog that ignores the keyboard.
        try:
            if opener.winfo_exists():
                opener.grab_set()
        except Exception:
            pass


@dataclass(frozen=True, slots=True)
class SetupOutcome:
    """What the user settled on, for the page that opened the dialog."""

    provider: str | None = None
    verified: bool = False

    @property
    def completed(self) -> bool:
        return self.provider is not None


class TextToCadSetupDialog(ctk.CTkToplevel):
    """Collect, verify and store one cloud planner credential."""

    def __init__(
        self,
        parent: ctk.CTkBaseClass,
        *,
        provider: str = RECOMMENDED_CLOUD_PROVIDER,
        manage: bool = False,
        discovery: PlannerModelDiscovery | None = None,
        credential_backend: object | None = None,
        browser: Callable[[str], object] | None = None,
    ) -> None:
        super().__init__(parent, fg_color=COLORS["surface"])
        self._discovery = discovery
        self._backend = credential_backend
        self._browser = browser or webbrowser.open
        self._manage = manage
        self._provider = provider if provider in CLOUD_PROVIDERS else RECOMMENDED_CLOUD_PROVIDER
        self.outcome = SetupOutcome()
        self._verifying = False
        self._help_window: TextToCadHelpDialog | None = None
        self._advanced_visible = manage
        self._wrapping: list[tuple[ctk.CTkLabel, int]] = []
        self._status_is_error = False

        self.title("Text-to-CAD setup" if not manage else "Text-to-CAD API keys")
        self.resizable(False, False)
        self.transient(parent.winfo_toplevel())

        self.provider_var = ctk.StringVar(value=PROVIDER_LABELS[self._provider])
        self.key_var = ctk.StringVar(value="")
        self.status_var = ctk.StringVar(value="")

        self._build()
        self._refresh_provider_copy()
        self._refresh_saved_keys()
        # Editing the key clears a stale rejection. The callback is told only
        # that the variable changed; it never reads the value.
        self.key_var.trace_add("write", self._key_edited)
        self.bind("<Configure>", self._reflow)
        self.grab_set()
        self.key_entry.focus_set()
        self.after(0, self._centre_on_parent)

    # ---------------------------------------------------------------- building

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=0, column=0, padx=BODY_PADX, pady=(24, 10), sticky="nsew")
        # Reserve a column wide enough for a full-width paragraph, so the
        # window *asks* to be that wide. Widening it afterwards with geometry()
        # does not work on a window that is not resizable and not yet mapped,
        # which is how a 453px paragraph ended up inside 415px of dialog.
        # grid minsize is in real pixels while a wraplength is in logical ones.
        body.grid_columnconfigure(
            0, weight=1, minsize=physical_width(self, CONTENT_WRAP + WRAP_MARGIN)
        )

        ctk.CTkLabel(
            body,
            text=SETUP_TITLE,
            text_color=COLORS["ink"],
            font=FONTS["title"],
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")
        intro = ctk.CTkLabel(
            body,
            text=SETUP_INTRO,
            text_color=COLORS["muted"],
            font=FONTS["body_small"],
            wraplength=CONTENT_WRAP,
            justify="left",
            anchor="w",
        )
        intro.grid(row=1, column=0, pady=(6, 12), sticky="ew")
        self._wrapping.append((intro, CONTENT_WRAP))

        self.recommendation_label = ctk.CTkLabel(
            body,
            text="",
            text_color=COLORS["ink"],
            font=FONTS["body_small"],
            anchor="w",
        )
        self.recommendation_label.grid(row=2, column=0, sticky="ew")

        ctk.CTkLabel(
            body,
            text="API key",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            anchor="w",
        ).grid(row=3, column=0, pady=(12, 3), sticky="ew")
        self.key_entry = ctk.CTkEntry(
            body,
            textvariable=self.key_var,
            # Masked like a password. CTkEntry keeps the platform paste
            # bindings, so Ctrl+V and the context menu still work.
            show="•",
            height=34,
            font=FONTS["body_small"],
            placeholder_text="Paste your key here",
        )
        self.key_entry.grid(row=4, column=0, sticky="ew")
        self.key_entry.bind("<Return>", lambda _event: self._verify())

        self.status_label = ctk.CTkLabel(
            body,
            textvariable=self.status_var,
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            wraplength=CONTENT_WRAP,
            justify="left",
            anchor="w",
        )
        self.status_label.grid(row=5, column=0, pady=(8, 0), sticky="ew")
        self._wrapping.append((self.status_label, CONTENT_WRAP))

        actions = ctk.CTkFrame(body, fg_color="transparent")
        actions.grid(row=6, column=0, pady=(14, 0), sticky="ew")
        actions.grid_columnconfigure(1, weight=1)
        self.get_key_button = ctk.CTkButton(
            actions,
            text="Get an API key",
            width=140,
            height=34,
            font=FONTS["button"],
            fg_color=COLORS["surface"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            command=self._open_provider_console,
        )
        self.get_key_button.grid(row=0, column=0, sticky="w")
        self.verify_button = ctk.CTkButton(
            actions,
            text="Verify & Continue",
            width=160,
            height=34,
            font=FONTS["button"],
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            text_color=COLORS["on_primary"],
            command=self._verify,
        )
        self.verify_button.grid(row=0, column=2, sticky="e")

        self.help_toggle = ctk.CTkButton(
            body,
            text="?  API usage, privacy, and other providers",
            height=26,
            font=FONTS["caption"],
            fg_color="transparent",
            hover_color=COLORS["control_hover"],
            text_color=COLORS["muted"],
            anchor="w",
            command=self.open_help,
        )
        self.help_toggle.grid(row=7, column=0, pady=(12, 0), sticky="ew")

        self.advanced_toggle = ctk.CTkButton(
            body,
            text="Other providers and offline use",
            height=26,
            font=FONTS["caption"],
            fg_color="transparent",
            hover_color=COLORS["control_hover"],
            text_color=COLORS["muted"],
            anchor="w",
            command=self._toggle_advanced,
        )
        self.advanced_toggle.grid(row=9, column=0, pady=(2, 0), sticky="ew")

        self.advanced_panel = ctk.CTkFrame(body, fg_color="transparent")
        self.advanced_panel.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            self.advanced_panel,
            text="Provider",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
        ).grid(row=0, column=0, padx=(0, 6), pady=(8, 0), sticky="w")
        self.provider_menu = ctk.CTkOptionMenu(
            self.advanced_panel,
            variable=self.provider_var,
            values=[PROVIDER_LABELS[item] for item in CLOUD_PROVIDERS],
            width=170,
            height=28,
            font=FONTS["caption"],
            command=self._provider_changed,
        )
        self.provider_menu.grid(row=0, column=1, pady=(8, 0), sticky="w")
        self.saved_keys_label = ctk.CTkLabel(
            self.advanced_panel,
            text="",
            text_color=COLORS["muted"],
            font=FONTS["caption"],
            wraplength=410,
            justify="left",
            anchor="w",
        )
        self.saved_keys_label.grid(row=1, column=0, columnspan=2, pady=(8, 0), sticky="ew")
        self._wrapping.append((self.saved_keys_label, 410))
        self.remove_button = ctk.CTkButton(
            self.advanced_panel,
            text="Remove saved key",
            width=150,
            height=28,
            font=FONTS["caption"],
            fg_color=COLORS["surface"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            command=self._remove_saved_key,
        )
        self.remove_button.grid(row=2, column=0, pady=(8, 0), sticky="w")
        self.local_button = ctk.CTkButton(
            self.advanced_panel,
            text="Use Local Ollama instead",
            width=190,
            height=28,
            font=FONTS["caption"],
            fg_color=COLORS["surface"],
            hover_color=COLORS["control_hover"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["ink"],
            command=self._choose_local,
        )
        self.local_button.grid(row=2, column=1, pady=(8, 0), sticky="w")
        local_note = ctk.CTkLabel(
            self.advanced_panel,
            text=provider_facts(LOCAL_OLLAMA).capability_note,
            text_color=COLORS["subtle"],
            font=FONTS["caption"],
            wraplength=410,
            justify="left",
            anchor="w",
        )
        local_note.grid(row=3, column=0, columnspan=2, pady=(6, 0), sticky="ew")
        self._wrapping.append((local_note, 410))

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=1, column=0, padx=BODY_PADX, pady=(4, 20), sticky="ew")
        footer.grid_columnconfigure(0, weight=1)
        self.close_button = ctk.CTkButton(
            footer,
            text="Close" if self._manage else "Not now",
            width=90,
            height=30,
            font=FONTS["caption"],
            fg_color="transparent",
            hover_color=COLORS["control_hover"],
            text_color=COLORS["muted"],
            command=self._dismiss,
        )
        self.close_button.grid(row=0, column=0, sticky="w")
        self.protocol("WM_DELETE_WINDOW", self._dismiss)
        if self._advanced_visible:
            self.advanced_panel.grid(row=10, column=0, sticky="ew")

    # ----------------------------------------------------------------- display

    def _refresh_provider_copy(self) -> None:
        facts = provider_facts(self._provider)
        prefix = "Provider" if self._manage else "Recommended provider"
        self.recommendation_label.configure(text=f"{prefix}: {facts.label}")
        self.get_key_button.configure(state="normal" if facts.api_key_url else "disabled")

    def _refresh_saved_keys(self) -> None:
        lines = []
        for provider_id in CLOUD_PROVIDERS:
            resolved = resolve_credential(provider_id, backend=self._backend)
            label = PROVIDER_LABELS[provider_id]
            if resolved.found:
                lines.append(f"{label}: configured from {resolved.source_description}")
            else:
                lines.append(f"{label}: no key configured")
        if not credential_store_available(backend=self._backend):
            lines.append(
                "This computer's credential store is unavailable, so a key "
                "entered here cannot be saved. Set it as an environment "
                "variable instead."
            )
        self.saved_keys_label.configure(text="\n".join(lines))
        current = resolve_credential(self._provider, backend=self._backend)
        self.remove_button.configure(
            state="normal" if current.source == CREDENTIAL_STORE else "disabled"
        )

    @property
    def _help_visible(self) -> bool:
        window = self._help_window
        return bool(window is not None and window.winfo_exists())

    def open_help(self) -> TextToCadHelpDialog:
        """Show the privacy and provider notes in their own window."""

        if self._help_visible:
            self._help_window.focus_set()
            return self._help_window
        self._help_window = TextToCadHelpDialog(self)
        return self._help_window

    def _toggle_advanced(self) -> None:
        self._set_advanced_visible(not self._advanced_visible)
        self._centre_on_parent()

    def _set_advanced_visible(self, visible: bool) -> None:
        self._advanced_visible = visible
        if visible:
            self.advanced_panel.grid(row=10, column=0, sticky="ew")
            self._refresh_saved_keys()
        else:
            self.advanced_panel.grid_remove()

    def _window_floor(self) -> int:
        """Narrowest this window may be and still hold a full paragraph."""

        return physical_width(self, CONTENT_WRAP + 2 * BODY_PADX + WRAP_MARGIN)

    def _interior_width(self) -> int:
        """Pixels a body paragraph has to wrap inside, padding excluded."""

        # Both readings describe the real window: one what it has been given,
        # the other what it asked for and will get.
        width = max(self.winfo_width(), self.winfo_reqwidth())
        return width - 2 * physical_width(self, BODY_PADX)

    def _reflow(self, _event: object = None) -> None:
        fit_wrapping(self._wrapping, available=self._interior_width())

    def _centre_on_parent(self) -> None:
        if not self.winfo_exists():
            return
        self.update_idletasks()
        # Wide enough to hold a full-width paragraph with its margins. Picking
        # this independently of the wrap the paragraphs are written to is what
        # let a message wrap wider than the window that had to show it.
        width = max(self.winfo_reqwidth(), self._window_floor())
        # Settle the wrapping at the width the window is going to have before
        # measuring its height. Narrowing a paragraph can cost it an extra
        # line, and a height measured before that happens leaves the last line
        # below the bottom of a window that cannot be resized.
        self.geometry(f"{width}x{self.winfo_reqheight()}")
        fit_wrapping(
            self._wrapping, available=width - 2 * physical_width(self, BODY_PADX)
        )
        self.update_idletasks()
        height = self.winfo_reqheight()
        parent = self.master.winfo_toplevel()
        x = parent.winfo_rootx() + max(0, (parent.winfo_width() - width) // 2)
        y = parent.winfo_rooty() + max(0, (parent.winfo_height() - height) // 3)
        self.geometry(f"{width}x{height}+{max(0, x)}+{max(0, y)}")

    # ----------------------------------------------------------------- actions

    def _set_status(self, message: str, *, tone: str = "neutral") -> None:
        """Put one sentence under the key field, in the colour that fits it."""

        self._status_is_error = tone == "error"
        self.status_label.configure(
            text_color={
                "error": COLORS["danger"],
                "success": COLORS["success"],
            }.get(tone, COLORS["muted"])
        )
        self.status_var.set(message)
        # A longer sentence must not reach past the window it is written in,
        # nor off the bottom of a window that was sized without it.
        fit_wrapping(self._wrapping, available=self._interior_width())
        self._resize_to_content()

    def _resize_to_content(self) -> None:
        """Grow the fixed-size window to whatever it now has to show."""

        if not self.winfo_exists():
            return
        self.update_idletasks()
        width = max(self.winfo_width(), self.winfo_reqwidth(), self._window_floor())
        height = self.winfo_reqheight()
        if (width, height) != (self.winfo_width(), self.winfo_height()):
            self.geometry(f"{width}x{height}")

    def _key_edited(self, *_args: object) -> None:
        """Drop a stale rejection as soon as the user edits the key.

        This runs from a Tk variable trace. The new value is deliberately never
        read, so the secret cannot reach a log, a status string or an error.
        """

        if self._status_is_error:
            self._set_status("")

    def _provider_changed(self, label: str) -> None:
        for provider_id in CLOUD_PROVIDERS:
            if PROVIDER_LABELS[provider_id] == label:
                self._provider = provider_id
                break
        self.key_var.set("")
        self._set_status("")
        self._refresh_provider_copy()
        self._refresh_saved_keys()

    def _open_provider_console(self) -> None:
        url = provider_facts(self._provider).api_key_url
        if not url:
            return
        try:
            self._browser(url)
        except Exception:
            # An unavailable browser is not worth a dialog of its own; show the
            # address so the user can open it themselves.
            self._set_status(f"Open this address to create a key: {url}")
            return
        self._set_status(
            "Create a key in the browser window, then paste it above."
        )

    def _choose_local(self) -> None:
        self.outcome = SetupOutcome(provider=LOCAL_OLLAMA, verified=False)
        self._close()

    def _remove_saved_key(self) -> None:
        removed = delete_credential(self._provider, backend=self._backend)
        label = PROVIDER_LABELS[self._provider]
        self._set_status(
            f"The saved {label} key was removed from this computer."
            if removed
            else f"There was no saved {label} key to remove."
        )
        self._refresh_saved_keys()

    def _set_busy(self, busy: bool) -> None:
        self._verifying = busy
        state = "disabled" if busy else "normal"
        self.verify_button.configure(
            state=state, text="Verifying…" if busy else "Verify & Continue"
        )
        self.key_entry.configure(state=state)
        self.provider_menu.configure(state=state)

    def _verify(self) -> None:
        if self._verifying:
            return
        candidate = self.key_var.get().strip()
        provider_id = self._provider
        if not candidate:
            self._set_status("Enter an API key to continue.", tone="error")
            return
        self._set_busy(True)
        self._set_status(f"Checking the key with {PROVIDER_LABELS[provider_id]}…")

        def worker() -> None:
            try:
                verification = verify_cloud_credential(
                    provider_id, candidate, discovery=self._discovery
                )
            except Exception as exc:
                self._schedule(lambda error=exc: self._verification_failed(provider_id, error))
                return
            self._schedule(
                lambda result=verification: self._verification_succeeded(
                    provider_id, candidate, result
                )
            )

        threading.Thread(target=worker, daemon=True).start()

    def _schedule(self, callback: Callable[[], object]) -> None:
        """Hand a result back to the Tk thread, tolerating a closed dialog.

        The provider call can still be in flight when the window goes away, and
        `after` on a destroyed widget raises rather than being ignored.
        """

        try:
            self.after(0, callback)
        except (tk.TclError, RuntimeError):
            pass

    def _verification_failed(self, provider_id: str, error: Exception) -> None:
        if not self.winfo_exists():
            return
        self._set_busy(False)
        if isinstance(error, ModelDiscoveryError):
            message = friendly_discovery_message(provider_id, error)
        else:
            message = (
                "We couldn't verify this API key. Check the key and your "
                "internet connection, then try again."
            )
        self._set_status(message, tone="error")
        # The rejected key is never stored, and the dialog stays open so the
        # user can correct it.

    def _verification_succeeded(
        self,
        provider_id: str,
        api_key: str,
        verification: CredentialVerification,
    ) -> None:
        if not self.winfo_exists():
            return
        try:
            store_credential(provider_id, api_key, backend=self._backend)
        except (CredentialStoreUnavailable, ValueError, KeyError) as exc:
            self._set_busy(False)
            self._set_status(
                f"The key was accepted by {PROVIDER_LABELS[provider_id]} but could "
                f"not be saved on this computer. {exc}",
                tone="error",
            )
            return
        self.outcome = SetupOutcome(provider=provider_id, verified=True)
        self._set_status(f"{VERIFIED_HEADLINE}\n{VERIFIED_DETAIL}", tone="success")
        self._set_busy(False)
        self.key_var.set("")
        self._refresh_saved_keys()
        if self._manage:
            return
        # Let the confirmation stay on screen long enough to read before the
        # builder appears behind it.
        self.after(650, self._close)

    def _dismiss(self) -> None:
        if self._verifying:
            return
        if self._manage and self.outcome.completed:
            self._close()
            return
        self.outcome = SetupOutcome(provider=None, verified=False)
        self._close()

    def _close(self) -> None:
        if not self.winfo_exists():
            return
        if self._help_visible:
            self._help_window.close()
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()

    def describe_ui_state(self) -> list[str]:
        """Summarise the dialog for the blind GUI contract."""

        return [
            f"Dialog: {SETUP_TITLE}",
            f"Provider: {PROVIDER_LABELS[self._provider]}",
            "API key entry is masked",
            f"Status: {self.status_var.get() or 'none'}",
            f"Help window open: {self._help_visible}",
            f"Other providers visible: {self._advanced_visible}",
        ]


def run_setup_dialog(
    parent: ctk.CTkBaseClass,
    *,
    provider: str = RECOMMENDED_CLOUD_PROVIDER,
    manage: bool = False,
    discovery: PlannerModelDiscovery | None = None,
    credential_backend: object | None = None,
    browser: Callable[[str], object] | None = None,
) -> SetupOutcome:
    """Show the setup dialog modally and return what the user settled on."""

    dialog = TextToCadSetupDialog(
        parent,
        provider=provider,
        manage=manage,
        discovery=discovery,
        credential_backend=credential_backend,
        browser=browser,
    )
    parent.wait_window(dialog)
    return dialog.outcome
