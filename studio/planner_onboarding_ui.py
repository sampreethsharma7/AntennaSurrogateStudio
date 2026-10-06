"""First-run setup for the Text-to-CAD cloud planner.

An RF engineer should be able to paste a key, have it checked against the
provider, and get on with designing. Nothing here asks the user to know what a
`.env` file is, what an endpoint is, or what an HTTP status means: the dialog
takes a key, verifies it for real, and either continues or says one plain
sentence about what to fix.

The dialog deliberately stays small. Pricing, privacy and the other providers
live behind a question-mark panel, because an onboarding screen that opens with
configuration documentation is one the user closes.
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
from studio.theme import COLORS, FONTS


SETUP_TITLE = "Set up Text-to-CAD"
SETUP_INTRO = (
    "Text-to-CAD uses a cloud AI model to interpret your antenna design requests."
)
VERIFIED_HEADLINE = "✓ API key verified"
VERIFIED_DETAIL = "Text-to-CAD is ready."
# Caps the scrolled help panel so the dialog always fits a laptop screen.
HELP_PANEL_HEIGHT = 250

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
        self._help_visible = False
        self._advanced_visible = manage

        self.title("Text-to-CAD setup" if not manage else "Text-to-CAD API keys")
        self.resizable(False, False)
        self.transient(parent.winfo_toplevel())

        self.provider_var = ctk.StringVar(value=PROVIDER_LABELS[self._provider])
        self.key_var = ctk.StringVar(value="")
        self.status_var = ctk.StringVar(value="")

        self._build()
        self._refresh_provider_copy()
        self._refresh_saved_keys()
        self.grab_set()
        self.key_entry.focus_set()
        self.after(0, self._centre_on_parent)

    # ---------------------------------------------------------------- building

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=0, column=0, padx=26, pady=(24, 10), sticky="nsew")
        body.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            body,
            text=SETUP_TITLE,
            text_color=COLORS["ink"],
            font=FONTS["title"],
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            body,
            text=SETUP_INTRO,
            text_color=COLORS["muted"],
            font=FONTS["body_small"],
            wraplength=420,
            justify="left",
            anchor="w",
        ).grid(row=1, column=0, pady=(6, 12), sticky="ew")

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
            wraplength=420,
            justify="left",
            anchor="w",
        )
        self.status_label.grid(row=5, column=0, pady=(8, 0), sticky="ew")

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
            command=self._toggle_help,
        )
        self.help_toggle.grid(row=7, column=0, pady=(12, 0), sticky="ew")

        # Scrolled and height-capped: the five sections run to roughly 750px of
        # content, which would push the dialog past the bottom of a 1366x768
        # laptop screen and put the primary action out of reach.
        self.help_panel = ctk.CTkScrollableFrame(
            body,
            height=HELP_PANEL_HEIGHT,
            fg_color=COLORS["surface_alt"],
            corner_radius=8,
            border_width=1,
            border_color=COLORS["border"],
        )
        self.help_panel.grid_columnconfigure(0, weight=1)
        for index, (heading, detail) in enumerate(HELP_SECTIONS):
            ctk.CTkLabel(
                self.help_panel,
                text=heading,
                text_color=COLORS["ink"],
                font=FONTS["caption"],
                anchor="w",
            ).grid(row=index * 2, column=0, padx=12, pady=(10 if index else 12, 1), sticky="ew")
            ctk.CTkLabel(
                self.help_panel,
                text=detail,
                text_color=COLORS["muted"],
                font=FONTS["caption"],
                wraplength=400,
                justify="left",
                anchor="w",
            ).grid(row=index * 2 + 1, column=0, padx=12, sticky="ew")
        ctk.CTkLabel(
            self.help_panel,
            text="",
            font=FONTS["caption"],
        ).grid(row=len(HELP_SECTIONS) * 2, column=0, pady=(0, 6))

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
        ctk.CTkLabel(
            self.advanced_panel,
            text=provider_facts(LOCAL_OLLAMA).capability_note,
            text_color=COLORS["subtle"],
            font=FONTS["caption"],
            wraplength=410,
            justify="left",
            anchor="w",
        ).grid(row=3, column=0, columnspan=2, pady=(6, 0), sticky="ew")

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=1, column=0, padx=26, pady=(4, 20), sticky="ew")
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

    def _toggle_help(self) -> None:
        self._help_visible = not self._help_visible
        if self._help_visible:
            # One secondary panel at a time: together they are taller than a
            # laptop screen, and nobody needs to read both at once.
            self._set_advanced_visible(False)
            self.help_panel.grid(row=8, column=0, pady=(6, 0), sticky="ew")
        else:
            self.help_panel.grid_remove()
        self._centre_on_parent()

    def _toggle_advanced(self) -> None:
        self._set_advanced_visible(not self._advanced_visible)
        if self._advanced_visible and self._help_visible:
            self._help_visible = False
            self.help_panel.grid_remove()
        self._centre_on_parent()

    def _set_advanced_visible(self, visible: bool) -> None:
        self._advanced_visible = visible
        if visible:
            self.advanced_panel.grid(row=10, column=0, sticky="ew")
            self._refresh_saved_keys()
        else:
            self.advanced_panel.grid_remove()

    def _centre_on_parent(self) -> None:
        if not self.winfo_exists():
            return
        self.update_idletasks()
        parent = self.master.winfo_toplevel()
        width = max(self.winfo_reqwidth(), 470)
        height = self.winfo_reqheight()
        x = parent.winfo_rootx() + max(0, (parent.winfo_width() - width) // 2)
        y = parent.winfo_rooty() + max(0, (parent.winfo_height() - height) // 3)
        self.geometry(f"{width}x{height}+{max(0, x)}+{max(0, y)}")

    # ----------------------------------------------------------------- actions

    def _provider_changed(self, label: str) -> None:
        for provider_id in CLOUD_PROVIDERS:
            if PROVIDER_LABELS[provider_id] == label:
                self._provider = provider_id
                break
        self.key_var.set("")
        self.status_var.set("")
        self.status_label.configure(text_color=COLORS["muted"])
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
            self.status_var.set(f"Open this address to create a key: {url}")
            return
        self.status_var.set(
            "Create a key in the browser window, then paste it above."
        )

    def _choose_local(self) -> None:
        self.outcome = SetupOutcome(provider=LOCAL_OLLAMA, verified=False)
        self._close()

    def _remove_saved_key(self) -> None:
        removed = delete_credential(self._provider, backend=self._backend)
        label = PROVIDER_LABELS[self._provider]
        self.status_label.configure(text_color=COLORS["muted"])
        self.status_var.set(
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
            self.status_label.configure(text_color=COLORS["danger"])
            self.status_var.set("Enter an API key to continue.")
            return
        self._set_busy(True)
        self.status_label.configure(text_color=COLORS["muted"])
        self.status_var.set(f"Checking the key with {PROVIDER_LABELS[provider_id]}…")

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
        self.status_label.configure(text_color=COLORS["danger"])
        self.status_var.set(message)
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
            self.status_label.configure(text_color=COLORS["danger"])
            self.status_var.set(
                f"The key was accepted by {PROVIDER_LABELS[provider_id]} but could "
                f"not be saved on this computer. {exc}"
            )
            return
        self.outcome = SetupOutcome(provider=provider_id, verified=True)
        self.status_label.configure(text_color=COLORS["success"])
        self.status_var.set(f"{VERIFIED_HEADLINE}\n{VERIFIED_DETAIL}")
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
            f"Help panel visible: {self._help_visible}",
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
