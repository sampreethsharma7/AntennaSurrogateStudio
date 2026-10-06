"""First-run Text-to-CAD credential setup, end to end through the GUI."""

import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from studio.antenna_builder_ui import DEFAULT_MODELS, cloud_onboarding_required
from studio.planner_credentials import (
    GEMINI,
    GROQ,
    KEYRING_SERVICE,
    LOCAL_OLLAMA,
)
from studio.planner_model_discovery import (
    ModelDiscoveryError,
    PlannerModelDiscovery,
    friendly_discovery_message,
)
from studio.planner_onboarding_ui import (
    HELP_SECTIONS,
    SETUP_INTRO,
    SETUP_TITLE,
    VERIFIED_DETAIL,
    VERIFIED_HEADLINE,
    SetupOutcome,
    TextToCadHelpDialog,
    TextToCadSetupDialog,
)
from studio.theme import COLORS
from studio.project_store import ProjectStore
from studio.ui import StudioApp


GUI_MAY_BE_AVAILABLE = (
    os.name == "nt"
    or os.sys.platform == "darwin"
    or bool(os.environ.get("DISPLAY"))
)

SECRET = "AIzaPretendGeminiKey1234567890"

# A 1366x768 laptop does not give a window all 768 rows: the taskbar and this
# window's own title bar come out of the budget first, and reqheight counts
# neither of them.
LAPTOP_CHROME = 80


def rendered_width(label):
    """Width Tk needs for the widest wrapped line this label will draw.

    A CTkLabel is a frame around a real Tk label, and only that inner widget
    knows how wide the text came out once it was wrapped. Asking it is what
    distinguishes a paragraph that fits from one that is about to be cut off
    by the window edge -- which is the defect this measures.
    """

    return label._label.winfo_reqwidth()


def allotted_width(label):
    """Width the layout actually gave this label on screen."""

    return label.winfo_width()


class FakeCredentialStore:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.writes = []

    def get_password(self, service, account):
        return self.values.get((service, account))

    def set_password(self, service, account, password):
        self.writes.append((service, account, password))
        self.values[(service, account)] = password

    def delete_password(self, service, account):
        del self.values[(service, account)]


class _ImmediateThread:
    """Run the verification worker inline so the test stays deterministic."""

    def __init__(self, *, target, daemon):
        self.target = target
        self.daemon = daemon

    def start(self):
        self.target()


def catalog_opener(payload):
    def opener(request, timeout=None):
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    return opener


def failing_opener(error):
    def opener(request, timeout=None):
        raise error

    return opener


GEMINI_CATALOG = {
    "models": [
        {
            "name": "models/gemini-3.8-flash",
            "displayName": "Gemini 3.8 Flash",
            "supportedGenerationMethods": ["generateContent"],
        }
    ]
}


class OnboardingDecisionTests(unittest.TestCase):
    """The gate is a pure decision, so it is tested without a window."""

    @classmethod
    def setUpClass(cls):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        cls.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        cls.store = ProjectStore(Path(cls.temp_dir.name) / "library")

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def _fresh_project(self):
        project = self.store.create_project(self._testMethodName)
        return self.store.open_project(project.path, touch=False)

    def test_a_first_run_with_no_cloud_credential_needs_setup(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(
                cloud_onboarding_required(
                    self._fresh_project(),
                    env_file="absent.env",
                    credential_backend=FakeCredentialStore(),
                )
            )

    def test_a_stored_credential_skips_setup(self):
        store = FakeCredentialStore({(KEYRING_SERVICE, "gemini"): SECRET})
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(
                cloud_onboarding_required(
                    self._fresh_project(),
                    env_file="absent.env",
                    credential_backend=store,
                )
            )

    def test_a_process_environment_credential_skips_setup(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": SECRET}, clear=True):
            self.assertFalse(
                cloud_onboarding_required(
                    self._fresh_project(),
                    env_file="absent.env",
                    credential_backend=FakeCredentialStore(),
                )
            )

    def test_an_env_file_credential_skips_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(f"GEMINI_API_KEY={SECRET}\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertFalse(
                    cloud_onboarding_required(
                        self._fresh_project(),
                        env_file=env_file,
                        credential_backend=FakeCredentialStore(),
                    )
                )

    def test_a_recorded_offline_choice_is_never_interrupted(self):
        # Reopening an existing design must not demand a cloud key the user
        # already declined.
        project = self.store.update_project(
            self._fresh_project(),
            {"antenna_builder": {"planner_provider": LOCAL_OLLAMA}},
        )
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(
                cloud_onboarding_required(
                    project,
                    env_file="absent.env",
                    credential_backend=FakeCredentialStore(),
                )
            )

    def test_a_recorded_cloud_choice_is_honoured_only_while_its_key_exists(self):
        """A stored provider settles the question; a stored provider with no
        key does not.

        Keys get removed from the API keys dialog, revoked at the provider, or
        left behind on another machine. Treating the recorded choice alone as
        settled sent the user into the builder with a dead credential, and the
        request then failed inside the planner telling them to set an
        environment variable -- which is the ending this dialog exists to
        prevent.
        """

        project = self.store.update_project(
            self._fresh_project(),
            {"antenna_builder": {"planner_provider": GEMINI}},
        )
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(
                cloud_onboarding_required(
                    project,
                    env_file="absent.env",
                    credential_backend=FakeCredentialStore(
                        {(KEYRING_SERVICE, "gemini"): SECRET}
                    ),
                ),
                "a usable key must not be interrupted",
            )
            self.assertTrue(
                cloud_onboarding_required(
                    project,
                    env_file="absent.env",
                    credential_backend=FakeCredentialStore(),
                ),
                "a recorded cloud provider whose key is gone must ask again",
            )
            # Another provider's key is no substitute for the one in use.
            self.assertTrue(
                cloud_onboarding_required(
                    project,
                    env_file="absent.env",
                    credential_backend=FakeCredentialStore(
                        {(KEYRING_SERVICE, "groq"): SECRET}
                    ),
                )
            )

    def test_a_legacy_project_recorded_only_as_a_backend_is_not_interrupted(self):
        project = self.store.update_project(
            self._fresh_project(),
            {"antenna_builder": {"planner_backend": "local_qwen"}},
        )
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(
                cloud_onboarding_required(
                    project,
                    env_file="absent.env",
                    credential_backend=FakeCredentialStore(),
                )
            )

    def test_no_project_means_nothing_to_set_up(self):
        self.assertFalse(cloud_onboarding_required(None))


@unittest.skipUnless(GUI_MAY_BE_AVAILABLE, "A desktop display is required.")
class SetupDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        cls.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        cls.store = ProjectStore(Path(cls.temp_dir.name) / "library")
        try:
            cls.app = StudioApp(project_store=cls.store)
        except Exception as exc:
            cls.temp_dir.cleanup()
            raise unittest.SkipTest(f"A desktop display is not available: {exc}") from exc
        cls.app.withdraw()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "app"):
            cls.app.destroy()
        cls.temp_dir.cleanup()

    def setUp(self):
        self.app.withdraw()
        self.backend = FakeCredentialStore()
        self.opened_urls = []
        self.dialog = None

    def tearDown(self):
        if self.dialog is not None and self.dialog.winfo_exists():
            self.dialog.destroy()
        self.app.update()

    def _dialog(self, *, payload=None, error=None, manage=False, provider=GEMINI):
        opener = failing_opener(error) if error is not None else catalog_opener(
            payload if payload is not None else GEMINI_CATALOG
        )
        self.dialog = TextToCadSetupDialog(
            self.app,
            provider=provider,
            manage=manage,
            discovery=PlannerModelDiscovery(opener=opener),
            credential_backend=self.backend,
            browser=self.opened_urls.append,
        )
        self.app.update()
        return self.dialog

    def _verify(self, dialog):
        with patch("studio.planner_onboarding_ui.threading.Thread", _ImmediateThread):
            dialog._verify()
        self.app.update()

    def test_the_dialog_opens_simple_recommending_gemini_with_a_masked_entry(self):
        dialog = self._dialog()
        self.assertEqual(
            dialog.recommendation_label.cget("text"), "Recommended provider: Gemini"
        )
        self.assertEqual(SETUP_TITLE, "Set up Text-to-CAD")
        self.assertIn("cloud AI model", SETUP_INTRO)
        self.assertIn("antenna design requests", SETUP_INTRO)
        # The key field behaves like a password field.
        self.assertEqual(dialog.key_entry.cget("show"), "•")
        self.assertEqual(dialog.verify_button.cget("text"), "Verify & Continue")
        self.assertEqual(dialog.get_key_button.cget("text"), "Get an API key")
        # The secondary material starts put away so the dialog stays small.
        self.assertFalse(dialog._help_visible)
        self.assertFalse(dialog._advanced_visible)
        self.app.update_idletasks()
        collapsed_height = dialog.winfo_reqheight()
        # Help opens a window of its own. The primary dialog must not grow,
        # because unfolding this material in place pushed the primary action
        # off the bottom of a laptop screen.
        help_window = dialog.open_help()
        self.app.update()
        self.assertTrue(dialog._help_visible)
        self.assertEqual(dialog.winfo_reqheight(), collapsed_height)
        help_window.close()
        self.app.update()
        self.assertFalse(dialog._help_visible)
        self.assertEqual(dialog.winfo_reqheight(), collapsed_height)

    def test_the_help_window_explains_storage_privacy_and_cost_without_promises(self):
        dialog = self._dialog()
        self.app.update_idletasks()
        collapsed = dialog.winfo_reqheight()
        help_window = dialog.open_help()
        self.app.update()
        self.assertTrue(dialog._help_visible)
        self.assertEqual(dialog.winfo_reqheight(), collapsed)
        self.addCleanup(help_window.close)

        from studio.planner_onboarding_ui import HELP_SECTIONS

        text = " ".join(detail for _heading, detail in HELP_SECTIONS).casefold()
        self.assertIn("credential store", text)
        self.assertIn("provider you choose", text)
        self.assertIn("retention", text)
        self.assertIn("local ollama", text)
        self.assertIn("spending limit", text)
        # No specific cost promise, and no claim that the Studio supplies credit.
        for forbidden in ("$1", "$2", "maximum", "free credit", "we provide"):
            self.assertNotIn(forbidden, text)

    def test_get_an_api_key_opens_the_official_console_in_the_system_browser(self):
        dialog = self._dialog()
        dialog._open_provider_console()
        self.assertEqual(self.opened_urls, ["https://aistudio.google.com/apikey"])
        self.assertIn("paste it above", dialog.status_var.get())

    def test_a_verified_key_is_stored_confirmed_and_reported_as_the_outcome(self):
        dialog = self._dialog()
        dialog.key_var.set(SECRET)
        self._verify(dialog)

        self.assertEqual(dialog.outcome, SetupOutcome(provider=GEMINI, verified=True))
        self.assertIn(VERIFIED_HEADLINE, dialog.status_var.get())
        self.assertIn("Text-to-CAD is ready", dialog.status_var.get())
        self.assertEqual(
            self.backend.values[(KEYRING_SERVICE, "gemini")], SECRET
        )
        # The field is cleared, so the secret is not left on screen or readable
        # back out of the widget.
        self.assertEqual(dialog.key_var.get(), "")

    def test_a_rejected_key_keeps_the_dialog_open_and_stores_nothing(self):
        dialog = self._dialog(
            error=urllib.error.HTTPError("https://example.invalid", 401, "Unauthorized", {}, None)
        )
        dialog.key_var.set("wrong-key")
        self._verify(dialog)

        self.assertFalse(dialog.outcome.completed)
        self.assertTrue(dialog.winfo_exists())
        self.assertEqual(self.backend.writes, [])
        message = dialog.status_var.get()
        self.assertIn("couldn't verify this API key", message)
        for forbidden in ("401", "HTTP", "wrong-key", "Traceback", "{"):
            self.assertNotIn(forbidden, message)
        # The field keeps what was typed so the user can correct it.
        self.assertEqual(dialog.key_var.get(), "wrong-key")

    def test_a_network_failure_is_reported_as_recoverable(self):
        dialog = self._dialog(error=urllib.error.URLError("getaddrinfo failed"))
        dialog.key_var.set(SECRET)
        self._verify(dialog)

        self.assertFalse(dialog.outcome.completed)
        self.assertEqual(self.backend.writes, [])
        message = dialog.status_var.get()
        self.assertIn("could not be reached", message)
        self.assertIn("internet connection", message)
        self.assertNotIn("getaddrinfo", message)

    def test_an_empty_key_is_refused_without_contacting_the_provider(self):
        dialog = self._dialog(error=AssertionError("the provider must not be contacted"))
        dialog.key_var.set("   ")
        self._verify(dialog)
        self.assertEqual(dialog.status_var.get(), "Enter an API key to continue.")
        self.assertEqual(self.backend.writes, [])

    def test_the_offline_path_selects_local_ollama_with_its_capability_caveat(self):
        dialog = self._dialog()
        dialog._toggle_advanced()
        self.app.update()
        self.assertTrue(dialog._advanced_visible)
        self.assertEqual(dialog.advanced_panel.winfo_manager(), "grid")
        self.assertEqual(dialog.local_button.cget("text"), "Use Local Ollama instead")

        dialog._choose_local()
        self.assertEqual(dialog.outcome, SetupOutcome(provider=LOCAL_OLLAMA, verified=False))
        self.assertEqual(self.backend.writes, [])

    def test_other_providers_remain_selectable_and_openai_is_absent(self):
        dialog = self._dialog()
        dialog._toggle_advanced()
        self.app.update()
        offered = list(dialog.provider_menu.cget("values"))
        self.assertEqual(offered, ["Gemini", "Groq", "OpenRouter"])
        self.assertNotIn("OpenAI", offered)

        dialog._provider_changed("Groq")
        self.assertEqual(dialog._provider, GROQ)
        self.assertEqual(dialog.recommendation_label.cget("text"), "Recommended provider: Groq")

    def test_switching_provider_clears_any_typed_key(self):
        dialog = self._dialog()
        dialog.key_var.set(SECRET)
        dialog._provider_changed("Groq")
        self.assertEqual(dialog.key_var.get(), "")

    def test_manage_mode_reports_the_active_source_and_removes_a_stored_key(self):
        self.backend.values[(KEYRING_SERVICE, "gemini")] = SECRET
        with patch.dict(os.environ, {}, clear=True):
            dialog = self._dialog(manage=True)
            summary = dialog.saved_keys_label.cget("text")
            self.assertIn("Gemini: configured from this computer's credential store", summary)
            self.assertIn("Groq: no key configured", summary)
            self.assertNotIn(SECRET, summary)
            self.assertEqual(str(dialog.remove_button.cget("state")), "normal")

            dialog._remove_saved_key()
            self.app.update()
            self.assertNotIn((KEYRING_SERVICE, "gemini"), self.backend.values)
            self.assertIn("was removed", dialog.status_var.get())
            self.assertIn("Gemini: no key configured", dialog.saved_keys_label.cget("text"))

    def test_manage_mode_names_the_shadowing_source_rather_than_hiding_it(self):
        # A stored key that an environment variable outranks is the one case
        # where "I saved it but it is not being used" would otherwise be a
        # mystery.
        self.backend.values[(KEYRING_SERVICE, "gemini")] = SECRET
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}, clear=True):
            dialog = self._dialog(manage=True)
            summary = dialog.saved_keys_label.cget("text")
            self.assertIn("Gemini: configured from an environment variable", summary)
            self.assertEqual(str(dialog.remove_button.cget("state")), "disabled")

    def test_the_dialog_never_reveals_a_stored_secret(self):
        self.backend.values[(KEYRING_SERVICE, "gemini")] = SECRET
        with patch.dict(os.environ, {}, clear=True):
            dialog = self._dialog(manage=True)
            rendered = " ".join([
                dialog.saved_keys_label.cget("text"),
                dialog.status_var.get(),
                dialog.key_var.get(),
                dialog.recommendation_label.cget("text"),
                " ".join(dialog.describe_ui_state()),
            ])
        self.assertNotIn(SECRET, rendered)

    def test_the_dialog_fits_a_laptop_viewport(self):
        dialog = self._dialog()
        scale = max(1.0, float(self.app.ui_scaling))
        width_budget = round(1366 * scale)
        # A real laptop also spends part of its screen on a taskbar and on this
        # window's own title bar, neither of which reqheight counts.
        height_budget = round((768 - LAPTOP_CHROME) * scale)
        longest = friendly_discovery_message(
            GEMINI, ModelDiscoveryError("rejected", kind="http", status=401)
        )
        for label, prepare in (
            ("collapsed", lambda: None),
            ("rejected key shown", lambda: dialog._set_status(longest, tone="error")),
            ("other providers open", dialog._toggle_advanced),
            ("other providers open with a rejection", lambda: None),
        ):
            with self.subTest(state=label):
                prepare()
                self.app.update_idletasks()
                self.assertLessEqual(dialog.winfo_reqwidth(), width_budget)
                self.assertLessEqual(dialog.winfo_reqheight(), height_budget)

    def test_every_verification_message_fits_the_width_it_is_given(self):
        """No status sentence may run past the edge of a fixed-width dialog.

        The dialog is sized once, while the status line is still empty, and is
        not resizable. A message that wraps to a width nobody checked against
        the window loses its last characters, which is what a first-time user
        saw: "It looks incomplete or i" / "not in the format Gemini expects."
        """

        dialog = self._dialog()
        self.app.update()
        messages = {
            "incomplete key": ("http", 400),
            "authentication rejected": ("http", 401),
            "access forbidden": ("http", 403),
            "model missing": ("http", 404),
            "rate limited": ("http", 429),
            "provider unavailable": ("http", 503),
            "network failure": ("network", None),
            "empty catalog": ("no_models", None),
        }
        rendered = {
            name: friendly_discovery_message(
                GEMINI, ModelDiscoveryError(name, kind=kind, status=status)
            )
            for name, (kind, status) in messages.items()
        }
        rendered["verified"] = VERIFIED_HEADLINE + "\n" + VERIFIED_DETAIL
        rendered["checking"] = "Checking the key with Gemini…"
        rendered["console opened"] = (
            "Open this address to create a key: https://aistudio.google.com/apikey"
        )

        for name, message in rendered.items():
            with self.subTest(message=name):
                dialog._set_status(message)
                self.app.update()
                self.assertLessEqual(
                    rendered_width(dialog.status_label),
                    allotted_width(dialog.status_label),
                    f"{name!r} is wider than the dialog can show: {message!r}",
                )

    def test_the_help_window_text_fits_the_width_it_is_given(self):
        """The privacy explanation has to be readable, not cut off mid-word."""

        dialog = self._dialog()
        help_window = dialog.open_help()
        self.addCleanup(help_window.close)
        self.app.update()
        self.assertTrue(help_window._wrapping, "no wrapped help paragraphs found")
        for index, (label, _declared) in enumerate(help_window._wrapping):
            with self.subTest(section=HELP_SECTIONS[index][0]):
                self.assertLessEqual(
                    rendered_width(label),
                    allotted_width(label),
                    f"help section {HELP_SECTIONS[index][0]!r} is clipped",
                )

    def test_editing_the_key_clears_a_stale_rejection(self):
        dialog = self._dialog(
            error=urllib.error.HTTPError(
                "https://example.invalid", 401, "Unauthorized", {}, None
            )
        )
        dialog.key_var.set("wrong-key")
        self._verify(dialog)
        self.assertIn("couldn't verify this API key", dialog.status_var.get())
        self.assertTrue(dialog._status_is_error)

        # Correcting the key puts the dialog back to neutral, so the red
        # sentence does not sit there contradicting what is in the field.
        dialog.key_var.set("wrong-key-corrected")
        self.app.update()
        self.assertEqual(dialog.status_var.get(), "")
        self.assertFalse(dialog._status_is_error)
        self.assertEqual(
            dialog.status_label.cget("text_color"), COLORS["muted"]
        )
        # Clearing the field entirely is also an edit, and must not resurrect it.
        dialog.key_var.set("")
        self.app.update()
        self.assertEqual(dialog.status_var.get(), "")

    def test_a_successful_verification_survives_the_field_being_cleared(self):
        """Clearing the key on success must not wipe the confirmation."""

        dialog = self._dialog()
        dialog.key_var.set(SECRET)
        self._verify(dialog)
        self.assertEqual(dialog.key_var.get(), "")
        self.assertIn(VERIFIED_HEADLINE, dialog.status_var.get())
        self.assertFalse(dialog._status_is_error)

    def test_the_help_window_fits_a_laptop_viewport(self):
        """Sized against a laptop screen, not whichever screen runs the tests.

        The window takes its height from the display it opens on, so proving it
        fits a 1366x768 laptop means telling it that is the display it is on.
        """

        dialog = self._dialog()
        scale = max(1.0, float(self.app.ui_scaling))
        laptop_height = round(768 * scale)
        with patch.object(
            TextToCadHelpDialog, "winfo_screenheight", lambda _self: laptop_height
        ):
            help_window = dialog.open_help()
            self.addCleanup(help_window.close)
            self.app.update()
        self.assertLessEqual(help_window.winfo_width(), round(1366 * scale))
        self.assertLessEqual(
            help_window.winfo_height(), round((768 - LAPTOP_CHROME) * scale)
        )
        # Still big enough to read rather than a letterbox.
        self.assertGreaterEqual(help_window.winfo_height(), round(300 * scale))


@unittest.skipUnless(GUI_MAY_BE_AVAILABLE, "A desktop display is required.")
class BuilderEntryGatingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        cls.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        cls.store = ProjectStore(Path(cls.temp_dir.name) / "library")
        try:
            cls.app = StudioApp(project_store=cls.store)
        except Exception as exc:
            cls.temp_dir.cleanup()
            raise unittest.SkipTest(f"A desktop display is not available: {exc}") from exc
        cls.app.withdraw()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "app"):
            cls.app.destroy()
        cls.temp_dir.cleanup()

    def setUp(self):
        self.app.withdraw()
        self.project = self.store.create_project(self._testMethodName)
        self.app.set_project(
            self.store.open_project(self.project.path, touch=False),
            target_page="design_start",
        )
        self.app.update()
        self.start_page = self.app.design_start_page
        self.backend = FakeCredentialStore()
        self.start_page.credential_backend = self.backend
        self.calls = []
        # Adopting a provider asks it for its model catalog. That is right in
        # the application and wrong in a test suite, which must never reach the
        # network, so the request is recorded instead of made.
        patcher = patch.object(
            self.app.antenna_builder_page, "_refresh_models_async"
        )
        self.model_refreshes = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.app.withdraw()

    def _runner(self, outcome):
        def runner(*args, **kwargs):
            self.calls.append(kwargs.get("provider"))
            return outcome

        return runner

    def test_entering_the_builder_without_a_credential_opens_setup(self):
        self.start_page.setup_dialog_runner = self._runner(
            SetupOutcome(provider=GEMINI, verified=True)
        )
        with patch.dict(os.environ, {}, clear=True):
            self.start_page.choose_template()
        self.app.update()

        # Setup ran, and it recommended Gemini.
        self.assertEqual(self.calls, [GEMINI])
        self.assertEqual(self.app.active_page, "antenna_builder")
        reopened = self.store.open_project(self.project.path, touch=False)
        self.assertEqual(
            reopened.manifest["antenna_builder"]["planner_provider"], GEMINI
        )

    def test_the_builder_adopts_the_provider_that_setup_verified(self):
        """The workspace must point at the planner the user just set up.

        The builder reads its planner choice when the project is opened, which
        is before setup has run, and nothing re-reads it afterwards. Recording
        the choice in the manifest is therefore not enough on its own: someone
        who had just verified a Gemini key was handed a workspace still set to
        Local Ollama, so their first request went to a planner they never chose
        and failed if no local model was running.
        """

        builder = self.app.antenna_builder_page
        self.assertEqual(builder._provider_id(), LOCAL_OLLAMA)
        self.start_page.setup_dialog_runner = self._runner(
            SetupOutcome(provider=GEMINI, verified=True)
        )
        with patch.dict(os.environ, {}, clear=True):
            self.start_page.choose_template()
            self.app.update()

        self.assertEqual(builder._provider_id(), GEMINI)
        self.assertEqual(builder._selected_model_id(), DEFAULT_MODELS[GEMINI])
        # It asked the provider for its catalog rather than reaching out itself.
        self.assertTrue(self.model_refreshes.called)

    def test_choosing_the_offline_planner_leaves_the_builder_on_it(self):
        builder = self.app.antenna_builder_page
        self.start_page.setup_dialog_runner = self._runner(
            SetupOutcome(provider=LOCAL_OLLAMA, verified=False)
        )
        with patch.dict(os.environ, {}, clear=True):
            self.start_page.choose_template()
            self.app.update()

        self.assertEqual(builder._provider_id(), LOCAL_OLLAMA)
        self.assertEqual(builder._selected_model_id(), DEFAULT_MODELS[LOCAL_OLLAMA])

    def test_dismissing_setup_keeps_the_user_on_the_start_page(self):
        # Proceeding would hand over a workspace whose first request fails deep
        # inside the planner with a message about an environment variable.
        self.start_page.setup_dialog_runner = self._runner(SetupOutcome())
        with patch.dict(os.environ, {}, clear=True):
            self.start_page.choose_template()
        self.app.update()

        self.assertEqual(self.app.active_page, "design_start")
        reopened = self.store.open_project(self.project.path, touch=False)
        # A new project already carries an antenna_builder scaffold, so what
        # must be absent is a recorded planner choice.
        self.assertNotIn(
            "planner_provider", reopened.manifest.get("antenna_builder", {})
        )
        self.assertEqual(reopened.manifest["design_start"]["choice"], None)

    def test_choosing_local_ollama_at_setup_enters_the_builder_offline(self):
        self.start_page.setup_dialog_runner = self._runner(
            SetupOutcome(provider=LOCAL_OLLAMA, verified=False)
        )
        with patch.dict(os.environ, {}, clear=True):
            self.start_page.choose_template()
        self.app.update()

        self.assertEqual(self.app.active_page, "antenna_builder")
        reopened = self.store.open_project(self.project.path, touch=False)
        self.assertEqual(
            reopened.manifest["antenna_builder"]["planner_provider"], LOCAL_OLLAMA
        )
        self.assertEqual(self.app.antenna_builder_page.planner_var.get(), "Local Ollama")

    def test_an_existing_credential_opens_the_builder_without_asking(self):
        def refuse(*args, **kwargs):
            raise AssertionError("setup must not run when a credential exists")

        self.start_page.setup_dialog_runner = refuse
        with patch.dict(os.environ, {"GEMINI_API_KEY": SECRET}, clear=True):
            self.start_page.choose_template()
        self.app.update()
        self.assertEqual(self.app.active_page, "antenna_builder")

    def test_no_secret_reaches_the_project_record_or_the_planner_audit(self):
        self.start_page.setup_dialog_runner = self._runner(
            SetupOutcome(provider=GEMINI, verified=True)
        )
        with patch.dict(os.environ, {}, clear=True):
            self.start_page.choose_template()
        self.app.update()

        manifest_path = self.project.path / "project.json"
        rendered = manifest_path.read_text(encoding="utf-8")
        self.assertNotIn(SECRET, rendered)
        self.assertNotIn("api_key", rendered.casefold())

        audit = self.project.path / "design" / "planner_ab.jsonl"
        if audit.exists():
            self.assertNotIn(SECRET, audit.read_text(encoding="utf-8"))

        # And nothing anywhere under the project directory holds it.
        for path in self.project.path.rglob("*"):
            if path.is_file():
                try:
                    body = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                self.assertNotIn(SECRET, body, f"{path} contains the API key")


if __name__ == "__main__":
    unittest.main()
