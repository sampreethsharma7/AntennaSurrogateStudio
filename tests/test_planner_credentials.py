"""Credential resolution, storage and provider-side verification."""

import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from studio.antenna_llm_planner import (
    load_gemini_api_key,
    load_groq_api_key,
    load_openrouter_api_key,
)
import studio.planner_credentials as credentials
from studio.planner_credentials import (
    CREDENTIAL_STORE,
    ENV_FILE,
    ENVIRONMENT,
    GEMINI,
    GROQ,
    KEYRING_SERVICE,
    LOCAL_OLLAMA,
    OPENROUTER,
    CredentialStoreUnavailable,
    ResolvedCredential,
    configured_cloud_providers,
    delete_credential,
    has_credential,
    redact,
    resolve_credential,
    store_credential,
)
from studio.planner_model_discovery import (
    PROVIDER_LABELS,
    ModelDiscoveryError,
    PlannerModelDiscovery,
    friendly_discovery_message,
    verify_cloud_credential,
)


class FakeCredentialStore:
    """An in-memory stand-in for `keyring`, isolated from the real machine."""

    def __init__(self, values=None):
        self.values = dict(values or {})
        self.writes = []
        self.deletes = []

    def get_password(self, service, account):
        return self.values.get((service, account))

    def set_password(self, service, account, password):
        self.writes.append((service, account, password))
        self.values[(service, account)] = password

    def delete_password(self, service, account):
        self.deletes.append((service, account))
        del self.values[(service, account)]


class BrokenCredentialStore:
    """A store whose backend fails, as a locked or misconfigured one would."""

    def get_password(self, service, account):
        raise RuntimeError("the credential store is locked")

    def set_password(self, service, account, password):
        raise RuntimeError("the credential store is locked")

    def delete_password(self, service, account):
        raise RuntimeError("the credential store is locked")


def gemini_catalog(models=("gemini-3.8-flash",)):
    return {
        "models": [
            {
                "name": f"models/{model}",
                "displayName": model,
                "supportedGenerationMethods": ["generateContent"],
            }
            for model in models
        ]
    }


def recording_opener(payload, calls):
    def opener(request, timeout=None):
        calls.append({
            "url": request.full_url,
            "method": request.get_method(),
            "body": request.data,
            "headers": {str(k).lower(): str(v) for k, v in request.headers.items()},
        })
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    return opener


def failing_opener(error):
    def opener(request, timeout=None):
        raise error

    return opener


class CredentialPrecedenceTests(unittest.TestCase):
    def setUp(self):
        self.store = FakeCredentialStore({(KEYRING_SERVICE, "gemini"): "stored-key"})

    def test_process_environment_wins_over_every_other_source(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("GEMINI_API_KEY=file-key\n", encoding="utf-8")
            with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}, clear=True):
                resolved = resolve_credential(
                    GEMINI, env_file=env_file, backend=self.store
                )
        self.assertEqual(resolved.api_key, "env-key")
        self.assertEqual(resolved.source, ENVIRONMENT)

    def test_env_file_wins_over_the_credential_store(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("GEMINI_API_KEY='file-key'\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                resolved = resolve_credential(
                    GEMINI, env_file=env_file, backend=self.store
                )
        self.assertEqual(resolved.api_key, "file-key")
        self.assertEqual(resolved.source, ENV_FILE)

    def test_credential_store_is_used_when_nothing_else_supplies_a_key(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "absent.env"
            with patch.dict(os.environ, {}, clear=True):
                resolved = resolve_credential(
                    GEMINI, env_file=missing, backend=self.store, use_credential_store=True
                )
        self.assertEqual(resolved.api_key, "stored-key")
        self.assertEqual(resolved.source, CREDENTIAL_STORE)

    def test_nothing_configured_reports_no_source(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "absent.env"
            with patch.dict(os.environ, {}, clear=True):
                resolved = resolve_credential(
                    GEMINI, env_file=missing, backend=FakeCredentialStore()
                )
        self.assertFalse(resolved.found)
        self.assertIsNone(resolved.source)
        self.assertEqual(resolved.source_description, "no configured source")

    def test_local_ollama_needs_no_credential_and_is_never_gated(self):
        resolved = resolve_credential(LOCAL_OLLAMA, backend=self.store)
        self.assertFalse(resolved.found)
        self.assertFalse(credentials.provider(LOCAL_OLLAMA).needs_credential)
        with self.assertRaises(KeyError):
            store_credential(LOCAL_OLLAMA, "irrelevant", backend=self.store)

    def test_openrouter_keeps_its_legacy_variable_spelling(self):
        with patch.dict(os.environ, {"OPEN_ROUTER_API_KEY": "legacy"}, clear=True):
            self.assertEqual(
                resolve_credential(OPENROUTER, backend=FakeCredentialStore()).api_key,
                "legacy",
            )

    def test_configured_cloud_providers_lists_only_usable_ones(self):
        store = FakeCredentialStore({(KEYRING_SERVICE, "groq"): "groq-key"})
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "absent.env"
            with patch.dict(os.environ, {}, clear=True):
                configured = configured_cloud_providers(env_file=missing, backend=store)
        self.assertEqual(configured, (GROQ,))


class BackwardCompatibilityTests(unittest.TestCase):
    """The pre-dialog loaders must keep behaving exactly as they did."""

    def test_env_file_credentials_still_load_without_a_credential_store(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "# local only\nGEMINI_API_KEY='file-key'\nGROQ_API_KEY=groq-file\n"
                "OPENROUTER_API_KEY=router-file\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(load_gemini_api_key(env_file=env_file), "file-key")
                self.assertEqual(load_groq_api_key(env_file=env_file), "groq-file")
                self.assertEqual(
                    load_openrouter_api_key(env_file=env_file), "router-file"
                )

    def test_process_environment_credentials_still_load(self):
        environment = {
            "GEMINI_API_KEY": "env-gemini",
            "GROQ_API_KEY": "env-groq",
            "OPENROUTER_API_KEY": "env-router",
        }
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(load_gemini_api_key(), "env-gemini")
            self.assertEqual(load_groq_api_key(), "env-groq")
            self.assertEqual(load_openrouter_api_key(), "env-router")

    def test_an_explicit_env_file_does_not_reach_into_the_credential_store(self):
        # Callers that name a file are saying where the credential should come
        # from. Quietly answering from the machine's store instead would make
        # that request meaningless, and would leak a developer's real key into
        # tests that deliberately configure none.
        store = FakeCredentialStore({(KEYRING_SERVICE, "gemini"): "stored-key"})
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "absent.env"
            with patch.object(credentials, "_keyring_backend", return_value=store):
                with patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(load_gemini_api_key(env_file=missing), "")

    def test_loaders_consult_the_credential_store_when_no_file_is_named(self):
        store = FakeCredentialStore({
            (KEYRING_SERVICE, "gemini"): "stored-gemini",
            (KEYRING_SERVICE, "groq"): "stored-groq",
            (KEYRING_SERVICE, "openrouter"): "stored-router",
        })
        with patch.object(credentials, "_keyring_backend", return_value=store):
            with patch.dict(os.environ, {}, clear=True):
                with patch.object(
                    credentials, "read_environment_or_env_file", return_value=""
                ):
                    self.assertEqual(load_gemini_api_key(), "stored-gemini")
                    self.assertEqual(load_groq_api_key(), "stored-groq")
                    self.assertEqual(load_openrouter_api_key(), "stored-router")


class CredentialStorageTests(unittest.TestCase):
    def test_storing_replacing_and_removing_a_credential(self):
        store = FakeCredentialStore()
        store_credential(GEMINI, "first-key", backend=store)
        self.assertEqual(
            resolve_credential(
                GEMINI, env_file=Path("absent.env"), backend=store
            ).api_key,
            "first-key",
        )

        store_credential(GEMINI, "second-key", backend=store)
        self.assertEqual(
            resolve_credential(
                GEMINI, env_file=Path("absent.env"), backend=store
            ).api_key,
            "second-key",
        )

        self.assertTrue(delete_credential(GEMINI, backend=store))
        self.assertFalse(
            has_credential(GEMINI, env_file=Path("absent.env"), backend=store)
        )
        # Removing a credential that is already gone is the caller's intent, so
        # it reports no change rather than failing.
        self.assertFalse(delete_credential(GEMINI, backend=store))

    def test_keys_are_held_per_provider_under_one_service_name(self):
        store = FakeCredentialStore()
        store_credential(GEMINI, "gemini-key", backend=store)
        store_credential(GROQ, "groq-key", backend=store)
        accounts = {account for _service, account, _password in store.writes}
        services = {service for service, _account, _password in store.writes}
        self.assertEqual(services, {KEYRING_SERVICE})
        self.assertEqual(accounts, {"gemini", "groq"})

    def test_an_empty_key_is_refused_before_it_reaches_the_store(self):
        store = FakeCredentialStore()
        with self.assertRaises(ValueError):
            store_credential(GEMINI, "   ", backend=store)
        self.assertEqual(store.writes, [])

    def test_a_broken_store_degrades_instead_of_breaking_the_application(self):
        broken = BrokenCredentialStore()
        # Reading must fall through to the other sources rather than raise.
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}, clear=True):
            self.assertEqual(
                resolve_credential(GEMINI, backend=broken).api_key, "env-key"
            )
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(
                resolve_credential(
                    GEMINI, env_file=Path("absent.env"), backend=broken
                ).found
            )
        # Saving, however, must say so plainly: the user is waiting on it.
        with self.assertRaises(CredentialStoreUnavailable):
            store_credential(GEMINI, "new-key", backend=broken)

    def test_a_missing_keyring_package_is_reported_as_unavailable(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict("sys.modules", {"keyring": None}):
                with self.assertRaises(CredentialStoreUnavailable) as caught:
                    credentials._keyring_backend()
        self.assertIn("keyring", str(caught.exception))

    def test_the_credential_store_can_be_switched_off_entirely(self):
        with patch.dict(
            os.environ, {credentials.DISABLE_KEYRING_VARIABLE: "1"}, clear=True
        ):
            self.assertTrue(credentials.credential_store_disabled())
            self.assertFalse(credentials.credential_store_available())
            with self.assertRaises(CredentialStoreUnavailable):
                credentials._keyring_backend()


class SecretRedactionTests(unittest.TestCase):
    def test_a_resolved_credential_never_prints_its_own_secret(self):
        resolved = ResolvedCredential(GEMINI, "AIzaTotallyRealLookingKey", ENVIRONMENT)
        for rendered in (repr(resolved), str(resolved), f"{resolved}"):
            self.assertNotIn("AIzaTotallyRealLookingKey", rendered)
            self.assertIn("redacted", rendered)

    def test_redact_describes_a_key_without_revealing_it(self):
        self.assertEqual(redact("abcdefgh"), "<8 characters redacted>")
        self.assertEqual(redact(""), "<no key>")
        self.assertEqual(redact(None), "<no key>")


class CredentialVerificationTests(unittest.TestCase):
    def test_a_valid_key_verifies_and_sends_no_design_or_project_context(self):
        calls = []
        discovery = PlannerModelDiscovery(
            opener=recording_opener(gemini_catalog(), calls)
        )
        result = verify_cloud_credential(GEMINI, "candidate-key", discovery=discovery)

        self.assertEqual(result.provider, GEMINI)
        self.assertEqual(result.model_count, 1)
        self.assertTrue(result.recommended_model_available)

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call["method"], "GET")
        # The decisive property: there is no request body at all, so no
        # geometry, design state, parameters or instruction can be in it.
        self.assertIsNone(call["body"])
        self.assertEqual(call["headers"].get("x-goog-api-key"), "candidate-key")
        # The key must authenticate in a header, never in a logged URL.
        self.assertNotIn("candidate-key", call["url"])
        for forbidden in ("design", "antenna", "patch", "recipe", "instruction", "project"):
            self.assertNotIn(forbidden, call["url"].casefold())

    def test_verification_reports_when_the_recommended_model_is_absent(self):
        discovery = PlannerModelDiscovery(
            opener=recording_opener(gemini_catalog(("some-other-model",)), [])
        )
        result = verify_cloud_credential(GEMINI, "candidate-key", discovery=discovery)
        self.assertEqual(result.model_count, 1)
        self.assertFalse(result.recommended_model_available)

    def test_a_rejected_key_raises_and_is_never_stored(self):
        store = FakeCredentialStore()
        error = urllib.error.HTTPError(
            "https://example.invalid", 401, "Unauthorized", {}, None
        )
        discovery = PlannerModelDiscovery(opener=failing_opener(error))
        with self.assertRaises(ModelDiscoveryError) as caught:
            verify_cloud_credential(GEMINI, "bad-key", discovery=discovery)
        self.assertEqual(caught.exception.status, 401)
        self.assertEqual(store.writes, [])

        message = friendly_discovery_message(GEMINI, caught.exception)
        self.assertIn("couldn't verify this API key", message)
        self.assertNotIn("bad-key", message)
        self.assertNotIn("401", message)
        self.assertNotIn("HTTP", message)

    def test_a_network_failure_is_a_friendly_recoverable_message(self):
        discovery = PlannerModelDiscovery(
            opener=failing_opener(urllib.error.URLError("getaddrinfo failed"))
        )
        with self.assertRaises(ModelDiscoveryError) as caught:
            verify_cloud_credential(GEMINI, "candidate-key", discovery=discovery)
        self.assertEqual(caught.exception.kind, "network")
        message = friendly_discovery_message(GEMINI, caught.exception)
        self.assertIn("could not be reached", message)
        self.assertIn("internet connection", message)
        self.assertNotIn("getaddrinfo", message)

    def test_an_account_without_model_access_is_explained_as_such(self):
        discovery = PlannerModelDiscovery(
            opener=recording_opener({"models": []}, [])
        )
        with self.assertRaises(ModelDiscoveryError) as caught:
            verify_cloud_credential(GEMINI, "candidate-key", discovery=discovery)
        self.assertEqual(caught.exception.kind, "no_models")
        self.assertIn(
            "does not currently have access",
            friendly_discovery_message(GEMINI, caught.exception),
        )

    def test_every_cloud_provider_verifies_through_its_own_catalog(self):
        payloads = {
            GEMINI: (
                gemini_catalog(),
                "generativelanguage.googleapis.com",
                "x-goog-api-key",
            ),
            GROQ: (
                {"data": [{"id": "openai/gpt-oss-120b", "active": True}]},
                "api.groq.com",
                "authorization",
            ),
            OPENROUTER: (
                {"data": [{"id": "nvidia/nemotron-3-ultra-550b-a55b:free",
                           "name": "Nemotron"}]},
                "openrouter.ai",
                "authorization",
            ),
        }
        for provider_id, (payload, host, header) in payloads.items():
            with self.subTest(provider=provider_id):
                calls = []
                discovery = PlannerModelDiscovery(
                    opener=recording_opener(payload, calls)
                )
                result = verify_cloud_credential(
                    provider_id, "candidate-key", discovery=discovery
                )
                self.assertTrue(result.recommended_model_available)
                self.assertIn(host, calls[0]["url"])
                self.assertIsNone(calls[0]["body"])
                self.assertIn("candidate-key", calls[0]["headers"][header])

    def test_local_ollama_cannot_be_verified_as_a_cloud_credential(self):
        with self.assertRaises(ModelDiscoveryError) as caught:
            verify_cloud_credential(LOCAL_OLLAMA, "irrelevant")
        self.assertEqual(caught.exception.kind, "config")

    def test_friendly_messages_never_expose_provider_internals(self):
        noisy = ModelDiscoveryError(
            "provider returned HTTP 403 for key AIzaSecret at "
            "https://generativelanguage.googleapis.com/v1beta/models?key=AIzaSecret",
            kind="http",
            status=403,
        )
        message = friendly_discovery_message(GEMINI, noisy)
        for forbidden in ("AIzaSecret", "http", "403", "{", "Traceback"):
            self.assertNotIn(forbidden, message)


class ProviderCatalogAgreementTests(unittest.TestCase):
    def test_discovery_and_credentials_describe_the_same_providers(self):
        # The two modules are imported separately by the UI; a label or id that
        # drifts between them would show the user one provider and configure
        # another.
        self.assertEqual(PROVIDER_LABELS, credentials.PROVIDER_LABELS)

    def test_every_cloud_provider_can_tell_the_user_where_to_get_a_key(self):
        for provider_id in credentials.CLOUD_PROVIDERS:
            with self.subTest(provider=provider_id):
                facts = credentials.provider(provider_id)
                self.assertTrue(facts.api_key_url.startswith("https://"))
                self.assertTrue(facts.keyring_account)
                self.assertTrue(facts.environment_variables)

    def test_gemini_is_the_recommended_onboarding_provider(self):
        self.assertEqual(credentials.RECOMMENDED_CLOUD_PROVIDER, GEMINI)
        self.assertIn(GEMINI, credentials.CLOUD_PROVIDERS)

    def test_no_openai_provider_is_offered(self):
        for provider_id in credentials.PROVIDERS:
            self.assertNotIn("openai", provider_id)
        for label in credentials.PROVIDER_LABELS.values():
            self.assertNotIn("openai", label.casefold())


if __name__ == "__main__":
    unittest.main()
