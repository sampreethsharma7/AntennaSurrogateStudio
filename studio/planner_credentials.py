"""Cloud planner credentials: where they come from, and where the GUI puts them.

Credentials have always been read from the process environment or an ignored
`.env` file, which assumes the user is comfortable editing files. The setup
dialog adds a third source -- the operating system credential store -- without
displacing the first two, so a developer's exported variable and an existing
user's `.env` keep working untouched.

Resolution order, highest first:

1. the process environment, so an explicitly exported variable always wins;
2. an ignored `.env` beside the working directory or the repository root;
3. the operating system credential store, where the setup dialog saves keys.

The order is deliberate. An environment variable is an explicit, visible act by
whoever launched the application, and silently overriding it with a value saved
months earlier in a GUI would be surprising. The credentials panel therefore
reports which source is in effect, so a stored key that is being shadowed is
visible rather than mysterious.

Keys are held per provider, never per project: a credential belongs to the
machine and its owner, not to a design. Nothing here writes a secret to a
project file, a log, or a `.env`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


LOCAL_OLLAMA = "local_ollama"
GEMINI = "gemini_cloud"
GROQ = "groq_cloud"
OPENROUTER = "openrouter_cloud"

# One service name for every provider, with the provider as the account, so the
# operating system shows a single recognisable Antenna Surrogate Studio entry
# per provider rather than one opaque blob.
KEYRING_SERVICE = "AntennaSurrogateStudio"

ENVIRONMENT = "environment"
ENV_FILE = "env_file"
CREDENTIAL_STORE = "credential_store"

SOURCE_DESCRIPTIONS = {
    ENVIRONMENT: "an environment variable set for this session",
    ENV_FILE: "the ignored .env file in the project root",
    CREDENTIAL_STORE: "this computer's credential store",
}

# Set this to any non-empty value to keep the application away from the
# operating system credential store entirely, for example on a locked-down
# machine or a build agent. Resolution then uses the environment and `.env`
# only, exactly as it did before the setup dialog existed.
DISABLE_KEYRING_VARIABLE = "ANTENNA_STUDIO_NO_CREDENTIAL_STORE"


class CredentialStoreUnavailable(RuntimeError):
    """Raised when the operating system credential store cannot be used."""


class CredentialBackend(Protocol):
    """The slice of `keyring` this module depends on."""

    def get_password(self, service: str, account: str) -> str | None: ...

    def set_password(self, service: str, account: str, password: str) -> None: ...

    def delete_password(self, service: str, account: str) -> None: ...


@dataclass(frozen=True, slots=True)
class PlannerProvider:
    """Static facts about one planner transport."""

    provider_id: str
    label: str
    environment_variables: tuple[str, ...] = ()
    keyring_account: str = ""
    api_key_url: str = ""
    capability_note: str = ""

    @property
    def needs_credential(self) -> bool:
        return bool(self.environment_variables)

    @property
    def primary_environment_variable(self) -> str:
        return self.environment_variables[0] if self.environment_variables else ""


PROVIDERS: dict[str, PlannerProvider] = {
    LOCAL_OLLAMA: PlannerProvider(
        LOCAL_OLLAMA,
        "Local Ollama",
        capability_note=(
            "Runs entirely on this computer and needs no API key. Local models "
            "are smaller, so they interpret fewer design requests correctly "
            "than the cloud planners."
        ),
    ),
    GEMINI: PlannerProvider(
        GEMINI,
        "Gemini",
        environment_variables=("GEMINI_API_KEY",),
        keyring_account="gemini",
        api_key_url="https://aistudio.google.com/apikey",
    ),
    GROQ: PlannerProvider(
        GROQ,
        "Groq",
        environment_variables=("GROQ_API_KEY",),
        keyring_account="groq",
        api_key_url="https://console.groq.com/keys",
    ),
    OPENROUTER: PlannerProvider(
        OPENROUTER,
        "OpenRouter",
        # The second spelling has been accepted since before the setup dialog
        # existed; dropping it would break those users' environments.
        environment_variables=("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY"),
        keyring_account="openrouter",
        api_key_url="https://openrouter.ai/settings/keys",
    ),
}

PROVIDER_LABELS = {provider_id: item.label for provider_id, item in PROVIDERS.items()}
CLOUD_PROVIDERS = tuple(
    provider_id for provider_id, item in PROVIDERS.items() if item.needs_credential
)
RECOMMENDED_CLOUD_PROVIDER = GEMINI


@dataclass(frozen=True, slots=True, repr=False)
class ResolvedCredential:
    """One credential lookup result, which never prints its own secret."""

    provider: str
    api_key: str
    source: str | None

    @property
    def found(self) -> bool:
        return bool(self.api_key)

    @property
    def source_description(self) -> str:
        return SOURCE_DESCRIPTIONS.get(self.source or "", "no configured source")

    def __repr__(self) -> str:
        # A credential reaches tracebacks, debuggers and log lines by accident.
        # The length distinguishes "present" from "empty" without leaking it.
        return (
            f"ResolvedCredential(provider={self.provider!r}, source={self.source!r}, "
            f"api_key=<{len(self.api_key)} characters redacted>)"
        )


def read_environment_or_env_file(
    name: str,
    *,
    env_file: str | Path | None = None,
) -> str:
    """Read one credential from the process environment or an ignored `.env`.

    This is the original pre-dialog lookup, unchanged: the environment first,
    then `.env` beside the working directory and beside the repository root.
    It lives here rather than beside the planners so that the credential store
    can sit behind the same resolution order without an import cycle.
    """

    direct = os.environ.get(name, "").strip()
    if direct:
        return direct
    candidates = [Path(env_file)] if env_file is not None else [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]
    for path in dict.fromkeys(candidates):
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip() == name:
                return value.strip().strip('"\'')
    return ""


def provider(provider_id: str) -> PlannerProvider:
    try:
        return PROVIDERS[provider_id]
    except KeyError:
        raise KeyError(f"Unknown planner provider: {provider_id}") from None


def credential_store_disabled() -> bool:
    return bool(os.environ.get(DISABLE_KEYRING_VARIABLE, "").strip())


def _keyring_backend() -> CredentialBackend:
    """Return the real `keyring` module, or explain why it cannot be used.

    `keyring` is imported here rather than at module scope so the application
    still starts, and `.env` credentials still work, on a machine where the
    package or a usable backend is missing.
    """

    if credential_store_disabled():
        raise CredentialStoreUnavailable(
            f"{DISABLE_KEYRING_VARIABLE} is set, so the credential store is not used."
        )
    try:
        import keyring
    except ImportError as exc:
        raise CredentialStoreUnavailable(
            "The 'keyring' package is not installed, so keys cannot be saved on "
            "this computer. Install it with 'pip install -r requirements.txt', or "
            "set the API key as an environment variable instead."
        ) from exc
    return keyring


def _resolve_backend(backend: CredentialBackend | None) -> CredentialBackend:
    return backend if backend is not None else _keyring_backend()


def credential_store_available(*, backend: CredentialBackend | None = None) -> bool:
    """Report whether a key could be saved, without raising."""

    try:
        _resolve_backend(backend)
    except CredentialStoreUnavailable:
        return False
    return True


def _store_failure(action: str, exc: Exception) -> CredentialStoreUnavailable:
    # Backend errors carry service and account names, never the secret, but they
    # are still implementation noise, so callers get one plain sentence.
    return CredentialStoreUnavailable(
        f"This computer's credential store could not {action} the key: {exc}"
    )


def read_stored_credential(
    provider_id: str,
    *,
    backend: CredentialBackend | None = None,
) -> str:
    """Read one provider's key from the credential store, or return an empty string."""

    item = provider(provider_id)
    if not item.needs_credential:
        return ""
    try:
        resolved = _resolve_backend(backend)
    except CredentialStoreUnavailable:
        return ""
    try:
        stored = resolved.get_password(KEYRING_SERVICE, item.keyring_account)
    except Exception:
        # A locked or misconfigured store must not stop the application from
        # falling back to the environment or `.env`.
        return ""
    return (stored or "").strip()


def store_credential(
    provider_id: str,
    api_key: str,
    *,
    backend: CredentialBackend | None = None,
) -> None:
    """Save one provider's key in the operating system credential store."""

    item = provider(provider_id)
    if not item.needs_credential:
        raise KeyError(f"{item.label} does not use an API key.")
    cleaned = (api_key or "").strip()
    if not cleaned:
        raise ValueError("An API key is required.")
    resolved = _resolve_backend(backend)
    try:
        resolved.set_password(KEYRING_SERVICE, item.keyring_account, cleaned)
    except Exception as exc:
        raise _store_failure("save", exc) from exc


def delete_credential(
    provider_id: str,
    *,
    backend: CredentialBackend | None = None,
) -> bool:
    """Remove one provider's stored key, reporting whether anything was removed."""

    item = provider(provider_id)
    if not item.needs_credential:
        return False
    try:
        resolved = _resolve_backend(backend)
    except CredentialStoreUnavailable:
        return False
    try:
        resolved.delete_password(KEYRING_SERVICE, item.keyring_account)
    except Exception:
        # keyring raises PasswordDeleteError when there is nothing to delete,
        # which is already the outcome the caller asked for.
        return False
    return True


def resolve_credential(
    provider_id: str,
    *,
    env_file: str | Path | None = None,
    backend: CredentialBackend | None = None,
    use_credential_store: bool = True,
) -> ResolvedCredential:
    """Resolve one provider's key in the documented precedence order."""

    item = provider(provider_id)
    if not item.needs_credential:
        return ResolvedCredential(provider_id, "", None)

    for name in item.environment_variables:
        direct = os.environ.get(name, "").strip()
        if direct:
            return ResolvedCredential(provider_id, direct, ENVIRONMENT)

    for name in item.environment_variables:
        # The reader checks the process environment too, which the loop above
        # has already rejected, so a hit here came from a file.
        from_file = read_environment_or_env_file(name, env_file=env_file).strip()
        if from_file:
            return ResolvedCredential(provider_id, from_file, ENV_FILE)

    if use_credential_store:
        stored = read_stored_credential(provider_id, backend=backend)
        if stored:
            return ResolvedCredential(provider_id, stored, CREDENTIAL_STORE)

    return ResolvedCredential(provider_id, "", None)


def has_credential(
    provider_id: str,
    *,
    env_file: str | Path | None = None,
    backend: CredentialBackend | None = None,
) -> bool:
    return resolve_credential(provider_id, env_file=env_file, backend=backend).found


def configured_cloud_providers(
    *,
    env_file: str | Path | None = None,
    backend: CredentialBackend | None = None,
) -> tuple[str, ...]:
    """Every cloud provider that currently has a usable key, in menu order."""

    return tuple(
        provider_id
        for provider_id in CLOUD_PROVIDERS
        if has_credential(provider_id, env_file=env_file, backend=backend)
    )


def redact(value: Any) -> str:
    """Describe a key without revealing it, for logs and error text."""

    text = str(value or "").strip()
    if not text:
        return "<no key>"
    return f"<{len(text)} characters redacted>"
