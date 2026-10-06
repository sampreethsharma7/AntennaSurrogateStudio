"""Provider model discovery for the antenna planner configuration UI."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from studio.assistant import local_ollama_base_url
from studio.planner_credentials import (
    GEMINI,
    GROQ,
    LOCAL_OLLAMA,
    OPENROUTER,
    PROVIDER_LABELS,
    provider as provider_facts,
    resolve_credential,
)


# Provider identity lives in `planner_credentials`, which also knows where each
# provider's key comes from. These names are re-exported because the builder UI
# and its tests have imported them from here since before that module existed.
__all__ = [
    "DEFAULT_MODELS",
    "GEMINI",
    "GROQ",
    "LABEL_TO_PROVIDER",
    "LOCAL_OLLAMA",
    "OPENROUTER",
    "PROVIDER_LABELS",
    "CredentialVerification",
    "ModelDiscoveryError",
    "PlannerModel",
    "PlannerModelDiscovery",
    "TESTED_MODELS",
    "friendly_discovery_message",
    "verify_cloud_credential",
]

LABEL_TO_PROVIDER = {label: provider for provider, label in PROVIDER_LABELS.items()}

TESTED_MODELS = {
    LOCAL_OLLAMA: frozenset({"qwen3:8b"}),
    GEMINI: frozenset({"gemini-3.8-flash"}),
    GROQ: frozenset({"openai/gpt-oss-120b"}),
    OPENROUTER: frozenset({"nvidia/nemotron-3-ultra-550b-a55b:free"}),
}

DEFAULT_MODELS = {provider: next(iter(models)) for provider, models in TESTED_MODELS.items()}


class ModelDiscoveryError(RuntimeError):
    """Raised when a configured provider cannot return a usable model catalog.

    `kind` and `status` let a caller translate the failure into user-facing
    wording without parsing the message, which carries provider detail that
    belongs in a developer log rather than in a dialog.
    """

    def __init__(self, message: str, *, kind: str = "provider", status: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status


@dataclass(frozen=True, slots=True)
class PlannerModel:
    provider: str
    model_id: str
    display_name: str
    tested: bool
    free: bool | None = None

    @property
    def menu_label(self) -> str:
        status = "Tested" if self.tested else "Untested"
        identity = (
            self.display_name
            if self.display_name == self.model_id
            else f"{self.display_name} ({self.model_id})"
        )
        return f"{identity}  ·  {status}"


_IRRELEVANT_TOKENS = (
    "embed", "embedding", "whisper", "transcribe", "speech", "tts",
    "audio", "guard", "moderation", "rerank",
)


def _obviously_irrelevant(model_id: str, metadata: Mapping[str, Any] | None = None) -> bool:
    normalized = model_id.casefold()
    if any(token in normalized for token in _IRRELEVANT_TOKENS):
        return True
    metadata = metadata or {}
    architecture = metadata.get("architecture")
    if isinstance(architecture, Mapping):
        outputs = architecture.get("output_modalities")
        if isinstance(outputs, list) and outputs and "text" not in {str(value).casefold() for value in outputs}:
            return True
        modality = str(architecture.get("modality", "")).casefold()
        if modality and "text" not in modality:
            return True
    return False


def _read_json(
    request: urllib.request.Request,
    *,
    timeout: int,
    opener: Callable[..., Any],
) -> dict[str, Any]:
    try:
        with opener(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ModelDiscoveryError(
            f"provider returned HTTP {exc.code}", kind="http", status=exc.code
        ) from exc
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        raise ModelDiscoveryError(
            str(exc.reason) if isinstance(exc, urllib.error.URLError) else str(exc),
            kind="network",
        ) from exc
    if not isinstance(payload, dict):
        raise ModelDiscoveryError(
            "provider returned a malformed model catalog", kind="payload"
        )
    return payload


def _planner_model(provider: str, model_id: str, display_name: str | None = None, *, free: bool | None = None) -> PlannerModel:
    return PlannerModel(
        provider,
        model_id,
        (display_name or model_id).strip(),
        model_id in TESTED_MODELS.get(provider, ()),
        free,
    )


def parse_ollama_models(payload: Mapping[str, Any]) -> tuple[PlannerModel, ...]:
    models = []
    for item in payload.get("models", ()):
        if not isinstance(item, Mapping):
            continue
        model_id = str(item.get("model") or item.get("name") or "").strip()
        if model_id and not _obviously_irrelevant(model_id, item):
            models.append(_planner_model(LOCAL_OLLAMA, model_id))
    return _sorted_unique(models)


def parse_gemini_models(payload: Mapping[str, Any]) -> tuple[PlannerModel, ...]:
    models = []
    for item in payload.get("models", ()):
        if not isinstance(item, Mapping):
            continue
        methods = {str(value) for value in item.get("supportedGenerationMethods", ())}
        if "generateContent" not in methods:
            continue
        model_id = str(item.get("baseModelId") or item.get("name") or "").removeprefix("models/").strip()
        if model_id and not _obviously_irrelevant(model_id, item):
            models.append(_planner_model(GEMINI, model_id, str(item.get("displayName") or model_id)))
    return _sorted_unique(models)


def parse_groq_models(payload: Mapping[str, Any]) -> tuple[PlannerModel, ...]:
    models = []
    for item in payload.get("data", ()):
        if not isinstance(item, Mapping) or item.get("active") is False:
            continue
        model_id = str(item.get("id") or "").strip()
        if model_id and not _obviously_irrelevant(model_id, item):
            models.append(_planner_model(GROQ, model_id))
    return _sorted_unique(models)


def _openrouter_is_free(item: Mapping[str, Any]) -> bool:
    model_id = str(item.get("id") or "")
    if model_id.endswith(":free"):
        return True
    pricing = item.get("pricing")
    if not isinstance(pricing, Mapping):
        return False
    try:
        return float(pricing.get("prompt", 1)) == 0.0 and float(pricing.get("completion", 1)) == 0.0
    except (TypeError, ValueError):
        return False


def parse_openrouter_models(payload: Mapping[str, Any], *, free_only: bool) -> tuple[PlannerModel, ...]:
    models = []
    for item in payload.get("data", ()):
        if not isinstance(item, Mapping):
            continue
        model_id = str(item.get("id") or "").strip()
        free = _openrouter_is_free(item)
        if not model_id or _obviously_irrelevant(model_id, item) or (free_only and not free):
            continue
        models.append(_planner_model(OPENROUTER, model_id, str(item.get("name") or model_id), free=free))
    return _sorted_unique(models)


def _sorted_unique(models: list[PlannerModel]) -> tuple[PlannerModel, ...]:
    by_id = {model.model_id: model for model in models}
    return tuple(sorted(by_id.values(), key=lambda item: (not item.tested, item.display_name.casefold(), item.model_id)))


class PlannerModelDiscovery:
    def __init__(self, *, timeout: int = 12, opener: Callable[..., Any] = urllib.request.urlopen) -> None:
        self.timeout = timeout
        self._opener = opener

    def discover(
        self,
        provider: str,
        *,
        ollama_base_url: str,
        free_only: bool = False,
        api_key: str | None = None,
    ) -> tuple[PlannerModel, ...]:
        """List the text-generation models one provider will actually serve.

        `api_key` tries a candidate key that has not been saved anywhere yet,
        which is what the setup dialog verifies with. Left unset, the key is
        resolved from the environment, `.env`, then the credential store.
        """

        def credential() -> str:
            if api_key is not None:
                candidate = api_key.strip()
                if not candidate:
                    raise ModelDiscoveryError("no API key was supplied", kind="config")
                return candidate
            resolved = resolve_credential(provider)
            if not resolved.found:
                raise ModelDiscoveryError(
                    f"{provider_facts(provider).label} has no API key configured",
                    kind="config",
                )
            return resolved.api_key

        if provider == LOCAL_OLLAMA:
            try:
                base = local_ollama_base_url(ollama_base_url)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ModelDiscoveryError(str(exc)) from exc
            payload = _read_json(
                urllib.request.Request(f"{base}/api/tags", headers={"Accept": "application/json"}),
                timeout=self.timeout,
                opener=self._opener,
            )
            models = parse_ollama_models(payload)
        elif provider == GEMINI:
            key = credential()
            models_list: list[PlannerModel] = []
            token = ""
            while True:
                query = urllib.parse.urlencode({"pageSize": 1000, **({"pageToken": token} if token else {})})
                payload = _read_json(
                    urllib.request.Request(
                        f"https://generativelanguage.googleapis.com/v1beta/models?{query}",
                        headers={"Accept": "application/json", "x-goog-api-key": key},
                    ),
                    timeout=self.timeout,
                    opener=self._opener,
                )
                models_list.extend(parse_gemini_models(payload))
                token = str(payload.get("nextPageToken") or "")
                if not token:
                    break
            models = _sorted_unique(models_list)
        elif provider == GROQ:
            key = credential()
            payload = _read_json(
                urllib.request.Request(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Accept": "application/json", "Authorization": f"Bearer {key}"},
                ),
                timeout=self.timeout,
                opener=self._opener,
            )
            models = parse_groq_models(payload)
        elif provider == OPENROUTER:
            key = credential()
            payload = _read_json(
                urllib.request.Request(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Accept": "application/json", "Authorization": f"Bearer {key}"},
                ),
                timeout=self.timeout,
                opener=self._opener,
            )
            models = parse_openrouter_models(payload, free_only=free_only)
        else:
            raise ModelDiscoveryError(
                f"Unknown planner provider: {provider}", kind="config"
            )
        if not models:
            raise ModelDiscoveryError(
                "provider returned no compatible text-generation models", kind="no_models"
            )
        return models


@dataclass(frozen=True, slots=True)
class CredentialVerification:
    """What one provider confirmed about a candidate key."""

    provider: str
    model_count: int
    recommended_model: str
    recommended_model_available: bool


def verify_cloud_credential(
    provider: str,
    api_key: str,
    *,
    discovery: PlannerModelDiscovery | None = None,
) -> CredentialVerification:
    """Confirm with the provider that a candidate key really works.

    This authenticates against the provider's own model catalog, which is the
    smallest authenticated request each of them offers and the same call the
    Refresh button already makes. It proves three things at once: the key is
    accepted, the provider is reachable, and the account can actually see
    models.

    No antenna geometry, project state, design parameters or user instruction
    are involved -- the request body is empty and the only thing sent is the
    key itself, to the provider the user picked.

    Raises `ModelDiscoveryError` when the key is rejected, the provider cannot
    be reached, or the account has no usable model.
    """

    facts = provider_facts(provider)
    if not facts.needs_credential:
        raise ModelDiscoveryError(
            f"{facts.label} does not use an API key.", kind="config"
        )
    client = discovery or PlannerModelDiscovery()
    models = client.discover(
        provider,
        # Only the Ollama branch reads this, and that branch is unreachable for
        # a provider that needs a credential.
        ollama_base_url="",
        free_only=False,
        api_key=api_key,
    )
    recommended = DEFAULT_MODELS[provider]
    return CredentialVerification(
        provider=provider,
        model_count=len(models),
        recommended_model=recommended,
        recommended_model_available=any(model.model_id == recommended for model in models),
    )


def friendly_discovery_message(provider: str, error: ModelDiscoveryError) -> str:
    """Translate a provider failure into one sentence a normal user can act on.

    The original message stays on the exception for the developer log. It can
    carry provider wording, HTTP status text and URLs, none of which helps an
    RF engineer decide what to do next.
    """

    label = PROVIDER_LABELS.get(provider, "The provider")
    status = getattr(error, "status", None)
    kind = getattr(error, "kind", "provider")

    if kind == "network":
        return (
            f"{label} could not be reached. Check your internet connection and "
            "try again."
        )
    if kind == "no_models":
        return (
            "This account does not currently have access to a usable model. "
            "Check the account's model access with the provider, then try again."
        )
    if kind == "config":
        return "Enter an API key to continue."
    if status in (401, 403):
        return (
            "We couldn't verify this API key. Check that it was copied in full "
            f"and is still active in your {label} account."
        )
    if status == 400:
        return (
            "We couldn't verify this API key. It looks incomplete or is not in "
            f"the format {label} expects."
        )
    if status == 404:
        return (
            "This account does not currently have access to the required model."
        )
    if status == 429:
        return (
            f"{label} is rate limiting this key right now. Wait a moment and "
            "try again."
        )
    if status is not None and 500 <= status < 600:
        return (
            f"{label} reported a temporary server problem. Try again in a "
            "few minutes."
        )
    return (
        f"We couldn't verify this API key with {label}. Check the key and your "
        "internet connection, then try again."
    )
