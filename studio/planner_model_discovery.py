"""Provider model discovery for the antenna planner configuration UI."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from studio.antenna_llm_planner import (
    load_gemini_api_key,
    load_groq_api_key,
    load_openrouter_api_key,
)
from studio.assistant import local_ollama_base_url


LOCAL_OLLAMA = "local_ollama"
GEMINI = "gemini_cloud"
GROQ = "groq_cloud"
OPENROUTER = "openrouter_cloud"

PROVIDER_LABELS = {
    LOCAL_OLLAMA: "Local Ollama",
    GEMINI: "Gemini",
    GROQ: "Groq",
    OPENROUTER: "OpenRouter",
}
LABEL_TO_PROVIDER = {label: provider for provider, label in PROVIDER_LABELS.items()}

TESTED_MODELS = {
    LOCAL_OLLAMA: frozenset({"qwen3:8b"}),
    GEMINI: frozenset({"gemini-3.8-flash"}),
    GROQ: frozenset({"openai/gpt-oss-120b"}),
    OPENROUTER: frozenset({"nvidia/nemotron-3-ultra-550b-a55b:free"}),
}

DEFAULT_MODELS = {provider: next(iter(models)) for provider, models in TESTED_MODELS.items()}


class ModelDiscoveryError(RuntimeError):
    """Raised when a configured provider cannot return a usable model catalog."""


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
        raise ModelDiscoveryError(f"provider returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        raise ModelDiscoveryError(str(exc.reason) if isinstance(exc, urllib.error.URLError) else str(exc)) from exc
    if not isinstance(payload, dict):
        raise ModelDiscoveryError("provider returned a malformed model catalog")
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
    ) -> tuple[PlannerModel, ...]:
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
            key = load_gemini_api_key()
            if not key:
                raise ModelDiscoveryError("GEMINI_API_KEY is not configured")
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
            key = load_groq_api_key()
            if not key:
                raise ModelDiscoveryError("GROQ_API_KEY is not configured")
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
            key = load_openrouter_api_key()
            if not key:
                raise ModelDiscoveryError("OPENROUTER_API_KEY is not configured")
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
            raise ModelDiscoveryError(f"Unknown planner provider: {provider}")
        if not models:
            raise ModelDiscoveryError("provider returned no compatible text-generation models")
        return models
