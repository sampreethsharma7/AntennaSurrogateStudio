"""Machine-local Studio settings with merge-safe atomic persistence."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from studio.project_store import atomic_write_json


SETTINGS_SCHEMA_VERSION = 1
DEFAULT_APPEARANCE_MODE = "light"
VALID_APPEARANCE_MODES = frozenset({"light", "dark"})
_SETTINGS_LOCK = threading.RLock()


def studio_settings_path(library_root: str | Path) -> Path:
    return Path(library_root) / "studio_settings.json"


def normalize_appearance_mode(
    value: object,
    *,
    default: str = DEFAULT_APPEARANCE_MODE,
) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in VALID_APPEARANCE_MODES else default


def load_studio_settings(library_root: str | Path) -> dict[str, Any]:
    path = studio_settings_path(library_root)
    with _SETTINGS_LOCK:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return payload if isinstance(payload, dict) else {}


def update_studio_settings(
    library_root: str | Path,
    changes: dict[str, Any],
) -> dict[str, Any]:
    path = studio_settings_path(library_root)
    with _SETTINGS_LOCK:
        payload = load_studio_settings(library_root)
        payload["schema_version"] = SETTINGS_SCHEMA_VERSION
        _deep_merge(payload, changes)
        atomic_write_json(path, payload)
        return payload


def load_appearance_mode(library_root: str | Path) -> str:
    payload = load_studio_settings(library_root)
    ui_settings = payload.get("ui", {})
    if not isinstance(ui_settings, dict):
        return DEFAULT_APPEARANCE_MODE
    return normalize_appearance_mode(ui_settings.get("appearance_mode"))


def save_appearance_mode(
    library_root: str | Path,
    mode: str,
) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_APPEARANCE_MODES:
        raise ValueError("Appearance mode must be 'light' or 'dark'.")
    update_studio_settings(
        library_root,
        {"ui": {"appearance_mode": normalized}},
    )
    return normalized


def _deep_merge(target: dict[str, Any], changes: dict[str, Any]) -> None:
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
