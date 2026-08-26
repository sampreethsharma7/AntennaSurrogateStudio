"""Opt-in, local-only SnowBuddy transcripts for development evaluation."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from studio.project_store import Project, utc_now


APP_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_CHANNEL = "development"
TRUTHY_VALUES = {"1", "true", "yes", "on"}


def development_logging_enabled() -> bool:
    """Require two deliberate switches so production cannot log by accident."""

    channel = os.environ.get("ANTENNA_STUDIO_BUILD_CHANNEL", "").strip().lower()
    enabled = os.environ.get("SNOWBUDDY_DEVELOPMENT_LOG", "").strip().lower()
    return channel == DEVELOPMENT_CHANNEL and enabled in TRUTHY_VALUES


def default_development_log_path() -> Path:
    override = os.environ.get("SNOWBUDDY_DEVELOPMENT_LOG_PATH", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return APP_ROOT / ".development" / "snowbuddy_sessions.jsonl"


class DevelopmentConversationLog:
    """Append development conversations without affecting the user workflow."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        path: str | Path | None = None,
    ):
        self.enabled = development_logging_enabled() if enabled is None else enabled
        self.path = Path(path).expanduser().resolve() if path else default_development_log_path()
        self.session_id = str(uuid.uuid4())
        self.last_error = ""

    def record(
        self,
        *,
        project: Project | None,
        question: str,
        response: str,
        model: str,
        used_local_model: bool,
        live_ui_state: str,
    ) -> bool:
        if not self.enabled:
            return False

        event: dict[str, Any] = {
            "schema_version": 1,
            "created_at": utc_now(),
            "session_id": self.session_id,
            "mode": "focus" if project else "welcome",
            "project_id": (
                str(project.manifest.get("project_id") or "") if project else None
            ),
            "project_name": project.name if project else None,
            "model": model,
            "response_source": "ollama" if used_local_model else "built_in_fallback",
            "question": question,
            "response": response,
            "live_ui_state": live_ui_state,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(event, ensure_ascii=False))
                handle.write("\n")
        except OSError as exc:
            self.last_error = str(exc)
            return False
        self.last_error = ""
        return True
