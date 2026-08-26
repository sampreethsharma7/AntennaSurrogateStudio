"""Portable project and recent-library persistence."""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_SCHEMA_VERSION = 1
WELCOME_SESSION_LIMIT = 50
PROJECT_SUBDIRECTORIES = (
    "data/raw",
    "data/prepared",
    "data/registered",
    "data/templates",
    "data/generated",
    "models",
    "books",
    "analysis",
    "inference",
    "inverse_design",
    "assistant",
    "logs",
)


class ProjectError(RuntimeError):
    """Raised when a project cannot be created, opened, or persisted."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_library_root() -> Path:
    override = os.environ.get("ANTENNA_STUDIO_LIBRARY", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / "Documents" / "Antenna Surrogate Studio Library"


def project_slug(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "-", name.strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-_")
    return cleaned or "untitled-antenna-project"


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


@dataclass(slots=True)
class Project:
    path: Path
    manifest: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.manifest.get("name") or self.path.name)

    @property
    def description(self) -> str:
        return str(self.manifest.get("description") or "")

    @property
    def workflow_stage(self) -> str:
        return str(self.manifest.get("workflow", {}).get("stage", "project_created"))

    @property
    def status_label(self) -> str:
        return {
            "project_created": "New project",
            "data_discovered": "In progress",
            "data_prepared": "Data ready",
            "dataset_registered": "Dataset registered",
            "model_trained": "Model trained",
            "model_saved": "Model saved",
        }.get(self.workflow_stage, "In progress")

    @property
    def last_opened_at(self) -> str:
        return str(self.manifest.get("last_opened_at") or self.manifest.get("created_at") or "")


class ProjectStore:
    """Owns the library index and portable project manifests."""

    def __init__(self, library_root: str | Path | None = None):
        self.library_root = Path(library_root or default_library_root()).expanduser().resolve()
        self.projects_root = self.library_root / "projects"
        self.index_path = self.library_root / "library_index.json"
        self.legacy_welcome_chat_path = (
            self.library_root / "assistant" / "welcome_chat_history.json"
        )
        self.welcome_sessions_root = (
            self.library_root / "assistant" / "welcome_sessions"
        )
        self.welcome_sessions_index_path = (
            self.library_root / "assistant" / "welcome_sessions_index.json"
        )
        self.welcome_session_id = ""
        self.welcome_chat_path = self.legacy_welcome_chat_path
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self.start_welcome_session()

    @property
    def welcome_session_count(self) -> int:
        return len(self._read_welcome_session_index().get("sessions", []))

    def start_welcome_session(self) -> str:
        """Create a fresh local Welcome conversation and retain prior sessions."""

        index = self._read_welcome_session_index()
        sessions = list(index.get("sessions", []))
        if not sessions and self.legacy_welcome_chat_path.exists():
            legacy_messages = self._load_chat_path(self.legacy_welcome_chat_path)
            if legacy_messages:
                legacy_id = f"legacy-{uuid.uuid4().hex[:8]}"
                legacy_path = self.welcome_sessions_root / f"{legacy_id}.json"
                atomic_write_json(
                    legacy_path,
                    {
                        "schema_version": 1,
                        "session_id": legacy_id,
                        "created_at": legacy_messages[0].get("created_at") or utc_now(),
                        "messages": legacy_messages[-200:],
                    },
                )
                sessions.append(
                    {
                        "session_id": legacy_id,
                        "created_at": legacy_messages[0].get("created_at") or utc_now(),
                        "path": legacy_path.name,
                        "label": "Imported welcome history",
                    }
                )

        created_at = utc_now()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        session_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"
        session_path = self.welcome_sessions_root / f"{session_id}.json"
        atomic_write_json(
            session_path,
            {
                "schema_version": 1,
                "session_id": session_id,
                "created_at": created_at,
                "messages": [],
            },
        )
        sessions.insert(
            0,
            {
                "session_id": session_id,
                "created_at": created_at,
                "path": session_path.name,
                "label": "Welcome session",
            },
        )
        atomic_write_json(
            self.welcome_sessions_index_path,
            {
                "schema_version": 1,
                "sessions": sessions[:WELCOME_SESSION_LIMIT],
            },
        )
        self.welcome_session_id = session_id
        self.welcome_chat_path = session_path
        return session_id

    def create_project(self, name: str, description: str = "") -> Project:
        display_name = name.strip()
        if not display_name:
            raise ProjectError("Enter a project name.")

        base_slug = project_slug(display_name)
        project_path = self.projects_root / base_slug
        suffix = 2
        while project_path.exists():
            project_path = self.projects_root / f"{base_slug}-{suffix}"
            suffix += 1

        try:
            for relative in PROJECT_SUBDIRECTORIES:
                (project_path / relative).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ProjectError(f"Could not create the project folder: {exc}") from exc

        now = utc_now()
        manifest = {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "project_id": str(uuid.uuid4()),
            "name": display_name,
            "slug": project_path.name,
            "description": description.strip(),
            "created_at": now,
            "updated_at": now,
            "last_opened_at": now,
            "workflow": {
                "stage": "project_created",
                "completed_steps": 1,
                "total_steps": 5,
                "next_action": "Load and prepare antenna data.",
            },
            "ui": {
                "last_page": "data",
            },
            "data_prep": {},
            "dataset_registry": {
                "dataset_count": 0,
                "active_dataset_id": None,
                "index": "data/registered/index.json",
            },
            "model_library": {
                "schema_version": 1,
                "book_count": 0,
                "active_book_id": None,
                "index": "books/index.json",
            },
            "inverse_design": {
                "schema_version": 1,
                "run_count": 0,
                "latest_run_id": None,
                "index": "inverse_design/index.json",
            },
        }
        self._write_manifest(project_path, manifest)
        self._ensure_chat_file(project_path)
        project = Project(project_path, manifest)
        self._register_recent(project)
        return project

    def open_project(self, path: str | Path, *, touch: bool = True) -> Project:
        project_path = Path(path).expanduser()
        if project_path.is_file() and project_path.name == "project.json":
            project_path = project_path.parent
        project_path = project_path.resolve()
        manifest_path = project_path / "project.json"
        if not manifest_path.exists():
            raise ProjectError("That folder is not an Antenna Surrogate Studio project.")

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProjectError(f"The project manifest could not be read: {exc}") from exc

        if not isinstance(manifest, dict) or not manifest.get("project_id"):
            raise ProjectError("The project manifest is missing its project identity.")

        schema = int(manifest.get("schema_version", 0))
        if schema > PROJECT_SCHEMA_VERSION:
            raise ProjectError(
                f"This project uses schema {schema}, newer than this Studio supports."
            )

        for relative in PROJECT_SUBDIRECTORIES:
            (project_path / relative).mkdir(parents=True, exist_ok=True)
        self._ensure_chat_file(project_path)

        workflow = dict(manifest.get("workflow") or {})
        raw_model_library = manifest.get("model_library")
        model_library = raw_model_library if isinstance(raw_model_library, dict) else {}
        if (
            workflow.get("stage") == "model_saved"
            and int(model_library.get("book_count") or 0) > 0
        ):
            has_active_book = bool(model_library.get("active_book_id"))
            workflow.update(
                {
                    "completed_steps": 5,
                    "total_steps": 5,
                    "next_action": (
                        "Run a prediction with the active Model Book."
                        if has_active_book
                        else "Open Model Library and set a Model Book as active."
                    ),
                }
            )
            manifest["workflow"] = workflow

        if touch:
            manifest["last_opened_at"] = utc_now()
            manifest["updated_at"] = manifest["last_opened_at"]
            self._write_manifest(project_path, manifest)

        project = Project(project_path, manifest)
        if touch:
            self._register_recent(project)
        return project

    def update_project(self, project: Project, changes: dict[str, Any]) -> Project:
        current = self.open_project(project.path, touch=False).manifest
        _deep_merge(current, changes)
        current["updated_at"] = utc_now()
        self._write_manifest(project.path, current)
        updated = Project(project.path, current)
        self._register_recent(updated)
        return updated

    def recent_projects(self, limit: int = 5) -> list[Project]:
        index = self._read_index()
        projects: list[Project] = []
        clean_entries: list[dict[str, str]] = []
        for entry in index.get("recent", []):
            raw_path = entry.get("path") if isinstance(entry, dict) else None
            if not raw_path:
                continue
            try:
                project = self.open_project(raw_path, touch=False)
            except ProjectError:
                continue
            projects.append(project)
            clean_entries.append(
                {"path": str(project.path), "last_opened_at": project.last_opened_at}
            )
            if len(projects) >= limit:
                break
        if clean_entries != index.get("recent", [])[: len(clean_entries)]:
            atomic_write_json(self.index_path, {"recent": clean_entries})
        return projects

    def load_chat(self, project: Project) -> list[dict[str, str]]:
        return self._load_chat_path(
            project.path / "assistant" / "chat_history.json"
        )

    def load_welcome_chat(self) -> list[dict[str, str]]:
        return self._load_chat_path(self.welcome_chat_path)

    def _load_chat_path(self, chat_path: Path) -> list[dict[str, str]]:
        try:
            payload = json.loads(chat_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        messages = payload.get("messages", []) if isinstance(payload, dict) else []
        return [
            {
                "role": str(item.get("role", "")),
                "content": str(item.get("content", "")),
                "created_at": str(item.get("created_at", "")),
            }
            for item in messages
            if isinstance(item, dict)
            and item.get("role") in {"user", "assistant"}
            and item.get("content")
        ]

    def append_chat(self, project: Project, role: str, content: str) -> dict[str, str]:
        return self._append_chat_path(
            project.path / "assistant" / "chat_history.json",
            self.load_chat(project),
            role,
            content,
        )

    def append_welcome_chat(self, role: str, content: str) -> dict[str, str]:
        return self._append_chat_path(
            self.welcome_chat_path,
            self.load_welcome_chat(),
            role,
            content,
        )

    def _append_chat_path(
        self,
        chat_path: Path,
        messages: list[dict[str, str]],
        role: str,
        content: str,
    ) -> dict[str, str]:
        if role not in {"user", "assistant"}:
            raise ValueError("Chat role must be 'user' or 'assistant'.")
        message = {"role": role, "content": content.strip(), "created_at": utc_now()}
        messages.append(message)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "messages": messages[-200:],
        }
        if chat_path == self.welcome_chat_path:
            payload.update(
                {
                    "session_id": self.welcome_session_id,
                    "created_at": self._welcome_session_created_at(),
                }
            )
        atomic_write_json(chat_path, payload)
        return message

    def clear_chat(self, project: Project) -> None:
        atomic_write_json(
            project.path / "assistant" / "chat_history.json",
            {"schema_version": 1, "messages": []},
        )

    def clear_welcome_chat(self) -> None:
        atomic_write_json(
            self.welcome_chat_path,
            {
                "schema_version": 1,
                "session_id": self.welcome_session_id,
                "created_at": self._welcome_session_created_at(),
                "messages": [],
            },
        )

    def _write_manifest(self, project_path: Path, manifest: dict[str, Any]) -> None:
        try:
            atomic_write_json(project_path / "project.json", manifest)
        except OSError as exc:
            raise ProjectError(f"Could not save the project: {exc}") from exc

    def _ensure_chat_file(self, project_path: Path) -> None:
        chat_path = project_path / "assistant" / "chat_history.json"
        if not chat_path.exists():
            atomic_write_json(chat_path, {"schema_version": 1, "messages": []})

    def _read_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {"recent": []}
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"recent": []}
        return payload if isinstance(payload, dict) else {"recent": []}

    def _read_welcome_session_index(self) -> dict[str, Any]:
        if not self.welcome_sessions_index_path.exists():
            return {"schema_version": 1, "sessions": []}
        try:
            payload = json.loads(
                self.welcome_sessions_index_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return {"schema_version": 1, "sessions": []}
        if not isinstance(payload, dict) or not isinstance(
            payload.get("sessions", []), list
        ):
            return {"schema_version": 1, "sessions": []}
        return payload

    def _welcome_session_created_at(self) -> str:
        for entry in self._read_welcome_session_index().get("sessions", []):
            if (
                isinstance(entry, dict)
                and entry.get("session_id") == self.welcome_session_id
            ):
                return str(entry.get("created_at") or utc_now())
        return utc_now()

    def _register_recent(self, project: Project) -> None:
        index = self._read_index()
        canonical = os.path.normcase(str(project.path.resolve()))
        entries = []
        for entry in index.get("recent", []):
            if not isinstance(entry, dict) or not entry.get("path"):
                continue
            other = os.path.normcase(str(Path(entry["path"]).expanduser().resolve()))
            if other != canonical:
                entries.append(entry)
        entries.insert(
            0,
            {
                "path": str(project.path),
                "last_opened_at": project.last_opened_at or utc_now(),
            },
        )
        atomic_write_json(self.index_path, {"recent": entries[:20]})


def _deep_merge(target: dict[str, Any], changes: dict[str, Any]) -> None:
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
