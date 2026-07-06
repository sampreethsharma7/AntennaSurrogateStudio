from pathlib import Path
from typing import Optional

from app.utils.versioning import utc_timestamp


class ProjectLogger:
    def __init__(self, project_dir: Optional[Path] = None):
        self.project_dir = project_dir

    def bind_project(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        (project_dir / "logs").mkdir(parents=True, exist_ok=True)

    def write(self, level: str, message: str) -> str:
        line = f"[{utc_timestamp()}] {level.upper()}: {message}"
        if self.project_dir:
            with (self.project_dir / "logs" / "app.log").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return line
