from pathlib import Path
import json
import shutil
import zipfile

from app.core.project_manifest import ProjectManifest
from app.utils.paths import PROJECTS_DIR
from app.utils.validation_helpers import sanitize_name
from app.utils.versioning import utc_timestamp


PROJECT_SUBDIRS = [
    "data/original",
    "data/processed",
    "models",
    "analysis/plots",
    "analysis/plot_data",
    "predictions/exported_predictions",
    "assistant",
    "logs",
    "backups",
]


class ProjectManager:
    def __init__(self, projects_dir: Path = PROJECTS_DIR):
        self.projects_dir = projects_dir
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def create_project(self, name: str, project_type: str, description: str = "", antenna_category: str = "", units: str = "") -> Path:
        base_name = sanitize_name(name)
        project_dir = self.projects_dir / base_name
        if project_dir.exists():
            project_dir = self.projects_dir / f"{base_name}_{utc_timestamp().replace(':', '').replace('-', '')}"
        for rel in PROJECT_SUBDIRS:
            (project_dir / rel).mkdir(parents=True, exist_ok=True)
        ProjectManifest(
            project_name=name.strip() or base_name,
            project_type=project_type,
            description=description,
            antenna_category=antenna_category,
            units_preference=units,
        ).save(project_dir)
        (project_dir / "assistant" / "conversation_history.json").write_text("[]", encoding="utf-8")
        return project_dir

    def load_manifest(self, project_dir: Path) -> ProjectManifest:
        return ProjectManifest.from_file(project_dir / "project.json")

    def recent_projects(self) -> list[Path]:
        projects = [p for p in self.projects_dir.iterdir() if (p / "project.json").exists()]
        return sorted(projects, key=lambda p: (p / "project.json").stat().st_mtime, reverse=True)

    def export_project_zip(self, project_dir: Path, destination: Path) -> Path:
        zip_path = destination / f"{project_dir.name}_bundle.zip"
        ignored = {".venv", ".venv_py311", "__pycache__", ".pytest_cache"}
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("README_INSIDE_PROJECT_ZIP.txt", "Portable Antenna Surrogate Studio project bundle.\n")
            for path in project_dir.rglob("*"):
                if any(part in ignored for part in path.parts):
                    continue
                zf.write(path, path.relative_to(project_dir.parent))
        return zip_path

    def import_project_zip(self, zip_path: Path) -> Path:
        target = self.projects_dir / zip_path.stem.replace("_bundle", "")
        if target.exists():
            target = self.projects_dir / f"{target.name}_{utc_timestamp().replace(':', '').replace('-', '')}"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(self.projects_dir)
        if not (target / "project.json").exists():
            candidates = [p for p in self.projects_dir.iterdir() if (p / "project.json").exists()]
            if not candidates:
                raise ValueError("The ZIP did not contain a valid project.json manifest.")
            target = max(candidates, key=lambda p: p.stat().st_mtime)
        return target

    def backup_project(self, project_dir: Path, reason: str) -> Path:
        backup_dir = project_dir / "backups" / f"backup_{reason}_{utc_timestamp().replace(':', '')}"
        shutil.copytree(project_dir, backup_dir, ignore=shutil.ignore_patterns("backups"))
        return backup_dir
