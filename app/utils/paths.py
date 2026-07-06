from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[2]
PROJECTS_DIR = APP_DIR / "projects"
EXAMPLES_DIR = APP_DIR / "examples"
ASSETS_DIR = APP_DIR / "assets"


def ensure_app_dirs() -> None:
    for path in (PROJECTS_DIR, EXAMPLES_DIR, ASSETS_DIR):
        path.mkdir(parents=True, exist_ok=True)
