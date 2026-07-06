from pathlib import Path
import json


def list_model_versions(manifest) -> list[dict]:
    return sorted(manifest.model_versions, key=lambda item: int(item.get("version", 0)))


def set_active_model(project_dir: Path, manifest, version: int) -> None:
    if version not in [int(item["version"]) for item in manifest.model_versions]:
        raise ValueError(f"Model version {version} does not exist.")
    manifest.active_model_version = version
    (project_dir / "models" / "active_model.json").write_text(json.dumps({"active_model_version": version}, indent=2), encoding="utf-8")
    manifest.save(project_dir)
