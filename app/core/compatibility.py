from app import PROJECT_SCHEMA_VERSION


def assess_project_compatibility(manifest) -> tuple[str, str]:
    if manifest.project_schema_version == PROJECT_SCHEMA_VERSION:
        return "compatible", "Project schema matches this application version."
    if str(manifest.project_schema_version).split(".")[0] == PROJECT_SCHEMA_VERSION.split(".")[0]:
        return "migration_optional", "A minor schema migration may be useful. Back up before changing files."
    return "migration_required", "This project uses a different major schema version. Preserve files and migrate only with approval."
