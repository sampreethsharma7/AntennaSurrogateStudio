from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import uuid
from typing import Optional

from app import APP_VERSION, PROJECT_SCHEMA_VERSION
from app.utils.versioning import utc_timestamp


@dataclass
class ProjectManifest:
    project_name: str
    project_type: str
    description: str = ""
    antenna_category: str = ""
    units_preference: str = ""
    project_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    creation_timestamp: str = field(default_factory=utc_timestamp)
    last_modified_timestamp: str = field(default_factory=utc_timestamp)
    application_version: str = APP_VERSION
    project_schema_version: str = PROJECT_SCHEMA_VERSION
    data_paths: dict = field(default_factory=dict)
    selected_input_columns: list[str] = field(default_factory=list)
    selected_output_columns: list[str] = field(default_factory=list)
    input_units: dict = field(default_factory=dict)
    output_units: dict = field(default_factory=dict)
    input_bounds: dict = field(default_factory=dict)
    output_axis_metadata: dict = field(default_factory=dict)
    model_versions: list[dict] = field(default_factory=list)
    active_model_version: Optional[int] = None
    compatibility_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_file(cls, path: Path) -> "ProjectManifest":
        data = json.loads(path.read_text(encoding="utf-8"))
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in data.items() if key in allowed})

    def save(self, project_dir: Path) -> None:
        self.last_modified_timestamp = utc_timestamp()
        (project_dir / "project.json").write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
