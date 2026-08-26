"""Project-local registration for validated training datasets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from studio.dataset_validation import (
    DatasetValidationResult,
    validate_dataset,
)
from studio.parser_engine import TrainingRequest
from studio.project_store import atomic_write_json, utc_now


REGISTRY_SCHEMA_VERSION = 1
DATASET_SCHEMA_VERSION = 1


class DatasetRegistrationError(RuntimeError):
    """Raised when a validated dataset cannot be registered safely."""


@dataclass(slots=True)
class RegisteredDataset:
    """A verified, immutable dataset snapshot owned by one project."""

    dataset_id: str
    name: str
    created_at: str
    fingerprint_sha256: str
    input_csv_path: Path
    output_csv_path: Path
    input_sha256: str
    output_sha256: str
    feature_columns: list[str]
    target_columns: list[str]
    sample_id_column: str | None
    sample_count: int
    feature_count: int
    target_count: int
    manifest_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "created_at": self.created_at,
            "fingerprint_sha256": self.fingerprint_sha256,
            "input_csv_path": str(self.input_csv_path),
            "output_csv_path": str(self.output_csv_path),
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "feature_columns": list(self.feature_columns),
            "target_columns": list(self.target_columns),
            "sample_id_column": self.sample_id_column,
            "sample_count": self.sample_count,
            "feature_count": self.feature_count,
            "target_count": self.target_count,
            "manifest_path": str(self.manifest_path),
        }


def register_dataset(
    project_path: str | Path,
    validation: DatasetValidationResult,
    *,
    name: str | None = None,
) -> RegisteredDataset:
    """Revalidate and register an immutable dataset snapshot in a project."""

    project_root, project_manifest = _open_project(project_path)
    current_validation = validate_dataset(
        TrainingRequest(
            input_csv_path=validation.input_csv_path,
            output_csv_path=validation.output_csv_path,
            feature_columns=list(validation.feature_columns),
            target_columns=list(validation.target_columns),
            sample_id_column=validation.sample_id_column,
        )
    )

    input_hash = _sha256(current_validation.input_csv_path)
    output_hash = _sha256(current_validation.output_csv_path)
    fingerprint = _dataset_fingerprint(
        current_validation,
        input_hash,
        output_hash,
    )
    dataset_id = f"dataset-{fingerprint[:12]}"
    registry_root = project_root / "data" / "registered"
    registry_root.mkdir(parents=True, exist_ok=True)
    index_path = registry_root / "index.json"
    index = _read_index(index_path)
    destination = registry_root / dataset_id

    if destination.exists():
        registered = _load_registered_dataset(destination / "dataset.json")
        if registered.fingerprint_sha256 != fingerprint:
            raise DatasetRegistrationError(
                f"Dataset ID collision detected for '{dataset_id}'."
            )
    else:
        display_name = (name or "").strip() or (
            f"Dataset {len(index['datasets']) + 1}"
        )
        registered = _create_snapshot(
            destination,
            display_name,
            dataset_id,
            fingerprint,
            current_validation,
            input_hash,
            output_hash,
        )

    summary = _index_entry(project_root, registered)
    existing_entries = [
        entry
        for entry in index["datasets"]
        if entry.get("dataset_id") != dataset_id
    ]
    existing_entries.append(summary)
    updated_index = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "active_dataset_id": dataset_id,
        "datasets": existing_entries,
    }
    try:
        atomic_write_json(index_path, updated_index)
        _update_project_manifest(
            project_root,
            project_manifest,
            dataset_count=len(existing_entries),
            active_dataset_id=dataset_id,
        )
    except OSError as exc:
        raise DatasetRegistrationError(
            f"The dataset registry could not be saved: {exc}"
        ) from exc
    return registered


def list_registered_datasets(
    project_path: str | Path,
) -> list[RegisteredDataset]:
    """Load all registered datasets in registration order."""

    project_root, _ = _open_project(project_path)
    index = _read_index(project_root / "data" / "registered" / "index.json")
    registered: list[RegisteredDataset] = []
    for entry in index["datasets"]:
        manifest_value = entry.get("manifest")
        if not isinstance(manifest_value, str) or not manifest_value:
            raise DatasetRegistrationError(
                "The dataset registry contains an invalid manifest path."
            )
        manifest_path = project_root / Path(manifest_value)
        registered.append(_load_registered_dataset(manifest_path))
    return registered


def get_registered_dataset(
    project_path: str | Path,
    dataset_id: str,
) -> RegisteredDataset:
    """Load one registered dataset by its stable content-based ID."""

    for dataset in list_registered_datasets(project_path):
        if dataset.dataset_id == dataset_id:
            return dataset
    raise DatasetRegistrationError(
        f"Registered dataset '{dataset_id}' was not found."
    )


def _open_project(project_path: str | Path) -> tuple[Path, dict[str, Any]]:
    project_root = Path(project_path).expanduser()
    if project_root.is_file() and project_root.name == "project.json":
        project_root = project_root.parent
    project_root = project_root.resolve()
    manifest_path = project_root / "project.json"
    if not manifest_path.is_file():
        raise DatasetRegistrationError(
            "Datasets can only be registered inside an Antenna Surrogate "
            "Studio project."
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetRegistrationError(
            f"The project manifest could not be read: {exc}"
        ) from exc
    if not isinstance(payload, dict) or not payload.get("project_id"):
        raise DatasetRegistrationError(
            "The project manifest is missing its project identity."
        )
    return project_root, payload


def _read_index(index_path: Path) -> dict[str, Any]:
    if not index_path.exists():
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "active_dataset_id": None,
            "datasets": [],
        }
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetRegistrationError(
            f"The dataset registry index could not be read: {exc}"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != REGISTRY_SCHEMA_VERSION
        or not isinstance(payload.get("datasets"), list)
    ):
        raise DatasetRegistrationError(
            "The dataset registry index has an unsupported structure."
        )
    return payload


def _create_snapshot(
    destination: Path,
    name: str,
    dataset_id: str,
    fingerprint: str,
    validation: DatasetValidationResult,
    input_hash: str,
    output_hash: str,
) -> RegisteredDataset:
    registry_root = destination.parent
    staging = Path(
        tempfile.mkdtemp(prefix=f".{dataset_id}.", dir=registry_root)
    )
    created_at = utc_now()
    try:
        input_snapshot = staging / "inputs.csv"
        output_snapshot = staging / "outputs.csv"
        shutil.copy2(validation.input_csv_path, input_snapshot)
        shutil.copy2(validation.output_csv_path, output_snapshot)
        if _sha256(input_snapshot) != input_hash or _sha256(output_snapshot) != output_hash:
            raise DatasetRegistrationError(
                "The source dataset changed while it was being registered."
            )

        record = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "name": name,
            "created_at": created_at,
            "fingerprint_sha256": fingerprint,
            "files": {
                "input": {"path": "inputs.csv", "sha256": input_hash},
                "output": {"path": "outputs.csv", "sha256": output_hash},
            },
            "contract": {
                "feature_columns": list(validation.feature_columns),
                "target_columns": list(validation.target_columns),
                "sample_id_column": validation.sample_id_column,
            },
            "shape": {
                "sample_count": validation.sample_count,
                "feature_count": validation.feature_count,
                "target_count": validation.target_count,
            },
        }
        atomic_write_json(staging / "dataset.json", record)
        try:
            os.replace(staging, destination)
        except OSError as exc:
            if not destination.exists():
                raise DatasetRegistrationError(
                    f"The dataset snapshot could not be finalized: {exc}"
                ) from exc
        return _load_registered_dataset(destination / "dataset.json")
    except OSError as exc:
        raise DatasetRegistrationError(
            f"The dataset snapshot could not be created: {exc}"
        ) from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _load_registered_dataset(manifest_path: Path) -> RegisteredDataset:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetRegistrationError(
            f"The registered dataset manifest could not be read: {exc}"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != DATASET_SCHEMA_VERSION
        or not payload.get("dataset_id")
    ):
        raise DatasetRegistrationError(
            "The registered dataset manifest has an unsupported structure."
        )
    try:
        files = payload["files"]
        contract = payload["contract"]
        shape = payload["shape"]
        input_path = manifest_path.parent / files["input"]["path"]
        output_path = manifest_path.parent / files["output"]["path"]
        input_hash = str(files["input"]["sha256"])
        output_hash = str(files["output"]["sha256"])
        feature_columns = [str(value) for value in contract["feature_columns"]]
        target_columns = [str(value) for value in contract["target_columns"]]
    except (KeyError, TypeError) as exc:
        raise DatasetRegistrationError(
            "The registered dataset manifest is incomplete."
        ) from exc

    if (
        not input_path.is_file()
        or not output_path.is_file()
        or _sha256(input_path) != input_hash
        or _sha256(output_path) != output_hash
    ):
        raise DatasetRegistrationError(
            f"Registered dataset '{payload['dataset_id']}' failed its integrity check."
        )

    return RegisteredDataset(
        dataset_id=str(payload["dataset_id"]),
        name=str(payload.get("name") or payload["dataset_id"]),
        created_at=str(payload.get("created_at") or ""),
        fingerprint_sha256=str(payload.get("fingerprint_sha256") or ""),
        input_csv_path=input_path,
        output_csv_path=output_path,
        input_sha256=input_hash,
        output_sha256=output_hash,
        feature_columns=feature_columns,
        target_columns=target_columns,
        sample_id_column=(
            str(contract["sample_id_column"])
            if contract.get("sample_id_column") is not None
            else None
        ),
        sample_count=int(shape["sample_count"]),
        feature_count=int(shape["feature_count"]),
        target_count=int(shape["target_count"]),
        manifest_path=manifest_path,
    )


def _dataset_fingerprint(
    validation: DatasetValidationResult,
    input_hash: str,
    output_hash: str,
) -> str:
    identity = {
        "input_sha256": input_hash,
        "output_sha256": output_hash,
        "feature_columns": list(validation.feature_columns),
        "target_columns": list(validation.target_columns),
        "sample_id_column": validation.sample_id_column,
        "sample_count": validation.sample_count,
    }
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DatasetRegistrationError(
            f"Could not read dataset file for fingerprinting: {path}: {exc}"
        ) from exc
    return digest.hexdigest()


def _index_entry(
    project_root: Path,
    dataset: RegisteredDataset,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset.dataset_id,
        "name": dataset.name,
        "created_at": dataset.created_at,
        "fingerprint_sha256": dataset.fingerprint_sha256,
        "sample_count": dataset.sample_count,
        "feature_count": dataset.feature_count,
        "target_count": dataset.target_count,
        "manifest": dataset.manifest_path.relative_to(project_root).as_posix(),
    }


def _update_project_manifest(
    project_root: Path,
    manifest: dict[str, Any],
    *,
    dataset_count: int,
    active_dataset_id: str,
) -> None:
    manifest["dataset_registry"] = {
        "dataset_count": dataset_count,
        "active_dataset_id": active_dataset_id,
        "index": "data/registered/index.json",
    }
    manifest["updated_at"] = utc_now()
    atomic_write_json(project_root / "project.json", manifest)
