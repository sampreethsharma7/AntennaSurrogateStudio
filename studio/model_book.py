"""Project-local promotion and loading of reusable trained Model Books."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from studio.dataset_registry import (
    DatasetRegistrationError,
    get_registered_dataset,
)
from studio.model_training import TRAINING_COMPLETED
from studio.output_axis import (
    OutputAxisMetadata,
    infer_output_axis,
    output_axis_from_dict,
)
from studio.project_store import atomic_write_json, utc_now


MODEL_BOOK_SCHEMA_VERSION = 1
MODEL_BOOK_VERSION = "1.0"
MODEL_BOOK_INDEX_SCHEMA_VERSION = 1
MODEL_BOOK_MANIFEST_NAME = "model_book.json"
MODEL_BOOK_MODEL_TYPES = {
    "linear_regression": "sklearn.linear_model.LinearRegression",
    "xgboost": "xgboost.sklearn.XGBRegressor",
    "neural_network": "sklearn.pipeline.Pipeline[StandardScaler,MLPRegressor]",
    "ensemble_ai_engine": "studio.ensemble.WeightedEnsembleRegressor",
}


class ModelBookError(RuntimeError):
    """Raised when a training run cannot be saved or loaded as a Model Book."""


@dataclass(slots=True)
class ModelBook:
    """Verified metadata and artifact paths for one reusable surrogate model."""

    project_path: Path
    directory: Path
    manifest_path: Path
    model_artifact_path: Path
    book_id: str
    name: str
    version: str
    created_at: str
    model_name: str
    model_type: str
    feature_columns: list[str]
    target_columns: list[str]
    sample_id_column: str | None
    parameters_used: dict[str, Any]
    training_mode: str
    search_level: str | None
    dataset_id: str
    dataset_fingerprint: str
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    source_run_id: str
    source_run_number: int
    source_trained_at: str
    model_sha256: str
    output_axis: OutputAxisMetadata | None = None
    ensemble_components: list[dict[str, Any]] | None = None
    component_artifact_paths: dict[str, Path] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "book_id": self.book_id,
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "feature_columns": list(self.feature_columns),
            "target_columns": list(self.target_columns),
            "sample_id_column": self.sample_id_column,
            "parameters_used": dict(self.parameters_used),
            "training_mode": self.training_mode,
            "search_level": self.search_level,
            "dataset_id": self.dataset_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "validation_metrics": dict(self.validation_metrics),
            "test_metrics": dict(self.test_metrics),
            "source_run_id": self.source_run_id,
            "source_run_number": self.source_run_number,
            "source_trained_at": self.source_trained_at,
            "model_sha256": self.model_sha256,
            "output_axis": (
                self.output_axis.to_dict() if self.output_axis is not None else None
            ),
            "ensemble_components": (
                [dict(component) for component in self.ensemble_components]
                if self.ensemble_components is not None
                else None
            ),
            "component_artifact_paths": (
                {
                    name: str(path)
                    for name, path in self.component_artifact_paths.items()
                }
                if self.component_artifact_paths is not None
                else None
            ),
            "manifest_path": str(self.manifest_path),
            "model_artifact_path": str(self.model_artifact_path),
        }


@dataclass(slots=True)
class ModelBookLibraryEntry:
    """One index entry, including a friendly error for an invalid book."""

    book_id: str
    name: str
    is_active: bool
    book: ModelBook | None = None
    error_message: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.book is not None and self.error_message is None


@dataclass(slots=True)
class ModelBookLibrary:
    """Project-local Model Book index resolved for browsing."""

    project_path: Path
    active_book_id: str | None
    entries: list[ModelBookLibraryEntry]

    @property
    def valid_book_count(self) -> int:
        return sum(entry.is_valid for entry in self.entries)


def save_model_book(
    project_path: str | Path,
    source_run_id: str,
    name: str,
) -> ModelBook:
    """Copy one completed training run into a new immutable Model Book."""

    project_root, project_manifest = _read_project(project_path)
    display_name = _validated_name(name)
    books_root = project_root / "books"
    books_root.mkdir(parents=True, exist_ok=True)
    index_path = books_root / "index.json"
    index = _read_index(index_path)
    if any(
        str(entry.get("name") or "").casefold() == display_name.casefold()
        for entry in index["books"]
        if isinstance(entry, dict)
    ):
        raise ModelBookError(
            f"A Model Book named '{display_name}' already exists. "
            "Choose a different name."
        )

    run_summary, run_record, run_directory = _load_completed_run(
        project_root,
        project_manifest,
        source_run_id,
    )
    source = _collect_source_metadata(
        project_root,
        run_summary,
        run_record,
        run_directory,
    )
    book_number = _next_book_number(books_root, index["books"])
    book_id = f"book-{book_number:04d}"
    destination = books_root / book_id
    if destination.exists():
        raise ModelBookError(
            f"The Model Book folder '{book_id}' already exists. No files were overwritten."
        )

    created_at = utc_now()
    model_filename = source["model_path"].name
    model_hash = _sha256(source["model_path"])
    payload = {
        "schema_version": MODEL_BOOK_SCHEMA_VERSION,
        "model_book_version": MODEL_BOOK_VERSION,
        "book_id": book_id,
        "name": display_name,
        "created_at": created_at,
        "model": {
            "name": source["model_name"],
            "type": source["model_type"],
            "artifact": {
                "path": model_filename,
                "sha256": model_hash,
            },
        },
        "interface": {
            "feature_columns": source["feature_columns"],
            "target_columns": source["target_columns"],
            "sample_id_column": source["sample_id_column"],
            "output_axis": source["output_axis"].to_dict(),
        },
        "training": {
            "mode": source["training_mode"],
            "search_level": source["search_level"],
            "parameters_used": source["parameters_used"],
            "split": source["split"],
        },
        "dataset": {
            "dataset_id": source["dataset_id"],
            "fingerprint_sha256": source["dataset_fingerprint"],
        },
        "performance": {
            "validation_metrics": source["validation_metrics"],
            "test_metrics": source["test_metrics"],
        },
        "source": {
            "run_id": source["run_id"],
            "run_number": source["run_number"],
            "trained_at": source["trained_at"],
            "run_manifest": (
                (run_directory / "run.json")
                .relative_to(project_root)
                .as_posix()
            ),
        },
    }
    if source.get("ensemble_components"):
        payload["model"]["ensemble"] = {
            "weighting_method": "inverse_validation_rmse",
            "components": [
                {
                    **{
                        key: value
                        for key, value in component.items()
                        if key not in {"artifact_source"}
                    },
                    "artifact": {
                        "path": f"components/{component['artifact_source'].name}",
                        "sha256": _sha256(component["artifact_source"]),
                    },
                }
                for component in source["ensemble_components"]
            ],
        }

    staging = Path(tempfile.mkdtemp(prefix=f".{book_id}.", dir=books_root))
    try:
        staged_model = staging / model_filename
        shutil.copy2(source["model_path"], staged_model)
        if _sha256(staged_model) != model_hash:
            raise ModelBookError(
                "The trained model changed while the Model Book was being saved."
            )
        for component in source.get("ensemble_components") or []:
            component_directory = staging / "components"
            component_directory.mkdir(exist_ok=True)
            source_path = component["artifact_source"]
            staged_component = component_directory / source_path.name
            shutil.copy2(source_path, staged_component)
            if _sha256(staged_component) != _sha256(source_path):
                raise ModelBookError(
                    "An Ensemble component changed while the Model Book was being saved."
                )
        atomic_write_json(staging / MODEL_BOOK_MANIFEST_NAME, payload)
        os.replace(staging, destination)
    except OSError as exc:
        raise ModelBookError(f"The Model Book could not be saved: {exc}") from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    entry = {
        "book_id": book_id,
        "name": display_name,
        "version": MODEL_BOOK_VERSION,
        "created_at": created_at,
        "source_run_id": source["run_id"],
        "model_name": source["model_name"],
        "manifest": (
            destination / MODEL_BOOK_MANIFEST_NAME
        ).relative_to(project_root).as_posix(),
    }
    updated_entries = [*index["books"], entry]
    active_book_id = index.get("active_book_id")
    updated_index = {
        "schema_version": MODEL_BOOK_INDEX_SCHEMA_VERSION,
        "active_book_id": active_book_id,
        "books": updated_entries,
    }
    try:
        atomic_write_json(index_path, updated_index)
        project_manifest["model_library"] = {
            "schema_version": MODEL_BOOK_INDEX_SCHEMA_VERSION,
            "book_count": len(updated_entries),
            "active_book_id": active_book_id,
            "index": "books/index.json",
        }
        workflow = dict(project_manifest.get("workflow") or {})
        workflow.update(
            {
                "stage": "model_saved",
                "completed_steps": 5,
                "total_steps": 5,
                "next_action": (
                    "Run a prediction with the active Model Book."
                    if active_book_id
                    else "Open Model Library and set the saved Model Book as active."
                ),
            }
        )
        project_manifest["workflow"] = workflow
        project_manifest["updated_at"] = created_at
        atomic_write_json(project_root / "project.json", project_manifest)
    except OSError as exc:
        raise ModelBookError(
            "The Model Book files were created, but the project index could not be updated."
        ) from exc

    return load_model_book(project_root, book_id)


def load_model_book(
    project_path: str | Path,
    book_id_or_name: str,
) -> ModelBook:
    """Load and integrity-check one saved Model Book without loading its estimator."""

    project_root, _ = _read_project(project_path)
    lookup = str(book_id_or_name or "").strip()
    if not lookup:
        raise ModelBookError("A Model Book ID or name is required.")
    index = _read_index(project_root / "books" / "index.json")
    entry = next(
        (
            item
            for item in index["books"]
            if isinstance(item, dict)
            and (
                str(item.get("book_id") or "") == lookup
                or str(item.get("name") or "").casefold() == lookup.casefold()
            )
        ),
        None,
    )
    if entry is None:
        raise ModelBookError(f"Model Book '{lookup}' was not found.")
    manifest_value = entry.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise ModelBookError("The Model Book index contains an invalid manifest path.")
    manifest_path = _safe_project_path(project_root, manifest_value)
    payload = _read_json(manifest_path, "Model Book manifest")
    book = _model_book_from_payload(project_root, manifest_path, payload)
    if book.book_id != str(entry.get("book_id") or ""):
        raise ModelBookError(
            "The Model Book identity does not match the project index."
        )
    return book


def list_model_books(project_path: str | Path) -> list[ModelBook]:
    """Load all Model Books in saved order for a future library page."""

    project_root, _ = _read_project(project_path)
    index = _read_index(project_root / "books" / "index.json")
    return [
        load_model_book(project_root, str(entry.get("book_id") or ""))
        for entry in index["books"]
        if isinstance(entry, dict)
    ]


def load_model_library(project_path: str | Path) -> ModelBookLibrary:
    """Load the book index while retaining invalid entries for clear UI errors."""

    project_root, _ = _read_project(project_path)
    index = _read_index(project_root / "books" / "index.json")
    raw_active = index.get("active_book_id")
    active_book_id = str(raw_active) if raw_active is not None else None
    entries: list[ModelBookLibraryEntry] = []
    for position, raw_entry in enumerate(index["books"], start=1):
        if not isinstance(raw_entry, dict):
            entries.append(
                ModelBookLibraryEntry(
                    book_id=f"invalid-entry-{position}",
                    name=f"Invalid Model Book entry {position}",
                    is_active=False,
                    error_message="The Model Book index entry has an invalid structure.",
                )
            )
            continue
        book_id = str(raw_entry.get("book_id") or "").strip()
        name = str(raw_entry.get("name") or book_id or f"Model Book {position}")
        try:
            book = load_model_book(project_root, book_id)
        except ModelBookError as exc:
            entries.append(
                ModelBookLibraryEntry(
                    book_id=book_id or f"invalid-entry-{position}",
                    name=name,
                    is_active=bool(book_id and book_id == active_book_id),
                    error_message=str(exc),
                )
            )
        else:
            entries.append(
                ModelBookLibraryEntry(
                    book_id=book.book_id,
                    name=book.name,
                    is_active=book.book_id == active_book_id,
                    book=book,
                )
            )
    return ModelBookLibrary(
        project_path=project_root,
        active_book_id=active_book_id,
        entries=entries,
    )


def set_active_model_book(
    project_path: str | Path,
    book_id: str,
) -> ModelBook:
    """Select one valid Model Book and persist it in the index and project."""

    project_root, project_manifest = _read_project(project_path)
    book = load_model_book(project_root, book_id)
    index_path = project_root / "books" / "index.json"
    index = _read_index(index_path)
    index["active_book_id"] = book.book_id
    selected_at = utc_now()
    try:
        atomic_write_json(index_path, index)
        library_state = dict(project_manifest.get("model_library") or {})
        library_state.update(
            {
                "schema_version": MODEL_BOOK_INDEX_SCHEMA_VERSION,
                "book_count": len(index["books"]),
                "active_book_id": book.book_id,
                "index": "books/index.json",
            }
        )
        project_manifest["model_library"] = library_state
        workflow = dict(project_manifest.get("workflow") or {})
        if workflow.get("stage") == "model_saved":
            workflow["next_action"] = "Run a prediction with the active Model Book."
            project_manifest["workflow"] = workflow
        project_manifest["updated_at"] = selected_at
        atomic_write_json(project_root / "project.json", project_manifest)
    except OSError as exc:
        raise ModelBookError(
            "The active Model Book selection could not be saved."
        ) from exc
    return book


def _collect_source_metadata(
    project_root: Path,
    run_summary: dict[str, Any],
    run_record: dict[str, Any],
    run_directory: Path,
) -> dict[str, Any]:
    dataset_id = str(run_record.get("dataset_id") or "").strip()
    if not dataset_id:
        raise ModelBookError("The completed run does not identify its dataset.")
    try:
        dataset = get_registered_dataset(project_root, dataset_id)
    except DatasetRegistrationError as exc:
        raise ModelBookError(
            "The registered dataset for this run is missing or failed its integrity check."
        ) from exc

    model_name = str(run_record.get("model_name") or "").strip()
    model_type = MODEL_BOOK_MODEL_TYPES.get(model_name)
    if model_type is None:
        raise ModelBookError(
            f"The completed run uses an unsupported model type: {model_name or 'missing'}."
        )
    model_path = _required_artifact(
        project_root,
        run_record,
        "model",
        "trained model",
    )
    metrics_path = _required_artifact(
        project_root,
        run_record,
        "metrics",
        "metrics.json",
    )
    training_config_path = _required_artifact(
        project_root,
        run_record,
        "training_config",
        "training_config.json",
    )
    metrics = _test_metrics(_read_json(metrics_path, "metrics.json"))
    training_config = _read_json(training_config_path, "training_config.json")
    training_mode = str(training_config.get("training_mode") or "").lower()
    if training_mode not in {"auto", "custom"}:
        raise ModelBookError("The completed run has invalid training-mode metadata.")
    raw_search_level = training_config.get("search_level")
    search_level = str(raw_search_level).lower() if raw_search_level is not None else None
    if (
        model_name in {"linear_regression", "neural_network"}
        and training_mode == "auto"
        and search_level not in {"medium", "high"}
    ):
        raise ModelBookError("The completed Auto run has invalid search-level metadata.")
    if training_mode == "custom" and search_level is not None:
        raise ModelBookError("The completed Custom run cannot contain an Auto search level.")
    if (
        model_name == "xgboost"
        and training_mode == "auto"
        and search_level not in {None, "medium", "high"}
    ):
        raise ModelBookError(
            "The completed XGBoost Auto run has invalid search-level metadata."
        )
    if model_name == "ensemble_ai_engine" and (
        training_mode != "auto" or search_level != "high"
    ):
        raise ModelBookError(
            "The completed Ensemble AI Engine run must use Auto High mode."
        )
    parameters = _model_parameters(
        training_config.get("parameters_used"),
        model_name,
    )

    configuration = run_record.get("configuration")
    if not isinstance(configuration, dict):
        raise ModelBookError("The completed run is missing its training configuration.")
    test_size = _finite_float(configuration.get("test_size"), "test split size")
    random_state = configuration.get("random_state")
    if not isinstance(random_state, int) or isinstance(random_state, bool):
        raise ModelBookError("The completed run has an invalid random state.")

    validation_metrics: dict[str, float] = {}
    ensemble_components: list[dict[str, Any]] = []
    if model_name == "ensemble_ai_engine":
        ensemble_path = _required_artifact(
            project_root,
            run_record,
            "ensemble_results",
            "ensemble_results.json",
        )
        ensemble = _read_json(ensemble_path, "ensemble_results.json")
        validation_metrics["RMSE"] = _finite_float(
            ensemble.get("ensemble_validation_rmse"),
            "ensemble validation RMSE",
        )
        raw_components = ensemble.get("component_results")
        artifacts = run_record.get("artifacts")
        component_artifacts = (
            artifacts.get("component_models")
            if isinstance(artifacts, dict)
            else None
        )
        if not isinstance(raw_components, list) or len(raw_components) < 2:
            raise ModelBookError(
                "The completed Ensemble run has insufficient component metadata."
            )
        if not isinstance(component_artifacts, dict):
            raise ModelBookError(
                "The completed Ensemble run is missing component model artifacts."
            )
        for raw_component in raw_components:
            if not isinstance(raw_component, dict):
                raise ModelBookError(
                    "The completed Ensemble component metadata is invalid."
                )
            component_name = str(raw_component.get("model_name") or "")
            relative_artifact = component_artifacts.get(component_name)
            if not isinstance(relative_artifact, str) or not relative_artifact:
                raise ModelBookError(
                    f"The completed Ensemble is missing the {component_name} artifact."
                )
            artifact_source = _safe_project_path(project_root, relative_artifact)
            if not artifact_source.is_file():
                raise ModelBookError(
                    f"The completed Ensemble component artifact is missing: {component_name}."
                )
            ensemble_components.append(
                {
                    "model_name": component_name,
                    "parameters_used": _model_parameters(
                        raw_component.get("parameters_used"),
                        component_name,
                    ),
                    "weight": _finite_float(
                        raw_component.get("weight"),
                        f"{component_name} weight",
                    ),
                    "validation_rmse": _finite_float(
                        raw_component.get("validation_rmse"),
                        f"{component_name} validation RMSE",
                    ),
                    "source_run_id": str(raw_component.get("source_run_id") or ""),
                    "artifact_source": artifact_source,
                }
            )
    elif training_mode == "auto" and search_level in {"medium", "high"}:
        auto_path = _required_artifact(
            project_root,
            run_record,
            "auto_search_results",
            "auto_search_results.json",
        )
        auto_search = _read_json(auto_path, "auto_search_results.json")
        validation_metrics["RMSE"] = _finite_float(
            auto_search.get("best_validation_rmse"),
            "validation RMSE",
        )

    run_id = str(run_record.get("run_id") or "").strip()
    run_number = run_record.get("run_number")
    if not run_id or not isinstance(run_number, int) or isinstance(run_number, bool):
        raise ModelBookError("The completed run is missing its run identity.")
    if run_id != str(run_summary.get("run_id") or ""):
        raise ModelBookError("The saved run identity does not match the project record.")

    trained_at = str(run_record.get("trained_at") or "").strip()
    if not trained_at:
        raise ModelBookError("The completed run is missing its training timestamp.")

    return {
        "model_path": model_path,
        "model_name": model_name,
        "model_type": model_type,
        "feature_columns": list(dataset.feature_columns),
        "target_columns": list(dataset.target_columns),
        "output_axis": infer_output_axis(dataset.target_columns),
        "sample_id_column": dataset.sample_id_column,
        "parameters_used": parameters,
        "training_mode": training_mode,
        "search_level": search_level,
        "split": {"test_size": test_size, "random_state": random_state},
        "dataset_id": dataset.dataset_id,
        "dataset_fingerprint": dataset.fingerprint_sha256,
        "validation_metrics": validation_metrics,
        "test_metrics": metrics,
        "run_id": run_id,
        "run_number": run_number,
        "trained_at": trained_at,
        "ensemble_components": ensemble_components,
    }


def _load_completed_run(
    project_root: Path,
    project_manifest: dict[str, Any],
    source_run_id: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    run_id = str(source_run_id or "").strip()
    if not run_id:
        raise ModelBookError("Select a completed training run to save as a Model Book.")
    training = project_manifest.get("model_training")
    records = training.get("runs") if isinstance(training, dict) else None
    if not isinstance(records, list):
        raise ModelBookError("No completed training run is available to save.")
    summary = next(
        (
            record
            for record in records
            if isinstance(record, dict) and record.get("run_id") == run_id
        ),
        None,
    )
    if summary is None:
        raise ModelBookError(f"Training run '{run_id}' was not found.")
    if summary.get("status") != TRAINING_COMPLETED:
        raise ModelBookError(
            "Only successfully completed training runs can be saved as a Model Book."
        )
    artifacts = summary.get("artifacts")
    relative_manifest = artifacts.get("run_manifest") if isinstance(artifacts, dict) else None
    if not isinstance(relative_manifest, str) or not relative_manifest:
        raise ModelBookError("The completed run does not reference run.json.")
    manifest_path = _safe_project_path(project_root, relative_manifest)
    run_record = _read_json(manifest_path, "run.json")
    if run_record.get("status") != TRAINING_COMPLETED:
        raise ModelBookError(
            "Only successfully completed training runs can be saved as a Model Book."
        )
    return dict(summary), run_record, manifest_path.parent


def _model_book_from_payload(
    project_root: Path,
    manifest_path: Path,
    payload: dict[str, Any],
) -> ModelBook:
    if payload.get("schema_version") != MODEL_BOOK_SCHEMA_VERSION:
        raise ModelBookError("The Model Book manifest has an unsupported structure.")
    if payload.get("model_book_version") != MODEL_BOOK_VERSION:
        raise ModelBookError("The Model Book uses an unsupported version.")
    try:
        book_id = str(payload["book_id"])
        name = str(payload["name"])
        created_at = str(payload["created_at"])
        model = payload["model"]
        interface = payload["interface"]
        training = payload["training"]
        dataset = payload["dataset"]
        performance = payload["performance"]
        source = payload["source"]
        artifact = model["artifact"]
    except (KeyError, TypeError) as exc:
        raise ModelBookError("The Model Book manifest is incomplete.") from exc
    if not book_id or not name or not created_at:
        raise ModelBookError("The Model Book identity is incomplete.")

    directory = manifest_path.parent.resolve()
    model_path = (directory / str(artifact.get("path") or "")).resolve()
    if not model_path.is_relative_to(directory) or not model_path.is_file():
        raise ModelBookError("The Model Book's trained model artifact is missing.")
    expected_hash = str(artifact.get("sha256") or "")
    if not expected_hash or _sha256(model_path) != expected_hash:
        raise ModelBookError("The Model Book's trained model failed its integrity check.")

    feature_columns = _string_list(interface.get("feature_columns"), "input features")
    target_columns = _string_list(interface.get("target_columns"), "target outputs")
    try:
        output_axis = output_axis_from_dict(
            interface.get("output_axis"),
            target_columns,
        )
    except ValueError as exc:
        raise ModelBookError(str(exc)) from exc
    model_name = str(model.get("name") or "").strip()
    model_type = str(model.get("type") or "").strip()
    if not model_name or not model_type:
        raise ModelBookError("The Model Book model identity is incomplete.")
    expected_model_type = MODEL_BOOK_MODEL_TYPES.get(model_name)
    if expected_model_type is None or model_type != expected_model_type:
        raise ModelBookError("The Model Book model type is unsupported or inconsistent.")
    parameters = _model_parameters(training.get("parameters_used"), model_name)
    ensemble_components: list[dict[str, Any]] | None = None
    component_artifact_paths: dict[str, Path] | None = None
    if model_name == "ensemble_ai_engine":
        raw_ensemble = model.get("ensemble")
        raw_components = (
            raw_ensemble.get("components")
            if isinstance(raw_ensemble, dict)
            else None
        )
        if (
            not isinstance(raw_ensemble, dict)
            or raw_ensemble.get("weighting_method") != "inverse_validation_rmse"
            or not isinstance(raw_components, list)
            or len(raw_components) < 2
        ):
            raise ModelBookError(
                "The Ensemble Model Book has invalid component metadata."
            )
        ensemble_components = []
        component_artifact_paths = {}
        for raw_component in raw_components:
            if not isinstance(raw_component, dict):
                raise ModelBookError(
                    "The Ensemble Model Book component metadata is invalid."
                )
            component_name = str(raw_component.get("model_name") or "")
            if (
                component_name not in parameters["weights"]
                or component_name in component_artifact_paths
            ):
                raise ModelBookError(
                    "The Ensemble Model Book identifies an invalid component."
                )
            component_artifact = raw_component.get("artifact")
            if not isinstance(component_artifact, dict):
                raise ModelBookError(
                    "The Ensemble Model Book component artifact is missing."
                )
            component_path = (
                directory / str(component_artifact.get("path") or "")
            ).resolve()
            if not component_path.is_relative_to(directory) or not component_path.is_file():
                raise ModelBookError(
                    f"The Ensemble component artifact is missing: {component_name}."
                )
            component_hash = str(component_artifact.get("sha256") or "")
            if not component_hash or _sha256(component_path) != component_hash:
                raise ModelBookError(
                    f"The Ensemble component failed its integrity check: {component_name}."
                )
            component_parameters = _model_parameters(
                raw_component.get("parameters_used"),
                component_name,
            )
            component_weight = _finite_float(
                raw_component.get("weight"),
                f"{component_name} weight",
            )
            if (
                component_parameters != parameters["components"][component_name]
                or not math.isclose(
                    component_weight,
                    parameters["weights"][component_name],
                    abs_tol=1e-12,
                )
            ):
                raise ModelBookError(
                    "The Ensemble component metadata does not match its training record."
                )
            ensemble_components.append(
                {
                    "model_name": component_name,
                    "parameters_used": component_parameters,
                    "weight": component_weight,
                    "validation_rmse": _finite_float(
                        raw_component.get("validation_rmse"),
                        f"{component_name} validation RMSE",
                    ),
                    "source_run_id": str(raw_component.get("source_run_id") or ""),
                    "artifact_sha256": component_hash,
                }
            )
            component_artifact_paths[component_name] = component_path
    training_mode = str(training.get("mode") or "")
    if training_mode not in {"auto", "custom"}:
        raise ModelBookError("The Model Book has invalid training-mode metadata.")
    search_value = training.get("search_level")
    search_level = str(search_value) if search_value is not None else None
    if model_name == "linear_regression" and training_mode == "auto":
        if search_level not in {"medium", "high"}:
            raise ModelBookError(
                "The Model Book has invalid Auto search-level metadata."
            )
    elif model_name == "neural_network":
        if training_mode == "auto" and search_level not in {"medium", "high"}:
            raise ModelBookError(
                "The Model Book has invalid Neural Network Auto search-level metadata."
            )
        if training_mode == "custom" and search_level is not None:
            raise ModelBookError(
                "The Model Book Custom Neural Network metadata cannot contain an "
                "Auto search level."
            )
    elif model_name == "xgboost":
        if training_mode == "auto" and search_level not in {
            None,
            "medium",
            "high",
        }:
            raise ModelBookError(
                "The Model Book has invalid XGBoost Auto search-level metadata."
            )
        if training_mode == "custom" and search_level is not None:
            raise ModelBookError(
                "The Model Book Custom XGBoost metadata cannot contain an Auto "
                "search level."
            )
    elif model_name == "ensemble_ai_engine":
        if training_mode != "auto" or search_level != "high":
            raise ModelBookError(
                "The Ensemble Model Book must use Auto High mode."
            )
    test_metrics = _test_metrics(performance.get("test_metrics"))
    validation_metrics = _numeric_metrics(
        performance.get("validation_metrics"),
        allow_empty=True,
    )
    run_number = source.get("run_number")
    if not isinstance(run_number, int) or isinstance(run_number, bool) or run_number < 1:
        raise ModelBookError("The Model Book has invalid source-run metadata.")

    dataset_id = str(dataset.get("dataset_id") or "").strip()
    dataset_fingerprint = str(dataset.get("fingerprint_sha256") or "").strip()
    source_run_id = str(source.get("run_id") or "").strip()
    source_trained_at = str(source.get("trained_at") or "").strip()
    if not dataset_id or not dataset_fingerprint:
        raise ModelBookError("The Model Book dataset identity is incomplete.")
    if not source_run_id or not source_trained_at:
        raise ModelBookError("The Model Book source-run identity is incomplete.")

    return ModelBook(
        project_path=project_root,
        directory=directory,
        manifest_path=manifest_path,
        model_artifact_path=model_path,
        book_id=book_id,
        name=name,
        version=MODEL_BOOK_VERSION,
        created_at=created_at,
        model_name=model_name,
        model_type=model_type,
        feature_columns=feature_columns,
        target_columns=target_columns,
        sample_id_column=(
            str(interface["sample_id_column"])
            if interface.get("sample_id_column") is not None
            else None
        ),
        parameters_used=parameters,
        training_mode=training_mode,
        search_level=search_level,
        dataset_id=dataset_id,
        dataset_fingerprint=dataset_fingerprint,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        source_run_id=source_run_id,
        source_run_number=run_number,
        source_trained_at=source_trained_at,
        model_sha256=expected_hash,
        output_axis=output_axis,
        ensemble_components=ensemble_components,
        component_artifact_paths=component_artifact_paths,
    )


def _read_project(project_path: str | Path) -> tuple[Path, dict[str, Any]]:
    project_root = Path(project_path).expanduser().resolve()
    if project_root.is_file() and project_root.name == "project.json":
        project_root = project_root.parent
    payload = _read_json(project_root / "project.json", "project manifest")
    if not payload.get("project_id"):
        raise ModelBookError("Model Books can only be saved inside a Studio project.")
    return project_root, payload


def _read_index(index_path: Path) -> dict[str, Any]:
    if not index_path.exists():
        return {
            "schema_version": MODEL_BOOK_INDEX_SCHEMA_VERSION,
            "active_book_id": None,
            "books": [],
        }
    payload = _read_json(index_path, "Model Book index")
    if (
        payload.get("schema_version") != MODEL_BOOK_INDEX_SCHEMA_VERSION
        or not isinstance(payload.get("books"), list)
    ):
        raise ModelBookError("The Model Book index has an unsupported structure.")
    return payload


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelBookError(f"The required {label} is missing.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelBookError(f"The saved {label} is malformed or unreadable.") from exc
    if not isinstance(payload, dict):
        raise ModelBookError(f"The saved {label} has an invalid structure.")
    return payload


def _required_artifact(
    project_root: Path,
    run_record: dict[str, Any],
    artifact_name: str,
    label: str,
) -> Path:
    artifacts = run_record.get("artifacts")
    relative = artifacts.get(artifact_name) if isinstance(artifacts, dict) else None
    if not isinstance(relative, str) or not relative:
        raise ModelBookError(f"The completed run does not reference {label}.")
    path = _safe_project_path(project_root, relative)
    if not path.is_file():
        raise ModelBookError(f"The required {label} artifact is missing.")
    return path


def _safe_project_path(project_root: Path, relative: str) -> Path:
    path = (project_root / relative).resolve()
    if not path.is_relative_to(project_root):
        raise ModelBookError("An artifact path points outside the project.")
    return path


def _validated_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ModelBookError("Enter a Model Book name.")
    display_name = name.strip()
    if len(display_name) > 120:
        raise ModelBookError("Model Book names must be 120 characters or fewer.")
    return display_name


def _next_book_number(books_root: Path, entries: list[Any]) -> int:
    numbers: set[int] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        value = str(entry.get("book_id") or "")
        if value.startswith("book-") and value[5:].isdigit():
            numbers.add(int(value[5:]))
    for path in books_root.glob("book-*"):
        if path.is_dir() and path.name[5:].isdigit():
            numbers.add(int(path.name[5:]))
    return max(numbers, default=0) + 1


def _model_parameters(value: Any, model_name: str) -> dict[str, Any]:
    if model_name == "linear_regression":
        return _boolean_parameters(value)
    if model_name == "neural_network":
        return _neural_network_parameters(value)
    if model_name == "ensemble_ai_engine":
        return _ensemble_parameters(value)
    if model_name != "xgboost":
        raise ModelBookError(f"Unsupported saved model parameters: {model_name}.")
    if not isinstance(value, dict):
        raise ModelBookError("The XGBoost parameters are missing.")
    required = (
        "objective",
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "random_state",
        "n_jobs",
        "tree_method",
        "verbosity",
    )
    missing = [name for name in required if name not in value]
    if missing:
        raise ModelBookError(
            f"The saved XGBoost parameter '{missing[0]}' is missing."
        )
    parameters: dict[str, Any] = {}
    for name in required:
        raw = value[name]
        if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
            raise ModelBookError(
                f"The saved XGBoost parameter '{name}' is invalid."
            )
        if isinstance(raw, float) and not math.isfinite(raw):
            raise ModelBookError(
                f"The saved XGBoost parameter '{name}' is not finite."
            )
        parameters[name] = raw
    return parameters


def _ensemble_parameters(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelBookError("The Ensemble parameters are missing.")
    weights = value.get("weights")
    components = value.get("components")
    if not isinstance(weights, dict) or not isinstance(components, dict):
        raise ModelBookError("The Ensemble weights or components are missing.")
    if len(weights) < 2 or set(weights) != set(components):
        raise ModelBookError(
            "The Ensemble requires matching metadata for at least two components."
        )
    validated_weights: dict[str, float] = {}
    validated_components: dict[str, dict[str, Any]] = {}
    for component_name, raw_weight in weights.items():
        if component_name not in {
            "linear_regression",
            "xgboost",
            "neural_network",
        }:
            raise ModelBookError(
                f"The Ensemble component is unsupported: {component_name}."
            )
        weight = _finite_float(raw_weight, f"{component_name} weight")
        if weight < 0.0:
            raise ModelBookError("Ensemble weights cannot be negative.")
        validated_weights[component_name] = weight
        validated_components[component_name] = _model_parameters(
            components[component_name],
            component_name,
        )
    if not math.isclose(sum(validated_weights.values()), 1.0, abs_tol=1e-9):
        raise ModelBookError("The Ensemble weights are not normalized.")
    return {
        "weights": validated_weights,
        "components": validated_components,
    }


def _neural_network_parameters(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelBookError("The Neural Network parameters are missing.")
    required = (
        "hidden_layer_sizes",
        "activation",
        "learning_rate_init",
        "batch_size",
        "max_iter",
        "solver",
        "random_state",
        "shuffle",
        "early_stopping",
        "tol",
        "n_iter_no_change",
    )
    missing = [name for name in required if name not in value]
    if missing:
        raise ModelBookError(
            f"The saved Neural Network parameter '{missing[0]}' is missing."
        )
    layers = value["hidden_layer_sizes"]
    if (
        not isinstance(layers, list)
        or not layers
        or any(
            isinstance(width, bool) or not isinstance(width, int) or width < 1
            for width in layers
        )
    ):
        raise ModelBookError(
            "The saved Neural Network hidden-layer architecture is invalid."
        )
    if (
        not isinstance(value["activation"], str)
        or value["activation"] not in {"relu", "tanh", "logistic", "identity"}
    ):
        raise ModelBookError("The saved Neural Network activation is invalid.")
    parameters = dict(value)
    parameters["hidden_layer_sizes"] = list(layers)
    for name in ("learning_rate_init", "tol"):
        parameters[name] = _finite_float(value[name], name)
    for name in ("batch_size", "max_iter", "random_state", "n_iter_no_change"):
        raw = value[name]
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ModelBookError(
                f"The saved Neural Network parameter '{name}' is invalid."
            )
    for name in ("shuffle", "early_stopping"):
        if not isinstance(value[name], bool):
            raise ModelBookError(
                f"The saved Neural Network parameter '{name}' is invalid."
            )
    if value["solver"] != "adam":
        raise ModelBookError("The saved Neural Network solver is unsupported.")
    return parameters


def _boolean_parameters(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict):
        raise ModelBookError("The Linear Regression parameters are missing.")
    result: dict[str, bool] = {}
    for name in ("fit_intercept", "positive"):
        raw = value.get(name)
        if not isinstance(raw, bool):
            raise ModelBookError(f"The saved parameter '{name}' is not Boolean.")
        result[name] = raw
    return result


def _numeric_metrics(value: Any, *, allow_empty: bool = False) -> dict[str, float]:
    if not isinstance(value, dict) or (not value and not allow_empty):
        raise ModelBookError("The saved performance metrics are missing.")
    metrics: dict[str, float] = {}
    for name, raw in value.items():
        metrics[str(name)] = _finite_float(raw, str(name))
    return metrics


def _test_metrics(value: Any) -> dict[str, float]:
    metrics = _numeric_metrics(value)
    missing = [name for name in ("MAE", "RMSE", "R²") if name not in metrics]
    if missing:
        raise ModelBookError(
            f"The saved test metrics are missing {missing[0]}."
        )
    return metrics


def _string_list(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ModelBookError(f"The Model Book's {label} are missing or invalid.")
    return [item.strip() for item in value]


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelBookError(f"The saved {label} value is invalid.") from exc
    if not math.isfinite(number):
        raise ModelBookError(f"The saved {label} value is not finite.")
    return number


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ModelBookError(f"The model artifact could not be read: {exc}") from exc
    return digest.hexdigest()
