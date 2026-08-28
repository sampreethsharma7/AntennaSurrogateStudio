"""Single-sample inference using the active project-local Model Book."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from studio.ensemble import WeightedEnsembleRegressor
from studio.model_book import (
    ModelBook,
    ModelBookError,
    load_model_book,
    load_model_library,
)
from studio.project_store import atomic_write_json, utc_now


INFERENCE_COMPLETED = "INFERENCE_COMPLETED"
INFERENCE_FAILED = "INFERENCE_FAILED"
INFERENCE_SCHEMA_VERSION = 1


class InferenceError(RuntimeError):
    """Raised internally for a safe, user-facing inference failure."""


@dataclass(slots=True)
class InferenceRequest:
    """One named input sample to evaluate with the active Model Book."""

    inputs: dict[str, object]
    model_book_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, dict):
            raise ValueError("Inference inputs must be provided as a dictionary.")
        invalid_names = [
            name
            for name in self.inputs
            if not isinstance(name, str) or not name.strip()
        ]
        if invalid_names:
            raise ValueError("Input feature names must be non-empty strings.")
        self.inputs = dict(self.inputs)

        if self.model_book_id is not None:
            if (
                not isinstance(self.model_book_id, str)
                or not self.model_book_id.strip()
            ):
                raise ValueError("The Model Book ID must be a non-empty string.")
            self.model_book_id = self.model_book_id.strip()


@dataclass(slots=True)
class InferenceResult:
    """Structured outcome of one local Model Book prediction."""

    success: bool
    status: str
    model_book_id: str | None
    model_book_name: str | None
    model_name: str | None
    feature_order: list[str] = field(default_factory=list)
    target_order: list[str] = field(default_factory=list)
    input_values: dict[str, float] = field(default_factory=dict)
    predictions: dict[str, float] = field(default_factory=dict)
    error_message: str | None = None
    run_id: str | None = None
    created_at: str | None = None
    artifact_directory: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "model_book_id": self.model_book_id,
            "model_book_name": self.model_book_name,
            "model_name": self.model_name,
            "feature_order": list(self.feature_order),
            "target_order": list(self.target_order),
            "input_values": dict(self.input_values),
            "predictions": dict(self.predictions),
            "error_message": self.error_message,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "artifact_directory": (
                str(self.artifact_directory)
                if self.artifact_directory is not None
                else None
            ),
        }


@dataclass(slots=True)
class InferenceHistory:
    """Valid project-local predictions plus isolated record errors."""

    runs: list[InferenceResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ModelBookPredictor:
    """Loaded Model Book evaluator reusable by inference and optimization.

    This class owns Model Book validation, artifact loading, saved feature order,
    and prediction shape checks.  Higher-level workflows can evaluate many
    samples without knowing which estimator family the book contains.
    """

    book: ModelBook
    estimator: Any

    @classmethod
    def load_active(
        cls,
        project_path: str | Path,
        model_book_id: str | None = None,
    ) -> "ModelBookPredictor":
        book = _active_model_book(project_path, model_book_id)
        return cls(book=book, estimator=_load_estimator(book))

    def predict(self, inputs: dict[str, object]) -> dict[str, float]:
        ordered_values = _validated_inputs(self.book, inputs)
        return _predict(self.estimator, self.book, ordered_values)

    def ordered_inputs(self, inputs: dict[str, object]) -> dict[str, float]:
        ordered_values = _validated_inputs(self.book, inputs)
        return {
            name: value
            for name, value in zip(
                self.book.feature_columns,
                ordered_values,
                strict=True,
            )
        }


def submit_inference_request(
    request: InferenceRequest,
    *,
    project_path: str | Path | None = None,
) -> InferenceResult:
    """Validate and evaluate one sample with the active saved Model Book.

    The optional request Model Book ID is a stale-selection guard: when supplied,
    it must identify the project's current active book. Every successful
    prediction is preserved as an immutable project-local inference run.
    """

    if not isinstance(request, InferenceRequest):
        raise TypeError("A validated InferenceRequest is required.")

    requested_book_id = getattr(request, "model_book_id", None)
    try:
        validated_request = InferenceRequest(
            inputs=dict(request.inputs),
            model_book_id=request.model_book_id,
        )
    except (TypeError, ValueError) as exc:
        return _failed_result(requested_book_id, str(exc))

    if project_path is None:
        return _failed_result(
            validated_request.model_book_id,
            "Inference requires an open Antenna Surrogate Studio project.",
        )

    try:
        predictor = ModelBookPredictor.load_active(
            project_path,
            validated_request.model_book_id,
        )
        book = predictor.book
        ordered_inputs = predictor.ordered_inputs(validated_request.inputs)
        predictions = predictor.predict(ordered_inputs)
        result = InferenceResult(
            success=True,
            status=INFERENCE_COMPLETED,
            model_book_id=book.book_id,
            model_book_name=book.name,
            model_name=book.model_name,
            feature_order=list(book.feature_columns),
            target_order=list(book.target_columns),
            input_values=ordered_inputs,
            predictions=predictions,
            error_message=None,
        )
        _save_completed_inference(
            Path(project_path).expanduser().resolve(),
            validated_request,
            book,
            result,
        )
        return result
    except (InferenceError, ModelBookError, OSError) as exc:
        return _failed_result(validated_request.model_book_id, str(exc))
    except ImportError:
        return _failed_result(
            validated_request.model_book_id,
            "Inference dependencies are not installed. Run the Studio installer "
            "to add scikit-learn, XGBoost, NumPy, and joblib.",
        )
    except Exception:
        return _failed_result(
            validated_request.model_book_id,
            "Inference failed because an unexpected local error occurred.",
        )


def load_inference_runs(
    project_path: str | Path,
    *,
    model_book_id: str | None = None,
) -> InferenceHistory:
    """Load every valid saved prediction, preserving chronological order.

    A malformed index prevents trustworthy enumeration and raises one friendly
    error. A damaged individual run is isolated in ``errors`` so remaining
    predictions can still be restored.
    """

    project_root = Path(project_path).expanduser().resolve()
    index_path = project_root / "inference" / "index.json"
    if not index_path.exists():
        return InferenceHistory()
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InferenceError(
            "The inference history index is malformed or unreadable."
        ) from exc
    if not isinstance(index, dict) or not isinstance(index.get("runs"), list):
        raise InferenceError("The inference history index has invalid metadata.")

    history = InferenceHistory()
    seen: set[str] = set()
    for raw_entry in index["runs"]:
        if not isinstance(raw_entry, dict):
            history.errors.append("An inference history entry has invalid metadata.")
            continue
        run_id = str(raw_entry.get("run_id") or "").strip()
        if (
            not run_id.startswith("inference-")
            or not run_id[10:].isdigit()
            or run_id in seen
        ):
            history.errors.append(
                f"Inference history entry '{run_id or 'unknown'}' has an invalid run ID."
            )
            continue
        seen.add(run_id)
        result_path = project_root / "inference" / "runs" / run_id / "result.json"
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            result = _inference_result_from_payload(payload, project_root, run_id)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            history.errors.append(
                f"Inference run '{run_id}' is malformed or unreadable."
            )
            continue
        entry_book_id = str(raw_entry.get("model_book_id") or "").strip()
        if entry_book_id and entry_book_id != result.model_book_id:
            history.errors.append(
                f"Inference run '{run_id}' does not match its history entry."
            )
            continue
        if model_book_id is not None and result.model_book_id != model_book_id:
            continue
        history.runs.append(result)
    return history


def _inference_result_from_payload(
    payload: Any,
    project_root: Path,
    run_id: str,
) -> InferenceResult:
    if not isinstance(payload, dict):
        raise ValueError("Inference result metadata must be an object.")
    if (
        payload.get("run_id") != run_id
        or payload.get("success") is not True
        or payload.get("status") != INFERENCE_COMPLETED
    ):
        raise ValueError("Inference result metadata is inconsistent.")
    model_book_id = str(payload.get("model_book_id") or "").strip()
    feature_order = payload.get("feature_order")
    target_order = payload.get("target_order")
    raw_inputs = payload.get("input_values")
    raw_predictions = payload.get("predictions")
    if (
        not model_book_id
        or not isinstance(feature_order, list)
        or not isinstance(target_order, list)
        or not isinstance(raw_inputs, dict)
        or not isinstance(raw_predictions, dict)
        or not feature_order
        or not target_order
        or any(not isinstance(name, str) or not name for name in feature_order)
        or any(not isinstance(name, str) or not name for name in target_order)
        or list(raw_inputs) != feature_order
        or list(raw_predictions) != target_order
    ):
        raise ValueError("Inference result fields are incomplete.")
    input_values = {name: float(raw_inputs[name]) for name in feature_order}
    predictions = {name: float(raw_predictions[name]) for name in target_order}
    if not all(math.isfinite(value) for value in (*input_values.values(), *predictions.values())):
        raise ValueError("Inference result values must be finite.")
    created_at = str(payload.get("created_at") or "").strip() or None
    return InferenceResult(
        success=True,
        status=INFERENCE_COMPLETED,
        model_book_id=model_book_id,
        model_book_name=str(payload.get("model_book_name") or "").strip() or None,
        model_name=str(payload.get("model_name") or "").strip() or None,
        feature_order=list(feature_order),
        target_order=list(target_order),
        input_values=input_values,
        predictions=predictions,
        error_message=None,
        run_id=run_id,
        created_at=created_at,
        artifact_directory=(project_root / "inference" / "runs" / run_id).resolve(),
    )


def _save_completed_inference(
    project_root: Path,
    request: InferenceRequest,
    book: ModelBook,
    result: InferenceResult,
) -> None:
    runs_root = project_root / "inference" / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_number = 1
    while (runs_root / f"inference-{run_number:04d}").exists():
        run_number += 1
    run_id = f"inference-{run_number:04d}"
    created_at = utc_now()
    destination = runs_root / run_id
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs_root))
    result.run_id = run_id
    result.created_at = created_at
    result.artifact_directory = destination
    try:
        atomic_write_json(
            staging / "request.json",
            {
                "schema_version": INFERENCE_SCHEMA_VERSION,
                "created_at": created_at,
                "model_book_id": book.book_id,
                "requested_model_book_id": request.model_book_id,
                "inputs": dict(result.input_values),
            },
        )
        payload = result.to_dict()
        payload["schema_version"] = INFERENCE_SCHEMA_VERSION
        payload["artifact_directory"] = f"inference/runs/{run_id}"
        payload["model_book"] = {
            "book_id": book.book_id,
            "name": book.name,
            "model_name": book.model_name,
            "dataset_fingerprint": book.dataset_fingerprint,
        }
        payload["output_axis"] = (
            book.output_axis.to_dict() if book.output_axis is not None else None
        )
        atomic_write_json(staging / "result.json", payload)
        axis_values = (
            book.output_axis.values
            if book.output_axis is not None
            and len(book.output_axis.values) == len(result.target_order)
            else tuple(float(index) for index in range(1, len(result.target_order) + 1))
        )
        with (staging / "prediction.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["Output coordinate", "Output variable", "Predicted value"])
            for coordinate, target in zip(axis_values, result.target_order, strict=True):
                writer.writerow([coordinate, target, result.predictions[target]])
        os.replace(staging, destination)
    except OSError as exc:
        raise InferenceError(
            f"The prediction was calculated but could not be saved: {exc}"
        ) from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    index_path = project_root / "inference" / "index.json"
    index: dict[str, Any] = {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "latest_run_id": None,
        "runs": [],
    }
    if index_path.exists():
        try:
            loaded = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InferenceError(
                "The prediction was saved, but its inference history index is invalid."
            ) from exc
        if not isinstance(loaded, dict) or not isinstance(loaded.get("runs"), list):
            raise InferenceError(
                "The prediction was saved, but its inference history index is invalid."
            )
        index = loaded
    index["latest_run_id"] = run_id
    index["runs"] = [
        *index.get("runs", []),
        {
            "run_id": run_id,
            "created_at": created_at,
            "model_book_id": book.book_id,
            "model_book_name": book.name,
            "result": f"runs/{run_id}/result.json",
        },
    ]
    atomic_write_json(index_path, index)

    manifest_path = project_root / "project.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InferenceError(
            "The prediction was saved, but project state could not be updated."
        ) from exc
    manifest["inference"] = {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "run_count": len(index["runs"]),
        "latest_run_id": run_id,
        "index": "inference/index.json",
    }
    manifest["updated_at"] = created_at
    atomic_write_json(manifest_path, manifest)


def _active_model_book(
    project_path: str | Path,
    requested_book_id: str | None,
) -> ModelBook:
    library = load_model_library(project_path)
    active_book_id = library.active_book_id
    if not active_book_id:
        raise InferenceError(
            "No active Model Book is selected. Select a valid Model Book as active first."
        )
    if requested_book_id is not None and requested_book_id != active_book_id:
        raise InferenceError(
            f"Model Book '{requested_book_id}' is not the active Model Book. "
            "Select it as active before running inference."
        )
    return load_model_book(project_path, active_book_id)


def _validated_inputs(
    book: ModelBook,
    raw_inputs: dict[str, object],
) -> list[float]:
    required = list(book.feature_columns)
    missing = [name for name in required if name not in raw_inputs]
    if missing:
        raise InferenceError(
            "Missing required input"
            f"{'s' if len(missing) != 1 else ''}: {', '.join(missing)}."
        )
    unexpected = [name for name in raw_inputs if name not in required]
    if unexpected:
        raise InferenceError(
            "Unexpected input"
            f"{'s' if len(unexpected) != 1 else ''}: {', '.join(unexpected)}."
        )

    ordered: list[float] = []
    for name in required:
        raw_value = raw_inputs[name]
        if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
            raise InferenceError(f"Input '{name}' must be a numeric value.")
        number = float(raw_value)
        if not math.isfinite(number):
            raise InferenceError(f"Input '{name}' must be a finite numeric value.")
        ordered.append(number)
    return ordered


def _load_estimator(book: ModelBook) -> Any:
    if book.model_name not in {
        "linear_regression",
        "xgboost",
        "neural_network",
        "ensemble_ai_engine",
    }:
        raise InferenceError(
            f"Inference does not support Model Book type '{book.model_name}'."
        )
    try:
        with warnings.catch_warnings():
            # Joblib reconstructs some scikit-learn arrays through NumPy's
            # deprecated shape setter. It is external serialization noise, not
            # an invalid Model Book; keep all other warnings visible.
            warnings.filterwarnings(
                "ignore",
                message=r"Setting the shape on a NumPy array has been deprecated.*",
                category=DeprecationWarning,
            )
            estimator = joblib.load(book.model_artifact_path)
    except Exception as exc:
        raise InferenceError(
            "The active Model Book's model artifact could not be loaded."
        ) from exc
    if book.model_name == "linear_regression":
        expected_type: type[Any] = LinearRegression
        expected_label = "Linear Regression"
    elif book.model_name == "xgboost":
        from xgboost import XGBRegressor

        expected_type = XGBRegressor
        expected_label = "XGBoost"
    elif book.model_name == "ensemble_ai_engine":
        expected_type = WeightedEnsembleRegressor
        expected_label = "Ensemble AI Engine"
    else:
        expected_type = Pipeline
        expected_label = "Neural Network"
    if not isinstance(estimator, expected_type):
        raise InferenceError(
            "The active Model Book artifact does not contain the expected "
            f"{expected_label} model."
        )
    if book.model_name == "neural_network":
        neural_step = getattr(estimator, "named_steps", {}).get("neural_network")
        if not isinstance(neural_step, MLPRegressor):
            raise InferenceError(
                "The active Model Book artifact does not contain the expected "
                "Neural Network model."
            )
    if book.model_name == "ensemble_ai_engine":
        if (
            estimator.weights != book.parameters_used.get("weights")
            or estimator.component_parameters
            != book.parameters_used.get("components")
            or set(estimator.component_models) != set(estimator.weights)
        ):
            raise InferenceError(
                "The active Ensemble Model Book artifact does not match its saved "
                "component weights and parameters."
            )
    expected_features = len(book.feature_columns)
    artifact_features = getattr(estimator, "n_features_in_", None)
    if artifact_features != expected_features:
        raise InferenceError(
            "The active Model Book artifact does not match its required input features."
        )
    return estimator


def _predict(
    estimator: Any,
    book: ModelBook,
    ordered_values: list[float],
) -> dict[str, float]:
    sample = np.asarray([ordered_values], dtype=float)
    try:
        raw_predictions = np.asarray(estimator.predict(sample), dtype=float)
    except Exception as exc:
        raise InferenceError(
            "The active Model Book could not produce a prediction for these inputs."
        ) from exc

    if raw_predictions.ndim == 1:
        values = raw_predictions.tolist()
    elif raw_predictions.ndim == 2 and raw_predictions.shape[0] == 1:
        values = raw_predictions[0].tolist()
    else:
        raise InferenceError(
            "The active Model Book returned an invalid single-sample prediction."
        )
    if len(values) != len(book.target_columns):
        raise InferenceError(
            "The active Model Book prediction does not match its saved outputs."
        )

    predictions: dict[str, float] = {}
    for target, raw_value in zip(book.target_columns, values, strict=True):
        value = float(raw_value)
        if not math.isfinite(value):
            raise InferenceError(
                f"The active Model Book returned a non-finite value for '{target}'."
            )
        predictions[target] = value
    return predictions


def _failed_result(
    model_book_id: str | None,
    error_message: str,
) -> InferenceResult:
    return InferenceResult(
        success=False,
        status=INFERENCE_FAILED,
        model_book_id=model_book_id,
        model_book_name=None,
        model_name=None,
        error_message=error_message,
    )
