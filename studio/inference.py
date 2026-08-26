"""Single-sample inference using the active project-local Model Book."""

from __future__ import annotations

import math
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


INFERENCE_COMPLETED = "INFERENCE_COMPLETED"
INFERENCE_FAILED = "INFERENCE_FAILED"


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
    it must identify the project's current active book. No project files are
    created or modified by inference.
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
        return InferenceResult(
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
