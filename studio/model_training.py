"""Validated configuration and deterministic local surrogate training."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from studio.dataset_registry import (
    DatasetRegistrationError,
    RegisteredDataset,
    get_registered_dataset,
)
from studio.dataset_validation import DatasetValidationError, validate_dataset
from studio.ensemble import (
    ENSEMBLE_COMPONENT_ORDER,
    WeightedEnsembleRegressor,
    normalize_inverse_rmse_weights,
)
from studio.parser_engine import TrainingRequest
from studio.project_store import atomic_write_json, utc_now


SUPPORTED_MODEL_NAMES = frozenset(
    {"linear_regression", "xgboost", "neural_network", "ensemble_ai_engine"}
)
SUPPORTED_TRAINING_MODES = frozenset({"auto", "custom"})
SUPPORTED_SEARCH_LEVELS = frozenset({"medium", "high"})
LINEAR_REGRESSION_PARAMETERS = frozenset({"fit_intercept", "positive"})
XGBOOST_CUSTOM_PARAMETER_NAMES = (
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
)
NEURAL_NETWORK_CUSTOM_PARAMETER_NAMES = (
    "hidden_layer_sizes",
    "activation",
    "learning_rate_init",
    "batch_size",
    "max_iter",
)
TRAINING_COMPLETED = "TRAINING_COMPLETED"
TRAINING_FAILED = "TRAINING_FAILED"
MODEL_ARTIFACT_NAME = "linear_regression_model.joblib"
XGBOOST_MODEL_ARTIFACT_NAME = "xgboost_model.joblib"
NEURAL_NETWORK_MODEL_ARTIFACT_NAME = "neural_network_model.joblib"
ENSEMBLE_MODEL_ARTIFACT_NAME = "ensemble_ai_engine_model.joblib"
MODEL_ARTIFACT_NAMES = {
    "linear_regression": MODEL_ARTIFACT_NAME,
    "xgboost": XGBOOST_MODEL_ARTIFACT_NAME,
    "neural_network": NEURAL_NETWORK_MODEL_ARTIFACT_NAME,
    "ensemble_ai_engine": ENSEMBLE_MODEL_ARTIFACT_NAME,
}
METRICS_ARTIFACT_NAME = "metrics.json"
PREDICTIONS_ARTIFACT_NAME = "test_predictions.csv"
TRAINING_CONFIG_ARTIFACT_NAME = "training_config.json"
AUTO_SEARCH_ARTIFACT_NAME = "auto_search_results.json"
ENSEMBLE_RESULTS_ARTIFACT_NAME = "ensemble_results.json"
RUN_MANIFEST_NAME = "run.json"
RUNS_SCHEMA_VERSION = 1
AUTO_SEARCH_CONFIGURATIONS = {
    "medium": (
        {"fit_intercept": True, "positive": False},
        {"fit_intercept": False, "positive": False},
    ),
    "high": (
        {"fit_intercept": True, "positive": False},
        {"fit_intercept": False, "positive": False},
        {"fit_intercept": True, "positive": True},
        {"fit_intercept": False, "positive": True},
    ),
}
XGBOOST_AUTO_SEARCH_CONFIGURATIONS = {
    "medium": (
        {
            "n_estimators": 64,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },
        {
            "n_estimators": 48,
            "max_depth": 3,
            "learning_rate": 0.12,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
        {
            "n_estimators": 96,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 1.0,
            "colsample_bytree": 0.9,
        },
    ),
    "high": (
        {
            "n_estimators": 64,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },
        {
            "n_estimators": 48,
            "max_depth": 3,
            "learning_rate": 0.12,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
        {
            "n_estimators": 96,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 1.0,
            "colsample_bytree": 0.9,
        },
        {
            "n_estimators": 128,
            "max_depth": 3,
            "learning_rate": 0.04,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
        },
        {
            "n_estimators": 80,
            "max_depth": 6,
            "learning_rate": 0.075,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 40,
            "max_depth": 2,
            "learning_rate": 0.15,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    ),
}
NEURAL_NETWORK_AUTO_SEARCH_CONFIGURATIONS = {
    "medium": (
        {
            "hidden_layer_sizes": [32],
            "activation": "relu",
            "learning_rate_init": 0.001,
            "batch_size": 8,
            "max_iter": 160,
        },
        {
            "hidden_layer_sizes": [64, 32],
            "activation": "relu",
            "learning_rate_init": 0.001,
            "batch_size": 8,
            "max_iter": 180,
        },
        {
            "hidden_layer_sizes": [64],
            "activation": "tanh",
            "learning_rate_init": 0.001,
            "batch_size": 8,
            "max_iter": 180,
        },
    ),
    "high": (
        {
            "hidden_layer_sizes": [32],
            "activation": "relu",
            "learning_rate_init": 0.001,
            "batch_size": 8,
            "max_iter": 160,
        },
        {
            "hidden_layer_sizes": [64, 32],
            "activation": "relu",
            "learning_rate_init": 0.001,
            "batch_size": 8,
            "max_iter": 180,
        },
        {
            "hidden_layer_sizes": [64],
            "activation": "tanh",
            "learning_rate_init": 0.001,
            "batch_size": 8,
            "max_iter": 180,
        },
        {
            "hidden_layer_sizes": [96, 48],
            "activation": "relu",
            "learning_rate_init": 0.0005,
            "batch_size": 8,
            "max_iter": 220,
        },
        {
            "hidden_layer_sizes": [48, 24],
            "activation": "tanh",
            "learning_rate_init": 0.0005,
            "batch_size": 4,
            "max_iter": 220,
        },
        {
            "hidden_layer_sizes": [32, 32, 16],
            "activation": "relu",
            "learning_rate_init": 0.002,
            "batch_size": 4,
            "max_iter": 180,
        },
    ),
}
AUTO_SEARCH_REQUESTED_FOLDS = {"medium": 3, "high": 5}
XGBOOST_BASELINE_PARAMETERS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 64,
    "learning_rate": 0.1,
    "max_depth": 4,
    "min_child_weight": 1.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": 1,
    "tree_method": "hist",
    "verbosity": 0,
}
XGBOOST_CUSTOM_DEFAULTS: dict[str, int | float] = {
    name: XGBOOST_BASELINE_PARAMETERS[name]
    for name in XGBOOST_CUSTOM_PARAMETER_NAMES
}
NEURAL_NETWORK_BASELINE_PARAMETERS: dict[str, Any] = {
    "hidden_layer_sizes": [64, 32],
    "activation": "relu",
    "learning_rate_init": 0.001,
    "batch_size": 8,
    "max_iter": 180,
    "solver": "adam",
    "random_state": 42,
    "shuffle": False,
    "early_stopping": False,
    "tol": 0.0001,
    "n_iter_no_change": 20,
}
NEURAL_NETWORK_CUSTOM_DEFAULTS: dict[str, Any] = {
    name: (
        list(NEURAL_NETWORK_BASELINE_PARAMETERS[name])
        if name == "hidden_layer_sizes"
        else NEURAL_NETWORK_BASELINE_PARAMETERS[name]
    )
    for name in NEURAL_NETWORK_CUSTOM_PARAMETER_NAMES
}


@dataclass(slots=True)
class ModelTrainingRequest:
    """Validated configuration passed to the future training backend."""

    model_name: str
    training_mode: str
    search_level: str | None = None
    custom_hyperparameters: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("A model name is required.")
        self.model_name = self.model_name.strip()
        if self.model_name not in SUPPORTED_MODEL_NAMES:
            raise ValueError(f"Unsupported model: {self.model_name}")

        if (
            not isinstance(self.training_mode, str)
            or not self.training_mode.strip()
        ):
            raise ValueError("A training mode is required.")
        self.training_mode = self.training_mode.strip()
        if self.training_mode not in SUPPORTED_TRAINING_MODES:
            raise ValueError(
                f"Unsupported training mode: {self.training_mode}"
            )

        if (
            self.custom_hyperparameters is not None
            and not isinstance(self.custom_hyperparameters, dict)
        ):
            raise ValueError("Custom hyperparameters must be a dictionary.")

        if self.model_name == "ensemble_ai_engine":
            self._validate_ensemble_configuration()
        elif self.model_name == "xgboost":
            self._validate_xgboost_configuration()
        elif self.model_name == "neural_network":
            self._validate_neural_network_configuration()
        elif self.training_mode == "auto":
            self._validate_auto_mode()
        else:
            self._validate_custom_mode()

    def _validate_ensemble_configuration(self) -> None:
        if self.training_mode != "auto":
            raise ValueError("Ensemble AI Engine supports Auto High mode only.")
        if isinstance(self.search_level, str):
            self.search_level = self.search_level.strip()
        if self.search_level != "high":
            raise ValueError("Ensemble AI Engine requires Auto High search.")
        if self.custom_hyperparameters:
            raise ValueError(
                "Custom hyperparameters cannot be provided for Ensemble AI Engine."
            )
        if self.custom_hyperparameters is not None:
            self.custom_hyperparameters = dict(self.custom_hyperparameters)

    def _validate_neural_network_configuration(self) -> None:
        if self.training_mode == "auto":
            self._validate_auto_mode()
            return
        if self.search_level is not None:
            raise ValueError("Search level cannot be used in Custom mode.")
        self._validate_neural_network_custom_mode()

    def _validate_neural_network_custom_mode(self) -> None:
        if not self.custom_hyperparameters:
            raise ValueError("Custom Neural Network mode requires hyperparameters.")
        parameters = dict(self.custom_hyperparameters)
        unknown = [
            name
            for name in parameters
            if name not in NEURAL_NETWORK_CUSTOM_PARAMETER_NAMES
        ]
        if unknown:
            raise ValueError(f"Unsupported Neural Network parameter: {unknown[0]}")
        missing = [
            name
            for name in NEURAL_NETWORK_CUSTOM_PARAMETER_NAMES
            if name not in parameters
        ]
        if missing:
            raise ValueError(
                "Custom Neural Network mode requires all parameters: "
                + ", ".join(NEURAL_NETWORK_CUSTOM_PARAMETER_NAMES)
                + "."
            )

        layers = parameters["hidden_layer_sizes"]
        if (
            not isinstance(layers, (list, tuple))
            or not layers
            or len(layers) > 8
            or any(
                isinstance(width, bool)
                or not isinstance(width, int)
                or not 1 <= width <= 4096
                for width in layers
            )
        ):
            raise ValueError(
                "Neural Network hidden_layer_sizes must contain 1 to 8 integer "
                "layer widths between 1 and 4096."
            )
        parameters["hidden_layer_sizes"] = [int(width) for width in layers]

        activation = parameters["activation"]
        if (
            not isinstance(activation, str)
            or activation not in {"relu", "tanh", "logistic", "identity"}
        ):
            raise ValueError(
                "Neural Network activation must be relu, tanh, logistic, or identity."
            )

        learning_rate = parameters["learning_rate_init"]
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(float(learning_rate))
            or not 0.0 < float(learning_rate) <= 1.0
        ):
            raise ValueError(
                "Neural Network learning_rate_init must be greater than 0 and "
                "no greater than 1."
            )
        parameters["learning_rate_init"] = float(learning_rate)

        for name, minimum, maximum in (
            ("batch_size", 1, 65536),
            ("max_iter", 1, 100000),
        ):
            value = parameters[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise ValueError(
                    f"Neural Network parameter '{name}' must be an integer "
                    f"between {minimum} and {maximum}."
                )
        self.custom_hyperparameters = parameters

    def _validate_xgboost_configuration(self) -> None:
        if self.training_mode == "auto":
            self._validate_auto_mode()
            return
        if self.search_level is not None:
            raise ValueError("Search level cannot be used in Custom mode.")
        self._validate_xgboost_custom_mode()

    def _validate_xgboost_custom_mode(self) -> None:
        if not self.custom_hyperparameters:
            raise ValueError("Custom XGBoost mode requires hyperparameters.")
        parameters = dict(self.custom_hyperparameters)
        unknown = [
            name
            for name in parameters
            if name not in XGBOOST_CUSTOM_PARAMETER_NAMES
        ]
        if unknown:
            raise ValueError(f"Unsupported XGBoost parameter: {unknown[0]}")
        missing = [
            name
            for name in XGBOOST_CUSTOM_PARAMETER_NAMES
            if name not in parameters
        ]
        if missing:
            raise ValueError(
                "Custom XGBoost mode requires all parameters: "
                + ", ".join(XGBOOST_CUSTOM_PARAMETER_NAMES)
                + "."
            )

        integer_ranges = {
            "n_estimators": (1, 5000),
            "max_depth": (1, 64),
        }
        for name, (minimum, maximum) in integer_ranges.items():
            value = parameters[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"XGBoost parameter '{name}' must be an integer between "
                    f"{minimum} and {maximum}."
                )
            if not minimum <= value <= maximum:
                raise ValueError(
                    f"XGBoost parameter '{name}' must be between {minimum} "
                    f"and {maximum}."
                )

        for name in ("learning_rate", "subsample", "colsample_bytree"):
            value = parameters[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"XGBoost parameter '{name}' must be a finite number."
                )
            number = float(value)
            if not 0.0 < number <= 1.0:
                raise ValueError(
                    f"XGBoost parameter '{name}' must be greater than 0 and "
                    "no greater than 1."
                )
            parameters[name] = number
        self.custom_hyperparameters = parameters

    def _validate_auto_mode(self) -> None:
        if (
            not isinstance(self.search_level, str)
            or not self.search_level.strip()
        ):
            raise ValueError("Auto mode requires a search level.")
        self.search_level = self.search_level.strip()
        if self.search_level not in SUPPORTED_SEARCH_LEVELS:
            raise ValueError(
                f"Unsupported Auto search level: {self.search_level}"
            )
        if self.custom_hyperparameters:
            raise ValueError(
                "Custom hyperparameters cannot be provided in Auto mode."
            )
        if self.custom_hyperparameters is not None:
            self.custom_hyperparameters = dict(self.custom_hyperparameters)

    def _validate_custom_mode(self) -> None:
        if self.search_level is not None:
            raise ValueError("Search level cannot be used in Custom mode.")
        if not self.custom_hyperparameters:
            raise ValueError("Custom mode requires hyperparameters.")

        parameters = dict(self.custom_hyperparameters)
        unknown_parameters = [
            name
            for name in parameters
            if name not in LINEAR_REGRESSION_PARAMETERS
        ]
        if unknown_parameters:
            raise ValueError(
                "Unsupported Linear Regression parameter: "
                f"{unknown_parameters[0]}"
            )

        missing_parameters = [
            name
            for name in ("fit_intercept", "positive")
            if name not in parameters
        ]
        if missing_parameters:
            raise ValueError(
                "Custom mode requires both fit_intercept and positive."
            )

        for name in ("fit_intercept", "positive"):
            if not isinstance(parameters[name], bool):
                raise ValueError(
                    f"Linear Regression parameter '{name}' must be Boolean."
                )
        self.custom_hyperparameters = parameters


def resolve_linear_regression_parameters(
    request: ModelTrainingRequest,
) -> dict[str, bool]:
    """Return the exact validated estimator parameters for one request."""

    if request.training_mode == "auto":
        return {"fit_intercept": True, "positive": False}
    parameters = request.custom_hyperparameters
    assert parameters is not None
    return {
        "fit_intercept": cast(bool, parameters["fit_intercept"]),
        "positive": cast(bool, parameters["positive"]),
    }


def resolve_xgboost_parameters(
    request: ModelTrainingRequest,
) -> dict[str, Any]:
    """Return the exact deterministic XGBoost estimator configuration."""

    parameters = dict(XGBOOST_BASELINE_PARAMETERS)
    if request.training_mode == "custom":
        custom = request.custom_hyperparameters
        assert custom is not None
        for name in XGBOOST_CUSTOM_PARAMETER_NAMES:
            parameters[name] = custom[name]
    return parameters


def resolve_neural_network_parameters(
    request: ModelTrainingRequest,
) -> dict[str, Any]:
    """Return the exact reproducible MLP configuration for one request."""

    parameters = dict(NEURAL_NETWORK_BASELINE_PARAMETERS)
    parameters["hidden_layer_sizes"] = list(
        NEURAL_NETWORK_BASELINE_PARAMETERS["hidden_layer_sizes"]
    )
    if request.training_mode == "custom":
        custom = request.custom_hyperparameters
        assert custom is not None
        for name in NEURAL_NETWORK_CUSTOM_PARAMETER_NAMES:
            value = custom[name]
            parameters[name] = list(value) if name == "hidden_layer_sizes" else value
    return parameters


@dataclass(slots=True)
class ModelTrainingResult:
    """Structured outcome returned by a supported training backend."""

    success: bool
    status: str
    model_name: str
    training_rows: int
    test_rows: int
    metrics: dict[str, float]
    predictions: list[dict[str, str | float]]
    error_message: str | None
    training_mode: str | None = None
    parameters_used: dict[str, Any] = field(default_factory=dict)
    search_level: str | None = None
    configurations_evaluated: int = 0
    cross_validation_folds: int | None = None
    search_results: list[dict[str, Any]] = field(default_factory=list)
    best_parameters: dict[str, Any] = field(default_factory=dict)
    best_validation_rmse: float | None = None
    test_metrics: dict[str, float] = field(default_factory=dict)
    model_artifact_path: Path | None = None
    metrics_artifact_path: Path | None = None
    predictions_artifact_path: Path | None = None
    training_config_artifact_path: Path | None = None
    auto_search_results_artifact_path: Path | None = None
    ensemble_results_artifact_path: Path | None = None
    component_results: list[dict[str, Any]] = field(default_factory=list)
    component_failures: list[dict[str, str]] = field(default_factory=list)
    ensemble_weights: dict[str, float] = field(default_factory=dict)
    ensemble_validation_rmse: float | None = None
    best_individual_model: str | None = None
    best_individual_validation_rmse: float | None = None
    ensemble_improved_on_best: bool | None = None
    dataset_id: str | None = None
    run_number: int | None = None
    run_id: str | None = None
    run_directory: Path | None = None


@dataclass(slots=True)
class _SavedRunArtifacts:
    run_number: int
    run_id: str
    run_directory: Path
    model_path: Path
    metrics_path: Path
    predictions_path: Path
    training_config_path: Path
    auto_search_results_path: Path | None
    ensemble_results_path: Path | None


@dataclass(slots=True)
class _AutoSearchOutcome:
    search_level: str
    configurations_evaluated: int
    cross_validation_folds: int
    configurations_tested: list[dict[str, Any]]
    search_results: list[dict[str, Any]]
    best_parameters: dict[str, Any]
    best_validation_rmse: float


class ModelTrainingError(RuntimeError):
    """Raised internally for a user-facing training failure."""


def submit_model_training_request(
    request: ModelTrainingRequest,
    *,
    project_path: str | Path | None = None,
) -> ModelTrainingResult:
    """Validate and execute the supported basic training configuration.

    The immutable active Stage 0 registered dataset is the only accepted data
    source. Expected failures are returned without exposing a traceback.
    """

    if not isinstance(request, ModelTrainingRequest):
        raise TypeError("A validated ModelTrainingRequest is required.")

    model_name = str(getattr(request, "model_name", "") or "")
    try:
        validated_request = ModelTrainingRequest(
            model_name=request.model_name,
            training_mode=request.training_mode,
            search_level=request.search_level,
            custom_hyperparameters=(
                dict(request.custom_hyperparameters)
                if request.custom_hyperparameters is not None
                else None
            ),
        )
    except (TypeError, ValueError) as exc:
        return _failed_result(model_name, str(exc))

    unsupported_message = _unsupported_execution_message(validated_request)
    if unsupported_message:
        return _failed_result(validated_request.model_name, unsupported_message)
    parameters_used: dict[str, Any] | None = None
    if validated_request.training_mode == "custom":
        if validated_request.model_name == "xgboost":
            parameters_used = resolve_xgboost_parameters(validated_request)
        elif validated_request.model_name == "neural_network":
            parameters_used = resolve_neural_network_parameters(validated_request)
        else:
            parameters_used = resolve_linear_regression_parameters(validated_request)
    if project_path is None:
        return _failed_result(
            validated_request.model_name,
            "Training requires an open Antenna Surrogate Studio project.",
        )

    try:
        project_root, dataset = _active_registered_dataset(project_path)
        validation = validate_dataset(
            TrainingRequest(
                input_csv_path=dataset.input_csv_path,
                output_csv_path=dataset.output_csv_path,
                feature_columns=list(dataset.feature_columns),
                target_columns=list(dataset.target_columns),
                sample_id_column=dataset.sample_id_column,
            )
        )
        if validation.sample_count < 5:
            if validated_request.model_name == "linear_regression":
                raise ModelTrainingError(
                    "At least 5 usable rows are required to train Linear "
                    f"Regression; the registered dataset has {validation.sample_count}. "
                    "Fewer rows cannot provide a safe 2-fold validation after the "
                    "test split."
                )
            model_label = (
                "XGBoost"
                if validated_request.model_name == "xgboost"
                else (
                    "a Neural Network"
                    if validated_request.model_name == "neural_network"
                    else "Ensemble AI Engine"
                )
            )
            raise ModelTrainingError(
                f"At least 5 usable rows are required to train {model_label}; "
                f"the registered dataset has {validation.sample_count}."
            )

        features, targets, sample_ids = _load_training_rows(dataset)
        if validated_request.model_name == "linear_regression":
            return _fit_linear_regression(
                project_root,
                dataset,
                features,
                targets,
                sample_ids,
                training_mode=validated_request.training_mode,
                search_level=validated_request.search_level,
                parameters_used=parameters_used,
            )
        if validated_request.model_name == "xgboost":
            return _fit_xgboost(
                project_root,
                dataset,
                features,
                targets,
                sample_ids,
                training_mode=validated_request.training_mode,
                search_level=validated_request.search_level,
                parameters_used=parameters_used,
            )
        if validated_request.model_name == "ensemble_ai_engine":
            return _fit_ensemble_ai_engine(
                project_root,
                dataset,
                features,
                targets,
                sample_ids,
            )
        return _fit_neural_network(
            project_root,
            dataset,
            features,
            targets,
            sample_ids,
            training_mode=validated_request.training_mode,
            search_level=validated_request.search_level,
            parameters_used=parameters_used,
        )
    except (
        DatasetRegistrationError,
        DatasetValidationError,
        ModelTrainingError,
        OSError,
    ) as exc:
        return _failed_result(validated_request.model_name, str(exc))
    except ImportError:
        return _failed_result(
            validated_request.model_name,
            "Training dependencies are not installed. Run the Studio installer "
            "to add scikit-learn, XGBoost, NumPy, and joblib.",
        )
    except Exception:
        return _failed_result(
            validated_request.model_name,
            "Training failed because an unexpected local error occurred.",
        )


def _unsupported_execution_message(
    request: ModelTrainingRequest,
) -> str | None:
    if request.model_name not in SUPPORTED_MODEL_NAMES:
        return f"Model '{request.model_name}' is not supported for training."
    if request.training_mode not in {"auto", "custom"}:
        return f"Training mode '{request.training_mode}' is not supported."
    return None


def _active_registered_dataset(
    project_path: str | Path,
) -> tuple[Path, RegisteredDataset]:
    project_root = Path(project_path).expanduser()
    if project_root.is_file() and project_root.name == "project.json":
        project_root = project_root.parent
    project_root = project_root.resolve()
    manifest_path = project_root / "project.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelTrainingError(
            "Training requires an open Antenna Surrogate Studio project."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelTrainingError(
            "The project manifest could not be read for training."
        ) from exc

    registry = manifest.get("dataset_registry")
    dataset_id = (
        registry.get("active_dataset_id") if isinstance(registry, dict) else None
    )
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ModelTrainingError(
            "Training cannot start because no Stage 0 validated dataset is "
            "registered for this project."
        )
    return project_root, get_registered_dataset(project_root, dataset_id)


def _load_training_rows(
    dataset: RegisteredDataset,
) -> tuple[list[list[float]], list[list[float]], list[str]]:
    input_headers, input_rows = _read_csv_rows(dataset.input_csv_path)
    output_headers, output_rows = _read_csv_rows(dataset.output_csv_path)

    missing_features = [
        name for name in dataset.feature_columns if name not in input_headers
    ]
    if missing_features:
        raise ModelTrainingError(
            "Selected input feature columns were not found: "
            f"{', '.join(missing_features)}."
        )
    missing_targets = [
        name for name in dataset.target_columns if name not in output_headers
    ]
    if missing_targets:
        raise ModelTrainingError(
            "Selected output target columns were not found: "
            f"{', '.join(missing_targets)}."
        )

    feature_indexes = [input_headers.index(name) for name in dataset.feature_columns]
    target_indexes = [output_headers.index(name) for name in dataset.target_columns]
    features = [
        [float(row[index].strip()) for index in feature_indexes]
        for row in input_rows
    ]
    targets = [
        [float(row[index].strip()) for index in target_indexes]
        for row in output_rows
    ]

    if dataset.sample_id_column is not None:
        sample_index = input_headers.index(dataset.sample_id_column)
        sample_ids = [row[sample_index].strip() for row in input_rows]
    else:
        sample_ids = [
            f"Sample_{row_number:06d}"
            for row_number in range(1, len(input_rows) + 1)
        ]
    return features, targets, sample_ids


def _read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        headers = [value.strip() for value in next(reader)]
        rows = [
            row
            for row in reader
            if row and any(value.strip() for value in row)
        ]
    return headers, rows


def _resolve_cross_validation_folds(
    training_row_count: int,
    search_level: str,
) -> int:
    """Return the deterministic usable fold count for an Auto search."""

    requested_folds = AUTO_SEARCH_REQUESTED_FOLDS[search_level]
    actual_folds = min(requested_folds, training_row_count)
    if actual_folds < 2:
        raise ModelTrainingError(
            "Auto search requires at least 2 training rows for 2-fold "
            "cross-validation."
        )
    return actual_folds


def _cross_validate_linear_regression_configuration(
    training_features: Any,
    training_targets: Any,
    parameters: dict[str, bool],
    fold_count: int,
) -> list[float]:
    """Evaluate one configuration using only the fixed training partition."""

    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold

    fold_rmse: list[float] = []
    splitter = KFold(n_splits=fold_count, shuffle=False)
    for fit_indices, validation_indices in splitter.split(training_features):
        candidate = LinearRegression(**parameters)
        candidate.fit(
            training_features[fit_indices],
            training_targets[fit_indices],
        )
        validation_predictions = candidate.predict(
            training_features[validation_indices]
        )
        rmse = float(
            math.sqrt(
                mean_squared_error(
                    training_targets[validation_indices],
                    np.asarray(validation_predictions, dtype=float),
                )
            )
        )
        if not math.isfinite(rmse):
            raise ValueError("Cross-validation produced a non-finite RMSE.")
        fold_rmse.append(rmse)
    return fold_rmse


def _run_linear_regression_auto_search(
    training_features: Any,
    training_targets: Any,
    search_level: str,
) -> _AutoSearchOutcome:
    """Run deterministic training-only CV and select one configuration."""

    configurations = AUTO_SEARCH_CONFIGURATIONS[search_level]
    fold_count = _resolve_cross_validation_folds(
        len(training_features),
        search_level,
    )
    search_results: list[dict[str, Any]] = []
    successful: list[tuple[float, dict[str, bool]]] = []

    for raw_parameters in configurations:
        parameters = dict(raw_parameters)
        try:
            fold_rmse = _cross_validate_linear_regression_configuration(
                training_features,
                training_targets,
                parameters,
                fold_count,
            )
            mean_validation_rmse = float(sum(fold_rmse) / len(fold_rmse))
            if not math.isfinite(mean_validation_rmse):
                raise ValueError(
                    "Cross-validation produced a non-finite mean RMSE."
                )
            search_results.append(
                {
                    "parameters": parameters,
                    "fold_rmse": fold_rmse,
                    "mean_validation_rmse": mean_validation_rmse,
                    "success": True,
                    "error_message": None,
                }
            )
            successful.append((mean_validation_rmse, parameters))
        except Exception as exc:
            search_results.append(
                {
                    "parameters": parameters,
                    "fold_rmse": [],
                    "mean_validation_rmse": None,
                    "success": False,
                    "error_message": (
                        "This configuration could not be evaluated: "
                        f"{str(exc) or type(exc).__name__}"
                    ),
                }
            )

    if not successful:
        raise ModelTrainingError(
            "Auto search failed because all Linear Regression configurations "
            "could not be evaluated."
        )

    best_validation_rmse, best_parameters = min(
        successful,
        key=lambda candidate: (
            candidate[0],
            candidate[1]["positive"],
            not candidate[1]["fit_intercept"],
        ),
    )
    return _AutoSearchOutcome(
        search_level=search_level,
        configurations_evaluated=len(configurations),
        cross_validation_folds=fold_count,
        configurations_tested=[dict(item) for item in configurations],
        search_results=search_results,
        best_parameters=dict(best_parameters),
        best_validation_rmse=best_validation_rmse,
    )


def _xgboost_estimator_parameters(
    tunable_parameters: dict[str, int | float],
) -> dict[str, Any]:
    """Merge one bounded search candidate with deterministic runtime settings."""

    parameters = dict(XGBOOST_BASELINE_PARAMETERS)
    parameters.update(tunable_parameters)
    return parameters


def _cross_validate_xgboost_configuration(
    training_features: Any,
    training_targets: Any,
    parameters: dict[str, Any],
    fold_count: int,
) -> list[float]:
    """Evaluate one XGBoost candidate using only the training partition."""

    import numpy as np
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold
    from xgboost import XGBRegressor

    fold_rmse: list[float] = []
    splitter = KFold(n_splits=fold_count, shuffle=False)
    for fit_indices, validation_indices in splitter.split(training_features):
        candidate = XGBRegressor(**parameters)
        candidate.fit(
            training_features[fit_indices],
            training_targets[fit_indices],
        )
        validation_predictions = np.asarray(
            candidate.predict(training_features[validation_indices]),
            dtype=float,
        )
        rmse = float(
            math.sqrt(
                mean_squared_error(
                    training_targets[validation_indices],
                    validation_predictions,
                )
            )
        )
        if not math.isfinite(rmse):
            raise ValueError("Cross-validation produced a non-finite RMSE.")
        fold_rmse.append(rmse)
    return fold_rmse


def _run_xgboost_auto_search(
    training_features: Any,
    training_targets: Any,
    search_level: str,
) -> _AutoSearchOutcome:
    """Run bounded deterministic XGBoost CV on the training partition only."""

    tunable_configurations = XGBOOST_AUTO_SEARCH_CONFIGURATIONS[search_level]
    configurations = [
        _xgboost_estimator_parameters(dict(candidate))
        for candidate in tunable_configurations
    ]
    fold_count = _resolve_cross_validation_folds(
        len(training_features),
        search_level,
    )
    search_results: list[dict[str, Any]] = []
    successful: list[tuple[float, int, dict[str, Any]]] = []

    for index, parameters in enumerate(configurations):
        try:
            fold_rmse = _cross_validate_xgboost_configuration(
                training_features,
                training_targets,
                parameters,
                fold_count,
            )
            mean_validation_rmse = float(sum(fold_rmse) / len(fold_rmse))
            if not math.isfinite(mean_validation_rmse):
                raise ValueError(
                    "Cross-validation produced a non-finite mean RMSE."
                )
            search_results.append(
                {
                    "parameters": dict(parameters),
                    "fold_rmse": fold_rmse,
                    "mean_validation_rmse": mean_validation_rmse,
                    "success": True,
                    "error_message": None,
                }
            )
            successful.append((mean_validation_rmse, index, dict(parameters)))
        except Exception as exc:
            search_results.append(
                {
                    "parameters": dict(parameters),
                    "fold_rmse": [],
                    "mean_validation_rmse": None,
                    "success": False,
                    "error_message": (
                        "This configuration could not be evaluated: "
                        f"{str(exc) or type(exc).__name__}"
                    ),
                }
            )

    if not successful:
        raise ModelTrainingError(
            "Auto search failed because all XGBoost configurations could not "
            "be evaluated."
        )

    best_validation_rmse, _candidate_index, best_parameters = min(
        successful,
        key=lambda candidate: (candidate[0], candidate[1]),
    )
    return _AutoSearchOutcome(
        search_level=search_level,
        configurations_evaluated=len(configurations),
        cross_validation_folds=fold_count,
        configurations_tested=[dict(item) for item in configurations],
        search_results=search_results,
        best_parameters=dict(best_parameters),
        best_validation_rmse=best_validation_rmse,
    )


def _neural_network_estimator_parameters(
    tunable_parameters: dict[str, Any],
) -> dict[str, Any]:
    """Merge one bounded candidate with fixed reproducibility settings."""

    parameters = dict(NEURAL_NETWORK_BASELINE_PARAMETERS)
    parameters["hidden_layer_sizes"] = list(
        NEURAL_NETWORK_BASELINE_PARAMETERS["hidden_layer_sizes"]
    )
    parameters.update(tunable_parameters)
    parameters["hidden_layer_sizes"] = list(parameters["hidden_layer_sizes"])
    return parameters


def _make_neural_network_estimator(parameters: dict[str, Any]) -> Any:
    """Create the saved standardized Neural Network regression pipeline."""

    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        steps=(
            ("scale_inputs", StandardScaler()),
            ("neural_network", MLPRegressor(**parameters)),
        )
    )


def _fit_neural_network_estimator(estimator: Any, features: Any, targets: Any) -> None:
    """Fit for the configured epoch budget without emitting convergence noise."""

    import warnings
    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings(
            "ignore",
            message=r"Got `batch_size` less than 1 or larger than sample size.*",
            category=UserWarning,
        )
        estimator.fit(features, targets)


def _cross_validate_neural_network_configuration(
    training_features: Any,
    training_targets: Any,
    parameters: dict[str, Any],
    fold_count: int,
) -> list[float]:
    """Evaluate one reproducible Neural Network on training-only folds."""

    import numpy as np
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold

    fold_rmse: list[float] = []
    splitter = KFold(n_splits=fold_count, shuffle=False)
    for fit_indices, validation_indices in splitter.split(training_features):
        candidate = _make_neural_network_estimator(parameters)
        _fit_neural_network_estimator(
            candidate,
            training_features[fit_indices],
            training_targets[fit_indices],
        )
        validation_predictions = np.asarray(
            candidate.predict(training_features[validation_indices]),
            dtype=float,
        )
        rmse = float(
            math.sqrt(
                mean_squared_error(
                    training_targets[validation_indices],
                    validation_predictions,
                )
            )
        )
        if not math.isfinite(rmse):
            raise ValueError("Cross-validation produced a non-finite RMSE.")
        fold_rmse.append(rmse)
    return fold_rmse


def _run_neural_network_auto_search(
    training_features: Any,
    training_targets: Any,
    search_level: str,
) -> _AutoSearchOutcome:
    """Select a bounded Neural Network configuration by validation RMSE."""

    tunable_configurations = NEURAL_NETWORK_AUTO_SEARCH_CONFIGURATIONS[search_level]
    configurations = [
        _neural_network_estimator_parameters(dict(candidate))
        for candidate in tunable_configurations
    ]
    fold_count = _resolve_cross_validation_folds(
        len(training_features),
        search_level,
    )
    search_results: list[dict[str, Any]] = []
    successful: list[tuple[float, int, dict[str, Any]]] = []
    for index, parameters in enumerate(configurations):
        try:
            fold_rmse = _cross_validate_neural_network_configuration(
                training_features,
                training_targets,
                parameters,
                fold_count,
            )
            mean_validation_rmse = float(sum(fold_rmse) / len(fold_rmse))
            if not math.isfinite(mean_validation_rmse):
                raise ValueError(
                    "Cross-validation produced a non-finite mean RMSE."
                )
            search_results.append(
                {
                    "parameters": dict(parameters),
                    "fold_rmse": fold_rmse,
                    "mean_validation_rmse": mean_validation_rmse,
                    "success": True,
                    "error_message": None,
                }
            )
            successful.append((mean_validation_rmse, index, dict(parameters)))
        except Exception as exc:
            search_results.append(
                {
                    "parameters": dict(parameters),
                    "fold_rmse": [],
                    "mean_validation_rmse": None,
                    "success": False,
                    "error_message": (
                        "This configuration could not be evaluated: "
                        f"{str(exc) or type(exc).__name__}"
                    ),
                }
            )
    if not successful:
        raise ModelTrainingError(
            "Auto search failed because all Neural Network configurations "
            "could not be evaluated."
        )
    best_validation_rmse, _candidate_index, best_parameters = min(
        successful,
        key=lambda candidate: (candidate[0], candidate[1]),
    )
    return _AutoSearchOutcome(
        search_level=search_level,
        configurations_evaluated=len(configurations),
        cross_validation_folds=fold_count,
        configurations_tested=[dict(item) for item in configurations],
        search_results=search_results,
        best_parameters=dict(best_parameters),
        best_validation_rmse=best_validation_rmse,
    )


def _make_component_estimator(model_name: str, parameters: dict[str, Any]) -> Any:
    """Create one already-selected component configuration for ensemble CV."""

    if model_name == "linear_regression":
        from sklearn.linear_model import LinearRegression

        return LinearRegression(**parameters)
    if model_name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(**parameters)
    if model_name == "neural_network":
        return _make_neural_network_estimator(parameters)
    raise ModelTrainingError(f"Unsupported ensemble component: {model_name}.")


def _fit_component_estimator(
    model_name: str,
    estimator: Any,
    features: Any,
    targets: Any,
) -> None:
    if model_name == "neural_network":
        _fit_neural_network_estimator(estimator, features, targets)
    else:
        estimator.fit(features, targets)


def _component_fold_predictions(
    model_name: str,
    training_features: Any,
    training_targets: Any,
    parameters: dict[str, Any],
    fold_count: int,
) -> list[Any]:
    """Return validation predictions for the shared deterministic folds."""

    import numpy as np
    from sklearn.model_selection import KFold

    predictions: list[Any] = []
    splitter = KFold(n_splits=fold_count, shuffle=False)
    for fit_indices, validation_indices in splitter.split(training_features):
        estimator = _make_component_estimator(model_name, parameters)
        _fit_component_estimator(
            model_name,
            estimator,
            training_features[fit_indices],
            training_targets[fit_indices],
        )
        values = np.asarray(
            estimator.predict(training_features[validation_indices]),
            dtype=float,
        )
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError("Cross-validation produced invalid predictions.")
        predictions.append(values)
    return predictions


def _ensemble_validation_rmse(
    training_features: Any,
    training_targets: Any,
    component_fold_predictions: dict[str, list[Any]],
    weights: dict[str, float],
    fold_count: int,
) -> tuple[float, list[float]]:
    """Calculate ensemble mean fold RMSE on the training partition only."""

    import numpy as np
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold

    fold_rmse: list[float] = []
    splitter = KFold(n_splits=fold_count, shuffle=False)
    for fold_index, (_fit_indices, validation_indices) in enumerate(
        splitter.split(training_features)
    ):
        combined = sum(
            component_fold_predictions[model_name][fold_index] * weights[model_name]
            for model_name in component_fold_predictions
        )
        actual = np.asarray(training_targets[validation_indices], dtype=float)
        if actual.ndim == 1:
            actual = actual.reshape(-1, 1)
        score = float(math.sqrt(mean_squared_error(actual, combined)))
        if not math.isfinite(score):
            raise ValueError("Ensemble validation produced a non-finite RMSE.")
        fold_rmse.append(score)
    return float(sum(fold_rmse) / len(fold_rmse)), fold_rmse


def _fit_ensemble_ai_engine(
    project_root: Path,
    dataset: RegisteredDataset,
    features: list[list[float]],
    targets: list[list[float]],
    sample_ids: list[str],
) -> ModelTrainingResult:
    """Train every individual family in Auto High and combine valid models."""

    import joblib
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    component_results: list[dict[str, Any]] = []
    component_failures: list[dict[str, str]] = []
    successful_results: dict[str, ModelTrainingResult] = {}
    for model_name in ENSEMBLE_COMPONENT_ORDER:
        result = submit_model_training_request(
            ModelTrainingRequest(
                model_name=model_name,
                training_mode="auto",
                search_level="high",
                custom_hyperparameters=None,
            ),
            project_path=project_root,
        )
        if (
            not result.success
            or result.best_validation_rmse is None
            or result.model_artifact_path is None
        ):
            component_failures.append(
                {
                    "model_name": model_name,
                    "error_message": result.error_message
                    or "The component did not produce valid Auto High evidence.",
                }
            )
            continue
        successful_results[model_name] = result

    if len(successful_results) < 2:
        failed_names = ", ".join(
            failure["model_name"] for failure in component_failures
        ) or "unknown components"
        raise ModelTrainingError(
            "Ensemble AI Engine requires at least two valid component models. "
            f"Failed components: {failed_names}."
        )

    component_models: dict[str, Any] = {}
    component_artifacts: dict[str, Path] = {}
    for model_name in ENSEMBLE_COMPONENT_ORDER:
        result = successful_results.get(model_name)
        if result is None or result.model_artifact_path is None:
            continue
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Setting the shape on a NumPy array has been deprecated.*",
                    category=DeprecationWarning,
                )
                component_models[model_name] = joblib.load(
                    result.model_artifact_path
                )
            component_artifacts[model_name] = result.model_artifact_path
        except Exception:
            component_failures.append(
                {
                    "model_name": model_name,
                    "error_message": "The trained component artifact could not be loaded.",
                }
            )
            successful_results.pop(model_name, None)

    if len(successful_results) < 2:
        raise ModelTrainingError(
            "Ensemble AI Engine could not load at least two valid component models."
        )

    component_parameters = {
        name: dict(result.parameters_used)
        for name, result in successful_results.items()
    }

    feature_values = np.asarray(features, dtype=float)
    target_values = np.asarray(targets, dtype=float)
    fit_targets: Any = (
        target_values[:, 0] if target_values.shape[1] == 1 else target_values
    )
    (
        training_features,
        test_features,
        training_targets,
        test_targets,
        _training_sample_ids,
        test_sample_ids,
    ) = train_test_split(
        feature_values,
        fit_targets,
        sample_ids,
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )
    fold_count = _resolve_cross_validation_folds(len(training_features), "high")
    component_fold_predictions: dict[str, list[Any]] = {}
    for model_name in tuple(successful_results):
        try:
            component_fold_predictions[model_name] = _component_fold_predictions(
                model_name,
                training_features,
                training_targets,
                component_parameters[model_name],
                fold_count,
            )
        except Exception:
            component_failures.append(
                {
                    "model_name": model_name,
                    "error_message": (
                        "The selected component could not reproduce validation "
                        "predictions for the ensemble."
                    ),
                }
            )
            successful_results.pop(model_name, None)
            component_parameters.pop(model_name, None)
            component_models.pop(model_name, None)
            component_artifacts.pop(model_name, None)
    if len(successful_results) < 2:
        raise ModelTrainingError(
            "Ensemble AI Engine requires at least two components with valid "
            "cross-validation predictions."
        )
    validation_scores = {
        name: float(result.best_validation_rmse)
        for name, result in successful_results.items()
        if result.best_validation_rmse is not None
    }
    weights = normalize_inverse_rmse_weights(validation_scores)
    ensemble_validation_rmse, ensemble_fold_rmse = _ensemble_validation_rmse(
        training_features,
        training_targets,
        component_fold_predictions,
        weights,
        fold_count,
    )

    model = WeightedEnsembleRegressor(
        component_models=component_models,
        weights=weights,
        component_parameters=component_parameters,
    )
    predicted_targets = np.asarray(model.predict(test_features), dtype=float)
    actual_targets = np.asarray(test_targets, dtype=float)
    if predicted_targets.ndim == 1:
        predicted_targets = predicted_targets.reshape(-1, 1)
        actual_targets = actual_targets.reshape(-1, 1)
    mae = float(mean_absolute_error(actual_targets, predicted_targets))
    rmse = float(math.sqrt(mean_squared_error(actual_targets, predicted_targets)))
    r_squared = (
        0.0
        if len(test_sample_ids) < 2
        else float(r2_score(actual_targets, predicted_targets, force_finite=True))
    )
    metrics = {"MAE": mae, "RMSE": rmse, "R\N{SUPERSCRIPT TWO}": r_squared}
    predictions = _prediction_records(
        test_sample_ids,
        dataset.target_columns,
        actual_targets,
        predicted_targets,
    )

    best_individual_model = min(
        validation_scores,
        key=lambda name: (
            validation_scores[name],
            ENSEMBLE_COMPONENT_ORDER.index(name),
        ),
    )
    best_individual_rmse = validation_scores[best_individual_model]
    improved = ensemble_validation_rmse < best_individual_rmse - 1e-12
    for model_name in ENSEMBLE_COMPONENT_ORDER:
        result = successful_results.get(model_name)
        if result is None:
            continue
        component_results.append(
            {
                "model_name": model_name,
                "status": TRAINING_COMPLETED,
                "source_run_id": result.run_id,
                "source_run_number": result.run_number,
                "validation_rmse": validation_scores[model_name],
                "weight": weights[model_name],
                "parameters_used": dict(result.parameters_used),
            }
        )
    parameters_used: dict[str, Any] = {
        "weights": dict(weights),
        "components": {
            name: dict(parameters)
            for name, parameters in component_parameters.items()
        },
    }
    ensemble_metadata = {
        "search_level": "high",
        "weighting_method": "inverse_validation_rmse",
        "cross_validation_folds": fold_count,
        "component_results": component_results,
        "component_failures": component_failures,
        "weights": dict(weights),
        "ensemble_fold_rmse": ensemble_fold_rmse,
        "ensemble_validation_rmse": ensemble_validation_rmse,
        "best_individual_model": best_individual_model,
        "best_individual_validation_rmse": best_individual_rmse,
        "ensemble_improved_on_best": improved,
        "test_metrics": dict(metrics),
    }
    saved_run = _save_training_artifacts(
        project_root,
        dataset,
        model,
        metrics,
        predictions,
        model_name="ensemble_ai_engine",
        training_mode="auto",
        search_level="high",
        parameters_used=parameters_used,
        auto_search=None,
        training_rows=len(training_features),
        test_rows=len(test_features),
        ensemble_metadata=ensemble_metadata,
        component_model_artifacts=component_artifacts,
    )
    return ModelTrainingResult(
        success=True,
        status=TRAINING_COMPLETED,
        model_name="ensemble_ai_engine",
        training_rows=len(training_features),
        test_rows=len(test_features),
        metrics=metrics,
        predictions=predictions,
        error_message=None,
        training_mode="auto",
        parameters_used=parameters_used,
        search_level="high",
        configurations_evaluated=len(component_results) + len(component_failures),
        cross_validation_folds=fold_count,
        search_results=[dict(item) for item in component_results],
        best_parameters=parameters_used,
        best_validation_rmse=ensemble_validation_rmse,
        test_metrics=dict(metrics),
        model_artifact_path=saved_run.model_path,
        metrics_artifact_path=saved_run.metrics_path,
        predictions_artifact_path=saved_run.predictions_path,
        training_config_artifact_path=saved_run.training_config_path,
        ensemble_results_artifact_path=saved_run.ensemble_results_path,
        component_results=component_results,
        component_failures=component_failures,
        ensemble_weights=dict(weights),
        ensemble_validation_rmse=ensemble_validation_rmse,
        best_individual_model=best_individual_model,
        best_individual_validation_rmse=best_individual_rmse,
        ensemble_improved_on_best=improved,
        dataset_id=dataset.dataset_id,
        run_number=saved_run.run_number,
        run_id=saved_run.run_id,
        run_directory=saved_run.run_directory,
    )


def _fit_linear_regression(
    project_root: Path,
    dataset: RegisteredDataset,
    features: list[list[float]],
    targets: list[list[float]],
    sample_ids: list[str],
    *,
    training_mode: str,
    search_level: str | None,
    parameters_used: dict[str, bool] | None,
) -> ModelTrainingResult:
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    feature_values = np.asarray(features, dtype=float)
    target_values = np.asarray(targets, dtype=float)
    fit_targets: Any = (
        target_values[:, 0] if target_values.shape[1] == 1 else target_values
    )
    (
        training_features,
        test_features,
        training_targets,
        test_targets,
        _training_sample_ids,
        test_sample_ids,
    ) = train_test_split(
        feature_values,
        fit_targets,
        sample_ids,
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )

    auto_search: _AutoSearchOutcome | None = None
    if training_mode == "auto":
        if search_level is None:
            raise ModelTrainingError("Auto mode requires a search level.")
        auto_search = _run_linear_regression_auto_search(
            training_features,
            training_targets,
            search_level,
        )
        resolved_parameters = dict(auto_search.best_parameters)
    else:
        if parameters_used is None:
            raise ModelTrainingError(
                "Custom mode requires validated Linear Regression parameters."
            )
        resolved_parameters = dict(parameters_used)

    model = LinearRegression(**resolved_parameters)
    model.fit(training_features, training_targets)
    predicted_targets = np.asarray(model.predict(test_features), dtype=float)
    actual_targets = np.asarray(test_targets, dtype=float)
    if predicted_targets.ndim == 1:
        predicted_targets = predicted_targets.reshape(-1, 1)
        actual_targets = actual_targets.reshape(-1, 1)

    mae = float(mean_absolute_error(actual_targets, predicted_targets))
    rmse = float(math.sqrt(mean_squared_error(actual_targets, predicted_targets)))
    if len(test_sample_ids) < 2:
        r_squared = 0.0
    else:
        r_squared = float(
            r2_score(actual_targets, predicted_targets, force_finite=True)
        )
    metrics = {"MAE": mae, "RMSE": rmse, "R²": r_squared}
    predictions = _prediction_records(
        test_sample_ids,
        dataset.target_columns,
        actual_targets,
        predicted_targets,
    )
    saved_run = _save_training_artifacts(
        project_root,
        dataset,
        model,
        metrics,
        predictions,
        model_name="linear_regression",
        training_mode=training_mode,
        search_level=search_level,
        parameters_used=resolved_parameters,
        auto_search=auto_search,
        training_rows=len(training_features),
        test_rows=len(test_features),
    )
    return ModelTrainingResult(
        success=True,
        status=TRAINING_COMPLETED,
        model_name="linear_regression",
        training_rows=len(training_features),
        test_rows=len(test_features),
        metrics=metrics,
        predictions=predictions,
        error_message=None,
        training_mode=training_mode,
        parameters_used=resolved_parameters,
        search_level=search_level,
        configurations_evaluated=(
            auto_search.configurations_evaluated if auto_search else 0
        ),
        cross_validation_folds=(
            auto_search.cross_validation_folds if auto_search else None
        ),
        search_results=(
            [dict(result) for result in auto_search.search_results]
            if auto_search
            else []
        ),
        best_parameters=(
            dict(auto_search.best_parameters)
            if auto_search
            else resolved_parameters
        ),
        best_validation_rmse=(
            auto_search.best_validation_rmse if auto_search else None
        ),
        test_metrics=dict(metrics),
        model_artifact_path=saved_run.model_path,
        metrics_artifact_path=saved_run.metrics_path,
        predictions_artifact_path=saved_run.predictions_path,
        training_config_artifact_path=saved_run.training_config_path,
        auto_search_results_artifact_path=(
            saved_run.auto_search_results_path
        ),
        dataset_id=dataset.dataset_id,
        run_number=saved_run.run_number,
        run_id=saved_run.run_id,
        run_directory=saved_run.run_directory,
    )


def _fit_xgboost(
    project_root: Path,
    dataset: RegisteredDataset,
    features: list[list[float]],
    targets: list[list[float]],
    sample_ids: list[str],
    *,
    training_mode: str,
    search_level: str | None,
    parameters_used: dict[str, Any] | None,
) -> ModelTrainingResult:
    """Fit one validated deterministic XGBoost configuration on the shared split."""

    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor

    feature_values = np.asarray(features, dtype=float)
    target_values = np.asarray(targets, dtype=float)
    fit_targets: Any = (
        target_values[:, 0] if target_values.shape[1] == 1 else target_values
    )
    (
        training_features,
        test_features,
        training_targets,
        test_targets,
        _training_sample_ids,
        test_sample_ids,
    ) = train_test_split(
        feature_values,
        fit_targets,
        sample_ids,
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )

    auto_search: _AutoSearchOutcome | None = None
    if training_mode == "auto":
        if search_level is None:
            raise ModelTrainingError("Auto mode requires a search level.")
        auto_search = _run_xgboost_auto_search(
            training_features,
            training_targets,
            search_level,
        )
        resolved_parameters = dict(auto_search.best_parameters)
    else:
        if parameters_used is None:
            raise ModelTrainingError(
                "Custom mode requires validated XGBoost parameters."
            )
        resolved_parameters = dict(parameters_used)
    model = XGBRegressor(**resolved_parameters)
    model.fit(training_features, training_targets)
    predicted_targets = np.asarray(model.predict(test_features), dtype=float)
    actual_targets = np.asarray(test_targets, dtype=float)
    if predicted_targets.ndim == 1:
        predicted_targets = predicted_targets.reshape(-1, 1)
        actual_targets = actual_targets.reshape(-1, 1)

    mae = float(mean_absolute_error(actual_targets, predicted_targets))
    rmse = float(math.sqrt(mean_squared_error(actual_targets, predicted_targets)))
    if len(test_sample_ids) < 2:
        r_squared = 0.0
    else:
        r_squared = float(
            r2_score(actual_targets, predicted_targets, force_finite=True)
        )
    metrics = {"MAE": mae, "RMSE": rmse, "R²": r_squared}
    predictions = _prediction_records(
        test_sample_ids,
        dataset.target_columns,
        actual_targets,
        predicted_targets,
    )
    saved_run = _save_training_artifacts(
        project_root,
        dataset,
        model,
        metrics,
        predictions,
        model_name="xgboost",
        training_mode=training_mode,
        search_level=search_level,
        parameters_used=resolved_parameters,
        auto_search=auto_search,
        training_rows=len(training_features),
        test_rows=len(test_features),
    )
    return ModelTrainingResult(
        success=True,
        status=TRAINING_COMPLETED,
        model_name="xgboost",
        training_rows=len(training_features),
        test_rows=len(test_features),
        metrics=metrics,
        predictions=predictions,
        error_message=None,
        training_mode=training_mode,
        parameters_used=resolved_parameters,
        search_level=search_level,
        configurations_evaluated=(
            auto_search.configurations_evaluated if auto_search else 0
        ),
        cross_validation_folds=(
            auto_search.cross_validation_folds if auto_search else None
        ),
        search_results=(
            [dict(result) for result in auto_search.search_results]
            if auto_search
            else []
        ),
        best_parameters=(
            dict(auto_search.best_parameters)
            if auto_search
            else dict(resolved_parameters)
        ),
        best_validation_rmse=(
            auto_search.best_validation_rmse if auto_search else None
        ),
        test_metrics=dict(metrics),
        model_artifact_path=saved_run.model_path,
        metrics_artifact_path=saved_run.metrics_path,
        predictions_artifact_path=saved_run.predictions_path,
        training_config_artifact_path=saved_run.training_config_path,
        auto_search_results_artifact_path=saved_run.auto_search_results_path,
        dataset_id=dataset.dataset_id,
        run_number=saved_run.run_number,
        run_id=saved_run.run_id,
        run_directory=saved_run.run_directory,
    )


def _fit_neural_network(
    project_root: Path,
    dataset: RegisteredDataset,
    features: list[list[float]],
    targets: list[list[float]],
    sample_ids: list[str],
    *,
    training_mode: str,
    search_level: str | None,
    parameters_used: dict[str, Any] | None,
) -> ModelTrainingResult:
    """Fit one reproducible standardized Neural Network on the shared split."""

    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    feature_values = np.asarray(features, dtype=float)
    target_values = np.asarray(targets, dtype=float)
    fit_targets: Any = (
        target_values[:, 0] if target_values.shape[1] == 1 else target_values
    )
    (
        training_features,
        test_features,
        training_targets,
        test_targets,
        _training_sample_ids,
        test_sample_ids,
    ) = train_test_split(
        feature_values,
        fit_targets,
        sample_ids,
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )

    auto_search: _AutoSearchOutcome | None = None
    if training_mode == "auto":
        if search_level is None:
            raise ModelTrainingError("Auto mode requires a search level.")
        auto_search = _run_neural_network_auto_search(
            training_features,
            training_targets,
            search_level,
        )
        resolved_parameters = dict(auto_search.best_parameters)
    else:
        if parameters_used is None:
            raise ModelTrainingError(
                "Custom mode requires validated Neural Network parameters."
            )
        resolved_parameters = dict(parameters_used)
    resolved_parameters["hidden_layer_sizes"] = list(
        resolved_parameters["hidden_layer_sizes"]
    )

    model = _make_neural_network_estimator(resolved_parameters)
    _fit_neural_network_estimator(model, training_features, training_targets)
    predicted_targets = np.asarray(model.predict(test_features), dtype=float)
    actual_targets = np.asarray(test_targets, dtype=float)
    if predicted_targets.ndim == 1:
        predicted_targets = predicted_targets.reshape(-1, 1)
        actual_targets = actual_targets.reshape(-1, 1)

    mae = float(mean_absolute_error(actual_targets, predicted_targets))
    rmse = float(math.sqrt(mean_squared_error(actual_targets, predicted_targets)))
    if len(test_sample_ids) < 2:
        r_squared = 0.0
    else:
        r_squared = float(
            r2_score(actual_targets, predicted_targets, force_finite=True)
        )
    metrics = {"MAE": mae, "RMSE": rmse, "R\N{SUPERSCRIPT TWO}": r_squared}
    predictions = _prediction_records(
        test_sample_ids,
        dataset.target_columns,
        actual_targets,
        predicted_targets,
    )
    saved_run = _save_training_artifacts(
        project_root,
        dataset,
        model,
        metrics,
        predictions,
        model_name="neural_network",
        training_mode=training_mode,
        search_level=search_level,
        parameters_used=resolved_parameters,
        auto_search=auto_search,
        training_rows=len(training_features),
        test_rows=len(test_features),
    )
    return ModelTrainingResult(
        success=True,
        status=TRAINING_COMPLETED,
        model_name="neural_network",
        training_rows=len(training_features),
        test_rows=len(test_features),
        metrics=metrics,
        predictions=predictions,
        error_message=None,
        training_mode=training_mode,
        parameters_used=resolved_parameters,
        search_level=search_level,
        configurations_evaluated=(
            auto_search.configurations_evaluated if auto_search else 0
        ),
        cross_validation_folds=(
            auto_search.cross_validation_folds if auto_search else None
        ),
        search_results=(
            [dict(result) for result in auto_search.search_results]
            if auto_search
            else []
        ),
        best_parameters=(
            dict(auto_search.best_parameters)
            if auto_search
            else dict(resolved_parameters)
        ),
        best_validation_rmse=(
            auto_search.best_validation_rmse if auto_search else None
        ),
        test_metrics=dict(metrics),
        model_artifact_path=saved_run.model_path,
        metrics_artifact_path=saved_run.metrics_path,
        predictions_artifact_path=saved_run.predictions_path,
        training_config_artifact_path=saved_run.training_config_path,
        auto_search_results_artifact_path=saved_run.auto_search_results_path,
        dataset_id=dataset.dataset_id,
        run_number=saved_run.run_number,
        run_id=saved_run.run_id,
        run_directory=saved_run.run_directory,
    )


def _prediction_records(
    sample_ids: list[str],
    target_columns: list[str],
    actual_targets: Any,
    predicted_targets: Any,
) -> list[dict[str, str | float]]:
    records: list[dict[str, str | float]] = []
    for row_index, sample_id in enumerate(sample_ids):
        for target_index, target_name in enumerate(target_columns):
            actual = float(actual_targets[row_index, target_index])
            predicted = float(predicted_targets[row_index, target_index])
            records.append(
                {
                    "sample_id": str(sample_id),
                    "target_name": target_name,
                    "actual_value": actual,
                    "predicted_value": predicted,
                    "residual": actual - predicted,
                }
            )
    return records


def _save_training_artifacts(
    project_root: Path,
    dataset: RegisteredDataset,
    model: Any,
    metrics: dict[str, float],
    predictions: list[dict[str, str | float]],
    *,
    model_name: str,
    training_mode: str,
    search_level: str | None,
    parameters_used: dict[str, Any],
    auto_search: _AutoSearchOutcome | None,
    training_rows: int,
    test_rows: int,
    ensemble_metadata: dict[str, Any] | None = None,
    component_model_artifacts: dict[str, Path] | None = None,
) -> _SavedRunArtifacts:
    import joblib

    models_root = project_root / "models"
    runs_root = models_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    manifest_path = project_root / "project.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing_runs = _existing_run_records(project_root, manifest, runs_root)
    existing_numbers = {
        int(record["run_number"])
        for record in existing_runs
        if isinstance(record.get("run_number"), int)
    }
    existing_numbers.update(_run_folder_numbers(runs_root))
    run_number = max(existing_numbers, default=0) + 1
    run_id = f"run-{run_number:04d}"
    final_run_directory = runs_root / run_id
    try:
        model_artifact_name = MODEL_ARTIFACT_NAMES[model_name]
    except KeyError as exc:
        raise ModelTrainingError(
            f"No artifact convention is registered for model '{model_name}'."
        ) from exc
    final_model = final_run_directory / model_artifact_name
    final_metrics = final_run_directory / METRICS_ARTIFACT_NAME
    final_predictions = final_run_directory / PREDICTIONS_ARTIFACT_NAME
    final_training_config = (
        final_run_directory / TRAINING_CONFIG_ARTIFACT_NAME
    )
    final_auto_search_results = (
        final_run_directory / AUTO_SEARCH_ARTIFACT_NAME
        if auto_search is not None
        else None
    )
    final_ensemble_results = (
        final_run_directory / ENSEMBLE_RESULTS_ARTIFACT_NAME
        if ensemble_metadata is not None
        else None
    )
    final_component_models = {
        name: final_run_directory / "components" / MODEL_ARTIFACT_NAMES[name]
        for name in (component_model_artifacts or {})
    }
    final_run_manifest = final_run_directory / RUN_MANIFEST_NAME
    trained_at = utc_now()
    configuration = {
        "training_mode": training_mode,
        "search_level": search_level,
        "parameters_used": dict(parameters_used),
        "test_size": 0.20,
        "random_state": 42,
    }
    if model_name == "linear_regression":
        configuration.update(
            {
                "fit_intercept": parameters_used["fit_intercept"],
                "positive": parameters_used["positive"],
            }
        )
    training_config = {
        "model_name": model_name,
        "training_mode": training_mode,
        "search_level": search_level,
        "parameters_used": dict(parameters_used),
    }
    if ensemble_metadata is not None:
        training_config["ensemble"] = {
            "weighting_method": ensemble_metadata["weighting_method"],
            "weights": dict(ensemble_metadata["weights"]),
            "components": [
                dict(component) for component in ensemble_metadata["component_results"]
            ],
        }
    auto_search_payload = (
        {
            "model_name": model_name,
            "search_level": auto_search.search_level,
            "configurations_evaluated": auto_search.configurations_evaluated,
            "configurations_tested": [
                dict(configuration)
                for configuration in auto_search.configurations_tested
            ],
            "cross_validation_folds": auto_search.cross_validation_folds,
            "search_results": auto_search.search_results,
            "best_parameters": dict(auto_search.best_parameters),
            "best_validation_rmse": auto_search.best_validation_rmse,
            "test_metrics": dict(metrics),
        }
        if auto_search is not None
        else None
    )
    artifact_paths = {
        "model": final_model.relative_to(project_root).as_posix(),
        "metrics": final_metrics.relative_to(project_root).as_posix(),
        "predictions": final_predictions.relative_to(project_root).as_posix(),
        "training_config": final_training_config.relative_to(
            project_root
        ).as_posix(),
        "run_manifest": final_run_manifest.relative_to(project_root).as_posix(),
    }
    if final_auto_search_results is not None:
        artifact_paths["auto_search_results"] = (
            final_auto_search_results.relative_to(project_root).as_posix()
        )
    if final_ensemble_results is not None:
        artifact_paths["ensemble_results"] = (
            final_ensemble_results.relative_to(project_root).as_posix()
        )
        artifact_paths["component_models"] = {
            name: path.relative_to(project_root).as_posix()
            for name, path in final_component_models.items()
        }
    run_record = {
        "schema_version": RUNS_SCHEMA_VERSION,
        "run_number": run_number,
        "run_id": run_id,
        "display_name": f"Run {run_number}",
        "status": TRAINING_COMPLETED,
        "model_name": model_name,
        "dataset_id": dataset.dataset_id,
        "trained_at": trained_at,
        "configuration": configuration,
        "parameters_used": dict(parameters_used),
        "training_rows": training_rows,
        "test_rows": test_rows,
        "metrics": dict(metrics),
        "artifacts": artifact_paths,
    }
    if auto_search_payload is not None:
        run_record["auto_search"] = auto_search_payload
    if ensemble_metadata is not None:
        run_record["ensemble"] = dict(ensemble_metadata)

    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs_root))
    try:
        staged_model = staging / model_artifact_name
        staged_metrics = staging / METRICS_ARTIFACT_NAME
        staged_predictions = staging / PREDICTIONS_ARTIFACT_NAME
        staged_training_config = staging / TRAINING_CONFIG_ARTIFACT_NAME
        staged_auto_search_results = staging / AUTO_SEARCH_ARTIFACT_NAME
        staged_ensemble_results = staging / ENSEMBLE_RESULTS_ARTIFACT_NAME
        joblib.dump(model, staged_model)
        atomic_write_json(staged_metrics, metrics)
        with staged_predictions.open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "sample_id",
                    "target_name",
                    "actual_value",
                    "predicted_value",
                    "residual",
                ),
            )
            writer.writeheader()
            writer.writerows(predictions)
        atomic_write_json(staged_training_config, training_config)
        if auto_search_payload is not None:
            atomic_write_json(
                staged_auto_search_results,
                auto_search_payload,
            )
        if ensemble_metadata is not None:
            atomic_write_json(staged_ensemble_results, ensemble_metadata)
            staged_components = staging / "components"
            staged_components.mkdir()
            for name, source_path in (component_model_artifacts or {}).items():
                shutil.copy2(
                    source_path,
                    staged_components / MODEL_ARTIFACT_NAMES[name],
                )
        atomic_write_json(staging / RUN_MANIFEST_NAME, run_record)
        if final_run_directory.exists():
            raise ModelTrainingError(
                f"Training run folder already exists: {final_run_directory}"
            )
        os.replace(staging, final_run_directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    workflow = dict(manifest.get("workflow") or {})
    workflow.update(
        {
            "stage": "model_trained",
            "completed_steps": 3,
            "next_action": "Review and save the trained model as a book.",
        }
    )
    manifest["workflow"] = workflow
    manifest["model_training"] = {
        "schema_version": RUNS_SCHEMA_VERSION,
        "run_count": len(existing_runs) + 1,
        "latest_run_number": run_number,
        "latest_run_id": run_id,
        "runs": [*existing_runs, run_record],
        "status": TRAINING_COMPLETED,
        "model_name": model_name,
        "dataset_id": dataset.dataset_id,
        "trained_at": trained_at,
        "configuration": configuration,
        "parameters_used": dict(parameters_used),
        "training_rows": training_rows,
        "test_rows": test_rows,
        "metrics": dict(metrics),
        "artifacts": dict(run_record["artifacts"]),
    }
    if auto_search_payload is not None:
        manifest["model_training"]["auto_search"] = auto_search_payload
    if ensemble_metadata is not None:
        manifest["model_training"]["ensemble"] = dict(ensemble_metadata)
    manifest["updated_at"] = trained_at
    atomic_write_json(manifest_path, manifest)
    return _SavedRunArtifacts(
        run_number=run_number,
        run_id=run_id,
        run_directory=final_run_directory,
        model_path=final_model,
        metrics_path=final_metrics,
        predictions_path=final_predictions,
        training_config_path=final_training_config,
        auto_search_results_path=final_auto_search_results,
        ensemble_results_path=final_ensemble_results,
    )


def _existing_run_records(
    project_root: Path,
    manifest: dict[str, Any],
    runs_root: Path,
) -> list[dict[str, Any]]:
    training_state = manifest.get("model_training")
    if not isinstance(training_state, dict):
        return []

    records_by_number: dict[int, dict[str, Any]] = {}
    raw_records = training_state.get("runs")
    if isinstance(raw_records, list):
        for record in raw_records:
            if not isinstance(record, dict):
                continue
            run_number = record.get("run_number")
            if isinstance(run_number, int) and run_number > 0:
                records_by_number[run_number] = dict(record)

    for child in runs_root.iterdir():
        run_manifest = child / RUN_MANIFEST_NAME
        if not child.is_dir() or not run_manifest.is_file():
            continue
        try:
            record = json.loads(run_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_number = record.get("run_number") if isinstance(record, dict) else None
        if isinstance(run_number, int) and run_number > 0:
            records_by_number[run_number] = record

    if not records_by_number:
        legacy_record = _preserve_legacy_training_result(
            project_root,
            training_state,
            runs_root,
        )
        if legacy_record is not None:
            records_by_number[1] = legacy_record

    return [records_by_number[number] for number in sorted(records_by_number)]


def _run_folder_numbers(runs_root: Path) -> set[int]:
    numbers: set[int] = set()
    for child in runs_root.iterdir():
        if not child.is_dir() or not child.name.startswith("run-"):
            continue
        suffix = child.name.removeprefix("run-")
        if suffix.isdigit() and int(suffix) > 0:
            numbers.add(int(suffix))
    return numbers


def _preserve_legacy_training_result(
    project_root: Path,
    training_state: dict[str, Any],
    runs_root: Path,
) -> dict[str, Any] | None:
    if training_state.get("status") != TRAINING_COMPLETED:
        return None
    artifacts = training_state.get("artifacts")
    if not isinstance(artifacts, dict):
        return None

    source_paths: dict[str, Path] = {}
    for key in ("model", "metrics", "predictions"):
        relative_path = artifacts.get(key)
        if not isinstance(relative_path, str) or not relative_path:
            return None
        source = (project_root / relative_path).resolve()
        if not source.is_relative_to(project_root) or not source.is_file():
            return None
        source_paths[key] = source

    run_number = 1
    run_id = "run-0001"
    final_directory = runs_root / run_id
    final_paths = {
        "model": final_directory / MODEL_ARTIFACT_NAME,
        "metrics": final_directory / METRICS_ARTIFACT_NAME,
        "predictions": final_directory / PREDICTIONS_ARTIFACT_NAME,
        "training_config": final_directory / TRAINING_CONFIG_ARTIFACT_NAME,
        "run_manifest": final_directory / RUN_MANIFEST_NAME,
    }
    legacy_configuration = dict(training_state.get("configuration") or {})
    raw_parameters = training_state.get("parameters_used")
    if not isinstance(raw_parameters, dict):
        raw_parameters = {
            "fit_intercept": legacy_configuration.get("fit_intercept", True),
            "positive": legacy_configuration.get("positive", False),
        }
    legacy_parameters = {
        "fit_intercept": bool(raw_parameters.get("fit_intercept", True)),
        "positive": bool(raw_parameters.get("positive", False)),
    }
    legacy_training_mode = str(
        legacy_configuration.get("training_mode") or "auto"
    )
    legacy_record = {
        "schema_version": RUNS_SCHEMA_VERSION,
        "run_number": run_number,
        "run_id": run_id,
        "display_name": "Run 1",
        "status": TRAINING_COMPLETED,
        "model_name": training_state.get("model_name", "linear_regression"),
        "dataset_id": training_state.get("dataset_id"),
        "trained_at": training_state.get("trained_at", ""),
        "configuration": legacy_configuration,
        "parameters_used": legacy_parameters,
        "training_rows": int(training_state.get("training_rows") or 0),
        "test_rows": int(training_state.get("test_rows") or 0),
        "metrics": dict(training_state.get("metrics") or {}),
        "artifacts": {
            key: path.relative_to(project_root).as_posix()
            for key, path in final_paths.items()
        },
        "migrated_from_legacy_layout": True,
    }

    if final_directory.exists():
        run_manifest = final_directory / RUN_MANIFEST_NAME
        if not run_manifest.is_file():
            return None
        try:
            existing = json.loads(run_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return existing if isinstance(existing, dict) else None

    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}-legacy.", dir=runs_root))
    try:
        shutil.copy2(source_paths["model"], staging / MODEL_ARTIFACT_NAME)
        shutil.copy2(source_paths["metrics"], staging / METRICS_ARTIFACT_NAME)
        shutil.copy2(
            source_paths["predictions"],
            staging / PREDICTIONS_ARTIFACT_NAME,
        )
        atomic_write_json(
            staging / TRAINING_CONFIG_ARTIFACT_NAME,
            {
                "model_name": legacy_record["model_name"],
                "training_mode": legacy_training_mode,
                "parameters_used": legacy_parameters,
            },
        )
        atomic_write_json(staging / RUN_MANIFEST_NAME, legacy_record)
        os.replace(staging, final_directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return legacy_record


def _failed_result(model_name: str, error_message: str) -> ModelTrainingResult:
    return ModelTrainingResult(
        success=False,
        status=TRAINING_FAILED,
        model_name=model_name,
        training_rows=0,
        test_rows=0,
        metrics={},
        predictions=[],
        error_message=error_message,
    )
