"""Read-only training-result loading, comparison, and deterministic insights."""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from studio.dataset_registry import (
    DatasetRegistrationError,
    RegisteredDataset,
    get_registered_dataset,
)


TRAINING_COMPLETED = "TRAINING_COMPLETED"
CUSTOM_VALIDATION_RMSE_TOLERANCE = 0.01
EXPECTED_PREDICTION_COLUMNS = (
    "Sample ID",
    "Actual",
    "Predicted",
    "Residual",
    "Absolute Error",
)


class TrainingResultsError(RuntimeError):
    """Raised when saved result artifacts cannot be displayed safely."""


@dataclass(frozen=True, slots=True)
class PredictionResult:
    sample_id: str
    target_name: str
    actual_value: float
    predicted_value: float
    residual: float
    absolute_error: float


@dataclass(frozen=True, slots=True)
class AutoCandidateResult:
    parameters: dict[str, Any]
    fold_rmse: tuple[float, ...]
    mean_validation_rmse: float | None
    success: bool
    error_message: str | None
    selected: bool


@dataclass(frozen=True, slots=True)
class CustomRecommendation:
    comparable_auto_run_id: str
    suggested_parameters: dict[str, bool]
    suggested_validation_rmse: float
    custom_validation_rmse: float | None
    auto_test_metrics: dict[str, float]
    recommendation: str
    explanation: str
    relative_validation_difference: float | None
    tolerance: float = CUSTOM_VALIDATION_RMSE_TOLERANCE


@dataclass(slots=True)
class TrainingResultsView:
    project_path: Path
    run_directory: Path
    run_id: str
    run_number: int
    model_name: str
    training_mode: str
    search_level: str | None
    parameters_used: dict[str, Any]
    test_size: float
    random_state: int
    training_rows: int
    test_rows: int
    dataset_id: str
    dataset_fingerprint: str
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    sample_input_values: dict[str, dict[str, float]]
    trained_at: str
    metrics: dict[str, float]
    validation_rmse: float | None
    predictions: list[PredictionResult]
    auto_candidates: list[AutoCandidateResult] = field(default_factory=list)
    configurations_evaluated: int = 0
    cross_validation_folds: int | None = None
    custom_recommendation: CustomRecommendation | None = None
    custom_guidance: str = ""
    insights: list[str] = field(default_factory=list)
    residual_interpretation: str = ""
    predictions_path: Path | None = None
    target_unit: str | None = None
    ensemble_components: list[dict[str, Any]] = field(default_factory=list)
    ensemble_failures: list[dict[str, str]] = field(default_factory=list)
    ensemble_weights: dict[str, float] = field(default_factory=dict)
    best_individual_model: str | None = None
    best_individual_validation_rmse: float | None = None
    ensemble_improved_on_best: bool | None = None

    @property
    def median_absolute_error(self) -> float:
        return float(
            statistics.median(
                prediction.absolute_error for prediction in self.predictions
            )
        )

    @property
    def largest_error_prediction(self) -> PredictionResult:
        return max(
            self.predictions,
            key=lambda prediction: (
                prediction.absolute_error,
                prediction.sample_id,
            ),
        )

    @property
    def prediction_sample_ids(self) -> tuple[str, ...]:
        """Return test-sample IDs in their saved prediction order."""

        return tuple(dict.fromkeys(row.sample_id for row in self.predictions))

    def predictions_for_sample(self, sample_id: str) -> tuple[PredictionResult, ...]:
        """Return one test sample's outputs in registered target-column order."""

        matching = [
            row for row in self.predictions if row.sample_id == sample_id
        ]
        target_order = {
            target_name: index
            for index, target_name in enumerate(self.target_columns)
        }
        return tuple(
            row
            for _, row in sorted(
                enumerate(matching),
                key=lambda item: (
                    target_order.get(
                        item[1].target_name,
                        0
                        if not item[1].target_name
                        and len(self.target_columns) == 1
                        else len(target_order),
                    ),
                    item[0],
                ),
            )
        )

    @property
    def recommendation_title(self) -> str:
        if self.model_name == "ensemble_ai_engine":
            return (
                "Ensemble Recommended"
                if self.ensemble_improved_on_best
                else "Ensemble Evaluated"
            )
        if self.model_name == "neural_network":
            return (
                "Recommended Configuration Selected"
                if self.training_mode == "auto"
                else "Custom Configuration Evaluated"
            )
        if self.model_name == "xgboost":
            return (
                "Recommended Configuration Selected"
                if self.training_mode == "auto" and self.search_level is not None
                else (
                    "Fixed Baseline Evaluated"
                    if self.training_mode == "auto"
                    else "Custom Configuration Evaluated"
                )
            )
        return (
            "Recommended Configuration Selected"
            if self.training_mode == "auto"
            else "Custom Configuration Evaluated"
        )

    @property
    def recommendation_statement(self) -> str:
        if self.model_name == "ensemble_ai_engine":
            if self.ensemble_improved_on_best:
                return "Ensemble AI Engine improved on the best individual model"
            return "Best individual model remains recommended"
        if self.model_name == "neural_network":
            return (
                "Recommended Neural Network configuration selected"
                if self.training_mode == "auto"
                else "Your custom Neural Network configuration was evaluated"
            )
        if self.model_name == "xgboost":
            return (
                "Recommended XGBoost configuration selected"
                if self.training_mode == "auto" and self.search_level is not None
                else (
                    "XGBoost fixed baseline configuration evaluated"
                    if self.training_mode == "auto"
                    else "Your custom XGBoost configuration was evaluated"
                )
            )
        return (
            "Recommended Linear Regression configuration selected"
            if self.training_mode == "auto"
            else "Your custom configuration was evaluated"
        )


def load_latest_training_results(
    project_path: str | Path,
    *,
    run_id: str | None = None,
) -> TrainingResultsView | None:
    """Load one completed run, defaulting to the latest, without fitting a model."""

    project_root, project_manifest = _read_project(project_path)
    run_summaries = _completed_run_summaries(project_manifest)
    if not run_summaries:
        return None
    if run_id is None:
        selected_summary = max(
            run_summaries,
            key=lambda record: int(record.get("run_number") or 0),
        )
    else:
        selected_summary = next(
            (
                summary
                for summary in run_summaries
                if str(summary.get("run_id") or "") == run_id
            ),
            None,
        )
        if selected_summary is None:
            raise TrainingResultsError(
                f"The completed training run '{run_id}' is not available."
            )
    run_record, run_directory = _load_run_record(project_root, selected_summary)
    dataset = _load_run_dataset(project_root, run_record)
    metrics = _load_metrics(project_root, run_record)
    predictions_path = _artifact_path(
        project_root,
        run_record,
        "predictions",
        "test_predictions.csv",
    )
    predictions = _load_predictions(predictions_path)
    sample_input_values = _load_sample_input_values(dataset)
    missing_input_samples = sorted(
        {
            prediction.sample_id
            for prediction in predictions
            if prediction.sample_id not in sample_input_values
        }
    )
    if missing_input_samples:
        raise TrainingResultsError(
            "Saved predictions could not be matched to the registered input "
            f"data for sample: {missing_input_samples[0]}."
        )
    training_config = _load_json_artifact(
        _artifact_path(
            project_root,
            run_record,
            "training_config",
            "training_config.json",
        ),
        "training configuration",
    )
    model_name = str(run_record.get("model_name") or "linear_regression")
    parameters = _model_parameters(
        training_config.get("parameters_used"),
        model_name,
    )
    training_mode = str(
        training_config.get("training_mode")
        or run_record.get("configuration", {}).get("training_mode")
        or ""
    ).lower()
    if training_mode not in {"auto", "custom"}:
        raise TrainingResultsError(
            "The latest run has an unsupported training-mode record."
        )

    auto_candidates: list[AutoCandidateResult] = []
    ensemble_components: list[dict[str, Any]] = []
    ensemble_failures: list[dict[str, str]] = []
    ensemble_weights: dict[str, float] = {}
    best_individual_model: str | None = None
    best_individual_validation_rmse: float | None = None
    ensemble_improved_on_best: bool | None = None
    validation_rmse: float | None = None
    configurations_evaluated = 0
    cross_validation_folds: int | None = None
    search_level = training_config.get("search_level")
    has_auto_search = training_mode == "auto" and (
        model_name == "linear_regression" or search_level is not None
    )
    if model_name == "ensemble_ai_engine":
        if training_mode != "auto" or str(search_level).lower() != "high":
            raise TrainingResultsError(
                "Ensemble AI Engine results require Auto High metadata."
            )
        ensemble = _load_json_artifact(
            _artifact_path(
                project_root,
                run_record,
                "ensemble_results",
                "ensemble_results.json",
            ),
            "Ensemble results",
        )
        ensemble_components = _ensemble_component_results(
            ensemble.get("component_results")
        )
        ensemble_failures = _ensemble_failures(
            ensemble.get("component_failures")
        )
        configurations_evaluated = len(ensemble_components) + len(
            ensemble_failures
        )
        cross_validation_folds = _positive_int(
            ensemble.get("cross_validation_folds"),
            "cross-validation folds",
        )
        validation_rmse = _finite_float(
            ensemble.get("ensemble_validation_rmse"),
            "ensemble validation RMSE",
        )
        ensemble_weights = {
            component["model_name"]: component["weight"]
            for component in ensemble_components
        }
        saved_weights = parameters.get("weights")
        if ensemble_weights != saved_weights:
            raise TrainingResultsError(
                "The ensemble weights do not match training_config.json."
            )
        saved_components = parameters.get("components")
        loaded_components = {
            component["model_name"]: component["parameters_used"]
            for component in ensemble_components
        }
        if loaded_components != saved_components:
            raise TrainingResultsError(
                "The ensemble component parameters do not match training_config.json."
            )
        best_individual_model = str(
            ensemble.get("best_individual_model") or ""
        )
        if best_individual_model not in ensemble_weights:
            raise TrainingResultsError(
                "The Ensemble results identify an invalid best individual model."
            )
        best_individual_validation_rmse = _finite_float(
            ensemble.get("best_individual_validation_rmse"),
            "best individual validation RMSE",
        )
        improved_value = ensemble.get("ensemble_improved_on_best")
        if not isinstance(improved_value, bool):
            raise TrainingResultsError(
                "The Ensemble recommendation evidence is invalid."
            )
        ensemble_improved_on_best = improved_value
        search_level = "high"
    elif has_auto_search:
        auto_search = _load_json_artifact(
            _artifact_path(
                project_root,
                run_record,
                "auto_search_results",
                "auto_search_results.json",
            ),
            "Auto search results",
        )
        search_level = str(auto_search.get("search_level") or "").lower()
        if search_level not in {"medium", "high"}:
            raise TrainingResultsError(
                "The Auto search artifact has an unsupported search level."
            )
        configurations_evaluated = _nonnegative_int(
            auto_search.get("configurations_evaluated"),
            "configurations evaluated",
        )
        cross_validation_folds = _positive_int(
            auto_search.get("cross_validation_folds"),
            "cross-validation folds",
        )
        validation_rmse = _finite_float(
            auto_search.get("best_validation_rmse"),
            "best validation RMSE",
        )
        selected_parameters = _model_parameters(
            auto_search.get("best_parameters"),
            model_name,
        )
        if selected_parameters != parameters:
            raise TrainingResultsError(
                "The selected Auto parameters do not match training_config.json."
            )
        auto_candidates = _auto_candidates(
            auto_search.get("search_results"),
            selected_parameters,
            model_name,
        )
        if configurations_evaluated != len(auto_candidates):
            raise TrainingResultsError(
                "The Auto search configuration count is inconsistent."
            )
    elif model_name == "linear_regression":
        if search_level is not None:
            raise TrainingResultsError(
                "A Custom run cannot contain an Auto search level."
            )
    elif model_name in {"xgboost", "neural_network"}:
        if training_mode == "custom" and search_level is not None:
            raise TrainingResultsError(
                "A Custom run cannot contain an Auto search level."
            )
    else:
        raise TrainingResultsError(
            f"The latest run uses an unsupported model: {model_name}."
        )

    run_id = str(run_record.get("run_id") or run_directory.name)
    run_number = _positive_int(run_record.get("run_number"), "run number")
    configuration = run_record.get("configuration")
    if not isinstance(configuration, dict):
        raise TrainingResultsError(
            "The saved run is missing its split configuration."
        )
    test_size = _finite_float(configuration.get("test_size"), "test split size")
    random_state = configuration.get("random_state")
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise TrainingResultsError(
            "The saved run has an invalid split random state."
        )
    view = TrainingResultsView(
        project_path=project_root,
        run_directory=run_directory,
        run_id=run_id,
        run_number=run_number,
        model_name=model_name,
        training_mode=training_mode,
        search_level=str(search_level) if search_level is not None else None,
        parameters_used=parameters,
        test_size=test_size,
        random_state=random_state,
        training_rows=_nonnegative_int(
            run_record.get("training_rows"), "training rows"
        ),
        test_rows=_nonnegative_int(run_record.get("test_rows"), "test rows"),
        dataset_id=dataset.dataset_id,
        dataset_fingerprint=dataset.fingerprint_sha256,
        feature_columns=tuple(dataset.feature_columns),
        target_columns=tuple(dataset.target_columns),
        sample_input_values=sample_input_values,
        trained_at=str(run_record.get("trained_at") or ""),
        metrics=metrics,
        validation_rmse=validation_rmse,
        predictions=predictions,
        auto_candidates=auto_candidates,
        configurations_evaluated=configurations_evaluated,
        cross_validation_folds=cross_validation_folds,
        predictions_path=predictions_path,
        target_unit=_target_unit(project_manifest, dataset.target_columns),
        ensemble_components=ensemble_components,
        ensemble_failures=ensemble_failures,
        ensemble_weights=ensemble_weights,
        best_individual_model=best_individual_model,
        best_individual_validation_rmse=best_individual_validation_rmse,
        ensemble_improved_on_best=ensemble_improved_on_best,
    )
    if training_mode == "custom" and model_name == "linear_regression":
        view.custom_recommendation = _find_custom_recommendation(
            project_root,
            project_manifest,
            run_record,
            dataset,
            parameters,
        )
        if view.custom_recommendation is None:
            view.custom_guidance = (
                "No comparable Auto result is available for this dataset. "
                "Run Auto training to generate a recommendation."
            )
        else:
            view.validation_rmse = (
                view.custom_recommendation.custom_validation_rmse
            )
    view.residual_interpretation = _residual_interpretation(view)
    view.insights = _plain_language_insights(view)
    return view


def metric_card_data(view: TrainingResultsView) -> list[dict[str, Any]]:
    """Return the four fixed, plain-language metric-card definitions."""

    unit_suffix = f" {view.target_unit}" if view.target_unit else ""
    r_squared = view.metrics["R²"]
    r_squared_meaning = (
        f"The model explains {r_squared * 100:.1f}% of the variation in the "
        "held-out test data."
        if r_squared >= 0
        else (
            "The negative value means the held-out predictions perform worse "
            "than predicting the test-data mean."
        )
    )
    return [
        {
            "name": "R²",
            "value": r_squared,
            "display_value": f"{r_squared:.6g}",
            "meaning": r_squared_meaning,
            "direction": "Higher is better.",
        },
        {
            "name": "RMSE",
            "value": view.metrics["RMSE"],
            "display_value": f"{view.metrics['RMSE']:.6g}{unit_suffix}",
            "meaning": (
                "Typical test prediction error, with larger errors weighted "
                "more heavily."
            ),
            "direction": "Lower is better.",
        },
        {
            "name": "MAE",
            "value": view.metrics["MAE"],
            "display_value": f"{view.metrics['MAE']:.6g}{unit_suffix}",
            "meaning": "Average absolute prediction error on the test data.",
            "direction": "Lower is better.",
        },
        {
            "name": "Validation RMSE",
            "value": view.validation_rmse,
            "display_value": (
                f"{view.validation_rmse:.6g}{unit_suffix}"
                if view.validation_rmse is not None
                else "Not available"
            ),
            "meaning": (
                "Training-only cross-validation error used for configuration "
                "selection."
                if view.validation_rmse is not None
                else "No comparable training-only validation score is available."
            ),
            "direction": "Lower is better.",
        },
    ]


def prediction_table_rows(view: TrainingResultsView) -> list[dict[str, Any]]:
    """Return sortable prediction-table data using the backend residual rule."""

    return [
        {
            "Sample ID": prediction.sample_id,
            "Actual": prediction.actual_value,
            "Predicted": prediction.predicted_value,
            "Residual": prediction.residual,
            "Absolute Error": prediction.absolute_error,
        }
        for prediction in view.predictions
    ]


def _read_project(project_path: str | Path) -> tuple[Path, dict[str, Any]]:
    project_root = Path(project_path).expanduser().resolve()
    if project_root.is_file() and project_root.name == "project.json":
        project_root = project_root.parent
    try:
        payload = json.loads(
            (project_root / "project.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingResultsError(
            "The project manifest could not be read for Training Results."
        ) from exc
    if not isinstance(payload, dict) or not payload.get("project_id"):
        raise TrainingResultsError(
            "The project manifest is invalid for Training Results."
        )
    return project_root, payload


def _completed_run_summaries(project_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    training = project_manifest.get("model_training")
    if not isinstance(training, dict):
        return []
    records = training.get("runs")
    if not isinstance(records, list):
        return []
    return [
        dict(record)
        for record in records
        if isinstance(record, dict)
        and record.get("status") == TRAINING_COMPLETED
        and isinstance(record.get("run_number"), int)
    ]


def _load_run_record(
    project_root: Path,
    run_summary: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    artifacts = run_summary.get("artifacts")
    relative_manifest = (
        artifacts.get("run_manifest") if isinstance(artifacts, dict) else None
    )
    if isinstance(relative_manifest, str) and relative_manifest:
        manifest_path = _safe_project_path(project_root, relative_manifest)
    else:
        run_id = str(run_summary.get("run_id") or "")
        manifest_path = project_root / "models" / "runs" / run_id / "run.json"
    payload = _load_json_artifact(manifest_path, "run manifest")
    if payload.get("status") != TRAINING_COMPLETED:
        raise TrainingResultsError(
            "The latest run is not recorded as a completed training run."
        )
    return payload, manifest_path.parent


def _load_run_dataset(
    project_root: Path,
    run_record: dict[str, Any],
) -> RegisteredDataset:
    dataset_id = str(run_record.get("dataset_id") or "")
    if not dataset_id:
        raise TrainingResultsError(
            "The latest run does not identify its registered dataset."
        )
    try:
        return get_registered_dataset(project_root, dataset_id)
    except DatasetRegistrationError as exc:
        raise TrainingResultsError(
            "The registered dataset for this run is missing or failed its "
            "integrity check."
        ) from exc


def _load_metrics(
    project_root: Path,
    run_record: dict[str, Any],
) -> dict[str, float]:
    payload = _load_json_artifact(
        _artifact_path(
            project_root,
            run_record,
            "metrics",
            "metrics.json",
        ),
        "metrics",
    )
    return {
        name: _finite_float(payload.get(name), name)
        for name in ("MAE", "RMSE", "R²")
    }


def _load_predictions(path: Path) -> list[PredictionResult]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {
                "sample_id",
                "actual_value",
                "predicted_value",
                "residual",
            }
            if not required.issubset(reader.fieldnames or []):
                raise TrainingResultsError(
                    "test_predictions.csv is missing required columns."
                )
            predictions: list[PredictionResult] = []
            for row_number, row in enumerate(reader, start=2):
                actual = _finite_float(
                    row.get("actual_value"),
                    f"actual value on prediction row {row_number}",
                )
                predicted = _finite_float(
                    row.get("predicted_value"),
                    f"predicted value on prediction row {row_number}",
                )
                saved_residual = _finite_float(
                    row.get("residual"),
                    f"residual on prediction row {row_number}",
                )
                residual = actual - predicted
                if not math.isclose(
                    saved_residual,
                    residual,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise TrainingResultsError(
                        "test_predictions.csv contains a residual that does not "
                        "equal Actual − Predicted."
                    )
                predictions.append(
                    PredictionResult(
                        sample_id=str(row.get("sample_id") or ""),
                        target_name=str(row.get("target_name") or ""),
                        actual_value=actual,
                        predicted_value=predicted,
                        residual=residual,
                        absolute_error=abs(residual),
                    )
                )
    except TrainingResultsError:
        raise
    except (OSError, csv.Error) as exc:
        raise TrainingResultsError(
            "test_predictions.csv could not be read."
        ) from exc
    if not predictions:
        raise TrainingResultsError(
            "test_predictions.csv contains no test predictions."
        )
    return predictions


def _load_sample_input_values(
    dataset: RegisteredDataset,
) -> dict[str, dict[str, float]]:
    """Read registered feature values for result display without changing data."""

    try:
        with dataset.input_csv_path.open(
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            required = set(dataset.feature_columns)
            if dataset.sample_id_column is not None:
                required.add(dataset.sample_id_column)
            if not required.issubset(headers):
                raise TrainingResultsError(
                    "The registered input CSV is missing columns required to "
                    "display test-sample inputs."
                )
            sample_inputs: dict[str, dict[str, float]] = {}
            for row_index, row in enumerate(reader, start=1):
                sample_id = (
                    str(row.get(dataset.sample_id_column) or "").strip()
                    if dataset.sample_id_column is not None
                    else f"Sample_{row_index:06d}"
                )
                if not sample_id:
                    raise TrainingResultsError(
                        "The registered input CSV contains an empty sample ID."
                    )
                sample_inputs[sample_id] = {
                    feature: _finite_float(
                        row.get(feature),
                        f"input value for '{feature}' in sample '{sample_id}'",
                    )
                    for feature in dataset.feature_columns
                }
    except TrainingResultsError:
        raise
    except (OSError, csv.Error) as exc:
        raise TrainingResultsError(
            "The registered input CSV could not be read for result display."
        ) from exc
    return sample_inputs


def _auto_candidates(
    raw_results: Any,
    selected_parameters: dict[str, Any],
    model_name: str = "linear_regression",
) -> list[AutoCandidateResult]:
    if not isinstance(raw_results, list):
        raise TrainingResultsError(
            "The Auto search artifact has no configuration results."
        )
    candidates: list[AutoCandidateResult] = []
    for raw in raw_results:
        if not isinstance(raw, dict):
            raise TrainingResultsError(
                "The Auto search artifact contains an invalid configuration."
            )
        parameters = _model_parameters(raw.get("parameters"), model_name)
        success = raw.get("success") is True
        if success:
            mean_rmse = _finite_float(
                raw.get("mean_validation_rmse"),
                "mean validation RMSE",
            )
            fold_values = raw.get("fold_rmse")
            if not isinstance(fold_values, list) or not fold_values:
                raise TrainingResultsError(
                    "A successful Auto configuration has no fold scores."
                )
            fold_rmse = tuple(
                _finite_float(value, "fold RMSE") for value in fold_values
            )
            error_message = None
        else:
            mean_rmse = None
            fold_rmse = ()
            error_message = str(
                raw.get("error_message") or "Configuration evaluation failed."
            )
        candidates.append(
            AutoCandidateResult(
                parameters=parameters,
                fold_rmse=fold_rmse,
                mean_validation_rmse=mean_rmse,
                success=success,
                error_message=error_message,
                selected=success and parameters == selected_parameters,
            )
        )
    if model_name == "linear_regression":
        candidates.sort(
            key=lambda candidate: (
                not candidate.success,
                candidate.mean_validation_rmse
                if candidate.mean_validation_rmse is not None
                else math.inf,
                candidate.parameters["positive"],
                not candidate.parameters["fit_intercept"],
            )
        )
    else:
        candidates.sort(
            key=lambda candidate: (
                not candidate.success,
                candidate.mean_validation_rmse
                if candidate.mean_validation_rmse is not None
                else math.inf,
            )
        )
    if not any(candidate.selected for candidate in candidates):
        raise TrainingResultsError(
            "The selected Auto configuration is missing from the search results."
        )
    return candidates


def _find_custom_recommendation(
    project_root: Path,
    project_manifest: dict[str, Any],
    custom_run: dict[str, Any],
    custom_dataset: RegisteredDataset,
    custom_parameters: dict[str, bool],
) -> CustomRecommendation | None:
    compatible: list[
        tuple[float, int, dict[str, Any], dict[str, Any], dict[str, float]]
    ] = []
    custom_configuration = custom_run.get("configuration")
    if not isinstance(custom_configuration, dict):
        return None
    for summary in _completed_run_summaries(project_manifest):
        if summary.get("run_id") == custom_run.get("run_id"):
            continue
        try:
            run_record, _ = _load_run_record(project_root, summary)
            configuration = run_record.get("configuration")
            if not isinstance(configuration, dict):
                continue
            if str(configuration.get("training_mode") or "") != "auto":
                continue
            if run_record.get("model_name") != custom_run.get("model_name"):
                continue
            auto_dataset = _load_run_dataset(project_root, run_record)
            if not _datasets_are_compatible(custom_dataset, auto_dataset):
                continue
            if not _split_is_compatible(custom_configuration, configuration):
                continue
            auto_search = _load_json_artifact(
                _artifact_path(
                    project_root,
                    run_record,
                    "auto_search_results",
                    "auto_search_results.json",
                ),
                "Auto search results",
            )
            best_validation = _finite_float(
                auto_search.get("best_validation_rmse"),
                "best validation RMSE",
            )
            auto_metrics = _load_metrics(project_root, run_record)
            compatible.append(
                (
                    best_validation,
                    -int(run_record.get("run_number") or 0),
                    run_record,
                    auto_search,
                    auto_metrics,
                )
            )
        except TrainingResultsError:
            continue
    if not compatible:
        return None

    best_validation, _, auto_run, auto_search, auto_metrics = min(compatible)
    suggested_parameters = _boolean_parameters(auto_search.get("best_parameters"))
    custom_validation = _matching_validation_rmse(
        auto_search.get("search_results"),
        custom_parameters,
    )
    relative_difference: float | None = None
    if custom_validation is None:
        recommendation = (
            "Suggested configuration available; validation comparison unavailable."
        )
        explanation = (
            "This compatible Auto run did not evaluate your exact Custom "
            "configuration. Run High Auto training to obtain validation evidence "
            "for all four Boolean configurations."
        )
    else:
        denominator = max(abs(best_validation), 1e-12)
        relative_difference = (
            custom_validation - best_validation
        ) / denominator
        if relative_difference > CUSTOM_VALIDATION_RMSE_TOLERANCE:
            recommendation = "Suggestion: Use the Auto-selected configuration."
            explanation = (
                "The suggested configuration achieved a meaningfully lower "
                "validation RMSE on the same dataset and compatible split."
            )
        elif relative_difference < -CUSTOM_VALIDATION_RMSE_TOLERANCE:
            recommendation = (
                "Your Custom configuration currently has the stronger "
                "validation result."
            )
            explanation = (
                "Its validation RMSE is lower on the same dataset and compatible "
                "split."
            )
        else:
            recommendation = (
                "Your Custom configuration performs similarly to the "
                "Auto-selected configuration."
            )
            explanation = (
                "The relative validation-RMSE difference is within the named "
                f"{CUSTOM_VALIDATION_RMSE_TOLERANCE:.0%} tolerance."
            )
    return CustomRecommendation(
        comparable_auto_run_id=str(auto_run.get("run_id") or ""),
        suggested_parameters=suggested_parameters,
        suggested_validation_rmse=best_validation,
        custom_validation_rmse=custom_validation,
        auto_test_metrics=auto_metrics,
        recommendation=recommendation,
        explanation=explanation,
        relative_validation_difference=relative_difference,
    )


def _matching_validation_rmse(
    raw_results: Any,
    parameters: dict[str, bool],
) -> float | None:
    if not isinstance(raw_results, list):
        return None
    for raw in raw_results:
        if not isinstance(raw, dict) or raw.get("success") is not True:
            continue
        try:
            candidate_parameters = _boolean_parameters(raw.get("parameters"))
        except TrainingResultsError:
            continue
        if candidate_parameters == parameters:
            return _finite_float(
                raw.get("mean_validation_rmse"),
                "mean validation RMSE",
            )
    return None


def _datasets_are_compatible(
    left: RegisteredDataset,
    right: RegisteredDataset,
) -> bool:
    return (
        left.fingerprint_sha256 == right.fingerprint_sha256
        and left.feature_columns == right.feature_columns
        and left.target_columns == right.target_columns
    )


def _split_is_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("test_size") == right.get("test_size")
        and left.get("random_state") == right.get("random_state")
    )


def _residual_interpretation(view: TrainingResultsView) -> str:
    mean_residual = statistics.fmean(
        prediction.residual for prediction in view.predictions
    )
    scale = max(view.metrics["RMSE"], 1e-12)
    if abs(mean_residual) <= 0.10 * scale:
        return (
            "Residuals are centered near zero in the held-out samples; this does "
            "not by itself prove statistical validity."
        )
    direction = (
        "positive, so actual values tend to exceed predictions"
        if mean_residual > 0
        else "negative, so predictions tend to exceed actual values"
    )
    return (
        f"The mean residual is {mean_residual:.6g} and leans {direction}. "
        "Inspect the plot for possible directional bias."
    )


def _plain_language_insights(view: TrainingResultsView) -> list[str]:
    insights: list[str] = []
    if view.custom_recommendation is not None:
        insights.append(view.custom_recommendation.recommendation)
    if view.model_name == "ensemble_ai_engine":
        best_label = (view.best_individual_model or "best individual").replace(
            "_", " "
        ).title()
        insights.append(
            "Ensemble AI Engine has the lower validation RMSE and is recommended."
            if view.ensemble_improved_on_best
            else f"{best_label} retains the lower validation RMSE; the Ensemble is not recommended."
        )
    if view.test_rows < 10:
        insights.append(
            f"Only {view.test_rows} held-out samples support these test metrics; "
            "treat the result as preliminary."
        )
    if view.validation_rmse is not None and view.validation_rmse > 0:
        relative_gap = abs(
            view.metrics["RMSE"] - view.validation_rmse
        ) / view.validation_rmse
        if relative_gap > 0.50:
            insights.append(
                "Validation and test RMSE differ by more than 50%; performance "
                "may vary across data subsets."
            )
    median_error = view.median_absolute_error
    largest = view.largest_error_prediction
    if largest.absolute_error > max(3 * median_error, 1e-12):
        insights.append(
            f"Sample {largest.sample_id} has an absolute error more than three "
            "times the median, so a small number of samples may dominate error."
        )
    if view.training_mode == "auto":
        successful = [
            candidate
            for candidate in view.auto_candidates
            if candidate.success and candidate.mean_validation_rmse is not None
        ]
        if len(successful) > 1 and successful[0].mean_validation_rmse is not None:
            best = successful[0].mean_validation_rmse
            second = successful[1].mean_validation_rmse
            if (
                (best == 0 and second > 0)
                or (best > 0 and (second - best) / best > 0.10)
            ):
                insights.append(
                    "The selected Auto configuration reduced mean validation "
                    "RMSE by more than 10% versus the next configuration."
                )
    if len(insights) < 3:
        insights.append(view.residual_interpretation)
    return insights[:3]


def _target_unit(
    project_manifest: dict[str, Any],
    target_columns: list[str],
) -> str | None:
    prep = project_manifest.get("data_prep")
    units = prep.get("target_units") if isinstance(prep, dict) else None
    if not isinstance(units, dict) or not target_columns:
        return None
    values = [units.get(column) for column in target_columns]
    if not all(isinstance(value, str) and value.strip() for value in values):
        return None
    normalized = {str(value).strip() for value in values}
    return normalized.pop() if len(normalized) == 1 else None


def _artifact_path(
    project_root: Path,
    run_record: dict[str, Any],
    artifact_name: str,
    expected_filename: str,
) -> Path:
    artifacts = run_record.get("artifacts")
    relative = artifacts.get(artifact_name) if isinstance(artifacts, dict) else None
    if not isinstance(relative, str) or not relative:
        raise TrainingResultsError(
            f"The latest run does not reference {expected_filename}."
        )
    path = _safe_project_path(project_root, relative)
    if not path.is_file():
        raise TrainingResultsError(
            f"The required result artifact is missing: {expected_filename}."
        )
    return path


def _safe_project_path(project_root: Path, relative: str) -> Path:
    path = (project_root / relative).resolve()
    if not path.is_relative_to(project_root):
        raise TrainingResultsError(
            "A result artifact path points outside the project."
        )
    return path


def _load_json_artifact(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TrainingResultsError(
            f"The required {label} artifact is missing."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingResultsError(
            f"The saved {label} artifact is malformed or unreadable."
        ) from exc
    if not isinstance(payload, dict):
        raise TrainingResultsError(
            f"The saved {label} artifact has an invalid structure."
        )
    return payload


def _model_parameters(value: Any, model_name: str) -> dict[str, Any]:
    if model_name == "linear_regression":
        return _boolean_parameters(value)
    if model_name == "neural_network":
        return _neural_network_parameters(value)
    if model_name == "ensemble_ai_engine":
        return _ensemble_parameters(value)
    if model_name != "xgboost":
        raise TrainingResultsError(
            f"The saved model parameters use an unsupported model: {model_name}."
        )
    if not isinstance(value, dict):
        raise TrainingResultsError("The saved XGBoost parameters are missing.")
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
        raise TrainingResultsError(
            f"The saved XGBoost parameter '{missing[0]}' is missing."
        )
    parameters: dict[str, Any] = {}
    for name in required:
        raw = value[name]
        if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
            raise TrainingResultsError(
                f"The saved XGBoost parameter '{name}' is invalid."
            )
        if isinstance(raw, float) and not math.isfinite(raw):
            raise TrainingResultsError(
                f"The saved XGBoost parameter '{name}' is not finite."
            )
        parameters[name] = raw
    return parameters


def _ensemble_parameters(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingResultsError("The saved Ensemble parameters are missing.")
    weights = value.get("weights")
    components = value.get("components")
    if not isinstance(weights, dict) or not isinstance(components, dict):
        raise TrainingResultsError(
            "The saved Ensemble weights or component parameters are missing."
        )
    if len(weights) < 2 or set(weights) != set(components):
        raise TrainingResultsError(
            "The saved Ensemble requires matching metadata for at least two components."
        )
    validated_weights: dict[str, float] = {}
    validated_components: dict[str, dict[str, Any]] = {}
    for model_name in weights:
        if model_name not in {
            "linear_regression",
            "xgboost",
            "neural_network",
        }:
            raise TrainingResultsError(
                f"The saved Ensemble component is unsupported: {model_name}."
            )
        weight = _finite_float(weights[model_name], f"{model_name} weight")
        if weight < 0.0:
            raise TrainingResultsError("Saved Ensemble weights cannot be negative.")
        validated_weights[model_name] = weight
        validated_components[model_name] = _model_parameters(
            components[model_name],
            model_name,
        )
    if not math.isclose(sum(validated_weights.values()), 1.0, abs_tol=1e-9):
        raise TrainingResultsError("Saved Ensemble weights are not normalized.")
    return {
        "weights": validated_weights,
        "components": validated_components,
    }


def _ensemble_component_results(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) < 2:
        raise TrainingResultsError(
            "Ensemble results require at least two completed components."
        )
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, dict):
            raise TrainingResultsError(
                "The Ensemble component result has an invalid structure."
            )
        model_name = str(raw.get("model_name") or "")
        if model_name in seen or model_name not in {
            "linear_regression",
            "xgboost",
            "neural_network",
        }:
            raise TrainingResultsError(
                "The Ensemble component result identifies an invalid model."
            )
        seen.add(model_name)
        run_number = raw.get("source_run_number")
        run_id = str(raw.get("source_run_id") or "")
        if not run_id or isinstance(run_number, bool) or not isinstance(run_number, int):
            raise TrainingResultsError(
                "The Ensemble component result has invalid source-run metadata."
            )
        results.append(
            {
                "model_name": model_name,
                "status": "TRAINING_COMPLETED",
                "source_run_id": run_id,
                "source_run_number": run_number,
                "validation_rmse": _finite_float(
                    raw.get("validation_rmse"),
                    f"{model_name} validation RMSE",
                ),
                "weight": _finite_float(raw.get("weight"), f"{model_name} weight"),
                "parameters_used": _model_parameters(
                    raw.get("parameters_used"),
                    model_name,
                ),
            }
        )
    return results


def _ensemble_failures(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TrainingResultsError("The Ensemble component failures are invalid.")
    failures: list[dict[str, str]] = []
    for raw in value:
        if not isinstance(raw, dict):
            raise TrainingResultsError(
                "The Ensemble component failure has an invalid structure."
            )
        model_name = str(raw.get("model_name") or "").strip()
        message = str(raw.get("error_message") or "").strip()
        if not model_name or not message:
            raise TrainingResultsError(
                "The Ensemble component failure is incomplete."
            )
        failures.append({"model_name": model_name, "error_message": message})
    return failures


def _neural_network_parameters(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingResultsError("The saved Neural Network parameters are missing.")
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
        raise TrainingResultsError(
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
        raise TrainingResultsError(
            "The saved Neural Network hidden-layer architecture is invalid."
        )
    if (
        not isinstance(value["activation"], str)
        or value["activation"] not in {"relu", "tanh", "logistic", "identity"}
    ):
        raise TrainingResultsError(
            "The saved Neural Network activation is invalid."
        )
    parameters = dict(value)
    parameters["hidden_layer_sizes"] = list(layers)
    for name in ("learning_rate_init", "tol"):
        parameters[name] = _finite_float(value[name], name)
    for name in ("batch_size", "max_iter", "random_state", "n_iter_no_change"):
        raw = value[name]
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TrainingResultsError(
                f"The saved Neural Network parameter '{name}' is invalid."
            )
    for name in ("shuffle", "early_stopping"):
        if not isinstance(value[name], bool):
            raise TrainingResultsError(
                f"The saved Neural Network parameter '{name}' is invalid."
            )
    if value["solver"] != "adam":
        raise TrainingResultsError("The saved Neural Network solver is unsupported.")
    return parameters


def _boolean_parameters(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict):
        raise TrainingResultsError(
            "The saved Linear Regression parameters are missing."
        )
    parameters: dict[str, bool] = {}
    for name in ("fit_intercept", "positive"):
        raw = value.get(name)
        if not isinstance(raw, bool):
            raise TrainingResultsError(
                f"The saved parameter '{name}' is not Boolean."
            )
        parameters[name] = raw
    return parameters


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TrainingResultsError(
            f"The saved {label} value is invalid."
        ) from exc
    if not math.isfinite(number):
        raise TrainingResultsError(
            f"The saved {label} value is not finite."
        )
    return number


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise TrainingResultsError(f"The saved {label} value is invalid.")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TrainingResultsError(f"The saved {label} value is invalid.")
    return value
