"""Artifact-backed comparison across supported surrogate model families."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from studio.training_results import (
    TRAINING_COMPLETED,
    TrainingResultsError,
    TrainingResultsView,
    load_latest_training_results,
)


MODEL_FAMILY_ORDER = (
    "linear_regression",
    "xgboost",
    "neural_network",
    "ensemble_ai_engine",
)
MODEL_DISPLAY_NAMES = {
    "linear_regression": "Linear Regression",
    "xgboost": "XGBoost",
    "neural_network": "Neural Network",
    "ensemble_ai_engine": "Ensemble AI Engine",
}
VALIDATION_TIE_ABSOLUTE_TOLERANCE = 1e-12


class ModelComparisonError(RuntimeError):
    """Raised when saved run evidence cannot produce a safe comparison."""


@dataclass(frozen=True, slots=True)
class ComparableModelRun:
    run_id: str
    run_number: int
    model_name: str
    model_family: str
    training_mode: str
    search_level: str | None
    parameters_used: dict[str, Any]
    validation_rmse: float | None
    test_rmse: float
    mae: float
    r_squared: float
    trained_at: str


@dataclass(frozen=True, slots=True)
class ModelFamilyComparison:
    model_name: str
    display_name: str
    best_run: ComparableModelRun | None
    compatible_run_count: int
    missing_validation_count: int


@dataclass(frozen=True, slots=True)
class ModelComparisonResult:
    anchor_run_id: str
    dataset_id: str
    dataset_fingerprint: str
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    test_size: float
    random_state: int
    families: tuple[ModelFamilyComparison, ...]
    recommended_model: str | None
    recommendation_title: str
    recommendation_reason: str
    compatible_run_count: int
    incompatible_run_count: int
    invalid_run_count: int

    def family(self, model_name: str) -> ModelFamilyComparison:
        return next(item for item in self.families if item.model_name == model_name)


def compare_compatible_model_runs(
    project_path: str | Path,
    *,
    anchor_run_id: str | None = None,
) -> ModelComparisonResult:
    """Compare the best validation-backed run from each compatible family."""

    project_root, summaries = _completed_run_summaries(project_path)
    if not summaries:
        raise ModelComparisonError(
            "No completed training runs are available for model comparison."
        )
    if anchor_run_id is None:
        anchor_run_id = str(
            max(summaries, key=lambda item: int(item.get("run_number") or 0)).get(
                "run_id"
            )
            or ""
        )
    try:
        anchor = load_latest_training_results(
            project_root,
            run_id=anchor_run_id,
        )
    except TrainingResultsError as exc:
        raise ModelComparisonError(str(exc)) from exc
    if anchor is None:
        raise ModelComparisonError(
            "The selected comparison run is not available."
        )

    compatible: list[ComparableModelRun] = []
    incompatible_count = 0
    invalid_count = 0
    for summary in sorted(
        summaries,
        key=lambda item: int(item.get("run_number") or 0),
    ):
        run_id = str(summary.get("run_id") or "")
        try:
            view = load_latest_training_results(project_root, run_id=run_id)
        except TrainingResultsError:
            invalid_count += 1
            continue
        if view is None:
            invalid_count += 1
            continue
        if not _runs_are_compatible(anchor, view):
            incompatible_count += 1
            continue
        if view.model_name not in MODEL_FAMILY_ORDER:
            incompatible_count += 1
            continue
        compatible.append(_comparison_run(view))

    families: list[ModelFamilyComparison] = []
    for model_name in MODEL_FAMILY_ORDER:
        runs = [run for run in compatible if run.model_name == model_name]
        valid = [run for run in runs if run.validation_rmse is not None]
        best = min(
            valid,
            key=lambda run: (float(run.validation_rmse), run.run_number),
            default=None,
        )
        families.append(
            ModelFamilyComparison(
                model_name=model_name,
                display_name=MODEL_DISPLAY_NAMES[model_name],
                best_run=best,
                compatible_run_count=len(runs),
                missing_validation_count=len(runs) - len(valid),
            )
        )

    recommended_model, title, reason = _recommendation(tuple(families))
    return ModelComparisonResult(
        anchor_run_id=anchor.run_id,
        dataset_id=anchor.dataset_id,
        dataset_fingerprint=anchor.dataset_fingerprint,
        feature_columns=anchor.feature_columns,
        target_columns=anchor.target_columns,
        test_size=anchor.test_size,
        random_state=anchor.random_state,
        families=tuple(families),
        recommended_model=recommended_model,
        recommendation_title=title,
        recommendation_reason=reason,
        compatible_run_count=len(compatible),
        incompatible_run_count=incompatible_count,
        invalid_run_count=invalid_count,
    )


def _recommendation(
    families: tuple[ModelFamilyComparison, ...],
) -> tuple[str | None, str, str]:
    available = [family for family in families if family.best_run is not None]
    if len(available) < 2:
        missing = [
            family.display_name
            for family in families
            if family.best_run is None
        ]
        return (
            None,
            "No Model Recommendation Yet",
            "Validation-backed compatible runs are required for at least two "
            f"model families. Missing: {', '.join(missing)}. Test metrics alone are "
            "not used to recommend a model.",
        )

    ranked = sorted(
        available,
        key=lambda family: (
            float(family.best_run.validation_rmse),
            MODEL_FAMILY_ORDER.index(family.model_name),
        ),
    )
    preferred, runner_up = ranked[:2]
    preferred_score = float(preferred.best_run.validation_rmse)
    runner_up_score = float(runner_up.best_run.validation_rmse)
    tied = math.isclose(
        preferred_score,
        runner_up_score,
        rel_tol=0.0,
        abs_tol=VALIDATION_TIE_ABSOLUTE_TOLERANCE,
    )
    if tied:
        tied_families = [
            family
            for family in available
            if math.isclose(
                float(family.best_run.validation_rmse),
                preferred_score,
                rel_tol=0.0,
                abs_tol=VALIDATION_TIE_ABSOLUTE_TOLERANCE,
            )
        ]
        preferred = min(
            tied_families,
            key=lambda family: MODEL_FAMILY_ORDER.index(family.model_name),
        )
        return (
            preferred.model_name,
            f"Recommended Model: {preferred.display_name}",
            "Validation RMSE is tied within the deterministic tolerance; the "
            "earlier, simpler model family in the supported order is preferred.",
        )
    return (
        preferred.model_name,
        f"Recommended Model: {preferred.display_name}",
        f"{preferred.display_name} has the lower compatible validation RMSE "
        f"({preferred.best_run.validation_rmse:.6g} versus "
        f"{runner_up.best_run.validation_rmse:.6g} for {runner_up.display_name}). "
        "Held-out test metrics are shown "
        "for context but were not used for this recommendation.",
    )


def _runs_are_compatible(
    anchor: TrainingResultsView,
    candidate: TrainingResultsView,
) -> bool:
    return (
        anchor.dataset_id == candidate.dataset_id
        and anchor.dataset_fingerprint == candidate.dataset_fingerprint
        and anchor.feature_columns == candidate.feature_columns
        and anchor.target_columns == candidate.target_columns
        and anchor.test_size == candidate.test_size
        and anchor.random_state == candidate.random_state
    )


def _comparison_run(view: TrainingResultsView) -> ComparableModelRun:
    validation = view.validation_rmse
    if validation is not None and not math.isfinite(validation):
        validation = None
    return ComparableModelRun(
        run_id=view.run_id,
        run_number=view.run_number,
        model_name=view.model_name,
        model_family=MODEL_DISPLAY_NAMES[view.model_name],
        training_mode=view.training_mode,
        search_level=view.search_level,
        parameters_used=dict(view.parameters_used),
        validation_rmse=validation,
        test_rmse=view.metrics["RMSE"],
        mae=view.metrics["MAE"],
        r_squared=view.metrics["R²"],
        trained_at=view.trained_at,
    )


def _completed_run_summaries(
    project_path: str | Path,
) -> tuple[Path, list[dict[str, Any]]]:
    project_root = Path(project_path).expanduser().resolve()
    if project_root.is_file() and project_root.name == "project.json":
        project_root = project_root.parent
    try:
        manifest = json.loads(
            (project_root / "project.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelComparisonError(
            "The project manifest could not be read for model comparison."
        ) from exc
    training = manifest.get("model_training") if isinstance(manifest, dict) else None
    records = training.get("runs") if isinstance(training, dict) else None
    if not isinstance(records, list):
        return project_root, []
    return project_root, [
        dict(record)
        for record in records
        if isinstance(record, dict)
        and record.get("status") == TRAINING_COMPLETED
        and isinstance(record.get("run_number"), int)
        and isinstance(record.get("run_id"), str)
    ]
