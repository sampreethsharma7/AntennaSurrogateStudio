"""Generic single-objective inverse design using saved surrogate Model Books."""

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
from typing import Any, Callable

import numpy as np

from studio.inference import InferenceError, ModelBookPredictor
from studio.model_book import ModelBookError
from studio.project_store import atomic_write_json, utc_now


INVERSE_DESIGN_COMPLETED = "INVERSE_DESIGN_COMPLETED"
INVERSE_DESIGN_FAILED = "INVERSE_DESIGN_FAILED"
INVERSE_DESIGN_SCHEMA_VERSION = 2
SUPPORTED_OBJECTIVE_GOALS = {"minimize", "maximize", "target"}
SUPPORTED_OBJECTIVE_AGGREGATIONS = {"single", "mean"}
SUPPORTED_CONSTRAINT_OPERATORS = {
    "greater_than_or_equal",
    "less_than_or_equal",
    "within_range",
}
DEFAULT_MAX_ITERATIONS = 80
DEFAULT_POPULATION_MULTIPLIER = 10
DEFAULT_RANDOM_SEED = 42


class InverseDesignError(RuntimeError):
    """A safe failure that can be displayed without a traceback."""


@dataclass(slots=True)
class InverseDesignHistory:
    """Valid saved searches plus isolated per-run restoration errors."""

    runs: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _name(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value.strip()


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a numeric value.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite numeric value.")
    return number


@dataclass(slots=True)
class InverseDesignObjective:
    """One scalar objective derived from one or more saved outputs."""

    output_name: str | None
    goal: str
    target_value: float | None = None
    aggregation: str = "single"
    output_names: list[str] | None = None

    def __post_init__(self) -> None:
        self.goal = _name(self.goal, "Objective goal").lower()
        if self.goal not in SUPPORTED_OBJECTIVE_GOALS:
            raise ValueError(
                "Objective goal must be minimize, maximize, or target."
            )
        if self.goal == "target":
            if self.target_value is None:
                raise ValueError("A target objective requires a target value.")
            self.target_value = _finite_number(
                self.target_value,
                "Objective target value",
            )
        elif self.target_value is not None:
            raise ValueError(
                "A target value can only be used with the target objective."
            )
        self.aggregation = _name(
            self.aggregation,
            "Objective aggregation",
        ).lower()
        if self.aggregation not in SUPPORTED_OBJECTIVE_AGGREGATIONS:
            raise ValueError("Objective aggregation must be single or mean.")

        raw_names = list(self.output_names or [])
        normalized_names = [
            _name(value, "Objective output name") for value in raw_names
        ]
        duplicates = [
            name
            for index, name in enumerate(normalized_names)
            if name in normalized_names[:index]
        ]
        if duplicates:
            raise ValueError(
                "Duplicate objective outputs were provided: "
                + ", ".join(dict.fromkeys(duplicates))
                + "."
            )
        if self.aggregation == "single":
            if self.output_name is None and len(normalized_names) == 1:
                self.output_name = normalized_names[0]
            self.output_name = _name(self.output_name, "Objective output")
            if normalized_names and normalized_names != [self.output_name]:
                raise ValueError(
                    "A single-output objective must identify exactly one output."
                )
            self.output_names = [self.output_name]
        else:
            if self.output_name is not None:
                raise ValueError(
                    "A mean objective must use output_names instead of output_name."
                )
            if len(normalized_names) < 2:
                raise ValueError(
                    "A mean objective requires at least two ordered outputs."
                )
            self.output_names = normalized_names

    @property
    def selected_outputs(self) -> list[str]:
        return list(self.output_names or [])

    @property
    def display_name(self) -> str:
        if self.aggregation == "single":
            return str(self.output_name)
        return (
            f"Mean of {self.output_names[0]} through {self.output_names[-1]}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_name": self.output_name,
            "output_names": self.selected_outputs,
            "aggregation": self.aggregation,
            "goal": self.goal,
            "target_value": self.target_value,
        }


@dataclass(slots=True)
class OutputConstraint:
    """One generic condition evaluated against one scalar output aggregate."""

    output_name: str | None
    operator: str
    value: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    aggregation: str = "single"
    output_names: list[str] | None = None

    def __post_init__(self) -> None:
        self.operator = _name(self.operator, "Constraint operator").lower()
        if self.operator not in SUPPORTED_CONSTRAINT_OPERATORS:
            raise ValueError(
                "Constraint operator must be greater_than_or_equal, "
                "less_than_or_equal, or within_range."
            )
        if self.operator == "within_range":
            if self.lower_bound is None or self.upper_bound is None:
                raise ValueError(
                    "A within-range constraint requires lower and upper values."
                )
            self.lower_bound = _finite_number(
                self.lower_bound,
                "Constraint lower value",
            )
            self.upper_bound = _finite_number(
                self.upper_bound,
                "Constraint upper value",
            )
            if self.lower_bound > self.upper_bound:
                raise ValueError(
                    "Constraint lower value cannot exceed its upper value."
                )
            if self.value is not None:
                raise ValueError(
                    "A within-range constraint cannot also use a threshold value."
                )
        else:
            if self.value is None:
                raise ValueError("This constraint requires a threshold value.")
            self.value = _finite_number(self.value, "Constraint threshold")
            if self.lower_bound is not None or self.upper_bound is not None:
                raise ValueError(
                    "A threshold constraint cannot also use lower or upper values."
                )
        self.aggregation = _name(
            self.aggregation,
            "Constraint aggregation",
        ).lower()
        if self.aggregation not in SUPPORTED_OBJECTIVE_AGGREGATIONS:
            raise ValueError("Constraint aggregation must be single or mean.")
        normalized_names = [
            _name(value, "Constraint output name")
            for value in list(self.output_names or [])
        ]
        duplicates = [
            name
            for index, name in enumerate(normalized_names)
            if name in normalized_names[:index]
        ]
        if duplicates:
            raise ValueError(
                "Duplicate constraint outputs were provided: "
                + ", ".join(dict.fromkeys(duplicates))
                + "."
            )
        if self.aggregation == "single":
            if self.output_name is None and len(normalized_names) == 1:
                self.output_name = normalized_names[0]
            self.output_name = _name(self.output_name, "Constraint output")
            if normalized_names and normalized_names != [self.output_name]:
                raise ValueError(
                    "A single-output constraint must identify exactly one output."
                )
            self.output_names = [self.output_name]
        else:
            if self.output_name is not None:
                raise ValueError(
                    "A mean constraint must use output_names instead of output_name."
                )
            if len(normalized_names) < 2:
                raise ValueError(
                    "A mean constraint requires at least two ordered outputs."
                )
            self.output_names = normalized_names

    @property
    def selected_outputs(self) -> list[str]:
        return list(self.output_names or [])

    @property
    def display_name(self) -> str:
        if self.aggregation == "single":
            return str(self.output_name)
        return f"Mean of {self.output_names[0]} through {self.output_names[-1]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_name": self.output_name,
            "output_names": self.selected_outputs,
            "aggregation": self.aggregation,
            "operator": self.operator,
            "value": self.value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }


@dataclass(slots=True)
class InverseDesignRequest:
    """Validated, model-independent configuration for one design search."""

    variable_bounds: dict[str, tuple[float, float]]
    fixed_inputs: dict[str, float]
    objective: InverseDesignObjective
    constraints: list[OutputConstraint] = field(default_factory=list)
    model_book_id: str | None = None
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    population_multiplier: int = DEFAULT_POPULATION_MULTIPLIER
    random_seed: int = DEFAULT_RANDOM_SEED

    def __post_init__(self) -> None:
        if not isinstance(self.variable_bounds, dict):
            raise ValueError("Variable bounds must be provided as a dictionary.")
        if not self.variable_bounds:
            raise ValueError("At least one input must be allowed to vary.")
        if not isinstance(self.fixed_inputs, dict):
            raise ValueError("Fixed inputs must be provided as a dictionary.")
        if not isinstance(self.objective, InverseDesignObjective):
            raise ValueError("A validated inverse-design objective is required.")
        if not isinstance(self.constraints, list) or any(
            not isinstance(item, OutputConstraint) for item in self.constraints
        ):
            raise ValueError(
                "Constraints must be provided as OutputConstraint objects."
            )

        normalized_bounds: dict[str, tuple[float, float]] = {}
        for raw_name, raw_bounds in self.variable_bounds.items():
            feature_name = _name(raw_name, "Variable input name")
            if (
                not isinstance(raw_bounds, (tuple, list))
                or len(raw_bounds) != 2
            ):
                raise ValueError(
                    f"Variable '{feature_name}' requires lower and upper bounds."
                )
            lower = _finite_number(raw_bounds[0], f"Lower bound for '{feature_name}'")
            upper = _finite_number(raw_bounds[1], f"Upper bound for '{feature_name}'")
            if lower >= upper:
                raise ValueError(
                    f"Lower bound for '{feature_name}' must be less than its upper bound."
                )
            normalized_bounds[feature_name] = (lower, upper)

        normalized_fixed: dict[str, float] = {}
        for raw_name, raw_value in self.fixed_inputs.items():
            feature_name = _name(raw_name, "Fixed input name")
            normalized_fixed[feature_name] = _finite_number(
                raw_value,
                f"Fixed value for '{feature_name}'",
            )
        overlap = [
            name for name in normalized_bounds if name in normalized_fixed
        ]
        if overlap:
            raise ValueError(
                "An input cannot be both variable and fixed: "
                + ", ".join(overlap)
                + "."
            )

        if self.model_book_id is not None:
            self.model_book_id = _name(self.model_book_id, "Model Book ID")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or not 1 <= self.max_iterations <= 10_000
        ):
            raise ValueError("Maximum iterations must be an integer from 1 to 10000.")
        if (
            isinstance(self.population_multiplier, bool)
            or not isinstance(self.population_multiplier, int)
            or not 4 <= self.population_multiplier <= 100
        ):
            raise ValueError(
                "Population multiplier must be an integer from 4 to 100."
            )
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("Random seed must be an integer.")

        self.variable_bounds = normalized_bounds
        self.fixed_inputs = normalized_fixed
        self.constraints = list(self.constraints)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_book_id": self.model_book_id,
            "variable_bounds": {
                name: {"lower": bounds[0], "upper": bounds[1]}
                for name, bounds in self.variable_bounds.items()
            },
            "fixed_inputs": dict(self.fixed_inputs),
            "objective": self.objective.to_dict(),
            "constraints": [item.to_dict() for item in self.constraints],
            "optimizer": {
                "name": "differential_evolution",
                "max_iterations": self.max_iterations,
                "population_multiplier": self.population_multiplier,
                "random_seed": self.random_seed,
            },
        }


@dataclass(slots=True)
class ConstraintEvaluation:
    output_name: str
    operator: str
    predicted_value: float
    satisfied: bool
    violation: float
    value: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    aggregation: str = "single"
    output_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_name": self.output_name,
            "output_names": list(self.output_names),
            "aggregation": self.aggregation,
            "operator": self.operator,
            "predicted_value": self.predicted_value,
            "satisfied": self.satisfied,
            "violation": self.violation,
            "value": self.value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }


@dataclass(slots=True)
class InverseDesignResult:
    """Structured best-design result plus immutable artifact locations."""

    success: bool
    status: str
    model_book_id: str | None = None
    model_book_name: str | None = None
    model_name: str | None = None
    run_id: str | None = None
    best_inputs: dict[str, float] = field(default_factory=dict)
    predicted_outputs: dict[str, float] = field(default_factory=dict)
    objective: dict[str, Any] = field(default_factory=dict)
    objective_value: float | None = None
    objective_score: float | None = None
    target_gap: float | None = None
    constraint_evaluations: list[dict[str, Any]] = field(default_factory=list)
    feasible: bool = False
    evaluations: int = 0
    iterations: int = 0
    optimizer_message: str | None = None
    artifact_directory: Path | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": INVERSE_DESIGN_SCHEMA_VERSION,
            "success": self.success,
            "status": self.status,
            "run_id": self.run_id,
            "model_book_id": self.model_book_id,
            "model_book_name": self.model_book_name,
            "model_name": self.model_name,
            "best_inputs": dict(self.best_inputs),
            "predicted_outputs": dict(self.predicted_outputs),
            "objective": dict(self.objective),
            "objective_value": self.objective_value,
            "objective_score": self.objective_score,
            "target_gap": self.target_gap,
            "constraint_evaluations": list(self.constraint_evaluations),
            "feasible": self.feasible,
            "evaluations": self.evaluations,
            "iterations": self.iterations,
            "optimizer_message": self.optimizer_message,
            "artifact_directory": (
                str(self.artifact_directory)
                if self.artifact_directory is not None
                else None
            ),
            "error_message": self.error_message,
        }


@dataclass(slots=True)
class _Evaluation:
    inputs: dict[str, float]
    outputs: dict[str, float]
    objective_score: float
    constraint_evaluations: list[ConstraintEvaluation]

    @property
    def feasible(self) -> bool:
        return all(item.satisfied for item in self.constraint_evaluations)


class _SurrogateObjectiveEvaluator:
    """Translate optimizer vectors into generic Model Book evaluations."""

    def __init__(
        self,
        predictor: ModelBookPredictor,
        request: InverseDesignRequest,
    ):
        self.predictor = predictor
        self.request = request
        self.variable_names = list(request.variable_bounds)
        self.cache: dict[tuple[float, ...], _Evaluation] = {}
        self.failures: dict[tuple[float, ...], str] = {}

    def _inputs(self, vector: np.ndarray) -> dict[str, float]:
        values = dict(self.request.fixed_inputs)
        values.update(
            {
                name: float(value)
                for name, value in zip(self.variable_names, vector, strict=True)
            }
        )
        return {
            name: values[name]
            for name in self.predictor.book.feature_columns
        }

    def evaluate(self, vector: np.ndarray) -> _Evaluation:
        key = tuple(float(value) for value in vector)
        if key in self.cache:
            return self.cache[key]
        if key in self.failures:
            raise InverseDesignError(self.failures[key])
        inputs = self._inputs(vector)
        try:
            outputs = self.predictor.predict(inputs)
        except (InferenceError, ModelBookError, OSError, ValueError) as exc:
            self.failures[key] = str(exc)
            raise InverseDesignError(str(exc)) from exc
        objective_output = _aggregated_objective_value(
            outputs,
            self.request.objective,
        )
        objective_score = _objective_score(
            objective_output,
            self.request.objective,
        )
        constraints = [
            _constraint_evaluation(
                _aggregated_constraint_value(outputs, item),
                item,
            )
            for item in self.request.constraints
        ]
        evaluation = _Evaluation(
            inputs=inputs,
            outputs=outputs,
            objective_score=objective_score,
            constraint_evaluations=constraints,
        )
        self.cache[key] = evaluation
        return evaluation

    def objective(self, vector: np.ndarray) -> float:
        try:
            return self.evaluate(vector).objective_score
        except InverseDesignError:
            return 1.0e100

    def constraint_value(self, vector: np.ndarray, index: int) -> float:
        constraint = self.request.constraints[index]
        try:
            return _aggregated_constraint_value(
                self.evaluate(vector).outputs,
                constraint,
            )
        except InverseDesignError:
            return (
                -1.0e100
                if constraint.operator == "greater_than_or_equal"
                else 1.0e100
            )


def submit_inverse_design_request(
    request: InverseDesignRequest,
    *,
    project_path: str | Path | None = None,
    predictor_factory: Callable[
        [str | Path, str | None], ModelBookPredictor
    ] | None = None,
) -> InverseDesignResult:
    """Run one deterministic Differential Evolution search."""

    if not isinstance(request, InverseDesignRequest):
        raise TypeError("A validated InverseDesignRequest is required.")
    if project_path is None:
        return _failed(
            request.model_book_id,
            "Inverse design requires an open Antenna Surrogate Studio project.",
        )
    try:
        validated = InverseDesignRequest(
            model_book_id=request.model_book_id,
            variable_bounds=dict(request.variable_bounds),
            fixed_inputs=dict(request.fixed_inputs),
            objective=InverseDesignObjective(**request.objective.to_dict()),
            constraints=[
                OutputConstraint(**item.to_dict()) for item in request.constraints
            ],
            max_iterations=request.max_iterations,
            population_multiplier=request.population_multiplier,
            random_seed=request.random_seed,
        )
    except (TypeError, ValueError) as exc:
        return _failed(request.model_book_id, str(exc))

    try:
        factory = predictor_factory or ModelBookPredictor.load_active
        predictor = factory(project_path, validated.model_book_id)
        _validate_against_book(validated, predictor)
        from scipy.optimize import NonlinearConstraint, differential_evolution

        evaluator = _SurrogateObjectiveEvaluator(predictor, validated)
        nonlinear_constraints = tuple(
            NonlinearConstraint(
                lambda vector, index=index: evaluator.constraint_value(
                    vector,
                    index,
                ),
                *_constraint_bounds(constraint),
            )
            for index, constraint in enumerate(validated.constraints)
        )
        with warnings.catch_warnings():
            # SciPy's generic constrained-polish warnings are expected for
            # linear surrogates and infeasible searches. The explicit
            # feasibility check below remains the user-facing authority.
            warnings.filterwarnings(
                "ignore",
                message=r"differential evolution didn't find a solution.*",
                category=UserWarning,
                module=r"scipy\.optimize\._differentialevolution",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"delta_grad == 0\.0.*",
                category=UserWarning,
                module=r"scipy\.optimize\._differentiable_functions",
            )
            optimizer_result = differential_evolution(
                evaluator.objective,
                bounds=list(validated.variable_bounds.values()),
                constraints=nonlinear_constraints,
                maxiter=validated.max_iterations,
                popsize=validated.population_multiplier,
                seed=validated.random_seed,
                polish=True,
                workers=1,
                updating="immediate",
            )
        if not evaluator.cache:
            raise InverseDesignError(
                "All candidate designs failed during surrogate prediction."
            )
        final_evaluation = evaluator.evaluate(np.asarray(optimizer_result.x))
        if not final_evaluation.feasible:
            raise InverseDesignError(
                "No design satisfying all output constraints was found within "
                "the configured input bounds and search budget."
            )
        objective_value = _aggregated_objective_value(
            final_evaluation.outputs,
            validated.objective,
        )
        target_gap = (
            abs(objective_value - float(validated.objective.target_value))
            if validated.objective.goal == "target"
            else None
        )
        result = InverseDesignResult(
            success=True,
            status=INVERSE_DESIGN_COMPLETED,
            model_book_id=predictor.book.book_id,
            model_book_name=predictor.book.name,
            model_name=predictor.book.model_name,
            best_inputs=final_evaluation.inputs,
            predicted_outputs=final_evaluation.outputs,
            objective=validated.objective.to_dict(),
            objective_value=objective_value,
            objective_score=final_evaluation.objective_score,
            target_gap=target_gap,
            constraint_evaluations=[
                item.to_dict()
                for item in final_evaluation.constraint_evaluations
            ],
            feasible=True,
            evaluations=int(getattr(optimizer_result, "nfev", len(evaluator.cache))),
            iterations=int(getattr(optimizer_result, "nit", 0)),
            optimizer_message=str(getattr(optimizer_result, "message", "")),
        )
        _save_completed_run(
            Path(project_path).resolve(),
            validated,
            predictor,
            result,
            evaluator,
        )
        return result
    except ImportError:
        return _failed(
            validated.model_book_id,
            "Inverse-design dependencies are not installed. Run the Studio "
            "installer to add SciPy.",
        )
    except (InverseDesignError, InferenceError, ModelBookError, OSError) as exc:
        return _failed(validated.model_book_id, str(exc))
    except Exception:
        return _failed(
            validated.model_book_id,
            "Inverse design failed because an unexpected local error occurred.",
        )


def load_inverse_design_run(
    project_path: str | Path,
    run_id: str | None = None,
) -> dict[str, Any] | None:
    """Load a saved result, or the latest result when no ID is provided."""

    project_root = Path(project_path).expanduser().resolve()
    index_path = project_root / "inverse_design" / "index.json"
    if not index_path.exists():
        return None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InverseDesignError(
            "The inverse-design run index is malformed or unreadable."
        ) from exc
    selected = str(run_id or index.get("latest_run_id") or "").strip()
    if not selected:
        return None
    if not selected.startswith("inverse-") or not selected[8:].isdigit():
        raise InverseDesignError("The inverse-design run ID is invalid.")
    result_path = project_root / "inverse_design" / "runs" / selected / "result.json"
    if not result_path.exists():
        raise InverseDesignError(
            f"Inverse-design result '{selected}' is missing."
        )
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InverseDesignError(
            f"Inverse-design result '{selected}' is malformed or unreadable."
        ) from exc
    if not isinstance(payload, dict) or payload.get("run_id") != selected:
        raise InverseDesignError(
            f"Inverse-design result '{selected}' has invalid metadata."
        )
    return payload


def load_inverse_design_runs(
    project_path: str | Path,
    *,
    model_book_id: str | None = None,
) -> InverseDesignHistory:
    """Load every valid completed search in chronological index order.

    A damaged individual result is skipped and reported without hiding the
    remaining project history. Results from another Model Book are filtered out.
    """

    project_root = Path(project_path).expanduser().resolve()
    index_path = project_root / "inverse_design" / "index.json"
    if not index_path.exists():
        return InverseDesignHistory()
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InverseDesignError(
            "The inverse-design run index is malformed or unreadable."
        ) from exc
    if not isinstance(index, dict) or not isinstance(index.get("runs"), list):
        raise InverseDesignError("The inverse-design run index has invalid metadata.")

    history = InverseDesignHistory()
    seen: set[str] = set()
    for raw_entry in index["runs"]:
        if not isinstance(raw_entry, dict):
            history.errors.append(
                "An inverse-design history entry has invalid metadata."
            )
            continue
        run_id = str(raw_entry.get("run_id") or "").strip()
        if (
            not run_id.startswith("inverse-")
            or not run_id[8:].isdigit()
            or run_id in seen
        ):
            history.errors.append(
                f"Inverse-design history entry '{run_id or 'unknown'}' has an invalid run ID."
            )
            continue
        seen.add(run_id)
        try:
            payload = load_inverse_design_run(project_root, run_id)
        except InverseDesignError:
            history.errors.append(
                f"Inverse-design run '{run_id}' is malformed or unreadable."
            )
            continue
        if payload is None:
            history.errors.append(f"Inverse-design run '{run_id}' is missing.")
            continue
        entry_book_id = str(raw_entry.get("model_book_id") or "").strip()
        payload_book_id = str(payload.get("model_book_id") or "").strip()
        if entry_book_id and entry_book_id != payload_book_id:
            history.errors.append(
                f"Inverse-design run '{run_id}' does not match its history entry."
            )
            continue
        if (
            payload.get("success") is not True
            or payload.get("status") != INVERSE_DESIGN_COMPLETED
            or not payload_book_id
        ):
            history.errors.append(
                f"Inverse-design run '{run_id}' is not a valid completed result."
            )
            continue
        if model_book_id is not None and payload_book_id != model_book_id:
            continue
        history.runs.append(payload)
    return history


def _validate_against_book(
    request: InverseDesignRequest,
    predictor: ModelBookPredictor,
) -> None:
    features = list(predictor.book.feature_columns)
    supplied = [*request.variable_bounds, *request.fixed_inputs]
    missing = [name for name in features if name not in supplied]
    unexpected = [name for name in supplied if name not in features]
    if missing:
        raise InverseDesignError(
            "Every required Model Book input must be variable or fixed. Missing: "
            + ", ".join(missing)
            + "."
        )
    if unexpected:
        raise InverseDesignError(
            "Unknown Model Book input"
            f"{'s' if len(unexpected) != 1 else ''}: {', '.join(unexpected)}."
        )
    outputs = set(predictor.book.target_columns)
    invalid_objective_outputs = [
        name for name in request.objective.selected_outputs if name not in outputs
    ]
    if invalid_objective_outputs:
        raise InverseDesignError(
            "Objective output"
            f"{'s' if len(invalid_objective_outputs) != 1 else ''} not saved in "
            "the active Model Book: "
            + ", ".join(invalid_objective_outputs)
            + "."
        )
    invalid_constraints = [
        name
        for item in request.constraints
        for name in item.selected_outputs
        if name not in outputs
    ]
    if invalid_constraints:
        raise InverseDesignError(
            "Constraint output"
            f"{'s' if len(invalid_constraints) != 1 else ''} not saved in the "
            f"active Model Book: {', '.join(invalid_constraints)}."
        )


def _objective_score(value: float, objective: InverseDesignObjective) -> float:
    if objective.goal == "minimize":
        return value
    if objective.goal == "maximize":
        return -value
    return abs(value - float(objective.target_value))


def _aggregated_objective_value(
    outputs: dict[str, float],
    objective: InverseDesignObjective,
) -> float:
    values = [float(outputs[name]) for name in objective.selected_outputs]
    if objective.aggregation == "single":
        return values[0]
    return float(np.mean(values))


def _aggregated_constraint_value(
    outputs: dict[str, float],
    constraint: OutputConstraint,
) -> float:
    values = [float(outputs[name]) for name in constraint.selected_outputs]
    if constraint.aggregation == "single":
        return values[0]
    return float(np.mean(values))


def _constraint_bounds(constraint: OutputConstraint) -> tuple[float, float]:
    if constraint.operator == "greater_than_or_equal":
        return float(constraint.value), np.inf
    if constraint.operator == "less_than_or_equal":
        return -np.inf, float(constraint.value)
    return float(constraint.lower_bound), float(constraint.upper_bound)


def _constraint_evaluation(
    predicted: float,
    constraint: OutputConstraint,
) -> ConstraintEvaluation:
    if constraint.operator == "greater_than_or_equal":
        violation = max(0.0, float(constraint.value) - predicted)
    elif constraint.operator == "less_than_or_equal":
        violation = max(0.0, predicted - float(constraint.value))
    else:
        violation = max(
            0.0,
            float(constraint.lower_bound) - predicted,
            predicted - float(constraint.upper_bound),
        )
    scale = max(
        1.0,
        abs(predicted),
        abs(float(constraint.value or 0.0)),
        abs(float(constraint.lower_bound or 0.0)),
        abs(float(constraint.upper_bound or 0.0)),
    )
    return ConstraintEvaluation(
        output_name=constraint.display_name,
        operator=constraint.operator,
        predicted_value=predicted,
        satisfied=violation <= 1.0e-8 * scale,
        violation=violation,
        value=constraint.value,
        lower_bound=constraint.lower_bound,
        upper_bound=constraint.upper_bound,
        aggregation=constraint.aggregation,
        output_names=constraint.selected_outputs,
    )


def _save_completed_run(
    project_root: Path,
    request: InverseDesignRequest,
    predictor: ModelBookPredictor,
    result: InverseDesignResult,
    evaluator: _SurrogateObjectiveEvaluator,
) -> None:
    runs_root = project_root / "inverse_design" / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_number = 1
    while (runs_root / f"inverse-{run_number:04d}").exists():
        run_number += 1
    run_id = f"inverse-{run_number:04d}"
    created_at = utc_now()
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs_root))
    destination = runs_root / run_id
    result.run_id = run_id
    result.artifact_directory = destination
    try:
        atomic_write_json(
            staging / "request.json",
            {
                "schema_version": INVERSE_DESIGN_SCHEMA_VERSION,
                "created_at": created_at,
                **request.to_dict(),
            },
        )
        result_payload = result.to_dict()
        result_payload["artifact_directory"] = f"inverse_design/runs/{run_id}"
        result_payload.update(
            {
                "created_at": created_at,
                "model_book": {
                    "book_id": predictor.book.book_id,
                    "name": predictor.book.name,
                    "model_name": predictor.book.model_name,
                    "dataset_fingerprint": predictor.book.dataset_fingerprint,
                },
                "optimizer": request.to_dict()["optimizer"],
                "unique_candidate_count": len(evaluator.cache),
                "failed_candidate_count": len(evaluator.failures),
            }
        )
        atomic_write_json(staging / "result.json", result_payload)
        with (staging / "best_prediction.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["Output variable", "Predicted value"])
            writer.writerows(result.predicted_outputs.items())
        with (staging / "evaluation_trace.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    *evaluator.variable_names,
                    "objective_score",
                    "feasible",
                    "total_constraint_violation",
                ]
            )
            for vector, evaluation in evaluator.cache.items():
                writer.writerow(
                    [
                        *vector,
                        evaluation.objective_score,
                        evaluation.feasible,
                        sum(
                            item.violation
                            for item in evaluation.constraint_evaluations
                        ),
                    ]
                )
        os.replace(staging, destination)
    except OSError as exc:
        raise InverseDesignError(
            f"The inverse-design result could not be saved: {exc}"
        ) from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    index_path = project_root / "inverse_design" / "index.json"
    index: dict[str, Any] = {
        "schema_version": INVERSE_DESIGN_SCHEMA_VERSION,
        "latest_run_id": None,
        "runs": [],
    }
    if index_path.exists():
        try:
            loaded = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InverseDesignError(
                "The inverse-design result was saved, but its run index is invalid."
            ) from exc
        if not isinstance(loaded, dict) or not isinstance(loaded.get("runs", []), list):
            raise InverseDesignError(
                "The inverse-design result was saved, but its run index is invalid."
            )
        index = loaded
    index["latest_run_id"] = run_id
    index["runs"] = [
        *index.get("runs", []),
        {
            "run_id": run_id,
            "created_at": created_at,
            "model_book_id": predictor.book.book_id,
            "model_book_name": predictor.book.name,
            "objective": request.objective.to_dict(),
            "result": f"runs/{run_id}/result.json",
        },
    ]
    atomic_write_json(index_path, index)

    manifest_path = project_root / "project.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InverseDesignError(
            "The inverse-design result was saved, but project state could not be updated."
        ) from exc
    manifest["inverse_design"] = {
        "schema_version": INVERSE_DESIGN_SCHEMA_VERSION,
        "run_count": len(index["runs"]),
        "latest_run_id": run_id,
        "index": "inverse_design/index.json",
    }
    manifest["updated_at"] = created_at
    atomic_write_json(manifest_path, manifest)


def _failed(model_book_id: str | None, message: str) -> InverseDesignResult:
    return InverseDesignResult(
        success=False,
        status=INVERSE_DESIGN_FAILED,
        model_book_id=model_book_id,
        error_message=message,
    )
