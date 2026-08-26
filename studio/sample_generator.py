"""Generic deterministic Latin Hypercube input-sample generation."""

from __future__ import annotations

import csv
import math
import os
import tempfile
from dataclasses import dataclass
from numbers import Real
from pathlib import Path


MAX_LHS_SAMPLES = 100_000
MAX_RANDOM_SEED = 2**32 - 1


def _variable_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Each LHS variable requires a non-empty name.")
    name = value.strip()
    if "\n" in name or "\r" in name:
        raise ValueError("LHS variable names cannot contain line breaks.")
    normalized = name.casefold().replace("_", " ").replace("-", " ")
    if " ".join(normalized.split()) == "sample id":
        raise ValueError(
            "The variable name 'sample_id' is reserved by the Studio CSV format."
        )
    return name


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a numeric value.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite numeric value.")
    return number


@dataclass(frozen=True, slots=True)
class LHSVariable:
    """One user-named simulation input and its allowed numeric range."""

    name: str
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        name = _variable_name(self.name)
        minimum = _finite_number(self.minimum, f"Minimum for '{name}'")
        maximum = _finite_number(self.maximum, f"Maximum for '{name}'")
        if minimum >= maximum:
            raise ValueError(
                f"Minimum for '{name}' must be less than its maximum."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)


@dataclass(slots=True)
class LHSSampleGenerationRequest:
    """Validated configuration for one Latin Hypercube sample set."""

    variables: list[LHSVariable]
    sample_count: int
    random_seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.variables, list) or not self.variables:
            raise ValueError("Add at least one LHS variable before generating samples.")
        if any(not isinstance(item, LHSVariable) for item in self.variables):
            raise ValueError("LHS variables must use the validated LHSVariable contract.")
        normalized = [item.name.casefold() for item in self.variables]
        duplicates = [
            item.name
            for index, item in enumerate(self.variables)
            if normalized[index] in normalized[:index]
        ]
        if duplicates:
            raise ValueError(
                "Duplicate LHS variable names were provided: "
                + ", ".join(dict.fromkeys(duplicates))
                + "."
            )
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or not 1 <= self.sample_count <= MAX_LHS_SAMPLES
        ):
            raise ValueError(
                f"Sample count must be an integer from 1 to {MAX_LHS_SAMPLES:,}."
            )
        if self.random_seed is not None and (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or not 0 <= self.random_seed <= MAX_RANDOM_SEED
        ):
            raise ValueError(
                f"Random seed must be an integer from 0 to {MAX_RANDOM_SEED:,}, "
                "or left blank."
            )
        self.variables = list(self.variables)


@dataclass(slots=True)
class LHSSampleSet:
    """Generated numeric rows in exact variable order."""

    variable_names: list[str]
    rows: list[list[float]]
    random_seed: int | None

    @property
    def sample_count(self) -> int:
        return len(self.rows)


def generate_lhs_samples(request: LHSSampleGenerationRequest) -> LHSSampleSet:
    """Generate one SciPy Latin Hypercube sample set."""

    if not isinstance(request, LHSSampleGenerationRequest):
        raise TypeError("A validated LHSSampleGenerationRequest is required.")
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(
        d=len(request.variables),
        scramble=True,
        seed=request.random_seed,
    )
    unit_samples = sampler.random(n=request.sample_count)
    scaled = qmc.scale(
        unit_samples,
        [item.minimum for item in request.variables],
        [item.maximum for item in request.variables],
    )
    return LHSSampleSet(
        variable_names=[item.name for item in request.variables],
        rows=[[float(value) for value in row] for row in scaled.tolist()],
        random_seed=request.random_seed,
    )


def write_lhs_inputs_csv(
    destination: str | Path,
    sample_set: LHSSampleSet,
) -> Path:
    """Atomically save a generated sample set as a Studio-compatible input CSV."""

    if not isinstance(sample_set, LHSSampleSet) or not sample_set.rows:
        raise ValueError("Generate at least one LHS sample before exporting.")
    expected_columns = len(sample_set.variable_names)
    if not sample_set.variable_names or any(
        len(row) != expected_columns for row in sample_set.rows
    ):
        raise ValueError("Generated LHS rows do not match the variable columns.")
    path = Path(destination).expanduser()
    if path.suffix.lower() != ".csv":
        raise ValueError("LHS input samples must be exported as a .csv file.")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(handle.name)
    try:
        with handle:
            writer = csv.writer(handle)
            writer.writerow(sample_set.variable_names)
            for row in sample_set.rows:
                writer.writerow(format(float(value), ".15g") for value in row)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path.resolve()
