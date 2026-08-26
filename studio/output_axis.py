"""Structured, deterministic metadata for ordered surrogate-model outputs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable


_AXIS_NUMBER = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
_EXPLICIT_AXIS = re.compile(
    rf"\bat\s+(.+?)\s+({_AXIS_NUMBER.pattern})"
    r"(?:\s+([^\s]+))?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class OutputAxisMetadata:
    """Describe the ordered coordinate associated with a model's outputs."""

    label: str
    unit: str | None
    values: tuple[float, ...]
    source: str

    @property
    def display_label(self) -> str:
        return f"{self.label} ({self.unit})" if self.unit else self.label

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "unit": self.unit,
            "values": list(self.values),
            "source": self.source,
        }


def infer_output_axis(target_columns: Iterable[str]) -> OutputAxisMetadata:
    """Infer only reliable meaning and preserve target-column order."""

    targets = tuple(str(name) for name in target_columns)
    explicit = [_EXPLICIT_AXIS.search(target) for target in targets]
    if targets and all(match is not None for match in explicit):
        labels = tuple(match.group(1).strip() for match in explicit if match)
        units = tuple((match.group(3) or "").strip() for match in explicit if match)
        values = tuple(float(match.group(2)) for match in explicit if match)
        if (
            len({label.casefold() for label in labels}) == 1
            and len({unit.casefold() for unit in units}) == 1
            and len(set(values)) == len(values)
        ):
            return OutputAxisMetadata(
                label=labels[0],
                unit=units[0] or None,
                values=values,
                source="target_columns",
            )

    joined = " ".join(targets).lower()
    if "freq" in joined:
        label = "Frequency"
        unit = next(
            (
                displayed
                for token, displayed in (
                    ("ghz", "GHz"),
                    ("mhz", "MHz"),
                    ("khz", "kHz"),
                    ("hz", "Hz"),
                )
                if token in joined
            ),
            None,
        )
    elif "theta" in joined:
        label = "Theta"
        unit = "deg" if any(token in joined for token in ("deg", "degree", "°")) else None
    elif "phi" in joined:
        label = "Phi"
        unit = "deg" if any(token in joined for token in ("deg", "degree", "°")) else None
    else:
        label = "Output coordinate"
        unit = None

    inferred: list[float] = []
    for target in targets:
        matches = _AXIS_NUMBER.findall(target)
        if not matches:
            inferred = []
            break
        inferred.append(float(matches[-1]))
    if len(inferred) == len(targets) and len(set(inferred)) == len(inferred):
        values = tuple(inferred)
        source = "target_columns"
    else:
        values = tuple(float(index) for index in range(1, len(targets) + 1))
        source = "output_index"
    return OutputAxisMetadata(label=label, unit=unit, values=values, source=source)


def output_axis_from_dict(
    value: Any,
    target_columns: Iterable[str],
) -> OutputAxisMetadata:
    """Validate saved metadata, deriving it for backward-compatible books."""

    targets = tuple(str(name) for name in target_columns)
    if value is None:
        return infer_output_axis(targets)
    if not isinstance(value, dict):
        raise ValueError("The Model Book output-axis metadata is invalid.")
    label = str(value.get("label") or "").strip()
    raw_unit = value.get("unit")
    unit = str(raw_unit).strip() if raw_unit is not None else None
    source = str(value.get("source") or "").strip()
    raw_values = value.get("values")
    if not label or not source or not isinstance(raw_values, list):
        raise ValueError("The Model Book output-axis metadata is incomplete.")
    try:
        values = tuple(float(item) for item in raw_values)
    except (TypeError, ValueError) as exc:
        raise ValueError("The Model Book output-axis values must be numeric.") from exc
    if len(values) != len(targets):
        raise ValueError("The Model Book output axis does not match its saved outputs.")
    if not all(math.isfinite(item) for item in values):
        raise ValueError("The Model Book output-axis values must be finite.")
    if len(set(values)) != len(values):
        raise ValueError("The Model Book output-axis values must be unique.")
    return OutputAxisMetadata(label=label, unit=unit or None, values=values, source=source)
