"""Provider-neutral results and helpers for read-only antenna analysis tools."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from studio.antenna_design import AntennaDesign
from studio.antenna_engineering import EngineeringDesignRef, _redact
from studio.antenna_geometry import evaluate_geometry, transformed_point


ANALYSIS_RESULT_SCHEMA_VERSION = 1
SPEED_OF_LIGHT_MM_GHZ = 299.792458
_TRANSIENT_DESIGN_KEYS = frozenset({
    "design_id", "revision", "validation", "validation_timestamp", "timestamp",
    "timestamp_utc", "audit", "audit_metadata",
})


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _semantic_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_value(item)
            for key, item in value.items()
            if str(key).casefold() not in _TRANSIENT_DESIGN_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_semantic_value(item) for item in value]
    return value


def semantic_design_payload(design: AntennaDesign | None) -> Any:
    """Return canonical engineering content without identity or validation bookkeeping."""

    if design is None:
        return None
    payload = design.to_dict()
    metadata = payload.get("metadata", ())
    if isinstance(metadata, (list, tuple)):
        payload["metadata"] = [
            item for item in metadata
            if not (
                isinstance(item, (list, tuple)) and item
                and str(item[0]).casefold() in _TRANSIENT_DESIGN_KEYS
            )
        ]
    return _semantic_value(payload)


def semantic_design_hash(design: AntennaDesign | None) -> str:
    return stable_hash(semantic_design_payload(design))


def normalize_analysis_arguments(value: Any) -> Any:
    """Canonicalize JSON-like analysis arguments without changing their meaning."""

    if isinstance(value, Mapping):
        return {str(key): normalize_analysis_arguments(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_analysis_arguments(item) for item in value]
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return value


def stable_analysis_id(tool_name: str, tool_version: str, arguments: Mapping[str, Any], design_hash: str) -> str:
    return "analysis-v1-" + stable_hash({
        "tool_name": tool_name,
        "tool_version": tool_version,
        "arguments": normalize_analysis_arguments(arguments),
        "semantic_design_hash": design_hash,
    })


@dataclass(frozen=True, slots=True)
class AnalysisMeasurement:
    value: float
    unit: str
    tolerance: float | None = None
    uncertainty: float | None = None

    def __post_init__(self) -> None:
        for name in ("value", "tolerance", "uncertainty"):
            value = getattr(self, name)
            if value is None and name != "value":
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Analysis measurement {name} must be finite and numeric.")
            if name != "value" and value < 0:
                raise ValueError(f"Analysis measurement {name} must be nonnegative.")
        if not isinstance(self.unit, str):
            raise ValueError("Analysis measurement unit must be a string.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "tolerance": self.tolerance,
            "uncertainty": self.uncertainty,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnalysisMeasurement":
        expected = {"value", "unit", "tolerance", "uncertainty"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("AnalysisMeasurement does not match the required schema.")
        return cls(payload["value"], payload["unit"], payload["tolerance"], payload["uncertainty"])


@dataclass(frozen=True, slots=True)
class EngineeringAnalysisResult:
    tool_name: str
    tool_version: str
    analysis_id: str
    working_design_ref: EngineeringDesignRef
    semantic_design_hash: str
    status: str
    measurements: Mapping[str, AnalysisMeasurement] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    applicability: str = ""
    limitations: tuple[str, ...] = ()
    message: str = ""
    schema_version: int = ANALYSIS_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("tool_name", "tool_version", "analysis_id", "semantic_design_hash", "applicability", "message"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string.")
        if self.schema_version != ANALYSIS_RESULT_SCHEMA_VERSION:
            raise ValueError("Unsupported engineering analysis result schema version.")
        if re.fullmatch(r"analysis-v1-[0-9a-f]{64}", self.analysis_id) is None:
            raise ValueError("analysis_id must be a deterministic analysis-v1 identifier.")
        if re.fullmatch(r"[0-9a-f]{64}", self.semantic_design_hash) is None:
            raise ValueError("semantic_design_hash must be a lowercase SHA-256 digest.")
        if not isinstance(self.working_design_ref, EngineeringDesignRef):
            raise ValueError("working_design_ref must identify the analyzed design revision.")
        if self.status not in {"completed", "not_applicable", "unknown", "failed"}:
            raise ValueError("Invalid engineering analysis status.")
        if not isinstance(self.measurements, Mapping):
            raise ValueError("measurements must be a mapping.")
        for key, value in self.measurements.items():
            if not isinstance(key, str) or not key.strip() or not isinstance(value, AnalysisMeasurement):
                raise ValueError("measurements must map nonempty names to AnalysisMeasurement values.")
        object.__setattr__(self, "measurements", MappingProxyType(dict(self.measurements)))
        for name in ("assumptions", "limitations"):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)) or any(not isinstance(item, str) or not item.strip() for item in values):
                raise ValueError(f"{name} must contain nonempty strings.")
            object.__setattr__(self, name, tuple(values))

    def to_dict(self) -> dict[str, Any]:
        return _redact({
            "schema_version": self.schema_version,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "analysis_id": self.analysis_id,
            "working_design_ref": self.working_design_ref.to_dict(),
            "semantic_design_hash": self.semantic_design_hash,
            "status": self.status,
            "measurements": {key: value.to_dict() for key, value in self.measurements.items()},
            "assumptions": list(self.assumptions),
            "applicability": self.applicability,
            "limitations": list(self.limitations),
            "message": self.message,
        })

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EngineeringAnalysisResult":
        expected = {
            "schema_version", "tool_name", "tool_version", "analysis_id", "working_design_ref",
            "semantic_design_hash", "status", "measurements", "assumptions", "applicability",
            "limitations", "message",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("EngineeringAnalysisResult does not match the required schema.")
        measurements = payload["measurements"]
        if not isinstance(measurements, Mapping):
            raise ValueError("Analysis measurements must be an object.")
        return cls(
            tool_name=payload["tool_name"],
            tool_version=payload["tool_version"],
            analysis_id=payload["analysis_id"],
            working_design_ref=EngineeringDesignRef.from_dict(payload["working_design_ref"]),
            semantic_design_hash=payload["semantic_design_hash"],
            status=payload["status"],
            measurements={key: AnalysisMeasurement.from_dict(value) for key, value in measurements.items()},
            assumptions=tuple(payload["assumptions"]),
            applicability=payload["applicability"],
            limitations=tuple(payload["limitations"]),
            message=payload["message"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class RectangularPatchBaseline:
    """First-order rectangular microstrip-patch dimensions in millimetres."""

    width_mm: float
    length_mm: float
    effective_epsilon_r: float
    fringing_extension_mm: float


def rectangular_patch_baseline_dimensions(
    frequency_ghz: float,
    epsilon_r: float,
    substrate_thickness_mm: float,
) -> RectangularPatchBaseline:
    """Evaluate the conventional dominant-mode rectangular-patch approximation."""

    values = (frequency_ghz, epsilon_r, substrate_thickness_mm)
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("Patch-baseline inputs must be finite numeric values.")
    if frequency_ghz <= 0 or epsilon_r <= 0 or substrate_thickness_mm <= 0:
        raise ValueError("Frequency, relative permittivity, and substrate thickness must be positive.")
    width_mm = SPEED_OF_LIGHT_MM_GHZ / (2.0 * frequency_ghz) * math.sqrt(2.0 / (epsilon_r + 1.0))
    width_to_height = width_mm / substrate_thickness_mm
    effective_epsilon_r = (
        (epsilon_r + 1.0) / 2.0
        + (epsilon_r - 1.0) / 2.0 / math.sqrt(1.0 + 12.0 / width_to_height)
    )
    fringing_extension_mm = 0.412 * substrate_thickness_mm * (
        (effective_epsilon_r + 0.3) * (width_to_height + 0.264)
        / ((effective_epsilon_r - 0.258) * (width_to_height + 0.8))
    )
    length_mm = (
        SPEED_OF_LIGHT_MM_GHZ / (2.0 * frequency_ghz * math.sqrt(effective_epsilon_r))
        - 2.0 * fringing_extension_mm
    )
    if not all(math.isfinite(value) and value > 0 for value in (
        width_mm, length_mm, effective_epsilon_r, fringing_extension_mm,
    )):
        raise ValueError("Patch-baseline equations did not produce a physical finite result.")
    return RectangularPatchBaseline(
        width_mm, length_mm, effective_epsilon_r, fringing_extension_mm,
    )


_RECTANGULAR_PATCH_ASSUMPTIONS = (
    "The estimate is a conventional rectangular microstrip-patch dominant fundamental-mode baseline.",
    "The substrate is treated as homogeneous with the canonical relative permittivity and thickness.",
    "The ground and dielectric support are idealized by the first-order closed-form approximation.",
    "estimated_patch_width and estimated_patch_length are analytical_estimate values.",
    "current_patch_width and current_patch_length are current_canonical_dimension values from per-element patch_width_mm and patch_length_mm.",
)

_RECTANGULAR_PATCH_LIMITATIONS = (
    "Slots, loading, finite-ground effects, and feed perturbations are not modeled by the analytical baseline.",
    "Array mutual coupling is not modeled; an array result describes one radiating element.",
    "The calculation does not estimate impedance match, gain, bandwidth, efficiency, or polarization.",
    "No full-wave simulation or validation is performed, and matching the estimate does not guarantee resonance at the reference frequency.",
)


def _canonical_substrate_epsilon_r(design: AntennaDesign) -> tuple[float | None, str]:
    substrate_ids = {item.material_id for item in design.geometry if "substrate" in item.tags}
    candidates = tuple(
        material for material in design.materials
        if material.kind == "dielectric" and (not substrate_ids or material.material_id in substrate_ids)
    )
    if len(candidates) != 1:
        return None, "missing" if not candidates else "ambiguous"
    epsilon_r = candidates[0].epsilon_r
    if type(epsilon_r) not in (int, float) or not math.isfinite(epsilon_r) or epsilon_r <= 0:
        return None, "missing or invalid"
    return float(epsilon_r), ""


def _radiator_has_composed_modification(design: AntennaDesign) -> bool:
    """Use canonical composition and evaluated Boolean provenance, not feature names."""

    composed_boolean_ids = {
        str(arguments["operation_id"])
        for group in design.composed_operations
        for call in group.calls
        if call.name in {"boolean.subtract", "boolean.union"}
        for arguments in (call.arguments(),)
        if "operation_id" in arguments
    }
    query = evaluate_geometry(design)
    evaluated_by_id = {item.object_id: item for item in query.objects}
    composed_affects_radiator = any(
        item.semantic_role == "radiating_patch_conductor"
        and bool(composed_boolean_ids.intersection(item.boolean_history))
        for item in query.objects
    )
    tagged_tool_affects_radiator = any(
        record.status in {"completed", "partially_evaluated"}
        and evaluated_by_id.get(record.target_id) is not None
        and evaluated_by_id[record.target_id].semantic_role == "radiating_patch_conductor"
        and any(
            tool_id in evaluated_by_id
            and bool({"planner_created", "boolean_tool", "custom_geometry"}.intersection(
                evaluated_by_id[tool_id].tags
            ))
            for tool_id in record.tool_ids
        )
        for record in query.booleans
    )
    active_modifiers = tuple(
        value.strip() for value in design.metadata_map().get("active_modifiers", "").split(",")
        if value.strip()
    )
    return composed_affects_radiator or tagged_tool_affects_radiator or bool(active_modifiers)


def rectangular_patch_baseline_analysis(
    design: AntennaDesign,
    arguments: dict[str, Any],
) -> EngineeringAnalysisResult:
    """Compare one canonical rectangular patch element with a closed-form baseline."""

    tool_name = "engineering.rectangular_patch_baseline"
    tool_version = "1"
    design_hash = semantic_design_hash(design)
    analysis_id = stable_analysis_id(tool_name, tool_version, arguments, design_hash)
    design_ref = EngineeringDesignRef(design.design_id, design.revision)
    if design.family != "rectangular_inset_patch":
        return EngineeringAnalysisResult(
            tool_name, tool_version, analysis_id, design_ref, design_hash, "not_applicable", {},
            _RECTANGULAR_PATCH_ASSUMPTIONS,
            f"Not applicable to antenna family {design.family!r}; a supported rectangular microstrip-patch design is required.",
            _RECTANGULAR_PATCH_LIMITATIONS,
            "The rectangular-patch analytical baseline was not evaluated for this antenna family.",
        )

    measurements: dict[str, AnalysisMeasurement] = {}
    canonical_values: dict[str, float] = {}
    for key, measurement_name, unit in (
        ("frequency_ghz", "frequency", "GHz"),
        ("substrate_thickness_mm", "substrate_thickness", "mm"),
        ("patch_width_mm", "current_patch_width", "mm"),
        ("patch_length_mm", "current_patch_length", "mm"),
    ):
        try:
            value = float(design.value(key))
        except (KeyError, TypeError, ValueError, ArithmeticError):
            continue
        if math.isfinite(value):
            canonical_values[key] = value
            measurements[measurement_name] = AnalysisMeasurement(value, unit)

    epsilon_r, epsilon_reason = _canonical_substrate_epsilon_r(design)
    if epsilon_r is not None:
        measurements["epsilon_r"] = AnalysisMeasurement(epsilon_r, "1")
    missing = tuple(
        label for key, label in (
            ("frequency_ghz", "frequency"),
            ("substrate_thickness_mm", "substrate thickness"),
            ("patch_width_mm", "current patch width"),
            ("patch_length_mm", "current patch length"),
        ) if key not in canonical_values or canonical_values[key] <= 0
    )
    if epsilon_r is None or missing:
        requirements = list(missing)
        if epsilon_r is None:
            requirements.append(f"an unambiguous positive substrate relative permittivity ({epsilon_reason})")
        return EngineeringAnalysisResult(
            tool_name, tool_version, analysis_id, design_ref, design_hash, "unknown", measurements,
            _RECTANGULAR_PATCH_ASSUMPTIONS,
            "Unresolved: " + ", ".join(requirements) + " is required from the canonical design/material system.",
            _RECTANGULAR_PATCH_LIMITATIONS,
            "The rectangular-patch baseline could not be calculated because required canonical data is unavailable.",
        )

    try:
        estimate = rectangular_patch_baseline_dimensions(
            canonical_values["frequency_ghz"], epsilon_r,
            canonical_values["substrate_thickness_mm"],
        )
    except ValueError as exc:
        return EngineeringAnalysisResult(
            tool_name, tool_version, analysis_id, design_ref, design_hash, "unknown", measurements,
            _RECTANGULAR_PATCH_ASSUMPTIONS, f"Unresolved analytical baseline: {exc}",
            _RECTANGULAR_PATCH_LIMITATIONS,
            "The canonical inputs did not produce a physical first-order patch estimate.",
        )

    current_width = canonical_values["patch_width_mm"]
    current_length = canonical_values["patch_length_mm"]
    width_difference = current_width - estimate.width_mm
    length_difference = current_length - estimate.length_mm
    measurements.update({
        "estimated_patch_width": AnalysisMeasurement(estimate.width_mm, "mm"),
        "estimated_patch_length": AnalysisMeasurement(estimate.length_mm, "mm"),
        "width_difference": AnalysisMeasurement(width_difference, "mm"),
        "length_difference": AnalysisMeasurement(length_difference, "mm"),
        "width_percent_difference": AnalysisMeasurement(100.0 * width_difference / estimate.width_mm, "%"),
        "length_percent_difference": AnalysisMeasurement(100.0 * length_difference / estimate.length_mm, "%"),
        "effective_dielectric_constant": AnalysisMeasurement(estimate.effective_epsilon_r, "1"),
        "fringing_extension": AnalysisMeasurement(estimate.fringing_extension_mm, "mm"),
    })
    modified = _radiator_has_composed_modification(design)
    array_text = (
        f" The {design.array.rows}x{design.array.columns} array is represented by one element; total array footprint is not analyzed."
        if design.array.element_count > 1 else ""
    )
    if modified:
        applicability = (
            "Approximate modified topology: the conventional unmodified rectangular-patch baseline is reference-only "
            "because evaluated/canonical provenance records composed radiator geometry."
        )
        limitations = (*_RECTANGULAR_PATCH_LIMITATIONS,
                       "The current radiator has composed geometry that the closed-form baseline does not represent.")
    else:
        applicability = "Conventional unmodified rectangular-patch analytical baseline for one radiating element."
        limitations = _RECTANGULAR_PATCH_LIMITATIONS
    return EngineeringAnalysisResult(
        tool_name, tool_version, analysis_id, design_ref, design_hash, "completed", measurements,
        _RECTANGULAR_PATCH_ASSUMPTIONS, applicability + array_text, limitations,
        "Calculated the first-order analytical width and length and compared them with the current canonical per-element patch dimensions.",
    )


@dataclass(frozen=True, slots=True)
class DipoleGeometryBaseline:
    """Evaluated centerline dimensions for one canonical two-arm dipole element."""

    arm_lengths_mm: tuple[float, float]
    feed_gap_mm: float
    arm_radii_mm: tuple[float, float]
    total_conductor_length_mm: float
    total_dipole_length_mm: float
    tip_to_tip_span_mm: float
    element: tuple[int, int] | None
    modified: bool


def _cylinder_centerline(design: AntennaDesign, evaluated_object) -> tuple[
    tuple[float, float, float], tuple[float, float, float], float,
]:
    canonical = next(
        (item for item in design.geometry if item.object_id == evaluated_object.object_id), None,
    )
    if canonical is None or evaluated_object.primitive != "cylinder":
        raise ValueError("A dipole arm is not a canonical cylinder.")
    dimensions = dict(evaluated_object.resolved_dimensions)
    required = {"center_1", "center_2", "radius", "start", "end"}
    if not required.issubset(dimensions):
        raise ValueError("A dipole arm has unresolved cylinder dimensions.")
    center_1 = dimensions["center_1"]
    center_2 = dimensions["center_2"]
    start = dimensions["start"]
    end = dimensions["end"]
    radius = dimensions["radius"]
    if not all(math.isfinite(value) for value in (center_1, center_2, start, end, radius)):
        raise ValueError("A dipole arm has non-finite cylinder dimensions.")
    if end <= start or radius <= 0:
        raise ValueError("A dipole arm must have positive centerline length and radius.")
    if evaluated_object.axis == "x":
        local_start, local_end = (start, center_1, center_2), (end, center_1, center_2)
    elif evaluated_object.axis == "y":
        local_start, local_end = (center_1, start, center_2), (center_1, end, center_2)
    elif evaluated_object.axis == "z":
        local_start, local_end = (center_1, center_2, start), (center_1, center_2, end)
    else:
        raise ValueError("A dipole arm has an unsupported cylinder axis.")
    return transformed_point(canonical, local_start), transformed_point(canonical, local_end), radius


def evaluated_dipole_geometry(design: AntennaDesign) -> DipoleGeometryBaseline:
    """Resolve physical dimensions from evaluated arm centerlines for one element."""

    query = evaluate_geometry(design)
    all_arms = tuple(
        item for item in query.objects
        if item.semantic_role == "radiating_arm" and item.is_physical is True
    )
    if not all_arms:
        raise ValueError("No resolved physical radiating arms are available.")
    elements = tuple(sorted({item.element for item in all_arms}, key=lambda value: value or (0, 0)))
    selected_element = (1, 1) if (1, 1) in elements else elements[0]
    arms = tuple(sorted(
        (item for item in all_arms if item.element == selected_element),
        key=lambda item: item.object_id,
    ))
    if len(arms) != 2:
        raise ValueError("Exactly two resolved physical radiating arms are required for one dipole element.")
    centerlines = tuple(_cylinder_centerline(design, item) for item in arms)
    arm_lengths = tuple(math.dist(start, end) for start, end, _radius in centerlines)
    radii = tuple(radius for _start, _end, radius in centerlines)
    if not all(math.isfinite(value) and value > 0 for value in (*arm_lengths, *radii)):
        raise ValueError("Resolved dipole arm dimensions are not positive and finite.")
    first_endpoints = centerlines[0][:2]
    second_endpoints = centerlines[1][:2]
    feed_gap = min(math.dist(first, second) for first in first_endpoints for second in second_endpoints)
    all_endpoints = (*first_endpoints, *second_endpoints)
    tip_to_tip_span = max(
        math.dist(first, second)
        for index, first in enumerate(all_endpoints)
        for second in all_endpoints[index + 1:]
    )
    total_conductor_length = sum(arm_lengths)
    total_dipole_length = total_conductor_length + feed_gap
    modified = any(item.boolean_history or len(item.source_primitive_ids) > 1 for item in arms)
    modified = modified or any(
        group.source_family == "dipole"
        and any(call.name.startswith(("geometry.", "boolean.")) for call in group.calls)
        for group in design.composed_operations
    )
    return DipoleGeometryBaseline(
        arm_lengths, feed_gap, radii, total_conductor_length,
        total_dipole_length, tip_to_tip_span, selected_element, modified,
    )


_DIPOLE_ASSUMPTIONS = (
    "lambda0 is the free-space wavelength c/f using c = 299792458 m/s.",
    "The analytical references are lambda0/2 for total dipole length and lambda0/4 for each arm.",
    "Arm lengths, feed gap, radii, and tip-to-tip span are current_canonical_dimension values extracted from evaluated physical cylinder centerlines.",
    "lambda0/2 and lambda0/4 measurements are analytical_reference values, not predicted resonant dimensions.",
)

_DIPOLE_LIMITATIONS = (
    "The free-space half-wave and quarter-wave quantities are reference baselines, not exact physical resonant lengths.",
    "Conductor diameter, feed gap, end effects, nearby materials, supports, and environment can change actual resonance.",
    "Input impedance, bandwidth, efficiency, gain, balun behavior, and radiation-pattern validity are not evaluated.",
    "No full-wave simulation or validation is performed.",
)


def dipole_baseline_analysis(
    design: AntennaDesign,
    arguments: dict[str, Any],
) -> EngineeringAnalysisResult:
    """Compare evaluated dipole dimensions with free-space wavelength references."""

    tool_name = "engineering.dipole_baseline"
    tool_version = "1"
    design_hash = semantic_design_hash(design)
    analysis_id = stable_analysis_id(tool_name, tool_version, arguments, design_hash)
    design_ref = EngineeringDesignRef(design.design_id, design.revision)
    if design.family != "dipole":
        return EngineeringAnalysisResult(
            tool_name, tool_version, analysis_id, design_ref, design_hash, "not_applicable", {},
            _DIPOLE_ASSUMPTIONS,
            f"Not applicable to antenna family {design.family!r}; a supported dipole design is required.",
            _DIPOLE_LIMITATIONS,
            "The dipole electrical-length baseline was not evaluated for this antenna family.",
        )

    measurements: dict[str, AnalysisMeasurement] = {}
    try:
        frequency_ghz = float(design.value("frequency_ghz"))
    except (KeyError, TypeError, ValueError, ArithmeticError):
        frequency_ghz = math.nan
    if not math.isfinite(frequency_ghz) or frequency_ghz <= 0:
        return EngineeringAnalysisResult(
            tool_name, tool_version, analysis_id, design_ref, design_hash, "unknown", measurements,
            _DIPOLE_ASSUMPTIONS,
            "Unresolved: a positive finite canonical reference frequency is required.",
            _DIPOLE_LIMITATIONS,
            "The dipole baseline could not be calculated because frequency is unavailable.",
        )
    measurements["frequency"] = AnalysisMeasurement(frequency_ghz, "GHz")
    lambda0_mm = SPEED_OF_LIGHT_MM_GHZ / frequency_ghz
    measurements["lambda0"] = AnalysisMeasurement(lambda0_mm, "mm")
    try:
        geometry = evaluated_dipole_geometry(design)
    except ValueError as exc:
        return EngineeringAnalysisResult(
            tool_name, tool_version, analysis_id, design_ref, design_hash, "unknown", measurements,
            _DIPOLE_ASSUMPTIONS, f"Unresolved canonical dipole geometry: {exc}",
            _DIPOLE_LIMITATIONS,
            "The dipole baseline could not be calculated because required physical geometry is unavailable.",
        )

    half_wave = lambda0_mm / 2.0
    quarter_wave = lambda0_mm / 4.0
    difference = geometry.total_dipole_length_mm - half_wave
    arm_1, arm_2 = geometry.arm_lengths_mm
    radius_1, radius_2 = geometry.arm_radii_mm
    measurements.update({
        "current_total_dipole_length": AnalysisMeasurement(geometry.total_dipole_length_mm, "mm"),
        "current_total_dipole_length_lambda_ratio": AnalysisMeasurement(geometry.total_dipole_length_mm / lambda0_mm, "lambda0"),
        "current_total_conductor_length": AnalysisMeasurement(geometry.total_conductor_length_mm, "mm"),
        "current_total_conductor_length_lambda_ratio": AnalysisMeasurement(geometry.total_conductor_length_mm / lambda0_mm, "lambda0"),
        "current_tip_to_tip_span": AnalysisMeasurement(geometry.tip_to_tip_span_mm, "mm"),
        "current_arm_1_length": AnalysisMeasurement(arm_1, "mm"),
        "current_arm_2_length": AnalysisMeasurement(arm_2, "mm"),
        "current_arm_1_length_lambda_ratio": AnalysisMeasurement(arm_1 / lambda0_mm, "lambda0"),
        "current_arm_2_length_lambda_ratio": AnalysisMeasurement(arm_2 / lambda0_mm, "lambda0"),
        "half_wave_length_reference": AnalysisMeasurement(half_wave, "mm"),
        "quarter_wave_arm_reference": AnalysisMeasurement(quarter_wave, "mm"),
        "total_length_difference_from_half_wave": AnalysisMeasurement(difference, "mm"),
        "total_length_percent_difference_from_half_wave": AnalysisMeasurement(100.0 * difference / half_wave, "%"),
        "current_feed_gap": AnalysisMeasurement(geometry.feed_gap_mm, "mm"),
        "current_feed_gap_lambda_ratio": AnalysisMeasurement(geometry.feed_gap_mm / lambda0_mm, "lambda0"),
        "current_arm_1_radius": AnalysisMeasurement(radius_1, "mm"),
        "current_arm_2_radius": AnalysisMeasurement(radius_2, "mm"),
        "current_arm_1_diameter": AnalysisMeasurement(2.0 * radius_1, "mm"),
        "current_arm_2_diameter": AnalysisMeasurement(2.0 * radius_2, "mm"),
        "current_arm_1_radius_lambda_ratio": AnalysisMeasurement(radius_1 / lambda0_mm, "lambda0"),
        "current_arm_2_radius_lambda_ratio": AnalysisMeasurement(radius_2 / lambda0_mm, "lambda0"),
        "current_arm_1_diameter_lambda_ratio": AnalysisMeasurement(2.0 * radius_1 / lambda0_mm, "lambda0"),
        "current_arm_2_diameter_lambda_ratio": AnalysisMeasurement(2.0 * radius_2 / lambda0_mm, "lambda0"),
    })
    if math.isclose(radius_1, radius_2, rel_tol=1e-12, abs_tol=1e-12):
        measurements.update({
            "current_conductor_radius": AnalysisMeasurement(radius_1, "mm"),
            "current_conductor_diameter": AnalysisMeasurement(2.0 * radius_1, "mm"),
            "current_conductor_radius_lambda_ratio": AnalysisMeasurement(radius_1 / lambda0_mm, "lambda0"),
            "current_conductor_diameter_lambda_ratio": AnalysisMeasurement(2.0 * radius_1 / lambda0_mm, "lambda0"),
        })
    array_text = (
        f" The {design.array.rows}x{design.array.columns} array is represented by element {geometry.element}; array coupling is not analyzed."
        if design.array.element_count > 1 else ""
    )
    if geometry.modified:
        applicability = (
            "Approximate modified topology: the simple two-arm free-space baseline is reference-only because "
            "evaluated/canonical provenance records composed conductor geometry."
        )
        limitations = (*_DIPOLE_LIMITATIONS,
                       "The current conductors contain composed geometry that the centerline baseline does not model.")
    else:
        applicability = "Simple two-arm dipole free-space electrical-length baseline."
        limitations = _DIPOLE_LIMITATIONS
    return EngineeringAnalysisResult(
        tool_name, tool_version, analysis_id, design_ref, design_hash, "completed", measurements,
        _DIPOLE_ASSUMPTIONS, applicability + array_text, limitations,
        "Measured the current canonical dipole centerlines and compared total and arm electrical lengths with free-space half-wave and quarter-wave references.",
    )


def design_summary_analysis(design: AntennaDesign, arguments: dict[str, Any]) -> EngineeringAnalysisResult:
    """Neutral runtime proof: summarize canonical scope without RF inference."""

    design_hash = semantic_design_hash(design)
    elements = max(1, design.array.rows * design.array.columns)
    measurements = {
        "geometry_object_count": AnalysisMeasurement(len(design.geometry), "count"),
        "port_count": AnalysisMeasurement(len(design.ports), "count"),
        "array_element_count": AnalysisMeasurement(elements, "count"),
        "boolean_operation_count": AnalysisMeasurement(len(design.booleans), "count"),
        "composed_feature_count": AnalysisMeasurement(len(design.composed_operations), "count"),
    }
    return EngineeringAnalysisResult(
        tool_name="engineering.design_summary",
        tool_version="1",
        analysis_id=stable_analysis_id("engineering.design_summary", "1", arguments, design_hash),
        working_design_ref=EngineeringDesignRef(design.design_id, design.revision),
        semantic_design_hash=design_hash,
        status="completed",
        measurements=measurements,
        assumptions=("Counts are derived from the current canonical AntennaDesign.",),
        applicability="Canonical design inventory and array scope only.",
        limitations=("This analysis does not estimate or validate RF performance.",),
        message="Counted the current canonical design objects, ports, elements, Boolean operations, and composed features.",
    )


_ARRAY_SPACING_LIMITATIONS = (
    "Ideal periodic-array principal-plane analysis only.",
    "Finite-array effects are not included.",
    "Element-pattern suppression and mutual coupling are not included.",
    "Feed-network amplitude, phase, and fabrication errors are not included.",
    "Substrate, surface-wave, and full-wave effects are not included.",
)


def _visible_spatial_orders(spacing_lambda: float, scan_angle_deg: float) -> tuple[tuple[int, float], ...]:
    """Return visible nonzero orders for sin(theta_m)=sin(theta_0)+m*lambda/d."""

    sine_0 = math.sin(math.radians(scan_angle_deg))
    minimum = math.ceil((-1.0 - sine_0) * spacing_lambda - 1e-12)
    maximum = math.floor((1.0 - sine_0) * spacing_lambda + 1e-12)
    visible = []
    for order in range(minimum, maximum + 1):
        if order == 0:
            continue
        sine_m = sine_0 + order / spacing_lambda
        if abs(sine_m) <= 1.0 + 1e-12:
            sine_m = max(-1.0, min(1.0, sine_m))
            visible.append((order, math.degrees(math.asin(sine_m))))
    return tuple(visible)


def array_spacing_analysis(design: AntennaDesign, arguments: dict[str, Any]) -> EngineeringAnalysisResult:
    """Measure canonical rectangular-array spacing in free-space wavelengths."""

    design_hash = semantic_design_hash(design)
    analysis_id = stable_analysis_id("engineering.array_spacing", "1", arguments, design_hash)
    design_ref = EngineeringDesignRef(design.design_id, design.revision)
    rows = design.array.rows
    columns = design.array.columns
    measurements: dict[str, AnalysisMeasurement] = {
        "rows": AnalysisMeasurement(rows, "count"),
        "columns": AnalysisMeasurement(columns, "count"),
    }
    active_axes = tuple(axis for axis, count in (("row", rows), ("column", columns)) if count > 1)
    inactive_axes = tuple(axis for axis, count in (("row", rows), ("column", columns)) if count == 1)
    if not active_axes:
        return EngineeringAnalysisResult(
            "engineering.array_spacing", "1", analysis_id, design_ref, design_hash, "not_applicable",
            measurements,
            ("Array dimensions come from the canonical ArraySpec.",),
            "Not applicable: a 1x1 design has no periodic row or column spacing axis.",
            _ARRAY_SPACING_LIMITATIONS,
            "Electrical-spacing analysis is not applicable to a single-element design.",
        )

    try:
        frequency_ghz = float(design.value("frequency_ghz"))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        frequency_ghz = math.nan
    spacing_mm = design.array.spacing_mm
    if not math.isfinite(frequency_ghz) or frequency_ghz <= 0:
        return EngineeringAnalysisResult(
            "engineering.array_spacing", "1", analysis_id, design_ref, design_hash, "unknown",
            measurements,
            ("Array dimensions and spacing come from the canonical ArraySpec.",),
            "Unresolved: the canonical operating/reference frequency is missing, non-finite, or non-positive.",
            _ARRAY_SPACING_LIMITATIONS,
            "Electrical spacing could not be calculated because frequency is unresolved.",
        )
    measurements["frequency"] = AnalysisMeasurement(frequency_ghz, "GHz")
    wavelength_mm = SPEED_OF_LIGHT_MM_GHZ / frequency_ghz
    measurements["lambda0"] = AnalysisMeasurement(wavelength_mm, "mm")
    if not isinstance(spacing_mm, (int, float)) or isinstance(spacing_mm, bool) or not math.isfinite(spacing_mm) or spacing_mm <= 0:
        return EngineeringAnalysisResult(
            "engineering.array_spacing", "1", analysis_id, design_ref, design_hash, "unknown",
            measurements,
            ("Frequency is the canonical frequency_ghz parameter.",),
            "Unresolved: canonical spacing_mm is missing, non-finite, or non-positive for an active array axis.",
            _ARRAY_SPACING_LIMITATIONS,
            "Electrical spacing could not be calculated because array spacing is unresolved.",
        )

    spacing_lambda = spacing_mm / wavelength_mm
    for axis in active_axes:
        measurements[f"{axis}_spacing"] = AnalysisMeasurement(spacing_mm, "mm")
        measurements[f"{axis}_spacing_lambda"] = AnalysisMeasurement(spacing_lambda, "lambda0")

    scan_angle = arguments.get("scan_angle_deg")
    requested_axis = arguments.get("principal_axis", "all_active")
    inactive_text = (
        " ".join(f"The {axis} axis is not applicable because its element count is 1." for axis in inactive_axes)
        if inactive_axes else "Both principal axes are active."
    )
    assumptions = [
        "Frequency and physical spacing are read from the current canonical AntennaDesign.",
        "lambda0 is the free-space wavelength c/f using c = 299792458 m/s.",
    ]
    if scan_angle is None:
        if "principal_axis" in arguments:
            return EngineeringAnalysisResult(
                "engineering.array_spacing", "1", analysis_id, design_ref, design_hash, "unknown",
                measurements, tuple(assumptions),
                "A principal_axis was supplied without scan_angle_deg; no visible-order test was performed. " + inactive_text,
                _ARRAY_SPACING_LIMITATIONS,
                "Electrical spacing was calculated, but the scan assumption is incomplete.",
            )
        return EngineeringAnalysisResult(
            "engineering.array_spacing", "1", analysis_id, design_ref, design_hash, "completed",
            measurements, tuple(assumptions),
            "Electrical spacing is reported for active canonical array axes. " + inactive_text,
            (*_ARRAY_SPACING_LIMITATIONS,
             "No grating-lobe conclusion is made without a scan-angle and progressive-phase assumption."),
            "Calculated free-space electrical spacing. Grating-lobe behavior was not classified because no scan assumption was supplied.",
        )

    scan_angle = float(scan_angle)
    selected_axes = active_axes if requested_axis == "all_active" else (requested_axis,)
    unavailable = tuple(axis for axis in selected_axes if axis not in active_axes)
    if unavailable:
        return EngineeringAnalysisResult(
            "engineering.array_spacing", "1", analysis_id, design_ref, design_hash, "not_applicable",
            measurements, tuple(assumptions),
            f"The requested principal axis {unavailable[0]!r} is inactive in the {rows}x{columns} array. " + inactive_text,
            _ARRAY_SPACING_LIMITATIONS,
            "The requested principal-axis scan analysis is not applicable to this array layout.",
        )

    assumptions.extend((
        f"A progressive phase steers the principal beam to {scan_angle:.12g} degrees on each selected principal axis.",
        "Visible orders use sin(theta_m) = sin(theta_0) + m*lambda0/d with nonzero integer m.",
    ))
    findings = []
    for axis in selected_axes:
        orders = _visible_spatial_orders(spacing_lambda, scan_angle)
        measurements[f"{axis}_visible_nonzero_order_count"] = AnalysisMeasurement(len(orders), "count")
        for order, angle in orders:
            order_label = f"neg_{abs(order)}" if order < 0 else f"pos_{order}"
            measurements[f"{axis}_visible_order_m_{order_label}_angle"] = AnalysisMeasurement(angle, "deg")
        if orders:
            details = ", ".join(f"m={order} at {angle:.4g} deg" for order, angle in orders)
            findings.append(f"{axis}: visible nonzero order(s) {details}")
        else:
            findings.append(f"{axis}: no visible nonzero integer order")
    return EngineeringAnalysisResult(
        "engineering.array_spacing", "1", analysis_id, design_ref, design_hash, "completed",
        measurements, tuple(assumptions),
        f"Ideal periodic principal-plane test at {scan_angle:.12g} degrees for {', '.join(selected_axes)} axis/axes. " + inactive_text,
        _ARRAY_SPACING_LIMITATIONS,
        "Electrical spacing and visible spatial orders were calculated under the stated ideal assumptions: " + "; ".join(findings) + ".",
    )
