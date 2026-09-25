"""Passive engineering-inspection protocol. No checks, geometry, or repairs run here.

Reports are derived facts about an explicitly identified working state, not
canonical state or instructions. Capability allowlists must come from the runtime,
never from serialized report data. Messages describe conditions, not repair advice;
this module deliberately does not attempt to classify prose.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import InitVar, dataclass, field, fields
from types import MappingProxyType
from typing import Any, Collection, Mapping


ENGINEERING_REPORT_SCHEMA_VERSION = 1
_SENSITIVE = ("api_key", "apikey", "authorization", "cookie", "password", "secret", "token", "x_goog_api_key")


def _text(value: Any, name: str, *, empty: bool = False) -> None:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{name} must be a nonempty string." if not empty else f"{name} must be a string.")


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple of strings.")
    for item in value:
        _text(item, name)
    return tuple(value)


def _hash(value: Any, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")


def _payload(cls: type, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {item.name for item in fields(cls)}:
        raise ValueError(f"{cls.__name__} does not match the required schema.")
    return dict(value)


def _redact(value: Any) -> Any:
    """Match trajectory key redaction, also scrubbing labelled credentials in text."""
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if any(
                marker in str(key).casefold().replace("-", "_") for marker in _SENSITIVE
            ) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)\bBearer\s+[^\s,;]+", "Bearer [REDACTED]", value)
        value = re.sub(
            r"(?i)(api[_-]?key|x[_-]goog[_-]api[_-]key|authorization|cookie|token|password|secret)(\s*[:=]\s*)[^\s,;}&]+",
            r"\1\2[REDACTED]", value,
        )
        return re.sub(r"(?i)([?&](?:key|api_key|token)=)[^&\s]+", r"\1[REDACTED]", value)
    return value


class _Record:
    def to_dict(self) -> dict[str, Any]:
        def encode(value: Any) -> Any:
            if isinstance(value, _Record):
                return {item.name: encode(getattr(value, item.name)) for item in fields(value)}
            if isinstance(value, Mapping):
                return {key: encode(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [encode(item) for item in value]
            return value
        return _redact(encode(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]):
        return cls(**_payload(cls, payload))


@dataclass(frozen=True, slots=True)
class EngineeringDesignRef(_Record):
    design_id: str
    revision: int

    def __post_init__(self) -> None:
        _text(self.design_id, "design_id")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("revision must be a nonnegative integer.")


@dataclass(frozen=True, slots=True)
class EngineeringMeasurement(_Record):
    value: float
    unit: str
    tolerance: float | None = None

    def __post_init__(self) -> None:
        for name, value in (("value", self.value), ("tolerance", self.tolerance)):
            if name == "tolerance" and value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Measurement {name} must be finite and numeric.")
        if self.tolerance is not None and self.tolerance < 0:
            raise ValueError("Measurement tolerance must be nonnegative.")
        _text(self.unit, "unit", empty=True)  # "" or "1" may describe dimensionless ratios.


@dataclass(frozen=True, slots=True)
class EngineeringSource(_Record):
    method: str
    applicability: str
    evaluated_geometry_hash: str | None = None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.method, "method")
        _text(self.applicability, "applicability")
        if self.evaluated_geometry_hash is not None:
            _hash(self.evaluated_geometry_hash, "evaluated_geometry_hash")
        object.__setattr__(self, "limitations", _strings(self.limitations, "limitations"))


def stable_observation_id(
    check_id: str, category: str, *, affected_objects: tuple[str, ...] = (),
    affected_ports: tuple[str, ...] = (), affected_elements: tuple[tuple[int, int], ...] = (),
    relationship_key: str = "",
) -> str:
    """Order-independent relationship identity, excluding measurements/version/state.

    relationship_key is an optional semantic discriminator (e.g. positive terminal),
    not an occurrence ID. Directional relationships must encode direction there.
    """
    _text(check_id, "check_id")
    _text(category, "category")
    _text(relationship_key, "relationship_key", empty=True)
    objects = _strings(affected_objects, "affected_objects")
    ports = _strings(affected_ports, "affected_ports")
    if not isinstance(affected_elements, (tuple, list)):
        raise ValueError("affected_elements must contain row/column pairs.")
    elements = []
    for pair in affected_elements:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2 or any(type(x) is not int or x < 1 for x in pair):
            raise ValueError("affected_elements must contain positive integer row/column pairs.")
        elements.append(tuple(pair))
    payload = [check_id, category, sorted(set(objects)), sorted(set(ports)), sorted(set(elements)), relationship_key]
    return "eng-v1-" + hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class EngineeringObservation(_Record):
    check_id: str
    check_version: str
    severity: str
    category: str
    message: str
    source: EngineeringSource
    affected_objects: tuple[str, ...] = ()
    affected_ports: tuple[str, ...] = ()
    affected_elements: tuple[tuple[int, int], ...] = ()
    measured_values: Mapping[str, EngineeringMeasurement] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    capability_refs: tuple[str, ...] = ()
    relationship_key: str = ""
    observation_id: str = field(init=False)
    installed_capability_ids: InitVar[Collection[str]] = ()

    def __post_init__(self, installed_capability_ids: Collection[str]) -> None:
        for name in ("check_id", "check_version", "category", "message"):
            _text(getattr(self, name), name)
        if self.severity not in ("info", "warning", "blocking"):
            raise ValueError("Invalid engineering severity.")
        if not isinstance(self.source, EngineeringSource):
            raise ValueError("source must be EngineeringSource.")
        for name in ("affected_objects", "affected_ports", "assumptions", "capability_refs"):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        identity = stable_observation_id(
            self.check_id, self.category, affected_objects=self.affected_objects,
            affected_ports=self.affected_ports, affected_elements=self.affected_elements,
            relationship_key=self.relationship_key,
        )
        object.__setattr__(self, "affected_elements", tuple(tuple(pair) for pair in self.affected_elements))
        object.__setattr__(self, "observation_id", identity)
        if not isinstance(self.measured_values, Mapping):
            raise ValueError("measured_values must be a measurement mapping.")
        for name, value in self.measured_values.items():
            _text(name, "measurement name")
            if not isinstance(value, EngineeringMeasurement):
                raise ValueError("measured_values entries must be EngineeringMeasurement.")
        object.__setattr__(self, "measured_values", MappingProxyType(dict(self.measured_values)))
        self.validate_capabilities(installed_capability_ids)

    def validate_capabilities(self, installed_capability_ids: Collection[str]) -> None:
        if isinstance(installed_capability_ids, str) or not set(self.capability_refs) <= set(installed_capability_ids):
            raise ValueError("capability_refs must reference verified installed capabilities.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, installed_capability_ids: Collection[str] = ()):
        data = _payload(cls, payload)
        identity = data.pop("observation_id")
        data["source"] = EngineeringSource.from_dict(data["source"])
        if not isinstance(data["measured_values"], Mapping):
            raise ValueError("measured_values must be a measurement mapping.")
        data["measured_values"] = {key: EngineeringMeasurement.from_dict(value) for key, value in data["measured_values"].items()}
        result = cls(**data, installed_capability_ids=installed_capability_ids)
        if identity != result.observation_id:
            raise ValueError("observation_id does not match its semantic relationship.")
        return result


@dataclass(frozen=True, slots=True)
class EngineeringCheckCoverage(_Record):
    check_id: str
    check_version: str
    status: str
    applicability: str
    reason: str
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("check_id", "check_version", "applicability", "reason"):
            _text(getattr(self, name), name)
        if self.status not in ("completed", "not_applicable", "unknown", "failed"):
            raise ValueError("Invalid engineering coverage status.")
        object.__setattr__(self, "limitations", _strings(self.limitations, "limitations"))


@dataclass(frozen=True, slots=True)
class EngineeringReport(_Record):
    working_design_ref: EngineeringDesignRef | None
    semantic_design_hash: str
    check_suite_version: str
    findings: tuple[EngineeringObservation, ...] = ()
    coverage: tuple[EngineeringCheckCoverage, ...] = ()
    schema_version: int = ENGINEERING_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != ENGINEERING_REPORT_SCHEMA_VERSION:
            raise ValueError("Unsupported engineering report schema version.")
        if self.working_design_ref is not None and not isinstance(self.working_design_ref, EngineeringDesignRef):
            raise ValueError("working_design_ref must be EngineeringDesignRef or None.")
        _hash(self.semantic_design_hash, "semantic_design_hash")
        _text(self.check_suite_version, "check_suite_version")
        for name, expected in (("findings", EngineeringObservation), ("coverage", EngineeringCheckCoverage)):
            items = getattr(self, name)
            if not isinstance(items, (list, tuple)) or any(not isinstance(item, expected) for item in items):
                raise ValueError(f"Invalid engineering {name}.")
            object.__setattr__(self, name, tuple(items))
        ids = [item.observation_id for item in self.findings]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate engineering observation IDs.")
        checks = [(item.check_id, item.check_version) for item in self.coverage]
        if len(checks) != len(set(checks)):
            raise ValueError("Duplicate engineering check coverage.")
        for finding in self.findings:
            if (finding.check_id, finding.check_version) not in checks:
                raise ValueError("Every finding requires matching check coverage.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, installed_capability_ids: Collection[str] = ()):
        data = _payload(cls, payload)
        if data["working_design_ref"] is not None:
            data["working_design_ref"] = EngineeringDesignRef.from_dict(data["working_design_ref"])
        if not isinstance(data["findings"], list) or not isinstance(data["coverage"], list):
            raise ValueError("findings and coverage must be arrays.")
        data["findings"] = tuple(EngineeringObservation.from_dict(item, installed_capability_ids=installed_capability_ids) for item in data["findings"])
        data["coverage"] = tuple(EngineeringCheckCoverage.from_dict(item) for item in data["coverage"])
        return cls(**data)

    def validate_capabilities(self, installed_capability_ids: Collection[str]) -> None:
        for finding in self.findings:
            finding.validate_capabilities(installed_capability_ids)
