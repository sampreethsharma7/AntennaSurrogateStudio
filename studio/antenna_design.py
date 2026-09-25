"""Solver-neutral antenna design graph used by agents, previews, and adapters."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping


SCHEMA_VERSION = 2


class DesignValidationError(ValueError):
    """Raised when a canonical design object is invalid."""


Scalar = float | int | str


@dataclass(frozen=True, slots=True)
class DesignParameter:
    key: str
    name: str
    label: str
    value: float
    unit: str = ""
    expression: str | None = None
    editable: bool = True
    sweepable: bool = False
    integer: bool = False
    minimum: float | None = None
    maximum: float | None = None
    sweep_lower_factor: float = 0.9
    sweep_upper_factor: float = 1.1


@dataclass(frozen=True, slots=True)
class MaterialSpec:
    material_id: str
    name: str
    kind: str
    epsilon_r: float = 1.0
    loss_tangent: float = 0.0
    conductivity_s_per_m: float | None = None


@dataclass(frozen=True, slots=True)
class TransformSpec:
    translate_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotate_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class GeometryObject:
    object_id: str
    primitive: str
    name: str
    material_id: str
    dimensions: tuple[tuple[str, Scalar], ...]
    axis: str = "z"
    transform: TransformSpec = field(default_factory=TransformSpec)
    component: str = "Antenna"
    tags: tuple[str, ...] = ()

    def dimension_map(self) -> dict[str, Scalar]:
        return dict(self.dimensions)


@dataclass(frozen=True, slots=True)
class BooleanOperation:
    operation_id: str
    operation: str
    target_id: str
    tool_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PortSpec:
    port_id: str
    name: str
    kind: str
    positive_point: tuple[Scalar, Scalar, Scalar]
    negative_point: tuple[Scalar, Scalar, Scalar]
    impedance_ohms: float = 50.0
    element_index: int = 1


EXCITATION_STRATEGIES = (
    "single_element_feed", "independent_ports", "corporate_feed", "series_feed",
    "custom", "unresolved", "legacy_recipe",
)
EXCITATION_REALIZATION_STATUSES = ("realized", "unresolved", "unsupported", "legacy")
FEED_NETWORK_TYPES = ("none", "unresolved", "independent", "corporate", "series", "custom")


@dataclass(frozen=True, slots=True)
class ExcitationAssignment:
    """Solver-neutral intended channel assignment for one array element."""

    element_row: int
    element_column: int
    excitation_id: str
    port_id: str | None = None
    status: str = "unresolved"


@dataclass(frozen=True, slots=True)
class ExcitationDefinition:
    """Excitation intent, realization truth, and optional physical-port linkage."""

    strategy: str = "unresolved"
    realization_status: str = "unresolved"
    feed_network: str = "unresolved"
    element_assignments: tuple[ExcitationAssignment, ...] = ()
    unresolved_requirements: tuple[str, ...] = ()

    @property
    def excitation_realized(self) -> bool:
        return self.realization_status == "realized"

    @property
    def realizing_port_ids(self) -> tuple[str, ...]:
        """Ports explicitly linked to a fully realized current excitation intent."""

        if not self.excitation_realized:
            return ()
        return tuple(
            assignment.port_id
            for assignment in self.element_assignments
            if assignment.status == "realized" and assignment.port_id is not None
        )


@dataclass(frozen=True, slots=True)
class ArraySpec:
    rows: int = 1
    columns: int = 1
    spacing_mm: float = 0.0
    row_vector: tuple[float, float, float] = (0.0, 1.0, 0.0)
    column_vector: tuple[float, float, float] = (1.0, 0.0, 0.0)

    @property
    def element_count(self) -> int:
        return self.rows * self.columns


def element_coordinate(array: ArraySpec, ordinal: int) -> tuple[int, int] | None:
    """Map the existing 1-based recipe port ordinal to canonical array coordinates."""

    if ordinal < 1 or array.columns < 1 or ordinal > array.element_count:
        return None
    return ((ordinal - 1) // array.columns + 1, (ordinal - 1) % array.columns + 1)


def _recipe_excitation_definition(
    array: ArraySpec,
    ports: tuple[PortSpec, ...],
) -> ExcitationDefinition:
    """Describe recipe-generated ports factually without inferring array feed intent."""

    if array.element_count == 1 and len(ports) == 1 and element_coordinate(array, ports[0].element_index) == (1, 1):
        return ExcitationDefinition(
            strategy="single_element_feed",
            realization_status="realized",
            feed_network="none",
            element_assignments=(ExcitationAssignment(
                1, 1, "excitation_1_1", ports[0].port_id, "realized",
            ),),
        )
    assignments = tuple(
        ExcitationAssignment(
            coordinates[0], coordinates[1], f"legacy_excitation_{coordinates[0]}_{coordinates[1]}",
            port.port_id, "legacy",
        )
        for port in ports
        if (coordinates := element_coordinate(array, port.element_index)) is not None
    )
    requirements = () if len(assignments) == array.element_count else (
        "Legacy recipe ports do not map completely to the current array layout.",
    )
    return ExcitationDefinition(
        strategy="legacy_recipe",
        realization_status="legacy",
        feed_network="unresolved",
        element_assignments=assignments,
        unresolved_requirements=requirements,
    )


def recipe_excitation_definition(design: "AntennaDesign") -> ExcitationDefinition:
    """Public recipe finalizer kept separate from physical feed generation."""

    return _recipe_excitation_definition(design.array, design.ports)


@dataclass(frozen=True, slots=True)
class SimulationSetup:
    frequency_min_ghz: Scalar
    frequency_max_ghz: Scalar
    boundary: str = "expanded_open"
    solver_family: str = "time_domain"


@dataclass(frozen=True, slots=True)
class ValidationRecord:
    stage: str
    passed: bool
    messages: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SemanticGeometryTarget:
    """Stable selector for geometry owned by a recipe element."""

    role: str
    element_row: int
    element_column: int


@dataclass(frozen=True, slots=True)
class ComposedTargetSelector:
    """Role-based target set for one logical persisted feature."""

    role: str
    scope: str = "single"
    elements: tuple[tuple[int, int], ...] = ((1, 1),)


@dataclass(frozen=True, slots=True)
class ComposedToolCall:
    """One immutable registered call captured after successful execution."""

    operation_id: str
    name: str
    arguments_json: str
    semantic_target: SemanticGeometryTarget | None = None

    @classmethod
    def create(
        cls,
        name: str,
        arguments: Mapping[str, Any],
        *,
        semantic_target: SemanticGeometryTarget | None = None,
        operation_id: str | None = None,
    ) -> "ComposedToolCall":
        normalized_arguments = json.dumps(dict(arguments), sort_keys=True, separators=(",", ":"))
        if operation_id is None:
            target_payload = asdict(semantic_target) if semantic_target is not None else None
            identity_payload = json.dumps(
                {"name": str(name), "arguments": json.loads(normalized_arguments), "semantic_target": target_payload},
                sort_keys=True,
                separators=(",", ":"),
            )
            operation_id = "call_" + hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()[:16]
        return cls(
            operation_id=str(operation_id),
            name=str(name),
            arguments_json=normalized_arguments,
            semantic_target=semantic_target,
        )

    def arguments(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.arguments_json)
        except (TypeError, ValueError) as exc:
            raise DesignValidationError("A composed tool call contains malformed arguments.") from exc
        if not isinstance(payload, dict):
            raise DesignValidationError("Composed tool-call arguments must be a JSON object.")
        return payload


@dataclass(frozen=True, slots=True)
class ComposedOperationGroup:
    """Replayable, recipe-independent primitive composition."""

    group_id: str
    source_recipe_id: str
    source_family: str
    calls: tuple[ComposedToolCall, ...]
    target_selector: ComposedTargetSelector | None = None
    coordinate_frame: str = "world"


def _composed_calls_from_payload(payloads: Any) -> tuple[ComposedToolCall, ...]:
    """Load legacy calls with deterministic IDs; disambiguate only identical legacy calls."""

    calls: list[ComposedToolCall] = []
    legacy_counts: dict[str, int] = {}
    for payload in payloads:
        semantic_payload = payload.get("semantic_target")
        semantic_target = (
            SemanticGeometryTarget(
                role=str(semantic_payload["role"]),
                element_row=int(semantic_payload["element_row"]),
                element_column=int(semantic_payload["element_column"]),
            )
            if semantic_payload is not None else None
        )
        explicit_id = payload.get("operation_id")
        call = ComposedToolCall.create(
            str(payload["name"]),
            payload["arguments"],
            operation_id=str(explicit_id) if explicit_id is not None else None,
            semantic_target=semantic_target,
        )
        if explicit_id is None:
            occurrence = legacy_counts.get(call.operation_id, 0) + 1
            legacy_counts[call.operation_id] = occurrence
            if occurrence > 1:
                suffix = f"_{occurrence}"
                call = replace(
                    call,
                    operation_id=f"{call.operation_id[:64 - len(suffix)]}{suffix}",
                )
        calls.append(call)
    return tuple(calls)


def _composed_group_from_payload(payload: Mapping[str, Any]) -> ComposedOperationGroup:
    """Load one group and give legacy groups an explicit single-element scope."""

    calls = _composed_calls_from_payload(payload["calls"])
    selector_payload = payload.get("target_selector")
    if selector_payload is not None:
        selector = ComposedTargetSelector(
            role=str(selector_payload["role"]),
            scope=str(selector_payload.get("scope", "single")),
            elements=tuple(
                (int(element[0]), int(element[1]))
                for element in selector_payload.get("elements", ((1, 1),))
            ),
        )
    else:
        targets = {
            (call.semantic_target.role,
             call.semantic_target.element_row,
             call.semantic_target.element_column)
            for call in calls
            if call.semantic_target is not None
        }
        selector = None
        if len(targets) == 1:
            role, row, column = next(iter(targets))
            selector = ComposedTargetSelector(role, "single", ((row, column),))
    return ComposedOperationGroup(
        group_id=str(payload["group_id"]),
        source_recipe_id=str(payload["source_recipe_id"]),
        source_family=str(payload["source_family"]),
        target_selector=selector,
        coordinate_frame=str(payload.get("coordinate_frame", "world")),
        calls=calls,
    )


@dataclass(frozen=True, slots=True)
class AntennaDesign:
    """Canonical design state; no field contains a solver command."""

    schema_version: int
    design_id: str
    revision: int
    recipe_id: str
    family: str
    display_name: str
    parameters: tuple[DesignParameter, ...]
    materials: tuple[MaterialSpec, ...]
    geometry: tuple[GeometryObject, ...]
    booleans: tuple[BooleanOperation, ...]
    ports: tuple[PortSpec, ...]
    array: ArraySpec
    simulation: SimulationSetup
    excitation: ExcitationDefinition = field(default_factory=ExcitationDefinition)
    composed_operations: tuple[ComposedOperationGroup, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()
    validation: tuple[ValidationRecord, ...] = ()

    @classmethod
    def empty(
        cls,
        *,
        recipe_id: str,
        family: str,
        display_name: str,
        design_id: str | None = None,
        revision: int = 0,
    ) -> "AntennaDesign":
        return cls(
            schema_version=SCHEMA_VERSION,
            design_id=design_id or uuid.uuid4().hex,
            revision=revision,
            recipe_id=recipe_id,
            family=family,
            display_name=display_name,
            parameters=(),
            materials=(),
            geometry=(),
            booleans=(),
            ports=(),
            array=ArraySpec(),
            simulation=SimulationSetup(1.0, 2.0),
        )

    @classmethod
    def starting_design(cls) -> "AntennaDesign":
        from studio.antenna_agent import create_default_agent

        return create_default_agent().create_design("inset_patch")

    def parameter_map(self) -> dict[str, DesignParameter]:
        return {parameter.key: parameter for parameter in self.parameters}

    def parameter(self, key: str) -> DesignParameter:
        try:
            return self.parameter_map()[key]
        except KeyError as exc:
            raise DesignValidationError(f"Unknown design parameter: {key}.") from exc

    def value(self, key: str) -> float:
        return resolve_parameter_values(self)[key]

    def metadata_map(self) -> dict[str, str]:
        return dict(self.metadata)

    @property
    def topology_label(self) -> str:
        if self.array.element_count == 1:
            return self.display_name
        kind = "linear" if self.array.rows == 1 or self.array.columns == 1 else "planar"
        return f"{self.array.rows} × {self.array.columns} {kind} {self.display_name.lower()} array"

    @property
    def template_id(self) -> str:
        return self.recipe_id

    @property
    def frequency_ghz(self) -> float:
        return self.value("frequency_ghz")

    @property
    def material(self) -> str:
        return self.metadata_map().get("substrate_material", "No substrate")

    @property
    def epsilon_r(self) -> float:
        substrates = [material for material in self.materials if material.kind == "dielectric"]
        return substrates[0].epsilon_r if substrates else 1.0

    @property
    def loss_tangent(self) -> float:
        substrates = [material for material in self.materials if material.kind == "dielectric"]
        return substrates[0].loss_tangent if substrates else 0.0

    @property
    def array_rows(self) -> int:
        return self.array.rows

    @property
    def array_columns(self) -> int:
        return self.array.columns

    @property
    def element_spacing_mm(self) -> float:
        return self.array.spacing_mm

    @property
    def element_spacing_lambda(self) -> float:
        wavelength = 299.792458 / self.frequency_ghz
        return self.array.spacing_mm / wavelength if wavelength else 0.0

    @property
    def substrate_thickness_mm(self) -> float:
        return self.value("substrate_thickness_mm")

    @property
    def copper_thickness_mm(self) -> float:
        return self.value("copper_thickness_mm")

    @property
    def patch_length_mm(self) -> float:
        return self.value("patch_length_mm")

    @property
    def patch_width_mm(self) -> float:
        return self.value("patch_width_mm")

    @property
    def feed_width_mm(self) -> float:
        return self.value("feed_width_mm")

    @property
    def inset_depth_mm(self) -> float:
        return self.value("inset_depth_mm")

    @property
    def inset_gap_mm(self) -> float:
        return self.value("inset_gap_mm")

    @property
    def board_margin_mm(self) -> float:
        return self.value("board_margin_mm")

    @property
    def estimated_resonance_ghz(self) -> float:
        return self.frequency_ghz

    @property
    def board_width_mm(self) -> float:
        return design_bounds(self)[1] - design_bounds(self)[0]

    @property
    def board_length_mm(self) -> float:
        return design_bounds(self)[3] - design_bounds(self)[2]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["composed_operations"] = [
            {
                "group_id": group.group_id,
                "source_recipe_id": group.source_recipe_id,
                "source_family": group.source_family,
                "target_selector": asdict(group.target_selector) if group.target_selector else None,
                "coordinate_frame": group.coordinate_frame,
                "calls": [
                    {
                        "operation_id": call.operation_id,
                        "name": call.name,
                        "arguments": call.arguments(),
                        "semantic_target": asdict(call.semantic_target) if call.semantic_target else None,
                    }
                    for call in group.calls
                ],
            }
            for group in self.composed_operations
        ]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AntennaDesign":
        if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
            raise DesignValidationError("Unsupported antenna design schema version.")
        try:
            return cls(
                schema_version=SCHEMA_VERSION,
                design_id=str(payload["design_id"]),
                revision=int(payload["revision"]),
                recipe_id=str(payload["recipe_id"]),
                family=str(payload["family"]),
                display_name=str(payload["display_name"]),
                parameters=tuple(DesignParameter(**item) for item in payload["parameters"]),
                materials=tuple(MaterialSpec(**item) for item in payload["materials"]),
                geometry=tuple(
                    GeometryObject(
                        object_id=str(item["object_id"]),
                        primitive=str(item["primitive"]),
                        name=str(item["name"]),
                        material_id=str(item["material_id"]),
                        dimensions=tuple((str(k), v) for k, v in item["dimensions"]),
                        axis=str(item.get("axis", "z")),
                        transform=TransformSpec(
                            translate_mm=tuple(item.get("transform", {}).get("translate_mm", (0.0, 0.0, 0.0))),
                            rotate_deg=tuple(item.get("transform", {}).get("rotate_deg", (0.0, 0.0, 0.0))),
                        ),
                        component=str(item.get("component", "Antenna")),
                        tags=tuple(item.get("tags", ())),
                    )
                    for item in payload["geometry"]
                ),
                booleans=tuple(
                    BooleanOperation(
                        operation_id=str(item["operation_id"]),
                        operation=str(item["operation"]),
                        target_id=str(item["target_id"]),
                        tool_ids=tuple(item["tool_ids"]),
                    )
                    for item in payload.get("booleans", ())
                ),
                ports=tuple(
                    PortSpec(
                        **{
                            **item,
                            "positive_point": tuple(item["positive_point"]),
                            "negative_point": tuple(item["negative_point"]),
                        }
                    )
                    for item in payload["ports"]
                ),
                array=ArraySpec(
                    rows=int(payload["array"]["rows"]),
                    columns=int(payload["array"]["columns"]),
                    spacing_mm=float(payload["array"]["spacing_mm"]),
                    row_vector=tuple(payload["array"].get("row_vector", (0.0, 1.0, 0.0))),
                    column_vector=tuple(payload["array"].get("column_vector", (1.0, 0.0, 0.0))),
                ),
                simulation=SimulationSetup(**payload["simulation"]),
                excitation=(
                    ExcitationDefinition(
                        strategy=str(payload["excitation"].get("strategy", "unresolved")),
                        realization_status=str(payload["excitation"].get("realization_status", "unresolved")),
                        feed_network=str(payload["excitation"].get("feed_network", "unresolved")),
                        element_assignments=tuple(
                            ExcitationAssignment(
                                element_row=int(item["element_row"]),
                                element_column=int(item["element_column"]),
                                excitation_id=str(item["excitation_id"]),
                                port_id=str(item["port_id"]) if item.get("port_id") is not None else None,
                                status=str(item.get("status", "unresolved")),
                            )
                            for item in payload["excitation"].get("element_assignments", ())
                        ),
                        unresolved_requirements=tuple(
                            str(item) for item in payload["excitation"].get("unresolved_requirements", ())
                        ),
                    )
                    if payload.get("excitation") is not None
                    else _recipe_excitation_definition(
                        ArraySpec(
                            rows=int(payload["array"]["rows"]),
                            columns=int(payload["array"]["columns"]),
                            spacing_mm=float(payload["array"]["spacing_mm"]),
                            row_vector=tuple(payload["array"].get("row_vector", (0.0, 1.0, 0.0))),
                            column_vector=tuple(payload["array"].get("column_vector", (1.0, 0.0, 0.0))),
                        ),
                        tuple(
                            PortSpec(
                                **{
                                    **item,
                                    "positive_point": tuple(item["positive_point"]),
                                    "negative_point": tuple(item["negative_point"]),
                                }
                            )
                            for item in payload["ports"]
                        ),
                    )
                ),
                composed_operations=tuple(
                    _composed_group_from_payload(group)
                    for group in payload.get("composed_operations", ())
                ),
                metadata=tuple((str(k), str(v)) for k, v in payload.get("metadata", ())),
                validation=tuple(
                    ValidationRecord(
                        stage=str(item["stage"]),
                        passed=bool(item["passed"]),
                        messages=tuple(item.get("messages", ())),
                    )
                    for item in payload.get("validation", ())
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DesignValidationError("The saved antenna design is malformed.") from exc


def _eval_node(node: ast.AST, values: Mapping[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, values)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in values:
        return float(values[node.id])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand, values)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left = _eval_node(node.left, values)
        right = _eval_node(node.right, values)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right
    raise DesignValidationError("Expressions may contain only parameter names and basic arithmetic.")


def evaluate_scalar(value: Scalar, parameter_values: Mapping[str, float]) -> float:
    if isinstance(value, bool):
        raise DesignValidationError("Boolean values are not valid geometry dimensions.")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            tree = ast.parse(value, mode="eval")
            number = _eval_node(tree, parameter_values)
        except RecursionError as exc:
            raise DesignValidationError(
                "Parameter expression is too deeply nested."
            ) from exc
        except (SyntaxError, ZeroDivisionError) as exc:
            raise DesignValidationError(f"Invalid parameter expression: {value}.") from exc
    else:
        raise DesignValidationError("Geometry dimensions must be numeric or parameter expressions.")
    if not math.isfinite(number):
        raise DesignValidationError("Geometry dimensions must resolve to finite values.")
    return number


def resolve_parameter_values(design: AntennaDesign) -> dict[str, float]:
    values = {parameter.key: float(parameter.value) for parameter in design.parameters}
    pending = {parameter.key: parameter.expression for parameter in design.parameters if parameter.expression}
    for _ in range(len(pending) + 1):
        progressed = False
        for key, expression in list(pending.items()):
            try:
                values[key] = evaluate_scalar(str(expression), values)
            except DesignValidationError:
                continue
            del pending[key]
            progressed = True
        if not pending or not progressed:
            break
    if pending:
        raise DesignValidationError(
            "Unresolved parameter expression(s): " + ", ".join(sorted(pending)) + "."
        )
    return values


def resolved_dimensions(design: AntennaDesign, geometry: GeometryObject) -> dict[str, float]:
    values = resolve_parameter_values(design)
    return {key: evaluate_scalar(value, values) for key, value in geometry.dimensions}


def _rotate_vector(
    vector: tuple[float, float, float],
    angles_deg: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Apply canonical X, then Y, then Z rotations about the global origin."""

    x, y, z = vector
    ax, ay, az = (math.radians(value) for value in angles_deg)
    y, z = y * math.cos(ax) - z * math.sin(ax), y * math.sin(ax) + z * math.cos(ax)
    x, z = x * math.cos(ay) + z * math.sin(ay), -x * math.sin(ay) + z * math.cos(ay)
    x, y = x * math.cos(az) - y * math.sin(az), x * math.sin(az) + y * math.cos(az)
    return x, y, z


def object_bounds(design: AntennaDesign, geometry: GeometryObject) -> tuple[float, float, float, float, float, float]:
    dimensions = resolved_dimensions(design, geometry)
    tx, ty, tz = geometry.transform.translate_mm
    if geometry.primitive in {"box", "sheet_rectangle"}:
        corners = [
            _rotate_vector((x, y, z), geometry.transform.rotate_deg)
            for x in (dimensions["x_min"], dimensions["x_max"])
            for y in (dimensions["y_min"], dimensions["y_max"])
            for z in (dimensions["z_min"], dimensions["z_max"])
        ]
        return (
            min(point[0] for point in corners) + tx,
            max(point[0] for point in corners) + tx,
            min(point[1] for point in corners) + ty,
            max(point[1] for point in corners) + ty,
            min(point[2] for point in corners) + tz,
            max(point[2] for point in corners) + tz,
        )
    if geometry.primitive in {"cylinder", "sheet_circle"}:
        radius = dimensions["radius"]
        axis = geometry.axis.casefold()
        center_1 = dimensions.get("center_1", 0.0)
        center_2 = dimensions.get("center_2", 0.0)
        start = dimensions["start"]
        end = dimensions["end"]
        if axis == "x":
            first, second, local_axis = (start, center_1, center_2), (end, center_1, center_2), (1.0, 0.0, 0.0)
        elif axis == "y":
            first, second, local_axis = (center_1, start, center_2), (center_1, end, center_2), (0.0, 1.0, 0.0)
        else:
            first, second, local_axis = (center_1, center_2, start), (center_1, center_2, end), (0.0, 0.0, 1.0)
        first = _rotate_vector(first, geometry.transform.rotate_deg)
        second = _rotate_vector(second, geometry.transform.rotate_deg)
        rotated_axis = _rotate_vector(local_axis, geometry.transform.rotate_deg)
        radial = tuple(radius * math.sqrt(max(0.0, 1.0 - component * component)) for component in rotated_axis)
        offsets = (tx, ty, tz)
        return tuple(
            value
            for index in range(3)
            for value in (
                min(first[index], second[index]) - radial[index] + offsets[index],
                max(first[index], second[index]) + radial[index] + offsets[index],
            )
        )
    raise DesignValidationError(f"Unsupported primitive for bounds: {geometry.primitive}.")


def design_bounds(design: AntennaDesign) -> tuple[float, float, float, float, float, float]:
    subtract_tools = {
        tool_id
        for operation in design.booleans
        if operation.operation == "subtract"
        for tool_id in operation.tool_ids
    }
    visible_geometry = [item for item in design.geometry if item.object_id not in subtract_tools]
    if not visible_geometry:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    bounds = [object_bounds(design, item) for item in visible_geometry]
    return (
        min(item[0] for item in bounds),
        max(item[1] for item in bounds),
        min(item[2] for item in bounds),
        max(item[3] for item in bounds),
        min(item[4] for item in bounds),
        max(item[5] for item in bounds),
    )


def with_validation(design: AntennaDesign, records: tuple[ValidationRecord, ...]) -> AntennaDesign:
    return replace(design, validation=records)
