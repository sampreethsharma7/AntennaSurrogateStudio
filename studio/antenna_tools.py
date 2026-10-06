"""Composable antenna tools and the discoverable capability registry."""

from __future__ import annotations

import importlib
import math
import pkgutil
import re
from dataclasses import dataclass, replace
from importlib import metadata
from typing import Any, Callable, Iterable, Protocol

from studio.antenna_analysis import (
    EngineeringAnalysisResult,
    array_spacing_analysis,
    design_summary_analysis,
    dipole_baseline_analysis,
    rectangular_patch_baseline_analysis,
)

from studio.antenna_design import (
    AntennaDesign,
    ArraySpec,
    BooleanOperation,
    DesignParameter,
    GeometryObject,
    MaterialSpec,
    PortSpec,
    PortCrossSection,
    SimulationSetup,
    TransformSpec,
)


class CapabilityError(ValueError):
    """Raised when a requested tool or recipe is unavailable or invalid."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    category: str
    description: str
    validation_level: str
    handler: Callable[[AntennaDesign, dict[str, Any]], AntennaDesign | EngineeringAnalysisResult]
    planner_exposed: bool = False
    argument_schema: dict[str, Any] | None = None
    effect: str = "design_action"
    tool_version: str = "1"
    requires_design: bool = True

    def __post_init__(self) -> None:
        if self.effect not in {"design_action", "analysis"}:
            raise ValueError("Registered tool effect must be design_action or analysis.")
        if not isinstance(self.tool_version, str) or not self.tool_version.strip():
            raise ValueError("Registered tool version must be a nonempty string.")


class Recipe(Protocol):
    recipe_id: str
    family: str
    aliases: tuple[str, ...]
    required_tools: tuple[str, ...]

    def build(
        self,
        registry: "ToolRegistry",
        values: dict[str, Any],
        *,
        design_id: str | None = None,
        revision: int = 0,
    ) -> tuple[AntennaDesign, tuple[ToolCall, ...]]: ...

    def defaults(self) -> dict[str, Any]: ...

    def parameter_definitions(self) -> tuple[Any, ...]: ...


class DesignModifier(Protocol):
    """A reusable composition applied after a base antenna recipe."""

    modifier_id: str
    aliases: tuple[str, ...]
    applicable_families: tuple[str, ...]
    required_tools: tuple[str, ...]

    def defaults(self) -> dict[str, Any]: ...

    def parameter_definitions(self) -> tuple[Any, ...]: ...

    def apply(
        self,
        registry: "ToolRegistry",
        design: AntennaDesign,
        values: dict[str, Any],
    ) -> tuple[AntennaDesign, tuple[ToolCall, ...]]: ...


class ToolRegistry:
    """Runtime registry; capability modules can register without core edits."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._recipes: dict[str, Recipe] = {}
        self._modifiers: dict[str, DesignModifier] = {}

    def register_tool(self, definition: ToolDefinition) -> None:
        if definition.name in self._tools:
            raise CapabilityError(f"Tool already registered: {definition.name}.")
        self._tools[definition.name] = definition

    def register_recipe(self, recipe: Recipe) -> None:
        if recipe.recipe_id in self._recipes:
            raise CapabilityError(f"Recipe already registered: {recipe.recipe_id}.")
        missing = sorted(set(recipe.required_tools) - set(self._tools))
        if missing:
            raise CapabilityError(
                f"Recipe {recipe.recipe_id} requires unavailable tools: {', '.join(missing)}."
            )
        self._recipes[recipe.recipe_id] = recipe

    def register_modifier(self, modifier: DesignModifier) -> None:
        if modifier.modifier_id in self._modifiers:
            raise CapabilityError(f"Modifier already registered: {modifier.modifier_id}.")
        missing = sorted(set(modifier.required_tools) - set(self._tools))
        if missing:
            raise CapabilityError(
                f"Modifier {modifier.modifier_id} requires unavailable tools: {', '.join(missing)}."
            )
        self._modifiers[modifier.modifier_id] = modifier

    def execute(self, design: AntennaDesign, call: ToolCall) -> AntennaDesign:
        try:
            definition = self._tools[call.name]
        except KeyError as exc:
            raise CapabilityError(f"Tool is not installed: {call.name}.") from exc
        if definition.effect != "design_action":
            raise CapabilityError(f"Tool {call.name} is read-only analysis and cannot execute as a design action.")
        result = definition.handler(design, dict(call.arguments))
        if not isinstance(result, AntennaDesign):
            raise CapabilityError(f"Design-action tool {call.name} returned an invalid result.")
        return result

    def execute_analysis(self, design: AntennaDesign, call: ToolCall) -> EngineeringAnalysisResult:
        try:
            definition = self._tools[call.name]
        except KeyError as exc:
            raise CapabilityError(f"Tool is not installed: {call.name}.") from exc
        if definition.effect != "analysis":
            raise CapabilityError(f"Tool {call.name} is a design action and cannot execute as analysis.")
        result = definition.handler(design, dict(call.arguments))
        if not isinstance(result, EngineeringAnalysisResult):
            raise CapabilityError(f"Analysis tool {call.name} returned an invalid result.")
        return result

    def execute_all(
        self,
        design: AntennaDesign,
        calls: Iterable[ToolCall],
    ) -> AntennaDesign:
        for call in calls:
            design = self.execute(design, call)
        return design

    def tool(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise CapabilityError(f"Tool is not installed: {name}.") from exc

    def recipe(self, recipe_id: str) -> Recipe:
        try:
            return self._recipes[recipe_id]
        except KeyError as exc:
            raise CapabilityError(f"Antenna recipe is not installed: {recipe_id}.") from exc

    def recipes(self) -> tuple[Recipe, ...]:
        return tuple(self._recipes.values())

    def modifier(self, modifier_id: str) -> DesignModifier:
        try:
            return self._modifiers[modifier_id]
        except KeyError as exc:
            raise CapabilityError(f"Design modifier is not installed: {modifier_id}.") from exc

    def modifiers(self) -> tuple[DesignModifier, ...]:
        return tuple(self._modifiers.values())

    def tools(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._tools.values())

    def find_recipe(self, text: str) -> Recipe | None:
        normalized = text.casefold()
        matches = [
            recipe
            for recipe in self._recipes.values()
            if recipe.family.casefold() in normalized
            or any(alias.casefold() in normalized for alias in recipe.aliases)
        ]
        return max(matches, key=lambda item: max((len(alias) for alias in item.aliases), default=0)) if matches else None


def validate_tool_arguments(schema: dict[str, Any], payload: Any, *, path: str = "arguments") -> None:
    """Validate the JSON-Schema subset used by registered planner tools."""

    expected = schema.get("type")
    expected_types = tuple(expected) if isinstance(expected, list) else (expected,)

    def matches_type(kind: str) -> bool:
        if kind == "object":
            return isinstance(payload, dict)
        if kind == "array":
            return isinstance(payload, (list, tuple))
        if kind == "string":
            return isinstance(payload, str)
        if kind == "number":
            return isinstance(payload, (int, float)) and not isinstance(payload, bool) and math.isfinite(float(payload))
        if kind == "integer":
            return isinstance(payload, int) and not isinstance(payload, bool)
        if kind == "boolean":
            return isinstance(payload, bool)
        return False

    if expected and not any(matches_type(kind) for kind in expected_types):
        raise CapabilityError(f"{path} does not match the registered schema type.")
    if "const" in schema and payload != schema["const"]:
        raise CapabilityError(f"{path} does not match the registered constant.")
    if "enum" in schema and payload not in schema["enum"]:
        raise CapabilityError(f"{path} is not one of the registered values.")
    if isinstance(payload, str):
        if schema.get("minLength") is not None and len(payload) < int(schema["minLength"]):
            raise CapabilityError(f"{path} is too short.")
        if schema.get("maxLength") is not None and len(payload) > int(schema["maxLength"]):
            raise CapabilityError(f"{path} is too long.")
        if schema.get("pattern") and re.fullmatch(str(schema["pattern"]), payload) is None:
            raise CapabilityError(f"{path} does not match the registered identifier pattern.")
    if isinstance(payload, (int, float)) and not isinstance(payload, bool):
        if schema.get("minimum") is not None and payload < schema["minimum"]:
            raise CapabilityError(f"{path} is below the registered minimum.")
        if schema.get("maximum") is not None and payload > schema["maximum"]:
            raise CapabilityError(f"{path} is above the registered maximum.")
    if isinstance(payload, dict):
        properties = schema.get("properties", {})
        missing = set(schema.get("required", ())) - set(payload)
        if missing:
            raise CapabilityError(f"{path} is missing required field(s): {', '.join(sorted(missing))}.")
        if schema.get("additionalProperties") is False:
            extra = set(payload) - set(properties)
            if extra:
                raise CapabilityError(f"{path} contains unregistered field(s): {', '.join(sorted(extra))}.")
        for key, value in payload.items():
            if key in properties:
                validate_tool_arguments(properties[key], value, path=f"{path}.{key}")
    if isinstance(payload, (list, tuple)):
        if schema.get("minItems") is not None and len(payload) < int(schema["minItems"]):
            raise CapabilityError(f"{path} contains too few items.")
        if schema.get("maxItems") is not None and len(payload) > int(schema["maxItems"]):
            raise CapabilityError(f"{path} contains too many items.")
        if schema.get("uniqueItems") and len({repr(item) for item in payload}) != len(payload):
            raise CapabilityError(f"{path} must contain unique items.")
        item_schema = schema.get("items")
        if item_schema:
            for index, value in enumerate(payload):
                validate_tool_arguments(item_schema, value, path=f"{path}[{index}]")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise CapabilityError(f"{label} must be numeric.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CapabilityError(f"{label} must be numeric.") from exc
    if not math.isfinite(number):
        raise CapabilityError(f"{label} must be finite.")
    return number


def _parameter(design: AntennaDesign, args: dict[str, Any]) -> AntennaDesign:
    parameter = DesignParameter(
        key=str(args["key"]),
        name=str(args.get("name", args["key"])),
        label=str(args.get("label", args["key"])),
        value=_finite(args["value"], str(args["key"])),
        unit=str(args.get("unit", "")),
        expression=str(args["expression"]) if args.get("expression") else None,
        editable=bool(args.get("editable", True)),
        sweepable=bool(args.get("sweepable", False)),
        integer=bool(args.get("integer", False)),
        minimum=float(args["minimum"]) if args.get("minimum") is not None else None,
        maximum=float(args["maximum"]) if args.get("maximum") is not None else None,
        sweep_lower_factor=float(args.get("sweep_lower_factor", 0.9)),
        sweep_upper_factor=float(args.get("sweep_upper_factor", 1.1)),
    )
    parameters = [item for item in design.parameters if item.key != parameter.key]
    parameters.append(parameter)
    return replace(design, parameters=tuple(parameters))


def _material(design: AntennaDesign, args: dict[str, Any]) -> AntennaDesign:
    material = MaterialSpec(
        material_id=str(args["material_id"]),
        name=str(args["name"]),
        kind=str(args["kind"]),
        epsilon_r=_finite(args.get("epsilon_r", 1.0), "epsilon_r"),
        loss_tangent=_finite(args.get("loss_tangent", 0.0), "loss_tangent"),
        conductivity_s_per_m=(
            _finite(args["conductivity_s_per_m"], "conductivity")
            if args.get("conductivity_s_per_m") is not None
            else None
        ),
    )
    materials = [item for item in design.materials if item.material_id != material.material_id]
    materials.append(material)
    return replace(design, materials=tuple(materials))


def _geometry(primitive: str) -> Callable[[AntennaDesign, dict[str, Any]], AntennaDesign]:
    required_by_primitive = {
        "box": {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"},
        "sheet_rectangle": {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"},
        "cylinder": {"center_1", "center_2", "radius", "start", "end"},
        "sheet_circle": {"center_1", "center_2", "radius", "start", "end"},
    }

    def add(design: AntennaDesign, args: dict[str, Any]) -> AntennaDesign:
        object_id = str(args["object_id"])
        if any(item.object_id == object_id for item in design.geometry):
            raise CapabilityError(f"Geometry object already exists: {object_id}.")
        material_id = str(args["material_id"])
        if material_id not in {item.material_id for item in design.materials}:
            raise CapabilityError(f"Unknown material: {material_id}.")
        dimensions = dict(args.get("dimensions", {}))
        missing = required_by_primitive[primitive] - set(dimensions)
        if missing:
            raise CapabilityError(
                f"{primitive} is missing dimension(s): {', '.join(sorted(missing))}."
            )
        axis = str(args.get("axis", "z")).casefold()
        if primitive in {"cylinder", "sheet_circle"} and axis not in {"x", "y", "z"}:
            raise CapabilityError("Cylinder axis must be x, y, or z.")
        geometry = GeometryObject(
            object_id=object_id,
            primitive=primitive,
            name=str(args.get("name", object_id)),
            material_id=material_id,
            dimensions=tuple((str(key), value) for key, value in dimensions.items()),
            axis=axis,
            component=str(args.get("component", "Antenna")),
            tags=tuple(str(tag) for tag in args.get("tags", ())),
        )
        return replace(design, geometry=(*design.geometry, geometry))

    return add


def _translate(design: AntennaDesign, args: dict[str, Any]) -> AntennaDesign:
    object_id = str(args["object_id"])
    delta = tuple(_finite(value, "translation") for value in args["offset_mm"])
    if len(delta) != 3:
        raise CapabilityError("Translation requires three coordinates.")
    found = False
    geometry = []
    for item in design.geometry:
        if item.object_id == object_id:
            current = item.transform.translate_mm
            item = replace(
                item,
                transform=replace(
                    item.transform,
                    translate_mm=tuple(current[index] + delta[index] for index in range(3)),
                ),
            )
            found = True
        geometry.append(item)
    if not found:
        raise CapabilityError(f"Unknown geometry object: {object_id}.")
    return replace(design, geometry=tuple(geometry))


def _rotate(design: AntennaDesign, args: dict[str, Any]) -> AntennaDesign:
    object_id = str(args["object_id"])
    angles = tuple(_finite(value, "rotation") for value in args["angles_deg"])
    if len(angles) != 3:
        raise CapabilityError("Rotation requires three angles.")
    found = False
    geometry = []
    for item in design.geometry:
        if item.object_id == object_id:
            current = item.transform.rotate_deg
            item = replace(
                item,
                transform=replace(
                    item.transform,
                    rotate_deg=tuple(current[index] + angles[index] for index in range(3)),
                ),
            )
            found = True
        geometry.append(item)
    if not found:
        raise CapabilityError(f"Unknown geometry object: {object_id}.")
    return replace(design, geometry=tuple(geometry))


def _duplicate(design: AntennaDesign, args: dict[str, Any]) -> AntennaDesign:
    source_id = str(args["source_id"])
    new_id = str(args["new_id"])
    if any(item.object_id == new_id for item in design.geometry):
        raise CapabilityError(f"Geometry object already exists: {new_id}.")
    source = next((item for item in design.geometry if item.object_id == source_id), None)
    if source is None:
        raise CapabilityError(f"Unknown geometry object: {source_id}.")
    copied = replace(source, object_id=new_id, name=str(args.get("name", new_id)))
    design = replace(design, geometry=(*design.geometry, copied))
    if args.get("offset_mm") is not None:
        design = _translate(design, {"object_id": new_id, "offset_mm": args["offset_mm"]})
    return design


def _boolean(operation: str) -> Callable[[AntennaDesign, dict[str, Any]], AntennaDesign]:
    def add(design: AntennaDesign, args: dict[str, Any]) -> AntennaDesign:
        known = {item.object_id for item in design.geometry}
        target = str(args["target_id"])
        tools = tuple(str(value) for value in args["tool_ids"])
        missing = ({target, *tools} - known)
        if missing:
            raise CapabilityError("Boolean references unknown object(s): " + ", ".join(sorted(missing)) + ".")
        if target in tools:
            raise CapabilityError("A Boolean target cannot also be its tool object.")
        operation_id = str(args.get("operation_id", f"{operation}_{len(design.booleans) + 1}"))
        if any(item.operation_id == operation_id for item in design.booleans):
            raise CapabilityError(f"Boolean operation already exists: {operation_id}.")
        item = BooleanOperation(
            operation_id=operation_id,
            operation=operation,
            target_id=target,
            tool_ids=tools,
        )
        return replace(design, booleans=(*design.booleans, item))

    return add


def _assign_material(design: AntennaDesign, args: dict[str, Any]) -> AntennaDesign:
    object_id = str(args["object_id"])
    material_id = str(args["material_id"])
    if material_id not in {item.material_id for item in design.materials}:
        raise CapabilityError(f"Unknown material: {material_id}.")
    found = False
    geometry = []
    for item in design.geometry:
        if item.object_id == object_id:
            item = replace(item, material_id=material_id)
            found = True
        geometry.append(item)
    if not found:
        raise CapabilityError(f"Unknown geometry object: {object_id}.")
    return replace(design, geometry=tuple(geometry))


def _port(design: AntennaDesign, args: dict[str, Any]) -> AntennaDesign:
    port_id = str(args["port_id"])
    if any(item.port_id == port_id for item in design.ports):
        raise CapabilityError(f"Port already exists: {port_id}.")
    positive = tuple(args["positive_point"])
    negative = tuple(args["negative_point"])
    if len(positive) != 3 or len(negative) != 3:
        raise CapabilityError("Port endpoints require three coordinates.")
    cross_section_payload = args.get("cross_section")
    cross_section = None
    if cross_section_payload is not None:
        if not isinstance(cross_section_payload, dict):
            raise CapabilityError("Port cross-section must be a structured geometry reference.")
        cross_section = PortCrossSection(
            geometry_id=str(cross_section_payload["geometry_id"]),
            face=str(cross_section_payload["face"]),
        )
    port = PortSpec(
        port_id=port_id,
        name=str(args.get("name", port_id)),
        kind=str(args.get("kind", "discrete")),
        positive_point=positive,
        negative_point=negative,
        impedance_ohms=_finite(args.get("impedance_ohms", 50.0), "port impedance"),
        element_index=int(args.get("element_index", len(design.ports) + 1)),
        signal_terminal=(str(args["signal_terminal"]) if args.get("signal_terminal") is not None else None),
        reference_terminal=(str(args["reference_terminal"]) if args.get("reference_terminal") is not None else None),
        cross_section=cross_section,
        mode_count=int(args.get("mode_count", 1)),
    )
    return replace(design, ports=(*design.ports, port))


def _frequency(design: AntennaDesign, args: dict[str, Any]) -> AntennaDesign:
    return replace(
        design,
        simulation=SimulationSetup(
            frequency_min_ghz=args["minimum_ghz"],
            frequency_max_ghz=args["maximum_ghz"],
            boundary=str(args.get("boundary", "expanded_open")),
            solver_family=str(args.get("solver_family", "time_domain")),
        ),
    )


def _array(design: AntennaDesign, args: dict[str, Any]) -> AntennaDesign:
    rows = int(args.get("rows", 1))
    columns = int(args.get("columns", 1))
    if rows < 1 or columns < 1 or rows * columns > 64:
        raise CapabilityError("Arrays require 1-64 elements with positive rows and columns.")
    return replace(
        design,
        array=ArraySpec(rows=rows, columns=columns, spacing_mm=_finite(args.get("spacing_mm", 0.0), "array spacing")),
    )


def _metadata(design: AntennaDesign, args: dict[str, Any]) -> AntennaDesign:
    metadata_values = dict(design.metadata)
    metadata_values[str(args["key"])] = str(args["value"])
    return replace(design, metadata=tuple(metadata_values.items()))


def register_primitive_tools(registry: ToolRegistry) -> None:
    identifier = {"type": "string", "pattern": r"[A-Za-z][A-Za-z0-9_]{0,63}"}
    tool_identifier = {"type": "string", "pattern": r"[a-z][a-z0-9_]{0,58}_tool"}
    operation_identifier = {"type": "string", "pattern": r"[a-z][a-z0-9_]{0,54}_(?:subtract|union)"}
    scalar = {"type": ["number", "string"]}
    parameter_expression = {
        "type": "string",
        "minLength": 1,
        "maxLength": 160,
        "pattern": r".*[A-Za-z_].*",
    }
    tags = {
        "type": "array",
        "minItems": 1,
        "maxItems": 4,
        "uniqueItems": True,
        "items": {"type": "string", "enum": ["planner_created", "boolean_tool", "slot", "custom_geometry"]},
    }
    conductor_z_min = {
        "type": "string",
        "const": "substrate_thickness_mm",
    }
    conductor_z_max = {
        "type": "string",
        "enum": [
            "substrate_thickness_mm+copper_thickness_mm",
            "substrate_thickness_mm + copper_thickness_mm",
        ],
    }
    cylinder_dimensions = {
        "type": "object", "additionalProperties": False,
        "required": ["center_1", "center_2", "radius", "start", "end"],
        "properties": {
            "center_1": scalar,
            "center_2": scalar,
            "radius": parameter_expression,
            "start": conductor_z_min,
            "end": conductor_z_max,
        },
    }
    rectangle_dimensions = {
        "type": "object", "additionalProperties": False,
        "required": ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"],
        "properties": {
            "x_min": parameter_expression,
            "x_max": parameter_expression,
            "y_min": parameter_expression,
            "y_max": parameter_expression,
            "z_min": conductor_z_min,
            "z_max": conductor_z_max,
        },
    }
    parameter_schema = {
        "type": "object", "additionalProperties": False,
        "required": ["key", "label", "value", "unit", "sweepable"],
        "properties": {
            "key": {"type": "string", "pattern": r"[a-z][a-z0-9_]{0,63}"},
            "label": {"type": "string", "minLength": 1, "maxLength": 80},
            "value": {"type": "number"},
            "unit": {"type": "string", "enum": ["", "mm"]},
            "expression": {"type": "string", "minLength": 1, "maxLength": 160},
            "editable": {"type": "boolean"},
            "sweepable": {"type": "boolean"},
            "minimum": {"type": "number"},
            "maximum": {"type": "number"},
            "sweep_lower_factor": {"type": "number", "minimum": 0.01, "maximum": 1.0},
            "sweep_upper_factor": {"type": "number", "minimum": 1.0, "maximum": 10.0},
        },
    }
    cylinder_schema = {
        "type": "object", "additionalProperties": False,
        "required": ["object_id", "material_id", "axis", "tags", "dimensions"],
        "properties": {
            "object_id": tool_identifier, "material_id": {"type": "string", "const": "copper"},
            "axis": {"type": "string", "enum": ["z"]}, "tags": tags,
            "dimensions": cylinder_dimensions,
        },
    }
    rectangle_schema = {
        "type": "object", "additionalProperties": False,
        "required": ["object_id", "material_id", "tags", "dimensions"],
        "properties": {
            "object_id": tool_identifier, "material_id": {"type": "string", "const": "copper"},
            "tags": tags, "dimensions": rectangle_dimensions,
        },
    }
    translate_schema = {
        "type": "object", "additionalProperties": False,
        "required": ["object_id", "offset_mm"],
        "properties": {
            "object_id": tool_identifier,
            "offset_mm": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}},
        },
    }
    rotate_schema = {
        "type": "object", "additionalProperties": False,
        "required": ["object_id", "angles_deg"],
        "properties": {
            "object_id": tool_identifier,
            "angles_deg": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}},
        },
    }
    duplicate_schema = {
        "type": "object", "additionalProperties": False,
        "required": ["source_id", "new_id", "offset_mm"],
        "properties": {
            "source_id": tool_identifier, "new_id": tool_identifier,
            "offset_mm": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}},
        },
    }
    boolean_schema = {
        "type": "object", "additionalProperties": False,
        "required": ["operation_id", "target_id", "tool_ids"],
        "properties": {
            "operation_id": operation_identifier, "target_id": identifier,
            "tool_ids": {"type": "array", "minItems": 1, "maxItems": 16, "uniqueItems": True, "items": tool_identifier},
        },
    }
    definitions = (
        ("parameter.create", "parameter", "Create a new numeric design parameter for planned geometry.", _parameter, parameter_schema),
        ("material.define", "em", "Define a solver-neutral material.", _material, None),
        ("material.assign", "em", "Assign a material to a geometry object.", _assign_material, None),
        ("geometry.box", "geometry", "Create an axis-aligned box.", _geometry("box"), None),
        ("geometry.rectangle_sheet", "geometry", "Create an x-y rectangular Boolean tool on the radiating-conductor z layer; x/y bounds are offsets in the selected Boolean target's local element frame.", _geometry("sheet_rectangle"), rectangle_schema),
        ("geometry.cylinder", "geometry", "Create a z-axis circular Boolean tool on the radiating-conductor z layer; center_1/center_2 are x/y offsets in the selected Boolean target's local element frame.", _geometry("cylinder"), cylinder_schema),
        ("geometry.circle_sheet", "geometry", "Create thin circular geometry with its center expressed in the selected Boolean target's local element frame.", _geometry("sheet_circle"), cylinder_schema),
        ("geometry.translate", "transform", "Translate geometry created by the same planner composition.", _translate, translate_schema),
        ("geometry.rotate", "transform", "Rotate geometry created by the same planner composition.", _rotate, rotate_schema),
        ("geometry.duplicate", "transform", "Duplicate planner-created geometry and translate the copy.", _duplicate, duplicate_schema),
        ("boolean.subtract", "boolean", "Subtract planner-created tools from a validated radiating conductor target.", _boolean("subtract"), boolean_schema),
        ("boolean.union", "boolean", "Union planner-created tools into a validated radiating conductor target.", _boolean("union"), boolean_schema),
        ("em.port.create", "em", "Create an excitation port.", _port, None),
        ("em.frequency.setup", "em", "Set the simulation frequency range.", _frequency, None),
        ("array.configure", "array", "Record array rows, columns, and spacing.", _array, None),
        ("design.metadata.set", "design", "Set solver-neutral design metadata.", _metadata, None),
    )
    for name, category, description, handler, planner_schema in definitions:
        registry.register_tool(
            ToolDefinition(
                name, category, description, "primitive", handler,
                planner_exposed=planner_schema is not None,
                argument_schema=planner_schema,
            )
        )

    registry.register_tool(ToolDefinition(
        "engineering.design_summary",
        "engineering_analysis",
        "Inspect the current canonical design inventory without modifying it.",
        "analysis",
        design_summary_analysis,
        planner_exposed=True,
        argument_schema={"type": "object", "additionalProperties": False, "properties": {}},
        effect="analysis",
        tool_version="1",
        requires_design=True,
    ))
    registry.register_tool(ToolDefinition(
        "engineering.array_spacing",
        "engineering_analysis",
        "Calculate canonical row/column spacing in free-space wavelengths; optionally test visible integer spatial orders for one scan angle in ideal periodic principal planes. This does not model finite-array, element-pattern, coupling, feed-network, substrate, surface-wave, or full-wave effects.",
        "analysis",
        array_spacing_analysis,
        planner_exposed=True,
        argument_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "scan_angle_deg": {"type": "number", "minimum": -90, "maximum": 90},
                "principal_axis": {"type": "string", "enum": ["all_active", "row", "column"]},
            },
        },
        effect="analysis",
        tool_version="1",
        requires_design=True,
    ))
    registry.register_tool(ToolDefinition(
        "engineering.rectangular_patch_baseline",
        "engineering_analysis",
        "Estimate conventional first-order rectangular microstrip-patch dimensions from the canonical frequency, substrate relative permittivity, and substrate thickness, then compare them with one current canonical radiating element. Modified radiators remain reference-only; this is not full-wave RF validation.",
        "analysis",
        rectangular_patch_baseline_analysis,
        planner_exposed=True,
        argument_schema={"type": "object", "additionalProperties": False, "properties": {}},
        effect="analysis",
        tool_version="1",
        requires_design=True,
    ))
    registry.register_tool(ToolDefinition(
        "engineering.dipole_baseline",
        "engineering_analysis",
        "Measure the current canonical dipole arm centerlines, feed gap, and conductor thickness and compare their electrical lengths with free-space lambda0/2 and lambda0/4 references. These references do not establish physical resonance or full-wave performance.",
        "analysis",
        dipole_baseline_analysis,
        planner_exposed=True,
        argument_schema={"type": "object", "additionalProperties": False, "properties": {}},
        effect="analysis",
        tool_version="1",
        requires_design=True,
    ))


def discover_capabilities(
    registry: ToolRegistry,
    *,
    package_name: str = "studio.antenna_capabilities",
) -> None:
    package = importlib.import_module(package_name)
    for module_info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        module = importlib.import_module(module_info.name)
        register = getattr(module, "register_capabilities", None)
        if callable(register):
            register(registry)
    try:
        entry_points = metadata.entry_points(group="antenna_surrogate_studio.antenna_capabilities")
    except TypeError:
        entry_points = metadata.entry_points().get("antenna_surrogate_studio.antenna_capabilities", ())
    for entry_point in entry_points:
        register = entry_point.load()
        register(registry)


def create_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_primitive_tools(registry)
    discover_capabilities(registry)
    return registry
