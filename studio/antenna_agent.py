"""Executor for schema-constrained LLM plans over registered antenna tools."""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from studio.antenna_analysis import EngineeringAnalysisResult
from studio.antenna_excitation import (
    INDEPENDENT_LOCAL_PORT_MODEL,
    realize_excitation,
    supports_independent_port_synthesis,
)
from studio.antenna_design import (
    AntennaDesign,
    ComposedOperationGroup,
    ComposedTargetSelector,
    ComposedToolCall,
    DesignValidationError,
    ExcitationAssignment,
    ExcitationDefinition,
    SemanticGeometryTarget,
    element_coordinate,
    evaluate_scalar,
    object_bounds,
    recipe_excitation_definition,
    resolve_parameter_values,
    resolved_dimensions,
)
from studio.antenna_llm_planner import LLMToolPlan, PlannedToolCall
from studio.antenna_modifiers import active_modifier_ids
from studio.antenna_recipes import MATERIALS, _circular_radius, _patch_dimensions
from studio.antenna_tools import (
    CapabilityError,
    ToolCall,
    ToolRegistry,
    create_tool_registry,
    validate_tool_arguments,
)
from studio.antenna_validation import validate_design


class AgentInstructionError(CapabilityError):
    """Raised when a plan cannot execute with installed capabilities."""


class CapabilityUnavailableError(AgentInstructionError):
    """Raised when a requested capability is not installed."""


class PlannerClarificationRequired(AgentInstructionError):
    """Raised when the LLM asks one question before changing geometry."""


class PlannerRefusal(AgentInstructionError):
    """Raised when installed capabilities cannot represent the request."""


def _solver_parameter_name(key: str) -> str:
    """Derive a stable CST-safe display name without asking the model to spell one."""

    parts = key.split("_")
    if parts and parts[-1].lower() in {"mm", "ghz", "deg"}:
        parts = parts[:-1]
    return "".join(part[:1].upper() + part[1:] for part in parts) or key


@dataclass(frozen=True, slots=True)
class PlannerToolDefinition:
    name: str
    description: str
    argument_schema: dict[str, Any]
    handler: Callable[["_PlanningContext", dict[str, Any]], None] | None = None
    primitive_tool_name: str | None = None
    effect: str = "design_action"
    tool_version: str = "1"
    requires_design: bool = True


@dataclass(slots=True)
class _PlanningContext:
    source_design: AntennaDesign
    recipe: Any
    values: dict[str, Any]
    modifier_ids: list[str]
    explicit_keys: set[str]
    changes: list[str]
    composed_groups: list[ComposedOperationGroup]
    composed_parameters: dict[str, Any]
    explicit_updates: dict[str, Any]
    excitation: ExcitationDefinition
    excitation_explicit: bool = False
    reset_or_family_change: bool = False
    dimension_driver_changed: bool = False


@dataclass(frozen=True, slots=True)
class AgentPlan:
    recipe_id: str
    intent: str
    parameter_updates: tuple[tuple[str, Any], ...]
    planned_tools: tuple[str, ...]
    modifier_ids: tuple[str, ...] = ()
    planner_calls: tuple[PlannedToolCall, ...] = ()
    planner_message: str = ""


@dataclass(frozen=True, slots=True)
class AgentResult:
    design: AntennaDesign
    plan: AgentPlan
    executed_tools: tuple[ToolCall, ...]
    changes: tuple[str, ...]

    @property
    def summary(self) -> str:
        return "Updated " + "; ".join(self.changes) + "."


class AntennaDesignAgent:
    """Validate and execute LLM-produced calls against runtime registries."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self._planner_tools: dict[str, PlannerToolDefinition] = {}
        self._register_planner_tools()

    def _register(self, definition: PlannerToolDefinition) -> None:
        if definition.name in self._planner_tools:
            raise CapabilityError(f"Planner tool already registered: {definition.name}.")
        self._planner_tools[definition.name] = definition

    def _register_planner_tools(self) -> None:
        recipe_ids = [item.recipe_id for item in self.registry.recipes()]
        modifier_ids = [item.modifier_id for item in self.registry.modifiers()]
        parameter_keys = sorted(
            {item.key for recipe in self.registry.recipes() for item in recipe.parameter_definitions()}
            | {item.key for modifier in self.registry.modifiers() for item in modifier.parameter_definitions()}
        )
        self._register(PlannerToolDefinition(
            "design.reset", "Reset the current recipe to validated defaults and remove modifiers.",
            {"type": "object", "additionalProperties": False, "properties": {}}, self._tool_reset,
        ))
        self._register(PlannerToolDefinition(
            "recipe.select", "Select an installed antenna recipe; this resets values and modifiers.",
            {"type": "object", "additionalProperties": False, "required": ["recipe_id"], "properties": {"recipe_id": {"type": "string", "enum": recipe_ids}}},
            self._tool_select_recipe,
        ))
        self._register(PlannerToolDefinition(
            "parameter.set", "Set one allowlisted parameter on the selected recipe or active modifier.",
            {"type": "object", "additionalProperties": False, "required": ["key", "value"], "properties": {"key": {"type": "string", "enum": parameter_keys}, "value": {"type": ["number", "string"]}}},
            self._tool_set_parameter,
        ))
        self._register(PlannerToolDefinition(
            "excitation.set_strategy",
            "Record solver-neutral excitation/feed intent without claiming or creating unsupported physical synthesis.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["strategy"],
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": [
                            "single_element_feed", "independent_ports", "corporate_feed",
                            "series_feed", "custom", "unresolved",
                        ],
                    },
                },
            },
            self._tool_set_excitation_strategy,
        ))
        self._register(PlannerToolDefinition(
            "composition.set_scope",
            "Change one persisted composed feature to one element, selected elements, or all matching array elements.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["group_id", "scope", "elements"],
                "properties": {
                    "group_id": {"type": "string", "pattern": r"[A-Za-z][A-Za-z0-9_]{0,63}"},
                    "scope": {"type": "string", "enum": ["single", "selected", "all"]},
                    "elements": {
                        "type": "array",
                        "maxItems": 64,
                        "uniqueItems": True,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["row", "column"],
                            "properties": {
                                "row": {"type": "integer", "minimum": 1, "maximum": 64},
                                "column": {"type": "integer", "minimum": 1, "maximum": 64},
                            },
                        },
                    },
                },
            },
            self._tool_set_composition_scope,
        ))
        composition_identifier = {"type": "string", "pattern": r"[A-Za-z][A-Za-z0-9_]{0,63}"}
        self._register(PlannerToolDefinition(
            "composition.update_operation",
            "Update selected arguments of one stable operation inside an existing persisted composed feature; the original primitive schema is revalidated.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["group_id", "operation_id", "argument_updates"],
                "properties": {
                    "group_id": dict(composition_identifier),
                    "operation_id": dict(composition_identifier),
                    "argument_updates": {"type": "object", "minProperties": 1},
                },
            },
            self._tool_update_composition_operation,
        ))
        self._register(PlannerToolDefinition(
            "composition.delete",
            "Delete exactly one persisted composed feature while preserving the recipe and other composed features.",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["group_id"],
                "properties": {"group_id": dict(composition_identifier)},
            },
            self._tool_delete_composition,
        ))
        for name, description, handler in (
            ("modifier.apply", "Apply one installed compatible geometry modifier.", self._tool_apply_modifier),
            ("modifier.remove", "Remove one currently active geometry modifier.", self._tool_remove_modifier),
        ):
            self._register(PlannerToolDefinition(
                name, description,
                {"type": "object", "additionalProperties": False, "required": ["modifier_id"], "properties": {"modifier_id": {"type": "string", "enum": modifier_ids}}},
                handler,
            ))
        for tool in self.registry.tools():
            if not tool.planner_exposed or tool.argument_schema is None:
                continue
            self._register(PlannerToolDefinition(
                tool.name,
                tool.description,
                tool.argument_schema,
                primitive_tool_name=tool.name,
                effect=tool.effect,
                tool_version=tool.tool_version,
                requires_design=tool.requires_design,
            ))

    def available_capabilities(self) -> dict[str, tuple[str, ...]]:
        return {
            "tools": tuple(item.name for item in self.registry.tools()),
            "recipes": tuple(item.recipe_id for item in self.registry.recipes()),
            "families": tuple(item.family for item in self.registry.recipes()),
            "modifiers": tuple(item.modifier_id for item in self.registry.modifiers()),
            "planner_tools": tuple(self._planner_tools),
        }

    def capability_manifest(self, design: AntennaDesign | None) -> dict[str, Any]:
        """Describe the live registries and exact callable schemas to the LLM."""

        has_design = design is not None
        callable_tools = []
        target_ids = [
            item.object_id
            for item in (design.geometry if design is not None else ())
            if self._semantic_role(item) == "radiating_patch_conductor"
        ]
        for item in self._planner_tools.values():
            if not has_design:
                if item.effect == "analysis" and item.requires_design:
                    continue
                if item.effect != "analysis" and item.name not in {"recipe.select", "parameter.set"}:
                    continue
            if item.name.startswith("composition.") and not design.composed_operations:
                continue
            schema = copy.deepcopy(item.argument_schema)
            if item.primitive_tool_name in {"boolean.subtract", "boolean.union"}:
                schema["properties"]["target_id"]["enum"] = target_ids
            elif item.name == "parameter.set":
                if design is None:
                    schema["properties"]["key"]["enum"] = sorted({
                        parameter.key
                        for recipe in self.registry.recipes()
                        for parameter in recipe.parameter_definitions()
                    })
                else:
                    schema["properties"]["key"]["enum"] = sorted({
                        *schema["properties"]["key"].get("enum", ()),
                        *design.parameter_map(),
                    })
            elif item.name in {"composition.set_scope", "composition.delete"}:
                schema["properties"]["group_id"]["enum"] = [
                    group.group_id for group in design.composed_operations
                ]
            elif item.name == "composition.update_operation":
                schema["properties"]["group_id"]["enum"] = [
                    group.group_id for group in design.composed_operations
                ]
                schema["properties"]["operation_id"]["enum"] = [
                    call.operation_id
                    for group in design.composed_operations
                    for call in group.calls
                ]
            callable_tools.append({
                "name": item.name,
                "kind": "primitive" if item.primitive_tool_name else "design_action",
                "effect": item.effect,
                "tool_version": item.tool_version,
                "description": item.description,
                "arguments": schema,
            })
        return {
            "callable_tools": callable_tools,
            "recipes": [{
                "recipe_id": recipe.recipe_id,
                "family": recipe.family,
                "display_name": recipe.display_name,
                "aliases": list(recipe.aliases),
                "parameters": [{
                    "key": parameter.key, "name": parameter.name, "label": parameter.label,
                    "unit": parameter.unit, "kind": parameter.kind, "minimum": parameter.minimum,
                    "maximum": parameter.maximum, "choices": list(parameter.choices),
                } for parameter in recipe.parameter_definitions()],
                "compiled_primitive_tools": list(recipe.required_tools),
            } for recipe in self.registry.recipes()],
            "modifiers": [{
                "modifier_id": modifier.modifier_id,
                "aliases": list(modifier.aliases),
                "applicable_families": list(modifier.applicable_families),
                "parameters": [{
                    "key": parameter.key, "name": parameter.name, "label": parameter.label,
                    "unit": parameter.unit, "minimum": parameter.minimum, "maximum": parameter.maximum,
                } for parameter in modifier.parameter_definitions()],
                "compiled_primitive_tools": list(modifier.required_tools),
            } for modifier in self.registry.modifiers()],
            "deterministic_primitive_tools": [{
                "name": tool.name,
                "category": tool.category,
                "description": tool.description,
                "llm_callable": tool.planner_exposed and (has_design or not tool.requires_design),
                "effect": tool.effect,
                "tool_version": tool.tool_version,
            } for tool in self.registry.tools()],
            "semantic_geometry_objects": (
                [self._semantic_geometry_entry(design, item) for item in design.geometry]
                if design is not None else []
            ),
            "boolean_relationships": [{
                "operation_id": item.operation_id,
                "operation": item.operation,
                "target_id": item.target_id,
                "tool_ids": list(item.tool_ids),
            } for item in design.booleans] if design is not None else [],
            "composed_features": (
                [self._composed_feature_entry(group) for group in design.composed_operations]
                if design is not None else []
            ),
            "excitation": self._excitation_entry(design) if design is not None else None,
            "current_recipe_id": design.recipe_id if design is not None else None,
            "current_modifier_ids": list(active_modifier_ids(design)) if design is not None else [],
        }

    @staticmethod
    def _excitation_entry(design: AntennaDesign) -> dict[str, Any]:
        excitation = design.excitation
        physical_port_ids = tuple(port.port_id for port in design.ports)
        realizing_port_ids = excitation.realizing_port_ids
        realizing_port_id_set = set(realizing_port_ids)
        synthesis_capability = {
            "single_element_feed": "installed_for_single_element_recipe",
            "independent_ports": (
                "supported_and_realizable"
                if supports_independent_port_synthesis(design)
                else "unsupported_for_current_design"
            ),
            "corporate_feed": "not_installed",
            "series_feed": "not_installed",
            "custom": "not_installed",
            "unresolved": "not_selected",
            "legacy_recipe": "legacy_recipe_geometry",
        }.get(excitation.strategy, "unknown")
        return {
            "strategy": excitation.strategy,
            "requested_strategy": excitation.strategy,
            "realization_status": excitation.realization_status,
            "excitation_realized": excitation.excitation_realized,
            "feed_network": excitation.feed_network,
            "element_assignments": [
                {
                    "element": [assignment.element_row, assignment.element_column],
                    "excitation_id": assignment.excitation_id,
                    "port_id": assignment.port_id,
                    "status": assignment.status,
                }
                for assignment in excitation.element_assignments
            ],
            "realizing_port_ids": list(realizing_port_ids),
            "other_physical_port_ids": [
                port_id for port_id in physical_port_ids
                if port_id not in realizing_port_id_set
            ],
            "current_physical_port_ids": list(physical_port_ids),
            "current_strategy_synthesis": synthesis_capability,
            "available_strategy_capabilities": [
                {
                    "strategy": "single_element_feed",
                    "physical_synthesis": "installed_for_single_element_recipe",
                },
                {
                    "strategy": "independent_ports",
                    "physical_synthesis": (
                        "supported_and_realizable"
                        if supports_independent_port_synthesis(design)
                        else "unsupported_for_current_design"
                    ),
                    "realization_model": (
                        INDEPENDENT_LOCAL_PORT_MODEL
                        if supports_independent_port_synthesis(design)
                        else None
                    ),
                },
                {"strategy": "corporate_feed", "physical_synthesis": "not_installed"},
                {"strategy": "series_feed", "physical_synthesis": "not_installed"},
                {"strategy": "custom", "physical_synthesis": "not_installed"},
            ],
            "unresolved_requirements": list(excitation.unresolved_requirements),
        }

    def planner_tool_effect(self, name: str) -> str:
        try:
            return self._planner_tools[name].effect
        except KeyError as exc:
            raise CapabilityUnavailableError(f"The LLM requested an unregistered planning tool: {name}.") from exc

    def execute_analysis_batch(
        self,
        design: AntennaDesign | None,
        calls: tuple[PlannedToolCall, ...],
    ) -> tuple[EngineeringAnalysisResult, ...]:
        if design is None:
            raise CapabilityUnavailableError("Engineering analysis requires a current canonical design.")
        if not calls:
            raise AgentInstructionError("An analysis batch requires at least one tool call.")
        results = []
        for call in calls:
            try:
                definition = self._planner_tools[call.name]
            except KeyError as exc:
                raise CapabilityUnavailableError(
                    f"The LLM requested an unregistered planning tool: {call.name}."
                ) from exc
            if definition.effect != "analysis" or definition.primitive_tool_name is None:
                raise AgentInstructionError("An analysis batch may contain only registered analysis tools.")
            try:
                validate_tool_arguments(definition.argument_schema, call.arguments, path=f"{call.name}.arguments")
            except CapabilityError as exc:
                raise AgentInstructionError(f"Analysis call {call.name} failed schema validation: {exc}") from exc
            result = self.registry.execute_analysis(
                design, ToolCall(definition.primitive_tool_name, dict(call.arguments)),
            )
            if result.tool_name != call.name or result.tool_version != definition.tool_version:
                raise AgentInstructionError(f"Analysis tool {call.name} returned mismatched identity metadata.")
            # Normalize through the strict, redacting provider-neutral contract
            # before any result enters planner context, cache, or trajectory.
            results.append(EngineeringAnalysisResult.from_dict(result.to_dict()))
        return tuple(results)

    @staticmethod
    def _composed_feature_entry(group: ComposedOperationGroup) -> dict[str, Any]:
        selector = group.target_selector
        if selector is None:
            legacy_targets = [call.semantic_target for call in group.calls if call.semantic_target is not None]
            selector = ComposedTargetSelector(
                role=legacy_targets[0].role,
                scope="single",
                elements=((legacy_targets[0].element_row, legacy_targets[0].element_column),),
            ) if legacy_targets else None
        parameter_calls = [call for call in group.calls if call.name == "parameter.create"]
        return {
            "group_id": group.group_id,
            "source_family": group.source_family,
            "coordinate_frame": group.coordinate_frame,
            "target_selector": (
                {
                    "role": selector.role,
                    "scope": selector.scope,
                    "elements": [list(element) for element in selector.elements],
                }
                if selector else None
            ),
            "parameters": [
                {
                    "operation_id": call.operation_id,
                    "key": call.arguments().get("key"),
                    "value": call.arguments().get("value"),
                    "unit": call.arguments().get("unit", ""),
                    "expression": call.arguments().get("expression"),
                }
                for call in parameter_calls
            ],
            "tool_sequence": [call.name for call in group.calls],
            "created_parameters": [call.arguments().get("key") for call in parameter_calls],
            "operations": [
                {
                    "operation_id": call.operation_id,
                    "tool_name": call.name,
                    "editable_arguments": call.arguments(),
                    "semantic_target": (
                        {
                            "role": call.semantic_target.role,
                            "element": [call.semantic_target.element_row, call.semantic_target.element_column],
                        }
                        if call.semantic_target else None
                    ),
                }
                for call in group.calls
            ],
        }

    @staticmethod
    def _semantic_role(item) -> str:
        tags = set(item.tags)
        if "ground" in tags:
            return "ground"
        if "substrate" in tags:
            return "substrate"
        if "probe_feed" in tags or item.object_id.endswith("_feed"):
            return "feed"
        if "patch_element" in tags and item.object_id.endswith("_patch"):
            return "radiating_patch_conductor"
        if "circular_patch_element" in tags:
            return "radiating_patch_conductor"
        if "dipole_arm" in tags:
            return "radiating_arm"
        if "patch_element" in tags:
            return "radiating_conductor_piece"
        if "planner_created" in tags:
            return "planner_created_geometry"
        return "supporting_geometry"

    def _semantic_geometry_entry(self, design: AntennaDesign, item) -> dict[str, Any]:
        entry = {
            "object_id": item.object_id,
            "name": item.name,
            "role": self._semantic_role(item),
            "primitive": item.primitive,
            "material_id": item.material_id,
            "tags": list(item.tags),
            "dimensions": item.dimension_map(),
        }
        try:
            entry["bounds_mm"] = list(object_bounds(design, item))
        except (KeyError, DesignValidationError):
            entry["bounds_mm"] = None
        if entry["role"] == "radiating_patch_conductor":
            bounds = entry["bounds_mm"]
            if bounds:
                center_y = (bounds[2] + bounds[3]) / 2
                if design.family == "rectangular_inset_patch":
                    center_y -= design.value("inset_depth_mm") / 2
                entry["element_center_mm"] = [(bounds[0] + bounds[1]) / 2, center_y]
                entry["boolean_target"] = True
        return entry

    @staticmethod
    def _element_coordinates(design: AntennaDesign, item) -> tuple[int, int] | None:
        element_index = None
        for tag in item.tags:
            match = re.fullmatch(r"element_(\d+)", tag)
            if match:
                element_index = int(match.group(1))
                break
        if element_index is None or design.array.columns < 1:
            return None
        return (
            (element_index - 1) // design.array.columns + 1,
            (element_index - 1) % design.array.columns + 1,
        )

    def _semantic_target_for_object(self, design: AntennaDesign, object_id: str) -> SemanticGeometryTarget:
        item = next((candidate for candidate in design.geometry if candidate.object_id == object_id), None)
        if item is None:
            raise CapabilityUnavailableError(f"Boolean target does not exist: {object_id}.")
        coordinates = self._element_coordinates(design, item)
        role = self._semantic_role(item)
        if coordinates is None or role != "radiating_patch_conductor":
            raise CapabilityUnavailableError(
                f"Boolean target {object_id!r} is not a radiating patch conductor with a stable element selector."
            )
        return SemanticGeometryTarget(role, coordinates[0], coordinates[1])

    def _resolve_semantic_target(self, design: AntennaDesign, target: SemanticGeometryTarget) -> str:
        matches = [
            item.object_id
            for item in design.geometry
            if self._semantic_role(item) == target.role
            and self._element_coordinates(design, item) == (target.element_row, target.element_column)
        ]
        if len(matches) != 1:
            raise CapabilityUnavailableError(
                "Stored composition target "
                f"{target.role} element [{target.element_row},{target.element_column}] "
                "cannot be resolved uniquely in the rebuilt antenna."
            )
        return matches[0]

    @staticmethod
    def _axis_offset_expression(index: int, count: int) -> str:
        return "0" if count == 1 else f"({index - 1}-({count}-1)/2)*element_spacing_mm"

    def _target_origin_expressions(
        self,
        design: AntennaDesign,
        target: SemanticGeometryTarget,
    ) -> tuple[str, str]:
        row_offset = self._axis_offset_expression(target.element_row, design.array.rows)
        column_offset = self._axis_offset_expression(target.element_column, design.array.columns)

        def component(row_value: float, column_value: float) -> str:
            terms: list[str] = []
            for coefficient, expression in ((row_value, row_offset), (column_value, column_offset)):
                if abs(coefficient) <= 1e-12 or expression == "0":
                    continue
                if abs(coefficient - 1.0) <= 1e-12:
                    terms.append(expression)
                elif abs(coefficient + 1.0) <= 1e-12:
                    terms.append(f"-({expression})")
                else:
                    terms.append(f"{coefficient:.12g}*({expression})")
            return "+".join(terms) or "0"

        return (
            component(design.array.row_vector[0], design.array.column_vector[0]),
            component(design.array.row_vector[1], design.array.column_vector[1]),
        )

    @staticmethod
    def _offset_in_frame(value: Any, origin: str) -> Any:
        if origin == "0":
            return value
        if isinstance(value, (int, float)):
            return f"({origin})+({float(value):.12g})"
        return f"({origin})+({value})"

    @staticmethod
    def _localize_coordinate(value: Any, origin_expression: str, origin_value: float) -> Any:
        if isinstance(value, (int, float)):
            return float(value) - origin_value
        try:
            return float(str(value)) - origin_value
        except ValueError:
            pass
        if origin_expression == "0":
            return value
        return f"({value})-({origin_expression})"

    def _geometry_arguments_in_frame(
        self,
        name: str,
        arguments: dict[str, Any],
        origin: tuple[str, str],
        *,
        to_world: bool,
        numeric_origin: tuple[float, float] = (0.0, 0.0),
    ) -> dict[str, Any]:
        result = copy.deepcopy(arguments)
        dimensions = result.get("dimensions")
        if not isinstance(dimensions, dict):
            return result
        if name in {"geometry.cylinder", "geometry.circle_sheet"}:
            fields = (("center_1", 0), ("center_2", 1))
        elif name == "geometry.rectangle_sheet":
            fields = (("x_min", 0), ("x_max", 0), ("y_min", 1), ("y_max", 1))
        else:
            fields = ()
        for field, axis in fields:
            dimensions[field] = (
                self._offset_in_frame(dimensions[field], origin[axis])
                if to_world
                else self._localize_coordinate(dimensions[field], origin[axis], numeric_origin[axis])
            )
        return result

    def _selector_for_group(self, group: ComposedOperationGroup) -> ComposedTargetSelector:
        if group.target_selector is not None:
            return group.target_selector
        targets = [call.semantic_target for call in group.calls if call.semantic_target is not None]
        if not targets:
            raise CapabilityUnavailableError(
                f"Stored composition {group.group_id!r} has no semantic target selector."
            )
        target = targets[0]
        return ComposedTargetSelector(
            target.role,
            "single",
            ((target.element_row, target.element_column),),
        )

    def _convert_group_to_target_local(
        self,
        design: AntennaDesign,
        group: ComposedOperationGroup,
    ) -> ComposedOperationGroup:
        if group.coordinate_frame == "target_local":
            return group
        selector = self._selector_for_group(group)
        if len(selector.elements) != 1:
            raise CapabilityUnavailableError(
                f"Stored composition {group.group_id!r} cannot infer one source frame."
            )
        row, column = selector.elements[0]
        target = SemanticGeometryTarget(selector.role, row, column)
        origin = self._target_origin_expressions(design, target)
        values = resolve_parameter_values(design)
        numeric_origin = (
            evaluate_scalar(origin[0], values),
            evaluate_scalar(origin[1], values),
        )
        calls = tuple(
            ComposedToolCall.create(
                call.name,
                self._geometry_arguments_in_frame(
                    call.name,
                    call.arguments(),
                    origin,
                    to_world=False,
                    numeric_origin=numeric_origin,
                ),
                semantic_target=call.semantic_target,
                operation_id=call.operation_id,
            )
            for call in group.calls
        )
        return replace(group, calls=calls, target_selector=selector, coordinate_frame="target_local")

    def _resolve_target_selector(
        self,
        design: AntennaDesign,
        selector: ComposedTargetSelector,
    ) -> tuple[SemanticGeometryTarget, ...]:
        matching: dict[tuple[int, int], list[str]] = {}
        for item in design.geometry:
            coordinates = self._element_coordinates(design, item)
            if coordinates is not None and self._semantic_role(item) == selector.role:
                matching.setdefault(coordinates, []).append(item.object_id)
        duplicate = next((coordinates for coordinates, ids in matching.items() if len(ids) != 1), None)
        if duplicate is not None:
            raise CapabilityUnavailableError(
                f"Stored composition target {selector.role} element {list(duplicate)} cannot be resolved uniquely."
            )
        requested = tuple(sorted(matching)) if selector.scope == "all" else selector.elements
        if not requested:
            raise CapabilityUnavailableError(
                f"Stored composition target role {selector.role!r} did not match any array elements."
            )
        missing = [coordinates for coordinates in requested if coordinates not in matching]
        if missing:
            formatted = ", ".join(f"[{row},{column}]" for row, column in missing)
            raise CapabilityUnavailableError(
                f"Stored composition target {selector.role} cannot resolve selected element(s) {formatted}."
            )
        return tuple(SemanticGeometryTarget(selector.role, row, column) for row, column in requested)

    @staticmethod
    def _instance_identifier(identifier: str, row: int, column: int, *, operation: bool = False) -> str:
        suffix = f"_r{row}c{column}"
        ending = ""
        for candidate in (("_subtract", "_union") if operation else ("_tool",)):
            if identifier.endswith(candidate):
                identifier, ending = identifier[:-len(candidate)], candidate
                break
        maximum = 64
        available = maximum - len(suffix) - len(ending)
        return f"{identifier[:available]}{suffix}{ending}"

    def _instantiate_group_calls(
        self,
        design: AntennaDesign,
        group: ComposedOperationGroup,
        targets: tuple[SemanticGeometryTarget, ...],
    ) -> tuple[PlannedToolCall, ...]:
        parameter_calls = [
            PlannedToolCall(call.name, call.arguments())
            for call in group.calls
            if call.name == "parameter.create"
        ]
        template_calls = [call for call in group.calls if call.name != "parameter.create"]
        planned: list[PlannedToolCall] = list(parameter_calls)
        for instance_index, target in enumerate(targets):
            preserve_ids = instance_index == 0
            id_map: dict[str, str] = {}
            for call in template_calls:
                arguments = call.arguments()
                if call.name in {"geometry.rectangle_sheet", "geometry.cylinder", "geometry.circle_sheet"}:
                    object_id = str(arguments["object_id"])
                    id_map[object_id] = object_id if preserve_ids else self._instance_identifier(
                        object_id, target.element_row, target.element_column
                    )
                elif call.name == "geometry.duplicate":
                    new_id = str(arguments["new_id"])
                    id_map[new_id] = new_id if preserve_ids else self._instance_identifier(
                        new_id, target.element_row, target.element_column
                    )
            origin = self._target_origin_expressions(design, target)
            target_id = self._resolve_semantic_target(design, target)
            for call in template_calls:
                arguments = call.arguments()
                if group.coordinate_frame == "target_local":
                    arguments = self._geometry_arguments_in_frame(
                        call.name, arguments, origin, to_world=True
                    )
                if call.name in {"geometry.rectangle_sheet", "geometry.cylinder", "geometry.circle_sheet"}:
                    arguments["object_id"] = id_map[str(arguments["object_id"])]
                elif call.name in {"geometry.translate", "geometry.rotate"}:
                    arguments["object_id"] = id_map[str(arguments["object_id"])]
                elif call.name == "geometry.duplicate":
                    arguments["source_id"] = id_map[str(arguments["source_id"])]
                    arguments["new_id"] = id_map[str(arguments["new_id"])]
                elif call.name in {"boolean.subtract", "boolean.union"}:
                    arguments["target_id"] = target_id
                    arguments["tool_ids"] = [id_map[str(tool_id)] for tool_id in arguments["tool_ids"]]
                    if not preserve_ids:
                        arguments["operation_id"] = self._instance_identifier(
                            str(arguments["operation_id"]),
                            target.element_row,
                            target.element_column,
                            operation=True,
                        )
                planned.append(PlannedToolCall(call.name, arguments))
        return tuple(planned)

    def _capture_operation_group(
        self,
        design: AntennaDesign,
        calls: tuple[PlannedToolCall, ...],
    ) -> ComposedOperationGroup:
        existing_ids = {group.group_id for group in design.composed_operations}
        sequence = len(existing_ids) + 1
        group_id = f"composition_{sequence}"
        while group_id in existing_ids:
            sequence += 1
            group_id = f"composition_{sequence}"
        captured = []
        operation_id_counts: dict[str, int] = {}
        semantic_targets: list[SemanticGeometryTarget] = []
        for call in calls:
            semantic_target = None
            if call.name in {"boolean.subtract", "boolean.union"}:
                semantic_target = self._semantic_target_for_object(
                    design,
                    str(call.arguments["target_id"]),
                )
                semantic_targets.append(semantic_target)
            captured_call = ComposedToolCall.create(
                call.name,
                call.arguments,
                semantic_target=semantic_target,
            )
            occurrence = operation_id_counts.get(captured_call.operation_id, 0) + 1
            operation_id_counts[captured_call.operation_id] = occurrence
            if occurrence > 1:
                suffix = f"_{occurrence}"
                captured_call = replace(
                    captured_call,
                    operation_id=f"{captured_call.operation_id[:64 - len(suffix)]}{suffix}",
                )
            captured.append(captured_call)
        unique_targets = {
            (target.role, target.element_row, target.element_column) for target in semantic_targets
        }
        if len(unique_targets) != 1:
            raise CapabilityUnavailableError(
                "One persisted composed feature must use one semantic radiating target."
            )
        role, row, column = next(iter(unique_targets))
        return ComposedOperationGroup(
            group_id=group_id,
            source_recipe_id=design.recipe_id,
            source_family=design.family,
            calls=tuple(captured),
            target_selector=ComposedTargetSelector(role, "single", ((row, column),)),
            coordinate_frame="target_local",
        )

    def _replay_composed_operations(
        self,
        design: AntennaDesign,
        groups: tuple[ComposedOperationGroup, ...],
    ) -> tuple[AntennaDesign, tuple[ToolCall, ...], tuple[str, ...]]:
        if not groups:
            return design, (), ()
        replayed_calls: list[ToolCall] = []
        changes: list[str] = []
        rebuilt = replace(design, composed_operations=(), validation=())
        for group in groups:
            if group.source_family != rebuilt.family:
                raise CapabilityUnavailableError(
                    f"Stored composition {group.group_id!r} was created for {group.source_family} "
                    f"and is not compatible with {rebuilt.family}; the topology change was not applied."
                )
            for stored_call in group.calls:
                definition = self._planner_tools.get(stored_call.name)
                if definition is None or definition.primitive_tool_name is None:
                    raise CapabilityUnavailableError(
                        f"Stored composition {group.group_id!r} requires unavailable primitive {stored_call.name!r}."
                    )
            selector = self._selector_for_group(group)
            targets = self._resolve_target_selector(rebuilt, selector)
            planned_calls = self._instantiate_group_calls(rebuilt, group, targets)
            try:
                rebuilt, executed, _group_changes = self._execute_primitive_calls(
                    rebuilt,
                    tuple(planned_calls),
                )
            except (AgentInstructionError, DesignValidationError) as exc:
                raise CapabilityUnavailableError(
                    f"Stored composition {group.group_id!r} is no longer compatible with the rebuilt design: {exc}"
                ) from exc
            replayed_calls.extend(executed)
            changes.append(f"stored composition {group.group_id} replayed")
        rebuilt = validate_design(replace(
            rebuilt,
            composed_operations=groups,
            validation=(),
        ))
        return rebuilt, tuple(replayed_calls), tuple(changes)

    def _recipe(self, hint: str):
        normalized = hint.casefold().replace("_", " ").strip()
        for recipe in self.registry.recipes():
            candidates = {recipe.recipe_id.casefold(), recipe.recipe_id.casefold().replace("_", " "), recipe.family.casefold(), recipe.family.casefold().replace("_", " "), *(alias.casefold() for alias in recipe.aliases)}
            if normalized in candidates or any(candidate in normalized for candidate in candidates):
                return recipe
        raise CapabilityUnavailableError(f"No installed antenna recipe matches '{hint}'.")

    def create_design(self, recipe_hint: str = "inset_patch", values: dict[str, Any] | None = None, *, design_id: str | None = None, revision: int = 0) -> AntennaDesign:
        recipe = self._recipe(recipe_hint)
        merged = recipe.defaults()
        if values:
            merged.update(values)
        design, _calls = recipe.build(self.registry, merged, design_id=design_id, revision=revision)
        return design

    def parameter_definitions_for_design(self, design: AntennaDesign) -> tuple[Any, ...]:
        rows = list(self.registry.recipe(design.recipe_id).parameter_definitions())
        for modifier_id in active_modifier_ids(design):
            rows.extend(self.registry.modifier(modifier_id).parameter_definitions())
        return tuple(rows)

    def values_for_design(self, design: AntennaDesign) -> dict[str, Any]:
        recipe = self.registry.recipe(design.recipe_id)
        definitions = self.parameter_definitions_for_design(design)
        keys = {definition.key for definition in definitions}
        values = {parameter.key: parameter.value for parameter in design.parameters if parameter.key in keys}
        if "material" in keys:
            values["material"] = design.metadata_map().get("substrate_material", "FR4")
        defaults = recipe.defaults()
        for modifier_id in active_modifier_ids(design):
            defaults.update(self.registry.modifier(modifier_id).defaults())
        return {**defaults, **values}

    @staticmethod
    def _composed_parameter_map(design: AntennaDesign) -> dict[str, Any]:
        keys = {
            str(call.arguments().get("key"))
            for group in design.composed_operations
            for call in group.calls
            if call.name == "parameter.create" and call.arguments().get("key")
        }
        parameters = design.parameter_map()
        return {key: parameters[key] for key in keys if key in parameters}

    def _build_with_modifiers(self, recipe, values: dict[str, Any], modifier_ids: tuple[str, ...], *, design_id: str, revision: int) -> tuple[AntennaDesign, tuple[ToolCall, ...]]:
        built, base_calls = recipe.build(self.registry, values, design_id=design_id, revision=revision)
        calls = list(base_calls)
        for modifier_id in modifier_ids:
            built, modifier_calls = self.registry.modifier(modifier_id).apply(self.registry, built, values)
            calls.extend(modifier_calls)
        return built, tuple(calls)

    @staticmethod
    def _port_for_element(design: AntennaDesign, row: int, column: int):
        return next(
            (
                port
                for port in design.ports
                if element_coordinate(design.array, port.element_index) == (row, column)
            ),
            None,
        )

    def _excitation_for_strategy(
        self,
        design: AntennaDesign,
        strategy: str,
    ) -> ExcitationDefinition:
        coordinates = tuple(
            (row, column)
            for row in range(1, design.array.rows + 1)
            for column in range(1, design.array.columns + 1)
        )
        if strategy == "single_element_feed":
            port = self._port_for_element(design, 1, 1)
            fully_realized = design.array.element_count == 1 and port is not None
            return ExcitationDefinition(
                strategy=strategy,
                realization_status="realized" if fully_realized else "unresolved",
                feed_network="none",
                element_assignments=(ExcitationAssignment(
                    1,
                    1,
                    "excitation_1_1",
                    port.port_id if fully_realized else None,
                    "realized" if fully_realized else "unresolved",
                ),),
                unresolved_requirements=(
                    ()
                    if fully_realized
                    else ("Single-element feed intent does not define excitation for every array element.",)
                ),
            )
        if strategy == "independent_ports":
            supported = supports_independent_port_synthesis(design)
            return ExcitationDefinition(
                strategy=strategy,
                realization_status="unresolved" if supported else "unsupported",
                feed_network="independent",
                element_assignments=tuple(
                    ExcitationAssignment(row, column, f"excitation_{row}_{column}")
                    for row, column in coordinates
                ),
                unresolved_requirements=(
                    (
                        "Independent local-port realization is pending deterministic synthesis and validation."
                        if supported
                        else "Independent local-port synthesis is installed only for rectangular inset-fed patch designs."
                    ),
                ),
            )
        if strategy in {"corporate_feed", "series_feed"}:
            network = "corporate" if strategy == "corporate_feed" else "series"
            return ExcitationDefinition(
                strategy=strategy,
                realization_status="unsupported",
                feed_network=network,
                element_assignments=tuple(
                    ExcitationAssignment(row, column, f"{network}_network")
                    for row, column in coordinates
                ),
                unresolved_requirements=(
                    f"{network.capitalize()} feed-network physical synthesis is not installed.",
                ),
            )
        if strategy == "custom":
            return ExcitationDefinition(
                strategy=strategy,
                realization_status="unresolved",
                feed_network="custom",
                unresolved_requirements=(
                    "A custom excitation realization has not been defined.",
                ),
            )
        if strategy == "unresolved":
            return ExcitationDefinition(
                strategy=strategy,
                realization_status="unresolved",
                feed_network="unresolved",
                unresolved_requirements=("Excitation strategy remains unresolved.",),
            )
        raise CapabilityUnavailableError(f"Excitation strategy {strategy!r} is unavailable.")

    def _reconcile_excitation(
        self,
        excitation: ExcitationDefinition,
        rebuilt: AntennaDesign,
    ) -> ExcitationDefinition:
        """Rebuild assignments from explicit semantics instead of retaining stale coordinates."""

        if excitation.strategy == "legacy_recipe":
            return recipe_excitation_definition(rebuilt)
        return self._excitation_for_strategy(rebuilt, excitation.strategy)

    def update_parameters(self, design: AntennaDesign, updates: dict[str, Any], *, intent: str = "structured parameter edit") -> AgentResult:
        recipe = self.registry.recipe(design.recipe_id)
        definitions = self.parameter_definitions_for_design(design)
        definition_map = {item.key: item for item in definitions}
        composed_parameters = self._composed_parameter_map(design)
        allowed = {*definition_map, *composed_parameters}
        unknown = sorted(set(updates) - allowed)
        if unknown:
            raise AgentInstructionError("The requested fields are unavailable for this antenna: " + ", ".join(unknown) + ".")
        values = self.values_for_design(design)
        values.update({key: value for key, value in updates.items() if key in definition_map})
        groups = design.composed_operations
        for key, value in updates.items():
            if key in composed_parameters:
                groups = self._set_group_parameter(groups, key, value)
        modifier_ids = active_modifier_ids(design)
        built, calls = self._build_with_modifiers(recipe, values, modifier_ids, design_id=design.design_id, revision=design.revision + 1)
        built = realize_excitation(replace(
            built,
            excitation=self._reconcile_excitation(design.excitation, built),
            validation=(),
        ))
        built, replayed_calls, replay_changes = self._replay_composed_operations(
            built,
            groups,
        )
        calls = (*calls, *replayed_calls)
        changes = tuple(
            f"{(definition_map.get(key) or composed_parameters[key]).label} to {value}"
            for key, value in updates.items()
        )
        plan = AgentPlan(recipe.recipe_id, intent, tuple(updates.items()), tuple(dict.fromkeys((*recipe.required_tools, *(tool for modifier_id in modifier_ids for tool in self.registry.modifier(modifier_id).required_tools)))), modifier_ids)
        return AgentResult(built, plan, tuple(calls), (*changes, *replay_changes))

    @staticmethod
    def _require_exact(arguments: dict[str, Any], keys: set[str], tool: str) -> None:
        if set(arguments) != keys:
            raise AgentInstructionError(f"{tool} arguments do not match its registered schema.")

    def _tool_reset(self, context: _PlanningContext, arguments: dict[str, Any]) -> None:
        self._require_exact(arguments, set(), "design.reset")
        context.values = context.recipe.defaults()
        context.modifier_ids.clear()
        context.explicit_keys.clear()
        context.excitation = ExcitationDefinition()
        context.excitation_explicit = False
        context.reset_or_family_change = True
        context.changes.append(f"design reset to {context.recipe.display_name} defaults")

    def _tool_select_recipe(self, context: _PlanningContext, arguments: dict[str, Any]) -> None:
        self._require_exact(arguments, {"recipe_id"}, "recipe.select")
        recipe_id = arguments["recipe_id"]
        if not isinstance(recipe_id, str):
            raise AgentInstructionError("recipe.select requires a string recipe_id.")
        context.recipe = self.registry.recipe(recipe_id)
        context.values = context.recipe.defaults()
        context.modifier_ids.clear()
        context.explicit_keys.clear()
        context.excitation = ExcitationDefinition()
        context.excitation_explicit = False
        context.reset_or_family_change = True
        context.changes.append(f"antenna family to {context.recipe.display_name}")

    def _definitions(self, context: _PlanningContext) -> dict[str, Any]:
        rows = list(context.recipe.parameter_definitions())
        for modifier_id in context.modifier_ids:
            rows.extend(self.registry.modifier(modifier_id).parameter_definitions())
        return {**{item.key: item for item in rows}, **context.composed_parameters}

    @staticmethod
    def _set_group_parameter(
        groups: list[ComposedOperationGroup] | tuple[ComposedOperationGroup, ...],
        key: str,
        value: Any,
    ) -> tuple[ComposedOperationGroup, ...]:
        updated_groups: list[ComposedOperationGroup] = []
        found = False
        for group in groups:
            updated_calls: list[ComposedToolCall] = []
            for call in group.calls:
                arguments = call.arguments()
                if call.name == "parameter.create" and arguments.get("key") == key:
                    arguments["value"] = value
                    call = ComposedToolCall.create(
                        call.name,
                        arguments,
                        semantic_target=call.semantic_target,
                        operation_id=call.operation_id,
                    )
                    found = True
                updated_calls.append(call)
            updated_groups.append(replace(group, calls=tuple(updated_calls)))
        if not found:
            raise CapabilityUnavailableError(f"Persisted composed parameter {key!r} could not be resolved.")
        return tuple(updated_groups)

    def _tool_set_parameter(self, context: _PlanningContext, arguments: dict[str, Any]) -> None:
        self._require_exact(arguments, {"key", "value"}, "parameter.set")
        key = arguments["key"]
        definitions = self._definitions(context)
        if not isinstance(key, str) or key not in definitions:
            raise CapabilityUnavailableError(f"Parameter {key!r} is unavailable for the selected recipe and active modifiers.")
        value = arguments["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise AgentInstructionError(f"Parameter {key} must be a number or installed choice string.")
        if key in context.composed_parameters:
            context.composed_groups = list(self._set_group_parameter(context.composed_groups, key, value))
        else:
            context.values[key] = value
        context.explicit_keys.add(key)
        context.explicit_updates[key] = value
        context.dimension_driver_changed |= key in {"frequency_ghz", "material", "substrate_thickness_mm"}
        context.changes.append(f"{definitions[key].label} to {value}")

    def _tool_set_excitation_strategy(
        self,
        context: _PlanningContext,
        arguments: dict[str, Any],
    ) -> None:
        self._require_exact(arguments, {"strategy"}, "excitation.set_strategy")
        strategy = arguments["strategy"]
        if not isinstance(strategy, str):
            raise AgentInstructionError("excitation.set_strategy requires a string strategy.")
        context.excitation = self._excitation_for_strategy(context.source_design, strategy)
        context.excitation_explicit = True
        context.changes.append(f"excitation strategy intent to {strategy}")

    def _tool_set_composition_scope(self, context: _PlanningContext, arguments: dict[str, Any]) -> None:
        self._require_exact(arguments, {"group_id", "scope", "elements"}, "composition.set_scope")
        group_id = str(arguments["group_id"])
        scope = str(arguments["scope"])
        elements = tuple((int(item["row"]), int(item["column"])) for item in arguments["elements"])
        if scope == "all" and elements:
            raise AgentInstructionError("All-element composition scope must use an empty elements list.")
        if scope == "single" and len(elements) != 1:
            raise AgentInstructionError("Single-element composition scope requires exactly one selected element.")
        if scope == "selected" and not elements:
            raise AgentInstructionError("Selected-element composition scope requires at least one element.")
        if len(elements) != len(set(elements)):
            raise AgentInstructionError("Composition scope elements must be unique.")
        updated: list[ComposedOperationGroup] = []
        found = False
        for group in context.composed_groups:
            if group.group_id != group_id:
                updated.append(group)
                continue
            group = self._convert_group_to_target_local(context.source_design, group)
            selector = group.target_selector
            if selector is None:
                targets = [call.semantic_target for call in group.calls if call.semantic_target is not None]
                if not targets:
                    raise CapabilityUnavailableError(
                        f"Persisted composition {group_id!r} has no semantic target to scope."
                    )
                selector = ComposedTargetSelector(
                    targets[0].role,
                    "single",
                    ((targets[0].element_row, targets[0].element_column),),
                )
            updated.append(replace(
                group,
                target_selector=replace(selector, scope=scope, elements=elements),
            ))
            found = True
        if not found:
            raise CapabilityUnavailableError(f"Persisted composition {group_id!r} does not exist.")
        context.composed_groups = updated
        context.changes.append(f"stored composition {group_id} scope set to {scope}")

    @staticmethod
    def _merge_argument_updates(current: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
        """Apply a bounded recursive argument patch without mutating stored JSON."""

        merged = copy.deepcopy(dict(current))
        for key, value in updates.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = AntennaDesignAgent._merge_argument_updates(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    @staticmethod
    def _symbolic_parameter_updates(
        current: Any,
        updates: Any,
        parameter_keys: set[str],
        values: dict[str, float],
    ) -> Any:
        """Redirect numeric edits of a direct symbolic leaf to its owning parameter."""

        if isinstance(current, Mapping) and isinstance(updates, Mapping):
            redirected: dict[str, Any] = {}
            for key, update in updates.items():
                redirected[key] = AntennaDesignAgent._symbolic_parameter_updates(
                    current.get(key), update, parameter_keys, values,
                )
            return redirected
        if (
            isinstance(current, str)
            and current in parameter_keys
            and isinstance(updates, (int, float))
            and not isinstance(updates, bool)
        ):
            number = float(updates)
            existing = values.get(current)
            if existing is not None and not math.isclose(existing, number, rel_tol=0.0, abs_tol=0.0):
                raise AgentInstructionError(
                    f"One composition edit assigned conflicting values to parameter {current!r}."
                )
            values[current] = number
            return current
        return copy.deepcopy(updates)

    def _tool_update_composition_operation(
        self,
        context: _PlanningContext,
        arguments: dict[str, Any],
    ) -> None:
        self._require_exact(
            arguments,
            {"group_id", "operation_id", "argument_updates"},
            "composition.update_operation",
        )
        group_id = str(arguments["group_id"])
        operation_id = str(arguments["operation_id"])
        updates = arguments["argument_updates"]
        if not isinstance(updates, dict) or not updates:
            raise AgentInstructionError("composition.update_operation requires nonempty argument_updates.")

        group_index = next(
            (index for index, group in enumerate(context.composed_groups) if group.group_id == group_id),
            None,
        )
        if group_index is None:
            raise CapabilityUnavailableError(f"Persisted composition {group_id!r} does not exist.")
        group = context.composed_groups[group_index]
        call_index = next(
            (index for index, call in enumerate(group.calls) if call.operation_id == operation_id),
            None,
        )
        if call_index is None:
            raise CapabilityUnavailableError(
                f"Persisted operation {operation_id!r} does not exist in composition {group_id!r}."
            )
        call = group.calls[call_index]
        definition = self._planner_tools.get(call.name)
        if definition is None or definition.primitive_tool_name is None:
            raise CapabilityUnavailableError(
                f"Persisted operation {operation_id!r} uses unavailable primitive {call.name!r}."
            )

        parameter_calls = {
            str(candidate.arguments()["key"]): candidate
            for candidate in group.calls
            if candidate.name == "parameter.create" and candidate.arguments().get("key")
        }
        redirected_values: dict[str, float] = {}
        redirected_updates = self._symbolic_parameter_updates(
            call.arguments(), updates, set(parameter_calls), redirected_values,
        )
        merged_arguments = self._merge_argument_updates(call.arguments(), redirected_updates)
        try:
            validate_tool_arguments(
                definition.argument_schema,
                merged_arguments,
                path=f"{call.name}.arguments",
            )
        except CapabilityError as exc:
            raise AgentInstructionError(
                f"Updated persisted operation {operation_id!r} failed its registered schema: {exc}"
            ) from exc

        updated_calls = list(group.calls)
        updated_calls[call_index] = ComposedToolCall.create(
            call.name,
            merged_arguments,
            semantic_target=call.semantic_target,
            operation_id=call.operation_id,
        )
        for parameter_key, value in redirected_values.items():
            parameter_index = next(
                index
                for index, candidate in enumerate(updated_calls)
                if candidate.operation_id == parameter_calls[parameter_key].operation_id
            )
            parameter_call = updated_calls[parameter_index]
            parameter_arguments = parameter_call.arguments()
            parameter_arguments["value"] = value
            parameter_definition = self._planner_tools["parameter.create"]
            try:
                validate_tool_arguments(
                    parameter_definition.argument_schema,
                    parameter_arguments,
                    path="parameter.create.arguments",
                )
            except CapabilityError as exc:
                raise AgentInstructionError(
                    f"Symbolic parameter {parameter_key!r} failed its registered schema: {exc}"
                ) from exc
            updated_calls[parameter_index] = ComposedToolCall.create(
                parameter_call.name,
                parameter_arguments,
                semantic_target=parameter_call.semantic_target,
                operation_id=parameter_call.operation_id,
            )
            if parameter_key in context.composed_parameters:
                context.composed_parameters[parameter_key] = replace(
                    context.composed_parameters[parameter_key], value=value,
                )

        context.composed_groups[group_index] = replace(group, calls=tuple(updated_calls))
        context.changes.append(
            f"stored composition {group_id} operation {operation_id} updated"
        )

    def _tool_delete_composition(self, context: _PlanningContext, arguments: dict[str, Any]) -> None:
        self._require_exact(arguments, {"group_id"}, "composition.delete")
        group_id = str(arguments["group_id"])
        retained = [group for group in context.composed_groups if group.group_id != group_id]
        if len(retained) == len(context.composed_groups):
            raise CapabilityUnavailableError(f"Persisted composition {group_id!r} does not exist.")
        context.composed_groups = retained
        retained_parameter_keys = {
            str(call.arguments()["key"])
            for group in retained
            for call in group.calls
            if call.name == "parameter.create" and call.arguments().get("key")
        }
        context.composed_parameters = {
            key: value
            for key, value in context.composed_parameters.items()
            if key in retained_parameter_keys
        }
        context.changes.append(f"stored composition {group_id} deleted")

    def _tool_apply_modifier(self, context: _PlanningContext, arguments: dict[str, Any]) -> None:
        self._require_exact(arguments, {"modifier_id"}, "modifier.apply")
        modifier_id = arguments["modifier_id"]
        if not isinstance(modifier_id, str):
            raise AgentInstructionError("modifier.apply requires a string modifier_id.")
        modifier = self.registry.modifier(modifier_id)
        if context.recipe.family not in modifier.applicable_families:
            raise CapabilityUnavailableError(f"Modifier {modifier_id} does not apply to {context.recipe.display_name}.")
        if modifier_id not in context.modifier_ids:
            context.modifier_ids.append(modifier_id)
            for key, value in modifier.defaults().items():
                context.values.setdefault(key, value)
            context.changes.append(f"modifier {modifier_id} applied")

    def _tool_remove_modifier(self, context: _PlanningContext, arguments: dict[str, Any]) -> None:
        self._require_exact(arguments, {"modifier_id"}, "modifier.remove")
        modifier_id = arguments["modifier_id"]
        if modifier_id not in context.modifier_ids:
            raise CapabilityUnavailableError(f"Modifier {modifier_id!r} is not active on this design.")
        context.modifier_ids.remove(modifier_id)
        context.changes.append(f"modifier {modifier_id} removed")

    @staticmethod
    def _reestimate(context: _PlanningContext) -> None:
        if not (context.reset_or_family_change or context.dimension_driver_changed):
            return
        frequency = float(context.values["frequency_ghz"])
        if context.recipe.family == "rectangular_inset_patch":
            material = MATERIALS[str(context.values["material"])]
            length, width = _patch_dimensions(frequency, material.epsilon_r, float(context.values["substrate_thickness_mm"]))
            for key, value in (("patch_length_mm", length), ("patch_width_mm", width), ("inset_depth_mm", round(length * 0.30, 4))):
                if key not in context.explicit_keys:
                    context.values[key] = value
        elif context.recipe.family == "circular_patch":
            material = MATERIALS[str(context.values["material"])]
            radius = _circular_radius(frequency, material.epsilon_r, float(context.values["substrate_thickness_mm"]))
            if "patch_radius_mm" not in context.explicit_keys:
                context.values["patch_radius_mm"] = radius
            if "feed_offset_mm" not in context.explicit_keys:
                context.values["feed_offset_mm"] = round(radius * 0.33, 4)
        elif context.recipe.family == "dipole":
            wavelength = 299.792458 / frequency
            if "feed_gap_mm" not in context.explicit_keys:
                context.values["feed_gap_mm"] = round(wavelength * 0.01, 4)
            if "arm_length_mm" not in context.explicit_keys:
                context.values["arm_length_mm"] = round((0.475 * wavelength - float(context.values["feed_gap_mm"])) / 2, 4)
            if "conductor_radius_mm" not in context.explicit_keys:
                context.values["conductor_radius_mm"] = round(wavelength / 300, 4)

    def _validate_primitive_applicability(
        self,
        design: AntennaDesign,
        call: PlannedToolCall,
        *,
        created_object_ids: set[str],
        consumed_tool_ids: set[str],
    ) -> None:
        arguments = call.arguments
        if call.name == "parameter.create":
            key = str(arguments["key"])
            name = str(arguments.get("name", _solver_parameter_name(key)))
            if key in design.parameter_map():
                raise AgentInstructionError(
                    f"Primitive parameter.create cannot replace existing parameter {key!r}; use parameter.set for installed parameters."
                )
            if name in {item.name for item in design.parameters}:
                raise AgentInstructionError(f"Exported parameter name already exists: {name}.")
            minimum = arguments.get("minimum")
            maximum = arguments.get("maximum")
            value = float(arguments["value"])
            if minimum is not None and value < float(minimum):
                raise AgentInstructionError(f"Parameter {key} is below its declared minimum.")
            if maximum is not None and value > float(maximum):
                raise AgentInstructionError(f"Parameter {key} is above its declared maximum.")
            if minimum is not None and maximum is not None and float(minimum) > float(maximum):
                raise AgentInstructionError(f"Parameter {key} has an invalid minimum/maximum range.")
            return

        if call.name in {"geometry.rectangle_sheet", "geometry.cylinder", "geometry.circle_sheet"}:
            tags = {str(value) for value in arguments["tags"]}
            if "planner_created" not in tags:
                raise AgentInstructionError(f"{call.name} must tag new geometry as planner_created.")
            return

        if call.name in {"geometry.translate", "geometry.rotate"}:
            if str(arguments["object_id"]) not in created_object_ids:
                raise AgentInstructionError(
                    "The planner may transform only geometry created in the same primitive phase."
                )
            return

        if call.name == "geometry.duplicate":
            if str(arguments["source_id"]) not in created_object_ids:
                raise AgentInstructionError("The planner may duplicate only geometry created in the same primitive phase.")
            return

        if call.name not in {"boolean.subtract", "boolean.union"}:
            return
        target_id = str(arguments["target_id"])
        by_id = {item.object_id: item for item in design.geometry}
        target = by_id.get(target_id)
        if target is None:
            raise CapabilityUnavailableError(f"Boolean target does not exist: {target_id}.")
        if self._semantic_role(target) != "radiating_patch_conductor":
            raise CapabilityUnavailableError(
                f"Primitive Boolean planning may target only a radiating patch conductor, not {target_id}."
            )
        target_bounds = object_bounds(design, target)
        tolerance = 1e-7
        for tool_id in (str(value) for value in arguments["tool_ids"]):
            if tool_id not in created_object_ids:
                raise CapabilityUnavailableError(
                    f"Boolean tool {tool_id!r} was not created in the same primitive plan."
                )
            if tool_id in consumed_tool_ids:
                raise AgentInstructionError(f"Boolean tool {tool_id!r} has already been consumed.")
            tool = by_id.get(tool_id)
            if tool is None:
                raise CapabilityUnavailableError(f"Boolean tool does not exist: {tool_id}.")
            if "boolean_tool" not in tool.tags:
                raise AgentInstructionError(f"Boolean tool {tool_id!r} must carry the boolean_tool tag.")
            tool_bounds = object_bounds(design, tool)
            if call.name == "boolean.subtract" and not all((
                target_bounds[0] - tolerance <= tool_bounds[0],
                tool_bounds[1] <= target_bounds[1] + tolerance,
                target_bounds[2] - tolerance <= tool_bounds[2],
                tool_bounds[3] <= target_bounds[3] + tolerance,
                target_bounds[4] - tolerance <= tool_bounds[4],
                tool_bounds[5] <= target_bounds[5] + tolerance,
            )):
                raise AgentInstructionError(
                    f"Subtract tool {tool_id!r} must remain fully inside radiating target {target_id!r}."
                )
            if call.name == "boolean.union" and tool.material_id != target.material_id:
                raise AgentInstructionError("Union tools must use the same material as the radiating target.")

    def _execute_primitive_calls(
        self,
        design: AntennaDesign,
        calls: tuple[PlannedToolCall, ...],
    ) -> tuple[AntennaDesign, tuple[ToolCall, ...], tuple[str, ...]]:
        preflight_errors: list[str] = []
        declared_ids: set[str] = set()
        material_ids = {item.material_id for item in design.materials}
        existing_ids = {item.object_id for item in design.geometry}
        existing_operations = {item.operation_id for item in design.booleans}
        for planned in calls:
            definition = self._planner_tools[planned.name]
            try:
                validate_tool_arguments(definition.argument_schema, planned.arguments, path=f"{planned.name}.arguments")
            except CapabilityError as exc:
                preflight_errors.append(str(exc))
            arguments = planned.arguments
            if planned.name in {"geometry.rectangle_sheet", "geometry.cylinder", "geometry.circle_sheet"}:
                object_id = str(arguments.get("object_id", ""))
                if object_id in existing_ids:
                    preflight_errors.append(f"new object_id {object_id!r} already exists")
                if object_id in material_ids:
                    preflight_errors.append(f"new object_id {object_id!r} must not reuse a material ID")
                declared_ids.add(object_id)
            elif planned.name == "geometry.duplicate":
                declared_ids.add(str(arguments.get("new_id", "")))
            elif planned.name in {"boolean.subtract", "boolean.union"}:
                operation_id = str(arguments.get("operation_id", ""))
                target_id = str(arguments.get("target_id", ""))
                tool_ids = {str(value) for value in arguments.get("tool_ids", ())}
                if operation_id in existing_operations or operation_id == target_id:
                    preflight_errors.append("operation_id must be new and distinct from target_id")
                if target_id in tool_ids:
                    preflight_errors.append("target_id must never appear in tool_ids")
                missing_tools = tool_ids - declared_ids
                if missing_tools:
                    preflight_errors.append(
                        "tool_ids must reference earlier planner-created geometry; missing: "
                        + ", ".join(sorted(missing_tools))
                    )
        if preflight_errors:
            raise AgentInstructionError(
                "Primitive plan preflight failed: " + "; ".join(dict.fromkeys(preflight_errors)) + "."
            )
        created_object_ids: set[str] = set()
        consumed_tool_ids: set[str] = set()
        created_parameter_keys: set[str] = set()
        executed: list[ToolCall] = []
        changes: list[str] = []
        for planned in calls:
            definition = self._planner_tools[planned.name]
            self._validate_primitive_applicability(
                design,
                planned,
                created_object_ids=created_object_ids,
                consumed_tool_ids=consumed_tool_ids,
            )
            tool_arguments = dict(planned.arguments)
            if planned.name == "parameter.create":
                tool_arguments["name"] = _solver_parameter_name(str(tool_arguments["key"]))
            call = ToolCall(definition.primitive_tool_name or planned.name, tool_arguments)
            try:
                design = self.registry.execute(design, call)
            except (CapabilityError, DesignValidationError) as exc:
                raise AgentInstructionError(f"Primitive call {planned.name} failed: {exc}") from exc
            executed.append(call)
            if planned.name == "parameter.create":
                created_parameter_keys.add(str(planned.arguments["key"]))
                changes.append(f"parameter {_solver_parameter_name(str(planned.arguments['key']))} created")
            elif planned.name in {"geometry.rectangle_sheet", "geometry.cylinder", "geometry.circle_sheet"}:
                created_object_ids.add(str(planned.arguments["object_id"]))
                changes.append(f"geometry {planned.arguments['object_id']} created")
            elif planned.name == "geometry.duplicate":
                created_object_ids.add(str(planned.arguments["new_id"]))
                changes.append(f"geometry {planned.arguments['new_id']} duplicated")
            elif planned.name.startswith("boolean."):
                tool_ids = {str(value) for value in planned.arguments["tool_ids"]}
                consumed_tool_ids.update(tool_ids)
                changes.append(
                    f"{planned.name.removeprefix('boolean.')} {', '.join(sorted(tool_ids))} from/into {planned.arguments['target_id']}"
                )
        unused_objects = created_object_ids - consumed_tool_ids
        if unused_objects:
            raise AgentInstructionError(
                "Planner-created geometry must be consumed by a Boolean operation: "
                + ", ".join(sorted(unused_objects)) + "."
            )
        if created_parameter_keys:
            expressions = " ".join(
                str(value)
                for item in design.geometry
                if item.object_id in created_object_ids
                for value in item.dimension_map().values()
            )
            unused_parameters = {key for key in created_parameter_keys if key not in expressions}
            if unused_parameters:
                raise AgentInstructionError(
                    "Planner-created parameters must drive planned geometry: "
                    + ", ".join(sorted(unused_parameters)) + "."
                )
        try:
            design = validate_design(replace(design, validation=()))
        except (CapabilityError, DesignValidationError) as exc:
            raise AgentInstructionError(f"The composed primitive design failed validation: {exc}") from exc
        return design, tuple(executed), tuple(changes)

    def _execute_initial_llm_plan(self, llm_plan: LLMToolPlan) -> AgentResult:
        """Create the first canonical design without a hidden default antenna."""

        if not llm_plan.calls or llm_plan.calls[0].name != "recipe.select":
            raise AgentInstructionError(
                "An empty project requires recipe.select as the first planning call."
            )
        first_call = llm_plan.calls[0]
        definition = self._planner_tools["recipe.select"]
        try:
            validate_tool_arguments(
                copy.deepcopy(definition.argument_schema),
                first_call.arguments,
                path="recipe.select.arguments",
            )
            recipe_id = first_call.arguments["recipe_id"]
            recipe = self.registry.recipe(recipe_id)
            candidate, _discarded_calls = recipe.build(
                self.registry,
                recipe.defaults(),
                revision=0,
            )
        except (KeyError, CapabilityError, DesignValidationError) as exc:
            raise AgentInstructionError(
                f"Initial recipe selection failed validation: {exc}"
            ) from exc

        context = _PlanningContext(
            source_design=candidate,
            recipe=recipe,
            values=recipe.defaults(),
            modifier_ids=[],
            explicit_keys=set(),
            changes=[],
            composed_groups=[],
            composed_parameters={},
            explicit_updates={},
            excitation=candidate.excitation,
            reset_or_family_change=True,
        )
        if definition.handler is None:
            raise CapabilityUnavailableError(
                "Planning tool recipe.select has no deterministic executor."
            )
        definition.handler(context, dict(first_call.arguments))

        for call in llm_plan.calls[1:]:
            if call.name != "parameter.set":
                raise AgentInstructionError(
                    "Only parameter.set may follow recipe.select while creating the first design. "
                    "Apply modifiers or composed geometry in a later turn."
                )
            parameter_definition = self._planner_tools["parameter.set"]
            try:
                validate_tool_arguments(
                    copy.deepcopy(parameter_definition.argument_schema),
                    call.arguments,
                    path="parameter.set.arguments",
                )
            except CapabilityError as exc:
                raise AgentInstructionError(
                    f"Planning call parameter.set failed schema validation: {exc}"
                ) from exc
            if parameter_definition.handler is None:
                raise CapabilityUnavailableError(
                    "Planning tool parameter.set has no deterministic executor."
                )
            parameter_definition.handler(context, dict(call.arguments))

        self._reestimate(context)
        try:
            built, executed = self._build_with_modifiers(
                context.recipe,
                context.values,
                (),
                design_id=candidate.design_id,
                revision=0,
            )
            built = validate_design(built)
        except (CapabilityError, DesignValidationError) as exc:
            raise AgentInstructionError(
                f"The initial antenna design failed validation: {exc}"
            ) from exc

        updates = tuple(
            (key, context.explicit_updates[key])
            for key in sorted(context.explicit_updates)
        )
        plan = AgentPlan(
            context.recipe.recipe_id,
            "LLM initial-design tool plan",
            updates,
            tuple(context.recipe.required_tools),
            (),
            llm_plan.calls,
            llm_plan.message,
        )
        return AgentResult(
            built,
            plan,
            tuple(executed),
            tuple(context.changes) or ("validated initial antenna design created",),
        )

    def execute_llm_plan(
        self,
        design: AntennaDesign | None,
        llm_plan: LLMToolPlan,
    ) -> AgentResult:
        if llm_plan.status == "clarify":
            raise PlannerClarificationRequired(llm_plan.message or "Clarify the antenna request before I change the design.")
        if llm_plan.status == "refuse":
            raise PlannerRefusal(llm_plan.message or "The installed validated capabilities cannot represent that request.")
        if llm_plan.status != "execute":
            raise AgentInstructionError("The LLM plan status is invalid.")
        if design is None:
            return self._execute_initial_llm_plan(llm_plan)
        recipe = self.registry.recipe(design.recipe_id)
        context = _PlanningContext(
            source_design=design,
            recipe=recipe,
            values=self.values_for_design(design),
            modifier_ids=list(active_modifier_ids(design)),
            explicit_keys=set(),
            changes=[],
            composed_groups=list(design.composed_operations),
            composed_parameters=self._composed_parameter_map(design),
            explicit_updates={},
            excitation=design.excitation,
        )
        primitive_calls: list[PlannedToolCall] = []
        primitive_phase = False
        configuration_used = False
        for call in llm_plan.calls:
            try:
                definition = self._planner_tools[call.name]
            except KeyError as exc:
                raise CapabilityUnavailableError(f"The LLM requested an unregistered planning tool: {call.name}.") from exc
            if definition.primitive_tool_name:
                primitive_phase = True
                primitive_calls.append(call)
                continue
            try:
                schema = copy.deepcopy(definition.argument_schema)
                if call.name == "parameter.set":
                    schema["properties"]["key"]["enum"] = sorted({
                        *schema["properties"]["key"].get("enum", ()),
                        *design.parameter_map(),
                    })
                elif call.name in {"composition.set_scope", "composition.delete"}:
                    schema["properties"]["group_id"]["enum"] = [
                        group.group_id for group in design.composed_operations
                    ]
                elif call.name == "composition.update_operation":
                    schema["properties"]["group_id"]["enum"] = [
                        group.group_id for group in design.composed_operations
                    ]
                    schema["properties"]["operation_id"]["enum"] = [
                        stored.operation_id
                        for group in design.composed_operations
                        for stored in group.calls
                    ]
                validate_tool_arguments(schema, call.arguments, path=f"{call.name}.arguments")
            except CapabilityError as exc:
                raise AgentInstructionError(f"Planning call {call.name} failed schema validation: {exc}") from exc
            if primitive_phase:
                raise AgentInstructionError("Recipe, parameter, and modifier actions must appear before primitive geometry calls.")
            if definition.handler is None:
                raise CapabilityUnavailableError(f"Planning tool {call.name} has no deterministic executor.")
            definition.handler(context, dict(call.arguments))
            configuration_used = True
        self._reestimate(context)
        modifier_ids = tuple(context.modifier_ids)
        if configuration_used:
            built, base_executed = self._build_with_modifiers(
                context.recipe,
                context.values,
                modifier_ids,
                design_id=design.design_id,
                revision=design.revision + 1,
            )
            if not (context.reset_or_family_change and not context.excitation_explicit):
                built = realize_excitation(replace(
                    built,
                    excitation=self._reconcile_excitation(context.excitation, built),
                    validation=(),
                ))
            built, replayed_executed, replay_changes = self._replay_composed_operations(
                built,
                tuple(context.composed_groups),
            )
        else:
            built = replace(design, revision=design.revision + 1, validation=())
            base_executed = ()
            replayed_executed = ()
            replay_changes = ()
        new_group = None
        if primitive_calls:
            if any(call.name in {"boolean.subtract", "boolean.union"} for call in primitive_calls):
                new_group = self._capture_operation_group(built, tuple(primitive_calls))
                targets = self._resolve_target_selector(built, self._selector_for_group(new_group))
                executable_primitive_calls = self._instantiate_group_calls(built, new_group, targets)
            else:
                executable_primitive_calls = tuple(primitive_calls)
            built, primitive_executed, primitive_changes = self._execute_primitive_calls(
                built,
                executable_primitive_calls,
            )
            if new_group is None:
                new_group = self._capture_operation_group(built, tuple(primitive_calls))
            built = validate_design(replace(
                built,
                composed_operations=(*built.composed_operations, new_group),
                validation=(),
            ))
        else:
            built, primitive_executed, primitive_changes = validate_design(built), (), ()
        executed = (*base_executed, *replayed_executed, *primitive_executed)
        updates = tuple((key, context.explicit_updates[key]) for key in sorted(context.explicit_updates))
        planned_primitives = tuple(dict.fromkeys((
            *context.recipe.required_tools,
            *(tool for modifier_id in modifier_ids for tool in self.registry.modifier(modifier_id).required_tools),
            *(call.name for call in primitive_calls),
        )))
        plan = AgentPlan(context.recipe.recipe_id, "LLM tool plan", updates, planned_primitives, modifier_ids, llm_plan.calls, llm_plan.message)
        return AgentResult(
            built,
            plan,
            tuple(executed),
            (*context.changes, *replay_changes, *primitive_changes) or ("validated tool plan applied",),
        )


def create_default_agent() -> AntennaDesignAgent:
    return AntennaDesignAgent(create_tool_registry())
