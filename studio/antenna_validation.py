"""Layered deterministic validation for solver-neutral antenna designs."""

from __future__ import annotations

import math
from dataclasses import replace

from studio.antenna_design import (
    AntennaDesign,
    DesignValidationError,
    EXCITATION_REALIZATION_STATUSES,
    EXCITATION_STRATEGIES,
    FEED_NETWORK_TYPES,
    ValidationRecord,
    element_coordinate,
    evaluate_scalar,
    object_bounds,
    resolve_parameter_values,
)


SUPPORTED_PRIMITIVES = frozenset({"box", "sheet_rectangle", "cylinder", "sheet_circle"})


def _record(stage: str, errors: list[str]) -> ValidationRecord:
    return ValidationRecord(stage=stage, passed=not errors, messages=tuple(errors))


def _circular_coax_errors(design: AntennaDesign, values: dict[str, float]) -> list[str]:
    """Validate the canonical probe/coax transition before any solver adapter runs."""

    errors: list[str] = []
    by_id = {item.object_id: item for item in design.geometry}
    materials = {item.material_id: item for item in design.materials}
    booleans = {
        (operation.operation, operation.target_id, tool_id)
        for operation in design.booleans
        for tool_id in operation.tool_ids
    }

    def tagged(element: int, role: str):
        element_tag = f"element_{element}"
        return [item for item in design.geometry if role in item.tags and element_tag in item.tags]

    def radius(item) -> float:
        return evaluate_scalar(item.dimension_map()["radius"], values)

    def center(item) -> tuple[float, float]:
        dimensions = item.dimension_map()
        return (
            evaluate_scalar(dimensions["center_1"], values) + item.transform.translate_mm[0],
            evaluate_scalar(dimensions["center_2"], values) + item.transform.translate_mm[1],
        )

    ground = by_id.get("ground")
    if ground is None:
        return ["The circular-patch coax feed requires a ground conductor."]
    if materials.get(ground.material_id) is None or materials[ground.material_id].kind != "conductor":
        errors.append("The circular-patch ground reference must be conductive.")

    for element in range(1, design.array.element_count + 1):
        port = next((item for item in design.ports if item.element_index == element), None)
        if port is None:
            continue
        if port.kind != "modal_cross_section":
            errors.append(
                f"{port.port_id} uses invalid axial discrete-port excitation; circular probe feeds require a modal coax cross-section."
            )
            continue
        if port.mode_count != 1:
            errors.append(f"{port.port_id} must excite exactly one coaxial mode.")
        if not port.signal_terminal or not port.reference_terminal:
            errors.append(f"{port.port_id} requires explicit signal and reference terminals.")
            continue
        if port.signal_terminal == port.reference_terminal:
            errors.append(f"{port.port_id} signal and reference terminals must be different conductors.")
        signal = by_id.get(port.signal_terminal)
        reference = by_id.get(port.reference_terminal)
        if signal is None or reference is None:
            errors.append(f"{port.port_id} references missing signal or reference conductor geometry.")
            continue
        if any(materials.get(item.material_id) is None or materials[item.material_id].kind != "conductor" for item in (signal, reference)):
            errors.append(f"{port.port_id} signal and reference terminals must resolve to conductors.")
        if signal.object_id == ground.object_id:
            errors.append(f"{port.port_id} probe and ground must be different conductor objects.")

        if port.cross_section is None:
            errors.append(f"{port.port_id} requires a modal cross-section geometry reference.")
            continue
        dielectric = by_id.get(port.cross_section.geometry_id)
        if dielectric is None:
            errors.append(f"{port.port_id} references a missing coax dielectric cross-section.")
            continue
        if port.cross_section.face not in {"z_min", "z_max"}:
            errors.append(f"{port.port_id} modal cross-section must reference a z-normal face.")
        dielectric_material = materials.get(dielectric.material_id)
        if dielectric_material is None or dielectric_material.kind != "dielectric":
            errors.append(f"{port.port_id} modal cross-section must resolve to dielectric geometry.")

        patches = tagged(element, "circular_patch_element")
        probes = tagged(element, "probe_feed")
        clearances = tagged(element, "ground_clearance_tool")
        dielectrics = tagged(element, "coax_dielectric")
        references = tagged(element, "coax_reference")
        if any(len(items) != 1 for items in (patches, probes, clearances, dielectrics, references)):
            errors.append(f"Circular-patch element {element} requires one patch, probe, clearance, coax dielectric, and coax reference conductor.")
            continue
        patch, probe, clearance, tagged_dielectric, tagged_reference = (
            patches[0], probes[0], clearances[0], dielectrics[0], references[0]
        )
        if signal.object_id != probe.object_id or reference.object_id != tagged_reference.object_id:
            errors.append(f"{port.port_id} terminals do not resolve to this element's probe and coax reference conductor.")
        if dielectric.object_id != tagged_dielectric.object_id:
            errors.append(f"{port.port_id} cross-section does not resolve to this element's coax dielectric.")

        probe_radius = radius(probe)
        clearance_radius = radius(clearance)
        dielectric_radius = radius(dielectric)
        reference_radius = radius(reference)
        if not probe_radius < clearance_radius:
            errors.append(f"Element {element} probe is not isolated by a larger ground clearance.")
        if abs(clearance_radius - dielectric_radius) > 1e-9:
            errors.append(f"Element {element} ground clearance and coax dielectric radii must coincide.")
        if not dielectric_radius < reference_radius:
            errors.append(f"Element {element} coax reference conductor must surround the dielectric.")
        if ("subtract", ground.object_id, clearance.object_id) not in booleans:
            errors.append(f"Element {element} probe lacks a Boolean ground-clearance hole.")
        if not any(
            operation.operation == "subtract"
            and operation.target_id == dielectric.object_id
            and probe.object_id not in operation.tool_ids
            and any("coax_pin_bore_tool" in by_id[tool_id].tags for tool_id in operation.tool_ids if tool_id in by_id)
            for operation in design.booleans
        ):
            errors.append(f"Element {element} coax dielectric lacks an isolated inner-pin bore.")
        if not any(
            operation.operation == "subtract"
            and operation.target_id == reference.object_id
            and any("coax_outer_bore_tool" in by_id[tool_id].tags for tool_id in operation.tool_ids if tool_id in by_id)
            for operation in design.booleans
        ):
            errors.append(f"Element {element} coax reference conductor lacks its dielectric bore.")

        patch_bounds = object_bounds(design, patch)
        probe_bounds = object_bounds(design, probe)
        dielectric_bounds = object_bounds(design, dielectric)
        reference_bounds = object_bounds(design, reference)
        patch_center = center(patch)
        probe_center = center(probe)
        dielectric_center = center(dielectric)
        reference_center = center(reference)
        radial_distance = math.hypot(probe_center[0] - patch_center[0], probe_center[1] - patch_center[1])
        probe_patch_overlap = min(probe_bounds[5], patch_bounds[5]) - max(probe_bounds[4], patch_bounds[4])
        if radial_distance + probe_radius > radius(patch) + 1e-9 or probe_patch_overlap <= 1e-9:
            errors.append(f"Element {element} probe does not contact the circular patch conductor.")
        ground_bounds = object_bounds(design, ground)
        clearance_bounds = object_bounds(design, clearance)
        if clearance_bounds[4] > ground_bounds[4] + 1e-9 or clearance_bounds[5] < ground_bounds[5] - 1e-9:
            errors.append(f"Element {element} ground clearance does not pass through the full ground thickness.")
        if any(
            math.hypot(point[0] - probe_center[0], point[1] - probe_center[1]) > 1e-9
            for point in (dielectric_center, reference_center)
        ):
            errors.append(f"Element {element} coax pin, dielectric, and reference conductor must be concentric.")
        if abs(reference_bounds[5] - ground_bounds[5]) > 1e-9 or reference_radius <= clearance_radius:
            errors.append(f"Element {element} coax reference conductor is not electrically joined to the ground plane.")

        cross_section_z = dielectric_bounds[4] if port.cross_section.face == "z_min" else dielectric_bounds[5]
        positive = tuple(evaluate_scalar(value, values) for value in port.positive_point)
        negative = tuple(evaluate_scalar(value, values) for value in port.negative_point)
        if abs(positive[2] - cross_section_z) > 1e-9 or abs(negative[2] - cross_section_z) > 1e-9:
            errors.append(f"{port.port_id} terminals must lie on its modal cross-section plane.")
        positive_radius = math.hypot(positive[0] - probe_center[0], positive[1] - probe_center[1])
        negative_radius = math.hypot(negative[0] - probe_center[0], negative[1] - probe_center[1])
        if positive_radius > probe_radius + 1e-9:
            errors.append(f"{port.port_id} signal terminal is outside the inner probe conductor.")
        if not dielectric_radius - 1e-9 <= negative_radius <= reference_radius + 1e-9:
            errors.append(f"{port.port_id} reference terminal is outside the coax outer conductor.")
        if not probe_bounds[4] - 1e-9 <= cross_section_z <= probe_bounds[5] + 1e-9:
            errors.append(f"{port.port_id} signal conductor does not intersect the modal cross-section.")
        if not reference_bounds[4] - 1e-9 <= cross_section_z <= reference_bounds[5] + 1e-9:
            errors.append(f"{port.port_id} reference conductor does not intersect the modal cross-section.")
    return errors


def validate_design(design: AntennaDesign, *, raise_on_error: bool = True) -> AntennaDesign:
    """Validate parameters, primitives, recipe structure, and simulation setup."""

    parameter_errors: list[str] = []
    keys = [item.key for item in design.parameters]
    names = [item.name for item in design.parameters]
    if len(keys) != len(set(keys)):
        parameter_errors.append("Parameter keys must be unique.")
    if len(names) != len(set(names)):
        parameter_errors.append("Exported parameter names must be unique.")
    try:
        values = resolve_parameter_values(design)
    except DesignValidationError as exc:
        parameter_errors.append(str(exc))
        values = {}
    for parameter in design.parameters:
        value = values.get(parameter.key, parameter.value)
        if not math.isfinite(float(value)):
            parameter_errors.append(f"{parameter.label} must be finite.")
        if parameter.minimum is not None and value < parameter.minimum:
            parameter_errors.append(f"{parameter.label} must be at least {parameter.minimum:g}.")
        if parameter.maximum is not None and value > parameter.maximum:
            parameter_errors.append(f"{parameter.label} must be at most {parameter.maximum:g}.")
        if parameter.integer and not float(value).is_integer():
            parameter_errors.append(f"{parameter.label} must be a whole number.")

    primitive_errors: list[str] = []
    material_ids = {item.material_id for item in design.materials}
    object_ids = [item.object_id for item in design.geometry]
    if len(object_ids) != len(set(object_ids)):
        primitive_errors.append("Geometry object IDs must be unique.")
    for item in design.geometry:
        if item.primitive not in SUPPORTED_PRIMITIVES:
            primitive_errors.append(f"Unsupported primitive {item.primitive} on {item.object_id}.")
            continue
        if item.material_id not in material_ids:
            primitive_errors.append(f"{item.object_id} references unknown material {item.material_id}.")
        try:
            bounds = object_bounds(design, item)
        except (KeyError, DesignValidationError) as exc:
            primitive_errors.append(f"{item.object_id}: {exc}")
            continue
        if not all(math.isfinite(value) for value in bounds):
            primitive_errors.append(f"{item.object_id} has non-finite bounds.")
        if not (bounds[0] < bounds[1] and bounds[2] < bounds[3] and bounds[4] < bounds[5]):
            primitive_errors.append(f"{item.object_id} must have positive extent on every axis.")
    known_objects = set(object_ids)
    for operation in design.booleans:
        if operation.operation not in {"subtract", "union"}:
            primitive_errors.append(f"Unsupported Boolean operation: {operation.operation}.")
        referenced = {operation.target_id, *operation.tool_ids}
        if not referenced <= known_objects:
            primitive_errors.append(f"{operation.operation_id} references missing geometry.")

    recipe_errors: list[str] = []
    if design.array.rows < 1 or design.array.columns < 1:
        recipe_errors.append("Array rows and columns must be positive.")
    if design.array.element_count > 64:
        recipe_errors.append("The experimental builder is limited to 64 elements.")
    if design.array.element_count > 1 and design.array.spacing_mm <= 0:
        recipe_errors.append("Replicated elements require positive spacing.")
    expected_ports = design.array.element_count
    if len(design.ports) != expected_ports:
        recipe_errors.append(
            f"{design.family} requires one port per element ({expected_ports} expected)."
        )
    for port in design.ports:
        try:
            positive = tuple(evaluate_scalar(value, values) for value in port.positive_point)
            negative = tuple(evaluate_scalar(value, values) for value in port.negative_point)
        except DesignValidationError as exc:
            recipe_errors.append(f"{port.port_id}: {exc}")
            continue
        if positive == negative:
            recipe_errors.append(f"{port.port_id} endpoints must be distinct.")
        if port.impedance_ohms <= 0:
            recipe_errors.append(f"{port.port_id} impedance must be positive.")
    tags = [tag for item in design.geometry for tag in item.tags]
    if design.family == "rectangular_inset_patch":
        if tags.count("patch_element") != expected_ports * 4:
            recipe_errors.append("Each inset patch must contain four composed conductor pieces.")
        if tags.count("substrate") != 1 or tags.count("ground") != 1:
            recipe_errors.append("The inset-patch recipe requires one substrate and one ground.")
    elif design.family == "circular_patch":
        if tags.count("circular_patch_element") != expected_ports:
            recipe_errors.append("Each circular-patch element requires one circular conductor.")
        if tags.count("probe_feed") != expected_ports:
            recipe_errors.append("Each circular-patch element requires one probe feed.")
        if values and not primitive_errors:
            recipe_errors.extend(_circular_coax_errors(design, values))
    elif design.family == "dipole":
        if tags.count("dipole_arm") != expected_ports * 2:
            recipe_errors.append("Each dipole element requires two conductor arms.")
    else:
        recipe_errors.append(f"No structural validator is installed for {design.family}.")

    excitation_errors: list[str] = []
    excitation = design.excitation
    if excitation.strategy not in EXCITATION_STRATEGIES:
        excitation_errors.append(f"Unknown excitation strategy: {excitation.strategy}.")
    if excitation.realization_status not in EXCITATION_REALIZATION_STATUSES:
        excitation_errors.append(
            f"Unknown excitation realization status: {excitation.realization_status}."
        )
    if excitation.feed_network not in FEED_NETWORK_TYPES:
        excitation_errors.append(f"Unknown feed-network intent: {excitation.feed_network}.")
    assignment_elements = [
        (assignment.element_row, assignment.element_column)
        for assignment in excitation.element_assignments
    ]
    if len(assignment_elements) != len(set(assignment_elements)):
        excitation_errors.append("Excitation assignments must identify each logical element at most once.")
    assigned_port_ids = [
        assignment.port_id
        for assignment in excitation.element_assignments
        if assignment.port_id is not None
    ]
    if len(assigned_port_ids) != len(set(assigned_port_ids)):
        excitation_errors.append("Excitation assignments must map physical ports one-to-one.")
    port_map = {port.port_id: port for port in design.ports}
    for assignment in excitation.element_assignments:
        coordinates = (assignment.element_row, assignment.element_column)
        if (
            assignment.element_row < 1 or assignment.element_column < 1
            or assignment.element_row > design.array.rows
            or assignment.element_column > design.array.columns
        ):
            excitation_errors.append(
                f"Excitation assignment {assignment.excitation_id!r} is outside the current array."
            )
        if not assignment.excitation_id:
            excitation_errors.append("Excitation assignments require a stable excitation ID.")
        if assignment.status not in EXCITATION_REALIZATION_STATUSES:
            excitation_errors.append(
                f"Excitation assignment {assignment.excitation_id!r} has an invalid status."
            )
        if assignment.port_id is not None:
            port = port_map.get(assignment.port_id)
            if port is None:
                excitation_errors.append(
                    f"Excitation assignment {assignment.excitation_id!r} references missing port {assignment.port_id!r}."
                )
            elif element_coordinate(design.array, port.element_index) != coordinates:
                excitation_errors.append(
                    f"Excitation assignment {assignment.excitation_id!r} references a port on another element."
                )
        if assignment.status == "realized" and assignment.port_id is None:
            excitation_errors.append(
                f"Realized excitation assignment {assignment.excitation_id!r} requires a physical port reference."
            )
    if excitation.strategy == "legacy_recipe" and excitation.realization_status != "legacy":
        excitation_errors.append("Legacy recipe excitation must use legacy realization status.")
    if excitation.realization_status == "legacy" and excitation.strategy != "legacy_recipe":
        excitation_errors.append("Legacy realization status is reserved for legacy recipe excitation.")
    expected_network = {
        "single_element_feed": "none",
        "independent_ports": "independent",
        "corporate_feed": "corporate",
        "series_feed": "series",
        "custom": "custom",
        "legacy_recipe": "unresolved",
    }.get(excitation.strategy)
    if expected_network is not None and excitation.feed_network != expected_network:
        excitation_errors.append(
            f"Excitation strategy {excitation.strategy!r} requires feed-network intent {expected_network!r}."
        )
    if excitation.realization_status == "realized":
        if excitation.strategy not in {"single_element_feed", "independent_ports"}:
            excitation_errors.append(
                "Only installed single-element or independent-port realization may be marked realized."
            )
        if any(assignment.status != "realized" for assignment in excitation.element_assignments):
            excitation_errors.append("Every assignment in a realized excitation must be realized.")
        if excitation.strategy == "single_element_feed":
            if design.array.element_count != 1 or len(excitation.element_assignments) != 1:
                excitation_errors.append(
                    "Realized single-element excitation requires exactly one assignment on a one-element design."
                )
        elif excitation.strategy == "independent_ports":
            expected_elements = {
                (row, column)
                for row in range(1, design.array.rows + 1)
                for column in range(1, design.array.columns + 1)
            }
            if set(assignment_elements) != expected_elements:
                excitation_errors.append(
                    "Realized independent excitation requires one assignment for every array element."
                )
            if set(assigned_port_ids) != set(port_map) or len(design.ports) != design.array.element_count:
                excitation_errors.append(
                    "Realized independent excitation requires exactly one mapped physical port per element."
                )
            if any(port.kind != "discrete" for port in design.ports):
                excitation_errors.append(
                    "Realized independent excitation currently requires discrete physical ports."
                )
            local_feeds = [
                item for item in design.geometry if "independent_local_feed" in item.tags
            ]
            local_feed_elements = [
                tag
                for item in local_feeds
                for tag in item.tags
                if tag.startswith("element_") and tag.removeprefix("element_").isdigit()
            ]
            if (
                len(local_feeds) != design.array.element_count
                or len(local_feed_elements) != design.array.element_count
                or len(set(local_feed_elements)) != design.array.element_count
            ):
                excitation_errors.append(
                    "Realized independent excitation requires one local feed conductor per element."
                )
            segments = []
            try:
                for port in design.ports:
                    positive = tuple(evaluate_scalar(value, values) for value in port.positive_point)
                    negative = tuple(evaluate_scalar(value, values) for value in port.negative_point)
                    direct = tuple(round(value, 12) for value in (*negative, *positive))
                    reverse = tuple(round(value, 12) for value in (*positive, *negative))
                    segments.append(min(direct, reverse))
            except (KeyError, DesignValidationError) as exc:
                excitation_errors.append(f"Independent excitation port endpoints could not be evaluated: {exc}")
            if len(segments) != len(set(segments)):
                excitation_errors.append("Realized independent excitation ports must be physically distinct.")
    elif any(assignment.status == "realized" for assignment in excitation.element_assignments):
        excitation_errors.append(
            "An excitation assignment cannot be marked realized when the current excitation is not realized."
        )
    if excitation.realization_status == "unsupported" and assigned_port_ids:
        excitation_errors.append(
            "Unsupported excitation intent cannot link physical ports as its realization."
        )
    if excitation.strategy in {"corporate_feed", "series_feed"} and excitation.realization_status != "unsupported":
        excitation_errors.append(
            f"{excitation.strategy} synthesis is not installed and must be marked unsupported."
        )

    simulation_errors: list[str] = []
    try:
        minimum = evaluate_scalar(design.simulation.frequency_min_ghz, values)
        maximum = evaluate_scalar(design.simulation.frequency_max_ghz, values)
        if minimum <= 0 or maximum <= minimum:
            simulation_errors.append("Simulation frequency range must be positive and ordered.")
    except DesignValidationError as exc:
        simulation_errors.append(str(exc))

    composition_errors: list[str] = []
    group_ids = [group.group_id for group in design.composed_operations]
    if len(group_ids) != len(set(group_ids)):
        composition_errors.append("Composed-operation group IDs must be unique.")
    for group in design.composed_operations:
        if not group.group_id or not group.source_recipe_id or not group.source_family:
            composition_errors.append("Composed-operation groups require stable identity and source topology.")
        if not group.calls:
            composition_errors.append(f"Composed-operation group {group.group_id!r} must contain calls.")
        operation_ids = [call.operation_id for call in group.calls]
        if any(not operation_id for operation_id in operation_ids):
            composition_errors.append(f"{group.group_id} contains a composed call without stable identity.")
        if len(operation_ids) != len(set(operation_ids)):
            composition_errors.append(f"{group.group_id} composed-operation IDs must be unique.")
        selector = group.target_selector
        if group.coordinate_frame not in {"world", "target_local"}:
            composition_errors.append(f"{group.group_id} has an invalid coordinate frame.")
        if selector is None:
            composition_errors.append(
                f"{group.group_id} must declare an explicit composed-feature target scope."
            )
        else:
            if not selector.role or selector.scope not in {"single", "selected", "all"}:
                composition_errors.append(f"{group.group_id} has an invalid target selector.")
            if selector.scope == "all" and selector.elements:
                composition_errors.append(f"{group.group_id} all-element scope must not store explicit elements.")
            if selector.scope == "single" and len(selector.elements) != 1:
                composition_errors.append(f"{group.group_id} single-element scope requires exactly one element.")
            if selector.scope == "selected" and not selector.elements:
                composition_errors.append(f"{group.group_id} selected-element scope requires elements.")
            if len(selector.elements) != len(set(selector.elements)) or any(
                row < 1 or column < 1 for row, column in selector.elements
            ):
                composition_errors.append(f"{group.group_id} contains invalid selected elements.")
        for call in group.calls:
            try:
                call.arguments()
            except DesignValidationError as exc:
                composition_errors.append(f"{group.group_id}: {exc}")
            target = call.semantic_target
            if target is not None and (
                not target.role or target.element_row < 1 or target.element_column < 1
            ):
                composition_errors.append(f"{group.group_id} contains an invalid semantic target.")

    records = (
        _record("parameter_update", parameter_errors),
        _record("primitive_geometry", primitive_errors),
        _record("antenna_recipe", recipe_errors),
        _record("excitation_definition", excitation_errors),
        _record("simulation_setup", simulation_errors),
        _record("composed_operations", composition_errors),
    )
    validated = replace(design, validation=records)
    errors = [message for record in records for message in record.messages]
    if errors and raise_on_error:
        raise DesignValidationError("; ".join(errors))
    return validated


def validate_adapter_input(design: AntennaDesign, adapter_name: str) -> ValidationRecord:
    errors: list[str] = []
    if any(not record.passed for record in design.validation):
        errors.append("The canonical design has not passed all prior validation stages.")
    return _record(f"{adapter_name}_adapter", errors)
