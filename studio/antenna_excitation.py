"""Deterministic physical realization of solver-neutral excitation intent."""

from __future__ import annotations

from dataclasses import replace

from studio.antenna_design import (
    AntennaDesign,
    DesignValidationError,
    ExcitationAssignment,
    ExcitationDefinition,
    GeometryObject,
    PortSpec,
    element_coordinate,
    evaluate_scalar,
    resolve_parameter_values,
)


INDEPENDENT_LOCAL_PORT_MODEL = "local_vertical_discrete_port"


def supports_independent_port_synthesis(design: AntennaDesign) -> bool:
    """Return whether the installed deterministic synthesizer supports this topology."""

    return (
        design.recipe_id == "inset_patch_v2"
        and design.family == "rectangular_inset_patch"
    )


def _replace_dimension(item: GeometryObject, key: str, value) -> GeometryObject:
    dimensions = item.dimension_map()
    if key not in dimensions:
        raise DesignValidationError(
            f"Independent-port synthesis requires {key!r} on {item.object_id!r}."
        )
    dimensions[key] = value
    tags = item.tags if "independent_local_feed" in item.tags else (*item.tags, "independent_local_feed")
    return replace(item, dimensions=tuple(dimensions.items()), tags=tags)


def _segment_key(design: AntennaDesign, port: PortSpec) -> tuple[tuple[float, ...], tuple[float, ...]]:
    values = resolve_parameter_values(design)
    positive = tuple(evaluate_scalar(value, values) for value in port.positive_point)
    negative = tuple(evaluate_scalar(value, values) for value in port.negative_point)
    direct = tuple(round(value, 12) for value in (*negative, *positive))
    reverse = tuple(round(value, 12) for value in (*positive, *negative))
    return (direct, reverse) if direct <= reverse else (reverse, direct)


def realize_excitation(design: AntennaDesign) -> AntennaDesign:
    """Realize the current excitation intent without selecting a strategy implicitly."""

    intent = design.excitation
    if intent.strategy != "independent_ports":
        return design
    if not supports_independent_port_synthesis(design):
        return replace(
            design,
            excitation=replace(
                intent,
                realization_status="unsupported",
                element_assignments=tuple(
                    replace(assignment, port_id=None, status="unresolved")
                    for assignment in intent.element_assignments
                ),
                unresolved_requirements=(
                    "Independent local-port synthesis is installed only for rectangular inset-fed patch designs.",
                ),
            ),
            validation=(),
        )

    geometry = list(design.geometry)
    ports = list(design.ports)
    assignments: list[ExcitationAssignment] = []
    for ordinal in range(1, design.array.element_count + 1):
        coordinates = element_coordinate(design.array, ordinal)
        if coordinates is None:
            raise DesignValidationError("Independent-port synthesis could not resolve an array element.")
        element_tag = f"element_{ordinal}"
        feed_indexes = [
            index
            for index, item in enumerate(geometry)
            if element_tag in item.tags and item.object_id.endswith("_feed")
        ]
        edge_candidates = [
            item
            for item in geometry
            if element_tag in item.tags
            and item.object_id.endswith(("_patch_left", "_patch_right"))
        ]
        port_indexes = [
            index for index, port in enumerate(ports) if port.element_index == ordinal
        ]
        if len(feed_indexes) != 1 or len(edge_candidates) != 2 or len(port_indexes) != 1:
            raise DesignValidationError(
                "Independent-port synthesis requires exactly one inset feed, two local inset-edge "
                f"conductors, and one recipe port for element {list(coordinates)}."
            )
        local_edges = [item.dimension_map().get("y_min") for item in edge_candidates]
        if local_edges[0] is None or local_edges[0] != local_edges[1]:
            raise DesignValidationError(
                f"Independent-port synthesis could not resolve one local feed edge for element {list(coordinates)}."
            )
        local_edge = local_edges[0]
        feed_index = feed_indexes[0]
        geometry[feed_index] = _replace_dimension(geometry[feed_index], "y_min", local_edge)

        port_index = port_indexes[0]
        port = ports[port_index]
        ports[port_index] = replace(
            port,
            kind="discrete",
            positive_point=(port.positive_point[0], local_edge, port.positive_point[2]),
            negative_point=(port.negative_point[0], local_edge, port.negative_point[2]),
        )
        assignments.append(ExcitationAssignment(
            coordinates[0],
            coordinates[1],
            f"excitation_{coordinates[0]}_{coordinates[1]}",
            port.port_id,
            "realized",
        ))

    realized = replace(
        design,
        geometry=tuple(geometry),
        ports=tuple(ports),
        excitation=ExcitationDefinition(
            strategy="independent_ports",
            realization_status="realized",
            feed_network="independent",
            element_assignments=tuple(assignments),
            unresolved_requirements=(),
        ),
        validation=(),
    )
    keys = [_segment_key(realized, port) for port in realized.ports]
    if len(keys) != len(set(keys)):
        raise DesignValidationError(
            "Independent-port synthesis produced coincident physical port segments."
        )
    return realized
