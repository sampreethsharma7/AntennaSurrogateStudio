"""Evaluate canonical geometry once; expose read-only queries and mesh that result."""

from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass, fields, replace

from shapely import affinity, constrained_delaunay_triangles
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely.errors import GEOSException
from shapely.geometry.base import BaseGeometry

from studio.antenna_design import (
    AntennaDesign,
    GeometryObject,
    DesignValidationError,
    evaluate_scalar,
    resolved_dimensions,
    resolve_parameter_values,
    object_bounds,
    TransformSpec,
)
from studio.antenna_engineering import EngineeringDesignRef
from studio.antenna_geometry_query import (
    GEOMETRY_TOLERANCES, EvaluatedBoolean, EvaluatedElement, EvaluatedGeometryObject,
    EvaluatedGeometryQueryResult, EvaluatedPort, GeometryApproximation, GeometryCoverage,
)


@dataclass(frozen=True, slots=True)
class GeometrySolid:
    name: str
    material: str
    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, ...], ...]
    color: str
    tags: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    evaluated_volume_mm3: float | None = None

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        return (
            min(point[0] for point in self.vertices),
            max(point[0] for point in self.vertices),
            min(point[1] for point in self.vertices),
            max(point[1] for point in self.vertices),
            min(point[2] for point in self.vertices),
            max(point[2] for point in self.vertices),
        )


@dataclass(frozen=True, slots=True)
class GeometryScene:
    state: AntennaDesign
    solids: tuple[GeometrySolid, ...]
    element_centers: tuple[tuple[float, float], ...]
    port_segments: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]
    scene_bounds: tuple[float, float, float, float, float, float]
    evaluated_operations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        return self.scene_bounds


@dataclass(slots=True)
class _PlanarSolid:
    object_id: str
    name: str
    material_id: str
    shape: Polygon | MultiPolygon
    z_min: float
    z_max: float
    tags: tuple[str, ...]
    source_ids: tuple[str, ...]


_BOX_FACES = (
    (3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4),
    (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
)


def _rotate_point(
    point: tuple[float, float, float],
    angles_deg: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Apply canonical X, then Y, then Z rotations around the global origin."""

    x, y, z = point
    ax, ay, az = (math.radians(value) for value in angles_deg)
    y, z = y * math.cos(ax) - z * math.sin(ax), y * math.sin(ax) + z * math.cos(ax)
    x, z = x * math.cos(ay) + z * math.sin(ay), -x * math.sin(ay) + z * math.cos(ay)
    x, y = x * math.cos(az) - y * math.sin(az), x * math.sin(az) + y * math.cos(az)
    return x, y, z


def transformed_point(item: GeometryObject, point: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = _rotate_point(point, item.transform.rotate_deg)
    tx, ty, tz = item.transform.translate_mm
    return x + tx, y + ty, z + tz


def _box_mesh(state: AntennaDesign, item: GeometryObject, values=None):
    values = resolved_dimensions(state, item) if values is None else values
    x0, x1 = values["x_min"], values["x_max"]
    y0, y1 = values["y_min"], values["y_max"]
    z0, z1 = values["z_min"], values["z_max"]
    vertices = (
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    )
    return tuple(transformed_point(item, point) for point in vertices), _BOX_FACES


def _cylinder_mesh(state: AntennaDesign, item: GeometryObject, segments: int = GEOMETRY_TOLERANCES.nonplanar_mesh_circle_segments, values=None):
    values = resolved_dimensions(state, item) if values is None else values
    radius = values["radius"]
    start, end = values["start"], values["end"]
    center_1, center_2 = values["center_1"], values["center_2"]
    vertices: list[tuple[float, float, float]] = []
    for position in (start, end):
        for index in range(segments):
            angle = 2 * math.pi * index / segments
            a, b = center_1 + radius * math.cos(angle), center_2 + radius * math.sin(angle)
            if item.axis == "x":
                point = (position, a, b)
            elif item.axis == "y":
                point = (a, position, b)
            else:
                point = (a, b, position)
            vertices.append(transformed_point(item, point))
    if item.axis == "y":
        faces: list[tuple[int, ...]] = [
            tuple(range(segments)),
            tuple(range(segments * 2 - 1, segments - 1, -1)),
        ]
    else:
        faces = [
            tuple(range(segments - 1, -1, -1)),
            tuple(range(segments, segments * 2)),
        ]
    for index in range(segments):
        nxt = (index + 1) % segments
        faces.append((index, nxt, segments + nxt, segments + index))
    return tuple(vertices), tuple(faces)


def _planar_solid(state: AntennaDesign, item: GeometryObject, values=None) -> _PlanarSolid | None:
    rx, ry, rz = item.transform.rotate_deg
    if abs(rx) > GEOMETRY_TOLERANCES.rotation_deg or abs(ry) > GEOMETRY_TOLERANCES.rotation_deg:
        return None
    values = resolved_dimensions(state, item) if values is None else values
    if item.primitive in {"box", "sheet_rectangle"}:
        shape = box(values["x_min"], values["y_min"], values["x_max"], values["y_max"])
        z_min, z_max = values["z_min"], values["z_max"]
    elif item.primitive in {"cylinder", "sheet_circle"} and item.axis == "z":
        shape = Point(values["center_1"], values["center_2"]).buffer(
            values["radius"], quad_segs=GEOMETRY_TOLERANCES.circle_quad_segments
        )
        z_min, z_max = values["start"], values["end"]
    else:
        return None
    if abs(rz) > GEOMETRY_TOLERANCES.rotation_deg:
        shape = affinity.rotate(shape, rz, origin=(0.0, 0.0), use_radians=False)
    tx, ty, tz = item.transform.translate_mm
    shape = affinity.translate(shape, xoff=tx, yoff=ty)
    return _PlanarSolid(
        item.object_id,
        item.name,
        item.material_id,
        shape,
        z_min + tz,
        z_max + tz,
        item.tags,
        (item.object_id,),
    )


def _same_layer(target: _PlanarSolid, tool: _PlanarSolid, operation: str) -> bool:
    tolerance = GEOMETRY_TOLERANCES.layer_mm
    if operation == "union":
        return abs(target.z_min - tool.z_min) <= tolerance and abs(target.z_max - tool.z_max) <= tolerance
    return tool.z_min <= target.z_min + tolerance and tool.z_max >= target.z_max - tolerance


def _polygon_parts(shape: Polygon | MultiPolygon):
    if shape.is_empty:
        return ()
    if isinstance(shape, Polygon):
        return (shape,)
    if isinstance(shape, MultiPolygon):
        return tuple(shape.geoms)
    return tuple(item for item in getattr(shape, "geoms", ()) if isinstance(item, Polygon))


def _extrude_shape(shape: Polygon | MultiPolygon, z_min: float, z_max: float):
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    vertex_indexes: dict[tuple[float, float, float], int] = {}

    def vertex(point: tuple[float, float], z: float) -> int:
        value = (float(point[0]), float(point[1]), float(z))
        key = tuple(round(coordinate, GEOMETRY_TOLERANCES.mesh_vertex_decimal_places) for coordinate in value)
        existing = vertex_indexes.get(key)
        if existing is not None:
            return existing
        vertices.append(value)
        index = len(vertices) - 1
        vertex_indexes[key] = index
        return index

    def counter_clockwise(
        coords: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        area_twice = sum(
            first[0] * second[1] - second[0] * first[1]
            for first, second in zip(coords, (*coords[1:], coords[0]))
        )
        return coords if area_twice >= 0 else list(reversed(coords))

    for raw_polygon in _polygon_parts(shape):
        # Shapely Boolean results may return either ring orientation. Normalize
        # every result so top/bottom and wall cells form one consistently wound
        # closed shell for downstream depth-buffered renderers.
        polygon = orient(raw_polygon, sign=1.0)
        for triangle in constrained_delaunay_triangles(polygon).geoms:
            coords = counter_clockwise(list(triangle.exterior.coords)[:3])
            bottom = tuple(vertex(point, z_min) for point in coords)
            top = tuple(vertex(point, z_max) for point in coords)
            faces.append(tuple(reversed(bottom)))
            faces.append(top)
        for ring in (polygon.exterior, *polygon.interiors):
            coords = list(ring.coords)
            for first, second in zip(coords, coords[1:]):
                a = vertex(first, z_min)
                b = vertex(second, z_min)
                c = vertex(second, z_max)
                d = vertex(first, z_max)
                faces.append((a, b, c, d))
    return tuple(vertices), tuple(faces)


def _solid_color(material_kind: str, tags: tuple[str, ...]) -> str:
    if "ground" in tags:
        return "#b87333"
    return "#e5943a" if material_kind == "conductor" else "#4e9b82"


def _geometry_hash(objects, elements, ports, booleans, coverage):
    """Content fingerprint, not a cross-GEOS-version geometric equivalence proof."""
    def encode(value):
        if isinstance(value, BaseGeometry):
            return value.normalize().wkb_hex
        if hasattr(value, "__dataclass_fields__"):
            return {field.name: encode(getattr(value, field.name)) for field in fields(value)}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        return value
    payload = {
        "schema_version": 1,
        "objects": sorted(objects, key=lambda item: item.object_id),
        "elements": elements, "ports": sorted(ports, key=lambda item: item.port_id),
        "booleans": booleans, "coverage": coverage, "tolerances": GEOMETRY_TOLERANCES,
    }
    return hashlib.sha256(json.dumps(
        {key: encode(value) for key, value in payload.items()}, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def evaluate_geometry(state: AntennaDesign) -> EvaluatedGeometryQueryResult:
    """Resolve canonical primitives/transforms/CSG without triangulation or checks.

    Invalid inputs produce explicit coverage, not invented empty geometry. Geometry
    is derived afresh; this API does not mutate, validate, or persist the design.
    """
    # Reuse existing canonical semantic conventions and the established state hash;
    # neither planner nor runner is invoked by these pure helper calls.
    from studio.antenna_agent import AntennaDesignAgent
    from studio.antenna_agent_runner import semantic_design_hash

    materials = {item.material_id: item for item in state.materials}
    planar: dict[str, _PlanarSolid] = {}
    sources: dict[str, _PlanarSolid] = {}
    local_shapes = {}
    dimensions = {}
    bounds = {}
    consumed: dict[str, str] = {}
    suppressed: set[str] = set()
    warnings: list[str] = []
    coverage: list[GeometryCoverage] = []
    object_status = {}
    histories = {item.object_id: () for item in state.geometry}
    approximations = {}
    boolean_records = []

    for item in state.geometry:
        status, reason = "completed", ""
        approximation = GeometryApproximation()
        try:
            values = resolved_dimensions(state, item)
            dimensions[item.object_id] = tuple(values.items())
            if item.primitive not in {"box", "sheet_rectangle", "cylinder", "sheet_circle"}:
                status, reason = "unsupported", f"unsupported primitive {item.primitive}"
            elif item.material_id not in materials:
                status, reason = "failed", "unresolved material reference"
            else:
                if item.primitive in {"box", "sheet_rectangle"}:
                    if any(values[f"{axis}_min"] > values[f"{axis}_max"] for axis in "xyz"):
                        raise DesignValidationError("Reversed primitive interval.")
                else:
                    if item.axis not in {"x", "y", "z"} or values["radius"] <= 0 or values["start"] > values["end"]:
                        raise DesignValidationError("Invalid cylinder interval, radius or axis.")
                    segments = GEOMETRY_TOLERANCES.circle_quad_segments * 4
                    approximation = GeometryApproximation(
                        "inscribed_circle_polygon", segments,
                        values["radius"] * (1 - math.cos(math.pi / segments)),
                        ("Curved boundaries are polygonal approximations; distances within chord error are uncertain.",),
                    )
                if not all(math.isfinite(value) for value in (*item.transform.translate_mm, *item.transform.rotate_deg)):
                    raise DesignValidationError("Non-finite transform.")
                bounds[item.object_id] = object_bounds(state, item)
                local = _planar_solid(state, replace(item, transform=TransformSpec()), values)
                local_shapes[item.object_id] = local.shape if local else None
                candidate = _planar_solid(state, item, values)
                if candidate is not None:
                    if candidate.shape.is_empty or not candidate.shape.is_valid:
                        raise DesignValidationError("Invalid or empty primitive planar shape.")
                    planar[item.object_id] = candidate
                    sources[item.object_id] = replace(candidate)
                else:
                    status = "partially_evaluated"
                    reason = "Resolved primitive/transform and analytic bounds only; world-XY extrusion queries and CSG unsupported."
                    approximation = GeometryApproximation(
                        "resolved_nonplanar_primitive", None, 0.0, (reason,)
                    )
        except (DesignValidationError, KeyError, ValueError, GEOSException, ArithmeticError) as exc:
            status, reason = "failed", str(exc)
        if status in {"failed", "unsupported"}:
            suppressed.add(item.object_id)
            warnings.append(f"{item.object_id} was not rendered: {reason}.")
        object_status[item.object_id] = (status, reason)
        approximations[item.object_id] = approximation

    for operation in state.booleans:
        target = planar.get(operation.target_id)
        tools = [planar.get(tool_id) for tool_id in operation.tool_ids]
        reason = ""
        if operation.operation not in {"union", "subtract"}:
            reason = "unsupported Boolean operation"
        elif target is None:
            reason = "target is not a common-axis planar extrusion"
        elif any(tool is None for tool in tools):
            reason = "one or more tools are not common-axis planar extrusions"
        elif any(not _same_layer(target, tool, operation.operation) for tool in tools if tool is not None):
            reason = "target and tools do not share a compatible extrusion layer"
        elif operation.target_id in consumed or any(tool_id in consumed for tool_id in operation.tool_ids):
            reason = "an input solid was already consumed by an earlier Boolean"
        elif operation.target_id in suppressed or any(tool_id in suppressed for tool_id in operation.tool_ids):
            reason = "an input has unresolved earlier geometry"
        elif not tools:
            reason = "no Boolean tools"
        status = "unsupported" if reason else "completed"
        valid_tools = [tool for tool in tools if tool is not None]
        result_shape = None
        if not reason:
            try:
                tool_shape = unary_union([tool.shape for tool in valid_tools])
                result_shape = target.shape.difference(tool_shape) if operation.operation == "subtract" else target.shape.union(tool_shape)
                if not result_shape.is_valid:
                    raise ValueError("invalid resolved Boolean shape")
            except (GEOSException, ValueError) as exc:
                reason, status = str(exc), "failed"
        if reason:
            involved = {operation.target_id, *operation.tool_ids}
            suppressed.update(involved)
            warnings.append(f"{operation.operation_id} was not rendered: {reason}.")
            boolean_records.append(EvaluatedBoolean(operation.operation_id, operation.operation, operation.target_id, operation.tool_ids, status, reason))
            coverage.append(GeometryCoverage("boolean", operation.operation_id, status, reason))
            continue
        assert target is not None
        target.shape = result_shape
        if operation.operation == "union":
            target.tags = tuple(dict.fromkeys((*target.tags, *(tag for tool in valid_tools for tag in tool.tags))))
            target.source_ids = tuple(dict.fromkeys((*target.source_ids, *(source for tool in valid_tools for source in tool.source_ids))))
        histories[target.object_id] = tuple(dict.fromkeys((
            *histories[target.object_id], *(entry for tool in valid_tools for entry in histories[tool.object_id]), operation.operation_id,
        )))
        curve_inputs = [approximations[target.object_id], *(approximations[tool.object_id] for tool in valid_tools)]
        approximations[target.object_id] = max(curve_inputs, key=lambda entry: entry.maximum_chord_error_mm)
        # Preserve the historical layer tolerance without claiming small Z offsets
        # were resolved in 3D by a planar Boolean.
        layer_ambiguous = any(
            (tool.z_min != target.z_min or tool.z_max != target.z_max) if operation.operation == "union"
            else (tool.z_min > target.z_min or tool.z_max < target.z_max)
            for tool in valid_tools
        )
        if layer_ambiguous:
            status, reason = "partially_evaluated", "Layer compatibility accepted within geometric tolerance; exact 3D layer difference unresolved."
        for tool_id in operation.tool_ids:
            consumed[tool_id] = operation.operation_id
        boolean_records.append(EvaluatedBoolean(operation.operation_id, operation.operation, operation.target_id, operation.tool_ids, status, reason))
        coverage.append(GeometryCoverage("boolean", operation.operation_id, status, reason))

    objects = []
    for item in state.geometry:
        candidate = planar.get(item.object_id)
        source = sources.get(item.object_id)
        status, reason = object_status[item.object_id]
        physical = item.object_id not in consumed
        if item.object_id in suppressed:
            physical, status, reason = None, "unsupported" if status != "failed" else status, reason or "Unresolved Boolean dependency."
        elif candidate is not None and candidate.shape.is_empty:
            physical = False
            if item.object_id not in consumed:
                warnings.append(f"{item.object_id} was not rendered because its evaluated geometry is empty.")
        final_shape = candidate.shape if candidate and physical is True else None
        object_bounds_value = bounds.get(item.object_id)
        if final_shape is not None:
            x0, y0, x1, y1 = final_shape.bounds
            object_bounds_value = (x0, x1, y0, y1, candidate.z_min, candidate.z_max)
        if physical is not None and any(record.status == "partially_evaluated" and record.operation_id in histories[item.object_id] for record in boolean_records):
            status = "partially_evaluated"
            reason = "Contains a Boolean evaluated within layer tolerance."
        coverage.append(GeometryCoverage("object", item.object_id, status, reason))
        objects.append(EvaluatedGeometryObject(
            item.object_id, item.name, item.primitive, AntennaDesignAgent._semantic_role(item),
            materials.get(item.material_id), item.component, item.tags, candidate.tags if candidate else item.tags,
            AntennaDesignAgent._element_coordinates(state, item),
            candidate.source_ids if candidate else (item.object_id,), histories[item.object_id],
            consumed.get(item.object_id), physical, status, item.transform, dimensions.get(item.object_id, ()), item.axis,
            local_shapes.get(item.object_id), source.shape if source else None, final_shape,
            object_bounds_value[4] if object_bounds_value else None, object_bounds_value[5] if object_bounds_value else None,
            object_bounds_value, approximations[item.object_id],
        ))

    elements = []
    array = state.array
    if array.rows >= 1 and array.columns >= 1 and math.isfinite(array.spacing_mm) and all(
        math.isfinite(value) for value in (*array.row_vector, *array.column_vector)
    ):
        for row in range(array.rows):
            for column in range(array.columns):
                origin = tuple(array.spacing_mm * (
                    (row - (array.rows - 1) / 2) * array.row_vector[axis]
                    + (column - (array.columns - 1) / 2) * array.column_vector[axis]
                ) for axis in range(3))
                index = (row + 1, column + 1)
                elements.append(EvaluatedElement(index, row * array.columns + column + 1, origin,
                    array.row_vector, array.column_vector, tuple(item.object_id for item in objects if item.element == index)))
        coverage.append(GeometryCoverage("layout", "array", "completed"))
    else:
        coverage.append(GeometryCoverage("layout", "array", "failed", "Invalid canonical array layout."))

    ports = []
    for port in state.ports:
        status, reason = "completed", ""
        positive = negative = None
        try:
            values = resolve_parameter_values(state)
            positive = tuple(evaluate_scalar(value, values) for value in port.positive_point)
            negative = tuple(evaluate_scalar(value, values) for value in port.negative_point)
            if len(positive) != 3 or len(negative) != 3:
                raise ValueError("Port endpoints must have three coordinates.")
        except (DesignValidationError, ValueError, ArithmeticError) as exc:
            status, reason = "failed", str(exc)
            positive = negative = None
        element = next((item.index for item in elements if item.ordinal == port.element_index), None)
        ports.append(EvaluatedPort(port.port_id, port.name, port.kind, port.element_index, element,
            positive, negative, port.impedance_ohms, status))
        coverage.append(GeometryCoverage("port", port.port_id, status, reason))
    return EvaluatedGeometryQueryResult(
        1, EngineeringDesignRef(state.design_id, state.revision), semantic_design_hash(state),
        _geometry_hash(objects, elements, ports, boolean_records, coverage), tuple(objects), tuple(elements),
        tuple(ports), tuple(boolean_records), tuple(coverage), state.excitation, warnings=tuple(warnings),
    )


def build_geometry_scene(state: AntennaDesign) -> GeometryScene:
    """Mesh the shared evaluated result; the renderer performs no CAD evaluation."""
    query = evaluate_geometry(state)
    by_id = {item.object_id: item for item in state.geometry}
    warnings = list(query.warnings)

    solids: list[GeometrySolid] = []
    for obj in query.physical_objects:
        item = by_id[obj.object_id]
        material = obj.material
        if obj.planar_shape is not None:
            vertices, faces = _extrude_shape(obj.planar_shape, obj.z_min, obj.z_max)
            tags, source_ids = obj.resolved_tags, obj.source_primitive_ids
            evaluated_volume = float(obj.planar_shape.area) * (obj.z_max - obj.z_min)
        elif item.primitive in {"box", "sheet_rectangle"}:
            vertices, faces = _box_mesh(state, item, dict(obj.resolved_dimensions))
            tags, source_ids = item.tags, (item.object_id,)
            evaluated_volume = None
        elif item.primitive in {"cylinder", "sheet_circle"}:
            vertices, faces = _cylinder_mesh(state, item, values=dict(obj.resolved_dimensions))
            tags, source_ids = item.tags, (item.object_id,)
            evaluated_volume = None
        else:
            warnings.append(f"{item.object_id} was not rendered: unsupported primitive {item.primitive}.")
            continue
        if not vertices or not faces:
            warnings.append(f"{item.object_id} was not rendered because its evaluated geometry is empty.")
            continue
        solids.append(
            GeometrySolid(
                item.name,
                material.name,
                vertices,
                faces,
                _solid_color(material.kind, tags),
                tags,
                source_ids,
                evaluated_volume,
            )
        )

    if solids:
        bounds = [solid.bounds for solid in solids]
        scene_bounds = (
            min(item[0] for item in bounds), max(item[1] for item in bounds),
            min(item[2] for item in bounds), max(item[3] for item in bounds),
            min(item[4] for item in bounds), max(item[5] for item in bounds),
        )
    else:
        scene_bounds = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    centers: list[tuple[float, float]] = []
    port_segments: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    for port in query.ports:
        positive, negative = port.positive_point, port.negative_point
        if positive is None or negative is None:
            warnings.append(f"{port.port_id} was not rendered: unresolved port endpoints.")
            continue
        centers.append(((positive[0] + negative[0]) / 2, (positive[1] + negative[1]) / 2))
        port_segments.append((negative, positive))
    return GeometryScene(
        state,
        tuple(solids),
        tuple(centers),
        tuple(port_segments),
        scene_bounds,
        tuple(item.operation_id for item in query.booleans if item.status in {"completed", "partially_evaluated"}),
        tuple(warnings),
    )


def mesh_volume(solid: GeometrySolid) -> float:
    """Signed-triangle volume magnitude used by focused preview tests."""

    volume = 0.0
    for face in solid.faces:
        if len(face) < 3:
            continue
        origin = solid.vertices[face[0]]
        for index in range(1, len(face) - 1):
            b = solid.vertices[face[index]]
            c = solid.vertices[face[index + 1]]
            volume += (
                origin[0] * (b[1] * c[2] - b[2] * c[1])
                + origin[1] * (b[2] * c[0] - b[0] * c[2])
                + origin[2] * (b[0] * c[1] - b[1] * c[0])
            ) / 6.0
    return abs(volume)
