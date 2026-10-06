"""Read-only derived geometry, before triangulation. Never persisted as design state.

Planar shapes are immutable Shapely 2 geometries in world XY millimetres.
Their predicates are explicitly *projected* predicates: callers must separately
consider Z intervals and approximation bounds. No engineering checks run here.
"""
from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import MultiPolygon, Polygon

from studio.antenna_design import ExcitationDefinition, MaterialSpec, TransformSpec
from studio.antenna_engineering import EngineeringDesignRef

Point3 = tuple[float, float, float]
Bounds = tuple[float, float, float, float, float, float]
PlanarShape = Polygon | MultiPolygon


@dataclass(frozen=True, slots=True)
class GeometryTolerances:
    layer_mm: float = 1e-7
    rotation_deg: float = 1e-10
    mesh_vertex_decimal_places: int = 12
    circle_quad_segments: int = 16
    nonplanar_mesh_circle_segments: int = 48


GEOMETRY_TOLERANCES = GeometryTolerances()


@dataclass(frozen=True, slots=True)
class GeometryCoverage:
    entity_kind: str
    entity_id: str
    status: str  # completed / partially_evaluated / unsupported / failed
    reason: str = ""


@dataclass(frozen=True, slots=True)
class GeometryApproximation:
    method: str = "exact_polygon"
    circle_segments: int | None = None
    maximum_chord_error_mm: float = 0.0
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluatedBoolean:
    operation_id: str
    operation: str
    target_id: str
    tool_ids: tuple[str, ...]
    status: str
    reason: str = ""


@dataclass(frozen=True, slots=True)
class EvaluatedGeometryObject:
    object_id: str
    name: str
    primitive: str
    semantic_role: str
    material: MaterialSpec | None
    component: str
    tags: tuple[str, ...]
    resolved_tags: tuple[str, ...]
    element: tuple[int, int] | None
    source_primitive_ids: tuple[str, ...]
    boolean_history: tuple[str, ...]
    consumed_by: str | None
    is_physical: bool | None  # None = unresolved, never interpreted as absent
    status: str
    transform: TransformSpec  # canonical X -> Y -> Z rotation, then translation
    resolved_dimensions: tuple[tuple[str, float], ...]
    axis: str
    local_planar_shape: PlanarShape | None  # primitive before transform/Booleans
    source_planar_shape: PlanarShape | None  # primitive in world XY, provenance only
    planar_shape: PlanarShape | None  # final physical shape ONLY
    z_min: float | None  # world-Z bounds; extrusion interval only when planar_shape exists
    z_max: float | None
    bounds: Bounds | None
    approximation: GeometryApproximation

    def require_planar_shape(self) -> PlanarShape:
        """Return a resolved physical XY shape, or explicitly reject unknown/consumed data.

        Supports Shapely intersects/intersection.area/distance/contains/covers and
        boundary.distance, including point queries into resolved holes. These are
        geometric operations, not physical-contact or tolerance classifications.
        """
        if self.is_physical is not True or self.planar_shape is None:
            raise ValueError(f"{self.object_id} has no resolved physical world-XY shape ({self.status}).")
        return self.planar_shape


@dataclass(frozen=True, slots=True)
class EvaluatedElement:
    index: tuple[int, int]  # 1-based (row, column)
    ordinal: int
    origin_mm: Point3
    row_vector: Point3
    column_vector: Point3
    object_ids: tuple[str, ...]
    # Layout origin is not a port position or the centroid of a decorated radiator.
    frame_source: str = "canonical_array_layout"

    def local_to_world(self, point: Point3) -> Point3:
        # Canonical ArraySpec translates instances; layout vectors specify placement,
        # not a rotation of the individual element's axes.
        return tuple(a + b for a, b in zip(self.origin_mm, point))

    def world_to_local(self, point: Point3) -> Point3:
        return tuple(a - b for a, b in zip(point, self.origin_mm))


@dataclass(frozen=True, slots=True)
class EvaluatedPort:
    port_id: str
    name: str
    kind: str
    element_index: int
    element: tuple[int, int] | None
    positive_point: Point3 | None
    negative_point: Point3 | None
    impedance_ohms: float
    status: str
    # PortSpec currently contains no target references. Do not infer attachments.
    canonical_references: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluatedGeometryQueryResult:
    schema_version: int
    design_ref: EngineeringDesignRef
    semantic_design_hash: str
    geometry_hash: str
    objects: tuple[EvaluatedGeometryObject, ...]
    elements: tuple[EvaluatedElement, ...]
    ports: tuple[EvaluatedPort, ...]
    booleans: tuple[EvaluatedBoolean, ...]
    coverage: tuple[GeometryCoverage, ...]
    excitation: ExcitationDefinition
    tolerances: GeometryTolerances = GEOMETRY_TOLERANCES
    warnings: tuple[str, ...] = ()

    @property
    def physical_objects(self) -> tuple[EvaluatedGeometryObject, ...]:
        """Known physical objects. Always consult coverage/unresolved_objects as well."""
        return tuple(item for item in self.objects if item.is_physical is True)

    @property
    def unresolved_objects(self) -> tuple[EvaluatedGeometryObject, ...]:
        return tuple(item for item in self.objects if item.is_physical is None)

    def object(self, object_id: str) -> EvaluatedGeometryObject:
        return next(item for item in self.objects if item.object_id == object_id)
