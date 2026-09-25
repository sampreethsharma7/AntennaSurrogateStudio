"""Read-only geometric evidence. No runner/planner integration or RF conclusions."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from itertools import combinations

from shapely.geometry import Point, Polygon
from shapely import get_coordinates

from studio.antenna_design import AntennaDesign
from studio.antenna_engineering import (
    EngineeringCheckCoverage, EngineeringDesignRef, EngineeringMeasurement as Measurement,
    EngineeringObservation, EngineeringReport, EngineeringSource,
)
from studio.antenna_geometry import evaluate_geometry
from studio.antenna_geometry_query import EvaluatedGeometryQueryResult

CONTACT_CHECK = "conductor_contact_clearance"
PORT_CHECK = "port_attachment_distinctness"
CONTAINMENT_CHECK = "substrate_support_containment"
EXCITATION_CHECK = "excitation_consistency"
COMPOSED_UNIFORMITY_CHECK = "composed_feature_uniformity"
CHECK_VERSIONS = {
    CONTACT_CHECK: "2",
    PORT_CHECK: "1",
    CONTAINMENT_CHECK: "1",
    EXCITATION_CHECK: "1",
    COMPOSED_UNIFORMITY_CHECK: "1",
}
SUITE_VERSION = "geometry_checks_v4"
_LIMITS = (
    "Geometry only: no impedance, modal excitation, calibration, or RF performance conclusion.",
    "Curved boundaries use the evaluator's polygonal approximation; near-boundary relationships may be unknown.",
)


@dataclass(frozen=True, slots=True)
class EngineeringCheckConfig:
    clearance_neighbors: int = 1

    def __post_init__(self):
        if type(self.clearance_neighbors) is not int or not 0 <= self.clearance_neighbors <= 8:
            raise ValueError("clearance_neighbors must be an integer from 0 to 8.")


def _selected_checks(checks):
    selected = tuple(CHECK_VERSIONS) if checks is None else tuple(checks)
    if len(set(selected)) != len(selected) or any(name not in CHECK_VERSIONS for name in selected):
        raise ValueError("Checks must be distinct installed engineering check IDs.")
    return tuple(sorted(selected))


def engineering_check_cache_key(query, *, checks=None, config=EngineeringCheckConfig()):
    """Future cache identity only. No cache or persistence is installed here."""
    payload = (query.geometry_hash, SUITE_VERSION,
               [(name, CHECK_VERSIONS[name]) for name in _selected_checks(checks)], asdict(config))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _elements(objects=(), ports=()):
    return tuple(sorted({item.element for item in (*objects, *ports) if item.element is not None}))


def _observation(query, check, category, message, *, objects=(), ports=(), severity="info",
                  measurements=None, relationship="", assumptions=(), elements=(),
                  source_method="resolved_planar_extrusions_and_evaluated_endpoints",
                  source_applicability="Supported world-XY conductive extrusions and geometrically evaluated port segments.",
                  source_limitations=_LIMITS):
    roles = tuple(f"{obj.object_id}: role={obj.semantic_role}, element={obj.element}." for obj in objects)
    return EngineeringObservation(
        check_id=check, check_version=CHECK_VERSIONS[check], severity=severity,
        category=category, message=message,
        source=EngineeringSource(
            method=source_method,
            applicability=source_applicability,
            evaluated_geometry_hash=query.geometry_hash, limitations=source_limitations,
        ),
        affected_objects=tuple(sorted({obj.object_id for obj in objects})),
        affected_ports=tuple(sorted({port.port_id for port in ports})),
        affected_elements=tuple(sorted(set(_elements(objects, ports)) | set(elements))), measured_values=measurements or {},
        relationship_key=relationship, assumptions=(*roles, *assumptions),
    )


def _coverage(check, status, reason):
    return EngineeringCheckCoverage(check, CHECK_VERSIONS[check], status,
        {CONTACT_CHECK: "Resolved conductor geometry.",
         PORT_CHECK: "Discrete-port geometric attachment and evaluated segment distinctness.",
         CONTAINMENT_CHECK: "Resolved planar conductor footprints and unambiguously associated support regions.",
         EXCITATION_CHECK: "Canonical per-element port indexing and evaluated physical segment groups.",
         COMPOSED_UNIFORMITY_CHECK: "Canonical composed-feature target scopes across logical array elements."}[check],
        reason, _LIMITS)


def _conductors(query):
    relevant = [obj for obj in query.objects if obj.is_physical is not False
                and (obj.material is None or obj.material.kind == "conductor")]
    known = [obj for obj in relevant if obj.status == "completed" and obj.is_physical is True
             and obj.planar_shape is not None and obj.z_min is not None and obj.z_max is not None]
    known_ids = {obj.object_id for obj in known}
    unknown = [obj for obj in relevant if obj.object_id not in known_ids]
    return sorted(known, key=lambda obj: obj.object_id), sorted(unknown, key=lambda obj: obj.object_id)


@dataclass(frozen=True, slots=True)
class _Relationship:
    xy_distance: float
    z_gap: float
    z_overlap: float
    area: float
    boundary_length: float
    contact: bool
    uncertain: bool

    def measurements(self, tolerance):
        return {
            "planar_clearance_mm": Measurement(self.xy_distance, "mm", tolerance),
            "z_clearance_mm": Measurement(self.z_gap, "mm", tolerance),
            "z_overlap_mm": Measurement(self.z_overlap, "mm", tolerance),
            "minimum_clearance_mm": Measurement(math.hypot(self.xy_distance, self.z_gap), "mm", tolerance),
            "xy_intersection_area_mm2": Measurement(self.area, "mm²"),
            "contact_within_tolerance": Measurement(int(self.contact), ""),
            "xy_shared_boundary_length_mm": Measurement(self.boundary_length, "mm"),
        }


def _relationship(a, b, tolerance, shape_a=None, shape_b=None):
    sa = a.planar_shape if shape_a is None else shape_a
    sb = b.planar_shape if shape_b is None else shape_b
    xy = sa.distance(sb)
    signed_z = min(a.z_max, b.z_max) - max(a.z_min, b.z_min)
    z_gap, z_overlap = max(0.0, -signed_z), max(0.0, signed_z)
    intersection = sa.intersection(sb)
    error = a.approximation.maximum_chord_error_mm + b.approximation.maximum_chord_error_mm
    contact = math.hypot(xy, z_gap) <= tolerance
    # Do not turn tessellation's near misses/tangencies into certified contact or
    # clearance. Deep interior overlap remains credible polygonal evidence.
    uncertain = error > tolerance and z_gap <= tolerance and (
        xy <= error + tolerance and (not contact or intersection.buffer(-error).is_empty)
    )
    return _Relationship(xy, z_gap, z_overlap, intersection.area,
                         sa.boundary.intersection(sb.boundary).length, contact, uncertain)


def _contributions(query, obj):
    """Attribute FINAL material to retained provenance footprints, never raw solids.

    Footprints are intersected with the resolved survivor. This is spatial
    attribution, not a reconstruction of the chronological ownership of material.
    Consumed subtraction tools are absent from retained source_primitive_ids.
    """
    result = []
    for source_id in sorted(obj.source_primitive_ids):
        source = query.object(source_id)
        footprint = source.source_planar_shape
        if footprint is not None:
            retained = footprint.intersection(obj.planar_shape)
            if not retained.is_empty and retained.area > 0:
                result.append((source, retained))
    return result


def check_conductor_contact_clearance(query, config=EngineeringCheckConfig()):
    conductors, unknown = _conductors(query)
    tolerance = query.tolerances.layer_mm
    findings, uncertain_pairs, clearances = [], [], []
    if unknown:
        findings.append(_observation(query, CONTACT_CHECK, "geometry_unknown",
            "Some conductor geometry cannot be inspected reliably for contact or clearance.", objects=tuple(unknown)))
    for a, b in combinations(conductors, 2):
        rel = _relationship(a, b, tolerance)
        if rel.uncertain:
            uncertain_pairs.append((a, b))
            findings.append(_observation(query, CONTACT_CHECK, "contact_unknown",
                "Conductor proximity lies within curved-boundary approximation uncertainty.",
                objects=(a, b), measurements=rel.measurements(tolerance)))
        elif rel.contact:
            members_a = {source.element for source, _ in _contributions(query, a) if source.element is not None}
            members_b = {source.element for source, _ in _contributions(query, b) if source.element is not None}
            same = len(members_a) == 1 and members_a == members_b
            # Select the strongest mixed-role attribution for this physical pair;
            # measurements of the whole physical pair remain separate.
            candidates = []
            for x, sx in _contributions(query, a):
                for y, sy in _contributions(query, b):
                    if x.semantic_role != y.semantic_role:
                        area = sx.intersection(sy).area
                        if area > tolerance ** 2:
                            candidates.append((area, x.object_id, y.object_id, x, y))
            measurements = rel.measurements(tolerance)
            measurements["resolved_conductor_xy_overlap_area_mm2"] = measurements.pop("xy_intersection_area_mm2")
            measurements["same_element"] = Measurement(int(same), "")
            objects = (a, b)
            message = (f"Resolved conductors {a.object_id} and {b.object_id} contact within geometric tolerance. "
                       "resolved_conductor_xy_overlap_area_mm2 measures the XY intersection of the complete final resolved conductors, including their retained union constituents.")
            if candidates:
                area, _, _, x, y = max(candidates, key=lambda entry: entry[:3])
                measurements["retained_constituent_xy_overlap_area_mm2"] = Measurement(area, "mm²")
                objects = tuple({obj.object_id: obj for obj in (a, b, x, y)}.values())
                message += (f" retained_constituent_xy_overlap_area_mm2 measures only the retained {x.semantic_role} "
                            f"({x.object_id}) / {y.semantic_role} ({y.object_id}) relationship, clipped to final material; "
                            "it is not the complete resolved-conductor overlap.")
            findings.append(_observation(query, CONTACT_CHECK, "conductor_contact", message,
                objects=objects, severity="info" if same else "warning", measurements=measurements,
                assumptions=("Same-element membership lowers severity but does not prove intended RF connectivity.",),
                relationship=f"physical_pair:{a.object_id}:{b.object_id}"))
        else:
            clearances.append((math.hypot(rel.xy_distance, rel.z_gap), a, b, rel))

    # At most k nearest noncontact neighbours per conductor. Distant disjoint
    # pairwise findings are suppressed; all credible contacts above are retained.
    chosen = {}
    for obj in conductors:
        nearest = sorted((entry for entry in clearances if obj.object_id in (entry[1].object_id, entry[2].object_id)),
                         key=lambda entry: (entry[0], entry[1].object_id, entry[2].object_id))
        for _, a, b, rel in nearest[:config.clearance_neighbors]:
            chosen[(a.object_id, b.object_id)] = (a, b, rel)
    for a, b, rel in chosen.values():
        findings.append(_observation(query, CONTACT_CHECK, "conductor_clearance",
            "Resolved conductors are separated; planar and Z clearances are reported independently.",
            objects=(a, b), measurements=rel.measurements(tolerance)))

    # A union is explicit canonical composition. Retained mixed-role footprints
    # provide structural evidence inside it, without pretending consumed pieces
    # remain independent physical solids.
    for obj in conductors:
        if len(obj.source_primitive_ids) < 2:
            continue
        grouped = {}
        uncertain_constituents = {}
        for (a, sa), (b, sb) in combinations(_contributions(query, obj), 2):
            if a.semantic_role == b.semantic_role:
                continue
            rel = _relationship(obj, obj, tolerance, sa, sb)
            if rel.uncertain:
                uncertain_pairs.append((a, b))
                uncertain_constituents.update({a.object_id: a, b.object_id: b})
                continue
            if not rel.contact:
                continue
            key = (a.semantic_role, b.semantic_role, a.element, b.element)
            score = (rel.area, rel.boundary_length, a.object_id, b.object_id)
            if key not in grouped or score > grouped[key][0]:
                grouped[key] = (score, a, b, rel)
        for _, a, b, rel in grouped.values():
            cross = a.element is not None and b.element is not None and a.element != b.element
            findings.append(_observation(query, CONTACT_CHECK, "union_constituent_contact",
                f"Retained {a.semantic_role} and {b.semantic_role} regions contact inside one canonical union.",
                objects=tuple({item.object_id: item for item in (obj, a, b)}.values()),
                severity="warning" if cross else "info", measurements=rel.measurements(tolerance),
                relationship=f"union:{obj.object_id}:{a.object_id}:{b.object_id}",
                assumptions=("Constituent footprints are clipped to final resolved material; they are not separate physical objects.",)))
        if uncertain_constituents:
            findings.append(_observation(query, CONTACT_CHECK, "union_contact_unknown",
                "Some retained constituent relationships lie within the survivor's curved-boundary approximation uncertainty.",
                objects=tuple(uncertain_constituents.values()), relationship=f"union:{obj.object_id}"))
    layout_unknown = any(c.entity_kind == "layout" and c.status != "completed" for c in query.coverage)
    status = "unknown" if unknown or uncertain_pairs or layout_unknown else "completed" if conductors else "not_applicable"
    return tuple(findings), _coverage(CONTACT_CHECK, status,
        f"Inspected {len(conductors)} resolved conductors; {len(unknown)} unresolved objects and {len(uncertain_pairs)} approximation-limited relationships. "
        f"Clearance output limited to {config.clearance_neighbors} nearest noncontact neighbours per conductor."
        + (" Canonical layout could not be fully evaluated." if layout_unknown else ""))


def _pieces(conductors):
    return [(obj, shape) for obj in conductors for shape in
            ((obj.planar_shape,) if isinstance(obj.planar_shape, Polygon) else tuple(obj.planar_shape.geoms))]


def _connected_components(pieces, tolerance):
    parents = list(range(len(pieces)))
    uncertain = False

    def root(index):
        while parents[index] != index:
            index = parents[index]
        return index

    for i, j in combinations(range(len(pieces)), 2):
        a, sa = pieces[i]
        b, sb = pieces[j]
        rel = _relationship(a, b, tolerance, sa, sb)
        uncertain = uncertain or rel.uncertain
        if rel.contact and not rel.uncertain:
            parents[root(j)] = root(i)
    return tuple(root(index) for index in range(len(pieces))), uncertain


def _endpoint_evidence(point, pieces, tolerance):
    p = Point(point[0], point[1])
    hits, uncertain, holes = [], [], []
    for index, (obj, shape) in enumerate(pieces):
        zgap = max(obj.z_min - point[2], point[2] - obj.z_max, 0.0)
        if zgap > tolerance:
            continue
        distance = shape.distance(p)
        error = obj.approximation.maximum_chord_error_mm
        if error > tolerance and shape.boundary.distance(p) <= error + tolerance:
            uncertain.append(index)
        elif math.hypot(distance, zgap) <= tolerance:
            hits.append(index)
        if any(Polygon(ring).covers(p) for ring in shape.interiors):
            holes.append(index)
    return tuple(hits), tuple(uncertain), tuple(holes)


def _segment_distance(a, b):
    direct = max(math.dist(a.positive_point, b.positive_point), math.dist(a.negative_point, b.negative_point))
    reverse = max(math.dist(a.positive_point, b.negative_point), math.dist(a.negative_point, b.positive_point))
    return min(direct, reverse), reverse < direct


def check_port_attachment_distinctness(query, config=EngineeringCheckConfig()):
    if not query.ports:
        return (), _coverage(PORT_CHECK, "not_applicable", "No canonical ports to inspect.")
    conductors, unknown = _conductors(query)
    tolerance = query.tolerances.layer_mm
    pieces = _pieces(conductors)
    components, component_uncertainty = _connected_components(pieces, tolerance)
    findings, incomplete = [], bool(unknown) or component_uncertainty
    if unknown:
        findings.append(_observation(query, PORT_CHECK, "geometry_unknown",
            "Unresolved conductor geometry limits attachment and connectivity conclusions.", objects=tuple(unknown)))
    if component_uncertainty:
        findings.append(_observation(query, PORT_CHECK, "conductor_connectivity_unknown",
            "Some possible conductor connections are limited by curved-boundary approximation; separate geometric components may not be conclusively distinct.",
            objects=tuple(conductors)))
    evaluated_ports = []
    for port in sorted(query.ports, key=lambda item: item.port_id):
        if port.status != "completed" or port.positive_point is None or port.negative_point is None:
            incomplete = True
            findings.append(_observation(query, PORT_CHECK, "port_unknown",
                "Port endpoints could not be fully evaluated.", ports=(port,)))
            continue
        evaluated_ports.append(port)
        if port.kind != "discrete":
            incomplete = True
            findings.append(_observation(query, PORT_CHECK, "port_unknown",
                "Attachment inspection currently supports discrete ports only; segment distinctness remains geometric.", ports=(port,)))
            continue
        if port.element is None:
            incomplete = True
            findings.append(_observation(query, PORT_CHECK, "port_element_unknown",
                "The port's canonical element ordinal does not resolve to a layout element.", ports=(port,)))
        terminal_hits = []
        for terminal, point in (("positive", port.positive_point), ("negative", port.negative_point)):
            hits, uncertain, holes = _endpoint_evidence(point, pieces, tolerance)
            terminal_hits.append(hits)
            incomplete = incomplete or bool(uncertain)
            hit_objects = tuple({pieces[index][0].object_id: pieces[index][0] for index in hits}.values())
            evidence_objects = {pieces[index][0].object_id: pieces[index][0] for index in (*hits, *uncertain, *holes)}
            hit_elements = set()
            for obj in hit_objects:
                for source, footprint in _contributions(query, obj):
                    if footprint.distance(Point(*point[:2])) <= tolerance:
                        evidence_objects[source.object_id] = source
                        if source.element is not None:
                            hit_elements.add(source.element)
            if not hits and unknown:
                evidence_objects.update({obj.object_id: obj for obj in unknown})
            objects = tuple(evidence_objects.values())
            measurements = {"attached_component_count": Measurement(len({components[index] for index in hits}), ""),
                            "inside_resolved_hole": Measurement(int(bool(holes)), "")}
            measurements.update({f"endpoint_{axis}_mm": Measurement(value, "mm", tolerance)
                                 for axis, value in zip("xyz", point)})
            if pieces:
                distance = min(math.hypot(shape.distance(Point(*point[:2])),
                    max(obj.z_min - point[2], point[2] - obj.z_max, 0.0)) for obj, shape in pieces)
                measurements["nearest_resolved_conductor_distance_mm"] = Measurement(distance, "mm", tolerance)
            if hits:
                category, severity = "port_terminal_attached", "info"
                message = f"The {terminal} terminal geometrically attaches to resolved conductive material."
            elif uncertain or unknown:
                category, severity = "port_attachment_unknown", "info"
                message = f"Attachment of the {terminal} terminal cannot be determined reliably."
            else:
                category, severity = "port_terminal_unattached", "warning"
                message = f"The {terminal} terminal does not attach to any resolved conductor within tolerance."
                if holes:
                    message += " Its location is inside a resolved Boolean hole."
            findings.append(_observation(query, PORT_CHECK, category, message, objects=objects, ports=(port,),
                severity=severity, measurements=measurements, relationship=terminal))
            if port.element is not None and hit_elements - {port.element}:
                findings.append(_observation(query, PORT_CHECK, "port_element_association",
                    f"The {terminal} terminal attaches to conductor geometry associated with another canonical element.",
                    objects=objects, ports=(port,), severity="warning", relationship=terminal,
                    assumptions=("Global conductors without element membership do not imply an element mismatch.",)))
        common = {components[index] for index in terminal_hits[0]} & {components[index] for index in terminal_hits[1]}
        if common:
            objects = tuple({pieces[index][0].object_id: pieces[index][0] for index in (*terminal_hits[0], *terminal_hits[1])}.values())
            findings.append(_observation(query, PORT_CHECK, "port_same_conductor_component",
                "Both terminals attach to the same geometrically connected conductor component; the excitation is geometrically suspicious.",
                objects=objects, ports=(port,), severity="warning",
                measurements={"terminal_separation_mm": Measurement(math.dist(port.positive_point, port.negative_point), "mm", tolerance)}))

    for a, b in combinations(evaluated_ports, 2):
        distance, reversed_order = _segment_distance(a, b)
        if distance <= tolerance:
            findings.append(_observation(query, PORT_CHECK, "coincident_port_segments",
                "Two ports occupy the same geometric segment within tolerance; this alone does not establish RF invalidity.",
                ports=(a, b), severity="warning", measurements={
                    "endpoint_pair_distance_mm": Measurement(distance, "mm", tolerance),
                    "reversed_endpoint_order": Measurement(int(reversed_order), ""),
                    "cross_element": Measurement(int(a.element is not None and b.element is not None and a.element != b.element), ""),
                }))
    # Complete-link groups avoid assuming tolerance proximity is transitive.
    groups = []
    for port in evaluated_ports:
        group = next((group for group in groups if all(_segment_distance(port, other)[0] <= tolerance for other in group)), None)
        if group is None:
            groups.append([port])
        else:
            group.append(port)
    findings.append(_observation(query, PORT_CHECK, "port_segment_summary",
        "Evaluated port segments are grouped by endpoint-pair coincidence, allowing reversed orientation.",
        ports=tuple(evaluated_ports), measurements={
            "canonical_port_count": Measurement(len(query.ports), ""),
            "evaluated_port_count": Measurement(len(evaluated_ports), ""),
            "distinct_segment_groups": Measurement(len(groups), ""),
        }, assumptions=("Coincidence groups use complete-link endpoint tolerance; no electrical equivalence is inferred.",)))
    return tuple(findings), _coverage(PORT_CHECK, "unknown" if incomplete else "completed",
        f"Inspected {len(evaluated_ports)} of {len(query.ports)} port segments; {len(groups)} distinct geometric groups. "
        "Attachment is geometric only; incomplete geometry or approximation limits remain unknown." if incomplete else
        f"Inspected {len(evaluated_ports)} port segments; {len(groups)} distinct geometric groups. Geometric attachment is not RF certification.")


def check_substrate_support_containment(query, config=EngineeringCheckConfig()):
    """Check final footprints; support association is explicit, conservative evidence."""
    supports = sorted((obj for obj in query.objects if obj.is_physical is not False and (
        obj.semantic_role == "substrate" or set(obj.tags) & {"support", "support_region"})), key=lambda obj: obj.object_id)
    if not supports:
        return (), _coverage(CONTAINMENT_CHECK, "not_applicable", "No canonical substrate/support region is declared; containment is not asserted.")
    conductors, unknown = _conductors(query)
    findings, incomplete = [], bool(unknown)
    tolerance = query.tolerances.layer_mm
    for obj in unknown:
        findings.append(_observation(query, CONTAINMENT_CHECK, "support_containment_unknown",
            "Conductor geometry is unresolved or unsupported for planar containment.", objects=(obj,)))
    for obj in conductors:
        # No nearest-board or bounding-box inference: require compatible element
        # membership and a touching/intersecting Z interval. Prefer local support.
        members = {source.element for source, _ in _contributions(query, obj) if source.element is not None}
        eligible = [support for support in supports if support.element is None or members == {support.element}]
        local = [support for support in eligible if support.element is not None]
        if local:
            eligible = local
        unresolved = [support for support in eligible if support.status != "completed" or support.planar_shape is None
                      or support.z_min is None or support.z_max is None]
        candidates = [support for support in eligible if support not in unresolved and
                      max(obj.z_min - support.z_max, support.z_min - obj.z_max, 0.0) <= tolerance]
        if unresolved or len(candidates) != 1:
            incomplete = True
            findings.append(_observation(query, CONTAINMENT_CHECK, "support_containment_unknown",
                "A unique resolved supporting region cannot be established from canonical membership and Z adjacency.",
                objects=(obj, *eligible)))
            continue
        support = candidates[0]
        shape, region = obj.planar_shape, support.planar_shape
        outside = shape.difference(region)
        excess = shape.difference(region.buffer(tolerance))
        error = obj.approximation.maximum_chord_error_mm + support.approximation.maximum_chord_error_mm
        credible_excess = shape.difference(region.buffer(tolerance + error))
        uncertain = not excess.is_empty and credible_excess.is_empty
        # A remote subtractive circular hole must not make an exact feed/board
        # edge unknown. Only curved positive contributors near that edge can
        # expand outside an otherwise containing exact polygonal support.
        if excess.is_empty:
            if support.approximation.maximum_chord_error_mm > tolerance:
                uncertain = shape.distance(region.boundary) <= error + tolerance
            for source, retained in _contributions(query, obj):
                if source.primitive in {"cylinder", "sheet_circle"}:
                    chord = source.approximation.maximum_chord_error_mm
                    uncertain |= chord > tolerance and retained.distance(region.boundary) <= chord + tolerance
        # Sample the complete resolved boundary (including interior rings). The
        # distance function is 1-Lipschitz: half a sample interval bounds the
        # unsampled boundary maximum. Never label this as an exact maximum.
        x0, y0, x1, y1 = shape.bounds
        sample_interval = max(math.hypot(x1-x0, y1-y0) / 256, tolerance)
        coords = get_coordinates(shape.boundary.segmentize(sample_interval))
        sampled_overhang = max((region.distance(Point(x, y)) for x, y in coords), default=0.0)
        measurements = {
            "outside_support_area_mm2": Measurement(outside.area, "mm²"),
            "outside_tolerance_area_mm2": Measurement(excess.area, "mm²"),
            "maximum_sampled_boundary_overhang_mm": Measurement(sampled_overhang, "mm", tolerance),
            "overhang_sampling_error_bound_mm": Measurement(sample_interval / 2, "mm"),
        }
        if outside.is_empty:
            measurements["minimum_support_edge_clearance_mm"] = Measurement(shape.distance(region.boundary), "mm", tolerance)
        if uncertain:
            category, severity = "support_containment_unknown", "info"
            message = "Containment lies within curved-boundary approximation uncertainty; no clean or overhang conclusion is available."
            incomplete = True
        elif not credible_excess.is_empty:
            category, severity = "support_overhang", "warning"
            message = "Resolved conductive material extends outside its associated support footprint beyond tolerance and approximation bounds."
        else:
            category, severity = "support_contained", "info"
            message = "Resolved conductor footprint is contained within support tolerance; exact edge termination is valid."
        contributors = tuple(source for source, retained in _contributions(query, obj)
                             if not excess.is_empty and not retained.intersection(excess).is_empty)
        findings.append(_observation(query, CONTAINMENT_CHECK, category, message,
            objects=tuple({item.object_id: item for item in (obj, support, *contributors)}.values()),
            severity=severity, measurements=measurements, relationship=f"support:{obj.object_id}:{support.object_id}",
            assumptions=("Support is inferred only from canonical support tags, compatible element membership, and touching/intersecting Z intervals.",
                         "This is footprint containment, not mechanical attachment or three-dimensional dielectric clearance certification.")))
    if any(s.status != "completed" or s.planar_shape is None for s in supports):
        incomplete = True
    return tuple(findings), _coverage(CONTAINMENT_CHECK, "unknown" if incomplete else "completed" if conductors else "not_applicable",
        f"Inspected {len(conductors)} planar conductors against declared support; ambiguous associations and unsupported geometry remain unknown.")


def check_excitation_consistency(query, config=EngineeringCheckConfig()):
    """Inspect the indexed excitation collection, not individual terminal quality."""
    conductors, unknown = _conductors(query)
    relevant = (*conductors, *unknown)
    represented = any(obj.element is not None for obj in relevant)
    if not represented and not query.ports:
        return (), _coverage(EXCITATION_CHECK, "not_applicable", "No represented conductive elements or indexed ports to inspect.")
    excitation = query.excitation
    explicit_intent = (
        excitation.strategy not in {"legacy_recipe", "unresolved"}
        and not (excitation.strategy == "single_element_feed" and len(query.elements) != 1)
    )
    represented_elements = {element.index for element in query.elements} if represented else set()
    expected = (
        {
            (assignment.element_row, assignment.element_column)
            for assignment in excitation.element_assignments
        }
        if explicit_intent
        else represented_elements
    )
    tolerance = query.tolerances.layer_mm
    incomplete = bool(unknown) or not represented or any(c.entity_kind == "layout" and c.status != "completed" for c in query.coverage)
    findings, evaluated = [], []
    associated = {element: [] for element in expected}
    assignment_by_port = {
        assignment.port_id: (assignment.element_row, assignment.element_column)
        for assignment in excitation.element_assignments
        if assignment.port_id is not None
    }
    realizing_port_ids = set(excitation.realizing_port_ids)
    canonical_port_ids = {port.port_id for port in query.ports}
    other_physical_port_ids = canonical_port_ids - realizing_port_ids
    for port in sorted(query.ports, key=lambda item: item.port_id):
        intended_element = assignment_by_port.get(port.port_id) if explicit_intent else port.element
        if intended_element in associated:
            associated[intended_element].append(port)
        if explicit_intent and port.port_id not in assignment_by_port:
            # Recipe compatibility ports remain physical canonical geometry, but
            # are not evidence for the current explicit excitation intent.
            continue
        if port.element not in represented_elements or port.status != "completed" or port.kind != "discrete" or port.positive_point is None or port.negative_point is None:
            incomplete = True
            findings.append(_observation(query, EXCITATION_CHECK, "excitation_association_unknown",
                "This port cannot be fully mapped to a represented element and supported discrete physical segment.", ports=(port,)))
        else:
            evaluated.append(port)
    for element in sorted(expected):
        if not associated[element]:
            findings.append(_observation(query, EXCITATION_CHECK, "excitation_missing_element_port",
                "A represented layout element has no canonically associated port; per-element excitation coverage is incomplete.",
                severity="warning", elements=(element,), relationship=f"element:{element}",
                measurements={"associated_port_count": Measurement(0, "")}))
    if explicit_intent and excitation.realization_status != "realized":
        findings.append(_observation(
            query,
            EXCITATION_CHECK,
            "excitation_intent_unrealized",
            "Canonical excitation intent is not physically realized by the current design; no missing feed network or ports are inferred or fabricated.",
            severity="warning",
            elements=tuple(sorted(expected)),
            relationship=f"excitation_strategy:{excitation.strategy}",
            measurements={
                "intended_assignment_count": Measurement(len(excitation.element_assignments), ""),
                "mapped_physical_port_count": Measurement(len(assignment_by_port), ""),
                "realizing_physical_port_count": Measurement(len(realizing_port_ids), ""),
                "other_physical_port_count": Measurement(len(other_physical_port_ids), ""),
            },
            assumptions=(
                f"Canonical strategy is {excitation.strategy} with realization status {excitation.realization_status}.",
                "This structural finding does not assess RF performance or synthesize feed geometry.",
            ),
        ))
    groups = []
    for port in evaluated:
        group = next((g for g in groups if all(_segment_distance(port, other)[0] <= tolerance for other in g)), None)
        if group is None:
            groups.append([port])
        else:
            group.append(port)
    pieces = _pieces(conductors)
    ambiguous_groups = 0
    for group in groups:
        declared = {port.element for port in group}
        touched, objects = set(), {}
        group_uncertain = False
        for port in group:
            for point in (port.positive_point, port.negative_point):
                hits, uncertain, _ = _endpoint_evidence(point, pieces, tolerance)
                group_uncertain |= bool(uncertain)
                for index in hits:
                    obj = pieces[index][0]
                    for source, footprint in _contributions(query, obj):
                        if source.element is not None and footprint.distance(Point(*point[:2])) <= tolerance:
                            touched.add(source.element)
                            objects.update({obj.object_id: obj, source.object_id: source})
        incomplete |= group_uncertain
        members = declared | touched
        if len(members) > 1:
            ambiguous_groups += 1
            findings.append(_observation(query, EXCITATION_CHECK, "excitation_shared_segment",
                "One physical excitation segment maps to multiple elements through canonical indexing and/or resolved conductor membership. "
                "The represented per-element ports are not geometrically independent at this segment; RF isolation and excitation weights are not assessed.",
                objects=tuple(objects.values()), ports=tuple(group), elements=tuple(members), severity="warning",
                relationship="segment_group:" + ":".join(port.port_id for port in group), measurements={
                    "nominal_port_count": Measurement(len(group), ""),
                    "declared_element_count": Measurement(len(declared), ""),
                    "resolved_contact_element_count": Measurement(len(touched), ""),
                    "associated_element_count": Measurement(len(members), ""),
                }))
    findings.append(_observation(query, EXCITATION_CHECK, "excitation_collection_summary",
        "Collection-level coverage of intended excitation assignments by linked physical ports and distinct evaluated segment groups. "
        "This does not establish impedance match, amplitude/phase, RF isolation, or corporate-feed behavior.",
        ports=tuple(query.ports), elements=tuple(expected), measurements={
            "expected_element_count": Measurement(len(expected), ""),
            "elements_with_indexed_port": Measurement(sum(bool(ports) for ports in associated.values()), ""),
            "intended_assignment_count": Measurement(len(excitation.element_assignments) if explicit_intent else len(expected), ""),
            "intent_mapped_physical_port_count": Measurement(len(assignment_by_port) if explicit_intent else sum(bool(ports) for ports in associated.values()), ""),
            "current_realizing_port_count": Measurement(len(realizing_port_ids), ""),
            "other_physical_port_count": Measurement(len(other_physical_port_ids), ""),
            "canonical_port_count": Measurement(len(query.ports), ""),
            "evaluated_indexed_port_count": Measurement(len(evaluated), ""),
            "physical_segment_group_count": Measurement(len(groups), ""),
            "multi_element_segment_group_count": Measurement(ambiguous_groups, ""),
        }, assumptions=(("Explicit canonical excitation assignments are used when present; otherwise legacy element-indexed port behavior is retained."
                         if explicit_intent else
                         "The current canonical port representation assigns an element ordinal to each discrete port; coverage is checked under that per-element representation."),
                        "Multiple distinct ports on one element are counted, not automatically classified as an error.",
                        "Physical ports not explicitly linked by the current realized excitation remain compatibility geometry and do not establish strategy realization.",
                        "No feed-network topology or installed repair capability is inferred from these counts.")))
    return tuple(findings), _coverage(EXCITATION_CHECK, "unknown" if incomplete else "completed",
        "Inspected element-indexed port coverage and physical-segment group associations; unresolved geometry/layout/ports limit conclusions." if incomplete else
        "Element-indexed collection inspected. Findings describe structural coverage, not RF excitation correctness.")


def check_composed_feature_uniformity(
    design: AntennaDesign,
    query: EvaluatedGeometryQueryResult,
    _config=EngineeringCheckConfig(),
):
    """Report persisted features whose explicit scope covers only part of an array."""

    if not query.objects:
        return (), _coverage(
            COMPOSED_UNIFORMITY_CHECK,
            "not_applicable",
            "No canonical geometry exists to associate with composed-feature scope.",
        )
    incomplete = any(item.status != "completed" for item in query.objects)
    if design.array.element_count <= 1 or not design.composed_operations:
        return (), _coverage(
            COMPOSED_UNIFORMITY_CHECK,
            "unknown" if incomplete else "completed",
            "Incomplete canonical geometry prevents a clean composed-feature scope conclusion."
            if incomplete else
            "No multi-element composed-feature scope comparison is required.",
        )

    expected = {
        (row, column)
        for row in range(1, design.array.rows + 1)
        for column in range(1, design.array.columns + 1)
    }
    findings = []
    for group in design.composed_operations:
        selector = group.target_selector
        if selector is None:
            covered = set()
            scope = "missing"
        elif selector.scope == "all":
            covered = set(expected)
            scope = "all"
        else:
            covered = set(selector.elements) & expected
            scope = selector.scope
        missing = expected - covered
        if not missing:
            continue
        findings.append(_observation(
            query,
            COMPOSED_UNIFORMITY_CHECK,
            "composed_feature_nonuniform",
            "A persisted composed feature is applied to only part of the array. "
            "The array elements are not geometrically uniform for this feature scope.",
            severity="warning",
            elements=tuple(sorted(missing)),
            relationship=group.group_id,
            measurements={
                "array_element_count": Measurement(len(expected), ""),
                "feature_element_count": Measurement(len(covered), ""),
                "missing_element_count": Measurement(len(missing), ""),
            },
            assumptions=(
                f"Composed feature {group.group_id} declares scope={scope!r} for role "
                f"{selector.role!r}." if selector is not None else
                f"Composed feature {group.group_id} has no explicit target selector.",
                "This check compares canonical feature scope; it does not infer intended RF symmetry.",
            ),
            source_method="canonical_composed_target_scope",
            source_applicability="Persisted composed-operation selectors and canonical array coordinates.",
            source_limitations=(
                "Scope uniformity does not establish electrical equivalence or RF performance.",
                "A deliberately asymmetric array may retain this warning and disclose that decision.",
            ),
        ))
    return tuple(findings), _coverage(
        COMPOSED_UNIFORMITY_CHECK,
        "completed",
        "Compared each persisted composed-feature selector with all canonical array elements.",
    )


def run_engineering_checks(design: AntennaDesign, *, checks=None, config=EngineeringCheckConfig()) -> EngineeringReport:
    """Evaluate once and combine independently covered checks without mutation."""
    selected = _selected_checks(checks)
    if not isinstance(config, EngineeringCheckConfig):
        raise ValueError("config must be EngineeringCheckConfig.")
    functions = {
        CONTACT_CHECK: check_conductor_contact_clearance,
        PORT_CHECK: check_port_attachment_distinctness,
        CONTAINMENT_CHECK: check_substrate_support_containment,
        EXCITATION_CHECK: check_excitation_consistency,
        COMPOSED_UNIFORMITY_CHECK: lambda query, config: check_composed_feature_uniformity(
            design, query, config
        ),
    }
    try:
        query = evaluate_geometry(design)
    except Exception as exc:
        from studio.antenna_agent_runner import semantic_design_hash
        return EngineeringReport(EngineeringDesignRef(design.design_id, design.revision), semantic_design_hash(design), SUITE_VERSION,
            coverage=tuple(_coverage(name, "failed", f"Geometry evaluation failed ({type(exc).__name__}); no clean conclusion is available.") for name in selected))
    findings, coverage = [], []
    for name in selected:
        try:
            observations, checked = functions[name](query, config)
        except Exception as exc:
            observations = ()
            checked = _coverage(name, "failed", f"Geometry inspection failed ({type(exc).__name__}); no clean conclusion is available.")
        findings.extend(observations)
        coverage.append(checked)
    return EngineeringReport(query.design_ref, query.semantic_design_hash, SUITE_VERSION,
        tuple(sorted(findings, key=lambda item: item.observation_id)), tuple(coverage))
