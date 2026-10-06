"""Deterministic planner projection; raw reports and check calculations stay intact."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from studio.antenna_design import AntennaDesign
from studio.antenna_engineering import EngineeringReport, _redact


GROUPING_VERSION = 1
GROUP_PREFIX = "eng-group-v1-"


def payload_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def engineering_report_hash(report: EngineeringReport) -> str:
    return payload_hash(report.to_dict())


@dataclass(frozen=True, slots=True)
class PlannerEngineeringSummary:
    """Immutable serialized projection, including full membership for audit."""

    _serialized: str

    def to_dict(self):
        return json.loads(self._serialized)

    @property
    def summary_hash(self):
        return payload_hash(self.to_dict())

    def to_planner_dict(self):
        result = self.to_dict()
        result["summary_hash"] = self.summary_hash
        for group in result["groups"]:
            # Exact membership and relationship detail are retained internally
            # and in audit. They are not required for a group acknowledgement.
            group.pop("constituent_observation_ids")
            group.pop("affected_relationships")
        return result


# Only categories with an explicit structured meaning in the installed checks
# are eligible for repeated-relationship grouping. Unknown future categories
# stay individual until their grouping semantics are reviewed.
_GROUPABLE = {
    ("conductor_contact_clearance", "conductor_contact"),
    ("conductor_contact_clearance", "conductor_clearance"),
    ("conductor_contact_clearance", "union_constituent_contact"),
    ("port_attachment_distinctness", "coincident_port_segments"),
    ("port_attachment_distinctness", "port_element_association"),
    ("port_attachment_distinctness", "port_terminal_attached"),
    ("port_attachment_distinctness", "port_terminal_unattached"),
    ("port_attachment_distinctness", "port_same_conductor_component"),
    ("substrate_support_containment", "support_contained"),
    ("substrate_support_containment", "support_overhang"),
    ("excitation_consistency", "excitation_shared_segment"),
    ("excitation_consistency", "excitation_missing_element_port"),
}


def summarize_engineering_report(report: EngineeringReport, design: AntennaDesign | None = None) -> PlannerEngineeringSummary:
    # Roles are canonical structured metadata, not parsed finding prose. This
    # reads object tags through the existing semantic-role helper; no geometry
    # reevaluation, tool execution, or design mutation occurs.
    from studio.antenna_agent import AntennaDesignAgent
    roles = {obj.object_id: AntennaDesignAgent._semantic_role(obj) for obj in design.geometry} if design else {}
    buckets = {}
    for finding in report.findings:
        relationship_type = finding.relationship_key.split(":", 1)[0]
        object_roles = tuple(sorted(roles[obj] for obj in finding.affected_objects if obj in roles))
        complete_roles = all(obj in roles for obj in finding.affected_objects)
        known = (finding.check_id, finding.category) in _GROUPABLE
        # Flags describe the relationship (e.g. within versus across elements),
        # unlike magnitudes, whose changes should not change group identity.
        flags = {key: measurement.value for key, measurement in finding.measured_values.items()
                 if key in {"same_element", "cross_element", "inside_resolved_hole", "contact_within_tolerance"}}
        signature = {
            "grouping_version": GROUPING_VERSION, "check_id": finding.check_id,
            "check_version": finding.check_version, "category": finding.category,
            "severity": finding.severity, "relationship_type": relationship_type,
            "semantic_roles": object_roles, "relationship_flags": flags,
            "measurement_schema": sorted((name, value.unit, value.tolerance) for name, value in finding.measured_values.items()),
            "source_method": finding.source.method, "source_applicability": finding.source.applicability,
            "source_limitations": finding.source.limitations,
        }
        if not known or not complete_roles:
            signature["individual_relationship"] = finding.observation_id
        identity = GROUP_PREFIX + payload_hash(signature)
        buckets.setdefault(identity, (signature, []))[1].append(finding)
    groups = []
    for identity, (signature, members) in sorted(buckets.items()):
        members = sorted(members, key=lambda item: item.observation_id)
        measurements = {}
        for key, first in sorted(members[0].measured_values.items()):
            values = [member.measured_values[key].value for member in members]
            low, high = min(values), max(values)
            common = high - low <= (first.tolerance or 0)
            measurements[key] = {"unit": first.unit, "tolerance": first.tolerance,
                "min": low, "max": high, "common_value": (low if low == high else (low + high) / 2) if common else None}
        groups.append({
            "group_id": identity, "check_id": signature["check_id"], "check_version": signature["check_version"],
            "category": signature["category"], "severity": signature["severity"],
            "relationship_type": signature["relationship_type"], "semantic_roles": signature["semantic_roles"],
            "relationship_flags": signature["relationship_flags"],
            "occurrence_count": len(members),
            "constituent_observation_ids": [f.observation_id for f in members],
            "affected_objects": sorted({obj for f in members for obj in f.affected_objects}),
            "affected_ports": sorted({port for f in members for port in f.affected_ports}),
            "affected_elements": sorted({element for f in members for element in f.affected_elements}),
            "affected_relationships": [{"observation_id": f.observation_id, "relationship_key": f.relationship_key,
                "objects": f.affected_objects, "ports": f.affected_ports, "elements": f.affected_elements} for f in members],
            "measurements": measurements,
            # One representative message supplies the check's existing meaning;
            # it is never a grouping key. Its IDs describe one occurrence only.
            "representative_message": members[0].message,
            "representative_message_scope": "one occurrence; use aggregate measurements and affected sets for the group",
            "limitations": signature["source_limitations"],
        })
    data = {
        "schema_version": 1, "grouping_version": GROUPING_VERSION,
        "engineering_report_hash": engineering_report_hash(report),
        "working_design_ref": report.working_design_ref.to_dict() if report.working_design_ref else None,
        "semantic_design_hash": report.semantic_design_hash, "check_suite_version": report.check_suite_version,
        "raw_finding_count": len(report.findings), "raw_warning_count": sum(f.severity == "warning" for f in report.findings),
        "grouped_warning_count": sum(group["severity"] == "warning" for group in groups),
        "groups": groups, "coverage": [item.to_dict() for item in report.coverage],
    }
    return PlannerEngineeringSummary(json.dumps(_redact(data), sort_keys=True, ensure_ascii=True))
