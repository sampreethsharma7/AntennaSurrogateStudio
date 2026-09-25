import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from studio.antenna_agent import create_default_agent
from studio.antenna_agent_runner import AntennaAgentRunner
from studio.antenna_engineering import EngineeringMeasurement
from studio.antenna_engineering_checks import run_engineering_checks
from studio.antenna_engineering_summary import (
    engineering_report_hash, payload_hash, summarize_engineering_report,
)
from studio.antenna_llm_planner import (
    AgentLoopBudgets, AgentStep, EngineeringDisposition, DeferredEngineeringFinding,
    GeminiSchemaConstrainedPlanner, OpenRouterNemotronPlanner, build_agent_step_exchange,
    parse_agent_step, validate_engineering_disposition,
)
from tests.test_antenna_agent_runner import ScriptedPlanner, step


class EngineeringSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = create_default_agent()
        cls.clean = cls.agent.create_design("inset_patch")
        cls.array = cls.agent.update_parameters(cls.clean, {"array_rows": 2, "array_columns": 3}).design
        cls.report = run_engineering_checks(cls.array)
        cls.summary = summarize_engineering_report(cls.report, cls.array)

    def warning_groups(self, summary=None):
        return [g for g in (summary or self.summary).to_dict()["groups"] if g["severity"] == "warning"]

    def disposition(self, summary=None):
        summary = summary or self.summary
        return EngineeringDisposition(summary.to_dict()["semantic_design_hash"],
            tuple(g["group_id"] for g in self.warning_groups(summary)),
            engineering_report_hash=summary.to_dict()["engineering_report_hash"])

    def test_fifteen_warnings_become_four_semantically_distinct_groups(self):
        groups = self.warning_groups()
        self.assertEqual(self.summary.to_dict()["raw_warning_count"], 15)
        self.assertEqual({g["category"]: g["occurrence_count"] for g in groups}, {
            "conductor_contact": 3, "coincident_port_segments": 3,
            "port_element_association": 6, "excitation_shared_segment": 3})
        self.assertEqual(len(groups), 4)

    def test_memberships_and_affected_relationships_are_preserved(self):
        raw = {f.observation_id: f.to_dict() for f in self.report.findings}
        represented = []
        for group in self.summary.to_dict()["groups"]:
            ids = group["constituent_observation_ids"]
            represented.extend(ids)
            self.assertEqual(group["occurrence_count"], len(ids))
            for field in ("affected_objects", "affected_ports", "affected_elements"):
                expected = {tuple(x) if isinstance(x, list) else x for identity in ids for x in raw[identity][field]}
                actual = {tuple(x) if isinstance(x, list) else x for x in group[field]}
                self.assertEqual(expected, actual)
            self.assertEqual({r["observation_id"] for r in group["affected_relationships"]}, set(ids))
        self.assertEqual(len(represented), len(set(represented)))
        self.assertEqual(set(represented), set(raw))

    def test_overlap_measurements_keep_separate_meanings(self):
        group = next(g for g in self.warning_groups() if g["category"] == "conductor_contact")
        self.assertAlmostEqual(group["measurements"]["retained_constituent_xy_overlap_area_mm2"]["common_value"], 60.4995)
        self.assertAlmostEqual(group["measurements"]["resolved_conductor_xy_overlap_area_mm2"]["common_value"], 107.4279)
        f = next(f for f in self.report.findings if f.category == "conductor_contact")
        a = replace(f, relationship_key="physical_pair:a", measured_values={"retained_constituent_xy_overlap_area_mm2": EngineeringMeasurement(1, "mm²")})
        b = replace(f, relationship_key="physical_pair:b", measured_values={"resolved_conductor_xy_overlap_area_mm2": EngineeringMeasurement(1, "mm²")})
        summary = summarize_engineering_report(replace(self.report, findings=(a, b)), self.array)
        self.assertEqual(len(summary.to_dict()["groups"]), 2)

    def test_units_severity_roles_and_relationship_type_prevent_incompatible_merge(self):
        f = next(f for f in self.report.findings if f.category == "conductor_contact")
        variants = (
            replace(f, severity="info", relationship_key="physical_pair:other"),
            replace(f, affected_objects=("ground",), relationship_key="physical_pair:other"),
            replace(f, relationship_key="different_relationship:other"),
            replace(f, relationship_key="physical_pair:other", measured_values={key: replace(value, unit="m²") for key, value in f.measured_values.items()}),
        )
        for other in variants:
            summary = summarize_engineering_report(replace(self.report, findings=(f, other)), self.array)
            self.assertEqual(len(summary.to_dict()["groups"]), 2)

    def test_ranges_and_common_values_respect_tolerance(self):
        f = next(f for f in self.report.findings if f.category == "coincident_port_segments")
        a = replace(f, relationship_key="test:a", measured_values={"distance": EngineeringMeasurement(1, "mm", .01)})
        for value, common in ((1.005, 1.0025), (2, None)):
            b = replace(a, relationship_key="test:b", measured_values={"distance": EngineeringMeasurement(value, "mm", .01)})
            groups = summarize_engineering_report(replace(self.report, findings=(a, b)), self.array).to_dict()["groups"]
            self.assertEqual(len(groups), 1)
            measurement = groups[0]["measurements"]["distance"]
            self.assertEqual(measurement["min"], 1)
            self.assertEqual(measurement["max"], value)
            self.assertEqual(measurement["common_value"], common)

    def test_group_ids_stable_across_revision_order_and_prose_changes(self):
        groups = self.summary.to_dict()["groups"]
        reordered = replace(self.report, findings=tuple(replace(f, message="Different wording.") for f in reversed(self.report.findings)))
        revised_design = replace(self.array, revision=99, metadata=(*self.array.metadata, ("audit", "later")))
        revised = run_engineering_checks(revised_design)
        for report, design in ((reordered, self.array), (revised, revised_design)):
            actual = summarize_engineering_report(report, design).to_dict()["groups"]
            self.assertEqual([g["group_id"] for g in groups], [g["group_id"] for g in actual])
        self.assertNotEqual(engineering_report_hash(self.report), engineering_report_hash(revised))

    def test_full_report_unchanged_and_compact_context_omits_member_ids(self):
        before = json.dumps(self.report.to_dict())
        summary = summarize_engineering_report(self.report, self.array)
        context = summary.to_planner_dict()
        self.assertEqual(json.dumps(self.report.to_dict()), before)
        self.assertEqual(context["coverage"], self.report.to_dict()["coverage"])
        self.assertTrue(all("constituent_observation_ids" not in g and "affected_relationships" not in g for g in context["groups"]))
        self.assertLess(len(json.dumps(context)), len(before) / 2)

    def test_clean_design_has_no_warning_groups(self):
        summary = summarize_engineering_report(run_engineering_checks(self.clean), self.clean)
        self.assertEqual(self.warning_groups(summary), [])
        self.assertEqual(summary.to_dict()["raw_warning_count"], 0)

    def test_summary_preserves_report_credential_redaction(self):
        finding = replace(self.report.findings[0], message="api_key=private-value")
        report = replace(self.report, findings=(finding,))
        summary = summarize_engineering_report(report, self.array)
        self.assertNotIn("private-value", json.dumps(summary.to_dict()))
        self.assertNotIn("private-value", json.dumps(summary.to_planner_dict()))

    def test_group_acknowledgement_and_deferment_cover_all_raw_warnings(self):
        disposition = self.disposition()
        self.assertEqual(validate_engineering_disposition(disposition, self.report, self.summary)["status"], "valid")
        deferred = replace(disposition, acknowledged_observation_ids=(), deferred=tuple(
            DeferredEngineeringFinding(identity, "User permits retained geometry.") for identity in disposition.acknowledged_observation_ids))
        self.assertEqual(validate_engineering_disposition(deferred, self.report, self.summary)["status"], "valid")
        parsed = parse_agent_step(AgentStep("finish", "Done.", engineering_disposition=deferred).to_dict(), callable_tool_names=())
        self.assertEqual(parsed.engineering_disposition, deferred)

    def test_one_group_expands_to_its_constituents_and_incomplete_is_rejected(self):
        groups = self.warning_groups()
        first = groups[0]
        selected = replace(self.report, findings=tuple(f for f in self.report.findings if f.observation_id in first["constituent_observation_ids"]))
        summary = summarize_engineering_report(selected, self.array)
        self.assertEqual(validate_engineering_disposition(self.disposition(summary), selected, summary)["status"], "valid")
        incomplete = replace(self.disposition(), acknowledged_observation_ids=(first["group_id"],))
        self.assertEqual(validate_engineering_disposition(incomplete, self.report, self.summary)["status"], "rejected")

    def test_mixing_group_and_member_or_repeating_group_cannot_double_count(self):
        disposition = self.disposition()
        identity = self.warning_groups()[0]["constituent_observation_ids"][0]
        for extra in (identity, disposition.acknowledged_observation_ids[0]):
            mixed = replace(disposition, acknowledged_observation_ids=(*disposition.acknowledged_observation_ids, extra))
            result = validate_engineering_disposition(mixed, self.report, self.summary)
            self.assertEqual(result["status"], "rejected")
            self.assertTrue(result["duplicate_observation_ids"])

    def test_stale_unknown_and_missing_report_hash_are_rejected(self):
        disposition = self.disposition()
        for changed in (replace(disposition, engineering_report_hash=None), replace(disposition, engineering_report_hash="a"*64),
                        replace(disposition, acknowledged_observation_ids=(*disposition.acknowledged_observation_ids, "eng-group-v1-unknown"))):
            self.assertEqual(validate_engineering_disposition(changed, self.report, self.summary)["status"], "rejected")
        revised = replace(self.report, working_design_ref=replace(self.report.working_design_ref, revision=100))
        summary = summarize_engineering_report(revised, self.array)
        self.assertTrue(validate_engineering_disposition(disposition, revised, summary)["stale_group_report_hash"])

    def test_runner_group_finish_audit_and_normal_retry(self):
        planner = ScriptedPlanner(AgentStep("finish", "Incomplete."),
            AgentStep("finish", "Retained with disclosed concerns.", engineering_disposition=self.disposition()))
        result = AntennaAgentRunner(self.agent).run(baseline_design=self.array, instruction="Preserve geometry.",
            project_memory=None, planner=planner)
        self.assertEqual(result.outcome, "finished")
        self.assertEqual([e.execution_status for e in result.trajectory], ["finish_rejected", "finished"])
        audit = result.trajectory[0].engineering_audit
        raw = audit["reports"][audit["before_report_hash"]]
        self.assertEqual(raw, self.report.to_dict())
        record = audit["summaries"][audit["before_summary_hash"]]
        self.assertEqual(payload_hash(record["summary"]), audit["before_summary_hash"])
        self.assertEqual(record["planner_context"], self.summary.to_planner_dict())
        self.assertEqual(result.trajectory[1].engineering_audit["summaries"], {})

    def test_providers_share_summary_and_no_raw_report_in_prompt(self):
        args = dict(instruction="Preserve the design.", current_design=self.array.to_dict(),
            capability_manifest=self.agent.capability_manifest(self.array), remaining_budgets=AgentLoopBudgets(),
            project_memory=None, agent_observation=None, engineering_report=self.report)
        expected = build_agent_step_exchange(**args)
        for cls in (GeminiSchemaConstrainedPlanner, OpenRouterNemotronPlanner):
            planner = cls.__new__(cls)
            captured = []
            def plan(**kwargs):
                captured.append(planner._exchange(**kwargs, execution_feedback=None))
                return AgentStep("finish", "Done.", engineering_disposition=self.disposition())
            with patch.object(planner, "plan", side_effect=plan):
                planner.plan_agent_step(**args)
            self.assertEqual(captured, [expected])
        context = json.loads(expected.user_content)
        self.assertNotIn("engineering_report", context)
        self.assertEqual(context["engineering_summary"], self.summary.to_planner_dict())
        args.pop("engineering_report")
        no_report = build_agent_step_exchange(**args)
        self.assertNotIn("engineering_summary", json.loads(no_report.user_content))
        self.assertNotIn("engineering_report_hash", no_report.schema["properties"]["engineering_disposition"]["properties"])


if __name__ == "__main__":
    unittest.main()
