"""Stage-3 protocol fixtures only: no engineering checks or provider requests."""

import hashlib
import json
import unittest
from dataclasses import replace

from studio.antenna_agent import create_default_agent
from studio.antenna_agent_runner import semantic_design_hash
from studio.antenna_engineering import (
    EngineeringCheckCoverage, EngineeringDesignRef, EngineeringMeasurement,
    EngineeringObservation, EngineeringReport, EngineeringSource, stable_observation_id,
)
from studio.antenna_llm_planner import (
    AgentLoopBudgets, AgentStep, AgentStepObservation, GeminiSchemaConstrainedPlanner,
    GroqSchemaConstrainedPlanner, OpenRouterNemotronPlanner, SchemaConstrainedLLMPlanner,
    build_agent_step_exchange, plan_json_schema,
)


class EngineeringProtocolTests(unittest.TestCase):
    def finding(self, **updates):
        data = dict(
            check_id="contact", check_version="1", severity="warning", category="contact",
            message="Two conductors overlap within the evaluated region.",
            source=EngineeringSource("planar intersection", "common-axis extrusions", "b" * 64,
                                     ("Polygon approximation; no RF performance assessment.",)),
            affected_objects=("conductor_a", "conductor_b"), affected_ports=("port_1",),
            affected_elements=((1, 1), (2, 1)),
            measured_values={"overlap_area": EngineeringMeasurement(1.25, "mm^2", 0.001)},
            assumptions=("Conductors share an extrusion layer.",),
        )
        data.update(updates)
        return EngineeringObservation(**data)

    def coverage(self, status="completed"):
        return EngineeringCheckCoverage("contact", "1", status, "common-axis extrusions",
                                        "Fixture assessment.", ("No full-wave analysis.",))

    def report(self, **updates):
        data = dict(working_design_ref=EngineeringDesignRef("design-a", 7),
                    semantic_design_hash="a" * 64, check_suite_version="1",
                    findings=(self.finding(),), coverage=(self.coverage(),))
        data.update(updates)
        return EngineeringReport(**data)

    def test_report_json_round_trip_preserves_exact_state_and_provenance(self):
        report = self.report()
        restored = EngineeringReport.from_dict(json.loads(json.dumps(report.to_dict(), allow_nan=False)))
        self.assertEqual(restored, report)
        self.assertEqual(restored.working_design_ref.revision, 7)
        self.assertEqual(restored.semantic_design_hash, "a" * 64)
        self.assertEqual(restored.findings[0].source, report.findings[0].source)
        self.assertEqual(restored.coverage[0].limitations, ("No full-wave analysis.",))

    def test_all_severities_are_passive_serializable_values(self):
        for severity in ("info", "warning", "blocking"):
            with self.subTest(severity=severity):
                report = self.report(findings=(self.finding(severity=severity),))
                self.assertEqual(EngineeringReport.from_dict(report.to_dict()).findings[0].severity, severity)

    def test_invalid_severity_rejected_directly_and_on_read(self):
        for severity in ("error", "WARN", None, 1):
            with self.subTest(severity=severity), self.assertRaises(ValueError):
                self.finding(severity=severity)
        payload = self.finding().to_dict()
        payload["severity"] = "critical"
        with self.assertRaises(ValueError):
            EngineeringObservation.from_dict(payload)

    def test_coverage_statuses_and_invalid_status(self):
        for status in ("completed", "not_applicable", "unknown", "failed"):
            with self.subTest(status=status):
                coverage = self.coverage(status)
                self.assertEqual(EngineeringCheckCoverage.from_dict(coverage.to_dict()), coverage)
        with self.assertRaises(ValueError):
            self.coverage("passed")

    def test_measurements_have_units_optional_tolerance_and_finite_numbers(self):
        measurements = {
            "overlap_area": EngineeringMeasurement(2, "mm^2", 0.01),
            "clearance_distance": EngineeringMeasurement(0.5, "mm"),
            "wavelength_ratio": EngineeringMeasurement(0.55, "1"),
            "endpoint_distance": EngineeringMeasurement(0, "mm", 1e-7),
        }
        finding = self.finding(measured_values=measurements)
        self.assertEqual(EngineeringObservation.from_dict(finding.to_dict()), finding)
        measurements.clear()
        self.assertEqual(len(finding.measured_values), 4)
        with self.assertRaises(TypeError):
            finding.measured_values["injected"] = EngineeringMeasurement(3, "mm")
        for value in (float("nan"), float("inf"), True, "3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                EngineeringMeasurement(value, "mm")
        for tolerance in (-1, float("nan"), True):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                EngineeringMeasurement(1, "mm", tolerance)

    def test_relationship_id_ignores_order_measurements_and_occurrence(self):
        first = self.finding()
        second = self.finding(affected_objects=tuple(reversed(first.affected_objects)),
                              affected_elements=tuple(reversed(first.affected_elements)),
                              check_version="2", severity="info", message="Changed measurement.",
                              measured_values={"overlap_area": EngineeringMeasurement(3, "mm^2")})
        self.assertEqual(first.observation_id, second.observation_id)
        self.assertNotEqual(first.observation_id, self.finding(affected_ports=("port_2",)).observation_id)
        self.assertNotEqual(first.observation_id, self.finding(relationship_key="positive_terminal").observation_id)
        report = self.report()
        moved = replace(report, working_design_ref=EngineeringDesignRef("design-a", 8), semantic_design_hash="c" * 64)
        self.assertEqual(report.findings[0].observation_id, moved.findings[0].observation_id)
        payload = first.to_dict()
        payload["observation_id"] = "invented"
        with self.assertRaises(ValueError):
            EngineeringObservation.from_dict(payload)

    def test_invalid_element_coordinates_are_rejected(self):
        for elements in (((0, 1),), ((1.5, 1),), ((True, 1),), ((1,),), "1,1"):
            with self.subTest(elements=elements), self.assertRaises(ValueError):
                stable_observation_id("contact", "contact", affected_elements=elements)

    def test_capability_refs_need_external_installed_allowlist_on_creation_and_restore(self):
        with self.assertRaises(ValueError):
            self.finding(capability_refs=("corporate_feed.synthesize",))
        finding = self.finding(capability_refs=("parameter.set",), installed_capability_ids={"parameter.set"})
        with self.assertRaises(ValueError):
            EngineeringObservation.from_dict(finding.to_dict())
        self.assertEqual(EngineeringObservation.from_dict(finding.to_dict(), installed_capability_ids={"parameter.set"}), finding)
        payload = finding.to_dict()
        payload["installed_capability_ids"] = ["parameter.set"]
        with self.assertRaises(ValueError):
            EngineeringObservation.from_dict(payload)
        report = self.report(findings=(finding,))
        self.assertEqual(EngineeringReport.from_dict(report.to_dict(), installed_capability_ids={"parameter.set"}), report)

    def test_empty_findings_do_not_imply_successful_inspection(self):
        reports = [self.report(findings=(), coverage=(self.coverage(status),))
                   for status in ("completed", "unknown", "failed", "not_applicable")]
        for report in reports:
            restored = EngineeringReport.from_dict(report.to_dict())
            self.assertEqual(restored.findings, ())
            self.assertEqual(restored.coverage, report.coverage)
        self.assertEqual(len({json.dumps(report.to_dict()) for report in reports}), 4)
        uninspected = self.report(findings=(), coverage=())
        self.assertEqual(uninspected.to_dict()["coverage"], [])

    def test_report_schema_state_and_coverage_consistency(self):
        for updates in ({"schema_version": 2}, {"schema_version": True}, {"semantic_design_hash": ""},
                        {"working_design_ref": {}}, {"coverage": ()},
                        {"findings": (self.finding(), self.finding())},
                        {"coverage": (self.coverage(), self.coverage())}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                self.report(**updates)
        for revision in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                EngineeringDesignRef("design-a", revision)
        empty = self.report(working_design_ref=None, semantic_design_hash=semantic_design_hash(None), findings=(), coverage=())
        self.assertEqual(EngineeringReport.from_dict(empty.to_dict()), empty)

    def exchange_inputs(self):
        agent = create_default_agent()
        design = agent.create_design("inset_patch")
        return dict(instruction="Inspect this design.", current_design=design.to_dict(),
                    capability_manifest=agent.capability_manifest(design), remaining_budgets=AgentLoopBudgets())

    def test_optional_report_is_separate_context_not_execution_observation(self):
        args = self.exchange_inputs()
        state = args["current_design"]
        report = self.report(working_design_ref=EngineeringDesignRef(state["design_id"], state["revision"]))
        baseline = build_agent_step_exchange(**args)
        enriched = build_agent_step_exchange(**args, engineering_report=report)
        context = json.loads(enriched.user_content)
        summary = context.pop("engineering_summary")
        self.assertEqual(summary["raw_finding_count"], len(report.findings))
        self.assertNotIn("engineering_report", context)
        context["agent_step_schema"]["properties"]["engineering_disposition"]["properties"].pop("engineering_report_hash")
        self.assertEqual(context, json.loads(baseline.user_content))
        self.assertNotIn("agent_observation", context)
        self.assertEqual(enriched.schema["properties"]["calls"], baseline.schema["properties"]["calls"])
        self.assertTrue(enriched.system_instruction.startswith(baseline.system_instruction))
        self.assertEqual(enriched.callable_names, baseline.callable_names)

    def test_exchange_rechecks_capabilities_and_revision(self):
        args = self.exchange_inputs()
        state = args["current_design"]
        with self.assertRaises(ValueError):
            build_agent_step_exchange(**args, engineering_report=self.report())
        # A previously installed tool is not automatically trusted in a new context.
        finding = self.finding(capability_refs=("removed_tool",), installed_capability_ids={"removed_tool"})
        report = self.report(working_design_ref=EngineeringDesignRef(state["design_id"], state["revision"]), findings=(finding,))
        with self.assertRaises(ValueError):
            build_agent_step_exchange(**args, engineering_report=report)

    def test_none_preserves_existing_context_for_every_provider(self):
        args = self.exchange_inputs()
        expected = build_agent_step_exchange(**args)
        self.assertEqual(expected, build_agent_step_exchange(**args, engineering_report=None))
        for cls in (SchemaConstrainedLLMPlanner, GeminiSchemaConstrainedPlanner,
                    GroqSchemaConstrainedPlanner, OpenRouterNemotronPlanner):
            with self.subTest(provider=cls.__name__):
                # No credentials or network: exercise each backend's existing common context path.
                planner = cls.__new__(cls)
                planner._agent_loop_remaining_budgets = args["remaining_budgets"]
                context_args = {key: value for key, value in args.items() if key != "remaining_budgets"}
                actual = planner._exchange(**context_args, project_memory=None,
                                           agent_observation=None, execution_feedback=None)
                self.assertEqual(actual, expected)
                self.assertNotIn("engineering_report", json.loads(actual.user_content))

    def test_context_and_output_contract_fingerprints(self):
        manifest = create_default_agent().capability_manifest(None)
        exchange = build_agent_step_exchange(instruction="Inspect installed capabilities.", current_design=None,
                                            capability_manifest=manifest, remaining_budgets=AgentLoopBudgets())
        # Deliberately re-frozen for AgentStep contract v3. Contract v3 prevents
        # refused requests from proposing active semantic project memory and adds
        # the matching provider-neutral instruction. The output schema, embedded
        # user content, and ToolPlan remain unchanged; the strict AgentStep parser
        # enforces the status-dependent rule after decoding.
        fixtures = (
            (exchange.user_content, "26ec12dc412004106e74b25b02d66b02c42b4272702dde2b09dc4e66e28ca2a8"),
            (exchange.system_instruction, "a6cabca28cde746fb76584818f2a2f12f31e5277279ee4bae4c45197b4a820f8"),
            (json.dumps(exchange.schema, sort_keys=True), "f06755d0110a353ae37da7b64772029fb066bf3ff241f24f4803a9ee47c8ecde"),
            (json.dumps(plan_json_schema(tuple(manifest["callable_tools"])), sort_keys=True),
             "553cc402a0599a9f119598def0485fb09ce3c3222402de3999eb29dd6d8aee19"),
        )
        for payload, expected in fixtures:
            self.assertEqual(hashlib.sha256(payload.encode()).hexdigest(), expected)
        self.assertEqual(set(AgentStep("finish", "Done.").to_dict()), {"schema_version", "status", "message", "calls"})
        with self.assertRaises(ValueError):
            AgentStepObservation("accepted", (), None, "a" * 64, AgentLoopBudgets())

    def test_serialization_scrubs_labelled_credentials_and_rejects_metadata(self):
        finding = self.finding(
            message="Request token=private-token failed.",
            source=EngineeringSource("inspection", "authorization=private-auth", limitations=("Bearer private-bearer",)),
            assumptions=("api_key=private-key",),
            measured_values={"secret_metadata": EngineeringMeasurement(987654321, "mm")},
        )
        serialized = json.dumps(self.report(findings=(finding,)).to_dict())
        for secret in ("private-token", "private-auth", "private-bearer", "private-key", "987654321"):
            self.assertNotIn(secret, serialized)
        self.assertIn("[REDACTED]", serialized)
        payload = self.report().to_dict()
        payload["metadata"] = {"API_KEY": "do-not-log-this"}
        with self.assertRaises(ValueError) as error:
            EngineeringReport.from_dict(payload)
        self.assertNotIn("do-not-log-this", str(error.exception))
        payload = self.finding().source.to_dict()
        payload["hidden_reasoning"] = "not part of this protocol"
        with self.assertRaises(ValueError):
            EngineeringSource.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
