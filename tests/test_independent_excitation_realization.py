import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from studio.antenna_agent import create_default_agent
from studio.antenna_builder import load_project_design, save_project_design
from studio.antenna_design import DesignValidationError, evaluate_scalar, resolve_parameter_values
from studio.antenna_engineering_checks import run_engineering_checks
from studio.antenna_geometry import evaluate_geometry
from studio.antenna_llm_planner import LLMToolPlan, PlannedToolCall
from studio.cst_antenna_adapter import CSTAdapter


def plan(*calls):
    return LLMToolPlan(
        "execute",
        "independent excitation realization test",
        tuple(PlannedToolCall(name, arguments) for name, arguments in calls),
    )


class IndependentExcitationRealizationTests(unittest.TestCase):
    def setUp(self):
        self.agent = create_default_agent()

    def _legacy(self, rows, columns):
        return self.agent.create_design(
            "inset_patch", {"array_rows": rows, "array_columns": columns},
        )

    def _realize(self, design):
        return self.agent.execute_llm_plan(
            design,
            plan(("excitation.set_strategy", {"strategy": "independent_ports"})),
        ).design

    @staticmethod
    def _segments(design):
        values = resolve_parameter_values(design)
        return tuple(
            (
                tuple(evaluate_scalar(value, values) for value in port.negative_point),
                tuple(evaluate_scalar(value, values) for value in port.positive_point),
            )
            for port in design.ports
        )

    @staticmethod
    def _warning_categories(design):
        return [
            finding.category
            for finding in run_engineering_checks(design).findings
            if finding.severity == "warning"
        ]

    def test_normal_single_patch_remains_backward_compatible(self):
        first = self.agent.create_design("inset_patch")
        second = self.agent.create_design("inset_patch")
        self.assertEqual(evaluate_geometry(first).geometry_hash, evaluate_geometry(second).geometry_hash)
        self.assertEqual(first.excitation.strategy, "single_element_feed")
        self.assertEqual(first.excitation.realization_status, "realized")
        feed = next(item for item in first.geometry if item.object_id == "element_1_1_feed")
        self.assertEqual(feed.dimension_map()["y_min"], "-board_length_mm/2")
        self.assertNotIn("independent_local_feed", feed.tags)

    def test_linear_and_planar_arrays_realize_one_distinct_local_port_per_element(self):
        for rows, columns, count in ((1, 4, 4), (3, 1, 3), (2, 3, 6)):
            with self.subTest(array=(rows, columns)):
                design = self._realize(self._legacy(rows, columns))
                assignments = design.excitation.element_assignments
                self.assertEqual(design.excitation.realization_status, "realized")
                self.assertTrue(design.excitation.excitation_realized)
                self.assertEqual(len(assignments), count)
                self.assertEqual(len(design.ports), count)
                self.assertEqual(len({item.port_id for item in assignments}), count)
                self.assertEqual({item.port_id for item in assignments}, {port.port_id for port in design.ports})
                self.assertEqual(len(set(self._segments(design))), count)
                self.assertEqual(
                    len([item for item in design.geometry if "independent_local_feed" in item.tags]),
                    count,
                )

    def test_2x3_engineering_checks_are_structurally_coherent(self):
        design = self._realize(self._legacy(2, 3))
        report = run_engineering_checks(design)
        warning_categories = {
            finding.category for finding in report.findings if finding.severity == "warning"
        }
        self.assertNotIn("conductor_contact", warning_categories)
        self.assertNotIn("coincident_port_segments", warning_categories)
        self.assertNotIn("port_element_association", warning_categories)
        self.assertNotIn("excitation_shared_segment", warning_categories)
        self.assertNotIn("excitation_missing_element_port", warning_categories)
        self.assertNotIn("excitation_intent_unrealized", warning_categories)
        self.assertNotIn("support_overhang", warning_categories)
        summary = next(
            finding for finding in report.findings
            if finding.category == "excitation_collection_summary"
        )
        self.assertEqual(summary.measured_values["expected_element_count"].value, 6)
        self.assertEqual(summary.measured_values["intent_mapped_physical_port_count"].value, 6)
        self.assertEqual(summary.measured_values["physical_segment_group_count"].value, 6)
        self.assertTrue(all(item.status == "completed" for item in report.coverage))

    def test_row_and_column_spacing_move_ports_with_local_element_frames(self):
        for rows, columns in ((3, 1), (1, 4)):
            with self.subTest(array=(rows, columns)):
                before = self._realize(self._legacy(rows, columns))
                after = self.agent.update_parameters(
                    before, {"element_spacing_lambda": 0.75},
                ).design
                before_query = evaluate_geometry(before)
                after_query = evaluate_geometry(after)

                def local_points(query):
                    origins = {element.index: element.origin_mm for element in query.elements}
                    return tuple(
                        tuple(round(value - origin, 10) for value, origin in zip(port.positive_point, origins[port.element]))
                        for port in query.ports
                    )

                self.assertEqual(local_points(before_query), local_points(after_query))
                self.assertNotEqual(
                    tuple(port.positive_point for port in before_query.ports),
                    tuple(port.positive_point for port in after_query.ports),
                )
                self.assertEqual(after.excitation.realization_status, "realized")

    def test_realized_array_resize_regenerates_ports_and_assignments(self):
        initial = self._realize(self._legacy(2, 3))
        one_by_four = self.agent.update_parameters(
            initial, {"array_rows": 1, "array_columns": 4},
        ).design
        self.assertEqual((one_by_four.array.rows, one_by_four.array.columns), (1, 4))
        self.assertEqual([port.port_id for port in one_by_four.ports], [f"port_{index}" for index in range(1, 5)])
        self.assertEqual(len(one_by_four.excitation.element_assignments), 4)
        self.assertEqual(len(set(self._segments(one_by_four))), 4)

        three_by_three = self.agent.update_parameters(
            initial, {"array_rows": 3, "array_columns": 3},
        ).design
        self.assertEqual(len(three_by_three.ports), 9)
        self.assertEqual(len(three_by_three.excitation.element_assignments), 9)
        self.assertEqual(len(set(self._segments(three_by_three))), 9)
        self.assertEqual(three_by_three.excitation.realization_status, "realized")

    def test_save_reopen_preserves_realized_mapping(self):
        design = self._realize(self._legacy(2, 3))
        root = Path.cwd() / ".test_runs"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            save_project_design(folder, design, [])
            restored, _messages = load_project_design(folder)
        self.assertEqual(restored.excitation, design.excitation)
        self.assertEqual(restored.ports, design.ports)
        self.assertEqual(evaluate_geometry(restored).geometry_hash, evaluate_geometry(design).geometry_hash)

    def test_cst_history_has_symbolic_local_feeds_and_distinct_discrete_ports(self):
        design = self._realize(self._legacy(2, 3))
        operations = CSTAdapter().history_operations(design)
        feed_operations = [
            item for item in operations
            if item.category == "primitive" and item.name.endswith("_feed")
        ]
        port_operations = [item for item in operations if item.category == "port"]
        self.assertEqual(len(feed_operations), 6)
        self.assertEqual(len(port_operations), 6)
        self.assertTrue(all("-BoardL/2" not in item.script for item in feed_operations))
        self.assertTrue(all("PatchL/2" in item.script for item in feed_operations))
        endpoint_rows = [
            tuple(row for row in operation.script.splitlines() if ".SetP" in row)
            for operation in port_operations
        ]
        self.assertEqual(len(endpoint_rows), len(set(endpoint_rows)))
        self.assertTrue(all("With DiscretePort" in item.script for item in port_operations))

    def test_unsupported_family_and_corporate_transition_never_claim_realization(self):
        circular = self.agent.create_design("circular_patch", {"array_columns": 2})
        circular_hash = evaluate_geometry(circular).geometry_hash
        unsupported = self._realize(circular)
        self.assertEqual(unsupported.excitation.strategy, "independent_ports")
        self.assertEqual(unsupported.excitation.realization_status, "unsupported")
        self.assertFalse(unsupported.excitation.excitation_realized)
        self.assertEqual(evaluate_geometry(unsupported).geometry_hash, circular_hash)
        self.assertFalse(any("independent_local_feed" in item.tags for item in unsupported.geometry))

        independent = self._realize(self._legacy(2, 3))
        corporate = self.agent.execute_llm_plan(
            independent,
            plan(("excitation.set_strategy", {"strategy": "corporate_feed"})),
        ).design
        self.assertEqual(corporate.excitation.realization_status, "unsupported")
        self.assertFalse(corporate.excitation.excitation_realized)
        self.assertFalse(any("independent_local_feed" in item.tags for item in corporate.geometry))
        self.assertEqual(
            evaluate_geometry(corporate).geometry_hash,
            evaluate_geometry(self._legacy(2, 3)).geometry_hash,
        )

    def test_synthesis_failure_is_transactional(self):
        baseline = self._legacy(2, 3)
        snapshot = baseline.to_dict()
        with patch(
            "studio.antenna_agent.realize_excitation",
            side_effect=DesignValidationError("injected synthesis failure"),
        ), self.assertRaisesRegex(DesignValidationError, "injected synthesis failure"):
            self._realize(baseline)
        self.assertEqual(baseline.to_dict(), snapshot)

    def test_legacy_2x3_changes_only_after_explicit_strategy_and_removes_known_findings(self):
        legacy = self._legacy(2, 3)
        legacy_hash = evaluate_geometry(legacy).geometry_hash
        legacy_warnings = self._warning_categories(legacy)
        self.assertEqual(legacy.excitation.strategy, "legacy_recipe")
        self.assertEqual(legacy_hash, "894aa73b151bf9d45d2a7a76ed04a1bd5ee415be36fcdaf5019617a112e2b612")
        self.assertEqual(legacy_warnings.count("conductor_contact"), 3)
        self.assertEqual(legacy_warnings.count("coincident_port_segments"), 3)
        self.assertEqual(legacy_warnings.count("port_element_association"), 6)
        self.assertEqual(legacy_warnings.count("excitation_shared_segment"), 3)

        independent = self._realize(legacy)
        independent_hash = evaluate_geometry(independent).geometry_hash
        self.assertEqual(independent_hash, "640cb10c9ebcff079faf9cd407cc0c3c9c451086a8277a0ccaf7d500329a990b")
        self.assertNotEqual(independent_hash, legacy_hash)
        self.assertEqual(self._warning_categories(independent), [])
        self.assertEqual(len(set(self._segments(legacy))), 3)
        self.assertEqual(len(set(self._segments(independent))), 6)


if __name__ == "__main__":
    unittest.main()
