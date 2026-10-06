import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from studio.antenna_agent import create_default_agent
from studio.antenna_analysis import semantic_design_hash
from studio.antenna_builder import load_project_design, save_project_design
from studio.antenna_design import DesignValidationError, ExcitationAssignment
from studio.antenna_engineering_checks import EXCITATION_CHECK, run_engineering_checks
from studio.antenna_geometry import evaluate_geometry
from studio.antenna_llm_planner import LLMToolPlan, PlannedToolCall
from studio.antenna_validation import validate_design
from studio.cst_antenna_adapter import CSTAdapter


def plan(*calls):
    return LLMToolPlan(
        "execute",
        "excitation lifecycle test",
        tuple(PlannedToolCall(name, arguments) for name, arguments in calls),
    )


class ExcitationLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.agent = create_default_agent()

    def _array(self, rows=2, columns=3):
        return self.agent.create_design(
            "inset_patch", {"array_rows": rows, "array_columns": columns},
        )

    def _strategy(self, design, strategy):
        return self.agent.execute_llm_plan(
            design,
            plan(("excitation.set_strategy", {"strategy": strategy})),
        ).design

    @staticmethod
    def _local_feeds(design):
        return tuple(
            item for item in design.geometry
            if "independent_local_feed" in item.tags
        )

    @staticmethod
    def _summary(design):
        report = run_engineering_checks(design, checks=[EXCITATION_CHECK])
        return report, next(
            item for item in report.findings
            if item.category == "excitation_collection_summary"
        )

    def test_transition_table_keeps_intent_realization_and_network_coherent(self):
        single = self.agent.create_design("inset_patch")
        legacy = self._array()
        independent = self._strategy(legacy, "independent_ports")
        corporate = self._strategy(independent, "corporate_feed")
        series = self._strategy(independent, "series_feed")
        custom = self._strategy(legacy, "custom")
        unresolved = self._strategy(legacy, "unresolved")
        cases = (
            (single, "single_element_feed", "realized", "none", 1, 1),
            (independent, "independent_ports", "realized", "independent", 6, 6),
            (corporate, "corporate_feed", "unsupported", "corporate", 6, 0),
            (series, "series_feed", "unsupported", "series", 6, 0),
            (custom, "custom", "unresolved", "custom", 0, 0),
            (unresolved, "unresolved", "unresolved", "unresolved", 0, 0),
            (legacy, "legacy_recipe", "legacy", "unresolved", 6, 0),
        )
        for design, strategy, status, network, assignment_count, realizing_count in cases:
            with self.subTest(strategy=strategy):
                self.assertEqual(design.excitation.strategy, strategy)
                self.assertEqual(design.excitation.realization_status, status)
                self.assertEqual(design.excitation.feed_network, network)
                self.assertEqual(len(design.excitation.element_assignments), assignment_count)
                self.assertEqual(len(design.excitation.realizing_port_ids), realizing_count)
                validate_design(replace(design, validation=()))

    def test_independent_to_corporate_to_independent_regenerates_without_stale_state(self):
        first = self._strategy(self._array(), "independent_ports")
        corporate = self._strategy(first, "corporate_feed")
        second = self._strategy(corporate, "independent_ports")
        self.assertEqual(corporate.excitation.realizing_port_ids, ())
        self.assertTrue(all(item.port_id is None for item in corporate.excitation.element_assignments))
        self.assertEqual(len(self._local_feeds(corporate)), 0)
        self.assertEqual(len(second.excitation.element_assignments), 6)
        self.assertEqual(len(set(second.excitation.realizing_port_ids)), 6)
        self.assertEqual(len(self._local_feeds(second)), 6)
        self.assertEqual(
            {item.port_id for item in second.excitation.element_assignments},
            {port.port_id for port in second.ports},
        )
        self.assertEqual(
            [item.category for item in run_engineering_checks(second).findings if item.severity == "warning"],
            [],
        )

    def test_independent_to_series_never_links_compatibility_ports(self):
        independent = self._strategy(self._array(), "independent_ports")
        series = self._strategy(independent, "series_feed")
        self.assertEqual(series.excitation.realization_status, "unsupported")
        self.assertEqual(series.excitation.realizing_port_ids, ())
        self.assertTrue(all(item.port_id is None for item in series.excitation.element_assignments))
        self.assertEqual(len(series.ports), 6)
        self.assertEqual(len(self._local_feeds(series)), 0)

    def test_unsupported_strategy_array_resize_refreshes_coordinates_only(self):
        corporate = self._strategy(self._array(1, 4), "corporate_feed")
        resized = self.agent.update_parameters(
            corporate, {"array_rows": 3, "array_columns": 2},
        ).design
        self.assertEqual(resized.excitation.strategy, "corporate_feed")
        self.assertEqual(resized.excitation.realization_status, "unsupported")
        self.assertEqual(
            [(item.element_row, item.element_column) for item in resized.excitation.element_assignments],
            [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)],
        )
        self.assertTrue(all(item.port_id is None for item in resized.excitation.element_assignments))
        self.assertEqual(len(self._local_feeds(resized)), 0)

    def test_realized_unsupported_and_unresolved_states_round_trip(self):
        states = (
            self._strategy(self._array(), "independent_ports"),
            self._strategy(self._array(), "corporate_feed"),
            self._strategy(self._array(), "unresolved"),
            self._strategy(self._array(), "custom"),
        )
        root = Path.cwd() / ".test_runs"
        root.mkdir(exist_ok=True)
        for state in states:
            with self.subTest(strategy=state.excitation.strategy):
                with tempfile.TemporaryDirectory(dir=root) as folder:
                    save_project_design(folder, state, [])
                    restored, _messages = load_project_design(folder)
                self.assertEqual(restored.excitation, state.excitation)
                self.assertEqual(restored.ports, state.ports)
                self.assertEqual(restored.geometry, state.geometry)
                self.assertEqual(semantic_design_hash(restored), semantic_design_hash(state))

    def test_unsupported_compatibility_ports_are_not_engineering_realization_evidence(self):
        corporate = self._strategy(self._array(), "corporate_feed")
        report, summary = self._summary(corporate)
        categories = {item.category for item in report.findings}
        self.assertIn("excitation_intent_unrealized", categories)
        self.assertEqual(summary.measured_values["canonical_port_count"].value, 6)
        self.assertEqual(summary.measured_values["intent_mapped_physical_port_count"].value, 0)
        self.assertEqual(summary.measured_values["current_realizing_port_count"].value, 0)
        self.assertEqual(summary.measured_values["other_physical_port_count"].value, 6)
        self.assertEqual(summary.measured_values["physical_segment_group_count"].value, 0)

    def test_missing_mapping_is_not_hidden_by_unrelated_physical_port_presence(self):
        independent = self._strategy(self._array(), "independent_ports")
        assignments = list(independent.excitation.element_assignments)
        assignments[0] = replace(assignments[0], port_id=None, status="unresolved")
        broken = replace(
            independent,
            excitation=replace(
                independent.excitation,
                realization_status="unresolved",
                element_assignments=tuple(assignments),
                unresolved_requirements=("One independent-port assignment is unresolved.",),
            ),
            validation=(),
        )
        report, summary = self._summary(broken)
        categories = {item.category for item in report.findings}
        self.assertIn("excitation_missing_element_port", categories)
        self.assertIn("excitation_intent_unrealized", categories)
        self.assertEqual(summary.measured_values["canonical_port_count"].value, 6)
        self.assertEqual(summary.measured_values["intent_mapped_physical_port_count"].value, 5)
        self.assertEqual(summary.measured_values["current_realizing_port_count"].value, 0)

    def test_unsupported_assignment_cannot_claim_a_physical_port(self):
        corporate = self._strategy(self._array(), "corporate_feed")
        assignments = list(corporate.excitation.element_assignments)
        assignments[0] = replace(assignments[0], port_id=corporate.ports[0].port_id)
        invalid = replace(
            corporate,
            excitation=replace(corporate.excitation, element_assignments=tuple(assignments)),
            validation=(),
        )
        with self.assertRaisesRegex(
            DesignValidationError,
            "Unsupported excitation intent cannot link physical ports",
        ):
            validate_design(invalid)

    def test_repeated_transitions_do_not_accumulate_synthesized_feeds(self):
        state = self._array()
        for _index in range(3):
            state = self._strategy(state, "independent_ports")
            self.assertEqual(len(self._local_feeds(state)), 6)
            self.assertEqual(len(state.ports), 6)
            state = self._strategy(state, "corporate_feed")
            self.assertEqual(len(self._local_feeds(state)), 0)
            self.assertEqual(len(state.ports), 6)

    def test_cst_export_keeps_realized_and_compatibility_ports_semantically_distinct(self):
        independent = self._strategy(self._array(), "independent_ports")
        corporate = self._strategy(independent, "corporate_feed")
        independent_ports = [
            item for item in CSTAdapter().history_operations(independent)
            if item.category == "port"
        ]
        compatibility_ports = [
            item for item in CSTAdapter().history_operations(corporate)
            if item.category == "port"
        ]
        self.assertEqual(len(independent_ports), len(independent.excitation.realizing_port_ids))
        self.assertEqual(len(compatibility_ports), len(corporate.ports))
        self.assertEqual(corporate.excitation.realizing_port_ids, ())
        exported_text = "\n".join(
            f"{item.name}\n{item.script}" for item in compatibility_ports
        ).casefold()
        self.assertNotIn("corporate", exported_text)
        self.assertNotIn("series", exported_text)

    def test_semantic_hash_changes_for_excitation_state_even_when_geometry_matches(self):
        legacy = self._array()
        corporate = self._strategy(legacy, "corporate_feed")
        series = self._strategy(corporate, "series_feed")
        self.assertEqual(
            evaluate_geometry(legacy).geometry_hash,
            evaluate_geometry(corporate).geometry_hash,
        )
        self.assertEqual(
            evaluate_geometry(corporate).geometry_hash,
            evaluate_geometry(series).geometry_hash,
        )
        self.assertEqual(len({
            semantic_design_hash(legacy),
            semantic_design_hash(corporate),
            semantic_design_hash(series),
        }), 3)

    def test_failed_transition_is_atomic(self):
        baseline = self._array()
        snapshot = baseline.to_dict()
        with patch(
            "studio.antenna_agent.realize_excitation",
            side_effect=DesignValidationError("injected lifecycle failure"),
        ), self.assertRaisesRegex(DesignValidationError, "injected lifecycle failure"):
            self._strategy(baseline, "independent_ports")
        self.assertEqual(baseline.to_dict(), snapshot)

    def test_manifest_explicitly_splits_realizing_and_other_physical_ports(self):
        independent = self._strategy(self._array(), "independent_ports")
        corporate = self._strategy(independent, "corporate_feed")
        independent_summary = self.agent.capability_manifest(independent)["excitation"]
        corporate_summary = self.agent.capability_manifest(corporate)["excitation"]
        self.assertEqual(len(independent_summary["realizing_port_ids"]), 6)
        self.assertEqual(independent_summary["other_physical_port_ids"], [])
        self.assertEqual(corporate_summary["requested_strategy"], "corporate_feed")
        self.assertEqual(corporate_summary["realization_status"], "unsupported")
        self.assertEqual(corporate_summary["realizing_port_ids"], [])
        self.assertEqual(len(corporate_summary["other_physical_port_ids"]), 6)
        self.assertEqual(corporate_summary["current_strategy_synthesis"], "not_installed")

    def test_single_element_and_legacy_behavior_remain_compatible(self):
        single = self.agent.create_design("inset_patch")
        legacy = self._array()
        self.assertEqual(single.excitation.strategy, "single_element_feed")
        self.assertEqual(single.excitation.realizing_port_ids, (single.ports[0].port_id,))
        self.assertEqual(legacy.excitation.strategy, "legacy_recipe")
        self.assertEqual(legacy.excitation.realizing_port_ids, ())
        resized = self.agent.update_parameters(
            self._strategy(self._array(1, 4), "corporate_feed"),
            {"array_rows": 1, "array_columns": 1},
        ).design
        self.assertEqual(resized.excitation.strategy, "corporate_feed")
        self.assertEqual(len(resized.excitation.element_assignments), 1)
        restored_single = self._strategy(resized, "single_element_feed")
        self.assertEqual(restored_single.excitation.realization_status, "realized")
        self.assertEqual(len(restored_single.excitation.realizing_port_ids), 1)


if __name__ == "__main__":
    unittest.main()
