import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from studio.antenna_agent import create_default_agent
from studio.antenna_agent_runner import AntennaAgentRunner
from studio.antenna_analysis import semantic_design_hash
from studio.antenna_builder import load_project_design, save_project_design
from studio.antenna_design import (
    AntennaDesign,
    DesignValidationError,
    ExcitationAssignment,
    ExcitationDefinition,
)
from studio.antenna_engineering_checks import EXCITATION_CHECK, run_engineering_checks
from studio.antenna_geometry import evaluate_geometry
from studio.antenna_llm_planner import (
    AgentLoopBudgets,
    AgentStep,
    LLMToolPlan,
    PlannedToolCall,
    build_agent_step_exchange,
)
from studio.antenna_validation import validate_design


def plan(*calls):
    return LLMToolPlan(
        "execute",
        "excitation test",
        tuple(PlannedToolCall(name, arguments) for name, arguments in calls),
    )


class ExcitationDefinitionTests(unittest.TestCase):
    def setUp(self):
        self.agent = create_default_agent()

    def _array(self, rows=2, columns=3):
        design = self.agent.create_design("inset_patch")
        return self.agent.execute_llm_plan(
            design,
            plan(
                ("parameter.set", {"key": "array_rows", "value": rows}),
                ("parameter.set", {"key": "array_columns", "value": columns}),
            ),
        ).design

    def _strategy(self, design, strategy):
        return self.agent.execute_llm_plan(
            design,
            plan(("excitation.set_strategy", {"strategy": strategy})),
        ).design

    @staticmethod
    def _physical_payload(design):
        payload = design.to_dict()
        return {
            key: payload[key]
            for key in ("parameters", "materials", "geometry", "booleans", "ports", "array", "simulation")
        }

    def test_new_single_element_recipes_have_explicit_realized_excitation(self):
        for recipe in ("inset_patch", "circular_patch", "dipole"):
            with self.subTest(recipe=recipe):
                design = self.agent.create_design(recipe)
                self.assertEqual(design.excitation.strategy, "single_element_feed")
                self.assertEqual(design.excitation.realization_status, "realized")
                self.assertTrue(design.excitation.excitation_realized)
                self.assertEqual(design.excitation.element_assignments[0].port_id, design.ports[0].port_id)

    def test_legacy_single_and_array_migrate_conservatively_without_geometry_change(self):
        single = self.agent.create_design("inset_patch")
        single_payload = single.to_dict()
        single_payload.pop("excitation")
        restored_single = AntennaDesign.from_dict(single_payload)
        self.assertEqual(restored_single.excitation.strategy, "single_element_feed")
        self.assertEqual(self._physical_payload(restored_single), self._physical_payload(single))

        array = self.agent.create_design("inset_patch", {"array_rows": 2, "array_columns": 3})
        array_payload = array.to_dict()
        array_payload.pop("excitation")
        restored_array = AntennaDesign.from_dict(array_payload)
        self.assertEqual(restored_array.excitation.strategy, "legacy_recipe")
        self.assertEqual(restored_array.excitation.realization_status, "legacy")
        self.assertNotIn(restored_array.excitation.strategy, {"independent_ports", "corporate_feed"})
        self.assertEqual(self._physical_payload(restored_array), self._physical_payload(array))

    def test_excitation_round_trip_and_semantic_hash_with_realized_geometry_change(self):
        baseline = self._array()
        before_geometry = evaluate_geometry(baseline).geometry_hash
        independent = self._strategy(baseline, "independent_ports")
        restored = AntennaDesign.from_dict(independent.to_dict())
        self.assertEqual(restored.excitation, independent.excitation)
        self.assertNotEqual(semantic_design_hash(baseline), semantic_design_hash(independent))
        self.assertNotEqual(evaluate_geometry(independent).geometry_hash, before_geometry)
        self.assertNotEqual(self._physical_payload(independent), self._physical_payload(baseline))

    def test_project_save_reopen_preserves_excitation(self):
        state = self._strategy(self._array(), "independent_ports")
        root = Path.cwd() / ".test_runs"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            save_project_design(folder, state, [{"role": "user", "content": "Use independent excitation."}])
            restored, messages = load_project_design(folder)
        self.assertEqual(restored.excitation, state.excitation)
        self.assertEqual(messages[0]["content"], "Use independent excitation.")

    def test_independent_realizes_locally_and_corporate_remains_unsupported(self):
        baseline = self._array()
        physical = self._physical_payload(baseline)
        independent = self._strategy(baseline, "independent_ports")
        self.assertEqual(independent.excitation.realization_status, "realized")
        self.assertEqual(len(independent.excitation.element_assignments), 6)
        self.assertTrue(all(item.port_id is not None for item in independent.excitation.element_assignments))
        self.assertNotEqual(self._physical_payload(independent), physical)

        corporate = self._strategy(independent, "corporate_feed")
        self.assertEqual(corporate.excitation.realization_status, "unsupported")
        self.assertEqual(corporate.excitation.feed_network, "corporate")
        self.assertFalse(corporate.excitation.excitation_realized)
        self.assertEqual(self._physical_payload(corporate), physical)

    def test_assignment_coordinate_port_and_duplicate_validation(self):
        design = self._array()
        cases = (
            replace(design, excitation=ExcitationDefinition(
                "independent_ports", "unresolved", "independent",
                (ExcitationAssignment(3, 1, "outside"),),
            )),
            replace(design, excitation=ExcitationDefinition(
                "independent_ports", "unresolved", "independent",
                (ExcitationAssignment(1, 1, "a"), ExcitationAssignment(1, 1, "b")),
            )),
            replace(design, excitation=ExcitationDefinition(
                "independent_ports", "unresolved", "independent",
                (ExcitationAssignment(1, 1, "missing", "no_such_port"),),
            )),
        )
        for invalid in cases:
            with self.subTest(excitation=invalid.excitation), self.assertRaises(DesignValidationError):
                validate_design(replace(invalid, validation=()))
        self.assertEqual(design.excitation, self._array().excitation)

    def test_array_resize_regenerates_explicit_assignments_without_stale_coordinates(self):
        independent = self._strategy(self._array(), "independent_ports")
        shrunk = self.agent.execute_llm_plan(
            independent,
            plan(
                ("parameter.set", {"key": "array_rows", "value": 1}),
                ("parameter.set", {"key": "array_columns", "value": 2}),
            ),
        ).design
        self.assertEqual(
            [(item.element_row, item.element_column) for item in shrunk.excitation.element_assignments],
            [(1, 1), (1, 2)],
        )
        expanded = self.agent.execute_llm_plan(
            shrunk,
            plan(("parameter.set", {"key": "array_columns", "value": 4})),
        ).design
        self.assertEqual(len(expanded.excitation.element_assignments), 4)
        self.assertTrue(all(item.port_id is not None for item in expanded.excitation.element_assignments))
        self.assertEqual(len(expanded.ports), 4)  # unchanged recipe compatibility geometry, not synthesized linkage

    def test_strategy_action_is_transactional_and_increments_revision(self):
        design = self._array()
        before = design.to_dict()
        result = self.agent.execute_llm_plan(
            design,
            plan(("excitation.set_strategy", {"strategy": "independent_ports"})),
        )
        self.assertEqual(result.design.revision, design.revision + 1)
        self.assertEqual(result.design.excitation.strategy, "independent_ports")
        self.assertEqual(design.to_dict(), before)
        with self.assertRaises(Exception):
            self.agent.execute_llm_plan(
                design,
                plan(("excitation.set_strategy", {"strategy": "invented"})),
            )
        self.assertEqual(design.to_dict(), before)

    def test_provisional_strategy_change_rolls_back_on_clarify_or_refuse(self):
        baseline = self._array()
        execute = AgentStep(
            "execute",
            "Record intent.",
            (PlannedToolCall("excitation.set_strategy", {"strategy": "independent_ports"}),),
        )

        class Planner:
            last_run_metadata = {}

            def __init__(self, terminal):
                self.responses = [execute, AgentStep(terminal, f"{terminal} after provisional intent")]

            def plan_agent_step(self, **_kwargs):
                return self.responses.pop(0)

        for status in ("clarify", "refuse"):
            with self.subTest(status=status):
                result = AntennaAgentRunner(self.agent).run(
                    baseline_design=baseline,
                    instruction="Change excitation and then stop.",
                    project_memory={"schema_version": 1},
                    planner=Planner(status),
                )
                self.assertEqual(result.outcome, status)
                self.assertIsNone(result.final_design)
                self.assertEqual(result.working_design_hash, semantic_design_hash(baseline))

    def test_problematic_2x3_geometry_fingerprint_changes_only_after_realization(self):
        baseline = self._array()
        before = evaluate_geometry(baseline)
        intended = self._strategy(baseline, "independent_ports")
        after = evaluate_geometry(intended)
        self.assertNotEqual(before.geometry_hash, after.geometry_hash)
        self.assertNotEqual(before.objects, after.objects)
        self.assertNotEqual(before.ports, after.ports)
        self.assertEqual(before.booleans, after.booleans)
        self.assertNotEqual(before.semantic_design_hash, after.semantic_design_hash)

    def test_excitation_consistency_uses_explicit_mapping_and_legacy_remains_compatible(self):
        baseline = self.agent.create_design(
            "inset_patch", {"array_rows": 2, "array_columns": 3},
        )
        self.assertEqual(baseline.excitation.strategy, "legacy_recipe")
        legacy = run_engineering_checks(baseline, checks=[EXCITATION_CHECK])
        explicit = run_engineering_checks(
            self._strategy(baseline, "independent_ports"), checks=[EXCITATION_CHECK],
        )
        legacy_categories = {item.category for item in legacy.findings}
        explicit_categories = {item.category for item in explicit.findings}
        self.assertIn("excitation_shared_segment", legacy_categories)
        self.assertNotIn("excitation_shared_segment", explicit_categories)
        self.assertNotIn("excitation_intent_unrealized", legacy_categories)
        self.assertNotIn("excitation_intent_unrealized", explicit_categories)
        summary = next(item for item in explicit.findings if item.category == "excitation_collection_summary")
        self.assertEqual(summary.measured_values["intended_assignment_count"].value, 6)
        self.assertEqual(summary.measured_values["intent_mapped_physical_port_count"].value, 6)

    def test_manifest_and_provider_neutral_exchange_expose_one_identical_schema(self):
        design = self._array()
        manifest = self.agent.capability_manifest(design)
        summary = manifest["excitation"]
        self.assertNotEqual(summary["current_physical_port_ids"], [])
        self.assertEqual(summary["strategy"], design.excitation.strategy)
        action = next(item for item in manifest["callable_tools"] if item["name"] == "excitation.set_strategy")
        self.assertIn("corporate_feed", action["arguments"]["properties"]["strategy"]["enum"])
        independent = next(
            item for item in summary["available_strategy_capabilities"]
            if item["strategy"] == "independent_ports"
        )
        self.assertEqual(independent["physical_synthesis"], "supported_and_realizable")
        common = dict(
            instruction="Use independent excitation.",
            current_design=design.to_dict(),
            capability_manifest=manifest,
            remaining_budgets=AgentLoopBudgets(),
        )
        gemini_exchange = build_agent_step_exchange(**common)
        nemotron_exchange = build_agent_step_exchange(**common)
        self.assertEqual(gemini_exchange.schema, nemotron_exchange.schema)
        self.assertEqual(gemini_exchange.user_content, nemotron_exchange.user_content)
        self.assertEqual(gemini_exchange.system_instruction, nemotron_exchange.system_instruction)


if __name__ == "__main__":
    unittest.main()
