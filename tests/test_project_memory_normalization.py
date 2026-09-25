import json
import shutil
import unittest
import uuid
from dataclasses import replace
from pathlib import Path

from studio.antenna_agent import create_default_agent
from studio.antenna_builder import (
    BuilderInteractionOutcome,
    BuilderProjectSession,
    ProjectMemory,
    ProjectMemoryItem,
    ProjectMemoryReducer,
    apply_semantic_memory_proposals,
    load_builder_session,
    save_builder_session,
)
from studio.antenna_design import ExcitationDefinition
from studio.antenna_llm_planner import PlannedToolCall, SemanticMemoryProposal
from studio.project_memory_normalization import SEMANTIC_NORMALIZATION_VERSION


def proposal(
    key,
    value,
    quote,
    *,
    kind="goal",
    action="upsert",
    unit=None,
    reference_id=None,
):
    return SemanticMemoryProposal(
        action=action,
        semantic_kind=kind,
        key=key,
        value=value,
        unit=unit,
        evidence_quote=quote,
        reference_id=reference_id,
    )


class ProjectMemoryNormalizationTests(unittest.TestCase):
    def apply(self, memory, item, instruction=None, *, design=None, turn_id="turn-1"):
        return apply_semantic_memory_proposals(
            memory,
            (item,),
            instruction=instruction or item.evidence_quote,
            planner_calls=(),
            design=design,
            turn_id=turn_id,
        )

    @staticmethod
    def semantic_items(memory):
        return tuple(
            item
            for collection in (memory.requirements, memory.decisions)
            for item in collection
            if item.source == "user_semantic"
        )

    def test_01_target_polarization_circular_is_canonical(self):
        memory, audit = self.apply(
            ProjectMemory.empty(),
            proposal("target_polarization", "circular", "I want circular polarization."),
        )
        item = self.semantic_items(memory)[0]
        self.assertEqual((item.key, item.value), ("target_polarization", "circular_polarization"))
        self.assertEqual(audit["rejected"], [])

    def test_02_desired_polarization_has_same_identity(self):
        memory, _audit = self.apply(
            ProjectMemory.empty(),
            proposal("desired_polarization", "circular", "I want circular polarization."),
        )
        self.assertEqual(self.semantic_items(memory)[0].key, "target_polarization")

    def test_03_rhcp_remains_distinct_from_generic_circular(self):
        generic, _ = self.apply(
            ProjectMemory.empty(),
            proposal("target_polarization", "circular", "I want circular polarization."),
        )
        rhcp, _ = self.apply(
            ProjectMemory.empty(),
            proposal("target_polarization", "RHCP", "I want RHCP."),
        )
        self.assertEqual(self.semantic_items(generic)[0].value, "circular_polarization")
        self.assertEqual(self.semantic_items(rhcp)[0].value, "rhcp")

    def test_04_lhcp_remains_distinct_from_rhcp(self):
        rhcp, _ = self.apply(
            ProjectMemory.empty(), proposal("target_polarization", "RHCP", "Use RHCP."),
        )
        lhcp, _ = self.apply(
            ProjectMemory.empty(), proposal("target_polarization", "LHCP", "Use LHCP."),
        )
        self.assertNotEqual(self.semantic_items(rhcp)[0].value, self.semantic_items(lhcp)[0].value)

    def test_05_board_width_unit_suffix_and_operator_are_canonical(self):
        memory, _ = self.apply(
            ProjectMemory.empty(),
            proposal(
                "max_board_width_mm", 90, "Keep the board under 90 mm.",
                kind="constraint", unit="millimeters",
            ),
        )
        item = self.semantic_items(memory)[0]
        self.assertEqual((item.key, item.value, item.unit), ("board_width_limit", 90.0, "mm"))
        self.assertEqual(item.constraint_operator, "max")

    def test_06_board_width_max_has_same_identity(self):
        memory, _ = self.apply(
            ProjectMemory.empty(),
            proposal(
                "board_width_max", 90, "Keep the board under 90 mm.",
                kind="constraint", unit="mm",
            ),
        )
        self.assertEqual(self.semantic_items(memory)[0].key, "board_width_limit")

    def test_07_later_board_alias_supersedes_existing_identity(self):
        first, _ = self.apply(
            ProjectMemory.empty(),
            proposal(
                "max_board_width_mm", 80, "Keep the board under 80 mm.",
                kind="constraint", unit="mm",
            ),
            turn_id="turn-1",
        )
        original = self.semantic_items(first)[0]
        second, audit = self.apply(
            first,
            proposal(
                "board_width_max", 100, "Up to 100 mm is acceptable.",
                kind="constraint", action="supersede", unit="mm",
                reference_id=original.item_id,
            ),
            turn_id="turn-2",
        )
        items = self.semantic_items(second)
        self.assertEqual([item.status for item in items], ["superseded", "active"])
        self.assertEqual(items[1].key, "board_width_limit")
        self.assertEqual(audit["superseded_ids"], [original.item_id])

    def test_08_future_feed_corporate_is_canonical(self):
        memory, _ = self.apply(
            ProjectMemory.empty(),
            proposal(
                "future_feed_network", "corporate", "Use a corporate feed later.",
                kind="future_intent",
            ),
        )
        item = self.semantic_items(memory)[0]
        self.assertEqual((item.key, item.value), ("feed_network_future_goal", "corporate_feed"))

    def test_09_explicit_future_corporate_distribution_is_canonical(self):
        memory, _ = self.apply(
            ProjectMemory.empty(),
            proposal(
                "corporate_distribution_network",
                "corporate distribution network",
                "A corporate distribution network is a later goal.",
                kind="future_intent",
            ),
        )
        item = self.semantic_items(memory)[0]
        self.assertEqual((item.key, item.value), ("feed_network_future_goal", "corporate_feed"))

    def test_10_realized_excitation_is_not_duplicated_as_future_intent(self):
        design = create_default_agent().create_design("inset_patch")
        design = replace(
            design,
            excitation=ExcitationDefinition(
                strategy="independent_ports",
                realization_status="realized",
                feed_network="independent",
            ),
        )
        memory, audit = self.apply(
            ProjectMemory.empty(),
            proposal(
                "future_feed_network", "independent", "Use independent ports later.",
                kind="future_intent",
            ),
            design=design,
        )
        self.assertEqual(self.semantic_items(memory), ())
        self.assertEqual(audit["rejected"][0]["reason"], "duplicates_current_canonical_state")

    def test_11_unknown_key_is_preserved_with_safe_generic_normalization(self):
        memory, _ = self.apply(
            ProjectMemory.empty(),
            proposal(
                "prototype_length_mm", 25, "Prototype length is 25 mm.",
                kind="constraint", unit="mm",
            ),
        )
        item = self.semantic_items(memory)[0]
        self.assertEqual((item.key, item.value, item.unit), ("prototype_length", 25, "mm"))

    def test_12_similarly_worded_distinct_keys_do_not_collapse(self):
        first, _ = self.apply(
            ProjectMemory.empty(),
            proposal(
                "board_width_target", 70, "Target board width is 70 mm.",
                kind="constraint", unit="mm",
            ),
            turn_id="turn-1",
        )
        second, _ = self.apply(
            first,
            proposal(
                "board_width_limit", 90, "Maximum board width is 90 mm.",
                kind="constraint", unit="mm",
            ),
            turn_id="turn-2",
        )
        active = {item.key for item in self.semantic_items(second) if item.status == "active"}
        self.assertEqual(active, {"board_width_target", "board_width_limit"})

    def test_13_invalid_evidence_is_rejected_before_normalization(self):
        memory, audit = self.apply(
            ProjectMemory.empty(),
            proposal("desired_polarization", "circular", "This quote is absent."),
            instruction="I want circular polarization.",
        )
        self.assertEqual(self.semantic_items(memory), ())
        self.assertEqual(audit["normalized"], [])
        self.assertEqual(audit["rejected"][0]["reason"], "evidence_quote_not_exact_current_turn_substring")

    def test_14_normalization_cannot_bypass_lexical_grounding(self):
        memory, audit = self.apply(
            ProjectMemory.empty(),
            proposal("desired_polarization", "circular", "Polarization matters."),
        )
        self.assertEqual(self.semantic_items(memory), ())
        self.assertEqual(audit["normalized"], [])
        self.assertEqual(audit["rejected"][0]["reason"], "value_not_lexically_grounded_in_evidence_quote")

    def test_15_original_proposal_and_normalization_are_auditable(self):
        memory, audit = self.apply(
            ProjectMemory.empty(),
            proposal("desired_polarization", "circular", "I want circular polarization."),
        )
        item = self.semantic_items(memory)[0]
        self.assertEqual((item.proposed_key, item.proposed_value), ("desired_polarization", "circular"))
        self.assertEqual(item.normalization_version, SEMANTIC_NORMALIZATION_VERSION)
        self.assertEqual(item.normalization_rule, "registered:polarization_target")
        self.assertEqual(audit["normalized"][0]["semantic_key"], "target_polarization")

    def test_16_save_reopen_preserves_identity_and_provenance(self):
        memory, _ = self.apply(
            ProjectMemory.empty(),
            proposal("desired_polarization", "circular", "I want circular polarization."),
        )
        directory = Path.cwd() / f".stage5d_persistence_{uuid.uuid4().hex}"
        directory.mkdir()
        try:
            session = BuilderProjectSession(memory=memory)
            save_builder_session(directory, session)
            restored = load_builder_session(directory)
        finally:
            shutil.rmtree(directory)
        self.assertEqual(restored.memory, memory)

    def test_17_supersede_and_resolve_work_through_alias_identity(self):
        first, _ = self.apply(
            ProjectMemory.empty(),
            proposal("desired_polarization", "circular", "I want circular polarization."),
            turn_id="turn-1",
        )
        original = self.semantic_items(first)[0]
        second, _ = self.apply(
            first,
            proposal(
                "polarization_target", "RHCP", "Make RHCP the target.",
                action="supersede", reference_id=original.item_id,
            ),
            turn_id="turn-2",
        )
        active = next(item for item in self.semantic_items(second) if item.status == "active")
        third, _ = self.apply(
            second,
            proposal(
                "desired_polarization", None, "Remove that polarization goal.",
                action="resolve", reference_id=active.item_id,
            ),
            turn_id="turn-3",
        )
        self.assertFalse(any(item.status == "active" for item in self.semantic_items(third)))

    def test_18_legacy_semantic_item_without_metadata_loads(self):
        payload = ProjectMemory.empty().to_dict()
        payload["requirements"] = [{
            "item_id": "semantic:desired_polarization:legacy:value",
            "key": "desired_polarization",
            "value": "circular",
            "unit": None,
            "status": "active",
            "source_turn_id": "legacy",
            "design_revision": 0,
            "source": "user_semantic",
            "semantic_kind": "goal",
            "evidence_quote": "I want circular polarization.",
            "supersedes_item_id": None,
            "created_design_revision": 0,
            "updated_design_revision": 0,
        }]
        item = ProjectMemory.from_dict(payload).requirements[0]
        self.assertIsNone(item.proposed_key)
        self.assertIsNone(item.normalization_version)

    def test_19_planner_context_deduplicates_legacy_aliases(self):
        items = (
            ProjectMemoryItem(
                "semantic:desired_polarization:one:value", "desired_polarization", "circular",
                source="user_semantic", semantic_kind="goal", evidence_quote="circular",
            ),
            ProjectMemoryItem(
                "semantic:target_polarization:two:value", "target_polarization", "circular_polarization",
                source="user_semantic", semantic_kind="goal", evidence_quote="circular polarization",
            ),
        )
        intent = ProjectMemory(requirements=items).to_planner_dict()["project_intent"]
        self.assertEqual(len(intent), 1)
        self.assertEqual(intent[0]["key"], "target_polarization")
        self.assertEqual(intent[0]["value"], "circular_polarization")

    def test_20_physical_state_reducer_behavior_is_unchanged(self):
        design = create_default_agent().create_design("inset_patch")
        memory = ProjectMemoryReducer().reduce(
            ProjectMemory.empty(),
            None,
            BuilderInteractionOutcome(
                status="executed",
                message="Created.",
                instruction="Create an inset patch at 2.45 GHz.",
                planner_calls=(
                    PlannedToolCall("recipe.select", {"recipe_id": "inset_patch_v2"}),
                    PlannedToolCall("parameter.set", {"key": "frequency_ghz", "value": 2.45}),
                ),
            ),
            design,
            "turn-physical",
        )
        requirement = next(item for item in memory.requirements if item.key == "target_frequency")
        self.assertEqual(requirement.source, "deterministic")
        self.assertIsNone(requirement.normalization_version)
        self.assertEqual(memory.canonical_ref.revision, design.revision)

    def test_recorded_g01_g02_g03_proposals_replay_to_canonical_intent(self):
        root = (
            Path(__file__).resolve().parents[1]
            / "benchmarks" / "results" / "v1"
            / "20260924_stage5b_hosted_v1" / "gemini"
        )
        results = {}
        for case_id in ("G01", "G02", "G03"):
            memory = ProjectMemory.empty()
            for line in (root / case_id / "trajectory.jsonl").read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                proposed = record.get("semantic_memory", {}).get("proposed", [])
                if not proposed:
                    continue
                proposals = tuple(SemanticMemoryProposal.from_unvalidated(item) for item in proposed)
                memory, _audit = apply_semantic_memory_proposals(
                    memory,
                    proposals,
                    instruction=record["user_request"],
                    planner_calls=(),
                    design=None,
                    turn_id=record["turn_id"],
                )
            results[case_id] = {
                item.key: (item.value, item.unit, item.constraint_operator)
                for item in self.semantic_items(memory)
                if item.status == "active"
            }
        self.assertEqual(
            results["G01"]["target_polarization"],
            ("circular_polarization", None, None),
        )
        self.assertEqual(results["G02"]["board_width_limit"], (90.0, "mm", "max"))
        self.assertEqual(
            results["G03"]["feed_network_future_goal"],
            ("corporate_feed", None, None),
        )

    def test_stage5e_polarization_alias_is_narrow_and_canonical(self):
        memory, audit = self.apply(
            ProjectMemory.empty(),
            proposal("polarization", "LHCP", "LHCP would be preferable.", kind="preference"),
        )
        item = self.semantic_items(memory)[0]
        self.assertEqual((item.key, item.value), ("target_polarization", "lhcp"))
        self.assertEqual(audit["rejected"], [])

    def test_stage5e_series_network_alias_requires_future_intent(self):
        text = "A series-fed network is something I may explore later."
        memory, _ = self.apply(
            ProjectMemory.empty(),
            proposal("series_fed_network", "explore later", text, kind="future_intent"),
        )
        self.assertEqual(
            (self.semantic_items(memory)[0].key, self.semantic_items(memory)[0].value),
            ("feed_network_future_goal", "series_feed"),
        )
        current, _ = self.apply(
            ProjectMemory.empty(),
            proposal("series_fed_network", "series-fed network", text, kind="goal"),
        )
        self.assertEqual(self.semantic_items(current)[0].key, "series_fed_network")

    def test_stage5e_board_dimension_alias_requires_board_maximum_evidence(self):
        text = "Keep the board under 85 mm."
        memory, _ = self.apply(
            ProjectMemory.empty(),
            proposal(
                "max_board_dimension_mm", 85, text,
                kind="constraint", unit="mm",
            ),
        )
        item = self.semantic_items(memory)[0]
        self.assertEqual(
            (item.key, item.value, item.unit, item.constraint_operator),
            ("board_width_limit", 85.0, "mm", "max"),
        )
        neutral, _ = self.apply(
            ProjectMemory.empty(),
            proposal(
                "max_board_dimension_mm", 85, "The board dimension is 85 mm.",
                kind="constraint", unit="mm",
            ),
        )
        self.assertEqual(self.semantic_items(neutral)[0].key, "max_board_dimension")
        substrate, _ = self.apply(
            ProjectMemory.empty(),
            proposal(
                "substrate_width_mm", 80, "The substrate width can be 80 mm.",
                kind="constraint", unit="mm",
            ),
        )
        self.assertEqual(self.semantic_items(substrate)[0].key, "substrate_width")

    def test_stage5e_corporate_feed_alias_requires_future_intent(self):
        text = "Corporate feed can come later."
        memory, _ = self.apply(
            ProjectMemory.empty(),
            proposal("corporate_feed", "later", text, kind="future_intent"),
        )
        item = self.semantic_items(memory)[0]
        self.assertEqual((item.key, item.value), ("feed_network_future_goal", "corporate_feed"))

    def test_boolean_semantic_placeholder_is_rejected_before_normalization(self):
        memory, audit = self.apply(
            ProjectMemory.empty(),
            proposal(
                "target_polarization", True,
                "RHCP is the eventual target.", kind="future_intent",
            ),
        )
        self.assertEqual(self.semantic_items(memory), ())
        self.assertEqual(audit["normalized"], [])
        self.assertEqual(audit["rejected"][0]["reason"], "boolean_placeholder_not_allowed")


if __name__ == "__main__":
    unittest.main()
