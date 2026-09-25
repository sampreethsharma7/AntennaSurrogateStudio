import json
import tempfile
import unittest
from pathlib import Path

from studio.antenna_builder import (
    BuilderProjectSession,
    ProjectMemory,
    apply_structured_state_patch,
    execute_builder_turn,
    load_builder_session,
    publish_builder_state_update,
    save_project_design,
)
from studio.antenna_design import AntennaDesign
from studio.antenna_llm_planner import LLMToolPlan, PlannedToolCall
from studio.antenna_tools import CapabilityError


def _plan(*calls, status="execute", message="test plan"):
    return LLMToolPlan(
        status,
        message,
        tuple(PlannedToolCall(name, arguments) for name, arguments in calls),
    )


def _initial_patch_plan():
    return _plan(
        ("recipe.select", {"recipe_id": "inset_patch_v2"}),
        ("parameter.set", {"key": "frequency_ghz", "value": 2.45}),
        ("parameter.set", {"key": "material", "value": "FR4"}),
    )


def _center_slot_plan(radius=3.0):
    return _plan(
        (
            "parameter.create",
            {
                "key": "slot_radius_mm",
                "label": "Slot radius",
                "value": radius,
                "unit": "mm",
                "sweepable": True,
            },
        ),
        (
            "geometry.cylinder",
            {
                "object_id": "center_slot_tool",
                "material_id": "copper",
                "axis": "z",
                "tags": ["planner_created", "boolean_tool", "slot"],
                "dimensions": {
                    "center_1": 0,
                    "center_2": 0,
                    "radius": "slot_radius_mm",
                    "start": "substrate_thickness_mm",
                    "end": "substrate_thickness_mm+copper_thickness_mm",
                },
            },
        ),
        (
            "boolean.subtract",
            {
                "operation_id": "center_slot_subtract",
                "target_id": "element_1_1_patch",
                "tool_ids": ["center_slot_tool"],
            },
        ),
    )


class _RecordingPlanner:
    def __init__(self, result):
        self.result = result
        self.request = None

    def plan(self, **request):
        self.request = request
        return self.result


class ProjectMemoryReducerTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        self.project_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_first_design_records_explicit_engineering_facts_and_revision_zero(self):
        session = BuilderProjectSession()

        result = execute_builder_turn(
            self.project_path,
            session,
            "Design an inset-fed rectangular patch at 2.45 GHz on FR4.",
            planner=_RecordingPlanner(_initial_patch_plan()),
            turn_id="turn-1",
        )

        memory = result.session.memory
        self.assertEqual(memory.canonical_ref.design_id, result.update.state.design_id)
        self.assertEqual(memory.canonical_ref.revision, 0)
        self.assertEqual(memory.requirements[0].key, "target_frequency")
        self.assertEqual(memory.requirements[0].value, 2.45)
        self.assertEqual(memory.requirements[0].unit, "GHz")
        self.assertEqual(
            {item.key for item in memory.decisions},
            {"antenna_family", "substrate_material"},
        )
        self.assertEqual(memory.assumptions, ())
        self.assertEqual(len(session.conversation), 2)
        self.assertEqual(session.conversation[0]["outcome"], "request")
        self.assertEqual(session.conversation[1]["outcome"], "executed")
        self.assertEqual(session.conversation[1]["design_revision"], 0)

    def test_recipe_defaults_are_labeled_as_assumptions_when_not_explicit(self):
        session = BuilderProjectSession()

        execute_builder_turn(
            self.project_path,
            session,
            "Create the supported inset patch recipe.",
            planner=_RecordingPlanner(_plan(
                ("recipe.select", {"recipe_id": "inset_patch_v2"}),
            )),
            turn_id="turn-defaults",
        )

        assumptions = {item.key: item for item in session.memory.assumptions}
        self.assertEqual(assumptions["default_frequency"].value, 2.45)
        self.assertEqual(assumptions["default_frequency"].unit, "GHz")
        self.assertEqual(assumptions["default_substrate_material"].value, "FR4")
        self.assertEqual(session.memory.requirements, ())

    def test_center_slot_is_reduced_after_publication_and_next_planner_gets_memory(self):
        session = BuilderProjectSession()
        execute_builder_turn(
            self.project_path,
            session,
            "Design an inset-fed rectangular patch at 2.45 GHz on FR4.",
            planner=_RecordingPlanner(_initial_patch_plan()),
            turn_id="turn-1",
        )
        slot_planner = _RecordingPlanner(_center_slot_plan())

        execute_builder_turn(
            self.project_path,
            session,
            "Add a 3 mm circular slot at the center.",
            planner=slot_planner,
            turn_id="turn-2",
        )

        supplied_memory = slot_planner.request["project_memory"]
        self.assertEqual(supplied_memory["canonical_ref"]["revision"], 0)
        self.assertNotIn("geometry", supplied_memory)
        self.assertNotIn("parameters", supplied_memory)
        self.assertEqual(session.memory.canonical_ref.revision, 1)
        change = session.memory.important_changes[-1].value
        self.assertEqual(
            change["created_parameters"],
            [{"key": "slot_radius_mm", "value": 3.0, "unit": "mm"}],
        )
        added = change["composed_features"]["added"]
        self.assertEqual(len(added), 1)
        self.assertIn("boolean.subtract", added[0]["operations"])

        next_planner = _RecordingPlanner(_plan(
            status="clarify",
            message="Which array layout should I use?",
        ))
        clarification = execute_builder_turn(
            self.project_path,
            session,
            "Continue the design.",
            planner=next_planner,
            turn_id="turn-3",
        )
        self.assertEqual(clarification.terminal_result.outcome, "clarify")
        self.assertEqual(
            next_planner.request["project_memory"]["canonical_ref"]["revision"],
            1,
        )
        self.assertTrue(next_planner.request["project_memory"]["important_changes"])

    def test_refusal_persists_limitation_without_advancing_canonical_reference(self):
        session = BuilderProjectSession()
        execute_builder_turn(
            self.project_path,
            session,
            "Design an inset-fed rectangular patch at 2.45 GHz on FR4.",
            planner=_RecordingPlanner(_initial_patch_plan()),
            turn_id="turn-1",
        )
        before_ref = session.memory.canonical_ref

        refusal = execute_builder_turn(
            self.project_path,
            session,
            "Turn this into a horn antenna.",
            planner=_RecordingPlanner(_plan(
                status="refuse",
                message="A horn antenna recipe is not installed.",
            )),
            turn_id="turn-2",
        )
        self.assertEqual(refusal.terminal_result.outcome, "refuse")

        self.assertEqual(session.memory.canonical_ref, before_ref)
        self.assertEqual(session.memory.limitations[-1].key, "unsupported_capability")
        self.assertEqual(session.conversation[-1]["outcome"], "refusal")
        self.assertEqual(session.conversation[-1]["design_revision"], 0)

    def test_failed_transaction_records_rejection_but_no_successful_design_change(self):
        session = BuilderProjectSession()
        execute_builder_turn(
            self.project_path,
            session,
            "Design an inset-fed rectangular patch at 2.45 GHz on FR4.",
            planner=_RecordingPlanner(_initial_patch_plan()),
            turn_id="turn-1",
        )
        before_design = session.design
        before_ref = session.memory.canonical_ref
        before_changes = session.memory.important_changes

        with self.assertRaises(CapabilityError):
            execute_builder_turn(
                self.project_path,
                session,
                "Set an invalid inset.",
                planner=_RecordingPlanner(_plan(
                    ("parameter.set", {"key": "inset_depth_mm", "value": 999.0}),
                )),
                turn_id="turn-2",
            )

        self.assertEqual(session.design, before_design)
        self.assertEqual(session.memory.canonical_ref, before_ref)
        self.assertEqual(session.memory.important_changes, before_changes)
        self.assertEqual(session.conversation[-1]["outcome"], "validation_rejection")
        self.assertEqual(session.memory.recent_context[-1].value["outcome"], "validation_rejection")

    def test_save_reopen_restores_design_conversation_memory_and_open_items(self):
        session = BuilderProjectSession()
        execute_builder_turn(
            self.project_path,
            session,
            "Design an inset-fed rectangular patch at 2.45 GHz on FR4.",
            planner=_RecordingPlanner(_initial_patch_plan()),
            turn_id="turn-1",
        )
        refusal = execute_builder_turn(
            self.project_path,
            session,
            "Build a horn.",
            planner=_RecordingPlanner(_plan(
                status="refuse",
                message="A horn antenna recipe is not installed.",
            )),
            turn_id="turn-2",
        )
        clarification = execute_builder_turn(
            self.project_path,
            session,
            "Choose a feed strategy.",
            planner=_RecordingPlanner(_plan(
                status="clarify",
                message="Should each element use an independent port?",
            )),
            turn_id="turn-3",
        )
        self.assertEqual(refusal.terminal_result.outcome, "refuse")
        self.assertEqual(clarification.terminal_result.outcome, "clarify")

        restored = load_builder_session(self.project_path)

        self.assertEqual(restored.design, session.design)
        self.assertEqual(restored.conversation, session.conversation)
        self.assertEqual(restored.memory, session.memory)
        self.assertEqual(restored.memory.canonical_ref.revision, 0)
        self.assertEqual(len(restored.memory.limitations), 1)
        self.assertEqual(len(restored.memory.open_questions), 1)
        self.assertEqual(len(restored.memory.recent_context), 3)
        payload = json.loads(
            (self.project_path / "design" / "builder_conversation.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["schema_version"], 2)

    def test_parameter_table_publication_records_only_explicit_canonical_changes(self):
        design = AntennaDesign.starting_design()
        session = BuilderProjectSession(design=design, memory=ProjectMemory.empty())
        update = apply_structured_state_patch(
            design,
            {"patch_width_mm": 40.0, "substrate_thickness_mm": 1.8},
        )

        publish_builder_state_update(
            self.project_path,
            session,
            update,
            description="Parameter table edit",
            turn_id="turn-table-1",
        )

        changed = session.memory.important_changes[-1].value["parameters"]
        self.assertEqual(
            {item["key"] for item in changed},
            {"patch_width_mm", "substrate_thickness_mm"},
        )
        self.assertEqual(session.memory.canonical_ref.revision, design.revision + 1)

    def test_recent_context_is_bounded_to_six_outcomes(self):
        session = BuilderProjectSession()
        for index in range(7):
            result = execute_builder_turn(
                self.project_path,
                session,
                f"Unsupported request {index}",
                planner=_RecordingPlanner(_plan(
                    status="refuse",
                    message=f"Capability {index} is unavailable.",
                )),
                turn_id=f"turn-{index}",
            )
            self.assertEqual(result.terminal_result.outcome, "refuse")

        self.assertEqual(len(session.memory.recent_context), 6)
        self.assertEqual(session.memory.recent_context[0].source_turn_id, "turn-1")
        self.assertEqual(session.memory.recent_context[-1].source_turn_id, "turn-6")

    def test_legacy_role_content_conversation_and_missing_memory_still_load(self):
        design = AntennaDesign.starting_design()
        save_project_design(
            self.project_path,
            design,
            [{"role": "user", "content": "legacy request"}],
        )
        (self.project_path / "design" / "project_memory.json").unlink(missing_ok=True)

        restored = load_builder_session(self.project_path)

        self.assertEqual(restored.conversation, [{"role": "user", "content": "legacy request"}])
        self.assertEqual(restored.memory.canonical_ref.design_id, design.design_id)
        self.assertEqual(restored.memory.canonical_ref.revision, design.revision)


if __name__ == "__main__":
    unittest.main()
