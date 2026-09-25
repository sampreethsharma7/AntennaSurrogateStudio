import json
import tempfile
import unittest
from pathlib import Path

from studio.antenna_builder import (
    BuilderProjectSession,
    CanonicalDesignRef,
    ProjectMemory,
    ProjectMemoryItem,
    apply_registered_tool_plan,
    load_builder_session,
    save_builder_session,
    save_project_design,
)
from studio.antenna_design import AntennaDesign
from studio.antenna_llm_planner import LLMToolPlan, PlannedToolCall


class BuilderProjectSessionTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        self.project_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_new_project_session_has_no_design_and_loading_creates_no_state_file(self):
        state_path = self.project_path / "design" / "antenna_state.json"

        session = load_builder_session(self.project_path)

        self.assertIsNone(session.design)
        self.assertEqual(session.conversation, [])
        self.assertEqual(session.memory, ProjectMemory.empty())
        self.assertFalse(state_path.exists())

    def test_empty_project_memory_round_trips_at_the_project_boundary(self):
        session = BuilderProjectSession(memory=ProjectMemory.empty())

        state_path, conversation_path, memory_path = save_builder_session(
            self.project_path,
            session,
        )
        restored = load_builder_session(self.project_path)

        self.assertIsNone(state_path)
        self.assertTrue(conversation_path.is_file())
        self.assertTrue(memory_path.is_file())
        self.assertEqual(restored.memory, ProjectMemory.empty())
        payload = json.loads(memory_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 1)
        self.assertIsNone(payload["canonical_ref"])
        self.assertTrue(all(payload[name] == [] for name in (
            "requirements",
            "decisions",
            "assumptions",
            "limitations",
            "open_questions",
            "important_changes",
            "recent_context",
        )))

    def test_session_conversation_round_trips_without_losing_provenance_fields(self):
        messages = [
            {
                "turn_id": "turn-1",
                "role": "user",
                "content": "Design an inset-fed patch.",
                "created_at": "2026-09-21T12:00:00+00:00",
            },
            {
                "turn_id": "turn-2",
                "role": "builder",
                "content": "Created the design.",
                "design_revision": 0,
            },
        ]
        session = BuilderProjectSession(conversation=messages)

        save_builder_session(self.project_path, session)
        restored = load_builder_session(self.project_path)

        self.assertEqual(restored.conversation, messages)

    def test_project_memory_items_round_trip_with_status_and_provenance(self):
        requirement = ProjectMemoryItem(
            item_id="requirement-target-frequency",
            key="target_frequency",
            value=2.45,
            unit="GHz",
            status="active",
            source_turn_id="turn-1",
            design_revision=0,
        )
        memory = ProjectMemory(
            canonical_ref=CanonicalDesignRef("design-1", 0),
            requirements=(requirement,),
        )

        save_builder_session(
            self.project_path,
            BuilderProjectSession(memory=memory),
        )
        restored = load_builder_session(self.project_path)

        self.assertEqual(restored.memory, memory)

    def test_existing_antenna_project_restores_design_unchanged(self):
        design = AntennaDesign.starting_design()
        save_project_design(self.project_path, design, [])

        restored = load_builder_session(self.project_path)

        self.assertEqual(restored.design, design)

    def test_legacy_project_without_memory_gets_only_a_canonical_reference(self):
        design = AntennaDesign.starting_design()
        save_project_design(self.project_path, design, [])
        memory_path = self.project_path / "design" / "project_memory.json"

        restored = load_builder_session(self.project_path)

        self.assertFalse(memory_path.exists())
        self.assertEqual(
            restored.memory.canonical_ref,
            CanonicalDesignRef(design.design_id, design.revision),
        )
        self.assertEqual(restored.memory.requirements, ())
        self.assertEqual(restored.memory.decisions, ())
        self.assertEqual(restored.memory.assumptions, ())

    def test_composed_operation_state_is_unchanged_by_session_round_trip(self):
        base = AntennaDesign.starting_design()
        plan = LLMToolPlan(
            "execute",
            "Add a center slot.",
            (
                PlannedToolCall(
                    "parameter.create",
                    {
                        "key": "slot_radius_mm",
                        "label": "Slot radius",
                        "value": 3.0,
                        "unit": "mm",
                        "sweepable": True,
                    },
                ),
                PlannedToolCall(
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
                PlannedToolCall(
                    "boolean.subtract",
                    {
                        "operation_id": "center_slot_subtract",
                        "target_id": "element_1_1_patch",
                        "tool_ids": ["center_slot_tool"],
                    },
                ),
            ),
        )
        composed = apply_registered_tool_plan(base, plan).state
        memory = ProjectMemory.empty(
            CanonicalDesignRef(composed.design_id, composed.revision)
        )

        save_builder_session(
            self.project_path,
            BuilderProjectSession(design=composed, memory=memory),
        )
        restored = load_builder_session(self.project_path)

        self.assertEqual(restored.design, composed)
        self.assertEqual(restored.design.composed_operations, composed.composed_operations)


if __name__ == "__main__":
    unittest.main()
