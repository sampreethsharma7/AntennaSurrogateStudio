import json
import unittest

from studio.antenna_agent import (
    AgentInstructionError,
    PlannerClarificationRequired,
    PlannerRefusal,
    create_default_agent,
)
from studio.antenna_builder import (
    ProjectMemory,
    ProjectMemoryItem,
    apply_registered_tool_plan,
    apply_text_instruction,
)
from studio.antenna_design import AntennaDesign
from studio.antenna_llm_planner import (
    LLMToolPlan,
    PlannedToolCall,
    build_planner_exchange,
)


def _plan(*calls, status="execute", message="test plan"):
    return LLMToolPlan(
        status,
        message,
        tuple(PlannedToolCall(name, arguments) for name, arguments in calls),
    )


class _RecordingPlanner:
    def __init__(self, result):
        self.result = result
        self.request = None

    def plan(self, **request):
        self.request = request
        return self.result


class EmptyDesignPlanningTests(unittest.TestCase):
    def test_valid_inset_patch_request_creates_first_design_from_null_context(self):
        planner = _RecordingPlanner(_plan(
            ("recipe.select", {"recipe_id": "inset_patch_v2"}),
            ("parameter.set", {"key": "frequency_ghz", "value": 2.45}),
            ("parameter.set", {"key": "material", "value": "FR4"}),
        ))
        remembered = ProjectMemory(
            canonical_ref=None,
            requirements=(ProjectMemoryItem(
                item_id="req-1",
                key="project_note",
                value="educational prototype",
                status="active",
                source_turn_id="turn-0",
            ),),
        )

        result = apply_text_instruction(
            None,
            "Design an inset-fed rectangular patch antenna at 2.45 GHz on FR4.",
            planner=planner,
            project_memory=remembered,
        )

        self.assertEqual(result.state.recipe_id, "inset_patch_v2")
        self.assertEqual(result.state.revision, 0)
        self.assertEqual(result.state.frequency_ghz, 2.45)
        self.assertEqual(result.state.material, "FR4")
        self.assertIsNone(planner.request["current_design"])
        self.assertEqual(planner.request["project_memory"], remembered.to_dict())
        self.assertEqual(
            [item["name"] for item in planner.request["capability_manifest"]["callable_tools"]],
            ["recipe.select", "parameter.set"],
        )
        parameter_tool = planner.request["capability_manifest"]["callable_tools"][1]
        self.assertNotIn(
            "corner_radius_ratio",
            parameter_tool["arguments"]["properties"]["key"]["enum"],
        )
        self.assertIsNone(planner.request["capability_manifest"]["current_recipe_id"])
        self.assertEqual(
            {item["recipe_id"] for item in planner.request["capability_manifest"]["recipes"]},
            {"inset_patch_v2", "circular_patch_v1", "dipole_v1"},
        )

    def test_provider_neutral_exchange_serializes_null_design_and_project_memory(self):
        manifest = create_default_agent().capability_manifest(None)
        memory = ProjectMemory.empty().to_dict()

        exchange = build_planner_exchange(
            instruction="Design an inset-fed patch at 2.45 GHz on FR4.",
            current_design=None,
            capability_manifest=manifest,
            project_memory=memory,
        )
        payload = json.loads(exchange.user_content)

        self.assertIsNone(payload["current_design"])
        self.assertEqual(payload["project_memory"], memory)
        self.assertEqual(set(exchange.callable_names), {"recipe.select", "parameter.set"})

    def test_circular_patch_request_creates_the_correct_family(self):
        result = apply_registered_tool_plan(None, _plan(
            ("recipe.select", {"recipe_id": "circular_patch_v1"}),
            ("parameter.set", {"key": "frequency_ghz", "value": 5.8}),
            ("parameter.set", {"key": "material", "value": "Rogers RT5880"}),
        ))

        self.assertEqual(result.state.recipe_id, "circular_patch_v1")
        self.assertEqual(result.state.family, "circular_patch")
        self.assertEqual(result.state.frequency_ghz, 5.8)

    def test_dipole_request_creates_the_correct_family(self):
        result = apply_registered_tool_plan(None, _plan(
            ("recipe.select", {"recipe_id": "dipole_v1"}),
            ("parameter.set", {"key": "frequency_ghz", "value": 1.0}),
        ))

        self.assertEqual(result.state.recipe_id, "dipole_v1")
        self.assertEqual(result.state.family, "dipole")
        self.assertEqual(result.state.frequency_ghz, 1.0)

    def test_capability_question_does_not_create_a_design(self):
        planner = _RecordingPlanner(_plan(
            status="clarify",
            message="I can build an inset-fed patch, circular patch, or simple dipole. Which should I create?",
        ))
        design = None

        with self.assertRaises(PlannerClarificationRequired):
            apply_text_instruction(
                design,
                "What antenna types can you currently build?",
                planner=planner,
            )

        self.assertIsNone(design)
        self.assertEqual(planner.request["current_design"], None)

    def test_unsupported_family_refusal_leaves_the_project_empty(self):
        planner = _RecordingPlanner(_plan(
            status="refuse",
            message="A horn antenna recipe is not installed.",
        ))
        design = None

        with self.assertRaises(PlannerRefusal):
            apply_text_instruction(
                design,
                "Design a horn antenna at 10 GHz.",
                planner=planner,
            )

        self.assertIsNone(design)

    def test_invalid_parameter_after_recipe_selection_rolls_back_to_none(self):
        design = None
        invalid = _plan(
            ("recipe.select", {"recipe_id": "inset_patch_v2"}),
            ("parameter.set", {"key": "inset_depth_mm", "value": 999.0}),
        )

        with self.assertRaisesRegex(AgentInstructionError, "initial antenna design failed"):
            apply_registered_tool_plan(design, invalid)

        self.assertIsNone(design)

    def test_design_dependent_geometry_before_recipe_selection_is_rejected(self):
        design = None
        invalid = _plan(
            ("geometry.cylinder", {
                "object_id": "slot_tool",
                "material_id": "copper",
                "axis": "z",
                "tags": ["planner_created", "boolean_tool", "slot"],
                "dimensions": {
                    "center_1": 0,
                    "center_2": 0,
                    "radius": 3,
                    "start": 0,
                    "end": 1,
                },
            }),
            ("recipe.select", {"recipe_id": "inset_patch_v2"}),
        )

        with self.assertRaisesRegex(AgentInstructionError, "recipe.select as the first"):
            apply_registered_tool_plan(design, invalid)

        self.assertIsNone(design)

    def test_existing_design_execution_path_keeps_revision_and_identity_behavior(self):
        original = AntennaDesign.starting_design()

        result = apply_registered_tool_plan(original, _plan(
            ("parameter.set", {"key": "patch_width_mm", "value": 40.0}),
        ))

        self.assertEqual(result.state.design_id, original.design_id)
        self.assertEqual(result.state.revision, original.revision + 1)
        self.assertEqual(result.state.patch_width_mm, 40.0)
        self.assertEqual(original.revision, 0)


if __name__ == "__main__":
    unittest.main()
