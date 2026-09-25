from dataclasses import replace
import unittest

from studio.antenna_agent import CapabilityUnavailableError, PlannerClarificationRequired, PlannerRefusal, create_default_agent
from studio.antenna_builder import apply_structured_state_patch
from studio.antenna_design import AntennaDesign, ComposedToolCall, object_bounds, resolved_dimensions
from studio.antenna_llm_planner import LLMToolPlan, PlannedToolCall
from studio.antenna_tools import CapabilityError, ToolCall, ToolDefinition, ToolRegistry, create_tool_registry
from studio.cst_antenna_adapter import CSTAdapter


def plan(*calls, status="execute", message="validated plan"):
    return LLMToolPlan(status, message, tuple(PlannedToolCall(name, args) for name, args in calls))


class AntennaAgentArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.agent = create_default_agent()

    def _center_circle_design(self):
        return self.agent.execute_llm_plan(self.agent.create_design("inset_patch"), plan(
            ("parameter.create", {
                "key": "slot_radius_mm", "label": "Slot radius",
                "value": 3.0, "unit": "mm", "sweepable": True,
            }),
            ("geometry.cylinder", {
                "object_id": "center_slot_tool", "material_id": "copper", "axis": "z",
                "tags": ["planner_created", "boolean_tool", "slot"],
                "dimensions": {
                    "center_1": 0, "center_2": 0, "radius": "slot_radius_mm",
                    "start": "substrate_thickness_mm", "end": "substrate_thickness_mm+copper_thickness_mm",
                },
            }),
            ("boolean.subtract", {
                "operation_id": "center_slot_subtract", "target_id": "element_1_1_patch",
                "tool_ids": ["center_slot_tool"],
            }),
        )).design

    def _two_circle_design(self):
        return self.agent.execute_llm_plan(self.agent.create_design("inset_patch"), plan(
            ("parameter.create", {
                "key": "slot_radius_mm", "label": "Slot radius",
                "value": 2.0, "unit": "mm", "sweepable": True,
            }),
            ("geometry.cylinder", {
                "object_id": "left_slot_tool", "material_id": "copper", "axis": "z",
                "tags": ["planner_created", "boolean_tool", "slot"],
                "dimensions": {
                    "center_1": -5, "center_2": 0, "radius": "slot_radius_mm",
                    "start": "substrate_thickness_mm", "end": "substrate_thickness_mm+copper_thickness_mm",
                },
            }),
            ("geometry.duplicate", {
                "source_id": "left_slot_tool", "new_id": "right_slot_tool", "offset_mm": [10, 0, 0],
            }),
            ("boolean.subtract", {
                "operation_id": "symmetric_slots_subtract", "target_id": "element_1_1_patch",
                "tool_ids": ["left_slot_tool", "right_slot_tool"],
            }),
        )).design

    def _rectangle_slot_design(self):
        return self.agent.execute_llm_plan(self.agent.create_design("inset_patch"), plan(
            ("parameter.create", {
                "key": "slot_width_mm", "label": "Slot width", "value": 6.0,
                "unit": "mm", "sweepable": True,
            }),
            ("parameter.create", {
                "key": "slot_height_mm", "label": "Slot height", "value": 1.0,
                "unit": "mm", "sweepable": True,
            }),
            ("geometry.rectangle_sheet", {
                "object_id": "center_rect_slot_tool", "material_id": "copper",
                "tags": ["planner_created", "boolean_tool", "slot"],
                "dimensions": {
                    "x_min": "-slot_width_mm/2", "x_max": "slot_width_mm/2",
                    "y_min": "-slot_height_mm/2", "y_max": "slot_height_mm/2",
                    "z_min": "substrate_thickness_mm", "z_max": "substrate_thickness_mm+copper_thickness_mm",
                },
            }),
            ("boolean.subtract", {
                "operation_id": "center_rect_slot_subtract", "target_id": "element_1_1_patch",
                "tool_ids": ["center_rect_slot_tool"],
            }),
        )).design

    def test_registry_exposes_primitives_recipes_modifiers_and_planner_tools(self):
        capabilities = self.agent.available_capabilities()
        self.assertEqual(capabilities["recipes"], ("inset_patch_v2", "circular_patch_v1", "dipole_v1"))
        self.assertEqual(capabilities["modifiers"], ("corner_circle_cutouts_v1",))
        self.assertEqual(capabilities["planner_tools"], (
            "design.reset", "recipe.select", "parameter.set", "excitation.set_strategy", "composition.set_scope",
            "composition.update_operation", "composition.delete", "modifier.apply", "modifier.remove",
            "parameter.create", "geometry.rectangle_sheet", "geometry.cylinder",
            "geometry.circle_sheet", "geometry.translate", "geometry.rotate", "geometry.duplicate",
            "boolean.subtract", "boolean.union", "engineering.design_summary", "engineering.array_spacing",
            "engineering.rectangular_patch_baseline", "engineering.dipole_baseline",
        ))
        self.assertTrue({"geometry.box", "geometry.cylinder", "boolean.subtract", "em.port.create"}.issubset(capabilities["tools"]))

    def test_runtime_manifest_contains_exact_schemas_and_primitive_inventory(self):
        design = self.agent.create_design("inset_patch")
        manifest = self.agent.capability_manifest(design)
        self.assertEqual(manifest["current_recipe_id"], "inset_patch_v2")
        self.assertEqual(
            [item["name"] for item in manifest["callable_tools"]],
            [name for name in self.agent.available_capabilities()["planner_tools"] if not name.startswith("composition.")],
        )
        self.assertIn("additionalProperties", manifest["callable_tools"][0]["arguments"])
        self.assertIn("geometry.cylinder", {item["name"] for item in manifest["deterministic_primitive_tools"]})
        primitive = next(item for item in manifest["callable_tools"] if item["name"] == "geometry.cylinder")
        self.assertEqual(primitive["kind"], "primitive")
        self.assertEqual(primitive["arguments"]["properties"]["material_id"]["const"], "copper")
        targets = next(item for item in manifest["callable_tools"] if item["name"] == "boolean.subtract")
        self.assertEqual(targets["arguments"]["properties"]["target_id"]["enum"], ["element_1_1_patch"])
        patch = next(item for item in manifest["semantic_geometry_objects"] if item["object_id"] == "element_1_1_patch")
        self.assertEqual(patch["role"], "radiating_patch_conductor")
        self.assertEqual(patch["element_center_mm"], [0.0, 0.0])

    def test_center_circle_slot_is_genuine_primitive_composition(self):
        starting = self.agent.create_design("inset_patch")
        result = self.agent.execute_llm_plan(starting, plan(
            ("recipe.select", {"recipe_id": "inset_patch_v2"}),
            ("parameter.set", {"key": "frequency_ghz", "value": 2.45}),
            ("parameter.set", {"key": "material", "value": "FR4"}),
            ("parameter.create", {
                "key": "slot_radius_mm", "label": "Slot radius",
                "value": 3.0, "unit": "mm", "sweepable": True, "minimum": 0.1, "maximum": 10.0,
            }),
            ("geometry.cylinder", {
                "object_id": "center_slot_tool", "material_id": "copper", "axis": "z",
                "tags": ["planner_created", "boolean_tool", "slot"],
                "dimensions": {
                    "center_1": 0, "center_2": 0, "radius": "slot_radius_mm",
                    "start": "substrate_thickness_mm", "end": "substrate_thickness_mm+copper_thickness_mm",
                },
            }),
            ("boolean.subtract", {
                "operation_id": "center_slot_subtract", "target_id": "element_1_1_patch", "tool_ids": ["center_slot_tool"],
            }),
        ))
        composed = result.design
        self.assertEqual(composed.value("slot_radius_mm"), 3.0)
        self.assertEqual(composed.metadata_map().get("active_modifiers", ""), "")
        self.assertEqual(composed.booleans[-1].operation_id, "center_slot_subtract")
        self.assertEqual(composed.booleans[-1].target_id, "element_1_1_patch")
        slot = next(item for item in composed.geometry if item.object_id == "center_slot_tool")
        self.assertEqual(resolved_dimensions(composed, slot)["center_1"], 0.0)
        self.assertEqual(resolved_dimensions(composed, slot)["center_2"], 0.0)
        self.assertEqual([call.name for call in result.plan.planner_calls[-3:]], [
            "parameter.create", "geometry.cylinder", "boolean.subtract",
        ])
        for unchanged_id in ("ground", "substrate", "element_1_1_feed"):
            self.assertEqual(
                next(item for item in starting.geometry if item.object_id == unchanged_id),
                next(item for item in composed.geometry if item.object_id == unchanged_id),
            )
        script = CSTAdapter().macro(composed)
        self.assertIn('StoreParameter "SlotRadius", "3"', script)
        self.assertIn('.Name "center_slot_tool"', script)
        self.assertIn('Solid.Subtract "Antenna:element_1_1_patch", "Antenna:center_slot_tool"', script)

    def test_two_symmetric_circle_slots_can_use_duplicate_and_one_subtraction(self):
        design = self.agent.create_design("inset_patch")
        result = self.agent.execute_llm_plan(design, plan(
            ("parameter.create", {
                "key": "slot_radius_mm", "label": "Slot radius",
                "value": 2.0, "unit": "mm", "sweepable": True,
            }),
            ("geometry.cylinder", {
                "object_id": "left_slot_tool", "material_id": "copper", "axis": "z",
                "tags": ["planner_created", "boolean_tool", "slot"],
                "dimensions": {
                    "center_1": -5, "center_2": 0, "radius": "slot_radius_mm",
                    "start": "substrate_thickness_mm", "end": "substrate_thickness_mm+copper_thickness_mm",
                },
            }),
            ("geometry.duplicate", {"source_id": "left_slot_tool", "new_id": "right_slot_tool", "offset_mm": [10, 0, 0]}),
            ("boolean.subtract", {
                "operation_id": "symmetric_slots_subtract", "target_id": "element_1_1_patch",
                "tool_ids": ["left_slot_tool", "right_slot_tool"],
            }),
        ))
        centers = sorted(
            object_bounds(result.design, item)[0:2]
            for item in result.design.geometry
            if item.object_id in {"left_slot_tool", "right_slot_tool"}
        )
        self.assertEqual(centers, [(-7.0, -3.0), (3.0, 7.0)])
        self.assertEqual(result.design.booleans[-1].tool_ids, ("left_slot_tool", "right_slot_tool"))

    def test_center_rectangular_slot_composes_sheet_and_subtraction(self):
        design = self.agent.create_design("inset_patch")
        result = self.agent.execute_llm_plan(design, plan(
            ("parameter.create", {"key": "slot_width_mm", "label": "Slot width", "value": 6.0, "unit": "mm", "sweepable": True}),
            ("parameter.create", {"key": "slot_height_mm", "label": "Slot height", "value": 1.0, "unit": "mm", "sweepable": True}),
            ("geometry.rectangle_sheet", {
                "object_id": "center_rect_slot_tool", "material_id": "copper",
                "tags": ["planner_created", "boolean_tool", "slot"],
                "dimensions": {
                    "x_min": "-slot_width_mm/2", "x_max": "slot_width_mm/2",
                    "y_min": "-slot_height_mm/2", "y_max": "slot_height_mm/2",
                    "z_min": "substrate_thickness_mm", "z_max": "substrate_thickness_mm+copper_thickness_mm",
                },
            }),
            ("boolean.subtract", {"operation_id": "center_rect_slot_subtract", "target_id": "element_1_1_patch", "tool_ids": ["center_rect_slot_tool"]}),
        ))
        self.assertEqual(result.design.value("slot_width_mm"), 6.0)
        self.assertEqual(result.design.value("slot_height_mm"), 1.0)
        self.assertEqual(result.design.booleans[-1].operation, "subtract")

    def test_center_slot_replays_after_patch_width_change_and_remains_in_cst(self):
        composed = self._center_circle_design()
        group = composed.composed_operations[0]
        stored_calls = list(group.calls)
        stale_arguments = stored_calls[-1].arguments()
        stale_arguments["target_id"] = "obsolete_recipe_object_id"
        stored_calls[-1] = ComposedToolCall.create(
            stored_calls[-1].name,
            stale_arguments,
            semantic_target=stored_calls[-1].semantic_target,
        )
        composed = replace(composed, composed_operations=(replace(group, calls=tuple(stored_calls)),))

        updated = self.agent.execute_llm_plan(composed, plan(
            ("parameter.set", {"key": "patch_width_mm", "value": 40.0}),
        )).design

        self.assertEqual(updated.patch_width_mm, 40.0)
        self.assertEqual(len(updated.composed_operations), 1)
        self.assertEqual(updated.booleans[-1].operation_id, "center_slot_subtract")
        self.assertIsNotNone(next(item for item in updated.geometry if item.object_id == "center_slot_tool"))
        script = CSTAdapter().macro(updated)
        self.assertIn('StoreParameter "SlotRadius", "3"', script)
        self.assertIn('Solid.Subtract "Antenna:element_1_1_patch", "Antenna:center_slot_tool"', script)

    def test_two_slots_replay_after_frequency_and_material_change(self):
        composed = self._two_circle_design()

        updated = self.agent.execute_llm_plan(composed, plan(
            ("parameter.set", {"key": "frequency_ghz", "value": 3.0}),
            ("parameter.set", {"key": "material", "value": "Rogers RT5880"}),
        )).design

        self.assertEqual(updated.frequency_ghz, 3.0)
        self.assertEqual(updated.material, "Rogers RT5880")
        self.assertEqual(updated.booleans[-1].tool_ids, ("left_slot_tool", "right_slot_tool"))
        self.assertEqual(
            {item.object_id for item in updated.geometry if "boolean_tool" in item.tags},
            {"left_slot_tool", "right_slot_tool"},
        )

    def test_rectangular_slot_replays_after_table_style_substrate_change(self):
        composed = self._rectangle_slot_design()

        updated = apply_structured_state_patch(
            composed,
            {"substrate_thickness_mm": 2.0},
        ).state

        slot = next(item for item in updated.geometry if item.object_id == "center_rect_slot_tool")
        self.assertEqual(updated.substrate_thickness_mm, 2.0)
        self.assertEqual(resolved_dimensions(updated, slot)["z_min"], 2.0)
        self.assertEqual(updated.booleans[-1].operation_id, "center_rect_slot_subtract")

    def test_incompatible_topology_change_refuses_without_losing_composition(self):
        composed = self._center_circle_design()

        with self.assertRaisesRegex(CapabilityUnavailableError, "topology change was not applied"):
            self.agent.execute_llm_plan(composed, plan(
                ("recipe.select", {"recipe_id": "dipole_v1"}),
            ))

        self.assertEqual(composed.family, "rectangular_inset_patch")
        self.assertEqual(composed.booleans[-1].operation_id, "center_slot_subtract")
        self.assertEqual(len(composed.composed_operations), 1)

    def test_failed_replay_leaves_previous_valid_design_unchanged(self):
        composed = self._center_circle_design()
        before = composed.to_dict()

        with self.assertRaisesRegex(CapabilityUnavailableError, "no longer compatible"):
            self.agent.update_parameters(composed, {"patch_width_mm": 5.5})

        self.assertEqual(composed.to_dict(), before)
        self.assertEqual(composed.patch_width_mm, self.agent.create_design("inset_patch").patch_width_mm)
        self.assertEqual(composed.booleans[-1].operation_id, "center_slot_subtract")

    def test_primitive_guard_rejects_invented_targets_base_transforms_and_partial_state(self):
        design = self.agent.create_design("inset_patch")
        with self.assertRaisesRegex(CapabilityUnavailableError, "target does not exist"):
            self.agent.execute_llm_plan(design, plan(
                ("geometry.cylinder", {
                    "object_id": "slot_tool", "material_id": "copper", "axis": "z",
                    "tags": ["planner_created", "boolean_tool", "slot"],
                    "dimensions": {"center_1": 0, "center_2": 0, "radius": "patch_width_mm/10", "start": "substrate_thickness_mm", "end": "substrate_thickness_mm+copper_thickness_mm"},
                }),
                ("boolean.subtract", {"operation_id": "bad_subtract", "target_id": "invented_patch", "tool_ids": ["slot_tool"]}),
            ))
        with self.assertRaisesRegex(CapabilityUnavailableError, "radiating patch conductor"):
            self.agent.execute_llm_plan(design, plan(
                ("geometry.cylinder", {
                    "object_id": "slot_tool", "material_id": "copper", "axis": "z",
                    "tags": ["planner_created", "boolean_tool", "slot"],
                    "dimensions": {"center_1": 0, "center_2": 0, "radius": "patch_width_mm/10", "start": "substrate_thickness_mm", "end": "substrate_thickness_mm+copper_thickness_mm"},
                }),
                ("boolean.subtract", {"operation_id": "bad_feed_subtract", "target_id": "element_1_1_feed", "tool_ids": ["slot_tool"]}),
            ))
        with self.assertRaisesRegex(CapabilityUnavailableError, "unregistered planning tool"):
            self.agent.execute_llm_plan(design, plan(("solver.start", {})))
        with self.assertRaisesRegex(CapabilityError, "identifier pattern"):
            self.agent.execute_llm_plan(design, plan(("geometry.translate", {"object_id": "element_1_1_patch", "offset_mm": [1, 0, 0]})))
        self.assertEqual(design.revision, 0)

    def test_external_primitive_can_register_without_core_changes(self):
        registry = ToolRegistry()
        registry.register_tool(ToolDefinition("example.noop", "example", "Test extension point.", "primitive", lambda design, _args: design))
        design = AntennaDesign.empty(recipe_id="example", family="example", display_name="Example")
        self.assertIs(registry.execute(design, ToolCall("example.noop", {})), design)

    def test_llm_recipe_plan_compiles_to_registered_primitive_calls(self):
        starting = self.agent.create_design("inset_patch")
        result = self.agent.execute_llm_plan(starting, plan(
            ("recipe.select", {"recipe_id": "circular_patch_v1"}),
            ("parameter.set", {"key": "frequency_ghz", "value": 5.8}),
            ("parameter.set", {"key": "material", "value": "Rogers RT5880"}),
        ))
        self.assertEqual(result.design.family, "circular_patch")
        self.assertEqual(result.design.frequency_ghz, 5.8)
        self.assertIn("geometry.cylinder", result.plan.planned_tools)
        self.assertIn("geometry.cylinder", [call.name for call in result.executed_tools])
        self.assertEqual(len(result.plan.planner_calls), 3)

    def test_geometry_transform_duplicate_and_boolean_primitives_still_compose(self):
        registry = create_tool_registry()
        design = AntennaDesign.empty(recipe_id="composition_test", family="composition_test", display_name="Composition test")
        design = registry.execute_all(design, (
            ToolCall("material.define", {"material_id": "pec", "name": "PEC", "kind": "conductor"}),
            ToolCall("geometry.box", {"object_id": "base", "material_id": "pec", "dimensions": {"x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10, "z_min": 0, "z_max": 1}}),
            ToolCall("geometry.duplicate", {"source_id": "base", "new_id": "copy", "offset_mm": (12, 0, 0)}),
            ToolCall("geometry.rotate", {"object_id": "copy", "angles_deg": (0, 0, 45)}),
            ToolCall("boolean.union", {"operation_id": "joined", "target_id": "base", "tool_ids": ("copy",)}),
        ))
        self.assertEqual(design.geometry[1].transform.rotate_deg, (0.0, 0.0, 45.0))
        self.assertEqual(design.booleans[0].operation, "union")

    def test_conversation_keeps_one_design_id_and_increments_revision(self):
        design = self.agent.create_design("inset_patch")
        circular = self.agent.execute_llm_plan(design, plan(("recipe.select", {"recipe_id": "circular_patch_v1"})))
        array = self.agent.execute_llm_plan(circular.design, plan(
            ("parameter.set", {"key": "array_rows", "value": 1}),
            ("parameter.set", {"key": "array_columns", "value": 4}),
            ("parameter.set", {"key": "element_spacing_lambda", "value": 0.6}),
        ))
        edited = self.agent.execute_llm_plan(array.design, plan(("parameter.set", {"key": "patch_radius_mm", "value": 15})))
        self.assertEqual({design.design_id, circular.design.design_id, array.design.design_id, edited.design.design_id}, {design.design_id})
        self.assertEqual((design.revision, circular.design.revision, array.design.revision, edited.design.revision), (0, 1, 2, 3))
        self.assertEqual(array.design.array.element_count, 4)
        self.assertEqual(edited.design.value("patch_radius_mm"), 15)

    def test_llm_clarification_and_refusal_never_change_state(self):
        design = self.agent.create_design("inset_patch")
        with self.assertRaisesRegex(PlannerClarificationRequired, "subtract or add"):
            self.agent.execute_llm_plan(design, plan(status="clarify", message="Should the circles subtract or add copper?"))
        with self.assertRaisesRegex(PlannerRefusal, "horn"):
            self.agent.execute_llm_plan(design, plan(status="refuse", message="A horn recipe is not installed."))
        self.assertEqual(design.revision, 0)

    def test_unregistered_or_cross_family_calls_are_rejected(self):
        design = self.agent.create_design("circular_patch")
        with self.assertRaisesRegex(CapabilityUnavailableError, "unregistered"):
            self.agent.execute_llm_plan(design, plan(("solver.start", {})))
        with self.assertRaisesRegex(CapabilityUnavailableError, "unavailable"):
            self.agent.execute_llm_plan(design, plan(("parameter.set", {"key": "conductor_radius_mm", "value": 0.8})))

    def test_corner_cutout_plan_uses_modifier_and_exact_corner_centers(self):
        design = self.agent.create_design("inset_patch")
        result = self.agent.execute_llm_plan(design, plan(
            ("design.reset", {}),
            ("modifier.apply", {"modifier_id": "corner_circle_cutouts_v1"}),
            ("parameter.set", {"key": "corner_radius_ratio", "value": 0.25}),
        ))
        modified = result.design
        self.assertEqual(result.plan.modifier_ids, ("corner_circle_cutouts_v1",))
        self.assertAlmostEqual(modified.value("corner_cutout_radius_mm"), modified.value("patch_width_mm") / 4)
        cutouts = [item for item in modified.geometry if "corner_cutout" in item.tags]
        centers = {(round(resolved_dimensions(modified, item)["center_1"], 6), round(resolved_dimensions(modified, item)["center_2"], 6)) for item in cutouts}
        half_width = round(modified.value("patch_width_mm") / 2, 6)
        half_length = round(modified.value("patch_length_mm") / 2, 6)
        self.assertEqual(centers, {(-half_width, -half_length), (-half_width, half_length), (half_width, -half_length), (half_width, half_length)})
        resized = self.agent.execute_llm_plan(modified, plan(("parameter.set", {"key": "patch_width_mm", "value": 40}))).design
        self.assertEqual(resized.value("corner_cutout_radius_mm"), 10.0)

    def test_invalid_recipe_update_is_rejected_before_geometry(self):
        design = self.agent.create_design("circular_patch")
        with self.assertRaisesRegex(CapabilityError, "inside the circular patch"):
            self.agent.execute_llm_plan(design, plan(
                ("parameter.set", {"key": "patch_radius_mm", "value": 5}),
                ("parameter.set", {"key": "feed_offset_mm", "value": 5}),
            ))

    def test_cst_remains_adapter_over_solver_neutral_design(self):
        circular = self.agent.execute_llm_plan(self.agent.create_design("inset_patch"), plan(
            ("recipe.select", {"recipe_id": "circular_patch_v1"}),
            ("parameter.set", {"key": "array_columns", "value": 2}),
            ("parameter.set", {"key": "element_spacing_lambda", "value": 0.6}),
        )).design
        script = CSTAdapter().macro(circular)
        self.assertIn('StoreParameter "PatchRadius"', script)
        self.assertEqual(script.count("With Port"), 2)
        self.assertNotIn("With DiscretePort", script)
        self.assertNotIn("Solver.Start", script)
        self.assertNotIn("solver_command", str(circular.to_dict()).casefold())

    def test_solver_neutral_state_round_trip(self):
        design = self.agent.create_design("dipole")
        self.assertEqual(AntennaDesign.from_dict(design.to_dict()), design)


if __name__ == "__main__":
    unittest.main()
