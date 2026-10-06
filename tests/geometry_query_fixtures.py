"""Deterministic canonical fixtures shared by query and pre-refactor mesh checks."""
from dataclasses import asdict, replace
import hashlib
import json

from studio.antenna_agent import create_default_agent
from studio.antenna_design import BooleanOperation, GeometryObject, TransformSpec
from studio.antenna_tools import ToolCall, create_tool_registry


def representative_designs():
    agent = create_default_agent()
    patch = agent.create_design("inset_patch")
    circle = GeometryObject("tool", "cylinder", "tool", "copper", (
        ("center_1", 0), ("center_2", 0), ("radius", 3),
        ("start", "substrate_thickness_mm"),
        ("end", "substrate_thickness_mm+copper_thickness_mm")), tags=("boolean_tool",))
    rectangle = replace(circle, primitive="sheet_rectangle", dimensions=(
        ("x_min", -3), ("x_max", 3), ("y_min", -.5), ("y_max", .5),
        ("z_min", "substrate_thickness_mm"),
        ("z_max", "substrate_thickness_mm+copper_thickness_mm")))

    def composition(tool, operation="subtract"):
        return replace(patch, geometry=(*patch.geometry, tool), booleans=(
            *patch.booleans, BooleanOperation("custom_boolean", operation, "element_1_1_patch", (tool.object_id,))))

    edge = replace(circle, dimensions=tuple(
        (key, "patch_width_mm/2" if key == "center_1" else value) for key, value in circle.dimensions))
    registry = create_tool_registry()
    translated = replace(patch, geometry=(*patch.geometry, circle))
    translated = registry.execute(translated, ToolCall("geometry.duplicate", {"source_id": "tool", "new_id": "copy", "offset_mm": [10, 0, 0]}))
    translated = registry.execute(translated, ToolCall("geometry.translate", {"object_id": "tool", "offset_mm": [-5, 2, 0]}))
    rotated = replace(rectangle, transform=TransformSpec((3, 4, 0), (0, 0, 30)))
    tilted = replace(circle, transform=TransformSpec(rotate_deg=(90, 0, 0)))
    return {
        "inset": patch,
        "circle_hole": composition(circle),
        "rectangle_hole": composition(rectangle),
        "edge_union": composition(edge, "union"),
        "circular": agent.create_design("circular_patch"),
        "dipole": agent.create_design("dipole"),
        "array_2x3": agent.update_parameters(patch, {"array_rows": 2, "array_columns": 3}).design,
        "translated_duplicate": translated,
        "rotated": replace(patch, geometry=(*patch.geometry, rotated)),
        "separate_z": replace(patch, geometry=(*patch.geometry, replace(circle, transform=TransformSpec((0, 0, 10))))),
        "nonplanar_boolean": composition(tilted),
        "tilted_primitive": replace(patch, geometry=(*patch.geometry, tilted)),
    }


def mesh_signature(scene):
    payload = {
        "solids": [asdict(solid) for solid in scene.solids],
        "element_centers": scene.element_centers,
        "port_segments": scene.port_segments,
        "bounds": scene.bounds,
        "operations": scene.evaluated_operations,
        "warnings": scene.warnings,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
