"""Validated geometry modifiers composed from registered primitive tools."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from studio.antenna_design import AntennaDesign
from studio.antenna_recipes import RecipeParameter
from studio.antenna_tools import CapabilityError, ToolCall, ToolRegistry
from studio.antenna_validation import validate_design


ACTIVE_MODIFIERS_METADATA_KEY = "active_modifiers"


def active_modifier_ids(design: AntennaDesign) -> tuple[str, ...]:
    raw = design.metadata_map().get(ACTIVE_MODIFIERS_METADATA_KEY, "")
    return tuple(item for item in raw.split(",") if item)


class CornerCircleCutoutModifier:
    """Cut four circles from a rectangular patch with centers on its corners."""

    modifier_id = "corner_circle_cutouts_v1"
    display_name = "Circular corner notches"
    aliases = ("circular corner cutouts", "corner circles", "circular fractal corners")
    applicable_families = ("rectangular_inset_patch",)
    required_tools = (
        "parameter.create",
        "geometry.cylinder",
        "boolean.subtract",
        "design.metadata.set",
    )
    parameter_rows = (
        RecipeParameter(
            "corner_radius_ratio",
            "CornerRadiusRatio",
            "Corner radius / patch width",
            "",
            0.25,
            sweepable=True,
            minimum=0.01,
            maximum=0.49,
            sweep_lower_factor=0.8,
            sweep_upper_factor=1.2,
        ),
    )

    def defaults(self) -> dict[str, Any]:
        return {item.key: item.default for item in self.parameter_rows}

    def parameter_definitions(self) -> tuple[RecipeParameter, ...]:
        return self.parameter_rows

    def apply(
        self,
        registry: ToolRegistry,
        design: AntennaDesign,
        values: dict[str, Any],
    ) -> tuple[AntennaDesign, tuple[ToolCall, ...]]:
        if design.family not in self.applicable_families:
            raise CapabilityError("Circular corner cutouts require a rectangular patch design.")
        ratio = float(values.get("corner_radius_ratio", 0.25))
        if not 0.01 <= ratio <= 0.49:
            raise CapabilityError("Corner radius ratio must be from 0.01 to 0.49 of patch width.")
        radius = design.value("patch_width_mm") * ratio
        if radius >= design.value("patch_length_mm") / 2:
            raise CapabilityError(
                "Corner cutout radius must remain less than half the patch length."
            )

        calls: list[ToolCall] = []

        def run(tool_name: str, **arguments: Any) -> None:
            nonlocal design
            call = ToolCall(tool_name, arguments)
            calls.append(call)
            design = registry.execute(design, call)

        run(
            "parameter.create",
            key="corner_radius_ratio",
            name="CornerRadiusRatio",
            label="Corner radius / patch width",
            value=ratio,
            unit="",
            editable=True,
            sweepable=True,
            minimum=0.01,
            maximum=0.49,
            sweep_lower_factor=0.8,
            sweep_upper_factor=1.2,
        )
        run(
            "parameter.create",
            key="corner_cutout_radius_mm",
            name="CornerRadius",
            label="Corner cutout radius",
            value=radius,
            unit="mm",
            expression="patch_width_mm*corner_radius_ratio",
            editable=False,
            sweepable=False,
        )

        element_patches = sorted(
            (
                item
                for item in design.geometry
                if item.object_id.endswith("_patch") and "patch_element" in item.tags
            ),
            key=lambda item: item.object_id,
        )
        for patch in element_patches:
            prefix = patch.object_id.removesuffix("_patch")
            dimensions = patch.dimension_map()
            corners = (
                ("top_left", dimensions["x_min"], dimensions["y_max"]),
                ("top_right", dimensions["x_max"], dimensions["y_max"]),
                # The lower edge belongs to the two side pieces before their union.
                ("bottom_left", dimensions["x_min"], f"{dimensions['y_min']}-inset_depth_mm"),
                ("bottom_right", dimensions["x_max"], f"{dimensions['y_min']}-inset_depth_mm"),
            )
            tool_ids: list[str] = []
            for corner, center_x, center_y in corners:
                object_id = f"{prefix}_corner_cutout_{corner}"
                tool_ids.append(object_id)
                run(
                    "geometry.cylinder",
                    object_id=object_id,
                    name=object_id,
                    material_id="copper",
                    axis="z",
                    tags=("boolean_tool", "corner_cutout"),
                    dimensions={
                        "radius": "corner_cutout_radius_mm",
                        "center_1": center_x,
                        "center_2": center_y,
                        "start": "substrate_thickness_mm",
                        "end": "substrate_thickness_mm+copper_thickness_mm",
                    },
                )
            run(
                "boolean.subtract",
                operation_id=f"{prefix}_corner_cutouts",
                target_id=patch.object_id,
                tool_ids=tuple(tool_ids),
            )

        active = tuple(dict.fromkeys((*active_modifier_ids(design), self.modifier_id)))
        run(
            "design.metadata.set",
            key=ACTIVE_MODIFIERS_METADATA_KEY,
            value=",".join(active),
        )
        return validate_design(replace(design, validation=())), tuple(calls)


def register_builtin_modifiers(registry: ToolRegistry) -> None:
    registry.register_modifier(CornerCircleCutoutModifier())
