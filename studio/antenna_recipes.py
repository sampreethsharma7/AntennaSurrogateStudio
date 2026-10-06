"""Antenna recipes composed exclusively from registered primitive tools."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from studio.antenna_design import AntennaDesign, recipe_excitation_definition
from studio.antenna_tools import CapabilityError, ToolCall, ToolRegistry
from studio.antenna_validation import validate_design


SPEED_OF_LIGHT_MM_GHZ = 299.792458
MAX_ARRAY_ELEMENTS = 64

# Element spacing has one independent variable, and which one it is decides
# whether the operating frequency may be swept.
#
# "lambda"   electrical spacing drives the physical one, so changing the
#            frequency moves every element and resizes the board.  Frequency is
#            therefore not sweepable: an LHS column varying it would vary
#            geometry in the same column and confound a surrogate fit.
# "fixed_mm" physical spacing in mm is independent and the electrical spacing is
#            reported from it.  Geometry no longer depends on frequency, so
#            frequency is sweepable and means only the operating point.
SPACING_MODE_LAMBDA = "lambda"
SPACING_MODE_FIXED_MM = "fixed_mm"
SPACING_MODES = (SPACING_MODE_LAMBDA, SPACING_MODE_FIXED_MM)
ELECTRICAL_SPACING_EXPRESSION = "299.792458/frequency_ghz*element_spacing_lambda"
PHYSICAL_SPACING_EXPRESSION = "element_spacing_mm*frequency_ghz/299.792458"


@dataclass(frozen=True, slots=True)
class MaterialDefinition:
    name: str
    epsilon_r: float
    loss_tangent: float


MATERIALS = {
    "FR4": MaterialDefinition("FR4", 4.4, 0.02),
    "Rogers RO4003C": MaterialDefinition("Rogers RO4003C", 3.55, 0.0027),
    "Rogers RT5880": MaterialDefinition("Rogers RT5880", 2.2, 0.0009),
}

MATERIAL_ALIASES = {
    "fr4": "FR4",
    "fr-4": "FR4",
    "ro4003c": "Rogers RO4003C",
    "rogers 4003": "Rogers RO4003C",
    "rogers ro4003c": "Rogers RO4003C",
    "rt5880": "Rogers RT5880",
    "5880": "Rogers RT5880",
    "rogers rt5880": "Rogers RT5880",
}


@dataclass(frozen=True, slots=True)
class RecipeParameter:
    key: str
    name: str
    label: str
    unit: str
    default: Any
    kind: str = "number"
    sweepable: bool = False
    minimum: float | None = None
    maximum: float | None = None
    sweep_lower_factor: float = 0.9
    sweep_upper_factor: float = 1.1
    choices: tuple[str, ...] = ()


def _patch_dimensions(frequency_ghz: float, epsilon_r: float, height_mm: float) -> tuple[float, float]:
    wavelength = SPEED_OF_LIGHT_MM_GHZ / frequency_ghz
    width = wavelength / 2 * math.sqrt(2 / (epsilon_r + 1))
    effective = (epsilon_r + 1) / 2 + (epsilon_r - 1) / 2 / math.sqrt(1 + 12 * height_mm / width)
    extension = 0.412 * height_mm * ((effective + 0.3) * (width / height_mm + 0.264)) / ((effective - 0.258) * (width / height_mm + 0.8))
    length = wavelength / (2 * math.sqrt(effective)) - 2 * extension
    return round(length, 4), round(width, 4)


def _microstrip_width_50_ohm(epsilon_r: float, height_mm: float) -> float:
    """Return the zero-thickness Hammerstad 50-ohm microstrip width in mm."""

    impedance = 50.0
    a = (
        impedance / 60.0 * math.sqrt((epsilon_r + 1.0) / 2.0)
        + (epsilon_r - 1.0) / (epsilon_r + 1.0) * (0.23 + 0.11 / epsilon_r)
    )
    narrow_ratio = 8.0 * math.exp(a) / (math.exp(2.0 * a) - 2.0)
    if narrow_ratio <= 2.0:
        width_height_ratio = narrow_ratio
    else:
        b = 377.0 * math.pi / (2.0 * impedance * math.sqrt(epsilon_r))
        width_height_ratio = 2.0 / math.pi * (
            b - 1.0 - math.log(2.0 * b - 1.0)
            + (epsilon_r - 1.0) / (2.0 * epsilon_r)
            * (math.log(b - 1.0) + 0.39 - 0.61 / epsilon_r)
        )
    return round(height_mm * width_height_ratio, 4)


def _circular_radius(frequency_ghz: float, epsilon_r: float, height_mm: float) -> float:
    wavelength = SPEED_OF_LIGHT_MM_GHZ / frequency_ghz
    effective = (epsilon_r + 1.0) / 2.0
    first = 1.8412 * wavelength / (2 * math.pi * math.sqrt(effective))
    correction = 1 + 0.5 * height_mm / max(first, 0.1)
    return round(first / correction, 4)


def _center(index: int, count: int, spacing_key: str = "element_spacing_mm") -> str:
    return "0" if count == 1 else f"({index}-({count}-1)/2)*{spacing_key}"


class _RecipeBase:
    parameter_rows: tuple[RecipeParameter, ...] = ()

    def defaults(self) -> dict[str, Any]:
        values = {item.key: item.default for item in self.parameter_rows}
        if "element_spacing_mm" in values:
            # Both modes start from the same geometry: the electrical spacing
            # seeds the physical one, so switching modes does not move anything
            # until something is actually edited.
            values["element_spacing_mm"] = round(
                SPEED_OF_LIGHT_MM_GHZ
                / float(values["frequency_ghz"])
                * float(values["element_spacing_lambda"]),
                4,
            )
        return values

    def parameter_definitions(self) -> tuple[RecipeParameter, ...]:
        return self.parameter_rows

    def _new_design(self, design_id: str | None, revision: int) -> AntennaDesign:
        return AntennaDesign.empty(
            recipe_id=self.recipe_id,
            family=self.family,
            display_name=self.display_name,
            design_id=design_id,
            revision=revision,
        )

    @staticmethod
    def _run(
        registry: ToolRegistry,
        design: AntennaDesign,
        calls: list[ToolCall],
        tool_name: str,
        **arguments: Any,
    ) -> AntennaDesign:
        call = ToolCall(tool_name, arguments)
        calls.append(call)
        return registry.execute(design, call)

    @staticmethod
    def _validate_array(values: dict[str, Any]) -> tuple[int, int, float, float]:
        rows = int(values["array_rows"])
        columns = int(values["array_columns"])
        if rows < 1 or columns < 1 or rows * columns > MAX_ARRAY_ELEMENTS:
            raise CapabilityError("Array rows and columns must describe 1-64 elements.")
        frequency = float(values["frequency_ghz"])
        wavelength = SPEED_OF_LIGHT_MM_GHZ / frequency
        if values.get("spacing_mode", SPACING_MODE_LAMBDA) == SPACING_MODE_FIXED_MM:
            # The physical dimension is authoritative, so a frequency edit must
            # leave it alone.  Bounding it electrically here would re-introduce
            # the coupling this mode exists to remove, so the millimetre value
            # is bounded directly and the electrical spacing is only reported.
            spacing_mm = float(values["element_spacing_mm"])
            if spacing_mm <= 0:
                raise CapabilityError("Element spacing must be greater than zero.")
            return rows, columns, spacing_mm / wavelength, spacing_mm
        spacing_lambda = float(values["element_spacing_lambda"])
        if not 0.2 <= spacing_lambda <= 2.0:
            raise CapabilityError("Element spacing must be from 0.2 to 2.0 wavelengths.")
        return rows, columns, spacing_lambda, wavelength * spacing_lambda

    @staticmethod
    def _spacing_registration(
        values: dict[str, Any],
    ) -> tuple[str | None, dict[str, dict[str, Any]]]:
        """Return the physical-spacing expression and per-parameter overrides.

        The electrical spacing seeds the physical one in both modes, so the
        starting geometry is identical; the modes differ only in which of the
        two stays independent afterwards, and therefore in whether frequency
        can be swept without moving the array.
        """

        if values.get("spacing_mode", SPACING_MODE_LAMBDA) == SPACING_MODE_FIXED_MM:
            spacing_mm = float(values["element_spacing_mm"])
            frequency = float(values["frequency_ghz"])
            return None, {
                "frequency_ghz": {"sweepable": True},
                "element_spacing_lambda": {
                    "editable": False,
                    "sweepable": False,
                    "expression": PHYSICAL_SPACING_EXPRESSION,
                    # Report what the fixed spacing actually is at this
                    # frequency, rather than the value that seeded it.
                    "value": round(spacing_mm * frequency / SPEED_OF_LIGHT_MM_GHZ, 6),
                    "minimum": None,
                    "maximum": None,
                },
                "element_spacing_mm": {
                    "editable": True,
                    "sweepable": True,
                    "minimum": 0.1,
                },
            }
        return ELECTRICAL_SPACING_EXPRESSION, {}

    def _common_start(
        self,
        registry: ToolRegistry,
        design: AntennaDesign,
        calls: list[ToolCall],
        values: dict[str, Any],
        derived: dict[str, tuple[str, float, str | None]],
        overrides: dict[str, dict[str, Any]] | None = None,
    ) -> AntennaDesign:
        overrides = overrides or {}
        definitions = {item.key: item for item in self.parameter_rows}
        for definition in self.parameter_rows:
            # A row that also appears in `derived` is registered by the loop
            # below, which knows its expression for the active spacing mode.
            if definition.kind == "choice" or definition.key in derived:
                continue
            raw = values[definition.key]
            arguments: dict[str, Any] = {
                "key": definition.key,
                "name": definition.name,
                "label": definition.label,
                "value": float(raw),
                "unit": definition.unit,
                "editable": True,
                "sweepable": definition.sweepable,
                "integer": definition.kind == "integer",
                "minimum": definition.minimum,
                "maximum": definition.maximum,
                "sweep_lower_factor": definition.sweep_lower_factor,
                "sweep_upper_factor": definition.sweep_upper_factor,
            }
            arguments.update(overrides.get(definition.key, {}))
            design = self._run(registry, design, calls, "parameter.create", **arguments)
        for key, (name, value, expression) in derived.items():
            arguments = {
                "key": key,
                "name": name,
                # Prefer the recipe row's wording when the key has one, so a
                # table row and its design parameter do not disagree.
                "label": definitions[key].label if key in definitions else key.replace("_", " ").title(),
                "value": value,
                "expression": expression,
                "editable": False,
                "sweepable": False,
            }
            arguments.update(overrides.get(key, {}))
            design = self._run(registry, design, calls, "parameter.create", **arguments)
        rows = int(values["array_rows"])
        columns = int(values["array_columns"])
        design = self._run(
            registry,
            design,
            calls,
            "array.configure",
            rows=rows,
            columns=columns,
            spacing_mm=derived["element_spacing_mm"][1],
        )
        design = self._run(
            registry,
            design,
            calls,
            "em.frequency.setup",
            minimum_ghz="0.65*frequency_ghz",
            maximum_ghz="1.35*frequency_ghz",
        )
        design = self._run(registry, design, calls, "design.metadata.set", key="recipe_id", value=self.recipe_id)
        design = self._run(
            registry,
            design,
            calls,
            "design.metadata.set",
            key="spacing_mode",
            value=str(values.get("spacing_mode", SPACING_MODE_LAMBDA)),
        )
        return design

    def normalize(self, values: dict[str, Any]) -> dict[str, Any]:
        result = self.defaults()
        result.update(values)
        for definition in self.parameter_rows:
            value = result[definition.key]
            if definition.kind == "choice":
                if value not in definition.choices:
                    raise CapabilityError(f"Unsupported {definition.label.lower()}: {value}.")
                continue
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise CapabilityError(f"{definition.label} must be numeric.") from exc
            if not math.isfinite(number):
                raise CapabilityError(f"{definition.label} must be finite.")
            if definition.kind == "integer":
                if not number.is_integer():
                    raise CapabilityError(f"{definition.label} must be a whole number.")
                result[definition.key] = int(number)
            else:
                result[definition.key] = number
            if definition.minimum is not None and number < definition.minimum:
                raise CapabilityError(f"{definition.label} must be at least {definition.minimum:g}.")
            if definition.maximum is not None and number > definition.maximum:
                raise CapabilityError(f"{definition.label} must be at most {definition.maximum:g}.")
        return result


class InsetPatchRecipe(_RecipeBase):
    recipe_id = "inset_patch_v2"
    family = "rectangular_inset_patch"
    display_name = "Inset-fed rectangular patch"
    aliases = ("inset-fed patch", "inset fed patch", "rectangular patch", "inset patch", "microstrip patch")
    required_tools = (
        "parameter.create", "material.define", "geometry.box", "em.port.create",
        "boolean.union", "em.frequency.setup", "array.configure", "design.metadata.set",
    )
    parameter_rows = (
        RecipeParameter("frequency_ghz", "FreqGHz", "Frequency", "GHz", 2.45, sweepable=False, minimum=0.1, maximum=100),
        RecipeParameter("material", "Material", "Substrate material", "", "FR4", kind="choice", choices=tuple(MATERIALS)),
        RecipeParameter("substrate_thickness_mm", "SubH", "Substrate thickness", "mm", 1.6, sweepable=True, minimum=0.05, maximum=20, sweep_lower_factor=0.8, sweep_upper_factor=1.2),
        RecipeParameter("copper_thickness_mm", "CopperT", "Copper thickness", "mm", 0.035, minimum=0.001, maximum=2),
        RecipeParameter("patch_length_mm", "PatchL", "Patch length", "mm", 28.8093, sweepable=True, minimum=0.1),
        RecipeParameter("patch_width_mm", "PatchW", "Patch width", "mm", 37.2343, sweepable=True, minimum=0.1),
        RecipeParameter("feed_width_mm", "FeedW", "Feed width", "mm", 3.0, sweepable=True, minimum=0.01, sweep_lower_factor=0.8, sweep_upper_factor=1.2),
        RecipeParameter("inset_depth_mm", "Inset", "Inset depth", "mm", 8.6428, sweepable=True, minimum=0.01, sweep_lower_factor=0.75, sweep_upper_factor=1.25),
        RecipeParameter("inset_gap_mm", "Gap", "Inset gap", "mm", 1.0, sweepable=True, minimum=0.01, sweep_lower_factor=0.75, sweep_upper_factor=1.25),
        RecipeParameter("board_margin_mm", "Margin", "Board margin", "mm", 7.0, minimum=0.1),
        RecipeParameter("array_rows", "ArrayRows", "Array rows", "", 1, kind="integer", minimum=1, maximum=16),
        RecipeParameter("array_columns", "ArrayCols", "Array columns", "", 1, kind="integer", minimum=1, maximum=16),
        RecipeParameter("element_spacing_lambda", "SpacingLambda", "Element spacing", "λ₀", 0.55, sweepable=True, minimum=0.2, maximum=2),
        RecipeParameter("spacing_mode", "SpacingMode", "Element spacing basis", "", SPACING_MODE_LAMBDA, kind="choice", choices=SPACING_MODES),
        RecipeParameter("element_spacing_mm", "ElementSpacing", "Physical element spacing", "mm", 0.0, minimum=0.1),
    )

    def defaults(self) -> dict[str, Any]:
        values = super().defaults()
        material = MATERIALS[values["material"]]
        length, width = _patch_dimensions(values["frequency_ghz"], material.epsilon_r, values["substrate_thickness_mm"])
        feed_width = _microstrip_width_50_ohm(
            material.epsilon_r, values["substrate_thickness_mm"]
        )
        values.update(
            patch_length_mm=length,
            patch_width_mm=width,
            feed_width_mm=feed_width,
            inset_depth_mm=round(length * 0.30, 4),
        )
        return values

    def normalize(self, values: dict[str, Any]) -> dict[str, Any]:
        result = super().normalize(values)
        rows, columns, _, spacing = self._validate_array(result)
        if rows * columns > MAX_ARRAY_ELEMENTS:
            raise CapabilityError("The array exceeds 64 elements.")
        if result["inset_depth_mm"] >= result["patch_length_mm"] / 2:
            raise CapabilityError("Inset depth must be less than half the patch length.")
        if result["feed_width_mm"] + 2 * result["inset_gap_mm"] >= result["patch_width_mm"]:
            raise CapabilityError("The feed and inset gaps must fit inside the patch width.")
        if columns > 1 and spacing <= result["patch_width_mm"]:
            raise CapabilityError("Column spacing must exceed patch width.")
        if rows > 1 and spacing <= result["patch_length_mm"]:
            raise CapabilityError("Row spacing must exceed patch length.")
        return result

    def build(self, registry: ToolRegistry, values: dict[str, Any], *, design_id: str | None = None, revision: int = 0) -> tuple[AntennaDesign, tuple[ToolCall, ...]]:
        values = self.normalize(values)
        rows, columns, _, spacing = self._validate_array(values)
        board_width = values["patch_width_mm"] + (columns - 1) * spacing + 2 * values["board_margin_mm"]
        board_length = values["patch_length_mm"] + (rows - 1) * spacing + 2 * values["board_margin_mm"]
        spacing_expression, spacing_overrides = self._spacing_registration(values)
        derived = {
            "element_spacing_mm": ("ElementSpacing", spacing, spacing_expression),
            "board_width_mm": ("BoardW", board_width, "patch_width_mm+(array_columns-1)*element_spacing_mm+2*board_margin_mm"),
            "board_length_mm": ("BoardL", board_length, "patch_length_mm+(array_rows-1)*element_spacing_mm+2*board_margin_mm"),
        }
        design = self._new_design(design_id, revision)
        calls: list[ToolCall] = []
        design = self._common_start(registry, design, calls, values, derived, spacing_overrides)
        material = MATERIALS[values["material"]]
        design = self._run(registry, design, calls, "material.define", material_id="copper", name="Copper", kind="conductor", conductivity_s_per_m=5.8e7)
        design = self._run(registry, design, calls, "material.define", material_id="substrate", name=material.name, kind="dielectric", epsilon_r=material.epsilon_r, loss_tangent=material.loss_tangent)
        design = self._run(registry, design, calls, "design.metadata.set", key="substrate_material", value=material.name)
        design = self._run(registry, design, calls, "geometry.box", object_id="ground", name="Ground", material_id="copper", tags=("ground",), dimensions={"x_min": "-board_width_mm/2", "x_max": "board_width_mm/2", "y_min": "-board_length_mm/2", "y_max": "board_length_mm/2", "z_min": "-copper_thickness_mm", "z_max": 0})
        design = self._run(registry, design, calls, "geometry.box", object_id="substrate", name="Substrate", material_id="substrate", tags=("substrate",), dimensions={"x_min": "-board_width_mm/2", "x_max": "board_width_mm/2", "y_min": "-board_length_mm/2", "y_max": "board_length_mm/2", "z_min": 0, "z_max": "substrate_thickness_mm"})
        element = 0
        for row in range(rows):
            cy = _center(row, rows)
            for column in range(columns):
                element += 1
                cx = _center(column, columns)
                prefix = f"element_{row + 1}_{column + 1}"
                x0, x1 = f"{cx}-patch_width_mm/2", f"{cx}+patch_width_mm/2"
                y0, y1 = f"{cy}-patch_length_mm/2", f"{cy}+patch_length_mm/2"
                inset_end = f"{y0}+inset_depth_mm"
                notch_left = f"{cx}-feed_width_mm/2-inset_gap_mm"
                notch_right = f"{cx}+feed_width_mm/2+inset_gap_mm"
                pieces = (
                    ("patch", x0, x1, inset_end, y1),
                    ("patch_left", x0, notch_left, y0, inset_end),
                    ("patch_right", notch_right, x1, y0, inset_end),
                    ("feed", f"{cx}-feed_width_mm/2", f"{cx}+feed_width_mm/2", "-board_length_mm/2", inset_end),
                )
                for suffix, xa, xb, ya, yb in pieces:
                    design = self._run(registry, design, calls, "geometry.box", object_id=f"{prefix}_{suffix}", name=f"{prefix}_{suffix}", material_id="copper", tags=("patch_element", f"element_{element}"), dimensions={"x_min": xa, "x_max": xb, "y_min": ya, "y_max": yb, "z_min": "substrate_thickness_mm", "z_max": "substrate_thickness_mm+copper_thickness_mm"})
                design = self._run(
                    registry,
                    design,
                    calls,
                    "boolean.union",
                    operation_id=f"{prefix}_conductor_union",
                    target_id=f"{prefix}_patch",
                    tool_ids=(
                        f"{prefix}_patch_left",
                        f"{prefix}_patch_right",
                        f"{prefix}_feed",
                    ),
                )
                design = self._run(registry, design, calls, "em.port.create", port_id=f"port_{element}", name=f"Port {element}", kind="discrete", positive_point=(cx, "-board_length_mm/2", "substrate_thickness_mm"), negative_point=(cx, "-board_length_mm/2", 0), element_index=element)
        design = replace(design, excitation=recipe_excitation_definition(design))
        return validate_design(design), tuple(calls)


class CircularPatchRecipe(_RecipeBase):
    recipe_id = "circular_patch_v1"
    family = "circular_patch"
    display_name = "Probe-fed circular patch"
    aliases = ("circular patch", "round patch", "circular microstrip patch")
    required_tools = (
        "parameter.create", "material.define", "geometry.box", "geometry.cylinder",
        "em.port.create", "em.frequency.setup", "array.configure", "design.metadata.set",
    )
    parameter_rows = (
        RecipeParameter("frequency_ghz", "FreqGHz", "Frequency", "GHz", 2.45, sweepable=False, minimum=0.1, maximum=100),
        RecipeParameter("material", "Material", "Substrate material", "", "FR4", kind="choice", choices=tuple(MATERIALS)),
        RecipeParameter("substrate_thickness_mm", "SubH", "Substrate thickness", "mm", 1.6, sweepable=True, minimum=0.05, maximum=20, sweep_lower_factor=0.8, sweep_upper_factor=1.2),
        RecipeParameter("copper_thickness_mm", "CopperT", "Copper thickness", "mm", 0.035, minimum=0.001, maximum=2),
        RecipeParameter("patch_radius_mm", "PatchRadius", "Patch radius", "mm", 16.4, sweepable=True, minimum=0.1),
        RecipeParameter("feed_offset_mm", "FeedOffset", "Probe offset", "mm", 5.5, sweepable=True, minimum=0),
        RecipeParameter("probe_radius_mm", "ProbeRadius", "Probe radius", "mm", 0.6, sweepable=True, minimum=0.05),
        RecipeParameter("board_margin_mm", "Margin", "Board margin", "mm", 7.0, minimum=0.1),
        RecipeParameter("array_rows", "ArrayRows", "Array rows", "", 1, kind="integer", minimum=1, maximum=16),
        RecipeParameter("array_columns", "ArrayCols", "Array columns", "", 1, kind="integer", minimum=1, maximum=16),
        RecipeParameter("element_spacing_lambda", "SpacingLambda", "Element spacing", "λ₀", 0.55, sweepable=True, minimum=0.2, maximum=2),
        RecipeParameter("spacing_mode", "SpacingMode", "Element spacing basis", "", SPACING_MODE_LAMBDA, kind="choice", choices=SPACING_MODES),
        RecipeParameter("element_spacing_mm", "ElementSpacing", "Physical element spacing", "mm", 0.0, minimum=0.1),
    )

    def defaults(self) -> dict[str, Any]:
        values = super().defaults()
        material = MATERIALS[values["material"]]
        radius = _circular_radius(values["frequency_ghz"], material.epsilon_r, values["substrate_thickness_mm"])
        values.update(patch_radius_mm=radius, feed_offset_mm=round(radius * 0.33, 4))
        return values

    def normalize(self, values: dict[str, Any]) -> dict[str, Any]:
        result = super().normalize(values)
        rows, columns, _, spacing = self._validate_array(result)
        if result["feed_offset_mm"] + result["probe_radius_mm"] >= result["patch_radius_mm"]:
            raise CapabilityError("The probe must remain inside the circular patch.")
        diameter = 2 * result["patch_radius_mm"]
        if (rows > 1 or columns > 1) and spacing <= diameter:
            raise CapabilityError("Array spacing must exceed the circular-patch diameter.")
        return result

    def build(self, registry: ToolRegistry, values: dict[str, Any], *, design_id: str | None = None, revision: int = 0) -> tuple[AntennaDesign, tuple[ToolCall, ...]]:
        values = self.normalize(values)
        rows, columns, _, spacing = self._validate_array(values)
        diameter = 2 * values["patch_radius_mm"]
        board_width = diameter + (columns - 1) * spacing + 2 * values["board_margin_mm"]
        board_length = diameter + (rows - 1) * spacing + 2 * values["board_margin_mm"]
        spacing_expression, spacing_overrides = self._spacing_registration(values)
        derived = {
            "element_spacing_mm": ("ElementSpacing", spacing, spacing_expression),
            "board_width_mm": ("BoardW", board_width, "2*patch_radius_mm+(array_columns-1)*element_spacing_mm+2*board_margin_mm"),
            "board_length_mm": ("BoardL", board_length, "2*patch_radius_mm+(array_rows-1)*element_spacing_mm+2*board_margin_mm"),
            "coax_dielectric_radius_mm": ("CoaxDielectricRadius", 3.5 * values["probe_radius_mm"], "3.5*probe_radius_mm"),
            "coax_outer_radius_mm": ("CoaxOuterRadius", 4.0 * values["probe_radius_mm"], "4*probe_radius_mm"),
            "coax_launch_length_mm": ("CoaxLength", 3.0 * values["substrate_thickness_mm"], "3*substrate_thickness_mm"),
        }
        design = self._new_design(design_id, revision)
        calls: list[ToolCall] = []
        design = self._common_start(registry, design, calls, values, derived, spacing_overrides)
        material = MATERIALS[values["material"]]
        design = self._run(registry, design, calls, "material.define", material_id="copper", name="Copper", kind="conductor", conductivity_s_per_m=5.8e7)
        design = self._run(registry, design, calls, "material.define", material_id="substrate", name=material.name, kind="dielectric", epsilon_r=material.epsilon_r, loss_tangent=material.loss_tangent)
        design = self._run(registry, design, calls, "material.define", material_id="coax_dielectric", name="PTFE", kind="dielectric", epsilon_r=2.1, loss_tangent=0.0002)
        design = self._run(registry, design, calls, "design.metadata.set", key="substrate_material", value=material.name)
        for object_id, material_id, tags, z0, z1 in (
            ("ground", "copper", ("ground",), "-copper_thickness_mm", 0),
            ("substrate", "substrate", ("substrate",), 0, "substrate_thickness_mm"),
        ):
            design = self._run(registry, design, calls, "geometry.box", object_id=object_id, name=object_id.title(), material_id=material_id, tags=tags, dimensions={"x_min": "-board_width_mm/2", "x_max": "board_width_mm/2", "y_min": "-board_length_mm/2", "y_max": "board_length_mm/2", "z_min": z0, "z_max": z1})
        element = 0
        for row in range(rows):
            cy = _center(row, rows)
            for column in range(columns):
                element += 1
                cx = _center(column, columns)
                probe_x = f"{cx}+feed_offset_mm"
                prefix = f"element_{element}"
                patch_id = f"{prefix}_patch"
                probe_id = f"{prefix}_probe"
                dielectric_id = f"{prefix}_coax_dielectric"
                outer_id = f"{prefix}_coax_outer"
                clearance_id = f"{prefix}_ground_clearance_tool"
                pin_bore_id = f"{prefix}_coax_pin_bore_tool"
                outer_bore_id = f"{prefix}_coax_outer_bore_tool"
                design = self._run(registry, design, calls, "geometry.cylinder", object_id=patch_id, name=f"CircularPatch_{element}", material_id="copper", axis="z", tags=("circular_patch_element", f"element_{element}"), dimensions={"center_1": cx, "center_2": cy, "radius": "patch_radius_mm", "start": "substrate_thickness_mm", "end": "substrate_thickness_mm+copper_thickness_mm"})
                design = self._run(registry, design, calls, "geometry.cylinder", object_id=probe_id, name=f"ProbePin_{element}", material_id="copper", axis="z", tags=("probe_feed", "coax_signal", f"element_{element}"), dimensions={"center_1": probe_x, "center_2": cy, "radius": "probe_radius_mm", "start": "-coax_launch_length_mm", "end": "substrate_thickness_mm+copper_thickness_mm"})
                design = self._run(registry, design, calls, "geometry.cylinder", object_id=clearance_id, name=f"GroundClearanceTool_{element}", material_id="coax_dielectric", axis="z", tags=("ground_clearance_tool", f"element_{element}"), dimensions={"center_1": probe_x, "center_2": cy, "radius": "coax_dielectric_radius_mm", "start": "-copper_thickness_mm", "end": 0})
                design = self._run(registry, design, calls, "geometry.cylinder", object_id=dielectric_id, name=f"CoaxDielectric_{element}", material_id="coax_dielectric", axis="z", tags=("coax_dielectric", f"element_{element}"), dimensions={"center_1": probe_x, "center_2": cy, "radius": "coax_dielectric_radius_mm", "start": "-coax_launch_length_mm", "end": 0})
                design = self._run(registry, design, calls, "geometry.cylinder", object_id=pin_bore_id, name=f"CoaxPinBoreTool_{element}", material_id="copper", axis="z", tags=("coax_pin_bore_tool", f"element_{element}"), dimensions={"center_1": probe_x, "center_2": cy, "radius": "probe_radius_mm", "start": "-coax_launch_length_mm", "end": 0})
                design = self._run(registry, design, calls, "geometry.cylinder", object_id=outer_id, name=f"CoaxOuter_{element}", material_id="copper", axis="z", tags=("coax_reference", f"element_{element}"), dimensions={"center_1": probe_x, "center_2": cy, "radius": "coax_outer_radius_mm", "start": "-coax_launch_length_mm", "end": 0})
                design = self._run(registry, design, calls, "geometry.cylinder", object_id=outer_bore_id, name=f"CoaxOuterBoreTool_{element}", material_id="coax_dielectric", axis="z", tags=("coax_outer_bore_tool", f"element_{element}"), dimensions={"center_1": probe_x, "center_2": cy, "radius": "coax_dielectric_radius_mm", "start": "-coax_launch_length_mm", "end": 0})
                design = self._run(registry, design, calls, "boolean.subtract", operation_id=f"{prefix}_ground_clearance_subtract", target_id="ground", tool_ids=(clearance_id,))
                design = self._run(registry, design, calls, "boolean.subtract", operation_id=f"{prefix}_coax_dielectric_subtract", target_id=dielectric_id, tool_ids=(pin_bore_id,))
                design = self._run(registry, design, calls, "boolean.subtract", operation_id=f"{prefix}_coax_outer_subtract", target_id=outer_id, tool_ids=(outer_bore_id,))
                reference_x = f"{probe_x}+(coax_dielectric_radius_mm+coax_outer_radius_mm)/2"
                design = self._run(
                    registry,
                    design,
                    calls,
                    "em.port.create",
                    port_id=f"port_{element}",
                    name=f"Port {element}",
                    kind="modal_cross_section",
                    positive_point=(probe_x, cy, "-coax_launch_length_mm"),
                    negative_point=(reference_x, cy, "-coax_launch_length_mm"),
                    signal_terminal=probe_id,
                    reference_terminal=outer_id,
                    cross_section={"geometry_id": dielectric_id, "face": "z_min"},
                    mode_count=1,
                    element_index=element,
                )
        design = replace(design, excitation=recipe_excitation_definition(design))
        return validate_design(design), tuple(calls)


class DipoleRecipe(_RecipeBase):
    recipe_id = "dipole_v1"
    family = "dipole"
    display_name = "Center-fed dipole"
    aliases = ("simple dipole", "half-wave dipole", "half wave dipole", "dipole")
    required_tools = (
        "parameter.create", "material.define", "geometry.cylinder", "em.port.create",
        "em.frequency.setup", "array.configure", "design.metadata.set",
    )
    parameter_rows = (
        RecipeParameter("frequency_ghz", "FreqGHz", "Frequency", "GHz", 2.45, sweepable=False, minimum=0.1, maximum=100),
        RecipeParameter("arm_length_mm", "ArmLength", "Arm length", "mm", 28.8, sweepable=True, minimum=0.1),
        RecipeParameter("conductor_radius_mm", "WireRadius", "Conductor radius", "mm", 0.4, sweepable=True, minimum=0.01),
        RecipeParameter("feed_gap_mm", "FeedGap", "Feed gap", "mm", 1.2, sweepable=True, minimum=0.01),
        RecipeParameter("array_rows", "ArrayRows", "Array rows", "", 1, kind="integer", minimum=1, maximum=16),
        RecipeParameter("array_columns", "ArrayCols", "Array columns", "", 1, kind="integer", minimum=1, maximum=16),
        RecipeParameter("element_spacing_lambda", "SpacingLambda", "Element spacing", "λ₀", 0.5, sweepable=True, minimum=0.2, maximum=2),
        RecipeParameter("spacing_mode", "SpacingMode", "Element spacing basis", "", SPACING_MODE_LAMBDA, kind="choice", choices=SPACING_MODES),
        RecipeParameter("element_spacing_mm", "ElementSpacing", "Physical element spacing", "mm", 0.0, minimum=0.1),
    )

    def defaults(self) -> dict[str, Any]:
        values = super().defaults()
        wavelength = SPEED_OF_LIGHT_MM_GHZ / values["frequency_ghz"]
        gap = wavelength * 0.01
        values.update(
            arm_length_mm=round((0.475 * wavelength - gap) / 2, 4),
            conductor_radius_mm=round(wavelength / 300, 4),
            feed_gap_mm=round(gap, 4),
        )
        return values

    def normalize(self, values: dict[str, Any]) -> dict[str, Any]:
        result = super().normalize(values)
        rows, columns, _, spacing = self._validate_array(result)
        total_length = 2 * result["arm_length_mm"] + result["feed_gap_mm"]
        if result["conductor_radius_mm"] * 2 >= result["feed_gap_mm"]:
            raise CapabilityError("Dipole feed gap must exceed the conductor diameter.")
        if rows > 1 and spacing <= total_length:
            raise CapabilityError("Dipole row spacing must exceed the element length.")
        if columns > 1 and spacing <= 2 * result["conductor_radius_mm"]:
            raise CapabilityError("Dipole column spacing must exceed the conductor diameter.")
        return result

    def build(self, registry: ToolRegistry, values: dict[str, Any], *, design_id: str | None = None, revision: int = 0) -> tuple[AntennaDesign, tuple[ToolCall, ...]]:
        values = self.normalize(values)
        rows, columns, _, spacing = self._validate_array(values)
        spacing_expression, spacing_overrides = self._spacing_registration(values)
        derived = {
            "element_spacing_mm": ("ElementSpacing", spacing, spacing_expression),
            "total_length_mm": ("DipoleLength", 2 * values["arm_length_mm"] + values["feed_gap_mm"], "2*arm_length_mm+feed_gap_mm"),
        }
        design = self._new_design(design_id, revision)
        calls: list[ToolCall] = []
        design = self._common_start(registry, design, calls, values, derived, spacing_overrides)
        design = self._run(registry, design, calls, "material.define", material_id="copper", name="Copper", kind="conductor", conductivity_s_per_m=5.8e7)
        design = self._run(registry, design, calls, "design.metadata.set", key="substrate_material", value="No substrate")
        element = 0
        for row in range(rows):
            cy = _center(row, rows)
            for column in range(columns):
                element += 1
                cx = _center(column, columns)
                lower_start = "-feed_gap_mm/2-arm_length_mm"
                lower_end = "-feed_gap_mm/2"
                upper_start = "feed_gap_mm/2"
                upper_end = "feed_gap_mm/2+arm_length_mm"
                for suffix, start, end in (("lower", lower_start, lower_end), ("upper", upper_start, upper_end)):
                    design = self._run(registry, design, calls, "geometry.cylinder", object_id=f"element_{element}_{suffix}_arm", name=f"Dipole_{element}_{suffix}", material_id="copper", axis="z", tags=("dipole_arm", f"element_{element}"), dimensions={"center_1": cx, "center_2": cy, "radius": "conductor_radius_mm", "start": start, "end": end})
                design = self._run(registry, design, calls, "em.port.create", port_id=f"port_{element}", name=f"Port {element}", kind="discrete", positive_point=(cx, cy, "feed_gap_mm/2"), negative_point=(cx, cy, "-feed_gap_mm/2"), element_index=element)
        design = replace(design, excitation=recipe_excitation_definition(design))
        return validate_design(design), tuple(calls)


def register_builtin_recipes(registry: ToolRegistry) -> None:
    registry.register_recipe(InsetPatchRecipe())
    registry.register_recipe(CircularPatchRecipe())
    registry.register_recipe(DipoleRecipe())
