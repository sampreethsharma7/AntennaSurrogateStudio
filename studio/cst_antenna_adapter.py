"""CST adapter for the solver-neutral antenna design graph."""

from __future__ import annotations

import re
from dataclasses import dataclass

from studio.antenna_design import AntennaDesign, GeometryObject, Scalar
from studio.antenna_validation import validate_adapter_input, validate_design


class CSTAdapterError(ValueError):
    """Raised when a canonical design cannot be represented by this adapter."""


@dataclass(frozen=True, slots=True)
class CSTHistoryOperation:
    """One replayable CST-native History List operation."""

    name: str
    script: str
    category: str


class CSTAdapter:
    adapter_id = "cst"

    def validate(self, design: AntennaDesign) -> None:
        validated = validate_design(design)
        record = validate_adapter_input(validated, self.adapter_id)
        if not record.passed:
            raise CSTAdapterError("; ".join(record.messages))

    @staticmethod
    def _expression(design: AntennaDesign, value: Scalar) -> str:
        if isinstance(value, (int, float)):
            return f"{float(value):.12g}"
        names = {parameter.key: parameter.name for parameter in design.parameters}
        return re.sub(
            r"\b[A-Za-z_]\w*\b",
            lambda match: names.get(match.group(0), match.group(0)),
            str(value),
        )

    @staticmethod
    def _material_name(design: AntennaDesign, material_id: str) -> str:
        material = next(item for item in design.materials if item.material_id == material_id)
        return "PEC" if material.kind == "conductor" else f"ASS_{material.material_id}"

    def _box(self, design: AntennaDesign, geometry: GeometryObject) -> list[str]:
        values = geometry.dimension_map()
        return [
            "With Brick",
            ".Reset",
            f'.Name "{geometry.name}"',
            f'.Component "{geometry.component}"',
            f'.Material "{self._material_name(design, geometry.material_id)}"',
            f'.Xrange "{self._expression(design, values["x_min"])}", "{self._expression(design, values["x_max"])}"',
            f'.Yrange "{self._expression(design, values["y_min"])}", "{self._expression(design, values["y_max"])}"',
            f'.Zrange "{self._expression(design, values["z_min"])}", "{self._expression(design, values["z_max"])}"',
            ".Create",
            "End With",
        ]

    def _cylinder(self, design: AntennaDesign, geometry: GeometryObject) -> list[str]:
        values = geometry.dimension_map()
        axis = geometry.axis.upper()
        centers = {
            "X": ("Ycenter", "Zcenter", "Xrange"),
            "Y": ("Xcenter", "Zcenter", "Yrange"),
            "Z": ("Xcenter", "Ycenter", "Zrange"),
        }[axis]
        return [
            "With Cylinder",
            ".Reset",
            f'.Name "{geometry.name}"',
            f'.Component "{geometry.component}"',
            f'.Material "{self._material_name(design, geometry.material_id)}"',
            f'.OuterRadius "{self._expression(design, values["radius"])}"',
            '.InnerRadius "0"',
            f'.Axis "{axis.lower()}"',
            f'.{centers[0]} "{self._expression(design, values["center_1"])}"',
            f'.{centers[1]} "{self._expression(design, values["center_2"])}"',
            f'.{centers[2]} "{self._expression(design, values["start"])}", "{self._expression(design, values["end"])}"',
            '.Segments "0"',
            ".Create",
            "End With",
        ]

    def parameter_values(self, design: AntennaDesign) -> tuple[tuple[str, str], ...]:
        """Return CST parameter names and expressions for non-history storage."""
        self.validate(design)
        return tuple(
            (
                parameter.name,
                self._expression(
                    design,
                    parameter.expression or f"{parameter.value:.12g}",
                ),
            )
            for parameter in design.parameters
        )

    @staticmethod
    def _operation(name: str, category: str, rows: list[str] | tuple[str, ...] | str) -> CSTHistoryOperation:
        script = rows if isinstance(rows, str) else "\n".join(rows)
        return CSTHistoryOperation(name=name, script=script, category=category)

    def history_operations(self, design: AntennaDesign) -> tuple[CSTHistoryOperation, ...]:
        """Build independently replayable, CST-native history operations.

        Parameters are deliberately excluded. Native project generation stores
        them through the CST project API before adding these history entries, so
        a later History List rebuild cannot overwrite a value edited in CST.

        This exports canonical physical ports, including recipe compatibility
        ports. Whether any port realizes the current excitation strategy remains
        defined exclusively by ExcitationDefinition assignments; unsupported
        strategy names are never applied to compatibility-port history entries.
        """
        self.validate(design)
        operations: list[CSTHistoryOperation] = [
            self._operation(
                "change units",
                "setup",
                [
                    "With Units",
                    '.Geometry "mm"',
                    '.Frequency "GHz"',
                    '.Time "ns"',
                    "End With",
                ],
            )
        ]
        for material in design.materials:
            if material.kind != "dielectric":
                continue
            operations.append(
                self._operation(
                        f"define material: ASS_{material.material_id}",
                    "material",
                    [
                        "With Material",
                        ".Reset",
                        f'.Name "ASS_{material.material_id}"',
                        '.Type "Normal"',
                        f'.Epsilon "{material.epsilon_r:.12g}"',
                        '.Mue "1"',
                        f'.TanD "{material.loss_tangent:.12g}"',
                        '.TanDGiven "True"',
                        f'.TanDFreq "{design.parameter("frequency_ghz").name}"',
                        '.TanDModel "ConstTanD"',
                        '.Colour "0.18", "0.48", "0.40"',
                        ".Create",
                        "End With",
                    ],
                )
            )
        for geometry in design.geometry:
            object_name = f"{geometry.component}:{geometry.name}"
            if geometry.primitive in {"box", "sheet_rectangle"}:
                primitive = "brick"
                rows = self._box(design, geometry)
            elif geometry.primitive in {"cylinder", "sheet_circle"}:
                primitive = "cylinder"
                rows = self._cylinder(design, geometry)
            else:
                raise CSTAdapterError(f"Unsupported CST primitive: {geometry.primitive}.")
            operations.append(
                self._operation(f"define {primitive}: {object_name}", "primitive", rows)
            )
            rx, ry, rz = geometry.transform.rotate_deg
            if any(abs(value) > 1e-12 for value in (rx, ry, rz)):
                operations.append(
                    self._operation(
                        f"Rotate: {object_name}",
                        "transform",
                        [
                            "With Transform",
                            ".Reset",
                            f'.Name "{object_name}"',
                            '.Origin "Free"',
                            '.Center "0", "0", "0"',
                            f'.Angle "{rx:.12g}", "{ry:.12g}", "{rz:.12g}"',
                            '.MultipleObjects "False"',
                            '.GroupObjects "False"',
                            '.Repetitions "1"',
                            '.MultipleSelection "False"',
                            '.Transform "Shape", "Rotate"',
                            "End With",
                        ],
                    )
                )
            tx, ty, tz = geometry.transform.translate_mm
            if any(abs(value) > 1e-12 for value in (tx, ty, tz)):
                operations.append(
                    self._operation(
                        f"Translate: {object_name}",
                        "transform",
                        [
                            "With Transform",
                            ".Reset",
                            f'.Name "{object_name}"',
                            f'.Vector "{tx:.12g}", "{ty:.12g}", "{tz:.12g}"',
                            '.UsePickedPoints "False"',
                            '.InvertPickedPoints "False"',
                            '.MultipleObjects "False"',
                            '.GroupObjects "False"',
                            '.Repetitions "1"',
                            '.Transform "Shape", "Translate"',
                            "End With",
                        ],
                    )
                )
        by_id = {item.object_id: item for item in design.geometry}
        for operation in design.booleans:
            target = by_id[operation.target_id]
            target_name = f"{target.component}:{target.name}"
            method = "Subtract" if operation.operation == "subtract" else "Add"
            for tool_id in operation.tool_ids:
                tool = by_id[tool_id]
                tool_name = f"{tool.component}:{tool.name}"
                operations.append(
                    self._operation(
                        f"{method}: {tool_name} from {target_name}"
                        if method == "Subtract"
                        else f"Union: {tool_name} with {target_name}",
                        "boolean",
                        f'Solid.{method} "{target_name}", "{tool_name}"',
                    )
                )
        for number, port in enumerate(design.ports, start=1):
            p1 = tuple(self._expression(design, value) for value in port.negative_point)
            p2 = tuple(self._expression(design, value) for value in port.positive_point)
            operations.append(
                self._operation(
                    f"define port: {port.name or f'P{number}'}",
                    "port",
                    [
                        "With DiscretePort",
                        ".Reset",
                        f'.PortNumber "{number}"',
                        '.Type "SParameter"',
                        f'.Impedance "{port.impedance_ohms:.12g}"',
                        '.Voltage "1"',
                        '.Current "1"',
                        '.Monitor "True"',
                        '.Radius "0"',
                        f'.SetP1 "False", "{p1[0]}", "{p1[1]}", "{p1[2]}"',
                        f'.SetP2 "False", "{p2[0]}", "{p2[1]}", "{p2[2]}"',
                        '.InvertDirection "False"',
                        '.LocalCoordinates "False"',
                        '.Wire ""',
                        '.Position "end1"',
                        ".Create",
                        "End With",
                    ],
                )
            )
        minimum = self._expression(design, design.simulation.frequency_min_ghz)
        maximum = self._expression(design, design.simulation.frequency_max_ghz)
        boundary = "expanded open" if design.simulation.boundary == "expanded_open" else design.simulation.boundary
        operations.extend(
            [
                self._operation(
                    "Set frequency range",
                    "simulation",
                    f'Solver.FrequencyRange "{minimum}", "{maximum}"',
                ),
                self._operation(
                    "Set boundaries",
                    "simulation",
                    [
                        "With Boundary",
                        f'.Xmin "{boundary}"',
                        f'.Xmax "{boundary}"',
                        f'.Ymin "{boundary}"',
                        f'.Ymax "{boundary}"',
                        f'.Zmin "{boundary}"',
                        f'.Zmax "{boundary}"',
                        "End With",
                    ],
                ),
                self._operation(
                    "Select solver: HF Time Domain",
                    "simulation",
                    'ChangeSolverType "HF Time Domain"',
                ),
            ]
        )
        return tuple(operations)

    def history(self, design: AntennaDesign) -> str:
        operations = self.history_operations(design)
        rows = [operations[0].script]
        rows.extend(
            f'StoreParameter "{name}", "{value}"'
            for name, value in self.parameter_values(design)
        )
        rows.extend(operation.script for operation in operations[1:])
        return "\n".join(rows)

    def macro(self, design: AntennaDesign) -> str:
        warning = (
            "' Experimental generated starting design. Validate mesh, ports, materials, and results before engineering use.\n"
            "' Array elements use independent ports. No array feed network is synthesized.\n"
            "' Array row/column parameter changes require regenerating this construction script.\n"
        )
        return warning + "Sub Main\n" + self.history(design) + "\nEnd Sub\n"
