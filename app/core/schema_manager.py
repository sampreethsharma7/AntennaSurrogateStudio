from pathlib import Path
import json
import re


def infer_output_axis(output_columns: list[str]) -> dict:
    values = []
    unit = "Output Index"
    for col in output_columns:
        match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(GHz|MHz|deg)", col, re.IGNORECASE)
        if not match:
            return {"kind": "index", "axis_values": list(range(len(output_columns))), "axis_unit": "Output Index"}
        values.append(float(match.group(1)))
        unit = match.group(2)
    kind = "frequency" if unit.lower() in {"ghz", "mhz"} else "angle"
    return {"kind": kind, "axis_values": values, "axis_unit": unit}


def save_schema(project_dir: Path, schema: dict) -> None:
    (project_dir / "data" / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
