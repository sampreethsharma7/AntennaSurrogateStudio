"""Antenna simulation discovery and preparation.

The two supported loaders preserve the workflows from
beamforming-inverse-search-tool while adding clearer validation and atomic
output writes.
"""

from __future__ import annotations

import csv
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable


PARAM_BLOCK_RE = re.compile(r"#Parameters\s*=\s*\{([^}]*)\}")
FILENAME_PHASE_RE = re.compile(
    r"A(-?\d+(?:\.\d+)?)_phi(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
QUOTED_HEADER_RE = re.compile(r'"([^"]+)"')

ProgressCallback = Callable[[str], None]
IMPORTED_OUTPUT_LABEL = "Imported output table"

LEGACY_GENERIC_INPUT_TEMPLATE_HEADER = ["Variable 1", "Variable 2"]
LEGACY_GENERIC_INPUT_TEMPLATE_ROWS = [
    [0.0, 0.0],
    [0.5, 1.0],
    [1.0, 0.5],
]
LEGACY_GENERIC_OUTPUT_TEMPLATE_HEADER = [
    "Response at coordinate min",
    "Response at coordinate midpoint",
    "Response at coordinate max",
]
LEGACY_GENERIC_OUTPUT_TEMPLATE_ROWS = [
    [-12.5, 7.8, -12.1],
    [-10.2, 8.4, -13.0],
    [-14.1, 6.9, -9.8],
]
TEMPLATE_INPUT_HEADER = [
    "Sample ID",
    "Input Parameter 1",
    "Input Parameter 2",
    "Input Parameter 3",
]
TEMPLATE_INPUT_ROWS: list[list[str | float]] = [
    ["Design_001", 0.0, 0.0, 0.0],
    ["Design_002", 0.5, 1.0, 1.5],
    ["Design_003", 1.0, 0.5, 2.0],
]
TEMPLATE_OUTPUT_HEADER = ["Sample ID", "Output 1", "Output 2", "Output 3"]
TEMPLATE_OUTPUT_ROWS: list[list[str | float]] = [
    ["Design_001", -12.5, 7.8, -12.1],
    ["Design_002", -10.2, 8.4, -13.0],
    ["Design_003", -14.1, 6.9, -9.8],
]
LEGACY_INPUT_TEMPLATE_HEADER = ["P1", "P2", "P3"]
LEGACY_INPUT_TEMPLATE_ROWS = [
    [0.0, 0.0, 0.0],
    [30.0, -30.0, 15.0],
    [60.0, 20.0, -45.0],
]
LEGACY_OUTPUT_TEMPLATE_HEADER = ["theta_-90", "theta_0", "theta_90"]
LEGACY_OUTPUT_TEMPLATE_ROWS = [
    [-12.5, 7.8, -12.1],
    [-10.2, 8.4, -13.0],
    [-14.1, 6.9, -9.8],
]
LEGACY_TEMPLATE_INSTRUCTIONS = (
    "ANTENNA SURROGATE STUDIO — INPUT/OUTPUT CSV TEMPLATE\n"
    "\n"
    "1. Keep one header row in each CSV.\n"
    "2. Put one simulation sample on each data row.\n"
    "3. The two files must have the same number of data rows.\n"
    "4. Every cell below the header must be numeric and finite.\n"
    "5. Input headers name design variables such as P1, P2, or Er.\n"
    "6. Output headers name response coordinates such as theta_-90.\n"
    "7. Row N in inputs must describe row N in outputs.\n"
    "\n"
    "You may rename these files. In Data Prep, choose Input + output "
    "files and point to both CSVs. The Studio loads the pair automatically.\n"
)
LEGACY_GENERIC_TEMPLATE_INSTRUCTIONS = (
    "ANTENNA SURROGATE STUDIO — GENERIC INPUT/OUTPUT CSV TEMPLATE\n"
    "\n"
    "These files are intentionally antenna- and solver-neutral.\n"
    "\n"
    "INPUT TABLE\n"
    "1. Rename Variable 1 and Variable 2 to your real design variables and "
    "include units when useful.\n"
    "2. Add or remove input columns as needed for the antenna being modeled.\n"
    "3. Put one complete simulation/design sample on each data row.\n"
    "\n"
    "OUTPUT TABLE\n"
    "4. Each output column is one response coordinate; each row contains the "
    "responses for the matching input sample.\n"
    "5. Rename the generic response headers to describe the quantity, axis "
    "value, and unit.\n"
    "6. Frequency-sweep example: S11 at 1 GHz, S11 at 5 GHz, "
    "S11 at 10 GHz (minimum to maximum frequency).\n"
    "7. Angular-sweep example: Gain at theta -90 deg, Gain at theta 0 deg, "
    "Gain at theta 90 deg (minimum to maximum theta).\n"
    "8. For a response over frequency and angle, create one column for each "
    "coordinate pair, for example Gain at 1 GHz / theta -90 deg.\n"
    "9. Scalar outputs are also valid, for example Efficiency (%) or "
    "Resonant frequency (GHz).\n"
    "\n"
    "PAIRING AND VALIDATION\n"
    "10. Keep one non-empty, unique header row in each CSV.\n"
    "11. The two files must contain the same number of data rows.\n"
    "12. Row N in inputs must describe the same sample as row N in outputs.\n"
    "13. Every cell below the headers must be numeric and finite.\n"
    "\n"
    "You may rename these files. In Data Prep, choose Input + output files "
    "and point to both CSVs. The Studio loads the pair automatically.\n"
)
TEMPLATE_INSTRUCTIONS = (
    "ANTENNA SURROGATE STUDIO — INPUT/OUTPUT CSV TEMPLATE\n"
    "\n"
    "These files are intentionally antenna- and solver-neutral.\n"
    "\n"
    "SAMPLE ID\n"
    "1. Keep Sample ID as the first column in both files.\n"
    "2. Use one unique ID per design, such as Design_001.\n"
    "3. The IDs must match in the same row order in both files. Sample ID is "
    "used to validate pairing, preserved for prediction traceability, and "
    "excluded from numeric model features and targets.\n"
    "\n"
    "INPUT TABLE\n"
    "4. Rename Input Parameter 1, 2, and 3 to the real design variables and "
    "include units when useful.\n"
    "5. Add or remove input-parameter columns as needed for the antenna.\n"
    "6. Put one complete simulation/design sample on each data row.\n"
    "\n"
    "OUTPUT TABLE\n"
    "7. Rename Output 1, 2, and 3 to the actual response coordinates.\n"
    "8. Each output column is one response coordinate; each row contains the "
    "responses for the matching input sample.\n"
    "9. Frequency example: S11 at 1 GHz, S11 at 5 GHz, S11 at 10 GHz.\n"
    "10. Angular example: Gain at theta -90 deg, Gain at theta 0 deg, "
    "Gain at theta 90 deg.\n"
    "11. For frequency and angle, use one column per coordinate pair, such as "
    "Gain at 1 GHz / theta -90 deg.\n"
    "12. Scalar outputs such as Efficiency (%) or Resonant frequency (GHz) "
    "are also valid.\n"
    "\n"
    "VALIDATION\n"
    "13. Keep one non-empty, unique header row in each CSV.\n"
    "14. The files must have the same Sample IDs and number of data rows.\n"
    "15. Every parameter and output cell must be numeric and finite.\n"
    "\n"
    "You may rename these files. In Data Prep, choose Input + output files "
    "and point to both CSVs. The Studio loads the pair automatically.\n"
)


class ParseError(RuntimeError):
    """Raised when source data is missing, malformed, or inconsistent."""


@dataclass(slots=True)
class DiscoveryResult:
    mode: str
    files: list[str]
    input_variables: list[str]
    output_variables: list[str]
    sample_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class PreparedResult:
    input_csv: str
    output_csv: str
    rows: int
    columns: int
    input_columns: int
    output_columns: int
    theta_points: int
    inputs: list[str]
    target_columns: list[str]
    output: str
    mode: str
    sample_id_column: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class CsvTable:
    headers: list[str]
    rows: list[list[float]]
    sample_ids: list[str] | None = None


@dataclass(frozen=True, slots=True)
class CoordinateAxis:
    """Meaning preserved from the first column of a #Parameters response table."""

    label: str
    unit: str | None = None


@dataclass(slots=True)
class TrainingRequest:
    """Validated configuration for a future model-training job.

    Validation is intentionally limited to the request itself. Constructing
    this object does not open, inspect, resolve, or require either CSV path.
    """

    input_csv_path: Path
    output_csv_path: Path
    feature_columns: list[str]
    target_columns: list[str]
    sample_id_column: str | None = None

    def __post_init__(self) -> None:
        if self.input_csv_path is None or not str(self.input_csv_path).strip():
            raise ValueError("A training input CSV path is required.")
        if self.output_csv_path is None or not str(self.output_csv_path).strip():
            raise ValueError("A training output CSV path is required.")
        if not self.feature_columns:
            raise ValueError("At least one feature column must be selected.")
        if any(
            not isinstance(column, str) or not column.strip()
            for column in self.feature_columns
        ):
            raise ValueError("Feature column names cannot be empty.")

        seen: set[str] = set()
        duplicates: list[str] = []
        for column in self.feature_columns:
            if column in seen and column not in duplicates:
                duplicates.append(column)
            seen.add(column)
        if duplicates:
            raise ValueError(
                "Duplicate feature columns were provided: "
                f"{', '.join(duplicates)}"
            )

        if not self.target_columns:
            raise ValueError("At least one target column must be selected.")
        if any(
            not isinstance(column, str) or not column.strip()
            for column in self.target_columns
        ):
            raise ValueError("Target column names cannot be empty.")

        seen_targets: set[str] = set()
        duplicate_targets: list[str] = []
        for column in self.target_columns:
            if column in seen_targets and column not in duplicate_targets:
                duplicate_targets.append(column)
            seen_targets.add(column)
        if duplicate_targets:
            raise ValueError(
                "Duplicate target columns were provided: "
                f"{', '.join(duplicate_targets)}"
            )

        feature_targets = [
            column
            for column in self.target_columns
            if column in self.feature_columns
        ]
        if feature_targets:
            raise ValueError(
                "Target columns cannot also be used as input features: "
                f"{', '.join(feature_targets)}"
            )
        if self.sample_id_column in self.feature_columns:
            raise ValueError(
                "The sample ID column cannot also be used as an input feature."
            )
        if (
            self.sample_id_column is not None
            and self.sample_id_column in self.target_columns
        ):
            raise ValueError(
                "The sample ID column cannot also be used as an output target."
            )


def find_text_files(path: str | Path) -> list[Path]:
    source = Path(path).expanduser()
    if source.is_file():
        if source.suffix.lower() != ".txt":
            raise ParseError("Choose a .txt simulation export.")
        return [source]
    if source.is_dir():
        files = sorted(
            (item for item in source.iterdir() if item.is_file() and item.suffix.lower() == ".txt"),
            key=lambda item: item.name.casefold(),
        )
        if not files:
            raise ParseError("No .txt simulation exports were found in that folder.")
        return files
    raise ParseError(f"The selected data path does not exist: {source}")


def sniff_data_format(path: str | Path, max_files: int = 8, max_lines: int = 60) -> str:
    files = find_text_files(path)
    filename_matches = sum(
        1 for file_path in files[:max_files] if FILENAME_PHASE_RE.search(file_path.name)
    )
    parameter_matches = 0
    for file_path in files[:max_files]:
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for index, line in enumerate(handle):
                if PARAM_BLOCK_RE.search(line):
                    parameter_matches += 1
                    break
                if index + 1 >= max_lines:
                    break
    if filename_matches and parameter_matches:
        raise ParseError(
            "The selection mixes filename sweeps and #Parameters data. "
            "Choose one source type or separate the exports."
        )
    if filename_matches:
        return "filename"
    if parameter_matches:
        return "parameters"
    return "unknown"


def discover(
    mode: str,
    path: str | Path,
    *,
    output_path: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> DiscoveryResult:
    if mode == "pair":
        if output_path is None:
            raise ParseError("Choose both an input CSV and an output CSV.")
        return discover_input_output_files(path, output_path, progress_callback)
    if mode == "filename":
        return discover_filename_format(path, progress_callback)
    if mode == "parameters":
        return discover_parameter_format(path, progress_callback)
    raise ParseError(f"Unsupported source mode: {mode}")


def prepare(
    mode: str,
    path: str | Path,
    selected_inputs: list[str],
    selected_output: str,
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    source_output_path: str | Path | None = None,
    phi_filter: float = 90.0,
    progress_callback: ProgressCallback | None = None,
) -> PreparedResult:
    if not selected_inputs:
        raise ParseError("Select at least one input variable.")
    if not selected_output:
        raise ParseError("Select one output variable.")
    if mode == "pair":
        if source_output_path is None:
            raise ParseError("Choose both an input CSV and an output CSV.")
        return prepare_from_input_output_files(
            path,
            source_output_path,
            selected_inputs,
            input_csv,
            output_csv,
            progress_callback=progress_callback,
        )
    if mode == "filename":
        return prepare_from_filename_format(
            path,
            selected_inputs,
            selected_output,
            input_csv,
            output_csv,
            phi_filter=phi_filter,
            progress_callback=progress_callback,
        )
    if mode == "parameters":
        return prepare_from_parameter_format(
            path,
            selected_inputs,
            selected_output,
            input_csv,
            output_csv,
            progress_callback=progress_callback,
        )
    raise ParseError(f"Unsupported source mode: {mode}")


def parse_float(value: str) -> float:
    number = float(str(value).strip())
    if not math.isfinite(number):
        raise ValueError("Values must be finite.")
    return number


def _read_numeric_csv(
    path: str | Path,
    label: str,
) -> tuple[list[str], list[list[float]]]:
    table = _read_csv_table(path, label, allow_sample_id=False)
    return table.headers, table.rows


def _read_csv_table(
    path: str | Path,
    label: str,
    *,
    allow_sample_id: bool,
) -> CsvTable:
    source = Path(path).expanduser()
    if not source.is_file():
        raise ParseError(f"The selected {label} CSV does not exist: {source}")
    if source.suffix.lower() != ".csv":
        raise ParseError(f"Choose a .csv file for the {label} table.")

    try:
        handle = source.open("r", newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise ParseError(f"The {label} CSV could not be opened: {exc}") from exc

    with handle:
        reader = csv.reader(handle)
        try:
            raw_header = next(reader)
        except StopIteration as exc:
            raise ParseError(f"The {label} CSV is empty.") from exc
        header = [item.strip() for item in raw_header]
        if not header or any(not item for item in header):
            raise ParseError(
                f"The {label} CSV header contains an empty column name."
            )
        normalized = [item.casefold() for item in header]
        if len(set(normalized)) != len(normalized):
            raise ParseError(
                f"The {label} CSV header contains duplicate column names."
            )

        sample_id_indexes = [
            index
            for index, item in enumerate(normalized)
            if item.replace("_", " ").replace("-", " ") == "sample id"
        ]
        if sample_id_indexes and not allow_sample_id:
            raise ParseError(
                f"The {label} CSV must contain numeric columns only."
            )
        if sample_id_indexes and sample_id_indexes != [0]:
            raise ParseError(
                f"Sample ID must be the first column in the {label} CSV."
            )
        has_sample_id = sample_id_indexes == [0]
        numeric_header = header[1:] if has_sample_id else header
        if not numeric_header:
            raise ParseError(
                f"The {label} CSV needs at least one numeric data column."
            )

        rows: list[list[float]] = []
        sample_ids: list[str] | None = [] if has_sample_id else None
        for line_number, raw_row in enumerate(reader, start=2):
            if not raw_row or all(not item.strip() for item in raw_row):
                continue
            if len(raw_row) != len(header):
                raise ParseError(
                    f"The {label} CSV row {line_number} has {len(raw_row)} "
                    f"columns; expected {len(header)}."
                )
            numeric_row = raw_row
            if has_sample_id:
                sample_id = raw_row[0].strip()
                if not sample_id:
                    raise ParseError(
                        f"The {label} CSV row {line_number} has an empty Sample ID."
                    )
                assert sample_ids is not None
                sample_ids.append(sample_id)
                numeric_row = raw_row[1:]
            try:
                rows.append([parse_float(item) for item in numeric_row])
            except ValueError as exc:
                raise ParseError(
                    f"The {label} CSV row {line_number} contains a non-numeric "
                    "or non-finite value."
                ) from exc

    if not rows:
        raise ParseError(f"The {label} CSV has no numeric sample rows.")
    if sample_ids is not None:
        normalized_ids = [item.casefold() for item in sample_ids]
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ParseError(
                f"The {label} CSV contains duplicate Sample ID values."
            )
    return CsvTable(numeric_header, rows, sample_ids)


def _read_input_output_pair(
    input_path: str | Path,
    output_path: str | Path,
) -> tuple[CsvTable, CsvTable]:
    input_table = _read_csv_table(
        input_path,
        "input",
        allow_sample_id=True,
    )
    output_table = _read_csv_table(
        output_path,
        "output",
        allow_sample_id=True,
    )
    if len(input_table.rows) != len(output_table.rows):
        raise ParseError(
            "Input and output CSVs must contain the same number of sample rows. "
            f"Found {len(input_table.rows)} input rows and "
            f"{len(output_table.rows)} output rows."
        )
    if (input_table.sample_ids is None) != (output_table.sample_ids is None):
        raise ParseError(
            "Use a Sample ID first column in both CSVs, or omit it from both."
        )
    if input_table.sample_ids is not None:
        assert output_table.sample_ids is not None
        for index, (input_id, output_id) in enumerate(
            zip(input_table.sample_ids, output_table.sample_ids),
            start=2,
        ):
            if input_id != output_id:
                raise ParseError(
                    f"Sample ID mismatch on CSV row {index}: input uses "
                    f"'{input_id}' but output uses '{output_id}'."
                )
    return input_table, output_table


def discover_input_output_files(
    input_path: str | Path,
    output_path: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> DiscoveryResult:
    input_table, output_table = _read_input_output_pair(input_path, output_path)
    if progress_callback:
        progress_callback(
            f"Validated {len(input_table.rows)} matched input/output samples."
        )
    return DiscoveryResult(
        mode="pair",
        files=[str(Path(input_path)), str(Path(output_path))],
        input_variables=input_table.headers,
        output_variables=output_table.headers,
        sample_count=len(input_table.rows),
    )


def prepare_from_input_output_files(
    source_input_csv: str | Path,
    source_output_csv: str | Path,
    selected_inputs: list[str],
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> PreparedResult:
    input_table, output_table = _read_input_output_pair(
        source_input_csv,
        source_output_csv,
    )
    missing = [item for item in selected_inputs if item not in input_table.headers]
    if missing:
        raise ParseError(
            f"Selected input(s) were not found: {', '.join(missing)}."
        )
    selected_indexes = [
        input_table.headers.index(item) for item in selected_inputs
    ]
    selected_rows = [
        [row[index] for index in selected_indexes] for row in input_table.rows
    ]
    _write_csv_pair(
        input_csv,
        output_csv,
        list(selected_inputs),
        output_table.headers,
        selected_rows,
        output_table.rows,
        sample_ids=input_table.sample_ids,
    )
    if progress_callback:
        progress_callback(
            f"Prepared {len(input_table.rows)} matched CSV samples."
        )
    return PreparedResult(
        input_csv=str(Path(input_csv)),
        output_csv=str(Path(output_csv)),
        rows=len(input_table.rows),
        columns=len(selected_inputs) + len(output_table.headers),
        input_columns=len(selected_inputs),
        output_columns=len(output_table.headers),
        theta_points=len(output_table.headers),
        inputs=list(selected_inputs),
        target_columns=list(output_table.headers),
        output=IMPORTED_OUTPUT_LABEL,
        mode="pair",
        sample_id_column=(
            "Sample ID" if input_table.sample_ids is not None else None
        ),
    )


def format_theta(theta: float) -> str:
    return str(int(theta)) if float(theta).is_integer() else f"{theta:g}"


def normalize_output_header(value: str) -> str:
    return re.sub(r"\s*\(\d+\)", "", value).strip()


def coordinate_axis_from_header(value: str) -> CoordinateAxis:
    """Describe a response-table coordinate without inventing physical meaning."""

    header = " ".join(str(value).strip().split())
    lowered = header.lower()
    if "freq" in lowered:
        label = "Frequency"
    elif "theta" in lowered:
        label = "Theta"
    elif "phi" in lowered:
        label = "Phi"
    else:
        label = re.sub(
            r"(?i)\s*(?:/|\[|\()\s*"
            r"(?:ghz|mhz|khz|hz|degrees?|deg|°|radians?|rad)"
            r"\s*(?:\]|\))?\s*$",
            "",
            header,
        ).strip(" /[]()")
        label = label or "Output coordinate"

    unit = next(
        (
            displayed
            for token, displayed in (
                ("ghz", "GHz"),
                ("mhz", "MHz"),
                ("khz", "kHz"),
                ("hz", "Hz"),
            )
            if re.search(rf"(?<![a-z]){token}(?![a-z])", lowered)
        ),
        None,
    )
    if unit is None and re.search(
        r"(?<![a-z])(?:degrees?|deg|°)(?![a-z])", lowered
    ):
        unit = "deg"
    if unit is None and re.search(
        r"(?<![a-z])(?:radians?|rad)(?![a-z])", lowered
    ):
        unit = "rad"
    return CoordinateAxis(label=label, unit=unit)


def parameter_target_columns(
    selected_output: str,
    coordinate_axis: CoordinateAxis,
    coordinate_values: Iterable[float],
) -> list[str]:
    """Build ordered, response-aware output names for a parsed sweep."""

    unit_suffix = f" {coordinate_axis.unit}" if coordinate_axis.unit else ""
    return [
        f"{selected_output} at {coordinate_axis.label} "
        f"{format_theta(value)}{unit_suffix}"
        for value in coordinate_values
    ]


def extract_filename_phase_variables(
    files: Iterable[Path],
) -> tuple[list[str], list[tuple[Path, dict[str, float]]]]:
    variables: list[str] = []
    rows: list[tuple[Path, dict[str, float]]] = []
    expected_count: int | None = None

    for file_path in files:
        matches = FILENAME_PHASE_RE.findall(file_path.name)
        if not matches:
            continue
        if expected_count is None:
            expected_count = len(matches)
        elif len(matches) != expected_count:
            raise ParseError(
                f"Inconsistent antenna-element count in filename: {file_path.name}. "
                f"Expected {expected_count}, found {len(matches)}."
            )

        row: dict[str, float] = {}
        for index, (amplitude, phase) in enumerate(matches, start=1):
            row[f"A{index}"] = parse_float(amplitude)
            row[f"P{index}"] = parse_float(phase)
        for column in row:
            if column not in variables:
                variables.append(column)
        rows.append((file_path, row))

    if not rows:
        raise ParseError(
            "No filename phase variables were found. "
            "Expected filenames containing tokens such as A0_phi-90."
        )
    return variables, rows


def discover_filename_format(
    path: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> DiscoveryResult:
    files = find_text_files(path)
    detected = sniff_data_format(path)
    if detected == "parameters":
        raise ParseError("This source contains #Parameters sweeps. Switch the source type.")
    variables, rows = extract_filename_phase_variables(files)
    outputs = discover_table_outputs(rows[0][0])
    if progress_callback:
        progress_callback(f"Discovered {len(rows)} filename-sweep samples.")
    return DiscoveryResult(
        mode="filename",
        files=[str(item) for item in files],
        input_variables=variables,
        output_variables=outputs,
        sample_count=len(rows),
    )


def discover_parameter_format(
    path: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> DiscoveryResult:
    files = find_text_files(path)
    detected = sniff_data_format(path)
    if detected == "filename":
        raise ParseError("This source uses filename phase sweeps. Switch the source type.")

    input_variables: list[str] = []
    output_variables: list[str] = []
    sample_count = 0
    for file_number, file_path in enumerate(files, start=1):
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                params = parse_parameter_line(line)
                if params:
                    sample_count += 1
                    for key in params:
                        if key not in input_variables:
                            input_variables.append(key)
                headers = parse_quoted_header(line)
                if len(headers) > 1:
                    for header in (normalize_output_header(item) for item in headers[1:]):
                        if header not in output_variables:
                            output_variables.append(header)
        if progress_callback:
            progress_callback(f"Scanned {file_number}/{len(files)} source file(s).")

    if not input_variables:
        raise ParseError("No #Parameters blocks were found.")
    if not output_variables:
        raise ParseError("No quoted output header was found after a #Parameters block.")
    return DiscoveryResult(
        mode="parameters",
        files=[str(item) for item in files],
        input_variables=input_variables,
        output_variables=output_variables,
        sample_count=sample_count,
    )


def parse_parameter_line(line: str) -> dict[str, float] | None:
    match = PARAM_BLOCK_RE.search(line)
    if not match:
        return None
    params: dict[str, float] = {}
    invalid: list[str] = []
    for part in match.group(1).split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        if not key:
            continue
        try:
            params[key] = parse_float(value)
        except ValueError:
            invalid.append(key)
    if invalid:
        raise ParseError(f"Non-numeric parameter value(s): {', '.join(invalid)}.")
    return params


def parse_quoted_header(line: str) -> list[str]:
    if not line.lstrip().startswith("#"):
        return []
    return QUOTED_HEADER_RE.findall(line)


def discover_table_outputs(file_path: str | Path) -> list[str]:
    with Path(file_path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lower_line = line.lower()
            if "theta" in lower_line and "phi" in lower_line and "abs(" in lower_line:
                return [
                    "Abs(Grlz)",
                    "Abs(Theta)",
                    "Phase(Theta)",
                    "Abs(Phi)",
                    "Phase(Phi)",
                    "Ax.Ratio",
                ]
            headers = parse_quoted_header(line)
            if len(headers) > 1:
                return [normalize_output_header(item) for item in headers[1:]]
    raise ParseError(f"Could not discover output columns in {file_path}.")


def output_index_for_table(output_name: str) -> int | None:
    return {
        "Abs(Grlz)": 2,
        "Abs(Theta)": 3,
        "Phase(Theta)": 4,
        "Abs(Phi)": 5,
        "Phase(Phi)": 6,
        "Ax.Ratio": 7,
    }.get(output_name)


def read_filename_table_output(
    file_path: str | Path,
    output_name: str,
    phi_filter: float = 90.0,
) -> tuple[list[float], list[float]]:
    column_index = output_index_for_table(output_name)
    if column_index is None:
        raise ParseError(f"Unsupported filename-table output: {output_name}")

    theta_values: list[float] = []
    output_values: list[float] = []
    found_filter_block = False
    with Path(file_path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) <= column_index:
                continue
            try:
                theta = parse_float(parts[0])
                phi = parse_float(parts[1])
                value = parse_float(parts[column_index])
            except ValueError:
                continue
            if math.isclose(phi, phi_filter, rel_tol=0.0, abs_tol=1e-6):
                found_filter_block = True
                theta_values.append(theta)
                output_values.append(value)
            elif found_filter_block:
                break
    if not theta_values:
        raise ParseError(f"No Phi = {phi_filter:g} rows were found in {file_path}.")
    return theta_values, output_values


def prepare_from_filename_format(
    path: str | Path,
    selected_inputs: list[str],
    selected_output: str,
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    phi_filter: float = 90.0,
    progress_callback: ProgressCallback | None = None,
) -> PreparedResult:
    files = find_text_files(path)
    if sniff_data_format(path) == "parameters":
        raise ParseError("This source contains #Parameters sweeps. Switch the source type.")
    available, phase_rows = extract_filename_phase_variables(files)
    missing_selected = [item for item in selected_inputs if item not in available]
    if missing_selected:
        raise ParseError(f"Selected input(s) were not found: {', '.join(missing_selected)}.")

    phase_rows.sort(
        key=lambda item: tuple(item[1][column] for column in selected_inputs)
    )
    rows: list[list[float]] = []
    theta_reference: list[float] | None = None
    for index, (file_path, phase_row) in enumerate(phase_rows, start=1):
        absent = [column for column in selected_inputs if column not in phase_row]
        if absent:
            raise ParseError(
                f"{file_path.name} is missing selected input(s): {', '.join(absent)}."
            )
        theta_values, output_values = read_filename_table_output(
            file_path, selected_output, phi_filter
        )
        theta_reference = _accept_theta_grid(theta_reference, theta_values, file_path)
        rows.append([phase_row[column] for column in selected_inputs] + output_values)
        if progress_callback and (
            index == 1 or index % 100 == 0 or index == len(phase_rows)
        ):
            progress_callback(f"Prepared {index}/{len(phase_rows)} sweep files.")

    _write_training_csvs(
        input_csv,
        output_csv,
        selected_inputs,
        theta_reference or [],
        rows,
    )
    target_columns = [
        f"theta_{format_theta(theta)}" for theta in (theta_reference or [])
    ]
    return PreparedResult(
        input_csv=str(Path(input_csv)),
        output_csv=str(Path(output_csv)),
        rows=len(rows),
        columns=len(selected_inputs) + len(theta_reference or []),
        input_columns=len(selected_inputs),
        output_columns=len(theta_reference or []),
        theta_points=len(theta_reference or []),
        inputs=list(selected_inputs),
        target_columns=target_columns,
        output=selected_output,
        mode="filename",
    )


def prepare_from_parameter_format(
    path: str | Path,
    selected_inputs: list[str],
    selected_output: str,
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> PreparedResult:
    files = find_text_files(path)
    if sniff_data_format(path) == "filename":
        raise ParseError("This source uses filename phase sweeps. Switch the source type.")

    rows: list[list[float]] = []
    coordinate_reference: list[float] | None = None
    coordinate_axis: CoordinateAxis | None = None
    for file_path in files:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        index = 0
        while index < len(lines):
            params = parse_parameter_line(lines[index])
            if not params:
                index += 1
                continue
            missing = [column for column in selected_inputs if column not in params]
            if missing:
                raise ParseError(
                    f"{file_path.name} is missing selected input(s): {', '.join(missing)}."
                )

            header_index = index + 1
            while header_index < len(lines):
                if parse_parameter_line(lines[header_index]):
                    raise ParseError(
                        f"No output header was found for a #Parameters block in {file_path.name}."
                    )
                if parse_quoted_header(lines[header_index]):
                    break
                header_index += 1
            if header_index >= len(lines):
                raise ParseError(
                    f"No output header was found after #Parameters in {file_path.name}."
                )

            headers = parse_quoted_header(lines[header_index])
            current_axis = coordinate_axis_from_header(headers[0])
            if coordinate_axis is None:
                coordinate_axis = current_axis
            elif (
                coordinate_axis.label.casefold() != current_axis.label.casefold()
                or (coordinate_axis.unit or "").casefold()
                != (current_axis.unit or "").casefold()
            ):
                raise ParseError(
                    f"Coordinate header mismatch in {file_path.name}: expected "
                    f"'{coordinate_axis.label}' but found '{current_axis.label}'."
                )
            normalized = [headers[0]] + [
                normalize_output_header(item) for item in headers[1:]
            ]
            if selected_output not in normalized[1:]:
                raise ParseError(
                    f"Output '{selected_output}' was not found in {file_path.name}."
                )
            output_index = normalized.index(selected_output)

            data_index = header_index + 1
            while data_index < len(lines) and lines[data_index].lstrip().startswith("#"):
                data_index += 1
            coordinate_values: list[float] = []
            output_values: list[float] = []
            while data_index < len(lines):
                parts = lines[data_index].split()
                if not parts or parts[0].startswith("#"):
                    break
                try:
                    coordinate_values.append(parse_float(parts[0]))
                    output_values.append(parse_float(parts[output_index]))
                except (ValueError, IndexError) as exc:
                    raise ParseError(
                        f"Invalid table row {data_index + 1} in {file_path.name}."
                    ) from exc
                data_index += 1
            if not coordinate_values:
                raise ParseError(f"No numeric output rows followed a block in {file_path.name}.")

            coordinate_reference = _accept_coordinate_grid(
                coordinate_reference,
                coordinate_values,
                file_path,
                coordinate_axis.label,
            )
            rows.append([params[column] for column in selected_inputs] + output_values)
            if progress_callback and (len(rows) == 1 or len(rows) % 100 == 0):
                progress_callback(f"Prepared {len(rows)} #Parameters samples.")
            index = max(data_index, index + 1)

    if not rows:
        raise ParseError("No training rows were prepared.")
    if coordinate_axis is None:
        raise ParseError("No response coordinate header was found.")
    target_columns = parameter_target_columns(
        selected_output,
        coordinate_axis,
        coordinate_reference or [],
    )
    _write_training_csvs(
        input_csv,
        output_csv,
        selected_inputs,
        coordinate_reference or [],
        rows,
        output_header=target_columns,
    )
    return PreparedResult(
        input_csv=str(Path(input_csv)),
        output_csv=str(Path(output_csv)),
        rows=len(rows),
        columns=len(selected_inputs) + len(coordinate_reference or []),
        input_columns=len(selected_inputs),
        output_columns=len(coordinate_reference or []),
        theta_points=len(coordinate_reference or []),
        inputs=list(selected_inputs),
        target_columns=target_columns,
        output=selected_output,
        mode="parameters",
    )


def _accept_theta_grid(
    reference: list[float] | None,
    values: list[float],
    file_path: Path,
) -> list[float]:
    return _accept_coordinate_grid(reference, values, file_path, "Theta")


def _accept_coordinate_grid(
    reference: list[float] | None,
    values: list[float],
    file_path: Path,
    coordinate_label: str,
) -> list[float]:
    if reference is None:
        return values
    matches = len(values) == len(reference) and all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=1e-7)
        for left, right in zip(reference, values)
    )
    if not matches:
        raise ParseError(f"{coordinate_label} grid mismatch in {file_path.name}.")
    return reference


def _write_training_csvs(
    input_csv: str | Path,
    output_csv: str | Path,
    selected_inputs: list[str],
    theta_values: list[float],
    rows: list[list[float]],
    *,
    output_header: list[str] | None = None,
) -> None:
    input_destination = Path(input_csv)
    output_destination = Path(output_csv)
    if input_destination.resolve() == output_destination.resolve():
        raise ParseError("Input and output tables must use different file paths.")

    input_count = len(selected_inputs)
    output_count = len(theta_values)
    expected_columns = input_count + output_count
    for index, row in enumerate(rows, start=1):
        if len(row) != expected_columns:
            raise ParseError(
                f"Prepared row {index} has {len(row)} columns; "
                f"expected {expected_columns}."
            )

    input_rows = [row[:input_count] for row in rows]
    output_rows = [row[input_count:] for row in rows]
    resolved_output_header = output_header or [
        f"theta_{format_theta(theta)}" for theta in theta_values
    ]
    if len(resolved_output_header) != output_count:
        raise ParseError("The prepared output header does not match the response grid.")
    _write_csv_pair(
        input_csv,
        output_csv,
        list(selected_inputs),
        resolved_output_header,
        input_rows,
        output_rows,
    )


def _write_csv_pair(
    input_csv: str | Path,
    output_csv: str | Path,
    input_header: list[str],
    output_header: list[str],
    input_rows: list[list[str | float]],
    output_rows: list[list[str | float]],
    *,
    sample_ids: list[str] | None = None,
) -> None:
    if len(input_rows) != len(output_rows):
        raise ParseError(
            "Input and output tables must contain the same number of rows."
        )
    if sample_ids is not None and len(sample_ids) != len(input_rows):
        raise ParseError(
            "Sample IDs must contain one value for every prepared sample row."
        )
    for label, header, table_rows in (
        ("input", input_header, input_rows),
        ("output", output_header, output_rows),
    ):
        for index, row in enumerate(table_rows, start=1):
            if len(row) != len(header):
                raise ParseError(
                    f"Prepared {label} row {index} has {len(row)} columns; "
                    f"expected {len(header)}."
                )

    input_destination = Path(input_csv)
    output_destination = Path(output_csv)
    if input_destination.resolve() == output_destination.resolve():
        raise ParseError("Input and output tables must use different file paths.")

    staged: list[tuple[Path, Path]] = []
    prepared_input_header: list[str] = list(input_header)
    prepared_output_header: list[str] = list(output_header)
    prepared_input_rows: list[list[str | float]] = [list(row) for row in input_rows]
    prepared_output_rows: list[list[str | float]] = [list(row) for row in output_rows]
    if sample_ids is not None:
        prepared_input_header.insert(0, "Sample ID")
        prepared_output_header.insert(0, "Sample ID")
        prepared_input_rows = [
            [sample_id, *row]
            for sample_id, row in zip(sample_ids, prepared_input_rows)
        ]
        prepared_output_rows = [
            [sample_id, *row]
            for sample_id, row in zip(sample_ids, prepared_output_rows)
        ]
    try:
        staged.append(
            _stage_csv(
                input_destination,
                prepared_input_header,
                prepared_input_rows,
            )
        )
        staged.append(
            _stage_csv(
                output_destination,
                prepared_output_header,
                prepared_output_rows,
            )
        )
        for temp_path, destination in staged:
            os.replace(temp_path, destination)
    except Exception:
        for temp_path, _destination in staged:
            temp_path.unlink(missing_ok=True)
        raise


def _stage_csv(
    destination: Path,
    header: list[str],
    rows: list[list[str | float]],
) -> tuple[Path, Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        delete=False,
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return temp_path, destination
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def write_input_output_templates(
    destination_folder: str | Path,
) -> tuple[Path, Path, Path]:
    folder = Path(destination_folder)
    folder.mkdir(parents=True, exist_ok=True)
    input_template = folder / "inputs_template.csv"
    output_template = folder / "outputs_template.csv"
    instructions = folder / "README.txt"

    input_needs_write = (
        not input_template.exists()
        or _matches_numeric_template(
            input_template,
            LEGACY_INPUT_TEMPLATE_HEADER,
            LEGACY_INPUT_TEMPLATE_ROWS,
        )
        or _matches_numeric_template(
            input_template,
            LEGACY_GENERIC_INPUT_TEMPLATE_HEADER,
            LEGACY_GENERIC_INPUT_TEMPLATE_ROWS,
        )
    )
    output_needs_write = (
        not output_template.exists()
        or _matches_numeric_template(
            output_template,
            LEGACY_OUTPUT_TEMPLATE_HEADER,
            LEGACY_OUTPUT_TEMPLATE_ROWS,
        )
        or _matches_numeric_template(
            output_template,
            LEGACY_GENERIC_OUTPUT_TEMPLATE_HEADER,
            LEGACY_GENERIC_OUTPUT_TEMPLATE_ROWS,
        )
    )

    if input_needs_write and output_needs_write:
        _write_csv_pair(
            input_template,
            output_template,
            TEMPLATE_INPUT_HEADER,
            TEMPLATE_OUTPUT_HEADER,
            TEMPLATE_INPUT_ROWS,
            TEMPLATE_OUTPUT_ROWS,
        )
    elif input_needs_write:
        _stage_and_replace_csv(
            input_template,
            TEMPLATE_INPUT_HEADER,
            TEMPLATE_INPUT_ROWS,
        )
    elif output_needs_write:
        _stage_and_replace_csv(
            output_template,
            TEMPLATE_OUTPUT_HEADER,
            TEMPLATE_OUTPUT_ROWS,
        )

    instructions_are_legacy = False
    if instructions.exists():
        try:
            existing_instructions = instructions.read_text(encoding="utf-8")
            instructions_are_legacy = existing_instructions in {
                LEGACY_TEMPLATE_INSTRUCTIONS,
                LEGACY_GENERIC_TEMPLATE_INSTRUCTIONS,
            } or (
                existing_instructions.startswith("ANTENNA SURROGATE STUDIO")
                and "select Analyze pair" in existing_instructions
            )
        except OSError:
            instructions_are_legacy = False
    if not instructions.exists() or instructions_are_legacy:
        _atomic_write_text(instructions, TEMPLATE_INSTRUCTIONS)
    return input_template, output_template, instructions


def _matches_numeric_template(
    path: Path,
    expected_header: list[str],
    expected_rows: list[list[float]],
) -> bool:
    if not path.exists():
        return False
    try:
        header, rows = _read_numeric_csv(path, "template")
    except (OSError, ParseError):
        return False
    return header == expected_header and rows == expected_rows


def _stage_and_replace_csv(
    destination: Path,
    header: list[str],
    rows: list[list[str | float]],
) -> None:
    temp_path, final_path = _stage_csv(destination, header, rows)
    try:
        os.replace(temp_path, final_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _atomic_write_text(destination: Path, content: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.write(content)
        os.replace(temp_path, destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
