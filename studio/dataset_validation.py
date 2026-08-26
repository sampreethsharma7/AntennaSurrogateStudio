"""Validation for paired input/output datasets used by model training.

This stage reads and validates the files described by ``TrainingRequest``.
It does not copy data, write reports, register datasets, or train models.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from studio.parser_engine import TrainingRequest


class DatasetValidationError(RuntimeError):
    """Raised when a requested training dataset is not usable."""


@dataclass(slots=True)
class DatasetValidationResult:
    """Verified dataset metadata passed to the future registration stage."""

    input_csv_path: Path
    output_csv_path: Path
    feature_columns: list[str]
    target_columns: list[str]
    sample_id_column: str | None
    sample_count: int
    feature_count: int
    target_count: int

    def to_dict(self) -> dict:
        return {
            "input_csv_path": str(self.input_csv_path),
            "output_csv_path": str(self.output_csv_path),
            "feature_columns": list(self.feature_columns),
            "target_columns": list(self.target_columns),
            "sample_id_column": self.sample_id_column,
            "sample_count": self.sample_count,
            "feature_count": self.feature_count,
            "target_count": self.target_count,
        }


@dataclass(slots=True)
class _ValidationTable:
    path: Path
    headers: list[str]
    rows: list[list[str]]


def validate_dataset(request: TrainingRequest) -> DatasetValidationResult:
    """Validate the two CSVs and selected columns in ``request``.

    Successful validation returns metadata only. Invalid datasets raise
    ``DatasetValidationError`` with a user-facing explanation.
    """

    input_table = _read_table(request.input_csv_path, "input")
    output_table = _read_table(request.output_csv_path, "output")

    missing_features = [
        column
        for column in request.feature_columns
        if column not in input_table.headers
    ]
    if missing_features:
        raise DatasetValidationError(
            "Selected input feature columns were not found: "
            f"{', '.join(missing_features)}."
        )

    missing_targets = [
        column
        for column in request.target_columns
        if column not in output_table.headers
    ]
    if missing_targets:
        raise DatasetValidationError(
            "Selected output target columns were not found: "
            f"{', '.join(missing_targets)}."
        )

    if len(input_table.rows) != len(output_table.rows):
        raise DatasetValidationError(
            "Input and output CSVs must contain the same number of sample rows. "
            f"Found {len(input_table.rows)} input rows and "
            f"{len(output_table.rows)} output rows."
        )

    _validate_numeric_columns(
        input_table,
        request.feature_columns,
        table_label="input",
        column_label="feature",
    )
    _validate_numeric_columns(
        output_table,
        request.target_columns,
        table_label="output",
        column_label="target",
    )

    if request.sample_id_column is not None:
        _validate_sample_ids(
            input_table,
            output_table,
            request.sample_id_column,
        )

    return DatasetValidationResult(
        input_csv_path=input_table.path,
        output_csv_path=output_table.path,
        feature_columns=list(request.feature_columns),
        target_columns=list(request.target_columns),
        sample_id_column=request.sample_id_column,
        sample_count=len(input_table.rows),
        feature_count=len(request.feature_columns),
        target_count=len(request.target_columns),
    )


def read_dataset_columns(
    input_csv_path: str | Path,
    output_csv_path: str | Path,
) -> tuple[list[str], list[str]]:
    """Read paired CSV headers for legacy prepared-project migration."""

    input_table = _read_table(Path(input_csv_path), "input")
    output_table = _read_table(Path(output_csv_path), "output")
    return list(input_table.headers), list(output_table.headers)


def _read_table(path: Path, label: str) -> _ValidationTable:
    source = Path(path).expanduser()
    if not source.is_file():
        raise DatasetValidationError(
            f"The training {label} CSV does not exist: {source}"
        )
    if source.suffix.lower() != ".csv":
        raise DatasetValidationError(
            f"The training {label} file must be a .csv file: {source}"
        )

    try:
        handle = source.open("r", newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise DatasetValidationError(
            f"The training {label} CSV could not be opened: {exc}"
        ) from exc

    with handle:
        reader = csv.reader(handle)
        try:
            raw_headers = next(reader)
        except StopIteration as exc:
            raise DatasetValidationError(
                f"The training {label} CSV is empty."
            ) from exc

        headers = [column.strip() for column in raw_headers]
        if not headers or any(not column for column in headers):
            raise DatasetValidationError(
                f"The training {label} CSV header contains an empty column name."
            )
        if len({column.casefold() for column in headers}) != len(headers):
            raise DatasetValidationError(
                f"The training {label} CSV header contains duplicate column names."
            )

        rows: list[list[str]] = []
        for row_number, row in enumerate(reader, start=2):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != len(headers):
                raise DatasetValidationError(
                    f"The training {label} CSV row {row_number} has "
                    f"{len(row)} columns; expected {len(headers)}."
                )
            rows.append(row)

    if not rows:
        raise DatasetValidationError(
            f"The training {label} CSV has no sample rows."
        )
    return _ValidationTable(path=source, headers=headers, rows=rows)


def _validate_numeric_columns(
    table: _ValidationTable,
    columns: list[str],
    *,
    table_label: str,
    column_label: str,
) -> None:
    indexes = [(column, table.headers.index(column)) for column in columns]
    for row_number, row in enumerate(table.rows, start=2):
        for column, index in indexes:
            raw_value = row[index].strip()
            if not raw_value:
                raise DatasetValidationError(
                    f"The training {table_label} CSV row {row_number} has an "
                    f"empty value in {column_label} column '{column}'."
                )
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise DatasetValidationError(
                    f"The training {table_label} CSV row {row_number} has a "
                    f"non-numeric value in {column_label} column '{column}'."
                ) from exc
            if not math.isfinite(value):
                raise DatasetValidationError(
                    f"The training {table_label} CSV row {row_number} has a "
                    f"non-finite value in {column_label} column '{column}'."
                )


def _validate_sample_ids(
    input_table: _ValidationTable,
    output_table: _ValidationTable,
    sample_id_column: str,
) -> None:
    if (
        sample_id_column not in input_table.headers
        or sample_id_column not in output_table.headers
    ):
        raise DatasetValidationError(
            f"The sample ID column '{sample_id_column}' must be present in both "
            "the input and output CSVs."
        )

    input_index = input_table.headers.index(sample_id_column)
    output_index = output_table.headers.index(sample_id_column)
    input_ids = [row[input_index].strip() for row in input_table.rows]
    output_ids = [row[output_index].strip() for row in output_table.rows]

    if any(not sample_id for sample_id in input_ids):
        raise DatasetValidationError(
            f"The input CSV contains an empty '{sample_id_column}' value."
        )
    if any(not sample_id for sample_id in output_ids):
        raise DatasetValidationError(
            f"The output CSV contains an empty '{sample_id_column}' value."
        )
    if len({sample_id.casefold() for sample_id in input_ids}) != len(input_ids):
        raise DatasetValidationError(
            f"The input CSV contains duplicate '{sample_id_column}' values."
        )
    if len({sample_id.casefold() for sample_id in output_ids}) != len(output_ids):
        raise DatasetValidationError(
            f"The output CSV contains duplicate '{sample_id_column}' values."
        )

    for row_number, (input_id, output_id) in enumerate(
        zip(input_ids, output_ids),
        start=2,
    ):
        if input_id != output_id:
            raise DatasetValidationError(
                f"Sample ID mismatch on CSV row {row_number}: input uses "
                f"'{input_id}' but output uses '{output_id}'."
            )
