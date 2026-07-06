import numpy as np
import pandas as pd
from typing import Optional


def validate_dataset(df: pd.DataFrame, input_columns: list[str], output_columns: list[str], sample_id: Optional[str] = None) -> list[dict]:
    findings: list[dict] = []
    selected = input_columns + output_columns
    if df.columns.duplicated().any():
        findings.append({"level": "error", "message": "Duplicate column names were found."})
    missing_cols = [c for c in selected if c not in df.columns]
    if missing_cols:
        findings.append({"level": "error", "message": f"Missing selected columns: {', '.join(missing_cols)}"})
        return findings
    if sample_id and sample_id in df.columns and df[sample_id].duplicated().any():
        findings.append({"level": "error", "message": f"Duplicate sample IDs were found in {sample_id}."})
    if len(df) < 50:
        findings.append({"level": "warning", "message": f"Only {len(df)} samples are available. Validation may be unreliable."})
    missing = df[selected].isna().sum()
    for col, count in missing[missing > 0].items():
        findings.append({"level": "error", "message": f"{col} has {int(count)} missing values."})
    for col in selected:
        if not pd.api.types.is_numeric_dtype(df[col]):
            findings.append({"level": "error", "message": f"{col} is not numeric."})
    for col in input_columns:
        if col in df.columns and df[col].nunique(dropna=True) <= 1:
            findings.append({"level": "warning", "message": f"{col} is constant and may not improve the model."})
    if input_columns and df.duplicated(subset=input_columns).any():
        findings.append({"level": "warning", "message": "Duplicate design samples were found. Check for inconsistent outputs."})
    numeric = df[selected].select_dtypes(include=[np.number])
    for col in numeric.columns:
        std = numeric[col].std()
        if std and np.isfinite(std):
            z = ((numeric[col] - numeric[col].mean()).abs() / std).max()
            if z > 6:
                findings.append({"level": "warning", "message": f"{col} contains a large outlier."})
    if not findings:
        findings.append({"level": "info", "message": "Dataset validation passed."})
    return findings
