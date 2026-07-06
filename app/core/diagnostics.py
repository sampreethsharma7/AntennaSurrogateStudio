import pandas as pd


def compute_diagnostics(df: pd.DataFrame, input_columns: list[str], output_columns: list[str], test_split: float = 0.2) -> dict:
    selected = input_columns + output_columns
    train_count = int(round(len(df) * (1 - test_split)))
    stats = df[selected].describe().transpose()[["min", "max", "mean", "std"]].round(6).to_dict("index")
    return {
        "total_samples": int(len(df)),
        "input_variable_count": len(input_columns),
        "output_variable_count": len(output_columns),
        "missing_value_count": int(df[selected].isna().sum().sum()),
        "duplicate_sample_count": int(df.duplicated(subset=input_columns).sum()) if input_columns else 0,
        "estimated_train_count": train_count,
        "estimated_test_count": len(df) - train_count,
        "summary_statistics": stats,
        "input_correlation": df[input_columns].corr(numeric_only=True).round(4).to_dict() if input_columns else {},
    }
