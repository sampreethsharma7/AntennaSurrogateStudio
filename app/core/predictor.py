from pathlib import Path
import json

import joblib
import pandas as pd

from app.utils.versioning import utc_timestamp


def load_active_model(project_dir: Path, manifest):
    version = manifest.active_model_version
    if not version:
        raise ValueError("No active model has been trained yet.")
    model = joblib.load(project_dir / "models" / f"model_v{version}.joblib")
    metadata = json.loads((project_dir / "models" / f"model_v{version}_metadata.json").read_text(encoding="utf-8"))
    return model, metadata


def predict_dataframe(project_dir: Path, manifest, input_df: pd.DataFrame) -> pd.DataFrame:
    model, metadata = load_active_model(project_dir, manifest)
    x = input_df[metadata["input_columns"]]
    y = model.predict(x)
    output_df = pd.DataFrame(y, columns=metadata["output_columns"])
    result = pd.concat([input_df.reset_index(drop=True), output_df], axis=1)
    prediction_dir = project_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    latest = prediction_dir / "latest_prediction.json"
    latest.write_text(result.to_json(orient="records", indent=2), encoding="utf-8")
    history_path = prediction_dir / "prediction_history.csv"
    run = result.copy()
    run.insert(0, "prediction_timestamp", utc_timestamp())
    if history_path.exists():
        run.to_csv(history_path, index=False, mode="a", header=False)
    else:
        run.to_csv(history_path, index=False)
    return result


def extrapolation_warnings(inputs: dict, manifest) -> list[str]:
    warnings = []
    for col, value in inputs.items():
        bounds = manifest.input_bounds.get(col)
        if bounds and (value < bounds["min"] or value > bounds["max"]):
            warnings.append(f"{col} = {value} is outside the training range of {bounds['min']} to {bounds['max']}. The prediction is extrapolating and may be unreliable.")
    return warnings
