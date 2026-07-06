from pathlib import Path
import hashlib
import json
import platform
import time

import joblib
import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from app import APP_VERSION, PROJECT_SCHEMA_VERSION
from app.core.evaluator import evaluate_predictions
from app.utils.versioning import next_model_version, utc_timestamp


DEFAULT_HYPERPARAMETERS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "min_child_weight": 1,
    "objective": "reg:squarederror",
}


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    data = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(data.tobytes()).hexdigest()


def train_xgboost_model(project_dir: Path, manifest, settings: dict, progress=None) -> dict:
    progress = progress or (lambda message: None)
    progress("Loading dataset")
    df = pd.read_csv(project_dir / "data" / "prepared_training_data.csv")
    x = df[manifest.selected_input_columns]
    y = df[manifest.selected_output_columns]
    seed = int(settings.get("random_seed", 42))
    test_split = float(settings.get("test_split", 0.2))
    params = {**DEFAULT_HYPERPARAMETERS, **settings.get("hyperparameters", {})}
    params["random_state"] = seed
    progress("Splitting train/test data")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, random_state=seed)
    progress("Training XGBoost outputs")
    model = MultiOutputRegressor(XGBRegressor(**params))
    start = time.time()
    model.fit(x_train, y_train)
    training_time = time.time() - start
    progress("Running validation")
    predictions = model.predict(x_test)
    metrics = evaluate_predictions(y_test, predictions, manifest.selected_output_columns)
    metrics["training_time_seconds"] = float(training_time)
    metrics["training_samples"] = int(len(x_train))
    metrics["test_samples"] = int(len(x_test))
    cv_folds = int(settings.get("cross_validation_folds", 0) or 0)
    if cv_folds > 1:
        scores = cross_val_score(model, x, y, cv=cv_folds, scoring="neg_root_mean_squared_error")
        metrics["cross_validation_rmse"] = [float(abs(v)) for v in scores]
    version = next_model_version(manifest.model_versions)
    model_id = f"xgb_{utc_timestamp().replace(':', '').replace('-', '').replace('Z', '')}_{version:03d}"
    model_path = project_dir / "models" / f"model_v{version}.joblib"
    metadata_path = project_dir / "models" / f"model_v{version}_metadata.json"
    progress("Saving artifacts")
    joblib.dump(model, model_path)
    metadata = {
        "model_id": model_id,
        "model_version": version,
        "app_version": APP_VERSION,
        "project_schema_version": PROJECT_SCHEMA_VERSION,
        "model_type": "xgboost_multioutput",
        "input_columns": manifest.selected_input_columns,
        "output_columns": manifest.selected_output_columns,
        "input_units": manifest.input_units,
        "output_units": manifest.output_units,
        "training_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "hyperparameters": params,
        "test_split": test_split,
        "random_seed": seed,
        "xgboost_version": xgboost.__version__,
        "scikit_learn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "training_timestamp": utc_timestamp(),
        "training_data_fingerprint": dataframe_fingerprint(df),
        "data_shape": list(df.shape),
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (project_dir / "models" / "active_model.json").write_text(json.dumps({"active_model_version": version}, indent=2), encoding="utf-8")
    (project_dir / "analysis" / f"training_report_v{version}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame([{"output": k, "rmse": v, "mae": metadata["metrics"]["per_output_mae"][k]} for k, v in metadata["metrics"]["per_output_rmse"].items()]).to_csv(project_dir / "analysis" / f"metrics_v{version}.csv", index=False)
    manifest.model_versions.append({"version": version, "model_id": model_id, "model_type": "XGBoost", "metadata_path": str(metadata_path.relative_to(project_dir)), "overall_rmse": metrics["overall_rmse"], "overall_r2": metrics["overall_r2"], "notes": ""})
    manifest.active_model_version = version
    manifest.save(project_dir)
    progress("Training complete")
    return metadata
