import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_predictions(y_true, y_pred, output_columns: list[str]) -> dict:
    err = np.asarray(y_pred) - np.asarray(y_true)
    sample_rmse = np.sqrt(np.mean(err ** 2, axis=1))
    per_output_rmse = np.sqrt(np.mean(err ** 2, axis=0))
    per_output_mae = np.mean(np.abs(err), axis=0)
    return {
        "overall_rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "overall_mae": float(mean_absolute_error(y_true, y_pred)),
        "overall_r2": float(r2_score(y_true, y_pred, multioutput="variance_weighted")),
        "per_output_rmse": dict(zip(output_columns, per_output_rmse.astype(float))),
        "per_output_mae": dict(zip(output_columns, per_output_mae.astype(float))),
        "per_sample_rmse": sample_rmse.astype(float).tolist(),
        "p95_sample_rmse": float(np.percentile(sample_rmse, 95)),
        "maximum_error": float(np.max(np.abs(err))),
    }
