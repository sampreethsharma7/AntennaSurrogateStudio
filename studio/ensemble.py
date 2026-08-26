"""Reusable weighted-ensemble estimator and validation-RMSE weighting helpers."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Mapping


ENSEMBLE_COMPONENT_ORDER = (
    "linear_regression",
    "xgboost",
    "neural_network",
)
INVERSE_RMSE_EPSILON = 1e-12


def normalize_inverse_rmse_weights(
    validation_rmse: Mapping[str, Real],
) -> dict[str, float]:
    """Return deterministic normalized inverse-validation-RMSE weights."""

    if not validation_rmse:
        raise ValueError("At least one validation RMSE is required.")
    inverse: dict[str, float] = {}
    for model_name, raw_score in validation_rmse.items():
        if isinstance(raw_score, bool) or not isinstance(raw_score, Real):
            raise ValueError(
                f"Validation RMSE for '{model_name}' must be numeric."
            )
        score = float(raw_score)
        if not math.isfinite(score) or score < 0.0:
            raise ValueError(
                f"Validation RMSE for '{model_name}' must be finite and non-negative."
            )
        inverse[model_name] = 1.0 / max(score, INVERSE_RMSE_EPSILON)
    total = sum(inverse.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Validation RMSE values could not produce valid weights.")
    return {name: value / total for name, value in inverse.items()}


class WeightedEnsembleRegressor:
    """Joblib-safe predictor that averages component outputs by saved weights."""

    def __init__(
        self,
        component_models: Mapping[str, Any],
        weights: Mapping[str, Real],
        component_parameters: Mapping[str, Mapping[str, Any]],
    ) -> None:
        import numpy as np

        self.component_models = dict(component_models)
        self.weights = {name: float(value) for name, value in weights.items()}
        self.component_parameters = {
            name: dict(parameters)
            for name, parameters in component_parameters.items()
        }
        if len(self.component_models) < 2:
            raise ValueError("An ensemble requires at least two component models.")
        if set(self.component_models) != set(self.weights):
            raise ValueError("Ensemble component models and weights do not match.")
        if set(self.component_models) != set(self.component_parameters):
            raise ValueError("Ensemble component parameters are incomplete.")
        if any(
            not math.isfinite(weight) or weight < 0.0
            for weight in self.weights.values()
        ) or not math.isclose(sum(self.weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError("Ensemble weights must be finite, non-negative, and normalized.")

        feature_counts = {
            getattr(model, "n_features_in_", None)
            for model in self.component_models.values()
        }
        if len(feature_counts) != 1 or None in feature_counts:
            raise ValueError(
                "Ensemble components do not share a valid input-feature count."
            )
        self.n_features_in_ = int(next(iter(feature_counts)))

        output_counts: set[int] = set()
        probe = np.zeros((1, self.n_features_in_), dtype=float)
        for model in self.component_models.values():
            prediction = np.asarray(model.predict(probe), dtype=float)
            if prediction.ndim == 1:
                output_counts.add(int(prediction.size))
            elif prediction.ndim == 2 and prediction.shape[0] == 1:
                output_counts.add(int(prediction.shape[1]))
            else:
                raise ValueError("An ensemble component returned an invalid output shape.")
        if len(output_counts) != 1:
            raise ValueError("Ensemble components do not share the same output shape.")
        self.n_outputs_ = int(next(iter(output_counts)))

    def predict(self, features: Any) -> Any:
        """Predict in saved component order without recalculating weights."""

        import numpy as np

        combined: Any = None
        single_output = self.n_outputs_ == 1
        for model_name, model in self.component_models.items():
            prediction = np.asarray(model.predict(features), dtype=float)
            if prediction.ndim == 1:
                normalized = prediction.reshape(-1, 1)
            elif prediction.ndim == 2:
                normalized = prediction
            else:
                raise ValueError(
                    f"Ensemble component '{model_name}' returned an invalid prediction."
                )
            weighted = normalized * self.weights[model_name]
            combined = weighted if combined is None else combined + weighted
        if combined is None:
            raise ValueError("The ensemble contains no component predictions.")
        return combined[:, 0] if single_output else combined
