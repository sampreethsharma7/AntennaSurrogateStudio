"""Transient state and contract mappings for the Model Training page."""

from __future__ import annotations

from dataclasses import dataclass, field

from studio.model_training import (
    NEURAL_NETWORK_CUSTOM_DEFAULTS,
    NEURAL_NETWORK_CUSTOM_PARAMETER_NAMES,
    XGBOOST_CUSTOM_DEFAULTS,
    XGBOOST_CUSTOM_PARAMETER_NAMES,
)


SUPPORTED_MODELS = (
    "Linear Regression",
    "XGBoost",
    "Neural Network",
    "Ensemble AI Engine",
)
TRAINING_MODES = ("Auto", "Custom")
AUTO_SEARCH_LEVELS = ("Medium", "High")
MODEL_REQUEST_NAMES = {
    "Linear Regression": "linear_regression",
    "XGBoost": "xgboost",
    "Neural Network": "neural_network",
    "Ensemble AI Engine": "ensemble_ai_engine",
}
TRAINING_MODE_REQUEST_NAMES = {"Auto": "auto", "Custom": "custom"}
SEARCH_LEVEL_REQUEST_NAMES = {"Medium": "medium", "High": "high"}
AUTO_SEARCH_DESCRIPTIONS = {
    "Medium": "Faster, lower-compute deterministic search",
    "High": "Slower, more thorough deterministic search",
}
DEFAULT_CUSTOM_HYPERPARAMETERS = {
    "fit_intercept": True,
    "positive": False,
}
TRAIN_BUTTON_LABEL = "Train Model"


@dataclass(slots=True)
class TrainingPageState:
    """Explicit, transient UI values for the training configuration page."""

    selected_model: str = SUPPORTED_MODELS[0]
    training_mode: str = TRAINING_MODES[0]
    search_level: str = AUTO_SEARCH_LEVELS[0]
    custom_hyperparameters: dict[str, bool] = field(
        default_factory=lambda: dict(DEFAULT_CUSTOM_HYPERPARAMETERS)
    )
    xgboost_custom_hyperparameters: dict[str, int | float] = field(
        default_factory=lambda: dict(XGBOOST_CUSTOM_DEFAULTS)
    )
    neural_network_custom_hyperparameters: dict[str, object] = field(
        default_factory=lambda: {
            name: list(value) if name == "hidden_layer_sizes" else value
            for name, value in NEURAL_NETWORK_CUSTOM_DEFAULTS.items()
        }
    )

    @property
    def auto_search_enabled(self) -> bool:
        return self.training_mode == "Auto" and not self.ensemble_mode_enabled

    @property
    def advanced_settings_enabled(self) -> bool:
        return self.training_mode == "Custom" and not self.ensemble_mode_enabled

    @property
    def fixed_baseline_enabled(self) -> bool:
        return False

    @property
    def ensemble_mode_enabled(self) -> bool:
        return self.selected_model == "Ensemble AI Engine"

    def set_model(self, model: str) -> None:
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported training-page model: {model}")
        self.selected_model = model
        if self.ensemble_mode_enabled:
            self.training_mode = "Auto"
            self.search_level = "High"

    def set_training_mode(self, mode: str) -> None:
        if mode not in TRAINING_MODES:
            raise ValueError(f"Unsupported training mode: {mode}")
        self.training_mode = mode

    def set_search_level(self, level: str) -> None:
        if level not in AUTO_SEARCH_LEVELS:
            raise ValueError(f"Unsupported auto search level: {level}")
        self.search_level = level

    def set_custom_hyperparameter(self, name: str, value: bool) -> None:
        if name not in DEFAULT_CUSTOM_HYPERPARAMETERS:
            raise ValueError(f"Unsupported custom hyperparameter: {name}")
        self.custom_hyperparameters[name] = bool(value)

    def set_xgboost_custom_hyperparameter(
        self,
        name: str,
        value: int | float,
    ) -> None:
        if name not in XGBOOST_CUSTOM_PARAMETER_NAMES:
            raise ValueError(f"Unsupported XGBoost hyperparameter: {name}")
        self.xgboost_custom_hyperparameters[name] = value

    def set_neural_network_custom_hyperparameter(
        self,
        name: str,
        value: object,
    ) -> None:
        if name not in NEURAL_NETWORK_CUSTOM_PARAMETER_NAMES:
            raise ValueError(f"Unsupported Neural Network hyperparameter: {name}")
        self.neural_network_custom_hyperparameters[name] = value

    def reset(self) -> None:
        self.selected_model = SUPPORTED_MODELS[0]
        self.training_mode = TRAINING_MODES[0]
        self.search_level = AUTO_SEARCH_LEVELS[0]
        self.custom_hyperparameters = dict(DEFAULT_CUSTOM_HYPERPARAMETERS)
        self.xgboost_custom_hyperparameters = dict(XGBOOST_CUSTOM_DEFAULTS)
        self.neural_network_custom_hyperparameters = {
            name: list(value) if name == "hidden_layer_sizes" else value
            for name, value in NEURAL_NETWORK_CUSTOM_DEFAULTS.items()
        }

    def snapshot(self) -> dict[str, object]:
        return {
            "selected_model": self.selected_model,
            "training_mode": self.training_mode,
            "search_level": self.search_level,
            "custom_hyperparameters": dict(self.custom_hyperparameters),
            "xgboost_custom_hyperparameters": dict(
                self.xgboost_custom_hyperparameters
            ),
            "neural_network_custom_hyperparameters": dict(
                self.neural_network_custom_hyperparameters
            ),
        }
