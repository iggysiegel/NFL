"""XGBoost-specific utilities for machine learning pipeline."""

import optuna
import xgboost as xgb


def get_hyperparams(trial: optuna.Trial) -> dict:
    """Return hyperparameter search space for XGBoost.

    Args:
        trial: Optuna trial object.

    Returns:
        dict: Hyperparameter search space for XGBoost.
    """
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 25, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    }


def build_model(hyperparams: dict, early_stopping: bool = False) -> xgb.XGBRegressor:
    """Initialize an XGBoost model with the specified hyperparameters.

    Args:
        hyperparams: Hyperparameters for the model.
        early_stopping: Whether to include early stopping (default: False).

    Returns:
        Initialized XGBoost regressor.
    """
    model_args = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 1000,
        **hyperparams,
    }

    if early_stopping:
        model_args["early_stopping_rounds"] = 50

    return xgb.XGBRegressor(**model_args)
