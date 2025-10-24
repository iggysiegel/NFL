"""Training module for machine learning pipeline.

This module provides a unified interface for training models with either direct or
residual prediction strategies.
"""

import optuna
import pandas as pd

from .utils import cross_validation, prepare_data
from .xgboost_utils import build_model, get_hyperparams


def objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> float:
    """Objective function for Optuna hyperparameter optimization.

    Args:
        trial: Optuna trial object.
        df: The input DataFrame containing the data.
        feature_cols: List of column names to be used as features.
        target_col: The column name to be used as the target variable.

    Returns:
        float: RMSE score.
    """
    hyperparams = get_hyperparams(trial)
    model = build_model(hyperparams, early_stopping=True)
    score, best_iteration = cross_validation(model, df, feature_cols, target_col)
    if best_iteration is not None:
        trial.set_user_attr("best_iteration", best_iteration)
    return score


def tune_hyperparameters(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_trials: int,
) -> optuna.Study:
    """Run Optuna hyperparameter tuning to minimize the objective function.

    Args:
        df: The input DataFrame containing the data.
        feature_cols: List of column names to be used as features.
        target_col: The column name to be used as the target variable.
        n_trials: Number of optimization trials.

    Returns:
        optuna.Study: Fitted Optuna study containing optimization results.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, df, feature_cols, target_col),
        n_trials=n_trials,
    )
    return study


# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
def train(
    df: pd.DataFrame,
    partition_season: int,
    partition_week: int,
    n_trials: int,
    feature_cols: list[str],
    target_col: str,
    model_type: str,
    baseline_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, object, dict]:
    """Train a model for NFL game prediction.

    Supports both direct and residual prediction strategies.

    Args:
        df: The input DataFrame containing the data.
        partition_season: The season to partition the data.
        partition_week: The week to partition the data.
        n_trials: Number of hyperparameter optimization trials.
        feature_cols: List of column names to be used as features.
        target_col: The column name to be used as the target variable.
        model_type: Either 'baseline' for direct prediction or 'residual'
            for predicting deviations from a baseline.
        baseline_col: Baseline column name (required for residual modeling).

    Returns:
        tuple containing:
        - Training dataframe with model predictions.
        - Testing dataframe with model predictions.
        - Trained model instance.
        - Best hyperparameters found via tuning.
    """
    if model_type not in ["baseline", "residual"]:
        raise ValueError(
            f"model_type must be 'baseline' or 'residual', got '{model_type}'"
        )

    if model_type == "residual" and baseline_col is None:
        raise ValueError("'baseline_col' must be provided for residual modeling.")

    # Create target column based on model type
    df = df.copy()
    if model_type == "residual":
        df["model_target"] = df[target_col] - df[baseline_col]
        training_target = "model_target"
    else:
        training_target = target_col

    # Prepare data
    df_train, df_test, x_train, x_test, y_train, _ = prepare_data(
        df, partition_season, partition_week, feature_cols, training_target
    )
    df_train = df_train.copy()
    df_test = df_test.copy()

    # Tune hyperparameters via cross-validation on training data
    study = tune_hyperparameters(df_train, feature_cols, training_target, n_trials)
    best_params = study.best_params
    best_iteration = study.best_trial.user_attrs.get("best_iteration", None)
    best_params["n_estimators"] = best_iteration

    # Fit model on all training data
    model = build_model(best_params, early_stopping=False)
    model.fit(x_train, y_train)

    # Generate predictions
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    # Reconstruct final predictions
    if model_type == "residual":
        df_train["model_predictions"] = df_train[baseline_col] + train_pred
        df_test["model_predictions"] = df_test[baseline_col] + test_pred
    else:
        df_train["model_predictions"] = train_pred
        df_test["model_predictions"] = test_pred

    return df_train, df_test, model, best_params
