"""Machine learning model pipeline for NFL game prediction.

This module provides a full workflow for preparing data, performing
hyperparameter optimization with Optuna, and generating out-of-sample predictions.
It supports both direct modeling of game results and residual modeling
(predicting deviations from a baseline metric such as the Vegas line).

Usage:
    from src.models.model_pipeline import main

    df_train, df_test, model, best_params = main(
        data=dataset,
        train_seasons=[2018, 2019, 2020, 2021],
        test_seasons=[2022],
        features=feature_list,
        model_class="XGB",
        n_trials=50,
        target_type="residual",
        baseline_col="home_line",
    )

Notes:
- Cross-validation is performed using a rolling-season approach
  to respect the temporal order of games.
"""

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


def prepare_data(
    data: pd.DataFrame,
    train_seasons: list[int],
    test_seasons: list[int],
    features: list[int],
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series
]:
    """
    Partition a dataset into training and testing sets, dropping rows with missing
    feature values, and performing standard scaling.

    Args:
        data: Input dataset containing all seasons and target column 'result'.
        train_seasons: List of season values to include in the training set.
        test_seasons: List of season values to include in the testing set.
        features: List of feature column names to use as predictors.

    Returns:
        tuple: Training dataframe, training feature matrix, training target vector,
            testing dataframe, testing feature matrix, testing target vector.
    """
    df_train = data[data["season"].isin(train_seasons)].dropna(subset=features)
    df_test = data[data["season"].isin(test_seasons)].dropna(subset=features)

    x_train = df_train[features]
    x_test = df_test[features]
    scaler = StandardScaler().fit(x_train)
    x_train = pd.DataFrame(
        scaler.transform(x_train), columns=features, index=x_train.index
    )
    x_test = pd.DataFrame(
        scaler.transform(x_test), columns=features, index=x_test.index
    )

    y_train = df_train["result"]
    y_test = df_test["result"]

    return df_train, x_train, y_train, df_test, x_test, y_test


def get_hyperparams(trial: optuna.Trial, model_class: str) -> dict:
    """
    Return hyperparameter search space for the given model.

    Args:
        trial: Optuna trial object.
        model_class: Model class.

    Returns:
        dict: Hyperparameter search space.

    Todo:
        Add support for span and revert.
    """
    if model_class == "XGB":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 25, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }
    if model_class == "Lasso":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True)}

    raise NotImplementedError(f"Model class not implemented for {model_class}")


def build_model(model_class: str, hyperparams: dict, early_stopping: bool):
    """
    Initialize a machine learning model with the specified hyperparameters.

    Args:
        model_class: Model class.
        hyperparams: Hyperparameters for the model.
        early_stopping: Whether to include early stopping.

    Returns:
        Initialized machine learning model.
    """
    if model_class == "XGB":
        model_args = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "n_estimators": 1000,
            **hyperparams,
        }
        if early_stopping:
            model_args["early_stopping_rounds"] = 50
        return xgb.XGBRegressor(**model_args)

    if model_class == "Lasso":
        return Lasso(**hyperparams, max_iter=5000)

    raise NotImplementedError(f"Model class not implemented for {model_class}")


def train_and_validate_fold(
    model: object,
    data: tuple[pd.DataFrame, pd.Series],
    seasons: pd.Series,
    train_seasons: list[int],
    val_season: int,
) -> tuple[float, int | None]:
    """Helper function to train and validate the model on a single fold."""
    train_idx = seasons[seasons.isin(train_seasons)].index
    val_idx = seasons[seasons == val_season].index

    x_train, y_train = data[0].loc[train_idx], data[1].loc[train_idx]
    x_val, y_val = data[0].loc[val_idx], data[1].loc[val_idx]

    try:
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    except TypeError:
        model.fit(x_train, y_train)

    preds = model.predict(x_val)
    rmse = metrics.root_mean_squared_error(y_val, preds)
    best_iter = getattr(model, "best_iteration", None)
    return rmse, best_iter


def cross_validation(
    model: object, x: pd.DataFrame, y: pd.Series, seasons: pd.Series
) -> tuple[float, int | None]:
    """
    Perform time-based cross-validation for the given model. For each fold, the model
    is trained on four consecutive seasons and validated on the subsequent season.

    Args:
        model: The machine learning model to evaluate.
        x: Feature matrix.
        y: Target vector.
        seasons: Series indicating the season for each row in the feature matrix.

    Returns:
        tuple: The mean RMSE score and the median number of boosting iterations.
    """
    all_seasons = list(range(seasons.min(), seasons.max() + 1))
    scores, best_iterations = [], []

    for i in range(len(all_seasons) - 4):
        train_seasons = all_seasons[i : i + 4]
        val_season = all_seasons[i + 4]

        rmse, best_iter = train_and_validate_fold(
            model, (x, y), seasons, train_seasons, val_season
        )
        scores.append(rmse)
        if best_iter is not None:
            best_iterations.append(best_iter)

    best_iterations = int(np.median(best_iterations)) if best_iterations else None
    return float(np.mean(scores)), best_iterations


def objective(
    trial: optuna.Trial,
    x: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    model_class: str,
) -> float:
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial: Optuna trial object.
        x: Feature matrix.
        y: Target vector.
        seasons: Series indicating the season for each row in X.
        model_class: Model class.

    Returns:
        float: RMSE score.
    """
    hyperparams = get_hyperparams(trial, model_class)
    model = build_model(model_class, hyperparams, early_stopping=True)
    score, best_iteration = cross_validation(model, x, y, seasons)
    if hasattr(model, "best_iteration"):
        trial.set_user_attr("best_iteration", best_iteration)
    return score


def tune_hyperparameters(
    x: pd.DataFrame, y: pd.Series, seasons: pd.Series, model_class: str, n_trials: int
) -> optuna.Study:
    """
    Run Optuna hyperparameter tuning to minimize the objective function.

    Args:
        x: Feature matrix.
        y: Target vector.
        seasons: Series indicating the season for each row in X.
        model_class: Model class.
        n_trials: Number of optimization trials.

    Returns:
        optuna.Study: Fitted Optuna study containing optimization results.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, x, y, seasons, model_class), n_trials=n_trials
    )
    return study


# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
def main(
    data: pd.DataFrame,
    train_seasons: list[int],
    test_seasons: list[int],
    features: list[str],
    model_class: str,
    n_trials: int,
    target_type: str,
    baseline_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, object, dict]:
    """
    Main function to prepare data, tune hyperparameters, fit the model, and store
    predictions. Supports direct or residual modeling.

    Args:
        data: Input dataset containing all seasons and target column 'result'.
        train_seasons: List of season values to include in the training set.
        test_seasons: List of season values to include in the testing set.
        features: List of feature column names to use as predictors.
        model_class: Model class.
        n_trials: Number of hyperparameter optimization trials.
        target_type: Type of modeling target either 'direct' for raw result prediction
            or 'residual' for modeling the difference between result and a baseline.
        baseline_col: Column name of baseline (required for residual modeling).

    Returns:
        tuple: Training dataframe with model predictions, testing dataframe with
            model predictions, trained model instance, best hyperparameters found
            via tuning.
    """
    if model_class not in ["XGB", "Lasso"]:
        raise NotImplementedError(f"Model class not implemented for {model_class}")

    if target_type == "residual" and baseline_col is None:
        raise ValueError("'baseline_col' must be provided for residual modeling.")

    # Prepare data
    df_train, x_train, y_train, df_test, x_test, y_test = prepare_data(
        data, train_seasons, test_seasons, features
    )

    # Residual modeling target
    if target_type == "residual":
        y_train = y_train - df_train[baseline_col]
        y_test = y_test - df_test[baseline_col]

    # Tune hyperparameters via cross-validation
    seasons = df_train["season"]
    study = tune_hyperparameters(x_train, y_train, seasons, model_class, n_trials)
    best_params = study.best_params
    best_iteration = study.best_trial.user_attrs.get("best_iteration", None)
    if best_iteration is not None:
        best_params["n_estimators"] = best_iteration

    # Fit model on all training data
    model = build_model(model_class, best_params, early_stopping=False)
    model.fit(x_train, y_train)

    # Predict
    model_train_pred = model.predict(x_train)
    model_test_pred = model.predict(x_test)

    # Reconstruct final predictions
    if target_type == "residual":
        df_train["model_predictions"] = df_train[baseline_col] + model_train_pred
        df_test["model_predictions"] = df_test[baseline_col] + model_test_pred
    else:
        df_train["model_predictions"] = model_train_pred
        df_test["model_predictions"] = model_test_pred

    return df_train, df_test, model, best_params
