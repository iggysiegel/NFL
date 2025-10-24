"""Utility functions for machine learning pipline."""

import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


def prepare_data(
    df: pd.DataFrame,
    partition_season: int,
    partition_week: int,
    feature_cols: list[str],
    target_col: str,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
]:
    """Prepares training and testing datasets from the provided DataFrame.

    Args:
        df: The input DataFrame containing the data.
        partition_season: The season to partition the data.
        partition_week: The week to partition the data.
        feature_cols: List of column names to be used as features.
        target_col: The column name to be used as the target variable.

    Returns:
        A tuple containing:
        - df_train: pd.DataFrame - Training dataset.
        - df_test: pd.DataFrame - Testing dataset.
        - x_train: pd.DataFrame - Training feature set.
        - x_test: pd.DataFrame - Testing feature set.
        - y_train: pd.Series - Training target variable.
        - y_test: pd.Series - Testing target variable.
    """
    # Split the data into training and testing sets based on season and week
    train_df = df[
        (df["season"] < partition_season)
        | ((df["season"] == partition_season) & (df["week"] < partition_week))
    ]
    test_df = df[
        (df["season"] > partition_season)
        | ((df["season"] == partition_season) & (df["week"] >= partition_week))
    ]

    # Extract features and target variables
    x_train = train_df[feature_cols]
    y_train = train_df[target_col]
    x_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Standardize the feature sets
    scaler = StandardScaler().fit(x_train)
    x_train = pd.DataFrame(
        scaler.transform(x_train), columns=feature_cols, index=x_train.index
    )
    x_test = pd.DataFrame(
        scaler.transform(x_test), columns=feature_cols, index=x_test.index
    )

    return train_df, test_df, x_train, x_test, y_train, y_test


def cross_validation(
    model: object, df: pd.DataFrame, feature_cols: list[str], target_col: str
) -> tuple[float, int | None]:
    """Perform a single train/validation split and evaluate the model.

    Args:
        model: The machine learning model to evaluate.
        df: The input DataFrame containing the data.
        feature_cols: List of column names to be used as features.
        target_col: The column name to be used as the target variable.

    Returns:
        tuple: The RMSE score and number of boosting iterations.
    """
    # Get unique season-week combinations sorted chronologically
    season_week_array = (
        df[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).values
    )

    # Calculate 80/20 split index
    split_idx = int(0.8 * len(season_week_array))
    partition_season = season_week_array[split_idx][0]
    partition_week = season_week_array[split_idx][1]

    # Create the training/validation split
    _, _, x_train, x_val, y_train, y_val = prepare_data(
        df, partition_season, partition_week, feature_cols, target_col
    )

    # Train the model
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    # Make predictions and calculate RMSE
    preds = model.predict(x_val)
    rmse = metrics.root_mean_squared_error(y_val, preds)
    best_iter = getattr(model, "best_iteration", None)

    return rmse, best_iter
