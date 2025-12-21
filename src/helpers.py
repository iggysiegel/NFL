"""A collection of helper functions to calculate model performance."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_all_predictions(prediction_dir: Path) -> pd.DataFrame:
    """Load all prediction files from directory.

    Args:
        prediction_dir: Directory containing prediction CSV files.
    """
    prediction_data = []
    for file_path in prediction_dir.glob("state_space_*.csv"):
        parts = file_path.stem.split("_")
        season = int(parts[-2])
        week = int(parts[-1])
        df = pd.read_csv(file_path)
        df["season"] = season
        df["week"] = week
        prediction_data.append(df)

    if not prediction_data:
        raise ValueError(f"No prediction files found in {prediction_dir}")

    return pd.concat(prediction_data, axis=0, ignore_index=True)


def calculate_spread_percentile(row: pd.Series, spread_col: str) -> int:
    """Percentile the spread falls at in the model's predictions.

    Args:
        row: A row containing the spread and prediction percentiles.
        spread_col: Column name of the spread to evaluate.

    Returns:
        A percentile index (0-99) for the spread.
    """
    spread = row[spread_col]
    percentiles = row[[f"prediction_ci_{i:02d}" for i in range(1, 100)]].values
    spread_idx = np.searchsorted(percentiles, spread, side="left")  # a[i-1] < v <= a[i]
    return spread_idx


def calculate_ats_performance(
    data: pd.DataFrame, confidence_threshold: float
) -> pd.DataFrame:
    """Add ATS outcomes for close and open spreads.

    Args:
        data: DataFrame containing game and prediction information.
        confidence_threshold: Minimum confidence required to bet.

    Returns:
        DataFrame with ATS results for each game.
    """
    data = data.copy()

    # Calculate confidence for each line type
    for line, spread_col in [("close", "spread_line"), ("open", "spread_line_open")]:
        data[f"spread_idx_{line}"] = data.apply(
            lambda row: (
                calculate_spread_percentile(row, spread_col)
                if pd.notna(row[spread_col])
                else np.nan
            ),
            axis=1,
        )
        data[f"confidence_{line}"] = (data[f"spread_idx_{line}"] - 50).abs() / 50

        # Keep track of results for each line type
        bets_placed = []
        bets_won = []

        for row in data.itertuples():
            confidence = getattr(row, f"confidence_{line}")
            spread_idx = getattr(row, f"spread_idx_{line}")
            spread_value = getattr(row, spread_col)

            # Skip missing values
            if pd.isna(spread_value) or pd.isna(spread_idx) or pd.isna(confidence):
                bets_placed.append(0)
                bets_won.append(0)
                continue

            # Only consider games above the confidence threshold
            if confidence < confidence_threshold:
                bets_placed.append(0)
                bets_won.append(0)
                continue

            # Model favors home team
            if spread_idx < 50:
                if row.result > spread_value:
                    bets_placed.append(1)
                    bets_won.append(1)
                elif row.result < spread_value:
                    bets_placed.append(1)
                    bets_won.append(0)
                else:
                    bets_placed.append(0)
                    bets_won.append(0)

            # Model favors away team
            elif spread_idx >= 50:
                if row.result < spread_value:
                    bets_placed.append(1)
                    bets_won.append(1)
                elif row.result > spread_value:
                    bets_placed.append(1)
                    bets_won.append(0)
                else:
                    bets_placed.append(0)
                    bets_won.append(0)

        data[f"bets_placed_{line}"] = bets_placed
        data[f"bets_won_{line}"] = bets_won

    return data


def calculate_seasonal_accuracy(ats_data: pd.DataFrame) -> pd.DataFrame:
    """Compute seasonal and overall ATS accuracy.

    Args:
        ats_data: DataFrame containing ATS output with columns:
            - bets_placed_close, bets_won_close
            - bets_placed_open, bets_won_open
            - season

    Returns:
        A DataFrame with summary ATS statistics by season.
    """
    results = []

    # Per-season ATS statistics
    for season, df in ats_data.groupby("season"):
        row = {
            "season": season,
            "bets_placed_close": df["bets_placed_close"].sum(),
            "bets_won_close": df["bets_won_close"].sum(),
            "bets_placed_open": df["bets_placed_open"].sum(),
            "bets_won_open": df["bets_won_open"].sum(),
        }
        row["accuracy_close"] = (
            row["bets_won_close"] / row["bets_placed_close"]
            if row["bets_placed_close"] > 0
            else np.nan
        )
        row["accuracy_open"] = (
            row["bets_won_open"] / row["bets_placed_open"]
            if row["bets_placed_open"] > 0
            else np.nan
        )
        results.append(row)

    # Overall ATS statistics
    overall = {
        "season": "overall",
        "bets_placed_close": ats_data["bets_placed_close"].sum(),
        "bets_won_close": ats_data["bets_won_close"].sum(),
        "bets_placed_open": ats_data["bets_placed_open"].sum(),
        "bets_won_open": ats_data["bets_won_open"].sum(),
    }
    overall["accuracy_close"] = (
        overall["bets_won_close"] / overall["bets_placed_close"]
        if overall["bets_placed_close"] > 0
        else np.nan
    )
    overall["accuracy_open"] = (
        overall["bets_won_open"] / overall["bets_placed_open"]
        if overall["bets_placed_open"] > 0
        else np.nan
    )
    results.append(overall)

    return pd.DataFrame(results)
