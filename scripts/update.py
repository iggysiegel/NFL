"""Update script to refresh predictions and report model performance.

Usage:
    python -m scripts.update
"""

import nflreadpy as nfl
import pandas as pd

from src.data import DataLoader
from src.helpers import (
    calculate_ats_performance,
    calculate_seasonal_accuracy,
    check_existence,
    load_all_predictions,
    print_accuracy_summary,
    print_game_predictions,
)
from src.paths import PREDICTION_DIR

CONFIDENCE_THRESHOLD = 0.6


def main():
    """Main execution function."""
    # Get current season and week
    current_season = nfl.get_current_season()
    current_week = nfl.get_current_week()

    print(f"Current Season: {current_season}, Week: {current_week}")

    # Validate if model and predictions exist
    if not check_existence(current_season, current_week):
        raise ValueError(
            "Model or predictions do not exist for the current season and week. "
            "Please run the prediction script first."
        )

    # Load game data
    game_data = DataLoader(2000, current_season).data

    # Load all predictions
    prediction_data = load_all_predictions(PREDICTION_DIR)

    # Merge game data with predictions
    merged_data = pd.merge(
        game_data,
        prediction_data,
        on=["season", "week", "home_team", "away_team"],
        how="inner",
    )

    # Filter for historical / latest games
    historical_data = merged_data[
        (merged_data["season"] < current_season)
        | (
            (merged_data["season"] == current_season)
            & (merged_data["week"] < current_week)
        )
    ].copy()

    latest_game_data = merged_data[
        (merged_data["season"] == current_season)
        & (merged_data["week"] == current_week)
    ].copy()

    # Print latest predictions with betting recommendations
    if not latest_game_data.empty:
        print_game_predictions(latest_game_data, CONFIDENCE_THRESHOLD)
    else:
        print("No games found for current season and week.")

    # Calculate ATS performance on historical data
    if not historical_data.empty:
        ats_data = calculate_ats_performance(historical_data, CONFIDENCE_THRESHOLD)
        accuracy_summary = calculate_seasonal_accuracy(ats_data)
        print_accuracy_summary(accuracy_summary)
    else:
        print("No historical data available for accuracy calculation.")

    print("\nUpdate complete.")


if __name__ == "__main__":
    main()
