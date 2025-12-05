"""Update script to refresh predictions for the current week and report historical model
performance.

Usage:
    Update with default confidence threshold:
        python -m scripts.update

    Update with custom confidence threshold:
        python -m scripts.update --confidence-threshold 0.7
"""

import argparse
from datetime import datetime

import nflreadpy as nfl
import pandas as pd

from src.data import DataLoader
from src.helpers import (
    calculate_ats_performance,
    calculate_seasonal_accuracy,
    load_all_predictions,
    print_accuracy_summary,
    print_game_predictions,
)
from src.model import StateSpaceModel
from src.paths import MODEL_DIR, PREDICTION_DIR


def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Refresh predictions and report model performance."
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Confidence threshold (float between 0 and 1).",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save current week's predictions to disk (default: False).",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "markdown"],
        default="text",
        help="Output format: 'text' or 'markdown' (default: text).",
    )
    args = parser.parse_args()
    confidence_threshold = args.confidence_threshold
    if not (0 <= confidence_threshold <= 1):
        raise ValueError(
            f"Confidence threshold must be between 0 and 1, got {confidence_threshold}."
        )

    # Get current season and week
    current_season = nfl.get_current_season()
    current_week = nfl.get_current_week()
    if args.format == "text":
        print(f"Current Season: {current_season}, Week: {current_week}")
    if args.format == "markdown":
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        print()
        print(f"**Current Season, Week:** {current_season}, {current_week}")
        print()
        print(f"**Last Updated:** {last_updated}")

    # Validate if model exists
    model_path = MODEL_DIR / f"model_{current_season}_{current_week:02d}.nc"
    if not model_path.exists():
        raise ValueError(
            f"Model file {model_path} does not exist. "
            "Please run the prediction script first."
        )
    model = StateSpaceModel()
    model.load(model_path)

    # Load game data
    game_data = DataLoader(2000, current_season).data

    # Load prediction data
    historical_predictions = load_all_predictions(PREDICTION_DIR)
    current_games = game_data[
        (game_data["season"] == current_season) & (game_data["week"] == current_week)
    ]
    current_predictions = model.predict(current_games)
    if args.save_predictions:
        output_path = (
            PREDICTION_DIR / f"state_space_{current_season}_{current_week:02d}.csv"
        )
        current_predictions.to_csv(output_path, index=False)
    prediction_data = pd.concat(
        [historical_predictions, current_predictions], ignore_index=True
    )

    # Merge game data and prediction data
    merged_data = pd.merge(
        game_data,
        prediction_data,
        on=["season", "week", "home_team", "away_team"],
        how="inner",
    )
    historical_data = merged_data[
        (merged_data["season"] < current_season)
        | (
            (merged_data["season"] == current_season)
            & (merged_data["week"] < current_week)
        )
    ]
    latest_game_data = merged_data[
        (merged_data["season"] == current_season)
        & (merged_data["week"] == current_week)
    ]

    # Print latest predictions with betting recommendations
    if not latest_game_data.empty:
        print_game_predictions(
            latest_game_data, confidence_threshold, format=args.format
        )
    else:
        if args.format == "text":
            print("No games found for current season and week.")
        if args.format == "markdown":
            print("## No games found for current season and week.")

    # Calculate ATS performance on historical data
    if not historical_data.empty:
        ats_data = calculate_ats_performance(historical_data, confidence_threshold)
        accuracy_summary = calculate_seasonal_accuracy(ats_data)
        print_accuracy_summary(accuracy_summary, format=args.format)
    else:
        if args.format == "text":
            print("No historical data available for accuracy calculation.")
        if args.format == "markdown":
            print("## No historical data available for accuracy calculation.")


if __name__ == "__main__":
    main()
