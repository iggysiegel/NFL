"""Update script to refresh predictions for the current week.

Usage:
    Output in text format:
        python -m scripts.update

    Output in markdown format:
        python -m scripts.update --format markdown
"""

import argparse

import nflreadpy as nfl

from src.data import DataLoader
from src.model import StateSpaceModel
from src.output import Formatter
from src.paths import MODEL_DIR, PREDICTION_DIR

CONFIDENCE_THRESHOLD = 0.5


def main():
    """Generate and display predictions for the current week."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Refresh predictions for the current week."
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "markdown"],
        default="text",
        help="Output format: 'text' or 'markdown'.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save current week's predictions to disk.",
    )
    args = parser.parse_args()

    # Get current season and week
    current_season = nfl.get_current_season()
    current_week = nfl.get_current_week()

    # Load the trained model
    model_path = MODEL_DIR / f"model_{current_season}_{current_week:02d}.nc"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file {model_path} does not exist. "
            "Run the prediction script first to train the model."
        )
    model = StateSpaceModel()
    model.load(model_path)

    # Load game data and generate predictions
    game_data = DataLoader(2000, current_season).data
    current_games = game_data[
        (game_data["season"] == current_season) & (game_data["week"] == current_week)
    ]
    current_predictions = model.predict(current_games)

    # Save predictions if requested
    if args.save_predictions:
        output_path = (
            PREDICTION_DIR / f"state_space_{current_season}_{current_week:02d}.csv"
        )
        current_predictions.to_csv(output_path, index=False)

    # Merge game data with predictions
    merged_data = current_games.merge(
        current_predictions,
        on=["season", "week", "home_team", "away_team"],
        how="inner",
    )

    # Format and display predictions
    formatter = Formatter(args.format, CONFIDENCE_THRESHOLD)
    formatter.print_header()
    formatter.print_game_predictions(merged_data)


if __name__ == "__main__":
    main()
