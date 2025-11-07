"""Generate weekly predictions using the state-space model."""

import argparse

import nflreadpy as nfl
from tqdm import tqdm

from src.data import load_data
from src.model import StateSpaceModel
from src.paths import PREDICTION_DIR


def predict(
    start_season: int = None,
    start_week: int = None,
    end_season: int = None,
    end_week: int = None,
):
    """Generate predictions for each week using a 5-year training set.

    If all parameters are None, predicts for the current week only.

    Args:
        start_season: First season to include in backtest (None = current week).
        start_week: First week to include in backtest (None = current week).
        end_season: Last season to include in backtest (None = current week).
        end_week: Last week to include in backtest (None = current week).
    """
    # Get current season/week
    current_season = nfl.get_current_season()
    current_week = nfl.get_current_week()

    # If all parameters None, predict current week only
    if all(x is None for x in [start_season, start_week, end_season, end_week]):
        print(
            f"No parameters provided - predicting current week ({current_season} "
            f"Week {current_week})"
        )
        start_season = current_season
        start_week = current_week
        end_season = current_season
        end_week = current_week

    # Validate inputs
    if (
        start_season is None
        or start_week is None
        or end_season is None
        or end_week is None
    ):
        raise ValueError(
            "Either provide all parameters (start_season, start_week, end_season, "
            "end_week) or none (to predict current week)"
        )

    # Check for future weeks
    if end_season > current_season or (
        end_season == current_season and end_week > current_week
    ):
        raise ValueError("Cannot predict future weeks")

    # Validate start <= end
    if (start_season > end_season) or (
        start_season == end_season and start_week > end_week
    ):
        raise ValueError(
            f"Start ({start_season} Week {start_week}) must be before or equal to "
            f"end ({end_season} Week {end_week})"
        )

    # Validate data availability
    if start_season < 2004:
        raise ValueError(
            f"Requested training start: {start_season}, data only avilable from 2004."
        )

    print(f"\n{'='*60}")
    print("BACKTEST CONFIGURATION")
    print(f"{'='*60}")
    print(
        f"Prediction range: {start_season} Week {start_week} to {end_season} "
        f"Week {end_week}"
    )
    print(f"Current week: {current_season} Week {current_week}")
    print("Training window: Rolling 5-year window")
    print(f"{'='*60}\n")

    # Load all data needed (including 5 years before start for training)
    try:
        data = load_data(start_season=start_season - 5, end_season=end_season)
    except ValueError as e:
        print(f"Error loading data: {e}")
        return

    # Build list of all season-week combinations to predict
    season_weeks = (
        data[["season", "week"]]
        .drop_duplicates()
        .sort_values(["season", "week"])
        .values.tolist()
    )

    # Filter to backtest range only
    season_weeks = [
        (s, w)
        for (s, w) in season_weeks
        if (s > start_season or (s == start_season and w >= start_week))
        and (s < end_season or (s == end_season and w <= end_week))
    ]

    if not season_weeks:
        print("No weeks found in the specified range.")
        return

    print(f"Found {len(season_weeks)} weeks to predict\n")

    for season, week in tqdm(season_weeks, desc="Backtesting"):
        # Check if predictions already exist
        output_path = PREDICTION_DIR / f"state_space_{season}_{week:02d}.csv"

        if output_path.exists():
            tqdm.write(
                f"Skipping Season {season} Week {week} - predictions already exist"
            )
            continue

        tqdm.write(f"Predicting for Season {season} Week {week}...")

        # Get training data
        train_start_season = season - 5
        train_data = data[
            (data["season"] >= train_start_season)
            & (
                (data["season"] < season)
                | ((data["season"] == season) & (data["week"] < week))
            )
        ]

        # Get testing data
        test_data = data[(data["season"] == season) & (data["week"] == week)]

        if train_data.empty or test_data.empty:
            tqdm.write(
                f"Warning: No data found for Season {season} Week {week}. Skipping."
            )
            continue

        # Initialize and fit model
        model = StateSpaceModel()

        try:
            model.fit(train_data)

            # Predict
            week_preds = model.predict(test_data)

            # Save predictions
            week_preds.to_csv(output_path, index=False)
            tqdm.write(f"Saved predictions to {output_path.name}")

        except Exception as e:
            tqdm.write(f"Error predicting Season {season} Week {week}: {e}")
            continue

    # Summary
    print(f"\n{'='*60}")
    print("BACKTEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total weeks: {len(season_weeks)}")
    print(f"Predictions saved to: {PREDICTION_DIR}")
    print(f"{'='*60}\n")


def main():
    """Parse command-line arguments and generate predictions."""
    parser = argparse.ArgumentParser(
        description="Generate predictions using the state-space model. "
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=None,
        help="First season to include in backtest. If omitted, predicts current week.",
    )
    parser.add_argument(
        "--start-week",
        type=int,
        default=None,
        help="First week to include in backtest. If omitted, predicts current week.",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=None,
        help="Last season to include in backtest. If omitted, predicts current week.",
    )
    parser.add_argument(
        "--end-week",
        type=int,
        default=None,
        help="Last week to include in backtest. If omitted, predicts current week.",
    )

    args = parser.parse_args()

    predict(
        start_season=args.start_season,
        start_week=args.start_week,
        end_season=args.end_season,
        end_week=args.end_week,
    )


if __name__ == "__main__":
    main()
