"""Generate weekly NFL predictions using the state-space model.

This script generates predictions for NFL games using a rolling 5-year
training window. It can either predict the current week or run a
backtest over a historical range of weeks.

Usage:
    Predict current week only:
        python -m scripts.predict

    Run backtest over specific range:
        python -m scripts.predict --start-season 2020 --start-week 1 \
                                  --end-season 2023 --end-week 18
"""

import argparse

import nflreadpy as nfl
from tqdm import tqdm

from src.data import DataLoader
from src.model import StateSpaceModel
from src.paths import PREDICTION_DIR


def predict(
    start_season: int = None,
    start_week: int = None,
    end_season: int = None,
    end_week: int = None,
) -> None:
    """Generate predictions for each week using a 5-year training set.

    If all parameters are None, predicts for the current week only.
    Otherwise, runs a backtest over the specified range of weeks,
    training a new model for each week using previous 5 years of data.

    Args:
        start_season: First season to include in backtest.
        start_week: First week to include in backtest.
        end_season: Last season to include in backtest.
        end_week: Last week to include in backtest.
    """
    # Get current season / week
    current_season = nfl.get_current_season()
    current_week = nfl.get_current_week()

    # If all parameters None, predict current week only
    if all(x is None for x in [start_season, start_week, end_season, end_week]):
        print(f"No parameters provided - predicting ({current_season} {current_week}).")
        start_season = current_season
        start_week = current_week
        end_season = current_season
        end_week = current_week

    # Validate that all parameters are provided together
    if (
        start_season is None
        or start_week is None
        or end_season is None
        or end_week is None
    ):
        raise ValueError(
            "Either provide all parameters (start_season, start_week, end_season, "
            "end_week) or none (to predict current week)."
        )

    # Validate for future weeks
    if end_season > current_season or (
        end_season == current_season and end_week > current_week
    ):
        raise ValueError("Cannot predict beyond current week.")

    # Validate start <= end
    if (start_season > end_season) or (
        start_season == end_season and start_week > end_week
    ):
        raise ValueError(
            f"Start ({start_season} Week {start_week}) must be before or equal to "
            f"end ({end_season} Week {end_week})."
        )

    # Validate data availability
    if start_season < 2005:
        raise ValueError(
            f"Requested training start: {start_season}, data only avilable from 2005."
        )

    print(f"\n{'='*60}")
    print(
        f"Prediction range: {start_season} Week {start_week} to {end_season} "
        f"Week {end_week}."
    )
    print(f"Current week: {current_season} Week {current_week}.")
    print(f"{'='*60}\n")

    # Load data
    try:
        data = DataLoader(start_season=start_season - 5, end_season=end_season).data
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

    season_weeks = [
        (s, w)
        for (s, w) in season_weeks
        if (s > start_season or (s == start_season and w >= start_week))
        and (s < end_season or (s == end_season and w <= end_week))
    ]

    if not season_weeks:
        print("No weeks found in the specified range.")
        return

    # Generate predictions for each week
    for season, week in tqdm(season_weeks):

        # Check if predictions already exist
        output_path = PREDICTION_DIR / f"state_space_{season}_{week:02d}.csv"

        if output_path.exists():
            tqdm.write(
                f"Skipping Season {season} Week {week} - prediction already exists."
            )
            continue

        tqdm.write(f"Predicting for Season {season} Week {week}...")

        # Get training data (5 years ending before current week)
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
            model.fit(
                data=train_data,
                draws=1000,
                tune=5000,
                target_accept=0.95,
                chains=4,
                cores=4,
                random_seed=42,
            )
            week_preds = model.predict(test_data)
            week_preds.to_csv(output_path, index=False)
            tqdm.write(f"Saved predictions to {output_path.name}.")

        except Exception as e:
            tqdm.write(f"Error predicting Season {season} Week {week}: {e}.")
            continue

    # Display summary
    print(f"\n{'='*60}")
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
