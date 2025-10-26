"""Run a rolling 5-year backtest using the state space model to generate weekly
predictions."""

import argparse

import pandas as pd
from tqdm import tqdm

from src.models.state_space import StateSpaceModel
from src.paths import PREDICTION_DIR, RAW_DIR


def run_backtest(start_season: int, start_week: int, end_season: int, end_week: int):
    """Run a rolling 5-year backtest and save predictions for each week."""
    data = pd.read_csv(RAW_DIR / "historical_games.csv")

    # Build list of all season-week combinations
    season_weeks = (
        data[["season", "week"]]
        .drop_duplicates()
        .sort_values(["season", "week"])
        .values.tolist()
    )

    # Limit to backtest range
    season_weeks = [
        (s, w)
        for (s, w) in season_weeks
        if (s > start_season or (s == start_season and w >= start_week))
        and (s < end_season or (s == end_season and w <= end_week))
    ]

    for season, week in tqdm(season_weeks, desc="Backtesting"):
        print(f"Predicting for Season {season} Week {week}...")

        # Define rolling 5-year training window
        train_start_season = season - 5

        train_data = data[
            (
                (data["season"] >= train_start_season)
                & (
                    (data["season"] < season)
                    | ((data["season"] == season) & (data["week"] < week))
                )
            )
        ]
        if train_data.empty:
            print("Warning! Not enough training data!")
            continue

        # Initialize and fit model
        model = StateSpaceModel()
        model.fit(train_data)

        # Testing data
        test_data = data[(data["season"] == season) & (data["week"] == week)]
        if test_data.empty:
            print("Warning! No games found to predict!")
            continue

        # Predict
        week_preds = model.predict(test_data)
        output_path = PREDICTION_DIR / f"state_space_{season}_{week}.csv"
        week_preds.to_csv(output_path, index=False)


def main():
    """Parse command-line arguments and execute the rolling backtest."""
    parser = argparse.ArgumentParser(
        description="Run backtest predictions using the state space model."
    )
    parser.add_argument(
        "--start-season",
        type=int,
        required=True,
        help="First season to include in backtest.",
    )
    parser.add_argument(
        "--start-week",
        type=int,
        required=True,
        help="First week to include in backtest.",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        required=True,
        help="Last season to include in backtest.",
    )
    parser.add_argument(
        "--end-week", type=int, required=True, help="Last week to include in backtest."
    )
    args = parser.parse_args()

    run_backtest(
        start_season=args.start_season,
        start_week=args.start_week,
        end_season=args.end_season,
        end_week=args.end_week,
    )


if __name__ == "__main__":
    main()
