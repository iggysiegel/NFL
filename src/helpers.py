"""A collection of helper functions to calculate and display model performance."""

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


def print_game_predictions(
    predictions: pd.DataFrame, confidence_threshold: float, format: str = "text"
) -> None:
    """Print predictions with betting recommendations.

    NOTE:
        Internally, predictions and spreads are (home - away).
        For display, signs are inverted to match standard convention.

    Args:
        predictions: DataFrame containing this week's game predictions.
        confidence_threshold: Minimum confidence required to bet.
        format: Output format, either 'text' or 'markdown'.
    """
    assert format in ["text", "markdown"], "Invalid format specified."

    # Calculate confidence for all games and sort by confidence
    game_data = []
    for idx in predictions.index:
        row = predictions.loc[idx]
        spread_idx_close = (
            calculate_spread_percentile(row, "spread_line")
            if pd.notna(row["spread_line"])
            else None
        )
        confidence_close = (
            abs(spread_idx_close - 50) / 50 if spread_idx_close is not None else None
        )
        game_data.append(
            {
                "row": row,
                "spread_idx": spread_idx_close,
                "confidence": confidence_close if confidence_close is not None else -1,
            }
        )
    game_data.sort(key=lambda x: x["confidence"], reverse=True)

    # Output header
    if format == "text":
        print("\n" + "=" * 60)
        print("LATEST GAME PREDICTIONS")
        print("=" * 60)
    else:
        print("")
        print("## Latest Game Predictions")

    # Iterate through games
    for game in game_data:
        row = game["row"]
        spread_idx_close = game["spread_idx"]
        confidence_close = game["confidence"] if game["confidence"] >= 0 else None

        # Invert signs for display
        display_prediction = -row["prediction_mean"]
        display_spread = -row["spread_line"] if pd.notna(row["spread_line"]) else None

        # Determine bet side and team name
        bet_side = None
        bet_team = None
        if spread_idx_close is not None:
            if spread_idx_close <= 50:
                bet_side = "HOME"
                bet_team = row["home_team"]
            else:
                bet_side = "AWAY"
                bet_team = row["away_team"]
        should_bet = (
            confidence_close is not None and confidence_close >= confidence_threshold
        )

        # Print game information
        if format == "text":
            print(f"\n{row['away_team']} @ {row['home_team']}")
            print(f"  Predicted Spread: {display_prediction:.1f}")
            if display_spread is not None:
                print(f"  Market Spread: {display_spread:.1f}")
            else:
                print("  Market Spread: N/A")
            if confidence_close is not None:
                print(f"  Confidence: {confidence_close:.1%}")
            else:
                print("  Confidence: N/A")
        else:
            print("")
            print(f"### {row['away_team']} @ {row['home_team']}")
            print("")
            print("| Metric | Value |")
            print("|--------|-------|")
            print(f"| **Predicted Spread** | {display_prediction:+.1f} |")
            if display_spread is not None:
                print(f"| **Market Spread** | {display_spread:+.1f} |")
            else:
                print("| **Market Spread** | N/A |")
            if confidence_close is not None:
                print(f"| **Confidence** | {confidence_close:.1%} |")
            else:
                print("| **Confidence** | N/A |")
            print()

        # Print betting recommendation
        if confidence_close is not None and should_bet:
            if format == "text":
                print(f"  RECOMMEND BET: {bet_side} ({bet_team})")
            else:
                print(f"**RECOMMEND BET: {bet_side} ({bet_team})**")

        if confidence_close is not None and not should_bet:
            # Calculate what spread would give us the threshold confidence
            if bet_side == "HOME":
                target_spread_idx = 50 - (confidence_threshold * 50)
            else:
                target_spread_idx = 50 + (confidence_threshold * 50)

            # Get the spread value at that percentile
            target_spread_internal = row[f"prediction_ci_{int(target_spread_idx):02d}"]

            # Invert for display
            target_spread_display = -target_spread_internal

            # Round to nearest 0.5
            if bet_side == "HOME":
                target_spread_display = np.ceil(target_spread_display * 2) / 2
            else:
                target_spread_display = np.floor(target_spread_display * 2) / 2

            if format == "text":
                print("  Skip betting (confidence below optimal level)")
                print(
                    f"  Bet {bet_side} ({bet_team}) if spread moves to "
                    f"{target_spread_display:.1f} or better"
                )
            else:
                print("Skip betting (confidence below optimal level)")
                print("")
                print(
                    f"Bet **{bet_side} ({bet_team})** if spread moves to "
                    f"**{target_spread_display:+.1f}** or better"
                )


def print_accuracy_summary(accuracy_df: pd.DataFrame, format: str = "text") -> None:
    """Print seasonal accuracy summary.

    Args:
        accuracy_df: DataFrame containing seasonal ATS accuracy.
        format: Output format, either 'text' or 'markdown'.
    """
    assert format in ["text", "markdown"], "Invalid format specified."

    # Text format
    if format == "text":
        print("\n" + "=" * 60)
        print("SEASONAL ATS ACCURACY")
        print("=" * 60)
        print(
            f"\n{'Season':<10} {'Close Bets':<12} {'Close Acc':<12} "
            f"{'Open Bets':<12} {'Open Acc':<12}"
        )
        print("-" * 60)

        for row in accuracy_df.itertuples():
            season_str = str(row.season) if row.season != "overall" else "OVERALL"
            close_acc = (
                f"{row.accuracy_close:.1%}" if pd.notna(row.accuracy_close) else "N/A"
            )
            open_acc = (
                f"{row.accuracy_open:.1%}" if pd.notna(row.accuracy_open) else "N/A"
            )

            print(
                f"{season_str:<10} "
                f"{row.bets_placed_close:<12.0f} "
                f"{close_acc:<12} "
                f"{row.bets_placed_open:<12.0f} "
                f"{open_acc:<12}"
            )

    else:
        print("")
        print("## Seasonal ATS Accuracy")
        print("")
        print("| Season | Close Bets | Close Acc | Open Bets | Open Acc |")
        print("|--------|------------|-----------|-----------|----------|")

        for row in accuracy_df.itertuples():
            if row.season == "overall":
                season_str = "**OVERALL**"
                close_bets = f"**{row.bets_placed_close:.0f}**"
                open_bets = f"**{row.bets_placed_open:.0f}**"
                close_acc = (
                    f"**{row.accuracy_close:.1%}**"
                    if pd.notna(row.accuracy_close)
                    else "**N/A**"
                )
                open_acc = (
                    f"**{row.accuracy_open:.1%}**"
                    if pd.notna(row.accuracy_open)
                    else "**N/A**"
                )
            else:
                season_str = str(row.season)
                close_bets = f"{row.bets_placed_close:.0f}"
                open_bets = f"{row.bets_placed_open:.0f}"
                close_acc = (
                    f"{row.accuracy_close:.1%}"
                    if pd.notna(row.accuracy_close)
                    else "N/A"
                )
                open_acc = (
                    f"{row.accuracy_open:.1%}" if pd.notna(row.accuracy_open) else "N/A"
                )
            print(
                f"| {season_str} | {close_bets} | {close_acc} | "
                f"{open_bets} | {open_acc} |"
            )
