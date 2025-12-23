"""Module for formatting and displaying model predictions."""

from datetime import datetime, timedelta, timezone

import nflreadpy as nfl
import pandas as pd

from src.helpers import calculate_spread_percentile


class Formatter:
    """Handles formatting output for different display targets."""

    def __init__(self, format: str, confidence_threshold: float):
        """Initialize formatter.

        Args:
            format: Output format, either 'text' or 'markdown'.
            confidence_threshold: Threshold for coloring confidence.
        """
        assert format in ["text", "markdown"], "Invalid format specified."
        self.format = format
        self.confidence_threshold = confidence_threshold

    def print_header(self) -> None:
        """Print header with current season and week info."""
        current_season = nfl.get_current_season()
        current_week = nfl.get_current_week()
        est = timezone(timedelta(hours=-5))
        current_time = datetime.now(est).strftime("%Y-%m-%d %H:%M:%S EST")
        nav_bar = (
            "[Home](index.html) | "
            "[Weekly Predictions](upcoming.html) | "
            "[Github](https://github.com/iggysiegel/NFL)"
        )

        if self.format == "text":
            print(f"Current Season: {current_season}, Week: {current_week}")

        if self.format == "markdown":
            print(nav_bar, "\n")
            print("---\n")
            print(f"**Current Season, Week:** {current_season}, {current_week}\n")
            print(f"**Last Updated:** {current_time}\n")
            print("---\n")

    def print_game_predictions(self, data: pd.DataFrame) -> None:
        """Print game predictions in the specified format.

        Internally, predictions and spreads are (home - away).
        For display, signs are inverted to match standard convention.

        Args:
            data: DataFrame containing game data and prediction data.
        """
        if data.empty:
            print("\nNo games found for current season and week.")
            return

        # Sort games by model confidence
        sorted_games = []
        for _, row in data.iterrows():
            spread_idx = calculate_spread_percentile(row, "spread_line")
            confidence = abs(spread_idx - 50) / 50
            sorted_games.append(
                {
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "display_spread": -row["spread_line"],
                    "display_prediction": -row["prediction_mean"],
                    "confidence": confidence,
                }
            )
        sorted_games.sort(key=lambda x: x["confidence"], reverse=True)

        # Text formatting
        if self.format == "text":
            print("\n" + "=" * 60)
            print("LATEST GAME PREDICTIONS")
            print("=" * 60)

            for game in sorted_games:
                self._print_game_prediction_text(game)

        # Markdown formatting
        elif self.format == "markdown":
            for i, game in enumerate(sorted_games):
                self._print_game_prediction_markdown(game)
                if i < len(sorted_games) - 1:
                    print("\n---\n")

    def _print_game_prediction_text(self, game):
        """Print a single game prediction in plain text format.

        Args:
            game: Dictionary containing game data.
        """
        print(f"\n{game['away_team']} @ {game['home_team']}")
        print(f"  Market Spread: {game['display_spread']:.1f}")
        print(f"  Predicted Spread: {game['display_prediction']:.1f}")
        print(f"  Model Confidence: {game['confidence']:.1%}")

    def _print_game_prediction_markdown(self, game):
        """Print a single game prediction in markdown format.

        Args:
            game: Dictionary containing game data.
        """
        away_team = game["away_team"]
        home_team = game["home_team"]
        confidence = game["confidence"]

        # Team logos and names
        away_logo_path = f"images/logos/{away_team}.png"
        home_logo_path = f"images/logos/{home_team}.png"
        print("<div align='center'>")
        print()
        print(
            f"<div style='display:flex; align-items:center; justify-content:center; "
            f"gap:40px; margin-bottom:12px;'>"
            f"<img src='{away_logo_path}' width='80' height='80' alt='{away_team}'>"
            f"<img src='{home_logo_path}' width='80' height='80' alt='{home_team}'>"
            f"</div>"
        )
        print()
        print(f"**{away_team}** @ **{home_team}**")
        print()
        print()

        # Game details
        if confidence >= self.confidence_threshold:
            confidence_color = "green"
        else:
            confidence_color = "red"
        print(f"**Market Spread:** {game['display_spread']:.1f}")
        print()
        print(f"**Predicted Spread:** {game['display_prediction']:.1f}")
        print()
        print(
            f"**Model Confidence:** "
            f"<span style='color:{confidence_color}; font-weight:bold;'"
            f">{confidence:.1%}</span>"
        )
        print()
        print("</div>")
