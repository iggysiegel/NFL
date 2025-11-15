"""Module for loading and preprocessing NFL schedule data."""

import nflreadpy as nfl
import pandas as pd


def load_data(start_season: int, end_season: int) -> pd.DataFrame:
    """Load and clean NFL schedule data for a range of seasons.

    Args:
        start_season: The first season to include.
        end_season: The last season to include.
    Returns:
        A cleaned DataFrame containing season, week, team names,
        location flags, QB information, spread lines, and results.
    """
    current_season = nfl.get_current_season()
    current_week = nfl.get_current_week()

    if start_season < 1999 or end_season > current_season:
        raise ValueError("Data available between 1999 and the current season only.")

    # Load schedule data
    data = nfl.load_schedules(list(range(start_season, end_season + 1))).to_pandas()

    # Normalize old team abbreviations
    team_name_map = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
    data["home_team"] = data["home_team"].replace(team_name_map)
    data["away_team"] = data["away_team"].replace(team_name_map)

    # Add neutral site flag
    data["is_neutral"] = data["location"].map({"Home": 0, "Neutral": 1})

    # Keep relevant columns
    data = data[
        [
            "season",
            "week",
            "home_team",
            "away_team",
            "is_neutral",
            "home_qb_id",
            "home_qb_name",
            "away_qb_id",
            "away_qb_name",
            "spread_line",
            "result",
        ]
    ]

    # Exclude future games from the current season (keeping upcoming week's games)
    if end_season == current_season:
        data = data[(data["season"] != current_season) | (data["week"] <= current_week)]

    return data
