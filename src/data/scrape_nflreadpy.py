"""Module for scraping NFL team statistics and game schedules using nflreadpy."""

import nflreadpy as nfl
import pandas as pd


def scrape_nflreadpy(
    start_season: int, end_season: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scrape NFL team statistics and game schedules for a range of seasons using
    nflreadpy.

    Data leakage is avoided by ensuring that, for the current season, only team
    statistics data from completed weeks is included.

    Args:
        start_season: The starting season (inclusive).
        end_season: The ending season (exclusive).
    Returns:
        A tuple containing two DataFrames:
        - team_stats: DataFrame with team statistics for the specified seasons.
        - game_schedules: DataFrame with game schedules for the specified seasons.
    """
    if end_season > nfl.get_current_season():
        raise ValueError("End season cannot be in the future.")

    team_stats = nfl.load_team_stats(
        list(range(start_season, end_season + 1))
    ).to_pandas()

    game_schedules = nfl.load_schedules(
        list(range(start_season, end_season + 1))
    ).to_pandas()

    if end_season == nfl.get_current_season():
        team_stats = team_stats[
            (team_stats["season"] != nfl.get_current_season())
            | (team_stats["week"] < nfl.get_current_week())
        ]
        game_schedules = game_schedules[
            (game_schedules["season"] != nfl.get_current_season())
            | (game_schedules["week"] <= nfl.get_current_week())
        ]
        game_schedules = game_schedules[game_schedules["result"].notnull()]

    return team_stats, game_schedules
