"""Module for scraping NFL team statistics and game schedules using nflreadpy."""

import nflreadpy as nfl
import pandas as pd


class NflReadPyScraper:
    """Scrape NFL team statistics and game schedules for a given season range using
    nflreadpy.

    This class avoids data leakage by ensuring that, for the current season, only
    completed weeks are included in the schedules.
    """

    def __init__(self, start_season: int, end_season: int):
        """
        Args:
            start_season: The starting season (inclusive).
            end_season: The ending season (inclusive).
        """
        self.start_season = start_season
        self.end_season = end_season

        current_season = nfl.get_current_season()
        if self.end_season > current_season:
            raise ValueError(f"End season cannot be in the future ({current_season}).")

    def scrape(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Scrape team statistics and game schedules.

        Returns:
            tuple: A tuple containing two DataFrames:
                - team_stats: Team statistics for the specified seasons.
                - game_schedules: Game schedules for the specified seasons.
        """
        seasons = list(range(self.start_season, self.end_season + 1))

        team_stats = nfl.load_team_stats(seasons).to_pandas()
        game_schedules = nfl.load_schedules(seasons).to_pandas()

        # For the current season, keep only completed games to avoid data leakage
        if self.end_season == nfl.get_current_season():
            current_week = nfl.get_current_week()
            game_schedules = game_schedules[
                (game_schedules["season"] != nfl.get_current_season())
                | (game_schedules["week"] <= current_week)
            ]
            game_schedules = game_schedules[game_schedules["result"].notnull()]

        return team_stats, game_schedules
