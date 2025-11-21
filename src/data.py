"""Module for loading and preprocessing NFL schedule data."""

import json

import nflreadpy as nfl
import numpy as np
import pandas as pd

from src.paths import SRC_DIR


class DataLoader:
    """Load and preprocess NFL schedule data for a range of seasons."""

    def __init__(self, start_season: int, end_season: int):
        """Initialize the DataLoader.

        Args:
            start_season: First season to load.
            end_season: Final season to load.
        """
        self.start_season = start_season
        self.end_season = end_season
        self.current_season = nfl.get_current_season()
        self.current_week = nfl.get_current_week()

        if self.start_season < 2000 or self.end_season > self.current_season:
            raise ValueError("Data available between 2000 and the current season only.")

        self.load_config()
        self.load_data()
        self.add_open_lines()
        self.add_game_flags()
        self.add_surface_advantage()
        self.add_rest_advantage()
        self.select_features()

    def load_config(self):
        """Load configuration file."""
        config_path = SRC_DIR / "config.json"
        with open(config_path, "r") as f:
            self.config = json.load(f)

    def load_data(self):
        """Load NFL schedule data for the given season range."""
        # Load schedule data
        seasons = list(range(self.start_season, self.end_season + 1))
        self.data = nfl.load_schedules(seasons).to_pandas()

        # Normalize old team abbreviations
        self.data["home_team"] = self.data["home_team"].replace(
            self.config["team_name_map"]
        )
        self.data["away_team"] = self.data["away_team"].replace(
            self.config["team_name_map"]
        )

        # Exclude future games this season but allow current week's games
        if self.end_season == self.current_season:
            self.data = self.data[
                (self.data["season"] != self.current_season)
                | (self.data["week"] <= self.current_week)
            ]

    def add_open_lines(self):
        """Add open spread line information."""
        lines = pd.read_csv(
            "https://raw.githubusercontent.com/greerreNFL/nfelomarket_data/"
            "refs/heads/main/Data/lines.csv"
        )
        lines = lines[["season", "week", "home_team", "away_team", "home_spread_open"]]
        lines["home_team"] = lines["home_team"].replace(self.config["team_name_map"])
        lines["away_team"] = lines["away_team"].replace(self.config["team_name_map"])
        lines = lines.rename(columns={"home_spread_open": "spread_line_open"})
        lines["spread_line_open"] = -lines["spread_line_open"]
        self.data = pd.merge(
            self.data,
            lines,
            on=["season", "week", "home_team", "away_team"],
            how="left",
        )

    def add_game_flags(self):
        """Add neutral site, divisional, and playoff flags."""
        self.data["is_neutral"] = self.data["location"].map({"Home": 0, "Neutral": 1})
        self.data = self.data.rename(columns={"div_game": "is_divisional"})
        self.data["is_playoff"] = np.where(self.data["game_type"] == "REG", 0, 1)

    def add_surface_advantage(self):
        """Compute primary surface and cross-surface flags."""
        # Map the game surface
        self.data["is_artificial"] = self.data["surface"].map(
            self.config["surface_map"]
        )

        # Fill missing surfaces using previous season's surface
        missing_mask = self.data["is_artificial"].isna()
        team_season_primary_surface = (
            self.data.dropna(subset=["is_artificial"])
            .groupby(["home_team", "season"])["is_artificial"]
            .mean()
            .round()
            .astype(int)
        )
        self.data.loc[missing_mask, "is_artificial"] = self.data.loc[
            missing_mask
        ].apply(
            lambda row: team_season_primary_surface.get(
                (row["home_team"], row["season"] - 1), 0
            ),
            axis=1,
        )

        # Compute home / away primary surface
        self.data["home_primary_surface"] = self.data.apply(
            lambda row: team_season_primary_surface.get(
                (row["home_team"], row["season"]), 0
            ),
            axis=1,
        )
        self.data["away_primary_surface"] = self.data.apply(
            lambda row: team_season_primary_surface.get(
                (row["away_team"], row["season"]), 0
            ),
            axis=1,
        )

        # Compute cross-surface flags
        self.data["home_grass_to_turf"] = (
            (self.data["home_primary_surface"] == 0)
            & (self.data["is_artificial"] == 1)
            & (self.data["is_neutral"] == 1)
        ).astype(int)
        self.data["home_turf_to_grass"] = (
            (self.data["home_primary_surface"] == 1)
            & (self.data["is_artificial"] == 0)
            & (self.data["is_neutral"] == 1)
        ).astype(int)
        self.data["away_grass_to_turf"] = (
            (self.data["away_primary_surface"] == 0) & (self.data["is_artificial"] == 1)
        ).astype(int)
        self.data["away_turf_to_grass"] = (
            (self.data["away_primary_surface"] == 1) & (self.data["is_artificial"] == 0)
        ).astype(int)

    def add_rest_advantage(self):
        """Compute rest advantage."""
        self.data["rest_advantage"] = (
            self.data["home_rest"] - self.data["away_rest"]
        ).clip(-4, 4)

    def select_features(self):
        """Restrict dataset to modeling features."""
        cols_to_keep = [
            "season",
            "week",
            "home_team",
            "away_team",
            "is_neutral",
            "is_divisional",
            "is_playoff",
            "is_artificial",
            "home_primary_surface",
            "away_primary_surface",
            "home_grass_to_turf",
            "home_turf_to_grass",
            "away_grass_to_turf",
            "away_turf_to_grass",
            "rest_advantage",
            "home_qb_id",
            "home_qb_name",
            "away_qb_id",
            "away_qb_name",
            "spread_line_open",
            "spread_line",
            "result",
        ]
        self.data = self.data[cols_to_keep]
