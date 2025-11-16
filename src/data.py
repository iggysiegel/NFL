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
        self.add_game_flags()
        self.add_rest_advantage()
        self.add_surface_advantage()
        self.add_tz_advantage()
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

    def add_game_flags(self):
        """Add neutral site, divisional, and playoff flags."""
        self.data["is_neutral"] = self.data["location"].map({"Home": 0, "Neutral": 1})
        self.data = self.data.rename(columns={"div_game": "is_divisional"})
        self.data["is_playoff"] = np.where(self.data["game_type"] == "REG", 0, 1)

    def add_rest_advantage(self):
        """Compute rest advantage."""
        self.data["rest_advantage"] = (
            self.data["home_rest"] - self.data["away_rest"]
        ).clip(-4, 4)

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
            (self.data["home_primary_surface"] == 0) & (self.data["is_artificial"] == 1)
        ).astype(int)
        self.data["home_turf_to_grass"] = (
            (self.data["home_primary_surface"] == 1) & (self.data["is_artificial"] == 0)
        ).astype(int)
        self.data["away_grass_to_turf"] = (
            (self.data["away_primary_surface"] == 0) & (self.data["is_artificial"] == 1)
        ).astype(int)
        self.data["away_turf_to_grass"] = (
            (self.data["away_primary_surface"] == 1) & (self.data["is_artificial"] == 0)
        ).astype(int)

    def add_tz_advantage(self):
        """Compute circadian timezone advantage for each game."""
        peak_time = pd.Timedelta(hours=14)

        tz_advantages = []

        for row in self.data.itertuples():
            kickoff = pd.Timestamp(row.gametime)
            season = row.season
            home = row.home_team
            away = row.away_team

            def resolve_offset(team):
                if team in self.config["timezone_offset_override"]:
                    override = self.config["timezone_offset_override"][team]
                    if season <= override["season"]:
                        return override["tz_override"]
                return self.config["timezone_offset"][team]

            home_offset = pd.Timedelta(hours=resolve_offset(home))
            away_offset = pd.Timedelta(hours=resolve_offset(away))

            kickoff_td = pd.Timedelta(hours=kickoff.hour, minutes=kickoff.minute)

            home_delta = abs((kickoff_td - home_offset) - peak_time)
            away_delta = abs((kickoff_td - away_offset) - peak_time)

            tz_advantages.append((away_delta - home_delta).total_seconds() / 3600.0)

        self.data["time_advantage"] = tz_advantages

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
            "time_advantage",
            "home_qb_id",
            "home_qb_name",
            "away_qb_id",
            "away_qb_name",
            "spread_line",
            "result",
        ]
        self.data = self.data[cols_to_keep]
