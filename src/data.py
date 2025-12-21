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
        self.add_qb_experience()
        self.update_current_week_qbs()
        self.select_features()

        # Exclude future games this season but allow current week's games
        self.data = self.data[self.data["season"] >= self.start_season]
        self.data = self.data[self.data["season"] <= self.end_season]
        if self.end_season == self.current_season:
            self.data = self.data[
                (self.data["season"] != self.current_season)
                | (self.data["week"] <= self.current_week)
            ]

    def load_config(self):
        """Load configuration file containing configuration mappings."""
        config_path = SRC_DIR / "config.json"
        with open(config_path, "r") as f:
            self.config = json.load(f)

    def load_data(self):
        """Load NFL schedule data and normalize team abbreviations."""
        # Load schedule data
        self.data = nfl.load_schedules().to_pandas()

        # Normalize old team abbreviations
        self.data["home_team"] = self.data["home_team"].replace(
            self.config["team_name_map"]
        )
        self.data["away_team"] = self.data["away_team"].replace(
            self.config["team_name_map"]
        )

    def add_open_lines(self):
        """Add opening spread lines from external data source."""
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
        """Add neutral site, divisional, and playoff game flags."""
        self.data["is_neutral"] = self.data["location"].map({"Home": 0, "Neutral": 1})
        self.data = self.data.rename(columns={"div_game": "is_divisional"})
        self.data["is_playoff"] = np.where(self.data["game_type"] == "REG", 0, 1)

    def add_surface_advantage(self):
        """Compute primary surface and identify cross-surface games.

        For each team-season, determine whether they primarily play on artificial turf
        or grass. Then identify where teams play on an unfamiliar surface.
        """
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
        """Calculate rest advantage as the difference in days of rest between teams.

        Positive values indicate home team had more rest.
        """
        self.data["rest_advantage"] = (
            self.data["home_rest"] - self.data["away_rest"]
        ).clip(-4, 4)

    def add_qb_experience(self):
        """Calculate cumulative game experience for each quarterback.

        Experience is the number of games started prior to each game.
        """
        qb_long = pd.concat(
            [
                self.data[["season", "week", "home_qb_id"]].rename(
                    columns={"home_qb_id": "qb_id"}
                ),
                self.data[["season", "week", "away_qb_id"]].rename(
                    columns={"away_qb_id": "qb_id"}
                ),
            ]
        )
        qb_long = qb_long.sort_values(["season", "week", "qb_id"])
        qb_long["experience"] = qb_long.groupby("qb_id").cumcount()

        home_exp = qb_long.rename(
            columns={"qb_id": "home_qb_id", "experience": "home_qb_experience"}
        )
        away_exp = qb_long.rename(
            columns={"qb_id": "away_qb_id", "experience": "away_qb_experience"}
        )

        self.data = pd.merge(
            self.data,
            home_exp[["season", "week", "home_qb_id", "home_qb_experience"]],
            on=["season", "week", "home_qb_id"],
            how="left",
        )
        self.data = pd.merge(
            self.data,
            away_exp[["season", "week", "away_qb_id", "away_qb_experience"]],
            on=["season", "week", "away_qb_id"],
            how="left",
        )

    def update_current_week_qbs(self):
        """Update QBs for current week using latest depth chart data."""
        # Load latest depth chart
        depth = nfl.load_depth_charts().to_pandas()
        latest_dt = depth["dt"].max()
        depth_latest = depth[
            (depth["dt"] == latest_dt)
            & (depth["pos_abb"] == "QB")
            & (depth["pos_rank"] == 1)
        ][["team", "player_name", "gsis_id"]]
        qb_mapping = depth_latest.set_index("team")[["player_name", "gsis_id"]].to_dict(
            "index"
        )

        # Function to update QB info for a row
        def update_qb_info(row):
            if (
                row["season"] == self.current_season
                and row["week"] == self.current_week
            ):
                if row["home_team"] in qb_mapping:
                    row["home_qb_name"] = qb_mapping[row["home_team"]]["player_name"]
                    row["home_qb_id"] = qb_mapping[row["home_team"]]["gsis_id"]
                if row["away_team"] in qb_mapping:
                    row["away_qb_name"] = qb_mapping[row["away_team"]]["player_name"]
                    row["away_qb_id"] = qb_mapping[row["away_team"]]["gsis_id"]
            return row

        # Apply the update
        self.data = self.data.apply(update_qb_info, axis=1)

    def select_features(self):
        """Restrict dataset to final modeling features."""
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
            "home_qb_experience",
            "away_qb_id",
            "away_qb_name",
            "away_qb_experience",
            "spread_line_open",
            "spread_line",
            "result",
        ]
        self.data = self.data[cols_to_keep]
