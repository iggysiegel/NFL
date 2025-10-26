"""Module for cleaning data scraped from nflreadpy.

This module provided classes to clean team statistics and game schedules data from
nflreadypy.
"""

import pandas as pd


class TeamStatCleaner:  # pylint: disable=too-few-public-methods
    """Clean features for team statistics."""

    def __init__(self, data: pd.DataFrame):
        """Initialize the TeamStatCleaner."""
        self.data = data.copy()
        self.static_features = [
            "season",
            "week",
            "team",
            "opponent_team",
        ]
        self.ewa_features = [
            "completions",
            "attempts",
            "passing_yards",
            "passing_tds",
            "passing_interceptions",
            "sacks_suffered",
            "sack_yards_lost",
            "sack_fumbles",
            "sack_fumbles_lost",
            "passing_air_yards",
            "passing_yards_after_catch",
            "passing_first_downs",
            "passing_epa",
            "passing_cpoe",
            "passing_2pt_conversions",
            "carries",
            "rushing_yards",
            "rushing_tds",
            "rushing_fumbles",
            "rushing_fumbles_lost",
            "rushing_first_downs",
            "rushing_epa",
            "rushing_2pt_conversions",
            "receptions",
            "targets",
            "receiving_yards",
            "receiving_tds",
            "receiving_fumbles",
            "receiving_fumbles_lost",
            "receiving_air_yards",
            "receiving_yards_after_catch",
            "receiving_first_downs",
            "receiving_epa",
            "receiving_2pt_conversions",
            "special_teams_tds",
            "def_tackles_solo",
            "def_tackles_with_assist",
            "def_tackle_assists",
            "def_tackles_for_loss",
            "def_tackles_for_loss_yards",
            "def_fumbles_forced",
            "def_sacks",
            "def_sack_yards",
            "def_qb_hits",
            "def_interceptions",
            "def_interception_yards",
            "def_pass_defended",
            "def_tds",
            "def_fumbles",
            "def_safeties",
            "misc_yards",
            "fumble_recovery_own",
            "fumble_recovery_yards_own",
            "fumble_recovery_opp",
            "fumble_recovery_yards_opp",
            "fumble_recovery_tds",
            "penalties",
            "penalty_yards",
            "timeouts",
            "punt_returns",
            "punt_return_yards",
            "kickoff_returns",
            "kickoff_return_yards",
            "fg_made",
            "fg_att",
            "fg_missed",
            "fg_blocked",
            "fg_long",
            "fg_pct",
            "fg_made_0_19",
            "fg_made_20_29",
            "fg_made_30_39",
            "fg_made_40_49",
            "fg_made_50_59",
            "fg_made_60_",
            "fg_missed_0_19",
            "fg_missed_20_29",
            "fg_missed_30_39",
            "fg_missed_40_49",
            "fg_missed_50_59",
            "fg_missed_60_",
            "fg_made_distance",
            "fg_missed_distance",
            "fg_blocked_distance",
            "pat_made",
            "pat_att",
            "pat_missed",
            "pat_blocked",
            "pat_pct",
            "gwfg_made",
            "gwfg_att",
            "gwfg_missed",
            "gwfg_blocked",
            "gwfg_distance",
        ]
        self.impute_missing_values()

    def impute_missing_values(self):
        """Impute missing values in EWA columns with column means."""
        self.data[self.ewa_features] = self.data[self.ewa_features].fillna(
            self.data[self.ewa_features].mean()
        )


class GameScheduleCleaner:
    """Clean features for game schedules."""

    def __init__(self, data: pd.DataFrame):
        """Initialize the GameScheduleCleaner."""
        self.data = data.copy()

        # Map historical team names to current ones
        team_name_map = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
        self.data["home_team"] = self.data["home_team"].replace(team_name_map)
        self.data["away_team"] = self.data["away_team"].replace(team_name_map)

        self.static_features = [
            "game_id",
            "season",
            "game_type",
            "week",
            "gameday",
            "weekday",
            "gametime",
            "opponent_team",
            "opponent_score",
            "team",
            "score",
            "location",
            "result",
            "total",
            "overtime",
            "old_game_id",
            "gsis",
            "nfl_detail_id",
            "pfr",
            "pff",
            "espn",
            "ftn",
            "opponent_rest",
            "rest",
            "opponent_moneyline",
            "moneyline",
            "spread_line",
            "opponent_spread_odds",
            "spread_odds",
            "total_line",
            "under_odds",
            "over_odds",
            "div_game",
            "roof",
            "surface",
            "temp",
            "wind",
            "opponent_qb_id",
            "qb_id",
            "opponent_qb_name",
            "qb_name",
            "opponent_coach",
            "coach",
            "referee",
            "stadium_id",
            "stadium",
            "is_home",
        ]
        self.ewa_features = [
            "score",
            "opponent_score",
            "result",
            "total",
            "rest",
            "opponent_rest",
            "opponent_moneyline",
            "moneyline",
            "spread_line",
            "opponent_spread_odds",
            "spread_odds",
            "total_line",
            "under_odds",
            "over_odds",
        ]
        self.convert_to_team_game_data()
        self.create_ewa_features()
        self.impute_missing_values()

    def convert_to_team_game_data(self):
        """Convert game-level data to team-game-level data."""
        home_data = self.data.rename(
            columns={
                "home_team": "team",
                "away_team": "opponent_team",
                "home_score": "score",
                "away_score": "opponent_score",
                "home_rest": "rest",
                "away_rest": "opponent_rest",
                "home_moneyline": "moneyline",
                "away_moneyline": "opponent_moneyline",
                "home_spread_odds": "spread_odds",
                "away_spread_odds": "opponent_spread_odds",
                "home_qb_id": "qb_id",
                "away_qb_id": "opponent_qb_id",
                "home_qb_name": "qb_name",
                "away_qb_name": "opponent_qb_name",
                "home_coach": "coach",
                "away_coach": "opponent_coach",
            }
        ).assign(is_home=True)

        away_data = self.data.rename(
            columns={
                "home_team": "opponent_team",
                "away_team": "team",
                "home_score": "opponent_score",
                "away_score": "score",
                "home_rest": "opponent_rest",
                "away_rest": "rest",
                "home_moneyline": "opponent_moneyline",
                "away_moneyline": "moneyline",
                "home_spread_odds": "opponent_spread_odds",
                "away_spread_odds": "spread_odds",
                "home_qb_id": "opponent_qb_id",
                "away_qb_id": "qb_id",
                "home_qb_name": "opponent_qb_name",
                "away_qb_name": "qb_name",
                "home_coach": "opponent_coach",
                "away_coach": "coach",
            }
        ).assign(
            is_home=False,
            result=lambda x: -x["result"],
            spread_line=lambda x: -x["spread_line"],
        )

        self.data = pd.concat([home_data, away_data], ignore_index=True)
        self.data = self.data.sort_values(["season", "week", "team"]).reset_index(
            drop=True
        )

    def create_ewa_features(self):
        """Create EWA features."""
        ewa_features = []
        for feature in self.ewa_features:
            self.data[f"{feature}_ewa"] = self.data[feature]
            ewa_features.append(f"{feature}_ewa")
        self.ewa_features = ewa_features

    def impute_missing_values(self):
        """Impute missing values in EWA columns with column means."""
        self.data[self.ewa_features] = self.data[self.ewa_features].fillna(
            self.data[self.ewa_features].mean()
        )
