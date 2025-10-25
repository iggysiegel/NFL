"""Module for processing data scraped from nflreadpy.

This module provided classes to clean and transform team statistics and game schedules,
compute exponentially weighted averages (EWA) for team stats, and combine these data
sources into a processed dataset ready for modeling or analysis.
"""

import pandas as pd


class TeamStatCleaner:
    """Clean features for team statistics, and compute EWA for numeric stats."""

    def __init__(self, df: pd.DataFrame, revert: float, span: int):
        """Initialize the TeamStatCleaner."""
        self.df = df.copy()
        self.revert = revert
        self.span = span
        self.metadata_cols = [
            "season",
            "week",
            "team",
            "opponent_team",
        ]
        self.numeric_cols = None
        self.feature_engineering()
        self.select_features()
        self.impute_missing_values()
        self.apply_ewa()

    def feature_engineering(self):
        """Create additional features from existing team statistics."""
        self.df["passing_epa_per_attempt"] = (
            self.df["passing_epa"] / self.df["attempts"]
        )
        self.df["completion_pct"] = self.df["completions"] / self.df["attempts"]
        self.df["passing_yards_per_attempt"] = (
            self.df["passing_yards"] / self.df["attempts"]
        )
        self.df["rushing_epa_per_carry"] = self.df["rushing_epa"] / self.df["carries"]
        self.df["rushing_yards_per_carry"] = (
            self.df["rushing_yards"] / self.df["carries"]
        )
        self.df["receiving_yards_after_catch_per_completion"] = (
            self.df["receiving_yards_after_catch"] / self.df["receptions"]
        )
        self.df["kickoff_return_avg_yards"] = (
            self.df["kickoff_return_yards"] / self.df["kickoff_returns"]
        )
        self.df["punt_return_avg_yards"] = (
            self.df["punt_return_yards"] / self.df["punt_returns"]
        )
        self.df["def_turnovers"] = (
            self.df["def_fumbles_forced"] + self.df["def_interceptions"]
        )
        self.df["def_avg_sack"] = self.df["def_sack_yards"] / self.df["def_sacks"]
        self.df["def_pressure"] = self.df["def_sacks"] + self.df["def_qb_hits"]
        self.df["def_avg_tackles_for_loss"] = (
            self.df["def_tackles_for_loss_yards"] / self.df["def_tackles_for_loss"]
        )
        self.df["turnovers"] = (
            self.df["passing_interceptions"]
            + self.df["sack_fumbles"]
            + self.df["rushing_fumbles"]
            + self.df["receiving_fumbles"]
        )
        self.df["avg_penalty"] = self.df["penalty_yards"] / self.df["penalties"]

    def select_features(self):
        """Select relevant features and handle missing values."""
        self.numeric_cols = [
            # Passing
            "passing_epa",
            "passing_epa_per_attempt",
            "completions",
            "completion_pct",
            "passing_cpoe",
            "passing_yards",
            "passing_yards_per_attempt",
            "passing_first_downs",
            "passing_tds",
            # Rushing
            "rushing_epa",
            "rushing_epa_per_carry",
            "carries",
            "rushing_yards",
            "rushing_yards_per_carry",
            "rushing_first_downs",
            "rushing_tds",
            # Receptions
            "receiving_epa",
            "receiving_yards_after_catch",
            "receiving_yards_after_catch_per_completion",
            # Special teams
            "kickoff_return_avg_yards",
            "punt_return_avg_yards",
            "fg_made",
            "fg_pct",
            "pat_pct",
            # Defense
            "def_turnovers",
            "def_sacks",
            "def_avg_sack",
            "def_qb_hits",
            "def_pressure",
            "def_tackles_for_loss",
            "def_avg_tackles_for_loss",
            "def_pass_defended",
            # Negative events
            "turnovers",
            "penalties",
            "penalty_yards",
            "avg_penalty",
            "sacks_suffered",
            "sack_yards_lost",
        ]
        self.df = self.df[self.metadata_cols + self.numeric_cols]

    def impute_missing_values(self):
        """Impute missing values in numeric columns with column means."""
        self.df[self.numeric_cols] = self.df[self.numeric_cols].fillna(
            self.df[self.numeric_cols].mean()
        )

    def apply_ewa(self):
        """Compute exponentially weighted averages (EWA) for numeric stats."""
        ewa_data = []

        # Compute league-wide and team-specific priors
        league_priors = (
            self.df.groupby("season")[self.numeric_cols].mean().reset_index()
        )
        team_priors = (
            self.df.groupby(["season", "team"])[self.numeric_cols].mean().reset_index()
        )

        for season, df_season in self.df.groupby("season"):
            for team, df_team in df_season.groupby("team"):

                # Skip first season since no prior data available
                if (season - 1) not in self.df["season"].values:
                    continue

                # Get league and team priors from previous season
                alpha = league_priors[league_priors["season"] == season - 1].copy()
                beta = team_priors[
                    (team_priors["season"] == season - 1)
                    & (team_priors["team"] == team)
                ].copy()

                # Combine league and team priors using revert factor
                prior_stats = dict(
                    zip(
                        self.numeric_cols,
                        (
                            self.revert * alpha[self.numeric_cols].values
                            + (1 - self.revert) * beta[self.numeric_cols].values
                        )[0],
                    )
                )

                # Create a synthetic 'week 0' prior row
                prior_row = pd.DataFrame(
                    {
                        "season": [season],
                        "week": [0],
                        "team": [team],
                        "opponent_team": ["prior"],
                        **prior_stats,
                    }
                )

                # Append prior row to actual games
                df_team_aug = pd.concat([prior_row, df_team], ignore_index=True)

                # Compute EWA and shift to use only past games
                ewa = (
                    df_team_aug[self.numeric_cols]
                    .ewm(span=self.span, adjust=False)
                    .mean()
                    .shift(1)
                )

                # Reassamble dataframe and remove synthetic week 0 row
                df_team_final = pd.concat(
                    [
                        df_team_aug[self.metadata_cols].reset_index(drop=True),
                        ewa.reset_index(drop=True),
                    ],
                    axis=1,
                )
                df_team_final = df_team_final[df_team_final["week"] > 0]

                ewa_data.append(df_team_final)

        # Combine all teams and sort by season/week
        self.df = (
            pd.concat(ewa_data, ignore_index=True)
            .sort_values(by=["season", "week"])
            .reset_index(drop=True)
        )


class GameScheduleCleaner:
    """Clean features for game schedules."""

    def __init__(self, df: pd.DataFrame):
        """Initialize the GameScheduleCleaner."""
        self.df = df.copy()
        self.feature_engineering()
        self.select_features()

    def feature_engineering(self):
        """Create additional features from existing game schedule data."""
        # Primetime games
        hours = self.df["gametime"].str.split(":").str[0].astype(int)
        is_prime_day = self.df["weekday"].isin(["Thursday", "Sunday", "Monday"])
        self.df["is_primetime"] = ((hours > 20) & is_prime_day).astype(int)

        # Neutral site games
        self.df["is_neutral"] = (self.df["location"] == "Neutral").astype(int)

        # Indoor games
        self.df["is_indoors"] = self.df["roof"].isin(["dome", "closed"]).astype(int)

        # Playing surface
        surfaces = self.df["surface"].fillna("").str.lower().str.strip()
        self.df["surface"] = surfaces.isin(["grass"]).astype(int)

        # Rest advantage
        self.df["rest"] = self.df["home_rest"] - self.df["away_rest"]

    def select_features(self):
        """Select relevant features from game schedule data."""
        cols = [
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "result",
            "is_primetime",
            "is_neutral",
            "div_game",
            "is_indoors",
            "surface",
            "rest",
            "away_moneyline",
            "home_moneyline",
            "spread_line",
            "away_spread_odds",
            "home_spread_odds",
            "total_line",
            "under_odds",
            "over_odds",
            "away_qb_id",
            "home_qb_id",
            "away_qb_name",
            "home_qb_name",
            "stadium_id",
            "stadium",
        ]
        self.df = self.df[cols]


class NflReadPyProcessor:
    """Process data from nflreadpy by combining team stats and game schedules."""

    def __init__(self, team_stats, game_schedules, revert, span):
        """Initialize the NflReadPyProcessor."""
        self.team_stat_cleaner = TeamStatCleaner(team_stats, revert, span)
        self.game_schedule_cleaner = GameScheduleCleaner(game_schedules)

    def get_team_stats(self, season: int, week: int, team: str) -> pd.DataFrame:
        """Return the stats for a given team and week, dropping metadata columns."""
        return (
            self.team_stat_cleaner.df[
                (self.team_stat_cleaner.df["season"] == season)
                & (self.team_stat_cleaner.df["week"] == week)
                & (self.team_stat_cleaner.df["team"] == team)
            ]
            .drop(columns=self.team_stat_cleaner.metadata_cols)
            .reset_index(drop=True)
        )

    def process(self) -> pd.DataFrame:
        """Combine cleaned team statistics and game schedules into a processed dataset.

        For each game, compute the home team's advantage over the away team
        by subtracting away stats from home stats.

        Returns:
            Processed game-level dataset with schedule info and stat differences.
        """
        processed_rows = []

        for i, game in enumerate(self.game_schedule_cleaner.df.itertuples()):
            parts = game.game_id.split("_")
            season = int(parts[0])
            week = int(parts[1])
            away_team = parts[2]
            home_team = parts[3]

            if season not in self.team_stat_cleaner.df["season"].values:
                continue

            away_stats = self.get_team_stats(season, week, away_team)
            home_stats = self.get_team_stats(season, week, home_team)

            advantage = home_stats - away_stats

            game_info = pd.DataFrame(
                [self.game_schedule_cleaner.df.iloc[i]]
            ).reset_index(drop=True)
            combined = pd.concat([game_info, advantage], axis=1)

            processed_rows.append(combined)

        return pd.concat(processed_rows, ignore_index=True)
