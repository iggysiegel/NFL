"""Module for processing data from all sources.

Merge all data sources, perform feature engineering, compute EWA for
selected features, and convert team-level data to game-level data.

Notes
-----
- Currently skip the first season since no prior data exists.
"""

import pandas as pd


class DataProcessor:
    """Process data from all sources."""

    def __init__(
        self, revert: float, span: int, team_stats: object, game_schedules: object
    ):
        self.revert = revert
        self.span = span
        self.team_stats = team_stats
        self.game_schedules = game_schedules

        self.data = pd.merge(
            self.team_stats.data,
            self.game_schedules.data,
            on=["season", "week", "team", "opponent_team"],
        )
        self.static_features = list(
            set(self.team_stats.static_features + self.game_schedules.static_features)
        )
        self.ewa_features = list(
            set(self.team_stats.ewa_features + self.game_schedules.ewa_features)
        )

        self.feature_engineering()
        self.apply_ewa()
        self.convert_to_game_data()

    def feature_engineering(self):
        """Create additional features."""
        self.data["passing_epa_per_attempt"] = (
            self.data["passing_epa"] / self.data["attempts"]
        )
        self.ewa_features.append("passing_epa_per_attempt")

        self.data["completion_pct"] = self.data["completions"] / self.data["attempts"]
        self.ewa_features.append("completion_pct")

        self.data["passing_yards_per_attempt"] = (
            self.data["passing_yards"] / self.data["attempts"]
        )
        self.ewa_features.append("passing_yards_per_attempt")

        self.data["rushing_epa_per_carry"] = (
            self.data["rushing_epa"] / self.data["carries"]
        )
        self.ewa_features.append("rushing_epa_per_carry")

        self.data["rushing_yards_per_carry"] = (
            self.data["rushing_yards"] / self.data["carries"]
        )
        self.ewa_features.append("rushing_yards_per_carry")

        self.data["receiving_yards_after_catch_per_completion"] = (
            self.data["receiving_yards_after_catch"] / self.data["receptions"]
        )
        self.ewa_features.append("receiving_yards_after_catch_per_completion")

        self.data["kickoff_return_avg_yards"] = (
            self.data["kickoff_return_yards"] / self.data["kickoff_returns"]
        )
        self.ewa_features.append("kickoff_return_avg_yards")

        self.data["punt_return_avg_yards"] = (
            self.data["punt_return_yards"] / self.data["punt_returns"]
        )
        self.ewa_features.append("punt_return_avg_yards")

        self.data["def_turnovers"] = (
            self.data["def_fumbles_forced"] + self.data["def_interceptions"]
        )
        self.ewa_features.append("def_turnovers")

        self.data["def_avg_sack"] = self.data["def_sack_yards"] / self.data["def_sacks"]
        self.ewa_features.append("def_avg_sack")

        self.data["def_pressure"] = self.data["def_sacks"] + self.data["def_qb_hits"]
        self.ewa_features.append("def_pressure")

        self.data["def_avg_tackles_for_loss"] = (
            self.data["def_tackles_for_loss_yards"] / self.data["def_tackles_for_loss"]
        )
        self.ewa_features.append("def_avg_tackles_for_loss")

        self.data["turnovers"] = (
            self.data["passing_interceptions"]
            + self.data["sack_fumbles"]
            + self.data["rushing_fumbles"]
            + self.data["receiving_fumbles"]
        )
        self.ewa_features.append("turnovers")

        self.data["avg_penalty"] = self.data["penalty_yards"] / self.data["penalties"]
        self.ewa_features.append("avg_penalty")

    def apply_ewa(self):
        """Compute exponentially weighted averages (EWA) for specified features."""
        ewa_data = []

        # Compute league-wide and team-specific priors
        league_priors = (
            self.data.groupby("season")[self.ewa_features].mean().reset_index()
        )
        team_priors = (
            self.data.groupby(["season", "team"])[self.ewa_features]
            .mean()
            .reset_index()
        )

        for season, df_season in self.data.groupby("season"):
            for team, df_team in df_season.groupby("team"):

                # Skip first season since no prior data available
                if (season - 1) not in self.data["season"].values:
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
                        self.ewa_features,
                        (
                            self.revert * alpha[self.ewa_features].values
                            + (1 - self.revert) * beta[self.ewa_features].values
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
                    df_team_aug[self.ewa_features]
                    .ewm(span=self.span, adjust=False)
                    .mean()
                    .shift(1)
                )

                # Reassamble dataframe and remove synthetic week 0 row
                df_team_final = pd.concat(
                    [
                        df_team_aug[self.static_features].reset_index(drop=True),
                        ewa.reset_index(drop=True),
                    ],
                    axis=1,
                )
                df_team_final = df_team_final[df_team_final["week"] > 0]

                ewa_data.append(df_team_final)

        # Combine all teams and sort by season/week
        self.data = (
            pd.concat(ewa_data, ignore_index=True)
            .sort_values(by=["season", "week", "team"])
            .reset_index(drop=True)
        )

    def _get_team_stats(self, season: int, week: int, team: str) -> pd.DataFrame:
        """Return the EWA stats for a given team, week, and season."""
        return (
            self.data[
                (self.data["season"] == season)
                & (self.data["week"] == week)
                & (self.data["team"] == team)
            ][self.ewa_features]
        ).reset_index(drop=True)

    def convert_to_game_data(self):
        """Convert team-game-level data to game-level data."""
        all_games = []

        for game_id in self.data["game_id"].unique():
            parts = game_id.split("_")
            season = int(parts[0])
            week = int(parts[1])
            away_team = parts[2]
            home_team = parts[3]

            away_ewa_stats = self._get_team_stats(season, week, away_team)
            home_ewa_stats = self._get_team_stats(season, week, home_team)
            advantage = home_ewa_stats - away_ewa_stats

            game_static_stats = self.data[
                (self.data["game_id"] == game_id) & (self.data["is_home"])
            ][self.static_features].reset_index(drop=True)

            combined = pd.concat([game_static_stats, advantage], axis=1)
            all_games.append(combined)

        self.data = pd.concat(all_games, ignore_index=True).sort_values(
            by=["season", "week", "team"]
        )

    def select_features(self):
        """TODO: Only keep relevant features"""
