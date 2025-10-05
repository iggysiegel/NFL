"""ELO rating system for NFL teams.

This module implements an ELO-based rating system to model
and forecast team performance over multiple NFL seasons.

Usage:
    from src.models.elo import ELO

    # Initialize and forecast
    elo_model = ELO(current_data, historical_elos)
    forecasted_data = elo_model.data

Notes:
- A future version will include the ability to dynamically adjust ELO hyperparameters.
"""

import math

import pandas as pd

K = 20.0  # The speed at which Elo ratings change
REVERT = 1 / 3.0  # Between seasons, a team retains 2/3 of its previous season's rating
BASELINE = 1500.0  # The average ELO rating
HFA = 65.0  # Home field advantage is worth 65 ELO points


class ELO:
    """Computes and updates ELO ratings for NFL teams across multiple seasons."""

    def __init__(self, data: pd.DataFrame, historical_elos: pd.DataFrame) -> None:
        """
        Initializes the ELO model and runs the full forecast.

        Args:
            data: Current season game-level dataframe. Must include:
                ['season', 'home_team', 'away_team', 'result']
            historical_elos: Historical ELO ratings dataframe.
        """
        self.data = data.copy()
        self.historical_elos = historical_elos.copy()
        if self.data["season"].min() < 1990 or self.data["season"].min() > 2021:
            raise ValueError("Data must start between 1990 and 2021.")
        self.initialize_elos()
        self.forecast()

    def initialize_elos(self) -> None:
        """
        Initializes each team's ELO rating based on the earliest season in the
        historical dataset.
        """
        self.elos = {}
        for team in self.data["home_team"].unique():
            sub_df = self.historical_elos[
                (
                    (self.historical_elos["team1"] == team)
                    | (self.historical_elos["team2"] == team)
                )
                & (self.historical_elos["season"] == self.data["season"].min())
            ].head(1)
            if sub_df["team1"].values[0] == team:
                elo = sub_df["elo1"].values[0]
            elif sub_df["team2"].values[0] == team:
                elo = sub_df["elo2"].values[0]
            else:
                raise ValueError("Team not found in historical ELOs.")
            self.elos[team] = {
                "season": self.data["season"].min(),
                "elo": elo,
            }

    def forecast(self) -> None:
        """
        Iteratively updates team ELO ratings and probabilities for each game.

        The following new columns are added to 'self.data':
            - 'home_elo', 'away_elo': Pre-game ELO ratings.
            - 'home_elo_prob', 'away_elo_prob': Win probabilities based on ELO.
            - 'elo_diff': Difference between home and away ELO ratings.
        """
        home_elo_list, away_elo_list = [], []
        home_elo_prob_list, away_elo_prob_list = [], []

        for row in self.data.itertuples():
            home_elo, away_elo, home_prob, away_prob = self._update_elo_for_game(row)
            home_elo_list.append(home_elo)
            away_elo_list.append(away_elo)
            home_elo_prob_list.append(home_prob)
            away_elo_prob_list.append(away_prob)

        self.data["home_elo"] = home_elo_list
        self.data["away_elo"] = away_elo_list
        self.data["home_elo_prob"] = home_elo_prob_list
        self.data["away_elo_prob"] = away_elo_prob_list
        self.data["elo_diff"] = self.data["home_elo"] - self.data["away_elo"]

    def _update_elo_for_game(self, row) -> tuple[float, float, float, float]:
        """
        Updates ELO ratings for a single game and returns pre-game values.

        Args:
            row: A row from the season game-level dataframe.

        Returns:
            Tuple of (home_elo, away_elo, home_elo_prob, away_elo_prob)
        """
        home_team, away_team = row.home_team, row.away_team

        # Handle season transitions
        for team in [home_team, away_team]:
            if self.elos[team]["season"] != row.season:
                self.elos[team]["elo"] = BASELINE * REVERT + self.elos[team]["elo"] * (
                    1 - REVERT
                )
                self.elos[team]["season"] = row.season

        # Compute probabilities
        home_elo, away_elo = self.elos[home_team]["elo"], self.elos[away_team]["elo"]
        elo_diff = home_elo - away_elo + HFA
        home_elo_prob = 1 / (1 + 10 ** (-elo_diff / 400))
        away_elo_prob = 1 - home_elo_prob

        # Compute rating shift
        point_diff = abs(row.result)
        partial = math.log(max(point_diff, 1) + 1.0)

        if row.result > 0:
            mult = partial * (2.2 / (elo_diff * 0.001 + 2.2))
            shift = (K * mult) * (1.0 - home_elo_prob)
        elif row.result < 0:
            mult = partial * (2.2 / (-elo_diff * 0.001 + 2.2))
            shift = K * mult * -home_elo_prob
        else:
            mult = partial * (2.2 / 1.0)
            shift = (K * mult) * (0.5 - home_elo_prob)

        # Apply updates
        self.elos[home_team]["elo"] += shift
        self.elos[away_team]["elo"] -= shift

        return home_elo, away_elo, home_elo_prob, away_elo_prob
