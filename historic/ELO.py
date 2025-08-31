import math
import pandas as pd
import numpy as np
import nfl_data_py as nfl

class ELO:
    """
    A NFL prediction model based on the ELO rating system.

    References:
    - https://github.com/fivethirtyeight/nfl-elo-game/tree/master
    - https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/
    """
    
    def __init__(self, start_season, end_season, end_week = None):
        """
        Initialize the ELO model.

        Inputs:
            - start_season: int, The first season to forecast
            - end_season: int, The last season to forecast
            - end_week: int, The last week to forecast
        """
        if start_season < 2010:
            raise ValueError("Odds data begins in 2010 season.")
        self.start_season = start_season
        self.end_season = end_season
        self.end_week = end_week

    def read_games(self):
        """
        Method to read and clean NFL game data from nfl_data_py architecture.
        """
        games_df = nfl.import_schedules(list(range(self.start_season, self.end_season + 1)))

        # Replace team names of teams that have moved or changed names
        games_df["home_team"] = games_df["home_team"].replace({"STL": "LA", "SD": "LAC", "OAK": "LV"})
        games_df["away_team"] = games_df["away_team"].replace({"STL": "LA", "SD": "LAC", "OAK": "LV"})

        # Drop games after the end week of the end season
        if self.end_week is not None:
            games_df.drop(games_df[(games_df["season"] == self.end_season) & (games_df["week"] > self.end_week)].index, inplace = True)

        self.games_df = games_df
        return self

    def forecast(self):
        """
        Method to forecast NFL games using the ELO rating system.
        """
        # Global parameters
        K = 20.0          # The speed at which Elo ratings change
        REVERT = 1/3.0    # Between seasons, a team retains 2/3 of its previous season's rating
        BASELINE = 1500.0 # The average ELO rating

        # Initialize ELO dictionary
        teams = {}
        for team in self.games_df["home_team"].unique():
            teams[team] = {
                "name": team,
                "season": None,
                "elo": None
            }

        # Historical 538 data
        historical_ELO = pd.read_csv("data/538_nfl_games.csv")

        # Forecasting algorithm
        self.games_df["home_elo"] = 0.0
        self.games_df["away_elo"] = 0.0
        self.games_df["home_elo_prob"] = 0.0
        self.games_df["away_elo_prob"] = 0.0

        for row in self.games_df.itertuples():

            home_team, away_team = teams[row.home_team], teams[row.away_team]
            
            # If we don't have an ELO rating, use historical 538 data
            if home_team["elo"] is None or away_team["elo"] is None:
                matched_game = historical_ELO[(historical_ELO["date"] == row.gameday) &
                                                (historical_ELO["team1"] == home_team["name"]) &
                                                (historical_ELO["team2"] == away_team["name"])]
                home_team["elo"] = matched_game["elo1"].values[0]
                away_team["elo"] = matched_game["elo2"].values[0]
                home_team["season"] = row.season
                away_team["season"] = row.season

            # Revert teams at the start of seasons
            else:
                for team in [home_team, away_team]:
                    if team["season"] and row.season != team["season"]:
                        team["elo"] = BASELINE * REVERT + team["elo"] * (1 - REVERT)
                    team["season"] = row.season

            # Update dataframe with ELO rating prior to game
            self.games_df.at[row.Index, "home_elo"] = home_team["elo"]
            self.games_df.at[row.Index, "away_elo"] = away_team["elo"]

            # Calculate ELO probabilities and update dataframe
            elo_diff = home_team["elo"] - away_team["elo"]
            home_team_prob = 1.0 / (math.pow(10.0, (-elo_diff / 400.0)) + 1.0)
            away_team_prob = 1.0 - home_team_prob
            self.games_df.at[row.Index, "home_elo_prob"] = home_team_prob
            self.games_df.at[row.Index, "away_elo_prob"] = away_team_prob

            # After game occurs, margin of victory is used as a K multiplier
            point_diff = abs(row.result)
            partial = (2.2 / (1.0 if row.result == 0.0 else ((elo_diff if row.result > 0.0 else -elo_diff) * 0.001 + 2.2)))
            mult = math.log(max(point_diff, 1) + 1.0) * partial

            # ELO shift based on K and the margin of victory multiplier
            if row.result < 0.0:
                shift = (K * mult) * (-home_team_prob)
            elif row.result == 0.0:
                shift = (K * mult) * (0.5 - home_team_prob)
            else:
                shift = (K * mult) * (1.0 - home_team_prob)
            home_team["elo"] += shift
            away_team["elo"] -= shift

        return self
    
    def evaluate_forecast(self):
        """
        Evaluates the quality of ELO predictions using the average Brier score.
        """
        brier_by_season = {}
        for row in self.games_df.itertuples():
            if row.result != 0.0:
                if row.season not in brier_by_season:
                    brier_by_season[row.season] = [0.0, 0]
                brier = (row.home_elo_prob - (row.result > 0)) ** 2
                brier_by_season[row.season][0] += brier
                brier_by_season[row.season][1] += 1

        average_brier = sum(brier for brier, _ in brier_by_season.values()) / sum(count for _, count in brier_by_season.values())
        print(f"Average Brier Score: {average_brier:.4f}\n")
        for season, (total_brier, count) in brier_by_season.items():
            season_average_brier = total_brier / count
            print(f"{season} Season: {season_average_brier:.4f}")