"""Clean nfl_data_py raw data.

This module provides reusable functions to clean and transform
nfl_data_py raw data into a structured format.

Usage:
    from src.processing.nfl_data_py_cleaner import clean_data

    # Process a raw dataframe
    cleaned_df = clean_data(raw_df)

Notes:
- Future versions may include additional transformations
  to leverage more features in the rich nfl_data_py dataset
"""

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a raw nfl_data_py play-by-play dataframe into a game-level dataframe.

    Args:
        df: Raw dataframe

    Returns:
        final_df: Cleaned dataframe
    """
    # Aggregate offensive stats
    # -------------------------
    team_offensive_data = df.groupby(["posteam", "season", "week"]).agg(
        {
            # Game Information
            "game_id": "first",  # Game ID
            "posteam_score": "max",  # Final score
            "home_team": "first",  # Home team flag
            # Drive Stats
            "drive": "nunique",  # Total number of drives
            # Efficiency Stats
            "epa": ["sum", "mean"],  # Total / average EPA
            "wp": "mean",  # Average win probability
            "score_differential": "mean",  # Average score differential
            "ydsnet": "mean",  # Average yards gained in a drive
            "yards_gained": "mean",  # Average yards gained on a play
            "ydstogo": "mean",  # Average yards to first down
            # Passing Stats
            "first_down_pass": "sum",  # Total first down passes
            "cpoe": "mean",  # Average cpoe
            "qb_dropback": "sum",  # Total dropbacks
            "qb_scramble": "sum",  # Total scrambles
            # Rushing Stats
            "first_down_rush": "sum",  # Total first down rushes
            # Receiving Stats
            "receiving_yards": ["sum", "mean"],  # Total / average receiving yards
            "yards_after_catch": ["sum", "mean"],  # Total / average YAC
            # Negative Plays
            "tackled_for_loss": "sum",  # Total TFLs
            "qb_hit": "sum",  # Total QB hits
            "sack": "sum",  # Total sacks
        }
    )

    # Standardize column names
    team_offensive_data.columns = [
        "_".join(col).strip() for col in team_offensive_data.columns
    ]
    team_offensive_data = team_offensive_data.rename(
        columns={"game_id_first": "game_id", "home_team_first": "home_team"}
    )
    team_offensive_data = team_offensive_data.reset_index()

    # Add "_off" suffix to all stats except identifiers
    team_offensive_data = team_offensive_data.rename(
        columns=lambda col: (
            col + "_off"
            if col not in ["posteam", "season", "week", "game_id", "home_team"]
            else col
        )
    )

    # Aggregate defensive stats
    # -------------------------
    team_mapping = (
        df[["season", "week", "posteam", "defteam"]]
        .drop_duplicates()
        .dropna()
        .reset_index(drop=True)
    )
    team_offensive_data = pd.merge(
        team_offensive_data, team_mapping, on=["season", "week", "posteam"], how="left"
    )
    team_defensive_data = team_offensive_data.copy()
    team_defensive_data = team_defensive_data.rename(
        columns=lambda col: col.replace("_off", "_def") if "_off" in col else col
    )

    # Merge offensive and defensive stats
    # -----------------------------------
    team_data = pd.merge(
        team_offensive_data,
        team_defensive_data,
        left_on=["season", "week", "defteam"],
        right_on=["season", "week", "posteam"],
        suffixes=("", "_def"),
    )
    team_data = team_data.drop(columns=["posteam_def", "defteam_def", "defteam"])

    # Convert from team-game structure to game-level structure
    # --------------------------------------------------------
    home_df = team_data[team_data["posteam"] == team_data["home_team"]].copy()
    away_df = team_data[team_data["posteam"] != team_data["home_team"]].copy()

    home_df = home_df.rename(
        columns={
            col: f"home_{col}"
            for col in home_df.columns
            if col not in ["posteam", "season", "week", "game_id", "home_team"]
        }
    )
    away_df = away_df.rename(
        columns={
            col: f"away_{col}"
            for col in away_df.columns
            if col not in ["posteam", "season", "week", "game_id", "home_team"]
        }
    )

    # Merge home and away
    final_df = home_df.merge(away_df, on=["season", "week", "game_id", "home_team"])
    final_df = final_df.rename(
        columns={
            "home_posteam_score_max_off": "home_final_score",
            "away_posteam_score_max_off": "away_final_score",
        }
    )

    # Reorder columns to put game_id first
    cols = ["game_id", "home_final_score", "away_final_score"] + [
        col
        for col in final_df.columns
        if col
        not in [
            "posteam_x",
            "posteam_y",
            "home_team",
            "game_id",
            "home_final_score",
            "away_final_score",
            "home_posteam_score_max_def",
            "away_posteam_score_max_def",
        ]
    ]
    final_df = final_df[cols]

    # Sort by season/week
    final_df = final_df.sort_values(by=["season", "week"]).reset_index(drop=True)

    return final_df
