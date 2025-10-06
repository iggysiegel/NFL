"""Perform feature engineering on processed data.

This module provides reusable functions to transform processed NFL game data
into engineered game-level features for modeling.

Usage:
    from src.features.feature_pipeline import main

    # Generate engineered features from 2010–2024
    features_df = main(2010, 2024)

Notes:
- Future versions may include automated feature selection and expanded
  feature engineering.
"""

# Imports, constants
# ------------------
import os
import re

import pandas as pd

from .contextual_features import add_contextual_features

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESS_DATA_DIR = os.path.join(MODULE_DIR, "../../data/processed")
METADATA_FEATURES = [
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "result",
    "home_line",
    "over_under",
    "roof",
    "surface",
    "attendance",
]


# Low-level helpers
# -----------------
def read_data(
    start_season: int, end_season: int, process_data_dir: str = PROCESS_DATA_DIR
) -> pd.DataFrame:
    """
    Read processed seasonal data and return a single concatenated DataFrame.

    Args:
        start_season: The first season to include.
        end_season: The last season to include.
        process_data_dir: Directory containing the processed files (optional).

    Returns:
        pd.DataFrame: A DataFrame containing data from all seasons in specified range.
    """
    df_out = pd.DataFrame()

    for season in range(start_season, end_season + 1):
        try:
            file_path = os.path.join(process_data_dir, f"{season}_processed.csv")
            temp_df = pd.read_csv(file_path)
            df_out = pd.concat([df_out, temp_df])
        except FileNotFoundError:
            print(f"Warning: Processed data for season {season} not found, skipping.")

    df_out["result"] = df_out["home_final_score"] - df_out["away_final_score"]

    return df_out


def convert_to_team_game(df_game: pd.DataFrame) -> pd.DataFrame:
    """
    Convert game-level DataFrame into team-game-level DataFrame.

    Args:
        df_game: Game-level dataframe.

    Returns:
        pd.DataFrame: Team-game-level DataFrame.
    """
    # Separate home and away features
    home_features = [
        c
        for c in df_game.columns
        if c.startswith("home") and c not in METADATA_FEATURES
    ]
    away_features = [
        c
        for c in df_game.columns
        if c.startswith("away") and c not in METADATA_FEATURES
    ]

    # Create home-game-level DataFrame
    df_home = df_game[
        ["game_id", "season", "week", "home_team", "away_final_score"] + home_features
    ].copy()
    df_home = df_home.rename(
        columns={
            "home_team": "team",
            "home_final_score": "home_final_score_off",
            "away_final_score": "home_final_score_def",
        }
    )
    df_home.columns = df_home.columns.str.replace("home_", "")

    # Create away-game-level DataFrame
    df_away = df_game[
        ["game_id", "season", "week", "away_team", "home_final_score"] + away_features
    ].copy()
    df_away = df_away.rename(
        columns={
            "away_team": "team",
            "away_final_score": "away_final_score_off",
            "home_final_score": "away_final_score_def",
        }
    )
    df_away.columns = df_away.columns.str.replace("away_", "")

    df_team_game = pd.concat([df_home, df_away]).sort_values(by=["season", "week"])
    return df_team_game


# pylint: disable=too-many-locals
def apply_ewa(df_team_game: pd.DataFrame, revert: float, span: int) -> pd.DataFrame:
    """
    Apply Exponential Weighted Average (EWA) algorithm to team-game-level DataFrame.

    For each team in each season, this function:
     1. Computes a prior using a weighted combination of the previous season's
        league-average and team-average stats (controlled by 'revert').
     2. Applies an EWA (controlled by 'span').
     3. Shift the EWA by one to ensure there is no leakage.

    Args:
        df_team_game: Team-game-level DataFrame.
        revert: Weight to revert to league averages. A value of 1 uses only league
            averages, 0 uses only team averages.
        span: Span parameter for the exponential weighted average.

    Returns:
        pd.DataFrame: Team-game-level DataFrame with EWA features applied, sorted by
            season and week.
    """
    # Compute league priors and team priors from previous season
    league_priors = (
        df_team_game.drop(columns=["game_id", "week", "team"])
        .groupby("season")
        .mean()
        .reset_index()
    )
    team_priors = (
        df_team_game.drop(columns=["game_id", "week"])
        .groupby(["season", "team"])
        .mean()
        .reset_index()
    )

    ewa_data = []

    for season, df_season in df_team_game.groupby("season"):
        df_season = df_season.sort_values(["team", "week"])

        for team, df_team in df_season.groupby("team"):
            # Skip first season since no prior data available
            if (season - 1) not in df_team_game["season"].values:
                continue

            # Create a synthetic 'week 0' prior row
            alpha = league_priors[league_priors["season"] == season - 1].drop(
                columns=["season"]
            )
            beta = team_priors[
                (team_priors["season"] == season - 1) & (team_priors["team"] == team)
            ].drop(columns=["season", "team"])
            combined = (revert * alpha.values) + ((1 - revert) * beta.values)
            prior_stats = dict(zip(alpha.columns, combined[0]))

            prior_row = pd.DataFrame(
                {
                    "game_id": ["prior"],
                    "season": [season],
                    "week": [0],
                    "team": [team],
                    **prior_stats,
                }
            )

            # Append prior row to actual games
            df_team_aug = pd.concat([prior_row, df_team])

            # Columns to exclude from EWA calculation
            exclude_cols = ["game_id", "season", "week", "team"]

            # Compute EWA and shift to use only past games
            ewa = (
                df_team_aug.drop(columns=exclude_cols)
                .ewm(span=span, adjust=False)
                .mean()
                .shift(1)
            )

            # Reassamble dataframe and remove synthetic week 0 row
            df_team_final = pd.concat([df_team_aug[exclude_cols], ewa], axis=1)
            df_team_final = df_team_final[df_team_final["week"] > 0]

            ewa_data.append(df_team_final)

    # Combine all teams and sort by season/week
    df_ewa = (
        pd.concat(ewa_data).sort_values(by=["season", "week"]).reset_index(drop=True)
    )
    return df_ewa


def convert_to_game(df_ewa: pd.DataFrame, processed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a team-game-level DataFrame with EWA features into a single game-level
    DataFrame, merging home and away stats and preserving original metadata.

    Args:
        ewa_df: Team-game-level DataFrame with EWA features.
        processed_df: Original processed game-level DataFrame for metadata.

    Returns:
        pd.DataFrame: Game-level DataFrame with EWA features and metadata.
    """
    df_ewa = df_ewa.copy()

    # Identify home team for each row and create a home indicator
    df_ewa["home_team"] = df_ewa["game_id"].str[-3:]
    df_ewa["is_home"] = (df_ewa["team"] == df_ewa["home_team"]).astype(int)

    # Split into home and away DataFrames
    df_home = df_ewa[df_ewa["is_home"] == 1]
    df_away = df_ewa[df_ewa["is_home"] == 0]

    # Merge home and away DataFrames on game_id
    exclude_cols = ["season", "week", "team", "home_team", "is_home"]
    features_to_keep = [col for col in df_ewa.columns if col not in exclude_cols]

    df_game = df_home[features_to_keep].merge(
        df_away[features_to_keep], on="game_id", suffixes=("_home", "_away")
    )
    df_game = processed_df[METADATA_FEATURES].merge(df_game, on="game_id")

    return df_game


def engineer_features(df_game_ewa: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer game-level features by computing home/away and offensive/defensive
    advantages, and removing highly correlated features.

    Args:
        df_game_ewa: Game-level DataFrame with EWA features and metadata.

    Returns:
        pd.DataFrame: DataFrame with final engineered features.
    """
    engineered = pd.DataFrame()
    features_to_process = list(df_game_ewa.columns)

    for feature in features_to_process:

        # Keep metadata features as-is
        if feature in METADATA_FEATURES:
            engineered[feature] = df_game_ewa[feature]

        # Offensive/defensive home-away advantage
        elif re.search(r"_(off|def)_", feature) and re.search(r"_(home|away)", feature):
            # Build standardized column names
            off_home = re.sub(r"_(off|def)_", "_off_", feature)
            off_home = re.sub(r"_(home|away)", "_home", off_home)
            def_home = re.sub(r"_(off|def)_", "_def_", feature)
            def_home = re.sub(r"_(home|away)", "_home", def_home)
            off_away = re.sub(r"_(off|def)_", "_off_", feature)
            off_away = re.sub(r"_(home|away)", "_away", off_away)
            def_away = re.sub(r"_(off|def)_", "_def_", feature)
            def_away = re.sub(r"_(home|away)", "_away", def_away)

            # Base name for engineered feature
            base_name = re.sub(r"_(off|def)_", "_", feature)
            base_name = re.sub(r"_(home|away)", "", base_name)

            # Compute home/away advantage
            engineered[f"{base_name}_adv_home"] = (
                df_game_ewa[off_home] - df_game_ewa[def_away]
            )
            engineered[f"{base_name}_adv_away"] = (
                df_game_ewa[off_away] - df_game_ewa[def_home]
            )

            # Remove from list to avoid double-counting
            features_to_process.remove(off_home)
            features_to_process.remove(def_home)
            features_to_process.remove(off_away)
            features_to_process.remove(def_away)

        # Home/away advantage
        elif re.search(r"_(home|away)", feature):
            # Build standardized column names
            home = re.sub(r"_(home|away)", "_home", feature)
            away = re.sub(r"_(home|away)", "_away", feature)

            # Base name for engineered feature
            base_name = re.sub(r"_(home|away)", "", feature)

            # Compute home/away advantage
            engineered[f"{base_name}_adv"] = df_game_ewa[home] - df_game_ewa[away]

            # Remove from list to avoid double-counting
            features_to_process.remove(home)
            features_to_process.remove(away)

        else:
            raise ValueError(f"Cannot parse feature: {feature}")

    return engineered


# High-level function
# -------------------
def main(
    start_season: int,
    end_season: int,
    revert: float = 1 / 3.0,
    span: int = 5,
    process_data_dir: str = PROCESS_DATA_DIR,
):
    """
    Main pipeline for preparing engineered features from processed game data.

    Args:
        start_season: The first season to include.
        end_season: The last season to include.
        revert: Weight to revert to league averages in EWA algorithm (optional).
        span: Span parameter for the EWA algorithm (optional).
        process_data_dir: Directory containing the processed files (optional).

    Returns:
        pd.DataFrame: Engineered feature DataFrame.
    """
    df_processed = read_data(start_season, end_season, process_data_dir)
    df_contextual = add_contextual_features(df_processed)
    df_team_game = convert_to_team_game(df_processed)
    df_ewa = apply_ewa(df_team_game, revert, span)
    df_ewa_game = convert_to_game(df_ewa, df_processed)
    df_engineer = engineer_features(df_ewa_game)

    cols_to_use = [
        c
        for c in df_engineer.columns
        if c not in df_contextual.columns or c == "game_id"
    ]
    df_merged = df_engineer[cols_to_use].merge(df_contextual, on="game_id", how="inner")
    return df_merged
