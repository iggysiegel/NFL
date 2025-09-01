"""Process Pro Football Reference raw data.

This module provides reusable functions to clean Pro Football Reference
raw data.

Usage:
    from src.processing.pro_football_reference_processor import clean_data

    # Process a raw dataframe
    processed_df = process_data(raw_df)

Notes:
- A future version will include the ability to handle weather data.
"""

# Imports, constants
# ------------------
import pandas as pd

COLUMN_RENAME_MAP = {
    "home_first downs": "home_first_downs",
    "home_net pass yards": "home_pass_yds_net",
    "home_total yards": "home_yds",
    "home_time of possession": "home_time_of_possession",
    "away_first downs": "away_first_downs",
    "away_net pass yards": "away_pass_yds_net",
    "away_total yards": "away_yds",
    "away_time of possession": "away_time_of_possession",
}

ROOF_MAP = {
    "outdoors": "outdoors",
    "retractable roof (open)": "outdoors",
    "dome": "indoors",
    "retractable roof (closed)": "indoors",
}

SURFACE_MAP = {
    "grass": "grass",
    "fieldturf": "turf",
    "astroplay": "turf",
    "sportturf": "turf",
    "matrixturf": "turf",
}

DASH_COLUMNS_MAP = {
    "home_rush-yds-tds": ["home_rush_attempts", "home_rush_yds", "home_rush_tds"],
    "home_cmp-att-yd-td-int": [
        "home_pass_cmp",
        "home_pass_attempts",
        "home_pass_yds",
        "home_pass_tds",
        "home_pass_ints",
    ],
    "home_sacked-yards": ["home_sacks", "home_sack_yards"],
    "home_fumbles-lost": ["home_fumbles", "home_fumbles_lost"],
    "home_penalties-yards": ["home_penalties", "home_penalties_yds"],
    "home_third down conv.": [
        "home_third_down_conversions",
        "home_third_down_attempts",
    ],
    "home_fourth down conv.": [
        "home_fourth_down_conversions",
        "home_fourth_down_attempts",
    ],
    "away_rush-yds-tds": ["away_rush_attempts", "away_rush_yds", "away_rush_tds"],
    "away_cmp-att-yd-td-int": [
        "away_pass_cmp",
        "away_pass_attempts",
        "away_pass_yds",
        "away_pass_tds",
        "away_pass_ints",
    ],
    "away_sacked-yards": ["away_sacks", "away_sack_yards"],
    "away_fumbles-lost": ["away_fumbles", "away_fumbles_lost"],
    "away_penalties-yards": ["away_penalties", "away_penalties_yds"],
    "away_third down conv.": [
        "away_third_down_conversions",
        "away_third_down_attempts",
    ],
    "away_fourth down conv.": [
        "away_fourth_down_conversions",
        "away_fourth_down_attempts",
    ],
}


# Low-level helpers
# -----------------
def clean_roof(value: str) -> str:
    """Clean 'roof' column"""
    if pd.isna(value):
        return None
    return ROOF_MAP.get(value.lower().strip())


def clean_surface(value: str) -> str:
    """Clean 'surface' column"""
    if pd.isna(value):
        return None
    return SURFACE_MAP.get(value.lower().strip())


def clean_time_of_possession(value: str) -> int:
    """Clean 'time_of_possession' column"""
    if pd.isna(value):
        return None
    parts = value.split(":")
    return 60 * int(parts[0]) + int(parts[1])


def clean_dash_column(df: pd.DataFrame, col_to_split: str, new_cols: list[str]) -> None:
    """Splits a column with dash-separated values into multiple columns, modifying
    the dataframe in place.

    Args:
        df: The dataframe containing the column to split
        col_to_split: The name of the column to split
        new_cols: List of new column names
    """
    n_splits = len(new_cols) - 1
    df[new_cols] = df[col_to_split].str.split("-", expand=True, n=n_splits)
    df[new_cols] = df[new_cols].apply(pd.to_numeric, errors="coerce")
    df.drop(columns=[col_to_split], inplace=True)


# High-level function
# -------------------
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a raw Pro Football Reference dataframe by standardizing column names,
    normalizing categorical variables, and splitting dash-separated statistics into
    separate columns.

    Args:
        df: Raw dataframe

    Returns:
        df: Cleaned dataframe
    """
    # Rename columns
    df = df.rename(columns=COLUMN_RENAME_MAP)

    # Roof, surface, time of possession
    df["roof"] = df["roof"].apply(clean_roof)
    df["surface"] = df["surface"].apply(clean_surface)
    for col in ["home_time_of_possession", "away_time_of_possession"]:
        df[col] = df[col].apply(clean_time_of_possession)

    # Dash-separated columns
    for col, new_cols in DASH_COLUMNS_MAP.items():
        clean_dash_column(df, col, new_cols)

    # Drop weather column
    df.drop(columns=["weather"], inplace=True)

    return df
