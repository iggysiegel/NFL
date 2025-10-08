"""A nfl_data_py Scraper.

This module provides reusable functions to scrape NFL season data from
nfl_data_py. Each season contains rich play by play data, with unique
game_id keys.

Usage:
    from src.data.nfl_data_py_scraper import scrape_season

    # Process a single season
    scrape_season(2024)

Notes:
- Team abbreviations are standardized for downstream processing.
- Output is CSVs with consistent schema across seasons.
"""

import os
from contextlib import redirect_stdout

import nfl_data_py as nfl
import pandas as pd

TEAM_HREFS = {
    "PHI": "phi",
    "IND": "clt",
    "NO": "nor",
    "TEN": "oti",
    "WAS": "was",
    "SEA": "sea",
    "CHI": "chi",
    "DEN": "den",
    "ATL": "atl",
    "NYJ": "nyj",
    "TB": "tam",
    "GB": "gnb",
    "CLE": "cle",
    "JAX": "jax",
    "MIA": "mia",
    "DAL": "dal",
    "KC": "kan",
    "DET": "det",
    "NE": "nwe",
    "CAR": "car",
    "SF": "sfo",
    "BUF": "buf",
    "MIN": "min",
    "BAL": "rav",
    "CIN": "cin",
    "NYG": "nyg",
    "LA": "ram",
    "LV": "rai",
    "LAC": "sdg",
    "PIT": "pit",
    "ARI": "crd",
    "HOU": "htx",
}


def scrape_season(season: int) -> pd.DataFrame:
    """Download, clean, and return play-by-play data for a single NFL season.

    Args:
        season: Year of the NFL season to scrape

    Returns:
        pbp: A DataFrame containing play-by-play data for the specified season
    """
    with open(os.devnull, "w", encoding="utf-8") as f, redirect_stdout(f):
        pbp = nfl.import_pbp_data([season])

    # Map teams to hrefs
    pbp["home_team"] = pbp["home_team"].replace(TEAM_HREFS)
    pbp["away_team"] = pbp["away_team"].replace(TEAM_HREFS)
    pbp["posteam"] = pbp["posteam"].replace(TEAM_HREFS)
    pbp["defteam"] = pbp["defteam"].replace(TEAM_HREFS)

    # Standardized game_id
    pbp = pbp.drop(columns=["game_id", "old_game_id"], errors="ignore")
    pbp = pbp.assign(
        game_id=pbp["game_date"].str.replace("-", "", regex=False)
        + "_"
        + pbp["away_team"]
        + "_"
        + pbp["home_team"]
    )

    # Move game_id to first column
    cols = ["game_id"] + [col for col in pbp.columns if col != "game_id"]
    pbp = pbp[cols]

    # Return CSV
    return pbp
