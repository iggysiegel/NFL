"""nfl_data_py Scraper.

Scrapes NFL seasons from nfl_data_py and saves each season's
play-by-play data as a CSV file. Each CSV contains all plays for a season,
with normalized game_id keys.

Usage:
    python nfl_data_py_scraper.py --start_season 2010 --end_season 2015

Arguments:
    --start_season    First season to scrape (inclusive).
    --end_season      Last season to scrape (inclusive).
"""

import argparse

import nfl_data_py as nfl


team_hrefs = {
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


def process_season(season: int) -> None:
    """Download, clean, and save play-by-play data for a single NFL season."""
    pbp = nfl.import_pbp_data([season])

    # Map teams to hrefs
    pbp["home_team"] = pbp["home_team"].replace(team_hrefs)
    pbp["away_team"] = pbp["away_team"].replace(team_hrefs)
    pbp["posteam"] = pbp["posteam"].replace(team_hrefs)
    pbp["defteam"] = pbp["defteam"].replace(team_hrefs)

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

    # Save CSV
    pbp.to_csv(f"raw/nfl_data_py_{season}.csv", index=False)


# Run the scraper from the command line
# -------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape NFL seasons from nfl_data_py")
    parser.add_argument(
        "--start_season", type=int, required=True, help="First season to scrape"
    )
    parser.add_argument(
        "--end_season", type=int, required=True, help="Last season to scrape"
    )

    args = parser.parse_args()

    for season_to_scrape in range(args.start_season, args.end_season + 1):
        process_season(season_to_scrape)
