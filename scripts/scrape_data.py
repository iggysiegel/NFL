"""Raw Data Scraper: scrape_data.py.

Scrape NFL data from multiple sources (e.g., NFLWeather.com, nfl_data_py,
Pro Football Reference) and save raw data in the raw data directory.

Output:
    - NFLWeather.com: JSON files
    - nfl_data_py: CSV files
    - Pro Football Reference: JSON files

Usage:
    From the project root, run:
        python -m scripts.scrape_data --start_season <START> --end_season <END>

Arguments:
    --start_season    First season to scrape (inclusive).
    --end_season      Last season to scrape (inclusive).
"""

import argparse
import json
import os

import pandas as pd

from src.data.nfl_data_py_scraper import scrape_season as ndp_scrape
from src.data.nfl_weather_scraper import NflWeatherScraper
from src.data.pro_football_reference_scraper import scrape_season as pfr_scrape

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(SCRIPT_DIR, "../data/raw")


def save_data(
    data: pd.DataFrame | dict, source: str, year: int, raw_data_dir: str
) -> None:
    """Save scraped NFL data to the raw data directory.

    Args:
        data: Data to save
        source: Data source name, used in filename
        year: Season year, used in filename
        raw_data_dir: Directory where file will be saved
    """
    os.makedirs(raw_data_dir, exist_ok=True)

    if isinstance(data, pd.DataFrame):
        filename = os.path.join(raw_data_dir, f"{source}_{year}.csv")
        data.to_csv(filename, index=False)
    elif isinstance(data, (dict, list)):
        filename = os.path.join(raw_data_dir, f"{source}_{year}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    else:
        raise TypeError(f"Cannot save data of type {type(data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape NFL seasons")
    parser.add_argument(
        "--start_season", type=int, required=True, help="First season to scrape"
    )
    parser.add_argument(
        "--end_season", type=int, required=True, help="Last season to scrape"
    )
    args = parser.parse_args()

    for season in range(args.start_season, args.end_season + 1):
        # NFLWeather.com
        weather_data = NflWeatherScraper().parse_season(season)
        save_data(weather_data, "weather", season, RAW_DATA_DIR)
        print(f"{season} Weather done.", flush=True)

        # nfl_data_py
        ndp_season = ndp_scrape(season)
        save_data(ndp_season, "nfl_data_py", season, RAW_DATA_DIR)

        # Pro Football Reference
        pfr_season = pfr_scrape(season)
        save_data(pfr_season, "pro_football_reference", season, RAW_DATA_DIR)
        print(f"{season} Pro Football Reference done.", flush=True)
