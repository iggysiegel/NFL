"""Data processor: process_data.py.

Process raw NFL data from multiple sources (e.g., Pro Football Reference JSON files,
NFLWeather.com JSON files, nfl_data_py CSV files) and save the processed data in the
processed data directory.

Output:
    - <SEASON>_processed.csv

Usage:
    From the project root, run:
        python -m scripts.process_data --start_season <START> --end_season <END>

Arguments:
    --start_season    First season to scrape (inclusive).
    --end_season      Last season to scrape (inclusive).
"""

import argparse
import os

import pandas as pd

from src.processing.nfl_data_py_processor import process_data as ndp_processor
from src.processing.nfl_weather_processor import \
    process_data as weather_processor
from src.processing.pro_football_reference_processor import \
    process_data as pfr_processor

# Directories
# -----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(SCRIPT_DIR, "../data/raw")
PROCESS_DATA_DIR = os.path.join(SCRIPT_DIR, "../data/processed")

# Configure sources
# -----------------
SOURCES = {
    "nfl_data_py": {
        "filename": "nfl_data_py_{}.csv",
        "processor": ndp_processor,
        "reader": lambda path: pd.read_csv(path, low_memory=False),
    },
    "weather": {
        "filename": "weather_{}.json",
        "processor": weather_processor,
        "reader": pd.read_json,
    },
    "pro_football_reference": {
        "filename": "pro_football_reference_{}.json",
        "processor": pfr_processor,
        "reader": pd.read_json,
    },
}


def main(start_season: int, end_season: int) -> None:
    """Process NFL data across multiple seasons.

    For each season in the given range:
    - Load raw data from each configured source
    - Process the data using the corresponding processor function
    - Merge all available sources on 'game_id'
    - Save the processed, merged DataFrame to the processed directory.

    Args:
        start_season: First season to process
        end_season: Las season to process
    """
    # Ensure processed directory exists
    os.makedirs(PROCESS_DATA_DIR, exist_ok=True)

    # Loop over all seasons
    for season in range(start_season, end_season + 1):
        print(f"Processing season {season}...")
        season_dfs = {}

        # Load and process each data source
        for source_name, source_info in SOURCES.items():
            file_path = os.path.join(
                RAW_DATA_DIR, source_info["filename"].format(season)
            )
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} does not exist, skipping {source_name}")
                continue

            df = source_info["reader"](file_path)
            df = source_info["processor"](df)
            season_dfs[source_name] = df

        if not season_dfs:
            print(f"Warning: No data available for season {season}")
            continue

        # Merge all DataFrames on 'game_id' key
        merged_df = None
        for df in season_dfs.values():
            if merged_df is None:
                merged_df = df
            else:
                merged_df = merged_df.merge(df, on="game_id")

        # Save final processed DataFrame
        output_path = os.path.join(PROCESS_DATA_DIR, f"{season}_processed.csv")
        merged_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process NFL seasons")
    parser.add_argument(
        "--start_season", type=int, required=True, help="First season to process"
    )
    parser.add_argument(
        "--end_season", type=int, required=True, help="Last season to process"
    )
    args = parser.parse_args()

    main(args.start_season, args.end_season)
