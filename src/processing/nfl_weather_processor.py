"""Process NFL Weather raw data.

This module provides functions to clean and process raw NFL weather data, including
generating game IDs, encoding weather conditions (windy, rainy, snow, severe), and
handling data.

Usage:
    from src.processing.nfl_weather_processor import process_data

    # Process a raw dataframe
    processed_df = process_data(raw_df)
"""

import os
from contextlib import redirect_stdout

import nfl_data_py as nfl
import pandas as pd

TEAM_HREFS_NDP = {
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

TEAM_HREFS_WEATHER = {
    "buccaneers": "tam",
    "titans": "oti",
    "patriots": "nwe",
    "patiots": "nwe",
    "bears": "chi",
    "redskins": "was",
    "football%20team": "was",
    "washington": "was",
    "jets": "nyj",
    "bills": "buf",
    "jaguars": "jax",
    "texans": "htx",
    "steelers": "pit",
    "chiefs": "kan",
    "cheifs": "kan",
    "eagles": "phi",
    "giants": "nyg",
    "seahawks": "sea",
    "rams": "ram",
    "saints": "nor",
    "lions": "det",
    "browns": "cle",
    "cowboys": "dal",
    "raiders": "rai",
    "bengals": "cin",
    "packers": "gnb",
    "vikings": "min",
    "colts": "clt",
    "panthers": "car",
    "falcons": "atl",
    "chargers": "sdg",
    "broncos": "den",
    "49ers": "sfo",
    "cardinals": "crd",
    "ravens": "rav",
    "dolphins": "mia",
}

WEEK_MAP = {
    "wildcard-weekend": 18,
    "wildcard-weekend-2": 18,
    "wild-card": 18,
    "divisional-playoffs": 19,
    "divisional-playoffs-2": 19,
    "divisional-round": 19,
    "conf-championships": 20,
    "conf-championships-2": 20,
    "%20conf-championships": 20,
    "conference-championships": 20,
    "conference-championship": 20,
    "superbowl": 21,
    "super-bowl": 21,
    "superbowl-2": 21,
}

STADIUM_COORDS = {
    "Bank of America Stadium, Charlotte, NC": (35.225960335894975, -80.8528629623183),
    "FedExField, Landover, MD": (38.9083494371315, -76.86395621793942),
    "TCF Bank Stadium, Minneapolis, MN": (44.97665399299416, -93.2245783905867),
    "Candlestick Park, San Francisco, CA": (37.781023484486404, -122.3921117220476),
    "Rogers Centre, Toronto, ON": (43.641723289659986, -79.38935310231281),
    "Highmark Stadium, Orchard Park, NY": (42.773872702014685, -78.7869079312028),
    "TIAA Bank Field, Jacksonville, FL": (30.324159920543618, -81.63738189139016),
    "FirstEnergy Stadium, Cleveland, OH": (41.506230233710994, -81.69956956196708),
    "Hubert H. Humphrey Metrodome, Minneapolis, MN": (
        44.96318033198384,
        -93.30641375990048,
    ),
    "Wembley Stadium, London, LO": (51.55621144935877, -0.27951041527012793),
    "M&T Bank Stadium, Baltimore, MD": (39.27806589194772, -76.62298661976958),
    "Lumen Field, Seattle, WA": (47.59531817925954, -122.33167159041363),
    "Nissan Stadium, Nashville, TN": (36.16654866074709, -86.77131600537857),
    "GEHA Field at Arrowhead Stadium, Kansas City, MO": (
        39.04906361591322,
        -94.48396664229905,
    ),
    "Lincoln Financial Field, Philadelphia, PA": (
        39.90385511978105,
        -75.16751337047857,
    ),
    "Paycor Stadium, Cincinnati, OH": (39.09559079722041, -84.51606843327228),
    "Ford Field, Detroit, MI": (42.34019669887877, -83.04556008890124),
    "Qualcomm Stadium, San Diego, CA": (32.784814410285904, -117.12353757593246),
    "State Farm Stadium, Glendale, AZ": (33.52778566092994, -112.26255930473224),
    "NRG Stadium, Houston, TX": (29.684815083189005, -95.41071813374656),
    "Caesars Superdome, New Orleans, LA": (29.951246895161553, -90.08122274722723),
    "AT&T Stadium, Arlington, TX": (32.74810485637789, -97.09339504709838),
    "Acrisure Stadium, Pittsburgh, PA": (40.44692806594956, -80.01580321970182),
    "Oakland–Alameda County Coliseum, Oakland, CA": (
        37.751626246990924,
        -122.20092133334762,
    ),
    "Raymond James Stadium, Tampa, FL": (27.976030152863633, -82.50327003196776),
    "MetLife Stadium, Rutherford, NJ": (40.813644411511326, -74.07468230989024),
    "Gillette Stadium, Foxborough, MA": (42.09101422797688, -71.26427130425994),
    "Lambeau Field, Green Bay, WI": (44.50149361417982, -88.06225121945309),
    "Soldier Field, Chicago, IL": (41.862456996504406, -87.61670986194561),
    "Hard Rock Stadium, Miami Gardens, FL": (25.95807519929362, -80.23898639157045),
    "Lucas Oil Stadium, Indianapolis, IN": (39.76030686033878, -86.16382333138297),
    "Empower Field at Mile High, Denver, CO": (39.743900671443996, -105.02018351974282),
    "Edward Jones Dome, St. Louis, MO": (38.632974028953754, -90.18856044864206),
    "Georgia Dome, Atlanta, GA": (33.78160288168175, -84.38731407588378),
    "Cleveland Browns Stadium, Cleveland, OH": (41.50429398781923, -81.68883638751281),
    "Levi's Stadium, Santa Clara, CA": (37.40358067727894, -121.96940959103847),
    "U.S. Bank Stadium, Minneapolis, MN": (44.973828228776284, -93.25749450407922),
    "Los Angeles Memorial Coliseum, Los Angeles, CA": (
        34.01387810904589,
        -118.28786769121578,
    ),
    "Twickenham Stadium, London, LO": (51.456152698045656, -0.34149693880946663),
    "Estadio Azteca, Mexico City, TL": (19.303073321622737, -99.15048478993783),
    "Mercedes-Benz Stadium, Atlanta, GA": (33.75559969728297, -84.40060123355686),
    "ROKiT Field at Dignity Health Sports Park, Carson, CA": (
        33.86474131318403,
        -118.26113207773084,
    ),
    "Tottenham Hotspur Stadium, London, LO": (51.60435263645574, -0.06623109568682027),
    "SoFi Stadium, Inglewood, CA": (33.953556568342414, -118.33906642005462),
    "Allegiant Stadium, Paradise, NV": (36.090992578036705, -115.18333574427751),
    "Allianz Arena, Munich, BAV": (48.21895273775025, 11.624624423121135),
    "Frankfurt Stadium, Frankfurt, HES": (50.06860133096438, 8.645117944680702),
    "Corinthians Arena, Sao Paulo, SAO": (-23.545087159706654, -46.474186289613584),
}


def load_game_id_dataframe(season: int) -> pd.DataFrame:
    """Generate a unique game ID DataFrame for a given season.

    Args:
        season: The NFL season year to load.

    Returns:
        A DataFrame with unique game IDs and corresponding team / week information.
    """
    # Suppress printed output from nfl_data_py
    with open(os.devnull, "w", encoding="utf-8") as f, redirect_stdout(f):
        df = nfl.import_pbp_data([season])
        df["away_team"] = df["away_team"].replace(TEAM_HREFS_NDP)
        df["home_team"] = df["home_team"].replace(TEAM_HREFS_NDP)
        df["game_id"] = (
            df["game_date"].str.replace("-", "", regex=False)
            + "_"
            + df["away_team"]
            + "_"
            + df["home_team"]
        )
        df = (
            df[["game_id", "away_team", "home_team", "season", "week"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        return df


def clean_game_id(game_id: str) -> tuple[int, int, str, str]:
    """Parse a game ID string and extract season, week, away team, and home team.

    Args:
        game_id: Game ID string from NFLWeather.com.

    Returns:
        A tuple of season, week, away_team, and home_team.
    """
    parts = game_id.split("_")

    season = int(parts[0])

    week = parts[1]
    if week in WEEK_MAP and season < 2021:
        week = WEEK_MAP[week]
    elif week in WEEK_MAP and season >= 2021:
        week = WEEK_MAP[week] + 1
    else:
        try:
            week = int(week.split("-")[1])
        except ValueError as exc:
            raise ValueError("Unable to parse NFLWeather.com week: {week}") from exc

    away_team, home_team = parts[2].split("-at-")
    away_team = TEAM_HREFS_WEATHER[away_team]
    home_team = TEAM_HREFS_WEATHER[home_team]

    return season, week, away_team, home_team


def clean_condition(text: str) -> tuple[int, int, int, int]:
    """Convert a weather condition string into a binary tuple representing
    weather categories.

    Categories (binary):
        - windy
        - rainy
        - snowy
        - severe

    Args:
        text: Weather condition string.

    Returns:
        A binary tuple representing (windy, rainy, snowy, severe).
    """
    if pd.isna(text) or text.lower() in ["na", "none"]:
        return (0, 0, 0, 0)

    t = text.lower()
    windy = 0
    rainy = 0
    snowy = 0
    severe = 0

    # Wind-related
    if "wind" in t or "breez" in t:
        windy = 1

    # Rain-related
    if "rain" in t or "drizzle" in t or "showers" in t:
        rainy = 1
    if "thunder" in t:
        rainy = 1
        if windy:
            severe = 1
    # Snow-related
    if "snow" in t or "flurries" in t:
        snowy = 1
    if "snow showers" in t:
        rainy = 0
        snowy = 1
    if "blizzard" in t or "heavy snow" in t or "sleet" in t or "wintry mix" in t:
        snowy = 1
        severe = 1

    # Severe events
    if (windy + rainy + snowy) > 0 and "heavy" in t:
        severe = 1
    if "light" in t:
        severe = 0

    return (windy, rainy, snowy, severe)


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process raw weather data into a processed DataFrame.

    Args:
        df (pd.DataFrame): Raw weather DataFrame.

    Returns:
        A processed DataFrame with columns:
            ["game_id", "weather_windy", "weather_rainy", "weather_snow",
            "weather_severe", "temp", "wind_mph", "precip_prob", "precip_prob_missing"]
    """
    df = df.T.reset_index()
    # Extract game meta-data from NFLWeather.com ID
    df[["season", "week", "away_team", "home_team"]] = (
        df["index"].apply(clean_game_id).apply(pd.Series)
    )
    # Merge with official game ID
    df_game_id = load_game_id_dataframe(df.iloc[0]["season"])
    df_merge = df_game_id.merge(df, how="inner")
    # Map stadium location to latitude and longitude
    df_merge["latitude"] = df_merge["location"].map(
        lambda x: STADIUM_COORDS[x][0] if x in STADIUM_COORDS else None
    )
    df_merge["longitude"] = df_merge["location"].map(
        lambda x: STADIUM_COORDS[x][1] if x in STADIUM_COORDS else None
    )
    missing_coords = df_merge[
        df_merge["latitude"].isna() | df_merge["longitude"].isna()
    ]
    if not missing_coords.empty:
        print(
            "Warning: Some stadium locations are missing coordinates:",
            missing_coords["location"].unique(),
        )
    # Clean condition, temp, wind, precipitation columns
    df_merge[["weather_windy", "weather_rainy", "weather_snow", "weather_severe"]] = (
        df_merge["condition"].apply(clean_condition).apply(pd.Series)
    )
    df_merge["temp"] = df_merge["temp"].str.replace(" °F", "")
    df_merge["temp"] = pd.to_numeric(df_merge["temp"], errors="coerce")
    df_merge["temp"] = df_merge["temp"].fillna(df_merge["temp"].mean())
    df_merge["wind_mph"] = (
        df_merge["wind_mph"].astype(float).fillna(df_merge["wind_mph"].mean())
    )
    df_merge["precip_prob_missing"] = df_merge["precip_prob"].isna().astype(int)
    df_merge["precip_prob"] = (
        df_merge["precip_prob"].astype(float).fillna(df_merge["precip_prob"].mean())
    )
    # Return DataFrame
    df_merge = df_merge[
        [
            "game_id",
            "latitude",
            "longitude",
            "weather_windy",
            "weather_rainy",
            "weather_snow",
            "weather_severe",
            "temp",
            "wind_mph",
            "precip_prob",
            "precip_prob_missing",
        ]
    ]
    return df_merge
