from datetime import datetime

import numpy as np


def haversine(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Calculate the Haversine distance between two points on Earth.

    Args:
        lat1: Latitude of the first point in degrees.
        lon1: Longitude of the first point in degrees.
        lat2: Latitude of the second point in degrees.
        lon2: Longitude of the second point in degrees.

    Returns:
        A float of the haversine distance in miles.
    """
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Earth radius in miles
    r = 3958.8
    return c * r


def travel_distance(df):
    df = df.copy()
    temp = {team: [37.77, -89.54] for team in df["home_team"].unique()}
    home_travel_distance = []
    away_travel_distance = []

    for row in df.itertuples():
        home_team = row.home_team
        away_team = row.away_team
        lat = row.latitude
        lon = row.longitude
        home_travel_distance.append(
            haversine(temp[home_team][0], temp[home_team][1], lat, lon)
        )
        away_travel_distance.append(
            haversine(temp[away_team][0], temp[away_team][1], lat, lon)
        )
        temp[home_team] = [lat, lon]
        temp[away_team] = [lat, lon]

    df["home_travel_distance"] = home_travel_distance
    df["away_travel_distance"] = away_travel_distance
    return df


def rest_days(df):
    df = df.copy()
    temp = {team: [None, None] for team in df["home_team"].unique()}
    home_rest_days = []
    away_rest_days = []

    for row in df.itertuples():
        game_date = datetime.strptime(row.game_id[:8], "%Y%m%d")

        if temp[row.home_team][0] != row.season:
            temp[row.home_team][0] = row.season
            home_rest_days.append(7)
            temp[row.home_team][1] = game_date
        else:
            time_delta = (game_date - temp[row.home_team][1]).days
            home_rest_days.append(time_delta)
            temp[row.home_team][1] = game_date

        if temp[row.away_team][0] != row.season:
            temp[row.away_team][0] = row.season
            away_rest_days.append(7)
            temp[row.away_team][1] = game_date
        else:
            time_delta = (game_date - temp[row.away_team][1]).days
            away_rest_days.append(time_delta)
            temp[row.away_team][1] = game_date

    df["home_rest_days"] = home_rest_days
    df["away_rest_days"] = away_rest_days
    return df


def is_indoors(df):
    df = df.copy()
    df["is_indoors"] = df["roof"].map({"outdoors": 0, "indoors": 1})
    df["is_indoors"] = df["is_indoors"].fillna(0)
    return df


def surface(df):
    df = df.copy()
    df["surface"] = df["surface"].map({"grass": 0, "turf": 1})
    df["surface"] = df["surface"].fillna(0)
    return df


def attendance(df):
    df = df.copy()
    df["attendance"] = df["attendance"].fillna(df["attendance"].mean())
    return df


def implied_score(df):
    df = df.copy()
    df["home_implied_score"] = (df["over_under"] - df["home_line"]) / 2
    df["away_implied_score"] = (df["over_under"] + df["home_line"]) / 2
    return df


def ats_pctg(df, n):
    df = df.copy()
    temp = {team: [0] * n for team in df["home_team"].unique()}
    home_ats_pctg = []
    away_ats_pctg = []
    for row in df.itertuples():
        home_ats_pctg.append(sum(temp[row.home_team]) / n)
        away_ats_pctg.append(sum(temp[row.away_team]) / n)
        temp[row.home_team].pop()
        temp[row.away_team].pop()
        if row.result > -row.home_line:
            temp[row.home_team].insert(0, 1)
            temp[row.away_team].insert(0, 0)
        elif row.result < row.home_line:
            temp[row.home_team].insert(0, 0)
            temp[row.away_team].insert(0, 1)
        else:
            temp[row.home_team].insert(0, 0)
            temp[row.away_team].insert(0, 0)
    df[f"home_ats_pctg_{n}"] = home_ats_pctg
    df[f"away_ats_pctg_{n}"] = away_ats_pctg
    return df


def ats_pctg_favorite(df, n):
    df = df.copy()
    temp = {team: [0] * n for team in df["home_team"].unique()}
    home_ats_pctg_fav = []
    away_ats_pctg_fav = []
    for row in df.itertuples():
        home_ats_pctg_fav.append(sum(temp[row.home_team]) / n)
        away_ats_pctg_fav.append(sum(temp[row.away_team]) / n)
        home_fav = row.home_line < 0
        away_fav = row.home_line > 0
        if home_fav:
            if row.result > -row.home_line:
                temp[row.home_team].pop()
                temp[row.home_team].insert(0, 1)
            else:
                temp[row.home_team].pop()
                temp[row.home_team].insert(0, 0)
        if away_fav:
            if row.result < -row.home_line:
                temp[row.away_team].pop()
                temp[row.away_team].insert(0, 1)
            else:
                temp[row.away_team].pop()
                temp[row.away_team].insert(0, 0)
    df[f"home_ats_pctg_fav_{n}"] = home_ats_pctg_fav
    df[f"away_ats_pctg_fav_{n}"] = away_ats_pctg_fav
    return df


def ats_pctg_underdog(df, n):
    df = df.copy()
    temp = {team: [0] * n for team in df["home_team"].unique()}
    home_ats_pctg_und = []
    away_ats_pctg_und = []
    for row in df.itertuples():
        home_ats_pctg_und.append(sum(temp[row.home_team]) / n)
        away_ats_pctg_und.append(sum(temp[row.away_team]) / n)
        home_und = row.home_line > 0
        away_und = row.home_line < 0
        if home_und:
            if row.result > -row.home_line:
                temp[row.home_team].pop()
                temp[row.home_team].insert(0, 1)
            else:
                temp[row.home_team].pop()
                temp[row.home_team].insert(0, 0)
        if away_und:
            if row.result < -row.home_line:
                temp[row.away_team].pop()
                temp[row.away_team].insert(0, 1)
            else:
                temp[row.away_team].pop()
                temp[row.away_team].insert(0, 0)
    df[f"home_ats_pctg_und_{n}"] = home_ats_pctg_und
    df[f"away_ats_pctg_und_{n}"] = away_ats_pctg_und
    return df


def win_pctg(df, n):
    df = df.copy()
    temp = {team: [0] * n for team in df["home_team"].unique()}
    home_win_pctg = []
    away_win_pctg = []
    for row in df.itertuples():
        home_win_pctg.append(sum(temp[row.home_team]) / n)
        away_win_pctg.append(sum(temp[row.away_team]) / n)
        temp[row.home_team].pop()
        temp[row.away_team].pop()
        if row.result > 0:
            temp[row.home_team].insert(0, 1)
            temp[row.away_team].insert(0, 0)
        elif row.result < 0:
            temp[row.home_team].insert(0, 0)
            temp[row.away_team].insert(0, 1)
        else:
            temp[row.home_team].insert(0, 0)
            temp[row.away_team].insert(0, 0)
    df[f"home_win_pctg_{n}"] = home_win_pctg
    df[f"away_win_pctg_{n}"] = away_win_pctg
    return df


def spread_differential(df, n):
    df = df.copy()
    temp = {team: [0] * n for team in df["home_team"].unique()}
    home_spread_diff = []
    away_spread_diff = []
    for row in df.itertuples():
        home_spread_diff.append(sum(temp[row.home_team]) / n)
        away_spread_diff.append(sum(temp[row.away_team]) / n)
        temp[row.home_team].pop()
        temp[row.home_team].insert(0, row.result + row.home_line)
        temp[row.away_team].pop()
        temp[row.away_team].insert(0, row.result - row.home_line)
    df[f"home_spread_diff_{n}"] = home_spread_diff
    df[f"away_spread_diff_{n}"] = away_spread_diff
    return df


def add_contextual_features(df):
    df = df.copy()
    df = travel_distance(df)
    df = rest_days(df)
    df = is_indoors(df)
    df = surface(df)
    df = attendance(df)
    df = implied_score(df)
    df = ats_pctg(df, 5)
    df = ats_pctg(df, 10)
    df = ats_pctg_favorite(df, 5)
    df = ats_pctg_favorite(df, 10)
    df = ats_pctg_underdog(df, 5)
    df = ats_pctg_underdog(df, 10)
    df = win_pctg(df, 5)
    df = win_pctg(df, 10)
    df = spread_differential(df, 5)
    df = spread_differential(df, 10)
    df = df[
        [
            "game_id",
            "home_travel_distance",
            "away_travel_distance",
            "home_rest_days",
            "away_rest_days",
            "is_indoors",
            "surface",
            "attendance",
            "home_implied_score",
            "away_implied_score",
            "home_ats_pctg_5",
            "away_ats_pctg_5",
            "home_ats_pctg_10",
            "away_ats_pctg_10",
            "home_ats_pctg_fav_5",
            "away_ats_pctg_fav_5",
            "home_ats_pctg_fav_10",
            "away_ats_pctg_fav_10",
            "home_ats_pctg_und_5",
            "away_ats_pctg_und_5",
            "home_ats_pctg_und_10",
            "away_ats_pctg_und_10",
            "home_win_pctg_5",
            "away_win_pctg_5",
            "home_win_pctg_10",
            "away_win_pctg_10",
            "home_spread_diff_5",
            "away_spread_diff_5",
            "home_spread_diff_10",
            "away_spread_diff_10",
        ]
    ]
    return df
