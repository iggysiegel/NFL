"""Pro Football Reference NFL Scraper.

Scrapes NFL seasons from Pro Football Reference and saves each season's
data as a JSON file. Each JSON file contains a list of dictionaries,
where each dictionary represents a single game and its stats.

Usage:
    python pro_football_reference.py --start_season 2010 --end_season 2015

Arguments:
    --start_season    First season to scrape (inclusive).
    --end_season      Last season to scrape (inclusive).
"""

# Imports, constants
# ------------------
import argparse
import datetime
import json
import random
import re
import time

import cloudscraper
from bs4 import BeautifulSoup


scraper = cloudscraper.create_scraper()
team_hrefs = {
    "Phoenix Cardinals": "crd",
    "Arizona Cardinals": "crd",
    "Atlanta Falcons": "atl",
    "Baltimore Ravens": "rav",
    "Buffalo Bills": "buf",
    "Carolina Panthers": "car",
    "Chicago Bears": "chi",
    "Cincinnati Bengals": "cin",
    "Cleveland Browns": "cle",
    "Dallas Cowboys": "dal",
    "Denver Broncos": "den",
    "Detroit Lions": "det",
    "Green Bay Packers": "gnb",
    "Houston Texans": "htx",
    "Indianapolis Colts": "clt",
    "Jacksonville Jaguars": "jax",
    "Kansas City Chiefs": "kan",
    "Los Angeles Raiders": "rai",
    "Oakland Raiders": "rai",
    "Las Vegas Raiders": "rai",
    "San Diego Chargers": "sdg",
    "Los Angeles Chargers": "sdg",
    "St. Louis Rams": "ram",
    "Los Angeles Rams": "ram",
    "Miami Dolphins": "mia",
    "Minnesota Vikings": "min",
    "New England Patriots": "nwe",
    "New Orleans Saints": "nor",
    "New York Giants": "nyg",
    "New York Jets": "nyj",
    "Philadelphia Eagles": "phi",
    "Pittsburgh Steelers": "pit",
    "San Francisco 49ers": "sfo",
    "Seattle Seahawks": "sea",
    "Tampa Bay Buccaneers": "tam",
    "Houston Oilers": "oti",
    "Tennessee Oilers": "oti",
    "Tennessee Titans": "oti",
    "Washington Redskins": "was",
    "Washington Football Team": "was",
    "Washington Commanders": "was",
}


# Low-level helpers
# -----------------
def get_soup(game_url: str) -> BeautifulSoup:
    """Return BeautifulSoup object for a given game URL."""
    response = scraper.get(game_url).text
    clean_response = re.sub(r"<!--|-->", "", response)
    soup = BeautifulSoup(clean_response, "lxml")
    return soup


def flatten_dict(d: dict, home_team: str, away_team: str) -> dict:
    """
    Helper function to flatten a nested dictionary so that keys are prefixed
    with 'home_' or 'away_'.

    Args:
        d: Nested dictionary
        home_team: Key in d corresponding to the home team
        away_team: Key in d corresponding to the away team

    Returns:
        flat: Flattened dictionary with prefixed keys
    """
    flat = {}

    for key, value in d[home_team].items():
        flat[f"home_{key}"] = value

    for key, value in d[away_team].items():
        flat[f"away_{key}"] = value

    return flat


# Single-game scraping helpers
# ----------------------------
def scrape_team_names(soup: BeautifulSoup) -> list[str]:
    """Scrape the home and away team names for a single game.

    Args:
        soup: BeautifulSoup object of the game page.

    Returns:
        A list containing the home and away team names.
    """
    team_names = soup.select('#inner_nav a[href^="/teams/"]')
    home_team = team_names[0].text.split("Schedule")[0].strip()
    away_team = team_names[1].text.split("Schedule")[0].strip()
    return [home_team, away_team]


def scrape_game_info(soup: BeautifulSoup) -> dict:
    """Scrape key game information for a single game.

    Args:
        soup: BeautifulSoup object of the game page.

    Returns:
        game_info: Dictionary containing home and away teams, game metadata,
        (roof, surface, weather, attendance), home line, and over/under.
    """
    # Initialize dictionary with all expected keys as None
    game_info = {
        "home_team": None,
        "away_team": None,
        "roof": None,
        "surface": None,
        "weather": None,
        "attendance": None,
        "home_line": None,
        "over_under": None,
    }

    # Scrape home and away teams
    home_team, away_team = scrape_team_names(soup)
    game_info["home_team"] = team_hrefs[home_team]
    game_info["away_team"] = team_hrefs[away_team]

    # Scrape game metadata and betting lines
    table = soup.find("table", id="game_info")
    for row in table.find_all("tr"):
        th = row.find("th")
        td = row.find("td")
        if not th or not td:
            continue
        field_name = th.text.strip().lower()

        if field_name == "roof":
            game_info["roof"] = td.text
        elif field_name == "surface":
            game_info["surface"] = td.text
        elif field_name == "weather":
            game_info["weather"] = td.text
        elif field_name == "attendance":
            game_info["attendance"] = int(td.text.replace(",", ""))
        elif field_name == "vegas line":
            vegas_favorite = " ".join(td.text.split()[0:-1])
            vegas_line = float(td.text.split()[-1])
            if vegas_favorite == home_team:
                game_info["home_line"] = vegas_line
            elif vegas_favorite == away_team:
                game_info["home_line"] = -vegas_line
            else:
                raise ValueError("Vegas favorite does not match either team.")
        elif field_name == "over/under":
            game_info["over_under"] = float(td.text.split()[0])

    return game_info


def scrape_expected_points_summary(soup: BeautifulSoup) -> dict:
    """Scrape expected points summary for a single game.

    Args:
        soup: BeautifulSoup object of the game page.

    Returns:
        expected_points: Dictionary containing expected points data for each
        team, keyed by team abbreviation.
    """
    table = soup.find("table", id="expected_points")
    expected_points = {}

    # Iterate over each row in the table
    for row in table.find_all("tr"):
        th = row.find("th", {"data-stat": "team_name"})
        if not th or th.get("scope") == "col":
            continue  # Skip headers

        # Map team name to abbreviation
        team_name = th.text.strip()
        for key, value in team_hrefs.items():
            if team_name in key:
                team_abbr = value
                break

        # Build the dictionary for this team
        expected_points[team_abbr] = {}
        for td in row.find_all("td"):
            expected_points[team_abbr][td["data-stat"]] = float(td.text)

    return expected_points


def scrape_team_stats(soup: BeautifulSoup) -> dict:
    """Scrape team stats for a single game.

    Args:
        soup: BeautifulSoup object of the game page.

    Returns:
        team_stats: Dictionary containing team stats for each team,
        keyed by team abbreviation.
    """
    table = soup.find("table", id="team_stats")
    team_stats = {}

    # Scrape team names
    home_team, away_team = scrape_team_names(soup)
    team_stats[team_hrefs[home_team]] = {}
    team_stats[team_hrefs[away_team]] = {}

    # Scrape the stat name and home / away values
    for row in table.find("tbody").find_all("tr"):
        stat_name = row.find("th").text.strip().lower()
        home_value = row.find("td", {"data-stat": "home_stat"}).text
        away_value = row.find("td", {"data-stat": "vis_stat"}).text
        team_stats[team_hrefs[home_team]][stat_name] = home_value
        team_stats[team_hrefs[away_team]][stat_name] = away_value

    return team_stats


# Multi-game / multi-week helpers
# -------------------------------
def get_number_of_weeks(season: int) -> int:
    """Return the number of completed weeks in a season."""
    # Expected weeks
    if season >= 2021:
        expected_weeks = 22
    elif season >= 1990:
        expected_weeks = 21
    else:
        raise ValueError("Warning: Data before 1990 may be incomplete or inaccurate.")

    # Return expected weeks if season is fully complete
    if datetime.datetime.now() >= datetime.datetime(season + 1, 3, 1):
        return expected_weeks

    # Otherwise, scrape in-progress season
    weeks = 0
    while True:
        url = (
            f"https://www.pro-football-reference.com/years/{season}/"
            f"week_{weeks + 1}.htm"
        )
        response = scraper.get(url)
        title = response.text.split("<title>")[1].split("</title>")[0]
        if "Preview" in title:
            break
        weeks += 1
        time.sleep(random.uniform(1, 3))

    return weeks


def get_week_links(season: int, week: int) -> list[str]:
    """Get the links to all games in a single week."""
    url = f"https://www.pro-football-reference.com/years/{season}/" f"week_{week}.htm"
    response = scraper.get(url)

    if response.status_code != 200:
        print(
            f"Failed to scrape week {week} of {season}. "
            f"Status code: {response.status_code}",
            flush=True,
        )
        return []

    soup = BeautifulSoup(response.text, "lxml")
    return [
        f"https://www.pro-football-reference.com{cell.a['href']}"
        for cell in soup.find_all("td", class_="gamelink")
    ]


def get_season_links(season: int) -> list[str]:
    """Get the links to all games in a season."""
    season_links = []

    number_of_weeks = get_number_of_weeks(season)

    for week in range(1, number_of_weeks + 1):
        week_links = get_week_links(season, week)
        season_links.extend(week_links)
        time.sleep(random.uniform(1, 3))

    return season_links


# High-level function
# -------------------
def scrape_season(season: int) -> list[dict]:
    """Scrape all games in a single season.

    Args:
        season: Year of the NFL season to scrape.

    Returns:
        season_data: List of dictionaries, each containing a single game.
            Each dictionary contains the keys:
                - 'game_id'
                - 'game_info'
                - 'expected_points'
                - 'team_stats'
    """
    season_data = []

    season_links = get_season_links(season)

    for game_url in season_links:
        try:
            soup = get_soup(game_url)
            game_info = scrape_game_info(soup)
            expected_points = scrape_expected_points_summary(soup)
            team_stats = scrape_team_stats(soup)

            home_team = game_info["home_team"]
            away_team = game_info["away_team"]
            game_date = game_url.split("/boxscores/")[1][:8]
            game_id = f"{game_date}_{away_team}_{home_team}"

            game_data = {}
            game_data["game_id"] = game_id
            game_data.update(game_info)
            game_data.update(flatten_dict(expected_points, home_team, away_team))
            game_data.update(flatten_dict(team_stats, home_team, away_team))

            season_data.append(game_data)

            time.sleep(random.uniform(1, 3))

        except Exception as e:  # Too general exception
            print(f"Failed to scrape {game_url}: {e}", flush=True)
            continue

    return season_data


# Run the scraper from the command line
# -------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape NFL seasons from Pro Football Reference"
    )
    parser.add_argument(
        "--start_season", type=int, required=True, help="First season to scrape"
    )
    parser.add_argument(
        "--end_season", type=int, required=True, help="Last season to scrape"
    )

    args = parser.parse_args()

    for season_to_scrape in range(args.start_year, args.end_year + 1):
        print(f"Scraping season {season_to_scrape}...", flush=True)
        season_to_scrape_data = scrape_season(season_to_scrape)

        with open(
            f"raw/pro_football_reference_{season_to_scrape}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(season_to_scrape_data, f, indent=2)
