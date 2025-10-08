"""A NFL Weather Scraper.

This module provides reusable functions to scrape NFL game-level weather
data from NFLWeather.com. It collects kickoff weather conditions,
temperature, wind, precipitation probability, and stadium location,
along with unique game_id keys.

Usage:
    from src.data.nfl_weather_scraper import NflWeatherScraper

    # Initialize scraper
    scraper = NflWeatherScraper()

    # Parse a single season
    season_data = scraper.parse_season(2024)

Notes:
- Future improvements:
  * Implement a script to automatically scrape multiple seasons.
  * Add ability to scrape current season for real-time model implementation.
"""

import random
import re
import time

import cloudscraper
from bs4 import BeautifulSoup


class NflWeatherScraper:
    """Scraper for NFL game weather data from NFLWeather.com."""

    def __init__(self):
        """Initialize the scraper with the base URL and a cloudscraper instance."""
        self.base_url = "https://www.nflweather.com/"
        self.scraper = cloudscraper.create_scraper()

    def get_soup(self, url: str) -> BeautifulSoup:
        """Return BeautifulSoup object for a given game URL."""
        response = self.scraper.get(url).text
        soup = BeautifulSoup(response, "lxml")
        return soup

    def parse_game(self, url: str) -> dict:
        """Parse weather data and stadium location for a single NFL game.

        Args:
            url: The URL of the game's page.

        Returns:
            A dictionary containing weather metrics and stadium location.
        """
        soup = self.get_soup(url)
        data = {
            "condition": None,
            "temp": None,
            "feels_like": None,
            "wind_mph": None,
            "precip_prob": None,
            "location": None,
        }

        # Weather report
        report = soup.find("div", class_="weather-report")
        if report:
            condition = report.find("p", class_="fw-bold")
            if condition:
                data["condition"] = condition.get_text(strip=True)
            temp = report.find("p", class_="weather-temperature")
            if temp:
                data["temp"] = temp.get_text(strip=True)
            data_elements = report.find_all("p", class_="weather-data")
            for p in data_elements:
                text = p.get_text(" ", strip=True)
                if "Feels Like" in text:
                    match = re.search(r"([\d.]+)\s*°F", text)
                    if match:
                        data["feels_like"] = float(match.group(1))
                elif "mph" in text and "Gusts" not in text:
                    match = re.search(r"([\d.]+)\s*mph", text)
                    if match:
                        data["wind_mph"] = float(match.group(1))
                elif "Prec." in text or "Precip" in text:
                    match = re.search(r"([\d.]+)\s*%", text)
                    if match:
                        data["precip_prob"] = float(match.group(1))

        # Location information
        stadium_container = soup.find("a", class_="stadium-container")
        if stadium_container:
            title_div = stadium_container.find("div", class_="title")
            if title_div:
                data["location"] = title_div.get_text(strip=True)

        return data

    def get_season_urls(self, season: int) -> list[str]:
        """Retrieve all game URLs for a given NFL season.

        Args:
            season: Year of the NFL season.

        Returns:
            A list of full URLs to all games in the season.
        """
        # List of NFLWeather.com week names in the season
        weeks_list = []
        if season < 2009:
            raise ValueError(
                "Warning: Data before 2009 may be incomplete or inaccurate."
            )
        elif season == 2010:
            for i in range(1, 18):
                weeks_list.append(f"week-{i}-2")
            weeks_list.extend(
                [
                    "wildcard-weekend-2",
                    "divisional-playoffs-2",
                    "conf-championships-2",
                    "superbowl-2",
                ]
            )
        elif season < 2019:
            for i in range(1, 18):
                weeks_list.append(f"week-{i}")
            weeks_list.extend(
                [
                    "wildcard-weekend",
                    "divisional-playoffs",
                    "conf-championships",
                    "superbowl",
                ]
            )
        elif season < 2021:
            for i in range(1, 18):
                weeks_list.append(f"week-{i}")
            weeks_list.extend(
                [
                    "wildcard-weekend",
                    "divisional-playoffs",
                    "%20conf-championships",
                    "superbowl",
                ]
            )
        elif season < 2023:
            for i in range(1, 19):
                weeks_list.append(f"week-{i}")
            weeks_list.extend(
                [
                    "wildcard-weekend",
                    "divisional-playoffs",
                    "%20conf-championships",
                    "superbowl",
                ]
            )
        else:
            for i in range(1, 19):
                weeks_list.append(f"week-{i}")
            weeks_list.extend(
                [
                    "wild-card",
                    "divisional-round",
                    "conference-championship",
                    "super-bowl",
                ]
            )

        # List of all game URLs in the season
        season_urls = []
        for week in weeks_list:
            week_url = self.base_url + f"week/{season}/{week}/"
            soup = self.get_soup(week_url)
            time.sleep(random.uniform(1, 2))
            week_urls = []
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                if href.startswith("/games/"):
                    week_urls.append(
                        self.base_url + f"games/{season}/{week}/" + href.split("/")[-1]
                    )
            week_urls = list(set(week_urls))
            season_urls.extend(week_urls)

        return season_urls

    def parse_season(self, season: int) -> dict[str, dict]:
        """Parse weather data for all games in a given NFL season.

        Args:
            season: Year of the NFL season.

        Returns:
            A dictionary mapping each game ID to its weather data.
        """
        data = {}
        season_urls = self.get_season_urls(season)
        for url in season_urls:
            game_data = self.parse_game(url)
            time.sleep(random.uniform(1, 2))
            game_id = ("_").join(url.split("/")[4:])
            data[game_id] = game_data
        return data
