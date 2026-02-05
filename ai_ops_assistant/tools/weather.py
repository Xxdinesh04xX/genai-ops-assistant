from typing import Any, Dict

import requests

WEATHER_CODE_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    80: "Rain showers",
    81: "Heavy rain showers",
    95: "Thunderstorm",
}


def _geocode_city(city: str) -> Dict[str, Any]:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()
    results = data.get("results", [])
    if not results:
        raise ValueError(f"No geocoding results for city: {city}")
    result = results[0]
    return {
        "name": result.get("name"),
        "country": result.get("country"),
        "latitude": result.get("latitude"),
        "longitude": result.get("longitude"),
    }


def get_current_weather(city: str) -> Dict[str, Any]:
    geo = _geocode_city(city)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": geo["latitude"],
        "longitude": geo["longitude"],
        "current": "temperature_2m,wind_speed_10m,weather_code",
    }
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

    current = data.get("current", {})
    code = current.get("weather_code")
    return {
        "location": f"{geo['name']}, {geo['country']}",
        "temperature_c": current.get("temperature_2m"),
        "wind_kph": current.get("wind_speed_10m"),
        "weather_code": code,
        "weather_summary": WEATHER_CODE_MAP.get(code, "Unknown conditions"),
        "source_url": response.url,
    }
