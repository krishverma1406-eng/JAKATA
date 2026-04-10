"""Current weather and forecast lookup with profile-based location fallback."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from config.settings import SETTINGS
from core.memory import Memory


TOOL_DEFINITION = {
    "name": "weather_tool",
    "description": "Get current weather and a short forecast using OpenWeatherMap.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["current", "forecast"],
                "description": "Weather action to perform.",
            },
            "location": {
                "type": "string",
                "description": "Optional city or location. If omitted, infer from the user profile.",
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    api_key = SETTINGS.weather_api_key.strip()
    if not api_key:
        return {"ok": False, "error": "OPENWEATHERMAP_API_KEY is not configured."}

    action = str(params.get("action", "")).strip().lower()
    location = str(params.get("location", "")).strip() or _infer_location()
    if not location:
        return {"ok": False, "error": "No location was provided and none could be inferred from the profile."}

    geo = _geocode_location(location, api_key)
    if not geo.get("ok"):
        return geo

    lat = geo["lat"]
    lon = geo["lon"]
    if action == "current":
        return _current_weather(location, lat, lon, api_key)
    if action == "forecast":
        return _forecast(location, lat, lon, api_key)
    return {"ok": False, "error": f"Unsupported action: {action}"}


def _infer_location() -> str:
    memory = Memory()
    hits = memory.recall("current city location where do i live currently", 8)
    for hit in hits:
        candidate = _extract_location_candidate(hit)
        if candidate:
            return candidate
    return ""


def _extract_location_candidate(text: str) -> str:
    cleaned = re.sub(r"[*_`#>\-\[\]]", " ", str(text or "")).strip()
    if not cleaned:
        return ""

    patterns = (
        r"(?i)\blocation\b[^A-Za-z]+([A-Za-z][A-Za-z .'-]*(?:,\s*[A-Za-z][A-Za-z .'-]*){0,2})",
        r"(?i)\b(?:live|lives|living|currently in|based in|study in|studies in)\b[^A-Za-z]+([A-Za-z][A-Za-z .'-]*(?:,\s*[A-Za-z][A-Za-z .'-]*){0,2})",
        r"(?i)\bin ([A-Za-z][A-Za-z .'-]*(?:,\s*[A-Za-z][A-Za-z .'-]*){0,2})",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if not match:
            continue
        candidate = re.sub(r"\s+", " ", match.group(1)).strip(" ,.-")
        candidate = re.sub(r"(?i)^(?:in|at|from)\s+", "", candidate).strip(" ,.-")
        if candidate:
            return candidate
    return ""


def _geocode_location(location: str, api_key: str) -> dict[str, Any]:
    request = Request(
        url=f"https://api.openweathermap.org/geo/1.0/direct?q={quote_plus(location)}&limit=1&appid={api_key}",
        method="GET",
    )
    payload = _load_json(request)
    if not payload["ok"]:
        return payload
    results = payload["data"]
    if not isinstance(results, list) or not results:
        return {"ok": False, "error": f"Location not found: {location}"}
    match = results[0]
    return {
        "ok": True,
        "lat": match.get("lat"),
        "lon": match.get("lon"),
        "resolved_location": ", ".join(
            part for part in [match.get("name"), match.get("state"), match.get("country")] if part
        ),
    }


def _current_weather(location: str, lat: float, lon: float, api_key: str) -> dict[str, Any]:
    units = SETTINGS.weather_default_units
    request = Request(
        url=(
            "https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={api_key}&units={quote_plus(units)}"
        ),
        method="GET",
    )
    payload = _load_json(request)
    if not payload["ok"]:
        return payload
    data = payload["data"]
    weather = (data.get("weather") or [{}])[0]
    main = data.get("main", {})
    wind = data.get("wind", {})
    return {
        "ok": True,
        "location": location,
        "description": weather.get("description", ""),
        "temperature": main.get("temp"),
        "feels_like": main.get("feels_like"),
        "humidity": main.get("humidity"),
        "wind_speed": wind.get("speed"),
        "units": units,
    }


def _forecast(location: str, lat: float, lon: float, api_key: str) -> dict[str, Any]:
    units = SETTINGS.weather_default_units
    request = Request(
        url=(
            "https://api.openweathermap.org/data/2.5/forecast"
            f"?lat={lat}&lon={lon}&appid={api_key}&units={quote_plus(units)}"
        ),
        method="GET",
    )
    payload = _load_json(request)
    if not payload["ok"]:
        return payload
    data = payload["data"]
    forecast_items = []
    for item in (data.get("list") or [])[:10]:
        weather = (item.get("weather") or [{}])[0]
        main = item.get("main", {})
        forecast_items.append(
            {
                "time": item.get("dt_txt"),
                "temperature": main.get("temp"),
                "description": weather.get("description", ""),
            }
        )
    return {"ok": True, "location": location, "units": units, "forecast": forecast_items}


def _load_json(request: Request) -> dict[str, Any]:
    try:
        with urlopen(request, timeout=30) as response:
            return {"ok": True, "data": json.loads(response.read().decode("utf-8"))}
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        return {"ok": False, "error": f"HTTP {exc.code}: {body}"}
    except URLError as exc:
        return {"ok": False, "error": str(exc)}
