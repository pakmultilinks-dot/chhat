"""Google Geocoding API client.

Turns a parsed Pakistani address into lat/lng.
Key from env: GOOGLE_MAPS_API_KEY (same key can enable Solar API).
"""
from __future__ import annotations

import os

import requests

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"


class GeocodeError(Exception):
    pass


def geocode(query: str, api_key: str | None = None) -> dict:
    key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not key:
        raise GeocodeError("GOOGLE_MAPS_API_KEY not set")
    r = requests.get(GEOCODE_URL, params={"address": query, "key": key,
                                          "region": "pk"},
                     timeout=20)
    r.raise_for_status()
    data = r.json()
    status = data.get("status")
    if status != "OK" or not data.get("results"):
        raise GeocodeError(f"geocoding failed: {status} "
                           f"({data.get('error_message', '')})")
    best = data["results"][0]
    loc = best["geometry"]["location"]
    return {
        "lat": loc["lat"],
        "lng": loc["lng"],
        "formatted_address": best["formatted_address"],
        "confidence": best["geometry"].get("location_type", "UNKNOWN"),
        "place_id": best.get("place_id"),
    }
