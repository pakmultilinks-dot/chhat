"""Roof imagery adapter — Google Solar API client with clearly-labeled DEMO mode.

DESIGN NOTE: this module is an *imagery source*, not intelligence. The brain
is our own rooftop segmentation model (wave 4, trained on open aerial data),
which will read roofs from any imagery source plugged in here. Google is a
swappable tap, never the product.

Real endpoints (need GOOGLE_SOLAR_API_KEY):
  - buildingInsights: roof segments, area, shading, solar potential
  - dataLayers:   aerial imagery / DSM / annual flux

DEMO mode (no key): returns a realistic mock Lahore rooftop so the whole
pipeline and UI can be exercised end-to-end. Every demo payload carries
"demo": True and the UI must show a DEMO badge whenever it is used.
"""
from __future__ import annotations

import math
import os
import random

import requests

BUILDING_INSIGHTS_URL = "https://solar.googleapis.com/v1/buildingInsights:findClosest"
DATA_LAYERS_URL = "https://solar.googleapis.com/v1/dataLayers:get"


class SolarError(Exception):
    pass


def _demo_roof(lat: float, lng: float, seed: int = 7) -> dict:
    """Realistic mock of a Lahore rooftop: 2-3 segments, parapet, one tank."""
    rng = random.Random(seed + int(abs(lat * 1000)) + int(abs(lng * 1000)))
    # typical Lahore house: 10 marla ~ 253 sqm plot, roof ~120-180 sqm
    roof_area = rng.uniform(120, 180)
    n_seg = rng.choice([2, 3])
    segments = []
    remaining = roof_area
    for i in range(n_seg):
        area = remaining / (n_seg - i) * rng.uniform(0.8, 1.2)
        remaining -= area
        segments.append({
            "pitch_degrees": rng.uniform(0, 8),
            "azimuth_degrees": rng.choice([90, 180, 270]),
            "area_m2": round(area, 1),
            "shade_fraction": rng.uniform(0.05, 0.25),
        })
    # obstructions typical for Pakistani rooftops
    obstructions = [{"type": "water_tank", "area_m2": round(rng.uniform(2, 4), 1)},
                    {"type": "mumty", "area_m2": round(rng.uniform(4, 8), 1)}]
    if rng.random() < 0.4:
        obstructions.append({"type": "parapet", "area_m2": round(rng.uniform(3, 6), 1)})
    usable = roof_area - sum(o["area_m2"] for o in obstructions)
    annual_kwh_per_kw = rng.uniform(1450, 1650)  # Lahore irradiance reality
    return {
        "demo": True,
        "lat": lat, "lng": lng,
        "roof_area_m2": round(roof_area, 1),
        "usable_area_m2": round(max(usable, 0), 1),
        "segments": segments,
        "obstructions": obstructions,
        "annual_kwh_per_kw": round(annual_kwh_per_kw, 0),
        "note": "DEMO roof: synthetic but realistic Lahore rooftop. "
                "Set GOOGLE_SOLAR_API_KEY for live Google Solar data.",
    }


def building_insights(lat: float, lng: float,
                      api_key: str | None = None) -> dict:
    key = api_key or os.environ.get("GOOGLE_SOLAR_API_KEY")
    if not key:
        return _demo_roof(lat, lng)
    r = requests.get(BUILDING_INSIGHTS_URL,
                     params={"location.latitude": lat, "location.longitude": lng,
                             "key": key},
                     timeout=30)
    if r.status_code == 404:
        raise SolarError("no solar data for this location (outside coverage)")
    r.raise_for_status()
    data = r.json()
    solar = data.get("solarPotential", {})
    roof_segments = []
    for seg in solar.get("roofSegmentStats", []):
        roof_segments.append({
            "pitch_degrees": seg.get("pitchDegrees"),
            "azimuth_degrees": seg.get("azimuthDegrees"),
            "area_m2": seg.get("stats", {}).get("areaMeters2"),
            "shade_fraction": None,
        })
    return {
        "demo": False,
        "lat": lat, "lng": lng,
        "roof_area_m2": solar.get("wholeRoofStats", {}).get("areaMeters2"),
        "usable_area_m2": None,  # computed by layout optimizer
        "segments": roof_segments,
        "obstructions": [],  # detected by layout optimizer from imagery
        "annual_kwh_per_kw": None,
        "raw": data,
    }


def roof_imagery_url(lat: float, lng: float, api_key: str | None = None,
                     zoom: int = 20, size: int = 512) -> str | None:
    """Aerial imagery tile URL for the roof visualization.

    Returns a Google dataLayers render URL when a key exists, else None
    (the web UI then draws a schematic roof instead).
    """
    key = api_key or os.environ.get("GOOGLE_SOLAR_API_KEY")
    if not key:
        return None
    r = requests.get(DATA_LAYERS_URL,
                     params={"location.latitude": lat, "location.longitude": lng,
                             "radiusMeters": 60, "view": "IMAGERY",
                             "requiredQuality": "HIGH", "key": key},
                     timeout=30)
    r.raise_for_status()
    return r.json().get("imageryRenderedUrl")
