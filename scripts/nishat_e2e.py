"""Nishat Colony end-to-end test: FULL pipeline on 3 address variants.

Flow per address: parse -> geocode query string -> (geocode: needs
GOOGLE_MAPS_API_KEY, skipped honestly if missing) -> demo roof at a
Lahore DEMO pin -> layout optimize -> quote -> Chhat Score.

Run: .venv/bin/python scripts/nishat_e2e.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chhat.geocode.client import GeocodeError, geocode
from chhat.layout.optimizer import optimize
from chhat.parser.predict import AddressParser
from chhat.quote.engine import chhat_score, load_config, quote
from chhat.solar.client import building_insights

ADDRESSES = [
    "House 123, Street 4, Block C, Nishat Colony, Lahore",
    "H#45 St 7 Nishat Colony Lahore",
    "Makan 89, Gali No 2, Nishat Colony, Lahore",
]

# DEMO pin: central Lahore (used only when no geocoding key is set)
DEMO_PIN = (31.5204, 74.3587)


def main() -> None:
    parser = AddressParser()
    config = load_config()
    max_kw = float(config.get("sizing", {}).get("max_system_kw", 10.0))

    for addr in ADDRESSES:
        print("=" * 70)
        print("INPUT :", addr)
        p = parser.parse(addr)
        print("PARSED:", {k: p[k] for k in ("house", "street", "block", "colony", "city")})
        q = parser.query_string(addr)
        print("GEOCODE QUERY:", q)

        try:
            g = geocode(q)
            lat, lng = g["lat"], g["lng"]
            print(f"GEOCODE: {lat:.5f}, {lng:.5f} ({g['confidence']})")
            print("  formatted:", g["formatted_address"])
        except GeocodeError as e:
            lat, lng = DEMO_PIN
            print(f"GEOCODE: skipped ({e}) -> DEMO pin {lat}, {lng}")

        roof = building_insights(lat, lng)
        lay = optimize(roof, max_system_kw=max_kw)
        quo = quote(lay, annual_kwh_per_kw=roof.get("annual_kwh_per_kw"))
        score = chhat_score(roof, lay, quo)

        print(f"ROOF  : {'DEMO' if roof.get('demo') else 'LIVE'} "
              f"{roof['roof_area_m2']} m2, usable {lay['usable_area_m2']} m2")
        print(f"LAYOUT: {lay['panel_count']} panels, {lay['system_kw']} kW")
        print(f"QUOTE : Rs {quo['equipment_cost_pkr']:,.0f} | "
              f"saves Rs {quo['annual_savings_pkr']:,.0f}/yr | "
              f"payback {quo['payback_years']} yrs")
        print(f"SCORE : {score['score']}/100")
    print("=" * 70)
    print("done.")


if __name__ == "__main__":
    main()
