"""Generate docs/pitch/golden-demos.json — 10 precomputed demo runs.

Each entry runs the REAL pipeline (parse -> demo pin -> demo roof ->
layout optimize -> quote -> Chhat Score). Nothing is invented: the roof
figures come from chhat.solar.client._demo_roof, seeded per pin.

Pins are APPROXIMATE landmark locations (demo pins, not surveyed).
All addresses are famous public/commercial landmarks — never a real home.

Run: .venv/bin/python scripts/golden_demos.py
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chhat.layout.optimizer import optimize
from chhat.parser.predict import AddressParser
from chhat.quote.engine import chhat_score, load_config, quote
from chhat.solar.client import building_insights

# (label, address, approx lat, approx lng) — famous public landmarks only.
GOLDEN = [
    ("Packages Mall — Lahore",
     "Packages Mall, Shahrah-e-Roomi, DHA Phase 5, Lahore", 31.4697, 74.3941),
    ("Emporium Mall — Lahore",
     "Emporium Mall, Abdul Haque Road, Johar Town, Lahore", 31.4695, 74.2728),
    ("Lucky One Mall — Karachi",
     "Lucky One Mall, Rashid Minhas Road, Karachi", 24.8876, 67.0894),
    ("Pearl Continental Hotel — Karachi",
     "Pearl Continental Hotel, Club Road, Karachi", 24.8516, 67.0283),
    ("Dolmen Mall Clifton — Karachi",
     "Dolmen Mall Clifton, Sea View Road, Karachi", 24.8025, 67.0297),
    ("Centaurus Mall — Islamabad",
     "Centaurus Mall, Jinnah Avenue, Islamabad", 33.7077, 73.0498),
    ("Safa Gold Mall — Islamabad",
     "Safa Gold Mall, F-7 Markaz, Islamabad", 33.6995, 73.0363),
    ("Serena Hotel — Islamabad",
     "Serena Hotel, Khayaban-e-Suharwardy, Islamabad", 33.7164, 73.0746),
    ("Bahria Town Phase 4 — Rawalpindi",
     "Bahria Town Phase 4, Rawalpindi", 33.5214, 73.0997),
    ("Clock Tower — Faisalabad",
     "Clock Tower, Faisalabad", 31.4187, 73.0791),
]


def main() -> None:
    parser = AddressParser()
    config = load_config()
    max_kw = float(config.get("sizing", {}).get("max_system_kw", 10.0))

    demos = []
    for label, address, lat, lng in GOLDEN:
        parsed = parser.parse(address)
        roof = building_insights(lat, lng)
        assert roof.get("demo"), "golden demos must use demo roof data"
        lay = optimize(roof, max_system_kw=max_kw)
        q = quote(lay, annual_kwh_per_kw=roof.get("annual_kwh_per_kw"))
        score = chhat_score(roof, lay, q)
        demos.append({
            "label": label,
            "address": address,
            "lat": lat,
            "lng": lng,
            "result": {
                "parsed": parsed,
                "geo": {"lat": lat, "lng": lng,
                        "formatted_address": address,
                        "confidence": "GOLDEN_DEMO",
                        "place_id": None},
                "roof": roof,
                "layout": lay,
                "quote": q,
                "score": score,
                "demo": True,
            },
        })
        print(f"{label}: score {score['score']}/100, "
              f"{lay['panel_count']} panels, {lay['system_kw']} kW, "
              f"Rs {q['equipment_cost_pkr']:,.0f}")

    out = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "note": ("Precomputed with the real Chhat pipeline in demo mode. "
                 "Pins are approximate landmark locations (demo pins). "
                 "All figures are estimates, not surveyed data."),
        "count": len(demos),
        "demos": demos,
    }
    path = os.path.join(os.path.dirname(__file__), "..", "docs", "pitch",
                        "golden-demos.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {path} ({os.path.getsize(path)//1024} KB)")


if __name__ == "__main__":
    main()
