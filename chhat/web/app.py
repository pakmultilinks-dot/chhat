"""Chhat web app — Satellite eye demo UI.

Flow: paste address -> parsed fields shown -> map pin confirm ->
roof visualization + Chhat Score + PKR quote.

Run: .venv/bin/python -m chhat.web.app   (http://localhost:5000)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from flask import Flask, jsonify, render_template, request

from chhat.geocode.client import GeocodeError, geocode
from chhat.layout.optimizer import optimize
from chhat.parser.predict import AddressParser
from chhat.quote.engine import chhat_score, load_config, quote
from chhat.solar.client import SolarError, building_insights

app = Flask(__name__)
_parser: AddressParser | None = None
_config = load_config()


def parser() -> AddressParser:
    global _parser
    if _parser is None:
        _parser = AddressParser()
    return _parser


STRINGS = {
    "en": {
        "title": "Chhat — AI rooftop solar intelligence",
        "tagline": "Paste your home address. We score your roof.",
        "address_ph": "House 123, Street 4, Block C, Nishat Colony, Lahore",
        "analyze": "Analyze my roof",
        "demo_badge": "DEMO ROOF — our segmentation model reads live roofs next; imagery sources plug in here",
        "step1": "Address", "step2": "Location", "step3": "Report",
        "parsed_title": "Parsed address", "confirm": "Confirm location →",
        "pin_hint": "Drag the pin to your exact rooftop, then confirm.",
        "score_title": "Chhat Score", "roof_title": "Roof layout",
        "quote_title": "Quote breakdown", "working": "Working…",
        "house": "House", "street": "Street", "block": "Block",
        "colony": "Colony", "city": "City",
        "system_size": "System size", "panels": "Panels (550W)",
        "yearly_gen": "Yearly generation", "payback": "Payback period",
        "yrs": "yrs", "system_cost": "System cost", "saved_year": "Saved / year",
        "savings_25y": "25-yr savings", "area_shade_payback": "Area / Shade / Payback",
        "usable_roof": "Usable roof",
    },
    "ur": {
        "title": "چھت — شمسی توانائی کی ذہین معلومات",
        "tagline": "اپنا پتہ لکھیں۔ ہم آپ کی چھت کا اسکور بتاتے ہیں۔",
        "address_ph": "مکان 123، گلی 4، بلاک C، نشاط کالونی، لاہور",
        "analyze": "میری چھت کا تجزیہ کریں",
        "demo_badge": "ڈیمو چھت — ہماری ماڈل جلد اصل چھتیں پڑھے گی",
        "step1": "پتہ", "step2": "مقام", "step3": "رپورٹ",
        "parsed_title": "پتے کی تفصیل", "confirm": "← مقام کی تصدیق کریں",
        "pin_hint": "پن کو اپنی ٹھیک چھت پر لگائیں، پھر تصدیق کریں۔",
        "score_title": "چھت اسکور", "roof_title": "چھت کا خاکہ",
        "quote_title": "قیمت کی تفصیل", "working": "کام جاری…",
        "house": "مکان", "street": "گلی", "block": "بلاک",
        "colony": "کالونی", "city": "شہر",
        "system_size": "سسٹم سائز", "panels": "پینل (550 واٹ)",
        "yearly_gen": "سالانہ بجلی", "payback": "واپسی کی مدت",
        "yrs": "سال", "system_cost": "سسٹم کی قیمت", "saved_year": "سالانہ بچت",
        "savings_25y": "25 سالہ بچت",
        "area_shade_payback": "رقبہ / سایہ / واپسی",
        "usable_roof": "قابل استعمال چھت",
    },
}


@app.get("/")
def index():
    lang = request.args.get("lang", "en")
    s = STRINGS.get(lang, STRINGS["en"])
    import json as _json
    return render_template("index.html", s=s, lang=lang,
                           ui_json=_json.dumps(s, ensure_ascii=False))


@app.post("/api/parse")
def api_parse():
    text = (request.json or {}).get("address", "").strip()
    if not text:
        return jsonify({"error": "empty address"}), 400
    p = parser().parse(text)
    return jsonify(p)


@app.post("/api/analyze")
def api_analyze():
    body = request.json or {}
    text = body.get("address", "").strip()
    lat, lng = body.get("lat"), body.get("lng")
    if not text:
        return jsonify({"error": "empty address"}), 400
    try:
        parsed = parser().parse(text)
        if lat is None or lng is None:
            g = geocode(parser().query_string(text))
            lat, lng = g["lat"], g["lng"]
            geo = g
        else:
            geo = {"lat": lat, "lng": lng, "formatted_address": text,
                   "confidence": "USER_PINNED", "place_id": None}
        roof = building_insights(lat, lng)
        max_kw = float(_config.get("sizing", {}).get("max_system_kw", 10.0))
        lay = optimize(roof, max_system_kw=max_kw)
        q = quote(lay, annual_kwh_per_kw=roof.get("annual_kwh_per_kw"))
        score = chhat_score(roof, lay, q)
        return jsonify({
            "parsed": parsed, "geo": geo, "roof": roof,
            "layout": lay, "quote": q, "score": score,
            "demo": bool(roof.get("demo")),
        })
    except (GeocodeError, SolarError) as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:  # noqa: BLE001 - surface honestly in demo
        return jsonify({"error": f"analysis failed: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
