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
from chhat.quote.engine import chhat_score, quote
from chhat.solar.client import SolarError, building_insights

app = Flask(__name__)
_parser: AddressParser | None = None


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
        "demo_badge": "DEMO DATA — set GOOGLE_SOLAR_API_KEY for live roof data",
    },
    "ur": {
        "title": "چھت — شمسی توانائی کی ذہین معلومات",
        "tagline": "اپنا پتہ لکھیں۔ ہم آپ کی چھت کا اسکور بتاتے ہیں۔",
        "address_ph": "مکان 123، گلی 4، بلاک C، نشاط کالونی، لاہور",
        "analyze": "میری چھت کا تجزیہ کریں",
        "demo_badge": "ڈیمو ڈیٹا — اصل ڈیٹا کے لیے GOOGLE_SOLAR_API_KEY درج کریں",
    },
}


@app.get("/")
def index():
    lang = request.args.get("lang", "en")
    s = STRINGS.get(lang, STRINGS["en"])
    return render_template("index.html", s=s, lang=lang)


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
        lay = optimize(roof)
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
