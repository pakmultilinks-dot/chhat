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
from chhat.solar.client import SolarError, building_insights, roof_imagery

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
        "demo_badge": "DEMO — all roof figures are estimates. Our own trained segmentation model reads the live imagery in step 4.",
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
        "est_tag": "(estimate)",
        "step4": "Roof AI",
        "roofai_title": "Roof Analysis — our own trained model",
        "roofai_hint": "Our TinyUNet rooftop segmenter, trained from scratch on Lahore aerial data, reads real satellite imagery of your pinned location.",
        "see_ai": "See what our AI sees →",
        "overlay_label": "Show predicted roofs",
        "conf_label": "Model confidence",
        "ok_badge": "Model confident — roofs detected",
        "unsure_badge": "Needs human review",
        "unsure_note": "Our model is not confident about this roof. Figures above are estimates — a human should verify before any real quote.",
        "back_report": "← Back to report",
    },
    "ur": {
        "title": "چھت — شمسی توانائی کی ذہین معلومات",
        "tagline": "اپنا پتہ لکھیں۔ ہم آپ کی چھت کا اسکور بتاتے ہیں۔",
        "address_ph": "مکان 123، گلی 4، بلاک C، نشاط کالونی، لاہور",
        "analyze": "میری چھت کا تجزیہ کریں",
        "demo_badge": "ڈیمو — چھت کے تمام اعداد تخمینہ ہیں۔ ہمارا اپنا تربیت یافتہ ماڈل مرحلہ 4 میں اصل تصویر پڑھتا ہے۔",
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
        "est_tag": "(تخمینہ)",
        "step4": "چھت AI",
        "roofai_title": "چھت کا تجزیہ — ہمارا اپنا تربیت یافتہ ماڈل",
        "roofai_hint": "ہمارا TinyUNet ماڈل، جو لاہور کی فضائی تصاویر پر تربیت یافتہ ہے، آپ کے مقام کی اصل سیٹلائٹ تصویر پڑھتا ہے۔",
        "see_ai": "دیکھیں ہمارا AI کیا دیکھتا ہے ←",
        "overlay_label": "متوقع چھتیں دکھائیں",
        "conf_label": "ماڈل کا اعتماد",
        "ok_badge": "ماڈل پراعتماد — چھتیں ملیں",
        "unsure_badge": "انسانی جائزے کی ضرورت",
        "unsure_note": "ہمارا ماڈل اس چھت کے بارے میں پراعتماد نہیں۔ اوپر دیے گئے اعداد تخمینہ ہیں — اصل قیمت سے پہلے انسانی تصدیق ضروری ہے۔",
        "back_report": "→ رپورٹ پر واپس",
    },
}


@app.get("/")
def index():
    lang = request.args.get("lang", "en")
    s = STRINGS.get(lang, STRINGS["en"])
    import json as _json
    return render_template("index.html", s=s, lang=lang,
                           ui_json=_json.dumps(s, ensure_ascii=False),
                           golden_json=_golden_json_text())


_GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "..", "..",
                            "docs", "pitch", "golden-demos.json")


def _golden_json_text() -> str:
    """Embedded golden demos so the offline fallback works with zero backend."""
    try:
        with open(_GOLDEN_PATH, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return '{"count": 0, "demos": []}'


@app.get("/golden-demos.json")
def golden_demos():
    """Serve the precomputed golden demo set (offline fallback data)."""
    from flask import Response
    return Response(_golden_json_text(), mimetype="application/json")


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


def _b64_png(bgr_img) -> str:
    import base64
    import cv2
    import numpy as np
    img = np.ascontiguousarray(bgr_img)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("png encode failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


@app.post("/api/roofseg")
def api_roofseg():
    """Run OUR trained rooftop segmenter on real imagery at the pinned point.

    Returns the satellite image, the predicted roof mask overlay (only when
    the model is confident), the confidence score, and the status. When the
    model is UNSURE we return no mask — the UI must show "needs human
    review", never a fake confident overlay.
    """
    body = request.json or {}
    lat, lng = body.get("lat"), body.get("lng")
    if lat is None or lng is None:
        return jsonify({"error": "missing lat/lng"}), 400
    try:
        from chhat.roofseg.predict import predict
        import numpy as np
        img, source = roof_imagery(float(lat), float(lng))
        if img is None or img.size == 0:
            return jsonify({"error": "imagery unavailable for this point"}), 502
        res = predict(img)
        out = {
            "status": res["status"],
            "confidence": round(res["confidence"], 3),
            "building_fraction": round(res["building_fraction"], 3),
            "source": source,
            "imagery": _b64_png(img),
            "overlay": None,
        }
        if res["mask"] is not None:
            ov = img.astype(np.float32)
            m = res["mask"] > 0
            ov[m] = ov[m] * 0.45 + np.array([0, 200, 0], np.float32) * 0.55  # BGR green
            out["overlay"] = _b64_png(ov.astype(np.uint8))
        return jsonify(out)
    except Exception as e:  # noqa: BLE001 - surface honestly in demo
        return jsonify({"error": f"roof analysis failed: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
