"""Net-billing-era quote engine.

Math for Pakistan's post-Oct-2026 net billing regime:
  - a self-consumed unit saves the retail tariff (Rs35-65)
  - an exported unit earns only the buyback rate (Rs8-11)
So the optimizer sizes for self-consumption, not maximum generation.

All prices come from chhat/quote/prices.yaml. Nothing is hardcoded.
"""
from __future__ import annotations

import os

import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "prices.yaml")


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def quote(layout: dict, annual_kwh_per_kw: float | None = None,
          config: dict | None = None) -> dict:
    cfg = config or load_config()
    eq, tf, gen, cm = (cfg["equipment"], cfg["tariff"],
                       cfg["generation"], cfg["consumption_model"])

    kw = layout["system_kw"]
    panels = layout["panel_count"]
    kwh_per_kw = annual_kwh_per_kw or gen["annual_kwh_per_kw"]
    annual_kwh = kw * kwh_per_kw

    scr = cm["self_consumption_ratio"]
    self_kwh = annual_kwh * scr
    export_kwh = annual_kwh * (1 - scr)

    annual_savings = (self_kwh * tf["retail_rs_per_unit"]
                      + export_kwh * tf["export_rs_per_unit"])

    cost = (panels * eq["panel_550w_pkr"]
            + kw * (eq["inverter_per_kw_pkr"] + eq["mounting_per_kw_pkr"]
                    + eq["labor_per_kw_pkr"])
            + eq["net_metering_fee_pkr"])

    payback_years = cost / annual_savings if annual_savings > 0 else float("inf")
    lifetime_savings_25y = annual_savings * 25 * (1 + tf["annual_tariff_hike"]) ** 12

    return {
        "system_kw": kw,
        "panel_count": panels,
        "annual_generation_kwh": round(annual_kwh, 0),
        "self_consumed_kwh": round(self_kwh, 0),
        "exported_kwh": round(export_kwh, 0),
        "equipment_cost_pkr": round(cost, 0),
        "annual_savings_pkr": round(annual_savings, 0),
        "payback_years": round(payback_years, 1),
        "lifetime_savings_25y_pkr": round(lifetime_savings_25y, 0),
        "tariff_used": tf["retail_rs_per_unit"],
        "export_rate_used": tf["export_rs_per_unit"],
    }


def chhat_score(roof: dict, layout: dict, q: dict,
                config: dict | None = None) -> dict:
    """0-100 rooftop solar viability score."""
    cfg = config or load_config()
    w = cfg["scoring"]

    usable = layout.get("usable_area_m2", 0)
    area_score = min(usable / 120.0, 1.0) * 100  # 120 sqm = great roof

    shades = [s.get("shade_fraction") or 0 for s in roof.get("segments", [])]
    avg_shade = sum(shades) / max(1, len(shades))
    shade_score = max(0, (1 - avg_shade * 2.5)) * 100

    kw = layout.get("system_kw", 0)
    fit_score = min(kw / 5.0, 1.0) * 100  # 5kW = ideal home system

    pb = q.get("payback_years", 99)
    payback_score = max(0, min(1, (8 - pb) / 6)) * 100  # <=2y:100, >=8y:0

    total = (w["w_usable_area"] * area_score + w["w_shade"] * shade_score
             + w["w_system_fit"] * fit_score + w["w_payback"] * payback_score)
    return {
        "score": round(total, 0),
        "breakdown": {
            "usable_area": round(area_score, 0),
            "shade": round(shade_score, 0),
            "system_fit": round(fit_score, 0),
            "payback": round(payback_score, 0),
        },
    }
