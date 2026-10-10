"""Panel layout optimizer.

Places standard solar panels on roof segments while avoiding obstructions
(water tanks, mumty/stair bulkheads, parapets) typical of Pakistani rooftops.

Simple, honest model: panels are 2.0m x 1.0m rectangles; each segment is
treated as a rectangle derived from its area (aspect 4:3); obstructions are
placed rectangles subtracted from usable space. Output: panel count and
per-panel coordinates in a normalized roof frame for visualization.
"""
from __future__ import annotations

import math

PANEL_W_M = 2.0   # standard 550W panel footprint
PANEL_H_M = 1.0
PANEL_WP = 550    # watts per panel
SETBACK_M = 0.3   # maintenance walkway margin


def _rect_for_area(area_m2: float) -> tuple[float, float]:
    w = math.sqrt(area_m2 * 4 / 3)
    return w, area_m2 / w


def optimize(roof: dict, max_system_kw: float = 10.0) -> dict:
    segments = roof.get("segments", [])
    obstructions = roof.get("obstructions", [])
    obstruction_area = sum(o.get("area_m2", 0) for o in obstructions)

    panels: list[dict] = []
    usable_total = 0.0
    seg_rects: list[dict] = []
    max_panels = int(max_system_kw * 1000 / PANEL_WP)
    for si, seg in enumerate(segments):
        if len(panels) >= max_panels:
            break
        area = seg.get("area_m2") or 0
        if area <= 0:
            continue
        w, h = _rect_for_area(area)
        # reserve obstruction space proportionally on the first segment
        reserve = obstruction_area if si == 0 else 0.0
        eff_area = max(area - reserve, 0)
        usable_total += eff_area
        ew, eh = _rect_for_area(eff_area)
        cols = max(int((ew - SETBACK_M) // (PANEL_W_M + 0.05)), 0)
        rows = max(int((eh - SETBACK_M) // (PANEL_H_M + 0.05)), 0)
        # stack segments vertically in the normalized roof frame
        oy = sum(r["h"] for r in seg_rects)
        seg_rects.append({"x": 0, "y": round(oy, 2), "w": round(ew, 2),
                          "h": round(eh, 2), "azimuth": seg.get("azimuth_degrees")})
        for r in range(rows):
            for c in range(cols):
                if len(panels) >= max_panels:
                    break
                panels.append({
                    "segment": si,
                    "x": round(c * (PANEL_W_M + 0.05) + SETBACK_M / 2, 2),
                    "y": round(oy + r * (PANEL_H_M + 0.05) + SETBACK_M / 2, 2),
                    "w": PANEL_W_M, "h": PANEL_H_M,
                    "azimuth": seg.get("azimuth_degrees"),
                })
            if len(panels) >= max_panels:
                break

    n_panels = len(panels)
    system_kw = round(n_panels * PANEL_WP / 1000, 2)
    return {
        "panel_count": n_panels,
        "system_kw": system_kw,
        "usable_area_m2": round(usable_total, 1),
        "obstruction_area_m2": round(obstruction_area, 1),
        "panels": panels,
        "segments": seg_rects,
        "panel_wp": PANEL_WP,
    }
