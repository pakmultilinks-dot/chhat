"""Build a CLEAN, human-verified dense-street fine-tuning set for Chhat.

Why: OSM building footprints are essentially ABSENT in dense Lahore
residential areas (probed 2026-10-11: Nishat Colony 7/km2, Walton 2/km2,
Samanabad 3/km2, Ichhra 23/km2, Model Town 4/km2). There is nothing to
"snap" there. So this script does the honest alternative:

  1. Pull Esri World Imagery at z19 (~0.3 m/px; measured 2026-10-11
     that z20 returns only gray placeholder tiles over Lahore) over
     dense Nishat Colony / Lahore Cantt blocks.
  2. Generate candidate roof masks with a morphological pipeline
     (gradient + watershed + size/shape filters).
  3. A human (me, the builder) VISUALLY inspects every candidate chip
     on a contact sheet and keeps ONLY chips where the mask cleanly
     matches visible roof edges (>=95% correct after auto-cleanup).
     Labels are then machine-proposed + human-verified: fully ours.

Step 1 (this script): download mosaics -> candidate masks -> contact
sheets for review. Step 2 (after review): scripts/keep_dense_chips.py
writes the kept chips into data/roofseg_dense/{train,val,test}.

Regions (dense residential, Lahore Cantt; neighborhood centers only,
no private addresses):
  - nishat_b: (31.4920, 74.3935)
  - nishat_w: (31.4900, 74.3860)
  - cantt_n:  (31.4980, 74.3980)
"""
import math
import os
import time

import cv2
import numpy as np
import requests

REGIONS = [
    ("nishat_b", 31.4920, 74.3935),
    ("nishat_w", 31.4900, 74.3860),
    ("cantt_n", 31.4980, 74.3980),
]
# NOTE (measured 2026-10-11): Esri has NO real z20 coverage over Lahore
# (returns flat gray placeholder tiles). z19 (~0.3 m/px) is the max
# usable zoom. Candidate-mask size thresholds below are in z19 pixels.
REGION_KM = 0.6
Z = 19
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "dense_clean")
TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
UA = {"User-Agent": "chhat-roofseg-dataset-builder (research project)"}


def latlon_to_tile(lat, lon, z):
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    lr = math.radians(lat)
    y = int((1 - math.asinh(math.tan(lr)) / math.pi) / 2 * n)
    return x, y


def _rough_candidates(img):
    """Step 1: morphological roof-region proposals (erode-based)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 5, 40, 40)
    grad = cv2.morphologyEx(g, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    _, edge = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(edge, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=2)
    inv = 255 - closed
    fg = cv2.erode(inv, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    # vegetation filter: drop components that are mostly trees/grass.
    # ExG = 2G-R-B is high for vegetation; roofs (concrete) are low.
    bgr = img.astype(np.float32)
    exg = 2 * bgr[:, :, 1] - bgr[:, :, 0] - bgr[:, :, 2]
    keep = np.zeros_like(fg)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        # z19 (~0.3 m/px): house roof ~8x12 m .. 25x30 m -> ~800 .. 12000 px
        if 800 <= area <= 12000:
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if max(w, h) / max(min(w, h), 1) <= 4.0:
                seg = (lab == i)
                if exg[seg].mean() < 25:  # not vegetation
                    keep[seg] = 255
    ffmask = np.zeros((keep.shape[0] + 2, keep.shape[1] + 2), np.uint8)
    ff = keep.copy()
    cv2.floodFill(ff, ffmask, (0, 0), 128)
    keep[ff == 0] = 255
    keep = cv2.morphologyEx(keep, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return keep


def _snap_to_edges(img, m):
    """Step 2: edge-aware refinement. Move mask boundaries to the nearest
    real image edges (Canny), so labels trace visible rooflines instead of
    blobby morphological approximations. Pixels far from any edge keep the
    safe interior."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 60, 150)
    dt = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    H, W = m.shape
    inner = cv2.erode(m, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero((m > 0) & (inner == 0))
    snapped = np.zeros_like(m)
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - 4), min(H, y + 5)
        x0, x1 = max(0, x - 4), min(W, x + 5)
        win = dt[y0:y1, x0:x1]
        jy, jx = np.unravel_index(np.argmin(win), win.shape)
        if win[jy, jx] <= 4:
            snapped[y0 + jy, x0 + jx] = 255
    thick = cv2.dilate(snapped, np.ones((3, 3), np.uint8))
    safe_interior = cv2.erode(m, np.ones((7, 7), np.uint8))
    near_orig = cv2.dilate(m, np.ones((7, 7), np.uint8))
    out = np.zeros_like(m)
    out[(safe_interior > 0) | ((thick > 0) & (near_orig > 0))] = 255
    ffmask = np.zeros((H + 2, W + 2), np.uint8)
    ff = out.copy()
    cv2.floodFill(ff, ffmask, (0, 0), 128)
    out[ff == 0] = 255
    return cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def candidate_mask(img):
    """Roof-candidate mask: morphological proposals + edge-aware boundary
    refinement. A human keeps only the clean chips (masks that trace
    visible roof edges, houses separated, no street spill)."""
    return _snap_to_edges(img, _rough_candidates(img))


def build_region(name, clat, clon):
    rdir = os.path.join(OUT, "regions", name)
    os.makedirs(rdir, exist_ok=True)
    mos_p = os.path.join(rdir, "mosaic.png")
    if os.path.exists(mos_p):
        mosaic = cv2.imread(mos_p)
        print(f"[{name}] cached mosaic {mosaic.shape}", flush=True)
    else:
        dlat = (REGION_KM / 2) / 111.0
        dlon = (REGION_KM / 2) / (111.0 * math.cos(math.radians(clat)))
        x0, ymin = latlon_to_tile(clat + dlat, clon - dlon, Z)
        x1, ymax = latlon_to_tile(clat - dlat, clon + dlon, Z)
        nx, ny = x1 - x0 + 1, ymax - ymin + 1
        print(f"[{name}] {nx}x{ny}={nx*ny} tiles", flush=True)
        mosaic = np.zeros((ny * 256, nx * 256, 3), np.uint8)
        ok = 0
        for iy in range(ny):
            for ix in range(nx):
                url = TILE_URL.format(z=Z, y=ymin + iy, x=x0 + ix)
                try:
                    rr = requests.get(url, headers=UA, timeout=30)
                    if rr.status_code == 200 and len(rr.content) > 4000:
                        im = cv2.imdecode(np.frombuffer(rr.content, np.uint8), cv2.IMREAD_COLOR)
                        if im is not None and im.shape[:2] == (256, 256):
                            mosaic[iy*256:(iy+1)*256, ix*256:(ix+1)*256] = im
                            ok += 1
                except Exception:
                    pass
                time.sleep(0.04)
        print(f"[{name}] tiles ok {ok}/{nx*ny}", flush=True)
        cv2.imwrite(mos_p, mosaic)

    # candidate masks + 256px chips (stride 192 -> less overlap, more variety)
    H, W = mosaic.shape[:2]
    cand_dir = os.path.join(rdir, "candidates")
    os.makedirs(cand_dir, exist_ok=True)
    n = 0
    thumbs = []
    for cy in range(0, H - 255, 192):
        for cx in range(0, W - 255, 192):
            chip = mosaic[cy:cy+256, cx:cx+256]
            if (chip == 0).mean() > 0.05:
                continue  # tile gaps
            m = candidate_mask(chip)
            cov = (m > 0).mean()
            if not (0.05 <= cov <= 0.90):
                continue
            cv2.imwrite(os.path.join(cand_dir, f"{name}_{cx}_{cy}_img.png"), chip)
            cv2.imwrite(os.path.join(cand_dir, f"{name}_{cx}_{cy}_msk.png"), m)
            ov = chip.copy()
            ov[m > 0] = (0, 255, 0)
            vis = cv2.addWeighted(chip, 0.55, ov, 0.45, 0)
            cv2.putText(vis, f"{name}_{cx}_{cy}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            thumbs.append((f"{name}_{cx}_{cy}", vis))
            n += 1
    print(f"[{name}] candidates: {n}", flush=True)

    # contact sheets (8 per row) for human review
    per_row = 8
    for s in range(0, len(thumbs), 48):
        batch = thumbs[s:s+48]
        rows = []
        for r in range(0, len(batch), per_row):
            row = batch[r:r+per_row]
            while len(row) < per_row:
                row.append(("", np.zeros((256, 256, 3), np.uint8)))
            rows.append(np.hstack([t[1] for t in row]))
        sheet = np.vstack(rows)
        cv2.imwrite(os.path.join(rdir, f"sheet_{s//48:02d}.png"), sheet)
        with open(os.path.join(rdir, f"sheet_{s//48:02d}.txt"), "w") as f:
            f.write("\n".join(t[0] for t in batch) + "\n")
    print(f"[{name}] sheets written", flush=True)


if __name__ == "__main__":
    for name, clat, clon in REGIONS:
        build_region(name, clat, clon)
    print("done ->", OUT)
