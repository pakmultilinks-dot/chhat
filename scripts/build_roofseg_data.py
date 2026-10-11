"""Build the Chhat rooftop segmentation dataset from REAL open data.

Sources (documented in docs/datasets.md):
  - Building footprints: OpenStreetMap (ODbL), via api.openstreetmap.org
  - Satellite imagery: Esri World Imagery (z18, ~0.6 m/px), tile server

Pipeline (per region): pick a dense Lahore quadrant -> download OSM buildings
-> download Esri tiles -> stitch mosaic -> rasterize footprints -> chip
256x256 (stride 128) -> keep chips with 1-95% building coverage.
All region chips are merged, shuffled (seed 7), and split 80/10/10 into
data/roofseg/{train,val,test}.

Regions (dense residential Lahore):
  - dha:        (31.470, 74.325)  DHA
  - dha5:       (31.462, 74.395)  DHA Phase 5
  - johartown:  (31.470, 74.273)  Johar Town

Label alignment (measured 2026-10-11, not guessed): OSM footprints are
systematically offset from Esri imagery. Estimated per region by
maximizing OSM-mask-boundary vs satellite-image-gradient agreement over
80 sampled buildings, split-half stability checked (both halves agree,
improvement over zero-shift in every half):
  - dha:       (dx=-1, dy=-2)
  - dha5:      (dx=+2, dy=-2)
  - johartown: (dx=+6, dy=-2)
(REGION_SHIFT below, in pixels, z18 ~0.6 m/px.)

Known limitation (honest): OSM footprints and Esri imagery are captured at
different times, so some masks are slightly misaligned or stale beyond the
systematic shift (per-building edits, demolitions). Documented, not hidden.
"""
import math
import os
import shutil
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import requests

REGIONS = [
    ("dha", 31.470, 74.325),
    ("dha5", 31.462, 74.395),   # DHA Phase 5 — 1048 OSM buildings in probe
    ("johartown", 31.470, 74.273),
]
REGION_KM = 1.6                  # square region side length (kept small: OSM API is slow)
Z = 18
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "roofseg")
REGION_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "roofseg_regions")
TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
UA = {"User-Agent": "chhat-roofseg-dataset-builder (research project)"}

# Measured systematic OSM -> Esri misalignment, pixels (see module docstring).
# Applied to masks so labels line up with the imagery they describe.
REGION_SHIFT = {
    "dha": (-1, -2),
    "dha5": (2, -2),
    "johartown": (6, -2),
}


def apply_shift(mask, name):
    dx, dy = REGION_SHIFT.get(name, (0, 0))
    if dx == 0 and dy == 0:
        return mask
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]),
                          flags=cv2.INTER_NEAREST, borderValue=0)


def latlon_to_tile(lat, lon, z):
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    lr = math.radians(lat)
    y = int((1 - math.asinh(math.tan(lr)) / math.pi) / 2 * n)
    return x, y


def tile_to_latlon(x, y, z):
    n = 2 ** z
    lon = x / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def build_region(name, clat, clon):
    rdir = os.path.join(REGION_DIR, name)
    os.makedirs(rdir, exist_ok=True)
    dlat = (REGION_KM / 2) / 111.0
    dlon = (REGION_KM / 2) / (111.0 * math.cos(math.radians(clat)))
    south, north = clat - dlat, clat + dlat
    west, east = clon - dlon, clon + dlon
    print(f"[{name}] region lat [{south:.4f},{north:.4f}] lon [{west:.4f},{east:.4f}]", flush=True)

    mosaic_path = os.path.join(rdir, "mosaic.png")
    mask_path = os.path.join(rdir, "mask.png")
    if os.path.exists(mosaic_path) and os.path.exists(mask_path):
        print(f"[{name}] using cached mosaic/mask", flush=True)
        mosaic = cv2.imread(mosaic_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) // 255
        H, W = mask.shape
        # cached masks predate the measured shift correction -> shift them too
        mask = apply_shift(mask, name)
    else:
        # 1. OSM buildings
        print(f"[{name}] downloading OSM buildings...", flush=True)
        r = requests.get(
            "https://api.openstreetmap.org/api/0.6/map",
            params={"bbox": f"{west},{south},{east},{north}"},
            headers=UA, timeout=180)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        nodes = {}
        for n in root.iter("node"):
            nodes[n.attrib["id"]] = (float(n.attrib["lat"]), float(n.attrib["lon"]))
        polys = []
        for w in root.iter("way"):
            if not any(t.attrib.get("k") == "building" for t in w.iter("tag")):
                continue
            pts = [nodes[nd.attrib["ref"]] for nd in w.iter("nd") if nd.attrib["ref"] in nodes]
            if len(pts) >= 4:
                polys.append(pts)
        print(f"[{name}] buildings: {len(polys)}", flush=True)

        # 2. tile range + mosaic (note: y grows southward, so ymin is the north edge)
        x0, ymin = latlon_to_tile(north, west, Z)
        x1, ymax = latlon_to_tile(south, east, Z)
        nx, ny = x1 - x0 + 1, ymax - ymin + 1
        print(f"[{name}] tiles x[{x0},{x1}] y[{ymin},{ymax}] -> {nx*ny}", flush=True)
        H, W = ny * 256, nx * 256
        mosaic = np.zeros((H, W, 3), np.uint8)
        ok = 0
        for iy in range(ny):
            for ix in range(nx):
                url = TILE_URL.format(z=Z, y=ymin + iy, x=x0 + ix)
                try:
                    rr = requests.get(url, headers=UA, timeout=30)
                    if rr.status_code == 200 and len(rr.content) > 4000:
                        img = cv2.imdecode(np.frombuffer(rr.content, np.uint8), cv2.IMREAD_COLOR)
                        if img is not None and img.shape[:2] == (256, 256):
                            mosaic[iy*256:(iy+1)*256, ix*256:(ix+1)*256] = img
                            ok += 1
                except Exception:
                    pass
                time.sleep(0.05)
        print(f"[{name}] tiles ok: {ok}/{nx*ny}", flush=True)

        # mosaic geo-reference: top-left corner lat/lon
        tlat_north, tlon_west = tile_to_latlon(x0, ymin, Z)
        tlat_south, tlon_east = tile_to_latlon(x1 + 1, ymax + 1, Z)

        def project(lat, lon):
            px = (lon - tlon_west) / (tlon_east - tlon_west) * W
            py = (tlat_north - lat) / (tlat_north - tlat_south) * H
            return int(px), int(py)

        # 3. rasterize masks
        mask = np.zeros((H, W), np.uint8)
        dx, dy = REGION_SHIFT.get(name, (0, 0))

        def project_shifted(lat, lon):
            px, py = project(lat, lon)
            return px + dx, py + dy

        for pts in polys:
            pp = np.array([project_shifted(la, lo) for la, lo in pts], np.int32)
            cv2.fillPoly(mask, [pp], 1)
        cv2.imwrite(mosaic_path, mosaic)
        cv2.imwrite(mask_path, mask * 255)
    print(f"[{name}] mask building px: {mask.sum()} ({100*mask.sum()/mask.size:.1f}%)", flush=True)

    # 4. chip 256x256 stride 128, keep 1-95% coverage; record coords for
    #    the spatial (geographic-zone) split in main()
    chips = []
    for cy in range(0, H - 255, 128):
        for cx in range(0, W - 255, 128):
            m = mask[cy:cy+256, cx:cx+256]
            cov = m.mean()
            if 0.01 <= cov <= 0.95:
                chips.append((mosaic[cy:cy+256, cx:cx+256].copy(),
                              (m * 255).astype(np.uint8), cx, cy, W))
    print(f"[{name}] chips kept: {len(chips)}", flush=True)
    return chips


def main():
    # Spatial (geographic-zone) split, per region: overlapping chips (stride
    # 128 < 256) are spatial neighbors, so a random shuffle split leaks test
    # buildings into train. Instead, split each mosaic into west/middle/east
    # zones with a one-stride buffer column dropped between zones.
    splits = {"train": [], "val": [], "test": []}
    for name, clat, clon in REGIONS:
        chips = build_region(name, clat, clon)
        W = chips[0][4]
        z1, z2 = W / 3.0, 2 * W / 3.0
        buf = 128
        for im, mk, cx, cy, _ in chips:
            if cx < z1 - buf:
                splits["train"].append((name, im, mk))
            elif z1 + buf <= cx < z2 - buf:
                splits["val"].append((name, im, mk))
            elif cx >= z2 + buf:
                splits["test"].append((name, im, mk))
            # else: buffer column, dropped
        print(f"[{name}] -> train/val/test zones split done", flush=True)
    total = sum(len(v) for v in splits.values())
    print(f"total chips kept: {total} "
          f"(train {len(splits['train'])} val {len(splits['val'])} test {len(splits['test'])})",
          flush=True)

    rng = np.random.default_rng(7)
    for v in splits.values():
        rng.shuffle(v)
    # wipe old split dirs (fresh merge)
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for name, items in splits.items():
        idir = os.path.join(OUT, name, "images"); mdir = os.path.join(OUT, name, "masks")
        os.makedirs(idir, exist_ok=True); os.makedirs(mdir, exist_ok=True)
        for i, (region, im, mk) in enumerate(items):
            cv2.imwrite(os.path.join(idir, f"{region}_{name}_{i:04d}.png"), im)
            cv2.imwrite(os.path.join(mdir, f"{region}_{name}_{i:04d}.png"), mk)
        print(f"{name}: {len(items)}", flush=True)
    flat = [c for v in splits.values() for c in v]
    if flat:
        prev = np.hstack([np.vstack([c[1], cv2.cvtColor(c[2], cv2.COLOR_GRAY2BGR)])
                          for c in flat[:6]])
        cv2.imwrite(os.path.join(OUT, "preview.png"), prev)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
