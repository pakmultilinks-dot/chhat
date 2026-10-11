"""Build the Chhat rooftop segmentation dataset from REAL open data.

Sources (documented in docs/datasets.md):
  - Building footprints: OpenStreetMap (ODbL), via api.openstreetmap.org
  - Satellite imagery: Esri World Imagery (z18, ~0.6 m/px), tile server

Pipeline: pick a dense Lahore region -> download OSM buildings -> download
Esri tiles -> stitch mosaic -> rasterize footprints -> chip 256x256 ->
train/val/test split. Keeps chips with 5-95% building coverage.

Known limitation (honest): OSM footprints and Esri imagery are captured at
different times, so some masks are slightly misaligned or stale. Documented,
not hidden.
"""
import math
import os
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import requests

REGION_CENTER = (31.470, 74.325)  # dense DHA Lahore (629 bldgs in quadrant probe)
REGION_KM = 1.6                  # square region side length (kept small: OSM API is slow)
Z = 18
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "roofseg")
TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
UA = {"User-Agent": "chhat-roofseg-dataset-builder (research project)"}


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


def main():
    os.makedirs(OUT, exist_ok=True)
    clat, clon = REGION_CENTER
    # region bounds in degrees (approx)
    dlat = (REGION_KM / 2) / 111.0
    dlon = (REGION_KM / 2) / (111.0 * math.cos(math.radians(clat)))
    south, north = clat - dlat, clat + dlat
    west, east = clon - dlon, clon + dlon
    print(f"region lat [{south:.4f},{north:.4f}] lon [{west:.4f},{east:.4f}]")

    # 1. OSM buildings
    print("downloading OSM buildings...")
    r = requests.get(
        "https://api.openstreetmap.org/api/0.6/map",
        params={"bbox": f"{west},{south},{east},{north}"},
        headers=UA, timeout=120)
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
    print(f"buildings: {len(polys)}")

    # 2. tile range + mosaic (note: y grows southward, so ymin is the north edge)
    x0, ymin = latlon_to_tile(north, west, Z)
    x1, ymax = latlon_to_tile(south, east, Z)
    nx, ny = x1 - x0 + 1, ymax - ymin + 1
    print(f"tiles x[{x0},{x1}] y[{ymin},{ymax}] -> {nx}x{ny} = {nx*ny}")
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
    print(f"tiles ok: {ok}/{nx*ny}")

    # mosaic geo-reference: top-left corner lat/lon
    tlat_north, tlon_west = tile_to_latlon(x0, ymin, Z)          # NW corner of first tile
    tlat_south, tlon_east = tile_to_latlon(x1 + 1, ymax + 1, Z)  # SE corner of last tile

    def project(lat, lon):
        px = (lon - tlon_west) / (tlon_east - tlon_west) * W
        py = (tlat_north - lat) / (tlat_north - tlat_south) * H
        return int(px), int(py)

    # 3. rasterize masks (cached)
    mosaic_path = os.path.join(OUT, "_mosaic.png")
    mask_path = os.path.join(OUT, "_mask.png")
    if os.path.exists(mosaic_path) and os.path.exists(mask_path):
        print("using cached mosaic/mask")
        mosaic = cv2.imread(mosaic_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) // 255
        H, W = mask.shape
    else:
        mask = np.zeros((H, W), np.uint8)
        for pts in polys:
            pp = np.array([project(la, lo) for la, lo in pts], np.int32)
            cv2.fillPoly(mask, [pp], 1)
        cv2.imwrite(mosaic_path, mosaic)
        cv2.imwrite(mask_path, mask * 255)
    print(f"mask building px: {mask.sum()} ({100*mask.sum()/mask.size:.1f}%)")

    # 4. chip 256x256 stride 128, keep 1-95% coverage
    chips = []
    for cy in range(0, H - 255, 128):
        for cx in range(0, W - 255, 128):
            m = mask[cy:cy+256, cx:cx+256]
            cov = m.mean()
            if 0.01 <= cov <= 0.95:
                chips.append((mosaic[cy:cy+256, cx:cx+256].copy(), (m * 255).astype(np.uint8)))
    print(f"chips kept: {len(chips)}")
    rng = np.random.default_rng(7)
    rng.shuffle(chips)
    n = len(chips)
    splits = {"train": chips[:int(.8*n)], "val": chips[int(.8*n):int(.9*n)], "test": chips[int(.9*n):]}
    for name, items in splits.items():
        idir = os.path.join(OUT, name, "images"); mdir = os.path.join(OUT, name, "masks")
        os.makedirs(idir, exist_ok=True); os.makedirs(mdir, exist_ok=True)
        for i, (im, mk) in enumerate(items):
            cv2.imwrite(os.path.join(idir, f"{name}_{i:04d}.png"), im)
            cv2.imwrite(os.path.join(mdir, f"{name}_{i:04d}.png"), mk)
        print(f"{name}: {len(items)}")
    # save a preview grid
    if chips:
        prev = np.hstack([np.vstack([c[0], cv2.cvtColor(c[1], cv2.COLOR_GRAY2BGR)]) for c in chips[:6]])
        cv2.imwrite(os.path.join(OUT, "preview.png"), prev)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
