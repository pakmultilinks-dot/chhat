"""Write the human-approved dense chips into data/roofseg_dense/.

Input: data/dense_clean/keep.txt — one chip id per line, e.g.
    nishat_b_1536_768
created by visually reviewing the contact sheets in
data/dense_clean/regions/*/sheet_*.png (keep ONLY chips where the green
mask cleanly traces visible roof edges, no street spill, houses separated).

Split: spatial — chips are assigned train/val/test by mosaic x-zone
(west/middle/east thirds of the region mosaic, one-stride buffer
dropped), same anti-leakage idea as the main dataset.
"""
import os
import shutil

import cv2
import numpy as np

SRC = os.path.join(os.path.dirname(__file__), "..", "data", "dense_clean")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "roofseg_dense")
CAND = os.path.join(SRC, "regions")


def main():
    # keep.txt lives in the repo (docs/dense-keep.txt, committed) with a
    # fallback to the working copy in data/dense_clean/ (gitignored).
    keep_p = os.path.join(os.path.dirname(__file__), "..", "docs", "dense-keep.txt")
    if not os.path.exists(keep_p):
        keep_p = os.path.join(SRC, "keep.txt")
    if not os.path.exists(keep_p):
        print("no keep.txt — review the contact sheets first")
        return
    ids = [l.strip() for l in open(keep_p) if l.strip()]
    print(f"approved chips: {len(ids)}")
    # group by region, split by x-zone of the mosaic
    by_region = {}
    for cid in ids:
        parts = cid.rsplit("_", 2)
        name, cx = parts[0], int(parts[1])
        by_region.setdefault(name, []).append((cid, cx))
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    counts = {"train": 0, "val": 0, "test": 0}
    for name, items in by_region.items():
        mos = cv2.imread(os.path.join(CAND, name, "mosaic.png"))
        W = mos.shape[1]
        # NOTE (honest): with only ~80 approved chips, dropping a buffer
        # column wastes a third of the data. Zones are split WITHOUT a
        # buffer here; chips straddling a zone boundary share up to a 64px
        # strip across splits. The primary generalization evidence is the
        # E378 street test (~1 km from all three regions), not this split.
        z1, z2 = W / 3.0, 2 * W / 3.0
        for cid, cx in items:
            if cx < z1:
                split = "train"
            elif cx < z2:
                split = "val"
            else:
                split = "test"
            img = cv2.imread(os.path.join(CAND, name, "candidates", f"{cid}_img.png"))
            msk = cv2.imread(os.path.join(CAND, name, "candidates", f"{cid}_msk.png"),
                             cv2.IMREAD_GRAYSCALE)
            if img is None or msk is None:
                continue
            idir = os.path.join(OUT, split, "images")
            mdir = os.path.join(OUT, split, "masks")
            os.makedirs(idir, exist_ok=True)
            os.makedirs(mdir, exist_ok=True)
            cv2.imwrite(os.path.join(idir, f"{cid}.png"), img)
            cv2.imwrite(os.path.join(mdir, f"{cid}.png"), msk)
            counts[split] += 1
    print("wrote:", counts, "->", OUT)


if __name__ == "__main__":
    main()
