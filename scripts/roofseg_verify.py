"""Verify the trained rooftop segmenter on test chips.

Runs chhat.roofseg.predict on 6 test images, saves side-by-side
(image | true mask | predicted mask | status) grid to docs/roofseg-verify.png,
and prints per-image status/confidence. This is the honesty check: we look at
what the model actually does, including UNSURE cases.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

from chhat.roofseg.predict import predict

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "roofseg", "test")


def main():
    imgs = sorted(glob.glob(os.path.join(ROOT, "images", "*.png")))[:6]
    rows = []
    for p in imgs:
        img = cv2.imread(p)
        true = cv2.imread(p.replace("/images/", "/masks/"), cv2.IMREAD_GRAYSCALE)
        r = predict(img)
        pred = r["mask"] if r["mask"] is not None else np.zeros_like(true)
        label = np.full((40, 256 * 3, 3), 255, np.uint8)
        cv2.putText(label, f"{os.path.basename(p)} {r['status']} conf={r['confidence']:.2f}",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        row = np.vstack([label,
                         np.hstack([img, cv2.cvtColor(true, cv2.COLOR_GRAY2BGR),
                                    cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)])])
        rows.append(row)
        print(os.path.basename(p), r["status"], f"conf={r['confidence']:.2f}",
              f"bldg_frac={r['building_fraction']:.2f}")
    grid = np.vstack(rows)
    out = os.path.join(os.path.dirname(__file__), "..", "docs", "roofseg-verify.png")
    cv2.imwrite(out, grid)
    print("saved", out)


if __name__ == "__main__":
    main()
