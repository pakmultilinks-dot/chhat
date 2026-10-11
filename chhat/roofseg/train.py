"""Train OUR rooftop segmentation model from scratch.

Data: data/roofseg/{train,val,test}/{images,masks} (built by
scripts/build_roofseg_data.py from OSM + Esri open data).

Model: TinyUNet (~0.5M params), 256x256, CPU. Loss: BCE + Dice.
Early stopping on val IoU. Honest metrics written to docs/roofseg-metrics.md.

Usage: .venv/bin/python -m chhat.roofseg.train [--epochs 30]
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .model import TinyUNet, count_params

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "roofseg")


class RoofSegDataset(Dataset):
    def __init__(self, split, augment=False):
        self.imgs = sorted(glob.glob(os.path.join(ROOT, split, "images", "*.png")))
        self.augment = augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        img = cv2.imread(p).astype(np.float32) / 255.0
        msk = cv2.imread(p.replace("/images/", "/masks/"), cv2.IMREAD_GRAYSCALE)
        msk = (msk > 127).astype(np.float32)
        if self.augment and np.random.rand() < 0.5:
            img, msk = img[:, ::-1].copy(), msk[:, ::-1].copy()
        if self.augment and np.random.rand() < 0.5:
            img, msk = img[::-1].copy(), msk[::-1].copy()
        img = torch.from_numpy(img.transpose(2, 0, 1))
        msk = torch.from_numpy(msk).unsqueeze(0)
        return img, msk


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    return (1 - (2 * inter + eps) / (union + eps)).mean()


def iou_score(logits, targets, eps=1e-6):
    preds = (torch.sigmoid(logits) > 0.5).float()
    inter = (preds * targets).sum()
    union = preds.sum() + targets.sum() - inter
    return float((inter + eps) / (union + eps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--resume", type=str, default=None,
                    help="path to checkpoint .pt to resume from (keeps its val_iou as best)")
    args = ap.parse_args()

    torch.manual_seed(7)
    np.random.seed(7)
    device = torch.device("cpu")
    torch.set_num_threads(2)

    train_ds = RoofSegDataset("train", augment=True)
    val_ds = RoofSegDataset("val")
    test_ds = RoofSegDataset("test")
    print(f"train {len(train_ds)} val {len(val_ds)} test {len(test_ds)}", flush=True)
    if len(train_ds) == 0:
        print("NO TRAINING DATA — run scripts/build_roofseg_data.py first", flush=True)
        sys.exit(2)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, num_workers=0)

    model = TinyUNet().to(device)
    print(f"params: {count_params(model):,}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    best_iou, bad, t0 = 0.0, 0, time.time()
    start_ep = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(sd)
        best_iou = float(ckpt.get("val_iou", 0.0)) if isinstance(ckpt, dict) else 0.0
        print(f"resumed from {args.resume}, best val IoU {best_iou:.4f}", flush=True)
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = bce(out, y) + dice_loss(out, y)
            loss.backward()
            opt.step()
            tot += loss.item()
        model.eval()
        ious = []
        with torch.no_grad():
            for x, y in val_dl:
                ious.append(iou_score(model(x.to(device)), y.to(device)))
        viou = float(np.mean(ious)) if ious else 0.0
        dt = time.time() - t0
        print(f"epoch {ep+1}/{args.epochs} loss {tot/len(train_dl):.4f} val_iou {viou:.4f} ({dt/60:.1f}m)", flush=True)
        if viou > best_iou + 1e-4:
            best_iou, bad = viou, 0
            torch.save({"state_dict": model.state_dict(), "val_iou": viou},
                       os.path.join(os.path.dirname(__file__), "..", "..", "models", "rooftop_seg.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {ep+1}, best val IoU {best_iou:.4f}", flush=True)
                break

    # final honest test evaluation
    ckpt = torch.load(os.path.join(os.path.dirname(__file__), "..", "..", "models", "rooftop_seg.pt"),
                      map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    test_dl = DataLoader(test_ds, batch_size=args.batch, num_workers=0)
    ious, dices = [], []
    with torch.no_grad():
        for x, y in test_dl:
            out = model(x.to(device))
            ious.append(iou_score(out, y.to(device)))
            probs = torch.sigmoid(out)
            inter = (probs * y.to(device)).sum()
            dices.append(float((2 * inter + 1e-6) / (probs.sum() + y.to(device).sum() + 1e-6)))
    tiou, tdice = float(np.mean(ious)), float(np.mean(dices))
    print(f"TEST IoU {tiou:.4f} Dice {tdice:.4f}", flush=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "roofseg-metrics.md"), "w") as f:
        f.write("# rooftop_seg metrics (honest)\n\n")
        f.write(f"- Model: TinyUNet from scratch, {count_params(model):,} params, 256x256, CPU\n")
        f.write(f"- Data: OSM footprints + Esri imagery, Lahore. train {len(train_ds)}, "
                f"val {len(val_ds)}, test {len(test_ds)}\n")
        f.write(f"- Best val IoU: {best_iou:.4f}\n")
        f.write(f"- **Test IoU: {tiou:.4f} | Test Dice: {tdice:.4f}**\n")
        f.write("- Caveat: OSM/Esri label noise (misalignment, stale footprints) caps the "
                "achievable score; predictions are confidence-gated (see chhat/roofseg/predict.py).\n")


if __name__ == "__main__":
    main()
