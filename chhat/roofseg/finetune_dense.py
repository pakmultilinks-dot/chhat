"""Fine-tune OUR rooftop segmenter on the clean dense-street set.

Starts from models/rooftop_seg.pt (trained on planned-area z18 data) and
fine-tunes on data/roofseg_dense/ (human-verified z20 Nishat Colony /
Lahore Cantt chips). Lower LR, early stopping on dense-val IoU.

Saves: models/rooftop_seg_dense.pt (NOT the production weights — those
are only replaced if the dense model proves better on the E378 street).
Metrics appended to docs/roofseg-metrics.md by the caller, honestly.

Usage: .venv/bin/python -m chhat.roofseg.finetune_dense [--epochs 20]
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .model import TinyUNet, count_params
from .train import dice_loss, iou_score

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "roofseg_mixed")
INIT = os.path.join(os.path.dirname(__file__), "..", "..", "models", "rooftop_seg.pt")
SAVE = os.path.join(os.path.dirname(__file__), "..", "..", "models", "rooftop_seg_joint.pt")


class DenseDataset(Dataset):
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
        if self.augment:
            if np.random.rand() < 0.5:
                img, msk = img[:, ::-1].copy(), msk[:, ::-1].copy()
            if np.random.rand() < 0.5:
                img, msk = img[::-1].copy(), msk[::-1].copy()
            k = np.random.choice([0, 1, 2, 3], p=[0.4, 0.2, 0.2, 0.2])
            if k:
                img = np.ascontiguousarray(np.rot90(img, k))
                msk = np.ascontiguousarray(np.rot90(msk, k))
            b = np.random.uniform(0.85, 1.15)
            c = np.random.uniform(0.9, 1.1)
            img = np.clip((img - 0.5) * c + 0.5 * b, 0, 1).astype(np.float32)
        return (torch.from_numpy(img.transpose(2, 0, 1)),
                torch.from_numpy(msk).unsqueeze(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--resume", type=str, default=None,
                    help="path to checkpoint .pt to resume from (keeps its val_iou as best)")
    args = ap.parse_args()

    torch.manual_seed(11)
    np.random.seed(11)
    device = torch.device("cpu")
    torch.set_num_threads(2)

    tr = DenseDataset("train", augment=True)
    va = DenseDataset("val")
    te = DenseDataset("test")
    print(f"dense train {len(tr)} val {len(va)} test {len(te)}", flush=True)
    if len(tr) == 0:
        print("NO DENSE DATA — build it first (scripts/build_dense_clean.py + keep_dense_chips.py)")
        sys.exit(2)
    tr_dl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=0)
    va_dl = DataLoader(va, batch_size=args.batch, num_workers=0)

    model = TinyUNet().to(device)
    ckpt = torch.load(INIT, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    print(f"fine-tuning from {INIT} ({count_params(model):,} params)", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    best, bad, t0 = 0.0, 0, time.time()
    if args.resume:
        rckpt = torch.load(args.resume, map_location="cpu")
        rsd = rckpt["state_dict"] if isinstance(rckpt, dict) and "state_dict" in rckpt else rckpt
        model.load_state_dict(rsd)
        best = float(rckpt.get("val_iou", 0.0)) if isinstance(rckpt, dict) else 0.0
        print(f"resumed from {args.resume}, best val IoU {best:.4f}", flush=True)
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in tr_dl:
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
            for x, y in va_dl:
                ious.append(iou_score(model(x.to(device)), y.to(device)))
        viou = float(np.mean(ious)) if ious else 0.0
        print(f"epoch {ep+1}/{args.epochs} loss {tot/len(tr_dl):.4f} val_iou {viou:.4f} ({(time.time()-t0)/60:.1f}m)", flush=True)
        if viou > best + 1e-4:
            best, bad = viou, 0
            torch.save({"state_dict": model.state_dict(), "val_iou": viou}, SAVE)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {ep+1}, best dense-val IoU {best:.4f}", flush=True)
                break

    ckpt = torch.load(SAVE, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    te_dl = DataLoader(te, batch_size=args.batch, num_workers=0)
    ious, dices = [], []
    with torch.no_grad():
        for x, y in te_dl:
            out = model(x.to(device))
            ious.append(iou_score(out, y.to(device)))
            pr = torch.sigmoid(out)
            yy = y.to(device)
            dices.append(float((2 * (pr * yy).sum() + 1e-6) / (pr.sum() + yy.sum() + 1e-6)))
    print(f"DENSE TEST IoU {np.mean(ious):.4f} Dice {np.mean(dices):.4f} -> {SAVE}", flush=True)


if __name__ == "__main__":
    main()
