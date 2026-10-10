"""Export trained address-parser weights from torch .pt to numpy .npz.

The .npz bundle lets the web app run inference with pure numpy
(no torch needed) — required for lightweight serverless deployment.

Run: .venv/bin/python scripts/export_numpy_weights.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from chhat.parser.train import LABELS

SRC = os.path.join(os.path.dirname(__file__), "..", "models", "address_parser.pt")
DST = os.path.join(os.path.dirname(__file__), "..", "models", "address_parser.npz")


def main() -> None:
    bundle = torch.load(SRC, map_location="cpu", weights_only=False)
    cfg = bundle["config"]
    sd = bundle["state_dict"]
    out = {
        "emb": sd["embedding.weight"].numpy(),
        "cls_w": sd["classifier.weight"].numpy(),
        "cls_b": sd["classifier.bias"].numpy(),
        "vocab_keys": np.array(list(bundle["vocab"].keys())),
        "vocab_vals": np.array(list(bundle["vocab"].values()), dtype=np.int64),
        "labels": np.array(bundle["labels"]),
        "emb_dim": np.int64(cfg.get("emb_dim", 64)),
        "hidden_dim": np.int64(cfg.get("hidden_dim", 128)),
        "num_layers": np.int64(cfg.get("num_layers", 2)),
    }
    n_layers = int(out["num_layers"])
    for l in range(n_layers):
        for d in ("", "_reverse"):
            for w in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                key = f"lstm.{w}_l{l}{d}"
                out[f"lstm_l{l}{d}_{w}"] = sd[key].numpy()
    np.savez_compressed(DST, **out)
    size_kb = os.path.getsize(DST) / 1024
    print(f"wrote {DST} ({size_kb:.0f} KB), labels={list(bundle['labels'])}")


if __name__ == "__main__":
    main()
