"""Inference + confidence gate for the Chhat rooftop segmenter.

predict(image) -> dict with:
  - mask: binary rooftop mask (H x W, uint8) or None when unsure
  - confidence: mean foreground probability (0..1)
  - status: "OK" | "UNSURE" | "NO_MODEL"
  - building_fraction: fraction of pixels predicted as rooftop

Anti-hallucination rule: if confidence < CONF_THRESHOLD, we return
status="UNSURE" with mask=None instead of a fake mask. The UI must show
"needs human review" in that case — never a confident-looking guess.
"""
import os

import cv2
import numpy as np
import torch

from .model import TinyUNet

CONF_THRESHOLD = 0.55
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "rooftop_seg.pt")

_model = None


def load_model(path=MODEL_PATH):
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(path):
        return None
    m = TinyUNet()
    ckpt = torch.load(path, map_location="cpu")
    m.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt)
    m.eval()
    _model = m
    return m


def predict(image, threshold=CONF_THRESHOLD):
    """image: BGR numpy array (any size). Returns dict (see module docstring)."""
    model = load_model()
    if model is None:
        return {"mask": None, "confidence": 0.0, "status": "NO_MODEL", "building_fraction": 0.0}
    h, w = image.shape[:2]
    inp = cv2.resize(image, (256, 256)).astype(np.float32) / 255.0
    inp = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(inp)).squeeze().numpy()
    prob_full = cv2.resize(prob, (w, h))
    bin_mask = (prob_full >= 0.5).astype(np.uint8)
    fg = bin_mask == 1
    confidence = float(prob_full[fg].mean()) if fg.any() else 0.0
    if confidence < threshold or not fg.any():
        return {"mask": None, "confidence": confidence,
                "status": "UNSURE", "building_fraction": float(fg.mean())}
    return {"mask": (bin_mask * 255).astype(np.uint8), "confidence": confidence,
            "status": "OK", "building_fraction": float(fg.mean())}
