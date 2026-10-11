"""Inference + confidence gate for the Chhat rooftop segmenters.

Two specialists, both trained from scratch:
  - v2 ("generalist"): TinyUNet on 676 chips, 3 planned Lahore regions
  - v3 ("dense specialist"): fine-tuned on 79 human-verified dense-street
    chips (Nishat Colony / Lahore Cantt)

predict(image, model_id) -> dict with mask, confidence, status, model.
predict_best(image) runs both and keeps the higher-confidence OK result;
  if both are UNSURE it returns UNSURE with no mask.

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
DENSE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..",
                                "models", "rooftop_seg_dense.pt")

# model_id -> loaded torch module (lazy, one per weights file)
_models: dict = {}

MODEL_LABELS = {
    "v2": "v2 generalist (planned + dense areas)",
    "v3": "v3 dense-street specialist (Nishat Colony / Lahore Cantt)",
}

MODEL_PATHS = {
    "v2": MODEL_PATH,
    "v3": DENSE_MODEL_PATH,
}


def load_model(path=MODEL_PATH, model_id="v2"):
    if model_id in _models:
        return _models[model_id]
    if not os.path.exists(path):
        return None
    m = TinyUNet()
    ckpt = torch.load(path, map_location="cpu")
    m.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt)
    m.eval()
    _models[model_id] = m
    return m


def _run(model, image):
    """Run one loaded model on a BGR image. Returns (prob_full, bin_mask)."""
    h, w = image.shape[:2]
    inp = cv2.resize(image, (256, 256)).astype(np.float32) / 255.0
    inp = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(inp)).squeeze().numpy()
    prob_full = cv2.resize(prob, (w, h))
    bin_mask = (prob_full >= 0.5).astype(np.uint8)
    return prob_full, bin_mask


def predict(image, threshold=CONF_THRESHOLD, model_id="v2"):
    """image: BGR numpy array (any size). model_id: "v2" or "v3".

    Returns dict with mask, confidence, status, building_fraction,
    plus "model" (which weights were used).
    """
    # Sanity: a blank/uniform frame is not real imagery (failed fetch) —
    # never hallucinate roofs on it.
    if image is None or image.size == 0 or float(image.std()) < 2.0:
        return {"mask": None, "confidence": 0.0, "status": "UNSURE",
                "building_fraction": 0.0, "model": model_id}
    model = load_model(MODEL_PATHS[model_id], model_id)
    if model is None:
        return {"mask": None, "confidence": 0.0, "status": "NO_MODEL",
                "building_fraction": 0.0, "model": model_id}
    prob_full, bin_mask = _run(model, image)
    fg = bin_mask == 1
    frac = float(fg.mean())
    confidence = float(prob_full[fg].mean()) if fg.any() else 0.0
    # Absurdity cap: no real 150m rooftop chip is >75% roof. Past that the
    # model is hallucinating on out-of-distribution input — refuse to guess.
    if confidence < threshold or not fg.any() or frac > 0.75:
        return {"mask": None, "confidence": confidence,
                "status": "UNSURE", "building_fraction": frac,
                "model": model_id}
    return {"mask": (bin_mask * 255).astype(np.uint8), "confidence": confidence,
            "status": "OK", "building_fraction": frac,
            "model": model_id}


def predict_best(image, threshold=CONF_THRESHOLD):
    """Run both specialists; keep the higher-confidence OK result.

    v2 = generalist, v3 = dense-street specialist. If both are UNSURE (or
    no weights exist), return UNSURE with no mask — the UI must show
    "needs human review", never a fake confident overlay.
    """
    results = []
    for mid in ("v2", "v3"):
        r = predict(image, threshold=threshold, model_id=mid)
        if r["status"] != "NO_MODEL":
            results.append(r)
    if not results:
        return {"mask": None, "confidence": 0.0, "status": "NO_MODEL",
                "building_fraction": 0.0, "model": None}
    ok = [r for r in results if r["status"] == "OK"]
    if ok:
        best = max(ok, key=lambda r: r["confidence"])
        return best
    # both unsure: report the more confident one, still UNSURE, no mask
    best = max(results, key=lambda r: r["confidence"])
    best["mask"] = None
    best["status"] = "UNSURE"
    return best
