"""Pure-numpy inference for the Chhat address parser — no torch required.

Loads weights exported by scripts/export_numpy_weights.py and runs the exact
same math as AddressTagger (embedding -> 2-layer BiLSTM -> linear ->
argmax). Used by the serverless deployment where torch is too heavy.

Verified parity: outputs identical labels to the torch model on the test
suite (see tests/test_numpy_parity.py).
"""
from __future__ import annotations

import os
import re

import numpy as np

DEFAULT_NPZ = os.path.join(os.path.dirname(__file__), "..", "..",
                          "models", "address_parser.npz")
# Fallback: download weights from the public repo on first run (serverless
# deployments ship code without the 3.5MB weight file).
DEFAULT_MODEL_URL = os.environ.get(
    "CHHAT_MODEL_URL",
    "https://raw.githubusercontent.com/pakmultilinks-dot/chhat"
    "/dev/satellite-eye/models/address_parser.npz",
)

# --- tiny text utils (duplicated from train.py to avoid importing torch) ---
PAD, UNK = "<pad>", "<unk>"


def tokenize_text(text: str) -> list[str]:
    return text.replace(",", " ,").split()


def spans_of(labels: list[str]) -> set[tuple[int, int, str]]:
    spans: set[tuple[int, int, str]] = set()
    start: int | None = None
    cur: str | None = None
    for i, lab in enumerate(labels):
        if lab.startswith("B-"):
            if start is not None and cur is not None:
                spans.add((start, i, cur))
            start, cur = i, lab[2:]
        elif lab.startswith("I-") and start is not None and cur == lab[2:]:
            continue
        else:
            if start is not None and cur is not None:
                spans.add((start, i, cur))
            start, cur = None, None
    if start is not None and cur is not None:
        spans.add((start, len(labels), cur))
    return spans


_HOUSE_PRE = re.compile(r"^(house#?|h\.?no\.?|h#|plot#?|flat#?|shop#?|p#)\s*", re.I)
_STREET_PRE = re.compile(r"^(street#?|st\.?\s*no\.?|st#?|gali#?|rd#?|no\.?)\s*", re.I)
_EDGE_JUNK = re.compile(r"^[#.\s]+|[#.\s]+$")


def _clean(value: str, kind: str) -> str:
    v = value.strip()
    if kind == "house":
        v = _HOUSE_PRE.sub("", v)
    elif kind == "street":
        v = _STREET_PRE.sub("", v)
    return _EDGE_JUNK.sub("", v)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _lstm_forward(x: np.ndarray, w_ih: np.ndarray, w_hh: np.ndarray,
                  b_ih: np.ndarray, b_hh: np.ndarray) -> np.ndarray:
    """Single-direction LSTM over (T, in) -> (T, hidden). Gate order i,f,g,o."""
    t, _ = x.shape
    h_dim = w_hh.shape[1]
    h = np.zeros(h_dim)
    c = np.zeros(h_dim)
    out = np.empty((t, h_dim))
    for i in range(t):
        gates = x[i] @ w_ih.T + b_ih + h @ w_hh.T + b_hh
        ig, fg, gg, og = np.split(gates, 4)
        ig, fg, og = _sigmoid(ig), _sigmoid(fg), _sigmoid(og)
        gg = np.tanh(gg)
        c = fg * c + ig * gg
        h = og * np.tanh(c)
        out[i] = h
    return out


def _ensure_weights(npz_path: str) -> str:
    """Return a local .npz path, downloading from the public repo if missing.

    Serverless deployments exclude the 3.5MB weight file; on first run it is
    fetched once to a writable cache (/tmp on Vercel, repo models/ locally).
    """
    if os.path.exists(npz_path):
        return npz_path
    import urllib.request

    for cache_dir in ("/tmp", os.path.dirname(npz_path)):
        try:
            os.makedirs(cache_dir, exist_ok=True)
            dest = os.path.join(cache_dir, "address_parser.npz")
            if os.path.exists(dest):
                return dest
            req = urllib.request.Request(DEFAULT_MODEL_URL,
                                         headers={"User-Agent": "chhat/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r, \
                    open(dest, "wb") as f:
                f.write(r.read())
            return dest
        except Exception:
            continue
    raise FileNotFoundError(
        f"address parser weights not found at {npz_path} and download failed; "
        "set CHHAT_MODEL_URL or ship models/address_parser.npz")


class NumpyAddressParser:
    def __init__(self, npz_path: str = DEFAULT_NPZ):
        npz_path = _ensure_weights(npz_path)
        z = np.load(npz_path, allow_pickle=False)
        self.vocab = dict(zip(z["vocab_keys"].tolist(), z["vocab_vals"].tolist()))
        self.labels = z["labels"].tolist()
        self.emb = z["emb"]
        self.cls_w = z["cls_w"]
        self.cls_b = z["cls_b"]
        self.n_layers = int(z["num_layers"])
        self.lstm_w: dict[str, np.ndarray] = {}
        for l in range(self.n_layers):
            for d in ("", "_reverse"):
                for w in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    self.lstm_w[f"{l}{d}_{w}"] = z[f"lstm_l{l}{d}_{w}"]

    def _forward(self, ids: list[int]) -> np.ndarray:
        x = self.emb[np.array(ids)]  # (T, E)
        for l in range(self.n_layers):
            fwd = _lstm_forward(
                x, self.lstm_w[f"{l}_weight_ih"], self.lstm_w[f"{l}_weight_hh"],
                self.lstm_w[f"{l}_bias_ih"], self.lstm_w[f"{l}_bias_hh"])
            rev = _lstm_forward(
                x[::-1], self.lstm_w[f"{l}_reverse_weight_ih"],
                self.lstm_w[f"{l}_reverse_weight_hh"],
                self.lstm_w[f"{l}_reverse_bias_ih"],
                self.lstm_w[f"{l}_reverse_bias_hh"])[::-1]
            x = np.concatenate([fwd, rev], axis=1)
        return x @ self.cls_w.T + self.cls_b  # (T, n_labels)

    def parse(self, text: str) -> dict:
        tokens = tokenize_text(text)
        ids = [self.vocab.get(t, self.vocab[UNK]) for t in tokens]
        logits = self._forward(ids)
        pred = list(np.argmax(logits, axis=1))
        labels = [self.labels[i] for i in pred]
        fields: dict[str, list[str]] = {}
        for start, end, tag in spans_of(labels):
            fields.setdefault(tag.lower(), []).append(" ".join(tokens[start:end]))
        return {
            "text": text,
            "tokens": tokens,
            "labels": labels,
            "house": _clean(" ".join(fields.get("house", [])), "house"),
            "street": _clean(" ".join(fields.get("street", [])), "street"),
            "block": _clean(" ".join(fields.get("block", [])), "block"),
            "colony": " ".join(fields.get("colony", [])),
            "city": " ".join(fields.get("city", [])),
        }

    def query_string(self, text: str) -> str:
        p = self.parse(text)
        bits = [b for b in [p["house"], p["street"], p["block"],
                            p["colony"], p["city"]] if b]
        return ", ".join(bits) if bits else text
