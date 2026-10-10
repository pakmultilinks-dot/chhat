"""Inference for the Chhat Pakistani-address parser.

Loads the trained bundle and turns raw address text into structured fields:
house, street, block, colony, city.
"""
from __future__ import annotations

import os

import torch

from .model import AddressTagger
from .train import LABELS, PAD, UNK, spans_of, tokenize_text

DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "..", "..",
                             "models", "address_parser.pt")


class AddressParser:
    def __init__(self, model_path: str = DEFAULT_MODEL):
        bundle = torch.load(model_path, map_location="cpu", weights_only=False)
        cfg = bundle["config"]
        self.vocab: dict[str, int] = bundle["vocab"]
        self.labels: list[str] = bundle["labels"]
        self.model = AddressTagger(len(self.vocab), len(self.labels), **cfg)
        self.model.load_state_dict(bundle["state_dict"])
        self.model.eval()

    def parse(self, text: str) -> dict:
        tokens = tokenize_text(text)
        ids = torch.tensor([[self.vocab.get(t, self.vocab[UNK]) for t in tokens]])
        mask = torch.ones_like(ids, dtype=torch.float)
        with torch.no_grad():
            pred = self.model(ids, mask).argmax(-1)[0].tolist()
        labels = [self.labels[i] for i in pred]
        fields: dict[str, list[str]] = {}
        for start, end, tag in spans_of(labels):
            fields.setdefault(tag.lower(), []).append(" ".join(tokens[start:end]))
        return {
            "text": text,
            "tokens": tokens,
            "labels": labels,
            "house": " ".join(fields.get("house", [])),
            "street": " ".join(fields.get("street", [])),
            "block": " ".join(fields.get("block", [])),
            "colony": " ".join(fields.get("colony", [])),
            "city": " ".join(fields.get("city", [])),
        }

    def query_string(self, text: str) -> str:
        """Build a geocoder-friendly query from parsed components."""
        p = self.parse(text)
        bits = [b for b in [p["house"], p["street"], p["block"],
                            p["colony"], p["city"]] if b]
        return ", ".join(bits) if bits else text
