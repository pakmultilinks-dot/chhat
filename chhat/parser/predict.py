"""Inference for the Chhat Pakistani-address parser.

Loads the trained bundle and turns raw address text into structured fields:
house, street, block, colony, city.
"""
from __future__ import annotations

import os
import re

import torch

from .model import AddressTagger
from .train import LABELS, PAD, UNK, spans_of, tokenize_text

DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "..", "..",
                             "models", "address_parser.pt")

# Descriptor prefixes people glue onto house/street numbers ("H#123",
# "St.No.3"). Stripped after tagging so UI + geocoder see clean values.
# NOTE: longest alternatives first — "street#?" must precede "st#?".
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
            "house": _clean(" ".join(fields.get("house", [])), "house"),
            "street": _clean(" ".join(fields.get("street", [])), "street"),
            "block": _clean(" ".join(fields.get("block", [])), "block"),
            "colony": " ".join(fields.get("colony", [])),
            "city": " ".join(fields.get("city", [])),
        }

    def query_string(self, text: str) -> str:
        """Build a geocoder-friendly query from parsed components."""
        p = self.parse(text)
        bits = [b for b in [p["house"], p["street"], p["block"],
                            p["colony"], p["city"]] if b]
        return ", ".join(bits) if bits else text


_PARSER: AddressParser | None = None


def parse_address(text: str, model_path: str = DEFAULT_MODEL) -> dict:
    """Parse a raw Pakistani address into structured fields.

    Simple one-call API: parse_address("H#123 St 4 Nishat Colony Lahore")
    -> {"house": "123", "street": "4", "block": "", "colony": "Nishat Colony",
        "city": "Lahore", ...}
    """
    global _PARSER
    if _PARSER is None:
        _PARSER = AddressParser(model_path)
    return _PARSER.parse(text)
