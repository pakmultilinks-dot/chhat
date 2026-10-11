"""Chhat roofseg: OUR OWN rooftop segmentation model.

Trained from scratch on open Lahore data (see docs/datasets.md) — no borrowed
brains. Every prediction carries a confidence score; low-confidence outputs
are gated to "UNSURE" instead of a fake mask. That is the anti-hallucination
mechanism: the model is allowed to say "I don't know".
"""
from .predict import predict, load_model

__all__ = ["predict", "load_model"]
