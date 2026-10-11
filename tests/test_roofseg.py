"""Tests for the dual-model rooftop inference + anti-hallucination gate.

Honesty contract: blank/uniform frames and absurd masks must return
UNSURE with no mask — never a confident-looking guess.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chhat.roofseg.predict import predict, predict_best


class TestRoofsegHonesty(unittest.TestCase):
    def test_blank_image_is_unsure(self):
        blank = np.zeros((256, 256, 3), np.uint8)
        for mid in ("v2", "v3"):
            r = predict(blank, model_id=mid)
            self.assertEqual(r["status"], "UNSURE")
            self.assertIsNone(r["mask"])

    def test_uniform_gray_is_unsure(self):
        gray = np.full((256, 256, 3), 128, np.uint8)
        r = predict_best(gray)
        self.assertEqual(r["status"], "UNSURE")
        self.assertIsNone(r["mask"])

    def test_predict_best_returns_model_id(self):
        rng = np.random.RandomState(0)
        noise = rng.randint(0, 255, (256, 256, 3)).astype(np.uint8)
        r = predict_best(noise)
        self.assertIn(r["status"], ("OK", "UNSURE"))
        self.assertIn(r["model"], ("v2", "v3", None))
        if r["status"] == "UNSURE":
            self.assertIsNone(r["mask"])
        if r["status"] == "OK":
            self.assertLessEqual(r["building_fraction"], 0.75)


if __name__ == "__main__":
    unittest.main()
