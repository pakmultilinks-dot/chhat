"""Tests for the Chhat address parser (run after training)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "address_parser.pt")

CASES = [
    # (input, expected fields)
    ("House 123, Street 4, Block C, Nishat Colony, Lahore",
     {"house": "123", "street": "4", "block": "C",
      "colony": "Nishat Colony", "city": "Lahore"}),
    ("45-B, Main Boulevard, DHA Phase 5, Lahore",
     {"house": "45-B", "colony": "DHA Phase 5", "city": "Lahore"}),
    ("House No. 78, Street No. 9, Model Town, Lahore",
     {"house": "78", "street": "9", "colony": "Model Town", "city": "Lahore"}),
    ("Plot 12, Block H, Johar Town, Lahore",
     {"house": "12", "block": "H", "colony": "Johar Town", "city": "Lahore"}),
    ("Flat 7, Building 3, Gulberg III, Lahore",
     {"house": "7", "colony": "Gulberg III", "city": "Lahore"}),
]


class TestParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(MODEL):
            raise unittest.SkipTest("model not trained yet")
        from chhat.parser.predict import AddressParser
        cls.p = AddressParser(MODEL)

    def test_cases(self):
        for text, want in CASES:
            with self.subTest(text=text):
                got = self.p.parse(text)
                for k, v in want.items():
                    self.assertEqual(got[k], v, f"{k}: {got[k]!r} != {v!r}")


if __name__ == "__main__":
    unittest.main()
