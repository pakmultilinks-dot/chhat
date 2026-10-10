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

# Messy real-world variants: missing commas, hash prefixes, Urdu-mixed
# descriptors, abbreviations, lowercase, landmarks, trailing country.
TRICKY_CASES = [
    ("H#123 St 4 Nishat Colony Lahore",
     {"house": "123", "street": "4", "colony": "Nishat Colony", "city": "Lahore"}),
    ("Makan 123 Gali 4 Nishat Colony Lahore",
     {"house": "123", "street": "4", "colony": "Nishat Colony", "city": "Lahore"}),
    ("house 123 street 4 block c nishat colony lahore",
     {"house": "123", "street": "4", "block": "c",
      "colony": "nishat colony", "city": "lahore"}),
    ("House 123 Street 4 Block C Nishat Colony Lahore",
     {"house": "123", "street": "4", "block": "C",
      "colony": "Nishat Colony", "city": "Lahore"}),
    ("Plot No 12 Block H Johar Town Lahore",
     {"house": "12", "block": "H", "colony": "Johar Town", "city": "Lahore"}),
    ("H.No. 45-B, St. No. 9, Model Town, Lahore",
     {"house": "45-B", "street": "9", "colony": "Model Town", "city": "Lahore"}),
    ("123/4 Nishat Colony Lahore",
     {"house": "123/4", "colony": "Nishat Colony", "city": "Lahore"}),
    ("Near Main Bazaar House 45 Street 2 Ichhra Lahore Pakistan",
     {"house": "45", "street": "2", "colony": "Ichhra", "city": "Lahore"}),
    ("Flat 7 Building 3 Gulberg III Lahore",
     {"house": "7", "colony": "Gulberg III", "city": "Lahore"}),
    ("12-B, St 7, Sector G-13, Islamabad",
     {"house": "12-B", "street": "7", "block": "G-13", "city": "Islamabad"}),
    ("H 45, St. 12, Blk C, DHA Phase 5, Lahore",
     {"house": "45", "street": "12", "block": "C",
      "colony": "DHA Phase 5", "city": "Lahore"}),
    ("Makaan No 78 Mohallah Samanabad Lahore",
     {"house": "78", "colony": "Samanabad", "city": "Lahore"}),
    ("P#12 St#4 Baghbanpura Lahore",
     {"house": "12", "street": "4", "colony": "Baghbanpura", "city": "Lahore"}),
    ("House#123/4 Street#59 Nishat Colony Lahore",
     {"house": "123/4", "street": "59", "colony": "Nishat Colony", "city": "Lahore"}),
    ("45, Main Boulevard, DHA Phase 5, Lahore",
     {"house": "45", "colony": "DHA Phase 5", "city": "Lahore"}),
    ("Shop 12, Main Road, Ichhra, Lahore",
     {"house": "12", "colony": "Ichhra", "city": "Lahore"}),
    ("H.No.7 St.No.3 Block B Valencia Town Lahore",
     {"house": "7", "street": "3", "block": "B",
      "colony": "Valencia Town", "city": "Lahore"}),
    ("Gali 10 Makan 55 Township Lahore",
     {"house": "55", "street": "10", "colony": "Township", "city": "Lahore"}),
    ("Plot 100, Block J, EME Society, Lahore",
     {"house": "100", "block": "J", "colony": "EME Society", "city": "Lahore"}),
    ("house no 9 street no 11 shadman lahore",
     {"house": "9", "street": "11", "colony": "shadman", "city": "lahore"}),
    ("Flat 5, Building 2, Bahria Town, Karachi",
     {"house": "5", "block": "2", "colony": "Bahria Town", "city": "Karachi"}),
    ("H# 200 St # 11 Green Town Lahore",
     {"house": "200", "street": "11", "colony": "Green Town", "city": "Lahore"}),
    ("Makan 123, Gali No 4, Nishat Colony, Lahore",
     {"house": "123", "street": "4", "colony": "Nishat Colony", "city": "Lahore"}),
    ("31-C, Street 8, Cavalry Ground, Lahore",
     {"house": "31-C", "street": "8", "colony": "Cavalry Ground", "city": "Lahore"}),
]


class TestParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(MODEL):
            raise unittest.SkipTest("model not trained yet")
        from chhat.parser.predict import AddressParser
        cls.p = AddressParser(MODEL)

    def _check(self, cases):
        for text, want in cases:
            with self.subTest(text=text):
                got = self.p.parse(text)
                for k, v in want.items():
                    self.assertEqual(got[k], v, f"{k}: {got[k]!r} != {v!r}")

    def test_cases(self):
        self._check(CASES)

    def test_tricky_cases(self):
        self._check(TRICKY_CASES)


if __name__ == "__main__":
    unittest.main()
