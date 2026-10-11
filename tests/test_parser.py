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


# Wave 8: multi-city tricky cases — Karachi / Islamabad / Rawalpindi / Faisalabad
# real-world messy formats. 28 cases.
MULTICITY_TRICKY_CASES = [
    # --- Karachi ---
    ("House R-123, Block 13, Gulshan-e-Iqbal, Karachi",
     {"house": "R-123", "block": "13", "colony": "Gulshan-e-Iqbal", "city": "Karachi"}),
    ("Plot 45-C, Khayaban-e-Ittehad, DHA Phase 6, Karachi",
     {"house": "45-C", "street": "Khayaban-e-Ittehad",
      "colony": "DHA Phase 6", "city": "Karachi"}),
    ("House 5/12, North Nazimabad, Karachi",
     {"house": "5/12", "colony": "North Nazimabad", "city": "Karachi"}),
    ("House 23, Block 5, Clifton, Karachi",
     {"house": "23", "block": "5", "colony": "Clifton", "city": "Karachi"}),
    ("Flat 12, Block 3, Gulshan-e-Iqbal, Karachi",
     {"house": "12", "block": "3", "colony": "Gulshan-e-Iqbal", "city": "Karachi"}),
    ("H#R-123 Block 13 Gulshan-e-Iqbal Karachi",
     {"house": "R-123", "block": "13", "colony": "Gulshan-e-Iqbal", "city": "Karachi"}),
    ("Makan R-45 Block 7 Gulshan-e-Iqbal Karachi",
     {"house": "R-45", "block": "7", "colony": "Gulshan-e-Iqbal", "city": "Karachi"}),
    ("house r-123 block 13 gulshan-e-iqbal karachi",
     {"house": "r-123", "block": "13", "colony": "gulshan-e-iqbal", "city": "karachi"}),
    ("Plot 45-C Khayaban-e-Ittehad DHA Phase 6 Karachi",
     {"house": "45-C", "street": "Khayaban-e-Ittehad",
      "colony": "DHA Phase 6", "city": "Karachi"}),
    ("Flat 7, Block 2, North Nazimabad, Karachi",
     {"house": "7", "block": "2", "colony": "North Nazimabad", "city": "Karachi"}),
    # --- Islamabad ---
    ("House 123, Street 45, Sector G-13/2, Islamabad",
     {"house": "123", "street": "45", "block": "G-13/2", "city": "Islamabad"}),
    ("House 8, Street 12, F-10/3, Islamabad",
     {"house": "8", "street": "12", "colony": "F-10/3", "city": "Islamabad"}),
    ("Plot 22, E-11/4, Islamabad",
     {"house": "22", "colony": "E-11/4", "city": "Islamabad"}),
    ("House 56, Margalla Town, Islamabad",
     {"house": "56", "colony": "Margalla Town", "city": "Islamabad"}),
    ("H 123 St 45 G-13/2 Islamabad",
     {"house": "123", "street": "45", "colony": "G-13/2", "city": "Islamabad"}),
    ("Makan 77 Gali 3 E-11 Islamabad",
     {"house": "77", "street": "3", "colony": "E-11", "city": "Islamabad"}),
    ("House 123 Street 45 Sector G-13/2 Islamabad",
     {"house": "123", "street": "45", "block": "G-13/2", "city": "Islamabad"}),
    ("House 9, Street 3, E-11/4, Islamabad",
     {"house": "9", "street": "3", "colony": "E-11/4", "city": "Islamabad"}),
    # --- Rawalpindi ---
    ("House 234, Block C, Satellite Town, Rawalpindi",
     {"house": "234", "block": "C", "colony": "Satellite Town", "city": "Rawalpindi"}),
    ("House 56, Street 7, Bahria Town, Rawalpindi",
     {"house": "56", "street": "7", "colony": "Bahria Town", "city": "Rawalpindi"}),
    ("Plot 89, Chaklala Scheme III, Rawalpindi",
     {"house": "89", "colony": "Chaklala Scheme III", "city": "Rawalpindi"}),
    ("H#234 Block C Satellite Town Rawalpindi",
     {"house": "234", "block": "C", "colony": "Satellite Town", "city": "Rawalpindi"}),
    ("Shop 5, Main Road, Satellite Town, Rawalpindi",
     {"house": "5", "colony": "Satellite Town", "city": "Rawalpindi"}),
    # --- Faisalabad ---
    ("House 78, Block B, Madina Town, Faisalabad",
     {"house": "78", "block": "B", "colony": "Madina Town", "city": "Faisalabad"}),
    ("Plot 12, Block A, DHA, Faisalabad",
     {"house": "12", "block": "A", "colony": "DHA", "city": "Faisalabad"}),
    ("House 33, Street 5, Model City, Faisalabad",
     {"house": "33", "street": "5", "colony": "Model City", "city": "Faisalabad"}),
    ("Makan 78 Gali 5 Madina Town Faisalabad",
     {"house": "78", "street": "5", "colony": "Madina Town", "city": "Faisalabad"}),
    ("House 100, Wadi-e-Sitara, Faisalabad",
     {"house": "100", "colony": "Wadi-e-Sitara", "city": "Faisalabad"}),
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

    def test_multicity_cases(self):
        self._check(MULTICITY_TRICKY_CASES)


if __name__ == "__main__":
    unittest.main()
