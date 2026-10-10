"""Parity: numpy inference must match torch inference exactly."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chhat.parser.infer_numpy import NumpyAddressParser
from chhat.parser.predict import AddressParser

CASES = [
    "House 123, Street 4, Block C, Nishat Colony, Lahore",
    "H#45 St.No.3 DHA Phase 5 Lahore",
    "Plot 12, Main Boulevard, Gulberg III, Lahore",
    "house 7 street 2 block b johar town lahore",
    "Flat 301, Sector F, DHA Phase 6, Lahore",
    "مکان 123، گلی 4، نشاط کالونی، لاہور",
    "Shop 9, Main Market, Model Town, Lahore",
    "123-B, Street 10, Block D, NFC Society, Lahore",
]


class TestNumpyParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch_p = AddressParser()
        cls.np_p = NumpyAddressParser()

    def test_labels_identical(self):
        for text in CASES:
            a = self.torch_p.parse(text)
            b = self.np_p.parse(text)
            self.assertEqual(a["labels"], b["labels"], f"labels differ: {text}")
            for f in ("house", "street", "block", "colony", "city"):
                self.assertEqual(a[f], b[f], f"field {f} differs: {text}")


if __name__ == "__main__":
    unittest.main()
