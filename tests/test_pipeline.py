"""Tests for layout optimizer and quote engine (no model needed)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chhat.layout.optimizer import optimize
from chhat.quote.engine import chhat_score, quote
from chhat.solar.client import building_insights


class TestPipeline(unittest.TestCase):
    def test_demo_roof_end_to_end(self):
        roof = building_insights(31.5204, 74.3587)  # Lahore, demo mode
        self.assertTrue(roof["demo"])
        lay = optimize(roof)
        self.assertGreater(lay["panel_count"], 0)
        self.assertGreater(lay["system_kw"], 0)
        q = quote(lay, annual_kwh_per_kw=roof["annual_kwh_per_kw"])
        self.assertGreater(q["equipment_cost_pkr"], 0)
        self.assertGreater(q["annual_savings_pkr"], 0)
        self.assertLess(q["payback_years"], 15)
        s = chhat_score(roof, lay, q)
        self.assertGreaterEqual(s["score"], 0)
        self.assertLessEqual(s["score"], 100)

    def test_quote_math(self):
        lay = {"panel_count": 10, "system_kw": 5.5, "usable_area_m2": 60}
        q = quote(lay, annual_kwh_per_kw=1550)
        # 10 panels * 42000 + 5.5kW * (45000+12000+8000) + 25000
        expect = 10 * 42000 + 5.5 * 65000 + 25000
        self.assertAlmostEqual(q["equipment_cost_pkr"], expect, delta=1)
        # savings: 70% self at 47.69, 30% export at 9.5
        gen = 5.5 * 1550
        expect_save = gen * 0.7 * 47.69 + gen * 0.3 * 9.5
        self.assertAlmostEqual(q["annual_savings_pkr"], expect_save, delta=1)


if __name__ == "__main__":
    unittest.main()
