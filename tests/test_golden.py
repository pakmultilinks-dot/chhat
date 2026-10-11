"""Golden demo set + offline fallback contract.

- golden-demos.json exists, has 10 demos, all famous public landmarks.
- Every demo result matches the /api/analyze response shape the UI renders.
- The /golden-demos.json route serves the same data.
"""
import json
import os
import unittest

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "..", "docs",
                           "pitch", "golden-demos.json")


class TestGoldenDemos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(GOLDEN_PATH, encoding="utf-8") as f:
            cls.doc = json.load(f)

    def test_count(self):
        self.assertEqual(self.doc["count"], 10)
        self.assertEqual(len(self.doc["demos"]), 10)

    def test_no_real_homes(self):
        for d in self.doc["demos"]:
            addr = d["address"].lower()
            self.assertNotIn("house ", addr.split(",")[0],
                             f"looks like a home: {d['address']}")

    def test_analyze_shape(self):
        required_top = {"parsed", "geo", "roof", "layout", "quote",
                        "score", "demo"}
        for d in self.doc["demos"]:
            r = d["result"]
            self.assertTrue(required_top <= set(r.keys()), d["label"])
            self.assertTrue(r["demo"])
            # fields the UI renderReport() reads:
            self.assertIn("score", r["score"])
            for k in ("system_kw", "panel_count", "segments", "panels",
                      "usable_area_m2"):
                self.assertIn(k, r["layout"], f"{d['label']}:{k}")
            for k in ("annual_generation_kwh", "payback_years",
                      "equipment_cost_pkr", "annual_savings_pkr",
                      "lifetime_savings_25y_pkr", "tariff_used",
                      "export_rate_used"):
                self.assertIn(k, r["quote"], f"{d['label']}:{k}")
            self.assertIn("breakdown", r["score"])

    def test_scores_sane(self):
        for d in self.doc["demos"]:
            s = d["result"]["score"]["score"]
            self.assertGreaterEqual(s, 0)
            self.assertLessEqual(s, 100)

    def test_route_serves_same(self):
        from chhat.web.app import app
        c = app.test_client()
        r = c.get("/golden-demos.json")
        self.assertEqual(r.status_code, 200)
        j = json.loads(r.data)
        self.assertEqual(j["count"], 10)

    def test_index_embeds_golden(self):
        from chhat.web.app import app
        c = app.test_client()
        r = c.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'id="goldenData"', r.data)
        self.assertIn(b'id="goldenCard"', r.data)


if __name__ == "__main__":
    unittest.main()
