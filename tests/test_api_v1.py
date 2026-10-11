"""Tests for the Chhat v1 API, middleware, and LLM client.

The LLM HTTP calls are MOCKED (no real network). Live LLM calls need
OPENROUTER_API_KEY set on the deployed host; this sandbox cannot carry
Authorization headers on egress, so live calls are not tested here.
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chhat.web import middleware
from chhat.llm import mode as llm_mode
from chhat.web.app import app


def _no_llm_key():
    """Context manager: environment without OPENROUTER_API_KEY."""
    env = dict(os.environ)
    env.pop("OPENROUTER_API_KEY", None)
    return mock.patch.dict(os.environ, env, clear=True)


class ApiV1Test(unittest.TestCase):
    def setUp(self):
        middleware.reset_rate_limit()
        llm_mode.reset_cache()
        self._saved_key = os.environ.pop("OPENROUTER_API_KEY", None)
        self.client = app.test_client()

    def tearDown(self):
        if self._saved_key is not None:
            os.environ["OPENROUTER_API_KEY"] = self._saved_key
        llm_mode.reset_cache()

    # ---------- health ----------
    def test_health(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertEqual(j["status"], "ok")
        self.assertIn("version", j)
        self.assertIn("timestamp", j)
        self.assertIn("parser", j["models_available"])
        self.assertIn("llm_configured", j)
        self.assertEqual(j["llm_mode"], "off")  # no key in test env

    def test_health_mode_server(self):
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        fake = mock.Mock(status_code=200)
        with mock.patch("chhat.llm.mode.requests.get", return_value=fake):
            r = self.client.get("/api/health")
        self.assertEqual(r.get_json()["llm_mode"], "server")

    def test_health_mode_direct_on_network_failure(self):
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        import requests as _rq
        with mock.patch("chhat.llm.mode.requests.get",
                        side_effect=_rq.ConnectionError("blocked")):
            r = self.client.get("/api/health")
        self.assertEqual(r.get_json()["llm_mode"], "direct")

    def test_mode_cached(self):
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        fake = mock.Mock(status_code=200)
        with mock.patch("chhat.llm.mode.requests.get",
                        return_value=fake) as g:
            self.assertEqual(llm_mode.get_mode(), "server")
            self.assertEqual(llm_mode.get_mode(), "server")
            self.assertEqual(g.call_count, 1)  # second call served from cache

    # ---------- page injects transport config ----------
    def test_index_injects_llm_config_off(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        html = r.get_data(as_text=True)
        self.assertIn('mode: "off"', html)
        self.assertNotIn("key:", html)  # key never rendered when off

    def test_index_direct_mode_injects_key(self):
        os.environ["OPENROUTER_API_KEY"] = "sk-test-direct-key"
        import requests as _rq
        with mock.patch("chhat.llm.mode.requests.get",
                        side_effect=_rq.ConnectionError("stripped")):
            r = self.client.get("/")
        html = r.get_data(as_text=True)
        self.assertIn('mode: "direct"', html)
        self.assertIn("sk-test-direct-key", html)  # preview-only exposure

    def test_index_server_mode_hides_key(self):
        os.environ["OPENROUTER_API_KEY"] = "sk-test-server-key"
        fake = mock.Mock(status_code=200)
        with mock.patch("chhat.llm.mode.requests.get",
                        return_value=fake):
            r = self.client.get("/")
        html = r.get_data(as_text=True)
        self.assertIn('mode: "server"', html)
        self.assertNotIn("sk-test-server-key", html)  # key stays secret

    # ---------- models ----------
    def test_models(self):
        r = self.client.get("/api/v1/models")
        self.assertEqual(r.status_code, 200)
        models = r.get_json()["models"]
        ids = [m["id"] for m in models]
        self.assertEqual(ids, ["address-parser", "roofseg-v2", "roofseg-v3"])
        v2 = next(m for m in models if m["id"] == "roofseg-v2")
        self.assertEqual(v2["metrics"]["test_iou"], 0.3604)
        v3 = next(m for m in models if m["id"] == "roofseg-v3")
        self.assertEqual(v3["metrics"]["dense_test_iou"], 0.3846)
        for m in models:
            self.assertTrue(m["trained_from_scratch"])

    # ---------- demos ----------
    def test_demos(self):
        r = self.client.get("/api/v1/demos")
        self.assertEqual(r.status_code, 200)
        self.assertIn("demos", r.get_json())

    def test_golden_demos_still_works(self):
        r = self.client.get("/golden-demos.json")
        self.assertEqual(r.status_code, 200)

    # ---------- docs ----------
    def test_docs(self):
        r = self.client.get("/api/docs")
        self.assertEqual(r.status_code, 200)
        html = r.get_data(as_text=True)
        self.assertIn("Chhat API", html)
        self.assertIn("direct", html)  # hybrid transport tradeoff documented

    def test_explain_button_on_page(self):
        r = self.client.get("/")
        html = r.get_data(as_text=True)
        self.assertIn('id="explainBtn"', html)
        self.assertIn('id="explainBox"', html)

    # ---------- explain ----------
    def test_explain_no_key_503(self):
        with _no_llm_key():
            r = self.client.post("/api/v1/explain",
                                 json={"report": {"score": {"score": 80}}, "lang": "en"})
        self.assertEqual(r.status_code, 503)
        self.assertIn("error", r.get_json())

    def test_explain_mocked(self):
        with mock.patch("chhat.llm.client.explain_report",
                        return_value=("Simple words.", "mock-model")):
            r = self.client.post("/api/v1/explain",
                                 json={"report": {"score": {"score": 80}}, "lang": "ur"})
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertEqual(j["explanation"], "Simple words.")
        self.assertEqual(j["model_used"], "mock-model")

    def test_explain_bad_report(self):
        r = self.client.post("/api/v1/explain", json={"lang": "en"})
        self.assertEqual(r.status_code, 400)

    def test_explain_bad_lang(self):
        r = self.client.post("/api/v1/explain",
                             json={"report": {"score": {}}, "lang": "xx"})
        self.assertEqual(r.status_code, 400)

    # ---------- ask ----------
    def test_ask_no_key_503(self):
        with _no_llm_key():
            r = self.client.post("/api/v1/ask",
                                 json={"question": "What is net billing?"})
        self.assertEqual(r.status_code, 503)

    def test_ask_mocked(self):
        with mock.patch("chhat.llm.client.answer_question",
                        return_value=("Net billing pays less.", "mock-model")):
            r = self.client.post("/api/v1/ask",
                                 json={"question": "What is net billing?",
                                       "report": {"score": {"score": 70}}})
        self.assertEqual(r.status_code, 200)
        self.assertIn("answer", r.get_json())

    def test_ask_missing_question(self):
        r = self.client.post("/api/v1/ask", json={"lang": "en"})
        self.assertEqual(r.status_code, 400)

    def test_ask_too_long(self):
        r = self.client.post("/api/v1/ask", json={"question": "x" * 2001})
        self.assertEqual(r.status_code, 400)

    # ---------- validation hardening ----------
    def test_parse_too_long(self):
        r = self.client.post("/api/parse", json={"address": "x" * 501})
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.get_json())

    def test_parse_still_works(self):
        r = self.client.post("/api/parse", json={
            "address": "House 123, Street 4, Block C, Nishat Colony, Lahore"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["house"], "123")

    def test_roofseg_bad_coords(self):
        r = self.client.post("/api/roofseg", json={"lat": 999, "lng": 74.3})
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/roofseg", json={"lat": "abc", "lng": 74.3})
        self.assertEqual(r.status_code, 400)

    def test_analyze_bad_coords(self):
        r = self.client.post("/api/analyze", json={
            "address": "House 123, Street 4, Nishat Colony, Lahore",
            "lat": 31.5, "lng": 999})
        self.assertEqual(r.status_code, 400)

    # ---------- rate limiting ----------
    def test_rate_limit(self):
        for _ in range(60):
            r = self.client.get("/api/health")
            self.assertEqual(r.status_code, 200)
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 429)
        self.assertIn("Retry-After", r.headers)
        self.assertIn("error", r.get_json())

    # ---------- error envelope ----------
    def test_api_404_envelope(self):
        r = self.client.get("/api/nope")
        self.assertEqual(r.status_code, 404)
        self.assertIn("error", r.get_json())


class LLMClientTest(unittest.TestCase):
    def test_not_configured_raises(self):
        from chhat.llm import client as llm
        with _no_llm_key():
            self.assertFalse(llm.is_configured())
            with self.assertRaises(llm.LLMNotConfigured):
                llm.complete([{"role": "user", "content": "hi"}])

    def test_default_model(self):
        from chhat.llm import client as llm
        with _no_llm_key():
            self.assertEqual(llm.default_model(), "claude-haiku-5.5:free")

    def test_explain_report_no_key(self):
        from chhat.llm import client as llm
        with _no_llm_key():
            with self.assertRaises(llm.LLMNotConfigured):
                llm.explain_report({"score": {"score": 80}}, "en")


if __name__ == "__main__":
    unittest.main()
