"""Chhat versioned API (v1).

Backwards compat: every original route (/, /api/parse, /api/analyze,
/api/requote, /api/roofseg, /golden-demos.json) keeps working unchanged.

New in v1:
  GET  /api/health        service + model availability
  GET  /api/v1/models     our own trained models (cards, metrics)
  GET  /api/v1/demos      golden demo list
  POST /api/v1/explain    LLM explains a report in homeowner words
  POST /api/v1/ask        LLM answers solar questions, grounded in a report
  GET  /api/docs          human-readable endpoint documentation
"""
from __future__ import annotations

import datetime as _dt
import os

from flask import Blueprint, jsonify, request

bp = Blueprint("api_v1", __name__)

VERSION = "1.0.0"

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MODEL_FILES = {
    "parser": os.path.join(_REPO_ROOT, "models", "address_parser.pt"),
    "roofseg_v2": os.path.join(_REPO_ROOT, "models", "rooftop_seg.pt"),
    "roofseg_v3": os.path.join(_REPO_ROOT, "models", "rooftop_seg_dense.pt"),
}

MODEL_CARDS = [
    {
        "id": "address-parser",
        "name": "Chhat address parser",
        "task": "Pakistani address entity extraction",
        "architecture": "BiLSTM (trained from scratch)",
        "metrics": {
            "entity_f1": 0.9999,
            "token_accuracy": 0.99995,
            "n_train": 10000,
            "n_test": 2000,
            "tricky_real_world_cases": 57,
        },
        "trained_from_scratch": True,
        "note": "Parses house/street/block/colony/city from messy Pakistani addresses.",
    },
    {
        "id": "roofseg-v2",
        "name": "v2 generalist rooftop segmenter",
        "task": "rooftop segmentation on aerial imagery",
        "architecture": "TinyUNet (trained from scratch)",
        "metrics": {"test_iou": 0.3604, "test_dice": 0.4562},
        "confidence_threshold": 0.55,
        "trained_from_scratch": True,
        "note": "Honesty gate: below 55% confidence it says UNSURE and shows no mask.",
    },
    {
        "id": "roofseg-v3",
        "name": "v3 dense-street specialist",
        "task": "rooftop segmentation in dense wall-to-wall streets",
        "architecture": "TinyUNet, fine-tuned on 79 human-verified dense chips",
        "metrics": {"dense_test_iou": 0.3846, "dense_test_dice": 0.3998},
        "confidence_threshold": 0.55,
        "trained_from_scratch": True,
        "note": "Specialist for Nishat Colony / Lahore Cantt style dense streets.",
    },
]


@bp.get("/api/health")
def health():
    """Service health + which model weights are available on this host."""
    from chhat.llm.mode import get_mode
    return jsonify({
        "status": "ok",
        "version": VERSION,
        "models_available": {k: os.path.exists(p)
                              for k, p in MODEL_FILES.items()},
        "llm_configured": bool(os.environ.get("OPENROUTER_API_KEY")),
        "llm_mode": get_mode(),
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    })


@bp.get("/api/v1/models")
def models():
    """Cards for our own trained models: task, metrics, thresholds."""
    return jsonify({"models": MODEL_CARDS})


@bp.get("/api/v1/demos")
def demos():
    """Golden demo list (same data as /golden-demos.json)."""
    from chhat.web.app import _golden_json_text
    import json as _json
    return jsonify(_json.loads(_golden_json_text()))


def _llm_unavailable(e) -> tuple:
    return jsonify({"error": str(e)}), 503


@bp.post("/api/v1/explain")
def explain():
    """Explain a Chhat report in simple homeowner words (LLM).

    Body: {"report": {...}, "lang": "en"|"ur"}.
    503 when the LLM is not configured — never a fabricated explanation.
    """
    from chhat.llm import client as llm
    body = request.get_json(silent=True) or {}
    report = body.get("report")
    lang = body.get("lang", "en")
    if not isinstance(report, dict) or not report:
        return jsonify({"error": "missing or invalid 'report' object"}), 400
    if lang not in ("en", "ur"):
        return jsonify({"error": "lang must be 'en' or 'ur'"}), 400
    try:
        text, model_used = llm.explain_report(report, lang)
    except llm.LLMNotConfigured as e:
        return _llm_unavailable(e)
    except llm.LLMError as e:
        return jsonify({"error": f"explanation failed: {e}"}), 502
    return jsonify({"explanation": text, "model_used": model_used})


@bp.post("/api/v1/ask")
def ask():
    """Answer a solar/rooftop question (LLM), grounded in an optional report.

    Body: {"question": "...", "lang": "en"|"ur", "report": {...optional}}.
    503 when the LLM is not configured — never a fabricated answer.
    """
    from chhat.llm import client as llm
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    lang = body.get("lang", "en")
    report = body.get("report")
    if not question:
        return jsonify({"error": "missing 'question'"}), 400
    if len(question) > 2000:
        return jsonify({"error": "question too long (max 2000 chars)"}), 400
    if lang not in ("en", "ur"):
        return jsonify({"error": "lang must be 'en' or 'ur'"}), 400
    if report is not None and not isinstance(report, dict):
        return jsonify({"error": "'report' must be an object"}), 400
    try:
        text, model_used = llm.answer_question(question, lang, report)
    except llm.LLMNotConfigured as e:
        return _llm_unavailable(e)
    except llm.LLMError as e:
        return jsonify({"error": f"answer failed: {e}"}), 502
    return jsonify({"answer": text, "model_used": model_used})


DOCS_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chhat API docs</title>
<style>
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:#040914;color:#e9f0fa;margin:0;padding:28px 18px;max-width:860px;margin-inline:auto}
h1{color:#fbbf24;font-size:26px}h2{color:#34d399;font-size:15px;letter-spacing:1px;text-transform:uppercase;margin-top:34px;border-bottom:1px solid #1e2f4d;padding-bottom:8px}
.ep{background:#0b1628;border:1px solid #1e2f4d;border-radius:12px;padding:16px;margin:14px 0}
.m{display:inline-block;font-weight:700;font-size:12px;padding:4px 10px;border-radius:6px;margin-right:8px}
.get{background:#14532d;color:#86efac}.post{background:#7c2d12;color:#fdba74}
code{color:#fde68a;font-size:13px;word-break:break-all}
pre{background:#060d1a;border:1px solid #1e2f4d;border-radius:8px;padding:12px;overflow-x:auto;font-size:12.5px;color:#cdd9ea}
.note{color:#93a4bd;font-size:13px}.warn{color:#f87171}
a{color:#fbbf24}
</style></head><body>
<h1>\u0686\u06be\u062a Chhat API</h1>
<p class="note">Version 1.0.0 &middot; base URL is the host root. All errors use the
envelope <code>{"error": "..."}</code> with proper HTTP status codes.
<code>/api/*</code> is rate-limited to 60 requests/minute per IP (HTTP 429 +
<code>Retry-After</code> when exceeded). The LLM endpoints need
<code>OPENROUTER_API_KEY</code> set on the host, otherwise they return 503.</p>
<p class="note"><b>LLM transport modes</b> (<code>/api/health</code> &rarr;
<code>llm_mode</code>): <code>server</code> means this host proxies LLM calls
itself (key stays secret, production setup). <code>direct</code> means the host
cannot reach OpenRouter (egress strips auth headers), so the page lets the
visitor's own browser call OpenRouter straight from their network &mdash; this
makes the LLM genuinely work on a phone preview. <span class="warn">Tradeoff:
direct mode puts the key in the served page, so anyone holding the preview URL
can see it. Direct mode is preview-only; production uses server mode with the
key in the environment.</span> <code>off</code> means no key is configured.</p>

<h2>Health &amp; models</h2>
<div class="ep"><span class="m get">GET</span><code>/api/health</code>
<pre>{
  "status": "ok",
  "version": "1.0.0",
  "models_available": {"parser": true, "roofseg_v2": true, "roofseg_v3": true},
  "llm_configured": false,
  "timestamp": "2026-10-11T15:45:00+00:00"
}</pre></div>
<div class="ep"><span class="m get">GET</span><code>/api/v1/models</code>
<p class="note">Cards for our own trained models: address-parser BiLSTM, roofseg v2
generalist (test IoU 0.3604), roofseg v3 dense specialist (dense IoU 0.3846).</p></div>
<div class="ep"><span class="m get">GET</span><code>/api/v1/demos</code>
<p class="note">Golden demo list (same data as <code>/golden-demos.json</code>).</p></div>

<h2>LLM enhancement layer</h2>
<div class="ep"><span class="m post">POST</span><code>/api/v1/explain</code>
<p class="note">Explains a Chhat report in simple homeowner words.</p>
<pre>Request:  {"report": {...}, "lang": "en"}
Response: {"explanation": "...", "model_used": "claude-haiku-5.5:free"}</pre></div>
<div class="ep"><span class="m post">POST</span><code>/api/v1/ask</code>
<p class="note">Answers solar questions for Pakistan; grounded in the report when supplied.</p>
<pre>Request:  {"question": "Is 10kW enough for a 3-bed house?", "lang": "en"}
Response: {"answer": "...", "model_used": "claude-haiku-5.5:free"}</pre>
<p class="warn">503 {"error": "LLM not configured: set the OPENROUTER_API_KEY ..."}
when no key is set. Never a fabricated answer.</p></div>

<h2>Core product (unchanged)</h2>
<div class="ep"><span class="m post">POST</span><code>/api/parse</code>
<pre>{"address": "House 123, Street 4, Block C, Nishat Colony, Lahore"}</pre></div>
<div class="ep"><span class="m post">POST</span><code>/api/analyze</code>
<pre>{"address": "...", "lat": 31.52, "lng": 74.35}</pre></div>
<div class="ep"><span class="m post">POST</span><code>/api/requote</code>
<pre>{"panels": 10, "annual_kwh_per_kw": 1550}</pre></div>
<div class="ep"><span class="m post">POST</span><code>/api/roofseg</code>
<pre>{"lat": 31.52, "lng": 74.35}</pre></div>
<div class="ep"><span class="m get">GET</span><code>/golden-demos.json</code>
<p class="note">Precomputed offline demo data.</p></div>
</body></html>
"""


@bp.get("/api/docs")
def docs():
    """Human-readable API documentation page."""
    from flask import Response
    return Response(DOCS_HTML, mimetype="text/html")
