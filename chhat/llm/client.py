"""OpenRouter chat-completions client for Chhat's LLM enhancement layer.

Security rules (hard):
- API key comes ONLY from the OPENROUTER_API_KEY environment variable.
- The key is NEVER written to logs, files, error messages, or the repo.
- If no key is configured, callers get LLMNotConfigured and must degrade
  honestly (HTTP 503 with a clear message), never a fabricated answer.

Note: some sandboxed hosts strip Authorization headers on egress; on such
hosts live LLM calls fail and the endpoints correctly report 503. Set the
env var on the deployed host for live answers.
"""
from __future__ import annotations

import os
import time

import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "claude-haiku-5.5:free"
TIMEOUT_S = 20


class LLMError(Exception):
    """The LLM call failed (network, timeout, bad response)."""


class LLMNotConfigured(LLMError):
    """No OPENROUTER_API_KEY is set on this host."""


def is_configured() -> bool:
    """True when an API key is available (value never exposed)."""
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def default_model() -> str:
    return os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)


def complete(messages: list[dict], max_tokens: int = 800,
             timeout: int = TIMEOUT_S) -> tuple[str, str]:
    """Call the chat model. Returns (text, model_used).

    Raises LLMNotConfigured when no key is set, LLMError otherwise.
    One automatic retry on transient failures.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMNotConfigured(
            "LLM not configured: set the OPENROUTER_API_KEY environment "
            "variable on the host to enable explanations and Q&A."
        )
    model = default_model()
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens}
    # Build headers without ever logging the key value.
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + api_key,
               "HTTP-Referer": "https://github.com/pakmultilinks-dot/chhat",
               "X-Title": "Chhat solar advisor"}
    last_err: Exception | None = None
    for attempt in (1, 2):
        try:
            r = requests.post(API_URL, json=payload, headers=headers,
                              timeout=timeout)
            if r.status_code == 429:
                last_err = LLMError("model rate limited, try again shortly")
                time.sleep(2)
                continue
            if r.status_code >= 500:
                last_err = LLMError(
                    f"model service error (HTTP {r.status_code})")
                time.sleep(1)
                continue
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            if not text or not text.strip():
                raise LLMError("model returned an empty answer")
            return text.strip(), data.get("model", model)
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = LLMError(f"model request failed: {type(e).__name__}")
            time.sleep(1)
        except requests.HTTPError as e:
            raise LLMError(f"model request failed (HTTP {e.response.status_code})")
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"model returned an unreadable response: {e}")
    raise last_err or LLMError("model request failed")


EXPLAIN_SYSTEM = (
    "You are Chhat's honest solar advisor for Pakistani homeowners. "
    "You explain a rooftop solar report in simple, friendly words. Rules: "
    "use ONLY the numbers given in the report; never invent prices, savings, "
    "or payback figures. If the report is marked as a demo/estimate, say so "
    "plainly. Keep it short (under 180 words). No marketing fluff, no emojis."
)

ASK_SYSTEM = (
    "You are Chhat's honest solar advisor for Pakistan (net-billing era, "
    "Rs 47.69/unit grid tariff context). Answer the homeowner's question "
    "directly. If a report is provided, ground your answer in its numbers "
    "and never invent prices beyond it. If you do not know something, say "
    "'I don't know' honestly instead of guessing. Keep answers under 180 "
    "words. No marketing fluff, no emojis."
)

EXPLAIN_SYSTEM_UR = (
    "آپ چھت کے دیانت دار سولر مشیر ہیں، پاکستانی گھر مالکان کے لیے۔ "
    "چھت کی سولر رپورٹ کو سادہ اور دوستانہ اردو میں سمجھائیں۔ اصول: "
    "صرف رپورٹ میں دیے گئے اعداد استعمال کریں؛ قیمتیں، بچت یا واپسی کی مدت "
    "خود نہ گھڑیں۔ اگر رپورٹ ڈیمو/تخمینہ ہے تو صاف کہیں۔ 180 الفاظ سے کم۔"
)

ASK_SYSTEM_UR = (
    "آپ چھت کے دیانت دار سولر مشیر ہیں، پاکستان کے لیے (نیٹ بلنگ کا دور)۔ "
    "گھر مالک کے سوال کا سیدھا جواب دیں۔ اگر رپورٹ دی گئی ہے تو اس کے اعداد "
    "پر بات رکھیں اور قیمتیں خود نہ گھڑیں۔ جو بات معلوم نہ ہو اس پر ایمانداری "
    "سے 'مجھے نہیں معلوم' کہیں۔ 180 الفاظ سے کم۔"
)


def explain_report(report: dict, lang: str = "en") -> tuple[str, str]:
    """Explain a Chhat report in homeowner words. Returns (text, model)."""
    lang = "ur" if lang == "ur" else "en"
    system = EXPLAIN_SYSTEM_UR if lang == "ur" else EXPLAIN_SYSTEM
    score = (report.get("score") or {})
    quote = (report.get("quote") or {})
    layout = (report.get("layout") or {})
    demo = bool(report.get("demo"))
    facts = (
        f"Chhat Score: {score.get('score', '?')}/100. "
        f"Panels: {layout.get('panel_count', '?')} "
        f"({layout.get('system_kw', '?')} kW). "
        f"System cost: Rs {quote.get('equipment_cost_pkr', '?')} "
        f"(estimate). Yearly savings: Rs {quote.get('annual_savings_pkr', '?')} "
        f"(estimate). Payback: {quote.get('payback_years', '?')} years. "
        f"Usable roof: {layout.get('usable_area_m2', '?')} m2. "
        f"{'This is a DEMO with estimated figures.' if demo else 'Figures from live analysis.'}"
    )
    ask = ("اس رپورٹ کو سادہ اردو میں سمجھائیں۔" if lang == "ur"
           else "Explain this report in simple words.")
    return complete(
        [{"role": "system", "content": system},
         {"role": "user", "content": facts + "\n\n" + ask}])


def answer_question(question: str, lang: str = "en",
                    report: dict | None = None) -> tuple[str, str]:
    """Answer a solar/rooftop question, optionally grounded in a report."""
    lang = "ur" if lang == "ur" else "en"
    system = ASK_SYSTEM_UR if lang == "ur" else ASK_SYSTEM
    ctx = ""
    if report:
        score = (report.get("score") or {})
        quote = (report.get("quote") or {})
        layout = (report.get("layout") or {})
        ctx = (
            "Homeowner's Chhat report — use these numbers if relevant: "
            f"Score {score.get('score', '?')}/100, "
            f"{layout.get('panel_count', '?')} panels "
            f"({layout.get('system_kw', '?')} kW), "
            f"cost Rs {quote.get('equipment_cost_pkr', '?')} (estimate), "
            f"payback {quote.get('payback_years', '?')} years. "
            + ("DEMO estimates. " if report.get("demo") else "")
        )
    return complete(
        [{"role": "system", "content": system},
         {"role": "user", "content": ctx + "Question: " + question}])
