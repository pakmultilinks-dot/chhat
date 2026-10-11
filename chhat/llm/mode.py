"""LLM transport mode detection for Chhat.

The server prefers to proxy LLM calls itself ("server" mode), keeping the
API key secret on the host. On hosts where egress strips Authorization
headers (this dev VM does), the server can never reach OpenRouter, so the
page falls back to "direct" mode: the visitor's own browser calls OpenRouter
straight from their network, which works fine on a normal phone/ISP.

  "server" - the host can reach OpenRouter with the configured key
  "direct" - a key is set but the host cannot complete the call itself
  "off"    - no OPENROUTER_API_KEY is set at all

The mode is probed lazily (one tiny auth/key call, 8s timeout) and cached
for 5 minutes so page renders and health checks stay fast.

Security: the key is NEVER rendered into the page except in "direct" mode,
and then only read from the OPENROUTER_API_KEY environment variable. It is
never hardcoded, never logged, never committed.
"""
from __future__ import annotations

import os
import time

import requests

_PROBE_URL = "https://openrouter.ai/api/v1/auth/key"
_PROBE_TIMEOUT_S = 8
_CACHE_TTL_S = 300

_cache: dict = {"mode": None, "at": 0.0}


def reset_cache() -> None:
    """Clear the cached mode (used by tests)."""
    _cache["mode"] = None
    _cache["at"] = 0.0


def get_model() -> str:
    return os.environ.get("OPENROUTER_MODEL", "claude-haiku-5.5:free")


def get_mode() -> str:
    """Return the current LLM transport mode: server | direct | off."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        return "off"
    now = time.time()
    if _cache["mode"] and now - _cache["at"] < _CACHE_TTL_S:
        return _cache["mode"]
    mode = _probe()
    _cache["mode"] = mode
    _cache["at"] = now
    return mode


def _probe() -> str:
    """One tiny authenticated call. 'server' only on HTTP 200."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    try:
        r = requests.get(
            _PROBE_URL,
            headers={"Authorization": "Bearer " + key},
            timeout=_PROBE_TIMEOUT_S,
        )
        return "server" if r.status_code == 200 else "direct"
    except Exception:
        # Network blocked, DNS failure, stripped auth header (401), etc.
        return "direct"
