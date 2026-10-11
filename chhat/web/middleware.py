"""Chhat web middleware: request logging + API rate limiting.

- Logs method, path (no query string), status, duration. Never logs bodies
  or query strings, so no addresses or other PII end up in logs.
- In-memory sliding-window rate limit on /api/* : 60 requests per minute
  per client IP. Exceeded requests get HTTP 429 with a Retry-After header
  and the standard {"error": ...} JSON envelope.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque

from flask import Flask, g, jsonify, request

log = logging.getLogger("chhat.api")

RATE_LIMIT = 60          # requests
RATE_WINDOW_S = 60       # per this many seconds

_hits: dict[str, deque] = defaultdict(deque)


def reset_rate_limit() -> None:
    """Clear all rate-limit state (used by tests)."""
    _hits.clear()


def _client_ip() -> str:
    return request.headers.get("X-Forwarded-For", request.remote_addr or "?").split(",")[0].strip()


def register(app: Flask) -> None:
    @app.before_request
    def _start_timer():
        g._t0 = time.perf_counter()

    @app.before_request
    def _rate_limit():
        if not request.path.startswith("/api/"):
            return None
        now = time.time()
        dq = _hits[_client_ip()]
        while dq and dq[0] <= now - RATE_WINDOW_S:
            dq.popleft()
        if len(dq) >= RATE_LIMIT:
            retry = int(dq[0] + RATE_WINDOW_S - now) + 1
            resp = jsonify({"error": "rate limit exceeded, try again shortly"})
            resp.status_code = 429
            resp.headers["Retry-After"] = str(retry)
            return resp
        dq.append(now)
        return None

    @app.after_request
    def _log_request(resp):
        try:
            ms = (time.perf_counter() - g.get("_t0", time.perf_counter())) * 1000
            # path only: query strings may carry addresses -> never logged
            log.info("%s %s -> %s (%.1f ms)",
                     request.method, request.path, resp.status_code, ms)
        except Exception:
            pass
        return resp

    @app.errorhandler(404)
    def _not_found(_e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "not found"}), 404
        return _e

    @app.errorhandler(405)
    def _method_not_allowed(_e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "method not allowed"}), 405
        return _e

    @app.errorhandler(500)
    def _internal(_e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "internal error"}), 500
        return _e
