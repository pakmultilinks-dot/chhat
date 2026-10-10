"""Vercel serverless entrypoint for the Chhat Flask app.

Vercel invokes this file for /api/index (and / via rewrites). The WSGI
environ carries PATH_INFO=/api/index, which Flask would 404 on, so a tiny
middleware strips the mount prefix before Flask sees the request.

The app runs on pure-numpy inference here (no torch on serverless);
see chhat/parser/infer_numpy.py. Model weights download once from the
public repo on cold start (see _ensure_weights).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chhat.web.app import app as flask_app  # noqa: E402


class _PrefixStripper:
    """Strip the /api/index mount prefix so Flask routes match."""

    def __init__(self, app, prefix="/api/index"):
        self.app = app
        self.prefix = prefix

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if path == self.prefix:
            environ["PATH_INFO"] = "/"
        elif path.startswith(self.prefix + "/"):
            environ["PATH_INFO"] = path[len(self.prefix):]
        return self.app(environ, start_response)


app = _PrefixStripper(flask_app)
