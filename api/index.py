"""Vercel serverless entrypoint for the Chhat Flask app.

The app runs on pure-numpy inference here (no torch on serverless);
see chhat/parser/infer_numpy.py. Model weights ship in models/.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chhat.web.app import app  # noqa: E402  (Vercel looks for `app`)
