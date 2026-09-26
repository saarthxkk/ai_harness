"""Vercel Serverless Function entry point for AI Harness."""

from __future__ import annotations

import pathlib
import sys

# Ensure repository root is in sys.path
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.server import HarnessRequestHandler


class handler(HarnessRequestHandler):
    """BaseHTTPRequestHandler subclass recognized by Vercel."""
    pass
