"""Tests for src/server.py — Web Dashboard HTTP Server."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
import urllib.error
from unittest.mock import patch, MagicMock

import pytest

from src.server import create_server


@pytest.fixture(scope="module")
def running_server():
    """Start an ephemeral test server in a background thread."""
    server = create_server(host="127.0.0.1", port=8990)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)  # allow server to start
    base_url = f"http://127.0.0.1:{port}"
    yield base_url
    server.shutdown()
    server.server_close()


class TestServerStaticEndpoints:
    """Test static file delivery."""

    def test_get_index_html(self, running_server):
        req = urllib.request.Request(f"{running_server}/")
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            content = resp.read().decode("utf-8")
            assert "AI HARNESS" in content
            assert "AUTONOMOUS PIPELINE LIFECYCLE" in content

    def test_get_styles_css(self, running_server):
        req = urllib.request.Request(f"{running_server}/styles.css")
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            content = resp.read().decode("utf-8")
            assert "--bg-canvas" in content
            assert "dark-theme" in content

    def test_get_app_js(self, running_server):
        req = urllib.request.Request(f"{running_server}/app.js")
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            content = resp.read().decode("utf-8")
            assert "renderEvidenceGraph" in content

    def test_get_unknown_returns_404(self, running_server):
        req = urllib.request.Request(f"{running_server}/non-existent.xyz")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 404


class TestServerApiEndpoints:
    """Test JSON API endpoints."""

    def test_api_status(self, running_server):
        req = urllib.request.Request(f"{running_server}/api/status")
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "ready"
            assert "model_name" in data
            assert "configured_providers" in data

    def test_cors_options(self, running_server):
        req = urllib.request.Request(f"{running_server}/api/status", method="OPTIONS")
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 204
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"

    def test_run_task_empty_body_fails(self, running_server):
        req = urllib.request.Request(
            f"{running_server}/api/run-task",
            data=json.dumps({"task": ""}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 400

    @patch("src.server.subprocess.run")
    def test_run_task_clamp_executes_demo(self, mock_subproc, running_server):
        mock_subproc.return_value = MagicMock(
            returncode=0,
            stdout="COUNTEREXAMPLE FOUND\nFINAL STATUS: VERIFIED",
            stderr="",
        )
        req = urllib.request.Request(
            f"{running_server}/api/run-task",
            data=json.dumps({"task": "Add clamp function"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["outcome"] == "VERIFIED"
            assert data["had_counterexample"] is True
            assert len(data["obligations"]) == 5
