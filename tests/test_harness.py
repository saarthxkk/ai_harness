"""Tests for the AI Coding-Agent Harness."""

import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest


# ── main.py smoke tests ─────────────────────────────────────────────


def test_main_missing_key():
    """main.py should exit non-zero when AI_API_KEY is not set."""
    env = {k: v for k, v in os.environ.items() if k != "AI_API_KEY"}
    result = subprocess.run(
        [sys.executable, "src/main.py"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "AI_API_KEY" in result.stderr


def test_main_with_key():
    """main.py should print 'Harness ready' and exit 0 when the key is set."""
    env = {**os.environ, "AI_API_KEY": "test-key-for-ci"}
    result = subprocess.run(
        [sys.executable, "src/main.py"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "Harness ready" in result.stdout


# ── model_client tests (mocked, no real network calls) ──────────────

from src.model_client import call_model, _validate_text_only  # noqa: E402


def _make_mock_response(text: str = "Hello from Claude", stop_reason: str = "end_turn"):
    """Build a fake ``anthropic.types.Message``-like object."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text

    response = MagicMock()
    response.content = [text_block]
    response.stop_reason = stop_reason
    response.model = "claude-sonnet-4-20250514"
    return response


@patch.dict(os.environ, {"AI_API_KEY": "test-key"})
@patch("src.model_client.anthropic.Anthropic")
def test_call_model_returns_text(mock_anthropic_cls):
    """call_model should return parsed content from the API response."""
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value = _make_mock_response("Test answer")

    messages = [{"role": "user", "content": "Say hi"}]
    result = call_model(messages)

    assert result.content[0].text == "Test answer"
    mock_client.messages.create.assert_called_once()

    # Verify the API was called with the right model and messages.
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["messages"] == messages
    assert call_kwargs["model"] == "claude-sonnet-4-20250514"


@patch.dict(os.environ, {"AI_API_KEY": "test-key"})
@patch("src.model_client.anthropic.Anthropic")
def test_call_model_with_tools(mock_anthropic_cls):
    """call_model should forward tool definitions to the API."""
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value = _make_mock_response(
        stop_reason="tool_use",
    )

    tools = [
        {
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
    ]
    messages = [{"role": "user", "content": "Read foo.txt"}]
    result = call_model(messages, tools=tools)

    assert result.stop_reason == "tool_use"
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["tools"] == tools


def test_call_model_rejects_image_content():
    """call_model must refuse messages containing image blocks."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"data": "base64..."}},
            ],
        }
    ]
    with pytest.raises(ValueError, match="Non-text content type"):
        _validate_text_only(messages)


def test_call_model_rejects_missing_key():
    """call_model must raise when AI_API_KEY is not set."""
    env = {k: v for k, v in os.environ.items() if k != "AI_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(EnvironmentError, match="AI_API_KEY"):
            call_model([{"role": "user", "content": "hi"}])


@patch.dict(os.environ, {"AI_API_KEY": "test-key"})
@patch("src.model_client.anthropic.Anthropic")
@patch("src.model_client.time.sleep")  # don't actually wait during tests
def test_call_model_retries_on_rate_limit(mock_sleep, mock_anthropic_cls):
    """call_model should retry up to 3 times on RateLimitError."""
    import anthropic as _anthropic

    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client

    rate_err = _anthropic.RateLimitError(
        message="rate limited",
        response=MagicMock(status_code=429, headers={}),
        body=None,
    )
    mock_client.messages.create.side_effect = [
        rate_err,
        rate_err,
        _make_mock_response("Recovered"),
    ]

    result = call_model([{"role": "user", "content": "hi"}])
    assert result.content[0].text == "Recovered"
    assert mock_client.messages.create.call_count == 3
    assert mock_sleep.call_count == 2  # slept between retries 1→2 and 2→3
