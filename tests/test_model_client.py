"""Tests for src/model_client.py — Multi-provider LLM API client."""

from __future__ import annotations

import io
import json
import os
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from src.model_client import (
    ContentBlock,
    ModelResponse,
    _convert_messages_to_openai,
    _convert_tools_to_openai,
    _validate_text_only,
    call_model,
    detect_provider,
    get_configured_providers,
)


# ---------------------------------------------------------------------------
# Helpers for mocking urllib.request.urlopen
# ---------------------------------------------------------------------------


def _mock_http_response(json_data: dict, status: int = 200):
    """Return a mock context manager mimicking urllib.request.urlopen response."""
    body_bytes = json.dumps(json_data).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = body_bytes
    mock_resp.status = status
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    return mock_resp


# ---------------------------------------------------------------------------
# Provider Detection Tests
# ---------------------------------------------------------------------------


class TestProviderDetection:
    """Tests for auto-detection and key resolution."""

    def test_detect_anthropic_by_ai_api_key(self):
        env = {"AI_API_KEY": "sk-ant-test"}
        p, m, k = detect_provider(config={}, environ=env)
        assert p == "anthropic"
        assert k == "sk-ant-test"
        assert "claude" in m

    def test_detect_anthropic_by_anthropic_api_key(self):
        env = {"ANTHROPIC_API_KEY": "sk-ant-test-2"}
        p, m, k = detect_provider(config={}, environ=env)
        assert p == "anthropic"
        assert k == "sk-ant-test-2"

    def test_detect_openai_by_env(self):
        env = {"OPENAI_API_KEY": "sk-proj-test"}
        p, m, k = detect_provider(config={}, environ=env)
        assert p == "openai"
        assert k == "sk-proj-test"
        assert "gpt" in m

    def test_detect_gemini_by_env(self):
        env = {"GEMINI_API_KEY": "AIzaSy-test"}
        p, m, k = detect_provider(config={}, environ=env)
        assert p == "gemini"
        assert k == "AIzaSy-test"
        assert "gemini" in m

    def test_detect_groq_by_env(self):
        env = {"GROQ_API_KEY": "gsk_test"}
        p, m, k = detect_provider(config={}, environ=env)
        assert p == "groq"
        assert k == "gsk_test"
        assert "llama" in m

    def test_detect_ollama_by_env(self):
        env = {"OLLAMA_BASE_URL": "http://localhost:11434/v1"}
        p, m, k = detect_provider(config={}, environ=env)
        assert p == "ollama"
        assert k == "http://localhost:11434/v1"

    def test_detect_by_model_name_hint(self):
        env = {"OPENAI_API_KEY": "sk-openai", "AI_API_KEY": "sk-ant"}
        cfg = {"model": {"name": "gpt-4o"}}
        p, m, k = detect_provider(config=cfg, environ=env)
        assert p == "openai"
        assert m == "gpt-4o"
        assert k == "sk-openai"

    def test_explicit_provider_in_config(self):
        env = {"OPENAI_API_KEY": "sk-openai", "AI_API_KEY": "sk-ant"}
        cfg = {"model": {"provider": "openai", "name": "custom-gpt"}}
        p, m, k = detect_provider(config=cfg, environ=env)
        assert p == "openai"
        assert m == "custom-gpt"
        assert k == "sk-openai"

    def test_configured_providers_list(self):
        env = {
            "OPENAI_API_KEY": "key1",
            "GEMINI_API_KEY": "key2",
            "GROQ_API_KEY": "key3",
        }
        providers = get_configured_providers(environ=env)
        assert "openai" in providers
        assert "gemini" in providers
        assert "groq" in providers
        assert "anthropic" not in providers


# ---------------------------------------------------------------------------
# Schema Conversion Tests
# ---------------------------------------------------------------------------


class TestSchemaConversion:
    """Tests for message and tool conversion to OpenAI format."""

    def test_convert_tools_to_openai(self):
        anthropic_tools = [
            {
                "name": "edit_file",
                "description": "Modify file contents",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ]
        openai_tools = _convert_tools_to_openai(anthropic_tools)
        assert len(openai_tools) == 1
        fn = openai_tools[0]["function"]
        assert openai_tools[0]["type"] == "function"
        assert fn["name"] == "edit_file"
        assert fn["description"] == "Modify file contents"
        assert fn["parameters"] == anthropic_tools[0]["input_schema"]

    def test_convert_messages_with_list_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Part 1"},
                    {"type": "text", "text": "Part 2"},
                ],
            }
        ]
        converted = _convert_messages_to_openai(messages)
        assert converted[0]["role"] == "user"
        assert "Part 1\nPart 2" in converted[0]["content"]


# ---------------------------------------------------------------------------
# OpenAI Provider Execution Tests
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    """Tests for calling OpenAI models."""

    @patch("src.model_client.urllib.request.urlopen")
    def test_openai_call_success(self, mock_urlopen):
        mock_payload = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "OpenAI generated answer"},
                    "finish_reason": "stop",
                }
            ],
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 15, "completion_tokens": 8},
        }
        mock_urlopen.return_value = _mock_http_response(mock_payload)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
            resp = call_model(
                messages=[{"role": "user", "content": "Hello GPT"}],
                provider="openai",
            )

        assert resp.text == "OpenAI generated answer"
        assert resp.content[0].text == "OpenAI generated answer"
        assert resp.stop_reason == "end_turn"
        assert resp.model == "gpt-4o"
        assert resp.usage["input_tokens"] == 15

    @patch("src.model_client.urllib.request.urlopen")
    def test_openai_tool_calls(self, mock_urlopen):
        mock_payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": json.dumps({"path": "src/main.py"}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "model": "gpt-4o",
        }
        mock_urlopen.return_value = _mock_http_response(mock_payload)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
            resp = call_model(
                messages=[{"role": "user", "content": "Read file"}],
                tools=[{"name": "read_file", "input_schema": {}}],
                provider="openai",
            )

        assert resp.stop_reason == "tool_use"
        assert len(resp.content) == 1
        block = resp.content[0]
        assert block.type == "tool_use"
        assert block.name == "read_file"
        assert block.input == {"path": "src/main.py"}
        assert block.id == "call_abc"


# ---------------------------------------------------------------------------
# Gemini Provider Execution Tests
# ---------------------------------------------------------------------------


class TestGeminiProvider:
    """Tests for calling Google Gemini models."""

    @patch("src.model_client.urllib.request.urlopen")
    def test_gemini_call_success(self, mock_urlopen):
        mock_payload = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Gemini response text"},
                    "finish_reason": "stop",
                }
            ],
            "model": "gemini-2.0-flash",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_urlopen.return_value = _mock_http_response(mock_payload)

        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSy-gemini-key"}, clear=True):
            resp = call_model(
                messages=[{"role": "user", "content": "Hi Gemini"}],
                provider="gemini",
            )

        assert resp.text == "Gemini response text"
        assert resp.model == "gemini-2.0-flash"
        # Check Authorization header was passed
        req_arg = mock_urlopen.call_args[0][0]
        assert req_arg.get_header("Authorization") == "Bearer AIzaSy-gemini-key"


# ---------------------------------------------------------------------------
# Groq Provider Execution Tests
# ---------------------------------------------------------------------------


class TestGroqProvider:
    """Tests for calling Groq models."""

    @patch("src.model_client.urllib.request.urlopen")
    def test_groq_call_success(self, mock_urlopen):
        mock_payload = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Groq fast output"},
                    "finish_reason": "stop",
                }
            ],
            "model": "llama-3.3-70b-versatile",
        }
        mock_urlopen.return_value = _mock_http_response(mock_payload)

        with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test"}, clear=True):
            resp = call_model(
                messages=[{"role": "user", "content": "Hi Groq"}],
                provider="groq",
            )

        assert resp.text == "Groq fast output"
        req_arg = mock_urlopen.call_args[0][0]
        assert req_arg.get_header("Authorization") == "Bearer gsk_test"


# ---------------------------------------------------------------------------
# Ollama Provider Execution Tests
# ---------------------------------------------------------------------------


class TestOllamaProvider:
    """Tests for calling local Ollama models."""

    @patch("src.model_client.urllib.request.urlopen")
    def test_ollama_call_success(self, mock_urlopen):
        mock_payload = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Local model output"},
                    "finish_reason": "stop",
                }
            ],
            "model": "llama3",
        }
        mock_urlopen.return_value = _mock_http_response(mock_payload)

        env = {"OLLAMA_BASE_URL": "http://localhost:11434/v1"}
        with patch.dict(os.environ, env, clear=True):
            resp = call_model(
                messages=[{"role": "user", "content": "Hello Ollama"}],
                provider="ollama",
            )

        assert resp.text == "Local model output"
        req_arg = mock_urlopen.call_args[0][0]
        assert "localhost:11434" in req_arg.full_url


# ---------------------------------------------------------------------------
# Retry and Error Handling Tests
# ---------------------------------------------------------------------------


class TestErrorAndRetryHandling:
    """Tests for error handling and retries in HTTP calls."""

    @patch("src.model_client.time.sleep")
    @patch("src.model_client.urllib.request.urlopen")
    def test_retry_on_429_rate_limit(self, mock_urlopen, mock_sleep):
        http_429 = urllib.error.HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=io.BytesIO(b'{"error": "rate limit"}'),
        )
        success_resp = _mock_http_response({
            "choices": [{"message": {"role": "assistant", "content": "Recovered"}}]
        })
        mock_urlopen.side_effect = [http_429, success_resp]

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-key"}, clear=True):
            resp = call_model([{"role": "user", "content": "hi"}], provider="openai")

        assert resp.text == "Recovered"
        assert mock_urlopen.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("src.model_client.urllib.request.urlopen")
    def test_client_error_401_raises_immediately(self, mock_urlopen):
        http_401 = urllib.error.HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=io.BytesIO(b'{"error": "invalid api key"}'),
        )
        mock_urlopen.side_effect = http_401

        with patch.dict(os.environ, {"OPENAI_API_KEY": "invalid-key"}, clear=True):
            with pytest.raises(RuntimeError, match="HTTP 401 error"):
                call_model([{"role": "user", "content": "hi"}], provider="openai")


# ---------------------------------------------------------------------------
# Contract Engine Compatibility Test
# ---------------------------------------------------------------------------


class TestContractEngineCompatibility:
    """Verify ContractEngine can parse output from ModelResponse."""

    def test_contract_engine_extracts_from_model_response(self):
        from src.contract_engine import ContractEngine

        json_contract = json.dumps({
            "task_goal": "Prevent locked user auth",
            "behavioral_requirements": ["Reject locked accounts"],
            "acceptance_criteria": ["Return 403 on locked"],
            "constraints": ["No plaintext passwords"],
            "non_goals": ["OAuth changes"],
            "affected_area": ["auth"],
            "expected_files": ["src/auth.py"],
            "risk_level": "LOW",
            "verification_plan": ["Run auth tests"],
            "proof_obligations": [
                {
                    "id": "OB-01",
                    "description": "Locked users rejected",
                    "type": "behavior",
                    "verification_method": "test",
                }
            ],
        })

        mock_response = ModelResponse(
            content=[ContentBlock(type="text", text=json_contract)],
            stop_reason="end_turn",
            model="gpt-4o",
        )

        extracted = ContractEngine._extract_response_text(mock_response)
        assert "Prevent locked user auth" in extracted
