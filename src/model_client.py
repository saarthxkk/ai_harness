"""Model Client — Multi-provider LLM API communication layer.

Supports:
- **Anthropic / Claude** (via official ``anthropic`` SDK)
- **OpenAI / GPT-4o / o1 / o3** (via OpenAI chat completions API)
- **Google Gemini** (via Gemini OpenAI-compatible API)
- **Groq** (ultra-fast Llama-3.3 inference)
- **Ollama** (local models, no API key required)

Features:
- Auto-provider detection based on active environment keys & config
- Text-only safety guard (refuses multi-modal image/audio/video content)
- Unified response model matching Anthropic Message interface
- Automatic exponential backoff retries on rate limits (429) & server errors (5xx)
- Zero external dependencies for OpenAI/Gemini/Groq/Ollama (uses standard library urllib)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
import os
import pathlib
import ssl
import time
import urllib.error
import urllib.request
from typing import Any

import anthropic
import yaml

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

_CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config.yaml"

_BLOCKED_CONTENT_TYPES = frozenset({"image", "image_url", "audio", "video"})

_MAX_RETRIES = 3
_BACKOFF_BASE = 2  # seconds

# Default models per provider
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "gemini": "gemini-2.0-flash",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3",
}

# Default API endpoints
DEFAULT_ENDPOINTS: dict[str, str] = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "ollama": "http://localhost:11434/v1/chat/completions",
}


def _load_config(path: pathlib.Path = _CONFIG_PATH) -> dict[str, Any]:
    """Parse *config.yaml* and return it as a plain dict."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
            return cfg if isinstance(cfg, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


# ---------------------------------------------------------------------------
# Normalized Response Types
# ---------------------------------------------------------------------------


@dataclass
class ContentBlock:
    """A normalized content block matching Anthropic message block structure."""
    type: str = "text"  # "text" | "tool_use"
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type}
        if self.type == "text":
            d["text"] = self.text
        elif self.type == "tool_use":
            d["id"] = self.id
            d["name"] = self.name
            d["input"] = self.input
        return d


@dataclass
class ModelResponse:
    """A normalized model response compatible with Anthropic's Message interface."""
    content: list[ContentBlock]
    stop_reason: str = "end_turn"
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Convenience property to extract primary text."""
        for block in self.content:
            if block.type == "text" and block.text:
                return block.text
        return ""


# ---------------------------------------------------------------------------
# Provider Detection & Key Resolution
# ---------------------------------------------------------------------------


def detect_provider(
    config: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
) -> tuple[str, str, str]:
    """Detect the active provider, model name, and API key or base endpoint.

    Returns
    -------
    tuple[provider, model_name, api_key_or_endpoint]
        provider : 'anthropic' | 'openai' | 'gemini' | 'groq' | 'ollama'
        model_name : configured or default model for that provider
        api_key_or_endpoint : secret key or base URL
    """
    cfg = config if config is not None else _load_config()
    env = environ if environ is not None else os.environ

    explicit_provider = cfg.get("model", {}).get("provider", "auto")
    config_model = cfg.get("model", {}).get("name", "")

    # 1. Check if provider is explicitly set
    if explicit_provider and explicit_provider.lower() != "auto":
        p = explicit_provider.lower()
        model = config_model or DEFAULT_MODELS.get(p, "gpt-4o")
        key = _resolve_key_for_provider(p, env)
        return p, model, key

    # 2. Check model name prefix hint
    if config_model:
        m = config_model.lower()
        if m.startswith("claude"):
            key = env.get("AI_API_KEY") or env.get("ANTHROPIC_API_KEY")
            if key:
                return "anthropic", config_model, key
        elif m.startswith(("gpt-", "o1", "o3", "chatgpt")):
            key = env.get("OPENAI_API_KEY")
            if key:
                return "openai", config_model, key
        elif m.startswith("gemini"):
            key = env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY")
            if key:
                return "gemini", config_model, key
        elif m.startswith("groq/") or (m.startswith("llama-") and env.get("GROQ_API_KEY")):
            key = env.get("GROQ_API_KEY")
            if key:
                return "groq", config_model.replace("groq/", ""), key

    # 3. Detect by available environment variables (priority order)
    if env.get("AI_API_KEY") or env.get("ANTHROPIC_API_KEY"):
        key = env.get("AI_API_KEY") or env.get("ANTHROPIC_API_KEY", "")
        model = config_model if (config_model and "claude" in config_model.lower()) else DEFAULT_MODELS["anthropic"]
        return "anthropic", model, key

    if env.get("OPENAI_API_KEY"):
        key = env.get("OPENAI_API_KEY", "")
        model = config_model if (config_model and ("gpt" in config_model.lower() or "o1" in config_model.lower() or "o3" in config_model.lower())) else DEFAULT_MODELS["openai"]
        return "openai", model, key

    if env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY"):
        key = env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY", "")
        model = config_model if (config_model and "gemini" in config_model.lower()) else DEFAULT_MODELS["gemini"]
        return "gemini", model, key

    if env.get("GROQ_API_KEY"):
        key = env.get("GROQ_API_KEY", "")
        model = config_model if (config_model and ("llama" in config_model.lower() or "mixtral" in config_model.lower())) else DEFAULT_MODELS["groq"]
        return "groq", model, key

    if env.get("OLLAMA_BASE_URL") or env.get("OLLAMA_HOST"):
        endpoint = env.get("OLLAMA_BASE_URL") or env.get("OLLAMA_HOST", "")
        model = config_model or DEFAULT_MODELS["ollama"]
        return "ollama", model, endpoint

    # Fallback default: anthropic (will raise missing key error if not configured)
    model = config_model or DEFAULT_MODELS["anthropic"]
    key = env.get("AI_API_KEY") or env.get("ANTHROPIC_API_KEY", "")
    return "anthropic", model, key


def _resolve_key_for_provider(provider: str, env: dict[str, str] | Any) -> str:
    """Resolve key or endpoint for a designated provider."""
    if provider == "anthropic":
        return env.get("AI_API_KEY") or env.get("ANTHROPIC_API_KEY", "")
    elif provider == "openai":
        return env.get("OPENAI_API_KEY", "")
    elif provider == "gemini":
        return env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY", "")
    elif provider == "groq":
        return env.get("GROQ_API_KEY", "")
    elif provider == "ollama":
        return env.get("OLLAMA_BASE_URL") or env.get("OLLAMA_HOST") or DEFAULT_ENDPOINTS["ollama"]
    return ""


def get_configured_providers(environ: dict[str, str] | None = None) -> list[str]:
    """Return a list of all providers that have keys or endpoints configured."""
    env = environ if environ is not None else os.environ
    configured: list[str] = []
    if env.get("AI_API_KEY") or env.get("ANTHROPIC_API_KEY"):
        configured.append("anthropic")
    if env.get("OPENAI_API_KEY"):
        configured.append("openai")
    if env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY"):
        configured.append("gemini")
    if env.get("GROQ_API_KEY"):
        configured.append("groq")
    if env.get("OLLAMA_BASE_URL") or env.get("OLLAMA_HOST"):
        configured.append("ollama")
    return configured


def _get_api_key() -> str:
    """Return the API key from the environment or raise with clear multi-provider instructions."""
    provider, _, key = detect_provider()
    if not key and provider != "ollama":
        raise EnvironmentError(
            "AI_API_KEY environment variable is not set. "
            "Supported keys: AI_API_KEY (Anthropic), OPENAI_API_KEY (OpenAI), "
            "GEMINI_API_KEY (Google), GROQ_API_KEY (Groq), or OLLAMA_BASE_URL."
        )
    return key


# ---------------------------------------------------------------------------
# Content-type guard
# ---------------------------------------------------------------------------


def _validate_text_only(messages: list[dict[str, Any]]) -> None:
    """Raise ``ValueError`` if any message contains non-text content."""
    for msg in messages:
        content = msg.get("content")
        # content can be a plain string (always OK) or a list of blocks.
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "text")
                    if block_type in _BLOCKED_CONTENT_TYPES:
                        raise ValueError(
                            f"Non-text content type '{block_type}' is not "
                            "supported.  This harness is text-only."
                        )


# ---------------------------------------------------------------------------
# Provider Call Handlers
# ---------------------------------------------------------------------------


def _call_anthropic(
    model_name: str,
    max_tokens: int,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    api_key: str,
) -> Any:
    """Dispatch request to Anthropic Claude SDK with retries."""
    client = anthropic.Anthropic(api_key=api_key)

    kwargs: dict[str, Any] = {
        "model": model_name,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools

    last_exc: BaseException | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError as exc:
            last_exc = exc
        except anthropic.APIConnectionError as exc:
            last_exc = exc
        except anthropic.APIStatusError as exc:
            # Retry only on transient server errors (5xx).
            if exc.status_code >= 500:
                last_exc = exc
            else:
                raise

        if attempt < _MAX_RETRIES - 1:
            time.sleep(_BACKOFF_BASE ** (attempt + 1))

    raise last_exc  # type: ignore[misc]


def _convert_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic tool definitions to OpenAI function tool format."""
    openai_tools = []
    for tool in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })
    return openai_tools


def _convert_messages_to_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure messages conform to standard OpenAI chat completions format."""
    openai_msgs: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            # Extract text blocks
            text_parts = []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    text_parts.append(b.get("text", ""))
                elif isinstance(b, str):
                    text_parts.append(b)
            content = "\n".join(text_parts)
        openai_msgs.append({"role": role, "content": content})
    return openai_msgs


def _call_openai_compatible(
    provider: str,
    endpoint: str,
    headers: dict[str, str],
    model_name: str,
    max_tokens: int,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> ModelResponse:
    """Execute an HTTP request to any OpenAI-compatible completions API."""
    formatted_messages = _convert_messages_to_openai(messages)

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": formatted_messages,
    }

    # Handle models with max_completion_tokens (e.g. OpenAI o1/o3) vs max_tokens
    if model_name.startswith(("o1", "o3")):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens

    if tools:
        payload["tools"] = _convert_tools_to_openai(tools)

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=req_data, headers=headers, method="POST")

    # SSL context with default certificates
    ctx = ssl.create_default_context()

    last_exc: BaseException | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=60, context=ctx) as response:
                body_bytes = response.read()
                data = json.loads(body_bytes.decode("utf-8"))

                # Parse choice
                choices = data.get("choices", [])
                if not choices:
                    return ModelResponse(
                        content=[ContentBlock(type="text", text="")],
                        stop_reason="end_turn",
                        model=data.get("model", model_name),
                    )

                first_choice = choices[0]
                msg = first_choice.get("message", {})
                finish_reason = first_choice.get("finish_reason", "stop")

                # Map finish_reason to Anthropic stop_reason
                stop_reason = "end_turn"
                if finish_reason == "tool_calls":
                    stop_reason = "tool_use"
                elif finish_reason == "length":
                    stop_reason = "max_tokens"

                blocks: list[ContentBlock] = []

                # Add text content if present
                if msg.get("content"):
                    blocks.append(ContentBlock(type="text", text=msg["content"]))

                # Add tool calls if present
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", {})
                        args_str = fn.get("arguments", "{}")
                        try:
                            args_dict = json.loads(args_str)
                        except Exception:
                            args_dict = {"raw": args_str}

                        blocks.append(ContentBlock(
                            type="tool_use",
                            id=tc.get("id", ""),
                            name=fn.get("name", ""),
                            input=args_dict,
                        ))

                if not blocks:
                    blocks.append(ContentBlock(type="text", text=""))

                usage = data.get("usage", {})
                return ModelResponse(
                    content=blocks,
                    stop_reason=stop_reason,
                    model=data.get("model", model_name),
                    usage={
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                    },
                )

        except urllib.error.HTTPError as exc:
            # Retry on rate limit (429) or server errors (5xx)
            if exc.code == 429 or exc.code >= 500:
                last_exc = exc
            else:
                err_body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
                raise RuntimeError(
                    f"{provider.upper()} API HTTP {exc.code} error: {exc.reason} - {err_body}"
                ) from exc
        except urllib.error.URLError as exc:
            last_exc = exc

        if attempt < _MAX_RETRIES - 1:
            time.sleep(_BACKOFF_BASE ** (attempt + 1))

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def call_model(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    provider: str | None = None,
) -> Any:
    """Send a chat-completion request to the active LLM provider.

    Supports Anthropic (Claude), OpenAI (GPT-4o), Google Gemini, Groq, and Ollama.
    Returns an object with ``.content``, ``.stop_reason``, and ``.model``.

    Parameters
    ----------
    messages:
        A list of message dicts in standard format
        (``{"role": "user"|"assistant"|"system", "content": ...}``).
        Only **text** content blocks are accepted.
    tools:
        Optional list of tool definitions in tool-use schema
        (each dict must have ``name``, ``description``, and ``input_schema`` keys).
    provider:
        Optional provider override ('anthropic', 'openai', 'gemini', 'groq', 'ollama').

    Returns
    -------
    anthropic.types.Message | ModelResponse
        The full API response object.

    Raises
    ------
    ValueError
        If any message contains image, audio, or video content.
    EnvironmentError
        If no valid API key is set for the active provider.
    """
    _validate_text_only(messages)

    config = _load_config()
    max_tokens: int = config.get("runtime", {}).get("max_tokens", 4096)

    detected_provider, model_name, key_or_endpoint = detect_provider(config)
    active_provider = (provider or detected_provider).lower()

    # Re-verify key if provider was overridden
    if provider and provider.lower() != detected_provider:
        key_or_endpoint = _resolve_key_for_provider(active_provider, os.environ)
        model_name = config.get("model", {}).get("name") or DEFAULT_MODELS.get(active_provider, "gpt-4o")

    # Ensure key exists for providers that require authentication
    if not key_or_endpoint and active_provider != "ollama":
        raise EnvironmentError(
            f"AI_API_KEY environment variable is not set. "
            f"Provider '{active_provider}' requires an API key in the environment."
        )

    # 1. Anthropic Provider
    if active_provider == "anthropic":
        return _call_anthropic(
            model_name=model_name,
            max_tokens=max_tokens,
            messages=messages,
            tools=tools,
            api_key=key_or_endpoint,
        )

    # 2. OpenAI Provider
    elif active_provider == "openai":
        endpoint = os.environ.get("OPENAI_BASE_URL", DEFAULT_ENDPOINTS["openai"])
        if not endpoint.endswith("/chat/completions"):
            endpoint = endpoint.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key_or_endpoint}",
        }
        return _call_openai_compatible(
            provider="openai",
            endpoint=endpoint,
            headers=headers,
            model_name=model_name,
            max_tokens=max_tokens,
            messages=messages,
            tools=tools,
        )

    # 3. Google Gemini Provider (OpenAI-compatible REST API)
    elif active_provider == "gemini":
        endpoint = os.environ.get("GEMINI_BASE_URL", DEFAULT_ENDPOINTS["gemini"])
        if not endpoint.endswith("/chat/completions"):
            endpoint = endpoint.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key_or_endpoint}",
        }
        return _call_openai_compatible(
            provider="gemini",
            endpoint=endpoint,
            headers=headers,
            model_name=model_name,
            max_tokens=max_tokens,
            messages=messages,
            tools=tools,
        )

    # 4. Groq Provider
    elif active_provider == "groq":
        endpoint = os.environ.get("GROQ_BASE_URL", DEFAULT_ENDPOINTS["groq"])
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key_or_endpoint}",
        }
        return _call_openai_compatible(
            provider="groq",
            endpoint=endpoint,
            headers=headers,
            model_name=model_name,
            max_tokens=max_tokens,
            messages=messages,
            tools=tools,
        )

    # 5. Ollama Local Provider
    elif active_provider == "ollama":
        base_url = (
            key_or_endpoint
            or os.environ.get("OLLAMA_BASE_URL")
            or os.environ.get("OLLAMA_HOST")
            or "http://localhost:11434"
        )
        if not base_url.endswith("/v1/chat/completions"):
            endpoint = base_url.rstrip("/") + "/v1/chat/completions"
        else:
            endpoint = base_url
        headers = {
            "Content-Type": "application/json",
        }
        return _call_openai_compatible(
            provider="ollama",
            endpoint=endpoint,
            headers=headers,
            model_name=model_name,
            max_tokens=max_tokens,
            messages=messages,
            tools=tools,
        )

    else:
        raise ValueError(
            f"Unsupported provider '{active_provider}'. Supported: "
            "anthropic, openai, gemini, groq, ollama."
        )
