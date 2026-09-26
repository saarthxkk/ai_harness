"""Model Client — LLM API communication layer (Anthropic / Claude).

Reads configuration from config.yaml and the API key from the
AI_API_KEY environment variable.  Exposes ``call_model()`` as the
single public entry-point for the rest of the harness.

**Text-only**: this module intentionally refuses any multi-modal
(image / audio / video) content blocks.
"""

from __future__ import annotations

import os
import pathlib
import time
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


def _load_config(path: pathlib.Path = _CONFIG_PATH) -> dict[str, Any]:
    """Parse *config.yaml* and return it as a plain dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _get_api_key() -> str:
    """Return the API key from the environment or raise."""
    key = os.environ.get("AI_API_KEY")
    if not key:
        raise EnvironmentError(
            "AI_API_KEY environment variable is not set. "
            "Export it before calling model_client functions."
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
# Public API
# ---------------------------------------------------------------------------


def call_model(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> anthropic.types.Message:
    """Send a chat-completion request to the Anthropic API.

    Parameters
    ----------
    messages:
        A list of message dicts in Anthropic's ``messages`` format
        (``{"role": "user"|"assistant", "content": ...}``).
        Only **text** content blocks are accepted.
    tools:
        Optional list of tool definitions in Anthropic's tool-use
        schema (each dict must have ``name``, ``description``, and
        ``input_schema`` keys).

    Returns
    -------
    anthropic.types.Message
        The full API response object.  Callers can inspect
        ``.content``, ``.stop_reason``, etc.

    Raises
    ------
    ValueError
        If any message contains image, audio, or video content.
    EnvironmentError
        If ``AI_API_KEY`` is not set.
    anthropic.APIError
        After exhausting all retries.
    """
    _validate_text_only(messages)

    config = _load_config()
    model_name: str = config["model"]["name"]
    max_tokens: int = config["runtime"]["max_tokens"]
    api_key = _get_api_key()

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
            response = client.messages.create(**kwargs)
            return response
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

        # Exponential back-off: 2s, 4s, 8s …
        if attempt < _MAX_RETRIES - 1:
            time.sleep(_BACKOFF_BASE ** (attempt + 1))

    # All retries exhausted — re-raise the last exception.
    raise last_exc  # type: ignore[misc]
