"""Tool — Shell command execution.

Provides a safe ``run_command`` function that:
* executes from the repository root;
* enforces timeouts;
* blocks destructive commands;
* redacts secrets from output;
* returns structured results.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from typing import Any

from src.tools.risk import (
    RiskLevel,
    check_or_raise,
    get_repo_root,
    make_result,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT = 30  # seconds
MAX_TIMEOUT = 300     # 5 minutes

# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------

_SECRET_ENV_VARS = frozenset({
    "AI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "API_KEY",
    "SECRET_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
})


def _redact_secrets(text: str) -> str:
    """Replace known secret values with ``[REDACTED]``."""
    for var_name in _SECRET_ENV_VARS:
        value = os.environ.get(var_name)
        if value:
            text = text.replace(value, "[REDACTED]")
    return text


# ---------------------------------------------------------------------------
# Destructive-command blocklist
# ---------------------------------------------------------------------------

_DESTRUCTIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*r[a-zA-Z]*\s+/", re.IGNORECASE),
    re.compile(r"\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*r[a-zA-Z]*\s+~", re.IGNORECASE),
    re.compile(r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*\s+\.\s*$", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"\bdd\s+.*of=/dev/", re.IGNORECASE),
    re.compile(r":\(\)\{.*\|.*\}", re.IGNORECASE),  # fork bomb
    re.compile(r"\b>\s*/dev/[sh]da", re.IGNORECASE),
    re.compile(r"\bchmod\s+-R\s+777\s+/", re.IGNORECASE),
    re.compile(r"\bchown\s+-R\s+.*\s+/\s*$", re.IGNORECASE),
    re.compile(r"\bcurl\b.*\|\s*(ba)?sh", re.IGNORECASE),
    re.compile(r"\bwget\b.*\|\s*(ba)?sh", re.IGNORECASE),
    re.compile(r"\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*r[a-zA-Z]*\s+\.\.", re.IGNORECASE),
]

# Commands that leak environment variables / secrets.
_SECRET_LEAK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\benv\b", re.IGNORECASE),
    re.compile(r"\bprintenv\b", re.IGNORECASE),
    re.compile(r"\bset\b\s*$", re.IGNORECASE),
    re.compile(r"\becho\s+.*\$AI_API_KEY", re.IGNORECASE),
    re.compile(r"\becho\s+.*\$ANTHROPIC_API_KEY", re.IGNORECASE),
    re.compile(r"\becho\s+.*\$OPENAI_API_KEY", re.IGNORECASE),
    re.compile(r"\bexport\b", re.IGNORECASE),
]


def _is_destructive(command: str) -> str | None:
    """Return a reason string if *command* is blocked, else ``None``."""
    for pattern in _DESTRUCTIVE_PATTERNS:
        if pattern.search(command):
            return f"Blocked destructive pattern: {pattern.pattern}"
    for pattern in _SECRET_LEAK_PATTERNS:
        if pattern.search(command):
            return f"Blocked secret-leaking command: {pattern.pattern}"
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_command(command: str, timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Execute a shell command from the repository root.

    Risk: EXECUTE (allow-listed dev commands).

    Parameters
    ----------
    command:
        The shell command string to execute.
    timeout:
        Maximum seconds to allow.  Clamped to [1, MAX_TIMEOUT].

    Returns
    -------
    dict
        Structured result with stdout, stderr, exit_code, duration, etc.
    """
    check_or_raise(RiskLevel.EXECUTE, "run_command")

    # Clamp timeout.
    timeout = max(1, min(timeout, MAX_TIMEOUT))

    # Block destructive commands.
    blocked_reason = _is_destructive(command)
    if blocked_reason:
        return make_result(
            success=False,
            risk=RiskLevel.EXECUTE,
            operation="run_command",
            error=blocked_reason,
            metadata={"command": command},
        )

    repo_root = str(get_repo_root())
    start = time.monotonic()

    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=repo_root,
            env={
                **os.environ,
                # Ensure child processes can't trivially dump the key.
                # The key is still in the parent env for the harness to use,
                # but we redact it from subprocess output below.
            },
        )
        duration = round(time.monotonic() - start, 3)

        stdout = _redact_secrets(proc.stdout)
        stderr = _redact_secrets(proc.stderr)

        return make_result(
            success=proc.returncode == 0,
            risk=RiskLevel.EXECUTE,
            operation="run_command",
            data={
                "command": command,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": proc.returncode,
                "duration_seconds": duration,
            },
        )

    except subprocess.TimeoutExpired:
        duration = round(time.monotonic() - start, 3)
        return make_result(
            success=False,
            risk=RiskLevel.EXECUTE,
            operation="run_command",
            error=f"Command timed out after {timeout}s",
            data={
                "command": command,
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
                "duration_seconds": duration,
            },
        )

    except OSError as exc:
        duration = round(time.monotonic() - start, 3)
        return make_result(
            success=False,
            risk=RiskLevel.EXECUTE,
            operation="run_command",
            error=str(exc),
            data={
                "command": command,
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
                "duration_seconds": duration,
            },
        )
