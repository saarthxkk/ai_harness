"""Tool risk classification and policy enforcement.

Every tool operation carries a risk level.  The harness uses these
levels to decide what may run autonomously and what must be blocked.

Risk levels
-----------
READ        – Read-only access (file reads, searches).  Always allowed.
WRITE       – Creates or modifies files.  Allowed inside the repo scope.
EXECUTE     – Runs a subprocess.  Allowed for allow-listed dev commands.
DESTRUCTIVE – Deletes files or could cause data loss.  Blocked unless the
              harness explicitly permits it.
NETWORK     – Makes outbound network requests.  Disabled by default.
"""

from __future__ import annotations

import enum
import pathlib
from typing import Any


# ---------------------------------------------------------------------------
# Risk enum
# ---------------------------------------------------------------------------

class RiskLevel(enum.Enum):
    READ = "READ"
    WRITE = "WRITE"
    EXECUTE = "EXECUTE"
    DESTRUCTIVE = "DESTRUCTIVE"
    NETWORK = "NETWORK"


# ---------------------------------------------------------------------------
# Policy defaults (can be overridden by the orchestrator)
# ---------------------------------------------------------------------------

_DEFAULT_POLICY: dict[RiskLevel, bool] = {
    RiskLevel.READ: True,
    RiskLevel.WRITE: True,
    RiskLevel.EXECUTE: True,
    RiskLevel.DESTRUCTIVE: False,
    RiskLevel.NETWORK: False,
}

_policy = dict(_DEFAULT_POLICY)


def set_policy(level: RiskLevel, allowed: bool) -> None:
    """Override the default policy for a risk level."""
    _policy[level] = allowed


def is_allowed(level: RiskLevel) -> bool:
    """Return ``True`` if operations at *level* are currently permitted."""
    return _policy.get(level, False)


def check_or_raise(level: RiskLevel, description: str = "") -> None:
    """Raise ``PermissionError`` if the risk level is blocked."""
    if not is_allowed(level):
        msg = f"Operation blocked by policy: risk={level.value}"
        if description:
            msg += f" ({description})"
        raise PermissionError(msg)


def reset_policy() -> None:
    """Restore default policy (useful in tests)."""
    global _policy
    _policy = dict(_DEFAULT_POLICY)


# ---------------------------------------------------------------------------
# Repository root resolution
# ---------------------------------------------------------------------------

def get_repo_root() -> pathlib.Path:
    """Return the resolved repository root (directory containing config.yaml)."""
    return pathlib.Path(__file__).resolve().parent.parent.parent


def resolve_safe_path(user_path: str, repo_root: pathlib.Path | None = None) -> pathlib.Path:
    """Resolve *user_path* relative to *repo_root* and verify it stays inside.

    Raises ``ValueError`` for paths that escape the repository via ``..``
    traversal or by being absolute paths outside the repo.
    """
    if repo_root is None:
        repo_root = get_repo_root()

    raw = pathlib.Path(user_path)

    # Block absolute paths that don't start with the repo root.
    if raw.is_absolute():
        resolved = raw.resolve()
    else:
        resolved = (repo_root / raw).resolve()

    # Ensure the resolved path is inside the repo root.
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError:
        raise ValueError(
            f"Path escapes the repository: {user_path!r} "
            f"(resolved to {resolved})"
        )

    return resolved


# ---------------------------------------------------------------------------
# Structured result helper
# ---------------------------------------------------------------------------

def make_result(
    *,
    success: bool,
    risk: RiskLevel,
    operation: str,
    path: str | None = None,
    data: Any = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured, machine-readable tool result dict."""
    result: dict[str, Any] = {
        "success": success,
        "risk": risk.value,
        "operation": operation,
    }
    if path is not None:
        result["path"] = path
    if data is not None:
        result["data"] = data
    if error is not None:
        result["error"] = error
    if metadata:
        result["metadata"] = metadata
    return result
