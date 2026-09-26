"""Tool — File Operations (list, read, write, create, delete, exists).

Every operation:
* validates that the target path stays inside the repository root;
* checks the risk-level policy before proceeding;
* returns a structured result dict via ``risk.make_result``.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

from src.tools.risk import (
    RiskLevel,
    check_or_raise,
    get_repo_root,
    make_result,
    resolve_safe_path,
)

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

MAX_READ_BYTES = 5 * 1024 * 1024   # 5 MiB
MAX_WRITE_BYTES = 5 * 1024 * 1024  # 5 MiB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_metadata(p: pathlib.Path) -> dict[str, Any]:
    """Return a dict of useful file metadata."""
    try:
        stat = p.stat()
        return {
            "size_bytes": stat.st_size,
            "is_file": p.is_file(),
            "is_dir": p.is_dir(),
        }
    except OSError:
        return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_files(path: str = ".") -> dict[str, Any]:
    """List files and directories under *path* (relative to repo root).

    Risk: READ
    """
    check_or_raise(RiskLevel.READ, "list_files")
    repo_root = get_repo_root()

    try:
        target = resolve_safe_path(path, repo_root)
    except ValueError as exc:
        return make_result(
            success=False,
            risk=RiskLevel.READ,
            operation="list_files",
            path=path,
            error=str(exc),
        )

    if not target.is_dir():
        return make_result(
            success=False,
            risk=RiskLevel.READ,
            operation="list_files",
            path=path,
            error=f"Not a directory: {path!r}",
        )

    entries: list[dict[str, Any]] = []
    try:
        for entry in sorted(target.iterdir()):
            rel = str(entry.relative_to(repo_root))
            entries.append({
                "name": entry.name,
                "relative_path": rel,
                "is_dir": entry.is_dir(),
            })
    except OSError as exc:
        return make_result(
            success=False,
            risk=RiskLevel.READ,
            operation="list_files",
            path=path,
            error=str(exc),
        )

    return make_result(
        success=True,
        risk=RiskLevel.READ,
        operation="list_files",
        path=path,
        data=entries,
        metadata={"count": len(entries)},
    )


def read_file(path: str) -> dict[str, Any]:
    """Read and return the UTF-8 contents of a file.

    Risk: READ
    """
    check_or_raise(RiskLevel.READ, "read_file")
    repo_root = get_repo_root()

    try:
        target = resolve_safe_path(path, repo_root)
    except ValueError as exc:
        return make_result(
            success=False,
            risk=RiskLevel.READ,
            operation="read_file",
            path=path,
            error=str(exc),
        )

    if not target.is_file():
        return make_result(
            success=False,
            risk=RiskLevel.READ,
            operation="read_file",
            path=path,
            error=f"File not found: {path!r}",
        )

    size = target.stat().st_size
    if size > MAX_READ_BYTES:
        return make_result(
            success=False,
            risk=RiskLevel.READ,
            operation="read_file",
            path=path,
            error=f"File too large ({size} bytes, limit {MAX_READ_BYTES})",
            metadata={"size_bytes": size, "limit_bytes": MAX_READ_BYTES},
        )

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return make_result(
            success=False,
            risk=RiskLevel.READ,
            operation="read_file",
            path=path,
            error=str(exc),
        )

    return make_result(
        success=True,
        risk=RiskLevel.READ,
        operation="read_file",
        path=path,
        data=content,
        metadata=_file_metadata(target),
    )


def write_file(path: str, content: str) -> dict[str, Any]:
    """Write *content* to an **existing** file (overwrite).

    Risk: WRITE
    """
    check_or_raise(RiskLevel.WRITE, "write_file")
    repo_root = get_repo_root()

    try:
        target = resolve_safe_path(path, repo_root)
    except ValueError as exc:
        return make_result(
            success=False,
            risk=RiskLevel.WRITE,
            operation="write_file",
            path=path,
            error=str(exc),
        )

    encoded = content.encode("utf-8")
    if len(encoded) > MAX_WRITE_BYTES:
        return make_result(
            success=False,
            risk=RiskLevel.WRITE,
            operation="write_file",
            path=path,
            error=f"Content too large ({len(encoded)} bytes, limit {MAX_WRITE_BYTES})",
        )

    if not target.exists():
        return make_result(
            success=False,
            risk=RiskLevel.WRITE,
            operation="write_file",
            path=path,
            error=f"File does not exist: {path!r}. Use create_file for new files.",
        )

    try:
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return make_result(
            success=False,
            risk=RiskLevel.WRITE,
            operation="write_file",
            path=path,
            error=str(exc),
        )

    return make_result(
        success=True,
        risk=RiskLevel.WRITE,
        operation="write_file",
        path=path,
        metadata=_file_metadata(target),
    )


def create_file(path: str, content: str = "") -> dict[str, Any]:
    """Create a new file with *content*.  Parent directories are created.

    Risk: WRITE
    """
    check_or_raise(RiskLevel.WRITE, "create_file")
    repo_root = get_repo_root()

    try:
        target = resolve_safe_path(path, repo_root)
    except ValueError as exc:
        return make_result(
            success=False,
            risk=RiskLevel.WRITE,
            operation="create_file",
            path=path,
            error=str(exc),
        )

    encoded = content.encode("utf-8")
    if len(encoded) > MAX_WRITE_BYTES:
        return make_result(
            success=False,
            risk=RiskLevel.WRITE,
            operation="create_file",
            path=path,
            error=f"Content too large ({len(encoded)} bytes, limit {MAX_WRITE_BYTES})",
        )

    if target.exists():
        return make_result(
            success=False,
            risk=RiskLevel.WRITE,
            operation="create_file",
            path=path,
            error=f"File already exists: {path!r}. Use write_file to overwrite.",
        )

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return make_result(
            success=False,
            risk=RiskLevel.WRITE,
            operation="create_file",
            path=path,
            error=str(exc),
        )

    return make_result(
        success=True,
        risk=RiskLevel.WRITE,
        operation="create_file",
        path=path,
        metadata=_file_metadata(target),
    )


def delete_file(path: str) -> dict[str, Any]:
    """Delete a file inside the repository.

    Risk: DESTRUCTIVE — blocked by default policy.
    """
    check_or_raise(RiskLevel.DESTRUCTIVE, "delete_file")
    repo_root = get_repo_root()

    try:
        target = resolve_safe_path(path, repo_root)
    except ValueError as exc:
        return make_result(
            success=False,
            risk=RiskLevel.DESTRUCTIVE,
            operation="delete_file",
            path=path,
            error=str(exc),
        )

    if not target.exists():
        return make_result(
            success=False,
            risk=RiskLevel.DESTRUCTIVE,
            operation="delete_file",
            path=path,
            error=f"File not found: {path!r}",
        )

    if not target.is_file():
        return make_result(
            success=False,
            risk=RiskLevel.DESTRUCTIVE,
            operation="delete_file",
            path=path,
            error=f"Not a file (directory deletion not supported): {path!r}",
        )

    try:
        target.unlink()
    except OSError as exc:
        return make_result(
            success=False,
            risk=RiskLevel.DESTRUCTIVE,
            operation="delete_file",
            path=path,
            error=str(exc),
        )

    return make_result(
        success=True,
        risk=RiskLevel.DESTRUCTIVE,
        operation="delete_file",
        path=path,
    )


def file_exists(path: str) -> dict[str, Any]:
    """Check whether a path exists inside the repository.

    Risk: READ
    """
    check_or_raise(RiskLevel.READ, "file_exists")
    repo_root = get_repo_root()

    try:
        target = resolve_safe_path(path, repo_root)
    except ValueError as exc:
        return make_result(
            success=False,
            risk=RiskLevel.READ,
            operation="file_exists",
            path=path,
            error=str(exc),
        )

    exists = target.exists()
    return make_result(
        success=True,
        risk=RiskLevel.READ,
        operation="file_exists",
        path=path,
        data=exists,
        metadata=_file_metadata(target) if exists else {},
    )
