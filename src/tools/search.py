"""Tool — Repository search (text search and filename search).

Searches are scoped to the repository root and automatically skip
directories that are not useful for code analysis (`.git`, `__pycache__`,
`node_modules`, etc.).

Every public function returns a structured result via ``risk.make_result``.
"""

from __future__ import annotations

import fnmatch
import os
import pathlib
import re
from typing import Any

from src.tools.risk import (
    RiskLevel,
    check_or_raise,
    get_repo_root,
    make_result,
    resolve_safe_path,
)

# ---------------------------------------------------------------------------
# Ignore patterns — directories and files skipped during search
# ---------------------------------------------------------------------------

IGNORED_DIRS: frozenset[str] = frozenset({
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".eggs",
    "*.egg-info",
})

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_RESULTS = 50
SNIPPET_CONTEXT_LINES = 2  # lines of context above/below each match

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _should_skip_dir(dirname: str) -> bool:
    """Return True if the directory name matches any ignore pattern."""
    for pattern in IGNORED_DIRS:
        if fnmatch.fnmatch(dirname, pattern):
            return True
    return False


def _read_lines_safe(filepath: pathlib.Path) -> list[str] | None:
    """Read file lines, returning ``None`` on decode/OS errors."""
    try:
        return filepath.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None


def _extract_snippet(
    lines: list[str], match_lineno: int, context: int = SNIPPET_CONTEXT_LINES
) -> str:
    """Return a snippet around *match_lineno* (1-indexed)."""
    start = max(0, match_lineno - 1 - context)
    end = min(len(lines), match_lineno + context)
    return "\n".join(lines[start:end])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def text_search(
    query: str,
    *,
    path: str = ".",
    glob_pattern: str = "*",
    max_results: int = DEFAULT_MAX_RESULTS,
    use_regex: bool = False,
) -> dict[str, Any]:
    """Search file contents for *query*.

    Risk: READ

    Parameters
    ----------
    query:
        The search string (literal or regex when *use_regex* is ``True``).
    path:
        Subdirectory to search within (relative to repo root).
    glob_pattern:
        File-name glob filter, e.g. ``"*.py"``.
    max_results:
        Cap the number of returned matches.
    use_regex:
        Treat *query* as a regular expression.

    Returns
    -------
    dict
        Structured result with a list of match dicts (file, line, content,
        snippet).
    """
    check_or_raise(RiskLevel.READ, "text_search")
    repo_root = get_repo_root()

    try:
        search_root = resolve_safe_path(path, repo_root)
    except ValueError as exc:
        return make_result(
            success=False,
            risk=RiskLevel.READ,
            operation="text_search",
            error=str(exc),
        )

    if not search_root.is_dir():
        return make_result(
            success=False,
            risk=RiskLevel.READ,
            operation="text_search",
            error=f"Not a directory: {path!r}",
        )

    # Compile pattern.
    try:
        if use_regex:
            pattern = re.compile(query)
        else:
            pattern = re.compile(re.escape(query))
    except re.error as exc:
        return make_result(
            success=False,
            risk=RiskLevel.READ,
            operation="text_search",
            error=f"Invalid regex: {exc}",
        )

    matches: list[dict[str, Any]] = []

    for dirpath, dirnames, filenames in os.walk(search_root):
        # Prune ignored directories in-place so os.walk skips them.
        dirnames[:] = [
            d for d in dirnames if not _should_skip_dir(d)
        ]

        for filename in filenames:
            if glob_pattern != "*" and not fnmatch.fnmatch(filename, glob_pattern):
                continue

            filepath = pathlib.Path(dirpath) / filename
            lines = _read_lines_safe(filepath)
            if lines is None:
                continue

            for lineno_0, line in enumerate(lines):
                if pattern.search(line):
                    lineno = lineno_0 + 1
                    rel_path = str(filepath.relative_to(repo_root))
                    matches.append({
                        "file": rel_path,
                        "line": lineno,
                        "content": line.rstrip(),
                        "snippet": _extract_snippet(lines, lineno),
                    })
                    if len(matches) >= max_results:
                        break
            if len(matches) >= max_results:
                break
        if len(matches) >= max_results:
            break

    return make_result(
        success=True,
        risk=RiskLevel.READ,
        operation="text_search",
        data=matches,
        metadata={
            "query": query,
            "total_matches": len(matches),
            "max_results": max_results,
            "truncated": len(matches) >= max_results,
        },
    )


def filename_search(
    query: str,
    *,
    path: str = ".",
    max_results: int = DEFAULT_MAX_RESULTS,
    use_regex: bool = False,
) -> dict[str, Any]:
    """Search for files whose **name** matches *query*.

    Risk: READ

    Parameters
    ----------
    query:
        Substring or regex to match against file/dir names.
    path:
        Subdirectory to search within (relative to repo root).
    max_results:
        Cap the number of returned matches.
    use_regex:
        Treat *query* as a regular expression.

    Returns
    -------
    dict
        Structured result with a list of matching relative paths.
    """
    check_or_raise(RiskLevel.READ, "filename_search")
    repo_root = get_repo_root()

    try:
        search_root = resolve_safe_path(path, repo_root)
    except ValueError as exc:
        return make_result(
            success=False,
            risk=RiskLevel.READ,
            operation="filename_search",
            error=str(exc),
        )

    if not search_root.is_dir():
        return make_result(
            success=False,
            risk=RiskLevel.READ,
            operation="filename_search",
            error=f"Not a directory: {path!r}",
        )

    try:
        if use_regex:
            pattern = re.compile(query)
        else:
            pattern = re.compile(re.escape(query))
    except re.error as exc:
        return make_result(
            success=False,
            risk=RiskLevel.READ,
            operation="filename_search",
            error=f"Invalid regex: {exc}",
        )

    matches: list[dict[str, Any]] = []

    for dirpath, dirnames, filenames in os.walk(search_root):
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]

        for name in filenames + dirnames:
            if pattern.search(name):
                entry_path = pathlib.Path(dirpath) / name
                rel_path = str(entry_path.relative_to(repo_root))
                matches.append({
                    "name": name,
                    "relative_path": rel_path,
                    "is_dir": entry_path.is_dir(),
                })
                if len(matches) >= max_results:
                    break
        if len(matches) >= max_results:
            break

    return make_result(
        success=True,
        risk=RiskLevel.READ,
        operation="filename_search",
        data=matches,
        metadata={
            "query": query,
            "total_matches": len(matches),
            "max_results": max_results,
            "truncated": len(matches) >= max_results,
        },
    )
