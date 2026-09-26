"""Repository Intelligence — Context Manager.

Replaces the simple conversation-history tracker with a full
**RepositoryContextManager** that can:

*  Build a lightweight repository map (README, config, deps, src, tests …).
*  Search for files relevant to a natural-language issue description.
*  Rank relevant files (filename, text search, imports, refs, tests, deps).
*  Return structured ``RelevantFile`` entries (path, reason, snippets, lines).
*  Respect a configurable context-token budget.
*  Cache repository metadata and refresh only what changed.
*  Build a dependency / impact map for any changed file.

The old ``ContextManager`` class is preserved for conversation tracking —
orchestrators can use both.
"""

from __future__ import annotations

import ast
import dataclasses
import fnmatch
import os
import pathlib
import re
from typing import Any

# ---------------------------------------------------------------------------
# Ignore patterns (shared with search.py, duplicated here to stay self-contained)
# ---------------------------------------------------------------------------

IGNORED_DIRS: frozenset[str] = frozenset({
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    "build", "dist", ".pytest_cache", ".mypy_cache", ".tox",
    ".eggs", "*.egg-info",
})


def _should_skip(dirname: str) -> bool:
    for pattern in IGNORED_DIRS:
        if fnmatch.fnmatch(dirname, pattern):
            return True
    return False


# ---------------------------------------------------------------------------
# Well-known file classifiers
# ---------------------------------------------------------------------------

_README_PATTERNS = re.compile(r"^readme(\.\w+)?$", re.IGNORECASE)
_CONFIG_PATTERNS = re.compile(
    r"^(config\.\w+|\..*rc|setup\.cfg|pyproject\.toml|"
    r"tsconfig\.json|package\.json|Makefile|Dockerfile|"
    r"\.env(\.\w+)?|tox\.ini|\.flake8|\.pylintrc)$",
    re.IGNORECASE,
)
_DEP_PATTERNS = re.compile(
    r"^(requirements.*\.txt|Pipfile(\.lock)?|poetry\.lock|"
    r"setup\.py|setup\.cfg|pyproject\.toml|package\.json|"
    r"package-lock\.json|yarn\.lock|go\.mod|Cargo\.toml)$",
    re.IGNORECASE,
)
_ENTRY_POINT_NAMES = frozenset({
    "main.py", "__main__.py", "app.py", "server.py", "cli.py",
    "manage.py", "wsgi.py", "asgi.py", "index.py", "run.py",
})
_TEST_DIR_NAMES = frozenset({"tests", "test", "spec", "specs"})
_SOURCE_DIR_NAMES = frozenset({"src", "lib", "app", "pkg", "core"})

# Rough chars-per-token estimate for budget enforcement.
CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FileInfo:
    """Cached metadata about a single repository file."""
    relative_path: str
    size_bytes: int
    category: str  # readme | config | dep | source | test | entry_point | other
    imports: list[str] = dataclasses.field(default_factory=list)
    definitions: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class RelevantFile:
    """A file selected as relevant to an issue, with evidence."""
    file_path: str
    relevance_score: float
    reason: str
    snippets: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    line_numbers: list[int] = dataclasses.field(default_factory=list)
    size_bytes: int = 0


@dataclasses.dataclass
class ImpactReport:
    """Estimated blast-radius of a set of changed files."""
    changed_files: list[str]
    directly_affected_files: list[str]
    indirectly_affected_files: list[str]
    related_tests: list[str]
    impact_level: str  # low | medium | high | critical
    reasons: list[str]


# ---------------------------------------------------------------------------
# Conversation tracker (preserved from the original implementation)
# ---------------------------------------------------------------------------

class ContextManager:
    """Maintains the message history and enforces the token window."""

    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        self.messages: list[dict] = []

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the conversation history."""
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        """Return the current message list."""
        return list(self.messages)

    def clear(self) -> None:
        """Reset the conversation history."""
        self.messages.clear()


# ---------------------------------------------------------------------------
# Repository Context Manager
# ---------------------------------------------------------------------------

class RepositoryContextManager:
    """Build and query a lightweight repository intelligence layer.

    Parameters
    ----------
    repo_root:
        Absolute path to the repository root.
    max_context_tokens:
        Token budget for context returned to the model.
    """

    def __init__(
        self,
        repo_root: str | pathlib.Path,
        max_context_tokens: int = 8192,
    ) -> None:
        self.repo_root = pathlib.Path(repo_root).resolve()
        self.max_context_tokens = max_context_tokens

        # Caches ---------------------------------------------------------
        self._file_cache: dict[str, FileInfo] = {}
        self._import_graph: dict[str, set[str]] = {}   # file → modules it imports
        self._reverse_deps: dict[str, set[str]] = {}    # module → files that import it
        self._repo_mapped: bool = False

    # ------------------------------------------------------------------
    # Repository mapping
    # ------------------------------------------------------------------

    def build_repo_map(self, *, force: bool = False) -> dict[str, Any]:
        """Walk the repository, classify every file, and cache the results.

        Returns a summary dict suitable for structured output.
        """
        if self._repo_mapped and not force:
            return self._summarise_map()

        self._file_cache.clear()
        self._import_graph.clear()
        self._reverse_deps.clear()

        for dirpath, dirnames, filenames in os.walk(self.repo_root):
            dirnames[:] = [d for d in dirnames if not _should_skip(d)]
            for fname in filenames:
                abs_path = pathlib.Path(dirpath) / fname
                rel = str(abs_path.relative_to(self.repo_root))
                try:
                    size = abs_path.stat().st_size
                except OSError:
                    continue

                category = self._classify(fname, rel)
                info = FileInfo(
                    relative_path=rel,
                    size_bytes=size,
                    category=category,
                )

                # For Python files extract imports and top-level definitions.
                if fname.endswith(".py"):
                    imports, defs = self._parse_python(abs_path)
                    info.imports = imports
                    info.definitions = defs
                    self._import_graph[rel] = set(imports)
                    for imp in imports:
                        self._reverse_deps.setdefault(imp, set()).add(rel)

                self._file_cache[rel] = info

        self._repo_mapped = True
        return self._summarise_map()

    def _summarise_map(self) -> dict[str, Any]:
        categories: dict[str, list[str]] = {}
        for info in self._file_cache.values():
            categories.setdefault(info.category, []).append(info.relative_path)
        return {
            "total_files": len(self._file_cache),
            "categories": {k: sorted(v) for k, v in categories.items()},
        }

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(filename: str, rel_path: str) -> str:
        if _README_PATTERNS.match(filename):
            return "readme"
        if _DEP_PATTERNS.match(filename):
            return "dep"
        if _CONFIG_PATTERNS.match(filename):
            return "config"
        if filename in _ENTRY_POINT_NAMES:
            return "entry_point"

        parts = pathlib.PurePosixPath(rel_path).parts
        for part in parts:
            if part.lower() in _TEST_DIR_NAMES or part.startswith("test"):
                return "test"
            if part.lower() in _SOURCE_DIR_NAMES:
                return "source"

        # Filename heuristics
        if filename.startswith("test_") or filename.endswith("_test.py"):
            return "test"
        if filename.endswith(".py"):
            return "source"

        return "other"

    # ------------------------------------------------------------------
    # Python AST helpers (safe, best-effort)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_python(filepath: pathlib.Path) -> tuple[list[str], list[str]]:
        """Return (imports, definitions) for a Python file.

        Falls back to empty lists on any parse error.
        """
        try:
            source = filepath.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(filepath))
        except (SyntaxError, ValueError, OSError):
            return [], []

        imports: list[str] = []
        definitions: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions.append(f"def {node.name}")
            elif isinstance(node, ast.ClassDef):
                definitions.append(f"class {node.name}")

        return imports, definitions

    # ------------------------------------------------------------------
    # Relevant-file search
    # ------------------------------------------------------------------

    def find_relevant_files(
        self,
        issue_text: str,
        *,
        max_files: int = 15,
    ) -> list[RelevantFile]:
        """Find and rank files relevant to *issue_text*.

        Scoring layers:
        1. Keyword matches in filename            (weight 3)
        2. Keyword matches in file content         (weight 2)
        3. Defines a function/class named in issue (weight 4)
        4. Is a test for a matching source file    (weight 1)
        5. Imports a matching module               (weight 1)
        """
        self.build_repo_map()

        keywords = self._extract_keywords(issue_text)
        if not keywords:
            return []

        scored: dict[str, tuple[float, list[str]]] = {}

        for rel_path, info in self._file_cache.items():
            score = 0.0
            reasons: list[str] = []

            # 1. Filename relevance
            fname_lower = pathlib.Path(rel_path).name.lower()
            for kw in keywords:
                if kw in fname_lower:
                    score += 3.0
                    reasons.append(f"filename contains '{kw}'")

            # 2. Content / text-search relevance
            abs_path = self.repo_root / rel_path
            content = self._read_safe(abs_path)
            if content is not None:
                content_lower = content.lower()
                for kw in keywords:
                    count = content_lower.count(kw)
                    if count:
                        score += min(count, 5) * 2.0
                        reasons.append(
                            f"content mentions '{kw}' ({count}×)"
                        )

            # 3. Definitions matching keywords
            for defn in info.definitions:
                defn_lower = defn.lower()
                for kw in keywords:
                    if kw in defn_lower:
                        score += 4.0
                        reasons.append(
                            f"defines {defn} (matches '{kw}')"
                        )

            # 4. Test references
            if info.category == "test":
                for kw in keywords:
                    if kw in fname_lower.replace("test_", ""):
                        score += 1.0
                        reasons.append(
                            f"test file related to '{kw}'"
                        )

            # 5. Import references
            for imp in info.imports:
                imp_lower = imp.lower()
                for kw in keywords:
                    if kw in imp_lower:
                        score += 1.0
                        reasons.append(
                            f"imports module matching '{kw}'"
                        )

            if score > 0:
                scored[rel_path] = (score, reasons)

        # Sort by score descending, then alphabetically for ties.
        ranked = sorted(
            scored.items(),
            key=lambda item: (-item[1][0], item[0]),
        )

        # Apply context budget.
        results: list[RelevantFile] = []
        budget_chars = self.max_context_tokens * CHARS_PER_TOKEN
        used_chars = 0

        for rel_path, (score, reasons) in ranked[:max_files]:
            info = self._file_cache[rel_path]
            # Estimate cost.
            file_chars = info.size_bytes  # rough: 1 byte ≈ 1 char for UTF-8 text
            if used_chars + file_chars > budget_chars and results:
                break
            used_chars += file_chars

            # Build snippets for keyword matches.
            snippets, line_numbers = self._extract_snippets(
                self.repo_root / rel_path, keywords
            )

            reason_text = "; ".join(dict.fromkeys(reasons))  # deduplicate, preserve order

            results.append(RelevantFile(
                file_path=rel_path,
                relevance_score=round(score, 2),
                reason=reason_text,
                snippets=snippets,
                line_numbers=line_numbers,
                size_bytes=info.size_bytes,
            ))

        return results

    # ------------------------------------------------------------------
    # Dependency / Impact Map
    # ------------------------------------------------------------------

    def build_impact_report(
        self,
        changed_files: list[str],
    ) -> ImpactReport:
        """Estimate the blast-radius of changes to *changed_files*.

        Uses the cached import graph to determine direct and indirect
        dependants, then locates related test files.
        """
        self.build_repo_map()

        direct: set[str] = set()
        indirect: set[str] = set()
        tests: set[str] = set()
        reasons: list[str] = []

        for changed in changed_files:
            info = self._file_cache.get(changed)
            if info is None:
                reasons.append(f"{changed}: not found in repository map")
                continue

            # Which modules does this file conceptually provide?
            module_names = self._file_to_module_names(changed)

            # Direct dependants: files that import this module.
            for mod_name in module_names:
                for dep_file in self._reverse_deps.get(mod_name, set()):
                    if dep_file not in changed_files:
                        dep_info = self._file_cache.get(dep_file)
                        if dep_info and dep_info.category == "test":
                            tests.add(dep_file)
                            reasons.append(
                                f"{dep_file}: test that imports {mod_name}"
                            )
                        else:
                            direct.add(dep_file)
                            reasons.append(
                                f"{dep_file}: imports {mod_name} (direct dependency)"
                            )

            # Indirect dependants: files that import *direct* dependants.
            for d_file in list(direct):
                d_modules = self._file_to_module_names(d_file)
                for d_mod in d_modules:
                    for id_file in self._reverse_deps.get(d_mod, set()):
                        if id_file not in changed_files and id_file not in direct:
                            id_info = self._file_cache.get(id_file)
                            if id_info and id_info.category == "test":
                                tests.add(id_file)
                                reasons.append(
                                    f"{id_file}: test for indirect dep {d_mod}"
                                )
                            else:
                                indirect.add(id_file)
                                reasons.append(
                                    f"{id_file}: indirectly affected via {d_file}"
                                )

            # Also find tests by naming convention: test_<stem>.py
            for changed in changed_files:
                stem = pathlib.Path(changed).stem
                for rel_path, finfo in self._file_cache.items():
                    if finfo.category == "test":
                        test_name = pathlib.Path(rel_path).stem
                        if (
                            test_name == f"test_{stem}"
                            or test_name == f"{stem}_test"
                        ):
                            tests.add(rel_path)
                            reasons.append(
                                f"{rel_path}: test file matches {changed} by name"
                            )

        # Determine impact level.
        total_affected = len(direct) + len(indirect) + len(tests)
        if total_affected == 0:
            level = "low"
        elif total_affected <= 3:
            level = "medium"
        elif total_affected <= 8:
            level = "high"
        else:
            level = "critical"

        # Deduplicate reasons while preserving order.
        seen: set[str] = set()
        unique_reasons: list[str] = []
        for r in reasons:
            if r not in seen:
                seen.add(r)
                unique_reasons.append(r)

        return ImpactReport(
            changed_files=sorted(changed_files),
            directly_affected_files=sorted(direct),
            indirectly_affected_files=sorted(indirect),
            related_tests=sorted(tests),
            impact_level=level,
            reasons=unique_reasons,
        )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def refresh_files(self, changed_paths: list[str]) -> None:
        """Re-scan only the listed files instead of rebuilding everything."""
        for rel_path in changed_paths:
            abs_path = self.repo_root / rel_path

            # Remove old data.
            old_info = self._file_cache.pop(rel_path, None)
            if old_info:
                self._import_graph.pop(rel_path, None)
                for imp in old_info.imports:
                    rev = self._reverse_deps.get(imp)
                    if rev:
                        rev.discard(rel_path)

            if not abs_path.exists():
                continue

            try:
                size = abs_path.stat().st_size
            except OSError:
                continue

            fname = abs_path.name
            category = self._classify(fname, rel_path)
            info = FileInfo(
                relative_path=rel_path,
                size_bytes=size,
                category=category,
            )

            if fname.endswith(".py"):
                imports, defs = self._parse_python(abs_path)
                info.imports = imports
                info.definitions = defs
                self._import_graph[rel_path] = set(imports)
                for imp in imports:
                    self._reverse_deps.setdefault(imp, set()).add(rel_path)

            self._file_cache[rel_path] = info

    def invalidate(self) -> None:
        """Drop all cached data — forces a fresh scan on next access."""
        self._file_cache.clear()
        self._import_graph.clear()
        self._reverse_deps.clear()
        self._repo_mapped = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Split issue text into lowercase search keywords.

        Filters out very short words and common stop-words.
        """
        stop = frozenset({
            "the", "and", "for", "that", "this", "with", "from",
            "are", "was", "were", "been", "have", "has", "had",
            "not", "but", "its", "you", "your", "can", "will",
            "should", "would", "could", "does", "did", "all",
            "into", "each", "than", "also", "when", "how",
            "add", "fix", "bug", "issue", "error", "file",
        })
        words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())
        seen: set[str] = set()
        result: list[str] = []
        for w in words:
            if len(w) >= 3 and w not in stop and w not in seen:
                seen.add(w)
                result.append(w)
        return result

    @staticmethod
    def _read_safe(filepath: pathlib.Path) -> str | None:
        try:
            return filepath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def _extract_snippets(
        self,
        filepath: pathlib.Path,
        keywords: list[str],
        context_lines: int = 2,
        max_snippets: int = 5,
    ) -> tuple[list[dict[str, Any]], list[int]]:
        """Return (snippets, line_numbers) for keyword matches."""
        content = self._read_safe(filepath)
        if content is None:
            return [], []

        lines = content.splitlines()
        matched_lines: list[int] = []

        for lineno_0, line in enumerate(lines):
            line_lower = line.lower()
            if any(kw in line_lower for kw in keywords):
                matched_lines.append(lineno_0 + 1)

        snippets: list[dict[str, Any]] = []
        for lineno in matched_lines[:max_snippets]:
            start = max(0, lineno - 1 - context_lines)
            end = min(len(lines), lineno + context_lines)
            snippets.append({
                "line": lineno,
                "text": "\n".join(lines[start:end]),
            })

        return snippets, matched_lines[:max_snippets]

    @staticmethod
    def _file_to_module_names(rel_path: str) -> list[str]:
        """Convert a relative file path to plausible Python module names.

        ``src/tools/file_ops.py`` → ``["src.tools.file_ops", "tools.file_ops", "file_ops"]``
        """
        p = pathlib.PurePosixPath(rel_path)
        if p.suffix != ".py":
            return []

        # Strip .py, convert slashes to dots.
        parts = list(p.with_suffix("").parts)
        results: list[str] = []
        for i in range(len(parts)):
            mod = ".".join(parts[i:])
            if mod != "__init__":
                results.append(mod)
        return results
