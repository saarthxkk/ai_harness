"""Metrics & Caching — Efficiency infrastructure for the harness.

Provides:
    ``ExecutionMetrics``  — structured counters for evaluation reporting.
    ``FileCache``         — deduplicates file reads with hash-based invalidation.
    ``VerificationCache`` — skips redundant verification runs when files haven't changed.
    ``ContextBudget``     — bounds the approximate context size sent to the model.

Design constraints:
    * Never sacrifice verification quality to reduce calls.
    * Only skip verification when file fingerprints prove nothing changed.
    * Invalidation is always conservative (invalidate on any doubt).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Execution Metrics
# ---------------------------------------------------------------------------

@dataclass
class ExecutionMetrics:
    """Structured metrics for a single task execution.

    All counters are incremented by the orchestrator and displayed
    in the final report.
    """
    model_calls: int = 0
    tool_calls: int = 0
    files_inspected: int = 0
    files_changed: int = 0
    verification_runs: int = 0
    falsification_runs: int = 0
    repair_attempts: int = 0
    stale_evidence_events: int = 0
    approximate_context_size: int = 0
    execution_duration: float = 0.0

    # Efficiency sub-metrics.
    cache_hits: int = 0
    cache_misses: int = 0
    skipped_verifications: int = 0
    skipped_rescans: int = 0

    def record_model_call(self, context_chars: int = 0) -> None:
        """Record a model API call with approximate context size."""
        self.model_calls += 1
        self.approximate_context_size += context_chars

    def record_tool_call(self) -> None:
        self.tool_calls += 1

    def record_file_inspected(self) -> None:
        self.files_inspected += 1

    def record_files_changed(self, count: int) -> None:
        self.files_changed += count

    def record_verification(self) -> None:
        self.verification_runs += 1

    def record_falsification(self) -> None:
        self.falsification_runs += 1

    def record_repair(self) -> None:
        self.repair_attempts += 1

    def record_stale_evidence(self) -> None:
        self.stale_evidence_events += 1

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        self.cache_misses += 1

    def record_skipped_verification(self) -> None:
        self.skipped_verifications += 1

    def record_skipped_rescan(self) -> None:
        self.skipped_rescans += 1

    @property
    def cache_hit_rate(self) -> float:
        """Return cache hit rate as a float in [0.0, 1.0]."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "files_inspected": self.files_inspected,
            "files_changed": self.files_changed,
            "verification_runs": self.verification_runs,
            "falsification_runs": self.falsification_runs,
            "repair_attempts": self.repair_attempts,
            "stale_evidence_events": self.stale_evidence_events,
            "approximate_context_size": self.approximate_context_size,
            "execution_duration": round(self.execution_duration, 3),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hit_rate, 3),
            "skipped_verifications": self.skipped_verifications,
            "skipped_rescans": self.skipped_rescans,
        }

    def summary_lines(self) -> list[str]:
        """Return human-readable metric lines for the TUI."""
        lines = [
            f"Model calls:          {self.model_calls}",
            f"Tool calls:           {self.tool_calls}",
            f"Files inspected:      {self.files_inspected}",
            f"Files changed:        {self.files_changed}",
            f"Verification runs:    {self.verification_runs}",
            f"Falsification runs:   {self.falsification_runs}",
            f"Repair attempts:      {self.repair_attempts}",
            f"Stale evidence:       {self.stale_evidence_events}",
            f"Context size:         ~{self.approximate_context_size} chars",
            f"Duration:             {self.execution_duration:.1f}s",
        ]
        if self.cache_hits + self.cache_misses > 0:
            lines.append(
                f"Cache hit rate:       {self.cache_hit_rate:.0%} "
                f"({self.cache_hits}/{self.cache_hits + self.cache_misses})"
            )
        if self.skipped_verifications > 0:
            lines.append(
                f"Skipped verifications: {self.skipped_verifications}"
            )
        if self.skipped_rescans > 0:
            lines.append(
                f"Skipped rescans:      {self.skipped_rescans}"
            )
        return lines


# ---------------------------------------------------------------------------
# File Cache — deduplicates reads with hash-based invalidation
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    """A single cached file entry."""
    content: str
    sha256: str
    size_bytes: int
    read_at: float


class FileCache:
    """In-memory content cache keyed by file path.

    On ``get(path)``, returns cached content if the hash matches.
    Avoids re-reading unchanged files from disk.

    Invalidation:
        * ``invalidate(path)`` — remove a single path.
        * ``invalidate_changed(paths)`` — remove only changed paths.
        * ``clear()`` — drop everything.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _CacheEntry] = {}

    def put(self, path: str, content: str) -> str:
        """Store content and return its SHA-256 hash."""
        h = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
        self._entries[path] = _CacheEntry(
            content=content,
            sha256=h,
            size_bytes=len(content.encode("utf-8", errors="replace")),
            read_at=time.time(),
        )
        return h

    def get(self, path: str) -> str | None:
        """Return cached content or ``None`` if not cached."""
        entry = self._entries.get(path)
        if entry is None:
            return None
        return entry.content

    def get_hash(self, path: str) -> str | None:
        """Return the cached SHA-256 hash or ``None``."""
        entry = self._entries.get(path)
        if entry is None:
            return None
        return entry.sha256

    def has(self, path: str) -> bool:
        return path in self._entries

    def is_unchanged(self, path: str, current_hash: str) -> bool:
        """Return ``True`` if cached hash matches *current_hash*."""
        entry = self._entries.get(path)
        if entry is None:
            return False
        return entry.sha256 == current_hash

    def invalidate(self, path: str) -> bool:
        """Remove a single path from the cache. Returns True if it existed."""
        return self._entries.pop(path, None) is not None

    def invalidate_changed(self, changed_paths: list[str]) -> int:
        """Invalidate only the listed paths. Returns count invalidated."""
        count = 0
        for p in changed_paths:
            if self.invalidate(p):
                count += 1
        return count

    def clear(self) -> None:
        """Drop all cached entries."""
        self._entries.clear()

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return len(self._entries)

    @property
    def total_bytes(self) -> int:
        """Total bytes of cached content."""
        return sum(e.size_bytes for e in self._entries.values())

    def paths(self) -> list[str]:
        """Return all cached paths."""
        return list(self._entries.keys())

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": self.size,
            "total_bytes": self.total_bytes,
            "paths": self.paths(),
        }


# ---------------------------------------------------------------------------
# Verification Cache — skip redundant verification
# ---------------------------------------------------------------------------

@dataclass
class _VerificationSnapshot:
    """Fingerprint snapshot at the time of a verification run."""
    file_hashes: dict[str, str]  # path → sha256
    passed: bool
    timestamp: float


class VerificationCache:
    """Tracks file fingerprints to avoid re-running verification
    when no files have changed since the last successful run.

    IMPORTANT: This never skips verification after a failed run.
    It only skips when ALL of these hold:
      1. The previous verification PASSED.
      2. No file in ``changed_files`` has a different hash.
      3. No new files were added to the changeset.

    This is conservative — if in doubt, re-verify.
    """

    def __init__(self) -> None:
        self._last_snapshot: _VerificationSnapshot | None = None

    def record(
        self, file_hashes: dict[str, str], passed: bool,
    ) -> None:
        """Record the outcome of a verification run."""
        self._last_snapshot = _VerificationSnapshot(
            file_hashes=dict(file_hashes),
            passed=passed,
            timestamp=time.time(),
        )

    def can_skip(self, current_hashes: dict[str, str]) -> bool:
        """Return ``True`` only if re-verification is provably unnecessary.

        Conservative: returns ``False`` on any doubt.
        """
        if self._last_snapshot is None:
            return False
        if not self._last_snapshot.passed:
            return False

        # Every file in the current changeset must have the same hash.
        for path, h in current_hashes.items():
            old_h = self._last_snapshot.file_hashes.get(path)
            if old_h is None:
                # New file added — must re-verify.
                return False
            if old_h != h:
                return False

        return True

    def invalidate(self) -> None:
        """Force re-verification on the next run."""
        self._last_snapshot = None

    @property
    def has_snapshot(self) -> bool:
        return self._last_snapshot is not None

    @property
    def last_passed(self) -> bool | None:
        if self._last_snapshot is None:
            return None
        return self._last_snapshot.passed


# ---------------------------------------------------------------------------
# Context Budget — bounds model context size
# ---------------------------------------------------------------------------

class ContextBudget:
    """Bounds the approximate context sent to the model.

    Tracks cumulative characters added and refuses additions
    that would exceed the budget.

    Parameters
    ----------
    max_chars : int
        Maximum total characters.  Default 100,000 (~25K tokens).
    """

    def __init__(self, max_chars: int = 100_000) -> None:
        self._max_chars = max_chars
        self._used_chars = 0
        self._items: list[str] = []

    def can_add(self, text: str) -> bool:
        """Return ``True`` if *text* fits within the remaining budget."""
        return self._used_chars + len(text) <= self._max_chars

    def add(self, text: str) -> bool:
        """Add *text* to the context if it fits.  Returns ``True`` if added."""
        if not self.can_add(text):
            return False
        self._used_chars += len(text)
        self._items.append(text)
        return True

    def force_add(self, text: str) -> None:
        """Add *text* unconditionally (for essential context)."""
        self._used_chars += len(text)
        self._items.append(text)

    @property
    def used(self) -> int:
        return self._used_chars

    @property
    def remaining(self) -> int:
        return max(0, self._max_chars - self._used_chars)

    @property
    def utilisation(self) -> float:
        if self._max_chars == 0:
            return 1.0
        return self._used_chars / self._max_chars

    @property
    def item_count(self) -> int:
        return len(self._items)

    def reset(self) -> None:
        """Clear all tracked context."""
        self._used_chars = 0
        self._items.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_chars": self._max_chars,
            "used_chars": self._used_chars,
            "remaining_chars": self.remaining,
            "utilisation": round(self.utilisation, 3),
            "items": self.item_count,
        }
