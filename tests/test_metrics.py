"""Tests for src/metrics.py — ExecutionMetrics, FileCache, VerificationCache, ContextBudget.

Covers:
  - ExecutionMetrics structured counters
  - ExecutionMetrics summary output
  - FileCache put/get/invalidation
  - FileCache hash-based unchanged detection
  - FileCache selective invalidation
  - VerificationCache skip logic
  - VerificationCache conservative invalidation
  - ContextBudget bounds
  - Orchestrator metrics integration
  - Verification cache skip in orchestrator
"""

import time
from unittest.mock import MagicMock

import pytest

from src.metrics import (
    ContextBudget,
    ExecutionMetrics,
    FileCache,
    VerificationCache,
)
from src.orchestrator import (
    Orchestrator,
    TaskOutcome,
    Phase,
)
from src.verifier import (
    CheckStatus,
    VerificationCheck,
    VerificationLevel,
    VerificationResult,
    VerificationStatus,
    Verifier,
)
from src.contract_engine import (
    ContractEngine,
    ContractValidationError,
    ObligationStatus,
    ObligationType,
    ProofObligation,
    RiskLevel,
    TaskContract,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_verification_result(
    *,
    passed: bool = True,
    status: VerificationStatus = VerificationStatus.VERIFIED,
    checks: list[VerificationCheck] | None = None,
    confidence: float = 0.95,
) -> VerificationResult:
    if checks is None:
        if passed:
            checks = [VerificationCheck(
                check_id="L2-SUITE", level=VerificationLevel.EXISTING_TESTS,
                description="Run test suite", status=CheckStatus.PASS,
                evidence="All tests passed",
            )]
        else:
            checks = [VerificationCheck(
                check_id="L2-SUITE", level=VerificationLevel.EXISTING_TESTS,
                description="Run test suite", status=CheckStatus.FAIL,
                evidence="1 test failed",
            )]
    return VerificationResult(
        passed=passed, status=status, command="pytest tests/",
        exit_code=0 if passed else 1, stdout="OK" if passed else "FAIL",
        stderr="", duration=1.0, checks=checks, failure_summary="",
        evidence=["evidence"], changed_files=["src/foo.py"],
        confidence=confidence,
    )


# ══════════════════════════════════════════════════════════════════════
# ExecutionMetrics tests
# ══════════════════════════════════════════════════════════════════════


class TestExecutionMetrics:
    def test_initial_zeros(self):
        m = ExecutionMetrics()
        assert m.model_calls == 0
        assert m.verification_runs == 0
        assert m.cache_hits == 0

    def test_record_model_call(self):
        m = ExecutionMetrics()
        m.record_model_call(500)
        m.record_model_call(300)
        assert m.model_calls == 2
        assert m.approximate_context_size == 800

    def test_record_tool_call(self):
        m = ExecutionMetrics()
        m.record_tool_call()
        m.record_tool_call()
        assert m.tool_calls == 2

    def test_record_verification(self):
        m = ExecutionMetrics()
        m.record_verification()
        assert m.verification_runs == 1

    def test_record_falsification(self):
        m = ExecutionMetrics()
        m.record_falsification()
        assert m.falsification_runs == 1

    def test_record_file_inspected(self):
        m = ExecutionMetrics()
        m.record_file_inspected()
        m.record_file_inspected()
        assert m.files_inspected == 2

    def test_record_files_changed(self):
        m = ExecutionMetrics()
        m.record_files_changed(3)
        m.record_files_changed(2)
        assert m.files_changed == 5

    def test_record_repair(self):
        m = ExecutionMetrics()
        m.record_repair()
        assert m.repair_attempts == 1

    def test_record_stale_evidence(self):
        m = ExecutionMetrics()
        m.record_stale_evidence()
        assert m.stale_evidence_events == 1

    def test_cache_hit_rate_zero(self):
        m = ExecutionMetrics()
        assert m.cache_hit_rate == 0.0

    def test_cache_hit_rate(self):
        m = ExecutionMetrics()
        m.record_cache_hit()
        m.record_cache_hit()
        m.record_cache_miss()
        assert abs(m.cache_hit_rate - 2 / 3) < 0.01

    def test_record_skipped_verification(self):
        m = ExecutionMetrics()
        m.record_skipped_verification()
        assert m.skipped_verifications == 1

    def test_record_skipped_rescan(self):
        m = ExecutionMetrics()
        m.record_skipped_rescan()
        assert m.skipped_rescans == 1

    def test_to_dict(self):
        m = ExecutionMetrics()
        m.record_model_call(100)
        m.record_verification()
        m.execution_duration = 5.0
        d = m.to_dict()
        assert d["model_calls"] == 1
        assert d["verification_runs"] == 1
        assert d["execution_duration"] == 5.0
        assert "cache_hit_rate" in d

    def test_summary_lines(self):
        m = ExecutionMetrics()
        m.record_model_call(100)
        m.record_verification()
        m.record_cache_hit()
        m.record_cache_miss()
        m.execution_duration = 2.5
        lines = m.summary_lines()
        assert any("Model calls:" in l for l in lines)
        assert any("Verification runs:" in l for l in lines)
        assert any("Duration:" in l for l in lines)
        assert any("Cache hit rate:" in l for l in lines)

    def test_summary_lines_no_cache(self):
        m = ExecutionMetrics()
        lines = m.summary_lines()
        assert not any("Cache hit rate:" in l for l in lines)

    def test_summary_skipped_verifications(self):
        m = ExecutionMetrics()
        m.record_skipped_verification()
        lines = m.summary_lines()
        assert any("Skipped verifications:" in l for l in lines)


# ══════════════════════════════════════════════════════════════════════
# FileCache tests
# ══════════════════════════════════════════════════════════════════════


class TestFileCache:
    def test_put_and_get(self):
        fc = FileCache()
        fc.put("src/foo.py", "print('hello')")
        assert fc.get("src/foo.py") == "print('hello')"

    def test_get_miss(self):
        fc = FileCache()
        assert fc.get("nonexistent.py") is None

    def test_put_returns_hash(self):
        fc = FileCache()
        h = fc.put("f.py", "content")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_get_hash(self):
        fc = FileCache()
        h1 = fc.put("f.py", "content")
        h2 = fc.get_hash("f.py")
        assert h1 == h2

    def test_get_hash_miss(self):
        fc = FileCache()
        assert fc.get_hash("missing.py") is None

    def test_has(self):
        fc = FileCache()
        assert not fc.has("f.py")
        fc.put("f.py", "c")
        assert fc.has("f.py")

    def test_is_unchanged_matching(self):
        fc = FileCache()
        h = fc.put("f.py", "content")
        assert fc.is_unchanged("f.py", h) is True

    def test_is_unchanged_different(self):
        fc = FileCache()
        fc.put("f.py", "content")
        assert fc.is_unchanged("f.py", "different_hash") is False

    def test_is_unchanged_missing(self):
        fc = FileCache()
        assert fc.is_unchanged("missing.py", "any") is False

    def test_invalidate(self):
        fc = FileCache()
        fc.put("f.py", "c")
        assert fc.invalidate("f.py") is True
        assert fc.get("f.py") is None
        assert fc.invalidate("f.py") is False

    def test_invalidate_changed(self):
        fc = FileCache()
        fc.put("a.py", "1")
        fc.put("b.py", "2")
        fc.put("c.py", "3")
        count = fc.invalidate_changed(["a.py", "c.py"])
        assert count == 2
        assert fc.get("a.py") is None
        assert fc.get("b.py") == "2"  # Untouched.
        assert fc.get("c.py") is None

    def test_invalidate_changed_preserves_unrelated(self):
        """Invalidation ONLY affects listed paths — unrelated files stay cached."""
        fc = FileCache()
        fc.put("src/important.py", "keep this")
        fc.put("src/changed.py", "old")
        fc.invalidate_changed(["src/changed.py"])
        assert fc.get("src/important.py") == "keep this"

    def test_clear(self):
        fc = FileCache()
        fc.put("a.py", "1")
        fc.put("b.py", "2")
        fc.clear()
        assert fc.size == 0

    def test_size(self):
        fc = FileCache()
        assert fc.size == 0
        fc.put("a.py", "x")
        fc.put("b.py", "y")
        assert fc.size == 2

    def test_total_bytes(self):
        fc = FileCache()
        fc.put("a.py", "hello")
        assert fc.total_bytes == 5

    def test_paths(self):
        fc = FileCache()
        fc.put("a.py", "1")
        fc.put("b.py", "2")
        assert sorted(fc.paths()) == ["a.py", "b.py"]

    def test_to_dict(self):
        fc = FileCache()
        fc.put("a.py", "1")
        d = fc.to_dict()
        assert d["entries"] == 1
        assert "total_bytes" in d

    def test_cache_reuse_avoids_reread(self):
        """Simulate cache reuse: first call reads, second call uses cache."""
        fc = FileCache()
        reads = {"count": 0}

        def read_file(path):
            reads["count"] += 1
            content = f"content of {path}"
            fc.put(path, content)
            return content

        # First read — cache miss.
        cached = fc.get("src/foo.py")
        assert cached is None
        content1 = read_file("src/foo.py")
        assert reads["count"] == 1

        # Second read — cache hit.
        cached = fc.get("src/foo.py")
        assert cached == content1
        assert reads["count"] == 1  # NOT incremented.

    def test_invalidation_forces_reread(self):
        """After invalidation, the cache no longer returns stale content."""
        fc = FileCache()
        fc.put("src/foo.py", "old content")
        fc.invalidate("src/foo.py")
        assert fc.get("src/foo.py") is None  # Must re-read.

    def test_same_content_same_hash(self):
        """Identical content always produces the same hash."""
        fc = FileCache()
        h1 = fc.put("a.py", "identical")
        h2 = fc.put("b.py", "identical")
        assert h1 == h2

    def test_different_content_different_hash(self):
        fc = FileCache()
        h1 = fc.put("a.py", "version1")
        h2 = fc.put("a.py", "version2")
        assert h1 != h2


# ══════════════════════════════════════════════════════════════════════
# VerificationCache tests
# ══════════════════════════════════════════════════════════════════════


class TestVerificationCache:
    def test_no_snapshot_cannot_skip(self):
        vc = VerificationCache()
        assert vc.can_skip({"f.py": "abc"}) is False

    def test_failed_run_cannot_skip(self):
        """A failed verification NEVER allows skipping."""
        vc = VerificationCache()
        vc.record({"f.py": "abc"}, passed=False)
        assert vc.can_skip({"f.py": "abc"}) is False

    def test_passed_unchanged_can_skip(self):
        """Passed + same hashes = safe to skip."""
        vc = VerificationCache()
        vc.record({"f.py": "abc123"}, passed=True)
        assert vc.can_skip({"f.py": "abc123"}) is True

    def test_passed_changed_cannot_skip(self):
        """Passed but hash changed = must re-verify."""
        vc = VerificationCache()
        vc.record({"f.py": "old_hash"}, passed=True)
        assert vc.can_skip({"f.py": "new_hash"}) is False

    def test_new_file_cannot_skip(self):
        """New file not in previous snapshot = must re-verify."""
        vc = VerificationCache()
        vc.record({"f.py": "abc"}, passed=True)
        assert vc.can_skip({"f.py": "abc", "g.py": "def"}) is False

    def test_invalidate_forces_reverification(self):
        """After invalidate(), can_skip always returns False."""
        vc = VerificationCache()
        vc.record({"f.py": "abc"}, passed=True)
        assert vc.can_skip({"f.py": "abc"}) is True
        vc.invalidate()
        assert vc.can_skip({"f.py": "abc"}) is False

    def test_has_snapshot(self):
        vc = VerificationCache()
        assert vc.has_snapshot is False
        vc.record({}, passed=True)
        assert vc.has_snapshot is True

    def test_last_passed(self):
        vc = VerificationCache()
        assert vc.last_passed is None
        vc.record({}, passed=True)
        assert vc.last_passed is True
        vc.record({}, passed=False)
        assert vc.last_passed is False

    def test_multiple_files_all_must_match(self):
        vc = VerificationCache()
        vc.record({"a.py": "h1", "b.py": "h2"}, passed=True)
        assert vc.can_skip({"a.py": "h1", "b.py": "h2"}) is True
        assert vc.can_skip({"a.py": "h1", "b.py": "changed"}) is False


# ══════════════════════════════════════════════════════════════════════
# ContextBudget tests
# ══════════════════════════════════════════════════════════════════════


class TestContextBudget:
    def test_initial_state(self):
        cb = ContextBudget(max_chars=1000)
        assert cb.used == 0
        assert cb.remaining == 1000
        assert cb.item_count == 0

    def test_can_add(self):
        cb = ContextBudget(max_chars=10)
        assert cb.can_add("hello") is True
        assert cb.can_add("hello world!") is False

    def test_add_within_budget(self):
        cb = ContextBudget(max_chars=100)
        assert cb.add("hello") is True
        assert cb.used == 5
        assert cb.remaining == 95

    def test_add_exceeds_budget(self):
        cb = ContextBudget(max_chars=5)
        assert cb.add("hello") is True
        assert cb.add("extra") is False
        assert cb.used == 5

    def test_force_add(self):
        cb = ContextBudget(max_chars=5)
        cb.force_add("hello world")  # Exceeds budget.
        assert cb.used == 11

    def test_utilisation(self):
        cb = ContextBudget(max_chars=100)
        cb.add("x" * 50)
        assert abs(cb.utilisation - 0.5) < 0.01

    def test_utilisation_zero_budget(self):
        cb = ContextBudget(max_chars=0)
        assert cb.utilisation == 1.0

    def test_reset(self):
        cb = ContextBudget(max_chars=100)
        cb.add("hello")
        cb.reset()
        assert cb.used == 0
        assert cb.item_count == 0

    def test_to_dict(self):
        cb = ContextBudget(max_chars=100)
        cb.add("hi")
        d = cb.to_dict()
        assert d["max_chars"] == 100
        assert d["used_chars"] == 2
        assert d["remaining_chars"] == 98


# ══════════════════════════════════════════════════════════════════════
# Orchestrator metrics integration
# ══════════════════════════════════════════════════════════════════════


class TestOrchestratorMetrics:
    def test_metrics_in_successful_result(self):
        """Successful run includes metrics in TaskResult."""
        vr = _make_verification_result(passed=True)
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=5, verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")

        assert result.metrics is not None
        assert result.metrics.verification_runs >= 1
        assert result.metrics.execution_duration >= 0

    def test_metrics_in_failed_result(self):
        """Failed run also includes metrics."""
        vr = _make_verification_result(
            passed=False, status=VerificationStatus.FAILED,
        )
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=2, verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")

        assert result.metrics is not None
        assert result.metrics.verification_runs >= 1
        assert result.metrics.files_changed >= 1

    def test_metrics_in_contract_failure(self):
        """Contract generation failure includes metrics."""
        ce = MagicMock(spec=ContractEngine)
        ce.generate_contract.side_effect = ContractValidationError("bad")

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=5, contract_engine=ce,
        )
        result = orch.run("Fix bug")

        assert result.metrics is not None
        assert result.metrics.model_calls == 1

    def test_metrics_count_repair_attempts(self):
        """Repair attempts are counted in metrics."""
        vr_fail = _make_verification_result(
            passed=False, status=VerificationStatus.FAILED,
        )
        vr_pass = _make_verification_result(passed=True)
        verifier = MagicMock(spec=Verifier)
        verifier.verify.side_effect = [vr_fail, vr_fail, vr_pass]

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=5, verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")

        assert result.metrics.repair_attempts == 2  # Attempts 2 and 3.

    def test_metrics_tool_calls_counted(self):
        """patch_fn invocations are counted as tool calls."""
        vr = _make_verification_result(passed=True)
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr

        calls = {"n": 0}

        def patch_fn(t, c, f, p):
            calls["n"] += 1
            return ["src/foo.py"]

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=5, verifier=verifier,
            patch_fn=patch_fn,
        )
        result = orch.run("Fix bug")

        assert result.metrics.tool_calls >= 1

    def test_metrics_files_changed_accumulated(self):
        """Files changed count accumulates across attempts."""
        vr_fail = _make_verification_result(
            passed=False, status=VerificationStatus.FAILED,
        )
        vr_pass = _make_verification_result(passed=True)
        verifier = MagicMock(spec=Verifier)
        verifier.verify.side_effect = [vr_fail, vr_pass]

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=5, verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py", "src/bar.py"],
        )
        result = orch.run("Fix bug")

        assert result.metrics.files_changed >= 4  # 2 files × 2 attempts

    def test_metrics_to_dict_in_result(self):
        """Metrics are included in TaskResult.to_dict()."""
        vr = _make_verification_result(passed=True)
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=5, verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")

        d = result.to_dict()
        assert "metrics" in d
        assert d["metrics"]["verification_runs"] >= 1

    def test_metrics_reset_between_runs(self):
        """Metrics reset on each run() call."""
        vr = _make_verification_result(passed=True)
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=5, verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )

        r1 = orch.run("Fix bug 1")
        r2 = orch.run("Fix bug 2")

        # Each run should have its own metrics, not accumulated.
        assert r1.metrics.verification_runs >= 1
        assert r2.metrics.verification_runs >= 1

    def test_cache_misses_on_fresh_run(self):
        """First run has cache misses (no prior cache)."""
        vr = _make_verification_result(passed=True)
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=5, verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")

        assert result.metrics.cache_misses >= 1
