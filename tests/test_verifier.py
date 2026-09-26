"""Tests for src/verifier.py — multi-level task verification."""

import hashlib
import time
from unittest.mock import MagicMock, patch

import pytest

from src.verifier import (
    CheckStatus,
    CommandResult,
    FileFingerprint,
    FalsificationInterface,
    VerificationCheck,
    VerificationLevel,
    VerificationResult,
    VerificationStatus,
    Verifier,
    compute_fingerprints,
    _aggregate_status,
    _compute_confidence,
    _obligation_to_command,
    _run_basic_validation,
    _run_contract_verification,
    _run_existing_tests,
    _run_regression_verification,
    _run_targeted_tests,
)
from src.contract_engine import (
    ObligationStatus,
    ObligationType,
    ProofObligation,
    RiskLevel,
    TaskContract,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_runner(
    *,
    success: bool = True,
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
    duration: float = 0.1,
    timed_out: bool = False,
):
    """Return a mock command runner that always returns the same result."""
    def runner(command: str, timeout: int) -> CommandResult:
        return CommandResult(
            success=success,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration=duration,
            timed_out=timed_out,
        )
    return runner


def _make_multi_runner(responses: list[dict]):
    """Return a runner that returns different results per call.

    Each dict in *responses* is passed as kwargs to CommandResult.
    If there are fewer responses than calls, the last response repeats.
    """
    idx = {"n": 0}

    def runner(command: str, timeout: int) -> CommandResult:
        i = min(idx["n"], len(responses) - 1)
        idx["n"] += 1
        return CommandResult(**responses[i])
    return runner


def _make_contract(
    obligations: list[ProofObligation] | None = None,
) -> TaskContract:
    """Build a minimal TaskContract for testing."""
    if obligations is None:
        obligations = [
            ProofObligation(
                id="OB-1",
                description="locked users cannot authenticate",
                type=ObligationType.BEHAVIOR,
                verification_method="pytest tests/test_auth.py -x -q",
            ),
            ProofObligation(
                id="OB-2",
                description="unlocked users can still authenticate",
                type=ObligationType.REGRESSION,
                verification_method="pytest tests/test_auth.py -k test_unlock",
            ),
        ]
    return TaskContract(
        task_goal="Fix login for locked accounts",
        behavioral_requirements=["Locked accounts must be rejected"],
        acceptance_criteria=["test_locked_user passes"],
        constraints=["Do not modify user model"],
        non_goals=["Do not change password hashing"],
        affected_area="authentication",
        expected_files=["src/auth/login.py"],
        risk_level=RiskLevel.MEDIUM,
        verification_plan="Run auth tests",
        proof_obligations=obligations,
    )


# ═══════════════════════════════════════════════════════════════════
# CommandResult
# ═══════════════════════════════════════════════════════════════════


class TestCommandResult:
    def test_success_result(self):
        r = CommandResult(success=True, stdout="ok", exit_code=0)
        assert r.success is True
        assert r.stdout == "ok"
        assert r.exit_code == 0
        assert r.timed_out is False

    def test_failure_result(self):
        r = CommandResult(success=False, stderr="err", exit_code=1)
        assert r.success is False
        assert r.stderr == "err"

    def test_timeout_result(self):
        r = CommandResult(success=False, timed_out=True, exit_code=-1)
        assert r.timed_out is True


# ═══════════════════════════════════════════════════════════════════
# FileFingerprint
# ═══════════════════════════════════════════════════════════════════


class TestFileFingerprint:
    def test_from_content(self):
        fp = FileFingerprint.from_content("src/foo.py", "print('hello')")
        assert fp.path == "src/foo.py"
        assert len(fp.sha256) == 64
        assert fp.size_bytes == len("print('hello')".encode())

    def test_to_dict(self):
        fp = FileFingerprint.from_content("a.py", "x = 1")
        d = fp.to_dict()
        assert d["path"] == "a.py"
        assert "sha256" in d
        assert "size_bytes" in d

    def test_same_content_same_hash(self):
        fp1 = FileFingerprint.from_content("a.py", "hello")
        fp2 = FileFingerprint.from_content("b.py", "hello")
        assert fp1.sha256 == fp2.sha256

    def test_different_content_different_hash(self):
        fp1 = FileFingerprint.from_content("a.py", "hello")
        fp2 = FileFingerprint.from_content("a.py", "world")
        assert fp1.sha256 != fp2.sha256


# ═══════════════════════════════════════════════════════════════════
# VerificationCheck
# ═══════════════════════════════════════════════════════════════════


class TestVerificationCheck:
    def test_to_dict_basic(self):
        c = VerificationCheck(
            check_id="L1-SYNTAX-1",
            level=VerificationLevel.BASIC_VALIDATION,
            description="Syntax check",
            status=CheckStatus.PASS,
            evidence="ok",
        )
        d = c.to_dict()
        assert d["check_id"] == "L1-SYNTAX-1"
        assert d["level"] == 1
        assert d["status"] == "PASS"

    def test_to_dict_with_obligation(self):
        c = VerificationCheck(
            check_id="L4-OB-1",
            level=VerificationLevel.CONTRACT_VERIFICATION,
            description="Obligation",
            status=CheckStatus.PASS,
            evidence="verified",
            obligation_id="OB-1",
            command="pytest test.py",
            stdout="1 passed",
            exit_code=0,
        )
        d = c.to_dict()
        assert d["obligation_id"] == "OB-1"
        assert d["command"] == "pytest test.py"


# ═══════════════════════════════════════════════════════════════════
# VerificationResult
# ═══════════════════════════════════════════════════════════════════


class TestVerificationResult:
    def _make_result(self, **overrides):
        defaults = dict(
            passed=True,
            status=VerificationStatus.VERIFIED,
            command="pytest",
            exit_code=0,
            stdout="ok",
            stderr="",
            duration=0.5,
            checks=[],
            failure_summary="",
            evidence=[],
            changed_files=["src/foo.py"],
            confidence=0.9,
            fingerprints=[
                FileFingerprint.from_content("src/foo.py", "original"),
            ],
        )
        defaults.update(overrides)
        return VerificationResult(**defaults)

    def test_to_dict(self):
        r = self._make_result()
        d = r.to_dict()
        assert d["passed"] is True
        assert d["status"] == "VERIFIED"
        assert d["confidence"] == 0.9
        assert len(d["fingerprints"]) == 1

    def test_is_stale_unchanged(self):
        r = self._make_result()
        current = [FileFingerprint.from_content("src/foo.py", "original")]
        assert r.is_stale(current) is False

    def test_is_stale_changed(self):
        r = self._make_result()
        current = [FileFingerprint.from_content("src/foo.py", "modified")]
        assert r.is_stale(current) is True

    def test_is_stale_deleted_file(self):
        r = self._make_result()
        assert r.is_stale([]) is True

    def test_mark_stale(self):
        r = self._make_result()
        r.mark_stale()
        assert r.status == VerificationStatus.STALE
        assert r.passed is False
        assert "stale" in r.failure_summary.lower()

    def test_stale_never_verified(self):
        """A stale result cannot be used as current proof."""
        r = self._make_result()
        r.mark_stale()
        assert r.status != VerificationStatus.VERIFIED


# ═══════════════════════════════════════════════════════════════════
# Aggregation helpers
# ═══════════════════════════════════════════════════════════════════


class TestAggregateStatus:
    def test_all_pass(self):
        checks = [
            VerificationCheck("c1", VerificationLevel.BASIC_VALIDATION,
                              "d", CheckStatus.PASS, "e"),
            VerificationCheck("c2", VerificationLevel.EXISTING_TESTS,
                              "d", CheckStatus.PASS, "e"),
        ]
        status, passed, summary = _aggregate_status(checks)
        assert status == VerificationStatus.VERIFIED
        assert passed is True
        assert summary == ""

    def test_any_fail(self):
        checks = [
            VerificationCheck("c1", VerificationLevel.BASIC_VALIDATION,
                              "d", CheckStatus.PASS, "e"),
            VerificationCheck("c2", VerificationLevel.EXISTING_TESTS,
                              "d", CheckStatus.FAIL, "broke"),
        ]
        status, passed, summary = _aggregate_status(checks)
        assert status == VerificationStatus.FAILED
        assert passed is False
        assert "broke" in summary

    def test_error_yields_unknown(self):
        checks = [
            VerificationCheck("c1", VerificationLevel.BASIC_VALIDATION,
                              "d", CheckStatus.PASS, "e"),
            VerificationCheck("c2", VerificationLevel.EXISTING_TESTS,
                              "d", CheckStatus.ERROR, "timeout"),
        ]
        status, passed, _ = _aggregate_status(checks)
        assert status == VerificationStatus.UNKNOWN
        assert passed is False

    def test_all_skip(self):
        checks = [
            VerificationCheck("c1", VerificationLevel.BASIC_VALIDATION,
                              "d", CheckStatus.SKIP, "e"),
        ]
        status, passed, _ = _aggregate_status(checks)
        assert status == VerificationStatus.UNKNOWN
        assert passed is False

    def test_empty_checks(self):
        status, passed, _ = _aggregate_status([])
        assert status == VerificationStatus.UNKNOWN
        assert passed is False

    def test_unknown_never_promoted_to_verified(self):
        """UNKNOWN must never be promoted to VERIFIED."""
        checks = [
            VerificationCheck("c1", VerificationLevel.BASIC_VALIDATION,
                              "d", CheckStatus.ERROR, "e"),
        ]
        status, _, _ = _aggregate_status(checks)
        assert status != VerificationStatus.VERIFIED

    def test_fail_takes_precedence_over_error(self):
        checks = [
            VerificationCheck("c1", VerificationLevel.BASIC_VALIDATION,
                              "d", CheckStatus.ERROR, "err"),
            VerificationCheck("c2", VerificationLevel.EXISTING_TESTS,
                              "d", CheckStatus.FAIL, "fail"),
        ]
        status, _, _ = _aggregate_status(checks)
        assert status == VerificationStatus.FAILED


class TestComputeConfidence:
    def test_all_pass(self):
        checks = [
            VerificationCheck("c1", VerificationLevel.BASIC_VALIDATION,
                              "d", CheckStatus.PASS, "e"),
            VerificationCheck("c2", VerificationLevel.EXISTING_TESTS,
                              "d", CheckStatus.PASS, "e"),
        ]
        conf = _compute_confidence(checks)
        assert conf == 1.0

    def test_all_fail(self):
        checks = [
            VerificationCheck("c1", VerificationLevel.BASIC_VALIDATION,
                              "d", CheckStatus.FAIL, "e"),
        ]
        conf = _compute_confidence(checks)
        assert conf == 0.0

    def test_mixed(self):
        checks = [
            VerificationCheck("c1", VerificationLevel.BASIC_VALIDATION,
                              "d", CheckStatus.PASS, "e"),
            VerificationCheck("c2", VerificationLevel.EXISTING_TESTS,
                              "d", CheckStatus.FAIL, "e"),
        ]
        conf = _compute_confidence(checks)
        assert 0.0 < conf < 1.0

    def test_skip_ignored(self):
        checks = [
            VerificationCheck("c1", VerificationLevel.BASIC_VALIDATION,
                              "d", CheckStatus.PASS, "e"),
            VerificationCheck("c2", VerificationLevel.EXISTING_TESTS,
                              "d", CheckStatus.SKIP, "e"),
        ]
        conf = _compute_confidence(checks)
        assert conf == 1.0

    def test_empty(self):
        assert _compute_confidence([]) == 0.0


# ═══════════════════════════════════════════════════════════════════
# Obligation to command extraction
# ═══════════════════════════════════════════════════════════════════


class TestObligationToCommand:
    def test_direct_pytest(self):
        ob = ProofObligation(
            id="OB-1", description="d",
            type=ObligationType.BEHAVIOR,
            verification_method="pytest tests/test_auth.py -x -q",
        )
        assert _obligation_to_command(ob) == "pytest tests/test_auth.py -x -q"

    def test_direct_python(self):
        ob = ProofObligation(
            id="OB-1", description="d",
            type=ObligationType.BEHAVIOR,
            verification_method="python -m pytest tests/",
        )
        assert _obligation_to_command(ob) == "python -m pytest tests/"

    def test_embedded_pytest(self):
        ob = ProofObligation(
            id="OB-1", description="d",
            type=ObligationType.BEHAVIOR,
            verification_method="Run pytest tests/test_auth.py to verify",
        )
        cmd = _obligation_to_command(ob)
        assert cmd is not None
        assert "pytest" in cmd

    def test_run_test_pattern(self):
        ob = ProofObligation(
            id="OB-1", description="d",
            type=ObligationType.BEHAVIOR,
            verification_method="Run test_locked_user to verify rejection",
        )
        cmd = _obligation_to_command(ob)
        assert cmd is not None
        assert "test_locked_user" in cmd

    def test_no_command(self):
        ob = ProofObligation(
            id="OB-1", description="d",
            type=ObligationType.BEHAVIOR,
            verification_method="Manual code review required",
        )
        assert _obligation_to_command(ob) is None


# ═══════════════════════════════════════════════════════════════════
# Level 1 — Basic validation
# ═══════════════════════════════════════════════════════════════════


class TestLevel1BasicValidation:
    def test_pass(self):
        runner = _make_runner(success=True, stdout="", exit_code=0)
        checks = _run_basic_validation(["src/foo.py"], runner, 30)
        assert len(checks) >= 2  # syntax + import
        assert all(c.status == CheckStatus.PASS for c in checks)

    def test_syntax_failure(self):
        runner = _make_runner(
            success=False, stderr="SyntaxError", exit_code=1,
        )
        checks = _run_basic_validation(["src/foo.py"], runner, 30)
        assert any(c.status == CheckStatus.FAIL for c in checks)

    def test_no_python_files(self):
        runner = _make_runner(success=True)
        checks = _run_basic_validation(["README.md"], runner, 30)
        assert len(checks) == 1
        assert checks[0].status == CheckStatus.SKIP

    def test_timeout(self):
        runner = _make_runner(timed_out=True, success=False, exit_code=-1)
        checks = _run_basic_validation(["src/foo.py"], runner, 5)
        assert any(c.status == CheckStatus.ERROR for c in checks)


# ═══════════════════════════════════════════════════════════════════
# Level 2 — Existing tests
# ═══════════════════════════════════════════════════════════════════


class TestLevel2ExistingTests:
    def test_pass(self):
        runner = _make_runner(
            success=True, stdout="5 passed", exit_code=0,
        )
        checks = _run_existing_tests(runner, 60)
        assert len(checks) == 1
        assert checks[0].status == CheckStatus.PASS
        assert "verified by executed test suite" in checks[0].evidence.lower()

    def test_failure(self):
        runner = _make_runner(
            success=False, stdout="2 failed", exit_code=1,
        )
        checks = _run_existing_tests(runner, 60)
        assert checks[0].status == CheckStatus.FAIL

    def test_timeout(self):
        runner = _make_runner(
            success=False, timed_out=True, exit_code=-1,
        )
        checks = _run_existing_tests(runner, 10)
        assert checks[0].status == CheckStatus.ERROR
        assert "timed out" in checks[0].evidence.lower()


# ═══════════════════════════════════════════════════════════════════
# Level 3 — Targeted tests
# ═══════════════════════════════════════════════════════════════════


class TestLevel3TargetedTests:
    def test_pass(self):
        runner = _make_runner(success=True, stdout="1 passed", exit_code=0)
        checks = _run_targeted_tests(["src/auth.py"], runner, 30)
        passed = [c for c in checks if c.status == CheckStatus.PASS]
        assert len(passed) >= 1

    def test_no_test_file(self):
        runner = _make_runner(success=False, exit_code=5)
        checks = _run_targeted_tests(["src/auth.py"], runner, 30)
        skipped = [c for c in checks if c.status == CheckStatus.SKIP]
        assert len(skipped) >= 1

    def test_failure(self):
        runner = _make_runner(
            success=False, stdout="FAILED", exit_code=1,
        )
        checks = _run_targeted_tests(["src/auth.py"], runner, 30)
        failed = [c for c in checks if c.status == CheckStatus.FAIL]
        assert len(failed) >= 1

    def test_includes_test_files_directly(self):
        runner = _make_runner(success=True, exit_code=0)
        checks = _run_targeted_tests(
            ["tests/test_auth.py"], runner, 30,
        )
        # The test file itself should be in the targets.
        commands = [c.command for c in checks if c.command]
        assert any("test_auth" in cmd for cmd in commands)


# ═══════════════════════════════════════════════════════════════════
# Level 4 — Contract verification
# ═══════════════════════════════════════════════════════════════════


class TestLevel4ContractVerification:
    def test_no_contract(self):
        runner = _make_runner(success=True)
        checks = _run_contract_verification(None, runner, 30)
        assert len(checks) == 1
        assert checks[0].status == CheckStatus.SKIP

    def test_empty_obligations(self):
        contract = _make_contract(obligations=[])
        runner = _make_runner(success=True)
        checks = _run_contract_verification(contract, runner, 30)
        assert len(checks) == 1
        assert checks[0].status == CheckStatus.SKIP

    def test_pass_obligations(self):
        contract = _make_contract()
        runner = _make_runner(
            success=True, stdout="1 passed", exit_code=0,
        )
        checks = _run_contract_verification(contract, runner, 30)
        assert len(checks) == 2
        for c in checks:
            assert c.status == CheckStatus.PASS
            assert c.obligation_id is not None
            assert "verified by executed test" in c.evidence.lower()

    def test_fail_obligation(self):
        contract = _make_contract()
        runner = _make_runner(
            success=False, stdout="FAILED", exit_code=1,
        )
        checks = _run_contract_verification(contract, runner, 30)
        assert all(c.status == CheckStatus.FAIL for c in checks)

    def test_non_executable_obligation(self):
        obligations = [
            ProofObligation(
                id="OB-1", description="manual check",
                type=ObligationType.BEHAVIOR,
                verification_method="Manual review of the diff",
            ),
        ]
        contract = _make_contract(obligations=obligations)
        runner = _make_runner(success=True)
        checks = _run_contract_verification(contract, runner, 30)
        assert checks[0].status == CheckStatus.SKIP

    def test_timeout_obligation(self):
        contract = _make_contract()
        runner = _make_runner(
            success=False, timed_out=True, exit_code=-1,
        )
        checks = _run_contract_verification(contract, runner, 10)
        assert all(c.status == CheckStatus.ERROR for c in checks)

    def test_multiple_obligations_mixed(self):
        """One obligation passes, the other fails."""
        obligations = [
            ProofObligation(
                id="OB-1", description="passes",
                type=ObligationType.BEHAVIOR,
                verification_method="pytest tests/test_good.py",
            ),
            ProofObligation(
                id="OB-2", description="fails",
                type=ObligationType.REGRESSION,
                verification_method="pytest tests/test_bad.py",
            ),
        ]
        contract = _make_contract(obligations=obligations)
        responses = [
            {"success": True, "stdout": "1 passed", "exit_code": 0},
            {"success": False, "stdout": "FAILED", "exit_code": 1},
        ]
        runner = _make_multi_runner(responses)
        checks = _run_contract_verification(contract, runner, 30)
        assert checks[0].status == CheckStatus.PASS
        assert checks[1].status == CheckStatus.FAIL


# ═══════════════════════════════════════════════════════════════════
# Level 5 — Regression
# ═══════════════════════════════════════════════════════════════════


class TestLevel5Regression:
    def test_pass(self):
        runner = _make_runner(success=True, stdout="10 passed", exit_code=0)
        checks = _run_regression_verification(
            ["src/auth.py"], runner, 60,
        )
        assert checks[0].status == CheckStatus.PASS
        assert "no regressions" in checks[0].evidence.lower()

    def test_regression_failure(self):
        runner = _make_runner(
            success=False, stdout="1 failed", exit_code=1,
        )
        checks = _run_regression_verification(
            ["src/auth.py"], runner, 60,
        )
        assert checks[0].status == CheckStatus.FAIL
        assert "regression" in checks[0].evidence.lower()

    def test_excludes_changed_test_files(self):
        runner = _make_runner(success=True, exit_code=0)
        checks = _run_regression_verification(
            ["src/auth.py", "tests/test_auth.py"], runner, 60,
        )
        assert checks[0].command is not None
        assert "--ignore=" in checks[0].command

    def test_no_tests_to_run(self):
        runner = _make_runner(success=False, exit_code=5)
        checks = _run_regression_verification(
            ["tests/test_only.py"], runner, 60,
        )
        assert checks[0].status == CheckStatus.SKIP

    def test_timeout(self):
        runner = _make_runner(
            success=False, timed_out=True, exit_code=-1,
        )
        checks = _run_regression_verification(
            ["src/auth.py"], runner, 10,
        )
        assert checks[0].status == CheckStatus.ERROR


# ═══════════════════════════════════════════════════════════════════
# Level 6 — Falsification interface
# ═══════════════════════════════════════════════════════════════════


class TestLevel6Falsification:
    def test_stub_returns_skip(self):
        fi = FalsificationInterface()
        runner = _make_runner(success=True)
        checks = fi.attempt_falsification(None, [], runner, 30)
        assert len(checks) == 1
        assert checks[0].status == CheckStatus.SKIP
        assert checks[0].level == VerificationLevel.FALSIFICATION

    def test_custom_implementation(self):
        """A subclass can override attempt_falsification."""
        class MyFalsifier(FalsificationInterface):
            def attempt_falsification(self, contract, changed_files, runner, timeout):
                return [VerificationCheck(
                    check_id="L6-CUSTOM",
                    level=VerificationLevel.FALSIFICATION,
                    description="Custom falsification",
                    status=CheckStatus.PASS,
                    evidence="No falsification found",
                )]

        fi = MyFalsifier()
        runner = _make_runner(success=True)
        checks = fi.attempt_falsification(None, [], runner, 30)
        assert checks[0].status == CheckStatus.PASS


# ═══════════════════════════════════════════════════════════════════
# Fingerprinting
# ═══════════════════════════════════════════════════════════════════


class TestFingerprinting:
    def test_compute_from_contents(self):
        contents = {"a.py": "x = 1", "b.py": "y = 2"}
        fps = compute_fingerprints(
            ["a.py", "b.py"], file_contents=contents,
        )
        assert len(fps) == 2
        assert fps[0].path == "a.py"
        assert fps[1].path == "b.py"

    def test_missing_file_skipped(self):
        """Files not on disk and not in file_contents are skipped."""
        fps = compute_fingerprints(
            ["/nonexistent/path.py"], repo_root="/tmp",
        )
        assert len(fps) == 0


# ═══════════════════════════════════════════════════════════════════
# Verifier — full pipeline
# ═══════════════════════════════════════════════════════════════════


class TestVerifierPass:
    """Full pipeline where everything passes."""

    def test_all_pass(self):
        runner = _make_runner(
            success=True, stdout="10 passed", exit_code=0,
        )
        v = Verifier(runner)
        result = v.verify(
            ["src/foo.py"],
            file_contents={"src/foo.py": "x = 1"},
        )
        assert result.passed is True
        assert result.status == VerificationStatus.VERIFIED
        assert result.confidence > 0.0
        assert len(result.checks) > 0
        assert len(result.fingerprints) == 1

    def test_evidence_language(self):
        """Evidence must say 'verified by executed test', not 'proven'."""
        runner = _make_runner(
            success=True, stdout="5 passed", exit_code=0,
        )
        v = Verifier(runner)
        result = v.verify(
            ["src/foo.py"],
            file_contents={"src/foo.py": "x = 1"},
        )
        for e in result.evidence:
            assert "proven correct" not in e.lower()


class TestVerifierFailure:
    """Full pipeline with failures."""

    def test_syntax_failure_aborts_early(self):
        runner = _make_runner(
            success=False, stderr="SyntaxError: invalid syntax",
            exit_code=1,
        )
        v = Verifier(runner)
        result = v.verify(
            ["src/bad.py"],
            file_contents={"src/bad.py": "def f("},
        )
        assert result.passed is False
        assert result.status == VerificationStatus.FAILED
        # Should abort before Level 2.
        levels = {c.level for c in result.checks}
        assert VerificationLevel.EXISTING_TESTS not in levels

    def test_test_suite_failure(self):
        responses = [
            # L1 syntax passes
            {"success": True, "stdout": "", "exit_code": 0},
            # L1 import passes
            {"success": True, "stdout": "", "exit_code": 0},
            # L2 test suite fails
            {"success": False, "stdout": "3 failed", "exit_code": 1},
            # L3+ — all fail too
            {"success": False, "stdout": "failed", "exit_code": 1},
        ]
        runner = _make_multi_runner(responses)
        v = Verifier(runner)
        result = v.verify(
            ["src/foo.py"],
            file_contents={"src/foo.py": "x = 1"},
        )
        assert result.passed is False
        assert result.status == VerificationStatus.FAILED


class TestVerifierUnknown:
    """Pipeline produces UNKNOWN."""

    def test_timeout_yields_unknown(self):
        runner = _make_runner(
            success=False, timed_out=True, exit_code=-1,
        )
        v = Verifier(runner)
        result = v.verify(
            ["README.md"],  # no .py files → L1 SKIP, then timeouts
            file_contents={"README.md": "# Hello"},
        )
        # L1 skips (no python), L2+ timeout → ERROR → UNKNOWN
        assert result.status == VerificationStatus.UNKNOWN
        assert result.passed is False

    def test_unknown_never_promoted(self):
        """UNKNOWN must never become VERIFIED."""
        runner = _make_runner(
            success=False, timed_out=True, exit_code=-1,
        )
        v = Verifier(runner)
        result = v.verify(
            ["README.md"],
            file_contents={"README.md": "# Hello"},
        )
        assert result.status != VerificationStatus.VERIFIED


class TestVerifierStaleEvidence:
    """Stale evidence detection."""

    def test_stale_after_file_change(self):
        runner = _make_runner(
            success=True, stdout="passed", exit_code=0,
        )
        v = Verifier(runner)
        result = v.verify(
            ["src/foo.py"],
            file_contents={"src/foo.py": "original_content"},
        )
        assert result.status == VerificationStatus.VERIFIED

        # Source file changes.
        new_fps = [
            FileFingerprint.from_content("src/foo.py", "modified_content"),
        ]
        is_stale = v.check_staleness(result, new_fps)
        assert is_stale is True
        assert result.status == VerificationStatus.STALE
        assert result.passed is False

    def test_not_stale_same_content(self):
        runner = _make_runner(
            success=True, stdout="passed", exit_code=0,
        )
        v = Verifier(runner)
        result = v.verify(
            ["src/foo.py"],
            file_contents={"src/foo.py": "same_content"},
        )
        same_fps = [
            FileFingerprint.from_content("src/foo.py", "same_content"),
        ]
        is_stale = v.check_staleness(result, same_fps)
        assert is_stale is False
        assert result.status == VerificationStatus.VERIFIED

    def test_stale_after_file_deletion(self):
        runner = _make_runner(
            success=True, stdout="passed", exit_code=0,
        )
        v = Verifier(runner)
        result = v.verify(
            ["src/foo.py"],
            file_contents={"src/foo.py": "content"},
        )
        is_stale = v.check_staleness(result, [])
        assert is_stale is True
        assert result.status == VerificationStatus.STALE


class TestVerifierContractIntegration:
    """Verifier updates contract obligations."""

    def test_obligations_updated_on_pass(self):
        contract = _make_contract()
        runner = _make_runner(
            success=True, stdout="1 passed", exit_code=0,
        )
        v = Verifier(runner)
        result = v.verify(
            ["src/auth/login.py"],
            contract=contract,
            file_contents={"src/auth/login.py": "pass"},
        )
        # Contract obligations should be updated.
        for ob in contract.proof_obligations:
            assert ob.status == ObligationStatus.PASS

    def test_obligations_updated_on_fail(self):
        contract = _make_contract()
        # L1 passes (syntax/import), L2 passes, L3 passes,
        # L4 fails (contract)
        call_count = {"n": 0}

        def runner(cmd, timeout):
            call_count["n"] += 1
            # First few calls are L1 syntax/import (pass)
            # Then L2 (pass), L3 (pass)
            # L4 contract checks should fail
            if "test_auth" in cmd:
                return CommandResult(
                    success=False, stdout="FAILED", exit_code=1,
                )
            return CommandResult(success=True, exit_code=0)

        v = Verifier(runner)
        result = v.verify(
            ["src/auth/login.py"],
            contract=contract,
            file_contents={"src/auth/login.py": "pass"},
        )
        failed_obs = [
            ob for ob in contract.proof_obligations
            if ob.status == ObligationStatus.FAIL
        ]
        assert len(failed_obs) == 2


class TestVerifierMaxLevel:
    """Verify that max_level limits execution."""

    def test_max_level_1(self):
        runner = _make_runner(success=True, exit_code=0)
        v = Verifier(runner)
        result = v.verify(
            ["src/foo.py"],
            max_level=VerificationLevel.BASIC_VALIDATION,
            file_contents={"src/foo.py": "x = 1"},
        )
        levels = {c.level for c in result.checks}
        assert VerificationLevel.EXISTING_TESTS not in levels
        assert VerificationLevel.CONTRACT_VERIFICATION not in levels

    def test_max_level_2(self):
        runner = _make_runner(success=True, exit_code=0)
        v = Verifier(runner)
        result = v.verify(
            ["src/foo.py"],
            max_level=VerificationLevel.EXISTING_TESTS,
            file_contents={"src/foo.py": "x = 1"},
        )
        levels = {c.level for c in result.checks}
        assert VerificationLevel.BASIC_VALIDATION in levels
        assert VerificationLevel.EXISTING_TESTS in levels
        assert VerificationLevel.CONTRACT_VERIFICATION not in levels


class TestVerifierRunLevel:
    """run_level runs a single verification level."""

    def test_run_level_1(self):
        runner = _make_runner(success=True, exit_code=0)
        v = Verifier(runner)
        checks = v.run_level(
            VerificationLevel.BASIC_VALIDATION, ["src/foo.py"],
        )
        assert all(c.level == VerificationLevel.BASIC_VALIDATION for c in checks)

    def test_run_level_2(self):
        runner = _make_runner(success=True, exit_code=0)
        v = Verifier(runner)
        checks = v.run_level(
            VerificationLevel.EXISTING_TESTS, ["src/foo.py"],
        )
        assert all(c.level == VerificationLevel.EXISTING_TESTS for c in checks)

    def test_run_level_6(self):
        runner = _make_runner(success=True, exit_code=0)
        v = Verifier(runner)
        checks = v.run_level(
            VerificationLevel.FALSIFICATION, ["src/foo.py"],
        )
        assert all(c.level == VerificationLevel.FALSIFICATION for c in checks)


class TestVerifierHistory:
    def test_results_stored(self):
        runner = _make_runner(success=True, exit_code=0)
        v = Verifier(runner)
        v.verify(["src/a.py"], file_contents={"src/a.py": "x"})
        v.verify(["src/b.py"], file_contents={"src/b.py": "y"})
        assert len(v.results) == 2


class TestVerifierTimeout:
    """Command timeout handling."""

    def test_timeout_in_level_2(self):
        responses = [
            # L1 syntax + import pass
            {"success": True, "stdout": "", "exit_code": 0},
            {"success": True, "stdout": "", "exit_code": 0},
            # L2 times out
            {"success": False, "stdout": "", "stderr": "", "exit_code": -1,
             "timed_out": True},
            # Remaining levels also time out
            {"success": False, "stdout": "", "stderr": "", "exit_code": -1,
             "timed_out": True},
        ]
        runner = _make_multi_runner(responses)
        v = Verifier(runner)
        result = v.verify(
            ["src/foo.py"],
            file_contents={"src/foo.py": "x = 1"},
        )
        assert result.status == VerificationStatus.UNKNOWN
        assert result.passed is False
        # Should have timeout evidence.
        timed = [
            c for c in result.checks
            if "timed out" in c.evidence.lower()
        ]
        assert len(timed) > 0


class TestVerifierCustomTestCommand:
    def test_custom_test_command(self):
        runner = _make_runner(success=True, stdout="passed", exit_code=0)
        v = Verifier(runner, test_command="make test")
        checks = v.run_level(
            VerificationLevel.EXISTING_TESTS, [],
        )
        assert checks[0].command == "make test"


class TestVerifierCustomFalsifier:
    def test_custom_falsifier(self):
        class StrictFalsifier(FalsificationInterface):
            def attempt_falsification(self, contract, changed_files, runner, timeout):
                return [VerificationCheck(
                    check_id="L6-STRICT",
                    level=VerificationLevel.FALSIFICATION,
                    description="Strict falsification",
                    status=CheckStatus.FAIL,
                    evidence="Found a counterexample",
                )]

        runner = _make_runner(success=True, exit_code=0)
        v = Verifier(runner, falsifier=StrictFalsifier())
        result = v.verify(
            ["src/foo.py"],
            file_contents={"src/foo.py": "x = 1"},
        )
        assert result.status == VerificationStatus.FAILED
        l6 = [c for c in result.checks if c.level == VerificationLevel.FALSIFICATION]
        assert len(l6) == 1
        assert l6[0].status == CheckStatus.FAIL


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_empty_changed_files(self):
        runner = _make_runner(success=True, exit_code=0)
        v = Verifier(runner)
        result = v.verify(
            [],
            file_contents={},
        )
        # Should still complete (L1 skips, rest runs).
        assert result.status in (
            VerificationStatus.VERIFIED,
            VerificationStatus.UNKNOWN,
        )

    def test_to_dict_round_trip(self):
        runner = _make_runner(success=True, stdout="ok", exit_code=0)
        v = Verifier(runner)
        result = v.verify(
            ["src/foo.py"],
            file_contents={"src/foo.py": "x = 1"},
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["status"] == "VERIFIED"
        assert isinstance(d["checks"], list)
        assert isinstance(d["fingerprints"], list)

    def test_multiple_changed_files(self):
        runner = _make_runner(success=True, exit_code=0)
        v = Verifier(runner)
        result = v.verify(
            ["src/a.py", "src/b.py", "src/c.py"],
            file_contents={
                "src/a.py": "a", "src/b.py": "b", "src/c.py": "c",
            },
        )
        assert len(result.fingerprints) == 3
        assert result.passed is True

    def test_stale_cannot_be_used_as_proof(self):
        """A stale result must not have VERIFIED status."""
        runner = _make_runner(success=True, exit_code=0)
        v = Verifier(runner)
        result = v.verify(
            ["src/foo.py"],
            file_contents={"src/foo.py": "original"},
        )
        assert result.status == VerificationStatus.VERIFIED

        result.mark_stale()
        assert result.status == VerificationStatus.STALE
        assert result.status != VerificationStatus.VERIFIED
        assert result.passed is False
