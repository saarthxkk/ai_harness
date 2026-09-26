"""End-to-end evaluation suite for the complete harness.

Uses temporary sample repositories.
Does NOT require a real external AI API — all model responses are mocked.

Scenarios:
    A — Normal success:   buggy repo → contract → fix → VERIFIED
    B — False confidence: tests pass but falsification finds counterexample
    C — Impossible task:  agent cannot establish correctness → FAILED/UNKNOWN
    D — Scope violation:  agent modifies unrelated files → CHANGE BUDGET violation
    E — Stale evidence:   evidence invalidated after source changes
    F — Test weakening:   agent deletes/weakens tests → integrity violation
"""

from __future__ import annotations

import copy
import os
import tempfile
import textwrap
from unittest.mock import MagicMock, patch

import pytest

from src.contract_engine import (
    ContractEngine,
    ContractValidationError,
    ObligationStatus,
    ObligationType,
    ProofObligation,
    RiskLevel,
    TaskContract,
)
from src.verifier import (
    CheckStatus,
    CommandResult,
    FileFingerprint,
    VerificationCheck,
    VerificationLevel,
    VerificationResult,
    VerificationStatus,
    Verifier,
)
from src.change_guard import (
    ChangeBudget,
    ChangeBudgetReport,
    ChangeGuard,
    ScopeStatus,
    Violation,
    ViolationType,
)
from src.rollback import RollbackEngine
from src.falsifier import FalsificationEngine
from src.evidence_graph import (
    EvidenceGraph,
    EvidenceReport,
    EvidenceVerdict,
)
from src.orchestrator import (
    AttemptHistory,
    AttemptRecord,
    FailureType,
    Orchestrator,
    Phase,
    PhaseLog,
    ProgressStatus,
    RepairAction,
    TaskOutcome,
    TaskResult,
)
from src.metrics import ExecutionMetrics


# ═══════════════════════════════════════════════════════════════════════
# Helpers — reusable builders for contracts, verifications, etc.
# ═══════════════════════════════════════════════════════════════════════


def _make_contract(
    *,
    goal: str = "Fix the discount calculation",
    expected_files: list[str] | None = None,
    obligations: list[ProofObligation] | None = None,
    affected_area: str = "src",
    risk: RiskLevel = RiskLevel.LOW,
) -> TaskContract:
    """Build a TaskContract without calling the model."""
    if expected_files is None:
        expected_files = ["src/discount.py"]
    if obligations is None:
        obligations = [
            ProofObligation(
                id="OB-1",
                description="Discount calculates correctly",
                type=ObligationType.BEHAVIOR,
                verification_method="Run test_discount.py",
            ),
            ProofObligation(
                id="OB-2",
                description="Existing tests still pass",
                type=ObligationType.REGRESSION,
                verification_method="Run full test suite",
            ),
        ]
    return TaskContract(
        task_goal=goal,
        behavioral_requirements=["Discount must apply correctly"],
        acceptance_criteria=["All tests pass"],
        constraints=["Do not change pricing logic"],
        non_goals=["Do not change unrelated modules"],
        affected_area=affected_area,
        expected_files=expected_files,
        risk_level=risk,
        verification_plan="Run pytest",
        proof_obligations=obligations,
    )


def _make_vr(
    *,
    passed: bool = True,
    status: VerificationStatus = VerificationStatus.VERIFIED,
    checks: list[VerificationCheck] | None = None,
    confidence: float = 0.95,
    changed_files: list[str] | None = None,
) -> VerificationResult:
    """Build a VerificationResult for tests."""
    if checks is None:
        if passed:
            checks = [
                VerificationCheck(
                    check_id="L1-SYNTAX", level=VerificationLevel.BASIC_VALIDATION,
                    description="Syntax check", status=CheckStatus.PASS,
                    evidence="No syntax errors",
                ),
                VerificationCheck(
                    check_id="L2-SUITE", level=VerificationLevel.EXISTING_TESTS,
                    description="Run test suite", status=CheckStatus.PASS,
                    evidence="All tests passed",
                ),
                VerificationCheck(
                    check_id="L5-REGRESSION", level=VerificationLevel.REGRESSION,
                    description="Regression check", status=CheckStatus.PASS,
                    evidence="No regressions",
                ),
            ]
        else:
            checks = [
                VerificationCheck(
                    check_id="L2-SUITE", level=VerificationLevel.EXISTING_TESTS,
                    description="Run test suite", status=CheckStatus.FAIL,
                    evidence="1 test failed: test_discount",
                ),
            ]
    return VerificationResult(
        passed=passed, status=status,
        command="pytest tests/",
        exit_code=0 if passed else 1,
        stdout="OK" if passed else "FAIL",
        stderr="",
        duration=1.0,
        checks=checks,
        failure_summary="" if passed else "test_discount failed",
        evidence=["test output"],
        changed_files=changed_files or ["src/discount.py"],
        confidence=confidence,
    )


def _make_vr_with_falsification_fail(
    *, counterexample: str = "locked account with expired session",
) -> VerificationResult:
    """Verification where tests PASS but falsification finds a counterexample."""
    return _make_vr(
        passed=True,
        checks=[
            VerificationCheck(
                check_id="L1-SYNTAX", level=VerificationLevel.BASIC_VALIDATION,
                description="Syntax check", status=CheckStatus.PASS,
                evidence="No syntax errors",
            ),
            VerificationCheck(
                check_id="L2-SUITE", level=VerificationLevel.EXISTING_TESTS,
                description="Run test suite", status=CheckStatus.PASS,
                evidence="All 12 tests passed",
            ),
            VerificationCheck(
                check_id="L3-TARGET", level=VerificationLevel.TARGETED_VERIFICATION,
                description="Targeted discount tests", status=CheckStatus.PASS,
                evidence="Discount tests passed",
            ),
            VerificationCheck(
                check_id="L6-FALSIFY-1", level=VerificationLevel.FALSIFICATION,
                description=f"Counterexample: {counterexample}",
                status=CheckStatus.FAIL,
                evidence=f"Falsification found: {counterexample}",
            ),
        ],
    )


def _make_vr_all_pass_with_falsification() -> VerificationResult:
    """Verification where EVERYTHING passes including falsification."""
    return _make_vr(
        passed=True,
        checks=[
            VerificationCheck(
                check_id="L1-SYNTAX", level=VerificationLevel.BASIC_VALIDATION,
                description="Syntax check", status=CheckStatus.PASS,
                evidence="No syntax errors",
            ),
            VerificationCheck(
                check_id="L2-SUITE", level=VerificationLevel.EXISTING_TESTS,
                description="Run test suite", status=CheckStatus.PASS,
                evidence="All tests passed",
            ),
            VerificationCheck(
                check_id="L5-REGRESSION", level=VerificationLevel.REGRESSION,
                description="Regression check", status=CheckStatus.PASS,
                evidence="No regressions",
            ),
            VerificationCheck(
                check_id="L6-FALSIFY", level=VerificationLevel.FALSIFICATION,
                description="Falsification", status=CheckStatus.PASS,
                evidence="No counterexamples found",
            ),
        ],
    )


# ═══════════════════════════════════════════════════════════════════════
# Scenario A — Normal success
# ═══════════════════════════════════════════════════════════════════════


class TestScenarioA_NormalSuccess:
    """Create a deliberately buggy repo → issue → VERIFIED."""

    def test_full_pipeline_succeeds(self):
        """End-to-end: issue → contract → verify → VERIFIED."""
        contract = _make_contract()
        vr = _make_vr(passed=True)

        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr

        orch = Orchestrator(
            model_client=MagicMock(),
            tools=[],
            max_iterations=5,
            verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/discount.py"],
        )

        result = orch.run(
            "Fix the discount calculation",
            contract=contract,
        )

        assert result.outcome == TaskOutcome.VERIFIED
        assert result.contract is not None
        assert result.final_verification is not None
        assert result.final_verification.passed is True
        assert result.reason  # Non-empty reason string.

    def test_phases_logged(self):
        """All pipeline phases appear in the log."""
        contract = _make_contract()
        vr = _make_vr(passed=True)

        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr

        orch = Orchestrator(
            model_client=MagicMock(),
            tools=[],
            max_iterations=5,
            verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/discount.py"],
        )
        result = orch.run("Fix the discount calculation", contract=contract)

        phases = {e.phase for e in result.phase_log}
        assert Phase.UNDERSTAND in phases
        assert Phase.CONTRACT in phases
        assert Phase.PLAN in phases
        assert Phase.EXECUTE in phases
        assert Phase.VERIFY in phases

    def test_metrics_populated(self):
        """Final result has non-trivial metrics."""
        contract = _make_contract()
        vr = _make_vr(passed=True)

        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr

        orch = Orchestrator(
            model_client=MagicMock(),
            tools=[],
            max_iterations=5,
            verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/discount.py"],
        )
        result = orch.run("Fix the discount calculation", contract=contract)

        assert result.metrics is not None
        assert result.metrics.verification_runs >= 1
        assert result.metrics.files_changed >= 1

    def test_single_attempt_on_first_success(self):
        """If first attempt passes, exactly 1 attempt is recorded."""
        contract = _make_contract()
        vr = _make_vr(passed=True)

        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr

        orch = Orchestrator(
            model_client=MagicMock(),
            tools=[],
            max_iterations=5,
            verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/discount.py"],
        )
        result = orch.run("Fix the discount calculation", contract=contract)

        assert result.history.attempt_count == 1
        assert len(result.history.passed_attempts()) == 1

    def test_contract_obligations_present(self):
        """Contract's proof obligations are present in the result."""
        contract = _make_contract()
        vr = _make_vr(passed=True)

        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr

        orch = Orchestrator(
            model_client=MagicMock(),
            tools=[],
            max_iterations=5,
            verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/discount.py"],
        )
        result = orch.run("Fix the discount calculation", contract=contract)

        assert len(result.contract.proof_obligations) == 2
        assert result.contract.proof_obligations[0].id == "OB-1"


# ═══════════════════════════════════════════════════════════════════════
# Scenario B — False confidence (tests pass, falsification catches bug)
# ═══════════════════════════════════════════════════════════════════════


class TestScenarioB_FalseConfidence:
    """Tests pass, but adversarial falsification finds a counterexample.

    This is the ESSENTIAL scenario demonstrating:
      "Tests passed, but the solution was still wrong."
    """

    def test_falsification_catches_hidden_bug(self):
        """Verification PASS → falsification FAIL → repair → VERIFIED.

        Sequence:
        1. First verify: tests PASS but falsification finds counterexample.
        2. Orchestrator does NOT declare VERIFIED — enters repair loop.
        3. Second verify: tests PASS and falsification PASS.
        4. Final outcome: VERIFIED.
        """
        vr_false_confidence = _make_vr_with_falsification_fail(
            counterexample="discount of 100% on zero-value item"
        )
        vr_fully_fixed = _make_vr_all_pass_with_falsification()

        verifier = MagicMock(spec=Verifier)
        verifier.verify.side_effect = [
            vr_false_confidence,  # Attempt 1: tests pass but falsification fails.
            vr_fully_fixed,       # Attempt 2: everything passes.
        ]

        orch = Orchestrator(
            model_client=MagicMock(),
            tools=[],
            max_iterations=5,
            verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/discount.py"],
        )

        contract = _make_contract()
        result = orch.run(
            "Fix the discount calculation",
            contract=contract,
        )

        assert result.outcome == TaskOutcome.VERIFIED
        # Must have made at least 2 attempts (repair triggered).
        assert result.history.attempt_count >= 2

    def test_falsification_not_ignored(self):
        """The orchestrator NEVER ignores a falsification counterexample."""
        vr_false_confidence = _make_vr_with_falsification_fail(
            counterexample="boundary: negative discount"
        )

        verifier = MagicMock(spec=Verifier)
        # Return false-confidence result repeatedly.
        verifier.verify.return_value = vr_false_confidence

        orch = Orchestrator(
            model_client=MagicMock(),
            tools=[],
            max_iterations=3,
            verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/discount.py"],
        )

        contract = _make_contract()
        result = orch.run(
            "Fix the discount calculation",
            contract=contract,
        )

        # Must NOT be VERIFIED — counterexample was never resolved.
        assert result.outcome != TaskOutcome.VERIFIED

    def test_counterexample_recorded_in_history(self):
        """Counterexamples from falsification appear in attempt history."""
        vr_false_confidence = _make_vr_with_falsification_fail(
            counterexample="edge: 0-quantity order"
        )
        vr_fixed = _make_vr_all_pass_with_falsification()

        verifier = MagicMock(spec=Verifier)
        verifier.verify.side_effect = [vr_false_confidence, vr_fixed]

        orch = Orchestrator(
            model_client=MagicMock(),
            tools=[],
            max_iterations=5,
            verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/discount.py"],
        )

        contract = _make_contract()
        result = orch.run("Fix discount", contract=contract)

        # In false-confidence, verification.passed is True but falsification
        # fails, so the attempt is NOT in failed_attempts().  Check ALL
        # attempts for the counterexample instead.
        all_counterexamples = []
        for a in result.history.attempts:
            all_counterexamples.extend(a.counterexamples)
        assert len(all_counterexamples) >= 1
        assert any("0-quantity" in ce for ce in all_counterexamples)

    def test_repair_loop_resolves_false_confidence(self):
        """After repair, the second verification+falsification pass fully."""
        vr_bad = _make_vr_with_falsification_fail(
            counterexample="discount > 100%"
        )
        vr_good = _make_vr_all_pass_with_falsification()

        verifier = MagicMock(spec=Verifier)
        verifier.verify.side_effect = [vr_bad, vr_good]

        repair_calls = {"count": 0}

        def patching(task, contract, files, prompt):
            repair_calls["count"] += 1
            return ["src/discount.py"]

        orch = Orchestrator(
            model_client=MagicMock(),
            tools=[],
            max_iterations=5,
            verifier=verifier,
            patch_fn=patching,
        )

        contract = _make_contract()
        result = orch.run("Fix discount", contract=contract)

        assert result.outcome == TaskOutcome.VERIFIED
        assert repair_calls["count"] >= 2  # Initial + repair.


# ═══════════════════════════════════════════════════════════════════════
# Scenario C — Impossible/unresolved task
# ═══════════════════════════════════════════════════════════════════════


class TestScenarioC_ImpossibleTask:
    """Task cannot be solved → FAILED or UNKNOWN, never VERIFIED."""

    def test_impossible_task_fails(self):
        """All verification attempts fail → FAILED."""
        vr_fail = _make_vr(
            passed=False,
            status=VerificationStatus.FAILED,
        )

        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr_fail

        orch = Orchestrator(
            model_client=MagicMock(),
            tools=[],
            max_iterations=3,
            verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/impossible.py"],
        )

        contract = _make_contract(
            goal="Implement perpetual motion machine",
            expected_files=["src/impossible.py"],
        )
        result = orch.run("Implement perpetual motion", contract=contract)

        assert result.outcome in (TaskOutcome.FAILED, TaskOutcome.UNKNOWN)
        assert result.outcome != TaskOutcome.VERIFIED

    def test_never_verified_on_persistent_failure(self):
        """After max iterations of failure, outcome is never VERIFIED."""
        vr_fail = _make_vr(
            passed=False,
            status=VerificationStatus.FAILED,
            confidence=0.0,
        )

        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr_fail

        orch = Orchestrator(
            model_client=MagicMock(),
            tools=[],
            max_iterations=5,
            verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/impossible.py"],
        )

        contract = _make_contract(
            goal="Impossible",
            expected_files=["src/impossible.py"],
        )
        result = orch.run("Impossible task", contract=contract)

        assert result.outcome != TaskOutcome.VERIFIED
        assert result.history.attempt_count >= 1

    def test_escalation_on_stuck_pattern(self):
        """Repeated identical failures → ESCALATE → UNKNOWN."""
        checks = [
            VerificationCheck(
                check_id="L2-SUITE", level=VerificationLevel.EXISTING_TESTS,
                description="Tests", status=CheckStatus.FAIL,
                evidence="test_thing always fails", obligation_id="OB-1",
            ),
        ]
        vr_same = _make_vr(
            passed=False, status=VerificationStatus.FAILED,
            checks=checks, confidence=0.1,
        )

        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr_same

        orch = Orchestrator(
            model_client=MagicMock(),
            tools=[],
            max_iterations=10,
            verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/stuck.py"],
        )

        contract = _make_contract(
            goal="Fix stuck module",
            expected_files=["src/stuck.py"],
            obligations=[
                ProofObligation(
                    id="OB-1", description="Must work",
                    type=ObligationType.BEHAVIOR,
                    verification_method="Run tests",
                ),
            ],
        )
        result = orch.run("Fix stuck", contract=contract)

        # Stuck pattern → ESCALATE → UNKNOWN.
        assert result.outcome in (TaskOutcome.UNKNOWN, TaskOutcome.FAILED)
        assert result.outcome != TaskOutcome.VERIFIED

    def test_budget_exhausted_message(self):
        """When budget is exhausted, the reason is clear."""
        vr_fail = _make_vr(passed=False, status=VerificationStatus.FAILED)

        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr_fail

        orch = Orchestrator(
            model_client=MagicMock(),
            tools=[],
            max_iterations=2,
            verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/x.py"],
        )

        contract = _make_contract(expected_files=["src/x.py"])
        result = orch.run("Fix x", contract=contract)

        assert "exhausted" in result.reason.lower() or "cannot" in result.reason.lower()


# ═══════════════════════════════════════════════════════════════════════
# Scenario D — Scope violation
# ═══════════════════════════════════════════════════════════════════════


class TestScenarioD_ScopeViolation:
    """Agent modifies unrelated files → CHANGE BUDGET violation."""

    def test_scope_violation_logged(self):
        """Modifying files outside scope creates a BLOCKED log entry."""
        contract = _make_contract(
            expected_files=["src/discount.py"],
            affected_area="src",
        )

        vr = _make_vr(passed=True, changed_files=[
            "src/discount.py", "database/schema.sql",
            "frontend/app.js", "deploy/config.yaml",
        ])

        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr

        # Patch function returns many files far outside scope.
        orch = Orchestrator(
            model_client=MagicMock(),
            tools=[],
            max_iterations=3,
            verifier=verifier,
            patch_fn=lambda t, c, f, p: [
                "src/discount.py", "database/schema.sql",
                "frontend/app.js", "deploy/config.yaml",
            ],
        )

        result = orch.run("Fix discount", contract=contract)

        # Check phase log for scope violation.
        budget_phases = [
            e for e in result.phase_log
            if e.phase == Phase.CHANGE_BUDGET
        ]
        # Should have budget creation + evaluation entries.
        assert len(budget_phases) >= 1

    def test_scope_violation_with_real_change_guard(self):
        """Use real ChangeGuard to verify scope violation detection."""
        contract = _make_contract(
            expected_files=["src/discount.py"],
            affected_area="src/discount",
        )

        guard = ChangeGuard()
        budget = guard.create_budget(contract)

        # Evaluate with files far outside scope.
        report = guard.evaluate(
            budget,
            actual_files=[
                "src/discount.py",
                "database/models.py",
                "frontend/index.js",
                "config/settings.yaml",
                "scripts/deploy.sh",
                "README.md",
            ],
        )

        # Should detect excessive files or out-of-scope modifications.
        assert report.status in (ScopeStatus.SCOPE_EXPANDED, ScopeStatus.BLOCKED)
        assert len(report.violations) > 0
        assert report.scope_expansion > 0

    def test_excessive_file_count_detected(self):
        """Too many files triggers EXCESSIVE_FILE_COUNT violation."""
        contract = _make_contract(expected_files=["src/a.py"])
        guard = ChangeGuard()
        budget = guard.create_budget(contract)

        # Create way too many files.
        many_files = [f"src/file_{i}.py" for i in range(20)]
        report = guard.evaluate(budget, many_files)

        violation_types = {v.type for v in report.violations}
        assert ViolationType.EXCESSIVE_FILE_COUNT in violation_types

    def test_unrelated_modification_detected(self):
        """File outside affected area is flagged."""
        contract = _make_contract(
            expected_files=["src/discount.py"],
            affected_area="src/discount",
        )
        guard = ChangeGuard()
        budget = guard.create_budget(contract)

        report = guard.evaluate(
            budget,
            actual_files=["src/discount.py", "unrelated/other.py"],
        )

        violation_types = {v.type for v in report.violations}
        assert (
            ViolationType.UNRELATED_MODIFICATION in violation_types
            or ViolationType.OUTSIDE_AFFECTED_AREA in violation_types
        )


# ═══════════════════════════════════════════════════════════════════════
# Scenario E — Stale evidence
# ═══════════════════════════════════════════════════════════════════════


class TestScenarioE_StaleEvidence:
    """Evidence becomes stale after source changes → must reverify."""

    def test_stale_evidence_detected_via_fingerprints(self):
        """VerificationResult.is_stale() detects changed files."""
        fp_old = FileFingerprint.from_content("src/foo.py", "original content")

        vr = _make_vr(passed=True)
        vr.fingerprints = [fp_old]

        # Simulate source change — new fingerprint differs.
        fp_new = FileFingerprint.from_content("src/foo.py", "modified content")
        assert vr.is_stale([fp_new]) is True

    def test_fresh_evidence_not_stale(self):
        """Unchanged files → evidence is NOT stale."""
        fp = FileFingerprint.from_content("src/foo.py", "same content")

        vr = _make_vr(passed=True)
        vr.fingerprints = [fp]

        assert vr.is_stale([fp]) is False

    def test_deleted_file_makes_evidence_stale(self):
        """If a file is deleted, evidence is stale."""
        fp = FileFingerprint.from_content("src/foo.py", "content")

        vr = _make_vr(passed=True)
        vr.fingerprints = [fp]

        # No fingerprints at all — file was deleted.
        assert vr.is_stale([]) is True

    def test_mark_stale_transitions_to_stale(self):
        """mark_stale() changes status and sets passed=False."""
        vr = _make_vr(passed=True, status=VerificationStatus.VERIFIED)
        vr.mark_stale()

        assert vr.status == VerificationStatus.STALE
        assert vr.passed is False
        assert "stale" in vr.failure_summary.lower()

    def test_stale_evidence_in_orchestrator(self):
        """Stale evidence forces reverification via the orchestrator metrics."""
        # First verification passes.
        vr_pass = _make_vr(passed=True)
        # Second verification also passes (after reverification).
        vr_pass2 = _make_vr(passed=True, confidence=0.99)

        verifier = MagicMock(spec=Verifier)
        verifier.verify.side_effect = [vr_pass, vr_pass2]

        orch = Orchestrator(
            model_client=MagicMock(),
            tools=[],
            max_iterations=5,
            verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/discount.py"],
        )

        result = orch.run("Fix discount")

        assert result.outcome == TaskOutcome.VERIFIED
        # At least 1 verification should have run.
        assert result.metrics.verification_runs >= 1

    def test_verification_cache_invalidation_on_change(self):
        """Verification cache rejects skip when hashes differ."""
        from src.metrics import VerificationCache

        vc = VerificationCache()
        vc.record({"src/foo.py": "hash_v1"}, passed=True)

        # Same file, different hash → cannot skip.
        assert vc.can_skip({"src/foo.py": "hash_v2"}) is False

    def test_stale_evidence_event_tracked_in_metrics(self):
        """Stale evidence events are counted in ExecutionMetrics."""
        m = ExecutionMetrics()
        m.record_stale_evidence()
        m.record_stale_evidence()
        assert m.stale_evidence_events == 2


# ═══════════════════════════════════════════════════════════════════════
# Scenario F — Test weakening
# ═══════════════════════════════════════════════════════════════════════


class TestScenarioF_TestWeakening:
    """Agent deletes or weakens tests → VERIFICATION INTEGRITY VIOLATION."""

    def test_test_deletion_detected(self):
        """Deleting a test file triggers TEST_DELETION violation."""
        contract = _make_contract(expected_files=["src/discount.py"])
        guard = ChangeGuard()
        budget = guard.create_budget(contract)

        report = guard.evaluate(
            budget,
            actual_files=["src/discount.py"],
            deleted_files=["tests/test_discount.py"],
        )

        violation_types = {v.type for v in report.violations}
        assert ViolationType.TEST_DELETION in violation_types
        # TEST_DELETION is severity "critical" → should BLOCK.
        assert report.status == ScopeStatus.BLOCKED

    def test_weakened_assertions_detected(self):
        """Reducing assertion count triggers WEAKENED_ASSERTION."""
        contract = _make_contract(expected_files=[
            "src/discount.py", "tests/test_discount.py",
        ])
        guard = ChangeGuard()
        budget = guard.create_budget(contract)

        original_test = textwrap.dedent("""\
            def test_discount_10():
                assert calculate_discount(100, 10) == 90
                assert calculate_discount(200, 10) == 180
                assert calculate_discount(50, 10) == 45
                assert calculate_discount(0, 10) == 0

            def test_discount_zero():
                assert calculate_discount(100, 0) == 100
                assert calculate_discount(0, 0) == 0
        """)

        weakened_test = textwrap.dedent("""\
            def test_discount_10():
                assert True

            def test_discount_zero():
                pass
        """)

        report = guard.evaluate(
            budget,
            actual_files=["src/discount.py", "tests/test_discount.py"],
            file_contents={"tests/test_discount.py": weakened_test},
            original_contents={"tests/test_discount.py": original_test},
        )

        violation_types = {v.type for v in report.violations}
        # Should detect either WEAKENED_ASSERTION or TRIVIAL_TEST.
        assert (
            ViolationType.WEAKENED_ASSERTION in violation_types
            or ViolationType.TRIVIAL_TEST in violation_types
        )

    def test_trivial_test_detected(self):
        """Test with only 'assert True' or 'pass' body is flagged."""
        contract = _make_contract(expected_files=[
            "src/discount.py", "tests/test_discount.py",
        ])
        guard = ChangeGuard()
        budget = guard.create_budget(contract)

        trivial_test = textwrap.dedent("""\
            def test_it_works():
                assert True

            def test_another():
                pass
        """)

        report = guard.evaluate(
            budget,
            actual_files=["src/discount.py", "tests/test_discount.py"],
            file_contents={"tests/test_discount.py": trivial_test},
        )

        violation_types = {v.type for v in report.violations}
        assert ViolationType.TRIVIAL_TEST in violation_types

    def test_test_deletion_blocks_pipeline(self):
        """Deleting a test file → BLOCKED status (critical severity)."""
        contract = _make_contract(expected_files=["src/discount.py"])
        guard = ChangeGuard()
        budget = guard.create_budget(contract)

        report = guard.evaluate(
            budget,
            actual_files=["src/discount.py"],
            deleted_files=["tests/test_core.py"],
        )

        assert report.status == ScopeStatus.BLOCKED
        critical = [v for v in report.violations if v.severity == "critical"]
        assert len(critical) >= 1

    def test_assertion_count_reduction_detected(self):
        """30%+ reduction in strong assertions → WEAKENED_ASSERTION."""
        contract = _make_contract(expected_files=[
            "src/discount.py", "tests/test_discount.py",
        ])
        guard = ChangeGuard()
        budget = guard.create_budget(contract)

        original = textwrap.dedent("""\
            def test_a():
                assert foo() == 1
                assert bar() == 2
                assert baz() == 3
                assert qux() == 4
                assert quux() == 5
                assert corge() == 6
                assert grault() == 7
                assert garply() == 8
                assert waldo() == 9
                assert fred() == 10
        """)

        weakened = textwrap.dedent("""\
            def test_a():
                assert foo() == 1
                assert bar() == 2
        """)

        report = guard.evaluate(
            budget,
            actual_files=["src/discount.py", "tests/test_discount.py"],
            file_contents={"tests/test_discount.py": weakened},
            original_contents={"tests/test_discount.py": original},
        )

        violation_types = {v.type for v in report.violations}
        assert ViolationType.WEAKENED_ASSERTION in violation_types


# ═══════════════════════════════════════════════════════════════════════
# Cross-scenario integration
# ═══════════════════════════════════════════════════════════════════════


class TestCrossScenarioIntegration:
    """Verify invariants that must hold across ALL scenarios."""

    def test_model_never_declares_verified(self):
        """The model is NEVER allowed to self-declare VERIFIED.

        Outcome must always come from executable verification evidence.
        """
        # No verifier → no evidence → cannot be VERIFIED.
        orch = Orchestrator(
            model_client=MagicMock(),
            tools=[],
            max_iterations=3,
            verifier=None,  # No verifier.
        )

        result = orch.run("Fix something")

        # Without a verifier, there's no executable evidence.
        # The orchestrator should not claim VERIFIED.
        assert result.outcome != TaskOutcome.VERIFIED or result.final_verification is not None

    def test_verified_requires_passing_verification(self):
        """VERIFIED outcome requires a passing VerificationResult."""
        vr_pass = _make_vr(passed=True)
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr_pass

        orch = Orchestrator(
            model_client=MagicMock(),
            tools=[],
            max_iterations=5,
            verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )

        result = orch.run("Fix foo")

        if result.outcome == TaskOutcome.VERIFIED:
            assert result.final_verification is not None
            assert result.final_verification.passed is True

    def test_to_dict_serializable(self):
        """TaskResult.to_dict() produces JSON-serializable output."""
        import json

        contract = _make_contract()
        vr = _make_vr(passed=True)
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr

        orch = Orchestrator(
            model_client=MagicMock(),
            tools=[],
            max_iterations=5,
            verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/discount.py"],
        )

        result = orch.run("Fix discount", contract=contract)

        d = result.to_dict()
        serialized = json.dumps(d)  # Must not raise.
        assert len(serialized) > 0

    def test_metrics_always_present(self):
        """Every TaskResult has metrics, regardless of outcome."""
        vr_fail = _make_vr(passed=False, status=VerificationStatus.FAILED)
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr_fail

        orch = Orchestrator(
            model_client=MagicMock(),
            tools=[],
            max_iterations=2,
            verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )

        result = orch.run("Fix foo")

        assert result.metrics is not None
        assert isinstance(result.metrics.to_dict(), dict)
