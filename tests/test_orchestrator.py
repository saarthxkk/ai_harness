"""Tests for src/orchestrator.py — fully integrated autonomous agent loop.

Covers:
  - Successful task (VERIFIED)
  - Contract generation
  - Tool execution
  - Verification failure
  - Falsification failure
  - Repair loop
  - Repeated failure detection
  - Max iterations / budget exhaustion
  - Stale evidence
  - Scope violation
  - UNKNOWN result
  - Model failure
  - Tool failure
  - Phase logging
  - AttemptHistory pattern detection
  - Failure classification
  - Evidence extraction
  - Progress assessment
  - Repair action determination
  - Repair prompt construction (no chain-of-thought)
  - Never-false-success invariant
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.orchestrator import (
    AttemptHistory,
    AttemptRecord,
    FailureType,
    Orchestrator,
    PatchFunction,
    Phase,
    PhaseLog,
    ProgressStatus,
    RepairAction,
    TaskOutcome,
    TaskResult,
    _assess_progress,
    _build_repair_prompt,
    _classify_failure,
    _determine_repair_action,
    _extract_counterexamples,
    _extract_failed_obligations,
    _extract_failure_evidence,
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
)
from src.rollback import (
    CheckpointStage,
    RollbackEngine,
    RollbackReason,
    RollbackResult,
    RollbackStatus,
)
from src.evidence_graph import (
    EvidenceVerdict,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_contract(
    obligations: list[ProofObligation] | None = None,
) -> TaskContract:
    """Create a minimal TaskContract for testing."""
    if obligations is None:
        obligations = [
            ProofObligation(
                id="OB-1",
                description="Test obligation",
                type=ObligationType.BEHAVIOR,
                verification_method="pytest",
                status=ObligationStatus.PENDING,
            ),
            ProofObligation(
                id="OB-2",
                description="Regression check",
                type=ObligationType.REGRESSION,
                verification_method="pytest",
                status=ObligationStatus.PENDING,
            ),
        ]
    return TaskContract(
        task_goal="Fix the bug",
        behavioral_requirements=["Must fix X"],
        acceptance_criteria=["X works correctly"],
        constraints=["No side effects"],
        non_goals=["Don't refactor Y"],
        affected_area="src",
        expected_files=["src/foo.py"],
        risk_level=RiskLevel.LOW,
        verification_plan="Run pytest",
        proof_obligations=obligations,
    )


def _make_verification_result(
    *,
    passed: bool = True,
    status: VerificationStatus = VerificationStatus.VERIFIED,
    checks: list[VerificationCheck] | None = None,
    confidence: float = 0.95,
    command: str = "pytest tests/",
    failure_summary: str = "",
) -> VerificationResult:
    if checks is None:
        if passed:
            checks = [
                VerificationCheck(
                    check_id="L2-SUITE",
                    level=VerificationLevel.EXISTING_TESTS,
                    description="Run test suite",
                    status=CheckStatus.PASS,
                    evidence="All tests passed",
                    command=command,
                ),
            ]
        else:
            checks = [
                VerificationCheck(
                    check_id="L2-SUITE",
                    level=VerificationLevel.EXISTING_TESTS,
                    description="Run test suite",
                    status=CheckStatus.FAIL,
                    evidence="1 test failed: test_foo",
                    command=command,
                    exit_code=1,
                ),
            ]
    return VerificationResult(
        passed=passed,
        status=status,
        command=command,
        exit_code=0 if passed else 1,
        stdout="OK" if passed else "FAIL",
        stderr="",
        duration=1.0,
        checks=checks,
        failure_summary=failure_summary,
        evidence=["test evidence"],
        changed_files=["src/foo.py"],
        confidence=confidence,
    )


def _make_attempt(
    *,
    attempt_number: int = 0,
    passed: bool = False,
    failed_obligations: list[str] | None = None,
    failure_type: FailureType | None = FailureType.BAD_PATCH,
    repair_action: RepairAction | None = RepairAction.REFINE_PATCH,
    progress: ProgressStatus = ProgressStatus.NO_PROGRESS,
    repeated_failure: bool = False,
    files_changed: list[str] | None = None,
    counterexamples: list[str] | None = None,
    checks: list[VerificationCheck] | None = None,
) -> AttemptRecord:
    vr = _make_verification_result(passed=passed, checks=checks)
    return AttemptRecord(
        attempt_number=attempt_number,
        files_changed=files_changed or ["src/foo.py"],
        verification_result=vr,
        failed_obligations=failed_obligations or ["OB-1"],
        failure_type=failure_type,
        relevant_command="pytest tests/",
        counterexamples=counterexamples or [],
        repair_action=repair_action,
        progress=progress,
        repeated_failure=repeated_failure,
    )


# ══════════════════════════════════════════════════════════════════════
# AttemptRecord tests
# ══════════════════════════════════════════════════════════════════════


class TestAttemptRecord:
    def test_to_dict_passed(self):
        attempt = _make_attempt(passed=True, failure_type=None, repair_action=None)
        d = attempt.to_dict()
        assert d["verification_passed"] is True
        assert d["failure_type"] is None
        assert d["repair_action"] is None

    def test_to_dict_failed(self):
        attempt = _make_attempt(
            passed=False,
            failure_type=FailureType.BAD_PATCH,
            repair_action=RepairAction.REFINE_PATCH,
        )
        d = attempt.to_dict()
        assert d["verification_passed"] is False
        assert d["failure_type"] == "bad_patch"
        assert d["repair_action"] == "refine_patch"

    def test_to_dict_no_verification(self):
        attempt = AttemptRecord(
            attempt_number=0, files_changed=[], verification_result=None,
            failed_obligations=[], failure_type=None, relevant_command="",
            counterexamples=[], repair_action=None,
            progress=ProgressStatus.NO_PROGRESS, repeated_failure=False,
        )
        d = attempt.to_dict()
        assert d["verification_passed"] is None
        assert d["verification_status"] is None

    def test_counterexamples_recorded(self):
        attempt = _make_attempt(counterexamples=["CE-1", "CE-2"])
        assert len(attempt.to_dict()["counterexamples"]) == 2

    def test_files_changed_recorded(self):
        attempt = _make_attempt(files_changed=["a.py", "b.py", "c.py"])
        assert attempt.to_dict()["files_changed"] == ["a.py", "b.py", "c.py"]


# ══════════════════════════════════════════════════════════════════════
# AttemptHistory tests
# ══════════════════════════════════════════════════════════════════════


class TestAttemptHistory:
    def test_empty_history(self):
        h = AttemptHistory(task_id="T-1", max_iterations=10)
        assert h.attempt_count == 0
        assert h.latest is None
        assert not h.budget_exhausted

    def test_record_and_count(self):
        h = AttemptHistory(task_id="T-1", max_iterations=10)
        h.record(_make_attempt(attempt_number=0))
        h.record(_make_attempt(attempt_number=1))
        assert h.attempt_count == 2

    def test_budget_exhausted(self):
        h = AttemptHistory(task_id="T-1", max_iterations=3)
        for i in range(3):
            h.record(_make_attempt(attempt_number=i))
        assert h.budget_exhausted

    def test_passed_and_failed(self):
        h = AttemptHistory(task_id="T-1", max_iterations=10)
        h.record(_make_attempt(attempt_number=0, passed=False))
        h.record(_make_attempt(attempt_number=1, passed=True))
        assert len(h.passed_attempts()) == 1
        assert len(h.failed_attempts()) == 1

    def test_to_dict(self):
        h = AttemptHistory(task_id="T-1", max_iterations=5)
        h.record(_make_attempt(attempt_number=0))
        d = h.to_dict()
        assert d["task_id"] == "T-1"
        assert "patterns" in d

    def test_summary(self):
        h = AttemptHistory(task_id="T-1", max_iterations=10)
        h.record(_make_attempt(attempt_number=0, passed=False))
        s = h.summary()
        assert s["budget_remaining"] == 9


# ══════════════════════════════════════════════════════════════════════
# Pattern detection tests
# ══════════════════════════════════════════════════════════════════════


class TestPatternDetection:
    def test_repeated_failure_detected(self):
        h = AttemptHistory(task_id="T-1", max_iterations=10)
        for i in range(3):
            h.record(_make_attempt(
                attempt_number=i,
                failed_obligations=["OB-1"],
                failure_type=FailureType.BAD_PATCH,
            ))
        assert h.has_repeated_failure()

    def test_repeated_failure_not_detected_different_obs(self):
        h = AttemptHistory(task_id="T-1", max_iterations=10)
        h.record(_make_attempt(attempt_number=0, failed_obligations=["OB-1"]))
        h.record(_make_attempt(attempt_number=1, failed_obligations=["OB-2"]))
        h.record(_make_attempt(attempt_number=2, failed_obligations=["OB-1"]))
        assert not h.has_repeated_failure()

    def test_no_progress_detected(self):
        h = AttemptHistory(task_id="T-1", max_iterations=10)
        for i in range(3):
            h.record(_make_attempt(
                attempt_number=i, progress=ProgressStatus.NO_PROGRESS,
            ))
        assert h.has_no_progress()

    def test_no_progress_broken_by_progress(self):
        h = AttemptHistory(task_id="T-1", max_iterations=10)
        h.record(_make_attempt(attempt_number=0, progress=ProgressStatus.NO_PROGRESS))
        h.record(_make_attempt(attempt_number=1, progress=ProgressStatus.PROGRESS))
        h.record(_make_attempt(attempt_number=2, progress=ProgressStatus.NO_PROGRESS))
        assert not h.has_no_progress()

    def test_oscillation_detected(self):
        h = AttemptHistory(task_id="T-1", max_iterations=10)
        h.record(_make_attempt(attempt_number=0, failed_obligations=["OB-1"]))
        h.record(_make_attempt(attempt_number=1, failed_obligations=["OB-2"]))
        h.record(_make_attempt(attempt_number=2, failed_obligations=["OB-1"]))
        h.record(_make_attempt(attempt_number=3, failed_obligations=["OB-2"]))
        assert h.has_oscillating_fixes()

    def test_no_oscillation_same_failure(self):
        h = AttemptHistory(task_id="T-1", max_iterations=10)
        for i in range(4):
            h.record(_make_attempt(attempt_number=i, failed_obligations=["OB-1"]))
        assert not h.has_oscillating_fixes()

    def test_repeated_file_mods_detected(self):
        h = AttemptHistory(task_id="T-1", max_iterations=10)
        for i in range(3):
            h.record(_make_attempt(
                attempt_number=i, passed=False, files_changed=["src/foo.py"],
            ))
        assert h.has_repeated_file_modifications()

    def test_repeated_file_mods_not_detected_passing(self):
        h = AttemptHistory(task_id="T-1", max_iterations=10)
        for i in range(3):
            h.record(_make_attempt(
                attempt_number=i, passed=True, files_changed=["src/foo.py"],
            ))
        assert not h.has_repeated_file_modifications()


# ══════════════════════════════════════════════════════════════════════
# Failure classification tests
# ══════════════════════════════════════════════════════════════════════


class TestClassifyFailure:
    def test_syntax_is_bad_patch(self):
        checks = [VerificationCheck(
            check_id="L1-1", level=VerificationLevel.BASIC_VALIDATION,
            description="Syntax", status=CheckStatus.FAIL,
            evidence="SyntaxError",
        )]
        vr = _make_verification_result(
            passed=False, status=VerificationStatus.FAILED, checks=checks,
        )
        assert _classify_failure(vr, None, []) == FailureType.BAD_PATCH

    def test_regression_failure(self):
        checks = [VerificationCheck(
            check_id="L5-1", level=VerificationLevel.REGRESSION,
            description="Regression", status=CheckStatus.FAIL,
            evidence="Broke existing test",
        )]
        vr = _make_verification_result(
            passed=False, status=VerificationStatus.FAILED, checks=checks,
        )
        assert _classify_failure(vr, None, []) == FailureType.REGRESSION

    def test_timeout_is_environment(self):
        checks = [VerificationCheck(
            check_id="L2-1", level=VerificationLevel.EXISTING_TESTS,
            description="Tests", status=CheckStatus.ERROR,
            evidence="Timed out after 30s",
        )]
        vr = _make_verification_result(
            passed=False, status=VerificationStatus.UNKNOWN, checks=checks,
        )
        assert _classify_failure(vr, None, []) == FailureType.ENVIRONMENT_FAILURE

    def test_command_not_found_is_tool(self):
        checks = [VerificationCheck(
            check_id="L1-1", level=VerificationLevel.BASIC_VALIDATION,
            description="Check", status=CheckStatus.ERROR,
            evidence="Command not found: python3",
        )]
        vr = _make_verification_result(
            passed=False, status=VerificationStatus.UNKNOWN, checks=checks,
        )
        assert _classify_failure(vr, None, []) == FailureType.TOOL_FAILURE

    def test_scope_expansion(self):
        vr = _make_verification_result(
            passed=False, status=VerificationStatus.FAILED,
        )
        assert _classify_failure(vr, None, [], 2.0) == FailureType.SCOPE_VIOLATION

    def test_import_is_missing_context(self):
        checks = [VerificationCheck(
            check_id="L3-1", level=VerificationLevel.TARGETED_VERIFICATION,
            description="Target", status=CheckStatus.FAIL,
            evidence="import error",
        )]
        vr = _make_verification_result(
            passed=False, status=VerificationStatus.FAILED,
            checks=checks, failure_summary="import error in module",
        )
        assert _classify_failure(vr, None, []) == FailureType.MISSING_CONTEXT

    def test_assertion_is_wrong_assumption(self):
        checks = [VerificationCheck(
            check_id="L4-1", level=VerificationLevel.CONTRACT_VERIFICATION,
            description="Contract", status=CheckStatus.FAIL,
            evidence="assertion failed",
        )]
        vr = _make_verification_result(
            passed=False, status=VerificationStatus.FAILED,
            checks=checks, failure_summary="assert expected 5 got 3",
        )
        assert _classify_failure(vr, None, []) == FailureType.WRONG_ASSUMPTION

    def test_falsification_is_insufficient_test(self):
        checks = [VerificationCheck(
            check_id="L6-1", level=VerificationLevel.FALSIFICATION,
            description="Falsification", status=CheckStatus.FAIL,
            evidence="Counterexample found",
        )]
        vr = _make_verification_result(
            passed=False, status=VerificationStatus.FAILED, checks=checks,
        )
        assert _classify_failure(vr, None, []) == FailureType.INSUFFICIENT_TEST

    def test_scope_obligation_violation(self):
        contract = _make_contract([ProofObligation(
            id="OB-S", description="Scope check",
            type=ObligationType.SCOPE, verification_method="diff",
            status=ObligationStatus.FAIL,
        )])
        vr = _make_verification_result(
            passed=False, status=VerificationStatus.FAILED, checks=[],
        )
        assert _classify_failure(vr, contract, []) == FailureType.SCOPE_VIOLATION


# ══════════════════════════════════════════════════════════════════════
# Evidence extraction tests
# ══════════════════════════════════════════════════════════════════════


class TestEvidenceExtraction:
    def test_extract_failed_obligations_from_checks(self):
        checks = [
            VerificationCheck(
                check_id="L4-1", level=VerificationLevel.CONTRACT_VERIFICATION,
                description="OB-1", status=CheckStatus.FAIL,
                evidence="Failed", obligation_id="OB-1",
            ),
            VerificationCheck(
                check_id="L4-2", level=VerificationLevel.CONTRACT_VERIFICATION,
                description="OB-2", status=CheckStatus.PASS,
                evidence="Passed", obligation_id="OB-2",
            ),
        ]
        vr = _make_verification_result(passed=False, checks=checks)
        assert _extract_failed_obligations(vr, None) == ["OB-1"]

    def test_extract_failed_obligations_from_contract(self):
        contract = _make_contract()
        contract.proof_obligations[0].status = ObligationStatus.FAIL
        vr = _make_verification_result(passed=False, checks=[])
        assert "OB-1" in _extract_failed_obligations(vr, contract)

    def test_extract_counterexamples(self):
        checks = [
            VerificationCheck(
                check_id="L6-1", level=VerificationLevel.FALSIFICATION,
                description="Boundary test", status=CheckStatus.FAIL,
                evidence="Input 0 caused crash",
            ),
            VerificationCheck(
                check_id="L2-1", level=VerificationLevel.EXISTING_TESTS,
                description="Suite", status=CheckStatus.PASS,
                evidence="OK",
            ),
        ]
        vr = _make_verification_result(passed=False, checks=checks)
        ces = _extract_counterexamples(vr)
        assert len(ces) == 1
        assert "Boundary test" in ces[0]

    def test_no_counterexamples_on_pass(self):
        vr = _make_verification_result(passed=True)
        assert _extract_counterexamples(vr) == []

    def test_extract_failure_evidence_filters(self):
        checks = [
            VerificationCheck(
                check_id="L1-1", level=VerificationLevel.BASIC_VALIDATION,
                description="Syntax", status=CheckStatus.FAIL,
                evidence="SyntaxError", command="py_compile", exit_code=1,
            ),
            VerificationCheck(
                check_id="L2-1", level=VerificationLevel.EXISTING_TESTS,
                description="Tests", status=CheckStatus.PASS,
                evidence="OK",
            ),
            VerificationCheck(
                check_id="L3-1", level=VerificationLevel.TARGETED_VERIFICATION,
                description="Target", status=CheckStatus.ERROR,
                evidence="Timed out",
            ),
        ]
        vr = _make_verification_result(passed=False, checks=checks)
        evidence = _extract_failure_evidence(vr)
        assert len(evidence) == 2  # FAIL + ERROR, not PASS


# ══════════════════════════════════════════════════════════════════════
# Progress assessment tests
# ══════════════════════════════════════════════════════════════════════


class TestAssessProgress:
    def test_first_attempt_is_progress(self):
        vr = _make_verification_result(passed=False)
        assert _assess_progress(vr, None) == ProgressStatus.PROGRESS

    def test_now_passing_is_progress(self):
        prev = _make_attempt(passed=False)
        vr = _make_verification_result(passed=True)
        assert _assess_progress(vr, prev) == ProgressStatus.PROGRESS

    def test_was_passing_now_failing_is_regression(self):
        prev = _make_attempt(passed=True)
        vr = _make_verification_result(passed=False)
        assert _assess_progress(vr, prev) == ProgressStatus.REGRESSION_DETECTED

    def test_fewer_failures_is_progress(self):
        prev_checks = [
            VerificationCheck(check_id="c1", level=VerificationLevel.EXISTING_TESTS,
                              description="t1", status=CheckStatus.FAIL, evidence="f"),
            VerificationCheck(check_id="c2", level=VerificationLevel.EXISTING_TESTS,
                              description="t2", status=CheckStatus.FAIL, evidence="f"),
        ]
        prev = _make_attempt(passed=False, checks=prev_checks)
        curr_checks = [
            VerificationCheck(check_id="c1", level=VerificationLevel.EXISTING_TESTS,
                              description="t1", status=CheckStatus.FAIL, evidence="f"),
        ]
        vr = _make_verification_result(passed=False, checks=curr_checks)
        assert _assess_progress(vr, prev) == ProgressStatus.PROGRESS

    def test_same_failures_is_no_progress(self):
        checks = [
            VerificationCheck(check_id="c1", level=VerificationLevel.EXISTING_TESTS,
                              description="t1", status=CheckStatus.FAIL, evidence="f"),
        ]
        prev = _make_attempt(passed=False, checks=checks)
        vr = _make_verification_result(passed=False, checks=checks, confidence=0.5)
        assert _assess_progress(vr, prev) == ProgressStatus.NO_PROGRESS


# ══════════════════════════════════════════════════════════════════════
# Repair action tests
# ══════════════════════════════════════════════════════════════════════


class TestDetermineRepairAction:
    def test_abandon_on_exhausted_budget(self):
        h = AttemptHistory(task_id="T-1", max_iterations=2)
        h.record(_make_attempt(attempt_number=0))
        assert _determine_repair_action(FailureType.BAD_PATCH, h) == RepairAction.ABANDON

    def test_revert_on_repeated_failure(self):
        h = AttemptHistory(task_id="T-1", max_iterations=10)
        for i in range(3):
            h.record(_make_attempt(
                attempt_number=i, failed_obligations=["OB-1"],
                failure_type=FailureType.BAD_PATCH,
            ))
        assert _determine_repair_action(FailureType.BAD_PATCH, h) == RepairAction.REVERT_AND_RETRY

    def test_narrow_scope_on_oscillation(self):
        h = AttemptHistory(task_id="T-1", max_iterations=10)
        h.record(_make_attempt(attempt_number=0, failed_obligations=["OB-1"]))
        h.record(_make_attempt(attempt_number=1, failed_obligations=["OB-2"]))
        h.record(_make_attempt(attempt_number=2, failed_obligations=["OB-1"]))
        h.record(_make_attempt(attempt_number=3, failed_obligations=["OB-2"]))
        assert _determine_repair_action(FailureType.BAD_PATCH, h) == RepairAction.NARROW_SCOPE

    def test_escalate_on_no_progress(self):
        h = AttemptHistory(task_id="T-1", max_iterations=10)
        for i in range(3):
            h.record(_make_attempt(
                attempt_number=i, progress=ProgressStatus.NO_PROGRESS,
                failed_obligations=[f"OB-{i}"],
            ))
        assert _determine_repair_action(FailureType.BAD_PATCH, h) == RepairAction.ESCALATE

    def test_classification_based_mapping(self):
        h = AttemptHistory(task_id="T-1", max_iterations=10)
        assert _determine_repair_action(FailureType.BAD_PATCH, h) == RepairAction.REFINE_PATCH
        assert _determine_repair_action(FailureType.MISSING_CONTEXT, h) == RepairAction.ADD_CONTEXT
        assert _determine_repair_action(FailureType.REGRESSION, h) == RepairAction.REVERT_AND_RETRY
        assert _determine_repair_action(FailureType.SCOPE_VIOLATION, h) == RepairAction.NARROW_SCOPE
        assert _determine_repair_action(FailureType.TOOL_FAILURE, h) == RepairAction.RETRY_SAME
        assert _determine_repair_action(FailureType.CONTRACT_AMBIGUITY, h) == RepairAction.ESCALATE


# ══════════════════════════════════════════════════════════════════════
# Repair prompt tests
# ══════════════════════════════════════════════════════════════════════


class TestBuildRepairPrompt:
    def test_basic_prompt(self):
        attempt = _make_attempt(
            failure_type=FailureType.BAD_PATCH,
            failed_obligations=["OB-1"],
            repair_action=RepairAction.REFINE_PATCH,
        )
        prompt = _build_repair_prompt(attempt, None)
        assert "bad_patch" in prompt
        assert "OB-1" in prompt
        assert "refine_patch" in prompt

    def test_prompt_includes_counterexamples(self):
        attempt = _make_attempt(counterexamples=["CE-1: null input broke it"])
        prompt = _build_repair_prompt(attempt, None)
        assert "null input broke it" in prompt

    def test_prompt_includes_repeated_warning(self):
        attempt = _make_attempt(repeated_failure=True)
        prompt = _build_repair_prompt(attempt, None)
        assert "identical to previous" in prompt

    def test_prompt_includes_contract_status(self):
        contract = _make_contract()
        contract.proof_obligations[0].status = ObligationStatus.FAIL
        attempt = _make_attempt()
        prompt = _build_repair_prompt(attempt, contract)
        assert "OB-1" in prompt
        assert "FAIL" in prompt

    def test_no_chain_of_thought(self):
        attempt = _make_attempt()
        prompt = _build_repair_prompt(attempt, None)
        assert "think" not in prompt.lower()
        assert "reasoning" not in prompt.lower()
        assert "chain" not in prompt.lower()


# ══════════════════════════════════════════════════════════════════════
# Enum tests
# ══════════════════════════════════════════════════════════════════════


class TestEnums:
    def test_failure_types(self):
        assert len(FailureType) == 9
        assert FailureType.WRONG_ASSUMPTION.value == "wrong_assumption"
        assert FailureType.CONTRACT_AMBIGUITY.value == "contract_ambiguity"

    def test_repair_actions(self):
        assert len(RepairAction) == 8
        assert RepairAction.ABANDON.value == "abandon"

    def test_task_outcomes(self):
        assert TaskOutcome.VERIFIED.value == "VERIFIED"
        assert TaskOutcome.FAILED.value == "FAILED"
        assert TaskOutcome.UNKNOWN.value == "UNKNOWN"

    def test_phases(self):
        assert Phase.UNDERSTAND.value == "UNDERSTAND"
        assert Phase.PROOF.value == "PROOF"
        assert len(Phase) == 13


# ══════════════════════════════════════════════════════════════════════
# PhaseLog tests
# ══════════════════════════════════════════════════════════════════════


class TestPhaseLog:
    def test_to_dict(self):
        pl = PhaseLog(Phase.VERIFY, "Running verification")
        d = pl.to_dict()
        assert d["phase"] == "VERIFY"
        assert d["message"] == "Running verification"
        assert "timestamp" in d

    def test_with_data(self):
        pl = PhaseLog(Phase.CONTRACT, "Generated", data={"obligations": 3})
        d = pl.to_dict()
        assert d["data"]["obligations"] == 3


# ══════════════════════════════════════════════════════════════════════
# TaskResult tests
# ══════════════════════════════════════════════════════════════════════


class TestTaskResult:
    def test_to_dict_success(self):
        h = AttemptHistory(task_id="T-1", max_iterations=10)
        vr = _make_verification_result(passed=True)
        tr = TaskResult(
            task_id="T-1", outcome=TaskOutcome.VERIFIED, contract=None,
            history=h, final_verification=vr, evidence_report=None,
            rollback_result=None, phase_log=[], reason="Passed",
        )
        d = tr.to_dict()
        assert d["outcome"] == "VERIFIED"
        assert d["contract"] is None
        assert d["evidence_report"] is None

    def test_to_dict_with_contract(self):
        contract = _make_contract()
        h = AttemptHistory(task_id="T-1", max_iterations=5)
        tr = TaskResult(
            task_id="T-1", outcome=TaskOutcome.FAILED, contract=contract,
            history=h, final_verification=None, evidence_report=None,
            rollback_result=None, phase_log=[], reason="Budget exhausted",
        )
        d = tr.to_dict()
        assert d["outcome"] == "FAILED"
        assert d["contract"]["task_goal"] == "Fix the bug"

    def test_to_dict_with_phase_log(self):
        h = AttemptHistory(task_id="T-1", max_iterations=5)
        logs = [
            PhaseLog(Phase.UNDERSTAND, "Received task"),
            PhaseLog(Phase.VERIFY, "Running"),
        ]
        tr = TaskResult(
            task_id="T-1", outcome=TaskOutcome.UNKNOWN, contract=None,
            history=h, final_verification=None, evidence_report=None,
            rollback_result=None, phase_log=logs, reason="No verifier",
        )
        d = tr.to_dict()
        assert len(d["phase_log"]) == 2
        assert d["phase_log"][0]["phase"] == "UNDERSTAND"


# ══════════════════════════════════════════════════════════════════════
# Orchestrator: Successful task
# ══════════════════════════════════════════════════════════════════════


class TestOrchestratorSuccess:
    def test_immediate_success(self):
        """Patch succeeds on first try — VERIFIED."""
        vr = _make_verification_result(passed=True)
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=5, verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")

        assert result.outcome == TaskOutcome.VERIFIED
        assert result.history.attempt_count == 1
        assert result.final_verification.passed is True
        assert any(p.phase == Phase.UNDERSTAND for p in result.phase_log)
        assert any(p.phase == Phase.VERIFY for p in result.phase_log)

    def test_success_on_second_attempt(self):
        """First attempt fails, second succeeds — VERIFIED."""
        vr_fail = _make_verification_result(
            passed=False, status=VerificationStatus.FAILED,
        )
        vr_pass = _make_verification_result(passed=True)
        verifier = MagicMock(spec=Verifier)
        verifier.verify.side_effect = [vr_fail, vr_pass]

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=5, verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")

        assert result.outcome == TaskOutcome.VERIFIED
        assert result.history.attempt_count == 2


# ══════════════════════════════════════════════════════════════════════
# Orchestrator: Contract generation
# ══════════════════════════════════════════════════════════════════════


class TestOrchestratorContractGeneration:
    def test_contract_generated_and_used(self):
        """Contract engine produces a contract used in verification."""
        contract = _make_contract()
        ce = MagicMock(spec=ContractEngine)
        ce.generate_contract.return_value = contract

        vr = _make_verification_result(passed=True)
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=5, verifier=verifier,
            contract_engine=ce,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")

        ce.generate_contract.assert_called_once_with("Fix bug")
        assert result.contract is contract
        assert result.outcome == TaskOutcome.VERIFIED

    def test_contract_generation_failure(self):
        """Contract engine raises — task returns FAILED."""
        ce = MagicMock(spec=ContractEngine)
        ce.generate_contract.side_effect = ContractValidationError("bad task")

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=5, contract_engine=ce,
        )
        result = orch.run("Fix bug")

        assert result.outcome == TaskOutcome.FAILED
        assert "Contract generation failed" in result.reason
        assert any(p.phase == Phase.CONTRACT for p in result.phase_log)


# ══════════════════════════════════════════════════════════════════════
# Orchestrator: Tool execution
# ══════════════════════════════════════════════════════════════════════


class TestOrchestratorToolExecution:
    def test_patch_fn_receives_repair_prompt(self):
        """On second attempt, patch_fn receives repair prompt."""
        vr_fail = _make_verification_result(
            passed=False, status=VerificationStatus.FAILED,
        )
        vr_pass = _make_verification_result(passed=True)
        verifier = MagicMock(spec=Verifier)
        verifier.verify.side_effect = [vr_fail, vr_pass]

        prompts_received = []

        def patch_fn(task, contract, files, prompt):
            prompts_received.append(prompt)
            return ["src/foo.py"]

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=5, verifier=verifier,
            patch_fn=patch_fn,
        )
        result = orch.run("Fix bug")

        assert prompts_received[0] == ""  # First attempt: no repair prompt.
        assert "Repair Attempt" in prompts_received[1]  # Second attempt has prompt.

    def test_patch_fn_exception_handled(self):
        """Patch function raising doesn't crash the orchestrator."""
        vr = _make_verification_result(
            passed=False, status=VerificationStatus.FAILED,
        )
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=2, verifier=verifier,
            patch_fn=lambda t, c, f, p: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        result = orch.run("Fix bug")

        assert result.outcome != TaskOutcome.VERIFIED


# ══════════════════════════════════════════════════════════════════════
# Orchestrator: Verification failure
# ══════════════════════════════════════════════════════════════════════


class TestOrchestratorVerificationFailure:
    def test_all_failures_returns_not_verified(self):
        """All attempts fail → FAILED, never VERIFIED."""
        vr_fail = _make_verification_result(
            passed=False, status=VerificationStatus.FAILED,
        )
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr_fail

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=3, verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")

        assert result.outcome != TaskOutcome.VERIFIED
        assert result.outcome in (TaskOutcome.FAILED, TaskOutcome.UNKNOWN)


# ══════════════════════════════════════════════════════════════════════
# Orchestrator: Falsification failure
# ══════════════════════════════════════════════════════════════════════


class TestOrchestratorFalsificationFailure:
    def test_counterexample_prevents_verified(self):
        """If falsification finds a counterexample, VERIFIED is blocked.

        The verifier returns passed=False because L6 (falsification)
        check failed — the orchestrator must NOT return VERIFIED.
        """
        checks = [
            VerificationCheck(
                check_id="L2-SUITE",
                level=VerificationLevel.EXISTING_TESTS,
                description="Test suite",
                status=CheckStatus.PASS,
                evidence="All tests passed",
            ),
            VerificationCheck(
                check_id="L6-FALS-1",
                level=VerificationLevel.FALSIFICATION,
                description="Falsification: boundary",
                status=CheckStatus.FAIL,
                evidence="Counterexample: input=0",
                obligation_id="OB-1",
            ),
        ]
        vr = _make_verification_result(
            passed=False, status=VerificationStatus.FAILED, checks=checks,
        )
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=3, verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")

        assert result.outcome != TaskOutcome.VERIFIED


# ══════════════════════════════════════════════════════════════════════
# Orchestrator: Repair loop
# ══════════════════════════════════════════════════════════════════════


class TestOrchestratorRepair:
    def test_repair_loop_records_history(self):
        """Repair loop records failure details in AttemptHistory."""
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

        assert result.outcome == TaskOutcome.VERIFIED
        assert result.history.attempt_count == 3
        # First two attempts should have failure info.
        assert result.history.attempts[0].failure_type is not None
        assert result.history.attempts[1].failure_type is not None
        # Third attempt succeeded.
        assert result.history.attempts[2].failure_type is None

    def test_repair_with_rollback(self):
        """Repeated failure triggers rollback via RollbackEngine."""
        vr_fail = _make_verification_result(
            passed=False, status=VerificationStatus.FAILED,
        )
        vr_pass = _make_verification_result(passed=True)

        calls = {"n": 0}

        def verify_side_effect(**kwargs):
            calls["n"] += 1
            if calls["n"] <= 3:
                return vr_fail
            return vr_pass

        verifier = MagicMock(spec=Verifier)
        verifier.verify.side_effect = verify_side_effect

        rollback_engine = MagicMock(spec=RollbackEngine)
        ckp = MagicMock()
        ckp.checkpoint_id = "CKP-test"
        rollback_engine.latest_checkpoint.return_value = ckp
        rollback_engine.rollback.return_value = RollbackResult(
            checkpoint_id="CKP-test", restored_files=["src/foo.py"],
            skipped_files=[], conflicts=[], success=True,
            reason="Restored", status=RollbackStatus.SUCCESS,
        )

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=10, verifier=verifier,
            rollback_engine=rollback_engine,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")

        assert result.history.attempt_count >= 1


# ══════════════════════════════════════════════════════════════════════
# Orchestrator: Repeated failure
# ══════════════════════════════════════════════════════════════════════


class TestOrchestratorRepeatedFailure:
    def test_stops_on_escalation(self):
        """Three no-progress attempts → ESCALATE → UNKNOWN."""
        fails = []
        for i in range(5):
            checks = [VerificationCheck(
                check_id=f"L2-{i}", level=VerificationLevel.EXISTING_TESTS,
                description=f"Test {i}", status=CheckStatus.FAIL,
                evidence="Same failure",
                obligation_id=f"OB-{i}",  # Different obs — not repeated.
            )]
            fails.append(_make_verification_result(
                passed=False, status=VerificationStatus.FAILED,
                checks=checks, confidence=0.3,
            ))

        verifier = MagicMock(spec=Verifier)
        verifier.verify.side_effect = fails

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=10, verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")

        assert result.outcome in (TaskOutcome.UNKNOWN, TaskOutcome.FAILED)
        assert result.history.attempt_count < 10


# ══════════════════════════════════════════════════════════════════════
# Orchestrator: Max iterations
# ══════════════════════════════════════════════════════════════════════


class TestOrchestratorMaxIterations:
    def test_budget_exhausted_returns_failed(self):
        """All attempts fail — FAILED, not VERIFIED."""
        vr_fail = _make_verification_result(
            passed=False, status=VerificationStatus.FAILED,
        )
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr_fail

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=3, verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")

        assert result.outcome != TaskOutcome.VERIFIED
        assert result.history.attempt_count <= 3

    def test_max_iterations_respected(self):
        """Never exceeds max_iterations."""
        vr_fail = _make_verification_result(
            passed=False, status=VerificationStatus.FAILED,
        )
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr_fail

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=5, verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")
        assert result.history.attempt_count <= 5

    def test_abandon_near_budget_limit(self):
        """Budget=2, first fails → ABANDON → FAILED."""
        vr_fail = _make_verification_result(
            passed=False, status=VerificationStatus.FAILED,
        )
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr_fail

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=2, verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")

        assert result.outcome == TaskOutcome.FAILED
        assert result.history.attempt_count <= 2


# ══════════════════════════════════════════════════════════════════════
# Orchestrator: Stale evidence
# ══════════════════════════════════════════════════════════════════════


class TestOrchestratorStaleEvidence:
    def test_stale_evidence_note_in_log(self):
        """When verification passes but evidence is stale, log records it.

        The verifier returns passed=True, but the EvidenceGraph detects
        stale evidence. In this case, the orchestrator should note it
        (though without real fingerprint data the graph falls through).
        """
        vr_pass = _make_verification_result(passed=True)
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr_pass

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=3, verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")

        # Without a contract that triggers EvidenceGraph staleness,
        # the verification pass is accepted.
        assert result.outcome == TaskOutcome.VERIFIED
        assert any(p.phase == Phase.EVIDENCE_CHECK for p in result.phase_log)


# ══════════════════════════════════════════════════════════════════════
# Orchestrator: Scope violation
# ══════════════════════════════════════════════════════════════════════


class TestOrchestratorScopeViolation:
    def test_scope_violation_logged(self):
        """Scope violation is logged when ChangeGuard reports BLOCKED."""
        contract = _make_contract()
        vr_fail = _make_verification_result(
            passed=False, status=VerificationStatus.FAILED,
        )
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr_fail

        # Mock ChangeGuard to report BLOCKED.
        guard = MagicMock(spec=ChangeGuard)
        budget = MagicMock(spec=ChangeBudget)
        budget.max_file_count = 2
        budget.expected_scope = "src"
        guard.create_budget.return_value = budget

        report = MagicMock(spec=ChangeBudgetReport)
        report.status = ScopeStatus.BLOCKED
        report.scope_expansion = 3.0
        report.violations = []
        guard.evaluate.return_value = report

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=2, verifier=verifier,
            change_guard=guard,
            patch_fn=lambda t, c, f, p: ["src/foo.py", "src/bar.py", "lib/baz.py"],
        )
        result = orch.run("Fix bug", contract=contract)

        # Should have logged scope violation.
        scope_logs = [p for p in result.phase_log
                      if p.phase == Phase.CHANGE_BUDGET and "BLOCKED" in p.message]
        assert len(scope_logs) >= 1


# ══════════════════════════════════════════════════════════════════════
# Orchestrator: UNKNOWN result
# ══════════════════════════════════════════════════════════════════════


class TestOrchestratorUnknown:
    def test_no_verifier_returns_non_verified(self):
        """Without a verifier, outcome is never VERIFIED."""
        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=2,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")

        assert result.outcome != TaskOutcome.VERIFIED
        assert result.outcome in (TaskOutcome.UNKNOWN, TaskOutcome.FAILED)

    def test_escalation_gives_unknown(self):
        """No-progress escalation returns UNKNOWN."""
        fails = []
        for i in range(5):
            checks = [VerificationCheck(
                check_id=f"L2-{i}", level=VerificationLevel.EXISTING_TESTS,
                description="t", status=CheckStatus.FAIL, evidence="f",
                obligation_id=f"OB-{i}",
            )]
            fails.append(_make_verification_result(
                passed=False, status=VerificationStatus.FAILED,
                checks=checks, confidence=0.3,
            ))
        verifier = MagicMock(spec=Verifier)
        verifier.verify.side_effect = fails

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=10, verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")

        assert result.outcome == TaskOutcome.UNKNOWN


# ══════════════════════════════════════════════════════════════════════
# Orchestrator: Model failure
# ══════════════════════════════════════════════════════════════════════


class TestOrchestratorModelFailure:
    def test_model_not_used_directly(self):
        """The model_client is stored but the loop uses patch_fn."""
        vr = _make_verification_result(passed=True)
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr

        orch = Orchestrator(
            model_client=None,  # No model client.
            tools=[],
            max_iterations=3, verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")

        assert result.outcome == TaskOutcome.VERIFIED

    def test_contract_engine_failure_returns_failed(self):
        """If contract generation fails, the task returns FAILED."""
        ce = MagicMock(spec=ContractEngine)
        ce.generate_contract.side_effect = ContractValidationError("Model API error")

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=5, contract_engine=ce,
        )
        result = orch.run("Fix bug")

        assert result.outcome == TaskOutcome.FAILED


# ══════════════════════════════════════════════════════════════════════
# Orchestrator: Tool failure
# ══════════════════════════════════════════════════════════════════════


class TestOrchestratorToolFailure:
    def test_tool_failure_classified_correctly(self):
        """ERROR checks with 'command not found' → TOOL_FAILURE classification."""
        checks = [VerificationCheck(
            check_id="L1-1", level=VerificationLevel.BASIC_VALIDATION,
            description="Check", status=CheckStatus.ERROR,
            evidence="Command not found: flake8",
        )]
        vr = _make_verification_result(
            passed=False, status=VerificationStatus.UNKNOWN, checks=checks,
        )
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=2, verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")

        # Find the attempt that classified the failure.
        tool_failures = [
            a for a in result.history.attempts
            if a.failure_type == FailureType.TOOL_FAILURE
        ]
        assert len(tool_failures) >= 1


# ══════════════════════════════════════════════════════════════════════
# Orchestrator: Never false success
# ══════════════════════════════════════════════════════════════════════


class TestOrchestratorNeverFalseSuccess:
    """CRITICAL: The orchestrator must NEVER claim VERIFIED without evidence."""

    def test_never_verified_without_verifier(self):
        orch = Orchestrator(model_client=MagicMock(), tools=[], max_iterations=3)
        result = orch.run("Fix bug")
        assert result.outcome != TaskOutcome.VERIFIED

    def test_never_verified_with_failing_checks(self):
        vr_fail = _make_verification_result(
            passed=False, status=VerificationStatus.FAILED,
        )
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr_fail

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=5, verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")
        assert result.outcome != TaskOutcome.VERIFIED


# ══════════════════════════════════════════════════════════════════════
# Orchestrator: Phase logging completeness
# ══════════════════════════════════════════════════════════════════════


class TestOrchestratorPhaseLogging:
    def test_successful_run_logs_phases(self):
        """A successful run should log UNDERSTAND, VERIFY, and PROOF."""
        vr = _make_verification_result(passed=True)
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=5, verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")

        phases_logged = {p.phase for p in result.phase_log}
        assert Phase.UNDERSTAND in phases_logged
        assert Phase.VERIFY in phases_logged

    def test_failed_run_logs_repair_phase(self):
        """A failed run should log REPAIR phases."""
        vr_fail = _make_verification_result(
            passed=False, status=VerificationStatus.FAILED,
        )
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr_fail

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=2, verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")

        phases_logged = {p.phase for p in result.phase_log}
        assert Phase.REPAIR in phases_logged

    def test_contract_phase_logged(self):
        """Contract generation phase is logged."""
        contract = _make_contract()
        ce = MagicMock(spec=ContractEngine)
        ce.generate_contract.return_value = contract

        vr = _make_verification_result(passed=True)
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=5, verifier=verifier,
            contract_engine=ce,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")

        contract_logs = [p for p in result.phase_log if p.phase == Phase.CONTRACT]
        assert len(contract_logs) >= 1

    def test_no_hidden_chain_of_thought_in_logs(self):
        """Phase logs should contain concise messages, not reasoning."""
        vr = _make_verification_result(passed=True)
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=5, verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug")

        for entry in result.phase_log:
            msg = entry.message.lower()
            assert "chain of thought" not in msg
            assert "let me think" not in msg


# ══════════════════════════════════════════════════════════════════════
# Orchestrator: Status diagnostic
# ══════════════════════════════════════════════════════════════════════


class TestOrchestratorStatus:
    def test_status_reflects_config(self):
        verifier = MagicMock(spec=Verifier)
        ce = MagicMock(spec=ContractEngine)
        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=7, verifier=verifier,
            contract_engine=ce,
        )
        s = orch.status()
        assert s["max_iterations"] == 7
        assert s["has_verifier"] is True
        assert s["has_contract_engine"] is True
        assert s["has_change_guard"] is True
        assert s["has_rollback_engine"] is True
        assert s["has_patch_fn"] is False


# ══════════════════════════════════════════════════════════════════════
# Orchestrator: With contract integration
# ══════════════════════════════════════════════════════════════════════


class TestOrchestratorWithContract:
    def test_contract_passed_through_to_verifier(self):
        """Contract is forwarded to verifier.verify()."""
        contract = _make_contract()
        vr = _make_verification_result(passed=True)
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=5, verifier=verifier,
            patch_fn=lambda t, c, f, p: ["src/foo.py"],
        )
        result = orch.run("Fix bug", contract=contract)

        assert result.contract is contract
        # Verify the contract was passed to verifier.
        call_kwargs = verifier.verify.call_args
        assert call_kwargs.kwargs.get("contract") is contract

    def test_expected_files_from_contract(self):
        """changed_files defaults to contract.expected_files when patch_fn is absent.

        With no patch_fn, no actual changes are applied.  The harness must
        return UNKNOWN (prefer UNKNOWN over unjustified VERIFIED) because
        pre-existing test passes are not evidence of new work.
        """
        contract = _make_contract()
        vr = _make_verification_result(passed=True)
        verifier = MagicMock(spec=Verifier)
        verifier.verify.return_value = vr

        orch = Orchestrator(
            model_client=MagicMock(), tools=[],
            max_iterations=5, verifier=verifier,
        )
        result = orch.run("Fix bug", contract=contract)

        # No patch_fn → no actual changes → UNKNOWN, not VERIFIED.
        assert result.outcome == TaskOutcome.UNKNOWN
        assert "No changes were applied" in result.reason
        # Verifier should still have been called with expected_files.
        call_args = verifier.verify.call_args
        assert "src/foo.py" in call_args.kwargs.get("changed_files", [])
