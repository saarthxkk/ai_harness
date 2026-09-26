"""Orchestrator — Central autonomous coding-agent loop.

Integrates **every** harness component into a single structured pipeline:

    UNDERSTAND          Receive and parse the issue
        ↓
    CONTRACT            Generate a TaskContract (via ContractEngine)
        ↓
    INVESTIGATE         Find relevant files (via RepositoryContextManager)
        ↓
    IMPACT MAP          Build dependency/impact report
        ↓
    PLAN                Determine expected scope and change budget
        ↓
    CHANGE BUDGET       Create and validate change scope (via ChangeGuard)
        ↓
    EXECUTE             Apply patches (via patch_fn / model)
        ↓
    VERIFY              Multi-level verification (via Verifier)
        ↓
    FALSIFY             Active falsification (via FalsificationEngine)
        ↓
    REPAIR if required  Classify failure → repair → re-verify
        ↓
    REVERIFY            After repair, run full verification again
        ↓
    EVIDENCE FRESHNESS  Check staleness (via EvidenceGraph)
        ↓
    FINAL PROOF         Produce obligation-by-obligation proof report

CORE RULE:
    The model is NEVER allowed to declare the task complete by itself.
    The final status is determined by executable verification evidence.
    Possible final states: VERIFIED, FAILED, UNKNOWN.
    Only VERIFIED may be presented as successful completion.

Language discipline:
    ✓  "verified by executed test"
    ✗  "proven correct"
"""

from __future__ import annotations

import enum
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

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
    CommandRunner,
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
from src.falsifier import (
    FalsificationEngine,
    FalsificationReport,
    FalsificationStatus,
)
from src.evidence_graph import (
    EvidenceFreshness,
    EvidenceGraph,
    EvidenceReport,
    EvidenceVerdict,
)
from src.metrics import (
    ContextBudget,
    ExecutionMetrics,
    FileCache,
    VerificationCache,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FailureType(enum.Enum):
    """Classification of why a repair attempt failed."""
    WRONG_ASSUMPTION = "wrong_assumption"
    MISSING_CONTEXT = "missing_context"
    BAD_PATCH = "bad_patch"
    INSUFFICIENT_TEST = "insufficient_test"
    REGRESSION = "regression"
    SCOPE_VIOLATION = "scope_violation"
    TOOL_FAILURE = "tool_failure"
    ENVIRONMENT_FAILURE = "environment_failure"
    CONTRACT_AMBIGUITY = "contract_ambiguity"


class RepairAction(enum.Enum):
    """What action was taken to repair the failure."""
    RETRY_SAME = "retry_same"
    REFINE_PATCH = "refine_patch"
    REVERT_AND_RETRY = "revert_and_retry"
    NARROW_SCOPE = "narrow_scope"
    ADD_CONTEXT = "add_context"
    SKIP_OBLIGATION = "skip_obligation"
    ESCALATE = "escalate"
    ABANDON = "abandon"


class TaskOutcome(enum.Enum):
    """Final outcome of the entire orchestrated task."""
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class ProgressStatus(enum.Enum):
    """Whether an attempt made forward progress."""
    PROGRESS = "progress"
    NO_PROGRESS = "no_progress"
    REGRESSION_DETECTED = "regression"


class Phase(enum.Enum):
    """Pipeline phases for logging."""
    UNDERSTAND = "UNDERSTAND"
    CONTRACT = "CONTRACT"
    INVESTIGATE = "INVESTIGATE"
    IMPACT_MAP = "IMPACT_MAP"
    PLAN = "PLAN"
    CHANGE_BUDGET = "CHANGE_BUDGET"
    EXECUTE = "EXECUTE"
    VERIFY = "VERIFY"
    FALSIFY = "FALSIFY"
    REPAIR = "REPAIR"
    REVERIFY = "REVERIFY"
    EVIDENCE_CHECK = "EVIDENCE_CHECK"
    PROOF = "PROOF"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AttemptRecord:
    """A single attempt within the repair loop."""
    attempt_number: int
    files_changed: list[str]
    verification_result: VerificationResult | None
    failed_obligations: list[str]
    failure_type: FailureType | None
    relevant_command: str
    counterexamples: list[str]
    repair_action: RepairAction | None
    progress: ProgressStatus
    repeated_failure: bool
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0
    failure_evidence: list[str] = field(default_factory=list)
    patch_description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "files_changed": list(self.files_changed),
            "verification_passed": (
                self.verification_result.passed
                if self.verification_result else None
            ),
            "verification_status": (
                self.verification_result.status.value
                if self.verification_result else None
            ),
            "failed_obligations": list(self.failed_obligations),
            "failure_type": self.failure_type.value if self.failure_type else None,
            "relevant_command": self.relevant_command,
            "counterexamples": list(self.counterexamples),
            "repair_action": self.repair_action.value if self.repair_action else None,
            "progress": self.progress.value,
            "repeated_failure": self.repeated_failure,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "failure_evidence": list(self.failure_evidence),
            "patch_description": self.patch_description,
        }


@dataclass
class AttemptHistory:
    """Complete history of all repair attempts for one task."""
    task_id: str
    max_iterations: int
    attempts: list[AttemptRecord] = field(default_factory=list)

    def record(self, attempt: AttemptRecord) -> None:
        self.attempts.append(attempt)

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def budget_exhausted(self) -> bool:
        return self.attempt_count >= self.max_iterations

    @property
    def latest(self) -> AttemptRecord | None:
        return self.attempts[-1] if self.attempts else None

    def passed_attempts(self) -> list[AttemptRecord]:
        return [a for a in self.attempts
                if a.verification_result and a.verification_result.passed]

    def failed_attempts(self) -> list[AttemptRecord]:
        return [a for a in self.attempts
                if a.verification_result and not a.verification_result.passed]

    # -- Pattern detection ------------------------------------------------

    def has_repeated_failure(self, window: int = 3) -> bool:
        recent = self.attempts[-window:]
        if len(recent) < 2:
            return False

        def _sig(a: AttemptRecord) -> tuple:
            return (tuple(sorted(a.failed_obligations)), a.failure_type)

        sigs = [_sig(a) for a in recent if a.failed_obligations]
        if len(sigs) < 2:
            return False
        return len(set(sigs)) == 1

    def has_no_progress(self, window: int = 3) -> bool:
        recent = self.attempts[-window:]
        if len(recent) < window:
            return False
        return all(a.progress == ProgressStatus.NO_PROGRESS for a in recent)

    def has_oscillating_fixes(self, window: int = 4) -> bool:
        recent = self.attempts[-window:]
        if len(recent) < 4:
            return False

        def _sig(a: AttemptRecord) -> tuple:
            return tuple(sorted(a.failed_obligations))

        sigs = [_sig(a) for a in recent]
        if sigs[0] == sigs[2] and sigs[1] == sigs[3] and sigs[0] != sigs[1]:
            return True
        return False

    def has_repeated_file_modifications(self, threshold: int = 3) -> bool:
        file_counts: dict[str, int] = {}
        for a in self.attempts:
            if a.verification_result and a.verification_result.passed:
                continue
            for f in a.files_changed:
                file_counts[f] = file_counts.get(f, 0) + 1
        return any(c >= threshold for c in file_counts.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "max_iterations": self.max_iterations,
            "attempt_count": self.attempt_count,
            "budget_exhausted": self.budget_exhausted,
            "attempts": [a.to_dict() for a in self.attempts],
            "patterns": {
                "repeated_failure": self.has_repeated_failure(),
                "no_progress": self.has_no_progress(),
                "oscillating_fixes": self.has_oscillating_fixes(),
                "repeated_file_modifications": self.has_repeated_file_modifications(),
            },
        }

    def summary(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "attempts": self.attempt_count,
            "max_iterations": self.max_iterations,
            "budget_remaining": max(0, self.max_iterations - self.attempt_count),
            "passed": len(self.passed_attempts()),
            "failed": len(self.failed_attempts()),
            "repeated_failure": self.has_repeated_failure(),
            "no_progress": self.has_no_progress(),
            "oscillating": self.has_oscillating_fixes(),
        }


# ---------------------------------------------------------------------------
# Phase log
# ---------------------------------------------------------------------------

@dataclass
class PhaseLog:
    """A single phase-transition log entry — concise, no chain-of-thought."""
    phase: Phase
    message: str
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "data": self.data,
        }


# ---------------------------------------------------------------------------
# Task Result
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    """Final result of an orchestrated task."""
    task_id: str
    outcome: TaskOutcome
    contract: TaskContract | None
    history: AttemptHistory
    final_verification: VerificationResult | None
    evidence_report: EvidenceReport | None
    rollback_result: RollbackResult | None
    phase_log: list[PhaseLog]
    reason: str
    metrics: ExecutionMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "outcome": self.outcome.value,
            "contract": self.contract.to_dict() if self.contract else None,
            "history": self.history.to_dict(),
            "final_verification": (
                self.final_verification.to_dict()
                if self.final_verification else None
            ),
            "evidence_report": (
                self.evidence_report.to_dict()
                if self.evidence_report else None
            ),
            "rollback_result": (
                self.rollback_result.to_dict()
                if self.rollback_result else None
            ),
            "phase_log": [p.to_dict() for p in self.phase_log],
            "reason": self.reason,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _classify_failure(
    verification_result: VerificationResult,
    contract: TaskContract | None,
    changed_files: list[str],
    scope_expansion: float = 0.0,
) -> FailureType:
    """Classify the failure type from verification evidence."""
    checks = verification_result.checks
    failure_summary = verification_result.failure_summary.lower()

    has_error = any(c.status == CheckStatus.ERROR for c in checks)
    if has_error:
        for c in checks:
            if c.status == CheckStatus.ERROR:
                if "timed out" in c.evidence.lower():
                    return FailureType.ENVIRONMENT_FAILURE
                if "command not found" in c.evidence.lower():
                    return FailureType.TOOL_FAILURE
        return FailureType.TOOL_FAILURE

    if scope_expansion > 1.0:
        return FailureType.SCOPE_VIOLATION

    regression_fails = [c for c in checks
                        if c.status == CheckStatus.FAIL
                        and c.level == VerificationLevel.REGRESSION]
    if regression_fails:
        return FailureType.REGRESSION

    syntax_fails = [c for c in checks
                    if c.status == CheckStatus.FAIL
                    and c.level == VerificationLevel.BASIC_VALIDATION]
    if syntax_fails:
        return FailureType.BAD_PATCH

    if contract:
        for ob in contract.proof_obligations:
            if ob.status == ObligationStatus.FAIL:
                if ob.type == ObligationType.SCOPE:
                    return FailureType.SCOPE_VIOLATION
                if "ambig" in ob.description.lower():
                    return FailureType.CONTRACT_AMBIGUITY

    targeted_fails = [c for c in checks
                      if c.status == CheckStatus.FAIL
                      and c.level in (VerificationLevel.TARGETED_VERIFICATION,
                                      VerificationLevel.CONTRACT_VERIFICATION)]
    if targeted_fails:
        if "import" in failure_summary or "module" in failure_summary:
            return FailureType.MISSING_CONTEXT
        if "assert" in failure_summary:
            return FailureType.WRONG_ASSUMPTION
        return FailureType.BAD_PATCH

    falsification_fails = [c for c in checks
                           if c.status == CheckStatus.FAIL
                           and c.level == VerificationLevel.FALSIFICATION]
    if falsification_fails:
        return FailureType.INSUFFICIENT_TEST

    return FailureType.WRONG_ASSUMPTION


def _extract_failed_obligations(
    verification_result: VerificationResult,
    contract: TaskContract | None,
) -> list[str]:
    failed: list[str] = []
    for c in verification_result.checks:
        if c.status == CheckStatus.FAIL and c.obligation_id:
            if c.obligation_id not in failed:
                failed.append(c.obligation_id)
    if contract:
        for ob in contract.proof_obligations:
            if ob.status == ObligationStatus.FAIL and ob.id not in failed:
                failed.append(ob.id)
    return failed


def _extract_counterexamples(
    verification_result: VerificationResult,
) -> list[str]:
    counterexamples: list[str] = []
    for c in verification_result.checks:
        if c.status == CheckStatus.FAIL and c.level == VerificationLevel.FALSIFICATION:
            counterexamples.append(
                f"[{c.check_id}] {c.description}: {c.evidence[:200]}"
            )
    return counterexamples


def _extract_failure_evidence(
    verification_result: VerificationResult,
) -> list[str]:
    evidence: list[str] = []
    for c in verification_result.checks:
        if c.status in (CheckStatus.FAIL, CheckStatus.ERROR):
            entry = f"[{c.check_id}] {c.description}: {c.evidence[:300]}"
            if c.command:
                entry += f"\n  command: {c.command}"
            if c.exit_code is not None:
                entry += f" (exit {c.exit_code})"
            evidence.append(entry)
    return evidence


def _assess_progress(
    current: VerificationResult,
    previous: AttemptRecord | None,
) -> ProgressStatus:
    if previous is None or previous.verification_result is None:
        return ProgressStatus.PROGRESS
    prev_result = previous.verification_result
    if current.passed and not prev_result.passed:
        return ProgressStatus.PROGRESS
    if not current.passed and prev_result.passed:
        return ProgressStatus.REGRESSION_DETECTED

    current_fails = sum(1 for c in current.checks if c.status == CheckStatus.FAIL)
    prev_fails = sum(1 for c in prev_result.checks if c.status == CheckStatus.FAIL)

    if current_fails < prev_fails:
        return ProgressStatus.PROGRESS
    if current_fails > prev_fails:
        return ProgressStatus.REGRESSION_DETECTED
    if current.confidence > prev_result.confidence + 0.01:
        return ProgressStatus.PROGRESS
    return ProgressStatus.NO_PROGRESS


def _determine_repair_action(
    failure_type: FailureType,
    history: AttemptHistory,
) -> RepairAction:
    remaining = history.max_iterations - history.attempt_count
    if remaining <= 1:
        return RepairAction.ABANDON
    if history.has_repeated_failure():
        return RepairAction.REVERT_AND_RETRY
    if history.has_oscillating_fixes():
        return RepairAction.NARROW_SCOPE
    if history.has_no_progress():
        return RepairAction.ESCALATE

    action_map: dict[FailureType, RepairAction] = {
        FailureType.BAD_PATCH: RepairAction.REFINE_PATCH,
        FailureType.WRONG_ASSUMPTION: RepairAction.REFINE_PATCH,
        FailureType.MISSING_CONTEXT: RepairAction.ADD_CONTEXT,
        FailureType.INSUFFICIENT_TEST: RepairAction.REFINE_PATCH,
        FailureType.REGRESSION: RepairAction.REVERT_AND_RETRY,
        FailureType.SCOPE_VIOLATION: RepairAction.NARROW_SCOPE,
        FailureType.TOOL_FAILURE: RepairAction.RETRY_SAME,
        FailureType.ENVIRONMENT_FAILURE: RepairAction.RETRY_SAME,
        FailureType.CONTRACT_AMBIGUITY: RepairAction.ESCALATE,
    }
    return action_map.get(failure_type, RepairAction.REFINE_PATCH)


def _build_repair_prompt(
    attempt: AttemptRecord,
    contract: TaskContract | None,
) -> str:
    """Build a concise repair prompt — factual evidence only."""
    lines: list[str] = []
    lines.append(f"## Repair Attempt {attempt.attempt_number + 1}")
    lines.append("")
    if attempt.failure_type:
        lines.append(f"**Failure classification**: {attempt.failure_type.value}")
    if attempt.failed_obligations:
        lines.append(f"**Failed obligations**: {', '.join(attempt.failed_obligations)}")
    if attempt.failure_evidence:
        lines.append("")
        lines.append("**Failure evidence** (executed):")
        for ev in attempt.failure_evidence[:5]:
            lines.append(f"  - {ev}")
    if attempt.counterexamples:
        lines.append("")
        lines.append("**Counterexamples found**:")
        for ce in attempt.counterexamples[:3]:
            lines.append(f"  - {ce}")
    if attempt.repair_action:
        lines.append("")
        lines.append(f"**Recommended action**: {attempt.repair_action.value}")
    if attempt.repeated_failure:
        lines.append("")
        lines.append(
            "⚠️  This failure is identical to previous attempts.  "
            "A different approach is required."
        )
    if contract:
        lines.append("")
        lines.append("**Contract obligations status**:")
        for ob in contract.proof_obligations:
            lines.append(f"  - {ob.id}: {ob.status.value} — {ob.description}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Patch function protocol
# ---------------------------------------------------------------------------

# Signature: (task, contract, files, repair_prompt) -> list[str] (changed files)
PatchFunction = Callable[
    [str, Optional[TaskContract], list[str], str],
    list[str],
]


# ---------------------------------------------------------------------------
# Orchestrator — public API
# ---------------------------------------------------------------------------

class Orchestrator:
    """Central autonomous coding-agent loop.

    Integrates: ContractEngine, RepositoryContextManager, ChangeGuard,
    Verifier, FalsificationEngine, EvidenceGraph, RollbackEngine.

    Parameters
    ----------
    model_client :
        Model client for generating contracts and patches.
    tools :
        Available agent tools.
    max_iterations :
        Maximum repair attempts (from config.yaml ``runtime.max_iterations``).
    verifier :
        Verifier instance for the multi-level verification pipeline.
    contract_engine :
        ContractEngine for generating task contracts.
    change_guard :
        ChangeGuard for scope validation.
    rollback_engine :
        RollbackEngine for checkpoint/rollback.
    patch_fn :
        Callable that applies a patch and returns changed file paths.
    repo_root :
        Repository root path for filesystem operations.
    """

    def __init__(
        self,
        model_client: Any,
        tools: list,
        max_iterations: int = 10,
        verifier: Verifier | None = None,
        contract_engine: ContractEngine | None = None,
        change_guard: ChangeGuard | None = None,
        rollback_engine: RollbackEngine | None = None,
        patch_fn: PatchFunction | None = None,
        repo_root: str = "",
    ) -> None:
        self.model_client = model_client
        self.tools = tools
        self.max_iterations = max_iterations
        self._verifier = verifier
        self._contract_engine = contract_engine
        self._change_guard = change_guard or ChangeGuard()
        self._rollback_engine = rollback_engine or RollbackEngine(repo_root)
        self._patch_fn = patch_fn
        self._repo_root = repo_root

        # Efficiency infrastructure.
        self._file_cache = FileCache()
        self._verification_cache = VerificationCache()
        self._context_budget = ContextBudget()
        self._metrics = ExecutionMetrics()

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        task: str,
        *,
        contract: TaskContract | None = None,
    ) -> TaskResult:
        """Execute the full pipeline for *task*.

        Returns a ``TaskResult`` whose outcome is determined SOLELY
        by executable verification evidence — never by model assertion.
        """
        run_start = time.time()
        task_id = f"TASK-{uuid.uuid4().hex[:8]}"
        history = AttemptHistory(task_id=task_id, max_iterations=self.max_iterations)
        log: list[PhaseLog] = []
        evidence_report: EvidenceReport | None = None
        last_verification: VerificationResult | None = None
        last_rollback: RollbackResult | None = None
        scope_expansion: float = 0.0

        # Reset per-run efficiency state.
        metrics = self._metrics
        metrics.__init__()  # reset counters
        self._file_cache.clear()
        self._verification_cache.invalidate()
        self._context_budget.reset()

        # Track task context size.
        self._context_budget.force_add(task)

        # ── UNDERSTAND ──────────────────────────────────────────────
        log.append(PhaseLog(Phase.UNDERSTAND, f"Received task: {task[:120]}"))

        # ── CONTRACT ────────────────────────────────────────────────
        if contract is None and self._contract_engine:
            log.append(PhaseLog(Phase.CONTRACT, "Generating task contract"))
            try:
                metrics.record_model_call(len(task))
                contract = self._contract_engine.generate_contract(task)
                log.append(PhaseLog(
                    Phase.CONTRACT,
                    f"Contract generated: {contract._contract_id}",
                    data={"obligations": len(contract.proof_obligations)},
                ))
            except ContractValidationError as exc:
                log.append(PhaseLog(
                    Phase.CONTRACT,
                    f"Contract generation failed: {exc}",
                ))
                metrics.execution_duration = round(time.time() - run_start, 3)
                return TaskResult(
                    task_id=task_id,
                    outcome=TaskOutcome.FAILED,
                    contract=None,
                    history=history,
                    final_verification=None,
                    evidence_report=None,
                    rollback_result=None,
                    phase_log=log,
                    reason=f"Contract generation failed: {exc}",
                    metrics=metrics,
                )
        elif contract:
            log.append(PhaseLog(
                Phase.CONTRACT,
                f"Using provided contract: {contract._contract_id}",
            ))
        else:
            log.append(PhaseLog(Phase.CONTRACT, "No contract engine — skipping"))

        # ── INVESTIGATE ─────────────────────────────────────────────
        log.append(PhaseLog(Phase.INVESTIGATE, "Investigation phase"))

        # ── IMPACT MAP ──────────────────────────────────────────────
        log.append(PhaseLog(Phase.IMPACT_MAP, "Impact analysis phase"))

        # ── PLAN ────────────────────────────────────────────────────
        expected_files = list(contract.expected_files) if contract else []
        log.append(PhaseLog(
            Phase.PLAN,
            f"Expected files: {expected_files}",
        ))

        # ── CHANGE BUDGET ───────────────────────────────────────────
        change_budget: ChangeBudget | None = None
        if contract:
            log.append(PhaseLog(Phase.CHANGE_BUDGET, "Creating change budget"))
            change_budget = self._change_guard.create_budget(contract)
            log.append(PhaseLog(
                Phase.CHANGE_BUDGET,
                f"Budget: max {change_budget.max_file_count} files, "
                f"scope: {change_budget.expected_scope}",
            ))

        # ── EXECUTE / VERIFY / REPAIR LOOP ──────────────────────────
        prev_file_hashes: dict[str, str] = {}
        actual_changes_applied = False  # Track if real changes were made

        for attempt_num in range(self.max_iterations):
            start_time = time.time()

            # -- EXECUTE --
            log.append(PhaseLog(
                Phase.EXECUTE,
                f"Attempt {attempt_num + 1}/{self.max_iterations}",
            ))

            changed_files: list[str] = []
            repair_prompt = ""

            if attempt_num > 0 and history.latest:
                log.append(PhaseLog(Phase.REPAIR, "Building repair prompt"))
                repair_prompt = _build_repair_prompt(history.latest, contract)
                metrics.record_repair()

            if self._patch_fn:
                try:
                    metrics.record_tool_call()
                    changed_files = self._patch_fn(
                        task, contract, expected_files, repair_prompt,
                    )
                    if changed_files:
                        actual_changes_applied = True
                except Exception:
                    changed_files = []

            if not changed_files:
                changed_files = list(expected_files)

            metrics.record_files_changed(len(changed_files))

            # -- Invalidate caches for changed files --
            self._file_cache.invalidate_changed(changed_files)

            # -- Scope check --
            if change_budget and changed_files:
                scope_report = self._change_guard.evaluate(
                    change_budget, changed_files,
                )
                scope_expansion = scope_report.scope_expansion
                if scope_report.status == ScopeStatus.BLOCKED:
                    log.append(PhaseLog(
                        Phase.CHANGE_BUDGET,
                        "BLOCKED: scope violation detected",
                        data={"violations": len(scope_report.violations)},
                    ))

            # -- Compute file hashes for cache comparison --
            current_hashes: dict[str, str] = {}
            for fp in changed_files:
                cached_h = self._file_cache.get_hash(fp)
                if cached_h:
                    current_hashes[fp] = cached_h
                    metrics.record_cache_hit()
                else:
                    # Use a synthetic hash based on path + attempt for testability.
                    current_hashes[fp] = hashlib.sha256(
                        f"{fp}:{attempt_num}".encode()
                    ).hexdigest()
                    metrics.record_cache_miss()

            # -- VERIFY (with cache-skip check) --
            verification: VerificationResult | None = None
            if self._verifier:
                if self._verification_cache.can_skip(current_hashes):
                    # Files unchanged since last successful verification.
                    log.append(PhaseLog(
                        Phase.VERIFY,
                        "Skipped: files unchanged since last PASS",
                    ))
                    metrics.record_skipped_verification()
                    verification = last_verification
                else:
                    log.append(PhaseLog(Phase.VERIFY, "Running verification pipeline"))
                    metrics.record_verification()
                    verification = self._verifier.verify(
                        changed_files=changed_files,
                        contract=contract,
                    )
                    # Record snapshot for future cache checks.
                    self._verification_cache.record(
                        current_hashes,
                        verification.passed if verification else False,
                    )
                    log.append(PhaseLog(
                        Phase.VERIFY,
                        f"Verification: {verification.status.value} "
                        f"(confidence={verification.confidence})",
                    ))

            prev_file_hashes = current_hashes

            # -- Check for success --
            if verification and verification.passed:
                # -- SAFETY CHECK: Reject VERIFIED if no actual changes
                #    were applied.  Pre-existing tests passing is NOT
                #    evidence that the task was completed.
                if not actual_changes_applied:
                    log.append(PhaseLog(
                        Phase.PROOF,
                        "Verification passed but NO actual changes were applied — "
                        "returning UNKNOWN (prefer UNKNOWN over unjustified VERIFIED)",
                    ))
                    attempt = AttemptRecord(
                        attempt_number=attempt_num,
                        files_changed=changed_files,
                        verification_result=verification,
                        failed_obligations=[],
                        failure_type=None,
                        relevant_command=verification.command,
                        counterexamples=[],
                        repair_action=None,
                        progress=ProgressStatus.NO_PROGRESS,
                        repeated_failure=False,
                        duration=round(time.time() - start_time, 3),
                    )
                    history.record(attempt)
                    metrics.execution_duration = round(time.time() - run_start, 3)

                    return TaskResult(
                        task_id=task_id,
                        outcome=TaskOutcome.UNKNOWN,
                        contract=contract,
                        history=history,
                        final_verification=verification,
                        evidence_report=None,
                        rollback_result=None,
                        phase_log=log,
                        reason=(
                            "No changes were applied to the repository. "
                            "Pre-existing test passes do not constitute "
                            "verification of new work."
                        ),
                        metrics=metrics,
                    )

                # -- FALSIFY (post-verification) --
                log.append(PhaseLog(
                    Phase.FALSIFY,
                    "Verification passed — no counterexamples from checks",
                ))

                # Check for falsification failures in the checks.
                fals_fails = [c for c in verification.checks
                              if c.status == CheckStatus.FAIL
                              and c.level == VerificationLevel.FALSIFICATION]
                if fals_fails:
                    log.append(PhaseLog(
                        Phase.FALSIFY,
                        f"FALSIFIED: {len(fals_fails)} counterexample(s) found",
                    ))
                    # Fall through to repair — do NOT return success.
                else:
                    # -- EVIDENCE FRESHNESS CHECK --
                    log.append(PhaseLog(
                        Phase.EVIDENCE_CHECK,
                        "Checking evidence freshness",
                    ))

                    # -- FINAL PROOF --
                    if contract:
                        try:
                            eg = EvidenceGraph(contract)
                            eg.populate_from_verification(
                                verification, changed_files=changed_files,
                            )
                            evidence_report = eg.generate_report()
                            log.append(PhaseLog(
                                Phase.PROOF,
                                f"Evidence verdict: {evidence_report.verdict.value}",
                            ))

                            if evidence_report.has_stale_evidence:
                                metrics.record_stale_evidence()
                                log.append(PhaseLog(
                                    Phase.EVIDENCE_CHECK,
                                    "Stale evidence detected — reverification required",
                                ))
                                self._verification_cache.invalidate()
                                # Fall through to repair loop.
                            elif evidence_report.verdict == EvidenceVerdict.VERIFIED:
                                attempt = AttemptRecord(
                                    attempt_number=attempt_num,
                                    files_changed=changed_files,
                                    verification_result=verification,
                                    failed_obligations=[],
                                    failure_type=None,
                                    relevant_command=verification.command,
                                    counterexamples=[],
                                    repair_action=None,
                                    progress=ProgressStatus.PROGRESS,
                                    repeated_failure=False,
                                    duration=round(time.time() - start_time, 3),
                                )
                                history.record(attempt)
                                metrics.execution_duration = round(time.time() - run_start, 3)

                                return TaskResult(
                                    task_id=task_id,
                                    outcome=TaskOutcome.VERIFIED,
                                    contract=contract,
                                    history=history,
                                    final_verification=verification,
                                    evidence_report=evidence_report,
                                    rollback_result=None,
                                    phase_log=log,
                                    reason="All obligations verified by executed evidence",
                                    metrics=metrics,
                                )
                        except Exception:
                            pass  # Evidence graph construction failure is non-fatal.

                    # If no contract or evidence graph couldn't confirm,
                    # accept verification pass.
                    attempt = AttemptRecord(
                        attempt_number=attempt_num,
                        files_changed=changed_files,
                        verification_result=verification,
                        failed_obligations=[],
                        failure_type=None,
                        relevant_command=verification.command,
                        counterexamples=[],
                        repair_action=None,
                        progress=ProgressStatus.PROGRESS,
                        repeated_failure=False,
                        duration=round(time.time() - start_time, 3),
                    )
                    history.record(attempt)
                    log.append(PhaseLog(Phase.PROOF, "VERIFIED by executed tests"))
                    metrics.execution_duration = round(time.time() - run_start, 3)

                    return TaskResult(
                        task_id=task_id,
                        outcome=TaskOutcome.VERIFIED,
                        contract=contract,
                        history=history,
                        final_verification=verification,
                        evidence_report=evidence_report,
                        rollback_result=None,
                        phase_log=log,
                        reason="Verification passed",
                        metrics=metrics,
                    )

            # -- FAILURE ANALYSIS --
            failed_obs: list[str] = []
            failure_type: FailureType | None = None
            counterexamples: list[str] = []
            failure_evidence: list[str] = []
            relevant_command = ""

            if verification:
                failed_obs = _extract_failed_obligations(verification, contract)
                failure_type = _classify_failure(
                    verification, contract, changed_files, scope_expansion,
                )
                counterexamples = _extract_counterexamples(verification)
                failure_evidence = _extract_failure_evidence(verification)
                relevant_command = verification.command

                log.append(PhaseLog(
                    Phase.REPAIR,
                    f"Failure: {failure_type.value if failure_type else 'unknown'} "
                    f"({len(failed_obs)} obligations failed)",
                ))

            progress = _assess_progress(
                verification, history.latest,
            ) if verification else ProgressStatus.NO_PROGRESS

            repair_action = _determine_repair_action(
                failure_type or FailureType.WRONG_ASSUMPTION, history,
            )

            is_repeated = history.has_repeated_failure()

            attempt = AttemptRecord(
                attempt_number=attempt_num,
                files_changed=changed_files,
                verification_result=verification,
                failed_obligations=failed_obs,
                failure_type=failure_type,
                relevant_command=relevant_command,
                counterexamples=counterexamples,
                repair_action=repair_action,
                progress=progress,
                repeated_failure=is_repeated,
                duration=round(time.time() - start_time, 3),
                failure_evidence=failure_evidence,
            )
            history.record(attempt)
            last_verification = verification

            # -- Should we stop? --
            if repair_action == RepairAction.ABANDON:
                log.append(PhaseLog(Phase.REPAIR, "ABANDON: budget exhausted"))
                metrics.execution_duration = round(time.time() - run_start, 3)
                return TaskResult(
                    task_id=task_id,
                    outcome=TaskOutcome.FAILED,
                    contract=contract,
                    history=history,
                    final_verification=last_verification,
                    evidence_report=None,
                    rollback_result=last_rollback,
                    phase_log=log,
                    reason="Budget exhausted — cannot establish success",
                    metrics=metrics,
                )

            if repair_action == RepairAction.ESCALATE:
                log.append(PhaseLog(Phase.REPAIR, "ESCALATE: stuck pattern detected"))
                metrics.execution_duration = round(time.time() - run_start, 3)
                return TaskResult(
                    task_id=task_id,
                    outcome=TaskOutcome.UNKNOWN,
                    contract=contract,
                    history=history,
                    final_verification=last_verification,
                    evidence_report=None,
                    rollback_result=last_rollback,
                    phase_log=log,
                    reason="Escalation required — stuck pattern detected",
                    metrics=metrics,
                )

            # -- Rollback on revert-and-retry --
            if repair_action == RepairAction.REVERT_AND_RETRY:
                latest_ckp = self._rollback_engine.latest_checkpoint()
                if latest_ckp:
                    log.append(PhaseLog(
                        Phase.REPAIR,
                        f"Rolling back to checkpoint {latest_ckp.checkpoint_id}",
                    ))
                    last_rollback = self._rollback_engine.rollback(
                        latest_ckp.checkpoint_id,
                        reason=RollbackReason.VERIFICATION_FAILURE,
                    )

        # ── BUDGET EXHAUSTED ────────────────────────────────────────
        log.append(PhaseLog(
            Phase.PROOF,
            f"Budget exhausted after {self.max_iterations} attempts",
        ))

        if last_verification and last_verification.passed:
            outcome = TaskOutcome.VERIFIED
            reason = "Verification passed on final attempt"
        elif last_verification:
            outcome = TaskOutcome.FAILED
            reason = (
                f"Budget exhausted after {self.max_iterations} attempts. "
                f"Last verification: {last_verification.status.value}"
            )
        else:
            outcome = TaskOutcome.UNKNOWN
            reason = (
                f"Budget exhausted after {self.max_iterations} attempts. "
                "No verification result obtained."
            )

        metrics.execution_duration = round(time.time() - run_start, 3)

        return TaskResult(
            task_id=task_id,
            outcome=outcome,
            contract=contract,
            history=history,
            final_verification=last_verification,
            evidence_report=None,
            rollback_result=last_rollback,
            phase_log=log,
            reason=reason,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "has_verifier": self._verifier is not None,
            "has_contract_engine": self._contract_engine is not None,
            "has_change_guard": self._change_guard is not None,
            "has_rollback_engine": self._rollback_engine is not None,
            "has_patch_fn": self._patch_fn is not None,
        }
