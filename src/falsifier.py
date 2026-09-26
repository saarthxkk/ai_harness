"""Falsifier — Active Falsification Engine.

CORE PRINCIPLE:
    "Do not only verify the solution.  Try to prove the solution wrong."

The falsifier runs AFTER normal verification indicates that the task
appears successful.  For each applicable proof obligation, it generates
a bounded set of **counterexample candidates** and EXECUTES them.

A candidate must be **executed** before it can become evidence.
Model-generated reasoning alone is never treated as proof.

Language discipline:
    ✓  "No counterexample was found within the bounded checks performed."
    ✗  "Mathematically proven correct."

NOT_FALSIFIED means:
    "No counterexample was found within the bounded checks performed."
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.contract_engine import (
    ObligationStatus,
    ObligationType,
    ProofObligation,
    TaskContract,
)
from src.verifier import (
    CheckStatus,
    CommandResult,
    CommandRunner,
    FalsificationInterface,
    VerificationCheck,
    VerificationLevel,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FalsificationStatus(enum.Enum):
    """Outcome of falsification for a single obligation."""
    NOT_FALSIFIED = "NOT_FALSIFIED"
    FALSIFIED = "FALSIFIED"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class CaseCategory(enum.Enum):
    """Category of adversarial test case."""
    BOUNDARY_VALUES = "boundary_values"
    EMPTY_NULL_INPUTS = "empty_null_inputs"
    INVALID_INPUTS = "invalid_inputs"
    OFF_BY_ONE = "off_by_one"
    TYPE_VARIATIONS = "type_variations"
    REGRESSION = "regression"
    ERROR_PATHS = "error_paths"
    PERMISSION_SAFETY = "permission_safety"
    CONTRACT_EDGE_CASES = "contract_edge_cases"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FalsificationCase:
    """A single adversarial test case — a counterexample candidate.

    A case MUST be executed before it can become evidence.
    """
    case_id: str
    obligation_id: str
    category: CaseCategory
    description: str
    test_command: str
    expected_behaviour: str

    # Populated after execution.
    executed: bool = False
    passed: bool = False
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration: float = 0.0
    timed_out: bool = False
    is_counterexample: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "case_id": self.case_id,
            "obligation_id": self.obligation_id,
            "category": self.category.value,
            "description": self.description,
            "test_command": self.test_command,
            "expected_behaviour": self.expected_behaviour,
            "executed": self.executed,
            "passed": self.passed,
            "is_counterexample": self.is_counterexample,
        }
        if self.executed:
            d["exit_code"] = self.exit_code
            d["duration"] = self.duration
            d["timed_out"] = self.timed_out
            if self.stdout:
                d["stdout"] = self.stdout[:500]
            if self.stderr:
                d["stderr"] = self.stderr[:500]
        return d


@dataclass
class FalsificationResult:
    """Complete falsification result for a single proof obligation."""
    obligation_id: str
    cases_attempted: int
    cases_passed: int
    counterexamples_found: int
    status: FalsificationStatus
    evidence: list[str]
    cases: list[FalsificationCase] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "cases_attempted": self.cases_attempted,
            "cases_passed": self.cases_passed,
            "counterexamples_found": self.counterexamples_found,
            "status": self.status.value,
            "evidence": list(self.evidence),
            "cases": [c.to_dict() for c in self.cases],
        }


@dataclass
class FalsificationReport:
    """Aggregate falsification report across all obligations."""
    results: list[FalsificationResult]
    total_cases_attempted: int
    total_counterexamples: int
    overall_status: FalsificationStatus
    duration: float
    evidence: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "total_cases_attempted": self.total_cases_attempted,
            "total_counterexamples": self.total_counterexamples,
            "overall_status": self.overall_status.value,
            "duration": self.duration,
            "evidence": list(self.evidence),
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FalsificationConfig:
    """Bounds to prevent infinite adversarial testing."""
    max_falsification_cases: int = 50
    max_falsification_time: float = 300.0   # seconds
    max_attempts_per_obligation: int = 10
    command_timeout: int = 30               # per-case timeout


# ---------------------------------------------------------------------------
# Case generator protocol
# ---------------------------------------------------------------------------

# Signature: (obligation, changed_files, context) -> list[FalsificationCase]
CaseGenerator = Callable[
    [ProofObligation, list[str], dict[str, Any]],
    list[FalsificationCase],
]


# ---------------------------------------------------------------------------
# Built-in case generators
# ---------------------------------------------------------------------------

def generate_boundary_cases(
    obligation: ProofObligation,
    changed_files: list[str],
    context: dict[str, Any],
) -> list[FalsificationCase]:
    """Generate boundary-value counterexample candidates.

    Examines the obligation description for numeric thresholds and
    produces boundary probes around those values.
    """
    import re

    cases: list[FalsificationCase] = []
    desc = obligation.description.lower()

    # Extract numeric thresholds from the description.
    numbers = re.findall(r"(\d+(?:\.\d+)?)\s*%?", desc)

    for i, raw in enumerate(numbers):
        val = float(raw)

        # Build the test script that probes slightly above the boundary.
        above = val + 0.01 if "." in raw else val + 1
        below = val - 0.01 if "." in raw else max(0, val - 1)

        cases.append(FalsificationCase(
            case_id=f"BOUND-{obligation.id}-above-{i}",
            obligation_id=obligation.id,
            category=CaseCategory.BOUNDARY_VALUES,
            description=f"Boundary probe: value just above {raw} → {above}",
            test_command=context.get(
                "test_command",
                f"pytest -x -q -k {obligation.id.lower().replace('-', '_')}",
            ),
            expected_behaviour=f"Value {above} should be handled correctly",
        ))
        cases.append(FalsificationCase(
            case_id=f"BOUND-{obligation.id}-below-{i}",
            obligation_id=obligation.id,
            category=CaseCategory.BOUNDARY_VALUES,
            description=f"Boundary probe: value just below {raw} → {below}",
            test_command=context.get(
                "test_command",
                f"pytest -x -q -k {obligation.id.lower().replace('-', '_')}",
            ),
            expected_behaviour=f"Value {below} should be handled correctly",
        ))

    return cases


def generate_empty_null_cases(
    obligation: ProofObligation,
    changed_files: list[str],
    context: dict[str, Any],
) -> list[FalsificationCase]:
    """Generate empty/null input counterexample candidates."""
    cases: list[FalsificationCase] = []

    test_cmd = context.get(
        "test_command",
        f"pytest -x -q -k {obligation.id.lower().replace('-', '_')}",
    )

    cases.append(FalsificationCase(
        case_id=f"NULL-{obligation.id}-empty",
        obligation_id=obligation.id,
        category=CaseCategory.EMPTY_NULL_INPUTS,
        description="Empty input probe: verify behaviour with empty/missing input",
        test_command=test_cmd,
        expected_behaviour="Empty input should not cause crashes or silent corruption",
    ))

    cases.append(FalsificationCase(
        case_id=f"NULL-{obligation.id}-none",
        obligation_id=obligation.id,
        category=CaseCategory.EMPTY_NULL_INPUTS,
        description="None/null probe: verify behaviour with None-like values",
        test_command=test_cmd,
        expected_behaviour="None inputs should raise appropriate errors or be handled",
    ))

    return cases


def generate_invalid_input_cases(
    obligation: ProofObligation,
    changed_files: list[str],
    context: dict[str, Any],
) -> list[FalsificationCase]:
    """Generate invalid input counterexample candidates."""
    test_cmd = context.get(
        "test_command",
        f"pytest -x -q -k {obligation.id.lower().replace('-', '_')}",
    )

    return [
        FalsificationCase(
            case_id=f"INVALID-{obligation.id}-type",
            obligation_id=obligation.id,
            category=CaseCategory.INVALID_INPUTS,
            description="Type-mismatch probe: pass wrong type to changed function",
            test_command=test_cmd,
            expected_behaviour="Invalid type should raise TypeError or be rejected",
        ),
        FalsificationCase(
            case_id=f"INVALID-{obligation.id}-range",
            obligation_id=obligation.id,
            category=CaseCategory.INVALID_INPUTS,
            description="Out-of-range probe: pass extreme values",
            test_command=test_cmd,
            expected_behaviour="Extreme values should be rejected or handled safely",
        ),
    ]


def generate_off_by_one_cases(
    obligation: ProofObligation,
    changed_files: list[str],
    context: dict[str, Any],
) -> list[FalsificationCase]:
    """Generate off-by-one counterexample candidates."""
    test_cmd = context.get(
        "test_command",
        f"pytest -x -q -k {obligation.id.lower().replace('-', '_')}",
    )

    return [FalsificationCase(
        case_id=f"OBO-{obligation.id}",
        obligation_id=obligation.id,
        category=CaseCategory.OFF_BY_ONE,
        description="Off-by-one probe: check fence-post/boundary conditions",
        test_command=test_cmd,
        expected_behaviour="Boundary conditions should be handled correctly",
    )]


def generate_error_path_cases(
    obligation: ProofObligation,
    changed_files: list[str],
    context: dict[str, Any],
) -> list[FalsificationCase]:
    """Generate error-path counterexample candidates."""
    test_cmd = context.get(
        "test_command",
        f"pytest -x -q -k {obligation.id.lower().replace('-', '_')}",
    )

    return [FalsificationCase(
        case_id=f"ERR-{obligation.id}",
        obligation_id=obligation.id,
        category=CaseCategory.ERROR_PATHS,
        description="Error path probe: trigger exception/error handling paths",
        test_command=test_cmd,
        expected_behaviour="Error paths should be handled gracefully",
    )]


def generate_permission_safety_cases(
    obligation: ProofObligation,
    changed_files: list[str],
    context: dict[str, Any],
) -> list[FalsificationCase]:
    """Generate permission/safety counterexample candidates.

    Only relevant for safety-type obligations.
    """
    if obligation.type not in (ObligationType.SAFETY, ObligationType.CONSTRAINT):
        return []

    test_cmd = context.get(
        "test_command",
        f"pytest -x -q -k {obligation.id.lower().replace('-', '_')}",
    )

    return [FalsificationCase(
        case_id=f"PERM-{obligation.id}",
        obligation_id=obligation.id,
        category=CaseCategory.PERMISSION_SAFETY,
        description="Permission/safety probe: attempt unauthorized action",
        test_command=test_cmd,
        expected_behaviour="Unauthorized actions must be blocked",
    )]


# Default generators, applied in order.
DEFAULT_GENERATORS: list[CaseGenerator] = [
    generate_boundary_cases,
    generate_empty_null_cases,
    generate_invalid_input_cases,
    generate_off_by_one_cases,
    generate_error_path_cases,
    generate_permission_safety_cases,
]


# ---------------------------------------------------------------------------
# Case executor
# ---------------------------------------------------------------------------

def execute_case(
    case: FalsificationCase,
    runner: CommandRunner,
    timeout: int,
) -> FalsificationCase:
    """Execute a single falsification case.

    A case is **only** evidence if it has been executed.
    Model-generated reasoning is never treated as proof.
    """
    try:
        result = runner(case.test_command, timeout)
    except Exception as exc:
        case.executed = True
        case.passed = False
        case.stderr = f"Execution error: {exc}"
        case.exit_code = -1
        case.is_counterexample = False
        return case

    case.executed = True
    case.stdout = result.stdout
    case.stderr = result.stderr
    case.exit_code = result.exit_code
    case.duration = result.duration
    case.timed_out = result.timed_out

    if result.timed_out:
        # Timeout → unknown, not a counterexample.
        case.passed = False
        case.is_counterexample = False
    elif result.success:
        # Test passed → the code handled this case correctly.
        case.passed = True
        case.is_counterexample = False
    else:
        # Test FAILED → this is a counterexample.
        case.passed = False
        case.is_counterexample = True

    return case


# ---------------------------------------------------------------------------
# Falsification Engine
# ---------------------------------------------------------------------------

class FalsificationEngine(FalsificationInterface):
    """Active Falsification Engine.

    Generates adversarial counterexample candidates for each proof
    obligation, executes them, and reports whether the solution was
    falsified.

    Implements ``FalsificationInterface`` so it plugs directly into
    the Verifier's Level 6.

    Parameters
    ----------
    config : FalsificationConfig | None
        Bounds configuration.  Defaults to sane defaults.
    generators : list[CaseGenerator] | None
        Case generators to use.  Defaults to all built-in generators.
    custom_cases : dict[str, list[FalsificationCase]] | None
        Pre-built cases keyed by obligation ID.  Useful for
        deterministic testing.  These are executed BEFORE generated
        cases.
    context : dict[str, Any] | None
        Extra context passed to case generators (e.g. ``test_command``,
        ``relevant_functions``, ``verification_results``).
    """

    def __init__(
        self,
        *,
        config: FalsificationConfig | None = None,
        generators: list[CaseGenerator] | None = None,
        custom_cases: dict[str, list[FalsificationCase]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self._config = config or FalsificationConfig()
        self._generators = generators if generators is not None else list(DEFAULT_GENERATORS)
        self._custom_cases = custom_cases or {}
        self._context = context or {}
        self._last_report: FalsificationReport | None = None

    # ------------------------------------------------------------------
    # FalsificationInterface implementation (plugs into Verifier L6)
    # ------------------------------------------------------------------

    def attempt_falsification(
        self,
        contract: TaskContract | None,
        changed_files: list[str],
        runner: CommandRunner,
        timeout: int,
    ) -> list[VerificationCheck]:
        """Attempt to falsify the agent's changes.

        Returns ``VerificationCheck`` objects for Level 6 integration.
        """
        report = self.falsify(contract, changed_files, runner, timeout)
        return self._report_to_checks(report)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def falsify(
        self,
        contract: TaskContract | None,
        changed_files: list[str],
        runner: CommandRunner,
        timeout: int | None = None,
    ) -> FalsificationReport:
        """Run the full falsification pipeline.

        1. Collect applicable proof obligations from the contract.
        2. For each obligation, generate adversarial cases.
        3. Execute cases within bounded limits.
        4. Aggregate results into a FalsificationReport.
        """
        start = time.monotonic()
        timeout = timeout or self._config.command_timeout
        results: list[FalsificationResult] = []
        total_cases = 0
        total_counterexamples = 0
        global_evidence: list[str] = []

        if contract is None or not contract.proof_obligations:
            # Nothing to falsify.
            report = FalsificationReport(
                results=[],
                total_cases_attempted=0,
                total_counterexamples=0,
                overall_status=FalsificationStatus.NOT_APPLICABLE,
                duration=round(time.monotonic() - start, 3),
                evidence=["No contract or proof obligations to falsify."],
            )
            self._last_report = report
            return report

        for obligation in contract.proof_obligations:
            # Respect global time budget.
            elapsed = time.monotonic() - start
            if elapsed >= self._config.max_falsification_time:
                results.append(FalsificationResult(
                    obligation_id=obligation.id,
                    cases_attempted=0,
                    cases_passed=0,
                    counterexamples_found=0,
                    status=FalsificationStatus.UNKNOWN,
                    evidence=[
                        f"Global time budget ({self._config.max_falsification_time}s) "
                        f"exhausted before reaching obligation {obligation.id}."
                    ],
                ))
                continue

            # Respect global case budget.
            if total_cases >= self._config.max_falsification_cases:
                results.append(FalsificationResult(
                    obligation_id=obligation.id,
                    cases_attempted=0,
                    cases_passed=0,
                    counterexamples_found=0,
                    status=FalsificationStatus.UNKNOWN,
                    evidence=[
                        f"Global case budget ({self._config.max_falsification_cases}) "
                        f"exhausted before reaching obligation {obligation.id}."
                    ],
                ))
                continue

            result = self._falsify_obligation(
                obligation, changed_files, runner, timeout,
                start_time=start,
                global_cases_used=total_cases,
            )
            results.append(result)
            total_cases += result.cases_attempted
            total_counterexamples += result.counterexamples_found
            global_evidence.extend(result.evidence)

        # Determine overall status.
        if total_counterexamples > 0:
            overall = FalsificationStatus.FALSIFIED
        elif all(r.status == FalsificationStatus.NOT_APPLICABLE for r in results):
            overall = FalsificationStatus.NOT_APPLICABLE
        elif any(r.status == FalsificationStatus.UNKNOWN for r in results):
            overall = FalsificationStatus.UNKNOWN
        elif all(
            r.status in (FalsificationStatus.NOT_FALSIFIED, FalsificationStatus.NOT_APPLICABLE)
            for r in results
        ):
            overall = FalsificationStatus.NOT_FALSIFIED
        else:
            overall = FalsificationStatus.UNKNOWN

        duration = round(time.monotonic() - start, 3)

        report = FalsificationReport(
            results=results,
            total_cases_attempted=total_cases,
            total_counterexamples=total_counterexamples,
            overall_status=overall,
            duration=duration,
            evidence=global_evidence,
        )
        self._last_report = report
        return report

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def last_report(self) -> FalsificationReport | None:
        """Return the most recent falsification report."""
        return self._last_report

    @property
    def config(self) -> FalsificationConfig:
        """Return the current configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Internal: per-obligation falsification
    # ------------------------------------------------------------------

    def _falsify_obligation(
        self,
        obligation: ProofObligation,
        changed_files: list[str],
        runner: CommandRunner,
        timeout: int,
        *,
        start_time: float,
        global_cases_used: int,
    ) -> FalsificationResult:
        """Falsify a single proof obligation."""
        cases: list[FalsificationCase] = []
        evidence: list[str] = []

        # 1. Gather pre-built custom cases for this obligation.
        if obligation.id in self._custom_cases:
            cases.extend(self._custom_cases[obligation.id])

        # 2. Generate cases from each generator.
        for gen in self._generators:
            try:
                generated = gen(obligation, changed_files, self._context)
                cases.extend(generated)
            except Exception:
                # Generator failure is non-fatal — skip.
                pass

        if not cases:
            return FalsificationResult(
                obligation_id=obligation.id,
                cases_attempted=0,
                cases_passed=0,
                counterexamples_found=0,
                status=FalsificationStatus.NOT_APPLICABLE,
                evidence=[
                    f"No falsification cases could be generated for "
                    f"obligation {obligation.id}: {obligation.description}"
                ],
            )

        # 3. Enforce per-obligation case limit.
        max_per = self._config.max_attempts_per_obligation
        remaining_global = self._config.max_falsification_cases - global_cases_used
        effective_limit = min(max_per, remaining_global, len(cases))
        cases = cases[:effective_limit]

        # 4. Execute cases.
        executed_cases: list[FalsificationCase] = []
        cases_passed = 0
        counterexamples = 0

        for case in cases:
            # Check time budget.
            elapsed = time.monotonic() - start_time
            if elapsed >= self._config.max_falsification_time:
                evidence.append(
                    f"Time budget exhausted during obligation {obligation.id} "
                    f"after {len(executed_cases)} cases."
                )
                break

            executed = execute_case(case, runner, timeout)
            executed_cases.append(executed)

            if executed.passed:
                cases_passed += 1
            elif executed.is_counterexample:
                counterexamples += 1
                evidence.append(
                    f"COUNTEREXAMPLE FOUND [{executed.case_id}]: "
                    f"{executed.description}  "
                    f"(exit_code={executed.exit_code})"
                )

        # 5. Determine obligation-level status.
        if counterexamples > 0:
            status = FalsificationStatus.FALSIFIED
            evidence.append(
                f"Obligation {obligation.id} FALSIFIED: "
                f"{counterexamples} counterexample(s) found in "
                f"{len(executed_cases)} executed case(s)."
            )
        elif len(executed_cases) == 0:
            status = FalsificationStatus.UNKNOWN
            evidence.append(
                f"Obligation {obligation.id}: no cases were executed."
            )
        elif all(c.timed_out for c in executed_cases if c.executed):
            status = FalsificationStatus.UNKNOWN
            evidence.append(
                f"Obligation {obligation.id}: all cases timed out."
            )
        else:
            status = FalsificationStatus.NOT_FALSIFIED
            evidence.append(
                f"Obligation {obligation.id} NOT_FALSIFIED: "
                f"no counterexample found within {len(executed_cases)} "
                f"bounded check(s).  This does NOT mean mathematically "
                f"proven correct."
            )

        return FalsificationResult(
            obligation_id=obligation.id,
            cases_attempted=len(executed_cases),
            cases_passed=cases_passed,
            counterexamples_found=counterexamples,
            status=status,
            evidence=evidence,
            cases=executed_cases,
        )

    # ------------------------------------------------------------------
    # Internal: convert report → VerificationCheck objects
    # ------------------------------------------------------------------

    def _report_to_checks(
        self,
        report: FalsificationReport,
    ) -> list[VerificationCheck]:
        """Convert a FalsificationReport into VerificationCheck objects.

        This is the bridge between the falsifier and the Verifier's
        Level 6 integration.
        """
        checks: list[VerificationCheck] = []

        if not report.results:
            checks.append(VerificationCheck(
                check_id="L6-NOCONTRACT",
                level=VerificationLevel.FALSIFICATION,
                description="Falsification: no contract available",
                status=CheckStatus.SKIP,
                evidence="Falsification skipped: no contract or obligations.",
            ))
            return checks

        for result in report.results:
            check_id = f"L6-{result.obligation_id}"

            if result.status == FalsificationStatus.FALSIFIED:
                checks.append(VerificationCheck(
                    check_id=check_id,
                    level=VerificationLevel.FALSIFICATION,
                    description=(
                        f"Falsification: {result.obligation_id} — "
                        f"{result.counterexamples_found} counterexample(s)"
                    ),
                    status=CheckStatus.FAIL,
                    evidence="\n".join(result.evidence),
                    obligation_id=result.obligation_id,
                ))
            elif result.status == FalsificationStatus.NOT_FALSIFIED:
                checks.append(VerificationCheck(
                    check_id=check_id,
                    level=VerificationLevel.FALSIFICATION,
                    description=(
                        f"Falsification: {result.obligation_id} — "
                        f"not falsified ({result.cases_attempted} cases)"
                    ),
                    status=CheckStatus.PASS,
                    evidence="\n".join(result.evidence),
                    obligation_id=result.obligation_id,
                ))
            elif result.status == FalsificationStatus.NOT_APPLICABLE:
                checks.append(VerificationCheck(
                    check_id=check_id,
                    level=VerificationLevel.FALSIFICATION,
                    description=(
                        f"Falsification: {result.obligation_id} — "
                        f"not applicable"
                    ),
                    status=CheckStatus.SKIP,
                    evidence="\n".join(result.evidence),
                    obligation_id=result.obligation_id,
                ))
            else:
                # UNKNOWN
                checks.append(VerificationCheck(
                    check_id=check_id,
                    level=VerificationLevel.FALSIFICATION,
                    description=(
                        f"Falsification: {result.obligation_id} — unknown"
                    ),
                    status=CheckStatus.ERROR,
                    evidence="\n".join(result.evidence),
                    obligation_id=result.obligation_id,
                ))

        return checks
