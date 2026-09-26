"""Tests for src/falsifier.py — Active Falsification Engine."""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.falsifier import (
    CaseCategory,
    DEFAULT_GENERATORS,
    FalsificationCase,
    FalsificationConfig,
    FalsificationEngine,
    FalsificationReport,
    FalsificationResult,
    FalsificationStatus,
    execute_case,
    generate_boundary_cases,
    generate_empty_null_cases,
    generate_error_path_cases,
    generate_invalid_input_cases,
    generate_off_by_one_cases,
    generate_permission_safety_cases,
)
from src.verifier import (
    CheckStatus,
    CommandResult,
    FalsificationInterface,
    VerificationCheck,
    VerificationLevel,
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
    duration: float = 0.05,
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


def _make_obligation(
    ob_id: str = "OB-1",
    description: str = "locked accounts must be rejected",
    ob_type: ObligationType = ObligationType.BEHAVIOR,
    method: str = "pytest tests/test_auth.py -x -q",
) -> ProofObligation:
    return ProofObligation(
        id=ob_id,
        description=description,
        type=ob_type,
        verification_method=method,
    )


def _make_contract(
    obligations: list[ProofObligation] | None = None,
) -> TaskContract:
    """Build a minimal TaskContract for testing."""
    if obligations is None:
        obligations = [
            _make_obligation(
                ob_id="OB-1",
                description="Discounts above 50% must be rejected",
            ),
            _make_obligation(
                ob_id="OB-2",
                description="Valid discounts must be accepted",
                ob_type=ObligationType.REGRESSION,
            ),
        ]
    return TaskContract(
        task_goal="Fix discount validation",
        behavioral_requirements=["Discounts above 50% must be rejected"],
        acceptance_criteria=["test_discount_validation passes"],
        constraints=["Do not modify pricing model"],
        non_goals=["Do not change tax calculation"],
        affected_area="pricing",
        expected_files=["src/pricing/discounts.py"],
        risk_level=RiskLevel.MEDIUM,
        verification_plan="Run pricing tests",
        proof_obligations=obligations,
    )


def _make_case(
    case_id: str = "CASE-1",
    obligation_id: str = "OB-1",
    category: CaseCategory = CaseCategory.BOUNDARY_VALUES,
    description: str = "boundary probe",
    test_command: str = "pytest -x -q -k test_boundary",
    expected: str = "should be handled",
) -> FalsificationCase:
    return FalsificationCase(
        case_id=case_id,
        obligation_id=obligation_id,
        category=category,
        description=description,
        test_command=test_command,
        expected_behaviour=expected,
    )


# ═══════════════════════════════════════════════════════════════════
# FalsificationCase
# ═══════════════════════════════════════════════════════════════════


class TestFalsificationCase:
    def test_initial_state(self):
        case = _make_case()
        assert case.executed is False
        assert case.passed is False
        assert case.is_counterexample is False
        assert case.exit_code is None

    def test_to_dict_unexecuted(self):
        case = _make_case()
        d = case.to_dict()
        assert d["case_id"] == "CASE-1"
        assert d["executed"] is False
        assert "exit_code" not in d
        assert "duration" not in d

    def test_to_dict_executed(self):
        case = _make_case()
        case.executed = True
        case.passed = True
        case.exit_code = 0
        case.duration = 0.5
        case.stdout = "ok"
        d = case.to_dict()
        assert d["executed"] is True
        assert d["exit_code"] == 0
        assert d["duration"] == 0.5
        assert d["stdout"] == "ok"

    def test_to_dict_counterexample(self):
        case = _make_case()
        case.executed = True
        case.passed = False
        case.is_counterexample = True
        case.exit_code = 1
        d = case.to_dict()
        assert d["is_counterexample"] is True
        assert d["passed"] is False


# ═══════════════════════════════════════════════════════════════════
# FalsificationResult
# ═══════════════════════════════════════════════════════════════════


class TestFalsificationResult:
    def test_basic_creation(self):
        r = FalsificationResult(
            obligation_id="OB-1",
            cases_attempted=5,
            cases_passed=4,
            counterexamples_found=1,
            status=FalsificationStatus.FALSIFIED,
            evidence=["found issue"],
        )
        assert r.obligation_id == "OB-1"
        assert r.status == FalsificationStatus.FALSIFIED

    def test_to_dict(self):
        r = FalsificationResult(
            obligation_id="OB-1",
            cases_attempted=3,
            cases_passed=3,
            counterexamples_found=0,
            status=FalsificationStatus.NOT_FALSIFIED,
            evidence=["all passed"],
            cases=[_make_case()],
        )
        d = r.to_dict()
        assert d["obligation_id"] == "OB-1"
        assert d["status"] == "NOT_FALSIFIED"
        assert len(d["cases"]) == 1


# ═══════════════════════════════════════════════════════════════════
# FalsificationReport
# ═══════════════════════════════════════════════════════════════════


class TestFalsificationReport:
    def test_basic_creation(self):
        report = FalsificationReport(
            results=[],
            total_cases_attempted=0,
            total_counterexamples=0,
            overall_status=FalsificationStatus.NOT_APPLICABLE,
            duration=0.0,
            evidence=[],
        )
        assert report.overall_status == FalsificationStatus.NOT_APPLICABLE

    def test_to_dict(self):
        result = FalsificationResult(
            obligation_id="OB-1",
            cases_attempted=2,
            cases_passed=2,
            counterexamples_found=0,
            status=FalsificationStatus.NOT_FALSIFIED,
            evidence=["ok"],
        )
        report = FalsificationReport(
            results=[result],
            total_cases_attempted=2,
            total_counterexamples=0,
            overall_status=FalsificationStatus.NOT_FALSIFIED,
            duration=0.1,
            evidence=["ok"],
        )
        d = report.to_dict()
        assert d["overall_status"] == "NOT_FALSIFIED"
        assert d["total_cases_attempted"] == 2


# ═══════════════════════════════════════════════════════════════════
# FalsificationConfig
# ═══════════════════════════════════════════════════════════════════


class TestFalsificationConfig:
    def test_defaults(self):
        cfg = FalsificationConfig()
        assert cfg.max_falsification_cases == 50
        assert cfg.max_falsification_time == 300.0
        assert cfg.max_attempts_per_obligation == 10
        assert cfg.command_timeout == 30

    def test_custom(self):
        cfg = FalsificationConfig(
            max_falsification_cases=5,
            max_falsification_time=10.0,
            max_attempts_per_obligation=2,
            command_timeout=5,
        )
        assert cfg.max_falsification_cases == 5
        assert cfg.max_falsification_time == 10.0


# ═══════════════════════════════════════════════════════════════════
# Case generators
# ═══════════════════════════════════════════════════════════════════


class TestBoundaryGenerator:
    def test_generates_cases_for_numeric_description(self):
        ob = _make_obligation(description="Discounts above 50% must be rejected")
        cases = generate_boundary_cases(ob, [], {})
        assert len(cases) >= 2
        assert all(c.category == CaseCategory.BOUNDARY_VALUES for c in cases)
        assert all(c.obligation_id == ob.id for c in cases)

    def test_no_numbers_no_cases(self):
        ob = _make_obligation(description="locked accounts must be rejected")
        cases = generate_boundary_cases(ob, [], {})
        assert len(cases) == 0

    def test_uses_context_test_command(self):
        ob = _make_obligation(description="max 100 items")
        ctx = {"test_command": "pytest custom_test.py"}
        cases = generate_boundary_cases(ob, [], ctx)
        assert all(c.test_command == "pytest custom_test.py" for c in cases)

    def test_float_boundary(self):
        ob = _make_obligation(description="rate must not exceed 3.5%")
        cases = generate_boundary_cases(ob, [], {})
        descs = [c.description for c in cases]
        assert any("3.5" in d for d in descs)


class TestEmptyNullGenerator:
    def test_generates_two_cases(self):
        ob = _make_obligation()
        cases = generate_empty_null_cases(ob, [], {})
        assert len(cases) == 2
        categories = {c.category for c in cases}
        assert categories == {CaseCategory.EMPTY_NULL_INPUTS}

    def test_case_ids_include_obligation_id(self):
        ob = _make_obligation(ob_id="OB-99")
        cases = generate_empty_null_cases(ob, [], {})
        assert all("OB-99" in c.case_id for c in cases)


class TestInvalidInputGenerator:
    def test_generates_two_cases(self):
        ob = _make_obligation()
        cases = generate_invalid_input_cases(ob, [], {})
        assert len(cases) == 2
        assert all(c.category == CaseCategory.INVALID_INPUTS for c in cases)


class TestOffByOneGenerator:
    def test_generates_one_case(self):
        ob = _make_obligation()
        cases = generate_off_by_one_cases(ob, [], {})
        assert len(cases) == 1
        assert cases[0].category == CaseCategory.OFF_BY_ONE


class TestErrorPathGenerator:
    def test_generates_one_case(self):
        ob = _make_obligation()
        cases = generate_error_path_cases(ob, [], {})
        assert len(cases) == 1
        assert cases[0].category == CaseCategory.ERROR_PATHS


class TestPermissionSafetyGenerator:
    def test_generates_for_safety_obligation(self):
        ob = _make_obligation(ob_type=ObligationType.SAFETY)
        cases = generate_permission_safety_cases(ob, [], {})
        assert len(cases) == 1
        assert cases[0].category == CaseCategory.PERMISSION_SAFETY

    def test_generates_for_constraint_obligation(self):
        ob = _make_obligation(ob_type=ObligationType.CONSTRAINT)
        cases = generate_permission_safety_cases(ob, [], {})
        assert len(cases) == 1

    def test_empty_for_behavior_obligation(self):
        ob = _make_obligation(ob_type=ObligationType.BEHAVIOR)
        cases = generate_permission_safety_cases(ob, [], {})
        assert len(cases) == 0

    def test_empty_for_regression_obligation(self):
        ob = _make_obligation(ob_type=ObligationType.REGRESSION)
        cases = generate_permission_safety_cases(ob, [], {})
        assert len(cases) == 0


class TestDefaultGenerators:
    def test_all_generators_present(self):
        assert len(DEFAULT_GENERATORS) == 6

    def test_all_generators_callable(self):
        for gen in DEFAULT_GENERATORS:
            assert callable(gen)


# ═══════════════════════════════════════════════════════════════════
# execute_case
# ═══════════════════════════════════════════════════════════════════


class TestExecuteCase:
    def test_passing_case(self):
        case = _make_case()
        runner = _make_runner(success=True, exit_code=0)
        result = execute_case(case, runner, timeout=30)
        assert result.executed is True
        assert result.passed is True
        assert result.is_counterexample is False
        assert result.exit_code == 0

    def test_failing_case_is_counterexample(self):
        case = _make_case()
        runner = _make_runner(success=False, exit_code=1, stderr="FAILED")
        result = execute_case(case, runner, timeout=30)
        assert result.executed is True
        assert result.passed is False
        assert result.is_counterexample is True
        assert result.exit_code == 1

    def test_timed_out_case(self):
        case = _make_case()
        runner = _make_runner(success=False, timed_out=True, exit_code=-1)
        result = execute_case(case, runner, timeout=5)
        assert result.executed is True
        assert result.passed is False
        assert result.is_counterexample is False
        assert result.timed_out is True

    def test_runner_exception(self):
        case = _make_case()
        def bad_runner(cmd, timeout):
            raise RuntimeError("boom")
        result = execute_case(case, bad_runner, timeout=30)
        assert result.executed is True
        assert result.passed is False
        assert result.is_counterexample is False
        assert result.exit_code == -1
        assert "boom" in result.stderr

    def test_stdout_stderr_captured(self):
        case = _make_case()
        runner = _make_runner(success=True, stdout="output", stderr="warnings")
        result = execute_case(case, runner, timeout=30)
        assert result.stdout == "output"
        assert result.stderr == "warnings"

    def test_duration_captured(self):
        case = _make_case()
        runner = _make_runner(success=True, duration=1.23)
        result = execute_case(case, runner, timeout=30)
        assert result.duration == 1.23


# ═══════════════════════════════════════════════════════════════════
# FalsificationEngine — basic
# ═══════════════════════════════════════════════════════════════════


class TestFalsificationEngineBasic:
    def test_implements_interface(self):
        engine = FalsificationEngine()
        assert isinstance(engine, FalsificationInterface)

    def test_no_contract(self):
        engine = FalsificationEngine()
        runner = _make_runner()
        report = engine.falsify(None, [], runner)
        assert report.overall_status == FalsificationStatus.NOT_APPLICABLE
        assert report.total_cases_attempted == 0

    def test_empty_obligations(self):
        contract = _make_contract(obligations=[])
        engine = FalsificationEngine()
        runner = _make_runner()
        report = engine.falsify(contract, [], runner)
        assert report.overall_status == FalsificationStatus.NOT_APPLICABLE

    def test_config_accessible(self):
        cfg = FalsificationConfig(max_falsification_cases=7)
        engine = FalsificationEngine(config=cfg)
        assert engine.config.max_falsification_cases == 7

    def test_last_report_initially_none(self):
        engine = FalsificationEngine()
        assert engine.last_report is None

    def test_last_report_set_after_falsify(self):
        engine = FalsificationEngine()
        runner = _make_runner()
        engine.falsify(None, [], runner)
        assert engine.last_report is not None


# ═══════════════════════════════════════════════════════════════════
# FalsificationEngine — successful falsification (no counterexamples)
# ═══════════════════════════════════════════════════════════════════


class TestSuccessfulFalsification:
    def test_all_cases_pass(self):
        """All generated adversarial cases pass → NOT_FALSIFIED."""
        contract = _make_contract()
        engine = FalsificationEngine()
        runner = _make_runner(success=True, stdout="all passed")
        report = engine.falsify(contract, ["src/pricing/discounts.py"], runner)
        assert report.overall_status == FalsificationStatus.NOT_FALSIFIED
        assert report.total_counterexamples == 0
        assert report.total_cases_attempted > 0

    def test_evidence_is_honest(self):
        """Evidence should say 'NOT_FALSIFIED' and disclaim proof."""
        contract = _make_contract()
        engine = FalsificationEngine()
        runner = _make_runner(success=True)
        report = engine.falsify(contract, [], runner)
        all_evidence = " ".join(report.evidence)
        assert "NOT_FALSIFIED" in all_evidence
        # The disclaimer "does NOT mean mathematically proven correct" is OK.
        # But there must never be a positive assertion of proof.
        assert "mathematically proven correct" not in all_evidence.lower().replace(
            "does not mean mathematically proven correct", ""
        )

    def test_individual_results(self):
        contract = _make_contract()
        engine = FalsificationEngine()
        runner = _make_runner(success=True)
        report = engine.falsify(contract, [], runner)
        assert len(report.results) == 2
        for r in report.results:
            assert r.status == FalsificationStatus.NOT_FALSIFIED
            assert r.counterexamples_found == 0


# ═══════════════════════════════════════════════════════════════════
# FalsificationEngine — counterexample found
# ═══════════════════════════════════════════════════════════════════


class TestCounterexampleFound:
    def test_single_counterexample(self):
        """If any case fails → FALSIFIED."""
        contract = _make_contract()
        # First call passes, rest fail.
        responses = [
            {"success": True, "stdout": "ok", "exit_code": 0, "duration": 0.1},
            {"success": False, "stderr": "FAILED", "exit_code": 1, "duration": 0.1},
        ]
        runner = _make_multi_runner(responses)
        engine = FalsificationEngine()
        report = engine.falsify(contract, [], runner)
        assert report.overall_status == FalsificationStatus.FALSIFIED
        assert report.total_counterexamples >= 1

    def test_all_cases_fail(self):
        """If every case fails → FALSIFIED."""
        contract = _make_contract()
        runner = _make_runner(success=False, exit_code=1, stderr="FAIL")
        engine = FalsificationEngine()
        report = engine.falsify(contract, [], runner)
        assert report.overall_status == FalsificationStatus.FALSIFIED
        assert report.total_counterexamples > 0

    def test_counterexample_evidence(self):
        """Evidence must mention the counterexample."""
        contract = _make_contract()
        runner = _make_runner(success=False, exit_code=1, stderr="FAIL")
        engine = FalsificationEngine()
        report = engine.falsify(contract, [], runner)
        all_evidence = " ".join(report.evidence)
        assert "COUNTEREXAMPLE" in all_evidence

    def test_falsified_result_per_obligation(self):
        contract = _make_contract()
        runner = _make_runner(success=False, exit_code=1)
        engine = FalsificationEngine()
        report = engine.falsify(contract, [], runner)
        for r in report.results:
            assert r.status == FalsificationStatus.FALSIFIED
            assert r.counterexamples_found > 0


# ═══════════════════════════════════════════════════════════════════
# FalsificationEngine — custom cases
# ═══════════════════════════════════════════════════════════════════


class TestCustomCases:
    def test_custom_cases_are_executed(self):
        custom = {
            "OB-1": [
                _make_case(
                    case_id="CUSTOM-1",
                    obligation_id="OB-1",
                    description="custom boundary test",
                    test_command="pytest -k test_custom",
                ),
            ],
        }
        contract = _make_contract()
        engine = FalsificationEngine(custom_cases=custom)
        runner = _make_runner(success=True)
        report = engine.falsify(contract, [], runner)
        # Custom case should be among the executed cases.
        ob1_result = next(r for r in report.results if r.obligation_id == "OB-1")
        case_ids = [c.case_id for c in ob1_result.cases]
        assert "CUSTOM-1" in case_ids

    def test_custom_counterexample(self):
        custom = {
            "OB-1": [
                _make_case(case_id="CUSTOM-BAD", obligation_id="OB-1"),
            ],
        }
        contract = _make_contract()
        engine = FalsificationEngine(
            custom_cases=custom,
            # Disable built-in generators for clarity.
            generators=[],
        )
        runner = _make_runner(success=False, exit_code=1)
        report = engine.falsify(contract, [], runner)
        ob1 = next(r for r in report.results if r.obligation_id == "OB-1")
        assert ob1.status == FalsificationStatus.FALSIFIED
        assert ob1.counterexamples_found == 1


# ═══════════════════════════════════════════════════════════════════
# FalsificationEngine — malformed generated case
# ═══════════════════════════════════════════════════════════════════


class TestMalformedCase:
    def test_generator_exception_is_non_fatal(self):
        """A generator that raises should not crash the engine."""
        def bad_gen(ob, files, ctx):
            raise ValueError("generator broke")

        contract = _make_contract()
        engine = FalsificationEngine(generators=[bad_gen])
        runner = _make_runner(success=True)
        # Should not raise.
        report = engine.falsify(contract, [], runner)
        # With no valid cases, obligations should be NOT_APPLICABLE.
        for r in report.results:
            assert r.status == FalsificationStatus.NOT_APPLICABLE

    def test_runner_exception_during_execution(self):
        """A runner that raises should mark the case as failed but not crash."""
        def crashing_runner(cmd, timeout):
            raise RuntimeError("runner crashed")

        custom = {
            "OB-1": [_make_case(case_id="CRASH-1", obligation_id="OB-1")],
        }
        contract = _make_contract()
        engine = FalsificationEngine(custom_cases=custom, generators=[])
        report = engine.falsify(contract, [], crashing_runner)
        ob1 = next(r for r in report.results if r.obligation_id == "OB-1")
        assert ob1.cases_attempted == 1
        assert ob1.cases[0].executed is True
        # Runner exception → not a counterexample, just an error.
        assert ob1.cases[0].is_counterexample is False


# ═══════════════════════════════════════════════════════════════════
# FalsificationEngine — timeout
# ═══════════════════════════════════════════════════════════════════


class TestTimeout:
    def test_case_timeout_is_unknown(self):
        """If all cases time out, obligation status should be UNKNOWN."""
        custom = {
            "OB-1": [_make_case(case_id="TO-1", obligation_id="OB-1")],
        }
        contract = _make_contract()
        engine = FalsificationEngine(custom_cases=custom, generators=[])
        runner = _make_runner(success=False, timed_out=True, exit_code=-1)
        report = engine.falsify(contract, [], runner)
        ob1 = next(r for r in report.results if r.obligation_id == "OB-1")
        assert ob1.status == FalsificationStatus.UNKNOWN
        assert ob1.counterexamples_found == 0

    def test_global_time_budget(self):
        """Obligations beyond the time budget should get UNKNOWN status."""
        cfg = FalsificationConfig(max_falsification_time=0.0)
        contract = _make_contract()
        engine = FalsificationEngine(config=cfg)
        runner = _make_runner(success=True)
        report = engine.falsify(contract, [], runner)
        # All obligations should be UNKNOWN because time exhausted.
        for r in report.results:
            assert r.status == FalsificationStatus.UNKNOWN
        # Evidence appears in individual results when budget is exhausted
        # before reaching the obligation.
        all_result_evidence = []
        for r in report.results:
            all_result_evidence.extend(r.evidence)
        assert any("exhausted" in e.lower() or "budget" in e.lower()
                   for e in all_result_evidence)


# ═══════════════════════════════════════════════════════════════════
# FalsificationEngine — unknown result
# ═══════════════════════════════════════════════════════════════════


class TestUnknownResult:
    def test_mixed_timeout_and_pass(self):
        """Mix of timeout and passing cases → NOT_FALSIFIED (not UNKNOWN)
        because at least some cases ran and passed."""
        custom = {
            "OB-1": [
                _make_case(case_id="PASS-1", obligation_id="OB-1"),
                _make_case(case_id="TO-1", obligation_id="OB-1"),
            ],
        }
        responses = [
            {"success": True, "exit_code": 0, "duration": 0.1},
            {"success": False, "timed_out": True, "exit_code": -1, "duration": 30.0},
        ]
        contract = _make_contract()
        engine = FalsificationEngine(custom_cases=custom, generators=[])
        runner = _make_multi_runner(responses)
        report = engine.falsify(contract, [], runner)
        ob1 = next(r for r in report.results if r.obligation_id == "OB-1")
        # At least one case passed, no counterexamples → NOT_FALSIFIED.
        assert ob1.status == FalsificationStatus.NOT_FALSIFIED

    def test_overall_unknown_when_any_obligation_unknown(self):
        """If one obligation is NOT_FALSIFIED and another is UNKNOWN,
        overall should be UNKNOWN."""
        obligations = [
            _make_obligation(ob_id="OB-1"),
            _make_obligation(ob_id="OB-2"),
        ]
        contract = _make_contract(obligations=obligations)
        # OB-1 gets custom cases, OB-2 has no generators → NOT_APPLICABLE.
        custom = {
            "OB-1": [_make_case(case_id="P1", obligation_id="OB-1")],
        }

        # We need OB-2 to be UNKNOWN.  Use a generator that only works for OB-2
        # but produces cases that all time out.
        def timeout_gen(ob, files, ctx):
            if ob.id == "OB-2":
                return [_make_case(case_id="TO-2", obligation_id="OB-2")]
            return []

        responses = [
            # OB-1 custom case passes.
            {"success": True, "exit_code": 0, "duration": 0.1},
            # OB-2 case times out.
            {"success": False, "timed_out": True, "exit_code": -1, "duration": 30.0},
        ]

        engine = FalsificationEngine(
            custom_cases=custom,
            generators=[timeout_gen],
        )
        runner = _make_multi_runner(responses)
        report = engine.falsify(contract, [], runner)
        assert report.overall_status == FalsificationStatus.UNKNOWN


# ═══════════════════════════════════════════════════════════════════
# FalsificationEngine — bounded execution
# ═══════════════════════════════════════════════════════════════════


class TestBoundedExecution:
    def test_max_attempts_per_obligation(self):
        """No more than max_attempts_per_obligation cases per obligation."""
        cfg = FalsificationConfig(max_attempts_per_obligation=2)
        # Create many custom cases.
        custom = {
            "OB-1": [_make_case(case_id=f"C-{i}", obligation_id="OB-1") for i in range(20)],
        }
        contract = _make_contract()
        engine = FalsificationEngine(config=cfg, custom_cases=custom, generators=[])
        runner = _make_runner(success=True)
        report = engine.falsify(contract, [], runner)
        ob1 = next(r for r in report.results if r.obligation_id == "OB-1")
        assert ob1.cases_attempted <= 2

    def test_max_global_cases(self):
        """Total cases across all obligations must not exceed max_falsification_cases."""
        cfg = FalsificationConfig(
            max_falsification_cases=3,
            max_attempts_per_obligation=10,
        )
        custom = {
            "OB-1": [_make_case(case_id=f"A-{i}", obligation_id="OB-1") for i in range(5)],
            "OB-2": [_make_case(case_id=f"B-{i}", obligation_id="OB-2") for i in range(5)],
        }
        contract = _make_contract()
        engine = FalsificationEngine(config=cfg, custom_cases=custom, generators=[])
        runner = _make_runner(success=True)
        report = engine.falsify(contract, [], runner)
        assert report.total_cases_attempted <= 3

    def test_global_budget_exhaustion_marks_remaining_unknown(self):
        """When global budget is hit, remaining obligations get UNKNOWN."""
        cfg = FalsificationConfig(max_falsification_cases=1)
        custom = {
            "OB-1": [_make_case(case_id="ONLY", obligation_id="OB-1")],
            "OB-2": [_make_case(case_id="SKIP", obligation_id="OB-2")],
        }
        contract = _make_contract()
        engine = FalsificationEngine(config=cfg, custom_cases=custom, generators=[])
        runner = _make_runner(success=True)
        report = engine.falsify(contract, [], runner)
        ob2 = next(r for r in report.results if r.obligation_id == "OB-2")
        assert ob2.status == FalsificationStatus.UNKNOWN
        assert ob2.cases_attempted == 0


# ═══════════════════════════════════════════════════════════════════
# FalsificationEngine — multiple obligations
# ═══════════════════════════════════════════════════════════════════


class TestMultipleObligations:
    def test_all_obligations_processed(self):
        obligations = [
            _make_obligation(ob_id="OB-1", description="check 50% threshold"),
            _make_obligation(ob_id="OB-2", description="check valid inputs"),
            _make_obligation(ob_id="OB-3", description="safety", ob_type=ObligationType.SAFETY),
        ]
        contract = _make_contract(obligations=obligations)
        engine = FalsificationEngine()
        runner = _make_runner(success=True)
        report = engine.falsify(contract, [], runner)
        result_ids = {r.obligation_id for r in report.results}
        assert result_ids == {"OB-1", "OB-2", "OB-3"}

    def test_mixed_results(self):
        """Some obligations pass, some fail."""
        obligations = [
            _make_obligation(ob_id="OB-1"),
            _make_obligation(ob_id="OB-2"),
        ]
        custom = {
            "OB-1": [_make_case(case_id="P1", obligation_id="OB-1")],
            "OB-2": [_make_case(case_id="F1", obligation_id="OB-2")],
        }
        # OB-1 passes, OB-2 fails.
        responses = [
            {"success": True, "exit_code": 0, "duration": 0.1},
            {"success": False, "exit_code": 1, "stderr": "FAIL", "duration": 0.1},
        ]
        contract = _make_contract(obligations=obligations)
        engine = FalsificationEngine(custom_cases=custom, generators=[])
        runner = _make_multi_runner(responses)
        report = engine.falsify(contract, [], runner)
        ob1 = next(r for r in report.results if r.obligation_id == "OB-1")
        ob2 = next(r for r in report.results if r.obligation_id == "OB-2")
        assert ob1.status == FalsificationStatus.NOT_FALSIFIED
        assert ob2.status == FalsificationStatus.FALSIFIED
        # Overall should be FALSIFIED (any counterexample taints overall).
        assert report.overall_status == FalsificationStatus.FALSIFIED


# ═══════════════════════════════════════════════════════════════════
# FalsificationEngine — Verifier integration (attempt_falsification)
# ═══════════════════════════════════════════════════════════════════


class TestVerifierIntegration:
    def test_attempt_falsification_returns_checks(self):
        contract = _make_contract()
        engine = FalsificationEngine()
        runner = _make_runner(success=True)
        checks = engine.attempt_falsification(contract, [], runner, timeout=30)
        assert isinstance(checks, list)
        assert all(isinstance(c, VerificationCheck) for c in checks)
        assert all(c.level == VerificationLevel.FALSIFICATION for c in checks)

    def test_no_contract_returns_skip(self):
        engine = FalsificationEngine()
        runner = _make_runner()
        checks = engine.attempt_falsification(None, [], runner, timeout=30)
        assert len(checks) == 1
        assert checks[0].status == CheckStatus.SKIP

    def test_falsified_returns_fail_check(self):
        contract = _make_contract()
        engine = FalsificationEngine()
        runner = _make_runner(success=False, exit_code=1)
        checks = engine.attempt_falsification(contract, [], runner, timeout=30)
        fail_checks = [c for c in checks if c.status == CheckStatus.FAIL]
        assert len(fail_checks) > 0

    def test_not_falsified_returns_pass_check(self):
        contract = _make_contract()
        engine = FalsificationEngine()
        runner = _make_runner(success=True)
        checks = engine.attempt_falsification(contract, [], runner, timeout=30)
        pass_checks = [c for c in checks if c.status == CheckStatus.PASS]
        assert len(pass_checks) > 0

    def test_unknown_returns_error_check(self):
        """Timeout-only cases → UNKNOWN → ERROR check."""
        custom = {
            "OB-1": [_make_case(case_id="TO", obligation_id="OB-1")],
        }
        obligations = [_make_obligation(ob_id="OB-1")]
        contract = _make_contract(obligations=obligations)
        engine = FalsificationEngine(custom_cases=custom, generators=[])
        runner = _make_runner(success=False, timed_out=True, exit_code=-1)
        checks = engine.attempt_falsification(contract, [], runner, timeout=30)
        error_checks = [c for c in checks if c.status == CheckStatus.ERROR]
        assert len(error_checks) >= 1

    def test_check_ids_include_obligation(self):
        contract = _make_contract()
        engine = FalsificationEngine()
        runner = _make_runner(success=True)
        checks = engine.attempt_falsification(contract, [], runner, timeout=30)
        for c in checks:
            assert c.check_id.startswith("L6-")


# ═══════════════════════════════════════════════════════════════════
# FalsificationEngine — with Verifier
# ═══════════════════════════════════════════════════════════════════


class TestWithVerifier:
    """Test that FalsificationEngine plugs into the Verifier correctly."""

    def test_verifier_uses_engine_at_level_6(self):
        from src.verifier import Verifier, VerificationLevel, VerificationStatus

        contract = _make_contract()
        engine = FalsificationEngine()
        runner = _make_runner(success=True, stdout="ok")

        verifier = Verifier(
            runner,
            falsifier=engine,
            default_timeout=30,
        )
        result = verifier.verify(
            changed_files=["src/pricing/discounts.py"],
            contract=contract,
            file_contents={"src/pricing/discounts.py": "# code"},
        )
        # Level 6 checks should be present.
        l6 = [c for c in result.checks if c.level == VerificationLevel.FALSIFICATION]
        assert len(l6) > 0
        # No counterexample → all L6 should be PASS or SKIP.
        for c in l6:
            assert c.status in (CheckStatus.PASS, CheckStatus.SKIP)

    def test_verifier_detects_falsification(self):
        from src.verifier import Verifier, VerificationLevel, VerificationStatus

        contract = _make_contract()
        engine = FalsificationEngine()
        runner = _make_runner(success=False, exit_code=1, stderr="FAIL")

        verifier = Verifier(
            runner,
            falsifier=engine,
            default_timeout=30,
        )
        result = verifier.verify(
            changed_files=["src/pricing/discounts.py"],
            contract=contract,
            file_contents={"src/pricing/discounts.py": "# code"},
        )
        # Result should be FAILED because L6 found counterexamples.
        assert result.status == VerificationStatus.FAILED
        assert result.passed is False


# ═══════════════════════════════════════════════════════════════════
# Enum & status coverage
# ═══════════════════════════════════════════════════════════════════


class TestEnums:
    def test_falsification_status_values(self):
        assert FalsificationStatus.NOT_FALSIFIED.value == "NOT_FALSIFIED"
        assert FalsificationStatus.FALSIFIED.value == "FALSIFIED"
        assert FalsificationStatus.UNKNOWN.value == "UNKNOWN"
        assert FalsificationStatus.NOT_APPLICABLE.value == "NOT_APPLICABLE"

    def test_case_category_values(self):
        assert CaseCategory.BOUNDARY_VALUES.value == "boundary_values"
        assert CaseCategory.EMPTY_NULL_INPUTS.value == "empty_null_inputs"
        assert CaseCategory.INVALID_INPUTS.value == "invalid_inputs"
        assert CaseCategory.OFF_BY_ONE.value == "off_by_one"
        assert CaseCategory.TYPE_VARIATIONS.value == "type_variations"
        assert CaseCategory.REGRESSION.value == "regression"
        assert CaseCategory.ERROR_PATHS.value == "error_paths"
        assert CaseCategory.PERMISSION_SAFETY.value == "permission_safety"
        assert CaseCategory.CONTRACT_EDGE_CASES.value == "contract_edge_cases"


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_obligation_with_no_applicable_generators(self):
        """If no generator produces cases and no custom cases exist,
        status should be NOT_APPLICABLE."""
        def empty_gen(ob, files, ctx):
            return []

        contract = _make_contract()
        engine = FalsificationEngine(generators=[empty_gen])
        runner = _make_runner(success=True)
        report = engine.falsify(contract, [], runner)
        for r in report.results:
            assert r.status == FalsificationStatus.NOT_APPLICABLE

    def test_single_obligation(self):
        obligations = [_make_obligation(ob_id="SINGLE")]
        contract = _make_contract(obligations=obligations)
        engine = FalsificationEngine()
        runner = _make_runner(success=True)
        report = engine.falsify(contract, [], runner)
        assert len(report.results) == 1
        assert report.results[0].obligation_id == "SINGLE"

    def test_duration_is_recorded(self):
        contract = _make_contract()
        engine = FalsificationEngine()
        runner = _make_runner(success=True)
        report = engine.falsify(contract, [], runner)
        assert report.duration >= 0

    def test_report_to_dict(self):
        contract = _make_contract()
        engine = FalsificationEngine()
        runner = _make_runner(success=True)
        report = engine.falsify(contract, [], runner)
        d = report.to_dict()
        assert "results" in d
        assert "overall_status" in d
        assert "total_cases_attempted" in d

    def test_context_passed_to_generators(self):
        """Verify that the context dict is passed to generators."""
        received = {}

        def spy_gen(ob, files, ctx):
            received.update(ctx)
            return []

        ctx = {"test_command": "pytest special.py", "extra": 42}
        contract = _make_contract()
        engine = FalsificationEngine(generators=[spy_gen], context=ctx)
        runner = _make_runner(success=True)
        engine.falsify(contract, [], runner)
        assert received.get("test_command") == "pytest special.py"
        assert received.get("extra") == 42

    def test_changed_files_passed_to_generators(self):
        """Verify that changed_files list reaches generators."""
        captured_files = []

        def file_spy(ob, files, ctx):
            captured_files.extend(files)
            return []

        contract = _make_contract()
        engine = FalsificationEngine(generators=[file_spy])
        runner = _make_runner(success=True)
        engine.falsify(contract, ["a.py", "b.py"], runner)
        # Each obligation triggers the generator, so files appear multiple times.
        assert "a.py" in captured_files
        assert "b.py" in captured_files
