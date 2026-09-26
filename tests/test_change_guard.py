"""Tests for the Change Guard (src/change_guard.py).

Covers:
    - Budget creation from contracts
    - Scope evaluation: within scope, expanded, blocked
    - Unexpected files / directories detection
    - Excessive file count
    - Unrelated modification detection
    - Outside-affected-area detection
    - Unjustified config changes
    - Test deletion detection
    - Test integrity heuristics:
        * trivial test functions (assert True / pass)
        * weakened assertions
        * trivial-only assertions
        * suspicious test modifications
    - Legitimate dependency recognition
    - Scope expansion report generation
    - Edge cases (empty inputs, no violations, config justified)
    - Serialisation
"""

from __future__ import annotations

import pytest

from src.contract_engine import RiskLevel, TaskContract, ProofObligation, ObligationType
from src.change_guard import (
    ChangeBudget,
    ChangeBudgetReport,
    ChangeGuard,
    ScopeExpansionReport,
    ScopeStatus,
    Violation,
    ViolationType,
    _compute_scope_description,
    _count_strong_assertions,
    _count_trivial_assertions,
    _extract_directories,
    _file_in_area,
    _has_trivial_test_functions,
    _is_config_path,
    _is_test_path,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_contract(**overrides) -> TaskContract:
    """Build a minimal TaskContract for testing."""
    defaults = dict(
        task_goal="Fix login for locked accounts",
        behavioral_requirements=["Reject locked accounts"],
        acceptance_criteria=["Locked → 403"],
        constraints=["No schema changes"],
        non_goals=["Don't change registration"],
        affected_area="auth",
        expected_files=["src/auth/login.py", "tests/test_login.py"],
        risk_level=RiskLevel.MEDIUM,
        verification_plan="Run auth tests",
        proof_obligations=[
            ProofObligation(
                id="OB-1",
                description="Reject locked",
                type=ObligationType.BEHAVIOR,
                verification_method="unit test",
            ),
        ],
    )
    defaults.update(overrides)
    return TaskContract(**defaults)


@pytest.fixture
def guard() -> ChangeGuard:
    return ChangeGuard()


@pytest.fixture
def contract() -> TaskContract:
    return _make_contract()


@pytest.fixture
def budget(guard: ChangeGuard, contract: TaskContract) -> ChangeBudget:
    return guard.create_budget(contract)


# ===================================================================
# Budget creation
# ===================================================================

class TestBudgetCreation:

    def test_budget_contains_expected_files(self, guard, contract):
        budget = guard.create_budget(contract)
        assert "src/auth/login.py" in budget.expected_files
        assert "tests/test_login.py" in budget.expected_files

    def test_budget_max_file_count_scales_with_risk(self, guard):
        low = guard.create_budget(_make_contract(risk_level=RiskLevel.LOW))
        high = guard.create_budget(_make_contract(risk_level=RiskLevel.HIGH))
        assert high.max_file_count > low.max_file_count

    def test_budget_max_file_count_minimum(self, guard, contract):
        budget = guard.create_budget(contract)
        # At least expected + 2
        assert budget.max_file_count >= len(contract.expected_files) + 2

    def test_budget_extra_allowed_files(self, guard, contract):
        budget = guard.create_budget(
            contract, extra_allowed_files=["src/auth/utils.py"],
        )
        assert "src/auth/utils.py" in budget.expected_files

    def test_budget_no_duplicate_extra_files(self, guard, contract):
        budget = guard.create_budget(
            contract, extra_allowed_files=["src/auth/login.py"],
        )
        assert budget.expected_files.count("src/auth/login.py") == 1

    def test_budget_allowed_directories(self, guard, contract):
        budget = guard.create_budget(contract)
        assert "src" in budget.allowed_directories or "tests" in budget.allowed_directories

    def test_budget_config_justified_flag(self, guard, contract):
        budget = guard.create_budget(contract, config_change_justified=True)
        assert budget.config_change_justified is True

    def test_budget_to_dict(self, guard, contract):
        budget = guard.create_budget(contract)
        d = budget.to_dict()
        assert "expected_files" in d
        assert "max_file_count" in d
        assert d["risk_level"] == "MEDIUM"

    def test_budget_affected_area(self, guard, contract):
        budget = guard.create_budget(contract)
        assert budget.affected_area == "auth"


# ===================================================================
# Within scope (no violations)
# ===================================================================

class TestWithinScope:

    def test_exact_expected_files(self, guard, budget):
        report = guard.evaluate(budget, ["src/auth/login.py", "tests/test_login.py"])
        assert report.status == ScopeStatus.WITHIN_SCOPE
        assert len(report.violations) == 0
        assert len(report.unexpected_files) == 0

    def test_subset_of_expected(self, guard, budget):
        report = guard.evaluate(budget, ["src/auth/login.py"])
        assert report.status == ScopeStatus.WITHIN_SCOPE

    def test_empty_actual_files(self, guard, budget):
        report = guard.evaluate(budget, [])
        assert report.status == ScopeStatus.WITHIN_SCOPE
        assert report.scope_expansion == 0.0

    def test_no_expansion(self, guard, budget):
        report = guard.evaluate(budget, ["src/auth/login.py"])
        assert report.scope_expansion == 0.0


# ===================================================================
# Unexpected files detection
# ===================================================================

class TestUnexpectedFiles:

    def test_unexpected_file_detected(self, guard, budget):
        actual = ["src/auth/login.py", "src/billing/invoice.py"]
        report = guard.evaluate(budget, actual)
        assert "src/billing/invoice.py" in report.unexpected_files

    def test_unexpected_files_list(self, guard, budget):
        actual = [
            "src/auth/login.py",
            "tests/test_login.py",
            "src/unrelated/foo.py",
            "src/unrelated/bar.py",
        ]
        report = guard.evaluate(budget, actual)
        assert len(report.unexpected_files) == 2


# ===================================================================
# Unexpected directories detection
# ===================================================================

class TestUnexpectedDirectories:

    def test_new_directory_flagged(self, guard, budget):
        actual = ["src/auth/login.py", "docs/guide.md"]
        report = guard.evaluate(budget, actual)
        dir_violations = [
            v for v in report.violations
            if v.type == ViolationType.UNEXPECTED_DIRECTORY
        ]
        assert len(dir_violations) >= 1
        assert any("docs" in v.file_path for v in dir_violations)

    def test_test_directory_not_flagged_when_expected(self, guard, budget):
        """tests/ is not flagged if contract already expects test files."""
        actual = ["src/auth/login.py", "tests/test_login.py", "tests/test_extra.py"]
        report = guard.evaluate(budget, actual)
        dir_violations = [
            v for v in report.violations
            if v.type == ViolationType.UNEXPECTED_DIRECTORY
            and v.file_path in ("tests", "test")
        ]
        assert len(dir_violations) == 0


# ===================================================================
# Excessive file count
# ===================================================================

class TestExcessiveFileCount:

    def test_exceeding_budget_flagged(self, guard):
        contract = _make_contract(
            expected_files=["src/auth/login.py"],
            risk_level=RiskLevel.LOW,
        )
        budget = guard.create_budget(contract)
        # Generate more files than budget allows.
        actual = [f"src/auth/file_{i}.py" for i in range(budget.max_file_count + 5)]
        report = guard.evaluate(budget, actual)
        excessive = [
            v for v in report.violations
            if v.type == ViolationType.EXCESSIVE_FILE_COUNT
        ]
        assert len(excessive) == 1
        assert "error" == excessive[0].severity

    def test_within_budget_not_flagged(self, guard, budget):
        report = guard.evaluate(budget, ["src/auth/login.py"])
        excessive = [
            v for v in report.violations
            if v.type == ViolationType.EXCESSIVE_FILE_COUNT
        ]
        assert len(excessive) == 0


# ===================================================================
# Outside affected area
# ===================================================================

class TestOutsideAffectedArea:

    def test_outside_area_flagged(self, guard, budget):
        actual = ["src/auth/login.py", "src/billing/payment.py"]
        report = guard.evaluate(budget, actual)
        area_violations = [
            v for v in report.violations
            if v.type == ViolationType.OUTSIDE_AFFECTED_AREA
        ]
        assert len(area_violations) >= 1

    def test_inside_area_not_flagged(self, guard, budget):
        actual = ["src/auth/login.py", "src/auth/utils.py"]
        report = guard.evaluate(budget, actual)
        area_violations = [
            v for v in report.violations
            if v.type == ViolationType.OUTSIDE_AFFECTED_AREA
        ]
        assert len(area_violations) == 0


# ===================================================================
# Unrelated modification
# ===================================================================

class TestUnrelatedModification:

    def test_unrelated_file_flagged(self, guard, budget):
        actual = ["src/auth/login.py", "src/billing/payment.py"]
        report = guard.evaluate(budget, actual)
        unrelated = [
            v for v in report.violations
            if v.type == ViolationType.UNRELATED_MODIFICATION
        ]
        assert len(unrelated) >= 1

    def test_test_file_not_flagged_as_unrelated(self, guard, budget):
        """Test files are not flagged as unrelated even if outside area."""
        actual = ["src/auth/login.py", "tests/test_other.py"]
        report = guard.evaluate(budget, actual)
        unrelated = [
            v for v in report.violations
            if v.type == ViolationType.UNRELATED_MODIFICATION
        ]
        assert len(unrelated) == 0


# ===================================================================
# Unjustified config change
# ===================================================================

class TestUnjustifiedConfigChange:

    def test_config_change_flagged(self, guard, budget):
        actual = ["src/auth/login.py", "config.yaml"]
        report = guard.evaluate(budget, actual)
        config_violations = [
            v for v in report.violations
            if v.type == ViolationType.UNJUSTIFIED_CONFIG_CHANGE
        ]
        assert len(config_violations) >= 1

    def test_config_change_justified(self, guard, contract):
        budget = guard.create_budget(contract, config_change_justified=True)
        actual = ["src/auth/login.py", "config.yaml"]
        report = guard.evaluate(budget, actual)
        config_violations = [
            v for v in report.violations
            if v.type == ViolationType.UNJUSTIFIED_CONFIG_CHANGE
        ]
        assert len(config_violations) == 0

    def test_various_config_files_detected(self, guard, budget):
        configs = [
            "pyproject.toml", "setup.cfg", ".env",
            "Dockerfile", "Makefile", "requirements.txt",
        ]
        for cfg in configs:
            actual = ["src/auth/login.py", cfg]
            report = guard.evaluate(budget, actual)
            config_violations = [
                v for v in report.violations
                if v.type == ViolationType.UNJUSTIFIED_CONFIG_CHANGE
            ]
            assert len(config_violations) >= 1, f"Failed for {cfg}"


# ===================================================================
# Test deletion
# ===================================================================

class TestTestDeletion:

    def test_deleted_test_flagged(self, guard, budget):
        report = guard.evaluate(
            budget,
            actual_files=["src/auth/login.py"],
            deleted_files=["tests/test_login.py"],
        )
        deletions = [
            v for v in report.violations
            if v.type == ViolationType.TEST_DELETION
        ]
        assert len(deletions) == 1
        assert deletions[0].severity == "critical"

    def test_deleted_non_test_not_flagged(self, guard, budget):
        report = guard.evaluate(
            budget,
            actual_files=["src/auth/login.py"],
            deleted_files=["src/auth/old_module.py"],
        )
        deletions = [
            v for v in report.violations
            if v.type == ViolationType.TEST_DELETION
        ]
        assert len(deletions) == 0

    def test_deleted_test_triggers_blocked(self, guard, budget):
        report = guard.evaluate(
            budget,
            actual_files=["src/auth/login.py"],
            deleted_files=["tests/test_login.py"],
        )
        assert report.status == ScopeStatus.BLOCKED


# ===================================================================
# Test integrity — trivial tests
# ===================================================================

class TestTrivialTests:

    def test_assert_true_only_detected(self, guard, budget):
        contents = {
            "tests/test_login.py": (
                "def test_locked():\n"
                "    assert True\n"
            ),
        }
        report = guard.evaluate(
            budget,
            actual_files=["src/auth/login.py", "tests/test_login.py"],
            file_contents=contents,
        )
        trivial = [
            v for v in report.violations
            if v.type == ViolationType.TRIVIAL_TEST
        ]
        assert len(trivial) >= 1

    def test_meaningful_test_not_flagged(self, guard, budget):
        contents = {
            "tests/test_login.py": (
                "def test_locked():\n"
                "    result = login(locked_user)\n"
                "    assert result.status == 403\n"
            ),
        }
        report = guard.evaluate(
            budget,
            actual_files=["src/auth/login.py", "tests/test_login.py"],
            file_contents=contents,
        )
        trivial = [
            v for v in report.violations
            if v.type == ViolationType.TRIVIAL_TEST
        ]
        assert len(trivial) == 0

    def test_pass_only_body_detected(self, guard, budget):
        contents = {
            "tests/test_login.py": (
                "def test_something():\n"
                "    pass\n"
            ),
        }
        report = guard.evaluate(
            budget,
            actual_files=["src/auth/login.py", "tests/test_login.py"],
            file_contents=contents,
        )
        trivial = [
            v for v in report.violations
            if v.type == ViolationType.TRIVIAL_TEST
        ]
        assert len(trivial) >= 1

    def test_trivial_only_no_strong_assertions(self, guard, budget):
        contents = {
            "tests/test_login.py": (
                "def test_a():\n"
                "    assert True\n"
                "\n"
                "def test_b():\n"
                "    assert True\n"
            ),
        }
        report = guard.evaluate(
            budget,
            actual_files=["src/auth/login.py", "tests/test_login.py"],
            file_contents=contents,
        )
        trivial = [
            v for v in report.violations
            if v.type == ViolationType.TRIVIAL_TEST
        ]
        assert len(trivial) >= 1


# ===================================================================
# Test integrity — weakened assertions
# ===================================================================

class TestWeakenedAssertions:

    def test_weakened_detected(self, guard, budget):
        original = {
            "tests/test_login.py": (
                "def test_login():\n"
                "    assert result.status == 200\n"
                "    assert user.name == 'Alice'\n"
                "    assert token is not None\n"
                "    assert 'session' in response\n"
            ),
        }
        new = {
            "tests/test_login.py": (
                "def test_login():\n"
                "    assert True\n"
            ),
        }
        report = guard.evaluate(
            budget,
            actual_files=["src/auth/login.py", "tests/test_login.py"],
            file_contents=new,
            original_contents=original,
        )
        weakened = [
            v for v in report.violations
            if v.type == ViolationType.WEAKENED_ASSERTION
        ]
        assert len(weakened) >= 1

    def test_adding_assertions_not_flagged(self, guard, budget):
        original = {
            "tests/test_login.py": (
                "def test_login():\n"
                "    assert result.status == 200\n"
            ),
        }
        new = {
            "tests/test_login.py": (
                "def test_login():\n"
                "    assert result.status == 200\n"
                "    assert user.name == 'Alice'\n"
                "    assert token is not None\n"
            ),
        }
        report = guard.evaluate(
            budget,
            actual_files=["src/auth/login.py", "tests/test_login.py"],
            file_contents=new,
            original_contents=original,
        )
        weakened = [
            v for v in report.violations
            if v.type == ViolationType.WEAKENED_ASSERTION
        ]
        assert len(weakened) == 0


# ===================================================================
# Test integrity — suspicious test modification
# ===================================================================

class TestSuspiciousTestModification:

    def test_unrelated_test_modification_flagged(self, guard):
        contract = _make_contract(
            affected_area="auth",
            expected_files=["src/auth/login.py"],
        )
        budget = guard.create_budget(contract)
        contents = {
            "tests/test_billing.py": (
                "def test_invoice():\n"
                "    assert invoice.total == 100\n"
            ),
        }
        report = guard.evaluate(
            budget,
            actual_files=["src/auth/login.py", "tests/test_billing.py"],
            file_contents=contents,
        )
        suspicious = [
            v for v in report.violations
            if v.type == ViolationType.SUSPICIOUS_TEST_MODIFICATION
        ]
        assert len(suspicious) >= 1

    def test_related_test_not_flagged(self, guard, budget):
        contents = {
            "tests/test_login.py": (
                "def test_locked():\n"
                "    assert result.status == 403\n"
            ),
        }
        report = guard.evaluate(
            budget,
            actual_files=["src/auth/login.py", "tests/test_login.py"],
            file_contents=contents,
        )
        suspicious = [
            v for v in report.violations
            if v.type == ViolationType.SUSPICIOUS_TEST_MODIFICATION
        ]
        assert len(suspicious) == 0


# ===================================================================
# Legitimate dependency changes
# ===================================================================

class TestLegitDependencyChanges:

    def test_init_py_in_expected_dir_is_dep(self, guard, budget):
        actual = [
            "src/auth/login.py",
            "src/auth/__init__.py",
            "tests/test_login.py",
        ]
        report = guard.evaluate(budget, actual)
        outside = [
            v for v in report.violations
            if v.type == ViolationType.OUTSIDE_AFFECTED_AREA
        ]
        assert len(outside) == 0

    def test_same_dir_file_is_dep(self, guard, budget):
        actual = [
            "src/auth/login.py",
            "src/auth/helpers.py",
            "tests/test_login.py",
        ]
        report = guard.evaluate(budget, actual)
        unrelated = [
            v for v in report.violations
            if v.type == ViolationType.UNRELATED_MODIFICATION
        ]
        assert len(unrelated) == 0

    def test_multi_file_change_not_auto_rejected(self, guard, budget):
        """A multi-file change within scope should not be blocked."""
        actual = [
            "src/auth/login.py",
            "src/auth/middleware.py",
            "tests/test_login.py",
        ]
        report = guard.evaluate(budget, actual)
        assert report.status == ScopeStatus.WITHIN_SCOPE


# ===================================================================
# Scope expansion
# ===================================================================

class TestScopeExpansion:

    def test_expansion_ratio_computed(self, guard, budget):
        actual = [
            "src/auth/login.py",
            "tests/test_login.py",
            "src/auth/a.py",
            "src/auth/b.py",
            "src/auth/c.py",
        ]
        report = guard.evaluate(budget, actual)
        assert report.scope_expansion > 0

    def test_no_expansion_when_within(self, guard, budget):
        report = guard.evaluate(budget, ["src/auth/login.py"])
        assert report.scope_expansion == 0.0

    def test_scope_expanded_status(self, guard):
        contract = _make_contract(
            expected_files=["src/auth/login.py"],
            risk_level=RiskLevel.LOW,
        )
        budget = guard.create_budget(contract)
        # Create many files to exceed 50% threshold
        actual = [f"src/auth/file_{i}.py" for i in range(4)]
        report = guard.evaluate(budget, actual)
        assert report.scope_expansion > 0.5
        assert report.status in (ScopeStatus.SCOPE_EXPANDED, ScopeStatus.BLOCKED)


# ===================================================================
# Scope expansion report
# ===================================================================

class TestScopeExpansionReport:

    def test_no_report_when_within_scope(self, guard, budget):
        report = guard.evaluate(budget, ["src/auth/login.py"])
        exp_report = guard.scope_expansion_report(report)
        assert exp_report is None

    def test_report_generated_on_expansion(self, guard):
        contract = _make_contract(
            expected_files=["src/auth/login.py"],
            risk_level=RiskLevel.LOW,
        )
        budget = guard.create_budget(contract)
        actual = [f"src/auth/f{i}.py" for i in range(5)]
        report = guard.evaluate(budget, actual)
        if report.status != ScopeStatus.WITHIN_SCOPE:
            exp_report = guard.scope_expansion_report(report)
            assert exp_report is not None
            assert exp_report.expansion_ratio > 0
            assert isinstance(exp_report.recommendation, str)

    def test_blocked_report_says_stop(self, guard, budget):
        report = guard.evaluate(
            budget,
            actual_files=["src/auth/login.py"],
            deleted_files=["tests/test_login.py"],
        )
        exp_report = guard.scope_expansion_report(report)
        assert exp_report is not None
        assert "STOP" in exp_report.recommendation

    def test_report_to_dict(self, guard, budget):
        report = guard.evaluate(
            budget,
            actual_files=["src/auth/login.py"],
            deleted_files=["tests/test_login.py"],
        )
        exp_report = guard.scope_expansion_report(report)
        d = exp_report.to_dict()
        assert "recommendation" in d
        assert "violations" in d
        assert "expansion_ratio" in d


# ===================================================================
# Report serialisation
# ===================================================================

class TestReportSerialisation:

    def test_report_to_dict_structure(self, guard, budget):
        report = guard.evaluate(budget, ["src/auth/login.py", "tests/test_login.py"])
        d = report.to_dict()
        assert "expected_files" in d
        assert "actual_files" in d
        assert "unexpected_files" in d
        assert "scope_expansion" in d
        assert "violations" in d
        assert d["status"] == "WITHIN_SCOPE"
        assert d["risk_level"] == "MEDIUM"

    def test_violation_to_dict(self):
        v = Violation(
            type=ViolationType.TEST_DELETION,
            file_path="tests/test_x.py",
            description="Test deleted",
            severity="critical",
        )
        d = v.to_dict()
        assert d["type"] == "test_deletion"
        assert d["severity"] == "critical"

    def test_report_properties(self, guard, budget):
        report = guard.evaluate(
            budget,
            actual_files=["src/auth/login.py"],
            deleted_files=["tests/test_login.py"],
        )
        assert report.has_violations is True
        assert len(report.critical_violations) >= 1

    def test_report_no_violations_properties(self, guard, budget):
        report = guard.evaluate(budget, ["src/auth/login.py"])
        assert report.has_violations is False
        assert len(report.critical_violations) == 0
        assert len(report.error_violations) == 0


# ===================================================================
# Helper function unit tests
# ===================================================================

class TestHelpers:

    def test_is_test_path_positive(self):
        assert _is_test_path("tests/test_login.py") is True
        assert _is_test_path("test/test_auth.py") is True
        assert _is_test_path("src/test_helper.py") is True
        assert _is_test_path("src/login_test.py") is True
        assert _is_test_path("specs/auth_spec.py") is True

    def test_is_test_path_negative(self):
        assert _is_test_path("src/auth/login.py") is False
        assert _is_test_path("src/contest.py") is False

    def test_is_config_path_positive(self):
        assert _is_config_path("config.yaml") is True
        assert _is_config_path("pyproject.toml") is True
        assert _is_config_path("Makefile") is True
        assert _is_config_path(".env") is True
        assert _is_config_path("requirements.txt") is True

    def test_is_config_path_negative(self):
        assert _is_config_path("src/auth/login.py") is False
        assert _is_config_path("src/config_module.py") is False

    def test_extract_directories(self):
        dirs = _extract_directories(["src/a.py", "tests/b.py", "root.py"])
        assert dirs == {"src", "tests"}

    def test_compute_scope_description(self):
        desc = _compute_scope_description(["src/a.py", "tests/b.py"])
        assert "src" in desc
        assert "tests" in desc

    def test_compute_scope_root_only(self):
        desc = _compute_scope_description(["file.py"])
        assert desc == "root-only"

    def test_file_in_area(self):
        assert _file_in_area("src/auth/login.py", "auth") is True
        assert _file_in_area("src/billing/pay.py", "auth") is False
        assert _file_in_area("src/auth/login.py", "") is True

    def test_file_in_area_multiword(self):
        assert _file_in_area(
            "src/auth/login.py", "authentication subsystem"
        ) is True  # "auth" keyword matches

    def test_count_strong_assertions(self):
        code = (
            "assert result.status == 200\n"
            "assert user.name == 'Alice'\n"
            "assert 'key' in data\n"
        )
        assert _count_strong_assertions(code) == 3

    def test_count_trivial_assertions(self):
        code = "assert True\nassert 1\nassert True, 'msg'\n"
        assert _count_trivial_assertions(code) == 3

    def test_has_trivial_test_functions(self):
        code = (
            "def test_foo():\n"
            "    assert True\n"
        )
        result = _has_trivial_test_functions(code)
        assert "test_foo" in result

    def test_has_trivial_test_pass(self):
        code = (
            "def test_bar():\n"
            "    pass\n"
        )
        result = _has_trivial_test_functions(code)
        assert "test_bar" in result

    def test_meaningful_test_not_trivial(self):
        code = (
            "def test_real():\n"
            "    result = do_thing()\n"
            "    assert result == 42\n"
        )
        result = _has_trivial_test_functions(code)
        assert len(result) == 0


# ===================================================================
# Status determination
# ===================================================================

class TestStatusDetermination:

    def test_critical_violation_blocks(self, guard, budget):
        """Any critical violation → BLOCKED."""
        report = guard.evaluate(
            budget,
            actual_files=["src/auth/login.py"],
            deleted_files=["tests/test_login.py"],
        )
        assert report.status == ScopeStatus.BLOCKED

    def test_error_violations_expand(self, guard):
        contract = _make_contract(
            expected_files=["src/auth/login.py"],
            risk_level=RiskLevel.LOW,
        )
        budget = guard.create_budget(contract)
        # Trivial test → error severity
        contents = {
            "tests/test_login.py": "def test_a():\n    assert True\n",
        }
        report = guard.evaluate(
            budget,
            actual_files=["src/auth/login.py", "tests/test_login.py"],
            file_contents=contents,
        )
        error_v = [v for v in report.violations if v.severity == "error"]
        if error_v:
            assert report.status in (
                ScopeStatus.SCOPE_EXPANDED, ScopeStatus.BLOCKED,
            )

    def test_warnings_only_within_scope(self, guard, budget):
        """Warnings alone don't trigger expansion if file count is small."""
        actual = ["src/auth/login.py", "tests/test_login.py", "src/auth/utils.py"]
        report = guard.evaluate(budget, actual)
        # utils.py is same dir as expected → dependency, no violation
        if all(v.severity == "warning" for v in report.violations):
            assert report.status == ScopeStatus.WITHIN_SCOPE


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:

    def test_no_expected_files(self, guard):
        contract = _make_contract(expected_files=[])
        budget = guard.create_budget(contract)
        report = guard.evaluate(budget, ["src/auth/login.py"])
        assert report.scope_expansion >= 0.0

    def test_single_expected_single_actual(self, guard):
        contract = _make_contract(expected_files=["src/auth/login.py"])
        budget = guard.create_budget(contract)
        report = guard.evaluate(budget, ["src/auth/login.py"])
        assert report.status == ScopeStatus.WITHIN_SCOPE

    def test_file_contents_none(self, guard, budget):
        """Passing no file_contents should not error."""
        report = guard.evaluate(budget, ["src/auth/login.py"])
        assert isinstance(report, ChangeBudgetReport)

    def test_deleted_files_none(self, guard, budget):
        report = guard.evaluate(budget, ["src/auth/login.py"], deleted_files=None)
        assert isinstance(report, ChangeBudgetReport)
