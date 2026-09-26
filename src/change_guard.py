"""Change Guard — Change Budget and Blast Radius Guard.

Before the agent modifies code it establishes an **expected change scope**
from the task contract, repository context, impact map, and risk level.

After modification the guard compares *actual* repository changes against
the expected scope and produces a ``ChangeBudgetReport`` that flags:

*  unexpected files / directories
*  excessive file count
*  unrelated modifications
*  suspicious test modifications (deletions, weakened assertions, trivial
   ``assert True`` stubs, unrelated test edits)
*  modifications outside the affected area
*  unjustified configuration changes
*  deletion of existing tests

When scope expands substantially the guard stops normal execution and
emits a concise scope-expansion report for the orchestrator to
re-evaluate the plan **autonomously** (no human confirmation required).
"""

from __future__ import annotations

import enum
import os
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from src.contract_engine import RiskLevel, TaskContract


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ScopeStatus(enum.Enum):
    """Result of the scope comparison."""
    WITHIN_SCOPE = "WITHIN_SCOPE"
    SCOPE_EXPANDED = "SCOPE_EXPANDED"
    BLOCKED = "BLOCKED"


class ViolationType(enum.Enum):
    """Categories of scope violations detected by the guard."""
    UNEXPECTED_FILE = "unexpected_file"
    UNEXPECTED_DIRECTORY = "unexpected_directory"
    EXCESSIVE_FILE_COUNT = "excessive_file_count"
    UNRELATED_MODIFICATION = "unrelated_modification"
    SUSPICIOUS_TEST_MODIFICATION = "suspicious_test_modification"
    OUTSIDE_AFFECTED_AREA = "outside_affected_area"
    UNJUSTIFIED_CONFIG_CHANGE = "unjustified_config_change"
    TEST_DELETION = "test_deletion"
    WEAKENED_ASSERTION = "weakened_assertion"
    TRIVIAL_TEST = "trivial_test"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    """A single scope violation detected by the guard."""
    type: ViolationType
    file_path: str
    description: str
    severity: str = "warning"  # "warning" | "error" | "critical"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "file_path": self.file_path,
            "description": self.description,
            "severity": self.severity,
        }


@dataclass
class ChangeBudget:
    """Pre-modification expected change scope."""
    expected_files: list[str]
    expected_scope: str
    max_file_count: int
    affected_area: str
    risk_level: RiskLevel
    allowed_directories: list[str] = field(default_factory=list)
    config_change_justified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_files": list(self.expected_files),
            "expected_scope": self.expected_scope,
            "max_file_count": self.max_file_count,
            "affected_area": self.affected_area,
            "risk_level": self.risk_level.value,
            "allowed_directories": list(self.allowed_directories),
            "config_change_justified": self.config_change_justified,
        }


@dataclass
class ChangeBudgetReport:
    """Post-modification comparison of actual vs. expected scope."""
    expected_files: list[str]
    actual_files: list[str]
    unexpected_files: list[str]
    expected_scope: str
    actual_scope: str
    scope_expansion: float  # 0.0 = no expansion, 1.0 = doubled, etc.
    risk_level: RiskLevel
    violations: list[Violation]
    status: ScopeStatus

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_files": list(self.expected_files),
            "actual_files": list(self.actual_files),
            "unexpected_files": list(self.unexpected_files),
            "expected_scope": self.expected_scope,
            "actual_scope": self.actual_scope,
            "scope_expansion": self.scope_expansion,
            "risk_level": self.risk_level.value,
            "violations": [v.to_dict() for v in self.violations],
            "status": self.status.value,
        }

    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0

    @property
    def critical_violations(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "critical"]

    @property
    def error_violations(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "error"]


@dataclass
class ScopeExpansionReport:
    """Concise report emitted when scope expands substantially."""
    original_scope: str
    expanded_scope: str
    expansion_ratio: float
    new_files: list[str]
    violations: list[Violation]
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_scope": self.original_scope,
            "expanded_scope": self.expanded_scope,
            "expansion_ratio": self.expansion_ratio,
            "new_files": list(self.new_files),
            "violations": [v.to_dict() for v in self.violations],
            "recommendation": self.recommendation,
        }


# ---------------------------------------------------------------------------
# Well-known patterns
# ---------------------------------------------------------------------------

_TEST_PATH_RE = re.compile(
    r"(^|/)tests?/|test_[^/]+\.py$|_test\.py$|spec[s]?/", re.IGNORECASE,
)

_CONFIG_FILE_RE = re.compile(
    r"(^|/)(config\.\w+|\..*rc|setup\.cfg|pyproject\.toml|"
    r"tsconfig\.json|package\.json|Makefile|Dockerfile|"
    r"\.env(\.\w+)?|tox\.ini|\.flake8|\.pylintrc|"
    r"requirements.*\.txt|Pipfile|poetry\.lock|Cargo\.toml|go\.mod|"
    r"\.gitignore|\.dockerignore)$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Test integrity heuristics (content-level)
# ---------------------------------------------------------------------------

# Patterns that suggest a test has been weakened or made trivial.
_TRIVIAL_ASSERT_RE = re.compile(
    r"^\s*assert\s+True\s*(,.*)?$"
    r"|^\s*assert\s+1\s*(,.*)?$"
    r"|^\s*self\.assertTrue\s*\(\s*True\s*\)"
    r"|^\s*assert\s+not\s+False\s*(,.*)?$",
    re.MULTILINE,
)

# Strong assertion patterns — if these disappear from a test, it may
# have been weakened.
_STRONG_ASSERT_RE = re.compile(
    r"assert\s+\S+.*[=!<>]"
    r"|assertEqual\s*\("
    r"|assertRaises\s*\("
    r"|assertIn\s*\("
    r"|assertIs\s*\("
    r"|assertAlmostEqual\s*\("
    r"|pytest\.raises\s*\("
    r"|assert\s+\S+\s+(in|not\s+in|is|is\s+not)\s+",
    re.MULTILINE,
)

# Detect entire test functions that are trivial stubs.
_TRIVIAL_TEST_FUNC_RE = re.compile(
    r"def\s+(test_\w+)\s*\([^)]*\)\s*:\s*\n"
    r"(?:\s*(?:#[^\n]*)?\n)*"  # optional comments/blanks
    r"\s*(?:assert\s+True|pass|\.\.\.)\s*$",
    re.MULTILINE,
)


def _is_test_path(path: str) -> bool:
    """Return ``True`` if *path* looks like a test file."""
    return bool(_TEST_PATH_RE.search(path))


def _is_config_path(path: str) -> bool:
    """Return ``True`` if *path* looks like a configuration file."""
    return bool(_CONFIG_FILE_RE.search(path))


def _count_strong_assertions(content: str) -> int:
    """Count meaningful assertion statements in content."""
    return len(_STRONG_ASSERT_RE.findall(content))


def _count_trivial_assertions(content: str) -> int:
    """Count trivially-true assertions in content."""
    return len(_TRIVIAL_ASSERT_RE.findall(content))


def _has_trivial_test_functions(content: str) -> list[str]:
    """Return names of test functions whose body is just ``assert True`` / ``pass``."""
    return _TRIVIAL_TEST_FUNC_RE.findall(content)


# ---------------------------------------------------------------------------
# Directory / scope helpers
# ---------------------------------------------------------------------------

def _extract_directories(file_paths: list[str]) -> set[str]:
    """Extract the set of first-level directories from a list of file paths."""
    dirs: set[str] = set()
    for fp in file_paths:
        parts = pathlib.PurePosixPath(fp).parts
        if len(parts) > 1:
            dirs.add(parts[0])
    return dirs


def _compute_scope_description(file_paths: list[str]) -> str:
    """Build a concise human-readable scope string."""
    dirs = sorted(_extract_directories(file_paths))
    if not dirs:
        return "root-only"
    return ", ".join(dirs)


def _file_in_area(file_path: str, affected_area: str) -> bool:
    """Check whether *file_path* is within the *affected_area*.

    Uses bidirectional substring matching so that both
    ``"auth" in "authentication"`` and ``"auth" in "src/auth/login.py"``
    return ``True``.
    """
    if not affected_area:
        return True
    normalised = file_path.replace("\\", "/").lower()
    area = affected_area.strip().lower().replace("\\", "/")
    # Split the affected area description into keywords.
    area_keywords = re.split(r"[\s/\\,]+", area)
    # Split the file path into components for bidirectional matching.
    path_parts = re.split(r"[/\\_.]", normalised)
    for kw in area_keywords:
        if len(kw) < 3:
            continue
        # Direct substring check: keyword in path.
        if kw in normalised:
            return True
        # Bidirectional: any path component is a prefix/substring of keyword
        # or keyword is a prefix/substring of a path component.
        for part in path_parts:
            if len(part) < 3:
                continue
            if part in kw or kw in part:
                return True
    # Also match exact directory prefix.
    if normalised.startswith(area + "/") or normalised == area:
        return True
    return False


# ---------------------------------------------------------------------------
# Risk-level multipliers for file count budgets
# ---------------------------------------------------------------------------

_RISK_FILE_MULTIPLIER: dict[RiskLevel, float] = {
    RiskLevel.LOW: 1.5,
    RiskLevel.MEDIUM: 2.0,
    RiskLevel.HIGH: 2.5,
    RiskLevel.CRITICAL: 3.0,
}

# Expansion threshold before triggering re-evaluation.
_EXPANSION_THRESHOLD = 0.5  # 50 % more files than expected


# ---------------------------------------------------------------------------
# ChangeGuard — public API
# ---------------------------------------------------------------------------

class ChangeGuard:
    """Validates agent modifications against the expected change scope.

    Typical lifecycle::

        guard = ChangeGuard()
        budget = guard.create_budget(contract)        # before modification
        report = guard.evaluate(budget, actual_files, file_contents)
        if report.status == ScopeStatus.BLOCKED:
            ...  # abort or escalate
    """

    # ------------------------------------------------------------------
    # Budget creation
    # ------------------------------------------------------------------

    def create_budget(
        self,
        contract: TaskContract,
        *,
        extra_allowed_files: list[str] | None = None,
        config_change_justified: bool = False,
    ) -> ChangeBudget:
        """Create a change budget from a task contract.

        The budget grants the agent a maximum file count derived from the
        contract's expected files, scaled by the risk multiplier.

        Parameters
        ----------
        contract:
            The task contract whose ``expected_files``, ``affected_area``,
            and ``risk_level`` inform the budget.
        extra_allowed_files:
            Additional file paths to whitelist (e.g. from the
            dependency-impact map).
        config_change_justified:
            Set ``True`` when the task explicitly calls for config edits.
        """
        expected = list(contract.expected_files)
        if extra_allowed_files:
            for f in extra_allowed_files:
                if f not in expected:
                    expected.append(f)

        multiplier = _RISK_FILE_MULTIPLIER.get(contract.risk_level, 2.0)
        max_files = max(int(len(expected) * multiplier), len(expected) + 2)

        allowed_dirs = sorted(_extract_directories(expected))

        return ChangeBudget(
            expected_files=expected,
            expected_scope=_compute_scope_description(expected),
            max_file_count=max_files,
            affected_area=contract.affected_area,
            risk_level=contract.risk_level,
            allowed_directories=allowed_dirs,
            config_change_justified=config_change_justified,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        budget: ChangeBudget,
        actual_files: list[str],
        file_contents: dict[str, str] | None = None,
        original_contents: dict[str, str] | None = None,
        deleted_files: list[str] | None = None,
    ) -> ChangeBudgetReport:
        """Compare actual changes against the budget.

        Parameters
        ----------
        budget:
            The pre-modification change budget.
        actual_files:
            List of file paths that were actually created or modified.
        file_contents:
            Optional mapping of ``path → new content`` for content-level
            heuristics (test integrity checks).
        original_contents:
            Optional mapping of ``path → original content`` so we can
            diff test strength.
        deleted_files:
            Paths of files that were deleted during the task.

        Returns
        -------
        ChangeBudgetReport
        """
        file_contents = file_contents or {}
        original_contents = original_contents or {}
        deleted_files = deleted_files or []

        violations: list[Violation] = []

        # -- Unexpected files --
        expected_set = set(budget.expected_files)
        unexpected = [f for f in actual_files if f not in expected_set]

        # -- Unexpected directories --
        expected_dirs = set(budget.allowed_directories)
        actual_dirs = _extract_directories(actual_files)
        new_dirs = actual_dirs - expected_dirs

        for d in sorted(new_dirs):
            # Don't flag test directories as unexpected if the contract
            # already expects test files.
            if d in ("tests", "test") and any(
                _is_test_path(ef) for ef in budget.expected_files
            ):
                continue
            violations.append(Violation(
                type=ViolationType.UNEXPECTED_DIRECTORY,
                file_path=d,
                description=f"Changes touch unexpected directory: {d}/",
                severity="warning",
            ))

        # -- Excessive file count --
        if len(actual_files) > budget.max_file_count:
            violations.append(Violation(
                type=ViolationType.EXCESSIVE_FILE_COUNT,
                file_path="",
                description=(
                    f"Modified {len(actual_files)} files, budget allows "
                    f"{budget.max_file_count}."
                ),
                severity="error",
            ))

        # -- Per-file checks --
        for fp in actual_files:
            if fp in expected_set:
                continue

            # Outside affected area?
            if not _file_in_area(fp, budget.affected_area):
                # Check if it's a legitimate dependency (test for
                # expected files, __init__.py, etc.).
                if not self._is_dependency_file(fp, budget):
                    violations.append(Violation(
                        type=ViolationType.OUTSIDE_AFFECTED_AREA,
                        file_path=fp,
                        description=(
                            f"File '{fp}' is outside the affected area "
                            f"'{budget.affected_area}'."
                        ),
                        severity="warning",
                    ))

            # Unjustified config change?
            if _is_config_path(fp) and not budget.config_change_justified:
                violations.append(Violation(
                    type=ViolationType.UNJUSTIFIED_CONFIG_CHANGE,
                    file_path=fp,
                    description=(
                        f"Configuration file '{fp}' was modified but the "
                        f"task does not justify config changes."
                    ),
                    severity="warning",
                ))

            # Unrelated modification?
            if (
                not _file_in_area(fp, budget.affected_area)
                and not _is_test_path(fp)
                and not _is_config_path(fp)
                and not self._is_dependency_file(fp, budget)
            ):
                violations.append(Violation(
                    type=ViolationType.UNRELATED_MODIFICATION,
                    file_path=fp,
                    description=f"File '{fp}' seems unrelated to the task.",
                    severity="warning",
                ))

        # -- Deleted files checks --
        for fp in deleted_files:
            if _is_test_path(fp):
                violations.append(Violation(
                    type=ViolationType.TEST_DELETION,
                    file_path=fp,
                    description=f"Test file '{fp}' was deleted.",
                    severity="critical",
                ))

        # -- Test integrity checks (content-level) --
        violations.extend(
            self._check_test_integrity(
                actual_files, file_contents, original_contents, budget,
            )
        )

        # -- Compute scope expansion --
        actual_scope = _compute_scope_description(actual_files)
        expansion = self._compute_expansion(budget, actual_files)

        # -- Determine status --
        status = self._determine_status(violations, expansion)

        return ChangeBudgetReport(
            expected_files=list(budget.expected_files),
            actual_files=list(actual_files),
            unexpected_files=unexpected,
            expected_scope=budget.expected_scope,
            actual_scope=actual_scope,
            scope_expansion=round(expansion, 3),
            risk_level=budget.risk_level,
            violations=violations,
            status=status,
        )

    # ------------------------------------------------------------------
    # Scope expansion report (for orchestrator re-evaluation)
    # ------------------------------------------------------------------

    def scope_expansion_report(
        self, report: ChangeBudgetReport,
    ) -> Optional[ScopeExpansionReport]:
        """Generate a concise scope-expansion report if warranted.

        Returns ``None`` if the scope did not expand substantially.
        The orchestrator should use this to decide whether to continue.
        """
        if report.status == ScopeStatus.WITHIN_SCOPE:
            return None

        critical = report.critical_violations
        errors = report.error_violations

        if report.status == ScopeStatus.BLOCKED:
            recommendation = (
                "STOP: critical scope violations detected. "
                "Re-evaluate the plan before proceeding."
            )
        elif len(critical) > 0:
            recommendation = (
                "STOP: critical violations require re-evaluation. "
                "Review the expanded scope and justify each new file."
            )
        elif len(errors) > 0:
            recommendation = (
                "CAUTION: scope expanded beyond budget. "
                "Verify that all new files are task-relevant."
            )
        else:
            recommendation = (
                "MINOR: scope expanded slightly. "
                "Verify unexpected files are legitimate dependencies."
            )

        return ScopeExpansionReport(
            original_scope=report.expected_scope,
            expanded_scope=report.actual_scope,
            expansion_ratio=report.scope_expansion,
            new_files=report.unexpected_files,
            violations=report.violations,
            recommendation=recommendation,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_dependency_file(file_path: str, budget: ChangeBudget) -> bool:
        """Heuristic: is this file a legitimate dependency of expected files?

        Recognised patterns:
        - ``__init__.py`` in a directory containing expected files
        - Test files when the contract expects source changes
        - Files sharing a directory with expected files
        """
        fp = pathlib.PurePosixPath(file_path)
        name = fp.name

        # __init__.py in a directory of expected files
        if name == "__init__.py":
            parent = str(fp.parent)
            for ef in budget.expected_files:
                ef_parent = str(pathlib.PurePosixPath(ef).parent)
                if ef_parent == parent or ef_parent.startswith(parent + "/"):
                    return True

        # Test file for an expected source file
        if _is_test_path(file_path):
            base = name.replace("test_", "").replace("_test", "")
            for ef in budget.expected_files:
                ef_name = pathlib.PurePosixPath(ef).name
                if base == ef_name:
                    return True

        # Same directory as an expected file
        fp_dir = str(fp.parent)
        for ef in budget.expected_files:
            if str(pathlib.PurePosixPath(ef).parent) == fp_dir:
                return True

        return False

    def _check_test_integrity(
        self,
        actual_files: list[str],
        file_contents: dict[str, str],
        original_contents: dict[str, str],
        budget: ChangeBudget,
    ) -> list[Violation]:
        """Run content-level heuristics on test files."""
        violations: list[Violation] = []

        for fp in actual_files:
            if not _is_test_path(fp):
                continue

            new_content = file_contents.get(fp)
            if new_content is None:
                continue

            # --- Trivial test functions ---
            trivial_funcs = _has_trivial_test_functions(new_content)
            for func_name in trivial_funcs:
                violations.append(Violation(
                    type=ViolationType.TRIVIAL_TEST,
                    file_path=fp,
                    description=(
                        f"Test function '{func_name}' has a trivial body "
                        f"(assert True / pass)."
                    ),
                    severity="error",
                ))

            # --- Excessive trivial assertions ---
            trivial_count = _count_trivial_assertions(new_content)
            strong_count = _count_strong_assertions(new_content)
            if trivial_count > 0 and strong_count == 0:
                violations.append(Violation(
                    type=ViolationType.TRIVIAL_TEST,
                    file_path=fp,
                    description=(
                        f"Test file contains {trivial_count} trivial "
                        f"assertion(s) and no meaningful assertions."
                    ),
                    severity="error",
                ))

            # --- Weakened assertions (comparison with original) ---
            old_content = original_contents.get(fp)
            if old_content is not None:
                old_strong = _count_strong_assertions(old_content)
                new_strong = _count_strong_assertions(new_content)
                if old_strong > 0 and new_strong < old_strong:
                    loss_pct = ((old_strong - new_strong) / old_strong) * 100
                    if loss_pct >= 30:
                        violations.append(Violation(
                            type=ViolationType.WEAKENED_ASSERTION,
                            file_path=fp,
                            description=(
                                f"Strong assertions reduced from {old_strong} "
                                f"to {new_strong} ({loss_pct:.0f}% loss)."
                            ),
                            severity="error",
                        ))

            # --- Suspicious test modification without task relationship ---
            if fp not in set(budget.expected_files):
                if not _file_in_area(fp, budget.affected_area):
                    if not self._is_dependency_file(fp, budget):
                        violations.append(Violation(
                            type=ViolationType.SUSPICIOUS_TEST_MODIFICATION,
                            file_path=fp,
                            description=(
                                f"Test file '{fp}' was modified but is not "
                                f"in the affected area or expected files."
                            ),
                            severity="warning",
                        ))

        return violations

    @staticmethod
    def _compute_expansion(
        budget: ChangeBudget, actual_files: list[str],
    ) -> float:
        """Compute the scope expansion ratio.

        Returns 0.0 when actual ≤ expected, and a positive float
        representing the proportional increase when actual exceeds expected.
        """
        expected_count = max(len(budget.expected_files), 1)
        actual_count = len(actual_files)
        if actual_count <= expected_count:
            return 0.0
        return (actual_count - expected_count) / expected_count

    @staticmethod
    def _determine_status(
        violations: list[Violation], expansion: float,
    ) -> ScopeStatus:
        """Decide the overall scope status.

        BLOCKED if there are any critical violations.
        SCOPE_EXPANDED if expansion exceeds threshold or there are errors.
        WITHIN_SCOPE otherwise.
        """
        if any(v.severity == "critical" for v in violations):
            return ScopeStatus.BLOCKED
        if expansion > _EXPANSION_THRESHOLD:
            return ScopeStatus.SCOPE_EXPANDED
        if any(v.severity == "error" for v in violations):
            return ScopeStatus.SCOPE_EXPANDED
        return ScopeStatus.WITHIN_SCOPE
