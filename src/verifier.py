"""Verifier — Multi-level task verification with structured evidence.

The verifier does NOT merely run tests.  It verifies the *task* by
executing a layered verification pipeline:

    LEVEL 1  Basic validation      (syntax, imports, project health)
    LEVEL 2  Existing tests        (run the project's relevant test suite)
    LEVEL 3  Targeted verification (run tests relevant to the issue)
    LEVEL 4  Contract verification (verify every proof obligation)
    LEVEL 5  Regression            (check behaviour outside changed code)
    LEVEL 6  Falsification         (clean interface for src/falsifier.py)

Every check produces **structured evidence** with deterministic file
fingerprints so that stale results can be detected and invalidated.

Language discipline:
    ✓  "verified by executed test"
    ✗  "proven correct"
"""

from __future__ import annotations

import enum
import hashlib
import os
import pathlib
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.contract_engine import (
    ObligationStatus,
    ObligationType,
    ProofObligation,
    TaskContract,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class VerificationStatus(enum.Enum):
    """Final verification outcome."""
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"
    STALE = "STALE"


class VerificationLevel(enum.Enum):
    """Ordered verification layers."""
    BASIC_VALIDATION = 1
    EXISTING_TESTS = 2
    TARGETED_VERIFICATION = 3
    CONTRACT_VERIFICATION = 4
    REGRESSION = 5
    FALSIFICATION = 6


class CheckStatus(enum.Enum):
    """Outcome of a single verification check."""
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FileFingerprint:
    """Deterministic fingerprint for a single file."""
    path: str
    sha256: str
    size_bytes: int
    mtime: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "mtime": self.mtime,
        }

    @classmethod
    def from_file(cls, path: str, repo_root: str = "") -> "FileFingerprint":
        """Compute a fingerprint from a real file on disk."""
        full = os.path.join(repo_root, path) if repo_root else path
        stat = os.stat(full)
        hasher = hashlib.sha256()
        with open(full, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                hasher.update(chunk)
        return cls(
            path=path,
            sha256=hasher.hexdigest(),
            size_bytes=stat.st_size,
            mtime=stat.st_mtime,
        )

    @classmethod
    def from_content(cls, path: str, content: str) -> "FileFingerprint":
        """Compute a fingerprint from in-memory content (testing)."""
        encoded = content.encode("utf-8")
        return cls(
            path=path,
            sha256=hashlib.sha256(encoded).hexdigest(),
            size_bytes=len(encoded),
            mtime=0.0,
        )


@dataclass
class VerificationCheck:
    """A single verification check with structured evidence."""
    check_id: str
    level: VerificationLevel
    description: str
    status: CheckStatus
    evidence: str
    obligation_id: str | None = None
    command: str | None = None
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "check_id": self.check_id,
            "level": self.level.value,
            "description": self.description,
            "status": self.status.value,
            "evidence": self.evidence,
        }
        if self.obligation_id:
            d["obligation_id"] = self.obligation_id
        if self.command:
            d["command"] = self.command
        if self.stdout:
            d["stdout"] = self.stdout
        if self.stderr:
            d["stderr"] = self.stderr
        if self.exit_code is not None:
            d["exit_code"] = self.exit_code
        d["duration"] = self.duration
        return d


@dataclass
class VerificationResult:
    """Complete verification result for a task."""
    passed: bool
    status: VerificationStatus
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration: float
    checks: list[VerificationCheck]
    failure_summary: str
    evidence: list[str]
    changed_files: list[str]
    confidence: float
    fingerprints: list[FileFingerprint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "status": self.status.value,
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration": self.duration,
            "checks": [c.to_dict() for c in self.checks],
            "failure_summary": self.failure_summary,
            "evidence": list(self.evidence),
            "changed_files": list(self.changed_files),
            "confidence": self.confidence,
            "fingerprints": [fp.to_dict() for fp in self.fingerprints],
        }

    def is_stale(self, current_fingerprints: list[FileFingerprint]) -> bool:
        """Return ``True`` if any relevant file has changed since verification.

        Compares recorded fingerprints against *current_fingerprints*.
        A mismatch in SHA-256 for any file means the evidence is stale.
        """
        current_map = {fp.path: fp.sha256 for fp in current_fingerprints}
        for fp in self.fingerprints:
            current_hash = current_map.get(fp.path)
            if current_hash is None:
                # File was deleted — evidence is stale.
                return True
            if current_hash != fp.sha256:
                return True
        return False

    def mark_stale(self) -> None:
        """Transition this result to STALE.

        A stale result cannot be used as current proof.
        """
        self.status = VerificationStatus.STALE
        self.passed = False
        self.failure_summary = (
            "Evidence is stale: source files changed after verification."
        )


# ---------------------------------------------------------------------------
# Command runner protocol
# ---------------------------------------------------------------------------

class CommandResult:
    """Minimal protocol for shell command results."""

    def __init__(
        self,
        *,
        success: bool,
        stdout: str = "",
        stderr: str = "",
        exit_code: int = 0,
        duration: float = 0.0,
        timed_out: bool = False,
    ) -> None:
        self.success = success
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.duration = duration
        self.timed_out = timed_out


# Type alias for the command runner callable.
# Signature: (command: str, timeout: int) -> CommandResult
CommandRunner = Callable[[str, int], CommandResult]


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------

def compute_fingerprints(
    file_paths: list[str],
    *,
    repo_root: str = "",
    file_contents: dict[str, str] | None = None,
) -> list[FileFingerprint]:
    """Compute fingerprints for a list of files.

    If *file_contents* is provided (testing), fingerprints are computed
    from in-memory content.  Otherwise the files are read from disk.
    """
    fingerprints: list[FileFingerprint] = []
    for path in file_paths:
        if file_contents and path in file_contents:
            fingerprints.append(
                FileFingerprint.from_content(path, file_contents[path])
            )
        else:
            try:
                fingerprints.append(
                    FileFingerprint.from_file(path, repo_root)
                )
            except (OSError, IOError):
                # File not accessible — skip.
                pass
    return fingerprints


# ---------------------------------------------------------------------------
# Individual level implementations
# ---------------------------------------------------------------------------

def _run_basic_validation(
    changed_files: list[str],
    runner: CommandRunner,
    timeout: int,
) -> list[VerificationCheck]:
    """LEVEL 1 — Syntax, imports, and basic project health."""
    checks: list[VerificationCheck] = []
    check_num = 0

    # Find Python files among changed files.
    py_files = [f for f in changed_files if f.endswith(".py")]

    if not py_files:
        checks.append(VerificationCheck(
            check_id="L1-SKIP",
            level=VerificationLevel.BASIC_VALIDATION,
            description="No Python files to validate",
            status=CheckStatus.SKIP,
            evidence="No .py files in changed file list",
        ))
        return checks

    # Syntax check via py_compile.
    for pf in py_files:
        check_num += 1
        cmd = f"python -m py_compile {pf}"
        result = runner(cmd, timeout)
        if result.timed_out:
            checks.append(VerificationCheck(
                check_id=f"L1-SYNTAX-{check_num}",
                level=VerificationLevel.BASIC_VALIDATION,
                description=f"Syntax check: {pf}",
                status=CheckStatus.ERROR,
                evidence=f"Syntax check timed out after {timeout}s",
                command=cmd,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                duration=result.duration,
            ))
        elif result.success:
            checks.append(VerificationCheck(
                check_id=f"L1-SYNTAX-{check_num}",
                level=VerificationLevel.BASIC_VALIDATION,
                description=f"Syntax check: {pf}",
                status=CheckStatus.PASS,
                evidence=f"Verified by executed syntax check (py_compile): {pf}",
                command=cmd,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                duration=result.duration,
            ))
        else:
            checks.append(VerificationCheck(
                check_id=f"L1-SYNTAX-{check_num}",
                level=VerificationLevel.BASIC_VALIDATION,
                description=f"Syntax check: {pf}",
                status=CheckStatus.FAIL,
                evidence=f"Syntax error in {pf}: {result.stderr.strip()}",
                command=cmd,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                duration=result.duration,
            ))

    # Import check for all changed Python files.
    for pf in py_files:
        check_num += 1
        module = pf.replace("/", ".").replace("\\", ".").removesuffix(".py")
        cmd = f"python -c \"import {module}\""
        result = runner(cmd, timeout)
        if result.timed_out:
            checks.append(VerificationCheck(
                check_id=f"L1-IMPORT-{check_num}",
                level=VerificationLevel.BASIC_VALIDATION,
                description=f"Import check: {module}",
                status=CheckStatus.ERROR,
                evidence=f"Import check timed out after {timeout}s",
                command=cmd,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                duration=result.duration,
            ))
        elif result.success:
            checks.append(VerificationCheck(
                check_id=f"L1-IMPORT-{check_num}",
                level=VerificationLevel.BASIC_VALIDATION,
                description=f"Import check: {module}",
                status=CheckStatus.PASS,
                evidence=f"Verified by executed import: {module}",
                command=cmd,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                duration=result.duration,
            ))
        else:
            checks.append(VerificationCheck(
                check_id=f"L1-IMPORT-{check_num}",
                level=VerificationLevel.BASIC_VALIDATION,
                description=f"Import check: {module}",
                status=CheckStatus.FAIL,
                evidence=f"Import failed for {module}: {result.stderr.strip()}",
                command=cmd,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                duration=result.duration,
            ))

    return checks


def _run_existing_tests(
    runner: CommandRunner,
    timeout: int,
    test_command: str = "pytest tests/ -x -q",
) -> list[VerificationCheck]:
    """LEVEL 2 — Run the project's existing test suite."""
    result = runner(test_command, timeout)

    if result.timed_out:
        return [VerificationCheck(
            check_id="L2-SUITE",
            level=VerificationLevel.EXISTING_TESTS,
            description="Run existing test suite",
            status=CheckStatus.ERROR,
            evidence=f"Test suite timed out after {timeout}s",
            command=test_command,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            duration=result.duration,
        )]

    if result.success:
        return [VerificationCheck(
            check_id="L2-SUITE",
            level=VerificationLevel.EXISTING_TESTS,
            description="Run existing test suite",
            status=CheckStatus.PASS,
            evidence=(
                f"Verified by executed test suite: {test_command}\n"
                f"Exit code 0.  Output:\n{result.stdout[:500]}"
            ),
            command=test_command,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            duration=result.duration,
        )]

    # Exit code 5 = no tests collected (e.g. empty test dir, no test files).
    # This is NOT a test failure — it means the project has no test suite.
    # Return SKIP so the verdict falls through to higher levels or UNKNOWN,
    # rather than falsely reporting FAILED.
    if result.exit_code == 5:
        return [VerificationCheck(
            check_id="L2-SUITE",
            level=VerificationLevel.EXISTING_TESTS,
            description="Run existing test suite",
            status=CheckStatus.SKIP,
            evidence=(
                f"No tests collected (exit code 5).  "
                f"The project may not have an existing test suite.\n"
                f"{result.stdout[:300]}"
            ),
            command=test_command,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            duration=result.duration,
        )]

    return [VerificationCheck(
        check_id="L2-SUITE",
        level=VerificationLevel.EXISTING_TESTS,
        description="Run existing test suite",
        status=CheckStatus.FAIL,
        evidence=(
            f"Test suite failed (exit code {result.exit_code}):\n"
            f"{result.stdout[:500]}\n{result.stderr[:500]}"
        ),
        command=test_command,
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.exit_code,
        duration=result.duration,
    )]


def _run_targeted_tests(
    changed_files: list[str],
    runner: CommandRunner,
    timeout: int,
) -> list[VerificationCheck]:
    """LEVEL 3 — Run tests relevant to the changed files."""
    checks: list[VerificationCheck] = []

    # Identify test files that correspond to changed source files.
    test_targets: list[str] = []
    for f in changed_files:
        basename = pathlib.PurePosixPath(f).stem
        # Source file → test file mapping heuristics.
        candidates = [
            f"tests/test_{basename}.py",
            f"tests/{basename}_test.py",
        ]
        # If the file is already a test file, include it directly.
        if re.search(r"(^|/)test_|_test\.py$", f):
            test_targets.append(f)
        else:
            test_targets.extend(candidates)

    if not test_targets:
        checks.append(VerificationCheck(
            check_id="L3-SKIP",
            level=VerificationLevel.TARGETED_VERIFICATION,
            description="No targeted tests identified",
            status=CheckStatus.SKIP,
            evidence="No test files could be mapped from changed files",
        ))
        return checks

    # Deduplicate.
    test_targets = sorted(set(test_targets))

    for i, target in enumerate(test_targets, 1):
        cmd = f"pytest {target} -x -q"
        result = runner(cmd, timeout)

        if result.timed_out:
            checks.append(VerificationCheck(
                check_id=f"L3-TARGET-{i}",
                level=VerificationLevel.TARGETED_VERIFICATION,
                description=f"Targeted test: {target}",
                status=CheckStatus.ERROR,
                evidence=f"Targeted test timed out after {timeout}s",
                command=cmd,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                duration=result.duration,
            ))
        elif result.success:
            checks.append(VerificationCheck(
                check_id=f"L3-TARGET-{i}",
                level=VerificationLevel.TARGETED_VERIFICATION,
                description=f"Targeted test: {target}",
                status=CheckStatus.PASS,
                evidence=(
                    f"Verified by executed targeted test: {target}\n"
                    f"{result.stdout[:300]}"
                ),
                command=cmd,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                duration=result.duration,
            ))
        else:
            # exit code 5 = no tests collected (file doesn't exist) → SKIP.
            if result.exit_code == 5:
                checks.append(VerificationCheck(
                    check_id=f"L3-TARGET-{i}",
                    level=VerificationLevel.TARGETED_VERIFICATION,
                    description=f"Targeted test: {target}",
                    status=CheckStatus.SKIP,
                    evidence=f"Test file not found or no tests collected: {target}",
                    command=cmd,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    exit_code=result.exit_code,
                    duration=result.duration,
                ))
            else:
                checks.append(VerificationCheck(
                    check_id=f"L3-TARGET-{i}",
                    level=VerificationLevel.TARGETED_VERIFICATION,
                    description=f"Targeted test: {target}",
                    status=CheckStatus.FAIL,
                    evidence=(
                        f"Targeted test failed: {target}\n"
                        f"{result.stdout[:300]}\n{result.stderr[:300]}"
                    ),
                    command=cmd,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    exit_code=result.exit_code,
                    duration=result.duration,
                ))

    return checks


def _run_contract_verification(
    contract: TaskContract | None,
    runner: CommandRunner,
    timeout: int,
) -> list[VerificationCheck]:
    """LEVEL 4 — Verify every proof obligation from the TaskContract."""
    checks: list[VerificationCheck] = []

    if contract is None:
        checks.append(VerificationCheck(
            check_id="L4-SKIP",
            level=VerificationLevel.CONTRACT_VERIFICATION,
            description="No task contract provided",
            status=CheckStatus.SKIP,
            evidence="Contract verification skipped: no contract available",
        ))
        return checks

    if not contract.proof_obligations:
        checks.append(VerificationCheck(
            check_id="L4-EMPTY",
            level=VerificationLevel.CONTRACT_VERIFICATION,
            description="Contract has no proof obligations",
            status=CheckStatus.SKIP,
            evidence="No proof obligations defined in the contract",
        ))
        return checks

    for ob in contract.proof_obligations:
        check_id = f"L4-{ob.id}"

        # Determine the verification command from the obligation's method.
        cmd = _obligation_to_command(ob)

        if cmd is None:
            # No executable verification method — mark as SKIP.
            checks.append(VerificationCheck(
                check_id=check_id,
                level=VerificationLevel.CONTRACT_VERIFICATION,
                description=f"Contract obligation: {ob.description}",
                status=CheckStatus.SKIP,
                evidence=(
                    f"Obligation {ob.id} has no executable verification "
                    f"method: {ob.verification_method}"
                ),
                obligation_id=ob.id,
            ))
            continue

        result = runner(cmd, timeout)

        if result.timed_out:
            checks.append(VerificationCheck(
                check_id=check_id,
                level=VerificationLevel.CONTRACT_VERIFICATION,
                description=f"Contract obligation: {ob.description}",
                status=CheckStatus.ERROR,
                evidence=(
                    f"Verification timed out for {ob.id} after {timeout}s"
                ),
                obligation_id=ob.id,
                command=cmd,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                duration=result.duration,
            ))
        elif result.success:
            checks.append(VerificationCheck(
                check_id=check_id,
                level=VerificationLevel.CONTRACT_VERIFICATION,
                description=f"Contract obligation: {ob.description}",
                status=CheckStatus.PASS,
                evidence=(
                    f"Obligation {ob.id} verified by executed test.\n"
                    f"Command: {cmd}\n"
                    f"Result: PASS\n"
                    f"Evidence: test output + affected code\n"
                    f"{result.stdout[:300]}"
                ),
                obligation_id=ob.id,
                command=cmd,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                duration=result.duration,
            ))
        else:
            checks.append(VerificationCheck(
                check_id=check_id,
                level=VerificationLevel.CONTRACT_VERIFICATION,
                description=f"Contract obligation: {ob.description}",
                status=CheckStatus.FAIL,
                evidence=(
                    f"Obligation {ob.id} FAILED.\n"
                    f"Command: {cmd}\n"
                    f"Exit code: {result.exit_code}\n"
                    f"{result.stdout[:300]}\n{result.stderr[:300]}"
                ),
                obligation_id=ob.id,
                command=cmd,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                duration=result.duration,
            ))

    return checks


def _obligation_to_command(ob: ProofObligation) -> str | None:
    """Extract a runnable command from a proof obligation's verification method.

    Returns ``None`` if the method doesn't contain an executable command.
    """
    method = ob.verification_method.strip()

    # Direct command patterns.
    if method.startswith("pytest ") or method.startswith("python "):
        return method

    # Look for pytest / python commands inside the text.
    m = re.search(r"(pytest\s+\S+(?:\s+-[^\n]*)?)", method)
    if m:
        return m.group(1).strip()

    m = re.search(r"(python\s+-[mc]\s+\S+)", method)
    if m:
        return m.group(1).strip()

    # Generic "run test_X" references.
    m = re.search(r"run\s+(test_\w+)", method, re.IGNORECASE)
    if m:
        return f"pytest -x -q -k {m.group(1)}"

    return None


def _run_regression_verification(
    changed_files: list[str],
    runner: CommandRunner,
    timeout: int,
    exclude_targets: list[str] | None = None,
) -> list[VerificationCheck]:
    """LEVEL 5 — Check behaviour outside the directly changed code."""
    checks: list[VerificationCheck] = []
    exclude = set(exclude_targets or [])

    # Run the full test suite but exclude changed test files to focus
    # on code *not* directly changed.
    changed_test_files = [
        f for f in changed_files
        if re.search(r"(^|/)test_|_test\.py$", f)
    ]

    if changed_test_files:
        # Ignore changed test files — focus on regression.
        ignore_args = " ".join(
            f"--ignore={tf}" for tf in changed_test_files if tf not in exclude
        )
        cmd = f"pytest tests/ -x -q {ignore_args}".strip()
    else:
        cmd = "pytest tests/ -x -q"

    result = runner(cmd, timeout)

    if result.timed_out:
        checks.append(VerificationCheck(
            check_id="L5-REGRESSION",
            level=VerificationLevel.REGRESSION,
            description="Regression check (unchanged tests)",
            status=CheckStatus.ERROR,
            evidence=f"Regression check timed out after {timeout}s",
            command=cmd,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            duration=result.duration,
        ))
    elif result.success:
        checks.append(VerificationCheck(
            check_id="L5-REGRESSION",
            level=VerificationLevel.REGRESSION,
            description="Regression check (unchanged tests)",
            status=CheckStatus.PASS,
            evidence=(
                f"Verified by executed regression suite.\n"
                f"No regressions detected in unchanged tests.\n"
                f"{result.stdout[:300]}"
            ),
            command=cmd,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            duration=result.duration,
        ))
    else:
        # exit code 5 = no tests collected → SKIP (all tests were excluded).
        if result.exit_code == 5:
            checks.append(VerificationCheck(
                check_id="L5-REGRESSION",
                level=VerificationLevel.REGRESSION,
                description="Regression check (unchanged tests)",
                status=CheckStatus.SKIP,
                evidence="No unchanged tests to run for regression",
                command=cmd,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                duration=result.duration,
            ))
        else:
            checks.append(VerificationCheck(
                check_id="L5-REGRESSION",
                level=VerificationLevel.REGRESSION,
                description="Regression check (unchanged tests)",
                status=CheckStatus.FAIL,
                evidence=(
                    f"Regression failure detected in unchanged tests.\n"
                    f"Exit code: {result.exit_code}\n"
                    f"{result.stdout[:300]}\n{result.stderr[:300]}"
                ),
                command=cmd,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                duration=result.duration,
            ))

    return checks


# ---------------------------------------------------------------------------
# Falsification interface (Level 6 — placeholder for src/falsifier.py)
# ---------------------------------------------------------------------------

class FalsificationInterface:
    """Clean interface for falsification (Level 6).

    This will be implemented by ``src/falsifier.py`` in a future module.
    The verifier calls ``attempt_falsification`` and uses the returned
    checks as Level 6 evidence.
    """

    def attempt_falsification(
        self,
        contract: TaskContract | None,
        changed_files: list[str],
        runner: CommandRunner,
        timeout: int,
    ) -> list[VerificationCheck]:
        """Attempt to falsify the agent's changes.

        Subclasses / future implementations should try to construct
        inputs that break the agent's modifications.

        Returns a list of ``VerificationCheck`` objects.
        """
        return [VerificationCheck(
            check_id="L6-STUB",
            level=VerificationLevel.FALSIFICATION,
            description="Falsification (not yet implemented)",
            status=CheckStatus.SKIP,
            evidence=(
                "Falsification level is a placeholder. "
                "Will connect to src/falsifier.py."
            ),
        )]


# ---------------------------------------------------------------------------
# Verifier — public API
# ---------------------------------------------------------------------------

class Verifier:
    """Multi-level task verifier with structured evidence.

    Parameters
    ----------
    runner : CommandRunner
        A callable with signature ``(command, timeout) -> CommandResult``
        that executes shell commands.  Injected for testability.
    falsifier : FalsificationInterface | None
        Optional falsification implementation.  Defaults to the stub.
    default_timeout : int
        Default per-command timeout in seconds.
    test_command : str
        Default command for running the project's test suite (Level 2).
    """

    def __init__(
        self,
        runner: CommandRunner,
        *,
        falsifier: FalsificationInterface | None = None,
        default_timeout: int = 60,
        test_command: str = "pytest tests/ -x -q",
    ) -> None:
        self._runner = runner
        self._falsifier = falsifier or FalsificationInterface()
        self._default_timeout = default_timeout
        self._test_command = test_command
        self._results: list[VerificationResult] = []

    # ------------------------------------------------------------------
    # Full verification pipeline
    # ------------------------------------------------------------------

    def verify(
        self,
        changed_files: list[str],
        *,
        contract: TaskContract | None = None,
        max_level: VerificationLevel = VerificationLevel.FALSIFICATION,
        timeout: int | None = None,
        file_contents: dict[str, str] | None = None,
        repo_root: str = "",
    ) -> VerificationResult:
        """Run the multi-level verification pipeline.

        Parameters
        ----------
        changed_files :
            List of files modified by the agent.
        contract :
            Optional TaskContract to verify against (Levels 4+).
        max_level :
            Highest verification level to execute.  Levels beyond this
            are skipped.
        timeout :
            Per-command timeout in seconds.
        file_contents :
            Optional mapping of path → content for fingerprinting
            (testing mode — avoids reading files from disk).
        repo_root :
            Repository root for file fingerprinting from disk.

        Returns
        -------
        VerificationResult
        """
        timeout = timeout or self._default_timeout
        start_time = time.monotonic()
        all_checks: list[VerificationCheck] = []
        all_evidence: list[str] = []
        primary_command = ""
        primary_exit_code = 0
        primary_stdout = ""
        primary_stderr = ""

        # -- LEVEL 1: Basic validation --
        if max_level.value >= VerificationLevel.BASIC_VALIDATION.value:
            l1_checks = _run_basic_validation(changed_files, self._runner, timeout)
            all_checks.extend(l1_checks)
            for c in l1_checks:
                if c.evidence:
                    all_evidence.append(c.evidence)

            # Abort early on syntax failures.
            if any(
                c.status == CheckStatus.FAIL
                and c.level == VerificationLevel.BASIC_VALIDATION
                for c in l1_checks
            ):
                return self._build_result(
                    all_checks, all_evidence, changed_files,
                    primary_command, primary_exit_code,
                    primary_stdout, primary_stderr,
                    start_time, file_contents=file_contents,
                    repo_root=repo_root,
                )

        # -- LEVEL 2: Existing tests --
        if max_level.value >= VerificationLevel.EXISTING_TESTS.value:
            l2_checks = _run_existing_tests(
                self._runner, timeout, self._test_command,
            )
            all_checks.extend(l2_checks)
            for c in l2_checks:
                if c.evidence:
                    all_evidence.append(c.evidence)
                if c.command:
                    primary_command = c.command
                if c.exit_code is not None:
                    primary_exit_code = c.exit_code
                if c.stdout:
                    primary_stdout = c.stdout
                if c.stderr:
                    primary_stderr = c.stderr

        # -- LEVEL 3: Targeted verification --
        if max_level.value >= VerificationLevel.TARGETED_VERIFICATION.value:
            l3_checks = _run_targeted_tests(
                changed_files, self._runner, timeout,
            )
            all_checks.extend(l3_checks)
            for c in l3_checks:
                if c.evidence:
                    all_evidence.append(c.evidence)

        # -- LEVEL 4: Contract verification --
        if max_level.value >= VerificationLevel.CONTRACT_VERIFICATION.value:
            l4_checks = _run_contract_verification(
                contract, self._runner, timeout,
            )
            all_checks.extend(l4_checks)
            for c in l4_checks:
                if c.evidence:
                    all_evidence.append(c.evidence)

                # Update contract obligations.
                if contract and c.obligation_id:
                    if c.status == CheckStatus.PASS:
                        contract.update_obligation(
                            c.obligation_id,
                            ObligationStatus.PASS,
                            c.evidence,
                        )
                    elif c.status == CheckStatus.FAIL:
                        contract.update_obligation(
                            c.obligation_id,
                            ObligationStatus.FAIL,
                            c.evidence,
                        )

        # -- LEVEL 5: Regression verification --
        if max_level.value >= VerificationLevel.REGRESSION.value:
            l5_checks = _run_regression_verification(
                changed_files, self._runner, timeout,
            )
            all_checks.extend(l5_checks)
            for c in l5_checks:
                if c.evidence:
                    all_evidence.append(c.evidence)

        # -- LEVEL 6: Falsification --
        if max_level.value >= VerificationLevel.FALSIFICATION.value:
            l6_checks = self._falsifier.attempt_falsification(
                contract, changed_files, self._runner, timeout,
            )
            all_checks.extend(l6_checks)
            for c in l6_checks:
                if c.evidence:
                    all_evidence.append(c.evidence)

        result = self._build_result(
            all_checks, all_evidence, changed_files,
            primary_command, primary_exit_code,
            primary_stdout, primary_stderr,
            start_time, file_contents=file_contents,
            repo_root=repo_root,
        )
        self._results.append(result)
        return result

    # ------------------------------------------------------------------
    # Single-level convenience methods
    # ------------------------------------------------------------------

    def run_level(
        self,
        level: VerificationLevel,
        changed_files: list[str],
        *,
        contract: TaskContract | None = None,
        timeout: int | None = None,
    ) -> list[VerificationCheck]:
        """Run a single verification level and return its checks."""
        timeout = timeout or self._default_timeout

        if level == VerificationLevel.BASIC_VALIDATION:
            return _run_basic_validation(changed_files, self._runner, timeout)
        elif level == VerificationLevel.EXISTING_TESTS:
            return _run_existing_tests(self._runner, timeout, self._test_command)
        elif level == VerificationLevel.TARGETED_VERIFICATION:
            return _run_targeted_tests(changed_files, self._runner, timeout)
        elif level == VerificationLevel.CONTRACT_VERIFICATION:
            return _run_contract_verification(contract, self._runner, timeout)
        elif level == VerificationLevel.REGRESSION:
            return _run_regression_verification(
                changed_files, self._runner, timeout,
            )
        elif level == VerificationLevel.FALSIFICATION:
            return self._falsifier.attempt_falsification(
                contract, changed_files, self._runner, timeout,
            )
        return []

    # ------------------------------------------------------------------
    # Staleness checking
    # ------------------------------------------------------------------

    def check_staleness(
        self,
        result: VerificationResult,
        current_fingerprints: list[FileFingerprint],
    ) -> bool:
        """Check if a verification result is stale.

        Returns ``True`` if any fingerprint has changed, and marks the
        result as STALE.
        """
        if result.is_stale(current_fingerprints):
            result.mark_stale()
            return True
        return False

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    @property
    def results(self) -> list[VerificationResult]:
        """Return all verification results from this session."""
        return list(self._results)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_result(
        self,
        checks: list[VerificationCheck],
        evidence: list[str],
        changed_files: list[str],
        primary_command: str,
        primary_exit_code: int,
        primary_stdout: str,
        primary_stderr: str,
        start_time: float,
        *,
        file_contents: dict[str, str] | None = None,
        repo_root: str = "",
    ) -> VerificationResult:
        """Aggregate checks into a final VerificationResult."""
        duration = round(time.monotonic() - start_time, 3)

        # Determine overall status.
        status, passed, failure_summary = _aggregate_status(checks)

        # Compute confidence.
        confidence = _compute_confidence(checks)

        # Compute fingerprints.
        fingerprints = compute_fingerprints(
            changed_files,
            repo_root=repo_root,
            file_contents=file_contents,
        )

        return VerificationResult(
            passed=passed,
            status=status,
            command=primary_command,
            exit_code=primary_exit_code,
            stdout=primary_stdout,
            stderr=primary_stderr,
            duration=duration,
            checks=checks,
            failure_summary=failure_summary,
            evidence=evidence,
            changed_files=list(changed_files),
            confidence=confidence,
            fingerprints=fingerprints,
        )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_status(
    checks: list[VerificationCheck],
) -> tuple[VerificationStatus, bool, str]:
    """Derive overall status from individual check outcomes.

    Rules:
    - Any FAIL → FAILED
    - All PASS/SKIP (at least one PASS) → VERIFIED
    - Otherwise → UNKNOWN
    - UNKNOWN is never promoted to VERIFIED
    """
    has_fail = False
    has_pass = False
    has_error = False
    failures: list[str] = []

    for c in checks:
        if c.status == CheckStatus.FAIL:
            has_fail = True
            failures.append(f"[{c.check_id}] {c.description}: {c.evidence}")
        elif c.status == CheckStatus.PASS:
            has_pass = True
        elif c.status == CheckStatus.ERROR:
            has_error = True
            failures.append(f"[{c.check_id}] {c.description}: {c.evidence}")

    if has_fail:
        return (
            VerificationStatus.FAILED,
            False,
            "; ".join(failures),
        )

    if has_error:
        return (
            VerificationStatus.UNKNOWN,
            False,
            "; ".join(failures),
        )

    if has_pass:
        return (
            VerificationStatus.VERIFIED,
            True,
            "",
        )

    # No PASS, no FAIL, no ERROR — all skipped or empty.
    return (
        VerificationStatus.UNKNOWN,
        False,
        "No verification checks produced a definitive result",
    )


def _compute_confidence(checks: list[VerificationCheck]) -> float:
    """Compute a confidence score from 0.0 to 1.0.

    Heuristic based on:
    - Proportion of PASS checks vs total non-skip checks.
    - Higher levels contribute more weight.
    """
    if not checks:
        return 0.0

    level_weights = {
        VerificationLevel.BASIC_VALIDATION: 0.1,
        VerificationLevel.EXISTING_TESTS: 0.2,
        VerificationLevel.TARGETED_VERIFICATION: 0.2,
        VerificationLevel.CONTRACT_VERIFICATION: 0.25,
        VerificationLevel.REGRESSION: 0.15,
        VerificationLevel.FALSIFICATION: 0.1,
    }

    total_weight = 0.0
    earned_weight = 0.0

    for c in checks:
        if c.status == CheckStatus.SKIP:
            continue
        w = level_weights.get(c.level, 0.1)
        total_weight += w
        if c.status == CheckStatus.PASS:
            earned_weight += w

    if total_weight == 0.0:
        return 0.0

    return round(earned_weight / total_weight, 3)
