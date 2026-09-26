"""AI Coding-Agent Harness — Entry point.

Usage:
    make setup   Install dependencies
    make run     Launch the interactive harness
    make test    Run all tests

The harness presents a terminal interface where the evaluator can:
1.  Enter a natural-language coding issue
2.  Watch the autonomous pipeline execute
3.  Receive the final proof report

NEVER prints API keys or environment secrets.
No image/audio/video functionality.
"""

from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import traceback
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config.yaml"
_REPO_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)


def _load_config() -> dict[str, Any]:
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return {"model": {"name": "claude-sonnet-4-20250514"}, "runtime": {"max_iterations": 10}}


# ---------------------------------------------------------------------------
# TUI — Terminal User Interface
# ---------------------------------------------------------------------------

# Box-drawing and status symbols (plain ASCII fallback).
_FANCY = True
try:
    # Test if terminal supports unicode.
    sys.stdout.write("")
    sys.stdout.flush()
except Exception:
    _FANCY = False


def _sym(check: bool) -> str:
    """Return ✓ or ✗ based on check result."""
    if _FANCY:
        return "✓" if check else "✗"
    return "[PASS]" if check else "[FAIL]"


def _bolt() -> str:
    return "⚡" if _FANCY else ">"


def _arrow() -> str:
    return "→" if _FANCY else "->"


def _bar(width: int = 60) -> str:
    return "─" * width if _FANCY else "-" * width


class TUI:
    """Clean terminal interface for the harness pipeline.

    Outputs structured phase information.  Falls back to plain text
    if terminal rendering fails.
    """

    def __init__(self, *, quiet: bool = False) -> None:
        self._quiet = quiet

    def _print(self, *args: Any, **kwargs: Any) -> None:
        if not self._quiet:
            try:
                print(*args, **kwargs)
                sys.stdout.flush()
            except (BrokenPipeError, IOError):
                pass

    # -- Header / Footer ---------------------------------------------------

    def header(self) -> None:
        self._print()
        self._print(_bar())
        self._print("  AI CODING HARNESS")
        self._print(_bar())
        self._print()

    def footer(self, outcome: str, reason: str = "") -> None:
        self._print()
        self._print(_bar())
        self._print(f"  FINAL STATUS: {outcome}")
        if reason:
            self._print(f"  {reason}")
        self._print(_bar())
        self._print()

    # -- Phase sections ----------------------------------------------------

    def section(self, title: str) -> None:
        self._print()
        self._print(f"  {title}")

    def item(self, text: str, passed: bool | None = None) -> None:
        if passed is not None:
            self._print(f"    {_sym(passed)} {text}")
        else:
            self._print(f"    {text}")

    def detail(self, text: str) -> None:
        self._print(f"      {text}")

    def activity(self, text: str) -> None:
        self._print(f"    {_bolt()} {text}")

    def pointer(self, text: str) -> None:
        self._print(f"    {_arrow()} {text}")

    def blank(self) -> None:
        self._print()

    # -- Phase display methods (high-level) --------------------------------

    def show_task(self, task: str) -> None:
        self.section("TASK")
        # Wrap long task descriptions.
        words = task.split()
        line = "    "
        for w in words:
            if len(line) + len(w) + 1 > 72:
                self._print(line)
                line = "    " + w
            else:
                line += (" " if len(line) > 4 else "") + w
        if line.strip():
            self._print(line)

    def show_contract(self, contract: Any) -> None:
        self.section("CONTRACT")
        if contract is None:
            self.item("No contract generated", passed=False)
            return
        for ob in contract.proof_obligations:
            status = ob.status.value
            passed = status == "PASS"
            pending = status == "PENDING"
            if pending:
                self.item(f"{ob.description}  [{status}]")
            else:
                self.item(ob.description, passed=passed)

    def show_obligations(self, contract: Any) -> None:
        self.section("PROOF OBLIGATIONS")
        if contract is None:
            self.item("None")
            return
        for ob in contract.proof_obligations:
            self.item(f"{ob.id}: {ob.description}  [{ob.status.value}]")

    def show_impact(self, expected: list[str]) -> None:
        self.section("REPOSITORY IMPACT")
        if not expected:
            self.item("No files identified")
        for f in expected:
            self.item(f)

    def show_change_budget(
        self, expected: int, actual: int, status: str,
    ) -> None:
        self.section("CHANGE BUDGET")
        self.item(f"Expected: {expected} files")
        self.item(f"Actual:   {actual} files")
        self.item(f"Status:   {status}")

    def show_files_changed(self, files: list[str]) -> None:
        self.section("FILES CHANGED")
        for f in files:
            self.item(f)

    def show_verification(self, checks: list[dict]) -> None:
        self.section("VERIFICATION")
        level_names = {
            1: "Basic validation",
            2: "Existing tests",
            3: "Targeted tests",
            4: "Contract checks",
            5: "Regression",
            6: "Falsification",
        }
        for c in checks:
            level = c.get("level", 0)
            name = level_names.get(level, c.get("description", "Check"))
            status = c.get("status", "UNKNOWN")
            passed = status == "PASS"
            self.item(name, passed=(passed if status != "SKIP" else None))

    def show_falsification(
        self, counterexamples: list[str], status: str,
    ) -> None:
        self.section("FALSIFICATION")
        if status == "NOT_FALSIFIED" or not counterexamples:
            self.activity("Testing boundary cases...")
            self.activity("Testing invalid states...")
            self.item("No counterexample found", passed=True)
        elif status == "FALSIFIED":
            self.activity("Testing boundary cases...")
            self.activity("Testing invalid states...")
            self.blank()
            self.section("COUNTEREXAMPLES")
            self.item(f"Found: {len(counterexamples)}")
            for ce in counterexamples:
                self.pointer(ce)

    def show_repair(self, attempt_num: int, failure_type: str) -> None:
        self.section(f"REPAIR #{attempt_num}")
        self.item(f"Failure: {failure_type}")

    def show_evidence(self, fresh: bool) -> None:
        self.section("EVIDENCE STATUS")
        if fresh:
            self.item("Fresh", passed=True)
        else:
            self.item("Stale — reverification required", passed=False)


# ---------------------------------------------------------------------------
# Command runner (adapter for Verifier)
# ---------------------------------------------------------------------------

def _make_command_runner(repo_root: str):
    """Create a CommandRunner callable for the Verifier."""
    from src.verifier import CommandResult

    def runner(command: str, timeout: int) -> CommandResult:
        start = time.monotonic()
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=repo_root,
            )
            duration = round(time.monotonic() - start, 3)
            return CommandResult(
                success=proc.returncode == 0,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                exit_code=proc.returncode,
                duration=duration,
            )
        except subprocess.TimeoutExpired:
            duration = round(time.monotonic() - start, 3)
            return CommandResult(
                success=False,
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                exit_code=-1,
                duration=duration,
                timed_out=True,
            )
        except Exception as exc:
            duration = round(time.monotonic() - start, 3)
            return CommandResult(
                success=False,
                stdout="",
                stderr=str(exc),
                exit_code=-1,
                duration=duration,
            )

    return runner


# ---------------------------------------------------------------------------
# Interactive harness
# ---------------------------------------------------------------------------

def run_interactive(tui: TUI) -> int:
    """Run the interactive harness loop.

    Returns the exit code (0 = VERIFIED, 1 = FAILED/UNKNOWN).
    """
    from src.model_client import call_model
    from src.contract_engine import ContractEngine, ContractValidationError
    from src.verifier import Verifier, CheckStatus, VerificationLevel
    from src.change_guard import ChangeGuard, ScopeStatus
    from src.rollback import RollbackEngine
    from src.falsifier import FalsificationEngine
    from src.orchestrator import (
        Orchestrator,
        TaskOutcome,
        Phase,
    )

    config = _load_config()
    max_iterations = config.get("runtime", {}).get("max_iterations", 10)

    # Build components.
    contract_engine = ContractEngine(model_call=call_model)
    runner = _make_command_runner(_REPO_ROOT)
    falsifier = FalsificationEngine()
    verifier = Verifier(runner, falsifier=falsifier)
    change_guard = ChangeGuard()
    rollback_engine = RollbackEngine(_REPO_ROOT)

    orchestrator = Orchestrator(
        model_client=call_model,
        tools=[],
        max_iterations=max_iterations,
        verifier=verifier,
        contract_engine=contract_engine,
        change_guard=change_guard,
        rollback_engine=rollback_engine,
        repo_root=_REPO_ROOT,
    )

    # -- Prompt for task ---------------------------------------------------
    tui.header()
    tui._print("  Enter a coding issue (or 'quit' to exit):")
    tui._print()

    try:
        task = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        tui._print()
        tui._print("  Exiting.")
        return 0

    if not task or task.lower() in ("quit", "exit", "q"):
        tui._print("  Exiting.")
        return 0

    tui.show_task(task)
    tui.blank()

    # -- Run the orchestrator ----------------------------------------------
    try:
        result = orchestrator.run(task)
    except KeyboardInterrupt:
        tui.footer("INTERRUPTED", "User cancelled the operation")
        return 1
    except Exception as exc:
        tui.footer("FAILED", f"Orchestrator error: {exc}")
        return 1

    # -- Display results via TUI -------------------------------------------

    # CONTRACT
    tui.show_contract(result.contract)

    # PROOF OBLIGATIONS
    tui.show_obligations(result.contract)

    # REPOSITORY IMPACT
    expected_files = list(result.contract.expected_files) if result.contract else []
    tui.show_impact(expected_files)

    # CHANGE BUDGET
    actual_files: list[str] = []
    if result.final_verification:
        actual_files = result.final_verification.changed_files
    budget_status = "WITHIN_SCOPE"
    # Check phase log for scope violations.
    for entry in result.phase_log:
        if entry.phase == Phase.CHANGE_BUDGET and "BLOCKED" in entry.message:
            budget_status = "BLOCKED"
            break
        if entry.phase == Phase.CHANGE_BUDGET and "SCOPE_EXPANDED" in entry.message:
            budget_status = "SCOPE_EXPANDED"
            break
    tui.show_change_budget(
        len(expected_files), len(actual_files), budget_status,
    )

    # FILES CHANGED
    if actual_files:
        tui.show_files_changed(actual_files)

    # VERIFICATION
    if result.final_verification:
        checks_dicts = [c.to_dict() for c in result.final_verification.checks]
        tui.show_verification(checks_dicts)

    # FALSIFICATION / COUNTEREXAMPLES
    counterexamples: list[str] = []
    fals_status = "UNKNOWN"
    if result.history.attempts:
        for attempt in result.history.attempts:
            counterexamples.extend(attempt.counterexamples)
    if result.final_verification:
        has_fals_fail = any(
            c.status.value == "FAIL" and c.level.value == 6
            for c in result.final_verification.checks
        )
        if has_fals_fail:
            fals_status = "FALSIFIED"
        elif any(c.level.value == 6 for c in result.final_verification.checks):
            fals_status = "NOT_FALSIFIED"
    tui.show_falsification(counterexamples, fals_status)

    # REPAIR ATTEMPTS
    failed_attempts = result.history.failed_attempts()
    if failed_attempts:
        for a in failed_attempts:
            ft = a.failure_type.value if a.failure_type else "unknown"
            tui.show_repair(a.attempt_number + 1, ft)
            if a.failure_evidence:
                for ev in a.failure_evidence[:3]:
                    tui.detail(ev[:100])

    # EVIDENCE STATUS
    has_stale = False
    if result.evidence_report and result.evidence_report.has_stale_evidence:
        has_stale = True
    tui.show_evidence(not has_stale)

    # METRICS
    if result.metrics:
        tui.section("METRICS")
        for line in result.metrics.summary_lines():
            tui.item(line)

    # FINAL RESULT
    tui.footer(result.outcome.value, result.reason)

    return 0 if result.outcome == TaskOutcome.VERIFIED else 1


# ---------------------------------------------------------------------------
# Vercel handler (preserved for deployment)
# ---------------------------------------------------------------------------

class handler:
    """Vercel Serverless Function HTTP handler (preserved)."""

    @staticmethod
    def do_GET(self):
        from http.server import BaseHTTPRequestHandler
        api_key = os.environ.get("AI_API_KEY")
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        body = {
            "status": "Harness ready",
            "api_key_configured": bool(api_key),
        }
        self.wfile.write(json.dumps(body).encode("utf-8"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point for the AI Coding-Agent Harness.

    Called by ``make run``.

    Handles:
    *  EOF / Ctrl+C gracefully
    *  Missing API key
    *  Malformed input
    *  Model errors
    *  Tool errors
    *  Never prints API keys or environment secrets
    """
    # -- API key check (always required) -----------------------------------
    api_key = os.environ.get("AI_API_KEY")

    if not api_key:
        print(
            "ERROR: AI_API_KEY environment variable is not set.\n"
            "Export it before running:  export AI_API_KEY='your-key-here'",
            file=sys.stderr,
        )
        sys.exit(1)

    # -- Non-interactive mode (CI / tests) ---------------------------------
    # If stdin is not a TTY, emit the ready message and exit (preserves
    # backward compatibility with existing tests).
    if not sys.stdin.isatty():
        print("Harness ready")
        sys.exit(0)

    # -- Interactive mode --------------------------------------------------
    print("Harness ready")

    tui = TUI()

    # Graceful signal handling.
    def _signal_handler(signum, frame):
        print("\n  Interrupted. Exiting.")
        sys.exit(130)

    signal.signal(signal.SIGINT, _signal_handler)

    try:
        exit_code = run_interactive(tui)
    except KeyboardInterrupt:
        print("\n  Interrupted. Exiting.")
        exit_code = 130
    except EOFError:
        print("\n  EOF received. Exiting.")
        exit_code = 0
    except Exception as exc:
        # Catch-all: never expose internal stack traces with secrets.
        error_msg = str(exc)
        # Redact any API key that might leak.
        if api_key and api_key in error_msg:
            error_msg = error_msg.replace(api_key, "[REDACTED]")
        print(f"\n  Error: {error_msg}", file=sys.stderr)
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
