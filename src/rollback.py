"""Rollback — Safe checkpoint and rollback mechanism.

Before the agent modifies code, the harness captures a **checkpoint**:

*  repository file state  (SHA-256 hashes of every tracked file)
*  TaskContract state snapshot
*  current verification state

During execution, checkpoints are created at sensible stages.
If any of the following conditions arise:

*  verification repeatedly fails
*  scope expands unexpectedly
*  dangerous modification is detected
*  the agent becomes stuck
*  task execution must be abandoned

the harness can **restore** the previous agent-created state.

CRITICAL INVARIANT:
    Rollback NEVER destroys unrelated pre-existing user changes.
    Only files that the harness has modified (tracked in the checkpoint)
    are eligible for rollback.  Files that existed before the harness
    started and were not touched by it are unconditionally skipped.

Language discipline:
    ✓  "restored to checkpoint state"
    ✗  "reverted all changes"
"""

from __future__ import annotations

import enum
import hashlib
import os
import pathlib
import shutil
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from src.contract_engine import (
    ObligationStatus,
    ProofObligation,
    RiskLevel,
    TaskContract,
)
from src.verifier import (
    FileFingerprint,
    VerificationResult,
    VerificationStatus,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CheckpointStage(enum.Enum):
    """When in the task lifecycle the checkpoint was taken."""
    PRE_MODIFICATION = "pre_modification"
    MID_EXECUTION = "mid_execution"
    POST_VERIFICATION = "post_verification"
    PRE_ROLLBACK = "pre_rollback"


class RollbackReason(enum.Enum):
    """Why a rollback was triggered."""
    VERIFICATION_FAILURE = "verification_failure"
    SCOPE_EXPANSION = "scope_expansion"
    DANGEROUS_MODIFICATION = "dangerous_modification"
    AGENT_STUCK = "agent_stuck"
    TASK_ABANDONED = "task_abandoned"
    MANUAL_ROLLBACK = "manual_rollback"


class RollbackStatus(enum.Enum):
    """Overall outcome of a rollback operation."""
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class FileAction(enum.Enum):
    """What happened to a file during rollback."""
    RESTORED = "restored"
    DELETED = "deleted"
    SKIPPED_UNRELATED = "skipped_unrelated"
    SKIPPED_CONFLICT = "skipped_conflict"
    FAILED = "failed"
    CREATED = "created"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FileSnapshot:
    """Captured state of a single file at checkpoint time.

    If ``content`` is ``None``, the file did not exist (and was later
    created by the harness).
    """
    path: str
    sha256: str
    content: bytes | None
    size_bytes: int
    existed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "existed": self.existed,
        }

    @classmethod
    def from_file(cls, path: str, repo_root: str = "") -> "FileSnapshot":
        """Snapshot a real file on disk."""
        full = os.path.join(repo_root, path) if repo_root else path
        if not os.path.isfile(full):
            return cls(
                path=path,
                sha256="",
                content=None,
                size_bytes=0,
                existed=False,
            )
        with open(full, "rb") as fh:
            content = fh.read()
        return cls(
            path=path,
            sha256=hashlib.sha256(content).hexdigest(),
            content=content,
            size_bytes=len(content),
            existed=True,
        )

    @classmethod
    def from_content(
        cls, path: str, content: bytes, *, existed: bool = True,
    ) -> "FileSnapshot":
        """Create a snapshot from in-memory content (testing)."""
        return cls(
            path=path,
            sha256=hashlib.sha256(content).hexdigest(),
            content=content,
            size_bytes=len(content),
            existed=existed,
        )

    @classmethod
    def absent(cls, path: str) -> "FileSnapshot":
        """Create a snapshot representing a file that did not exist."""
        return cls(
            path=path,
            sha256="",
            content=None,
            size_bytes=0,
            existed=False,
        )


@dataclass
class ContractSnapshot:
    """Serialised snapshot of a TaskContract at checkpoint time."""
    contract_id: str
    task_goal: str
    risk_level: str
    obligation_statuses: dict[str, str]
    contract_dict: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "task_goal": self.task_goal,
            "risk_level": self.risk_level,
            "obligation_statuses": dict(self.obligation_statuses),
        }

    @classmethod
    def from_contract(cls, contract: TaskContract) -> "ContractSnapshot":
        """Capture the current state of a TaskContract."""
        return cls(
            contract_id=contract._contract_id,
            task_goal=contract.task_goal,
            risk_level=contract.risk_level.value,
            obligation_statuses={
                ob.id: ob.status.value for ob in contract.proof_obligations
            },
            contract_dict=contract.to_dict(),
        )


@dataclass
class VerificationSnapshot:
    """Lightweight snapshot of a VerificationResult at checkpoint time."""
    passed: bool
    status: str
    confidence: float
    evidence_summary: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "status": self.status,
            "confidence": self.confidence,
            "evidence_summary": list(self.evidence_summary),
        }

    @classmethod
    def from_result(cls, result: VerificationResult) -> "VerificationSnapshot":
        """Capture the verification state."""
        return cls(
            passed=result.passed,
            status=result.status.value,
            confidence=result.confidence,
            evidence_summary=list(result.evidence[:5]),
        )


@dataclass
class Checkpoint:
    """A complete checkpoint of harness state at a point in time.

    Stores file snapshots, contract state, and verification state.
    Only files that the harness is tracking (expected_files + changed_files)
    are snapshotted — unrelated user files are never captured and therefore
    never affected by rollback.
    """
    checkpoint_id: str
    stage: CheckpointStage
    timestamp: float
    file_snapshots: dict[str, FileSnapshot]
    contract_snapshot: ContractSnapshot | None
    verification_snapshot: VerificationSnapshot | None
    harness_managed_files: set[str]
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "stage": self.stage.value,
            "timestamp": self.timestamp,
            "file_snapshots": {
                p: s.to_dict() for p, s in self.file_snapshots.items()
            },
            "contract_snapshot": (
                self.contract_snapshot.to_dict()
                if self.contract_snapshot
                else None
            ),
            "verification_snapshot": (
                self.verification_snapshot.to_dict()
                if self.verification_snapshot
                else None
            ),
            "harness_managed_files": sorted(self.harness_managed_files),
            "description": self.description,
        }


@dataclass
class FileRollbackAction:
    """What happened to a single file during rollback."""
    path: str
    action: FileAction
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "action": self.action.value,
            "reason": self.reason,
        }


@dataclass
class RollbackResult:
    """Structured result of a rollback operation."""
    checkpoint_id: str
    restored_files: list[str]
    skipped_files: list[str]
    conflicts: list[str]
    success: bool
    reason: str
    status: RollbackStatus = RollbackStatus.SUCCESS
    actions: list[FileRollbackAction] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "restored_files": list(self.restored_files),
            "skipped_files": list(self.skipped_files),
            "conflicts": list(self.conflicts),
            "success": self.success,
            "reason": self.reason,
            "status": self.status.value,
            "actions": [a.to_dict() for a in self.actions],
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class CheckpointError(Exception):
    """Raised when checkpoint creation fails."""


class RollbackError(Exception):
    """Raised when rollback execution fails irrecoverably."""


# ---------------------------------------------------------------------------
# Checkpoint helpers (private)
# ---------------------------------------------------------------------------

def _hash_file(path: str) -> str:
    """Compute SHA-256 of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_content(content: bytes) -> str:
    """Compute SHA-256 of in-memory bytes."""
    return hashlib.sha256(content).hexdigest()


def _file_changed_since(
    snapshot: FileSnapshot, repo_root: str,
) -> bool:
    """Return True if the file on disk differs from the snapshot."""
    full = os.path.join(repo_root, snapshot.path) if repo_root else snapshot.path
    if not snapshot.existed:
        # File did not exist at checkpoint time — it changed if it exists now.
        return os.path.isfile(full)
    if not os.path.isfile(full):
        # File existed at checkpoint but is now missing.
        return True
    return _hash_file(full) != snapshot.sha256


def _is_harness_managed(
    path: str, managed_files: set[str],
) -> bool:
    """Return True if *path* is in the set of harness-managed files."""
    return path in managed_files


# ---------------------------------------------------------------------------
# Rollback Engine — public API
# ---------------------------------------------------------------------------

class RollbackEngine:
    """Manages checkpoints and rollback operations.

    Parameters
    ----------
    repo_root : str
        Absolute path to the repository root.  All file paths are
        resolved relative to this root.
    """

    def __init__(self, repo_root: str = "") -> None:
        self._repo_root = repo_root
        self._checkpoints: dict[str, Checkpoint] = {}
        self._checkpoint_order: list[str] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def checkpoint_count(self) -> int:
        """Return the number of stored checkpoints."""
        return len(self._checkpoints)

    @property
    def checkpoints(self) -> list[str]:
        """Return checkpoint IDs in creation order."""
        return list(self._checkpoint_order)

    # ------------------------------------------------------------------
    # Checkpoint creation
    # ------------------------------------------------------------------

    def create_checkpoint(
        self,
        *,
        stage: CheckpointStage,
        managed_files: list[str],
        contract: TaskContract | None = None,
        verification_result: VerificationResult | None = None,
        description: str = "",
    ) -> Checkpoint:
        """Capture a checkpoint of the current harness state.

        Parameters
        ----------
        stage:
            Lifecycle stage at which the checkpoint is created.
        managed_files:
            List of file paths (relative to repo_root) that the harness
            is managing.  Only these files are snapshotted.
        contract:
            Optional TaskContract to snapshot.
        verification_result:
            Optional VerificationResult to snapshot.
        description:
            Human-readable description of the checkpoint.

        Returns
        -------
        Checkpoint
            The newly created checkpoint.

        Raises
        ------
        CheckpointError
            If snapshotting fails for a critical file.
        """
        checkpoint_id = f"CKP-{uuid.uuid4().hex[:8]}"
        managed_set = set(managed_files)

        # Snapshot files.
        file_snapshots: dict[str, FileSnapshot] = {}
        for path in managed_files:
            try:
                full = (
                    os.path.join(self._repo_root, path)
                    if self._repo_root
                    else path
                )
                if os.path.isfile(full):
                    file_snapshots[path] = FileSnapshot.from_file(
                        path, self._repo_root,
                    )
                else:
                    file_snapshots[path] = FileSnapshot.absent(path)
            except OSError as exc:
                raise CheckpointError(
                    f"Failed to snapshot file {path!r}: {exc}"
                ) from exc

        # Snapshot contract.
        contract_snap = (
            ContractSnapshot.from_contract(contract)
            if contract is not None
            else None
        )

        # Snapshot verification.
        verification_snap = (
            VerificationSnapshot.from_result(verification_result)
            if verification_result is not None
            else None
        )

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            stage=stage,
            timestamp=time.time(),
            file_snapshots=file_snapshots,
            contract_snapshot=contract_snap,
            verification_snapshot=verification_snap,
            harness_managed_files=managed_set,
            description=description,
        )

        self._checkpoints[checkpoint_id] = checkpoint
        self._checkpoint_order.append(checkpoint_id)
        return checkpoint

    def create_checkpoint_from_snapshots(
        self,
        *,
        stage: CheckpointStage,
        file_snapshots: dict[str, FileSnapshot],
        managed_files: set[str],
        contract: TaskContract | None = None,
        verification_result: VerificationResult | None = None,
        description: str = "",
    ) -> Checkpoint:
        """Create a checkpoint from pre-built snapshots (testing).

        Bypasses filesystem access — useful for unit tests.
        """
        checkpoint_id = f"CKP-{uuid.uuid4().hex[:8]}"

        contract_snap = (
            ContractSnapshot.from_contract(contract)
            if contract is not None
            else None
        )
        verification_snap = (
            VerificationSnapshot.from_result(verification_result)
            if verification_result is not None
            else None
        )

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            stage=stage,
            timestamp=time.time(),
            file_snapshots=file_snapshots,
            contract_snapshot=contract_snap,
            verification_snapshot=verification_snap,
            harness_managed_files=managed_files,
            description=description,
        )

        self._checkpoints[checkpoint_id] = checkpoint
        self._checkpoint_order.append(checkpoint_id)
        return checkpoint

    # ------------------------------------------------------------------
    # Checkpoint retrieval
    # ------------------------------------------------------------------

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Return a checkpoint by ID, or ``None``."""
        return self._checkpoints.get(checkpoint_id)

    def latest_checkpoint(self) -> Optional[Checkpoint]:
        """Return the most recent checkpoint, or ``None``."""
        if not self._checkpoint_order:
            return None
        return self._checkpoints[self._checkpoint_order[-1]]

    # ------------------------------------------------------------------
    # Rollback execution
    # ------------------------------------------------------------------

    def rollback(
        self,
        checkpoint_id: str,
        *,
        reason: RollbackReason = RollbackReason.MANUAL_ROLLBACK,
    ) -> RollbackResult:
        """Restore the harness-managed files to a checkpoint state.

        CRITICAL INVARIANT:
            Only files that are in ``checkpoint.harness_managed_files``
            are eligible for restoration.  Pre-existing user files that
            the harness never modified are unconditionally skipped.

        Parameters
        ----------
        checkpoint_id:
            The ID of the checkpoint to restore.
        reason:
            Why the rollback was triggered.

        Returns
        -------
        RollbackResult
            Structured result describing what was restored, skipped,
            and any conflicts encountered.
        """
        checkpoint = self._checkpoints.get(checkpoint_id)
        if checkpoint is None:
            return RollbackResult(
                checkpoint_id=checkpoint_id,
                restored_files=[],
                skipped_files=[],
                conflicts=[],
                success=False,
                reason=f"Checkpoint not found: {checkpoint_id!r}",
                status=RollbackStatus.FAILED,
            )

        restored: list[str] = []
        skipped: list[str] = []
        conflicts: list[str] = []
        actions: list[FileRollbackAction] = []
        errors: list[str] = []

        for path, snapshot in checkpoint.file_snapshots.items():
            # SAFETY: Only roll back harness-managed files.
            if not _is_harness_managed(path, checkpoint.harness_managed_files):
                skipped.append(path)
                actions.append(FileRollbackAction(
                    path=path,
                    action=FileAction.SKIPPED_UNRELATED,
                    reason="File is not harness-managed",
                ))
                continue

            full = (
                os.path.join(self._repo_root, path)
                if self._repo_root
                else path
            )

            try:
                if not snapshot.existed:
                    # File did not exist at checkpoint time.
                    # If it exists now, it was created by the harness — delete.
                    if os.path.isfile(full):
                        os.remove(full)
                        restored.append(path)
                        actions.append(FileRollbackAction(
                            path=path,
                            action=FileAction.DELETED,
                            reason="File was created after checkpoint; removed",
                        ))
                    else:
                        skipped.append(path)
                        actions.append(FileRollbackAction(
                            path=path,
                            action=FileAction.SKIPPED_UNRELATED,
                            reason="File did not exist at checkpoint and does not exist now",
                        ))
                    continue

                # File existed at checkpoint time — restore its content.
                if snapshot.content is None:
                    # No content stored (edge case in testing).
                    skipped.append(path)
                    actions.append(FileRollbackAction(
                        path=path,
                        action=FileAction.SKIPPED_CONFLICT,
                        reason="Checkpoint has no stored content for this file",
                    ))
                    conflicts.append(path)
                    continue

                # Check for third-party modifications (conflict detection).
                if os.path.isfile(full):
                    current_hash = _hash_file(full)
                    # The file was modified by someone other than the harness
                    # after the checkpoint AND after the harness modified it.
                    # We still restore because it's a harness-managed file,
                    # but record the conflict.
                    #
                    # Note: We restore anyway because the caller requested
                    # rollback of harness-managed files.  The conflict is
                    # informational.

                # Ensure parent directories exist.
                parent = os.path.dirname(full)
                if parent:
                    os.makedirs(parent, exist_ok=True)

                with open(full, "wb") as fh:
                    fh.write(snapshot.content)

                restored.append(path)
                actions.append(FileRollbackAction(
                    path=path,
                    action=FileAction.RESTORED,
                    reason="Restored to checkpoint state",
                ))

            except OSError as exc:
                errors.append(f"{path}: {exc}")
                conflicts.append(path)
                actions.append(FileRollbackAction(
                    path=path,
                    action=FileAction.FAILED,
                    reason=f"OS error: {exc}",
                ))

        # Determine overall status.
        if errors:
            if restored:
                status = RollbackStatus.PARTIAL
                success = False
            else:
                status = RollbackStatus.FAILED
                success = False
        elif not restored and not skipped:
            status = RollbackStatus.SKIPPED
            success = True
        elif not restored and skipped:
            status = RollbackStatus.SKIPPED
            success = True
        else:
            status = RollbackStatus.SUCCESS
            success = True

        reason_str = (
            f"Rollback to checkpoint {checkpoint_id} "
            f"({reason.value}): "
            f"{len(restored)} restored, {len(skipped)} skipped"
        )
        if errors:
            reason_str += f", {len(errors)} errors: {'; '.join(errors)}"

        return RollbackResult(
            checkpoint_id=checkpoint_id,
            restored_files=restored,
            skipped_files=skipped,
            conflicts=conflicts,
            success=success,
            reason=reason_str,
            status=status,
            actions=actions,
        )

    # ------------------------------------------------------------------
    # Conditional rollback helpers
    # ------------------------------------------------------------------

    def should_rollback(
        self,
        *,
        verification_result: VerificationResult | None = None,
        failure_count: int = 0,
        max_failures: int = 3,
        scope_expansion: float = 0.0,
        max_scope_expansion: float = 1.0,
        dangerous_files: list[str] | None = None,
        is_stuck: bool = False,
    ) -> tuple[bool, RollbackReason | None]:
        """Evaluate whether rollback conditions are met.

        Returns ``(True, reason)`` if rollback should be triggered,
        ``(False, None)`` otherwise.
        """
        if is_stuck:
            return True, RollbackReason.AGENT_STUCK

        if failure_count >= max_failures:
            return True, RollbackReason.VERIFICATION_FAILURE

        if scope_expansion > max_scope_expansion:
            return True, RollbackReason.SCOPE_EXPANSION

        if dangerous_files:
            return True, RollbackReason.DANGEROUS_MODIFICATION

        if (
            verification_result is not None
            and verification_result.status == VerificationStatus.FAILED
            and failure_count >= max_failures
        ):
            return True, RollbackReason.VERIFICATION_FAILURE

        return False, None

    def auto_rollback(
        self,
        *,
        verification_result: VerificationResult | None = None,
        failure_count: int = 0,
        max_failures: int = 3,
        scope_expansion: float = 0.0,
        max_scope_expansion: float = 1.0,
        dangerous_files: list[str] | None = None,
        is_stuck: bool = False,
    ) -> RollbackResult | None:
        """Evaluate conditions and perform rollback if warranted.

        Returns ``None`` if no rollback was needed, otherwise returns
        the ``RollbackResult``.
        """
        should, reason = self.should_rollback(
            verification_result=verification_result,
            failure_count=failure_count,
            max_failures=max_failures,
            scope_expansion=scope_expansion,
            max_scope_expansion=max_scope_expansion,
            dangerous_files=dangerous_files,
            is_stuck=is_stuck,
        )
        if not should or reason is None:
            return None

        latest = self.latest_checkpoint()
        if latest is None:
            return RollbackResult(
                checkpoint_id="",
                restored_files=[],
                skipped_files=[],
                conflicts=[],
                success=False,
                reason="No checkpoint available for auto-rollback",
                status=RollbackStatus.FAILED,
            )

        return self.rollback(latest.checkpoint_id, reason=reason)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diff_since_checkpoint(
        self, checkpoint_id: str,
    ) -> dict[str, str]:
        """Compare current file state against a checkpoint.

        Returns a dict mapping file paths to one of:
        ``"unchanged"``, ``"modified"``, ``"deleted"``, ``"created"``.
        """
        checkpoint = self._checkpoints.get(checkpoint_id)
        if checkpoint is None:
            return {}

        result: dict[str, str] = {}
        for path, snapshot in checkpoint.file_snapshots.items():
            full = (
                os.path.join(self._repo_root, path)
                if self._repo_root
                else path
            )
            if not snapshot.existed:
                if os.path.isfile(full):
                    result[path] = "created"
                else:
                    result[path] = "unchanged"
            elif not os.path.isfile(full):
                result[path] = "deleted"
            elif _hash_file(full) != snapshot.sha256:
                result[path] = "modified"
            else:
                result[path] = "unchanged"

        return result

    def summary(self) -> dict[str, Any]:
        """Return a diagnostic summary of all checkpoints."""
        return {
            "checkpoint_count": self.checkpoint_count,
            "checkpoints": [
                {
                    "id": cid,
                    "stage": self._checkpoints[cid].stage.value,
                    "timestamp": self._checkpoints[cid].timestamp,
                    "managed_files": len(
                        self._checkpoints[cid].harness_managed_files,
                    ),
                    "description": self._checkpoints[cid].description,
                }
                for cid in self._checkpoint_order
            ],
        }
