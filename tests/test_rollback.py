"""Tests for src/rollback.py — checkpoint and rollback mechanism."""

import hashlib
import os
import pathlib
import shutil
import tempfile
import time

import pytest

from src.rollback import (
    Checkpoint,
    CheckpointError,
    CheckpointStage,
    ContractSnapshot,
    FileAction,
    FileRollbackAction,
    FileSnapshot,
    RollbackEngine,
    RollbackError,
    RollbackReason,
    RollbackResult,
    RollbackStatus,
    VerificationSnapshot,
    _file_changed_since,
    _hash_content,
    _hash_file,
    _is_harness_managed,
)
from src.contract_engine import (
    ObligationStatus,
    ObligationType,
    ProofObligation,
    RiskLevel,
    TaskContract,
)
from src.verifier import (
    CheckStatus,
    FileFingerprint,
    VerificationCheck,
    VerificationLevel,
    VerificationResult,
    VerificationStatus,
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
        ]
    return TaskContract(
        task_goal="Test goal",
        behavioral_requirements=["Must do X"],
        acceptance_criteria=["X works"],
        constraints=["No side effects"],
        non_goals=["Don't change Y"],
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
    confidence: float = 0.95,
) -> VerificationResult:
    """Create a minimal VerificationResult for testing."""
    return VerificationResult(
        passed=passed,
        status=status,
        command="pytest",
        exit_code=0 if passed else 1,
        stdout="All tests passed" if passed else "1 failed",
        stderr="",
        duration=1.0,
        checks=[],
        failure_summary="" if passed else "Test failure",
        evidence=["test ran", "output captured"],
        changed_files=["src/foo.py"],
        confidence=confidence,
    )


class TempRepoMixin:
    """Mixin that creates a temporary directory for filesystem tests."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp(prefix="rollback_test_")

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _write(self, rel_path: str, content: str) -> str:
        """Write a file in the temp repo and return absolute path."""
        full = os.path.join(self._tmpdir, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as fh:
            fh.write(content)
        return full

    def _read(self, rel_path: str) -> str:
        """Read a file from the temp repo."""
        full = os.path.join(self._tmpdir, rel_path)
        with open(full, "r") as fh:
            return fh.read()

    def _exists(self, rel_path: str) -> bool:
        """Check if a file exists in the temp repo."""
        return os.path.isfile(os.path.join(self._tmpdir, rel_path))


# ══════════════════════════════════════════════════════════════════════
# FileSnapshot tests
# ══════════════════════════════════════════════════════════════════════


class TestFileSnapshot:
    """Tests for the FileSnapshot data class."""

    def test_from_content_basic(self):
        content = b"hello world"
        snap = FileSnapshot.from_content("src/foo.py", content)
        assert snap.path == "src/foo.py"
        assert snap.sha256 == hashlib.sha256(content).hexdigest()
        assert snap.size_bytes == len(content)
        assert snap.existed is True
        assert snap.content == content

    def test_from_content_non_existent(self):
        content = b"data"
        snap = FileSnapshot.from_content("new.py", content, existed=False)
        assert snap.existed is False
        assert snap.content == content

    def test_absent(self):
        snap = FileSnapshot.absent("missing.py")
        assert snap.path == "missing.py"
        assert snap.sha256 == ""
        assert snap.content is None
        assert snap.size_bytes == 0
        assert snap.existed is False

    def test_to_dict(self):
        snap = FileSnapshot.from_content("a.py", b"x")
        d = snap.to_dict()
        assert d["path"] == "a.py"
        assert "sha256" in d
        assert d["existed"] is True
        # Content is not serialised (privacy / size).
        assert "content" not in d


class TestFileSnapshotFilesystem(TempRepoMixin):
    """Tests for FileSnapshot.from_file with real filesystem."""

    def test_from_file_existing(self):
        self._write("src/hello.py", "print('hi')\n")
        snap = FileSnapshot.from_file("src/hello.py", self._tmpdir)
        assert snap.existed is True
        assert snap.content == b"print('hi')\n"
        assert snap.size_bytes == len(b"print('hi')\n")
        assert snap.sha256 == hashlib.sha256(b"print('hi')\n").hexdigest()

    def test_from_file_missing(self):
        snap = FileSnapshot.from_file("nonexistent.py", self._tmpdir)
        assert snap.existed is False
        assert snap.content is None


# ══════════════════════════════════════════════════════════════════════
# ContractSnapshot tests
# ══════════════════════════════════════════════════════════════════════


class TestContractSnapshot:
    """Tests for ContractSnapshot."""

    def test_from_contract(self):
        contract = _make_contract()
        snap = ContractSnapshot.from_contract(contract)
        assert snap.contract_id == contract._contract_id
        assert snap.task_goal == "Test goal"
        assert snap.risk_level == "LOW"
        assert snap.obligation_statuses["OB-1"] == "PENDING"

    def test_to_dict(self):
        contract = _make_contract()
        snap = ContractSnapshot.from_contract(contract)
        d = snap.to_dict()
        assert d["task_goal"] == "Test goal"
        assert "OB-1" in d["obligation_statuses"]


# ══════════════════════════════════════════════════════════════════════
# VerificationSnapshot tests
# ══════════════════════════════════════════════════════════════════════


class TestVerificationSnapshot:
    """Tests for VerificationSnapshot."""

    def test_from_result_passed(self):
        result = _make_verification_result(passed=True)
        snap = VerificationSnapshot.from_result(result)
        assert snap.passed is True
        assert snap.status == "VERIFIED"
        assert snap.confidence == 0.95

    def test_from_result_failed(self):
        result = _make_verification_result(
            passed=False,
            status=VerificationStatus.FAILED,
        )
        snap = VerificationSnapshot.from_result(result)
        assert snap.passed is False
        assert snap.status == "FAILED"

    def test_to_dict(self):
        result = _make_verification_result()
        snap = VerificationSnapshot.from_result(result)
        d = snap.to_dict()
        assert d["passed"] is True
        assert d["status"] == "VERIFIED"


# ══════════════════════════════════════════════════════════════════════
# Helper function tests
# ══════════════════════════════════════════════════════════════════════


class TestHelpers:
    """Tests for private helper functions."""

    def test_hash_content(self):
        h = _hash_content(b"hello")
        assert h == hashlib.sha256(b"hello").hexdigest()

    def test_is_harness_managed_true(self):
        assert _is_harness_managed("src/a.py", {"src/a.py", "src/b.py"})

    def test_is_harness_managed_false(self):
        assert not _is_harness_managed("src/c.py", {"src/a.py", "src/b.py"})


class TestHashFile(TempRepoMixin):
    """Tests for _hash_file with real filesystem."""

    def test_hash_file(self):
        path = self._write("test.txt", "content")
        assert _hash_file(path) == hashlib.sha256(b"content").hexdigest()


class TestFileChangedSince(TempRepoMixin):
    """Tests for _file_changed_since."""

    def test_unchanged(self):
        self._write("f.py", "original")
        snap = FileSnapshot.from_file("f.py", self._tmpdir)
        assert not _file_changed_since(snap, self._tmpdir)

    def test_modified(self):
        self._write("f.py", "original")
        snap = FileSnapshot.from_file("f.py", self._tmpdir)
        self._write("f.py", "modified")
        assert _file_changed_since(snap, self._tmpdir)

    def test_deleted(self):
        path = self._write("f.py", "original")
        snap = FileSnapshot.from_file("f.py", self._tmpdir)
        os.remove(path)
        assert _file_changed_since(snap, self._tmpdir)

    def test_created_since_absent(self):
        snap = FileSnapshot.absent("f.py")
        assert not _file_changed_since(snap, self._tmpdir)
        self._write("f.py", "new content")
        assert _file_changed_since(snap, self._tmpdir)


# ══════════════════════════════════════════════════════════════════════
# Checkpoint creation tests (in-memory)
# ══════════════════════════════════════════════════════════════════════


class TestCheckpointCreationInMemory:
    """Tests for checkpoint creation using in-memory snapshots."""

    def test_create_from_snapshots_basic(self):
        engine = RollbackEngine()
        snap = FileSnapshot.from_content("a.py", b"hello")
        ckp = engine.create_checkpoint_from_snapshots(
            stage=CheckpointStage.PRE_MODIFICATION,
            file_snapshots={"a.py": snap},
            managed_files={"a.py"},
            description="Before modification",
        )
        assert ckp.checkpoint_id.startswith("CKP-")
        assert ckp.stage == CheckpointStage.PRE_MODIFICATION
        assert "a.py" in ckp.file_snapshots
        assert "a.py" in ckp.harness_managed_files
        assert ckp.description == "Before modification"

    def test_checkpoint_with_contract(self):
        engine = RollbackEngine()
        contract = _make_contract()
        ckp = engine.create_checkpoint_from_snapshots(
            stage=CheckpointStage.MID_EXECUTION,
            file_snapshots={},
            managed_files=set(),
            contract=contract,
        )
        assert ckp.contract_snapshot is not None
        assert ckp.contract_snapshot.task_goal == "Test goal"

    def test_checkpoint_with_verification(self):
        engine = RollbackEngine()
        result = _make_verification_result()
        ckp = engine.create_checkpoint_from_snapshots(
            stage=CheckpointStage.POST_VERIFICATION,
            file_snapshots={},
            managed_files=set(),
            verification_result=result,
        )
        assert ckp.verification_snapshot is not None
        assert ckp.verification_snapshot.passed is True

    def test_checkpoint_count(self):
        engine = RollbackEngine()
        snap = FileSnapshot.from_content("a.py", b"v1")
        engine.create_checkpoint_from_snapshots(
            stage=CheckpointStage.PRE_MODIFICATION,
            file_snapshots={"a.py": snap},
            managed_files={"a.py"},
        )
        engine.create_checkpoint_from_snapshots(
            stage=CheckpointStage.MID_EXECUTION,
            file_snapshots={"a.py": snap},
            managed_files={"a.py"},
        )
        assert engine.checkpoint_count == 2
        assert len(engine.checkpoints) == 2

    def test_latest_checkpoint(self):
        engine = RollbackEngine()
        snap = FileSnapshot.from_content("a.py", b"v1")
        ckp1 = engine.create_checkpoint_from_snapshots(
            stage=CheckpointStage.PRE_MODIFICATION,
            file_snapshots={"a.py": snap},
            managed_files={"a.py"},
        )
        snap2 = FileSnapshot.from_content("a.py", b"v2")
        ckp2 = engine.create_checkpoint_from_snapshots(
            stage=CheckpointStage.MID_EXECUTION,
            file_snapshots={"a.py": snap2},
            managed_files={"a.py"},
        )
        latest = engine.latest_checkpoint()
        assert latest is not None
        assert latest.checkpoint_id == ckp2.checkpoint_id

    def test_latest_checkpoint_empty(self):
        engine = RollbackEngine()
        assert engine.latest_checkpoint() is None

    def test_get_checkpoint(self):
        engine = RollbackEngine()
        snap = FileSnapshot.from_content("a.py", b"v1")
        ckp = engine.create_checkpoint_from_snapshots(
            stage=CheckpointStage.PRE_MODIFICATION,
            file_snapshots={"a.py": snap},
            managed_files={"a.py"},
        )
        assert engine.get_checkpoint(ckp.checkpoint_id) is ckp
        assert engine.get_checkpoint("CKP-nonexistent") is None

    def test_checkpoint_to_dict(self):
        engine = RollbackEngine()
        contract = _make_contract()
        result = _make_verification_result()
        snap = FileSnapshot.from_content("a.py", b"v1")
        ckp = engine.create_checkpoint_from_snapshots(
            stage=CheckpointStage.PRE_MODIFICATION,
            file_snapshots={"a.py": snap},
            managed_files={"a.py"},
            contract=contract,
            verification_result=result,
            description="Full checkpoint",
        )
        d = ckp.to_dict()
        assert d["stage"] == "pre_modification"
        assert "a.py" in d["file_snapshots"]
        assert d["contract_snapshot"]["task_goal"] == "Test goal"
        assert d["verification_snapshot"]["passed"] is True
        assert "a.py" in d["harness_managed_files"]


# ══════════════════════════════════════════════════════════════════════
# Checkpoint creation tests (filesystem)
# ══════════════════════════════════════════════════════════════════════


class TestCheckpointCreationFilesystem(TempRepoMixin):
    """Tests for create_checkpoint with real filesystem access."""

    def test_create_checkpoint_existing_file(self):
        self._write("src/foo.py", "def foo(): pass\n")
        engine = RollbackEngine(self._tmpdir)
        ckp = engine.create_checkpoint(
            stage=CheckpointStage.PRE_MODIFICATION,
            managed_files=["src/foo.py"],
        )
        assert "src/foo.py" in ckp.file_snapshots
        snap = ckp.file_snapshots["src/foo.py"]
        assert snap.existed is True
        assert snap.content == b"def foo(): pass\n"

    def test_create_checkpoint_missing_file(self):
        engine = RollbackEngine(self._tmpdir)
        ckp = engine.create_checkpoint(
            stage=CheckpointStage.PRE_MODIFICATION,
            managed_files=["src/new.py"],
        )
        snap = ckp.file_snapshots["src/new.py"]
        assert snap.existed is False

    def test_create_checkpoint_multiple_files(self):
        self._write("a.py", "aaa")
        self._write("b.py", "bbb")
        engine = RollbackEngine(self._tmpdir)
        ckp = engine.create_checkpoint(
            stage=CheckpointStage.PRE_MODIFICATION,
            managed_files=["a.py", "b.py"],
        )
        assert len(ckp.file_snapshots) == 2
        assert ckp.file_snapshots["a.py"].content == b"aaa"
        assert ckp.file_snapshots["b.py"].content == b"bbb"

    def test_create_checkpoint_with_contract_and_verification(self):
        self._write("src/foo.py", "code")
        engine = RollbackEngine(self._tmpdir)
        contract = _make_contract()
        result = _make_verification_result()
        ckp = engine.create_checkpoint(
            stage=CheckpointStage.POST_VERIFICATION,
            managed_files=["src/foo.py"],
            contract=contract,
            verification_result=result,
            description="After first verification pass",
        )
        assert ckp.contract_snapshot is not None
        assert ckp.verification_snapshot is not None
        assert ckp.description == "After first verification pass"


# ══════════════════════════════════════════════════════════════════════
# Successful rollback tests
# ══════════════════════════════════════════════════════════════════════


class TestSuccessfulRollback(TempRepoMixin):
    """Tests for rollback that successfully restores state."""

    def test_rollback_restores_file_content(self):
        """Core test: checkpoint, modify, rollback restores original."""
        self._write("src/foo.py", "original content")
        engine = RollbackEngine(self._tmpdir)

        ckp = engine.create_checkpoint(
            stage=CheckpointStage.PRE_MODIFICATION,
            managed_files=["src/foo.py"],
        )

        # Agent modifies the file.
        self._write("src/foo.py", "agent modified content")
        assert self._read("src/foo.py") == "agent modified content"

        # Rollback.
        result = engine.rollback(ckp.checkpoint_id)

        assert result.success is True
        assert result.status == RollbackStatus.SUCCESS
        assert "src/foo.py" in result.restored_files
        assert self._read("src/foo.py") == "original content"

    def test_rollback_deletes_created_file(self):
        """File that didn't exist at checkpoint time is removed."""
        engine = RollbackEngine(self._tmpdir)

        ckp = engine.create_checkpoint(
            stage=CheckpointStage.PRE_MODIFICATION,
            managed_files=["src/new.py"],
        )

        # Agent creates the file.
        self._write("src/new.py", "new file content")
        assert self._exists("src/new.py")

        result = engine.rollback(ckp.checkpoint_id)

        assert result.success is True
        assert "src/new.py" in result.restored_files
        assert not self._exists("src/new.py")

    def test_rollback_multiple_files(self):
        """Rollback restores multiple modified files."""
        self._write("a.py", "orig_a")
        self._write("b.py", "orig_b")
        engine = RollbackEngine(self._tmpdir)

        ckp = engine.create_checkpoint(
            stage=CheckpointStage.PRE_MODIFICATION,
            managed_files=["a.py", "b.py"],
        )

        self._write("a.py", "modified_a")
        self._write("b.py", "modified_b")

        result = engine.rollback(ckp.checkpoint_id)

        assert result.success is True
        assert set(result.restored_files) == {"a.py", "b.py"}
        assert self._read("a.py") == "orig_a"
        assert self._read("b.py") == "orig_b"

    def test_rollback_result_structure(self):
        """Verify the RollbackResult has all required fields."""
        self._write("f.py", "v1")
        engine = RollbackEngine(self._tmpdir)
        ckp = engine.create_checkpoint(
            stage=CheckpointStage.PRE_MODIFICATION,
            managed_files=["f.py"],
        )
        self._write("f.py", "v2")
        result = engine.rollback(ckp.checkpoint_id)

        assert result.checkpoint_id == ckp.checkpoint_id
        assert isinstance(result.restored_files, list)
        assert isinstance(result.skipped_files, list)
        assert isinstance(result.conflicts, list)
        assert isinstance(result.success, bool)
        assert isinstance(result.reason, str)
        assert result.status == RollbackStatus.SUCCESS

        # Serialisation.
        d = result.to_dict()
        assert d["checkpoint_id"] == ckp.checkpoint_id
        assert d["success"] is True
        assert d["status"] == "SUCCESS"

    def test_rollback_to_earlier_checkpoint(self):
        """Rollback to the first of two checkpoints."""
        self._write("f.py", "v1")
        engine = RollbackEngine(self._tmpdir)

        ckp1 = engine.create_checkpoint(
            stage=CheckpointStage.PRE_MODIFICATION,
            managed_files=["f.py"],
        )

        self._write("f.py", "v2")
        ckp2 = engine.create_checkpoint(
            stage=CheckpointStage.MID_EXECUTION,
            managed_files=["f.py"],
        )

        self._write("f.py", "v3")

        # Rollback to first checkpoint.
        result = engine.rollback(ckp1.checkpoint_id)
        assert result.success is True
        assert self._read("f.py") == "v1"


# ══════════════════════════════════════════════════════════════════════
# Partial modification tests
# ══════════════════════════════════════════════════════════════════════


class TestPartialModification(TempRepoMixin):
    """Tests for rollback when only some files were modified."""

    def test_rollback_unchanged_file_still_restored(self):
        """Rollback restores even files that weren't modified (idempotent)."""
        self._write("a.py", "orig_a")
        self._write("b.py", "orig_b")
        engine = RollbackEngine(self._tmpdir)

        ckp = engine.create_checkpoint(
            stage=CheckpointStage.PRE_MODIFICATION,
            managed_files=["a.py", "b.py"],
        )

        # Only modify a.py.
        self._write("a.py", "modified_a")

        result = engine.rollback(ckp.checkpoint_id)

        assert result.success is True
        assert self._read("a.py") == "orig_a"
        assert self._read("b.py") == "orig_b"

    def test_rollback_mix_of_created_and_modified(self):
        """Mix of file creation and modification."""
        self._write("existing.py", "original")
        engine = RollbackEngine(self._tmpdir)

        ckp = engine.create_checkpoint(
            stage=CheckpointStage.PRE_MODIFICATION,
            managed_files=["existing.py", "new.py"],
        )

        self._write("existing.py", "changed")
        self._write("new.py", "brand new")

        result = engine.rollback(ckp.checkpoint_id)

        assert result.success is True
        assert self._read("existing.py") == "original"
        assert not self._exists("new.py")


# ══════════════════════════════════════════════════════════════════════
# Unrelated pre-existing changes tests
# ══════════════════════════════════════════════════════════════════════


class TestUnrelatedPreexistingChanges(TempRepoMixin):
    """CRITICAL: Rollback must NEVER affect files not managed by harness."""

    def test_unrelated_file_untouched(self):
        """User file that harness never touched must survive rollback."""
        # User's file — not managed by harness.
        self._write("user_code.py", "user's precious code")
        # Harness-managed file.
        self._write("src/task.py", "original task code")

        engine = RollbackEngine(self._tmpdir)

        ckp = engine.create_checkpoint(
            stage=CheckpointStage.PRE_MODIFICATION,
            managed_files=["src/task.py"],
            # Note: user_code.py is NOT in managed_files.
        )

        # Agent modifies only its file.
        self._write("src/task.py", "agent modified task code")

        result = engine.rollback(ckp.checkpoint_id)

        assert result.success is True
        # Harness file restored.
        assert self._read("src/task.py") == "original task code"
        # User file absolutely untouched.
        assert self._read("user_code.py") == "user's precious code"

    def test_user_file_modified_during_task_not_rolled_back(self):
        """User modifies their own file during task — must not be touched."""
        self._write("user_code.py", "user v1")
        self._write("src/task.py", "harness v1")

        engine = RollbackEngine(self._tmpdir)

        ckp = engine.create_checkpoint(
            stage=CheckpointStage.PRE_MODIFICATION,
            managed_files=["src/task.py"],
        )

        # Both user and agent make changes.
        self._write("user_code.py", "user v2")  # User change.
        self._write("src/task.py", "harness v2")  # Agent change.

        result = engine.rollback(ckp.checkpoint_id)

        assert result.success is True
        # Harness file restored.
        assert self._read("src/task.py") == "harness v1"
        # User's independent change preserved.
        assert self._read("user_code.py") == "user v2"

    def test_many_unrelated_files_preserved(self):
        """Multiple user files all survive rollback."""
        for i in range(5):
            self._write(f"user_{i}.py", f"user content {i}")
        self._write("harness.py", "harness original")

        engine = RollbackEngine(self._tmpdir)

        ckp = engine.create_checkpoint(
            stage=CheckpointStage.PRE_MODIFICATION,
            managed_files=["harness.py"],
        )

        self._write("harness.py", "harness modified")

        result = engine.rollback(ckp.checkpoint_id)

        assert result.success is True
        assert self._read("harness.py") == "harness original"
        for i in range(5):
            assert self._read(f"user_{i}.py") == f"user content {i}"


# ══════════════════════════════════════════════════════════════════════
# Failed rollback tests
# ══════════════════════════════════════════════════════════════════════


class TestFailedRollback:
    """Tests for rollback failure scenarios."""

    def test_rollback_nonexistent_checkpoint(self):
        """Rollback with invalid checkpoint ID returns failure."""
        engine = RollbackEngine()
        result = engine.rollback("CKP-doesnotexist")
        assert result.success is False
        assert result.status == RollbackStatus.FAILED
        assert "not found" in result.reason

    def test_rollback_no_content_stored(self):
        """Rollback when snapshot has no stored content records conflict."""
        engine = RollbackEngine()

        # Create a snapshot with existed=True but content=None.
        bad_snap = FileSnapshot(
            path="a.py",
            sha256="abc",
            content=None,
            size_bytes=100,
            existed=True,
        )
        ckp = engine.create_checkpoint_from_snapshots(
            stage=CheckpointStage.PRE_MODIFICATION,
            file_snapshots={"a.py": bad_snap},
            managed_files={"a.py"},
        )

        result = engine.rollback(ckp.checkpoint_id)

        # File is in conflicts because content is None.
        assert "a.py" in result.conflicts
        assert "a.py" in result.skipped_files


class TestFailedRollbackFilesystem(TempRepoMixin):
    """Tests for filesystem-level rollback failures."""

    def test_rollback_read_only_target(self):
        """Rollback fails gracefully on read-only file."""
        self._write("readonly.py", "original")
        engine = RollbackEngine(self._tmpdir)

        ckp = engine.create_checkpoint(
            stage=CheckpointStage.PRE_MODIFICATION,
            managed_files=["readonly.py"],
        )

        self._write("readonly.py", "modified")
        full = os.path.join(self._tmpdir, "readonly.py")
        os.chmod(full, 0o444)

        try:
            result = engine.rollback(ckp.checkpoint_id)
            # On some systems this may succeed (root), on others it fails.
            # The test verifies graceful handling either way.
            if not result.success:
                assert result.status in (
                    RollbackStatus.PARTIAL,
                    RollbackStatus.FAILED,
                )
                assert len(result.conflicts) > 0 or len(result.restored_files) == 0
        finally:
            # Clean up permissions for teardown.
            os.chmod(full, 0o644)


# ══════════════════════════════════════════════════════════════════════
# Interrupted execution tests
# ══════════════════════════════════════════════════════════════════════


class TestInterruptedExecution(TempRepoMixin):
    """Tests for rollback after interrupted/partial execution."""

    def test_rollback_after_partial_modification(self):
        """Agent modifies some files then checkpoint is needed."""
        self._write("a.py", "orig_a")
        self._write("b.py", "orig_b")
        self._write("c.py", "orig_c")

        engine = RollbackEngine(self._tmpdir)

        ckp = engine.create_checkpoint(
            stage=CheckpointStage.PRE_MODIFICATION,
            managed_files=["a.py", "b.py", "c.py"],
        )

        # Agent modifies a and b, but gets interrupted before c.
        self._write("a.py", "modified_a")
        self._write("b.py", "modified_b")
        # c.py remains unchanged.

        result = engine.rollback(ckp.checkpoint_id)

        assert result.success is True
        assert self._read("a.py") == "orig_a"
        assert self._read("b.py") == "orig_b"
        assert self._read("c.py") == "orig_c"

    def test_mid_execution_checkpoint_and_rollback(self):
        """Checkpoint taken mid-execution captures intermediate state."""
        self._write("f.py", "v1")
        engine = RollbackEngine(self._tmpdir)

        # Pre-modification checkpoint.
        ckp_pre = engine.create_checkpoint(
            stage=CheckpointStage.PRE_MODIFICATION,
            managed_files=["f.py"],
        )

        # Agent makes progress.
        self._write("f.py", "v2_partial")

        # Mid-execution checkpoint.
        ckp_mid = engine.create_checkpoint(
            stage=CheckpointStage.MID_EXECUTION,
            managed_files=["f.py"],
        )

        # Agent continues but breaks things.
        self._write("f.py", "v3_broken")

        # Rollback to mid-execution (preserves partial progress).
        result = engine.rollback(ckp_mid.checkpoint_id)
        assert result.success is True
        assert self._read("f.py") == "v2_partial"

        # Or rollback to pre-modification (full reset).
        self._write("f.py", "v3_broken_again")
        result2 = engine.rollback(ckp_pre.checkpoint_id)
        assert result2.success is True
        assert self._read("f.py") == "v1"

    def test_multiple_checkpoints_across_stages(self):
        """Multiple checkpoints at different stages all work."""
        self._write("f.py", "initial")
        engine = RollbackEngine(self._tmpdir)

        ckps = []
        for i, stage in enumerate([
            CheckpointStage.PRE_MODIFICATION,
            CheckpointStage.MID_EXECUTION,
            CheckpointStage.POST_VERIFICATION,
        ]):
            ckps.append(engine.create_checkpoint(
                stage=stage,
                managed_files=["f.py"],
                description=f"Stage {i}",
            ))
            self._write("f.py", f"version_{i + 1}")

        # Rollback to each stage works.
        for i, ckp in enumerate(ckps):
            if i == 0:
                expected = "initial"
            else:
                expected = f"version_{i}"
            result = engine.rollback(ckp.checkpoint_id)
            assert result.success is True
            assert self._read("f.py") == expected
            # Restore to allow next rollback test.
            self._write("f.py", "current")


# ══════════════════════════════════════════════════════════════════════
# Conditional rollback / should_rollback tests
# ══════════════════════════════════════════════════════════════════════


class TestShouldRollback:
    """Tests for the should_rollback condition evaluator."""

    def test_no_conditions_met(self):
        engine = RollbackEngine()
        should, reason = engine.should_rollback()
        assert should is False
        assert reason is None

    def test_agent_stuck(self):
        engine = RollbackEngine()
        should, reason = engine.should_rollback(is_stuck=True)
        assert should is True
        assert reason == RollbackReason.AGENT_STUCK

    def test_failure_count_exceeded(self):
        engine = RollbackEngine()
        should, reason = engine.should_rollback(
            failure_count=3, max_failures=3,
        )
        assert should is True
        assert reason == RollbackReason.VERIFICATION_FAILURE

    def test_failure_count_not_exceeded(self):
        engine = RollbackEngine()
        should, reason = engine.should_rollback(
            failure_count=2, max_failures=3,
        )
        assert should is False

    def test_scope_expansion(self):
        engine = RollbackEngine()
        should, reason = engine.should_rollback(
            scope_expansion=1.5, max_scope_expansion=1.0,
        )
        assert should is True
        assert reason == RollbackReason.SCOPE_EXPANSION

    def test_dangerous_files(self):
        engine = RollbackEngine()
        should, reason = engine.should_rollback(
            dangerous_files=[".env", "Makefile"],
        )
        assert should is True
        assert reason == RollbackReason.DANGEROUS_MODIFICATION

    def test_priority_agent_stuck_over_failures(self):
        """Agent stuck takes priority over failure count."""
        engine = RollbackEngine()
        should, reason = engine.should_rollback(
            is_stuck=True, failure_count=10, max_failures=3,
        )
        assert reason == RollbackReason.AGENT_STUCK


# ══════════════════════════════════════════════════════════════════════
# Auto-rollback tests
# ══════════════════════════════════════════════════════════════════════


class TestAutoRollback(TempRepoMixin):
    """Tests for auto_rollback integration."""

    def test_auto_rollback_no_trigger(self):
        engine = RollbackEngine(self._tmpdir)
        result = engine.auto_rollback()
        assert result is None

    def test_auto_rollback_no_checkpoint_available(self):
        engine = RollbackEngine(self._tmpdir)
        result = engine.auto_rollback(is_stuck=True)
        assert result is not None
        assert result.success is False
        assert "No checkpoint available" in result.reason

    def test_auto_rollback_triggered(self):
        self._write("f.py", "original")
        engine = RollbackEngine(self._tmpdir)

        engine.create_checkpoint(
            stage=CheckpointStage.PRE_MODIFICATION,
            managed_files=["f.py"],
        )

        self._write("f.py", "broken")

        result = engine.auto_rollback(
            failure_count=5, max_failures=3,
        )

        assert result is not None
        assert result.success is True
        assert self._read("f.py") == "original"

    def test_auto_rollback_uses_latest_checkpoint(self):
        self._write("f.py", "v1")
        engine = RollbackEngine(self._tmpdir)

        engine.create_checkpoint(
            stage=CheckpointStage.PRE_MODIFICATION,
            managed_files=["f.py"],
        )

        self._write("f.py", "v2")
        engine.create_checkpoint(
            stage=CheckpointStage.MID_EXECUTION,
            managed_files=["f.py"],
        )

        self._write("f.py", "v3")

        result = engine.auto_rollback(is_stuck=True)
        assert result is not None
        assert result.success is True
        # Should restore to v2 (latest checkpoint), not v1.
        assert self._read("f.py") == "v2"


# ══════════════════════════════════════════════════════════════════════
# Diff since checkpoint tests
# ══════════════════════════════════════════════════════════════════════


class TestDiffSinceCheckpoint(TempRepoMixin):
    """Tests for diff_since_checkpoint diagnostics."""

    def test_diff_unchanged(self):
        self._write("f.py", "content")
        engine = RollbackEngine(self._tmpdir)
        ckp = engine.create_checkpoint(
            stage=CheckpointStage.PRE_MODIFICATION,
            managed_files=["f.py"],
        )
        diff = engine.diff_since_checkpoint(ckp.checkpoint_id)
        assert diff["f.py"] == "unchanged"

    def test_diff_modified(self):
        self._write("f.py", "v1")
        engine = RollbackEngine(self._tmpdir)
        ckp = engine.create_checkpoint(
            stage=CheckpointStage.PRE_MODIFICATION,
            managed_files=["f.py"],
        )
        self._write("f.py", "v2")
        diff = engine.diff_since_checkpoint(ckp.checkpoint_id)
        assert diff["f.py"] == "modified"

    def test_diff_deleted(self):
        path = self._write("f.py", "content")
        engine = RollbackEngine(self._tmpdir)
        ckp = engine.create_checkpoint(
            stage=CheckpointStage.PRE_MODIFICATION,
            managed_files=["f.py"],
        )
        os.remove(path)
        diff = engine.diff_since_checkpoint(ckp.checkpoint_id)
        assert diff["f.py"] == "deleted"

    def test_diff_created(self):
        engine = RollbackEngine(self._tmpdir)
        ckp = engine.create_checkpoint(
            stage=CheckpointStage.PRE_MODIFICATION,
            managed_files=["new.py"],
        )
        self._write("new.py", "new content")
        diff = engine.diff_since_checkpoint(ckp.checkpoint_id)
        assert diff["new.py"] == "created"

    def test_diff_nonexistent_checkpoint(self):
        engine = RollbackEngine(self._tmpdir)
        diff = engine.diff_since_checkpoint("CKP-nope")
        assert diff == {}


# ══════════════════════════════════════════════════════════════════════
# Engine summary tests
# ══════════════════════════════════════════════════════════════════════


class TestEngineSummary:
    """Tests for the engine summary diagnostic."""

    def test_summary_empty(self):
        engine = RollbackEngine()
        s = engine.summary()
        assert s["checkpoint_count"] == 0
        assert s["checkpoints"] == []

    def test_summary_with_checkpoints(self):
        engine = RollbackEngine()
        snap = FileSnapshot.from_content("a.py", b"v1")
        engine.create_checkpoint_from_snapshots(
            stage=CheckpointStage.PRE_MODIFICATION,
            file_snapshots={"a.py": snap},
            managed_files={"a.py"},
            description="First",
        )
        engine.create_checkpoint_from_snapshots(
            stage=CheckpointStage.MID_EXECUTION,
            file_snapshots={"a.py": snap},
            managed_files={"a.py"},
            description="Second",
        )
        s = engine.summary()
        assert s["checkpoint_count"] == 2
        assert len(s["checkpoints"]) == 2
        assert s["checkpoints"][0]["stage"] == "pre_modification"
        assert s["checkpoints"][0]["description"] == "First"
        assert s["checkpoints"][1]["stage"] == "mid_execution"


# ══════════════════════════════════════════════════════════════════════
# RollbackResult serialisation tests
# ══════════════════════════════════════════════════════════════════════


class TestRollbackResultSerialisation:
    """Tests for RollbackResult.to_dict."""

    def test_to_dict_complete(self):
        result = RollbackResult(
            checkpoint_id="CKP-abc",
            restored_files=["a.py", "b.py"],
            skipped_files=["c.py"],
            conflicts=["d.py"],
            success=True,
            reason="Manual rollback",
            status=RollbackStatus.SUCCESS,
            actions=[
                FileRollbackAction("a.py", FileAction.RESTORED, "OK"),
                FileRollbackAction("c.py", FileAction.SKIPPED_UNRELATED, "Not managed"),
            ],
        )
        d = result.to_dict()
        assert d["checkpoint_id"] == "CKP-abc"
        assert d["restored_files"] == ["a.py", "b.py"]
        assert d["skipped_files"] == ["c.py"]
        assert d["conflicts"] == ["d.py"]
        assert d["success"] is True
        assert d["reason"] == "Manual rollback"
        assert d["status"] == "SUCCESS"
        assert len(d["actions"]) == 2
        assert d["actions"][0]["action"] == "restored"


# ══════════════════════════════════════════════════════════════════════
# Enum coverage tests
# ══════════════════════════════════════════════════════════════════════


class TestEnums:
    """Basic coverage of enum values."""

    def test_checkpoint_stage_values(self):
        assert CheckpointStage.PRE_MODIFICATION.value == "pre_modification"
        assert CheckpointStage.MID_EXECUTION.value == "mid_execution"
        assert CheckpointStage.POST_VERIFICATION.value == "post_verification"
        assert CheckpointStage.PRE_ROLLBACK.value == "pre_rollback"

    def test_rollback_reason_values(self):
        assert RollbackReason.VERIFICATION_FAILURE.value == "verification_failure"
        assert RollbackReason.SCOPE_EXPANSION.value == "scope_expansion"
        assert RollbackReason.DANGEROUS_MODIFICATION.value == "dangerous_modification"
        assert RollbackReason.AGENT_STUCK.value == "agent_stuck"
        assert RollbackReason.TASK_ABANDONED.value == "task_abandoned"
        assert RollbackReason.MANUAL_ROLLBACK.value == "manual_rollback"

    def test_rollback_status_values(self):
        assert RollbackStatus.SUCCESS.value == "SUCCESS"
        assert RollbackStatus.PARTIAL.value == "PARTIAL"
        assert RollbackStatus.FAILED.value == "FAILED"
        assert RollbackStatus.SKIPPED.value == "SKIPPED"

    def test_file_action_values(self):
        assert FileAction.RESTORED.value == "restored"
        assert FileAction.DELETED.value == "deleted"
        assert FileAction.SKIPPED_UNRELATED.value == "skipped_unrelated"
        assert FileAction.SKIPPED_CONFLICT.value == "skipped_conflict"
        assert FileAction.FAILED.value == "failed"
        assert FileAction.CREATED.value == "created"
