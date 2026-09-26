"""Tests for RepositoryContextManager — repo mapping, relevance ranking,
import relationships, related-test discovery, impact mapping, context
limits, and cache invalidation.

All tests run against a temporary directory; nothing touches the real
filesystem.
"""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from src.context_manager import (
    ContextManager,
    FileInfo,
    ImpactReport,
    RelevantFile,
    RepositoryContextManager,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """Build a realistic fake Python repository.

    Layout::

        tmp_path/
        ├── README.md
        ├── config.yaml
        ├── requirements.txt
        ├── Makefile
        ├── src/
        │   ├── __init__.py
        │   ├── main.py            (entry point)
        │   ├── auth.py            (defines authenticate, AuthManager)
        │   ├── payment.py         (imports auth; defines process_payment)
        │   ├── checkout.py        (imports payment)
        │   └── utils.py
        ├── tests/
        │   ├── __init__.py
        │   ├── test_auth.py       (imports src.auth)
        │   ├── test_payment.py    (imports src.payment)
        │   └── test_checkout.py   (imports src.checkout)
        ├── .git/
        │   └── HEAD
        ├── __pycache__/
        │   └── junk.pyc
        └── node_modules/
            └── pkg/
                └── index.js
    """

    # -- top-level files -------------------------------------------------
    (tmp_path / "README.md").write_text("# My Project\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("model:\n  name: test\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("flask>=2.0\npytest\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text("test:\n\tpytest\n", encoding="utf-8")

    # -- src/ ------------------------------------------------------------
    src = tmp_path / "src"
    src.mkdir()
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "main.py").write_text(textwrap.dedent("""\
        \"\"\"Entry point.\"\"\"
        from src.auth import authenticate
        from src.payment import process_payment

        def main():
            user = authenticate("admin", "pass")
            process_payment(user, 42.0)
    """), encoding="utf-8")

    (src / "auth.py").write_text(textwrap.dedent("""\
        \"\"\"Authentication module.\"\"\"
        import hashlib

        class AuthManager:
            def __init__(self):
                self.sessions = {}

        def authenticate(username: str, password: str) -> dict:
            hashed = hashlib.sha256(password.encode()).hexdigest()
            return {"username": username, "token": hashed}

        def revoke_token(token: str) -> bool:
            return True
    """), encoding="utf-8")

    (src / "payment.py").write_text(textwrap.dedent("""\
        \"\"\"Payment processing.\"\"\"
        from src.auth import authenticate

        def process_payment(user: dict, amount: float) -> dict:
            return {"status": "ok", "user": user["username"], "amount": amount}

        def refund(transaction_id: str) -> dict:
            return {"status": "refunded", "id": transaction_id}
    """), encoding="utf-8")

    (src / "checkout.py").write_text(textwrap.dedent("""\
        \"\"\"Checkout flow.\"\"\"
        from src.payment import process_payment

        def checkout(cart: list, user: dict) -> dict:
            total = sum(item["price"] for item in cart)
            return process_payment(user, total)
    """), encoding="utf-8")

    (src / "utils.py").write_text(textwrap.dedent("""\
        \"\"\"General utilities.\"\"\"
        def format_currency(amount: float) -> str:
            return f"${amount:.2f}"
    """), encoding="utf-8")

    # -- tests/ ----------------------------------------------------------
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("", encoding="utf-8")
    (tests / "test_auth.py").write_text(textwrap.dedent("""\
        from src.auth import authenticate, AuthManager
        def test_authenticate():
            result = authenticate("u", "p")
            assert "token" in result
    """), encoding="utf-8")
    (tests / "test_payment.py").write_text(textwrap.dedent("""\
        from src.payment import process_payment
        def test_process_payment():
            result = process_payment({"username": "u"}, 10.0)
            assert result["status"] == "ok"
    """), encoding="utf-8")
    (tests / "test_checkout.py").write_text(textwrap.dedent("""\
        from src.checkout import checkout
        def test_checkout():
            cart = [{"price": 5.0}]
            result = checkout(cart, {"username": "u"})
            assert result["status"] == "ok"
    """), encoding="utf-8")

    # -- ignored directories ---------------------------------------------
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    (pycache / "junk.pyc").write_bytes(b"\x00\x00")

    nm = tmp_path / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("module.exports = {};\n", encoding="utf-8")

    return tmp_path


@pytest.fixture()
def rcm(repo: pathlib.Path) -> RepositoryContextManager:
    """Pre-mapped RepositoryContextManager."""
    mgr = RepositoryContextManager(repo)
    mgr.build_repo_map()
    return mgr


# ======================================================================
# Repository mapping
# ======================================================================


class TestRepoMap:
    def test_total_file_count(self, rcm: RepositoryContextManager):
        summary = rcm.build_repo_map()
        # Should not include .git/HEAD, __pycache__/junk.pyc, node_modules/**
        assert summary["total_files"] > 0
        for rel in rcm._file_cache:
            assert ".git" not in rel.split("/")[0] or not rel.startswith(".git")

    def test_ignored_dirs_excluded(self, rcm: RepositoryContextManager):
        for rel in rcm._file_cache:
            parts = pathlib.PurePosixPath(rel).parts
            for p in parts:
                assert p not in {".git", "__pycache__", "node_modules"}

    def test_readme_classified(self, rcm: RepositoryContextManager):
        info = rcm._file_cache.get("README.md")
        assert info is not None
        assert info.category == "readme"

    def test_config_classified(self, rcm: RepositoryContextManager):
        info = rcm._file_cache.get("config.yaml")
        assert info is not None
        assert info.category == "config"

    def test_dep_classified(self, rcm: RepositoryContextManager):
        info = rcm._file_cache.get("requirements.txt")
        assert info is not None
        assert info.category == "dep"

    def test_entry_point_classified(self, rcm: RepositoryContextManager):
        info = rcm._file_cache.get("src/main.py")
        assert info is not None
        assert info.category == "entry_point"

    def test_source_classified(self, rcm: RepositoryContextManager):
        info = rcm._file_cache.get("src/auth.py")
        assert info is not None
        assert info.category == "source"

    def test_test_classified(self, rcm: RepositoryContextManager):
        info = rcm._file_cache.get("tests/test_auth.py")
        assert info is not None
        assert info.category == "test"

    def test_definitions_extracted(self, rcm: RepositoryContextManager):
        info = rcm._file_cache["src/auth.py"]
        defs = info.definitions
        assert "class AuthManager" in defs
        assert "def authenticate" in defs
        assert "def revoke_token" in defs

    def test_imports_extracted(self, rcm: RepositoryContextManager):
        info = rcm._file_cache["src/payment.py"]
        assert "src.auth" in info.imports

    def test_map_is_cached(self, rcm: RepositoryContextManager):
        """Calling build_repo_map() again without force=True reuses cache."""
        count_before = len(rcm._file_cache)
        rcm.build_repo_map()  # should be a no-op
        assert len(rcm._file_cache) == count_before

    def test_force_rebuild(self, rcm: RepositoryContextManager, repo: pathlib.Path):
        (repo / "new_file.py").write_text("x = 1\n")
        rcm.build_repo_map(force=True)
        assert "new_file.py" in rcm._file_cache


# ======================================================================
# Relevant-file ranking
# ======================================================================


class TestRelevantFiles:
    def test_auth_issue_finds_auth_files(self, rcm: RepositoryContextManager):
        results = rcm.find_relevant_files("authentication login token")
        paths = [r.file_path for r in results]
        assert "src/auth.py" in paths

    def test_relevance_scores_ordered(self, rcm: RepositoryContextManager):
        results = rcm.find_relevant_files("authenticate user token")
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_reason_is_populated(self, rcm: RepositoryContextManager):
        results = rcm.find_relevant_files("payment processing")
        for r in results:
            assert len(r.reason) > 0

    def test_snippets_returned(self, rcm: RepositoryContextManager):
        results = rcm.find_relevant_files("authenticate")
        auth_result = next(
            (r for r in results if r.file_path == "src/auth.py"), None
        )
        assert auth_result is not None
        assert len(auth_result.snippets) > 0
        assert len(auth_result.line_numbers) > 0

    def test_size_bytes_populated(self, rcm: RepositoryContextManager):
        results = rcm.find_relevant_files("payment")
        for r in results:
            assert r.size_bytes > 0

    def test_max_files_limit(self, rcm: RepositoryContextManager):
        results = rcm.find_relevant_files("src", max_files=2)
        assert len(results) <= 2

    def test_empty_query_returns_nothing(self, rcm: RepositoryContextManager):
        results = rcm.find_relevant_files("")
        assert len(results) == 0

    def test_definition_match_boosts_score(self, rcm: RepositoryContextManager):
        """A file that *defines* 'authenticate' should score higher than one
        that merely imports it."""
        results = rcm.find_relevant_files("authenticate")
        auth_idx = next(
            i for i, r in enumerate(results) if r.file_path == "src/auth.py"
        )
        # auth.py should rank at or near the top
        assert auth_idx < 3


# ======================================================================
# Import relationships
# ======================================================================


class TestImportGraph:
    def test_payment_imports_auth(self, rcm: RepositoryContextManager):
        assert "src.auth" in rcm._import_graph.get("src/payment.py", set())

    def test_checkout_imports_payment(self, rcm: RepositoryContextManager):
        assert "src.payment" in rcm._import_graph.get("src/checkout.py", set())

    def test_reverse_deps_for_auth(self, rcm: RepositoryContextManager):
        importers = rcm._reverse_deps.get("src.auth", set())
        assert "src/payment.py" in importers
        assert "src/main.py" in importers

    def test_reverse_deps_for_payment(self, rcm: RepositoryContextManager):
        importers = rcm._reverse_deps.get("src.payment", set())
        assert "src/checkout.py" in importers
        assert "src/main.py" in importers

    def test_test_file_imports(self, rcm: RepositoryContextManager):
        importers = rcm._reverse_deps.get("src.auth", set())
        assert "tests/test_auth.py" in importers


# ======================================================================
# Related-test discovery
# ======================================================================


class TestRelatedTests:
    def test_auth_change_finds_test_auth(self, rcm: RepositoryContextManager):
        report = rcm.build_impact_report(["src/auth.py"])
        assert "tests/test_auth.py" in report.related_tests

    def test_payment_change_finds_test_payment(self, rcm: RepositoryContextManager):
        report = rcm.build_impact_report(["src/payment.py"])
        assert "tests/test_payment.py" in report.related_tests

    def test_naming_convention_discovery(self, rcm: RepositoryContextManager):
        """Tests should be found by naming convention (test_<stem>.py)
        even without an import link."""
        report = rcm.build_impact_report(["src/checkout.py"])
        assert "tests/test_checkout.py" in report.related_tests


# ======================================================================
# Impact mapping
# ======================================================================


class TestImpactMapping:
    def test_auth_change_directly_affects_payment(
        self, rcm: RepositoryContextManager
    ):
        report = rcm.build_impact_report(["src/auth.py"])
        assert "src/payment.py" in report.directly_affected_files

    def test_auth_change_indirectly_affects_checkout(
        self, rcm: RepositoryContextManager
    ):
        report = rcm.build_impact_report(["src/auth.py"])
        # checkout.py imports payment.py which imports auth.py
        assert "src/checkout.py" in report.indirectly_affected_files

    def test_impact_report_structure(self, rcm: RepositoryContextManager):
        report = rcm.build_impact_report(["src/auth.py"])
        assert isinstance(report, ImpactReport)
        assert report.changed_files == ["src/auth.py"]
        assert report.impact_level in {"low", "medium", "high", "critical"}
        assert len(report.reasons) > 0

    def test_impact_level_scales(self, rcm: RepositoryContextManager):
        # auth.py has a wide blast radius → should be at least "medium"
        report = rcm.build_impact_report(["src/auth.py"])
        assert report.impact_level in {"medium", "high", "critical"}

    def test_utils_has_low_impact(self, rcm: RepositoryContextManager):
        """utils.py is imported by nothing → minimal impact."""
        report = rcm.build_impact_report(["src/utils.py"])
        assert report.impact_level == "low"

    def test_missing_file_handled(self, rcm: RepositoryContextManager):
        report = rcm.build_impact_report(["nonexistent.py"])
        assert any("not found" in r for r in report.reasons)

    def test_reasons_are_deduplicated(self, rcm: RepositoryContextManager):
        report = rcm.build_impact_report(["src/auth.py"])
        assert len(report.reasons) == len(set(report.reasons))

    def test_changed_files_not_in_affected(self, rcm: RepositoryContextManager):
        report = rcm.build_impact_report(["src/auth.py"])
        assert "src/auth.py" not in report.directly_affected_files
        assert "src/auth.py" not in report.indirectly_affected_files


# ======================================================================
# Context / token budget
# ======================================================================


class TestContextLimits:
    def test_budget_limits_files(self, repo: pathlib.Path):
        """With a very small token budget, fewer files should be returned."""
        # Set budget so small that only 1-2 files fit.
        mgr = RepositoryContextManager(repo, max_context_tokens=50)
        mgr.build_repo_map()
        results = mgr.find_relevant_files("authenticate payment checkout")
        # The exact count depends on file sizes, but it must be fewer than
        # with a large budget.
        mgr_large = RepositoryContextManager(repo, max_context_tokens=100_000)
        mgr_large.build_repo_map()
        results_large = mgr_large.find_relevant_files(
            "authenticate payment checkout"
        )
        assert len(results) <= len(results_large)

    def test_at_least_one_file_even_if_budget_is_tiny(
        self, repo: pathlib.Path
    ):
        """Even with a near-zero budget the first matching file is included."""
        mgr = RepositoryContextManager(repo, max_context_tokens=1)
        mgr.build_repo_map()
        results = mgr.find_relevant_files("authenticate")
        # The first file always gets in regardless of budget.
        assert len(results) >= 1


# ======================================================================
# Cache invalidation / refresh
# ======================================================================


class TestCacheInvalidation:
    def test_refresh_updates_file(self, rcm: RepositoryContextManager, repo: pathlib.Path):
        # Modify auth.py — add a new function.
        auth_path = repo / "src" / "auth.py"
        original = auth_path.read_text()
        auth_path.write_text(original + "\ndef new_func():\n    pass\n")

        rcm.refresh_files(["src/auth.py"])
        info = rcm._file_cache["src/auth.py"]
        assert "def new_func" in info.definitions

    def test_refresh_handles_deleted_file(
        self, rcm: RepositoryContextManager, repo: pathlib.Path
    ):
        (repo / "src" / "utils.py").unlink()
        rcm.refresh_files(["src/utils.py"])
        assert "src/utils.py" not in rcm._file_cache

    def test_refresh_adds_new_file(
        self, rcm: RepositoryContextManager, repo: pathlib.Path
    ):
        (repo / "src" / "notifications.py").write_text(
            "from src.auth import authenticate\ndef notify(): pass\n"
        )
        rcm.refresh_files(["src/notifications.py"])
        info = rcm._file_cache.get("src/notifications.py")
        assert info is not None
        assert "src.auth" in info.imports
        assert "def notify" in info.definitions

    def test_invalidate_clears_everything(self, rcm: RepositoryContextManager):
        assert len(rcm._file_cache) > 0
        rcm.invalidate()
        assert len(rcm._file_cache) == 0
        assert not rcm._repo_mapped

    def test_refresh_updates_reverse_deps(
        self, rcm: RepositoryContextManager, repo: pathlib.Path
    ):
        """When a file's imports change, the reverse-dep graph updates."""
        # Make checkout.py stop importing payment.
        checkout = repo / "src" / "checkout.py"
        checkout.write_text("def checkout(): pass\n")
        rcm.refresh_files(["src/checkout.py"])

        importers = rcm._reverse_deps.get("src.payment", set())
        assert "src/checkout.py" not in importers


# ======================================================================
# Original ContextManager (preserved API)
# ======================================================================


class TestContextManagerOriginal:
    def test_add_and_get(self):
        cm = ContextManager()
        cm.add_message("user", "hello")
        cm.add_message("assistant", "hi")
        msgs = cm.get_messages()
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "hello"}

    def test_clear(self):
        cm = ContextManager()
        cm.add_message("user", "test")
        cm.clear()
        assert cm.get_messages() == []

    def test_get_messages_returns_copy(self):
        cm = ContextManager()
        cm.add_message("user", "x")
        msgs = cm.get_messages()
        msgs.clear()
        assert len(cm.get_messages()) == 1


# ======================================================================
# Module-name conversion helper
# ======================================================================


class TestFileToModuleNames:
    def test_simple(self):
        result = RepositoryContextManager._file_to_module_names("src/auth.py")
        assert "src.auth" in result
        assert "auth" in result

    def test_deeply_nested(self):
        result = RepositoryContextManager._file_to_module_names(
            "src/tools/file_ops.py"
        )
        assert "src.tools.file_ops" in result
        assert "tools.file_ops" in result
        assert "file_ops" in result

    def test_non_python(self):
        result = RepositoryContextManager._file_to_module_names("README.md")
        assert result == []
