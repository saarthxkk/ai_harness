"""Tests for src/evidence_graph.py — Evidence Graph."""

import hashlib
import time

import pytest

from src.evidence_graph import (
    ChangeNode,
    CodeLocationNode,
    EvidenceFreshness,
    EvidenceGraph,
    EvidenceNode,
    EvidenceReport,
    EvidenceVerdict,
    FalsificationNode,
    GraphEdge,
    GraphNode,
    NodeType,
    ObligationNode,
    ObligationProofReport,
    RequirementNode,
    VerificationNode,
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
from src.falsifier import (
    FalsificationResult,
    FalsificationStatus,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _sha(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _make_obligation(
    ob_id: str = "OB-1",
    description: str = "Locked users cannot login",
    ob_type: ObligationType = ObligationType.BEHAVIOR,
    method: str = "pytest tests/test_auth.py -x -q",
    status: ObligationStatus = ObligationStatus.PENDING,
) -> ProofObligation:
    return ProofObligation(
        id=ob_id,
        description=description,
        type=ob_type,
        verification_method=method,
        status=status,
    )


def _make_contract(
    obligations: list[ProofObligation] | None = None,
    requirements: list[str] | None = None,
) -> TaskContract:
    if obligations is None:
        obligations = [
            _make_obligation(
                ob_id="OB-1",
                description="Locked users cannot login",
            ),
            _make_obligation(
                ob_id="OB-2",
                description="Unlocked users can still authenticate",
                ob_type=ObligationType.REGRESSION,
            ),
        ]
    if requirements is None:
        requirements = [
            "Locked accounts must be rejected",
            "Unlocked accounts must continue authenticating",
        ]
    return TaskContract(
        task_goal="Fix login for locked accounts",
        behavioral_requirements=requirements,
        acceptance_criteria=["test_locked_user passes"],
        constraints=["Do not modify user model"],
        non_goals=["Do not change password hashing"],
        affected_area="authentication",
        expected_files=["src/auth/login.py"],
        risk_level=RiskLevel.MEDIUM,
        verification_plan="Run auth tests",
        proof_obligations=obligations,
    )


def _make_fingerprint(path: str, content: str = "original") -> FileFingerprint:
    return FileFingerprint.from_content(path, content)


def _make_check(
    check_id: str = "L4-OB-1",
    level: VerificationLevel = VerificationLevel.CONTRACT_VERIFICATION,
    description: str = "Contract obligation: locked users",
    status: CheckStatus = CheckStatus.PASS,
    evidence: str = "Verified by executed test",
    obligation_id: str | None = "OB-1",
    command: str = "pytest tests/test_auth.py -x -q",
) -> VerificationCheck:
    return VerificationCheck(
        check_id=check_id,
        level=level,
        description=description,
        status=status,
        evidence=evidence,
        obligation_id=obligation_id,
        command=command,
    )


def _make_verification_result(
    checks: list[VerificationCheck] | None = None,
    changed_files: list[str] | None = None,
    fingerprints: list[FileFingerprint] | None = None,
) -> VerificationResult:
    if checks is None:
        checks = [_make_check()]
    if changed_files is None:
        changed_files = ["src/auth/login.py"]
    if fingerprints is None:
        fingerprints = [_make_fingerprint("src/auth/login.py")]
    return VerificationResult(
        passed=True,
        status=VerificationStatus.VERIFIED,
        command="pytest tests/ -x -q",
        exit_code=0,
        stdout="all passed",
        stderr="",
        duration=1.5,
        checks=checks,
        failure_summary="",
        evidence=["Verified by executed test"],
        changed_files=changed_files,
        confidence=0.95,
        fingerprints=fingerprints,
    )


def _make_falsification_result(
    obligation_id: str = "OB-1",
    status: FalsificationStatus = FalsificationStatus.NOT_FALSIFIED,
    cases: int = 5,
    counterexamples: int = 0,
) -> FalsificationResult:
    return FalsificationResult(
        obligation_id=obligation_id,
        cases_attempted=cases,
        cases_passed=cases - counterexamples,
        counterexamples_found=counterexamples,
        status=status,
        evidence=[f"Obligation {obligation_id}: {status.value}"],
    )


# ═══════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════


class TestEnums:
    def test_node_types(self):
        assert NodeType.REQUIREMENT.value == "requirement"
        assert NodeType.OBLIGATION.value == "obligation"
        assert NodeType.CODE_LOCATION.value == "code_location"
        assert NodeType.CHANGE.value == "change"
        assert NodeType.VERIFICATION.value == "verification"
        assert NodeType.EVIDENCE.value == "evidence"
        assert NodeType.FALSIFICATION.value == "falsification"

    def test_freshness(self):
        assert EvidenceFreshness.FRESH.value == "FRESH"
        assert EvidenceFreshness.STALE.value == "STALE"
        assert EvidenceFreshness.INVALIDATED.value == "INVALIDATED"
        assert EvidenceFreshness.UNKNOWN.value == "UNKNOWN"

    def test_verdict(self):
        assert EvidenceVerdict.VERIFIED.value == "VERIFIED"
        assert EvidenceVerdict.FAILED.value == "FAILED"
        assert EvidenceVerdict.STALE.value == "STALE"
        assert EvidenceVerdict.INCOMPLETE.value == "INCOMPLETE"
        assert EvidenceVerdict.UNKNOWN.value == "UNKNOWN"


# ═══════════════════════════════════════════════════════════════════
# Node data classes
# ═══════════════════════════════════════════════════════════════════


class TestGraphNode:
    def test_base_node(self):
        n = GraphNode(node_id="N1", node_type=NodeType.REQUIREMENT, label="test")
        d = n.to_dict()
        assert d["node_id"] == "N1"
        assert d["node_type"] == "requirement"
        assert d["label"] == "test"

    def test_requirement_node(self):
        n = RequirementNode(
            node_id="REQ-1",
            node_type=NodeType.REQUIREMENT,
            label="req",
            requirement_text="must do X",
        )
        d = n.to_dict()
        assert d["requirement_text"] == "must do X"
        assert n.node_type == NodeType.REQUIREMENT

    def test_obligation_node(self):
        n = ObligationNode(
            node_id="OB-1",
            node_type=NodeType.OBLIGATION,
            label="ob",
            obligation_id="OB-1",
            obligation_type="behavior",
            status="PENDING",
            verification_method="pytest test.py",
        )
        d = n.to_dict()
        assert d["obligation_id"] == "OB-1"
        assert d["obligation_type"] == "behavior"

    def test_code_location_node(self):
        n = CodeLocationNode(
            node_id="CODE-1",
            node_type=NodeType.CODE_LOCATION,
            label="auth.py:authenticate()",
            file_path="src/auth.py",
            function_name="authenticate",
            line_range=(10, 30),
        )
        d = n.to_dict()
        assert d["file_path"] == "src/auth.py"
        assert d["function_name"] == "authenticate"
        assert d["line_range"] == [10, 30]

    def test_code_location_no_range(self):
        n = CodeLocationNode(
            node_id="CODE-2",
            node_type=NodeType.CODE_LOCATION,
            label="file.py",
            file_path="file.py",
        )
        d = n.to_dict()
        assert "line_range" not in d

    def test_change_node(self):
        fp = _make_fingerprint("a.py")
        n = ChangeNode(
            node_id="CHG-1",
            node_type=NodeType.CHANGE,
            label="modified: a.py",
            file_path="a.py",
            change_type="modified",
            fingerprint=fp,
        )
        d = n.to_dict()
        assert d["file_path"] == "a.py"
        assert d["change_type"] == "modified"
        assert "fingerprint" in d

    def test_verification_node(self):
        n = VerificationNode(
            node_id="VER-1",
            node_type=NodeType.VERIFICATION,
            label="test check",
            check_id="L4-OB-1",
            level=4,
            command="pytest test.py",
            status="PASS",
            duration=0.5,
        )
        d = n.to_dict()
        assert d["check_id"] == "L4-OB-1"
        assert d["level"] == 4

    def test_evidence_node(self):
        fp = _make_fingerprint("a.py")
        n = EvidenceNode(
            node_id="EV-1",
            node_type=NodeType.EVIDENCE,
            label="test evidence",
            evidence_text="Verified by executed test",
            freshness=EvidenceFreshness.FRESH,
            fingerprints=[fp],
        )
        d = n.to_dict()
        assert d["evidence_text"] == "Verified by executed test"
        assert d["freshness"] == "FRESH"
        assert len(d["fingerprints"]) == 1

    def test_evidence_freshness_check(self):
        fp = _make_fingerprint("a.py", "original")
        n = EvidenceNode(
            node_id="EV-1",
            node_type=NodeType.EVIDENCE,
            label="test",
            evidence_text="test",
            fingerprints=[fp],
        )
        # Same hash → fresh.
        assert n.is_fresh({fp.path: fp.sha256})
        # Different hash → not fresh.
        assert not n.is_fresh({fp.path: _sha("changed")})
        # Missing file → not fresh.
        assert not n.is_fresh({})

    def test_evidence_invalidation(self):
        n = EvidenceNode(
            node_id="EV-1",
            node_type=NodeType.EVIDENCE,
            label="test",
            evidence_text="test",
        )
        n.invalidate("file changed")
        assert n.freshness == EvidenceFreshness.INVALIDATED
        assert n.metadata["invalidation_reason"] == "file changed"

    def test_falsification_node(self):
        n = FalsificationNode(
            node_id="FALS-1",
            node_type=NodeType.FALSIFICATION,
            label="not falsified",
            falsification_status="NOT_FALSIFIED",
            cases_attempted=5,
            counterexamples_found=0,
        )
        d = n.to_dict()
        assert d["falsification_status"] == "NOT_FALSIFIED"
        assert d["cases_attempted"] == 5


# ═══════════════════════════════════════════════════════════════════
# GraphEdge
# ═══════════════════════════════════════════════════════════════════


class TestGraphEdge:
    def test_basic(self):
        e = GraphEdge(source_id="A", target_id="B", relation="derives")
        d = e.to_dict()
        assert d["source_id"] == "A"
        assert d["target_id"] == "B"
        assert d["relation"] == "derives"

    def test_with_metadata(self):
        e = GraphEdge(
            source_id="A", target_id="B", relation="x",
            metadata={"weight": 1.0},
        )
        assert e.metadata["weight"] == 1.0


# ═══════════════════════════════════════════════════════════════════
# ObligationProofReport
# ═══════════════════════════════════════════════════════════════════


class TestObligationProofReport:
    def test_to_dict(self):
        r = ObligationProofReport(
            requirement="must reject locked",
            obligation_id="OB-1",
            obligation_description="locked users cannot login",
            relevant_files=["auth.py"],
            verification_performed="pytest test.py",
            result="PASS",
            evidence=["test passed"],
            evidence_freshness="FRESH",
            falsification_result="survived (5 cases, 0 counterexamples)",
        )
        d = r.to_dict()
        assert d["obligation_id"] == "OB-1"
        assert d["evidence_freshness"] == "FRESH"


# ═══════════════════════════════════════════════════════════════════
# Graph creation
# ═══════════════════════════════════════════════════════════════════


class TestGraphCreation:
    def test_creates_from_contract(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        # Should have 2 requirement nodes + 2 obligation nodes.
        req_nodes = graph.nodes_by_type(NodeType.REQUIREMENT)
        ob_nodes = graph.nodes_by_type(NodeType.OBLIGATION)
        assert len(req_nodes) == 2
        assert len(ob_nodes) == 2

    def test_edges_link_requirements_to_obligations(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        edges = graph.edges
        derive_edges = [e for e in edges if e.relation == "derives"]
        # With 2 reqs and 2 obs (same count), should be 1:1 mapping.
        assert len(derive_edges) == 2

    def test_len(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        assert len(graph) == 4  # 2 reqs + 2 obs

    def test_contains(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        assert "OB-1" in graph
        assert "OB-2" in graph
        assert "REQ-1" in graph
        assert "NONEXISTENT" not in graph

    def test_get_node(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        node = graph.get_node("OB-1")
        assert node is not None
        assert isinstance(node, ObligationNode)
        assert node.obligation_id == "OB-1"

    def test_get_nonexistent_node(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        assert graph.get_node("NOPE") is None

    def test_unequal_reqs_and_obs(self):
        """When req count != ob count, many-to-many mapping."""
        contract = _make_contract(
            requirements=["req A", "req B", "req C"],
            obligations=[_make_obligation(ob_id="OB-1")],
        )
        graph = EvidenceGraph(contract)
        derive_edges = [e for e in graph.edges if e.relation == "derives"]
        # 3 reqs × 1 ob = 3 edges.
        assert len(derive_edges) == 3

    def test_no_requirements(self):
        contract = _make_contract(requirements=[])
        graph = EvidenceGraph(contract)
        req_nodes = graph.nodes_by_type(NodeType.REQUIREMENT)
        assert len(req_nodes) == 0

    def test_no_obligations(self):
        contract = _make_contract(obligations=[])
        graph = EvidenceGraph(contract)
        ob_nodes = graph.nodes_by_type(NodeType.OBLIGATION)
        assert len(ob_nodes) == 0


# ═══════════════════════════════════════════════════════════════════
# Requirement-to-obligation mapping
# ═══════════════════════════════════════════════════════════════════


class TestRequirementMapping:
    def test_one_to_one(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        chain = graph.get_obligation_chain("OB-1")
        assert chain["requirement"] is not None
        assert "Locked" in chain["requirement"]["requirement_text"]

    def test_second_obligation(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        chain = graph.get_obligation_chain("OB-2")
        assert chain["requirement"] is not None
        assert "Unlocked" in chain["requirement"]["requirement_text"]

    def test_chain_has_obligation(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        chain = graph.get_obligation_chain("OB-1")
        assert chain["obligation"] is not None
        assert chain["obligation"]["obligation_id"] == "OB-1"


# ═══════════════════════════════════════════════════════════════════
# Code and evidence association
# ═══════════════════════════════════════════════════════════════════


class TestCodeEvidenceAssociation:
    def test_add_code_location(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        node_id = graph.add_code_location(
            "OB-1", "src/auth/login.py",
            function_name="authenticate",
            line_range=(10, 30),
        )
        assert node_id in graph
        node = graph.get_node(node_id)
        assert isinstance(node, CodeLocationNode)
        assert node.file_path == "src/auth/login.py"
        assert node.function_name == "authenticate"

    def test_add_change(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        code_id = graph.add_code_location("OB-1", "src/auth/login.py")
        fp = _make_fingerprint("src/auth/login.py")
        chg_id = graph.add_change(code_id, "src/auth/login.py", fingerprint=fp)
        assert chg_id in graph
        node = graph.get_node(chg_id)
        assert isinstance(node, ChangeNode)
        assert node.file_path == "src/auth/login.py"

    def test_add_verification(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        check = _make_check()
        ver_id = graph.add_verification("OB-1", check)
        assert ver_id in graph
        node = graph.get_node(ver_id)
        assert isinstance(node, VerificationNode)
        assert node.status == "PASS"

    def test_add_evidence(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        check = _make_check()
        ver_id = graph.add_verification("OB-1", check)
        fp = _make_fingerprint("src/auth/login.py")
        ev_id = graph.add_evidence(
            ver_id, "OB-1", "Verified by executed test",
            fingerprints=[fp],
        )
        assert ev_id in graph
        node = graph.get_node(ev_id)
        assert isinstance(node, EvidenceNode)
        assert node.evidence_text == "Verified by executed test"
        assert node.freshness == EvidenceFreshness.FRESH

    def test_add_falsification(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        fr = _make_falsification_result("OB-1")
        fals_id = graph.add_falsification("OB-1", fr)
        assert fals_id in graph
        node = graph.get_node(fals_id)
        assert isinstance(node, FalsificationNode)
        assert node.falsification_status == "NOT_FALSIFIED"

    def test_full_chain(self):
        """Build the complete chain: req → ob → code → change → ver → ev → fals."""
        contract = _make_contract()
        graph = EvidenceGraph(contract)

        code_id = graph.add_code_location(
            "OB-1", "src/auth/login.py", function_name="authenticate",
        )
        fp = _make_fingerprint("src/auth/login.py")
        graph.add_change(code_id, "src/auth/login.py", fingerprint=fp)

        check = _make_check()
        ver_id = graph.add_verification("OB-1", check)
        graph.add_evidence(
            ver_id, "OB-1", "Verified by executed test",
            fingerprints=[fp],
        )
        graph.add_falsification("OB-1", _make_falsification_result("OB-1"))

        chain = graph.get_obligation_chain("OB-1")
        assert chain["requirement"] is not None
        assert chain["obligation"] is not None
        assert len(chain["code_locations"]) >= 1
        assert len(chain["changes"]) >= 1
        assert len(chain["verifications"]) >= 1
        assert len(chain["evidence"]) >= 1
        assert chain["falsification"] is not None


# ═══════════════════════════════════════════════════════════════════
# Populate from verification result
# ═══════════════════════════════════════════════════════════════════


class TestPopulateFromVerification:
    def test_populate_creates_nodes(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        vr = _make_verification_result()
        graph.populate_from_verification(vr)
        # Should have verification + evidence nodes.
        ver_nodes = graph.nodes_by_type(NodeType.VERIFICATION)
        ev_nodes = graph.nodes_by_type(NodeType.EVIDENCE)
        assert len(ver_nodes) >= 1
        assert len(ev_nodes) >= 1

    def test_populate_creates_change_nodes(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        vr = _make_verification_result()
        graph.populate_from_verification(vr)
        chg_nodes = graph.nodes_by_type(NodeType.CHANGE)
        assert len(chg_nodes) >= 1

    def test_populate_from_falsification(self):
        from src.falsifier import FalsificationReport
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        report = FalsificationReport(
            results=[
                _make_falsification_result("OB-1"),
                _make_falsification_result("OB-2"),
            ],
            total_cases_attempted=10,
            total_counterexamples=0,
            overall_status=FalsificationStatus.NOT_FALSIFIED,
            duration=0.5,
            evidence=["all passed"],
        )
        graph.populate_from_falsification(report)
        fals_nodes = graph.nodes_by_type(NodeType.FALSIFICATION)
        assert len(fals_nodes) == 2


# ═══════════════════════════════════════════════════════════════════
# Evidence freshness
# ═══════════════════════════════════════════════════════════════════


class TestEvidenceFreshnessTracking:
    def test_fresh_evidence(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        fp = _make_fingerprint("src/auth/login.py", "original")
        check = _make_check()
        ver_id = graph.add_verification("OB-1", check)
        ev_id = graph.add_evidence(
            ver_id, "OB-1", "test passed",
            fingerprints=[fp],
        )
        current = {fp.path: fp.sha256}
        freshness = graph.check_freshness(current)
        assert freshness[ev_id] == EvidenceFreshness.FRESH

    def test_stale_evidence(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        fp = _make_fingerprint("src/auth/login.py", "original")
        check = _make_check()
        ver_id = graph.add_verification("OB-1", check)
        ev_id = graph.add_evidence(
            ver_id, "OB-1", "test passed",
            fingerprints=[fp],
        )
        # File changed.
        current = {fp.path: _sha("modified")}
        freshness = graph.check_freshness(current)
        assert freshness[ev_id] == EvidenceFreshness.STALE

    def test_deleted_file_is_stale(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        fp = _make_fingerprint("src/auth/login.py", "original")
        check = _make_check()
        ver_id = graph.add_verification("OB-1", check)
        ev_id = graph.add_evidence(
            ver_id, "OB-1", "test passed",
            fingerprints=[fp],
        )
        # File deleted.
        current = {}
        freshness = graph.check_freshness(current)
        assert freshness[ev_id] == EvidenceFreshness.STALE

    def test_no_fingerprints_unknown(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        check = _make_check()
        ver_id = graph.add_verification("OB-1", check)
        ev_id = graph.add_evidence(ver_id, "OB-1", "test passed")
        freshness = graph.check_freshness({})
        assert freshness[ev_id] == EvidenceFreshness.UNKNOWN

    def test_already_invalidated_stays_invalidated(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        fp = _make_fingerprint("a.py", "original")
        check = _make_check()
        ver_id = graph.add_verification("OB-1", check)
        ev_id = graph.add_evidence(
            ver_id, "OB-1", "test",
            fingerprints=[fp],
        )
        # Manually invalidate.
        node = graph.get_node(ev_id)
        node.invalidate("manual")
        freshness = graph.check_freshness({fp.path: fp.sha256})
        assert freshness[ev_id] == EvidenceFreshness.INVALIDATED


# ═══════════════════════════════════════════════════════════════════
# Stale evidence invalidation
# ═══════════════════════════════════════════════════════════════════


class TestStaleEvidenceInvalidation:
    def test_invalidate_stale(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        fp = _make_fingerprint("a.py", "original")
        check = _make_check()
        ver_id = graph.add_verification("OB-1", check)
        ev_id = graph.add_evidence(
            ver_id, "OB-1", "test passed",
            fingerprints=[fp],
        )
        # File changed.
        invalidated = graph.invalidate_stale_evidence({fp.path: _sha("changed")})
        assert ev_id in invalidated
        node = graph.get_node(ev_id)
        assert node.freshness == EvidenceFreshness.INVALIDATED

    def test_fresh_not_invalidated(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        fp = _make_fingerprint("a.py", "original")
        check = _make_check()
        ver_id = graph.add_verification("OB-1", check)
        ev_id = graph.add_evidence(
            ver_id, "OB-1", "test passed",
            fingerprints=[fp],
        )
        invalidated = graph.invalidate_stale_evidence({fp.path: fp.sha256})
        assert ev_id not in invalidated

    def test_has_stale_evidence(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        fp = _make_fingerprint("a.py", "original")
        check = _make_check()
        ver_id = graph.add_verification("OB-1", check)
        graph.add_evidence(
            ver_id, "OB-1", "test passed",
            fingerprints=[fp],
        )
        assert not graph.has_stale_evidence({fp.path: fp.sha256})
        assert graph.has_stale_evidence({fp.path: _sha("changed")})


# ═══════════════════════════════════════════════════════════════════
# Multiple verification runs
# ═══════════════════════════════════════════════════════════════════


class TestMultipleVerificationRuns:
    def test_two_verification_rounds(self):
        """Simulate: verify → code changes → re-verify."""
        contract = _make_contract()
        graph = EvidenceGraph(contract)

        # Round 1: verify.
        fp1 = _make_fingerprint("a.py", "v1")
        check1 = _make_check(check_id="L4-OB-1-R1")
        ver1 = graph.add_verification("OB-1", check1)
        ev1 = graph.add_evidence(
            ver1, "OB-1", "round 1 evidence",
            fingerprints=[fp1],
        )

        # Code changes.
        fp2 = _make_fingerprint("a.py", "v2")
        current = {fp2.path: fp2.sha256}

        # Old evidence should be stale.
        assert graph.has_stale_evidence(current)
        graph.invalidate_stale_evidence(current)
        old_node = graph.get_node(ev1)
        assert old_node.freshness == EvidenceFreshness.INVALIDATED

        # Round 2: re-verify.
        check2 = _make_check(check_id="L4-OB-1-R2")
        ver2 = graph.add_verification("OB-1", check2)
        ev2 = graph.add_evidence(
            ver2, "OB-1", "round 2 evidence",
            fingerprints=[fp2],
        )

        # New evidence is fresh.
        freshness = graph.check_freshness(current)
        assert freshness[ev2] == EvidenceFreshness.FRESH
        # Old evidence is still invalidated.
        assert freshness[ev1] == EvidenceFreshness.INVALIDATED

    def test_multiple_obligations_separate_evidence(self):
        """Each obligation gets its own evidence chain."""
        contract = _make_contract()
        graph = EvidenceGraph(contract)

        fp = _make_fingerprint("a.py", "content")

        for ob_id in ["OB-1", "OB-2"]:
            check = _make_check(
                check_id=f"L4-{ob_id}",
                obligation_id=ob_id,
            )
            ver_id = graph.add_verification(ob_id, check)
            graph.add_evidence(
                ver_id, ob_id, f"evidence for {ob_id}",
                fingerprints=[fp],
            )

        ev_nodes = graph.nodes_by_type(NodeType.EVIDENCE)
        assert len(ev_nodes) == 2

    def test_re_falsification_after_change(self):
        """After code change, add new falsification results."""
        contract = _make_contract()
        graph = EvidenceGraph(contract)

        # First falsification.
        fr1 = _make_falsification_result("OB-1")
        graph.add_falsification("OB-1", fr1)

        fals_nodes = graph.nodes_by_type(NodeType.FALSIFICATION)
        assert len(fals_nodes) == 1
        assert fals_nodes[0].falsification_status == "NOT_FALSIFIED"


# ═══════════════════════════════════════════════════════════════════
# Evidence report
# ═══════════════════════════════════════════════════════════════════


class TestEvidenceReport:
    def test_report_structure(self):
        contract = _make_contract(obligations=[
            _make_obligation(ob_id="OB-1", status=ObligationStatus.PASS),
        ])
        graph = EvidenceGraph(contract)
        fp = _make_fingerprint("a.py")
        check = _make_check()
        ver_id = graph.add_verification("OB-1", check)
        graph.add_evidence(
            ver_id, "OB-1", "verified",
            fingerprints=[fp],
        )
        report = graph.generate_report(
            current_fingerprints={fp.path: fp.sha256},
        )
        assert isinstance(report, EvidenceReport)
        assert report.task_goal == "Fix login for locked accounts"
        assert report.total_obligations == 1

    def test_report_to_dict(self):
        contract = _make_contract(obligations=[
            _make_obligation(ob_id="OB-1", status=ObligationStatus.PASS),
        ])
        graph = EvidenceGraph(contract)
        report = graph.generate_report()
        d = report.to_dict()
        assert "task_goal" in d
        assert "verdict" in d
        assert "obligations" in d
        assert "has_stale_evidence" in d

    def test_verified_verdict(self):
        contract = _make_contract(
            obligations=[
                _make_obligation(ob_id="OB-1", status=ObligationStatus.PASS),
                _make_obligation(ob_id="OB-2", status=ObligationStatus.PASS),
            ],
        )
        graph = EvidenceGraph(contract)
        fp = _make_fingerprint("a.py")
        for ob_id in ["OB-1", "OB-2"]:
            check = _make_check(check_id=f"L4-{ob_id}", obligation_id=ob_id)
            ver_id = graph.add_verification(ob_id, check)
            graph.add_evidence(
                ver_id, ob_id, "passed",
                fingerprints=[fp],
            )
        report = graph.generate_report(
            current_fingerprints={fp.path: fp.sha256},
        )
        assert report.verdict == EvidenceVerdict.VERIFIED
        assert report.obligations_verified == 2

    def test_failed_verdict(self):
        contract = _make_contract(
            obligations=[
                _make_obligation(ob_id="OB-1", status=ObligationStatus.FAIL),
            ],
        )
        graph = EvidenceGraph(contract)
        report = graph.generate_report()
        assert report.verdict == EvidenceVerdict.FAILED

    def test_stale_verdict(self):
        """Stale evidence → STALE verdict, never VERIFIED."""
        contract = _make_contract(
            obligations=[
                _make_obligation(ob_id="OB-1", status=ObligationStatus.PASS),
            ],
        )
        graph = EvidenceGraph(contract)
        fp = _make_fingerprint("a.py", "original")
        check = _make_check()
        ver_id = graph.add_verification("OB-1", check)
        graph.add_evidence(
            ver_id, "OB-1", "passed",
            fingerprints=[fp],
        )
        # File changed → stale.
        report = graph.generate_report(
            current_fingerprints={fp.path: _sha("changed")},
        )
        assert report.verdict == EvidenceVerdict.STALE
        assert report.has_stale_evidence is True
        assert report.obligations_stale == 1

    def test_stale_overrides_verified(self):
        """Even if all obligations PASS, stale evidence → STALE."""
        contract = _make_contract(
            obligations=[
                _make_obligation(ob_id="OB-1", status=ObligationStatus.PASS),
                _make_obligation(ob_id="OB-2", status=ObligationStatus.PASS),
            ],
        )
        graph = EvidenceGraph(contract)
        fp = _make_fingerprint("a.py", "original")
        for ob_id in ["OB-1", "OB-2"]:
            check = _make_check(check_id=f"L4-{ob_id}", obligation_id=ob_id)
            ver_id = graph.add_verification(ob_id, check)
            graph.add_evidence(
                ver_id, ob_id, "passed",
                fingerprints=[fp],
            )
        # One file stale.
        report = graph.generate_report(
            current_fingerprints={fp.path: _sha("changed")},
        )
        assert report.verdict == EvidenceVerdict.STALE

    def test_incomplete_verdict(self):
        contract = _make_contract(
            obligations=[
                _make_obligation(ob_id="OB-1", status=ObligationStatus.PASS),
                _make_obligation(ob_id="OB-2", status=ObligationStatus.PENDING),
            ],
        )
        graph = EvidenceGraph(contract)
        fp = _make_fingerprint("a.py")
        check = _make_check(check_id="L4-OB-1", obligation_id="OB-1")
        ver_id = graph.add_verification("OB-1", check)
        graph.add_evidence(
            ver_id, "OB-1", "passed",
            fingerprints=[fp],
        )
        report = graph.generate_report(
            current_fingerprints={fp.path: fp.sha256},
        )
        assert report.verdict == EvidenceVerdict.INCOMPLETE

    def test_unknown_verdict(self):
        contract = _make_contract(
            obligations=[
                _make_obligation(ob_id="OB-1", status=ObligationStatus.PENDING),
            ],
        )
        graph = EvidenceGraph(contract)
        report = graph.generate_report()
        assert report.verdict == EvidenceVerdict.UNKNOWN

    def test_report_includes_falsification(self):
        contract = _make_contract(
            obligations=[
                _make_obligation(ob_id="OB-1", status=ObligationStatus.PASS),
            ],
        )
        graph = EvidenceGraph(contract)
        fp = _make_fingerprint("a.py")
        check = _make_check()
        ver_id = graph.add_verification("OB-1", check)
        graph.add_evidence(
            ver_id, "OB-1", "passed",
            fingerprints=[fp],
        )
        graph.add_falsification("OB-1", _make_falsification_result("OB-1"))
        report = graph.generate_report(
            current_fingerprints={fp.path: fp.sha256},
        )
        assert "survived" in report.obligations[0].falsification_result

    def test_report_falsified_obligation(self):
        contract = _make_contract(
            obligations=[
                _make_obligation(ob_id="OB-1", status=ObligationStatus.FAIL),
            ],
        )
        graph = EvidenceGraph(contract)
        graph.add_falsification(
            "OB-1",
            _make_falsification_result(
                "OB-1",
                status=FalsificationStatus.FALSIFIED,
                counterexamples=2,
            ),
        )
        report = graph.generate_report()
        assert "FALSIFIED" in report.obligations[0].falsification_result

    def test_report_no_verification(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        report = graph.generate_report()
        for ob_report in report.obligations:
            assert ob_report.verification_performed == "none"

    def test_report_no_chain_of_thought(self):
        """Report should contain no hidden reasoning."""
        contract = _make_contract(
            obligations=[_make_obligation(ob_id="OB-1", status=ObligationStatus.PASS)],
        )
        graph = EvidenceGraph(contract)
        report = graph.generate_report()
        d = report.to_dict()
        serialised = str(d)
        # Must not contain reasoning markers.
        assert "chain-of-thought" not in serialised.lower()
        assert "thinking" not in serialised.lower()


# ═══════════════════════════════════════════════════════════════════
# Serialisation
# ═══════════════════════════════════════════════════════════════════


class TestSerialization:
    def test_to_dict(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        d = graph.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert "requirement_to_obligations" in d
        assert "obligation_to_code" in d
        assert "obligation_to_evidence" in d
        assert "obligation_to_falsification" in d

    def test_to_dict_with_full_chain(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        fp = _make_fingerprint("a.py")
        code_id = graph.add_code_location("OB-1", "a.py")
        graph.add_change(code_id, "a.py", fingerprint=fp)
        check = _make_check()
        ver_id = graph.add_verification("OB-1", check)
        graph.add_evidence(ver_id, "OB-1", "ok", fingerprints=[fp])
        graph.add_falsification("OB-1", _make_falsification_result("OB-1"))

        d = graph.to_dict()
        # Should have many nodes.
        assert len(d["nodes"]) >= 7  # 2 req + 2 ob + code + chg + ver + ev + fals


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_chain_for_nonexistent_obligation(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        chain = graph.get_obligation_chain("NONEXISTENT")
        assert chain["obligation"] is None
        assert chain["requirement"] is None

    def test_nodes_by_type_empty(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        assert graph.nodes_by_type(NodeType.FALSIFICATION) == []

    def test_multiple_code_locations_per_obligation(self):
        contract = _make_contract()
        graph = EvidenceGraph(contract)
        id1 = graph.add_code_location("OB-1", "a.py")
        id2 = graph.add_code_location("OB-1", "b.py")
        assert id1 != id2
        chain = graph.get_obligation_chain("OB-1")
        assert len(chain["code_locations"]) == 2

    def test_evidence_report_with_empty_contract(self):
        contract = _make_contract(obligations=[], requirements=[])
        graph = EvidenceGraph(contract)
        report = graph.generate_report()
        assert report.total_obligations == 0
        assert report.verdict in (EvidenceVerdict.INCOMPLETE, EvidenceVerdict.UNKNOWN)
