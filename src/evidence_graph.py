"""Evidence Graph — Connects requirements to executable evidence.

Builds a directed evidence graph:

    Requirement
        ↓
    Proof Obligation
        ↓
    Code Location
        ↓
    Change
        ↓
    Verification
        ↓
    Evidence
        ↓
    Falsification

Each node carries structured metadata.  Edges represent causal or
logical dependencies.  Evidence freshness is tracked via deterministic
file hashing — stale evidence is **never** allowed to support a final
VERIFIED result.

Language discipline:
    ✓  "evidence is fresh / stale / invalidated"
    ✗  "proven correct"
"""

from __future__ import annotations

import enum
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

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
    VerificationResult,
    VerificationStatus,
)
from src.falsifier import (
    FalsificationReport,
    FalsificationResult,
    FalsificationStatus,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class NodeType(enum.Enum):
    """Type of node in the evidence graph."""
    REQUIREMENT = "requirement"
    OBLIGATION = "obligation"
    CODE_LOCATION = "code_location"
    CHANGE = "change"
    VERIFICATION = "verification"
    EVIDENCE = "evidence"
    FALSIFICATION = "falsification"


class EvidenceFreshness(enum.Enum):
    """Freshness state of a piece of evidence."""
    FRESH = "FRESH"
    STALE = "STALE"
    INVALIDATED = "INVALIDATED"
    UNKNOWN = "UNKNOWN"


class EvidenceVerdict(enum.Enum):
    """Overall verdict of the evidence graph for a task."""
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    STALE = "STALE"
    INCOMPLETE = "INCOMPLETE"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    """Base node in the evidence graph."""
    node_id: str
    node_type: NodeType
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "label": self.label,
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp,
        }


@dataclass
class RequirementNode(GraphNode):
    """A behavioral requirement from the task contract."""
    requirement_text: str = ""

    def __post_init__(self):
        self.node_type = NodeType.REQUIREMENT

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["requirement_text"] = self.requirement_text
        return d


@dataclass
class ObligationNode(GraphNode):
    """A proof obligation derived from a requirement."""
    obligation_id: str = ""
    obligation_type: str = ""
    status: str = "PENDING"
    verification_method: str = ""

    def __post_init__(self):
        self.node_type = NodeType.OBLIGATION

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["obligation_id"] = self.obligation_id
        d["obligation_type"] = self.obligation_type
        d["status"] = self.status
        d["verification_method"] = self.verification_method
        return d


@dataclass
class CodeLocationNode(GraphNode):
    """A code location (file, function, class) relevant to an obligation."""
    file_path: str = ""
    function_name: str = ""
    class_name: str = ""
    line_range: tuple[int, int] | None = None

    def __post_init__(self):
        self.node_type = NodeType.CODE_LOCATION

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["file_path"] = self.file_path
        d["function_name"] = self.function_name
        d["class_name"] = self.class_name
        if self.line_range:
            d["line_range"] = list(self.line_range)
        return d


@dataclass
class ChangeNode(GraphNode):
    """A change made to a code location."""
    file_path: str = ""
    change_type: str = "modified"  # modified, added, deleted
    fingerprint: FileFingerprint | None = None

    def __post_init__(self):
        self.node_type = NodeType.CHANGE

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["file_path"] = self.file_path
        d["change_type"] = self.change_type
        if self.fingerprint:
            d["fingerprint"] = self.fingerprint.to_dict()
        return d


@dataclass
class VerificationNode(GraphNode):
    """A verification action (test run, check) that produces evidence."""
    check_id: str = ""
    level: int = 0
    command: str = ""
    status: str = "UNKNOWN"
    duration: float = 0.0

    def __post_init__(self):
        self.node_type = NodeType.VERIFICATION

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["check_id"] = self.check_id
        d["level"] = self.level
        d["command"] = self.command
        d["status"] = self.status
        d["duration"] = self.duration
        return d


@dataclass
class EvidenceNode(GraphNode):
    """Concrete evidence produced by a verification action."""
    evidence_text: str = ""
    freshness: EvidenceFreshness = EvidenceFreshness.UNKNOWN
    fingerprints: list[FileFingerprint] = field(default_factory=list)
    recorded_at: float = field(default_factory=time.time)

    def __post_init__(self):
        self.node_type = NodeType.EVIDENCE

    def is_fresh(self, current_fingerprints: dict[str, str]) -> bool:
        """Check if evidence is fresh against current file hashes.

        Returns ``True`` only if every fingerprinted file still matches.
        """
        for fp in self.fingerprints:
            current = current_fingerprints.get(fp.path)
            if current is None:
                # File deleted → stale.
                return False
            if current != fp.sha256:
                return False
        return True

    def invalidate(self, reason: str = "") -> None:
        """Mark this evidence as invalidated."""
        self.freshness = EvidenceFreshness.INVALIDATED
        if reason:
            self.metadata["invalidation_reason"] = reason

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["evidence_text"] = self.evidence_text
        d["freshness"] = self.freshness.value
        d["fingerprints"] = [fp.to_dict() for fp in self.fingerprints]
        d["recorded_at"] = self.recorded_at
        return d


@dataclass
class FalsificationNode(GraphNode):
    """Falsification result for an obligation."""
    falsification_status: str = "NOT_APPLICABLE"
    cases_attempted: int = 0
    counterexamples_found: int = 0

    def __post_init__(self):
        self.node_type = NodeType.FALSIFICATION

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["falsification_status"] = self.falsification_status
        d["cases_attempted"] = self.cases_attempted
        d["counterexamples_found"] = self.counterexamples_found
        return d


# ---------------------------------------------------------------------------
# Graph edge
# ---------------------------------------------------------------------------

@dataclass
class GraphEdge:
    """Directed edge between two graph nodes."""
    source_id: str
    target_id: str
    relation: str  # e.g. "derives", "changes", "verifies", "produces", "falsifies"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Obligation proof report (single obligation)
# ---------------------------------------------------------------------------

@dataclass
class ObligationProofReport:
    """Concise proof report for a single proof obligation."""
    requirement: str
    obligation_id: str
    obligation_description: str
    relevant_files: list[str]
    verification_performed: str
    result: str
    evidence: list[str]
    evidence_freshness: str
    falsification_result: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement": self.requirement,
            "obligation_id": self.obligation_id,
            "obligation_description": self.obligation_description,
            "relevant_files": list(self.relevant_files),
            "verification_performed": self.verification_performed,
            "result": self.result,
            "evidence": list(self.evidence),
            "evidence_freshness": self.evidence_freshness,
            "falsification_result": self.falsification_result,
        }


# ---------------------------------------------------------------------------
# Evidence proof report (full task)
# ---------------------------------------------------------------------------

@dataclass
class EvidenceReport:
    """Complete evidence report for a task — no hidden chain-of-thought."""
    task_goal: str
    verdict: EvidenceVerdict
    obligations: list[ObligationProofReport]
    total_obligations: int
    obligations_verified: int
    obligations_failed: int
    obligations_stale: int
    has_stale_evidence: bool
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_goal": self.task_goal,
            "verdict": self.verdict.value,
            "obligations": [o.to_dict() for o in self.obligations],
            "total_obligations": self.total_obligations,
            "obligations_verified": self.obligations_verified,
            "obligations_failed": self.obligations_failed,
            "obligations_stale": self.obligations_stale,
            "has_stale_evidence": self.has_stale_evidence,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Evidence Graph
# ---------------------------------------------------------------------------

class EvidenceGraph:
    """Directed graph connecting requirements → obligations → code →
    changes → verifications → evidence → falsification.

    The graph enforces evidence freshness: stale evidence is never
    allowed to support a final VERIFIED verdict.

    Parameters
    ----------
    contract : TaskContract
        The task contract whose obligations form the backbone.
    """

    def __init__(self, contract: TaskContract) -> None:
        self._contract = contract
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []
        self._requirement_to_obligations: dict[str, list[str]] = {}
        self._obligation_to_code: dict[str, list[str]] = {}
        self._obligation_to_evidence: dict[str, list[str]] = {}
        self._obligation_to_falsification: dict[str, str | None] = {}

        # Seed the graph from the contract.
        self._build_from_contract()

    # ------------------------------------------------------------------
    # Construction from contract
    # ------------------------------------------------------------------

    def _build_from_contract(self) -> None:
        """Populate requirement and obligation nodes from the contract."""
        for i, req_text in enumerate(self._contract.behavioral_requirements):
            req_id = f"REQ-{i + 1}"
            req_node = RequirementNode(
                node_id=req_id,
                node_type=NodeType.REQUIREMENT,
                label=req_text[:80],
                requirement_text=req_text,
            )
            self._add_node(req_node)
            self._requirement_to_obligations[req_id] = []

        for ob in self._contract.proof_obligations:
            ob_node = ObligationNode(
                node_id=ob.id,
                node_type=NodeType.OBLIGATION,
                label=ob.description[:80],
                obligation_id=ob.id,
                obligation_type=ob.type.value,
                status=ob.status.value,
                verification_method=ob.verification_method,
            )
            self._add_node(ob_node)
            self._obligation_to_code[ob.id] = []
            self._obligation_to_evidence[ob.id] = []
            self._obligation_to_falsification[ob.id] = None

        # Link requirements to obligations (heuristic: sequential mapping
        # or many-to-many if counts don't match).
        req_ids = list(self._requirement_to_obligations.keys())
        ob_ids = [ob.id for ob in self._contract.proof_obligations]

        if len(req_ids) == len(ob_ids):
            for req_id, ob_id in zip(req_ids, ob_ids):
                self._add_edge(req_id, ob_id, "derives")
                self._requirement_to_obligations[req_id].append(ob_id)
        elif req_ids and ob_ids:
            # Map all obligations to all requirements.
            for req_id in req_ids:
                for ob_id in ob_ids:
                    self._add_edge(req_id, ob_id, "derives")
                    self._requirement_to_obligations[req_id].append(ob_id)

    # ------------------------------------------------------------------
    # Node / edge management
    # ------------------------------------------------------------------

    def _add_node(self, node: GraphNode) -> None:
        self._nodes[node.node_id] = node

    def _add_edge(self, source: str, target: str, relation: str, **meta) -> None:
        self._edges.append(GraphEdge(
            source_id=source,
            target_id=target,
            relation=relation,
            metadata=meta,
        ))

    def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    @property
    def nodes(self) -> dict[str, GraphNode]:
        return dict(self._nodes)

    @property
    def edges(self) -> list[GraphEdge]:
        return list(self._edges)

    # ------------------------------------------------------------------
    # Add code locations
    # ------------------------------------------------------------------

    def add_code_location(
        self,
        obligation_id: str,
        file_path: str,
        *,
        function_name: str = "",
        class_name: str = "",
        line_range: tuple[int, int] | None = None,
    ) -> str:
        """Associate a code location with an obligation.

        Returns the node_id of the created CodeLocationNode.
        """
        node_id = f"CODE-{obligation_id}-{file_path.replace('/', '_')}"
        if function_name:
            node_id += f"-{function_name}"

        label = file_path
        if function_name:
            label += f":{function_name}()"

        node = CodeLocationNode(
            node_id=node_id,
            node_type=NodeType.CODE_LOCATION,
            label=label,
            file_path=file_path,
            function_name=function_name,
            class_name=class_name,
            line_range=line_range,
        )
        self._add_node(node)
        self._add_edge(obligation_id, node_id, "located_in")
        if obligation_id in self._obligation_to_code:
            self._obligation_to_code[obligation_id].append(node_id)
        return node_id

    # ------------------------------------------------------------------
    # Add changes
    # ------------------------------------------------------------------

    def add_change(
        self,
        code_node_id: str,
        file_path: str,
        *,
        change_type: str = "modified",
        fingerprint: FileFingerprint | None = None,
    ) -> str:
        """Record a change to a code location.

        Returns the node_id of the created ChangeNode.
        """
        node_id = f"CHG-{file_path.replace('/', '_')}-{uuid.uuid4().hex[:6]}"
        node = ChangeNode(
            node_id=node_id,
            node_type=NodeType.CHANGE,
            label=f"{change_type}: {file_path}",
            file_path=file_path,
            change_type=change_type,
            fingerprint=fingerprint,
        )
        self._add_node(node)
        self._add_edge(code_node_id, node_id, "changed")
        return node_id

    # ------------------------------------------------------------------
    # Add verification
    # ------------------------------------------------------------------

    def add_verification(
        self,
        obligation_id: str,
        check: VerificationCheck,
    ) -> str:
        """Record a verification check linked to an obligation.

        Returns the node_id of the created VerificationNode.
        """
        node_id = f"VER-{check.check_id}"
        node = VerificationNode(
            node_id=node_id,
            node_type=NodeType.VERIFICATION,
            label=check.description[:80],
            check_id=check.check_id,
            level=check.level.value,
            command=check.command or "",
            status=check.status.value,
            duration=check.duration,
        )
        self._add_node(node)
        self._add_edge(obligation_id, node_id, "verified_by")
        return node_id

    # ------------------------------------------------------------------
    # Add evidence
    # ------------------------------------------------------------------

    def add_evidence(
        self,
        verification_node_id: str,
        obligation_id: str,
        evidence_text: str,
        *,
        fingerprints: list[FileFingerprint] | None = None,
        freshness: EvidenceFreshness = EvidenceFreshness.FRESH,
    ) -> str:
        """Record evidence produced by a verification.

        Returns the node_id of the created EvidenceNode.
        """
        node_id = f"EV-{obligation_id}-{uuid.uuid4().hex[:6]}"
        node = EvidenceNode(
            node_id=node_id,
            node_type=NodeType.EVIDENCE,
            label=evidence_text[:80],
            evidence_text=evidence_text,
            freshness=freshness,
            fingerprints=fingerprints or [],
        )
        self._add_node(node)
        self._add_edge(verification_node_id, node_id, "produces")
        if obligation_id in self._obligation_to_evidence:
            self._obligation_to_evidence[obligation_id].append(node_id)
        return node_id

    # ------------------------------------------------------------------
    # Add falsification
    # ------------------------------------------------------------------

    def add_falsification(
        self,
        obligation_id: str,
        result: FalsificationResult,
    ) -> str:
        """Record a falsification result for an obligation.

        Returns the node_id of the created FalsificationNode.
        """
        node_id = f"FALS-{obligation_id}"
        label = f"falsification: {result.status.value}"
        if result.counterexamples_found > 0:
            label += f" ({result.counterexamples_found} counterexamples)"

        node = FalsificationNode(
            node_id=node_id,
            node_type=NodeType.FALSIFICATION,
            label=label,
            falsification_status=result.status.value,
            cases_attempted=result.cases_attempted,
            counterexamples_found=result.counterexamples_found,
        )
        self._add_node(node)

        # Link from the latest evidence node(s) for this obligation.
        evidence_ids = self._obligation_to_evidence.get(obligation_id, [])
        if evidence_ids:
            self._add_edge(evidence_ids[-1], node_id, "falsified_by")
        else:
            self._add_edge(obligation_id, node_id, "falsified_by")

        self._obligation_to_falsification[obligation_id] = node_id
        return node_id

    # ------------------------------------------------------------------
    # Populate from results
    # ------------------------------------------------------------------

    def populate_from_verification(
        self,
        result: VerificationResult,
        *,
        changed_files: list[str] | None = None,
    ) -> None:
        """Bulk-populate the graph from a VerificationResult.

        Links verification checks and evidence to obligations.
        Also creates change nodes for changed files.
        """
        changed = changed_files or result.changed_files

        # Create change nodes for each changed file.
        change_nodes: dict[str, str] = {}
        for fp_data in result.fingerprints:
            # Find or create a code node for this file.
            code_id = None
            for ob_id, code_ids in self._obligation_to_code.items():
                for cid in code_ids:
                    node = self.get_node(cid)
                    if isinstance(node, CodeLocationNode) and node.file_path == fp_data.path:
                        code_id = cid
                        break
                if code_id:
                    break

            if code_id is None:
                # Create a generic code node for this file.
                for ob in self._contract.proof_obligations:
                    code_id = self.add_code_location(ob.id, fp_data.path)
                    break

            if code_id:
                chg_id = self.add_change(
                    code_id, fp_data.path,
                    fingerprint=fp_data,
                )
                change_nodes[fp_data.path] = chg_id

        # Create verification and evidence nodes from checks.
        for check in result.checks:
            ob_id = check.obligation_id
            if ob_id is None:
                # Try to attach to the first obligation.
                if self._contract.proof_obligations:
                    ob_id = self._contract.proof_obligations[0].id
                else:
                    continue

            ver_id = self.add_verification(ob_id, check)
            if check.evidence:
                self.add_evidence(
                    ver_id, ob_id, check.evidence,
                    fingerprints=list(result.fingerprints),
                    freshness=EvidenceFreshness.FRESH,
                )

    def populate_from_falsification(
        self,
        report: FalsificationReport,
    ) -> None:
        """Bulk-populate falsification nodes from a FalsificationReport."""
        for fr_result in report.results:
            if fr_result.obligation_id in self._obligation_to_falsification:
                self.add_falsification(fr_result.obligation_id, fr_result)

    # ------------------------------------------------------------------
    # Evidence freshness
    # ------------------------------------------------------------------

    def check_freshness(
        self,
        current_fingerprints: dict[str, str],
    ) -> dict[str, EvidenceFreshness]:
        """Check freshness of all evidence nodes.

        Parameters
        ----------
        current_fingerprints : dict
            Mapping of file path → current SHA-256 hash.

        Returns
        -------
        dict
            Mapping of evidence node_id → freshness status.
        """
        results: dict[str, EvidenceFreshness] = {}

        for node_id, node in self._nodes.items():
            if not isinstance(node, EvidenceNode):
                continue

            if not node.fingerprints:
                # No fingerprints → freshness unknown.
                results[node_id] = EvidenceFreshness.UNKNOWN
                continue

            if node.freshness == EvidenceFreshness.INVALIDATED:
                results[node_id] = EvidenceFreshness.INVALIDATED
                continue

            if node.is_fresh(current_fingerprints):
                node.freshness = EvidenceFreshness.FRESH
                results[node_id] = EvidenceFreshness.FRESH
            else:
                node.freshness = EvidenceFreshness.STALE
                results[node_id] = EvidenceFreshness.STALE

        return results

    def invalidate_stale_evidence(
        self,
        current_fingerprints: dict[str, str],
    ) -> list[str]:
        """Invalidate all evidence whose fingerprints don't match.

        Returns a list of invalidated evidence node IDs.
        """
        freshness_map = self.check_freshness(current_fingerprints)
        invalidated: list[str] = []

        for node_id, freshness in freshness_map.items():
            if freshness == EvidenceFreshness.STALE:
                node = self._nodes[node_id]
                if isinstance(node, EvidenceNode):
                    node.invalidate("Source files changed after evidence was recorded")
                    invalidated.append(node_id)

        return invalidated

    def has_stale_evidence(
        self,
        current_fingerprints: dict[str, str],
    ) -> bool:
        """Return ``True`` if any evidence node is stale or invalidated."""
        freshness_map = self.check_freshness(current_fingerprints)
        return any(
            f in (EvidenceFreshness.STALE, EvidenceFreshness.INVALIDATED)
            for f in freshness_map.values()
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_obligation_chain(self, obligation_id: str) -> dict[str, Any]:
        """Return the full evidence chain for a single obligation.

        Returns a dict with keys: requirement, obligation, code_locations,
        changes, verifications, evidence, falsification.
        """
        chain: dict[str, Any] = {
            "obligation_id": obligation_id,
            "requirement": None,
            "obligation": None,
            "code_locations": [],
            "changes": [],
            "verifications": [],
            "evidence": [],
            "falsification": None,
        }

        # Find the obligation node.
        ob_node = self.get_node(obligation_id)
        if isinstance(ob_node, ObligationNode):
            chain["obligation"] = ob_node.to_dict()

        # Find the requirement that derives this obligation.
        for req_id, ob_ids in self._requirement_to_obligations.items():
            if obligation_id in ob_ids:
                req_node = self.get_node(req_id)
                if req_node:
                    chain["requirement"] = req_node.to_dict()
                break

        # Code locations.
        for code_id in self._obligation_to_code.get(obligation_id, []):
            node = self.get_node(code_id)
            if node:
                chain["code_locations"].append(node.to_dict())

        # Walk edges to find changes, verifications, evidence.
        for edge in self._edges:
            target = self.get_node(edge.target_id)
            if target is None:
                continue

            if isinstance(target, ChangeNode) and edge.source_id in [
                c for c in self._obligation_to_code.get(obligation_id, [])
            ]:
                chain["changes"].append(target.to_dict())

            if isinstance(target, VerificationNode) and edge.source_id == obligation_id:
                chain["verifications"].append(target.to_dict())

            if isinstance(target, EvidenceNode) and edge.target_id in (
                self._obligation_to_evidence.get(obligation_id, [])
            ):
                chain["evidence"].append(target.to_dict())

        # Falsification.
        fals_id = self._obligation_to_falsification.get(obligation_id)
        if fals_id:
            fals_node = self.get_node(fals_id)
            if fals_node:
                chain["falsification"] = fals_node.to_dict()

        return chain

    def nodes_by_type(self, node_type: NodeType) -> list[GraphNode]:
        """Return all nodes of a given type."""
        return [n for n in self._nodes.values() if n.node_type == node_type]

    # ------------------------------------------------------------------
    # Evidence report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        current_fingerprints: dict[str, str] | None = None,
    ) -> EvidenceReport:
        """Generate a concise evidence report for the task.

        Contains: requirement, obligation, relevant files, verification
        performed, result, evidence, evidence freshness, falsification.

        No hidden chain-of-thought is included.
        """
        # If fingerprints provided, check freshness first.
        if current_fingerprints is not None:
            self.check_freshness(current_fingerprints)

        ob_reports: list[ObligationProofReport] = []
        verified_count = 0
        failed_count = 0
        stale_count = 0
        any_stale = False

        for ob in self._contract.proof_obligations:
            chain = self.get_obligation_chain(ob.id)

            # Requirement text.
            req_text = ""
            if chain["requirement"]:
                req_text = chain["requirement"].get("requirement_text", "")

            # Relevant files.
            relevant_files = list({
                loc.get("file_path", "")
                for loc in chain["code_locations"]
                if loc.get("file_path")
            })

            # Verification performed.
            ver_descriptions = [
                v.get("command") or v.get("label", "")
                for v in chain["verifications"]
            ]
            verification_performed = "; ".join(ver_descriptions) if ver_descriptions else "none"

            # Result (from obligation status on the contract).
            result = ob.status.value

            # Evidence texts.
            evidence_texts = [
                e.get("evidence_text", "")
                for e in chain["evidence"]
                if e.get("evidence_text")
            ]

            # Evidence freshness — worst case across all evidence for this obligation.
            freshness_values = [
                e.get("freshness", "UNKNOWN")
                for e in chain["evidence"]
            ]
            if not freshness_values:
                evidence_freshness = "UNKNOWN"
            elif "INVALIDATED" in freshness_values:
                evidence_freshness = "INVALIDATED"
                any_stale = True
                stale_count += 1
            elif "STALE" in freshness_values:
                evidence_freshness = "STALE"
                any_stale = True
                stale_count += 1
            else:
                evidence_freshness = "FRESH"

            # Falsification result.
            fals_text = "not performed"
            if chain["falsification"]:
                fs = chain["falsification"].get("falsification_status", "NOT_APPLICABLE")
                cases = chain["falsification"].get("cases_attempted", 0)
                cex = chain["falsification"].get("counterexamples_found", 0)
                if fs == "NOT_FALSIFIED":
                    fals_text = f"survived ({cases} cases, 0 counterexamples)"
                elif fs == "FALSIFIED":
                    fals_text = f"FALSIFIED ({cex} counterexamples in {cases} cases)"
                elif fs == "UNKNOWN":
                    fals_text = "unknown"
                else:
                    fals_text = "not applicable"

            # Count verdicts.
            if ob.status == ObligationStatus.PASS and evidence_freshness == "FRESH":
                verified_count += 1
            elif ob.status == ObligationStatus.FAIL:
                failed_count += 1

            ob_reports.append(ObligationProofReport(
                requirement=req_text,
                obligation_id=ob.id,
                obligation_description=ob.description,
                relevant_files=relevant_files,
                verification_performed=verification_performed,
                result=result,
                evidence=evidence_texts,
                evidence_freshness=evidence_freshness,
                falsification_result=fals_text,
            ))

        # Overall verdict.
        total = len(self._contract.proof_obligations)
        if any_stale:
            verdict = EvidenceVerdict.STALE
        elif failed_count > 0:
            verdict = EvidenceVerdict.FAILED
        elif verified_count == total and total > 0:
            verdict = EvidenceVerdict.VERIFIED
        elif verified_count > 0 or total == 0:
            verdict = EvidenceVerdict.INCOMPLETE
        else:
            verdict = EvidenceVerdict.UNKNOWN

        return EvidenceReport(
            task_goal=self._contract.task_goal,
            verdict=verdict,
            obligations=ob_reports,
            total_obligations=total,
            obligations_verified=verified_count,
            obligations_failed=failed_count,
            obligations_stale=stale_count,
            has_stale_evidence=any_stale,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the entire graph to a JSON-safe dict."""
        return {
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
            "edges": [e.to_dict() for e in self._edges],
            "requirement_to_obligations": dict(self._requirement_to_obligations),
            "obligation_to_code": dict(self._obligation_to_code),
            "obligation_to_evidence": dict(self._obligation_to_evidence),
            "obligation_to_falsification": dict(self._obligation_to_falsification),
        }

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes
