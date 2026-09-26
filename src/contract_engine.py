"""Contract Engine — Converts natural-language issues into structured task contracts.

This module is the core differentiator of the AI coding-agent harness.
Before the agent modifies any code, the ContractEngine analyses the issue
and produces a **TaskContract** that describes:

* what must change  (behavioral_requirements, acceptance_criteria)
* what must NOT change  (non_goals, constraints)
* what must be proven  (proof_obligations with individual verification plans)

The contract is preserved throughout the entire task lifecycle.  The final
result reports every proof obligation individually — no obligation is
marked PASS simply because another test passed, and the contract is never
considered complete while any obligation is UNKNOWN.

**Privacy**: chain-of-thought from the model is never stored.  Only
requirements, concise decisions, evidence, and outcomes are persisted.
"""

from __future__ import annotations

import enum
import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ObligationType(enum.Enum):
    """Category of proof obligation."""
    BEHAVIOR = "behavior"
    REGRESSION = "regression"
    CONSTRAINT = "constraint"
    INTERFACE = "interface"
    SAFETY = "safety"
    SCOPE = "scope"


class ObligationStatus(enum.Enum):
    """Current verification state of a proof obligation."""
    PENDING = "PENDING"
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class RiskLevel(enum.Enum):
    """Task-level risk assessment."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProofObligation:
    """A single verifiable claim that the contract demands."""
    id: str
    description: str
    type: ObligationType
    verification_method: str
    status: ObligationStatus = ObligationStatus.PENDING
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "type": self.type.value,
            "verification_method": self.verification_method,
            "status": self.status.value,
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProofObligation":
        """Reconstruct a ProofObligation from a plain dict.

        Raises ``ContractValidationError`` if required fields are missing
        or enum values are invalid.
        """
        required = {"id", "description", "type", "verification_method"}
        missing = required - set(data.keys())
        if missing:
            raise ContractValidationError(
                f"ProofObligation missing required fields: {sorted(missing)}"
            )
        try:
            ob_type = ObligationType(data["type"])
        except ValueError:
            raise ContractValidationError(
                f"Invalid obligation type: {data['type']!r}"
            )

        status_raw = data.get("status", "PENDING")
        try:
            ob_status = ObligationStatus(status_raw)
        except ValueError:
            ob_status = ObligationStatus.PENDING

        return cls(
            id=str(data["id"]),
            description=str(data["description"]),
            type=ob_type,
            verification_method=str(data["verification_method"]),
            status=ob_status,
            evidence=list(data.get("evidence", [])),
        )


@dataclass
class TaskContract:
    """Structured task contract produced before code modification begins."""
    task_goal: str
    behavioral_requirements: list[str]
    acceptance_criteria: list[str]
    constraints: list[str]
    non_goals: list[str]
    affected_area: str
    expected_files: list[str]
    risk_level: RiskLevel
    verification_plan: str
    proof_obligations: list[ProofObligation]

    # Internal metadata
    _contract_id: str = field(default_factory=lambda: f"CTR-{uuid.uuid4().hex[:8]}")
    _original_issue: str = ""
    _ambiguities: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the contract to a JSON-safe dict."""
        return {
            "contract_id": self._contract_id,
            "task_goal": self.task_goal,
            "behavioral_requirements": list(self.behavioral_requirements),
            "acceptance_criteria": list(self.acceptance_criteria),
            "constraints": list(self.constraints),
            "non_goals": list(self.non_goals),
            "affected_area": self.affected_area,
            "expected_files": list(self.expected_files),
            "risk_level": self.risk_level.value,
            "verification_plan": self.verification_plan,
            "proof_obligations": [ob.to_dict() for ob in self.proof_obligations],
            "ambiguities": list(self._ambiguities),
        }

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    def is_complete(self) -> bool:
        """Return ``True`` only if every obligation is PASS or FAIL.

        An obligation with status UNKNOWN means the contract cannot be
        considered complete — the harness must investigate further.
        """
        for ob in self.proof_obligations:
            if ob.status in (ObligationStatus.PENDING, ObligationStatus.UNKNOWN):
                return False
        return len(self.proof_obligations) > 0

    def all_passed(self) -> bool:
        """Return ``True`` only if every obligation is PASS."""
        return (
            len(self.proof_obligations) > 0
            and all(ob.status == ObligationStatus.PASS for ob in self.proof_obligations)
        )

    def summary(self) -> dict[str, Any]:
        """Return a concise status report for every obligation."""
        counts = {s.value: 0 for s in ObligationStatus}
        for ob in self.proof_obligations:
            counts[ob.status.value] += 1
        return {
            "contract_id": self._contract_id,
            "task_goal": self.task_goal,
            "total_obligations": len(self.proof_obligations),
            "status_counts": counts,
            "is_complete": self.is_complete(),
            "all_passed": self.all_passed(),
            "ambiguities": list(self._ambiguities),
            "obligations": [
                {
                    "id": ob.id,
                    "description": ob.description,
                    "status": ob.status.value,
                }
                for ob in self.proof_obligations
            ],
        }

    # ------------------------------------------------------------------
    # Obligation mutation helpers
    # ------------------------------------------------------------------

    def get_obligation(self, obligation_id: str) -> Optional[ProofObligation]:
        """Return the obligation with the given ID, or ``None``."""
        for ob in self.proof_obligations:
            if ob.id == obligation_id:
                return ob
        return None

    def update_obligation(
        self,
        obligation_id: str,
        status: ObligationStatus,
        evidence: str | None = None,
    ) -> bool:
        """Update a single obligation's status and optionally add evidence.

        Returns ``True`` if the obligation was found and updated.
        """
        ob = self.get_obligation(obligation_id)
        if ob is None:
            return False
        ob.status = status
        if evidence:
            ob.evidence.append(evidence)
        return True


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ContractValidationError(Exception):
    """Raised when the model's output cannot be parsed into a valid contract."""


# ---------------------------------------------------------------------------
# Prompt construction (private)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a contract-generation engine for an AI coding agent.

Given a natural-language coding issue, produce a JSON object with these exact keys:
{
  "task_goal": "<one-sentence goal>",
  "behavioral_requirements": ["<requirement>", ...],
  "acceptance_criteria": ["<criterion>", ...],
  "constraints": ["<constraint>", ...],
  "non_goals": ["<what should NOT change>", ...],
  "affected_area": "<subsystem or module name>",
  "expected_files": ["<file path>", ...],
  "risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "verification_plan": "<how to verify the change>",
  "proof_obligations": [
    {
      "id": "OB-<N>",
      "description": "<what must be proven>",
      "type": "behavior" | "regression" | "constraint" | "interface" | "safety" | "scope",
      "verification_method": "<how to verify this specific obligation>"
    },
    ...
  ],
  "ambiguities": ["<any unclear aspect of the issue>", ...]
}

Rules:
- Every non-trivial issue MUST produce at least one regression obligation.
- Every non-trivial issue MUST produce at least one scope obligation.
- If the issue is ambiguous, identify the ambiguity in the "ambiguities" list rather than inventing requirements.
- Do NOT include chain-of-thought.  Output ONLY the JSON object.
- proof_obligations must have at least one entry.
- Each obligation must have a unique "id" starting with "OB-".
"""


def _build_messages(issue: str) -> list[dict[str, Any]]:
    """Build the messages list for the model call."""
    return [
        {"role": "user", "content": f"Generate a task contract for the following issue:\n\n{issue}"},
    ]


# ---------------------------------------------------------------------------
# Response parsing (private)
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from model output.

    Handles both raw JSON and JSON wrapped in markdown code fences.
    Raises ``ContractValidationError`` on failure.
    """
    # Try markdown code block first.
    match = _JSON_BLOCK_RE.search(text)
    candidate = match.group(1).strip() if match else text.strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        # Fall back: find the first { and last } as boundaries.
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ContractValidationError(
                "Model output does not contain valid JSON."
            )
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ContractValidationError(
                f"Failed to parse JSON from model output: {exc}"
            ) from exc

    if not isinstance(parsed, dict):
        raise ContractValidationError(
            f"Expected a JSON object, got {type(parsed).__name__}."
        )
    return parsed


def _validate_contract_data(data: dict[str, Any]) -> None:
    """Validate that the parsed JSON contains all required contract fields.

    Raises ``ContractValidationError`` with a clear message listing
    every problem.
    """
    errors: list[str] = []

    required_strings = ["task_goal", "affected_area", "verification_plan"]
    required_lists = [
        "behavioral_requirements",
        "acceptance_criteria",
        "constraints",
        "non_goals",
        "expected_files",
        "proof_obligations",
    ]

    for key in required_strings:
        if key not in data or not isinstance(data[key], str) or not data[key].strip():
            errors.append(f"Missing or empty required string field: '{key}'")

    for key in required_lists:
        if key not in data or not isinstance(data[key], list):
            errors.append(f"Missing or non-list required field: '{key}'")

    # risk_level validation
    risk_raw = data.get("risk_level", "")
    if risk_raw not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
        errors.append(
            f"Invalid or missing 'risk_level': {risk_raw!r}. "
            "Must be LOW, MEDIUM, HIGH, or CRITICAL."
        )

    # proof_obligations must not be empty
    obligations = data.get("proof_obligations", [])
    if isinstance(obligations, list) and len(obligations) == 0:
        errors.append("'proof_obligations' must contain at least one entry.")

    if errors:
        raise ContractValidationError(
            "Contract validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )


def _parse_contract(data: dict[str, Any], issue: str) -> TaskContract:
    """Convert validated JSON data into a ``TaskContract`` instance.

    Raises ``ContractValidationError`` if obligation parsing fails.
    """
    _validate_contract_data(data)

    obligations: list[ProofObligation] = []
    for i, ob_data in enumerate(data["proof_obligations"]):
        if not isinstance(ob_data, dict):
            raise ContractValidationError(
                f"proof_obligations[{i}] is not an object."
            )
        obligations.append(ProofObligation.from_dict(ob_data))

    # Check for duplicate obligation IDs.
    seen_ids: set[str] = set()
    for ob in obligations:
        if ob.id in seen_ids:
            raise ContractValidationError(
                f"Duplicate proof obligation ID: {ob.id!r}"
            )
        seen_ids.add(ob.id)

    return TaskContract(
        task_goal=data["task_goal"].strip(),
        behavioral_requirements=data["behavioral_requirements"],
        acceptance_criteria=data["acceptance_criteria"],
        constraints=data["constraints"],
        non_goals=data["non_goals"],
        affected_area=data["affected_area"].strip(),
        expected_files=data["expected_files"],
        risk_level=RiskLevel(data["risk_level"]),
        verification_plan=data["verification_plan"].strip(),
        proof_obligations=obligations,
        _original_issue=issue,
        _ambiguities=data.get("ambiguities", []),
    )


# ---------------------------------------------------------------------------
# Contract Engine — public API
# ---------------------------------------------------------------------------

class ContractEngine:
    """Generates and manages task contracts.

    Parameters
    ----------
    model_call : callable
        A function with the same signature as ``model_client.call_model``
        (``messages, tools=None`` → response object with ``.content``).
        Injected so the engine is testable without a real LLM.
    """

    def __init__(self, model_call: Callable[..., Any]) -> None:
        self._model_call = model_call
        self._contracts: dict[str, TaskContract] = {}

    # ------------------------------------------------------------------
    # Contract generation
    # ------------------------------------------------------------------

    def generate_contract(self, issue: str) -> TaskContract:
        """Analyse *issue* and return a validated ``TaskContract``.

        The issue text is sent to the LLM with a carefully scoped system
        prompt.  The response is parsed, validated, and stored internally.

        Raises
        ------
        ContractValidationError
            If the model returns unparseable or structurally invalid output.
        """
        if not issue or not issue.strip():
            raise ContractValidationError("Issue text must not be empty.")

        messages = _build_messages(issue.strip())
        response = self._model_call(
            [{"role": "user", "content": _SYSTEM_PROMPT}] + messages,
        )

        # Extract text content from the model response.
        raw_text = self._extract_response_text(response)
        data = _extract_json(raw_text)
        contract = _parse_contract(data, issue.strip())

        self._contracts[contract._contract_id] = contract
        return contract

    # ------------------------------------------------------------------
    # Contract generation from pre-parsed data (testing / offline)
    # ------------------------------------------------------------------

    def build_contract_from_data(
        self, data: dict[str, Any], issue: str = "",
    ) -> TaskContract:
        """Build a contract from already-parsed JSON data.

        Useful for testing or when the model output has already been
        captured.
        """
        contract = _parse_contract(data, issue)
        self._contracts[contract._contract_id] = contract
        return contract

    # ------------------------------------------------------------------
    # Contract lookup / management
    # ------------------------------------------------------------------

    def get_contract(self, contract_id: str) -> Optional[TaskContract]:
        """Retrieve a previously generated contract by ID."""
        return self._contracts.get(contract_id)

    def list_contracts(self) -> list[str]:
        """Return all stored contract IDs."""
        return list(self._contracts.keys())

    # ------------------------------------------------------------------
    # Obligation updates
    # ------------------------------------------------------------------

    def update_obligation(
        self,
        contract_id: str,
        obligation_id: str,
        status: ObligationStatus,
        evidence: str | None = None,
    ) -> bool:
        """Update an obligation on a stored contract.

        Returns ``True`` if the contract and obligation were found.
        """
        contract = self._contracts.get(contract_id)
        if contract is None:
            return False
        return contract.update_obligation(obligation_id, status, evidence)

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------

    def final_report(self, contract_id: str) -> dict[str, Any]:
        """Produce the final obligation-by-obligation report.

        The report never claims completeness if any obligation is UNKNOWN,
        and never marks an obligation PASS simply because another passed.
        """
        contract = self._contracts.get(contract_id)
        if contract is None:
            return {"error": f"Contract not found: {contract_id!r}"}
        return contract.summary()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """Pull the text string out of a model response object.

        Supports both Anthropic-style (response.content[0].text) and
        plain-string responses (for testing).
        """
        if isinstance(response, str):
            return response

        # Anthropic message object.
        if hasattr(response, "content"):
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text

        raise ContractValidationError(
            "Could not extract text from model response."
        )
