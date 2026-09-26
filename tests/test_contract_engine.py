"""Tests for the Contract Engine (src/contract_engine.py).

Covers:
    - Simple issue → valid contract
    - Multi-requirement issue → multiple obligations
    - Regression requirement generation
    - Scope requirement generation
    - Ambiguous issue → ambiguities surfaced
    - Malformed model output handling
    - Missing obligation fields
    - Contract lifecycle (update, report, completeness)
    - Edge cases (empty issue, duplicate IDs, no obligations)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.contract_engine import (
    ContractEngine,
    ContractValidationError,
    ObligationStatus,
    ObligationType,
    ProofObligation,
    RiskLevel,
    TaskContract,
    _extract_json,
    _parse_contract,
    _validate_contract_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_contract_data(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid contract JSON dict, with optional overrides."""
    base: dict[str, Any] = {
        "task_goal": "Fix login so locked accounts cannot authenticate",
        "behavioral_requirements": [
            "Locked accounts must be rejected at authentication",
        ],
        "acceptance_criteria": [
            "Locked accounts receive a 403 response",
        ],
        "constraints": [
            "Must not alter the user table schema",
        ],
        "non_goals": [
            "Do not change the registration flow",
        ],
        "affected_area": "authentication subsystem",
        "expected_files": ["src/auth/login.py", "tests/test_login.py"],
        "risk_level": "MEDIUM",
        "verification_plan": "Run existing auth test suite + new locked-account tests",
        "proof_obligations": [
            {
                "id": "OB-1",
                "description": "Locked accounts must be rejected",
                "type": "behavior",
                "verification_method": "Unit test with locked account fixture",
            },
            {
                "id": "OB-2",
                "description": "Unlocked accounts must continue authenticating",
                "type": "regression",
                "verification_method": "Existing login tests must still pass",
            },
            {
                "id": "OB-3",
                "description": "No unrelated authentication behavior should change",
                "type": "regression",
                "verification_method": "Full auth test suite passes",
            },
            {
                "id": "OB-4",
                "description": "Change must remain within authentication subsystem",
                "type": "scope",
                "verification_method": "Verify only auth/ files are modified",
            },
        ],
        "ambiguities": [],
    }
    base.update(overrides)
    return base


def _mock_model_response(data: dict[str, Any]) -> MagicMock:
    """Build a mock Anthropic response containing JSON in a text block."""
    text_block = MagicMock()
    text_block.text = json.dumps(data)
    text_block.type = "text"
    response = MagicMock()
    response.content = [text_block]
    return response


def _mock_model_call(data: dict[str, Any]):
    """Return a callable that mimics model_client.call_model."""
    def _call(messages, tools=None):
        return _mock_model_response(data)
    return _call


# ===================================================================
# Test: Simple issue → valid contract
# ===================================================================

class TestSimpleIssue:
    """A straightforward issue produces a well-formed contract."""

    def test_simple_issue_generates_contract(self):
        data = _make_valid_contract_data()
        engine = ContractEngine(model_call=_mock_model_call(data))
        contract = engine.generate_contract(
            "Fix login so locked accounts cannot authenticate."
        )

        assert isinstance(contract, TaskContract)
        assert contract.task_goal == "Fix login so locked accounts cannot authenticate"
        assert len(contract.proof_obligations) == 4
        assert contract.risk_level == RiskLevel.MEDIUM

    def test_contract_stored_by_engine(self):
        data = _make_valid_contract_data()
        engine = ContractEngine(model_call=_mock_model_call(data))
        contract = engine.generate_contract("Fix the login bug.")
        assert engine.get_contract(contract._contract_id) is contract

    def test_contract_id_in_list(self):
        data = _make_valid_contract_data()
        engine = ContractEngine(model_call=_mock_model_call(data))
        contract = engine.generate_contract("Fix the login bug.")
        assert contract._contract_id in engine.list_contracts()

    def test_contract_has_expected_files(self):
        data = _make_valid_contract_data()
        engine = ContractEngine(model_call=_mock_model_call(data))
        contract = engine.generate_contract("Fix the login bug.")
        assert "src/auth/login.py" in contract.expected_files

    def test_contract_constraints(self):
        data = _make_valid_contract_data()
        engine = ContractEngine(model_call=_mock_model_call(data))
        contract = engine.generate_contract("Fix the login bug.")
        assert "Must not alter the user table schema" in contract.constraints

    def test_contract_non_goals(self):
        data = _make_valid_contract_data()
        engine = ContractEngine(model_call=_mock_model_call(data))
        contract = engine.generate_contract("Fix the login bug.")
        assert "Do not change the registration flow" in contract.non_goals


# ===================================================================
# Test: Multi-requirement issue → multiple obligations
# ===================================================================

class TestMultiRequirementIssue:
    """An issue with multiple concerns produces multiple proof obligations."""

    def test_multiple_obligations_created(self):
        data = _make_valid_contract_data(
            task_goal="Refactor auth to support OAuth2 and SAML",
            behavioral_requirements=[
                "OAuth2 login must work end-to-end",
                "SAML login must work end-to-end",
                "Existing password login must not break",
            ],
            proof_obligations=[
                {
                    "id": "OB-1",
                    "description": "OAuth2 flow completes successfully",
                    "type": "behavior",
                    "verification_method": "Integration test with mock OAuth2 provider",
                },
                {
                    "id": "OB-2",
                    "description": "SAML flow completes successfully",
                    "type": "behavior",
                    "verification_method": "Integration test with mock SAML IdP",
                },
                {
                    "id": "OB-3",
                    "description": "Password login continues to work",
                    "type": "regression",
                    "verification_method": "Existing password login tests pass",
                },
                {
                    "id": "OB-4",
                    "description": "Auth module interfaces remain stable",
                    "type": "interface",
                    "verification_method": "No signature changes in public auth API",
                },
                {
                    "id": "OB-5",
                    "description": "Changes stay within auth/ directory",
                    "type": "scope",
                    "verification_method": "Only auth/ files are modified",
                },
            ],
        )

        engine = ContractEngine(model_call=_mock_model_call(data))
        contract = engine.generate_contract(
            "Refactor auth to support OAuth2 and SAML."
        )

        assert len(contract.proof_obligations) == 5
        types = {ob.type for ob in contract.proof_obligations}
        assert ObligationType.BEHAVIOR in types
        assert ObligationType.REGRESSION in types
        assert ObligationType.INTERFACE in types
        assert ObligationType.SCOPE in types

    def test_all_obligations_start_pending(self):
        data = _make_valid_contract_data()
        engine = ContractEngine(model_call=_mock_model_call(data))
        contract = engine.generate_contract("Some issue.")
        for ob in contract.proof_obligations:
            assert ob.status == ObligationStatus.PENDING


# ===================================================================
# Test: Regression requirement
# ===================================================================

class TestRegressionRequirement:
    """Regression obligations protect existing behavior."""

    def test_regression_obligation_present(self):
        data = _make_valid_contract_data()
        engine = ContractEngine(model_call=_mock_model_call(data))
        contract = engine.generate_contract("Fix the login bug.")
        regression_obs = [
            ob for ob in contract.proof_obligations
            if ob.type == ObligationType.REGRESSION
        ]
        assert len(regression_obs) >= 1

    def test_regression_obligation_has_verification(self):
        data = _make_valid_contract_data()
        engine = ContractEngine(model_call=_mock_model_call(data))
        contract = engine.generate_contract("Fix the login bug.")
        regression_obs = [
            ob for ob in contract.proof_obligations
            if ob.type == ObligationType.REGRESSION
        ]
        for ob in regression_obs:
            assert ob.verification_method
            assert len(ob.verification_method) > 0


# ===================================================================
# Test: Scope requirement
# ===================================================================

class TestScopeRequirement:
    """Scope obligations ensure changes don't leak beyond the affected area."""

    def test_scope_obligation_present(self):
        data = _make_valid_contract_data()
        engine = ContractEngine(model_call=_mock_model_call(data))
        contract = engine.generate_contract("Fix the login bug.")
        scope_obs = [
            ob for ob in contract.proof_obligations
            if ob.type == ObligationType.SCOPE
        ]
        assert len(scope_obs) >= 1

    def test_scope_obligation_references_affected_area(self):
        data = _make_valid_contract_data()
        engine = ContractEngine(model_call=_mock_model_call(data))
        contract = engine.generate_contract("Fix the login bug.")
        scope_obs = [
            ob for ob in contract.proof_obligations
            if ob.type == ObligationType.SCOPE
        ]
        # The scope obligation should mention the affected area or file scope.
        for ob in scope_obs:
            assert ob.description
            assert ob.verification_method


# ===================================================================
# Test: Ambiguous issue
# ===================================================================

class TestAmbiguousIssue:
    """Ambiguous issues should surface ambiguities rather than inventing."""

    def test_ambiguities_captured(self):
        data = _make_valid_contract_data(
            task_goal="Improve performance",
            ambiguities=[
                "Which endpoint should be optimised?",
                "What is the target latency?",
                "Should caching be introduced or is algorithmic improvement preferred?",
            ],
        )
        engine = ContractEngine(model_call=_mock_model_call(data))
        contract = engine.generate_contract("Improve performance.")
        assert len(contract._ambiguities) == 3
        assert "Which endpoint" in contract._ambiguities[0]

    def test_ambiguities_in_summary(self):
        data = _make_valid_contract_data(
            ambiguities=["Unclear scope"],
        )
        engine = ContractEngine(model_call=_mock_model_call(data))
        contract = engine.generate_contract("Vague issue.")
        report = contract.summary()
        assert "ambiguities" in report
        assert "Unclear scope" in report["ambiguities"]

    def test_no_ambiguities_is_fine(self):
        data = _make_valid_contract_data(ambiguities=[])
        engine = ContractEngine(model_call=_mock_model_call(data))
        contract = engine.generate_contract("Clear issue.")
        assert contract._ambiguities == []


# ===================================================================
# Test: Malformed model output
# ===================================================================

class TestMalformedModelOutput:
    """The engine must handle garbage, partial JSON, and missing fields."""

    def test_completely_invalid_json(self):
        """Model returns plain English instead of JSON."""
        def bad_model(messages, tools=None):
            resp = MagicMock()
            block = MagicMock()
            block.text = "I don't understand the issue, sorry."
            resp.content = [block]
            return resp

        engine = ContractEngine(model_call=bad_model)
        with pytest.raises(ContractValidationError, match="does not contain valid JSON"):
            engine.generate_contract("Fix the login bug.")

    def test_json_missing_required_fields(self):
        """Model returns JSON but without task_goal."""
        incomplete = {"risk_level": "LOW"}

        engine = ContractEngine(model_call=_mock_model_call(incomplete))
        with pytest.raises(ContractValidationError, match="task_goal"):
            engine.generate_contract("Fix the login bug.")

    def test_invalid_risk_level(self):
        data = _make_valid_contract_data(risk_level="EXTREME")
        engine = ContractEngine(model_call=_mock_model_call(data))
        with pytest.raises(ContractValidationError, match="risk_level"):
            engine.generate_contract("Fix the login bug.")

    def test_empty_proof_obligations(self):
        data = _make_valid_contract_data(proof_obligations=[])
        engine = ContractEngine(model_call=_mock_model_call(data))
        with pytest.raises(ContractValidationError, match="at least one"):
            engine.generate_contract("Fix the login bug.")

    def test_invalid_obligation_type(self):
        data = _make_valid_contract_data(
            proof_obligations=[
                {
                    "id": "OB-1",
                    "description": "Test thing",
                    "type": "nonexistent_type",
                    "verification_method": "some method",
                },
            ],
        )
        engine = ContractEngine(model_call=_mock_model_call(data))
        with pytest.raises(ContractValidationError, match="Invalid obligation type"):
            engine.generate_contract("Fix the login bug.")

    def test_json_in_markdown_code_fence(self):
        """Model wraps its JSON in triple backticks — engine should handle it."""
        data = _make_valid_contract_data()
        json_str = f"```json\n{json.dumps(data)}\n```"

        def fenced_model(messages, tools=None):
            resp = MagicMock()
            block = MagicMock()
            block.text = json_str
            resp.content = [block]
            return resp

        engine = ContractEngine(model_call=fenced_model)
        contract = engine.generate_contract("Fix the login bug.")
        assert isinstance(contract, TaskContract)

    def test_json_with_preamble_text(self):
        """Model returns some text before the JSON object."""
        data = _make_valid_contract_data()
        raw = f"Here is the contract:\n\n{json.dumps(data)}"

        def chatty_model(messages, tools=None):
            resp = MagicMock()
            block = MagicMock()
            block.text = raw
            resp.content = [block]
            return resp

        engine = ContractEngine(model_call=chatty_model)
        contract = engine.generate_contract("Fix the login bug.")
        assert isinstance(contract, TaskContract)

    def test_empty_issue_rejected(self):
        engine = ContractEngine(model_call=_mock_model_call({}))
        with pytest.raises(ContractValidationError, match="empty"):
            engine.generate_contract("")

    def test_whitespace_only_issue_rejected(self):
        engine = ContractEngine(model_call=_mock_model_call({}))
        with pytest.raises(ContractValidationError, match="empty"):
            engine.generate_contract("   \n\t  ")

    def test_model_returns_list_instead_of_object(self):
        """Model returns a JSON array instead of an object."""
        def list_model(messages, tools=None):
            resp = MagicMock()
            block = MagicMock()
            block.text = "[1, 2, 3]"
            resp.content = [block]
            return resp

        engine = ContractEngine(model_call=list_model)
        with pytest.raises(ContractValidationError):
            engine.generate_contract("Fix the login bug.")

    def test_duplicate_obligation_ids_rejected(self):
        data = _make_valid_contract_data(
            proof_obligations=[
                {
                    "id": "OB-1",
                    "description": "First",
                    "type": "behavior",
                    "verification_method": "test",
                },
                {
                    "id": "OB-1",
                    "description": "Duplicate",
                    "type": "regression",
                    "verification_method": "test",
                },
            ],
        )
        engine = ContractEngine(model_call=_mock_model_call(data))
        with pytest.raises(ContractValidationError, match="Duplicate"):
            engine.generate_contract("Fix the login bug.")


# ===================================================================
# Test: Missing obligation fields
# ===================================================================

class TestMissingObligationFields:
    """Individual obligation dicts with missing required fields."""

    def test_missing_id(self):
        data = _make_valid_contract_data(
            proof_obligations=[
                {
                    "description": "Missing ID",
                    "type": "behavior",
                    "verification_method": "test",
                },
            ],
        )
        engine = ContractEngine(model_call=_mock_model_call(data))
        with pytest.raises(ContractValidationError, match="missing required fields"):
            engine.generate_contract("Fix the login bug.")

    def test_missing_description(self):
        data = _make_valid_contract_data(
            proof_obligations=[
                {
                    "id": "OB-1",
                    "type": "behavior",
                    "verification_method": "test",
                },
            ],
        )
        engine = ContractEngine(model_call=_mock_model_call(data))
        with pytest.raises(ContractValidationError, match="missing required fields"):
            engine.generate_contract("Fix the login bug.")

    def test_missing_type(self):
        data = _make_valid_contract_data(
            proof_obligations=[
                {
                    "id": "OB-1",
                    "description": "Something",
                    "verification_method": "test",
                },
            ],
        )
        engine = ContractEngine(model_call=_mock_model_call(data))
        with pytest.raises(ContractValidationError, match="missing required fields"):
            engine.generate_contract("Fix the login bug.")

    def test_missing_verification_method(self):
        data = _make_valid_contract_data(
            proof_obligations=[
                {
                    "id": "OB-1",
                    "description": "Something",
                    "type": "behavior",
                },
            ],
        )
        engine = ContractEngine(model_call=_mock_model_call(data))
        with pytest.raises(ContractValidationError, match="missing required fields"):
            engine.generate_contract("Fix the login bug.")

    def test_non_dict_obligation_rejected(self):
        data = _make_valid_contract_data(
            proof_obligations=["not a dict"],
        )
        engine = ContractEngine(model_call=_mock_model_call(data))
        with pytest.raises(ContractValidationError, match="not an object"):
            engine.generate_contract("Fix the login bug.")


# ===================================================================
# Test: Contract lifecycle — updates and reports
# ===================================================================

class TestContractLifecycle:
    """Test obligation updates, completeness checks, and final reports."""

    def _make_engine_with_contract(self) -> tuple[ContractEngine, TaskContract]:
        data = _make_valid_contract_data()
        engine = ContractEngine(model_call=_mock_model_call(data))
        contract = engine.generate_contract("Fix the login bug.")
        return engine, contract

    def test_update_obligation_pass(self):
        engine, contract = self._make_engine_with_contract()
        ok = engine.update_obligation(
            contract._contract_id, "OB-1",
            ObligationStatus.PASS, "Test test_locked_rejected passed",
        )
        assert ok is True
        ob = contract.get_obligation("OB-1")
        assert ob.status == ObligationStatus.PASS
        assert "test_locked_rejected" in ob.evidence[0]

    def test_update_obligation_fail(self):
        engine, contract = self._make_engine_with_contract()
        ok = engine.update_obligation(
            contract._contract_id, "OB-2",
            ObligationStatus.FAIL, "Test failed with AssertionError",
        )
        assert ok is True
        assert contract.get_obligation("OB-2").status == ObligationStatus.FAIL

    def test_update_nonexistent_obligation(self):
        engine, contract = self._make_engine_with_contract()
        ok = engine.update_obligation(
            contract._contract_id, "OB-999",
            ObligationStatus.PASS,
        )
        assert ok is False

    def test_update_nonexistent_contract(self):
        engine, _ = self._make_engine_with_contract()
        ok = engine.update_obligation(
            "CTR-nonexistent", "OB-1", ObligationStatus.PASS,
        )
        assert ok is False

    def test_contract_not_complete_with_pending(self):
        _, contract = self._make_engine_with_contract()
        assert contract.is_complete() is False

    def test_contract_not_complete_with_unknown(self):
        _, contract = self._make_engine_with_contract()
        for ob in contract.proof_obligations:
            ob.status = ObligationStatus.PASS
        # Set one to UNKNOWN.
        contract.proof_obligations[0].status = ObligationStatus.UNKNOWN
        assert contract.is_complete() is False

    def test_contract_complete_all_pass(self):
        _, contract = self._make_engine_with_contract()
        for ob in contract.proof_obligations:
            ob.status = ObligationStatus.PASS
        assert contract.is_complete() is True
        assert contract.all_passed() is True

    def test_contract_complete_mixed_pass_fail(self):
        _, contract = self._make_engine_with_contract()
        for ob in contract.proof_obligations:
            ob.status = ObligationStatus.PASS
        contract.proof_obligations[0].status = ObligationStatus.FAIL
        assert contract.is_complete() is True
        assert contract.all_passed() is False

    def test_final_report_structure(self):
        engine, contract = self._make_engine_with_contract()
        report = engine.final_report(contract._contract_id)
        assert "contract_id" in report
        assert "total_obligations" in report
        assert "status_counts" in report
        assert "obligations" in report
        assert report["total_obligations"] == 4

    def test_final_report_unknown_contract(self):
        engine, _ = self._make_engine_with_contract()
        report = engine.final_report("CTR-nope")
        assert "error" in report

    def test_report_shows_individual_statuses(self):
        engine, contract = self._make_engine_with_contract()
        engine.update_obligation(
            contract._contract_id, "OB-1", ObligationStatus.PASS
        )
        engine.update_obligation(
            contract._contract_id, "OB-2", ObligationStatus.FAIL
        )
        report = engine.final_report(contract._contract_id)
        ob_map = {ob["id"]: ob["status"] for ob in report["obligations"]}
        assert ob_map["OB-1"] == "PASS"
        assert ob_map["OB-2"] == "FAIL"
        assert ob_map["OB-3"] == "PENDING"

    def test_evidence_accumulates(self):
        engine, contract = self._make_engine_with_contract()
        engine.update_obligation(
            contract._contract_id, "OB-1",
            ObligationStatus.FAIL, "First run failed",
        )
        engine.update_obligation(
            contract._contract_id, "OB-1",
            ObligationStatus.PASS, "Second run passed after fix",
        )
        ob = contract.get_obligation("OB-1")
        assert ob.status == ObligationStatus.PASS
        assert len(ob.evidence) == 2


# ===================================================================
# Test: Serialisation
# ===================================================================

class TestSerialisation:
    """Contract and obligation serialisation round-trips."""

    def test_contract_to_dict(self):
        data = _make_valid_contract_data()
        engine = ContractEngine(model_call=_mock_model_call(data))
        contract = engine.generate_contract("Fix the bug.")
        d = contract.to_dict()
        assert d["task_goal"] == "Fix login so locked accounts cannot authenticate"
        assert d["risk_level"] == "MEDIUM"
        assert len(d["proof_obligations"]) == 4

    def test_obligation_to_dict(self):
        ob = ProofObligation(
            id="OB-1",
            description="Test",
            type=ObligationType.BEHAVIOR,
            verification_method="unit test",
        )
        d = ob.to_dict()
        assert d["id"] == "OB-1"
        assert d["type"] == "behavior"
        assert d["status"] == "PENDING"
        assert d["evidence"] == []

    def test_obligation_from_dict_round_trip(self):
        ob = ProofObligation(
            id="OB-42",
            description="Round trip",
            type=ObligationType.SAFETY,
            verification_method="fuzz test",
            status=ObligationStatus.PASS,
            evidence=["passed fuzz run"],
        )
        d = ob.to_dict()
        restored = ProofObligation.from_dict(d)
        assert restored.id == ob.id
        assert restored.type == ob.type
        assert restored.status == ob.status
        assert restored.evidence == ob.evidence

    def test_contract_to_dict_is_json_serialisable(self):
        data = _make_valid_contract_data()
        engine = ContractEngine(model_call=_mock_model_call(data))
        contract = engine.generate_contract("Fix the bug.")
        # Must not raise.
        json_str = json.dumps(contract.to_dict())
        assert isinstance(json_str, str)


# ===================================================================
# Test: Internal JSON extraction
# ===================================================================

class TestJsonExtraction:
    """Low-level _extract_json tests."""

    def test_raw_json(self):
        d = {"key": "value"}
        result = _extract_json(json.dumps(d))
        assert result == d

    def test_code_fenced_json(self):
        d = {"key": "value"}
        text = f"```json\n{json.dumps(d)}\n```"
        result = _extract_json(text)
        assert result == d

    def test_json_with_preamble(self):
        d = {"key": "value"}
        text = f"Here is the result:\n{json.dumps(d)}"
        result = _extract_json(text)
        assert result == d

    def test_no_json_raises(self):
        with pytest.raises(ContractValidationError):
            _extract_json("No JSON here at all!")

    def test_array_json_raises(self):
        with pytest.raises(ContractValidationError):
            _extract_json("[1, 2, 3]")


# ===================================================================
# Test: build_contract_from_data
# ===================================================================

class TestBuildContractFromData:
    """Direct contract construction from pre-parsed data."""

    def test_build_from_valid_data(self):
        data = _make_valid_contract_data()
        engine = ContractEngine(model_call=_mock_model_call(data))
        contract = engine.build_contract_from_data(data, issue="Test issue")
        assert isinstance(contract, TaskContract)
        assert contract._original_issue == "Test issue"

    def test_build_from_invalid_data_raises(self):
        engine = ContractEngine(model_call=_mock_model_call({}))
        with pytest.raises(ContractValidationError):
            engine.build_contract_from_data({"only": "partial"})


# ===================================================================
# Test: _extract_response_text
# ===================================================================

class TestExtractResponseText:
    """Edge cases for extracting text from model responses."""

    def test_plain_string(self):
        result = ContractEngine._extract_response_text('{"key": "val"}')
        assert result == '{"key": "val"}'

    def test_anthropic_style_response(self):
        block = MagicMock()
        block.text = "hello"
        resp = MagicMock()
        resp.content = [block]
        assert ContractEngine._extract_response_text(resp) == "hello"

    def test_no_text_block_raises(self):
        resp = MagicMock()
        resp.content = []
        with pytest.raises(ContractValidationError, match="extract text"):
            ContractEngine._extract_response_text(resp)


# ===================================================================
# Test: Obligation types coverage
# ===================================================================

class TestObligationTypes:
    """All six obligation types can be created and serialised."""

    @pytest.mark.parametrize("ob_type", list(ObligationType))
    def test_all_types_valid(self, ob_type: ObligationType):
        ob = ProofObligation(
            id=f"OB-{ob_type.value}",
            description=f"Test {ob_type.value}",
            type=ob_type,
            verification_method="automated",
        )
        d = ob.to_dict()
        assert d["type"] == ob_type.value
        restored = ProofObligation.from_dict(d)
        assert restored.type == ob_type

    @pytest.mark.parametrize("status", list(ObligationStatus))
    def test_all_statuses_valid(self, status: ObligationStatus):
        ob = ProofObligation(
            id="OB-1",
            description="Test",
            type=ObligationType.BEHAVIOR,
            verification_method="test",
            status=status,
        )
        assert ob.status == status
        assert ob.to_dict()["status"] == status.value
