"""
CDSS Platform – Test Suite
pytest tests/test_cdss.py -v
"""
import json
import pytest
from pathlib import Path


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

NSTEMI_PAYLOAD = {
    "patientId": "PAT-CARD-001",
    "encounterId": "ENC-CARD-001",
    "userId": "cardiologist.local",
    "userRole": "CARDIOLOGIST",
    "query": "Recommend guideline-based considerations for NSTEMI management for this patient.",
    "consentVerified": True,
    "patientContext": {
        "age": 68,
        "sex": "male",
        "encounterType": "cardiology-consult",
        "diagnoses": ["NSTEMI", "type-2-diabetes", "chronic-kidney-disease"],
        "ecgFindings": ["ST depression in lateral leads"],
        "labs": {"troponin": "elevated and rising", "eGFR": "42", "potassium": "4.8"},
        "vitals": {"systolicBp": "138", "heartRate": "92"},
        "currentMedications": ["atorvastatin", "metoprolol"],
        "allergies": ["aspirin"],
        "contraindications": ["documented aspirin allergy"],
        "cardiacHistory": ["prior PCI"],
    },
}


# ─────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────

class TestSchemas:
    def test_valid_request_parses(self):
        from app.models.schemas import CDSSRecommendationRequest
        req = CDSSRecommendationRequest(**NSTEMI_PAYLOAD)
        assert req.patient_id == "PAT-CARD-001"
        assert req.user_role.value == "CARDIOLOGIST"
        assert req.consent_verified is True

    def test_aspirin_in_allergies(self):
        from app.models.schemas import CDSSRecommendationRequest
        req = CDSSRecommendationRequest(**NSTEMI_PAYLOAD)
        assert "aspirin" in req.patient_context.allergies

    def test_consent_required(self):
        from app.models.schemas import CDSSRecommendationRequest
        from pydantic import ValidationError
        bad = {**NSTEMI_PAYLOAD, "consentVerified": False}
        with pytest.raises(ValidationError):
            CDSSRecommendationRequest(**bad)

    def test_lab_values_parsed(self):
        from app.models.schemas import CDSSRecommendationRequest
        req = CDSSRecommendationRequest(**NSTEMI_PAYLOAD)
        assert req.patient_context.labs.egfr == "42"
        assert req.patient_context.labs.potassium == "4.8"


# ─────────────────────────────────────────────
# Safety Gate Tests
# ─────────────────────────────────────────────

class TestSafetyGate:
    def setup_method(self):
        import sys, os
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from dotenv import load_dotenv
        load_dotenv()

    def _make_request(self, allergies=None, egfr=None):
        from app.models.schemas import CDSSRecommendationRequest
        payload = json.loads(json.dumps(NSTEMI_PAYLOAD))
        if allergies is not None:
            payload["patientContext"]["allergies"] = allergies
        if egfr is not None:
            payload["patientContext"]["labs"]["eGFR"] = egfr
        return CDSSRecommendationRequest(**payload)

    def test_aspirin_blocked_in_response(self):
        from app.services.safety_gate import safety_gate
        req = self._make_request(allergies=["aspirin"])
        result = safety_gate.evaluate(
            request=req,
            llm_response_text="administer aspirin 300mg loading dose",
            rag_confidence=0.8,
        )
        assert result.passed is False
        assert any("aspirin" in f.lower() for f in result.flags)

    def test_no_aspirin_in_response_passes(self):
        from app.services.safety_gate import safety_gate
        from app.models.schemas import DecisionStatus
        req = self._make_request(allergies=["aspirin"])
        result = safety_gate.evaluate(
            request=req,
            llm_response_text="Recommend ticagrelor 180mg loading dose as alternative antiplatelet therapy.",
            rag_confidence=0.85,
        )
        assert result.passed is True
        assert result.decision_status in (DecisionStatus.PENDING_REVIEW, DecisionStatus.FLAGGED)

    def test_low_egfr_flags_renal_alert(self):
        from app.services.safety_gate import safety_gate
        req = self._make_request(egfr="25")
        result = safety_gate.evaluate(
            request=req,
            llm_response_text="Use prasugrel as antiplatelet.",
            rag_confidence=0.7,
        )
        assert any("renal" in f.lower() or "egfr" in f.lower() for f in result.flags)

    def test_prasugrel_blocked_with_low_egfr(self):
        from app.services.safety_gate import safety_gate
        req = self._make_request(egfr="25", allergies=[])
        result = safety_gate.evaluate(
            request=req,
            llm_response_text="recommend prasugrel as primary antiplatelet",
            rag_confidence=0.7,
        )
        assert result.passed is False

    def test_potassium_watch_flag(self):
        from app.services.safety_gate import safety_gate
        req = self._make_request()
        result = safety_gate.evaluate(
            request=req,
            llm_response_text="Initiate ACE inhibitor and monitor potassium.",
            rag_confidence=0.8,
        )
        assert any("potassium" in f.lower() for f in result.flags)

    def test_confidence_zero_when_blocked(self):
        from app.services.safety_gate import safety_gate
        req = self._make_request(allergies=["aspirin"])
        result = safety_gate.evaluate(
            request=req,
            llm_response_text="administer aspirin 300mg",
            rag_confidence=0.9,
        )
        assert result.confidence_score == 0.0


# ─────────────────────────────────────────────
# RAG Engine Tests
# ─────────────────────────────────────────────

class TestRAGEngine:
    def test_chunk_document(self):
        from app.rag.rag_engine import chunk_document
        text = " ".join(["This is a test sentence."] * 100)
        chunks = chunk_document(text, chunk_size=200, overlap=50)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) >= 50

    def test_classify_chunk_allergy(self):
        from app.rag.rag_engine import _classify_chunk
        text = "aspirin allergy protocol: do not administer aspirin"
        assert _classify_chunk(text) == "antiplatelet_allergy"

    def test_classify_chunk_renal(self):
        from app.rag.rag_engine import _classify_chunk
        text = "renal impairment eGFR dose adjustment CKD"
        assert _classify_chunk(text) == "renal"

    def test_classify_chunk_risk(self):
        from app.rag.rag_engine import _classify_chunk
        text = "risk stratification using GRACE score and troponin"
        assert _classify_chunk(text) == "risk_stratification"


# ─────────────────────────────────────────────
# Audit Logger Tests
# ─────────────────────────────────────────────

class TestAuditLogger:
    def test_audit_writes_json_line(self, tmp_path):
        import os
        os.environ["AUDIT_LOG_PATH"] = str(tmp_path / "test_audit.jsonl")

        from app.core.audit import AuditLogger
        logger = AuditLogger()
        logger.log_request(
            correlation_id="test-corr-001",
            user_id="test.user",
            user_role="CARDIOLOGIST",
            patient_id="PAT-001",
            encounter_id="ENC-001",
            endpoint="/api/v1/recommendations",
            query="Test query",
        )

        log_file = tmp_path / "test_audit.jsonl"
        assert log_file.exists()
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event"] == "cdss_request"
        assert entry["correlation_id"] == "test-corr-001"
        assert "_hash" in entry

    def test_audit_hash_tamper_detection(self, tmp_path):
        import os, hashlib
        os.environ["AUDIT_LOG_PATH"] = str(tmp_path / "audit2.jsonl")
        from app.core.audit import AuditLogger, _compute_hash

        logger = AuditLogger()
        logger.log_decision(
            correlation_id="test-001",
            patient_id="PAT-001",
            decision_status="pending_review",
            confidence=0.85,
            requires_human_review=True,
            recommendation_summary="Test recommendation",
        )

        log_file = tmp_path / "audit2.jsonl"
        entry = json.loads(log_file.read_text().strip())
        stored_hash = entry.pop("_hash")

        # Recompute – should match
        recomputed = _compute_hash(entry)
        assert recomputed == stored_hash


# ─────────────────────────────────────────────
# API Tests (FastAPI TestClient)
# ─────────────────────────────────────────────

class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        # Don't run lifespan for unit tests
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c

    def test_health_endpoint(self, client):
        res = client.get("/api/v1/health")
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
        assert "version" in data

    def test_recommendation_no_api_key_returns_503(self, client):
        """Without a real API key, expect 503."""
        import os
        original = os.environ.get("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = "your_anthropic_api_key_here"
        try:
            res = client.post("/api/v1/recommendations", json=NSTEMI_PAYLOAD)
            assert res.status_code == 503
        finally:
            if original:
                os.environ["ANTHROPIC_API_KEY"] = original

    def test_recommendation_no_consent_rejected(self, client):
        """Consent required."""
        payload = {**NSTEMI_PAYLOAD, "consentVerified": False}
        res = client.post("/api/v1/recommendations", json=payload)
        assert res.status_code == 422  # Pydantic validation error

    def test_get_missing_recommendation(self, client):
        res = client.get("/api/v1/recommendations/nonexistent-id-000")
        assert res.status_code == 404

    def test_review_invalid_action(self, client):
        res = client.post("/api/v1/recommendations/some-id/review", json={
            "correlation_id": "some-id",
            "reviewer_id": "dr.test",
            "reviewer_role": "CARDIOLOGIST",
            "action": "invalidaction",
        })
        assert res.status_code == 400
