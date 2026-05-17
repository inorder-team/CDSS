"""
CDSS Platform – Extended Test Suite
Tests for RAG retrieval, medication routes, encounter routes, and auth.
Run: pytest tests/ -v
"""
import pytest


# ─────────────────────────────────────────────
# RAG Integration Tests
# ─────────────────────────────────────────────

class TestRAGIntegration:
    """Tests that require real ChromaDB + sentence-transformers (no Anthropic key)."""

    @pytest.mark.rag
    def test_ingest_and_retrieve_nstemi(self, ingested_rag):
        docs = ingested_rag.retrieve(
            query="NSTEMI antiplatelet therapy aspirin allergy alternative",
            top_k=5,
        )
        # Should return results if guidelines are ingested
        if ingested_rag.collection_count() > 0:
            assert len(docs) > 0
            assert all(d.similarity_score >= 0.0 for d in docs)
            assert all(d.doc_id for d in docs)

    @pytest.mark.rag
    def test_retrieve_renal_guidance(self, ingested_rag):
        if ingested_rag.collection_count() == 0:
            pytest.skip("RAG not ingested")
        docs = ingested_rag.retrieve(
            query="eGFR 42 CKD anticoagulant dose adjustment NSTEMI",
            top_k=3,
        )
        assert isinstance(docs, list)
        # Check relevant tags appear
        tags = [d.relevance_tag for d in docs]
        assert any(t in ("renal", "antiplatelet", "general") for t in tags)

    @pytest.mark.rag
    def test_retrieve_human_review_mandate(self, ingested_rag):
        if ingested_rag.collection_count() == 0:
            pytest.skip("RAG not ingested")
        docs = ingested_rag.retrieve(
            query="human review cardiologist AI decision support requirement",
            top_k=3,
        )
        assert isinstance(docs, list)

    @pytest.mark.rag
    def test_retrieve_returns_evidence_documents(self, ingested_rag):
        if ingested_rag.collection_count() == 0:
            pytest.skip("RAG not ingested")
        from app.models.schemas import EvidenceDocument
        docs = ingested_rag.retrieve(query="NSTEMI risk stratification GRACE score")
        for doc in docs:
            assert isinstance(doc, EvidenceDocument)
            assert 0.0 <= doc.similarity_score <= 1.0
            assert len(doc.content_snippet) > 10

    @pytest.mark.rag
    def test_retrieve_with_low_threshold(self, ingested_rag):
        if ingested_rag.collection_count() == 0:
            pytest.skip("RAG not ingested")
        docs = ingested_rag.retrieve(
            query="potassium monitoring ACE inhibitor CKD",
            score_threshold=0.1,
            top_k=5,
        )
        assert isinstance(docs, list)

    @pytest.mark.rag
    def test_collection_count_positive_after_ingest(self, ingested_rag):
        if ingested_rag.collection_count() == 0:
            pytest.skip("RAG not ingested")
        assert ingested_rag.collection_count() > 0
        assert ingested_rag.is_ready


# ─────────────────────────────────────────────
# Medication Route Tests
# ─────────────────────────────────────────────

class TestMedicationRoutes:

    def test_get_ticagrelor_info(self, test_client):
        res = test_client.get("/api/v1/medications/ticagrelor")
        assert res.status_code == 200
        data = res.json()
        assert data["drug_name"] == "Ticagrelor"
        assert "P2Y12" in data["drug_class"]
        assert isinstance(data["contraindications"], list)

    def test_get_clopidogrel_info(self, test_client):
        res = test_client.get("/api/v1/medications/clopidogrel")
        assert res.status_code == 200
        data = res.json()
        assert data["drug_name"] == "Clopidogrel"

    def test_get_unknown_drug_404(self, test_client):
        res = test_client.get("/api/v1/medications/unknowndrugxyz")
        assert res.status_code == 404

    def test_drug_safety_check_aspirin_allergy(self, test_client):
        res = test_client.get(
            "/api/v1/medications/check",
            params={"drug": "aspirin", "patient_id": "PAT-001", "allergies": "aspirin"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["safe_to_use"] is False
        assert len(data["contraindications_triggered"]) > 0

    def test_drug_safety_check_prasugrel_low_egfr(self, test_client):
        res = test_client.get(
            "/api/v1/medications/check",
            params={"drug": "prasugrel", "patient_id": "PAT-001", "egfr": 25.0},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["safe_to_use"] is False
        assert any("30" in c or "contraindicated" in c.lower() for c in data["contraindications_triggered"])

    def test_drug_safety_check_enoxaparin_renal_warning(self, test_client):
        res = test_client.get(
            "/api/v1/medications/check",
            params={"drug": "enoxaparin", "patient_id": "PAT-001", "egfr": 25.0},
        )
        assert res.status_code == 200
        data = res.json()
        assert len(data["warnings"]) > 0
        assert "once daily" in data["dose_recommendation"].lower()

    def test_ticagrelor_safe_egfr42(self, test_client):
        res = test_client.get(
            "/api/v1/medications/check",
            params={"drug": "ticagrelor", "patient_id": "PAT-001", "egfr": 42.0},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["safe_to_use"] is True

    def test_interaction_check_dual_p2y12(self, test_client):
        res = test_client.post(
            "/api/v1/medications/interactions",
            json=["ticagrelor", "clopidogrel"],
        )
        assert res.status_code == 200
        data = res.json()
        assert len(data) > 0
        assert data[0]["severity"] == "avoid"

    def test_interaction_dual_anticoag(self, test_client):
        res = test_client.post(
            "/api/v1/medications/interactions",
            json=["fondaparinux", "enoxaparin"],
        )
        assert res.status_code == 200
        data = res.json()
        assert len(data) > 0
        assert data[0]["severity"] == "contraindicated"

    def test_no_interactions_single_drug(self, test_client):
        res = test_client.post(
            "/api/v1/medications/interactions",
            json=["ticagrelor"],
        )
        assert res.status_code == 200
        data = res.json()
        assert data == []


# ─────────────────────────────────────────────
# Encounter Route Tests
# ─────────────────────────────────────────────

class TestEncounterRoutes:

    def test_get_existing_encounter(self, test_client):
        res = test_client.get("/api/v1/encounters/ENC-CARD-001")
        assert res.status_code == 200
        data = res.json()
        assert data["encounter_id"] == "ENC-CARD-001"
        assert data["patient_id"] == "PAT-CARD-001"
        assert data["status"] == "active"
        assert isinstance(data["timeline_events"], list)
        assert len(data["timeline_events"]) > 0

    def test_get_missing_encounter(self, test_client):
        res = test_client.get("/api/v1/encounters/ENC-NONEXISTENT-999")
        assert res.status_code == 404

    def test_create_encounter(self, test_client):
        res = test_client.post(
            "/api/v1/encounters",
            json={
                "patient_id": "PAT-TEST-002",
                "encounter_id": "ENC-TEST-002",
                "encounter_type": "emergency",
                "clinician_id": "nurse.local",
                "clinician_role": "CDSS_NURSE",
                "primary_complaint": "Chest pain",
                "diagnoses": ["NSTEMI"],
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["encounter_id"] == "ENC-TEST-002"
        assert data["status"] == "active"

    def test_get_timeline(self, test_client):
        res = test_client.get("/api/v1/encounters/ENC-CARD-001/timeline")
        assert res.status_code == 200
        data = res.json()
        assert "timeline" in data
        assert isinstance(data["timeline"], list)
        assert len(data["timeline"]) > 0


# ─────────────────────────────────────────────
# Auth Route Tests
# ─────────────────────────────────────────────

class TestAuthRoutes:

    def test_login_valid_cardiologist(self, test_client):
        res = test_client.post(
            "/api/v1/auth/token",
            json={"username": "cardiologist.local", "password": "cardio123"},
        )
        assert res.status_code == 200
        data = res.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    def test_login_invalid_password(self, test_client):
        res = test_client.post(
            "/api/v1/auth/token",
            json={"username": "cardiologist.local", "password": "wrongpassword"},
        )
        assert res.status_code == 401

    def test_login_unknown_user(self, test_client):
        res = test_client.post(
            "/api/v1/auth/token",
            json={"username": "ghost.user", "password": "anything"},
        )
        assert res.status_code == 401

    def test_me_with_valid_token(self, test_client):
        # First login
        login = test_client.post(
            "/api/v1/auth/token",
            json={"username": "cardiologist.local", "password": "cardio123"},
        )
        token = login.json()["access_token"]
        # Then /me
        res = test_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["user_id"] == "cardiologist.local"
        assert data["role"] == "CARDIOLOGIST"

    def test_me_without_token(self, test_client):
        res = test_client.get("/api/v1/auth/me")
        assert res.status_code == 401

    def test_me_with_bad_token(self, test_client):
        res = test_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer totally_invalid_token"},
        )
        assert res.status_code == 401


# ─────────────────────────────────────────────
# Patient Route Tests
# ─────────────────────────────────────────────

class TestPatientRoutes:

    def test_get_existing_patient(self, test_client):
        res = test_client.get("/api/v1/patients/PAT-CARD-001")
        assert res.status_code == 200
        data = res.json()
        assert data["patient_id"] == "PAT-CARD-001"
        assert "aspirin" in data["allergies"]
        assert "NSTEMI" in data["active_diagnoses"]

    def test_get_missing_patient(self, test_client):
        res = test_client.get("/api/v1/patients/PAT-GHOST-999")
        assert res.status_code == 404

    def test_list_patient_encounters(self, test_client):
        res = test_client.get("/api/v1/patients/PAT-CARD-001/encounters")
        assert res.status_code == 200
        data = res.json()
        assert "encounters" in data
        assert len(data["encounters"]) > 0

    def test_list_patient_recommendations(self, test_client):
        res = test_client.get("/api/v1/patients/PAT-CARD-001/recommendations")
        assert res.status_code == 200
        data = res.json()
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)


# ─────────────────────────────────────────────
# Admin Route Tests
# ─────────────────────────────────────────────

class TestAdminRoutes:

    def test_get_stats(self, test_client):
        res = test_client.get("/api/v1/admin/stats")
        assert res.status_code == 200
        data = res.json()
        assert "app_name" in data
        assert "rag_chunk_count" in data
        assert isinstance(data["rag_chunk_count"], int)

    def test_get_audit_log(self, test_client):
        res = test_client.get("/api/v1/admin/audit?limit=10")
        assert res.status_code == 200
        data = res.json()
        assert "entries" in data
        assert "total" in data

    def test_list_audit_event_types(self, test_client):
        res = test_client.get("/api/v1/admin/audit/events")
        assert res.status_code == 200
        data = res.json()
        assert "event_types" in data

    def test_rag_reindex(self, test_client):
        res = test_client.post("/api/v1/admin/rag/reindex")
        # Should succeed (200) or fail if guidelines dir missing
        assert res.status_code in (200, 404)


# ─────────────────────────────────────────────
# Security / JWT Tests
# ─────────────────────────────────────────────

class TestSecurity:

    def test_create_and_decode_token(self):
        from app.core.security import create_access_token, decode_token
        token = create_access_token(user_id="test.user", role="CARDIOLOGIST", org_id="test-org")
        payload = decode_token(token)
        assert payload.sub == "test.user"
        assert payload.role == "CARDIOLOGIST"
        assert payload.org_id == "test-org"

    def test_revoked_token_rejected(self):
        from app.core.security import create_access_token, decode_token, revoke_token
        from fastapi import HTTPException
        token = create_access_token(user_id="test.user", role="CARDIOLOGIST")
        payload = decode_token(token)
        revoke_token(payload.jti)
        with pytest.raises(HTTPException) as exc:
            decode_token(token)
        assert exc.value.status_code == 401

    def test_expired_token_rejected(self):
        from jose import jwt
        from fastapi import HTTPException
        from app.core.security import decode_token
        from app.core.config import get_settings
        import time
        settings = get_settings()
        expired_payload = {
            "sub": "test.user",
            "role": "CARDIOLOGIST",
            "exp": int(time.time()) - 3600,   # expired 1 hour ago
            "iat": int(time.time()) - 7200,
        }
        token = jwt.encode(expired_payload, settings.secret_key, algorithm=settings.jwt_algorithm)
        with pytest.raises(HTTPException) as exc:
            decode_token(token)
        assert exc.value.status_code == 401

    def test_wrong_secret_rejected(self):
        from jose import jwt
        from fastapi import HTTPException
        from app.core.security import decode_token
        import time
        bad_token = jwt.encode(
            {"sub": "attacker", "role": "CDSS_ADMIN", "exp": int(time.time()) + 3600},
            "wrong_secret_key_12345",
            algorithm="HS256",
        )
        with pytest.raises(HTTPException):
            decode_token(bad_token)
