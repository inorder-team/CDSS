"""
CDSS Platform – API Integration Tests
Tests all API routes using FastAPI TestClient.
"""
import pytest
from fastapi.testclient import TestClient


class TestAuthRoutes:
    def test_login_valid_user(self, test_client):
        res = test_client.post("/api/v1/auth/token", json={
            "username": "cardiologist.local",
            "password": "cardio123"
        })
        assert res.status_code == 200
        data = res.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_login_invalid_password(self, test_client):
        res = test_client.post("/api/v1/auth/token", json={
            "username": "cardiologist.local",
            "password": "wrongpassword"
        })
        assert res.status_code == 401

    def test_login_unknown_user(self, test_client):
        res = test_client.post("/api/v1/auth/token", json={
            "username": "nobody.local",
            "password": "anything"
        })
        assert res.status_code == 401

    def test_me_with_valid_token(self, test_client):
        login = test_client.post("/api/v1/auth/token", json={
            "username": "cardiologist.local", "password": "cardio123"
        })
        token = login.json()["access_token"]
        res = test_client.get("/api/v1/auth/me",
                              headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        data = res.json()
        assert data["user_id"] == "cardiologist.local"
        assert data["role"] == "CARDIOLOGIST"

    def test_logout(self, test_client):
        login = test_client.post("/api/v1/auth/token", json={
            "username": "nurse.local", "password": "nurse123"
        })
        token = login.json()["access_token"]
        res = test_client.post("/api/v1/auth/logout",
                               headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200


class TestPatientRoutes:
    def test_get_known_patient(self, test_client):
        res = test_client.get("/api/v1/patients/PAT-CARD-001")
        assert res.status_code == 200
        data = res.json()
        assert data["patient_id"] == "PAT-CARD-001"
        assert "aspirin" in data["allergies"]

    def test_get_unknown_patient(self, test_client):
        res = test_client.get("/api/v1/patients/PAT-DOES-NOT-EXIST")
        assert res.status_code == 404

    def test_list_encounters(self, test_client):
        res = test_client.get("/api/v1/patients/PAT-CARD-001/encounters")
        assert res.status_code == 200
        data = res.json()
        assert "encounters" in data
        assert len(data["encounters"]) >= 1


class TestEncounterRoutes:
    def test_get_existing_encounter(self, test_client):
        res = test_client.get("/api/v1/encounters/ENC-CARD-001")
        assert res.status_code == 200
        data = res.json()
        assert data["encounter_id"] == "ENC-CARD-001"
        assert data["patient_id"] == "PAT-CARD-001"

    def test_create_encounter(self, test_client):
        res = test_client.post("/api/v1/encounters", json={
            "encounter_id": "ENC-TEST-999",
            "patient_id": "PAT-TEST-999",
            "encounter_type": "emergency",
            "diagnoses": ["NSTEMI"],
            "attending_clinician": "cardiologist.local"
        })
        assert res.status_code == 201
        data = res.json()
        assert data["encounter_id"] == "ENC-TEST-999"
        assert data["status"] == "active"

    def test_create_duplicate_encounter_fails(self, test_client):
        test_client.post("/api/v1/encounters", json={
            "encounter_id": "ENC-DUP-001",
            "patient_id": "PAT-DUP-001",
            "encounter_type": "cardiology-consult",
        })
        res = test_client.post("/api/v1/encounters", json={
            "encounter_id": "ENC-DUP-001",
            "patient_id": "PAT-DUP-001",
            "encounter_type": "cardiology-consult",
        })
        assert res.status_code == 400


class TestMedicationRoutes:
    def test_drug_allergy_check_conflict(self, test_client):
        res = test_client.get("/api/v1/medications/check",
                              params={"drugs": "aspirin", "allergies": "aspirin"})
        assert res.status_code == 200
        data = res.json()
        assert data[0]["allergy_conflict"] is True
        assert data[0]["safe_to_use"] is False

    def test_drug_allergy_check_safe(self, test_client):
        res = test_client.get("/api/v1/medications/check",
                              params={"drugs": "ticagrelor,atorvastatin", "allergies": "aspirin"})
        assert res.status_code == 200
        for item in res.json():
            assert item["allergy_conflict"] is False

    def test_renal_adjust_fondaparinux_low_egfr(self, test_client):
        res = test_client.get("/api/v1/medications/renal-adjust",
                              params={"drugs": "fondaparinux", "egfr": "18"})
        assert res.status_code == 200
        data = res.json()
        assert data[0]["contraindicated"] is True

    def test_renal_adjust_ticagrelor_moderate_ckd(self, test_client):
        res = test_client.get("/api/v1/medications/renal-adjust",
                              params={"drugs": "ticagrelor", "egfr": "42"})
        assert res.status_code == 200
        data = res.json()
        assert data[0]["contraindicated"] is False

    def test_renal_adjust_unknown_drug(self, test_client):
        res = test_client.get("/api/v1/medications/renal-adjust",
                              params={"drugs": "unknowndrugxyz", "egfr": "55"})
        assert res.status_code == 200
        data = res.json()
        assert "formulary" in data[0]["recommended_dose"].lower()


class TestAdminRoutes:
    def test_system_status(self, test_client):
        res = test_client.get("/api/v1/admin/system")
        assert res.status_code == 200
        data = res.json()
        assert "rag" in data
        assert "audit" in data
        assert "compliance" in data

    def test_rag_status(self, test_client):
        res = test_client.get("/api/v1/admin/rag/status")
        assert res.status_code == 200
        data = res.json()
        assert "chunk_count" in data
        assert "embedding_model" in data

    def test_get_config(self, test_client):
        res = test_client.get("/api/v1/admin/config")
        assert res.status_code == 200
        data = res.json()
        assert "llm_model" in data
        assert "safety_gate_enabled" in data


class TestAuditRoutes:
    def test_list_audit_events(self, test_client):
        res = test_client.get("/api/v1/audit?limit=10")
        assert res.status_code == 200
        data = res.json()
        assert "events" in data

    def test_missing_correlation_id(self, test_client):
        res = test_client.get("/api/v1/audit/nonexistent-correlation-id-xyz")
        assert res.status_code == 404


class TestHealthRoutes:
    def test_health(self, test_client):
        res = test_client.get("/api/v1/health")
        assert res.status_code == 200
        data = res.json()
        assert data["version"] is not None

    def test_rag_health(self, test_client):
        res = test_client.get("/api/v1/health/rag")
        assert res.status_code == 200

    def test_recommendations_list(self, test_client):
        res = test_client.get("/api/v1/recommendations")
        assert res.status_code == 200
        data = res.json()
        assert "recommendations" in data
