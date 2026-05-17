"""
CDSS Platform – pytest conftest.py
Shared fixtures for all tests.
"""
import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("APP_ENV", "testing")
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("ANTHROPIC_API_KEY", "test_key_not_real")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./data/test_cdss.db")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./data/test_chroma_db")
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("SAFETY_GATE_ENABLED", "true")

from dotenv import load_dotenv
load_dotenv()


@pytest.fixture(scope="session")
def nstemi_payload() -> dict:
    return {
        "patientId": "PAT-CARD-001",
        "encounterId": "ENC-CARD-001",
        "userId": "cardiologist.local",
        "userRole": "CARDIOLOGIST",
        "query": "Recommend guideline-based considerations for NSTEMI management for this patient.",
        "consentVerified": True,
        "patientContext": {
            "age": 68, "sex": "male", "encounterType": "cardiology-consult",
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


@pytest.fixture(scope="session")
def nstemi_request(nstemi_payload):
    from app.models.schemas import CDSSRecommendationRequest
    return CDSSRecommendationRequest(**nstemi_payload)


@pytest.fixture(scope="session")
def test_client():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def temp_audit_logger(tmp_path):
    os.environ["AUDIT_LOG_PATH"] = str(tmp_path / "test_audit.jsonl")
    from app.core.audit import AuditLogger
    return AuditLogger()


@pytest.fixture(scope="session")
def ingested_rag():
    from app.rag.rag_engine import CDSSRagEngine
    engine = CDSSRagEngine()
    guideline_dir = Path("./data/guidelines")
    if guideline_dir.exists() and any(guideline_dir.glob("*.txt")):
        engine.ingest_guidelines_directory(guideline_dir)
    return engine
