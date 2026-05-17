"""
CDSS Platform – Patient Routes (doc §12)

Permission guards per doc §12:
  GET  /api/v1/patients/{id}                  → requires cdss:patient:read
  POST /api/v1/patients/                       → requires cdss:patient:write
  GET  /api/v1/patients/{id}/encounters        → requires cdss:encounter:read
  GET  /api/v1/patients/{id}/recommendations   → requires cdss:recommendation:read
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.core.security import (
    TokenPayload,
    require_patient_read,
    require_patient_write,
    require_encounter_read,
    require_recommendation_read,
)

router = APIRouter(prefix="/patients", tags=["Patients"])


class PatientSummary(BaseModel):
    patient_id: str
    display_name: str
    age: int
    sex: str
    active_diagnoses: list[str]
    allergies: list[str]
    current_medications: list[str]


_PATIENTS = {
    "PAT-CARD-001": PatientSummary(
        patient_id="PAT-CARD-001",
        display_name="Patient A (Cardiology)",
        age=68, sex="male",
        active_diagnoses=["NSTEMI", "type-2-diabetes", "chronic-kidney-disease"],
        allergies=["aspirin"],
        current_medications=["atorvastatin", "metoprolol"],
    ),
}


@router.get(
    "/{patient_id}",
    response_model=PatientSummary,
    summary="Get Patient Summary",
    description="Requires scope: `cdss:patient:read`",
)
async def get_patient(
    patient_id: str,
    current_user: TokenPayload = Depends(require_patient_read),
) -> PatientSummary:
    patient = _PATIENTS.get(patient_id)
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Patient {patient_id} not found")
    return patient


@router.post(
    "/",
    summary="Register Patient",
    description="Requires scope: `cdss:patient:write`",
    status_code=201,
)
async def create_patient(
    body: PatientSummary,
    current_user: TokenPayload = Depends(require_patient_write),
) -> dict:
    _PATIENTS[body.patient_id] = body
    return {"message": "Patient registered", "patient_id": body.patient_id}


@router.get(
    "/{patient_id}/encounters",
    summary="List Patient Encounters",
    description="Requires scope: `cdss:encounter:read`",
)
async def list_encounters(
    patient_id: str,
    current_user: TokenPayload = Depends(require_encounter_read),
) -> dict:
    if patient_id not in _PATIENTS:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return {
        "patient_id": patient_id,
        "encounters": [
            {"encounter_id": "ENC-CARD-001", "encounter_type": "cardiology-consult",
             "date": "2026-05-08", "status": "active"},
        ],
    }


@router.get(
    "/{patient_id}/recommendations",
    summary="Recommendation History",
    description="Requires scope: `cdss:recommendation:read`",
)
async def list_patient_recommendations(
    patient_id: str,
    current_user: TokenPayload = Depends(require_recommendation_read),
) -> dict:
    from app.api.routes.recommendations import _pending_decisions
    recs = [
        {"correlation_id": r.correlation_id, "status": r.recommendation.decision_status,
         "confidence": r.recommendation.confidence_score,
         "generated_at": r.recommendation.generated_at}
        for r in _pending_decisions.values()
        if r.patient_id == patient_id
    ]
    return {"patient_id": patient_id, "total": len(recs), "recommendations": recs}
