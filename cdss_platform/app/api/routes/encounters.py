"""
CDSS Platform – Encounter Routes (doc §12)

Permission guards:
  GET  /api/v1/encounters/{id}           → cdss:encounter:read
  POST /api/v1/encounters                → cdss:encounter:write
  GET  /api/v1/encounters/{id}/timeline  → cdss:encounter:read
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.security import TokenPayload, require_encounter_read, require_encounter_write

router = APIRouter(prefix="/encounters", tags=["Encounters"])

class EncounterCreate(BaseModel):
    patient_id: str; encounter_id: str; encounter_type: str
    clinician_id: str; clinician_role: str
    primary_complaint: Optional[str] = None; diagnoses: list[str] = []

class EncounterDetail(BaseModel):
    encounter_id: str; patient_id: str; encounter_type: str
    clinician_id: str; status: str; created_at: datetime
    diagnoses: list[str]; timeline_events: list[dict]

_encounters: dict[str, EncounterDetail] = {
    "ENC-CARD-001": EncounterDetail(
        encounter_id="ENC-CARD-001", patient_id="PAT-CARD-001",
        encounter_type="cardiology-consult", clinician_id="cardiologist.local",
        status="active", created_at=datetime(2026, 5, 8, 9, 0, 0, tzinfo=timezone.utc),
        diagnoses=["NSTEMI", "type-2-diabetes", "chronic-kidney-disease"],
        timeline_events=[
            {"timestamp": "2026-05-08T09:00:00Z", "event_type": "admission", "description": "Patient admitted via ED", "actor": "ED_NURSE"},
            {"timestamp": "2026-05-08T09:10:00Z", "event_type": "ecg", "description": "12-lead ECG: ST depression lateral leads", "actor": "cardiologist.local"},
            {"timestamp": "2026-05-08T09:20:00Z", "event_type": "lab_result", "description": "Troponin-I elevated. eGFR 42. K+ 4.8.", "actor": "LAB_SYSTEM"},
        ],
    )
}

@router.get("/{encounter_id}", response_model=EncounterDetail,
            summary="Get Encounter", description="Requires `cdss:encounter:read`")
async def get_encounter(
    encounter_id: str,
    current_user: TokenPayload = Depends(require_encounter_read),
) -> EncounterDetail:
    enc = _encounters.get(encounter_id)
    if not enc:
        raise HTTPException(status_code=404, detail=f"Encounter {encounter_id} not found")
    return enc

@router.post("", response_model=EncounterDetail,
             summary="Create Encounter", description="Requires `cdss:encounter:write`")
async def create_encounter(
    body: EncounterCreate,
    current_user: TokenPayload = Depends(require_encounter_write),
) -> EncounterDetail:
    enc = EncounterDetail(
        encounter_id=body.encounter_id, patient_id=body.patient_id,
        encounter_type=body.encounter_type, clinician_id=body.clinician_id,
        status="active", created_at=datetime.now(timezone.utc),
        diagnoses=body.diagnoses,
        timeline_events=[{"timestamp": datetime.now(timezone.utc).isoformat(),
                          "event_type": "encounter_created",
                          "description": body.primary_complaint or "Encounter opened",
                          "actor": body.clinician_id}],
    )
    _encounters[body.encounter_id] = enc
    return enc

@router.get("/{encounter_id}/timeline",
            summary="Encounter Timeline", description="Requires `cdss:encounter:read`")
async def get_timeline(
    encounter_id: str,
    current_user: TokenPayload = Depends(require_encounter_read),
) -> dict:
    enc = _encounters.get(encounter_id)
    if not enc:
        raise HTTPException(status_code=404, detail=f"Encounter {encounter_id} not found")
    return {"encounter_id": encounter_id, "patient_id": enc.patient_id,
            "status": enc.status, "timeline": sorted(enc.timeline_events,
            key=lambda e: e.get("timestamp", ""), reverse=True)}
