"""
CDSS Platform – Medication Routes (doc §12)

Permission guards:
  GET  /api/v1/medications/check         → cdss:medication:read
  GET  /api/v1/medications/{drug_name}   → cdss:medication:read
  POST /api/v1/medications/interactions  → cdss:medication:read
  POST /api/v1/medications/prescribe     → cdss:medication:write
"""
from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.security import TokenPayload, require_medication_read, require_medication_write

router = APIRouter(prefix="/medications", tags=["Medications"])

DRUG_DB = {
    "ticagrelor": {"drug_name": "Ticagrelor", "class": "P2Y12 inhibitor (antiplatelet)", "standard_dose": "180mg loading, then 90mg twice daily", "renal_adjustment": "No dose adjustment required for eGFR >= 15", "contraindications": ["active bleeding", "prior intracranial haemorrhage", "severe hepatic impairment"], "interactions": ["strong CYP3A4 inhibitors", "digoxin", "simvastatin >40mg"], "monitoring": ["dyspnoea", "bleeding signs", "renal function"], "notes": "Preferred P2Y12 inhibitor for NSTEMI. NOT affected by aspirin allergy."},
    "clopidogrel": {"drug_name": "Clopidogrel", "class": "P2Y12 inhibitor (antiplatelet)", "standard_dose": "300-600mg loading, then 75mg daily", "renal_adjustment": "No dose adjustment required", "contraindications": ["active pathological bleeding", "severe hepatic impairment"], "interactions": ["omeprazole", "PPIs generally", "warfarin"], "monitoring": ["bleeding", "platelet count"], "notes": "Alternative antiplatelet. Safe in aspirin allergy."},
    "prasugrel": {"drug_name": "Prasugrel", "class": "P2Y12 inhibitor (antiplatelet)", "standard_dose": "60mg loading, then 10mg daily", "renal_adjustment": "AVOID if eGFR < 30", "contraindications": ["prior TIA or stroke", "active bleeding", "severe hepatic impairment", "eGFR <30"], "interactions": ["anticoagulants", "NSAIDs"], "monitoring": ["bleeding", "renal function"], "notes": "Contraindicated eGFR <30."},
    "fondaparinux": {"drug_name": "Fondaparinux", "class": "Factor Xa inhibitor (anticoagulant)", "standard_dose": "2.5mg subcutaneous once daily", "renal_adjustment": "Contraindicated if eGFR < 20", "contraindications": ["eGFR <20", "active major bleeding"], "interactions": ["other anticoagulants", "NSAIDs"], "monitoring": ["anti-Xa levels", "renal function"], "notes": "Preferred anticoagulant for NSTEMI."},
    "enoxaparin": {"drug_name": "Enoxaparin", "class": "Low-molecular-weight heparin (LMWH)", "standard_dose": "1mg/kg subcutaneous twice daily", "renal_adjustment": "eGFR < 30: 1mg/kg once daily", "contraindications": ["active major bleeding", "HIT", "eGFR <15"], "interactions": ["NSAIDs", "antiplatelet drugs", "warfarin"], "monitoring": ["anti-Xa levels", "platelet count"], "notes": "Dose-reduce to once daily in eGFR <30."},
    "atorvastatin": {"drug_name": "Atorvastatin", "class": "HMG-CoA reductase inhibitor (high-intensity statin)", "standard_dose": "40-80mg once daily", "renal_adjustment": "No dose adjustment required", "contraindications": ["active liver disease", "pregnancy"], "interactions": ["clarithromycin", "itraconazole", "HIV protease inhibitors"], "monitoring": ["LFTs", "CK if myopathy symptoms"], "notes": "Consider maximising to 80mg for NSTEMI."},
    "metoprolol": {"drug_name": "Metoprolol Succinate", "class": "Beta-1 selective adrenergic blocker", "standard_dose": "25-200mg once daily (extended-release)", "renal_adjustment": "No dose adjustment required", "contraindications": ["cardiogenic shock", "decompensated heart failure", "significant bradycardia", "2nd/3rd degree AV block"], "interactions": ["verapamil", "diltiazem", "clonidine"], "monitoring": ["heart rate (target 50-70 bpm)", "blood pressure"], "notes": "Optimise dose for heart rate control in NSTEMI."},
}
INTERACTIONS = {
    frozenset(["ticagrelor", "aspirin"]): {"severity": "monitor", "description": "Standard DAPT. Use low-dose aspirin (75mg).", "recommendation": "Use aspirin 75mg if combining."},
    frozenset(["ticagrelor", "clopidogrel"]): {"severity": "avoid", "description": "Dual P2Y12 – no benefit, increased bleeding.", "recommendation": "Use only one P2Y12 inhibitor."},
    frozenset(["fondaparinux", "enoxaparin"]): {"severity": "contraindicated", "description": "Two anticoagulants – markedly increased bleeding.", "recommendation": "Use only one anticoagulant."},
    frozenset(["metoprolol", "ticagrelor"]): {"severity": "monitor", "description": "Additive bradycardia risk.", "recommendation": "Monitor heart rate; adjust metoprolol dose if needed."},
}

class DrugInfo(BaseModel):
    drug_name: str; drug_class: str; standard_dose: str; renal_adjustment: str
    contraindications: list[str]; interactions: list[str]; monitoring: list[str]; notes: str

class SafetyCheckResult(BaseModel):
    drug: str; patient_id: str; safe_to_use: bool; warnings: list[str]
    contraindications_triggered: list[str]; dose_recommendation: str

class InteractionResult(BaseModel):
    drug_a: str; drug_b: str; severity: str; description: str; recommendation: str

class InteractionCheckRequest(BaseModel):
    drugs: list[str]

class PrescribeRequest(BaseModel):
    drug: str
    patient_id: str
    dose: str


@router.get("/check", response_model=SafetyCheckResult,
            summary="Drug Safety Check", description="Requires `cdss:medication:read`")
async def drug_safety_check(
    drug: str, patient_id: str,
    egfr: Optional[float] = None, allergies: Optional[str] = None,
    current_user: TokenPayload = Depends(require_medication_read),
) -> SafetyCheckResult:
    drug_key = drug.lower().replace(" ", "")
    info = DRUG_DB.get(drug_key)
    if not info:
        raise HTTPException(status_code=404, detail=f"Drug '{drug}' not found in formulary")
    warnings: list[str] = []; contraindications_triggered: list[str] = []; safe = True
    allergy_list = [a.strip().lower() for a in (allergies or "").split(",") if a.strip()]
    if drug_key in allergy_list:
        contraindications_triggered.append(f"Patient has documented allergy to {drug_key}"); safe = False
    if egfr is not None:
        if drug_key == "prasugrel" and egfr < 30:
            contraindications_triggered.append(f"Prasugrel contraindicated: eGFR {egfr} < 30"); safe = False
        elif drug_key == "fondaparinux" and egfr < 20:
            contraindications_triggered.append(f"Fondaparinux contraindicated: eGFR {egfr} < 20"); safe = False
        elif drug_key == "enoxaparin" and egfr < 30:
            warnings.append(f"Enoxaparin: reduce to once daily dosing for eGFR {egfr} < 30")
    dose = info["standard_dose"]
    if egfr and egfr < 30 and drug_key == "enoxaparin":
        dose = "1mg/kg subcutaneous ONCE daily (renal dose-reduced)"
    return SafetyCheckResult(drug=info["drug_name"], patient_id=patient_id, safe_to_use=safe,
                             warnings=warnings, contraindications_triggered=contraindications_triggered,
                             dose_recommendation=dose)


@router.get("/{drug_name}", response_model=DrugInfo,
            summary="Drug Info", description="Requires `cdss:medication:read`")
async def get_drug_info(
    drug_name: str,
    current_user: TokenPayload = Depends(require_medication_read),
) -> DrugInfo:
    key = drug_name.lower().replace(" ", "")
    info = DRUG_DB.get(key)
    if not info:
        raise HTTPException(status_code=404, detail=f"Drug '{drug_name}' not in formulary")
    return DrugInfo(drug_name=info["drug_name"], drug_class=info["class"], standard_dose=info["standard_dose"],
                    renal_adjustment=info["renal_adjustment"], contraindications=info["contraindications"],
                    interactions=info["interactions"], monitoring=info["monitoring"], notes=info["notes"])


@router.post("/interactions", response_model=list[InteractionResult],
             summary="Check Drug-Drug Interactions", description="Requires `cdss:medication:read`")
async def check_interactions(
    body: InteractionCheckRequest,
    current_user: TokenPayload = Depends(require_medication_read),
) -> list[InteractionResult]:
    results: list[InteractionResult] = []
    drug_keys = [d.lower().replace(" ", "") for d in body.drugs]
    for i, a in enumerate(drug_keys):
        for b in drug_keys[i + 1:]:
            interaction = INTERACTIONS.get(frozenset([a, b]))
            if interaction:
                results.append(InteractionResult(drug_a=a, drug_b=b, **interaction))
    return results


@router.post("/prescribe", summary="Prescribe Medication", description="Requires `cdss:medication:write`",
             status_code=201)
async def prescribe_medication(
    body: PrescribeRequest,
    current_user: TokenPayload = Depends(require_medication_write),
) -> dict:
    return {"message": "Prescription recorded", "drug": body.drug, "patient_id": body.patient_id,
            "dose": body.dose, "prescribed_by": current_user.sub}
