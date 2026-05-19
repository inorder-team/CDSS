"""
CDSS Platform – ACS Clinical Decision Support Endpoint
Endpoint  : POST /clinical-decision-support/acs/recommendations
Engine    : Drools KIE Server (python-drools-sdk) + Python fallback
Supports  : STEMI | NSTEMI | Diagnostic CAG
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from loguru import logger
from pydantic import BaseModel, Field

from app.core.audit import audit_logger
from app.core.config import get_settings
from app.core.security import TokenPayload, require_recommendation_create
from app.services.rules.drools_client import execute_acs_rules, DroolsRecommendation

settings = get_settings()
router = APIRouter()


# ────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas (mirrors drools_payload.docx exactly)
# ────────────────────────────────────────────────────────────────────────────

class RuleSet(BaseModel):
    name: str = "ACS_CAG_DECISION_PATHWAY"
    version: str = "1.0.0"
    executionMode: str = "STATELESS"


class PatientInfo(BaseModel):
    patientId: str
    mrn: Optional[str] = None
    ageYears: Optional[int] = None
    sex: Optional[str] = None


class EncounterInfo(BaseModel):
    encounterId: str
    encounterType: Optional[str] = None
    arrivalDateTime: Optional[str] = None
    firstMedicalContactDateTime: Optional[str] = None
    symptomOnsetDateTime: Optional[str] = None


class AcsCaseInput(BaseModel):
    caseId: str
    acsType: str  # "STEMI" | "NSTEMI" | "Diagnostic CAG"
    timiScore: Optional[int] = None
    graceScore: Optional[int] = None
    haemodynamicInstability: bool = False
    electricalInstability: bool = False
    recurrentIschaemia: bool = False
    dynamicStOrTChanges: bool = False
    largeTroponinRise: bool = False
    primaryPciFacilityAvailableCloseBy: bool = True
    expectedFmcToBalloonMinutes: Optional[int] = None
    delayedPresentation: bool = False
    lvDysfunction: bool = False
    viableMyocardium: Optional[bool] = None


class CagFindingInput(BaseModel):
    caseId: Optional[str] = None
    diagnosticCagCompleted: bool = False
    lesionCategory: Optional[str] = None
    maxDiameterStenosisPercent: Optional[float] = None
    numberOfEpicardialVesselsWithSignificantDisease: Optional[int] = None
    culpritVessel: bool = False
    ffrPerformed: bool = False
    ffrValue: Optional[float] = None
    stressImagingPerformed: bool = False
    stressImagingPositiveForIschaemia: Optional[bool] = None


class DiagnosisCodes(BaseModel):
    system: Optional[str] = None
    code: Optional[str] = None
    display: Optional[str] = None


class MedicationsGiven(BaseModel):
    dualAntiplateletTherapyGiven: bool = False
    anticoagulationGiven: bool = False
    statinGiven: bool = False
    betaBlockerGiven: bool = False
    tenecteplaseGiven: bool = False


class VitalsInput(BaseModel):
    systolicBloodPressureMmHg: Optional[float] = None
    diastolicBloodPressureMmHg: Optional[float] = None
    heartRateBpm: Optional[float] = None
    oxygenSaturationPercent: Optional[float] = None


class EcgInput(BaseModel):
    stElevationPresent: bool = False
    dynamicStTChangesPresent: bool = False
    sustainedVtVfPresent: bool = False
    highGradeAvBlockPresent: bool = False


class LabsInput(BaseModel):
    troponinPositive: bool = False
    troponinRisePattern: Optional[str] = None


class EchoInput(BaseModel):
    lvefPercent: Optional[float] = None
    lvDysfunctionPresent: bool = False


class EmrContext(BaseModel):
    diagnosisCodes: list[DiagnosisCodes] = []
    medicationsAlreadyGiven: Optional[MedicationsGiven] = None
    vitals: Optional[VitalsInput] = None
    ecg: Optional[EcgInput] = None
    labs: Optional[LabsInput] = None
    echo: Optional[EchoInput] = None


class AcsRecommendationRequest(BaseModel):
    """Full ACS recommendation request (matches drools_payload.docx)."""
    requestId: str = Field(default_factory=lambda: f"REQ-ACS-{uuid.uuid4().hex[:8].upper()}")
    sourceSystem: str = "EMR"
    tenantId: Optional[str] = None
    facilityId: Optional[str] = None
    requestedAt: Optional[str] = None
    ruleSet: Optional[RuleSet] = None
    patient: Optional[PatientInfo] = None
    encounter: Optional[EncounterInfo] = None
    acsCase: AcsCaseInput
    cagFinding: Optional[CagFindingInput] = None
    emrContext: Optional[EmrContext] = None


# ── Response ────────────────────────────────────────────────────────────────

class RecommendationItem(BaseModel):
    type: str
    code: str
    priority: str
    message: str
    rationale: str
    urgency: str
    source: str = "DROOLS_KIE_SERVER"


class AcsRecommendationResponse(BaseModel):
    requestId: str
    caseId: str
    acsType: str
    processedAt: str
    executionTimeMs: float
    rulesFired: int
    kieServerUsed: bool
    fallbackUsed: bool
    recommendations: list[RecommendationItem]
    nlpSummary: str
    disclaimer: str = (
        "These are AI/rule-engine-generated clinical decision support recommendations. "
        "They require cardiologist review and approval before any clinical action. "
        "Not a substitute for clinical judgment. For validated use only."
    )
    auditLogged: bool = True


# ────────────────────────────────────────────────────────────────────────────
# NLP Formatter – converts Drools output to human-readable clinical text
# ────────────────────────────────────────────────────────────────────────────

_PRIORITY_LABELS = {
    "CRITICAL": "🔴 CRITICAL",
    "URGENT": "🟠 URGENT",
    "HIGH": "🟡 HIGH",
    "MEDIUM": "🔵 MEDIUM",
    "LOW": "⚪ LOW",
}


def _format_nlp_summary(
    acs_type: str,
    recommendations: list[DroolsRecommendation],
) -> str:
    """Generate a human-readable clinical narrative from rule outputs."""
    if not recommendations:
        return (
            f"No specific rule pathway was triggered for ACS Type: {acs_type}. "
            "Please review the patient data and consult clinical guidelines directly."
        )

    lines = [f"**ACS Clinical Pathway Summary – {acs_type}**\n"]

    for i, rec in enumerate(recommendations, start=1):
        priority_label = _PRIORITY_LABELS.get(rec.priority, rec.priority)
        lines.append(
            f"{i}. [{priority_label}] **{rec.type.replace('_', ' ').title()}**\n"
            f"   → {rec.message}\n"
            f"   *Rationale:* {rec.rationale}\n"
            f"   *Urgency:* {rec.urgency}\n"
        )

    lines.append(
        "\n*All recommendations generated by the CDSS Drools Rule Engine "
        "following ESC 2023 ACS Guidelines. Requires cardiologist review before action.*"
    )
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# Endpoint
# ────────────────────────────────────────────────────────────────────────────

@router.post(
    "/acs/recommendations",
    response_model=AcsRecommendationResponse,
    status_code=status.HTTP_200_OK,
    tags=["ACS Clinical Pathways"],
    summary="ACS Clinical Decision Support – Drools Rule Engine",
    description=(
        "Executes deterministic clinical pathway rules for ACS using Drools KIE Server.\n\n"
        "**Supported ACS Types:**\n"
        "- `STEMI` – Primary PCI vs Pharmacoinvasive strategy\n"
        "- `NSTEMI` – Risk stratification, Early Invasive vs Medical\n"
        "- `Diagnostic CAG` – Post-angiography revascularisation decision (PCI/CABG/Medical)\n\n"
        "Rules are sourced from:\n"
        "- `rules/drools-rules/acs_triage.drl`\n"
        "- `rules/drools-rules/cag_decision.drl`\n"
        "- `rules/drools-rules/post_mi_viability.drl`\n\n"
        "Falls back to Python rule engine if KIE Server is unavailable."
    ),
)
async def acs_recommendations(
    request_body: AcsRecommendationRequest,
    http_request: Request,
    current_user: TokenPayload = Depends(require_recommendation_create),
) -> AcsRecommendationResponse:
    """
    POST /clinical-decision-support/acs/recommendations

    Validates the incoming ACS payload, executes Drools rules via KIE Server
    (or Python fallback), formats NLP summary, and returns structured response.
    """
    correlation_id = request_body.requestId
    acs_type = request_body.acsCase.acsType

    # Validate ACS type
    valid_acs_types = {"STEMI", "NSTEMI", "Diagnostic CAG", "UNSTABLE_ANGINA", "UA"}
    if acs_type not in valid_acs_types:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid acsType '{acs_type}'. Must be one of: {sorted(valid_acs_types)}",
        )

    logger.info(
        f"[ACS] Incoming request | requestId={correlation_id} "
        f"acsType={acs_type} "
        f"patient={request_body.patient.patientId if request_body.patient else 'N/A'} "
        f"user={current_user.sub} role={current_user.role}"
    )

    # Build payload dict for Drools client
    payload_dict = {
        "acsCase": request_body.acsCase.model_dump(),
        "cagFinding": request_body.cagFinding.model_dump() if request_body.cagFinding else None,
    }

    # Execute rules
    try:
        drools_result = await execute_acs_rules(
            request_payload=payload_dict,
            request_id=correlation_id,
        )
    except Exception as e:
        logger.exception(f"[ACS] Rule execution error | requestId={correlation_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rule engine error: {str(e)[:300]}",
        )

    # Audit log
        # Audit log – safe wrapper
    try:
        audit_logger.log_decision(
            correlation_id=correlation_id,
            patient_id=request_body.patient.patientId if request_body.patient else "UNKNOWN",
            decision_status="rules_executed",
            confidence=1.0,
            requires_human_review=True,
            recommendation_summary=f"ACS_TYPE={acs_type} RULES_FIRED={drools_result.rule_count_fired}",
        )
    except Exception:
        pass

    # Build response
    rec_items = [
        RecommendationItem(
            type=r.type,
            code=r.code,
            priority=r.priority,
            message=r.message,
            rationale=r.rationale,
            urgency=r.urgency,
            source="KIE_SERVER" if drools_result.kie_server_used else "PYTHON_FALLBACK_ENGINE",
        )
        for r in drools_result.recommendations
    ]

    nlp_summary = _format_nlp_summary(acs_type, drools_result.recommendations)

    logger.info(
        f"[ACS] Response ready | requestId={correlation_id} "
        f"rules_fired={drools_result.rule_count_fired} "
        f"time={drools_result.execution_time_ms}ms "
        f"kie={drools_result.kie_server_used}"
    )

    return AcsRecommendationResponse(
        requestId=correlation_id,
        caseId=drools_result.case_id,
        acsType=acs_type,
        processedAt=datetime.now(timezone.utc).isoformat(),
        executionTimeMs=drools_result.execution_time_ms,
        rulesFired=drools_result.rule_count_fired,
        kieServerUsed=drools_result.kie_server_used,
        fallbackUsed=drools_result.fallback_used,
        recommendations=rec_items,
        nlpSummary=nlp_summary,
    )
