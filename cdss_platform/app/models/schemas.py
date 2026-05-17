"""
CDSS Platform – Data Models
Full Pydantic v2 models for API request/response, clinical context, and pipeline objects.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class UserRole(str, Enum):
    CARDIOLOGIST = "CARDIOLOGIST"
    CLINICIAN = "CLINICIAN"
    NURSE = "NURSE"
    PHARMACIST = "PHARMACIST"
    PATIENT = "CDSS_PATIENT"
    ADMIN = "CDSS_ADMIN"
    MANAGEMENT = "CDSS_MANAGEMENT"


class QueryType(str, Enum):
    CLINICAL = "clinical"
    DIAGNOSTIC = "diagnostic"
    EMERGENCY = "emergency"
    PROTOCOL = "protocol"
    MEDICATION = "medication"


class DecisionStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    BLOCKED_SAFETY = "blocked_safety"
    FLAGGED = "flagged"


class AIPath(str, Enum):
    RAG_LLM = "rag_llm"
    LLM_REASONING = "llm_reasoning_rag"
    RULES_LLM = "rules_llm"
    EMERGENCY_HITT = "emergency_hitt_no_rag"


# ─────────────────────────────────────────────
# Sub-models
# ─────────────────────────────────────────────

class LabValues(BaseModel):
    troponin: Optional[str] = None
    egfr: Optional[str] = Field(None, alias="eGFR")
    potassium: Optional[str] = None
    creatinine: Optional[str] = None
    haemoglobin: Optional[str] = None
    platelets: Optional[str] = None
    inr: Optional[str] = Field(None, alias="INR")

    model_config = {"populate_by_name": True}


class Vitals(BaseModel):
    systolic_bp: list[str] = Field(default=[], alias="systolicBp")
    diastolic_bp: list[str] = Field(default=[], alias="diastolicBp")
    heart_rate: Optional[str] = Field(None, alias="heartRate")
    spo2: Optional[str] = None
    respiratory_rate: Optional[str] = Field(None, alias="respiratoryRate")
    temperature: Optional[str] = None

    model_config = {"populate_by_name": True}


class PatientContext(BaseModel):
    age: int
    sex: str
    encounter_type: str = Field(alias="encounterType")
    diagnoses: list[str] = []
    ecg_findings: list[str] = Field(default=[], alias="ecgFindings")
    labs: LabValues = Field(default_factory=LabValues)
    vitals: Vitals = Field(default_factory=Vitals)
    current_medications: list[str] = Field(default=[], alias="currentMedications")
    allergies: list[str] = []
    contraindications: list[str] = []
    cardiac_history: list[str] = Field(default=[], alias="cardiacHistory")

    model_config = {"populate_by_name": True}


# ─────────────────────────────────────────────
# Request / Response
# ─────────────────────────────────────────────

class CDSSRecommendationRequest(BaseModel):
    patient_id: str = Field(alias="patientId")
    encounter_id: str = Field(alias="encounterId")
    user_id: str = Field(alias="userId")
    user_role: UserRole = Field(alias="userRole")
    query: str
    consent_verified: bool = Field(alias="consentVerified")
    patient_context: PatientContext = Field(alias="patientContext")

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def validate_consent(self) -> "CDSSRecommendationRequest":
        if not self.consent_verified:
            raise ValueError("Consent must be verified before clinical decision support can be provided.")
        return self


class EvidenceDocument(BaseModel):
    doc_id: str
    source: str
    content_snippet: str
    similarity_score: float
    source_type: str = "guideline"
    relevance_tag: Optional[str] = None


class SafetyFlags(BaseModel):
    allergy_conflict: bool = False
    contraindication_detected: bool = False
    renal_dose_adjustment_required: bool = False
    high_bleeding_risk: bool = False
    haemodynamic_instability: bool = False
    requires_urgent_escalation: bool = False
    flags: list[str] = []


class ClinicalRecommendation(BaseModel):
    recommendation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    summary: str
    risk_stratification: str
    antiplatelet_guidance: str
    invasive_strategy: str
    adjunct_therapy: str
    monitoring_plan: str
    human_review_note: str
    ai_path_used: AIPath
    rag_driven: bool
    evidence_documents: list[EvidenceDocument] = []
    safety_flags: SafetyFlags = Field(default_factory=SafetyFlags)
    confidence_score: float = Field(ge=0.0, le=1.0)
    decision_status: DecisionStatus = DecisionStatus.PENDING_REVIEW
    requires_human_review: bool = True
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CDSSRecommendationResponse(BaseModel):
    correlation_id: str
    patient_id: str
    encounter_id: str
    query_type: QueryType
    recommendation: ClinicalRecommendation
    pipeline_latency_ms: float
    audit_logged: bool = True
    model_version: str
    disclaimer: str = (
        "This output is AI-generated clinical decision support. "
        "It requires cardiologist review before any care action. "
        "Not a substitute for clinical judgment."
    )


class HumanReviewRequest(BaseModel):
    correlation_id: str
    reviewer_id: str
    reviewer_role: str
    action: str  # "approve" | "reject" | "edit"
    notes: Optional[str] = None
    edited_summary: Optional[str] = None


class HumanReviewResponse(BaseModel):
    correlation_id: str
    final_status: DecisionStatus
    reviewed_by: str
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    audit_logged: bool = True


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    chroma_ready: bool
    llm_model: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
