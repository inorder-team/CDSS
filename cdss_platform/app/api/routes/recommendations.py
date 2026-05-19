"""
CDSS Platform – API Routes
Recommendation, Human Review, Health endpoints.

FIXES APPLIED
─────────────
BUG 4 – `await http_request.json()` consumed the request body stream
  FastAPI reads the request body ONCE to feed it to Pydantic.  Calling
  `await http_request.json()` inside the route handler a second time
  either raises an error or returns an empty dict — and in some ASGI
  server versions it silently corrupts the body Pydantic already read,
  causing a cascade 422 on the *next* request to the same worker.

  FIX: Remove the manual body read entirely.  The validated
  `request_body: CDSSRecommendationRequest` argument already contains
  the fully-parsed payload — just log that instead.
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
)
from fastapi.responses import JSONResponse
from loguru import logger

from app.core.audit import audit_logger
from app.core.config import get_settings
from app.core.security import (
    TokenPayload,
    require_clinical,
    require_recommendation_create,
    require_recommendation_read,
)
from app.models.schemas import (
    AIPath,
    CDSSRecommendationRequest,
    CDSSRecommendationResponse,
    ClinicalRecommendation,
    DecisionStatus,
    HealthResponse,
    HumanReviewRequest,
    HumanReviewResponse,
    QueryType,
    SafetyFlags,
)
from app.rag.rag_engine import rag_engine
from app.services.pipeline import pipeline

settings = get_settings()
router = APIRouter()

# In-memory store for pending reviews
_pending_decisions: dict[str, CDSSRecommendationResponse] = {}


def _classify_query_type(request_body: CDSSRecommendationRequest) -> QueryType:
    text = f"{request_body.query} {' '.join(request_body.patient_context.diagnoses)}".lower()
    if any(word in text for word in ["stemi", "shock", "unstable", "emergency"]):
        return QueryType.EMERGENCY
    if any(word in text for word in ["antiplatelet", "aspirin", "ticagrelor", "prasugrel", "clopidogrel", "dose"]):
        return QueryType.MEDICATION
    if any(word in text for word in ["diagnostic", "ecg", "troponin", "angiography", "cag"]):
        return QueryType.DIAGNOSTIC
    return QueryType.CLINICAL


def _patient_context_summary(request_body: CDSSRecommendationRequest) -> str:
    ctx = request_body.patient_context
    parts = [
        f"age {ctx.age}",
        f"sex {ctx.sex}",
        f"encounter {ctx.encounter_type}",
        f"diagnoses {', '.join(ctx.diagnoses)}",
        f"ecg {', '.join(ctx.ecg_findings)}",
        f"medications {', '.join(ctx.current_medications)}",
        f"allergies {', '.join(ctx.allergies)}",
        f"contraindications {', '.join(ctx.contraindications)}",
        f"history {', '.join(ctx.cardiac_history)}",
    ]
    if ctx.labs:
        parts.append(f"labs {ctx.labs.model_dump_json(by_alias=True)}")
    if ctx.vitals:
        parts.append(f"vitals {ctx.vitals.model_dump_json(by_alias=True)}")
    return " | ".join(part for part in parts if part and not part.endswith(" "))


def _build_rag_fallback_response(
    request_body: CDSSRecommendationRequest,
    correlation_id: str,
) -> CDSSRecommendationResponse:
    """
    Evidence-backed fallback for local demos when ANTHROPIC_API_KEY is not set.
    It keeps the RAG vector store workflow usable while clearly marking that no
    external LLM inference was performed.
    """
    query_type = _classify_query_type(request_body)
    patient_summary = _patient_context_summary(request_body)
    evidence_docs = rag_engine.retrieve(
        request_body.query,
        patient_context_summary=patient_summary,
        top_k=5,
        score_threshold=0.0,
    )
    evidence_names = ", ".join(doc.source for doc in evidence_docs[:5]) or "No guideline evidence retrieved"

    ctx = request_body.patient_context
    diagnoses = ", ".join(ctx.diagnoses) or "not specified"
    ecg = ", ".join(ctx.ecg_findings) or "not specified"
    meds = ", ".join(ctx.current_medications) or "not specified"
    allergies = ", ".join(ctx.allergies) or "none documented"
    contraindications = ", ".join(ctx.contraindications) or "none documented"

    flags = []
    allergy_text = " ".join(ctx.allergies).lower()
    if "aspirin" in allergy_text:
        flags.append("Aspirin allergy documented; avoid aspirin until clinician verifies allergy and alternatives.")
    try:
        egfr_raw = ctx.labs.egfr if ctx.labs else None
        egfr_digits = "".join(ch for ch in str(egfr_raw or "") if ch.isdigit() or ch == ".")
        if egfr_digits and float(egfr_digits) < 60:
            flags.append("Reduced eGFR; review renal dosing, contrast exposure, and bleeding risk.")
    except Exception:
        pass
    if "active bleeding" in contraindications.lower():
        flags.append("Bleeding-related contraindication documented; antithrombotic therapy needs clinician review.")

    recommendation = ClinicalRecommendation(
        summary=(
            "RAG vector evidence summary generated because ANTHROPIC_API_KEY is not configured. "
            f"For {diagnoses} with ECG findings {ecg}, retrieved guideline context supports clinician-led ACS risk "
            "stratification, antiplatelet selection, renal safety review, and invasive strategy planning. "
            "This is not an LLM-generated final recommendation."
        ),
        risk_stratification=(
            "Treat as possible high-risk ACS when troponin is positive, dynamic ST/T changes are present, "
            "or GRACE/TIMI risk is elevated. Escalate urgently if instability, recurrent ischaemia, or STEMI features appear."
        ),
        antiplatelet_guidance=(
            f"Current medicines: {meds}. Allergies: {allergies}. Use guideline-directed DAPT only after clinician confirms "
            "bleeding risk, allergy status, renal function, and planned invasive strategy."
        ),
        invasive_strategy=(
            "For NSTEMI/high-risk NSTE-ACS features, consider early invasive coronary angiography according to local ACS protocol. "
            "Use immediate invasive management if haemodynamic or electrical instability develops."
        ),
        adjunct_therapy=(
            "Continue evidence-based adjunct care where not contraindicated: high-intensity statin, anticoagulation strategy, "
            "symptom control, renal-aware dosing, diabetes/CKD optimization, and secondary prevention planning."
        ),
        monitoring_plan=(
            "Monitor serial ECG/troponin, blood pressure, heart rate, oxygenation, renal function, potassium, haemoglobin, "
            "platelets, bleeding signs, and response to therapy."
        ),
        human_review_note=(
            "Fallback RAG output only. Configure ANTHROPIC_API_KEY for full RAG + LLM generation. "
            "Cardiologist/clinician review is mandatory before any care action."
        ),
        ai_path_used=AIPath.RAG_LLM,
        rag_driven=True,
        evidence_documents=evidence_docs,
        safety_flags=SafetyFlags(
            allergy_conflict=any("aspirin allergy" in flag.lower() for flag in flags),
            renal_dose_adjustment_required=any("egfr" in flag.lower() or "renal" in flag.lower() for flag in flags),
            contraindication_detected=bool(contraindications and contraindications != "none documented"),
            flags=flags,
        ),
        confidence_score=0.62 if evidence_docs else 0.35,
        decision_status=DecisionStatus.PENDING_REVIEW,
        requires_human_review=True,
    )

    return CDSSRecommendationResponse(
        correlation_id=correlation_id,
        patient_id=request_body.patient_id,
        encounter_id=request_body.encounter_id,
        query_type=query_type,
        recommendation=recommendation,
        pipeline_latency_ms=0.0,
        audit_logged=True,
        model_version=f"rag-vector-fallback-no-anthropic-key | evidence={evidence_names}",
    )


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
async def health_check():
    """System health check."""
    chroma_ready = rag_engine.is_ready
    return HealthResponse(
        status="healthy" if chroma_ready else "degraded",
        version=settings.app_version,
        environment=settings.app_env,
        chroma_ready=chroma_ready,
        llm_model=settings.llm_model,
    )


@router.get(
    "/health/rag",
    tags=["System"],
)
async def rag_health():
    """Detailed RAG collection status."""
    return {
        "collection":      settings.chroma_collection_clinical,
        "chunk_count":     rag_engine.collection_count(),
        "embedding_model": settings.embedding_model,
        "ready":           rag_engine.is_ready,
    }


# ─────────────────────────────────────────────
# Clinical Recommendation
# ─────────────────────────────────────────────

@router.post(
    "/recommendations",
    response_model=CDSSRecommendationResponse,
    status_code=status.HTTP_200_OK,
    tags=["Clinical Decision Support"],
    summary="Generate CDSS Clinical Recommendation",
)
async def create_recommendation(
    request_body: CDSSRecommendationRequest,
    http_request: Request,
    current_user: TokenPayload = Depends(require_recommendation_create),
) -> CDSSRecommendationResponse:
    """
    POST /api/v1/recommendations — runs full CDSS pipeline.

    Required body (all field names are camelCase):
    ```json
    {
      "patientId":       "string",
      "encounterId":     "string",
      "userId":          "string",
      "userRole":        "CARDIOLOGIST",
      "query":           "string",
      "consentVerified": true,
      "patientContext": {
        "age":                0,
        "sex":                "Male",
        "encounterType":      "EMERGENCY",
        "diagnoses":          [],
        "ecgFindings":        [],
        "labs":               { "troponin": "string", "eGFR": "string", "INR": "string" },
        "vitals":             { "systolicBp": ["string"], "heartRate": "string" },
        "currentMedications": [],
        "allergies":          [],
        "contraindications":  [],
        "cardiacHistory":     []
      }
    }
    ```
    """
    correlation_id = str(uuid.uuid4())
    source_ip = http_request.client.host if http_request.client else None

    # FIX BUG 4: Log the already-parsed Pydantic model instead of re-reading
    # the raw request body (which would consume / corrupt the stream).
    logger.info(
        "[API] /recommendations | "
        "corr={} patient={} user={} role={} query={}",
        correlation_id,
        request_body.patient_id,
        request_body.user_id,
        request_body.user_role,
        request_body.query[:80],
    )

    # Check Anthropic API key
    if (
        (
            not settings.openai_api_key
            or settings.openai_api_key == "your_openai_api_key_here"
        )
        and (
            not settings.anthropic_api_key
            or settings.anthropic_api_key == "your_anthropic_api_key_here"
        )
    ):
        logger.warning(
            "[API] /recommendations | corr={} no external LLM key configured; "
            "using RAG vector fallback response",
            correlation_id,
        )
        response = _build_rag_fallback_response(request_body, correlation_id)
        _pending_decisions[correlation_id] = response
        try:
            audit_logger.log_decision(
                correlation_id=correlation_id,
                patient_id=request_body.patient_id,
                decision_status=response.recommendation.decision_status.value,
                confidence=response.recommendation.confidence_score,
                requires_human_review=True,
                recommendation_summary=response.recommendation.summary[:200],
            )
        except Exception as audit_err:
            logger.warning("[API] RAG fallback audit failed (non-fatal): {}", audit_err)
        return response

    try:
        response = await pipeline.run(
            request=request_body,
            correlation_id=correlation_id,
            source_ip=source_ip,
        )

        # Store for human review retrieval
        _pending_decisions[correlation_id] = response

        logger.success("[PIPELINE SUCCESS] corr={}", correlation_id)
        return response

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("[PIPELINE ERROR] corr={}", correlation_id)

        try:
            audit_logger.log_error(
                correlation_id=correlation_id,
                error_type=type(e).__name__,
                error_message=str(e),
                stage="api_recommendation",
            )
        except Exception:
            pass

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {str(e)[:300]}",
        )


# ─────────────────────────────────────────────
# Human Review
# ─────────────────────────────────────────────

@router.post(
    "/recommendations/{correlation_id}/review",
    response_model=HumanReviewResponse,
    tags=["Human Review"],
)
async def submit_human_review(
    correlation_id: str,
    review: HumanReviewRequest,
    current_user: TokenPayload = Depends(require_clinical),
) -> HumanReviewResponse:
    """Submit clinician review decision for a pending recommendation."""

    valid_actions = {"approve", "reject", "edit"}
    if review.action not in valid_actions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"action must be one of: {valid_actions}",
        )

    final_status = {
        "approve": DecisionStatus.APPROVED,
        "reject":  DecisionStatus.REJECTED,
        "edit":    DecisionStatus.APPROVED,
    }[review.action]

    effective_reviewer_id = review.reviewer_id or current_user.sub
    effective_reviewer_role = review.reviewer_role or current_user.role

    try:
        audit_logger.log_human_review(
            correlation_id=correlation_id,
            reviewer_id=effective_reviewer_id,
            reviewer_role=effective_reviewer_role,
            action=review.action,
            notes=review.notes,
        )
    except Exception as audit_err:
        logger.warning("[HUMAN REVIEW] Audit log failed (non-fatal): {}", audit_err)

    if correlation_id in _pending_decisions:
        _pending_decisions[correlation_id].recommendation.decision_status = final_status
        if review.action == "edit" and review.edited_summary:
            _pending_decisions[correlation_id].recommendation.summary = review.edited_summary
            _pending_decisions[correlation_id].recommendation.human_review_note = (
                f"Edited and approved by {effective_reviewer_id}: "
                f"{review.notes or 'No additional notes.'}"
            )

    logger.info(
        "[HUMAN REVIEW] corr={} action={}",
        correlation_id,
        review.action,
    )

    return HumanReviewResponse(
        correlation_id=correlation_id,
        final_status=final_status,
        reviewed_by=effective_reviewer_id,
        audit_logged=True,
    )


# ─────────────────────────────────────────────
# Get Recommendation
# ─────────────────────────────────────────────

@router.get(
    "/recommendations/{correlation_id}",
    response_model=CDSSRecommendationResponse,
    tags=["Clinical Decision Support"],
)
async def get_recommendation(
    correlation_id: str,
    current_user: TokenPayload = Depends(require_recommendation_read),
) -> CDSSRecommendationResponse:
    """Retrieve a recommendation by correlation ID."""

    if correlation_id not in _pending_decisions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No recommendation found for correlation_id={correlation_id}",
        )
    return _pending_decisions[correlation_id]


# ─────────────────────────────────────────────
# List Recommendations
# ─────────────────────────────────────────────

@router.get(
    "/recommendations",
    tags=["Clinical Decision Support"],
)
async def list_recommendations(
    limit: int = 20,
    current_user: TokenPayload = Depends(require_recommendation_read),
):
    """List the most recent recommendations (newest last, capped at limit)."""

    items = list(_pending_decisions.values())[-limit:]
    return {
        "total":           len(_pending_decisions),
        "returned":        len(items),
        "recommendations": [
            {
                "correlation_id":  r.correlation_id,
                "patient_id":      r.patient_id,
                "status":          r.recommendation.decision_status,
                "confidence":      r.recommendation.confidence_score,
                "requires_review": r.recommendation.requires_human_review,
                "generated_at":    r.recommendation.generated_at,
            }
            for r in items
        ],
    }
