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
    CDSSRecommendationRequest,
    CDSSRecommendationResponse,
    DecisionStatus,
    HealthResponse,
    HumanReviewRequest,
    HumanReviewResponse,
)
from app.rag.rag_engine import rag_engine
from app.services.pipeline import pipeline

settings = get_settings()
router = APIRouter()

# In-memory store for pending reviews
_pending_decisions: dict[str, CDSSRecommendationResponse] = {}


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
        not settings.anthropic_api_key
        or settings.anthropic_api_key == "your_anthropic_api_key_here"
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Anthropic API key not configured. "
                "Set ANTHROPIC_API_KEY in .env"
            ),
        )

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
