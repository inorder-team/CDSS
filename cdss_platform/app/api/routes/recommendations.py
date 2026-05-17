"""
CDSS Platform – API Routes
Recommendation, Human Review, Health endpoints.
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi import Depends
from loguru import logger

from app.core.audit import audit_logger
from app.core.security import TokenPayload, require_recommendation_create, require_recommendation_read, get_current_user, require_cardiologist
from app.core.config import get_settings
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

# In-memory store for pending reviews (use Redis/DB in full prod)
_pending_decisions: dict[str, CDSSRecommendationResponse] = {}


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check – checks RAG collection availability."""
    chroma_ready = rag_engine.is_ready
    chunk_count = rag_engine.collection_count()
    return HealthResponse(
        status="healthy" if chroma_ready else "degraded",
        version=settings.app_version,
        environment=settings.app_env,
        chroma_ready=chroma_ready,
        llm_model=settings.llm_model,
    )


@router.get("/health/rag", tags=["System"])
async def rag_health():
    """Detailed RAG collection status."""
    return {
        "collection": settings.chroma_collection_clinical,
        "chunk_count": rag_engine.collection_count(),
        "embedding_model": settings.embedding_model,
        "ready": rag_engine.is_ready,
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
    description=(
        "Runs the full 6-layer Clinical Intelligence Pipeline. "
        "Performs RAG retrieval, LLM inference via Claude, safety gate evaluation, "
        "and returns a structured recommendation in pending_review state awaiting "
        "cardiologist human review."
    ),
)
async def create_recommendation(
    request_body: CDSSRecommendationRequest,
    http_request: Request,
    current_user: TokenPayload = Depends(require_recommendation_create),
) -> CDSSRecommendationResponse:
    """
    POST /api/v1/recommendations
    Full CDSS pipeline endpoint.
    """
    correlation_id = str(uuid.uuid4())
    source_ip = http_request.client.host if http_request.client else None

    logger.info(
        f"[API] /recommendations | corr={correlation_id} "
        f"patient={request_body.patient_id} user={request_body.user_id} "
        f"role={request_body.user_role}"
    )

    if not settings.anthropic_api_key or settings.anthropic_api_key == "your_anthropic_api_key_here":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Anthropic API key not configured. Set ANTHROPIC_API_KEY in .env file.",
        )

    try:
        response = await pipeline.run(
            request=request_body,
            correlation_id=correlation_id,
            source_ip=source_ip,
        )
        # Store for human review
        _pending_decisions[correlation_id] = response
        return response

    except Exception as e:
        logger.exception(f"[API] Pipeline error | corr={correlation_id}")
        audit_logger.log_error(
            correlation_id=correlation_id,
            error_type=type(e).__name__,
            error_message=str(e),
            stage="api_recommendation",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {str(e)[:200]}",
        )


# ─────────────────────────────────────────────
# Human Review (Layer 5 – LangGraph node)
# ─────────────────────────────────────────────

@router.post(
    "/recommendations/{correlation_id}/review",
    response_model=HumanReviewResponse,
    tags=["Human Review"],
    summary="Submit Human Review Decision",
    description=(
        "Cardiologist submits approve/reject/edit decision. "
        "Logged immutably per HIPAA audit requirements."
    ),
)
async def submit_human_review(
    correlation_id: str,
    review: HumanReviewRequest,
    current_user: TokenPayload = Depends(require_cardiologist),
) -> HumanReviewResponse:
    """
    POST /api/v1/recommendations/{correlation_id}/review
    """
    if review.action not in ("approve", "reject", "edit"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="action must be one of: approve, reject, edit",
        )

    final_status = {
        "approve": DecisionStatus.APPROVED,
        "reject": DecisionStatus.REJECTED,
        "edit": DecisionStatus.APPROVED,
    }[review.action]

    audit_logger.log_human_review(
        correlation_id=correlation_id,
        reviewer_id=review.reviewer_id,
        reviewer_role=review.reviewer_role,
        action=review.action,
        notes=review.notes,
    )

    # Update stored decision status
    if correlation_id in _pending_decisions:
        _pending_decisions[correlation_id].recommendation.decision_status = final_status

    logger.info(f"[API] Human review submitted | corr={correlation_id} action={review.action}")

    return HumanReviewResponse(
        correlation_id=correlation_id,
        final_status=final_status,
        reviewed_by=review.reviewer_id,
        audit_logged=True,
    )


@router.get(
    "/recommendations/{correlation_id}",
    response_model=CDSSRecommendationResponse,
    tags=["Clinical Decision Support"],
)
async def get_recommendation(
    correlation_id: str,
    current_user: TokenPayload = Depends(require_recommendation_read),
) -> CDSSRecommendationResponse:
    """Retrieve a previously generated recommendation by correlation ID."""
    if correlation_id not in _pending_decisions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No recommendation found for correlation_id: {correlation_id}",
        )
    return _pending_decisions[correlation_id]


@router.get(
    "/recommendations",
    tags=["Clinical Decision Support"],
)
async def list_recommendations(
    limit: int = 20,
    current_user: TokenPayload = Depends(require_recommendation_read),
):
    """List recent recommendations (pending and reviewed)."""
    items = list(_pending_decisions.values())[-limit:]
    return {
        "total": len(_pending_decisions),
        "returned": len(items),
        "recommendations": [
            {
                "correlation_id": r.correlation_id,
                "patient_id": r.patient_id,
                "status": r.recommendation.decision_status,
                "confidence": r.recommendation.confidence_score,
                "requires_review": r.recommendation.requires_human_review,
                "generated_at": r.recommendation.generated_at,
            }
            for r in items
        ],
    }
