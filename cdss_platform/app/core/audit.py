"""
CDSS Platform – Immutable Audit Logger
HIPAA/NDHM/ABDM compliant. Every request, decision, and action is logged
with a correlation ID, user identity, timestamps, and decision outcome.
"""
import json
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger as loguru_logger

from app.core.config import get_settings

settings = get_settings()


def _compute_hash(record: dict) -> str:
    """SHA-256 of the record for tamper-evidence."""
    canonical = json.dumps(record, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class AuditLogger:
    """
    Append-only JSONL audit log.
    Each line is a self-contained JSON record with a tamper-evident hash.
    """

    def __init__(self):
        self.log_path = Path(settings.audit_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, record: dict) -> None:
        record["_hash"] = _compute_hash({k: v for k, v in record.items() if k != "_hash"})
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_request(
        self,
        *,
        correlation_id: str,
        user_id: str,
        user_role: str,
        patient_id: str,
        encounter_id: str,
        endpoint: str,
        query: str,
        ip_address: Optional[str] = None,
    ) -> None:
        record = {
            "event": "cdss_request",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": correlation_id,
            "user_id": user_id,
            "user_role": user_role,
            "patient_id": patient_id,
            "encounter_id": encounter_id,
            "endpoint": endpoint,
            "query_snippet": query[:120],
            "source_ip": ip_address,
        }
        self._write(record)
        loguru_logger.info(f"[AUDIT] REQUEST | corr={correlation_id} user={user_id} patient={patient_id}")

    def log_rag_retrieval(
        self,
        *,
        correlation_id: str,
        documents_retrieved: int,
        top_scores: list[float],
        collection: str,
    ) -> None:
        record = {
            "event": "rag_retrieval",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": correlation_id,
            "collection": collection,
            "documents_retrieved": documents_retrieved,
            "top_similarity_scores": top_scores,
        }
        self._write(record)

    def log_llm_call(
        self,
        *,
        correlation_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        rag_driven: bool,
    ) -> None:
        record = {
            "event": "llm_inference",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": correlation_id,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": round(latency_ms, 2),
            "rag_driven": rag_driven,
        }
        self._write(record)

    def log_safety_gate(
        self,
        *,
        correlation_id: str,
        passed: bool,
        flags: list[str],
        confidence_score: float,
    ) -> None:
        record = {
            "event": "safety_gate_evaluation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": correlation_id,
            "safety_passed": passed,
            "flags_raised": flags,
            "confidence_score": confidence_score,
        }
        self._write(record)
        if not passed:
            loguru_logger.warning(f"[SAFETY] GATE BLOCKED | corr={correlation_id} flags={flags}")

    def log_decision(
        self,
        *,
        correlation_id: str,
        patient_id: str,
        decision_status: str,
        confidence: float,
        requires_human_review: bool,
        recommendation_summary: str,
    ) -> None:
        record = {
            "event": "clinical_decision",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": correlation_id,
            "patient_id": patient_id,
            "decision_status": decision_status,
            "confidence_score": confidence,
            "requires_human_review": requires_human_review,
            "recommendation_summary": recommendation_summary[:200],
        }
        self._write(record)
        loguru_logger.info(
            f"[AUDIT] DECISION | corr={correlation_id} status={decision_status} "
            f"conf={confidence:.2f} human_review={requires_human_review}"
        )

    def log_human_review(
        self,
        *,
        correlation_id: str,
        reviewer_id: str,
        reviewer_role: str,
        action: str,
        notes: Optional[str] = None,
    ) -> None:
        record = {
            "event": "human_review",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": correlation_id,
            "reviewer_id": reviewer_id,
            "reviewer_role": reviewer_role,
            "action": action,
            "notes": notes,
        }
        self._write(record)
        loguru_logger.info(f"[AUDIT] HUMAN_REVIEW | corr={correlation_id} action={action} reviewer={reviewer_id}")

    def log_error(
        self,
        *,
        correlation_id: str,
        error_type: str,
        error_message: str,
        stage: str,
    ) -> None:
        record = {
            "event": "pipeline_error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": correlation_id,
            "stage": stage,
            "error_type": error_type,
            "error_message": error_message[:300],
        }
        self._write(record)
        loguru_logger.error(f"[AUDIT] ERROR | corr={correlation_id} stage={stage} err={error_message[:80]}")


# Singleton
audit_logger = AuditLogger()
