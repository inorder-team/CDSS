"""
CDSS Platform – Audit Routes
GET  /api/v1/audit                     – List recent audit events
GET  /api/v1/audit/{correlation_id}    – Events for a specific correlation ID
GET  /api/v1/audit/export              – Export audit log as JSONL
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from app.core.security import TokenPayload, require_audit_read
from fastapi.responses import PlainTextResponse

from app.core.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/audit", tags=["Audit"])


def _read_audit_log(n: int = 200) -> list[dict]:
    log_path = Path(settings.audit_log_path)
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    entries = []
    for line in lines[-n:]:
        try:
            entries.append(json.loads(line))
        except Exception:
            pass
    return entries


@router.get("", summary="List Recent Audit Events")
async def list_audit_events(limit: int = Query(50, le=500),
    current_user: TokenPayload = Depends(require_audit_read)) -> dict:
    events = _read_audit_log(limit)
    return {
        "total_returned": len(events),
        "events": list(reversed(events)),  # newest first
    }


@router.get("/export", response_class=PlainTextResponse, summary="Export Audit Log (JSONL)")
async def export_audit_log(current_user: TokenPayload = Depends(require_audit_read)) -> PlainTextResponse:
    log_path = Path(settings.audit_log_path)
    if not log_path.exists():
        return PlainTextResponse("")
    content = log_path.read_text(encoding="utf-8")
    return PlainTextResponse(
        content,
        media_type="application/x-ndjson",
        headers={"Content-Disposition": "attachment; filename=cdss_audit.jsonl"},
    )


@router.get("/{correlation_id}", summary="Audit Events for Correlation ID")
async def get_audit_by_correlation(correlation_id: str, current_user: TokenPayload = Depends(require_audit_read)) -> dict:
    events = _read_audit_log(5000)
    matched = [e for e in events if e.get("correlation_id") == correlation_id]
    if not matched:
        raise HTTPException(status_code=404, detail=f"No audit events for correlation_id={correlation_id}")
    return {"correlation_id": correlation_id, "events": matched}
