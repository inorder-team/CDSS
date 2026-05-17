"""
CDSS Platform – Admin Routes
GET  /api/v1/admin/stats          – System statistics
GET  /api/v1/admin/audit          – Query audit log
POST /api/v1/admin/rag/reindex    – Re-ingest guidelines into ChromaDB
DELETE /api/v1/admin/rag/reset    – Reset ChromaDB collection
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from app.core.security import TokenPayload, require_admin, require_audit_read
from pydantic import BaseModel

from app.core.config import get_settings
from app.rag.rag_engine import rag_engine

settings = get_settings()
router = APIRouter(prefix="/admin", tags=["Admin"])


class SystemStats(BaseModel):
    app_name: str
    version: str
    environment: str
    llm_model: str
    rag_collection: str
    rag_chunk_count: int
    rag_ready: bool
    audit_log_entries: int
    pending_recommendations: int


@router.get("/stats", response_model=SystemStats, summary="System Statistics")
async def get_stats(current_user: TokenPayload = Depends(require_admin)) -> SystemStats:
    from app.api.routes.recommendations import _pending_decisions

    audit_path = Path(settings.audit_log_path)
    audit_count = 0
    if audit_path.exists():
        audit_count = sum(1 for _ in audit_path.open())

    return SystemStats(
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        llm_model=settings.llm_model,
        rag_collection=settings.chroma_collection_clinical,
        rag_chunk_count=rag_engine.collection_count(),
        rag_ready=rag_engine.is_ready,
        audit_log_entries=audit_count,
        pending_recommendations=len(_pending_decisions),
    )


@router.get("/audit", summary="Query Audit Log")
async def query_audit_log(current_user: TokenPayload = Depends(require_audit_read),
    limit: int = Query(50, ge=1, le=500),
    event_type: Optional[str] = Query(None),
    patient_id: Optional[str] = Query(None),
):
    audit_path = Path(settings.audit_log_path)
    if not audit_path.exists():
        return {"entries": [], "total": 0}

    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    entries = []
    for line in reversed(lines):
        try:
            entry = json.loads(line)
            if event_type and entry.get("event") != event_type:
                continue
            if patient_id and entry.get("patient_id") != patient_id:
                continue
            entries.append(entry)
            if len(entries) >= limit:
                break
        except json.JSONDecodeError:
            continue

    return {"entries": entries, "total": len(lines), "returned": len(entries)}


@router.post("/rag/reindex", summary="Re-ingest Guidelines into ChromaDB")
async def reindex_rag(current_user: TokenPayload = Depends(require_admin)) -> dict:
    guideline_dir = Path("./data/guidelines")
    if not guideline_dir.exists():
        raise HTTPException(status_code=404, detail="data/guidelines directory not found")
    count = rag_engine.ingest_guidelines_directory(guideline_dir)
    return {"status": "success", "chunks_indexed": count, "collection": settings.chroma_collection_clinical}


@router.delete("/rag/reset", summary="Reset ChromaDB Collection")
async def reset_rag(current_user: TokenPayload = Depends(require_admin)) -> dict:
    try:
        client = rag_engine._get_client()
        client.delete_collection(settings.chroma_collection_clinical)
        rag_engine._collection = None
        rag_engine._ready = False
        return {"status": "collection_reset", "message": "Run /admin/rag/reindex to rebuild."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit/events", summary="List Unique Audit Event Types")
async def list_event_types(current_user: TokenPayload = Depends(require_audit_read)) -> dict:
    audit_path = Path(settings.audit_log_path)
    if not audit_path.exists():
        return {"event_types": []}
    types: set[str] = set()
    for line in audit_path.open():
        try:
            entry = json.loads(line)
            types.add(entry.get("event", "unknown"))
        except Exception:
            pass
    return {"event_types": sorted(types)}
