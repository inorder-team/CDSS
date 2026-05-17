"""
CDSS Platform – Database Models (SQLAlchemy Async)
Persistent storage for recommendations, audit events, and patient encounters.
Dev: SQLite via aiosqlite
Prod: swap DATABASE_URL to PostgreSQL in .env
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, Text, JSON, UniqueConstraint,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship

from app.core.config import get_settings

settings = get_settings()

# ─────────────────────────────────────────────
# Engine + Session Factory
# ─────────────────────────────────────────────

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    future=True,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ─────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────

class DBRecommendation(Base):
    """Stores every generated CDSS recommendation."""
    __tablename__ = "recommendations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    correlation_id = Column(String(36), unique=True, nullable=False, index=True)
    patient_id = Column(String(100), nullable=False, index=True)
    encounter_id = Column(String(100), nullable=False)
    user_id = Column(String(100), nullable=False)
    user_role = Column(String(50), nullable=False)
    query = Column(Text, nullable=False)
    query_type = Column(String(50))
    ai_path_used = Column(String(50))
    rag_driven = Column(Boolean, default=True)
    evidence_count = Column(Integer, default=0)
    confidence_score = Column(Float, default=0.0)
    decision_status = Column(String(50), default="pending_review")
    requires_human_review = Column(Boolean, default=True)
    summary = Column(Text)
    risk_stratification = Column(Text)
    antiplatelet_guidance = Column(Text)
    invasive_strategy = Column(Text)
    adjunct_therapy = Column(Text)
    monitoring_plan = Column(Text)
    human_review_note = Column(Text)
    safety_flags_json = Column(JSON, default=list)
    pipeline_latency_ms = Column(Float)
    llm_model = Column(String(100))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))

    reviews = relationship("DBHumanReview", back_populates="recommendation", cascade="all, delete-orphan")


class DBHumanReview(Base):
    """Records human review decisions (cardiologist approve/reject/edit)."""
    __tablename__ = "human_reviews"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    correlation_id = Column(String(36), ForeignKey("recommendations.correlation_id"), nullable=False)
    reviewer_id = Column(String(100), nullable=False)
    reviewer_role = Column(String(50), nullable=False)
    action = Column(String(20), nullable=False)   # approve | reject | edit
    notes = Column(Text)
    edited_summary = Column(Text)
    final_status = Column(String(50))
    reviewed_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    recommendation = relationship("DBRecommendation", back_populates="reviews")


class DBAuditEvent(Base):
    """Immutable audit trail (mirrors JSONL log in DB for queryability)."""
    __tablename__ = "audit_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(60), nullable=False, index=True)
    correlation_id = Column(String(36), index=True)
    user_id = Column(String(100), index=True)
    patient_id = Column(String(100), index=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    payload_json = Column(JSON)
    record_hash = Column(String(64))   # SHA-256 for tamper detection


class DBPatientEncounter(Base):
    """Encounter summary (lightweight; full PHI in EMR)."""
    __tablename__ = "patient_encounters"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String(100), nullable=False, index=True)
    encounter_id = Column(String(100), nullable=False)
    encounter_type = Column(String(100))
    diagnoses_json = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (UniqueConstraint("patient_id", "encounter_id"),)


# ─────────────────────────────────────────────
# DB Session Dependency
# ─────────────────────────────────────────────

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all tables (run on startup)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
