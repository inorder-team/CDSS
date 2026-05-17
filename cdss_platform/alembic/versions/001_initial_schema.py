"""Initial schema: recommendations, human_reviews, audit_events, patient_encounters

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-05-09
"""
from alembic import op
import sqlalchemy as sa

revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'patient_encounters',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('patient_id', sa.String(100), nullable=False, index=True),
        sa.Column('encounter_id', sa.String(100), nullable=False),
        sa.Column('encounter_type', sa.String(100)),
        sa.Column('diagnoses_json', sa.JSON),
        sa.Column('created_at', sa.DateTime(timezone=True)),
        sa.UniqueConstraint('patient_id', 'encounter_id', name='uq_patient_encounter'),
    )

    op.create_table(
        'recommendations',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('correlation_id', sa.String(36), unique=True, nullable=False, index=True),
        sa.Column('patient_id', sa.String(100), nullable=False, index=True),
        sa.Column('encounter_id', sa.String(100), nullable=False),
        sa.Column('user_id', sa.String(100), nullable=False),
        sa.Column('user_role', sa.String(50), nullable=False),
        sa.Column('query', sa.Text, nullable=False),
        sa.Column('query_type', sa.String(50)),
        sa.Column('ai_path_used', sa.String(50)),
        sa.Column('rag_driven', sa.Boolean, default=True),
        sa.Column('evidence_count', sa.Integer, default=0),
        sa.Column('confidence_score', sa.Float, default=0.0),
        sa.Column('decision_status', sa.String(50), default='pending_review'),
        sa.Column('requires_human_review', sa.Boolean, default=True),
        sa.Column('summary', sa.Text),
        sa.Column('risk_stratification', sa.Text),
        sa.Column('antiplatelet_guidance', sa.Text),
        sa.Column('invasive_strategy', sa.Text),
        sa.Column('adjunct_therapy', sa.Text),
        sa.Column('monitoring_plan', sa.Text),
        sa.Column('human_review_note', sa.Text),
        sa.Column('safety_flags_json', sa.JSON),
        sa.Column('pipeline_latency_ms', sa.Float),
        sa.Column('llm_model', sa.String(100)),
        sa.Column('created_at', sa.DateTime(timezone=True)),
        sa.Column('updated_at', sa.DateTime(timezone=True)),
    )

    op.create_table(
        'human_reviews',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('correlation_id', sa.String(36), sa.ForeignKey('recommendations.correlation_id'), nullable=False),
        sa.Column('reviewer_id', sa.String(100), nullable=False),
        sa.Column('reviewer_role', sa.String(50), nullable=False),
        sa.Column('action', sa.String(20), nullable=False),
        sa.Column('notes', sa.Text),
        sa.Column('edited_summary', sa.Text),
        sa.Column('final_status', sa.String(50)),
        sa.Column('reviewed_at', sa.DateTime(timezone=True)),
    )

    op.create_table(
        'audit_events',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('event_type', sa.String(60), nullable=False, index=True),
        sa.Column('correlation_id', sa.String(36), index=True),
        sa.Column('user_id', sa.String(100), index=True),
        sa.Column('patient_id', sa.String(100), index=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), index=True),
        sa.Column('payload_json', sa.JSON),
        sa.Column('record_hash', sa.String(64)),
    )


def downgrade() -> None:
    op.drop_table('audit_events')
    op.drop_table('human_reviews')
    op.drop_table('recommendations')
    op.drop_table('patient_encounters')
