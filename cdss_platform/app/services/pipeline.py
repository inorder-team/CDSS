"""
CDSS Platform – Clinical Intelligence Pipeline (CDS Engine)
Orchestrates the full 6-layer pipeline:
  Layer 1: Request Intake
  Layer 2: Validation & Target Object Construction
  Layer 3: AI Path Selection (RAG+LLM / Rules+LLM / Emergency)
  Layer 4: Pre-Response Safety Gate
  Layer 5: LangGraph-style Human-in-the-Loop (status = pending_review)
  Layer 6: JSON REST API Output
"""
from __future__ import annotations

import time
import uuid
from statistics import mean
from typing import Optional

import anthropic
from loguru import logger

from app.core.audit import audit_logger
from app.core.config import get_settings
from app.models.schemas import (
    AIPath,
    CDSSRecommendationRequest,
    CDSSRecommendationResponse,
    ClinicalRecommendation,
    DecisionStatus,
    EvidenceDocument,
    QueryType,
    SafetyFlags,
    UserRole,
)
from app.rag.rag_engine import rag_engine
from app.services.safety_gate import safety_gate

settings = get_settings()

# Singleton async Anthropic client — created once, reused across all requests
_anthropic_client: anthropic.AsyncAnthropic | None = None


def _get_anthropic_client() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _anthropic_client


# ─────────────────────────────────────────────
# Query Router (Layer 2)
# ─────────────────────────────────────────────

def _route_query(request: CDSSRecommendationRequest) -> QueryType:
    """Determine query type from context (mirrors diagram Query Router)."""
    q = request.query.lower()
    ctx = request.patient_context
    diagnoses_lower = [d.lower() for d in ctx.diagnoses]

    if any(term in q for term in ["emergency", "urgent", "instability", "shock"]):
        return QueryType.EMERGENCY
    if any(term in diagnoses_lower for term in ["nstemi", "stemi", "acs"]):
        return QueryType.CLINICAL
    if "medication" in q or "drug" in q or "dose" in q:
        return QueryType.MEDICATION
    if "diagnos" in q or "ecg" in q:
        return QueryType.DIAGNOSTIC
    return QueryType.PROTOCOL


def _select_ai_path(query_type: QueryType, has_evidence: bool) -> AIPath:
    """Select AI inference path (mirrors diagram AI Path selector)."""
    if query_type == QueryType.EMERGENCY:
        return AIPath.EMERGENCY_HITT
    if has_evidence:
        return AIPath.RAG_LLM
    return AIPath.LLM_REASONING


# ─────────────────────────────────────────────
# Prompt Builder (Layer 3)
# ─────────────────────────────────────────────

def _build_system_prompt() -> str:
    return """You are CDSS Clinical Intelligence – an expert AI clinical decision support engine integrated into a HIPAA/NDHM/ABDM compliant enterprise platform.

Your role: Provide structured, evidence-based, guideline-aligned clinical decision support for cardiologists managing NSTEMI/ACS patients.

CRITICAL RULES:
1. You MUST follow clinical guidelines exactly. Never contradict guideline-cited evidence.
2. ASPIRIN ALLERGY: If the patient has a documented aspirin allergy or contraindication to aspirin, you MUST NOT recommend aspirin as a direct clinical action. Instead, recommend guideline-compliant alternatives (clopidogrel monotherapy or ticagrelor) and flag that cardiologist validation is required.
3. RENAL IMPAIRMENT: Always account for eGFR when recommending antithrombotics. Prasugrel is contraindicated if eGFR <30.
4. HUMAN REVIEW: Every recommendation you generate MUST end with a clear statement that it requires cardiologist review before clinical action.
5. OUTPUT FORMAT: Return ONLY a valid JSON object with exactly these fields (no markdown, no preamble):
{
  "summary": "...",
  "risk_stratification": "...",
  "antiplatelet_guidance": "...",
  "invasive_strategy": "...",
  "adjunct_therapy": "...",
  "monitoring_plan": "...",
  "human_review_note": "...",
  "confidence_reasoning": "..."
}"""


def _build_user_prompt(
    request: CDSSRecommendationRequest,
    evidence_docs: list[EvidenceDocument],
) -> str:
    ctx = request.patient_context

    evidence_text = ""
    if evidence_docs:
        evidence_text = "\n\n--- RETRIEVED GUIDELINE EVIDENCE ---\n"
        for i, doc in enumerate(evidence_docs[:5], 1):
            evidence_text += (
                f"\n[Evidence {i}] Source: {doc.doc_id} | "
                f"Relevance: {doc.similarity_score:.2f} | Tag: {doc.relevance_tag}\n"
                f"{doc.content_snippet}\n"
            )

    labs = ctx.labs
    vitals = ctx.vitals

    return f"""CLINICAL QUERY: {request.query}

PATIENT PROFILE:
- Patient ID: {request.patient_id}
- Age: {ctx.age} years | Sex: {ctx.sex}
- Encounter Type: {ctx.encounter_type}
- Requesting Clinician Role: {request.user_role.value}

DIAGNOSES: {', '.join(ctx.diagnoses)}
ECG FINDINGS: {', '.join(ctx.ecg_findings)}

LABORATORY VALUES:
- Troponin: {labs.troponin or 'not provided'}
- eGFR: {labs.egfr or 'not provided'} mL/min/1.73m²
- Potassium: {labs.potassium or 'not provided'} mmol/L

VITAL SIGNS:
- Systolic BP: {vitals.systolic_bp or 'not provided'} mmHg
- Heart Rate: {vitals.heart_rate or 'not provided'} bpm

CURRENT MEDICATIONS: {', '.join(ctx.current_medications) or 'None documented'}
ALLERGIES: {', '.join(ctx.allergies) or 'None documented'}
CONTRAINDICATIONS: {', '.join(ctx.contraindications) or 'None'}
CARDIAC HISTORY: {', '.join(ctx.cardiac_history) or 'None'}

⚠️  CRITICAL ALLERGY FLAG: {'ASPIRIN ALLERGY DOCUMENTED – Do NOT recommend aspirin as direct action' if any('aspirin' in a.lower() for a in ctx.allergies) else 'No aspirin allergy on record'}

{evidence_text}

Based on the above patient data and retrieved guideline evidence, generate a structured NSTEMI management recommendation in the exact JSON format specified. Account for ALL patient-specific factors including allergies, renal function, and comorbidities."""


# ─────────────────────────────────────────────
# LLM Client (Anthropic)
# ─────────────────────────────────────────────

async def _call_claude(system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
    """
    Call Anthropic Claude asynchronously. Returns (text, prompt_tokens, completion_tokens).
    """
    client = _get_anthropic_client()
    message = await client.messages.create(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(
        block.text for block in message.content if hasattr(block, "text")
    )
    return (
        text,
        message.usage.input_tokens,
        message.usage.output_tokens,
    )


# ─────────────────────────────────────────────
# Response Parser
# ─────────────────────────────────────────────

def _parse_llm_json(raw: str) -> dict:
    """Parse JSON from LLM output, stripping markdown fences."""
    import json
    import re

    # Strip ```json ... ``` fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()

    # Find first { ... } block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {raw[:200]}")
    return json.loads(match.group())


# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────

class CDSSPipeline:

    async def run(
        self,
        request: CDSSRecommendationRequest,
        correlation_id: str,
        source_ip: Optional[str] = None,
    ) -> CDSSRecommendationResponse:
        """
        Execute the full 6-layer CDSS pipeline.
        """
        pipeline_start = time.perf_counter()

        # ── Layer 1: Request Intake + Audit ───────────────────────────────
        audit_logger.log_request(
            correlation_id=correlation_id,
            user_id=request.user_id,
            user_role=request.user_role.value,
            patient_id=request.patient_id,
            encounter_id=request.encounter_id,
            endpoint="/api/v1/recommendations",
            query=request.query,
            ip_address=source_ip,
        )

        # ── Layer 2: Query Routing ─────────────────────────────────────────
        query_type = _route_query(request)
        logger.info(f"[PIPELINE] query_type={query_type} corr={correlation_id}")

        # ── Layer 3a: RAG Retrieval ────────────────────────────────────────
        patient_summary = (
            f"age {request.patient_context.age} "
            f"{request.patient_context.sex} "
            f"diagnoses: {' '.join(request.patient_context.diagnoses)} "
            f"allergies: {' '.join(request.patient_context.allergies)} "
            f"eGFR: {request.patient_context.labs.egfr}"
        )

        evidence_docs: list[EvidenceDocument] = []
        rag_confidence = 0.0
        # Fix: compare QueryType to QueryType (not AIPath)
        rag_driven = query_type != QueryType.EMERGENCY

        if rag_driven:
            evidence_docs = rag_engine.retrieve(
                query=request.query,
                patient_context_summary=patient_summary,
                top_k=settings.rag_top_k,
            )
            if evidence_docs:
                rag_confidence = mean(d.similarity_score for d in evidence_docs)

            audit_logger.log_rag_retrieval(
                correlation_id=correlation_id,
                documents_retrieved=len(evidence_docs),
                top_scores=[d.similarity_score for d in evidence_docs[:3]],
                collection=settings.chroma_collection_clinical,
            )

        # ── Layer 3b: AI Path Selection ───────────────────────────────────
        ai_path = _select_ai_path(query_type, has_evidence=bool(evidence_docs))
        logger.info(f"[PIPELINE] ai_path={ai_path} rag_docs={len(evidence_docs)}")

        # ── Layer 3c: LLM Inference ───────────────────────────────────────
        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(request, evidence_docs)

        llm_start = time.perf_counter()
        raw_text, prompt_tokens, completion_tokens = await _call_claude(system_prompt, user_prompt)
        llm_latency = (time.perf_counter() - llm_start) * 1000

        audit_logger.log_llm_call(
            correlation_id=correlation_id,
            model=settings.llm_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=llm_latency,
            rag_driven=bool(evidence_docs),
        )

        # Parse structured output
        try:
            llm_data = _parse_llm_json(raw_text)
        except Exception as e:
            audit_logger.log_error(
                correlation_id=correlation_id,
                error_type="json_parse_error",
                error_message=str(e),
                stage="llm_response_parsing",
            )
            llm_data = {
                "summary": raw_text[:500],
                "risk_stratification": "Unable to parse structured output.",
                "antiplatelet_guidance": "Manual review required.",
                "invasive_strategy": "Manual review required.",
                "adjunct_therapy": "Manual review required.",
                "monitoring_plan": "Manual review required.",
                "human_review_note": "AI output parsing failed. Full cardiologist review mandatory.",
                "confidence_reasoning": "Parse error",
            }

        # ── Layer 4: Safety Gate ───────────────────────────────────────────
        gate_result = safety_gate.evaluate(
            request=request,
            llm_response_text=raw_text,
            rag_confidence=rag_confidence,
        )

        audit_logger.log_safety_gate(
            correlation_id=correlation_id,
            passed=gate_result.passed,
            flags=gate_result.flags,
            confidence_score=gate_result.confidence_score,
        )

        # If blocked, override content with safety message
        if not gate_result.passed:
            llm_data["summary"] = (
                f"⚠️ RECOMMENDATION BLOCKED BY SAFETY GATE: {gate_result.block_reason} "
                "A cardiologist must review this case manually."
            )
            llm_data["antiplatelet_guidance"] = (
                "Blocked – safety gate flagged a contraindication conflict. "
                "Cardiologist validation required before any antiplatelet therapy."
            )

        # ── Layer 5: Human-in-the-Loop (LangGraph node = pending_review) ──
        decision_status = gate_result.decision_status

        safety_flags_model = SafetyFlags(
            allergy_conflict=any("aspirin" in f.lower() and "allergy" in f.lower() for f in gate_result.flags),
            renal_dose_adjustment_required=any("renal" in f.lower() for f in gate_result.flags),
            haemodynamic_instability=any("haemodynamic" in f.lower() for f in gate_result.flags),
            requires_urgent_escalation=any("urgent" in f.lower() or "critical" in f.lower() for f in gate_result.flags),
            flags=gate_result.flags,
        )

        recommendation = ClinicalRecommendation(
            summary=llm_data.get("summary", ""),
            risk_stratification=llm_data.get("risk_stratification", ""),
            antiplatelet_guidance=llm_data.get("antiplatelet_guidance", ""),
            invasive_strategy=llm_data.get("invasive_strategy", ""),
            adjunct_therapy=llm_data.get("adjunct_therapy", ""),
            monitoring_plan=llm_data.get("monitoring_plan", ""),
            human_review_note=llm_data.get(
                "human_review_note",
                "This AI recommendation requires cardiologist review before any care action."
            ),
            ai_path_used=ai_path,
            rag_driven=bool(evidence_docs),
            evidence_documents=evidence_docs,
            safety_flags=safety_flags_model,
            confidence_score=gate_result.confidence_score,
            decision_status=decision_status,
            requires_human_review=True,
        )

        total_latency = (time.perf_counter() - pipeline_start) * 1000

        # ── Audit: Decision ───────────────────────────────────────────────
        audit_logger.log_decision(
            correlation_id=correlation_id,
            patient_id=request.patient_id,
            decision_status=decision_status.value,
            confidence=gate_result.confidence_score,
            requires_human_review=True,
            recommendation_summary=recommendation.summary[:200],
        )

        # ── Layer 6: JSON REST Response ───────────────────────────────────
        return CDSSRecommendationResponse(
            correlation_id=correlation_id,
            patient_id=request.patient_id,
            encounter_id=request.encounter_id,
            query_type=query_type,
            recommendation=recommendation,
            pipeline_latency_ms=round(total_latency, 2),
            audit_logged=True,
            model_version=settings.llm_model,
        )


# Singleton
pipeline = CDSSPipeline()
