"""
CDSS Platform – Safety Gate (Layer 4 Pre-Response Gate)
Validates AI output against clinical safety boundaries before returning to caller.
Aligns with diagram Layer 4: Confidence scoring, Safety gate, Block & Flag.
"""
from __future__ import annotations

import re
from typing import Optional

from loguru import logger

from app.models.schemas import (
    CDSSRecommendationRequest,
    DecisionStatus,
    SafetyFlags,
)
from app.core.config import get_settings

settings = get_settings()


class SafetyGateResult:
    def __init__(
        self,
        passed: bool,
        flags: list[str],
        confidence_score: float,
        decision_status: DecisionStatus,
        block_reason: Optional[str] = None,
    ):
        self.passed = passed
        self.flags = flags
        self.confidence_score = confidence_score
        self.decision_status = decision_status
        self.block_reason = block_reason


class CDSSSafetyGate:
    """
    Layer 4 pre-response safety gate.
    Applies clinical safety rules, LOINC thresholds, and allergy guards.
    """

    # Phrases that must NEVER appear in response if aspirin allergy is present
    ASPIRIN_UNSAFE_PHRASES = [
        "administer aspirin",
        "give aspirin",
        "start aspirin",
        "aspirin 300mg",
        "aspirin 75mg",
        "aspirin loading",
        "take aspirin",
    ]

    def evaluate(
        self,
        request: CDSSRecommendationRequest,
        llm_response_text: str,
        rag_confidence: float,
    ) -> SafetyGateResult:
        """
        Run all safety checks. Returns SafetyGateResult.
        """
        flags: list[str] = []
        safety_flags = SafetyFlags()
        blocked = False
        block_reason = None

        ctx = request.patient_context
        response_lower = llm_response_text.lower()

        # ── 1. Aspirin Allergy Hard Boundary ──────────────────────────────
        has_aspirin_allergy = any(
            "aspirin" in a.lower() for a in ctx.allergies
        ) or any(
            "aspirin" in c.lower() for c in ctx.contraindications
        )

        if has_aspirin_allergy:
            safety_flags.allergy_conflict = True
            for phrase in self.ASPIRIN_UNSAFE_PHRASES:
                if phrase in response_lower:
                    flags.append(f"CRITICAL: Response contains '{phrase}' despite documented aspirin allergy")
                    blocked = True
                    block_reason = (
                        "AI output contained aspirin recommendation despite documented allergy. "
                        "Output blocked per antiplatelet safety boundary (CARDIO-ACS-NSTEMI-2026 §4)."
                    )
                    safety_flags.allergy_conflict = True

        # ── 2. Renal Impairment Checks ────────────────────────────────────
        egfr_raw = ctx.labs.egfr
        if egfr_raw:
            try:
                egfr = float(egfr_raw)
                if egfr < 60:
                    safety_flags.renal_dose_adjustment_required = True
                    flags.append(f"RENAL_ALERT: eGFR {egfr} – dose adjustment required for renally-cleared antithrombotics")
                if egfr < 30:
                    flags.append("RENAL_CRITICAL: eGFR <30 – prasugrel contraindicated; use ticagrelor or clopidogrel; nephrology review required")
                    if "prasugrel" in response_lower:
                        flags.append("SAFETY: Response mentions prasugrel but eGFR <30 (contraindicated)")
                        blocked = True
                        block_reason = "Prasugrel recommended with eGFR <30 – absolute contraindication."
            except ValueError:
                pass

        # ── 3. Potassium Level Monitoring ─────────────────────────────────
        k_raw = ctx.labs.potassium
        if k_raw:
            try:
                k = float(k_raw)
                if k >= 5.0:
                    flags.append(f"ELECTROLYTE_ALERT: Potassium {k} mmol/L – avoid additional K+ supplementation; review ACE-I/ARB")
                elif k >= 4.8:
                    flags.append(f"ELECTROLYTE_WATCH: Potassium {k} mmol/L – borderline high; monitor closely with ACE-I/ARB use")
            except ValueError:
                pass

        # ── 4. Haemodynamic Instability Check ─────────────────────────────
        sbp_raw = ctx.vitals.systolic_bp
        if sbp_raw:
            try:
                sbp = float(sbp_raw)
                if sbp < 90:
                    safety_flags.haemodynamic_instability = True
                    safety_flags.requires_urgent_escalation = True
                    flags.append(f"HAEMODYNAMIC: SBP {sbp} mmHg – haemodynamic instability; urgent cardiology escalation")
                    if "nitrate" in response_lower:
                        flags.append("SAFETY: Nitrate mentioned with low SBP (<90) – contraindicated")
            except ValueError:
                pass

        # ── 5. Human Review Mandate ───────────────────────────────────────
        # Per guideline §10 and diagram Layer 5 (LangGraph Human-in-the-loop)
        # ALL AI ACS recommendations require human review
        requires_human_review = True

        # ── 6. Confidence Scoring ──────────────────────────────────────────
        confidence_score = self._compute_confidence(
            rag_confidence=rag_confidence,
            flags=flags,
            has_evidence=rag_confidence > 0,
            blocked=blocked,
        )

        # ── 7. Determine Status ───────────────────────────────────────────
        if blocked:
            decision_status = DecisionStatus.BLOCKED_SAFETY
        elif flags:
            decision_status = DecisionStatus.FLAGGED
        else:
            decision_status = DecisionStatus.PENDING_REVIEW

        safety_flags.flags = flags

        logger.info(
            f"[SAFETY_GATE] passed={not blocked} flags={len(flags)} "
            f"confidence={confidence_score:.2f} status={decision_status}"
        )

        return SafetyGateResult(
            passed=not blocked,
            flags=flags,
            confidence_score=confidence_score,
            decision_status=decision_status,
            block_reason=block_reason,
        )

    def _compute_confidence(
        self,
        rag_confidence: float,
        flags: list[str],
        has_evidence: bool,
        blocked: bool,
    ) -> float:
        """
        Simple confidence scoring based on RAG retrieval quality and safety flags.
        """
        if blocked:
            return 0.0

        base = rag_confidence if has_evidence else 0.5
        # Penalise for each flag
        penalty = len([f for f in flags if "CRITICAL" in f or "SAFETY" in f]) * 0.15
        penalty += len([f for f in flags if "ALERT" in f or "WATCH" in f]) * 0.05
        return max(0.0, min(1.0, base - penalty))


# Singleton
safety_gate = CDSSSafetyGate()
