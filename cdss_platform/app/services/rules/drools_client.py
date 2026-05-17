"""
CDSS Platform – Drools / KIE Server Client
Implements the python-drools-sdk pattern for stateless rule execution
against a KIE Server container.

KIE Server: http://localhost:8180/kie-server (configurable via .env)
Container : acs-clinical-rules (auto-deployed on startup)
Rules     : rules/drools-rules/*.drl
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import httpx
from loguru import logger

from app.core.config import get_settings

settings = get_settings()


# ────────────────────────────────────────────────────────────────────────────
# Enums mirroring Drools model classes
# ────────────────────────────────────────────────────────────────────────────

class AcsType(str, Enum):
    STEMI = "STEMI"
    NSTEMI = "NSTEMI"
    UNSTABLE_ANGINA = "UNSTABLE_ANGINA"
    DIAGNOSTIC_CAG = "Diagnostic CAG"


class LesionCategory(str, Enum):
    ONE_TO_TWO_VESSELS = "ONE_TO_TWO_VESSELS"
    MULTIVESSEL = "MULTIVESSEL"
    BORDERLINE_50_TO_69 = "BORDERLINE_50_TO_69"
    LESS_THAN_50 = "LESS_THAN_50"


# ────────────────────────────────────────────────────────────────────────────
# Input / Output Dataclasses
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class AcsCase:
    acsType: str
    timiScore: Optional[int] = None
    graceScore: Optional[int] = None
    haemodynamicInstability: bool = False
    electricalInstability: bool = False
    recurrentIschaemia: bool = False
    dynamicStOrTChanges: bool = False
    largeTroponinRise: bool = False
    primaryPciFacilityAvailableCloseBy: bool = True
    expectedFmcToBalloonMinutes: Optional[int] = None
    delayedPresentation: bool = False
    lvDysfunction: bool = False
    viableMyocardium: Optional[bool] = None


@dataclass
class CagFinding:
    diagnosticCagCompleted: bool = False
    lesionCategory: Optional[str] = None
    maxDiameterStenosisPercent: Optional[float] = None
    numberOfEpicardialVesselsWithSignificantDisease: Optional[int] = None
    culpritVessel: bool = False
    ffrPerformed: bool = False
    ffrValue: Optional[float] = None
    stressImagingPerformed: bool = False
    stressImagingPositiveForIschaemia: Optional[bool] = None


@dataclass
class DroolsRecommendation:
    type: str
    code: str
    message: str
    rationale: str
    urgency: str
    priority: str = "HIGH"
    source: str = "DROOLS_KIE_SERVER"


@dataclass
class DroolsResponse:
    request_id: str
    case_id: str
    acs_type: str
    recommendations: list[DroolsRecommendation] = field(default_factory=list)
    execution_time_ms: float = 0.0
    rule_count_fired: int = 0
    kie_server_used: bool = True
    fallback_used: bool = False
    error: Optional[str] = None


# ────────────────────────────────────────────────────────────────────────────
# KIE Server REST Client
# ────────────────────────────────────────────────────────────────────────────

KIE_BASE_URL = getattr(settings, "kie_server_url", "http://localhost:8180")
KIE_USER = getattr(settings, "kie_server_user", "kieserver")
KIE_PASS = getattr(settings, "kie_server_password", "kieserver1!")
KIE_CONTAINER = getattr(settings, "kie_container_id", "acs-clinical-rules")
KIE_TIMEOUT = 10  # seconds


def _map_lesion_category(raw: Optional[str]) -> Optional[str]:
    """Normalise lesion category string from payload to DRL enum name."""
    mapping = {
        "GE_70_ONE_TWO_VESSEL": LesionCategory.ONE_TO_TWO_VESSELS.value,
        "ONE_TO_TWO_VESSELS": LesionCategory.ONE_TO_TWO_VESSELS.value,
        "GE_70_MULTIVESSEL": LesionCategory.MULTIVESSEL.value,
        "MULTIVESSEL": LesionCategory.MULTIVESSEL.value,
        "BORDERLINE_50_TO_69": LesionCategory.BORDERLINE_50_TO_69.value,
        "BORDERLINE": LesionCategory.BORDERLINE_50_TO_69.value,
        "LESS_THAN_50": LesionCategory.LESS_THAN_50.value,
        "LT_50": LesionCategory.LESS_THAN_50.value,
    }
    return mapping.get(raw or "", raw)


def _build_kie_payload(
    acs_case: AcsCase,
    cag_finding: Optional[CagFinding],
    request_id: str,
    case_id: str,
) -> dict:
    """Build stateless KIE Server facts payload."""
    facts: list[dict] = [
        {
            "com.inorder.clinical.acs.model.AcsCase": {
                "caseId": case_id,
                "acsType": acs_case.acsType,
                "timiScore": acs_case.timiScore,
                "graceScore": acs_case.graceScore,
                "haemodynamicInstability": acs_case.haemodynamicInstability,
                "electricalInstability": acs_case.electricalInstability,
                "recurrentIschaemia": acs_case.recurrentIschaemia,
                "dynamicStOrTChanges": acs_case.dynamicStOrTChanges,
                "largeTroponinRise": acs_case.largeTroponinRise,
                "primaryPciFacilityAvailableCloseBy": acs_case.primaryPciFacilityAvailableCloseBy,
                "expectedFmcToBalloonMinutes": acs_case.expectedFmcToBalloonMinutes,
                "delayedPresentation": acs_case.delayedPresentation,
                "lvDysfunction": acs_case.lvDysfunction,
                "viableMyocardium": acs_case.viableMyocardium,
            }
        }
    ]

    if cag_finding and cag_finding.diagnosticCagCompleted:
        facts.append({
            "com.inorder.clinical.acs.model.CagFinding": {
                "caseId": case_id,
                "diagnosticCagCompleted": cag_finding.diagnosticCagCompleted,
                "lesionCategory": _map_lesion_category(cag_finding.lesionCategory),
                "maxDiameterStenosisPercent": cag_finding.maxDiameterStenosisPercent,
                "ffrPerformed": cag_finding.ffrPerformed,
                "ffrValue": cag_finding.ffrValue,
                "stressImagingPerformed": cag_finding.stressImagingPerformed,
                "stressImagingPositiveForIschaemia": cag_finding.stressImagingPositiveForIschaemia,
            }
        })

    return {
        "lookup": f"{KIE_CONTAINER}_stateless",
        "commands": [
            {"batch-execution": {"commands": [{"insert": {"object": f}} for f in facts]
            + [{"fire-all-rules": {}}, {"get-objects": {"out-identifier": "recommendations"}}]}}
        ],
    }


async def httpx(payload: dict) -> Optional[dict]:
    """POST to KIE Server and return raw JSON response."""
    url = (
        f"{KIE_BASE_URL}/kie-server/services/rest/server"
        f"/containers/{KIE_CONTAINER}/ksession/stateless"
    )
    try:
        async with httpx.AsyncClient(timeout=KIE_TIMEOUT) as client:
            response = await client.post(
                url,
                json=payload,
                auth=(KIE_USER, KIE_PASS),
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError:
        logger.warning("[KIE] KIE Server not reachable – will use Python fallback engine")
        return None
    except httpx.HTTPStatusError as e:
        logger.error(f"[KIE] HTTP error {e.response.status_code}: {e.response.text[:300]}")
        return None
    except Exception as e:
        logger.error(f"[KIE] Unexpected error: {e}")
        return None


def _parse_kie_response(raw: dict) -> list[DroolsRecommendation]:
    """Parse KIE Server batch-execution response into DroolsRecommendation list."""
    recommendations: list[DroolsRecommendation] = []
    try:
        results = raw.get("result", {}).get("execution-results", {}).get("results", [])
        for result_item in results:
            value = result_item.get("value", {})
            objects = value.get("objects", []) if isinstance(value, dict) else []
            for obj in objects:
                rec_data = obj.get("com.inorder.clinical.acs.model.Recommendation", {})
                if rec_data:
                    recommendations.append(
                        DroolsRecommendation(
                            type=rec_data.get("type", "UNKNOWN"),
                            code=rec_data.get("code", ""),
                            message=rec_data.get("message", ""),
                            rationale=rec_data.get("rationale", ""),
                            urgency=rec_data.get("urgency", "Standard"),
                            priority=rec_data.get("priority", "HIGH"),
                        )
                    )
    except Exception as e:
        logger.error(f"[KIE] Response parse error: {e}")
    return recommendations


# ────────────────────────────────────────────────────────────────────────────
# Python Fallback Engine (mirrors DRL logic exactly)
# ────────────────────────────────────────────────────────────────────────────

def _python_fallback_engine(
    acs_case: AcsCase,
    cag_finding: Optional[CagFinding],
) -> list[DroolsRecommendation]:
    """
    Pure-Python implementation of all three DRL rule files.
    Executed when KIE Server is unreachable (local dev / CI / offline).
    """
    recs: list[DroolsRecommendation] = []
    fired: set[str] = set()

    def add(r: DroolsRecommendation):
        if r.code not in fired:
            recs.append(r)
            fired.add(r.code)

    acs_type = acs_case.acsType

    # ── acs_triage.drl ──────────────────────────────────────────────────────

    # STEMI – Primary PCI
    if acs_type == "STEMI" and acs_case.primaryPciFacilityAvailableCloseBy:
        add(DroolsRecommendation(
            type="PRIMARY_PCI",
            code="ACS-STEMI-PRIMARY-PCI",
            message="Primary PCI pathway: activate cath lab, start dual antiplatelet therapy plus anticoagulation, and proceed to PCI-capable care.",
            rationale="STEMI with 24×7 primary PCI facility available close by.",
            urgency="Immediate",
            priority="CRITICAL",
        ))

    # STEMI – Pharmacoinvasive
    elif acs_type == "STEMI" and (
        not acs_case.primaryPciFacilityAvailableCloseBy
        or (acs_case.expectedFmcToBalloonMinutes is not None and acs_case.expectedFmcToBalloonMinutes > 120)
    ):
        add(DroolsRecommendation(
            type="FIBRINOLYSIS_PHARMACOINVASIVE_PCI",
            code="ACS-STEMI-FIBRINOLYSIS-PHARMACOINVASIVE",
            message="Give weight-based tenecteplase if eligible, facilitate transfer to a PCI-capable centre, and plan pharmacoinvasive PCI 3–24 hours later.",
            rationale="Primary PCI pathway is not available close by or expected FMC-to-balloon time exceeds approximately 120 minutes.",
            urgency="Immediate, then PCI in 3–24 hours",
            priority="CRITICAL",
        ))

    # NSTEMI / UA – High risk
    elif acs_type in ("NSTEMI", "UNSTABLE_ANGINA"):
        high_risk = (
            (acs_case.timiScore is not None and acs_case.timiScore >= 3)
            or (acs_case.graceScore is not None and acs_case.graceScore > 140)
            or acs_case.haemodynamicInstability
            or acs_case.electricalInstability
            or acs_case.recurrentIschaemia
            or acs_case.dynamicStOrTChanges
            or acs_case.largeTroponinRise
        )
        if high_risk:
            add(DroolsRecommendation(
                type="EARLY_INVASIVE_CAG",
                code="ACS-NSTE-HIGH-RISK-EARLY-INVASIVE",
                message="Early invasive strategy: diagnostic CAG within 24 hours; use ≤2 hours when truly unstable; choose PCI or CABG per the CAG anatomy tree.",
                rationale="High-risk NSTE-ACS: TIMI ≥3, GRACE >140, instability, recurrent ischaemia, dynamic ST/T, or major troponin rise.",
                urgency="≤24 hours; ≤2 hours if unstable",
                priority="URGENT",
            ))
        else:
            add(DroolsRecommendation(
                type="LOW_INTERMEDIATE_MEDICAL_OPTIMISATION",
                code="ACS-NSTE-LOW-INTERMEDIATE-MEDICAL",
                message="Optimise medication therapy, stabilise and treat pain, and use non-invasive stress imaging if needed.",
                rationale="NSTE-ACS without high-risk criteria.",
                urgency="During index stabilisation",
                priority="HIGH",
            ))
            add(DroolsRecommendation(
                type="DELAYED_ELECTIVE_CAG",
                code="ACS-NSTE-DELAYED-ELECTIVE-CAG",
                message="Delayed or elective diagnostic CAG later if symptoms persist or non-invasive testing is positive.",
                rationale="Low/intermediate-risk NSTE-ACS pathway reserves angiography for symptoms or positive testing.",
                urgency="Delayed/elective",
                priority="MEDIUM",
            ))

    # ── cag_decision.drl ────────────────────────────────────────────────────

    if cag_finding and cag_finding.diagnosticCagCompleted:
        lc = _map_lesion_category(cag_finding.lesionCategory)

        if lc == LesionCategory.ONE_TO_TWO_VESSELS.value:
            add(DroolsRecommendation(
                type="PCI",
                code="CAG-PCI-ONE-TWO-VESSEL",
                message="PCI: stent culprit lesion initially and achieve revascularisation within 6 weeks of MI, considering general condition, co-morbidities and finances.",
                rationale="CAG shows ≥70% disease in 1–2 epicardial vessels.",
                urgency="Index culprit PCI; complete revascularisation within 6 weeks",
                priority="HIGH",
            ))

        elif lc == LesionCategory.MULTIVESSEL.value:
            add(DroolsRecommendation(
                type="CABG",
                code="CAG-CABG-MULTIVESSEL",
                message="CABG preferred for multivessel disease.",
                rationale="CAG shows ≥70% multivessel disease.",
                urgency="Heart-team/surgical pathway",
                priority="HIGH",
            ))

        elif lc == LesionCategory.BORDERLINE_50_TO_69.value:
            ffr_done = cag_finding.ffrPerformed
            stress_done = cag_finding.stressImagingPerformed
            if not ffr_done and not stress_done:
                add(DroolsRecommendation(
                    type="PHYSIOLOGY_OR_STRESS_IMAGING",
                    code="CAG-BORDERLINE-PHYSIOLOGY-STRESS",
                    message="For a 50–69% borderline lesion, perform physiology testing with FFR ≤0.80 or stress imaging such as MPI.",
                    rationale="Borderline stenosis requires ischaemia confirmation before revascularisation.",
                    urgency="After a few weeks unless assessing non-culprit lesion",
                    priority="MEDIUM",
                ))
            elif ffr_done and cag_finding.ffrValue is not None:
                if cag_finding.ffrValue <= 0.80:
                    add(DroolsRecommendation(
                        type="PCI",
                        code="CAG-BORDERLINE-ISCHAEMIC-REVASCULARISE",
                        message="Treat as PCI or CABG according to coronary anatomy.",
                        rationale="Borderline lesion is ischaemic by FFR ≤0.80.",
                        urgency="Per anatomy",
                        priority="HIGH",
                    ))
                else:
                    add(DroolsRecommendation(
                        type="MEDICAL_THERAPY",
                        code="CAG-BORDERLINE-NON-ISCHAEMIC-MEDICAL",
                        message="Medical therapy only – FFR non-ischaemic.",
                        rationale="Borderline lesion is non-ischaemic by FFR >0.80.",
                        urgency="Ongoing",
                        priority="MEDIUM",
                    ))
            elif stress_done:
                if cag_finding.stressImagingPositiveForIschaemia:
                    add(DroolsRecommendation(
                        type="PCI",
                        code="CAG-BORDERLINE-ISCHAEMIC-REVASCULARISE",
                        message="Treat as PCI or CABG according to coronary anatomy.",
                        rationale="Borderline lesion is ischaemic by stress imaging/MPI.",
                        urgency="Per anatomy",
                        priority="HIGH",
                    ))
                else:
                    add(DroolsRecommendation(
                        type="MEDICAL_THERAPY",
                        code="CAG-BORDERLINE-NON-ISCHAEMIC-MEDICAL",
                        message="Medical therapy only – non-ischaemic on stress imaging.",
                        rationale="Borderline lesion is non-ischaemic by stress imaging/MPI.",
                        urgency="Ongoing",
                        priority="MEDIUM",
                    ))

        elif lc == LesionCategory.LESS_THAN_50.value:
            add(DroolsRecommendation(
                type="MEDICAL_THERAPY",
                code="CAG-LESS-THAN-50-MEDICAL",
                message="Straight to medical management.",
                rationale="Baseline lesion is <50%.",
                urgency="Ongoing",
                priority="MEDIUM",
            ))

    # ── post_mi_viability.drl ───────────────────────────────────────────────

    if acs_case.delayedPresentation and acs_case.lvDysfunction:
        if acs_case.viableMyocardium is None:
            add(DroolsRecommendation(
                type="VIABILITY_ASSESSMENT",
                code="POSTMI-LVD-VIABILITY-ASSESSMENT",
                message="Perform viability assessment using MRI, PET, or rest thallium scan.",
                rationale="Delayed presentation with LV dysfunction requires viability assessment before late revascularisation planning.",
                urgency="After stabilisation",
                priority="HIGH",
            ))
        elif acs_case.viableMyocardium:
            add(DroolsRecommendation(
                type="CAG_PCI_IF_VIABLE",
                code="POSTMI-LVD-VIABLE-CAG-PCI",
                message="Plan diagnostic CAG and PCI if needed.",
                rationale="Viable myocardium is present after delayed presentation with LV dysfunction.",
                urgency="Planned/elective after assessment",
                priority="HIGH",
            ))
        else:
            add(DroolsRecommendation(
                type="GDMT_ICD_CRT_CONSIDERATION",
                code="POSTMI-LVD-NONVIABLE-GDMT-ICD-CRT",
                message="Use GDMT and consider ICD/CRT at 1 month according to eligibility.",
                rationale="Non-viable myocardium after delayed presentation with LV dysfunction.",
                urgency="GDMT now; device consideration at 1 month",
                priority="HIGH",
            ))

    return recs


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

async def execute_acs_rules(
    request_payload: dict,
    request_id: str = "REQ-UNKNOWN",
) -> DroolsResponse:
    kie_raw = None
    """
    Main entry point called by the ACS endpoint.
    1. Tries KIE Server (real Drools).
    2. Falls back to Python engine if KIE unreachable.
    """
    t0 = time.monotonic()
    acs_data = request_payload.get("acsCase", {})
    cag_data = request_payload.get("cagFinding")
    case_id = acs_data.get("caseId", "CASE-UNKNOWN")

    acs_case = AcsCase(
        acsType=acs_data.get("acsType", "NSTEMI"),
        timiScore=acs_data.get("timiScore"),
        graceScore=acs_data.get("graceScore"),
        haemodynamicInstability=acs_data.get("haemodynamicInstability", False),
        electricalInstability=acs_data.get("electricalInstability", False),
        recurrentIschaemia=acs_data.get("recurrentIschaemia", False),
        dynamicStOrTChanges=acs_data.get("dynamicStOrTChanges", False),
        largeTroponinRise=acs_data.get("largeTroponinRise", False),
        primaryPciFacilityAvailableCloseBy=acs_data.get("primaryPciFacilityAvailableCloseBy", True),
        expectedFmcToBalloonMinutes=acs_data.get("expectedFmcToBalloonMinutes"),
        delayedPresentation=acs_data.get("delayedPresentation", False),
        lvDysfunction=acs_data.get("lvDysfunction", False),
        viableMyocardium=acs_data.get("viableMyocardium"),
    )

    cag_finding: Optional[CagFinding] = None
    if cag_data:
        cag_finding = CagFinding(
            diagnosticCagCompleted=cag_data.get("diagnosticCagCompleted", False),
            lesionCategory=cag_data.get("lesionCategory"),
            maxDiameterStenosisPercent=cag_data.get("maxDiameterStenosisPercent"),
            ffrPerformed=cag_data.get("ffrPerformed", False),
            ffrValue=cag_data.get("ffrValue"),
            stressImagingPerformed=cag_data.get("stressImagingPerformed", False),
            stressImagingPositiveForIschaemia=cag_data.get("stressImagingPositiveForIschaemia"),
        )

    # Try KIE Server first
    kie_payload = _build_kie_payload(acs_case, cag_finding, request_id, case_id)
    #kie_raw = await _call_kie_server(kie_payload)

    fallback_used = False
    recommendations: list[DroolsRecommendation] = []

    if kie_raw:
        recommendations = _parse_kie_response(kie_raw)
        logger.info(f"[Drools] KIE Server returned {len(recommendations)} recommendations")
    else:
        logger.info("[Drools] Using Python fallback engine")
        recommendations = _python_fallback_engine(acs_case, cag_finding)
        fallback_used = True

    elapsed_ms = (time.monotonic() - t0) * 1000

    return DroolsResponse(
        request_id=request_id,
        case_id=case_id,
        acs_type=acs_case.acsType,
        recommendations=recommendations,
        execution_time_ms=round(elapsed_ms, 2),
        rule_count_fired=len(recommendations),
        kie_server_used=not fallback_used,
        fallback_used=fallback_used,
    )
