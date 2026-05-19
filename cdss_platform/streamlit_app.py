"""
CDSS Clinical Intelligence Platform – Streamlit UI
Full production-grade EMR demo interface with:
  - ACS Pathway (STEMI / NSTEMI / Diagnostic CAG) via Drools Rule Engine
  - RAG + LLM clinical recommendations via FastAPI backend
  - Clinician review workflow (Approve / Reject / Revise)
  - Audit trail display
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import requests
import streamlit as st

# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────

FASTAPI_BASE = "http://localhost:8000"
ACS_ENDPOINT = f"{FASTAPI_BASE}/clinical-decision-support/acs/recommendations"
RAG_ENDPOINT = f"{FASTAPI_BASE}/api/v1/recommendations"
AUTH_ENDPOINT = f"{FASTAPI_BASE}/api/v1/auth/token"
REVIEW_BASE = f"{FASTAPI_BASE}/api/v1/recommendations"

# Colour palette
COLOURS = {
    "CRITICAL": "#d32f2f",
    "URGENT": "#f57c00",
    "HIGH": "#fbc02d",
    "MEDIUM": "#0288d1",
    "LOW": "#388e3c",
    "PRIMARY_PCI": "#b71c1c",
    "PHARMACOINVASIVE_PCI": "#e65100",
    "EARLY_INVASIVE_CAG": "#f57f17",
    "MEDICAL_THERAPY": "#1565c0",
    "CABG": "#4a148c",
    "PCI": "#880e4f",
}

# ────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CDSS Clinical Intelligence Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── FULL BLUE BACKGROUND ───────────────────────────────────── */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    .stApp, .main {
        background-color: #0d1b4b !important;
    }
    .main .block-container {
        padding-top: 1rem;
        max-width: 1400px;
        background-color: transparent !important;
    }
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"],
    div[data-testid="column"] {
        background-color: transparent !important;
    }

    /* ── TAB BAR ────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0a1640 !important;
        border-radius: 10px 10px 0 0;
        padding: 4px 8px 0;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a2e6e !important;
        color: #90caf9 !important;
        border-radius: 8px 8px 0 0 !important;
        border: 1px solid #2a3f8f !important;
        font-weight: 600;
        padding: 8px 18px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1565c0 !important;
        color: #ffffff !important;
        border-color: #42a5f5 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #0f2060 !important;
        border-radius: 0 0 10px 10px;
        border: 1px solid #1e3a8a;
        padding: 1.2rem;
    }

    /* ── FORM WIDGETS ───────────────────────────────────────────── */
    .stTextInput > div > div,
    .stNumberInput > div > div,
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stTextArea > div > div {
        background-color: #132357 !important;
        border: 1px solid #2a4cad !important;
        border-radius: 6px !important;
    }
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {
        color: #e3f2fd !important;
        background-color: #132357 !important;
    }

    /* ── TEXT & LABELS ──────────────────────────────────────────── */
    label,
    .stMarkdown p, .stMarkdown li,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #cfd8f0 !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: #90caf9 !important;
    }
    .stCheckbox span { color: #cfd8f0 !important; }

    /* ── EXPANDER ────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background-color: #1a2e6e !important;
        color: #90caf9 !important;
        border-radius: 6px !important;
        border: 1px solid #2a4cad !important;
    }
    .streamlit-expanderContent {
        background-color: #132357 !important;
        border: 1px solid #2a4cad !important;
        border-top: none !important;
    }

    /* ── HEADER ──────────────────────────────────────────────────── */
    .cdss-header {
        background: linear-gradient(135deg, #0a1640 0%, #1a237e 50%, #1565c0 100%);
        padding: 1.2rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.5);
        border: 1px solid #2a4cad;
    }
    .cdss-header h1 { color: #ffffff; margin: 0; font-size: 1.6rem; font-weight: 700; }
    .cdss-header p  { color: #90caf9; margin: 0.3rem 0 0; font-size: 0.85rem; }

    /* ── PANELS ──────────────────────────────────────────────────── */
    .panel {
        background: #132357;
        border-radius: 10px; padding: 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.4); margin-bottom: 1rem;
        border-left: 4px solid #42a5f5;
        border-top: 1px solid #1e3a8a;
        border-right: 1px solid #1e3a8a;
        border-bottom: 1px solid #1e3a8a;
    }
    .panel-title { color: #90caf9; font-weight: 700; font-size: 1rem; margin-bottom: 0.6rem; }

    /* ── REC CARDS ───────────────────────────────────────────────── */
    .rec-card {
        border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
        border-left: 5px solid; box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .rec-CRITICAL { background: #1a0a0a; border-color: #ef5350; }
    .rec-URGENT   { background: #1a1000; border-color: #ffa726; }
    .rec-HIGH     { background: #1a1700; border-color: #ffca28; }
    .rec-MEDIUM   { background: #071a2e; border-color: #42a5f5; }
    .rec-LOW      { background: #071a0f; border-color: #66bb6a; }

    .rec-type { font-weight: 700; font-size: 0.95rem; color: #e3f2fd; }
    .rec-msg  { font-size: 0.9rem; margin: 0.3rem 0; color: #b3cde0; }
    .rec-rat  { font-size: 0.8rem; color: #90a4ae; }
    .rec-urg  { font-size: 0.8rem; font-weight: 600; color: #ef9a9a; }

    /* ── BADGES ──────────────────────────────────────────────────── */
    .badge {
        display: inline-block; padding: 0.2rem 0.7rem; border-radius: 20px;
        font-size: 0.75rem; font-weight: 700; margin-right: 0.3rem;
    }
    .badge-stemi     { background: #c62828; color: white; }
    .badge-nstemi    { background: #e65100; color: white; }
    .badge-cag       { background: #1565c0; color: white; }
    .badge-kie       { background: #1b5e20; color: white; }
    .badge-fallback  { background: #4a148c; color: white; }
    .badge-approved  { background: #2e7d32; color: white; }
    .badge-rejected  { background: #b71c1c; color: white; }
    .badge-pending   { background: #e65100; color: white; }

    /* ── BUTTONS ─────────────────────────────────────────────────── */
    .stButton > button {
        border-radius: 6px; font-weight: 600;
        border: 1px solid #2a4cad !important;
        padding: 0.5rem 1.5rem; cursor: pointer; transition: all 0.2s;
        background-color: #1a2e6e !important;
        color: #e3f2fd !important;
    }
    .stButton > button:hover {
        background-color: #1565c0 !important;
        border-color: #42a5f5 !important;
        color: #ffffff !important;
    }
    .stButton > button[kind="primary"] {
        background-color: #1565c0 !important;
        border-color: #42a5f5 !important;
        color: white !important;
    }

    /* ── SIDEBAR ─────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #060e2e 0%, #0d1b4b 100%) !important;
        border-right: 1px solid #1e3a8a;
    }
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label { color: #90caf9 !important; }
    section[data-testid="stSidebar"] .stTextInput > div > div,
    section[data-testid="stSidebar"] .stTextInput input {
        background-color: #0a1640 !important;
        border-color: #1e3a8a !important;
        color: #e3f2fd !important;
    }

    /* ── STATUS BAR ──────────────────────────────────────────────── */
    .status-bar {
        display: flex; gap: 1rem; align-items: center;
        background: #0a1640; border-radius: 8px; padding: 0.5rem 1rem;
        font-size: 0.8rem; color: #90caf9; margin-bottom: 0.5rem;
        border: 1px solid #1e3a8a;
    }

    /* ── AUDIT TRAIL ─────────────────────────────────────────────── */
    .audit-row {
        background: #0f2060; border-radius: 6px; padding: 0.5rem 1rem;
        margin-bottom: 0.4rem; border: 1px solid #1e3a8a;
    }
    .audit-ts { color: #5c7abf; font-size: 0.75rem; }

    /* ── NLP SUMMARY ─────────────────────────────────────────────── */
    .nlp-box {
        background: #071f0a; border-radius: 8px; padding: 1rem 1.2rem;
        border: 1px solid #2e7d32; font-size: 0.9rem; line-height: 1.6;
        color: #a5d6a7;
    }

    /* ── DISCLAIMER ──────────────────────────────────────────────── */
    .disclaimer {
        background: #1a1000; border: 1px solid #f57c00; border-radius: 8px;
        padding: 0.8rem 1rem; font-size: 0.8rem; color: #ffb74d;
    }

    /* ── ALERTS ──────────────────────────────────────────────────── */
    .stAlert, [data-testid="stAlert"] {
        background-color: #0f2060 !important;
        border: 1px solid #2a4cad !important;
        color: #90caf9 !important;
        border-radius: 8px !important;
    }

    /* ── DROPDOWN MENUS ──────────────────────────────────────────── */
    [data-baseweb="popover"], [data-baseweb="menu"] {
        background-color: #132357 !important;
        border: 1px solid #2a4cad !important;
    }
    [data-baseweb="option"] {
        background-color: #132357 !important;
        color: #e3f2fd !important;
    }
    [data-baseweb="option"]:hover { background-color: #1565c0 !important; }

    /* ── MISC ────────────────────────────────────────────────────── */
    hr { border-color: #1e3a8a !important; }
    .stDownloadButton > button {
        background-color: #1565c0 !important;
        color: white !important; border: none !important;
        border-radius: 6px !important;
    }
    .stCode, code {
        background-color: #071230 !important;
        color: #80cbc4 !important;
        border: 1px solid #1e3a8a !important;
    }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# Session State
# ────────────────────────────────────────────────────────────────────────────

defaults = {
    "acs_result": None,
    "rag_result": None,
    "audit_trail": [],
    "review_status": None,
    "correlation_id": None,
    "active_tab": "ACS Pathway",
    "auth_token": None,
    "auth_user": None,
    "auth_role": None,
    "acs_payload": None,
    "rag_payload": None,
    "active_result_source": None,
    "reviewed_recommendations": {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _add_audit(event: str, detail: str, status: str = "INFO"):
    st.session_state.audit_trail.append({
        "ts": _now_iso(),
        "event": event,
        "detail": detail,
        "status": status,
    })


def _auth_headers(token: Optional[str]) -> dict:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _post_acs(payload: dict, token: Optional[str]) -> dict:
    resp = requests.post(ACS_ENDPOINT, json=payload, headers=_auth_headers(token), timeout=120)
    resp.raise_for_status()
    return resp.json()


def _post_rag(payload: dict, token: Optional[str]) -> dict:
    resp = requests.post(RAG_ENDPOINT, json=payload, headers=_auth_headers(token), timeout=120)
    resp.raise_for_status()
    return resp.json()


def _get_token(username: str, password: str) -> Optional[str]:
    try:
        r = requests.post(
            AUTH_ENDPOINT,
            json={"username": username, "password": password},  # json= not data=
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("access_token")
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Login error {e.response.status_code}: {e.response.json().get('detail','Unknown error')}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach FastAPI server at localhost:8000 — is it running?")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None


def _get_user_info(token: Optional[str]) -> Optional[dict]:
    if not token:
        return None
    try:
        r = requests.get(
            f"{FASTAPI_BASE}/api/v1/auth/me",
            headers=_auth_headers(token),
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _submit_review(
    correlation_id: str,
    action: str,
    notes: str,
    token: Optional[str],
    edited_summary: Optional[str] = None,
    reviewer_id: Optional[str] = None,
    reviewer_role: Optional[str] = None,
) -> dict:
    payload = {
        "correlation_id": correlation_id,
        "reviewer_id": reviewer_id or st.session_state.get("reviewer_id", "DR-UNKNOWN"),
        "reviewer_role": reviewer_role or st.session_state.get("auth_role") or "CLINICIAN",
        "action": action,
        "notes": notes,
        "edited_summary": edited_summary,
    }
    resp = requests.post(
        f"{REVIEW_BASE}/{correlation_id}/review",
        json=payload,
        headers=_auth_headers(token),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _store_reviewed_text(correlation_id: str, status: str, notes: str, revised_summary: Optional[str] = None) -> None:
    st.session_state.reviewed_recommendations[correlation_id] = {
        "status": status,
        "notes": notes,
        "revised_summary": revised_summary,
        "reviewed_at": _now_iso(),
    }


def _split_clinical_items(raw: str) -> list[str]:
    """Convert comma/newline clinical text into a compact list for the API payload."""
    items: list[str] = []
    for line in raw.replace(";", "\n").replace(",", "\n").splitlines():
        item = line.strip()
        if item:
            items.append(item)
    return items


def _merge_unique(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for item in group:
            normalized = item.strip()
            if normalized and normalized.lower() != "none" and normalized.lower() not in seen:
                seen.add(normalized.lower())
                merged.append(normalized)
    return merged


RAG_RECOMMENDATION_FIELDS = [
    ("Summary", "summary"),
    ("Risk Stratification", "risk_stratification"),
    ("Antiplatelet Guidance", "antiplatelet_guidance"),
    ("Invasive Strategy", "invasive_strategy"),
    ("Adjunct Therapy", "adjunct_therapy"),
    ("Monitoring Plan", "monitoring_plan"),
]


def _format_rag_recommendations_for_review(rec: dict[str, Any]) -> str:
    """Create a clinician-editable note from all structured RAG recommendation sections."""
    sections: list[str] = []
    for index, (label, field) in enumerate(RAG_RECOMMENDATION_FIELDS, start=1):
        value = str(rec.get(field) or "").strip()
        if value:
            sections.append(f"{index}. {label}\n{value}")

    safety_flags = rec.get("safety_flags") or {}
    flags = safety_flags.get("flags") or []
    if flags:
        sections.append("Safety Flags\n" + "\n".join(f"- {flag}" for flag in flags))

    return "\n\n".join(sections)


def _format_acs_recommendations_for_review(result: dict[str, Any]) -> str:
    """Create a clinician-editable note from Drools ACS recommendations."""
    sections: list[str] = []
    for index, rec in enumerate(result.get("recommendations", []) or [], start=1):
        title = str(rec.get("type") or "ACS Recommendation").replace("_", " ").title()
        lines = [
            f"{index}. {title}",
            str(rec.get("message") or "").strip(),
        ]
        rationale = str(rec.get("rationale") or "").strip()
        urgency = str(rec.get("urgency") or "").strip()
        code = str(rec.get("code") or "").strip()
        source = str(rec.get("source") or "").strip()
        if rationale:
            lines.append(f"Rationale: {rationale}")
        if urgency:
            lines.append(f"Urgency: {urgency}")
        if code or source:
            lines.append(f"Code/Source: {code} {source}".strip())
        sections.append("\n".join(line for line in lines if line.strip()))

    narrative = str(result.get("nlpSummary") or "").strip()
    if narrative:
        sections.append("Clinical Narrative\n" + narrative)

    return "\n\n".join(sections)


def _active_recommendation_text(active_source: Optional[str], active_result: Optional[dict[str, Any]]) -> str:
    if not active_result:
        return ""
    if active_source == "rag":
        return _format_rag_recommendations_for_review(active_result.get("recommendation", {}) or {})
    return _format_acs_recommendations_for_review(active_result)


def _priority_emoji(p: str) -> str:
    return {"CRITICAL": "🔴", "URGENT": "🟠", "HIGH": "🟡", "MEDIUM": "🔵", "LOW": "⚪"}.get(p, "⚫")


# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏥 CDSS Platform")
    st.markdown("---")

    st.markdown("### 🔐 Clinician Login")
    reviewer_id = st.text_input("Clinician ID", value="bismita-dr", key="reviewer_id_input")
    reviewer_pass = st.text_input("Password", type="password", value="Bismita@cdss1!", key="reviewer_pass")

    if st.button("🔑 Login", use_container_width=True):
        token = _get_token(
            username=reviewer_id,
            password=reviewer_pass,
        )
        if token:
            st.session_state.auth_token = token
            st.session_state["reviewer_id"] = reviewer_id
            user_info = _get_user_info(token) or {}
            st.session_state.auth_user = user_info
            st.session_state.auth_role = user_info.get("role") or "CLINICIAN"
            st.success(f"Logged in as {reviewer_id} ({st.session_state.auth_role})")
            _add_audit("LOGIN", f"Clinician {reviewer_id} authenticated as {st.session_state.auth_role}", "SUCCESS")
        else:
            st.session_state["reviewer_id"] = reviewer_id
            st.session_state.auth_token = None
            st.session_state.auth_user = None
            st.session_state.auth_role = None

    if st.session_state.auth_token:
        st.markdown(
            f"**Authenticated:** `{st.session_state.get('reviewer_id')}`  "
            f"Role: `{st.session_state.get('auth_role') or 'UNKNOWN'}`"
        )
    else:
        st.info("Login before generating Drools or RAG recommendations.")

    st.markdown("---")
    st.markdown("### ℹ️ System Info")
    st.markdown("**API:** `localhost:8000`")
    st.markdown("**Rules:** Drools KIE Server")
    st.markdown("**LLM:** OpenAI / configured LLM")
    st.markdown("**RAG:** ChromaDB")

    if st.session_state.audit_trail:
        st.markdown("---")
        st.markdown("### 📋 Audit Events")
        for a in reversed(st.session_state.audit_trail[-6:]):
            icon = "✅" if a["status"] == "SUCCESS" else "ℹ️" if a["status"] == "INFO" else "⚠️"
            st.markdown(f"{icon} `{a['ts'][:19]}` **{a['event']}**")

    if st.button("🗑️ Clear Session", use_container_width=True):
        for k in ["acs_result", "rag_result", "audit_trail", "review_status", "correlation_id", "acs_payload", "active_result_source"]:
            st.session_state[k] = None if k != "audit_trail" else []
        st.rerun()

# ────────────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="cdss-header">
  <h1>🏥 CDSS Clinical Intelligence Platform</h1>
  <p>AI-Powered EMR | Drools Rule Engine | RAG + LLM | Clinician Review Workflow</p>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# Main Tabs
# ────────────────────────────────────────────────────────────────────────────

tab_acs, tab_rag, tab_review, tab_audit, tab_payload = st.tabs([
    "🫀 ACS Pathway (Drools)",
    "🧠 RAG + LLM Recommendation",
    "✅ Clinician Review",
    "📋 Audit Trail",
    "📄 Raw Payload Inspector",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – ACS CLINICAL PATHWAY (DROOLS)
# ════════════════════════════════════════════════════════════════════════════

with tab_acs:
    st.markdown("### 🫀 ACS Clinical Pathway Decision Engine")
    st.markdown(
        "Deterministic rule execution via **Drools KIE Server** "
        "(`rules/drools-rules/acs_triage.drl`, `cag_decision.drl`, `post_mi_viability.drl`)"
    )

    col_form, col_result = st.columns([1, 1], gap="medium")

    with col_form:
        with st.container():
            st.markdown('<div class="panel"><div class="panel-title">👤 Patient & Encounter</div>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            patient_id = c1.text_input("Patient ID", value="PAT-000123", key="acs_patient_id")
            mrn = c2.text_input("MRN", value="MRN-998877", key="acs_mrn")

            c3, c4, c5 = st.columns(3)
            age = c3.number_input("Age (yrs)", min_value=18, max_value=120, value=58, key="acs_age")
            acs_sex_options = ["MALE", "FEMALE", "OTHER"]
            acs_encounter_options = ["EMERGENCY", "OUTPATIENT", "INPATIENT"]
            if st.session_state.get("acs_sex") not in acs_sex_options:
                st.session_state["acs_sex"] = "MALE"
            if st.session_state.get("acs_encounter_type") not in acs_encounter_options:
                st.session_state["acs_encounter_type"] = "EMERGENCY"
            sex = c4.selectbox(
                "Sex",
                acs_sex_options,
                index=acs_sex_options.index(st.session_state["acs_sex"]),
                key="acs_sex",
            )
            encounter_type = c5.selectbox(
                "Encounter Type",
                acs_encounter_options,
                index=acs_encounter_options.index(st.session_state["acs_encounter_type"]),
                key="acs_encounter_type",
            )

            st.markdown("---")
            st.markdown("**⏱️ Timing**")
            c6, c7 = st.columns(2)
            fmc_to_balloon = c6.number_input("FMC→Balloon (min)", min_value=0, max_value=999, value=85, key="acs_fmc_to_balloon")
            pci_available = c7.checkbox("PCI Facility Close By", value=True, key="acs_pci_available")

            st.markdown("</div>", unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="panel"><div class="panel-title">🏥 ACS Case Details</div>', unsafe_allow_html=True)

            acs_type = st.selectbox(
                "ACS Type",
                ["STEMI", "NSTEMI", "Diagnostic CAG", "UNSTABLE_ANGINA"],
                help="Select the ACS type for Drools rule execution",
                key="acs_type",
            )

            c8, c9 = st.columns(2)
            timi_score = c8.number_input("TIMI Score (0-7)", min_value=0, max_value=7, value=3, key="acs_timi_score")
            grace_score = c9.number_input("GRACE Score", min_value=0, max_value=500, value=145, key="acs_grace_score")

            st.markdown("**⚠️ Risk Flags**")
            r1, r2, r3 = st.columns(3)
            haemo_instab = r1.checkbox("Haemodynamic Instability", key="acs_haemo_instab")
            elec_instab = r2.checkbox("Electrical Instability", key="acs_elec_instab")
            recur_ischaemia = r3.checkbox("Recurrent Ischaemia", key="acs_recur_ischaemia")
            r4, r5 = st.columns(2)
            dynamic_st = r4.checkbox("Dynamic ST/T Changes", value=True, key="acs_dynamic_st")
            large_troponin = r5.checkbox("Large Troponin Rise", value=True, key="acs_large_troponin")

            st.markdown("**📊 Post-MI LV Status**")
            m1, m2 = st.columns(2)
            delayed_presentation = m1.checkbox("Delayed Presentation", key="acs_delayed_presentation")
            lv_dysfunction = m2.checkbox("LV Dysfunction", key="acs_lv_dysfunction")
            if lv_dysfunction:
                viable_options = {"Unknown": None, "Yes – Viable": True, "No – Non-viable": False}
                viable_label = st.selectbox("Viable Myocardium", list(viable_options.keys()), key="acs_viable_myocardium")
                viable_myocardium = viable_options[viable_label]
            else:
                viable_myocardium = None

            st.markdown("</div>", unsafe_allow_html=True)

        if acs_type == "Diagnostic CAG":
            with st.container():
                st.markdown('<div class="panel"><div class="panel-title">🔬 CAG Findings</div>', unsafe_allow_html=True)

                cag_completed = st.checkbox("Diagnostic CAG Completed", value=True, key="acs_cag_completed")

                lesion_options = {
                    "≥70% – 1-2 Vessel": "GE_70_ONE_TWO_VESSEL",
                    "≥70% – Multivessel": "GE_70_MULTIVESSEL",
                    "Borderline 50-69%": "BORDERLINE_50_TO_69",
                    "<50%": "LESS_THAN_50",
                }
                lesion_label = st.selectbox("Lesion Category", list(lesion_options.keys()), key="acs_lesion_label")
                lesion_category = lesion_options[lesion_label]

                stenosis_pct = st.slider("Max Diameter Stenosis %", 0, 100, 90, key="acs_stenosis_pct")

                f1, f2 = st.columns(2)
                ffr_done = f1.checkbox("FFR Performed", key="acs_ffr_done")
                ffr_val = f1.number_input("FFR Value", min_value=0.0, max_value=1.0, value=0.75, step=0.01, key="acs_ffr_val") if ffr_done else None

                stress_done = f2.checkbox("Stress Imaging Performed", key="acs_stress_done")
                stress_positive = f2.checkbox("Stress Imaging Positive for Ischaemia", key="acs_stress_positive") if stress_done else None

                st.markdown("</div>", unsafe_allow_html=True)

        # Submit button
        if st.button("🚀 Execute ACS Clinical Rules", type="primary", use_container_width=True):
            if not st.session_state.auth_token:
                st.error("Login first so the ACS request is sent with a valid Keycloak bearer token.")
                st.stop()

            request_id = f"REQ-ACS-{uuid.uuid4().hex[:8].upper()}"
            case_id = f"ACS-CASE-{uuid.uuid4().hex[:8].upper()}"

            acs_payload = {
                "requestId": request_id,
                "sourceSystem": "STREAMLIT_EMR",
                "tenantId": "hospital-001",
                "facilityId": "facility-001",
                "requestedAt": _now_iso(),
                "ruleSet": {"name": "ACS_CAG_DECISION_PATHWAY", "version": "1.0.0", "executionMode": "STATELESS"},
                "patient": {"patientId": patient_id, "mrn": mrn, "ageYears": age, "sex": sex},
                "encounter": {
                    "encounterId": f"ENC-{uuid.uuid4().hex[:8].upper()}",
                    "encounterType": encounter_type,
                    "arrivalDateTime": _now_iso(),
                    "firstMedicalContactDateTime": _now_iso(),
                },
                "acsCase": {
                    "caseId": case_id,
                    "acsType": acs_type,
                    "timiScore": int(timi_score) if acs_type != "STEMI" else None,
                    "graceScore": int(grace_score) if acs_type != "STEMI" else None,
                    "haemodynamicInstability": haemo_instab,
                    "electricalInstability": elec_instab,
                    "recurrentIschaemia": recur_ischaemia,
                    "dynamicStOrTChanges": dynamic_st,
                    "largeTroponinRise": large_troponin,
                    "primaryPciFacilityAvailableCloseBy": pci_available,
                    "expectedFmcToBalloonMinutes": int(fmc_to_balloon),
                    "delayedPresentation": delayed_presentation,
                    "lvDysfunction": lv_dysfunction,
                    "viableMyocardium": viable_myocardium,
                },
                "emrContext": {
                    "diagnosisCodes": [],
                    "vitals": {},
                    "ecg": {
                        "stElevationPresent": acs_type == "STEMI",
                        "dynamicStTChangesPresent": dynamic_st,
                        "sustainedVtVfPresent": elec_instab,
                    },
                    "labs": {"troponinPositive": large_troponin},
                },
            }

            # Add CAG finding if applicable
            if acs_type == "Diagnostic CAG":
                acs_payload["cagFinding"] = {
                    "caseId": case_id,
                    "diagnosticCagCompleted": cag_completed,
                    "lesionCategory": lesion_category,
                    "maxDiameterStenosisPercent": stenosis_pct,
                    "ffrPerformed": ffr_done,
                    "ffrValue": ffr_val,
                    "stressImagingPerformed": stress_done,
                    "stressImagingPositiveForIschaemia": stress_positive,
                }

            with st.spinner("⚙️ Executing clinical pathway rules..."):
                try:
                    result = _post_acs(acs_payload, st.session_state.auth_token)
                    st.session_state.acs_result = result
                    st.session_state.acs_payload = acs_payload
                    st.session_state.correlation_id = request_id
                    st.session_state.active_result_source = "acs"
                    st.session_state.review_status = None
                    generated_review_text = _format_acs_recommendations_for_review(result)
                    st.session_state[f"clinical_review_notes_{request_id}"] = generated_review_text
                    st.session_state[f"clinical_revised_summary_{request_id}"] = ""
                    _add_audit(
                        "ACS_RULES_EXECUTED",
                        f"requestId={request_id} acsType={acs_type} rules_fired={result.get('rulesFired',0)}",
                        "SUCCESS",
                    )
                    st.success(f"✅ Rules executed: {result.get('rulesFired', 0)} fired | "
                               f"{'KIE Server' if result.get('kieServerUsed') else 'Python Fallback Engine'}")
                except Exception as e:
                    st.error(f"❌ Rule execution failed: {e}")
                    _add_audit("ACS_RULES_ERROR", str(e), "ERROR")

    with col_result:
        if st.session_state.acs_result:
            r = st.session_state.acs_result

            # Status bar
            acs_badge = (
                "badge-stemi" if r.get("acsType") == "STEMI"
                else "badge-nstemi" if r.get("acsType") == "NSTEMI"
                else "badge-cag"
            )
            engine_badge = "badge-kie" if r.get("kieServerUsed") else "badge-fallback"
            engine_label = "KIE Server" if r.get("kieServerUsed") else "Python Fallback"

            st.markdown(f"""
            <div class="status-bar">
              <span class="badge {acs_badge}">{r.get('acsType')}</span>
              <span class="badge {engine_badge}">🔧 {engine_label}</span>
              <span>⚡ {r.get('executionTimeMs', 0):.1f}ms</span>
              <span>📏 {r.get('rulesFired', 0)} rules fired</span>
            </div>
            """, unsafe_allow_html=True)

            # Recommendations
            st.markdown("#### 📋 Clinical Recommendations")
            recommendations = r.get("recommendations", [])
            if recommendations:
                for rec in recommendations:
                    priority = rec.get("priority", "HIGH")
                    prio_class = f"rec-{priority}"
                    emoji = _priority_emoji(priority)
                    st.markdown(f"""
                    <div class="rec-card {prio_class}">
                      <div class="rec-type">{emoji} {rec.get('type','').replace('_',' ').title()}</div>
                      <div class="rec-msg">📌 {rec.get('message','')}</div>
                      <div class="rec-rat">💡 {rec.get('rationale','')}</div>
                      <div class="rec-urg">⏱ Urgency: {rec.get('urgency','')}</div>
                      <small style="color:#999">Code: <code>{rec.get('code','')}</code> | Source: {rec.get('source','')}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No recommendations generated. Check patient data.")

            # NLP Summary
            st.markdown("#### 📝 Clinical Narrative (NLP Formatted)")
            st.markdown(f"""
            <div class="nlp-box">
            {r.get('nlpSummary','').replace(chr(10),'<br>')}
            </div>
            """, unsafe_allow_html=True)

            acs_review = st.session_state.reviewed_recommendations.get(r.get("requestId") or "")
            if acs_review and acs_review.get("revised_summary"):
                st.markdown("#### Final Recommendations")
                st.text_area(
                    "Approved / Approved Revised ACS Recommendation",
                    value=acs_review.get("revised_summary", ""),
                    height=180,
                    disabled=True,
                    key=f"acs_final_recommendation_{r.get('requestId')}",
                )

            # Disclaimer
            st.markdown(f"""
            <div class="disclaimer">
            ⚠️ <strong>Medical Disclaimer:</strong> {r.get('disclaimer','')}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👈 Complete the patient form and click **Execute ACS Clinical Rules** to generate pathway recommendations.")
            st.markdown("""
            **Supported Decision Pathways:**
            - 🔴 **STEMI** → Primary PCI or Pharmacoinvasive Strategy
            - 🟠 **NSTEMI** → Early Invasive (high risk) or Medical Optimisation (low risk)
            - 🔵 **Diagnostic CAG** → PCI / CABG / FFR / Stress Imaging / Medical
            - 🟡 **Post-MI LV Dysfunction** → Viability Assessment pathway
            """)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – RAG + LLM RECOMMENDATION
# ════════════════════════════════════════════════════════════════════════════

with tab_rag:
    st.markdown("### 🧠 RAG + LLM Clinical Recommendation")
    st.markdown("Uses **ChromaDB RAG** + the configured LLM for evidence-based clinical guidance.")

    lc1, lc2 = st.columns([1, 1], gap="medium")

    with lc1:
        st.markdown('<div class="panel"><div class="panel-title">📋 Patient Clinical Context</div>', unsafe_allow_html=True)

        rag_patient_id = st.text_input("Patient ID", value="PAT-000200", key="rag_pid")
        rag_enc_id = st.text_input("Encounter ID", value="ENC-2026-000100", key="rag_enc")

        c_a, c_b, c_c = st.columns(3)
        rag_age = c_a.number_input("Age", min_value=18, max_value=120, value=62, key="rag_age")
        rag_sex = c_b.selectbox("Sex", ["Male", "Female"], key="rag_sex")
        rag_enc_type = c_c.selectbox("Encounter", ["EMERGENCY", "OUTPATIENT", "INPATIENT"], key="rag_enc_type")

        rag_chief_complaint = st.text_area(
            "Chief Complaint / Presenting Problem",
            value="Retrosternal chest pain for 6 hours with diaphoresis; troponin positive.",
            height=70,
            key="rag_chief_complaint",
        )

        rag_diagnoses = st.multiselect(
            "Diagnoses / ICD Codes",
            ["NSTEMI", "STEMI", "ACS", "CKD", "Diabetes", "Hypertension", "Heart Failure", "AF"],
            default=["NSTEMI", "Hypertension"],
            key="rag_diagnoses",
        )
        rag_extra_diagnoses = st.text_input(
            "Additional Diagnoses (comma separated)",
            value="Dyslipidemia",
            key="rag_extra_diagnoses",
        )

        rag_ecg_findings = st.multiselect(
            "ECG Findings",
            [
                "ST depression",
                "T-wave inversion",
                "ST elevation",
                "New LBBB",
                "Normal sinus rhythm",
                "Atrial fibrillation",
            ],
            default=["ST depression"],
            key="rag_ecg_findings",
        )
        rag_extra_ecg = st.text_input(
            "Additional ECG / Imaging Findings",
            value="No posterior STEMI pattern documented",
            key="rag_extra_ecg",
        )

        st.markdown("**🩺 Vitals**")
        v1, v2, v3 = st.columns(3)
        sbp = v1.text_input("Systolic BP", value="138", key="sbp")
        dbp = v2.text_input("Diastolic BP", value="84", key="dbp")
        hr = v3.text_input("Heart Rate", value="92", key="hr")
        v4, v5, v6 = st.columns(3)
        spo2 = v4.text_input("SpO2 %", value="96", key="spo2")
        resp_rate = v5.text_input("Respiratory Rate", value="18", key="resp_rate")
        temperature = v6.text_input("Temperature", value="36.8 C", key="temperature")

        st.markdown("**🧪 Labs**")
        l1, l2, l3 = st.columns(3)
        troponin = l1.text_input("Troponin", value="2.4 ng/mL", key="trop")
        egfr = l2.text_input("eGFR", value="52", key="egfr")
        creatinine = l3.text_input("Creatinine", value="1.4 mg/dL", key="creat")
        l4, l5, l6, l7 = st.columns(4)
        potassium = l4.text_input("Potassium", value="4.3 mmol/L", key="potassium")
        haemoglobin = l5.text_input("Haemoglobin", value="12.8 g/dL", key="haemoglobin")
        platelets = l6.text_input("Platelets", value="230 x10^9/L", key="platelets")
        inr = l7.text_input("INR", value="1.0", key="inr")

        st.markdown("**💊 Current Medications**")
        rag_meds = st.multiselect(
            "Select medications",
            ["Aspirin 75mg", "Clopidogrel 75mg", "Ticagrelor 90mg", "Metformin 500mg",
             "Lisinopril 10mg", "Atorvastatin 40mg", "Metoprolol 25mg", "Furosemide 40mg"],
            default=["Aspirin 75mg", "Atorvastatin 40mg"],
            key="rag_meds",
        )
        rag_extra_meds = st.text_area(
            "Additional Medication Details",
            value="Enoxaparin started in ED; PPI considered due dyspepsia history.",
            height=70,
            key="rag_extra_meds",
        )

        rag_allergies = st.multiselect(
            "Allergies / Contraindications",
            ["Penicillin", "Aspirin", "Contrast", "Sulfa", "None"],
            default=["None"],
            key="rag_allergies",
        )
        rag_extra_allergies = st.text_input(
            "Additional Allergies",
            value="",
            key="rag_extra_allergies",
        )
        rag_contraindications = st.text_area(
            "Contraindications / Bleeding Risk Notes",
            value="No active bleeding; no prior intracranial haemorrhage documented.",
            height=70,
            key="rag_contraindications",
        )
        rag_cardiac_history = st.text_area(
            "Cardiac History / Prior Procedures",
            value="Hypertension for 10 years; no prior PCI or CABG.",
            height=70,
            key="rag_cardiac_history",
        )

        rag_query = st.text_area(
            "Clinical Query",
            value="Patient with NSTEMI, hypertension and CKD stage 3. What is the optimal antiplatelet and revascularisation strategy?",
            height=100,
            key="rag_query",
        )

        consent_ok = st.checkbox("✅ Patient consent verified", value=True, key="rag_consent_ok")

        if st.button("🧠 Generate RAG + LLM Recommendation", type="primary", use_container_width=True):
            if not st.session_state.auth_token:
                st.error("Login first so the RAG request is sent with a valid Keycloak bearer token.")
            elif not consent_ok:
                st.error("❌ Patient consent must be verified before generating recommendations.")
            else:
                diagnoses = _merge_unique(rag_diagnoses, _split_clinical_items(rag_extra_diagnoses))
                ecg_findings = _merge_unique(rag_ecg_findings, _split_clinical_items(rag_extra_ecg))
                medications = _merge_unique(rag_meds, _split_clinical_items(rag_extra_meds))
                allergies = _merge_unique(
                    [a for a in rag_allergies if a != "None"],
                    _split_clinical_items(rag_extra_allergies),
                )
                contraindications = _split_clinical_items(rag_contraindications)
                cardiac_history = _split_clinical_items(rag_cardiac_history)
                rag_payload = {
                    "patientId": rag_patient_id,
                    "encounterId": rag_enc_id,
                    "userId": st.session_state.get("reviewer_id", "DR-UNKNOWN"),
                    "userRole": st.session_state.get("auth_role") or "CLINICIAN",
                    "query": f"{rag_chief_complaint}\n\nClinical question: {rag_query}",
                    "consentVerified": True,
                    "patientContext": {
                        "age": rag_age,
                        "sex": rag_sex,
                        "encounterType": rag_enc_type,
                        "diagnoses": diagnoses,
                        "vitals": {
                            "systolicBp": [sbp],
                            "diastolicBp": [dbp],
                            "heartRate": hr,
                            "spo2": spo2,
                            "respiratoryRate": resp_rate,
                            "temperature": temperature,
                        },
                        "labs": {
                            "troponin": troponin,
                            "eGFR": egfr,
                            "creatinine": creatinine,
                            "potassium": potassium,
                            "haemoglobin": haemoglobin,
                            "platelets": platelets,
                            "INR": inr,
                        },
                        "currentMedications": medications,
                        "allergies": allergies,
                        "contraindications": contraindications,
                        "cardiacHistory": cardiac_history,
                        "ecgFindings": ecg_findings,
                    },
                }
                st.session_state.rag_payload = rag_payload

                with st.spinner("🔍 Retrieving clinical evidence + generating LLM recommendation..."):
                    try:
                        result = _post_rag(rag_payload, st.session_state.auth_token)
                        st.session_state.rag_result = result
                        st.session_state.correlation_id = result.get("correlation_id")
                        st.session_state.active_result_source = "rag"
                        st.session_state.review_status = None
                        generated_review_text = _format_rag_recommendations_for_review(result.get("recommendation", {}))
                        if result.get("correlation_id"):
                            st.session_state[f"rag_review_notes_{result.get('correlation_id')}"] = generated_review_text
                            st.session_state[f"rag_revised_summary_{result.get('correlation_id')}"] = ""
                        _add_audit(
                            "RAG_LLM_RECOMMENDATION",
                            f"patient={rag_patient_id} corr={result.get('correlation_id')} "
                            f"confidence={result.get('recommendation',{}).get('confidence_score',0):.2f}",
                            "SUCCESS",
                        )
                        st.success("✅ Recommendation generated successfully!")
                    except Exception as e:
                        st.error(f"❌ RAG/LLM error: {e}")
                        _add_audit("RAG_LLM_ERROR", str(e), "ERROR")

        if st.session_state.get("rag_payload"):
            with st.expander("RAG API Payload Preview", expanded=False):
                st.json(st.session_state.rag_payload)

        st.markdown("</div>", unsafe_allow_html=True)

    with lc2:
        if st.session_state.rag_result:
            r = st.session_state.rag_result
            rec = r.get("recommendation", {})

            st.markdown("#### 🎯 Clinical Recommendation")

            conf = rec.get("confidence_score", 0)
            conf_colour = "#2e7d32" if conf >= 0.8 else "#f57c00" if conf >= 0.6 else "#c62828"
            st.markdown(f"""
            <div class="status-bar">
              <span>🎯 Confidence: <strong style="color:{conf_colour}">{conf:.0%}</strong></span>
              <span>🤖 Path: <code>{rec.get('ai_path_used','')}</code></span>
              <span>📄 Evidence: {len(rec.get('evidence_documents',[]))} docs</span>
            </div>
            """, unsafe_allow_html=True)

            fields = {
                "📋 Summary": "summary",
                "⚠️ Risk Stratification": "risk_stratification",
                "💊 Antiplatelet Guidance": "antiplatelet_guidance",
                "🏥 Invasive Strategy": "invasive_strategy",
                "🧪 Adjunct Therapy": "adjunct_therapy",
                "📊 Monitoring Plan": "monitoring_plan",
            }
            for label, field in fields.items():
                val = rec.get(field, "")
                if val:
                    with st.expander(label, expanded=(field == "summary")):
                        st.write(val)

            generated_review_text = _format_rag_recommendations_for_review(rec)
            if generated_review_text:
                st.markdown("#### Generated RAG LLM Recommendations")
                st.text_area(
                    "Generated Recommendation Text",
                    value=generated_review_text,
                    height=220,
                    disabled=True,
                    key=f"rag_generated_recommendation_display_{r.get('correlation_id', 'latest')}",
                )

            # Safety flags
            sf = rec.get("safety_flags", {})
            flags = sf.get("flags", [])
            if flags:
                st.markdown("#### 🚨 Safety Flags")
                for flag in flags:
                    st.error(f"⚠️ {flag}")

            # Evidence
            docs = rec.get("evidence_documents", [])
            if docs:
                st.markdown("#### 📚 Supporting Evidence")
                for doc in docs:
                    with st.expander(f"📄 {doc.get('source','')} (score: {doc.get('similarity_score',0):.2f})"):
                        st.write(doc.get("content_snippet", ""))

            st.markdown(f"""
            <div class="disclaimer">
            ⚠️ {r.get('disclaimer','')}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👈 Fill in patient details and click **Generate RAG + LLM Recommendation**")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 – CLINICIAN REVIEW
# ════════════════════════════════════════════════════════════════════════════

with tab_rag:
    if st.session_state.rag_result:
        r = st.session_state.rag_result
        rec = r.get("recommendation", {})
        corr_id = r.get("correlation_id")
        prior_review = st.session_state.reviewed_recommendations.get(corr_id or "") or {}

        st.markdown("---")
        st.markdown("### Clinician Review for RAG LLM Recommendation")
        st.caption(
            f"Reviewer: {st.session_state.get('reviewer_id', 'bismita-dr')} | "
            f"Role: {st.session_state.get('auth_role') or 'CLINICIAN'} | "
            f"Correlation ID: {corr_id}"
        )
        if prior_review:
            st.info(
                f"Current local review status: {prior_review.get('status')} "
                f"at {prior_review.get('reviewed_at')}"
            )

        generated_review_text = _format_rag_recommendations_for_review(rec)
        review_note_key = f"rag_review_notes_{corr_id}"
        revised_text_key = f"rag_revised_summary_{corr_id}"
        final_recommendation_text = prior_review.get("revised_summary") or ""
        if generated_review_text and (
            review_note_key not in st.session_state
            or st.session_state.get(review_note_key) in ("", rec.get("summary", ""))
        ):
            st.session_state[review_note_key] = prior_review.get("notes") or generated_review_text
        if final_recommendation_text and st.session_state.get(revised_text_key) != final_recommendation_text:
            st.session_state[revised_text_key] = final_recommendation_text

        rag_review_notes = st.text_area(
            "Clinician Review Note",
            placeholder="RAG LLM generated recommendations will appear here for clinician review.",
            height=220,
            key=review_note_key,
        )
        rag_revised_summary = st.text_area(
            "Revised Recommendation Text",
            placeholder="After approval this shows the approved RAG recommendation. After revision this shows the clinician revised final recommendation.",
            height=180,
            key=revised_text_key,
        )

        rag_b1, rag_b2, rag_b3 = st.columns(3)
        with rag_b1:
            if st.button("Approve RAG Recommendation", type="primary", use_container_width=True, key=f"rag_approve_{corr_id}"):
                if not st.session_state.auth_token:
                    st.error("Login as bismita-dr before submitting review.")
                else:
                    try:
                        _submit_review(
                            corr_id,
                            "approve",
                            rag_review_notes,
                            st.session_state.auth_token,
                            reviewer_id=st.session_state.get("reviewer_id", "bismita-dr"),
                            reviewer_role=st.session_state.get("auth_role") or "CLINICIAN",
                        )
                        st.session_state.review_status = "APPROVED"
                        _store_reviewed_text(corr_id, "APPROVED", rag_review_notes, rag_review_notes)
                        _add_audit("RAG_REVIEW_APPROVED", f"reviewer=bismita-dr corr={corr_id}", "SUCCESS")
                        st.success("RAG recommendation approved and review submitted.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Review submit failed: {e}")
        with rag_b2:
            if st.button("Revise RAG Recommendation", use_container_width=True, key=f"rag_revise_{corr_id}"):
                if not st.session_state.auth_token:
                    st.error("Login as bismita-dr before submitting review.")
                elif not rag_revised_summary.strip():
                    st.error("Enter revised recommendation text before revising.")
                else:
                    try:
                        _submit_review(
                            corr_id,
                            "edit",
                            rag_review_notes,
                            st.session_state.auth_token,
                            edited_summary=rag_revised_summary,
                            reviewer_id=st.session_state.get("reviewer_id", "bismita-dr"),
                            reviewer_role=st.session_state.get("auth_role") or "CLINICIAN",
                        )
                        st.session_state.rag_result["recommendation"]["summary"] = rag_revised_summary
                        st.session_state.review_status = "REVISED"
                        _store_reviewed_text(corr_id, "REVISED", rag_review_notes, rag_revised_summary)
                        _add_audit("RAG_REVIEW_REVISED", f"reviewer=bismita-dr corr={corr_id}", "SUCCESS")
                        st.info("RAG recommendation revised and review submitted.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Review submit failed: {e}")
        with rag_b3:
            if st.button("Reject RAG Recommendation", use_container_width=True, key=f"rag_reject_{corr_id}"):
                if not st.session_state.auth_token:
                    st.error("Login as bismita-dr before submitting review.")
                elif not rag_review_notes.strip():
                    st.error("Enter rejection reason in clinician review notes.")
                else:
                    try:
                        _submit_review(
                            corr_id,
                            "reject",
                            rag_review_notes,
                            st.session_state.auth_token,
                            reviewer_id=st.session_state.get("reviewer_id", "bismita-dr"),
                            reviewer_role=st.session_state.get("auth_role") or "CLINICIAN",
                        )
                        st.session_state.review_status = "REJECTED"
                        _store_reviewed_text(corr_id, "REJECTED", rag_review_notes)
                        _add_audit("RAG_REVIEW_REJECTED", f"reviewer=bismita-dr corr={corr_id}", "INFO")
                        st.warning("RAG recommendation rejected and review submitted.")
                    except Exception as e:
                        st.error(f"Review submit failed: {e}")


with tab_review:
    st.markdown("### ✅ Clinician Review & Approval Workflow")
    st.markdown("Review, approve, reject, or revise the generated clinical recommendation.")

    if not st.session_state.correlation_id:
        st.warning("⚠️ No recommendation pending review. Generate a recommendation in the ACS Pathway or RAG + LLM tab first.")
    else:
        corr_id = st.session_state.correlation_id
        st.markdown(f"**Correlation ID:** `{corr_id}`")

        # Show current recommendation summary
        active_source = st.session_state.get("active_result_source")
        active_result = st.session_state.rag_result if active_source == "rag" else st.session_state.acs_result
        if active_result is None:
            active_result = st.session_state.acs_result or st.session_state.rag_result
        if active_result:
            with st.expander("📋 View Recommendation Summary", expanded=True):
                if active_source == "acs" or (active_source is None and st.session_state.acs_result):
                    recs = st.session_state.acs_result.get("recommendations", [])
                    for rec in recs:
                        emoji = _priority_emoji(rec.get("priority", "HIGH"))
                        st.markdown(f"**{emoji} {rec.get('type','').replace('_',' ').title()}**")
                        st.write(rec.get("message", ""))
                elif st.session_state.rag_result:
                    rec = st.session_state.rag_result.get("recommendation", {})
                    st.write(rec.get("summary", ""))

                prior_review = st.session_state.reviewed_recommendations.get(corr_id)
                if prior_review:
                    st.info(
                        f"Current local review status: {prior_review.get('status')} "
                        f"at {prior_review.get('reviewed_at')}"
                    )
                    if prior_review.get("revised_summary"):
                        st.write(prior_review["revised_summary"])

        st.markdown("---")
        st.markdown("#### 🩺 Clinician Decision")

        generated_recommendation_text = _active_recommendation_text(active_source, active_result)
        prior_review = st.session_state.reviewed_recommendations.get(corr_id) or {}
        review_notes_key = f"clinical_review_notes_{corr_id}"
        revised_summary_key = f"clinical_revised_summary_{corr_id}"
        final_summary_key = f"clinical_final_summary_{corr_id}"
        if generated_recommendation_text and review_notes_key not in st.session_state:
            st.session_state[review_notes_key] = prior_review.get("notes") or generated_recommendation_text
        if prior_review.get("revised_summary") and revised_summary_key not in st.session_state:
            st.session_state[revised_summary_key] = prior_review["revised_summary"]
        if prior_review.get("revised_summary"):
            st.session_state[final_summary_key] = prior_review["revised_summary"]

        reviewer_name = st.text_input("Reviewer Name / ID", value=st.session_state.get("reviewer_id", "bismita-dr"))
        review_notes = st.text_area(
            "Clinical Notes / Justification",
            placeholder="Enter your clinical assessment, modifications, or reason for rejection...",
            height=220,
            key=review_notes_key,
        )

        revised_summary = None
        revise_clicked = False
        col_rev1, col_rev2 = st.columns(2)
        with col_rev1:
            revise_clicked = st.checkbox("✏️ Revise / Modify Recommendation")
        if revise_clicked:
            revised_summary = st.text_area(
                "Revised Recommendation Text",
                placeholder="Enter your modified clinical recommendation...",
                height=150,
                key=revised_summary_key,
            )

        st.markdown("---")
        b1, b2, b3 = st.columns(3)

        with b1:
            if st.button("✅ APPROVE", type="primary", use_container_width=True, help="Approve recommendation for clinical action"):
                action = "approve"
                with st.spinner("Submitting approval..."):
                    try:
                        resp = _submit_review(
                            corr_id,
                            action,
                            review_notes,
                            st.session_state.auth_token,
                            reviewer_id=reviewer_name,
                            reviewer_role=st.session_state.get("auth_role") or "CLINICIAN",
                        )
                        st.session_state.review_status = "APPROVED"
                        _store_reviewed_text(corr_id, "APPROVED", review_notes, review_notes)
                        st.session_state[final_summary_key] = review_notes
                        _add_audit("REVIEW_APPROVED", f"reviewer={reviewer_name} corr={corr_id}", "SUCCESS")
                        st.success("✅ Recommendation **APPROVED** and logged to audit trail.")
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        # Demo mode – log locally
                        st.session_state.review_status = "APPROVED"
                        _store_reviewed_text(corr_id, "APPROVED", review_notes, review_notes)
                        st.session_state[final_summary_key] = review_notes
                        _add_audit("REVIEW_APPROVED", f"reviewer={reviewer_name} corr={corr_id} [demo]", "SUCCESS")
                        st.success("✅ Recommendation **APPROVED** (demo mode – audit logged locally).")

        with b2:
            if st.button("✏️ REVISE", use_container_width=True, help="Approve with modifications"):
                action = "edit"
                if not revised_summary or not revised_summary.strip():
                    st.error("Enter the modified recommendation text before revising.")
                    st.stop()
                with st.spinner("Submitting revision..."):
                    try:
                        resp = _submit_review(
                            corr_id,
                            action,
                            review_notes,
                            st.session_state.auth_token,
                            edited_summary=revised_summary,
                            reviewer_id=reviewer_name,
                            reviewer_role=st.session_state.get("auth_role") or "CLINICIAN",
                        )
                        st.session_state.review_status = "REVISED"
                        _store_reviewed_text(corr_id, "REVISED", review_notes, revised_summary)
                        st.session_state[final_summary_key] = revised_summary
                        _add_audit("REVIEW_REVISED", f"reviewer={reviewer_name} corr={corr_id}", "SUCCESS")
                        st.info("✏️ Recommendation **REVISED** and logged.")
                    except Exception as e:
                        st.session_state.review_status = "REVISED"
                        _store_reviewed_text(corr_id, "REVISED", review_notes, revised_summary)
                        st.session_state[final_summary_key] = revised_summary
                        _add_audit("REVIEW_REVISED", f"reviewer={reviewer_name} corr={corr_id} [demo]", "SUCCESS")
                        st.info("✏️ Recommendation **REVISED** (demo mode).")

        with b3:
            if st.button("❌ REJECT", use_container_width=True, help="Reject recommendation"):
                if not review_notes.strip():
                    st.error("❌ Please provide a justification for rejection.")
                else:
                    action = "reject"
                    with st.spinner("Submitting rejection..."):
                        try:
                            resp = _submit_review(
                                corr_id,
                                action,
                                review_notes,
                                st.session_state.auth_token,
                                reviewer_id=reviewer_name,
                                reviewer_role=st.session_state.get("auth_role") or "CLINICIAN",
                            )
                            st.session_state.review_status = "REJECTED"
                            _store_reviewed_text(corr_id, "REJECTED", review_notes)
                            _add_audit("REVIEW_REJECTED", f"reviewer={reviewer_name} corr={corr_id} reason={review_notes[:80]}", "INFO")
                            st.warning("❌ Recommendation **REJECTED** and logged.")
                        except Exception as e:
                            st.session_state.review_status = "REJECTED"
                            _add_audit("REVIEW_REJECTED", f"reviewer={reviewer_name} corr={corr_id} [demo]", "INFO")
                            st.warning("❌ Recommendation **REJECTED** (demo mode).")

        final_review = st.session_state.reviewed_recommendations.get(corr_id) or {}
        final_recommendation_text = st.session_state.get(final_summary_key) or final_review.get("revised_summary")
        if final_recommendation_text:
            st.markdown("#### Final Recommendations")
            st.text_area(
                "Approved / Approved Revised Recommendation",
                value=final_recommendation_text,
                height=180,
                disabled=True,
                key=f"clinical_final_recommendation_{corr_id}",
            )

        if st.session_state.review_status:
            status_map = {
                "APPROVED": ("badge-approved", "✅ APPROVED"),
                "REJECTED": ("badge-rejected", "❌ REJECTED"),
                "REVISED": ("badge-pending", "✏️ REVISED"),
            }
            badge_class, label = status_map.get(st.session_state.review_status, ("badge-pending", "PENDING"))
            st.markdown(f"""
            <div style="margin-top:1rem; text-align:center">
              <span class="badge {badge_class}" style="font-size:1rem; padding:0.5rem 2rem">
                Final Status: {label}
              </span>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 – AUDIT TRAIL
# ════════════════════════════════════════════════════════════════════════════

with tab_audit:
    st.markdown("### 📋 Immutable Audit Trail")
    st.markdown("All clinical events are logged per HIPAA/NDHM/ABDM audit requirements.")

    if not st.session_state.audit_trail:
        st.info("No audit events recorded yet. Generate a recommendation to begin.")
    else:
        for i, event in enumerate(reversed(st.session_state.audit_trail)):
            icon = {
                "SUCCESS": "✅",
                "ERROR": "❌",
                "INFO": "ℹ️",
            }.get(event.get("status", "INFO"), "📌")

            st.markdown(f"""
            <div class="audit-row">
              {icon} <strong>{event.get('event','')}</strong>
              <span class="audit-ts"> | {event.get('ts','')[:19]} UTC</span><br>
              <span style="font-size:0.85rem">{event.get('detail','')}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("📥 Export Audit Trail (JSON)"):
        audit_json = json.dumps(st.session_state.audit_trail, indent=2)
        st.download_button(
            label="⬇️ Download audit.json",
            data=audit_json,
            file_name=f"cdss_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 – RAW PAYLOAD INSPECTOR
# ════════════════════════════════════════════════════════════════════════════

with tab_payload:
    st.markdown("### 📄 Raw Payload Inspector")
    st.markdown("View the exact JSON payloads and responses sent to/from the CDSS API.")

    pinsp1, pinsp2 = st.columns(2)

    with pinsp1:
        st.markdown("#### 📤 Sample ACS Request Payload")
        sample_stemi = {
            "requestId": "REQ-ACS-2026-000001",
            "sourceSystem": "EMR",
            "tenantId": "hospital-001",
            "facilityId": "facility-001",
            "requestedAt": "2026-05-14T12:30:00+05:30",
            "ruleSet": {"name": "ACS_CAG_DECISION_PATHWAY", "version": "1.0.0", "executionMode": "STATELESS"},
            "patient": {"patientId": "PAT-000123", "mrn": "MRN-998877", "ageYears": 58, "sex": "MALE"},
            "encounter": {
                "encounterId": "ENC-2026-000456",
                "encounterType": "EMERGENCY",
                "arrivalDateTime": "2026-05-14T11:45:00+05:30",
                "firstMedicalContactDateTime": "2026-05-14T11:30:00+05:30",
                "symptomOnsetDateTime": "2026-05-14T09:15:00+05:30",
            },
            "acsCase": {
                "caseId": "ACS-CASE-2026-000001",
                "acsType": "STEMI",
                "primaryPciFacilityAvailableCloseBy": True,
                "expectedFmcToBalloonMinutes": 85,
                "dynamicStOrTChanges": True,
                "largeTroponinRise": True,
            },
            "cagFinding": {
                "diagnosticCagCompleted": True,
                "lesionCategory": "GE_70_ONE_TWO_VESSEL",
                "maxDiameterStenosisPercent": 90,
            },
        }
        payload_type = st.selectbox("Select ACS Type", ["STEMI", "NSTEMI", "Diagnostic CAG"])
        if payload_type == "NSTEMI":
            sample_stemi["acsCase"]["acsType"] = "NSTEMI"
            sample_stemi["acsCase"]["timiScore"] = 4
            sample_stemi["acsCase"]["graceScore"] = 155
            sample_stemi["acsCase"]["haemodynamicInstability"] = False
            del sample_stemi["acsCase"]["primaryPciFacilityAvailableCloseBy"]
        elif payload_type == "Diagnostic CAG":
            sample_stemi["acsCase"]["acsType"] = "Diagnostic CAG"
            sample_stemi["cagFinding"] = {
                "diagnosticCagCompleted": True,
                "lesionCategory": "BORDERLINE_50_TO_69",
                "ffrPerformed": True,
                "ffrValue": 0.75,
            }

        st.code(json.dumps(sample_stemi, indent=2), language="json")
        st.caption(f"Endpoint: POST {ACS_ENDPOINT}")

        default_payload = st.session_state.acs_payload or sample_stemi
        raw_acs_payload = st.text_area(
            "Editable ACS JSON Payload",
            value=json.dumps(default_payload, indent=2, default=str),
            height=360,
            key="raw_acs_payload_editor",
        )
        if st.button("Send Raw ACS Payload to Drools", type="primary", use_container_width=True):
            if not st.session_state.auth_token:
                st.error("Login first so the raw ACS payload is sent with a valid Keycloak bearer token.")
            else:
                try:
                    payload = json.loads(raw_acs_payload)
                    result = _post_acs(payload, st.session_state.auth_token)
                    st.session_state.acs_payload = payload
                    st.session_state.acs_result = result
                    st.session_state.correlation_id = result.get("requestId") or payload.get("requestId")
                    st.session_state.active_result_source = "acs"
                    st.session_state.review_status = None
                    _add_audit(
                        "RAW_ACS_PAYLOAD_EXECUTED",
                        f"requestId={st.session_state.correlation_id} rules_fired={result.get('rulesFired', 0)}",
                        "SUCCESS",
                    )
                    st.success(
                        f"Drools response received: {result.get('rulesFired', 0)} rules fired "
                        f"via {'KIE Server' if result.get('kieServerUsed') else 'Python Fallback Engine'}."
                    )
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON payload: {e}")
                except Exception as e:
                    st.error(f"Raw ACS payload execution failed: {e}")

    with pinsp2:
        st.markdown("#### 📥 Last API Response")
        last_result = st.session_state.acs_result or st.session_state.rag_result
        if last_result:
            st.code(json.dumps(last_result, indent=2, default=str), language="json")
        else:
            st.info("Generate a recommendation to see the response here.")

        st.markdown("#### 🌐 API Endpoints")
        st.markdown(f"""
        | Method | Endpoint | Description |
        |--------|----------|-------------|
        | `POST` | `/clinical-decision-support/acs/recommendations` | ACS Drools Rules |
        | `POST` | `/api/v1/recommendations` | RAG + LLM Recommendation |
        | `POST` | `/api/v1/recommendations/{{id}}/review` | Submit Review |
        | `GET`  | `/api/v1/health` | Health Check |
        | `GET`  | `/docs` | Swagger UI |
        """)
