"""
CDSS Platform – Regex Clinical Rules
Purpose: Pattern-based extraction of clinically significant text from
         free-text EMR fields (medications, allergies, contraindications).
"""

import re
from typing import Optional

# ────────────────────────────────────────────────────────────────────────────
# Antiplatelet Medication Patterns
# ────────────────────────────────────────────────────────────────────────────

ANTIPLATELET_PATTERNS: list[dict] = [
    {
        "name": "Aspirin",
        "codes": ["aspirin", "asa", "acetylsalicylic acid", "ecotrin"],
        "pattern": re.compile(
            r"\b(aspirin|asa|acetylsalicylic\s+acid|ecotrin)\b",
            re.IGNORECASE,
        ),
        "clinical_flag": "ASPIRIN_PRESENT",
    },
    {
        "name": "Clopidogrel",
        "codes": ["clopidogrel", "plavix"],
        "pattern": re.compile(r"\b(clopidogrel|plavix)\b", re.IGNORECASE),
        "clinical_flag": "CLOPIDOGREL_PRESENT",
    },
    {
        "name": "Ticagrelor",
        "codes": ["ticagrelor", "brilinta", "brilique"],
        "pattern": re.compile(r"\b(ticagrelor|brilinta|brilique)\b", re.IGNORECASE),
        "clinical_flag": "TICAGRELOR_PRESENT",
    },
    {
        "name": "Prasugrel",
        "codes": ["prasugrel", "effient", "efient"],
        "pattern": re.compile(r"\b(prasugrel|effient|efient)\b", re.IGNORECASE),
        "clinical_flag": "PRASUGREL_PRESENT",
    },
    {
        "name": "Tenecteplase",
        "codes": ["tenecteplase", "tнк", "tnkase", "metalyse"],
        "pattern": re.compile(r"\b(tenecteplase|tnkase|metalyse)\b", re.IGNORECASE),
        "clinical_flag": "FIBRINOLYTIC_GIVEN",
    },
]

# ────────────────────────────────────────────────────────────────────────────
# Allergy / Contraindication Patterns
# ────────────────────────────────────────────────────────────────────────────

ALLERGY_PATTERNS: list[dict] = [
    {
        "name": "Aspirin Allergy",
        "pattern": re.compile(
            r"\b(allerg|intoleran|contraindic|hypersensitiv|react)\w*\s+(to\s+)?(aspirin|asa)\b",
            re.IGNORECASE,
        ),
        "clinical_flag": "ASPIRIN_ALLERGY",
        "severity": "HIGH",
    },
    {
        "name": "Heparin-Induced Thrombocytopenia",
        "pattern": re.compile(r"\b(hit|heparin.induced\s+thrombocytopenia)\b", re.IGNORECASE),
        "clinical_flag": "HIT_DOCUMENTED",
        "severity": "HIGH",
    },
    {
        "name": "Contrast Allergy",
        "pattern": re.compile(
            r"\b(allerg|react|hypersensitiv)\w*\s+(to\s+)?(contrast|iodine|iodinated)\b",
            re.IGNORECASE,
        ),
        "clinical_flag": "CONTRAST_ALLERGY",
        "severity": "MEDIUM",
    },
    {
        "name": "Prasugrel Contraindication – Stroke/TIA",
        "pattern": re.compile(r"\b(stroke|tia|transient\s+ischaemic)\b", re.IGNORECASE),
        "clinical_flag": "PRASUGREL_CONTRAINDICATED",
        "severity": "HIGH",
    },
]

# ────────────────────────────────────────────────────────────────────────────
# Risk Score Extraction
# ────────────────────────────────────────────────────────────────────────────

TIMI_SCORE_PATTERN = re.compile(r"\btimi\s*(?:score)?\s*[=:]\s*(\d+)\b", re.IGNORECASE)
GRACE_SCORE_PATTERN = re.compile(r"\bgrace\s*(?:score)?\s*[=:]\s*(\d+)\b", re.IGNORECASE)
EGFR_PATTERN = re.compile(r"\b(?:egfr|estimated\s+gfr)\s*[=:]\s*([\d.]+)\s*(?:ml/min)?\b", re.IGNORECASE)
TROPONIN_PATTERN = re.compile(r"\btroponin[\s\w]*[=:]\s*([\d.]+)\s*(ng/ml|ug/l|ng/l)?\b", re.IGNORECASE)
LVEF_PATTERN = re.compile(r"\b(?:ef|ejection\s+fraction|lvef)\s*[=:]\s*([\d.]+)\s*%?\b", re.IGNORECASE)

# ────────────────────────────────────────────────────────────────────────────
# ACS Type Detection
# ────────────────────────────────────────────────────────────────────────────

ACS_TYPE_PATTERNS: list[dict] = [
    {
        "type": "STEMI",
        "pattern": re.compile(
            r"\b(stemi|st[\s-]elevation\s+(mi|myocardial\s+infarction)|st\s+elevation\s+mi)\b",
            re.IGNORECASE,
        ),
    },
    {
        "type": "NSTEMI",
        "pattern": re.compile(
            r"\b(nstemi|non[\s-]st[\s-]elevation\s+(mi|myocardial\s+infarction))\b",
            re.IGNORECASE,
        ),
    },
    {
        "type": "UA",
        "pattern": re.compile(r"\b(unstable\s+angina|ua)\b", re.IGNORECASE),
    },
    {
        "type": "Diagnostic CAG",
        "pattern": re.compile(
            r"\b(diagnostic\s+cag|coronary\s+angiograph|elective\s+cag|cath\s+lab\s+referral)\b",
            re.IGNORECASE,
        ),
    },
]


# ────────────────────────────────────────────────────────────────────────────
# Extraction Functions
# ────────────────────────────────────────────────────────────────────────────

def extract_medications(text: str) -> list[str]:
    """Return list of clinical flags for detected antiplatelet agents."""
    flags = []
    for p in ANTIPLATELET_PATTERNS:
        if p["pattern"].search(text):
            flags.append(p["clinical_flag"])
    return flags


def extract_allergies(text: str) -> list[dict]:
    """Return list of detected allergy/contraindication entries."""
    found = []
    for p in ALLERGY_PATTERNS:
        if p["pattern"].search(text):
            found.append({"flag": p["clinical_flag"], "severity": p["severity"], "name": p["name"]})
    return found


def extract_acs_type(text: str) -> Optional[str]:
    """Detect ACS type from free text; returns first match."""
    for p in ACS_TYPE_PATTERNS:
        if p["pattern"].search(text):
            return p["type"]
    return None


def extract_scores(text: str) -> dict:
    """Extract numeric risk scores and key lab values from free text."""
    result: dict = {}

    m = TIMI_SCORE_PATTERN.search(text)
    if m:
        result["timi_score"] = int(m.group(1))

    m = GRACE_SCORE_PATTERN.search(text)
    if m:
        result["grace_score"] = int(m.group(1))

    m = EGFR_PATTERN.search(text)
    if m:
        result["egfr"] = float(m.group(1))

    m = TROPONIN_PATTERN.search(text)
    if m:
        result["troponin_value"] = m.group(1)

    m = LVEF_PATTERN.search(text)
    if m:
        result["lvef_percent"] = float(m.group(1))

    return result


def run_all_regex_rules(free_text: str) -> dict:
    """
    Master entry point: run all regex rules against free-text EMR input.
    Returns a structured dict consumed by the CDSS pipeline.
    """
    return {
        "detected_medications": extract_medications(free_text),
        "detected_allergies": extract_allergies(free_text),
        "detected_acs_type": extract_acs_type(free_text),
        "extracted_scores": extract_scores(free_text),
    }
