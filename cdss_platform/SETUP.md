# CDSS Platform – PyCharm Community Edition Setup

## Prerequisites
- Python 3.11+
- PyCharm Community Edition
- Internet access (for pip installs & Anthropic API)

---

## Step 1: Open Project in PyCharm
File → Open → select the `cdss_platform` folder

---

## Step 2: Create Virtual Environment
In PyCharm: Settings → Project → Python Interpreter → Add Interpreter → Virtualenv
- Location: `<project_root>/venv`
- Base: Python 3.11

Or in terminal:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

---

## Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

> Note: First install takes 5–10 minutes (downloads sentence-transformer model ~90MB)

---

## Step 4: Configure Environment
```bash
cp .env.example .env
```
Edit `.env` and set:
```
ANTHROPIC_API_KEY=sk-ant-your-real-key-here
```

---

## Step 5: Ingest Clinical Guidelines into ChromaDB
```bash
python scripts/manage.py ingest
```
Expected output:
```
[RAG] Ingesting: CARDIO-ACS-NSTEMI-2026.txt
[RAG] CARDIO-ACS-NSTEMI-2026.txt → 28 chunks
[RAG] Added 28 new chunks from CARDIO-ACS-NSTEMI-2026.txt
✓ Ingested 28 chunks into ChromaDB
```

---

## Step 6: Run Server
```bash
python run.py
```

Then open: http://localhost:8000

API docs: http://localhost:8000/docs

---

## Step 7: PyCharm Run Configurations

### Run Server
- Script: `run.py`
- Working directory: `<project_root>`

### Manage Script
- Script: `scripts/manage.py`
- Parameters: `ingest` / `test-rag` / `test-pipeline` / `audit`
- Working directory: `<project_root>`

### Run Tests
- Run → Edit Configurations → pytest
- Script: `tests/test_cdss.py`
- Working directory: `<project_root>`

---

## Project Structure
```
cdss_platform/
├── app/
│   ├── api/routes/
│   │   ├── recommendations.py   # POST /api/v1/recommendations
│   │   └── patients.py          # GET /api/v1/patients/**
│   ├── core/
│   │   ├── config.py            # Pydantic settings
│   │   ├── audit.py             # Immutable HIPAA audit log
│   │   ├── security.py          # JWT auth
│   │   └── middleware.py        # Rate limiting, correlation ID
│   ├── models/
│   │   ├── schemas.py           # All Pydantic models
│   │   └── database.py          # SQLAlchemy models
│   ├── rag/
│   │   └── rag_engine.py        # ChromaDB + sentence-transformers
│   ├── services/
│   │   ├── pipeline.py          # 6-layer CDS engine
│   │   └── safety_gate.py       # Layer 4 safety evaluation
│   └── main.py                  # FastAPI app
├── data/
│   ├── guidelines/              # Clinical guideline .txt files
│   └── chroma_db/               # ChromaDB persistent storage (auto-created)
├── frontend/templates/index.html # Clinical dashboard UI
├── logs/
│   ├── cdss.log                 # Application log
│   └── audit.jsonl              # Immutable audit trail
├── scripts/manage.py            # CLI management tool
├── tests/test_cdss.py           # pytest test suite
├── run.py                       # Server entry point
├── requirements.txt
└── .env                         # Your config (never commit)
```

---

## API Usage Example (curl)
```bash
curl -X POST http://localhost:8000/api/v1/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "patientId": "PAT-CARD-001",
    "encounterId": "ENC-CARD-001",
    "userId": "cardiologist.local",
    "userRole": "CARDIOLOGIST",
    "query": "Recommend guideline-based considerations for NSTEMI management.",
    "consentVerified": true,
    "patientContext": {
      "age": 68, "sex": "male",
      "encounterType": "cardiology-consult",
      "diagnoses": ["NSTEMI", "type-2-diabetes", "chronic-kidney-disease"],
      "ecgFindings": ["ST depression in lateral leads"],
      "labs": {"troponin": "elevated and rising", "eGFR": "42", "potassium": "4.8"},
      "vitals": {"systolicBp": "138", "heartRate": "92"},
      "currentMedications": ["atorvastatin", "metoprolol"],
      "allergies": ["aspirin"],
      "contraindications": ["documented aspirin allergy"],
      "cardiacHistory": ["prior PCI"]
    }
  }'
```

---

## Running Tests
```bash
pytest tests/ -v
# With coverage:
pip install pytest-cov
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## Troubleshooting
| Issue | Fix |
|-------|-----|
| `chromadb` install fails | `pip install chromadb==0.5.0 --no-cache-dir` |
| Sentence-transformer slow download | Wait, it downloads ~90MB model once |
| 503 on /recommendations | Set real ANTHROPIC_API_KEY in .env |
| Port 8000 in use | Change API_PORT in .env |
| RAG returns no results | Run `python scripts/manage.py ingest` first |
