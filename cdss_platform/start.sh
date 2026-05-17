#!/usr/bin/env bash
# ─────────────────────────────────────────────────────
# CDSS Platform – Quick Start Script
# Run this once to set up and start the server.
# ─────────────────────────────────────────────────────
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo "  CDSS Clinical Intelligence Platform – Setup"
echo "============================================================"

# 1. Check Python version
python_ver=$(python3 --version 2>&1)
echo "[CHECK] Python: $python_ver"

# 2. Create venv if missing
if [ ! -d "venv" ]; then
  echo "[SETUP] Creating virtual environment..."
  python3 -m venv venv
fi

# 3. Activate venv
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null || true

# 4. Install dependencies
echo "[SETUP] Installing dependencies (this may take a few minutes)..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# 5. Copy .env if missing
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "[SETUP] Created .env from .env.example"
  echo ""
  echo "  ⚠️  ACTION REQUIRED: Edit .env and set ANTHROPIC_API_KEY"
  echo ""
fi

# 6. Ingest guidelines
echo "[SETUP] Ingesting clinical guidelines into ChromaDB..."
python scripts/manage.py ingest

# 7. Start server
echo ""
echo "[START] Starting CDSS Platform..."
echo "  Dashboard: http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo "============================================================"
python run.py
