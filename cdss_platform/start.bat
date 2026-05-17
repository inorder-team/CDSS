@echo off
REM ─────────────────────────────────────────────────────
REM CDSS Platform – Quick Start (Windows)
REM ─────────────────────────────────────────────────────

cd /d "%~dp0"

echo ============================================================
echo   CDSS Clinical Intelligence Platform - Setup
echo ============================================================

REM Create venv if missing
if not exist "venv" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
)

REM Activate
call venv\Scripts\activate.bat

REM Install
echo [SETUP] Installing dependencies...
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

REM Copy .env
if not exist ".env" (
    copy .env.example .env
    echo [SETUP] Created .env - please set ANTHROPIC_API_KEY in .env
    pause
)

REM Ingest
echo [SETUP] Ingesting guidelines into ChromaDB...
python scripts\manage.py ingest

REM Start
echo.
echo [START] Starting server at http://localhost:8000
echo.
python run.py
pause
