"""
CDSS Platform – FastAPI Application (Full Production)

FIXES APPLIED
─────────────
BUG 3 – 422 errors were silent
  FastAPI's default 422 handler returns the Pydantic error list but the
  server log showed nothing — making it impossible to know which field
  failed without a network inspector.

  FIX: Add a RequestValidationError handler that logs the exact field
  path and error type at ERROR level before returning the 422 response.
  This also enriches the response body so the frontend can surface the
  exact problem to developers.
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

from app.core.config import get_settings
from app.core.middleware import (
    AccessLogMiddleware,
    CorrelationIDMiddleware,
    GatewayPredicateMiddleware,
    RateLimitMiddleware,
    TimingMiddleware,
)
from app.api.routes.recommendations import router as rec_router
from app.api.routes.auth import router as auth_router
from app.api.routes.patients import router as patient_router
from app.api.routes.medications import router as med_router
from app.api.routes.encounters import router as enc_router
from app.api.routes.admin import router as admin_router
from app.api.routes.audit import router as audit_router
from app.api.routes.acs_pathways import router as acs_router
from app.rag.rag_engine import rag_engine

settings = get_settings()

# ── Logging setup ─────────────────────────────────────────────────────────────
settings.logs_dir  # ensures ./logs/ is created
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
)
logger.add(
    str(settings.logs_dir / "cdss.log"),
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info(f"  {settings.app_name}  v{settings.app_version}")
    logger.info(f"  Environment : {settings.app_env}")
    logger.info(f"  LLM Model   : {settings.llm_model}")
    logger.info(f"  Keycloak    : {settings.keycloak_url}/realms/{settings.keycloak_realm}")
    logger.info("=" * 60)

    try:
        from app.models.database import init_db
        await init_db()
        logger.info("[STARTUP] Database tables ready")
    except Exception as e:
        logger.warning(f"[STARTUP] DB init skipped: {e}")

    guideline_dir = Path("./data/guidelines")
    if guideline_dir.exists() and any(guideline_dir.glob("*.txt")):
        logger.info("[STARTUP] Ingesting clinical guidelines into ChromaDB...")
        try:
            count = rag_engine.ingest_guidelines_directory(guideline_dir)
            logger.info(f"[STARTUP] RAG ready: {count} chunks indexed")
        except Exception as e:
            logger.error(f"[STARTUP] RAG ingestion error: {e}")
    else:
        logger.warning("[STARTUP] No guidelines found in data/guidelines/")

    yield
    logger.info("[SHUTDOWN] CDSS Platform shutting down.")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "## CDSS Clinical Intelligence Platform\n\n"
        "Enterprise RAG + Claude LLM + Safety Gate + Human-in-the-Loop\n\n"
        "**Auth:** `POST /api/v1/auth/token` → body: `{username, password}`\n\n"
        "Default user: `admin9` / `deepak@123"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Clinical Decision Support"},
        {"name": "ACS Clinical Pathways"},
        {"name": "Human Review"},
        {"name": "Authentication"},
        {"name": "Patients"},
        {"name": "Medications"},
        {"name": "Encounters"},
        {"name": "Admin"},
        {"name": "System"},
    ],
    lifespan=lifespan,
)

# ── Middleware (order matters — last added = outermost) ───────────────────────
_cors_origins = settings.allowed_origins_list
_allow_credentials = "*" not in _cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AccessLogMiddleware)
app.add_middleware(GatewayPredicateMiddleware)
app.add_middleware(TimingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(CorrelationIDMiddleware)

# ── Static / Templates ────────────────────────────────────────────────────────
static_dir    = Path("./frontend/static")
templates_dir = Path("./frontend/templates")
static_dir.mkdir(parents=True, exist_ok=True)
templates_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(rec_router,     prefix=settings.api_prefix)
app.include_router(auth_router,    prefix=settings.api_prefix)
app.include_router(patient_router, prefix=settings.api_prefix)
app.include_router(med_router,     prefix=settings.api_prefix)
app.include_router(enc_router,     prefix=settings.api_prefix)
app.include_router(admin_router,   prefix=settings.api_prefix)
app.include_router(audit_router,   prefix=settings.api_prefix)
app.include_router(
    acs_router,
    prefix="/clinical-decision-support",
    tags=["ACS Clinical Pathways"],
)


# ── Exception Handlers ────────────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    FIX BUG 3: Log the exact field path and error type for every 422.

    Server log shows e.g.:
      [VALIDATION 422] POST /api/v1/recommendations
        • body → patientContext → encounterType  missing  Field required
        • body → userRole  enum  Input should be 'CARDIOLOGIST' | 'CLINICIAN' | ...

    This makes debugging 422s instant — no network inspector needed.
    """
    errors = exc.errors()

    logger.error(
        "[VALIDATION 422] {} {}",
        request.method,
        request.url.path,
    )
    for err in errors:
        loc = " → ".join(str(l) for l in err.get("loc", []))
        logger.error(
            "  • {}  {}  {}",
            loc,
            err.get("type", ""),
            err.get("msg", ""),
        )

    # Also log the raw body so you can replay the exact request
    try:
        body = await request.json()
        logger.debug("[VALIDATION 422] Raw body: {}", body)
    except Exception:
        pass

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation failed",
            "detail": errors,
            "hint": (
                "Check that all field names use camelCase aliases "
                "(patientId, encounterId, userId, userRole, consentVerified, patientContext, "
                "encounterType, ecgFindings, currentMedications, cardiacHistory, "
                "systolicBp, diastolicBp, heartRate, respiratoryRate, eGFR, INR). "
                "userRole must be one of: CARDIOLOGIST, CLINICIAN, NURSE, PHARMACIST, "
                "CDSS_PATIENT, CDSS_ADMIN, CDSS_MANAGEMENT. "
                "consentVerified must be true."
            ),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "Unhandled error on {} {}: {}",
        request.method,
        request.url.path,
        exc,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Check server logs.",
        },
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})