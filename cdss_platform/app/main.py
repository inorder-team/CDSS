"""
CDSS Platform – FastAPI Application (Full Production)
"""
from __future__ import annotations
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

from app.core.config import get_settings
from app.core.middleware import AccessLogMiddleware, CorrelationIDMiddleware, GatewayPredicateMiddleware, RateLimitMiddleware, TimingMiddleware
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

settings.logs_dir
logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
logger.add(str(settings.logs_dir / "cdss.log"), rotation="10 MB", retention="30 days", level="DEBUG")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info(f"  {settings.app_name}  v{settings.app_version}")
    logger.info(f"  Environment : {settings.app_env}")
    logger.info(f"  LLM Model   : {settings.llm_model}")
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


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "## CDSS Clinical Intelligence Platform\n\n"
        "Enterprise RAG + Claude LLM + Safety Gate + Human-in-the-Loop\n\n"
        "**Auth:** POST /api/v1/auth/token  →  username: `cardiologist@hospital.local` / password: `SecurePass123!`"
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

_cors_origins = settings.allowed_origins_list
_allow_credentials = "*" not in _cors_origins
app.add_middleware(CORSMiddleware, allow_origins=_cors_origins, allow_credentials=_allow_credentials, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(AccessLogMiddleware)
app.add_middleware(GatewayPredicateMiddleware)
app.add_middleware(TimingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(CorrelationIDMiddleware)

static_dir = Path("./frontend/static")
templates_dir = Path("./frontend/templates")
static_dir.mkdir(parents=True, exist_ok=True)
templates_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

app.include_router(rec_router,     prefix=settings.api_prefix)
app.include_router(auth_router,    prefix=settings.api_prefix)
app.include_router(patient_router, prefix=settings.api_prefix)
app.include_router(med_router,     prefix=settings.api_prefix)
app.include_router(enc_router,     prefix=settings.api_prefix)
app.include_router(admin_router,   prefix=settings.api_prefix)
app.include_router(audit_router,   prefix=settings.api_prefix)
app.include_router(acs_router,     prefix="/clinical-decision-support", tags=["ACS Clinical Pathways"])


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.method} {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": "An unexpected error occurred. Check server logs."})
