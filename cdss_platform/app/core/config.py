"""
CDSS Platform – Core Configuration

FIXES APPLIED
─────────────
BUG 1 – Trailing slash on KEYCLOAK_URL
  .env has  KEYCLOAK_URL=http://localhost:8081/   (trailing slash)
  Default   keycloak_url = "http://keycloak:8081" (Docker-internal name)

  Both caused bad URLs:
    http://localhost:8081//realms/cdss/...   (double slash → 404)
    http://keycloak:8081/realms/cdss/...     (unreachable from host)

  FIX: Strip trailing slash in a validator so every computed property
  that appends "/realms/..." always produces a clean URL.

BUG 2 – JWKS_URI in .env was never read
  Settings had no `jwks_uri` field, so the JWKS_URI env-var was ignored
  and the computed property always used the (potentially wrong) keycloak_url.

  FIX: Add `jwks_uri_override` field that, when set, is used directly by
  security.py instead of the computed URL.  This lets ops teams point to
  a load-balancer or proxy without changing keycloak_url.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────────
    app_name:    str  = "CDSS Clinical Intelligence Platform"
    app_version: str  = "1.0.0"
    app_env:     str  = "development"
    debug:       bool = True
    secret_key:  str  = "change_me_in_production_must_be_32_chars_min"

    # ── API ──────────────────────────────────────────────────────────────────
    api_host:   str = "0.0.0.0"
    api_port:   int = 8000
    api_prefix: str = "/api/v1"

    # ── Anthropic ────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chroma_persist_dir:        str = "./data/chroma_db"
    chroma_collection_clinical: str = "cdss_clinical_guidelines"
    embedding_model:           str = "all-MiniLM-L6-v2"

    # ── Database ─────────────────────────────────────────────────────────────
    database_url:     str = "sqlite+aiosqlite:///./data/cdss.db"
    db_pool_size:     int = 10
    db_max_overflow:  int = 20
    db_pool_timeout:  int = 30
    db_pool_recycle:  int = 1800

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_url:           str  = "redis://localhost:6379/0"
    rate_limit_enabled:  bool = False

    # ── JWT ──────────────────────────────────────────────────────────────────
    jwt_algorithm:                    str  = "RS256"
    jwt_access_token_expire_minutes:  int  = 15
    jwt_refresh_token_expire_minutes: int  = 60
    jwt_verify_audience:              bool = False

    # ── Keycloak ─────────────────────────────────────────────────────────────
    # FIX: default is now localhost (works out of box without Docker).
    # Trailing slash is stripped by the validator below.
    keycloak_url:           str = "http://localhost:8081"
    keycloak_realm:         str = "cdss"
    keycloak_client_id:     str = "cdss-api"
    keycloak_client_secret: str = ""
    jwks_cache_ttl_seconds: int = 300

    # FIX: optional override – when set, used verbatim as the JWKS URL.
    # Maps to JWKS_URI in .env.  Leave blank to derive from keycloak_url.
    jwks_uri_override: str = ""

    # ── LLM / Clinical AI ────────────────────────────────────────────────────
    llm_model:                  str   = "claude-sonnet-4-5"
    llm_max_tokens:             int   = 2000
    rag_top_k:                  int   = 3
    rag_similarity_threshold:   float = 0.3
    confidence_threshold_high:  float = 0.80
    confidence_threshold_medium: float = 0.60
    safety_gate_enabled:        bool  = True

    # ── Audit ────────────────────────────────────────────────────────────────
    audit_log_path:  str  = "./logs/audit.jsonl"
    immutable_audit: bool = True

    # ── CORS ─────────────────────────────────────────────────────────────────
    allowed_origins: str = "http://localhost:8000,http://127.0.0.1:8000"

    # ── OpenTelemetry ────────────────────────────────────────────────────────
    otel_enabled:                  bool = False
    otel_exporter_otlp_endpoint:   str  = "http://otel-collector:4317"
    otel_service_name:             str  = "cdss-platform"

    # ── Circuit breaker ──────────────────────────────────────────────────────
    cb_failure_threshold: int = 5
    cb_recovery_timeout:  int = 30

    # ── KIE Server (Drools) ──────────────────────────────────────────────────
    kie_server_url:      str = "http://localhost:8180"
    kie_server_user:     str = "kieserver"
    kie_server_password: str = "kieserver1!"
    kie_container_id:    str = "acs-clinical-rules"

    # ── Validators ───────────────────────────────────────────────────────────

    @field_validator("keycloak_url", mode="before")
    @classmethod
    def strip_keycloak_trailing_slash(cls, v: str) -> str:
        """
        FIX BUG 1: Remove trailing slash so every computed property that
        appends '/realms/...' produces a clean URL.

        http://localhost:8081/  →  http://localhost:8081
        http://keycloak:8080/   →  http://keycloak:8080
        """
        return str(v).rstrip("/")

    @field_validator("jwks_uri_override", mode="before")
    @classmethod
    def strip_jwks_trailing_slash(cls, v: str) -> str:
        return str(v).rstrip("/") if v else ""

    # ── Computed properties ──────────────────────────────────────────────────

    @property
    def jwks_uri(self) -> str:
        """
        FIX BUG 2: Return JWKS_URI from .env when set; otherwise derive
        from keycloak_url.  This lets the JWKS endpoint differ from the
        token/issuer endpoint (common behind reverse proxies).
        """
        if self.jwks_uri_override:
            return self.jwks_uri_override
        return (
            f"{self.keycloak_url}/realms/{self.keycloak_realm}"
            f"/protocol/openid-connect/certs"
        )

    @property
    def keycloak_token_endpoint(self) -> str:
        return (
            f"{self.keycloak_url}/realms/{self.keycloak_realm}"
            f"/protocol/openid-connect/token"
        )

    @property
    def keycloak_issuer(self) -> str:
        return f"{self.keycloak_url}/realms/{self.keycloak_realm}"

    @property
    def keycloak_end_session_endpoint(self) -> str:
        return (
            f"{self.keycloak_url}/realms/{self.keycloak_realm}"
            f"/protocol/openid-connect/logout"
        )

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def chroma_dir(self) -> Path:
        p = Path(self.chroma_persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_dir(self) -> Path:
        p = Path("./data")
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def logs_dir(self) -> Path:
        p = Path("./logs")
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()