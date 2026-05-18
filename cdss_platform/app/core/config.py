"""
CDSS Platform – Core Configuration
"""
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        case_sensitive=False, extra="ignore",
    )

    # App
    app_name: str = "CDSS Clinical Intelligence Platform"
    app_version: str = "1.0.0"
    app_env: str = "development"
    debug: bool = True
    secret_key: str = "change_me_in_production_must_be_32_chars_min"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"

    # Anthropic
    anthropic_api_key: str = ""

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_clinical: str = "cdss_clinical_guidelines"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Database + connection pooling
    database_url: str = "sqlite+aiosqlite:///./data/cdss.db"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30
    db_pool_recycle: int = 1800

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    rate_limit_enabled: bool = False

    # JWT
    jwt_algorithm: str = "RS256"
    jwt_access_token_expire_minutes: int = 15
    jwt_refresh_token_expire_minutes: int = 60
    jwt_verify_audience: bool = False   # set True in prod when Keycloak sets aud

    # Keycloak
    keycloak_url: str = "http://keycloak:8081"
    keycloak_realm: str = "cdss"
    keycloak_client_id: str = "cdss-api"
    keycloak_client_secret: str = ""
    jwks_cache_ttl_seconds: int = 300

    # LLM / Clinical AI
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 2000
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.3
    confidence_threshold_high: float = 0.80
    confidence_threshold_medium: float = 0.60
    safety_gate_enabled: bool = True

    # Audit
    audit_log_path: str = "./logs/audit.jsonl"
    immutable_audit: bool = True

    # CORS
    allowed_origins: str = "http://localhost:8000,http://127.0.0.1:8000"

    # OpenTelemetry
    otel_enabled: bool = False
    otel_exporter_otlp_endpoint: str = "http://otel-collector:4317"
    otel_service_name: str = "cdss-platform"

    # Circuit breaker
    cb_failure_threshold: int = 5
    cb_recovery_timeout: int = 30

    # KIE Server (Drools)
    kie_server_url: str = "http://localhost:8180"
    kie_server_user: str = "kieserver"
    kie_server_password: str = "kieserver1!"
    kie_container_id: str = "acs-clinical-rules"

    # ── Computed properties ──────────────────────────────────────────────────

    @property
    def jwks_uri(self) -> str:
        return f"{self.keycloak_url}/realms/{self.keycloak_realm}/protocol/openid-connect/certs"

    @property
    def keycloak_token_endpoint(self) -> str:
        return f"{self.keycloak_url}/realms/{self.keycloak_realm}/protocol/openid-connect/token"

    @property
    def keycloak_issuer(self) -> str:
        return f"{self.keycloak_url}/realms/{self.keycloak_realm}"

    @property
    def keycloak_end_session_endpoint(self) -> str:
        return f"{self.keycloak_url}/realms/{self.keycloak_realm}/protocol/openid-connect/logout"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def chroma_dir(self) -> Path:
        p = Path(self.chroma_persist_dir); p.mkdir(parents=True, exist_ok=True); return p

    @property
    def data_dir(self) -> Path:
        p = Path("./data"); p.mkdir(parents=True, exist_ok=True); return p

    @property
    def logs_dir(self) -> Path:
        p = Path("./logs"); p.mkdir(parents=True, exist_ok=True); return p


@lru_cache
def get_settings() -> Settings:
    return Settings()
