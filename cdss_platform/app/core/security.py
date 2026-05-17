"""
CDSS Platform – Security Module
JWT validation via Keycloak RS256 JWKS.

Full implementation per doc requirements:
  - RS256 JWKS with TTL-based cache refresh
  - Redis-backed JTI revocation
  - audience (aud) validation
  - source_system + groups claim extraction
  - Fine-grained permission / scope checking (cdss:patient:read etc.)
  - Role-level RBAC + scope-level ABAC
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwk, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)

# ─────────────────────────────────────────────
# Permission scopes (doc §3 + §12)
# ─────────────────────────────────────────────
class Scope:
    PATIENT_READ          = "cdss:patient:read"
    PATIENT_WRITE         = "cdss:patient:write"
    ENCOUNTER_READ        = "cdss:encounter:read"
    ENCOUNTER_WRITE       = "cdss:encounter:write"
    RECOMMENDATION_READ   = "cdss:recommendation:read"
    RECOMMENDATION_CREATE = "cdss:recommendation:create"
    MEDICATION_READ       = "cdss:medication:read"
    MEDICATION_WRITE      = "cdss:medication:write"
    AUDIT_READ            = "cdss:audit:read"
    ADMIN                 = "cdss:admin"


# Role → default scopes mapping (used when JWT has no explicit scope claim)
_ROLE_SCOPES: dict[str, set[str]] = {
    "CARDIOLOGIST":    {Scope.PATIENT_READ, Scope.ENCOUNTER_READ, Scope.ENCOUNTER_WRITE,
                        Scope.RECOMMENDATION_READ, Scope.RECOMMENDATION_CREATE,
                        Scope.MEDICATION_READ},
    "CLINICIAN":       {Scope.PATIENT_READ, Scope.ENCOUNTER_READ, Scope.ENCOUNTER_WRITE,
                        Scope.RECOMMENDATION_READ, Scope.RECOMMENDATION_CREATE,
                        Scope.MEDICATION_READ},
    "CDSS_NURSE":      {Scope.PATIENT_READ, Scope.ENCOUNTER_READ, Scope.ENCOUNTER_WRITE,
                        Scope.MEDICATION_READ},
    "CDSS_PHARMACIST": {Scope.MEDICATION_READ, Scope.MEDICATION_WRITE},
    "CDSS_MANAGEMENT": {Scope.AUDIT_READ},
    "CDSS_ADMIN":      {Scope.PATIENT_READ, Scope.PATIENT_WRITE, Scope.ENCOUNTER_READ,
                        Scope.ENCOUNTER_WRITE, Scope.RECOMMENDATION_READ,
                        Scope.RECOMMENDATION_CREATE, Scope.MEDICATION_READ,
                        Scope.MEDICATION_WRITE, Scope.AUDIT_READ, Scope.ADMIN},
    "CDSS_USER":       {Scope.PATIENT_READ, Scope.RECOMMENDATION_READ},
    # External system roles (doc §2)
    "HMIS":            {Scope.PATIENT_READ, Scope.ENCOUNTER_WRITE, Scope.RECOMMENDATION_CREATE},
    "EMR":             {Scope.PATIENT_READ, Scope.ENCOUNTER_READ, Scope.ENCOUNTER_WRITE},
    "PHR":             {Scope.PATIENT_READ, Scope.RECOMMENDATION_READ},
    "DHA":             {Scope.AUDIT_READ},
}


# ─────────────────────────────────────────────
# Token Models
# ─────────────────────────────────────────────

class TokenPayload(BaseModel):
    sub: str
    role: str = "CDSS_USER"
    org_id: Optional[str] = None
    source_system: Optional[str] = None   # NEW – doc §7
    groups: list[str] = []                # NEW – doc §7 Keycloak groups
    scopes: set[str] = set()              # NEW – resolved permission scopes
    exp: Optional[int] = None
    iat: Optional[int] = None
    jti: Optional[str] = None
    iss: Optional[str] = None
    azp: Optional[str] = None            # client_id of calling system

    class Config:
        arbitrary_types_allowed = True


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


# ─────────────────────────────────────────────
# Redis-backed JTI Revocation (doc §10)
# ─────────────────────────────────────────────

class _JTIRevocationStore:
    def __init__(self) -> None:
        self._fallback: set[str] = set()
        self._redis: Any = None
        self._init_redis()

    def _init_redis(self) -> None:
        try:
            import redis as _redis
            client = _redis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=3)
            client.ping()
            self._redis = client
            logger.info("[JTI] Redis revocation store connected")
        except Exception as exc:
            logger.warning("[JTI] Redis unavailable (%s) – using in-process fallback", exc)

    def revoke(self, jti: str, ttl_seconds: int = 86400) -> None:
        if self._redis:
            try:
                self._redis.setex(f"cdss:revoked:{jti}", ttl_seconds, "1")
                return
            except Exception:
                pass
        self._fallback.add(jti)

    def is_revoked(self, jti: str) -> bool:
        if self._redis:
            try:
                return bool(self._redis.exists(f"cdss:revoked:{jti}"))
            except Exception:
                pass
        return jti in self._fallback


_jti_store = _JTIRevocationStore()


# ─────────────────────────────────────────────
# JWKS Cache with TTL refresh (doc §11)
# ─────────────────────────────────────────────

class _JWKSCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keys: dict[str, Any] = {}
        self._fetched_at: float = 0.0

    def _is_stale(self) -> bool:
        return (time.monotonic() - self._fetched_at) > settings.jwks_cache_ttl_seconds

    async def _fetch_and_cache(self) -> None:
        url = settings.jwks_uri
        logger.info("[JWKS] Fetching keys from %s", url)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Cannot reach Keycloak JWKS endpoint: {exc}",
            )
        new_keys: dict[str, Any] = {}
        for key_data in resp.json().get("keys", []):
            if key_data.get("use") == "sig":
                kid = key_data.get("kid", "default")
                new_keys[kid] = jwk.construct(key_data)
        self._keys = new_keys
        self._fetched_at = time.monotonic()
        logger.info("[JWKS] Cached %d signing key(s)", len(new_keys))

    async def get_key(self, kid: Optional[str]) -> Any:
        with self._lock:
            if not self._is_stale() and kid and kid in self._keys:
                return self._keys[kid]
        await self._fetch_and_cache()
        with self._lock:
            if kid and kid in self._keys:
                return self._keys[kid]
            if self._keys:
                return next(iter(self._keys.values()))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No suitable signing key found in Keycloak JWKS",
        )


_jwks_cache = _JWKSCache()


# ─────────────────────────────────────────────
# Core Token Decode
# ─────────────────────────────────────────────

async def decode_token(token: str) -> TokenPayload:
    """
    Validate Keycloak RS256 JWT per doc §11:
      signature → issuer → expiry → audience → JTI → claims extraction
    """
    try:
        unverified_header = jwt.get_unverified_header(token)
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail=f"Malformed JWT header: {exc}",
                            headers={"WWW-Authenticate": "Bearer"})

    kid = unverified_header.get("kid")
    alg = unverified_header.get("alg", "RS256")
    if alg != "RS256":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail=f"Unsupported algorithm '{alg}'. Only RS256 accepted.",
                            headers={"WWW-Authenticate": "Bearer"})

    public_key = await _jwks_cache.get_key(kid)

    try:
        data: dict = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            options={
                "verify_exp": True,
                "verify_iat": True,
                "verify_aud": settings.jwt_verify_audience,
                **({"audience": settings.keycloak_client_id} if settings.jwt_verify_audience else {}),
            },
        )
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail=f"Token validation failed: {exc}",
                            headers={"WWW-Authenticate": "Bearer"})

    # Issuer check
    token_issuer = data.get("iss", "")
    if token_issuer and token_issuer != settings.keycloak_issuer:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail=f"Issuer mismatch: expected '{settings.keycloak_issuer}'",
                            headers={"WWW-Authenticate": "Bearer"})

    # JTI revocation
    jti = data.get("jti")
    if jti and _jti_store.is_revoked(jti):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Token has been revoked")

    role = _extract_role(data)
    groups = _extract_groups(data)
    scopes = _extract_scopes(data, role)
    source_system = data.get("source_system") or data.get("azp")

    return TokenPayload(
        sub=data.get("sub", ""),
        role=role,
        org_id=data.get("org_id") or data.get("organization_id") or data.get("organization"),
        source_system=source_system,
        groups=groups,
        scopes=scopes,
        exp=data.get("exp"),
        iat=data.get("iat"),
        jti=jti,
        iss=data.get("iss"),
        azp=data.get("azp"),
    )


def _extract_role(claims: dict) -> str:
    """
    Extract role from Keycloak JWT (doc §2 realm roles + client roles).
    Accepts both bare names (NURSE) and CDSS_-prefixed names (CDSS_NURSE).
    """
    # Bare → normalised mapping so both forms resolve to the same scope set
    _NORMALISE = {
        "NURSE": "CDSS_NURSE",
        "PHARMACIST": "CDSS_PHARMACIST",
        "MANAGEMENT": "CDSS_MANAGEMENT",
        "PATIENT": "CDSS_PATIENT",
        "ADMIN": "CDSS_ADMIN",
        "USER": "CDSS_USER",
    }
    all_roles = {
        "CARDIOLOGIST", "CLINICIAN",
        "NURSE", "CDSS_NURSE",
        "CDSS_ADMIN", "CDSS_PHARMACIST", "PHARMACIST",
        "CDSS_USER", "CDSS_MANAGEMENT", "MANAGEMENT",
        "CDSS_PATIENT", "PATIENT",
        "HMIS", "EMR", "PHR", "DHA",
    }

    def _normalise(r: str) -> str:
        upper = r.upper()
        return _NORMALISE.get(upper, upper)

    client_id = settings.keycloak_client_id
    for r in claims.get("resource_access", {}).get(client_id, {}).get("roles", []):
        if r.upper() in all_roles:
            return _normalise(r)
    for r in claims.get("realm_access", {}).get("roles", []):
        if r.upper() in all_roles:
            return _normalise(r)
    flat = claims.get("role", "")
    if flat and flat.upper() in all_roles:
        return _normalise(flat)
    return "CDSS_USER"


def _extract_groups(claims: dict) -> list[str]:
    """Extract Keycloak group memberships (doc §7 groups claim)."""
    return [str(g).strip("/") for g in claims.get("groups", [])]


def _extract_scopes(claims: dict, role: str) -> set[str]:
    """
    Resolve permission scopes (doc §3 + §6 + §12).
    Priority: explicit scope claim > role-default scopes.
    """
    # Explicit scope string in JWT (set by Keycloak scope mapper)
    raw_scope = claims.get("scope", "")
    cdss_scopes = {s for s in raw_scope.split() if s.startswith("cdss:")}
    if cdss_scopes:
        return cdss_scopes
    # Fall back to role-derived defaults
    return _ROLE_SCOPES.get(role, _ROLE_SCOPES["CDSS_USER"]).copy()


def revoke_token(jti: str, ttl_seconds: int = 86400) -> None:
    _jti_store.revoke(jti, ttl_seconds)


# ─────────────────────────────────────────────
# FastAPI Dependencies
# ─────────────────────────────────────────────

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> TokenPayload:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Authorization header missing",
                            headers={"WWW-Authenticate": "Bearer"})
    return await decode_token(credentials.credentials)


def require_role(*allowed_roles: str):
    """Broad role-based guard."""
    def _check(current_user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
        if current_user.role not in allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                detail=f"Role '{current_user.role}' not authorised.")
        return current_user
    return _check


def require_scope(*required_scopes: str):
    """
    Fine-grained permission guard (doc §12).
    The caller must hold ALL of the listed scopes.
    """
    def _check(current_user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
        missing = set(required_scopes) - current_user.scopes
        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permissions: {', '.join(sorted(missing))}",
            )
        return current_user
    return _check


# ── Convenience scope guards (doc §12) ───────────────────────────────────────
require_patient_read          = require_scope(Scope.PATIENT_READ)
require_patient_write         = require_scope(Scope.PATIENT_READ, Scope.PATIENT_WRITE)
require_encounter_read        = require_scope(Scope.ENCOUNTER_READ)
require_encounter_write       = require_scope(Scope.ENCOUNTER_WRITE)
require_recommendation_read   = require_scope(Scope.RECOMMENDATION_READ)
require_recommendation_create = require_scope(Scope.RECOMMENDATION_READ, Scope.RECOMMENDATION_CREATE)
require_medication_read       = require_scope(Scope.MEDICATION_READ)
require_medication_write      = require_scope(Scope.MEDICATION_READ, Scope.MEDICATION_WRITE)
require_audit_read            = require_scope(Scope.AUDIT_READ)
require_admin                 = require_scope(Scope.ADMIN)

# Legacy role aliases (backward compat)
require_cardiologist = require_role("CARDIOLOGIST", "CDSS_ADMIN")
require_clinical     = require_role("CARDIOLOGIST", "CLINICIAN", "CDSS_ADMIN")

# Kept for test compat
DEV_USERS: dict = {}
def authenticate_user(user_id: str, password: str) -> Optional[dict]:
    return None
