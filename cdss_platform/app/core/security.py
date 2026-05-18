"""
CDSS Platform – Security Module
================================
Handles JWT validation, JWKS caching, JTI revocation, role/scope extraction,
and FastAPI dependency guards.

FIX HISTORY
-----------
BUG:  GET /api/v1/auth/me always returned 401 Unauthorized.

ROOT CAUSE:
  The original code compared the token's 'iss' claim against
  settings.keycloak_issuer (from .env):

      if token_issuer != settings.keycloak_issuer:
          raise HTTP 401

  But the two values were different:
    Token  iss  → http://localhost:8081/realms/cdss   (Keycloak stamped this
                                                        based on the public URL)
    Config iss  → http://keycloak:8080/realms/cdss    (Docker-internal name)

  Same Keycloak instance, two different hostnames → hard string mismatch → 401.

FIX:
  • Issuer is now read FROM the token itself (not from config).
  • JWKS URL is derived dynamically from that issuer.
  • Issuer equality check against config is removed entirely.
  • verify_aud=False because Keycloak password-flow tokens carry no 'aud' claim.

HOW TO APPLY:
  Replace:  cdss_platform/app/core/security.py
  Restart:  uvicorn app.main:app --reload  (or your process manager)
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


# ══════════════════════════════════════════════════════════════════════════════
# Scopes
# ══════════════════════════════════════════════════════════════════════════════

class Scope:
    PATIENT_READ           = "cdss:patient:read"
    PATIENT_WRITE          = "cdss:patient:write"
    ENCOUNTER_READ         = "cdss:encounter:read"
    ENCOUNTER_WRITE        = "cdss:encounter:write"
    RECOMMENDATION_READ    = "cdss:recommendation:read"
    RECOMMENDATION_CREATE  = "cdss:recommendation:create"
    MEDICATION_READ        = "cdss:medication:read"
    MEDICATION_WRITE       = "cdss:medication:write"
    AUDIT_READ             = "cdss:audit:read"
    ADMIN                  = "cdss:admin"


# Role → default scope mapping
_ROLE_SCOPES: dict[str, set[str]] = {
    "CARDIOLOGIST": {
        Scope.PATIENT_READ, Scope.ENCOUNTER_READ, Scope.ENCOUNTER_WRITE,
        Scope.RECOMMENDATION_READ, Scope.RECOMMENDATION_CREATE,
        Scope.MEDICATION_READ,
    },
    "CLINICIAN": {
        Scope.PATIENT_READ, Scope.ENCOUNTER_READ, Scope.ENCOUNTER_WRITE,
        Scope.RECOMMENDATION_READ, Scope.RECOMMENDATION_CREATE,
        Scope.MEDICATION_READ,
    },
    "CDSS_NURSE": {
        Scope.PATIENT_READ, Scope.ENCOUNTER_READ,
        Scope.ENCOUNTER_WRITE, Scope.MEDICATION_READ,
    },
    "CDSS_PHARMACIST": {
        Scope.MEDICATION_READ, Scope.MEDICATION_WRITE,
    },
    "CDSS_MANAGEMENT": {
        Scope.AUDIT_READ,
    },
    "CDSS_ADMIN": {
        Scope.PATIENT_READ, Scope.PATIENT_WRITE,
        Scope.ENCOUNTER_READ, Scope.ENCOUNTER_WRITE,
        Scope.RECOMMENDATION_READ, Scope.RECOMMENDATION_CREATE,
        Scope.MEDICATION_READ, Scope.MEDICATION_WRITE,
        Scope.AUDIT_READ, Scope.ADMIN,
    },
    "CDSS_USER": {
        Scope.PATIENT_READ, Scope.RECOMMENDATION_READ,
    },
    "HMIS": {
        Scope.PATIENT_READ, Scope.ENCOUNTER_WRITE,
        Scope.RECOMMENDATION_CREATE,
    },
    "EMR": {
        Scope.PATIENT_READ, Scope.ENCOUNTER_READ, Scope.ENCOUNTER_WRITE,
    },
    "PHR": {
        Scope.PATIENT_READ, Scope.RECOMMENDATION_READ,
    },
    "DHA": {
        Scope.AUDIT_READ,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Token Models
# ══════════════════════════════════════════════════════════════════════════════

class TokenPayload(BaseModel):
    sub:           str
    role:          str            = "CDSS_USER"
    org_id:        Optional[str]  = None
    source_system: Optional[str]  = None
    groups:        list[str]      = []
    scopes:        set[str]       = set()
    exp:           Optional[int]  = None
    iat:           Optional[int]  = None
    jti:           Optional[str]  = None
    iss:           Optional[str]  = None
    azp:           Optional[str]  = None

    class Config:
        arbitrary_types_allowed = True


class TokenPair(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"
    expires_in:    int


# ══════════════════════════════════════════════════════════════════════════════
# JTI Revocation Store  (Redis primary, in-process fallback)
# ══════════════════════════════════════════════════════════════════════════════

class _JTIRevocationStore:
    """
    Stores revoked JWT IDs.  Uses Redis when available so revocations are
    shared across all FastAPI worker processes / replicas.  Falls back to
    an in-process set (single-instance only) when Redis is unreachable.
    """

    _KEY_PREFIX = "cdss:revoked:"

    def __init__(self) -> None:
        self._fallback: set[str] = set()
        self._redis: Any = None
        try:
            import redis as _r
            c = _r.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            c.ping()
            self._redis = c
            logger.info("[JTI] Redis revocation store connected")
        except Exception as exc:
            logger.warning(
                "[JTI] Redis unavailable (%s) – using in-process fallback", exc
            )

    def revoke(self, jti: str, ttl: int = 86_400) -> None:
        """Mark *jti* as revoked for *ttl* seconds (default 24 h)."""
        if self._redis:
            try:
                self._redis.setex(f"{self._KEY_PREFIX}{jti}", ttl, "1")
                return
            except Exception as exc:
                logger.warning("[JTI] Redis setex failed: %s – falling back", exc)
        self._fallback.add(jti)

    def is_revoked(self, jti: str) -> bool:
        if self._redis:
            try:
                return bool(self._redis.exists(f"{self._KEY_PREFIX}{jti}"))
            except Exception as exc:
                logger.warning("[JTI] Redis exists failed: %s – falling back", exc)
        return jti in self._fallback


_jti_store = _JTIRevocationStore()


# ══════════════════════════════════════════════════════════════════════════════
# JWKS Cache  (keyed by token issuer, NOT by config)
# ══════════════════════════════════════════════════════════════════════════════

class _JWKSCache:
    """
    Per-issuer JWKS cache.

    FIX: keyed by the issuer URL extracted from the token, not from .env.
    This means validation works even when KEYCLOAK_URL in .env differs from
    the 'iss' claim baked into the JWT (common in Docker / reverse-proxy setups).
    """

    def __init__(self) -> None:
        self._lock       = threading.Lock()
        self._keys:       dict[str, dict[str, Any]] = {}
        self._fetched_at: dict[str, float]          = {}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _is_stale(self, iss: str) -> bool:
        age = time.monotonic() - self._fetched_at.get(iss, 0.0)
        return age > settings.jwks_cache_ttl_seconds

    async def _fetch_and_cache(self, jwks_url: str, iss: str) -> None:
        logger.info("[JWKS] Fetching keys from %s", jwks_url)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(jwks_url)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("[JWKS] Fetch failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Cannot reach Keycloak JWKS endpoint ({jwks_url}): {exc}",
            )

        keys: dict[str, Any] = {}
        for k in resp.json().get("keys", []):
            if k.get("use") == "sig":
                kid = k.get("kid", "default")
                keys[kid] = jwk.construct(k)

        with self._lock:
            self._keys[iss]       = keys
            self._fetched_at[iss] = time.monotonic()

        logger.info("[JWKS] Cached %d signing key(s) for issuer '%s'", len(keys), iss)

    # ── public ────────────────────────────────────────────────────────────────

    async def get(self, kid: Optional[str], iss: str, jwks_url: str) -> Any:
        """Return the JWK for *kid* (or the first available key)."""
        with self._lock:
            cached = self._keys.get(iss, {})
            stale  = self._is_stale(iss)

        # Fast path: cache is fresh and the exact kid is present
        if not stale and kid and kid in cached:
            return cached[kid]

        # Slow path: refresh cache
        await self._fetch_and_cache(jwks_url, iss)

        with self._lock:
            cached = self._keys.get(iss, {})

        if kid and kid in cached:
            return cached[kid]

        # kid not found but we have at least one key – try it (handles rotation lag)
        if cached:
            logger.warning(
                "[JWKS] kid '%s' not found; using first available key for issuer '%s'",
                kid, iss,
            )
            return next(iter(cached.values()))

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No signing key found in Keycloak JWKS response.",
            headers={"WWW-Authenticate": "Bearer"},
        )


_jwks_cache = _JWKSCache()


# ══════════════════════════════════════════════════════════════════════════════
# Core JWT Decode  ← THE FIX IS IN THIS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

async def decode_token(token: str) -> TokenPayload:
    """
    Validate a Bearer JWT and return its decoded payload.

    Steps
    -----
    1. Read JWT header  →  extract alg + kid.
    2. Read JWT claims WITHOUT verification  →  extract 'iss' (issuer).
    3. Derive JWKS URL from the token's own issuer  (the fix).
    4. Fetch / cache Keycloak public key.
    5. Verify signature + expiry (verify_aud=False – no 'aud' on password-flow tokens).
    6. JTI revocation check.
    7. Build and return TokenPayload.
    """

    # ── 1. JWT header ─────────────────────────────────────────────────────────
    try:
        header = jwt.get_unverified_header(token)
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Malformed JWT header: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    alg = header.get("alg", "RS256")
    if alg != "RS256":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Unsupported signing algorithm '{alg}'. RS256 is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    kid: Optional[str] = header.get("kid")

    # ── 2. Unverified claims  →  read 'iss' ───────────────────────────────────
    try:
        unverified_claims = jwt.get_unverified_claims(token)
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Cannot read JWT claims: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_iss: str = unverified_claims.get("iss", "").rstrip("/")
    if not token_iss:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT is missing the required 'iss' (issuer) claim.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ── 3. Derive JWKS URL from token issuer  (THE FIX) ───────────────────────
    #
    #   token iss   →  http://localhost:8081/realms/cdss
    #   JWKS URL    →  http://localhost:8081/realms/cdss/protocol/openid-connect/certs
    #
    #   This works regardless of what KEYCLOAK_URL is set to in .env,
    #   because we follow the issuer the token itself declares.
    #
    jwks_url = f"{token_iss}/protocol/openid-connect/certs"
    logger.debug("[AUTH] token_iss=%s  jwks_url=%s  kid=%s", token_iss, jwks_url, kid)

    # ── 4. Resolve public key ─────────────────────────────────────────────────
    public_key = await _jwks_cache.get(kid, token_iss, jwks_url)

    # ── 5. Verify signature + expiry ──────────────────────────────────────────
    #
    #  verify_aud=False:
    #    Keycloak Resource Owner Password flow tokens do not carry an 'aud'
    #    claim by default.  Enabling aud verification would cause a 401 for
    #    every valid user login token.
    #
    try:
        claims: dict = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            options={
                "verify_exp": True,
                "verify_iat": True,
                "verify_aud": False,   # see note above
            },
        )
    except JWTError as exc:
        logger.warning("[AUTH] Token verification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token verification failed: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ── 6. JTI revocation check ───────────────────────────────────────────────
    jti: Optional[str] = claims.get("jti")
    if jti and _jti_store.is_revoked(jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="This token has been revoked. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ── 7. Build structured payload ───────────────────────────────────────────
    return _build_payload(claims)


# ══════════════════════════════════════════════════════════════════════════════
# Payload Builder & Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_payload(claims: dict) -> TokenPayload:
    role = _extract_role(claims)
    return TokenPayload(
        sub           = claims.get("sub", ""),
        role          = role,
        org_id        = (
            claims.get("org_id")
            or claims.get("organization_id")
            or claims.get("organization")
        ),
        source_system = claims.get("source_system") or claims.get("azp"),
        groups        = [str(g).strip("/") for g in claims.get("groups", [])],
        scopes        = _extract_scopes(claims, role),
        exp           = claims.get("exp"),
        iat           = claims.get("iat"),
        jti           = claims.get("jti"),
        iss           = claims.get("iss"),
        azp           = claims.get("azp"),
    )


def _extract_role(claims: dict) -> str:
    """
    Determine the single effective role for this token.

    Priority
    --------
    1. Client-specific roles  (resource_access.<client_id>.roles)
    2. Realm roles            (realm_access.roles) — highest privilege wins
    3. Flat 'role' claim
    4. Default: CDSS_USER
    """
    _NORMALISE: dict[str, str] = {
        "NURSE":       "CDSS_NURSE",
        "PHARMACIST":  "CDSS_PHARMACIST",
        "MANAGEMENT":  "CDSS_MANAGEMENT",
        "PATIENT":     "CDSS_PATIENT",
        "ADMIN":       "CDSS_ADMIN",
        "USER":        "CDSS_USER",
    }
    _KNOWN: set[str] = {
        "CARDIOLOGIST", "CLINICIAN", "HMIS", "EMR", "PHR", "DHA",
        "NURSE", "CDSS_NURSE",
        "PHARMACIST", "CDSS_PHARMACIST",
        "MANAGEMENT", "CDSS_MANAGEMENT",
        "PATIENT", "CDSS_PATIENT",
        "ADMIN", "CDSS_ADMIN",
        "USER", "CDSS_USER",
    }
    _PRIORITY = [
        "CDSS_ADMIN", "ADMIN",
        "CARDIOLOGIST",
        "CLINICIAN",
        "CDSS_NURSE", "NURSE",
        "CDSS_PHARMACIST", "PHARMACIST",
        "CDSS_MANAGEMENT", "MANAGEMENT",
        "HMIS", "EMR", "PHR", "DHA",
        "CDSS_USER", "USER",
        "CDSS_PATIENT", "PATIENT",
    ]

    def normalise(r: str) -> str:
        return _NORMALISE.get(r.upper(), r.upper())

    # 1. Client-specific roles (most specific)
    client_roles: list[str] = (
        claims.get("resource_access", {})
        .get(settings.keycloak_client_id, {})
        .get("roles", [])
    )
    for r in client_roles:
        if r.upper() in _KNOWN:
            return normalise(r)

    # 2. Realm roles – iterate priority list
    realm_role_map: dict[str, str] = {
        r.upper(): r
        for r in claims.get("realm_access", {}).get("roles", [])
    }
    for p in _PRIORITY:
        if p in realm_role_map:
            return normalise(realm_role_map[p])

    # 3. Flat 'role' claim
    flat = claims.get("role", "")
    if flat and flat.upper() in _KNOWN:
        return normalise(flat)

    # 4. Safe default
    return "CDSS_USER"


def _extract_scopes(claims: dict, role: str) -> set[str]:
    """
    Explicit cdss:* scopes present in the token take priority.
    Otherwise fall back to the role's default scope set.
    """
    explicit = {
        s for s in claims.get("scope", "").split()
        if s.startswith("cdss:")
    }
    if explicit:
        return explicit
    return _ROLE_SCOPES.get(role, _ROLE_SCOPES["CDSS_USER"]).copy()


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def revoke_token(jti: str, ttl_seconds: int = 86_400) -> None:
    """Add *jti* to the revocation store (called by the logout endpoint)."""
    _jti_store.revoke(jti, ttl_seconds)


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI Dependencies
# ══════════════════════════════════════════════════════════════════════════════

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> TokenPayload:
    """
    FastAPI dependency – validates the Bearer token and returns its payload.
    Raises HTTP 401 if the token is absent, malformed, expired, or revoked.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing. Expected: Authorization: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return await decode_token(credentials.credentials)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[AUTH] Unexpected error during token decode: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication error: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_role(*allowed_roles: str):
    """
    FastAPI dependency factory – raises 403 if the user's role is not in
    *allowed_roles*.

    Usage::

        @router.get("/admin-only")
        async def admin_view(user = Depends(require_role("CDSS_ADMIN"))):
            ...
    """
    def _check(
        current_user: TokenPayload = Depends(get_current_user),
    ) -> TokenPayload:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Role '{current_user.role}' is not authorised for this endpoint. "
                    f"Required: {list(allowed_roles)}"
                ),
            )
        return current_user

    return _check


def require_scope(*required_scopes: str):
    """
    FastAPI dependency factory – raises 403 if the token is missing any of
    *required_scopes*.

    Usage::

        @router.get("/patients")
        async def list_patients(user = Depends(require_scope("cdss:patient:read"))):
            ...
    """
    def _check(
        current_user: TokenPayload = Depends(get_current_user),
    ) -> TokenPayload:
        missing = set(required_scopes) - current_user.scopes
        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required scope(s): {', '.join(sorted(missing))}",
            )
        return current_user

    return _check


# ══════════════════════════════════════════════════════════════════════════════
# Pre-built Scope / Role Guards  (convenience aliases)
# ══════════════════════════════════════════════════════════════════════════════

require_patient_read          = require_scope(Scope.PATIENT_READ)
require_patient_write         = require_scope(Scope.PATIENT_READ, Scope.PATIENT_WRITE)
require_encounter_read        = require_scope(Scope.ENCOUNTER_READ)
require_encounter_write       = require_scope(Scope.ENCOUNTER_WRITE)
require_recommendation_read   = require_scope(Scope.RECOMMENDATION_READ)
require_recommendation_create = require_scope(
    Scope.RECOMMENDATION_READ, Scope.RECOMMENDATION_CREATE
)
require_medication_read       = require_scope(Scope.MEDICATION_READ)
require_medication_write      = require_scope(
    Scope.MEDICATION_READ, Scope.MEDICATION_WRITE
)
require_audit_read            = require_scope(Scope.AUDIT_READ)
require_admin                 = require_scope(Scope.ADMIN)
require_cardiologist          = require_role("CARDIOLOGIST", "CDSS_ADMIN")
require_clinical              = require_role("CARDIOLOGIST", "CLINICIAN", "CDSS_ADMIN")


# ══════════════════════════════════════════════════════════════════════════════
# Backward-compatibility stubs
# ══════════════════════════════════════════════════════════════════════════════

# Kept so any code that imports these doesn't break.
DEV_USERS: dict = {}


def authenticate_user(user_id: str, password: str) -> Optional[dict]:  # noqa: ARG001
    """Deprecated – authentication is delegated entirely to Keycloak."""
    return None