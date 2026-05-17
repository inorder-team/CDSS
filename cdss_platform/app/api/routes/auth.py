"""
CDSS Platform – Auth Routes (Keycloak Proxy)

Implements doc requirements §8 + §9:
  POST /api/v1/auth/sessions  – System-to-system Client Credentials flow (doc §8, §9)
  POST /api/v1/auth/token     – User login via Resource Owner Password flow
  POST /api/v1/auth/refresh   – Refresh token proxy
  POST /api/v1/auth/logout    – JTI revocation + Keycloak end-session
  GET  /api/v1/auth/me        – Current user info decoded from JWT
  GET  /api/v1/auth/jwks      – Proxy Keycloak JWKS public keys
"""
from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.security import (
    TokenPair,
    TokenPayload,
    get_current_user,
    revoke_token,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])
settings = get_settings()


# ─────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class SessionRequest(BaseModel):
    """
    Doc §9: machine-to-machine token request.
    Client sends client_id + client_secret → CDSS proxies to Keycloak
    using Client Credentials grant (not password grant).
    """
    client_id: str
    client_secret: str


class RefreshRequest(BaseModel):
    refresh_token: str


class UserInfo(BaseModel):
    user_id: str
    role: str
    org_id: str | None = None
    source_system: str | None = None
    groups: list[str] = []
    scopes: list[str] = []
    issuer: str | None = None
    client: str | None = None


# ─────────────────────────────────────────────
# Internal Keycloak proxy helper
# ─────────────────────────────────────────────

async def _keycloak_token_request(form_data: dict) -> dict:
    """Forward form_data to Keycloak token endpoint, surface errors cleanly."""
    url = settings.keycloak_token_endpoint
    logger.info("[Auth] Forwarding %s grant to Keycloak", form_data.get("grant_type"))
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, data=form_data)
    except httpx.HTTPError as exc:
        logger.error("[Auth] Keycloak unreachable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable. Please try again later.",
        )

    if resp.status_code == 401:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not resp.is_success:
        ct = resp.headers.get("content-type", "")
        body = resp.json() if "application/json" in ct else {}
        detail = body.get("error_description") or body.get("error") or "Authentication failed"
        raise HTTPException(status_code=resp.status_code, detail=detail)

    return resp.json()


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@router.post(
    "/sessions",
    response_model=TokenPair,
    summary="System Session – Client Credentials Flow (doc §8, §9)",
    description=(
        "Machine-to-machine token endpoint for HMIS, EMR, PHR, DHA systems.\n\n"
        "Accepts `client_id` + `client_secret`, proxies to Keycloak using "
        "**Client Credentials Flow** (`grant_type=client_credentials`), "
        "and returns a signed RS256 access token.\n\n"
        "CDSS does not generate JWT tokens internally."
    ),
)
async def create_session(body: SessionRequest) -> TokenPair:
    data = await _keycloak_token_request({
        "grant_type": "client_credentials",
        "client_id": body.client_id,
        "client_secret": body.client_secret,
        "scope": "openid",
    })
    return TokenPair(
        access_token=data["access_token"],
        refresh_token=data.get("refresh_token", ""),
        token_type=data.get("token_type", "bearer"),
        expires_in=data.get("expires_in", settings.jwt_access_token_expire_minutes * 60),
    )


@router.post(
    "/token",
    response_model=TokenPair,
    summary="User Login – Resource Owner Password Flow (human users)",
    description=(
        "For user-facing login (clinicians, nurses, admins). "
        "Forwards credentials to Keycloak using Resource Owner Password flow.\n\n"
        "For system-to-system (HMIS/EMR/PHR/DHA) use **POST /auth/sessions** instead."
    ),
)
async def login(body: LoginRequest) -> TokenPair:
    data = await _keycloak_token_request({
        "grant_type": "password",
        "client_id": settings.keycloak_client_id,
        "client_secret": settings.keycloak_client_secret,
        "username": body.username,
        "password": body.password,
        "scope": "openid ",
    })
    return TokenPair(
        access_token=data["access_token"],
        refresh_token=data.get("refresh_token", ""),
        token_type=data.get("token_type", "bearer"),
        expires_in=data.get("expires_in", settings.jwt_access_token_expire_minutes * 60),
    )


@router.post(
    "/refresh",
    response_model=TokenPair,
    summary="Refresh Access Token",
)
async def refresh_token(body: RefreshRequest) -> TokenPair:
    data = await _keycloak_token_request({
        "grant_type": "refresh_token",
        "client_id": settings.keycloak_client_id,
        "client_secret": settings.keycloak_client_secret,
        "refresh_token": body.refresh_token,
    })
    return TokenPair(
        access_token=data["access_token"],
        refresh_token=data.get("refresh_token", body.refresh_token),
        token_type=data.get("token_type", "bearer"),
        expires_in=data.get("expires_in", settings.jwt_access_token_expire_minutes * 60),
    )


@router.post("/logout", summary="Logout – Revoke Token + Keycloak end-session")
async def logout(current_user: TokenPayload = Depends(get_current_user)) -> dict:
    """
    1. Adds JTI to Redis revocation store (immediate effect across all instances).
    2. Optionally calls Keycloak end-session endpoint.
    """
    if current_user.jti:
        # TTL = remaining token lifetime; default 24h
        ttl = max((current_user.exp or 0) - int(__import__("time").time()), 60)
        revoke_token(current_user.jti, ttl_seconds=ttl)
        logger.info("[Auth] Revoked JTI %s for %s", current_user.jti, current_user.sub)

    return {
        "message": "Logged out successfully",
        "user_id": current_user.sub,
        "note": (
            f"For full session termination also call: "
            f"GET {settings.keycloak_end_session_endpoint}"
        ),
    }


@router.get("/me", response_model=UserInfo, summary="Current User Info")
async def me(current_user: TokenPayload = Depends(get_current_user)) -> UserInfo:
    """Return decoded identity from the validated JWT (no DB lookup)."""
    return UserInfo(
        user_id=current_user.sub,
        role=current_user.role,
        org_id=current_user.org_id,
        source_system=current_user.source_system,
        groups=current_user.groups,
        scopes=sorted(current_user.scopes),
        issuer=current_user.iss,
        client=current_user.azp,
    )


@router.get("/jwks", summary="Keycloak JWKS Public Keys")
async def jwks() -> dict:
    """
    Proxy Keycloak JWKS so HMIS/EMR/PHR/DHA clients can verify tokens
    without direct Keycloak network access.
    """
    url = settings.jwks_uri
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        logger.error("[Auth] Failed to fetch JWKS: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to fetch public keys from Keycloak",
        )
