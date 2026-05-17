"""
CDSS Platform – Middleware Stack (doc §13, §14)

Order of execution per doc §14 (predicate-based routing):
  1. CorrelationIDMiddleware  – tag every request with a unique ID
  2. TimingMiddleware          – measure latency
  3. GatewayPredicateMiddleware – doc §14: token validation order enforcement
  4. RateLimitMiddleware       – doc §13: per client_id / org_id / route / IP
  5. AccessLogMiddleware       – structured audit-friendly access log
"""
from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import get_settings

settings = get_settings()


# ─────────────────────────────────────────────
# 1. Correlation ID
# ─────────────────────────────────────────────

class CorrelationIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response


# ─────────────────────────────────────────────
# 2. Timing
# ─────────────────────────────────────────────

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response


# ─────────────────────────────────────────────
# 3. Gateway Predicate Middleware (doc §14)
# Enforces strict order: token validation → authz → rate-limit → route
# Adds parsed identity to request.state so downstream middleware can use it.
# ─────────────────────────────────────────────

_PUBLIC_PATHS = {
    "/api/v1/auth/token",
    "/api/v1/auth/sessions",
    "/api/v1/auth/refresh",
    "/api/v1/auth/jwks",
    "/api/v1/health",
    "/api/v1/health/rag",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/",
}

class GatewayPredicateMiddleware(BaseHTTPMiddleware):
    """
    Doc §14: gate that enforces routing predicates.

    For protected paths:
      1. Validates Bearer token (calls decode_token).
      2. Attaches TokenPayload to request.state.user.
      3. Stores client_id and org_id for rate-limit keying.

    Routing itself is handled by FastAPI routers – this middleware ensures
    auth always precedes rate-limit and route dispatch.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # Skip auth for public paths
        if path in _PUBLIC_PATHS or path.startswith("/static"):
            return await call_next(request)

        # Extract and validate token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized", "detail": "Authorization header missing"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth_header[len("Bearer "):]
        try:
            from app.core.security import decode_token
            user = decode_token(token)
            request.state.user = user
            request.state.client_id = user.azp or "unknown"
            request.state.org_id = user.org_id or "unknown"
        except Exception as exc:
            status_code = getattr(exc, "status_code", 401)
            detail = getattr(exc, "detail", "Authentication failed")
            return JSONResponse(
                status_code=status_code,
                content={"error": "Unauthorized", "detail": detail},
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)


# ─────────────────────────────────────────────
# 4. Rate Limiter (doc §13)
# Keyed by client_id + org_id + route prefix + IP
# ─────────────────────────────────────────────

class _RateLimitStore:
    """Sliding window rate limit store."""
    def __init__(self):
        self._windows: dict[str, deque] = defaultdict(deque)

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> bool:
        now = time.time()
        window = self._windows[key]
        while window and window[0] < now - window_seconds:
            window.popleft()
        if len(window) >= max_requests:
            return False
        window.append(now)
        return True


_rate_store = _RateLimitStore()

# Doc §13 rate limits: (path_prefix, max_req, window_seconds)
RATE_LIMIT_RULES: list[tuple[str, int, int]] = [
    ("/api/v1/auth/sessions",      20,  60),   # doc §13: 20/min for /sessions
    ("/api/v1/auth/token",         20,  60),   # same limit for password flow
    ("/api/v1/admin",              50,  60),   # doc §13: 50/min admin APIs
    ("/api/v1/recommendations",   100,  60),   # doc §13: 100/min clinical APIs
    ("/api/v1/patients",          300,  60),   # doc §13: 300/min patient read
    ("/api/v1/",                  200,  60),   # general fallback
]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Doc §13: rate limiting per client_id + org_id + route + IP.
    Uses identity attached by GatewayPredicateMiddleware when available.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not settings.rate_limit_enabled:
            return await call_next(request)

        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"

        # Use identity claims if available (set by GatewayPredicateMiddleware)
        client_id = getattr(request.state, "client_id", client_ip)
        org_id = getattr(request.state, "org_id", "unknown")

        for prefix, max_req, window in RATE_LIMIT_RULES:
            if path.startswith(prefix):
                # Key = client_id:org_id:route_prefix (doc §13)
                key = f"{client_id}:{org_id}:{prefix}"
                if not _rate_store.is_allowed(key, max_req, window):
                    logger.warning(
                        "[RATE_LIMIT] Blocked client=%s org=%s ip=%s on %s",
                        client_id, org_id, client_ip, prefix,
                    )
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "Rate limit exceeded",
                            "detail": f"Max {max_req} requests per {window}s for this route",
                            "client_id": client_id,
                        },
                        headers={"Retry-After": str(window)},
                    )
                break

        return await call_next(request)


# ─────────────────────────────────────────────
# 5. Access Log (doc §15)
# Logs: timestamp, request_id, client_id, subject, org_id,
#       source_system, path, method, authz decision, status, latency
# Does NOT log: tokens, secrets, PHI payloads
# ─────────────────────────────────────────────

class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        corr = getattr(request.state, "correlation_id", "-")
        user = getattr(request.state, "user", None)

        logger.info(
            "{method} {path} → {status} ({ms:.0f}ms) "
            "corr={corr} client={client} sub={sub} org={org} src={src}",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            ms=elapsed_ms,
            corr=corr[:8] if corr else "-",
            client=getattr(user, "azp", "-") or "-",
            sub=getattr(user, "sub", "-") or "-",
            org=getattr(user, "org_id", "-") or "-",
            src=getattr(user, "source_system", "-") or "-",
        )
        return response
