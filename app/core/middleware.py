import secrets
import time
from typing import ClassVar

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp

from app.config import get_settings


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that measures request processing time and attaches
    X-Process-Time and X-Response-Time headers to the response.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}s"
        response.headers["X-Response-Time"] = f"{process_time * 1000:.2f}ms"
        return response


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware that enforces API Gateway token authentication on incoming requests.

    All endpoints are protected except:
    - Health endpoints (/health, /api/v1/health, /hello, /api/v1/hello)
    - Interactive API docs (/docs, /redoc, /openapi.json)
    - CORS pre-flight OPTIONS requests
    """

    DEFAULT_EXEMPT_PATHS: ClassVar[set[str]] = {
        "/health",
        "/api/v1/health",
        "/hello",
        "/api/v1/hello",
        "/favicon.ico",
    }

    DOC_PREFIXES: ClassVar[tuple[str, ...]] = (
        "/docs",
        "/redoc",
        "/openapi.json",
    )

    def __init__(
        self,
        app: ASGIApp,
        exempt_paths: set[str] | None = None,
    ) -> None:
        super().__init__(app)
        self.exempt_paths = (
            set(exempt_paths) if exempt_paths is not None else self.DEFAULT_EXEMPT_PATHS
        )

    @classmethod
    def is_path_exempt(cls, path: str) -> bool:
        """Check if a path is exempt from token authentication by default."""
        clean_path = path.rstrip("/") or "/"
        if clean_path in cls.DEFAULT_EXEMPT_PATHS:
            return True
        return clean_path.startswith(cls.DOC_PREFIXES)

    def _is_exempt(self, path: str) -> bool:
        clean_path = path.rstrip("/") or "/"
        if clean_path in self.exempt_paths:
            return True
        return clean_path.startswith(self.DOC_PREFIXES)

    @staticmethod
    def _extract_token(request: Request) -> str | None:
        """Extract auth token from standard headers (x-api-key, gateway-token, Authorization)."""
        # 1. Header: X-API-Key
        if api_key := request.headers.get("x-api-key"):
            return api_key.strip()

        # 2. Header: X-Gateway-Token
        if gw_token := request.headers.get("x-gateway-token"):
            return gw_token.strip()

        # 3. Header: Authorization (Bearer <token> or Token <token>)
        if auth_header := request.headers.get("authorization"):
            parts = auth_header.split(maxsplit=1)
            if len(parts) == 2 and parts[0].lower() in ("bearer", "token"):
                return parts[1].strip()
            return auth_header.strip()

        return None

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Pre-flight OPTIONS requests bypass auth to allow CORS negotiation
        if request.method == "OPTIONS":
            return await call_next(request)

        # Health checks and documentation endpoints bypass auth
        if self._is_exempt(request.url.path):
            return await call_next(request)

        settings = get_settings()
        expected_token = settings.api_gateway_token

        if not expected_token:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "detail": "API gateway authentication token is not configured on server.",
                    "error": "Server misconfiguration",
                },
            )

        provided_token = self._extract_token(request)

        if not provided_token or not secrets.compare_digest(provided_token, expected_token):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "detail": "Invalid or missing API gateway token.",
                    "error": "Unauthorized",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)
