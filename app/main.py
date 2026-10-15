import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.middleware import TimingMiddleware, TokenAuthMiddleware
from app.core.sentry import init_sentry
from app.repositories.firebase_client import firebase_service
from app.views.health import router as health_router
from app.views.resume import router as resume_router

settings = get_settings()

# Configure LangSmith tracing immediately on module load
if settings.langsmith_api_key and settings.langsmith_tracing:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    """Initialize application dependencies and handle graceful shutdown."""
    init_sentry()
    firebase_service.initialize()
    yield


def create_app() -> FastAPI:
    """FastAPI application factory."""
    settings = get_settings()

    app = FastAPI(
        title="CodeSnooper Agents",
        description="Agent workflows platform powered by FastAPI, LangChain, and LangGraph",
        version="0.1.0",
        lifespan=lifespan,
        swagger_ui_parameters={
            "persistAuthorization": True,
            "displayRequestDuration": True,
        },
    )

    app.add_middleware(TokenAuthMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.app_debug else settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time", "X-Response-Time"],
    )

    register_exception_handlers(app)
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(resume_router, prefix="/api/v1")

    def custom_openapi() -> dict[str, Any]:
        """Inject OpenAPI security schemes for Swagger UI Authorize support."""
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )

        openapi_schema.setdefault("components", {})
        openapi_schema["components"].setdefault("securitySchemes", {})
        openapi_schema["components"]["securitySchemes"]["HTTPBearer"] = {
            "type": "http",
            "scheme": "bearer",
            "description": "API Gateway Token (Bearer <token>)",
        }
        openapi_schema["components"]["securitySchemes"]["APIKeyHeader"] = {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API Gateway Token (X-API-Key header)",
        }

        # Apply security requirements to all endpoints except exempt ones (e.g. health)
        for path, path_item in openapi_schema.get("paths", {}).items():
            if not TokenAuthMiddleware.is_path_exempt(path):
                for method in path_item:
                    if method.lower() in ("get", "post", "put", "patch", "delete"):
                        path_item[method].setdefault(
                            "security",
                            [
                                {"HTTPBearer": []},
                                {"APIKeyHeader": []},
                            ],
                        )

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

    return app


app = create_app()
