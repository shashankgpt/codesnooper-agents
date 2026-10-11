from typing import Any, Literal

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

from app.config import get_settings

SentryLogLevel = Literal["fatal", "critical", "error", "warning", "info", "debug"]


def init_sentry() -> None:
    """Initialize Sentry SDK with FastAPI and Starlette integrations.

    Call once at app startup (e.g. in FastAPI lifespan).
    Silently skips if SENTRY_DSN is not configured.
    """
    settings = get_settings()
    if not settings.sentry_dsn:
        return

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.app_env,
        traces_sample_rate=settings.sentry_traces_sample_rate,
        integrations=[
            StarletteIntegration(transaction_style="endpoint"),
            FastApiIntegration(transaction_style="endpoint"),
        ],
        send_default_pii=False,
        release="codesnooper-agents@0.1.0",
    )


def capture_exception(exc: Exception) -> None:
    """Capture an exception and send to Sentry if initialized."""
    sentry_sdk.capture_exception(exc)


def capture_message(message: str, level: SentryLogLevel = "info") -> None:
    """Capture a message event and send to Sentry."""
    sentry_sdk.capture_message(message, level=level)


def set_context(key: str, value: dict[str, Any]) -> None:
    """Set custom context on current Sentry scope."""
    sentry_sdk.set_context(key, value)


def set_tag(key: str, value: str) -> None:
    """Set custom tag on current Sentry scope."""
    sentry_sdk.set_tag(key, value)
