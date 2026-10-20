"""Views and API route handlers."""

from app.views.health import router as health_router
from app.views.resume import router as resume_router

__all__ = ["health_router", "resume_router"]
