import asyncio
import logging
from functools import partial

from google import genai
from langsmith import Client

from app.config import Settings
from app.models.common import HealthResponse, HelloResponse
from app.repositories.firebase_client import firebase_service

logger = logging.getLogger(__name__)


class HealthController:
    """Controller for health status and greetings."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def get_hello(self) -> HelloResponse:
        """Construct welcome payload."""
        return HelloResponse(
            message="Hello from CodeSnooper Agents! 🚀",
            environment=self._settings.app_env,
            version="0.1.0",
        )

    def _check_gemini(self) -> bool:
        """Verify Gemini API connectivity and model availability."""
        if not self._settings.google_api_key:
            return False
        try:
            client = genai.Client(api_key=self._settings.google_api_key)
            client.models.get(model=self._settings.gemini_model)
            return True
        except Exception as exc:
            logger.warning("Gemini health check failed: %s", exc)
            return False

    def _check_langsmith(self) -> bool:
        """Verify LangSmith credentials and API connectivity without creating a run."""
        if not self._settings.langsmith_api_key or not self._settings.langsmith_tracing:
            return False

        client = None
        try:
            client = Client(
                api_key=self._settings.langsmith_api_key,
                api_url=self._settings.langsmith_endpoint,
            )
            # Consume one result so the authenticated request is actually sent.
            next(iter(client.list_projects(limit=1)), None)
            return True
        except Exception as exc:
            logger.warning("LangSmith connectivity check failed: %s", exc)
            return False
        finally:
            if client is not None:
                try:
                    client.close(timeout=10)
                except Exception as exc:
                    logger.warning("LangSmith client cleanup failed: %s", exc)

    async def get_health(self) -> HealthResponse:
        """Assess overall health including Firebase, Gemini, and LangSmith connectivity."""
        loop = asyncio.get_running_loop()
        firebase_task = loop.run_in_executor(None, partial(firebase_service.check_connection))
        gemini_task = loop.run_in_executor(None, partial(self._check_gemini))
        langsmith_task = loop.run_in_executor(None, partial(self._check_langsmith))

        firebase_ok, gemini_ok, langsmith_ok = await asyncio.gather(
            firebase_task, gemini_task, langsmith_task
        )
        overall_ok = firebase_ok and gemini_ok

        return HealthResponse(
            status="healthy" if overall_ok else "degraded",
            firebase_connected=firebase_ok,
            gemini_connected=gemini_ok,
            langsmith_connected=langsmith_ok,
            gemini_model=self._settings.gemini_model if gemini_ok else None,
        )
