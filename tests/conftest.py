from collections.abc import Generator
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(autouse=True)
def _mock_health_external_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock external network calls during health checks to keep tests offline and fast."""
    def _mock_true(*_args: object, **_kwargs: object) -> bool:
        return True

    monkeypatch.setattr(
        "app.controllers.health_controller.HealthController._check_gemini",
        _mock_true,
    )
    monkeypatch.setattr(
        "app.controllers.health_controller.HealthController._check_langsmith",
        _mock_true,
    )
    monkeypatch.setattr(
        "app.repositories.firebase_client.firebase_service.check_connection",
        _mock_true,
    )
    monkeypatch.setattr(
        "app.views.resume.firebase_service.async_create_or_get_resume_job",
        AsyncMock(return_value=("mock-job-id", True)),
    )
    monkeypatch.setattr(
        "app.views.resume.spawn_resume_parse",
        AsyncMock(),
    )


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Test client fixture for invoking endpoints."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
