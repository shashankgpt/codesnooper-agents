from collections.abc import Generator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models.portfolio import PortfolioEnhanceResponse


@pytest.fixture
def auth_client() -> Generator[TestClient, None, None]:
    """Test client configured with valid gateway credentials."""
    settings = Settings(
        api_gateway_token="test-secret-token-12345678901234567890",
        m2m_signing_secret="",
    )
    with (
        patch("app.core.middleware.get_settings", return_value=settings),
        patch("app.config.get_settings", return_value=settings),
        patch("app.views.portfolio.get_settings", return_value=settings),
    ):
        app = create_app()
        with TestClient(app) as test_client:
            yield test_client


def test_portfolio_routes_require_authentication(auth_client: TestClient) -> None:
    """Verify portfolio endpoints reject unauthenticated requests with 401."""
    resp = auth_client.post("/api/v1/portfolio/generate", json={"uid": "u1"})
    assert resp.status_code == 401

    resp_enh = auth_client.post(
        "/api/v1/portfolio/enhance",
        json={"uid": "u1", "section": "bio", "current_text": "hello"},
    )
    assert resp_enh.status_code == 401


def test_portfolio_generate_requires_matching_user_id(auth_client: TestClient) -> None:
    """Verify mismatched X-User-Id header returns 403 Forbidden."""
    headers = {
        "Authorization": "Bearer test-secret-token-12345678901234567890",
        "X-User-Id": "caller-uid",
    }
    resp = auth_client.post(
        "/api/v1/portfolio/generate",
        headers=headers,
        json={"uid": "different-target-uid"},
    )
    assert resp.status_code == 403
    assert "User ID does not match" in resp.json()["detail"]


def test_portfolio_generate_rejects_missing_profile(auth_client: TestClient) -> None:
    """Verify 400 Bad Request if user has no career profile."""
    headers = {
        "Authorization": "Bearer test-secret-token-12345678901234567890",
        "X-User-Id": "user-no-profile",
    }
    with patch(
        "app.views.portfolio.firebase_service.async_get",
        new_callable=AsyncMock,
    ) as mock_get:
        mock_get.return_value = None

        resp = auth_client.post(
            "/api/v1/portfolio/generate",
            headers=headers,
            json={"uid": "user-no-profile"},
        )
        assert resp.status_code == 400
        assert "No career profile found" in resp.json()["detail"]


def test_portfolio_generate_success_queues_job(auth_client: TestClient) -> None:
    """Verify valid request queues job and returns 202 Accepted with job_id."""
    headers = {
        "Authorization": "Bearer test-secret-token-12345678901234567890",
        "X-User-Id": "valid-user",
        "Idempotency-Key": "test-key-123",
    }
    with (
        patch(
            "app.views.portfolio.firebase_service.async_get",
            new_callable=AsyncMock,
        ) as mock_get,
        patch(
            "app.views.portfolio.firebase_service.async_create_or_get_resume_job",
            new_callable=AsyncMock,
        ) as mock_create_job,
        patch("app.views.portfolio.spawn_resume_parse", new_callable=AsyncMock) as mock_spawn,
    ):
        mock_get.return_value = {"full_name": "Valid Candidate"}
        mock_create_job.return_value = ("job-generated-999", True)

        resp = auth_client.post(
            "/api/v1/portfolio/generate",
            headers=headers,
            json={"uid": "valid-user"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["job_id"] == "job-generated-999"
        assert data["status"] == "queued"
        mock_spawn.assert_called_once()


def test_portfolio_enhance_success(auth_client: TestClient) -> None:
    """Verify enhance endpoint returns enhanced text and bullets."""
    headers = {
        "Authorization": "Bearer test-secret-token-12345678901234567890",
        "X-User-Id": "valid-user",
    }
    mock_response = PortfolioEnhanceResponse(
        success=True,
        section="bio",
        enhanced_text="Refined developer bio.",
        bullets=[],
    )

    with patch(
        "app.controllers.portfolio_controller.PortfolioController.enhance_section",
        new_callable=AsyncMock,
    ) as mock_controller_enh:
        mock_controller_enh.return_value = mock_response

        resp = auth_client.post(
            "/api/v1/portfolio/enhance",
            headers=headers,
            json={
                "uid": "valid-user",
                "section": "bio",
                "current_text": "Draft bio",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["enhanced_text"] == "Refined developer bio."


def test_portfolio_job_status_and_delete(auth_client: TestClient) -> None:
    """Verify job polling and deletion with ownership checks."""
    headers = {
        "Authorization": "Bearer test-secret-token-12345678901234567890",
        "X-User-Id": "job-owner",
    }
    job_record = {
        "uid": "job-owner",
        "status": "completed",
        "result": {"success": True, "portfolio_id": "port-1"},
    }

    with (
        patch(
            "app.views.portfolio.firebase_service.async_get",
            new_callable=AsyncMock,
        ) as mock_get,
        patch(
            "app.views.portfolio.firebase_service.async_delete",
            new_callable=AsyncMock,
        ) as mock_del,
    ):
        mock_get.return_value = job_record

        # 1. Poll status
        resp = auth_client.get("/api/v1/portfolio/jobs/job-123", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

        # 2. Delete job
        del_resp = auth_client.delete("/api/v1/portfolio/jobs/job-123", headers=headers)
        assert del_resp.status_code == 204
        mock_del.assert_called_once_with("resume_parse_jobs/job-123")
