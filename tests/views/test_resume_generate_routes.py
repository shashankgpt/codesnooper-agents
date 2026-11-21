from collections.abc import Generator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture
def auth_client() -> Generator[TestClient, None, None]:
    """Test client configured with gateway bearer credentials."""
    settings = Settings(
        api_gateway_token="test-secret-token-12345678901234567890",
        m2m_signing_secret="",
    )
    with (
        patch("app.core.middleware.get_settings", return_value=settings),
        patch("app.config.get_settings", return_value=settings),
    ):
        app = create_app()
        with TestClient(app) as test_client:
            yield test_client


def test_resume_generate_unauthenticated_returns_401(auth_client: TestClient) -> None:
    """Verify endpoint rejects unauthenticated requests with 401."""
    resp = auth_client.post(
        "/api/v1/resume/generate",
        json={"uid": "u1", "user_email": "u1@example.com"},
    )
    assert resp.status_code == 401


def test_resume_generate_missing_uid_header_returns_400(auth_client: TestClient) -> None:
    """Verify missing X-User-Id header returns 400 Bad Request."""
    headers = {"Authorization": "Bearer test-secret-token-12345678901234567890"}
    resp = auth_client.post(
        "/api/v1/resume/generate",
        headers=headers,
        json={"uid": "user-123", "user_email": "user@example.com"},
    )
    assert resp.status_code == 400
    assert "Missing X-User-Id header" in resp.json()["detail"]


def test_resume_generate_uid_mismatch_returns_403(auth_client: TestClient) -> None:
    """Verify mismatched X-User-Id header returns 403 Forbidden."""
    headers = {
        "Authorization": "Bearer test-secret-token-12345678901234567890",
        "X-User-Id": "header-user-id",
    }
    resp = auth_client.post(
        "/api/v1/resume/generate",
        headers=headers,
        json={"uid": "body-user-id", "user_email": "user@example.com"},
    )
    assert resp.status_code == 403
    assert "User ID does not match" in resp.json()["detail"]


def test_resume_generate_success_queues_job_and_returns_202(auth_client: TestClient) -> None:
    """Verify valid request queues job and returns 202 Accepted with job_id."""
    headers = {
        "Authorization": "Bearer test-secret-token-12345678901234567890",
        "X-User-Id": "valid-user",
        "Idempotency-Key": "test-key-123",
    }
    with (
        patch(
            "app.views.resume.firebase_service.async_create_or_get_resume_job",
            new_callable=AsyncMock,
        ) as mock_create_job,
        patch("app.views.resume.spawn_resume_parse", new_callable=AsyncMock) as mock_spawn,
    ):
        mock_create_job.return_value = ("job-generated-999", True)

        resp = auth_client.post(
            "/api/v1/resume/generate",
            headers=headers,
            json={
                "uid": "valid-user",
                "user_email": "valid@example.com",
                "template_id": "clean-ats",
                "target_role": "Backend Engineer",
            },
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["job_id"] == "job-generated-999"
        assert data["status"] == "queued"

        mock_create_job.assert_awaited_once()
        job_data = mock_create_job.call_args[0][2]
        assert job_data["uid"] == "valid-user"
        assert job_data["job_type"] == "resume_generate"

        mock_spawn.assert_awaited_once()
        spawn_args = mock_spawn.call_args
        assert spawn_args[0][0] == "job-generated-999"
        assert spawn_args[1]["agent_name"] == "resume_generate"
        assert spawn_args[0][1]["user_email"] == "valid@example.com"


def test_resume_generate_idempotency_key_reuse_returns_existing_job(
    auth_client: TestClient,
) -> None:
    """Verify reusing idempotency key returns existing job without duplicate spawn."""
    headers = {
        "Authorization": "Bearer test-secret-token-12345678901234567890",
        "X-User-Id": "valid-user",
        "Idempotency-Key": "test-key-existing",
    }
    with (
        patch(
            "app.views.resume.firebase_service.async_create_or_get_resume_job",
            new_callable=AsyncMock,
        ) as mock_create_job,
        patch("app.views.resume.spawn_resume_parse", new_callable=AsyncMock) as mock_spawn,
    ):
        mock_create_job.return_value = ("job-existing-111", False)

        resp = auth_client.post(
            "/api/v1/resume/generate",
            headers=headers,
            json={
                "uid": "valid-user",
                "user_email": "valid@example.com",
            },
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["job_id"] == "job-existing-111"
        assert data["status"] == "queued"
        mock_spawn.assert_not_awaited()


def test_resume_generate_invalid_idempotency_key_returns_400(auth_client: TestClient) -> None:
    """Verify invalid whitespace idempotency key returns 400 Bad Request."""
    headers = {
        "Authorization": "Bearer test-secret-token-12345678901234567890",
        "X-User-Id": "valid-user",
        "Idempotency-Key": "key with spaces",
    }
    resp = auth_client.post(
        "/api/v1/resume/generate",
        headers=headers,
        json={"uid": "valid-user", "user_email": "valid@example.com"},
    )
    assert resp.status_code == 400
    assert "Invalid Idempotency-Key" in resp.json()["detail"]
