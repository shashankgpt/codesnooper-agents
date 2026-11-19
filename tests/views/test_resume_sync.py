import hashlib
import hmac
import json
import time
from collections.abc import Generator
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models.resume import (
    ResumeSchema,
    ResumeSyncRequest,
)

SIGNING_SECRET = "super-secret-hmac-key-at-least-32-chars-long"


def hmac_headers(
    secret: str,
    method: str,
    path: str,
    body: bytes = b"",
    user_id: str = "uid-alice",
    ts_offset: int = 0,
) -> dict[str, str]:
    ts = int(time.time()) + ts_offset
    prefix = f"{ts}.{method.upper()}.{path}.".encode()
    signed_payload = prefix + body
    sig = hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()
    return {
        "X-Timestamp": str(ts),
        "X-Signature": sig,
        "X-User-Id": user_id,
        "Content-Type": "application/json",
    }


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    settings = Settings(m2m_signing_secret=SIGNING_SECRET)
    monkeypatch.setattr("app.core.middleware.get_settings", lambda: settings)
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_resume_sync_uid_mismatch_returns_403(client: TestClient) -> None:
    """X-User-Id header must match request.uid."""
    path = "/api/v1/resume/sync"
    body = json.dumps(
        {
            "uid": "user-123",
            "mode": "authoritative",
            "profile_updates": {"full_name": "Alice Smith"},
        }
    ).encode()
    headers = hmac_headers(SIGNING_SECRET, "POST", path, body, user_id="user-different")

    resp = client.post(path, content=body, headers=headers)
    assert resp.status_code == 403
    assert "User ID does not match" in resp.json()["detail"]


def test_resume_sync_success_enqueues_job(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /resume/sync accepts valid request and returns 202 queued job."""
    path = "/api/v1/resume/sync"
    body = json.dumps(
        {
            "uid": "user-123",
            "mode": "authoritative",
            "profile_updates": {
                "full_name": "Alice Smith",
                "summary": "Experienced engineer",
                "narrative_context": "Winner of Hackathon",
            },
        }
    ).encode()
    headers = hmac_headers(SIGNING_SECRET, "POST", path, body, user_id="user-123")

    mock_spawn = AsyncMock()
    mock_create_or_get = AsyncMock(return_value=("sync-job-999", True))

    monkeypatch.setattr("app.views.resume.spawn_resume_parse", mock_spawn)
    monkeypatch.setattr(
        "app.views.resume.firebase_service.async_create_or_get_resume_job",
        mock_create_or_get,
    )

    resp = client.post(path, content=body, headers=headers)
    assert resp.status_code == 202
    data = resp.json()
    assert data["job_id"] == "sync-job-999"
    assert data["status"] == "queued"
    mock_spawn.assert_awaited_once()
    assert mock_spawn.call_args[1]["agent_name"] == "resume_sync"


@pytest.mark.asyncio
async def test_run_resume_sync_job_authoritative(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_resume_sync_job in authoritative mode updates profile and completes job."""
    from app.core.resume_jobs import run_resume_sync_job

    mock_set = AsyncMock()
    mock_merge = AsyncMock()
    mock_get = AsyncMock(return_value={"full_name": "Alice Smith", "lastImportJobId": "import-abc"})

    monkeypatch.setattr("app.repositories.firebase_client.firebase_service.async_set", mock_set)
    monkeypatch.setattr("app.repositories.firebase_client.firebase_service.async_merge", mock_merge)
    monkeypatch.setattr("app.repositories.firebase_client.firebase_service.async_get", mock_get)

    payload = {
        "uid": "user-123",
        "mode": "authoritative",
        "profile_updates": {
            "full_name": "Alice Smith",
            "summary": "Lead Architect",
            "narrative_context": None,
            "skills": ["Python", "Modal"],
        },
    }

    result = await run_resume_sync_job("sync-job-1", payload)
    assert result["success"] is True
    assert result["mode"] == "authoritative"
    assert result["model_used"] == "deterministic"
    assert result["resume"]["full_name"] == "Alice Smith"

    # Verify Firebase calls
    mock_set.assert_awaited_once()
    saved_doc = mock_set.call_args[0][1]
    assert saved_doc["full_name"] == "Alice Smith"
    # Preserves lastImportJobId without clobbering
    assert saved_doc["lastImportJobId"] == "import-abc"
    mock_merge.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_resume_sync_job_additive_resume_builder_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_resume_sync_job handles Resume Builder payload with skills, experience, etc."""
    from app.core.resume_jobs import run_resume_sync_job
    from app.models.resume import ResumeCombineResponse

    mock_set = AsyncMock()
    mock_merge = AsyncMock()
    mock_get = AsyncMock(
        return_value={
            "full_name": "Alice Smith",
            "summary": "Existing summary",
            "experience": [
                {
                    "company": "Prior Corp",
                    "title": "Software Engineer",
                    "start_date": "2018",
                    "end_date": "2020",
                }
            ],
            "skills": ["Java"],
        }
    )

    monkeypatch.setattr("app.repositories.firebase_client.firebase_service.async_set", mock_set)
    monkeypatch.setattr("app.repositories.firebase_client.firebase_service.async_merge", mock_merge)
    monkeypatch.setattr("app.repositories.firebase_client.firebase_service.async_get", mock_get)

    # Mock controller.sync_profile to verify payload passes ResumeSyncJobRequest validation
    mock_sync = AsyncMock(
        return_value=ResumeCombineResponse(
            profile=ResumeSchema(
                full_name="Alice Smith",
                skills=["Java", "Python", "Go"],
                narrative_context="Custom Sections: Patents",
            ),
            model_used="gemini",
        )
    )
    monkeypatch.setattr(
        "app.controllers.resume_controller.ResumeController.sync_profile", mock_sync
    )

    payload = {
        "uid": "user-123",
        "mode": "additive",
        "source_kind": "resume",
        "profile_updates": {
            "personalInfo": {"fullName": "Alice Smith"},
            "skills": [{"category": "Languages", "items": ["Python", "Go"]}],
            "experience": [{"company": "New Corp", "role": "Senior Engineer", "startDate": "2021"}],
            "customSections": [{"title": "Patents", "content": "US Patent 12345"}],
        },
    }

    result = await run_resume_sync_job("sync-job-2", payload)
    assert result["success"] is True
    assert result["mode"] == "additive"
    assert result["source_kind"] == "resume"
    mock_sync.assert_awaited_once()
    called_request = mock_sync.call_args[0][0]
    assert called_request.mode == "additive"
    assert called_request.source_kind == "resume"
    assert "personalInfo" in called_request.profile_updates
    mock_set.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_resume_sync_job_additive_portfolio_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_resume_sync_job handles Portfolio payload with timelineEvents, achievements, etc."""
    from app.core.resume_jobs import run_resume_sync_job
    from app.models.resume import ResumeCombineResponse

    mock_set = AsyncMock()
    mock_merge = AsyncMock()
    mock_get = AsyncMock(return_value={"full_name": "Alice Smith", "skills": ["Python"]})

    monkeypatch.setattr("app.repositories.firebase_client.firebase_service.async_set", mock_set)
    monkeypatch.setattr("app.repositories.firebase_client.firebase_service.async_merge", mock_merge)
    monkeypatch.setattr("app.repositories.firebase_client.firebase_service.async_get", mock_get)

    mock_sync = AsyncMock(
        return_value=ResumeCombineResponse(
            profile=ResumeSchema(
                full_name="Alice Smith",
                skills=["Python", "TypeScript"],
                narrative_context="Portfolio Subheading: Tech Lead",
            ),
            model_used="gemini",
        )
    )
    monkeypatch.setattr(
        "app.controllers.resume_controller.ResumeController.sync_profile", mock_sync
    )

    payload = {
        "uid": "user-123",
        "mode": "additive",
        "source_kind": "portfolio",
        "profile_updates": {
            "name": "Alice Smith",
            "subHeading": "Tech Lead",
            "skills": [{"skillName": "TypeScript", "yearsOfExperience": 5}],
            "achievements": [{"achievementKey": "Stars", "achievementValue": "5000"}],
            "timelineEvents": [{"heading": "Launched Product", "eventStartDate": "2023"}],
        },
    }

    result = await run_resume_sync_job("sync-job-3", payload)
    assert result["success"] is True
    assert result["mode"] == "additive"
    assert result["source_kind"] == "portfolio"
    mock_sync.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_controller_sync_profile_residual_placeholder_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """sync_profile raises ValueError if residual PII placeholders are detected."""
    from app.config import get_settings
    from app.controllers.resume_controller import ResumeController

    controller = ResumeController(get_settings())
    request = ResumeSyncRequest(
        current_profile=None,
        profile_updates={
            "full_name": "{{PII_NAME_1}}",
            "summary": "Test summary",
        },
        mode="authoritative",
    )

    with pytest.raises(ValueError, match="Residual placeholders detected"):
        await controller.sync_profile(request)


@pytest.mark.asyncio
async def test_resume_controller_sync_profile_authoritative_reconcile_with_pii(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Directly test ResumeController.sync_profile authoritative reconciliation with PII."""
    from unittest.mock import AsyncMock, MagicMock

    from app.config import get_settings
    from app.controllers.resume_controller import ResumeController
    from app.models.resume import ReconciledNarrativeOutput

    mock_llm = MagicMock()
    mock_structured = AsyncMock()
    mock_llm.with_structured_output.return_value = mock_structured

    async def fake_ainvoke(prompt: list[object]) -> ReconciledNarrativeOutput:
        human_msg = str(getattr(prompt[1], "content", ""))
        assert "{{PII_NAME_1}}" in human_msg
        assert "Alice Smith" not in human_msg
        return ReconciledNarrativeOutput(
            narrative_context="### Highlights\n- {{PII_NAME_1}} led core platform architecture."
        )

    mock_structured.ainvoke = fake_ainvoke

    def _mock_llm_authoritative(_provider: str) -> MagicMock:
        return mock_llm

    monkeypatch.setattr(
        "app.controllers.resume_controller.create_llm_for_provider",
        _mock_llm_authoritative,
    )

    controller = ResumeController(get_settings())
    request = ResumeSyncRequest(
        current_profile=None,
        profile_updates={
            "full_name": "Alice Smith",
            "summary": "Engineering Director",
            "narrative_context": "### Highlights\n- Alice Smith led core platform architecture.",
            "skills": ["Python", "System Design"],
        },
        mode="authoritative",
    )

    resp = await controller.sync_profile(request)
    assert resp.profile.full_name == "Alice Smith"
    assert resp.profile.summary == "Engineering Director"
    assert resp.profile.skills == ["Python", "System Design"]
    assert resp.profile.narrative_context is not None
    assert "Alice Smith led core platform architecture." in resp.profile.narrative_context
    assert "{{PII_NAME_1}}" not in resp.profile.narrative_context
    assert resp.profile.raw_text is not None
    assert "## Additional Context & Highlights" in resp.profile.raw_text
    assert "Alice Smith led core platform architecture." in resp.profile.raw_text


@pytest.mark.asyncio
async def test_resume_controller_sync_profile_additive_builder_with_pii(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Directly test ResumeController.sync_profile additive mode with Resume Builder PII payload."""
    from unittest.mock import AsyncMock, MagicMock

    from app.config import get_settings
    from app.controllers.resume_controller import ResumeController
    from app.models.resume import ResumeExtractionSchema

    mock_llm = MagicMock()
    mock_structured = AsyncMock()
    mock_llm.with_structured_output.return_value = mock_structured

    async def fake_ainvoke(prompt: list[object]) -> ResumeExtractionSchema:
        human_msg = str(getattr(prompt[1], "content", ""))
        assert "{{PII_NAME_1}}" in human_msg
        assert "Alice Smith" not in human_msg
        return ResumeExtractionSchema(
            full_name="{{PII_NAME_1}}",
            skills=["Python", "Go", "Kubernetes"],
            narrative_context="### Patents\n- {{PII_NAME_1}} holds US Patent 98765.",
        )

    mock_structured.ainvoke = fake_ainvoke

    def _mock_llm_authoritative(_provider: str) -> MagicMock:
        return mock_llm

    monkeypatch.setattr(
        "app.controllers.resume_controller.create_llm_for_provider",
        _mock_llm_authoritative,
    )

    controller = ResumeController(get_settings())
    request = ResumeSyncRequest(
        current_profile=ResumeSchema(full_name="Alice Smith", skills=["Python"]),
        profile_updates={
            "personalInfo": {"fullName": "Alice Smith"},
            "skills": [{"category": "Cloud", "items": ["Kubernetes"]}],
            "customSections": [
                {"title": "Patents", "content": "Alice Smith holds US Patent 98765."}
            ],
        },
        mode="additive",
        source_kind="resume",
    )

    resp = await controller.sync_profile(request)
    assert resp.profile.full_name == "Alice Smith"
    assert resp.profile.skills == ["Python", "Go", "Kubernetes"]
    assert resp.profile.narrative_context is not None
    assert "Alice Smith holds US Patent 98765." in resp.profile.narrative_context
    assert "{{PII_NAME_1}}" not in resp.profile.narrative_context
    assert resp.profile.raw_text is not None
    assert "## Additional Context & Highlights" in resp.profile.raw_text


@pytest.mark.asyncio
async def test_run_resume_sync_job_authoritative_skips_when_profile_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_resume_sync_job authoritative mode marks job skipped if profile does not exist."""
    from app.core.resume_jobs import run_resume_sync_job

    mock_set = AsyncMock()
    mock_merge = AsyncMock()
    mock_get = AsyncMock(return_value=None)

    monkeypatch.setattr("app.repositories.firebase_client.firebase_service.async_set", mock_set)
    monkeypatch.setattr("app.repositories.firebase_client.firebase_service.async_merge", mock_merge)
    monkeypatch.setattr("app.repositories.firebase_client.firebase_service.async_get", mock_get)

    payload = {
        "uid": "user-deleted",
        "mode": "authoritative",
        "profile_updates": {"full_name": "Ghost User"},
    }

    result = await run_resume_sync_job("sync-job-deleted", payload)
    assert result.get("skipped") == "profile_not_found"
    mock_set.assert_not_awaited()
    mock_merge.assert_awaited_once()
    assert mock_merge.call_args[0][1]["status"] == "completed"
