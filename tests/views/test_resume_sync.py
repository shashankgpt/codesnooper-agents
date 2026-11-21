import hashlib
import hmac
import json
import time
from collections.abc import Generator
from typing import Any
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
    assert mock_merge.await_count == 2
    assert mock_merge.call_args_list[0][0][1]["status"] == "processing"
    assert mock_merge.call_args_list[1][0][1]["status"] == "completed"


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
    assert mock_merge.await_count == 2
    assert mock_merge.call_args_list[0][0][1]["status"] == "processing"
    assert mock_merge.call_args_list[1][0][1]["status"] == "completed"


@pytest.mark.asyncio
async def test_resume_controller_sync_profile_additive_preserves_rich_context_without_erasing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify additive sync prompt includes richness rules and merges bullets without erasing rich profile text."""
    from unittest.mock import AsyncMock, MagicMock

    from app.config import get_settings
    from app.controllers.resume_controller import ResumeController
    from app.models.resume import Experience, ResumeExtractionSchema

    mock_llm = MagicMock()
    mock_structured = AsyncMock()
    mock_llm.with_structured_output.return_value = mock_structured

    async def fake_ainvoke(prompt: list[object]) -> ResumeExtractionSchema:
        system_msg = str(getattr(prompt[0], "content", ""))
        # Verify hardened prompt instructions are sent to LLM
        assert "STRICT NON-DELETION" in system_msg
        assert "NEVER DOWNGRADE RICH CONTEXT" in system_msg
        assert "ENTITY DEDUPLICATION & MATCHING" in system_msg

        # LLM returns combined experience preserving rich context while appending new bullet
        return ResumeExtractionSchema(
            full_name="Alice Smith",
            skills=["Go", "Kafka", "CockroachDB", "Kubernetes", "AWS", "GCP"],
            experience=[
                Experience(
                    company="Acme Corp",
                    title="Lead Systems Architect",
                    start_date="2020",
                    end_date="Present",
                    description=(
                        "Lead Systems Architect responsible for core banking transactional pipeline.\n\n"
                        "Spearheaded migration of legacy monolith to event-driven microservices architecture using Go, Kafka, and CockroachDB.\n\n"
                        "Managed 15 engineers across platform and reliability squads, achieving 99.999% uptime."
                    ),
                    highlights=[
                        "Achieved 99.999% uptime across 4 regions",
                        "Spearheaded monolith to microservices migration",
                        "Reduced batch latency by 45% using streaming pipelines.",
                    ],
                )
            ],
            narrative_context="### Core Architecture\n- Led enterprise data pipeline redesign.",
        )

    mock_structured.ainvoke = fake_ainvoke

    def _mock_llm(_provider: str) -> MagicMock:
        return mock_llm

    monkeypatch.setattr(
        "app.controllers.resume_controller.create_llm_for_provider",
        _mock_llm,
    )

    controller = ResumeController(get_settings())
    rich_current = ResumeSchema(
        full_name="Alice Smith",
        skills=["Go", "Kafka", "CockroachDB", "Kubernetes"],
        experience=[
            Experience(
                company="Acme Corp",
                title="Lead Systems Architect",
                start_date="2020",
                end_date="Present",
                description=(
                    "Lead Systems Architect responsible for core banking transactional pipeline.\n\n"
                    "Spearheaded migration of legacy monolith to event-driven microservices architecture using Go, Kafka, and CockroachDB.\n\n"
                    "Managed 15 engineers across platform and reliability squads, achieving 99.999% uptime."
                ),
                highlights=[
                    "Achieved 99.999% uptime across 4 regions",
                    "Spearheaded monolith to microservices migration",
                ],
            )
        ],
        narrative_context="### Core Architecture\n- Led enterprise data pipeline redesign.",
    )

    condensed_resume_updates: dict[str, object] = {
        "experience": [
            {
                "company": "Acme Corp, Inc.",  # Case/punctuation variation
                "role": "Lead Systems Architect",
                "description": "<ul><li>Reduced batch latency by 45% using streaming pipelines.</li></ul>",
                "techStack": ["Go", "Kafka", "AWS"],
            }
        ],
        "skills": [{"category": "Cloud", "items": ["AWS", "GCP"]}],
    }

    request = ResumeSyncRequest(
        current_profile=rich_current,
        profile_updates=condensed_resume_updates,
        mode="additive",
        source_kind="resume",
    )

    resp = await controller.sync_profile(request)
    assert resp.profile.full_name == "Alice Smith"
    # Deduplicated single company entry
    assert len(resp.profile.experience) == 1
    exp = resp.profile.experience[0]
    assert exp.company == "Acme Corp"
    # Rich description was preserved, NOT overwritten by <ul><li> bullet
    assert "core banking transactional pipeline" in (exp.description or "")
    assert "Spearheaded migration" in (exp.description or "")
    # New bullet was appended to highlights
    assert "Reduced batch latency by 45% using streaming pipelines." in exp.highlights
    assert "Achieved 99.999% uptime across 4 regions" in exp.highlights
    # New skills merged
    assert "AWS" in resp.profile.skills
    assert "GCP" in resp.profile.skills


@pytest.mark.asyncio
async def test_concurrent_resume_sync_jobs_execute_sequentially_without_data_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two concurrent sync jobs for the same user execute in sequence without losing data."""
    import asyncio

    from app.core.resume_jobs import run_resume_sync_job
    from app.models.resume import ResumeCombineResponse

    uid = "user-concurrent-1"
    store: dict[str, dict[str, Any]] = {
        f"profile/{uid}": {
            "uid": uid,
            "full_name": "Alice Smith",
            "skills": ["Python"],
            "experience": [{"company": "Initial Corp", "title": "Dev"}],
        }
    }
    execution_order: list[str] = []

    async def fake_get(path: str) -> dict[str, Any] | None:
        data = store.get(path)
        return dict(data) if data else None

    async def fake_set(path: str, data: dict[str, Any]) -> None:
        store[path] = dict(data)

    async def fake_merge(path: str, data: dict[str, Any]) -> None:
        existing = store.get(path, {})
        store[path] = {**existing, **data}

    monkeypatch.setattr("app.repositories.firebase_client.firebase_service.async_get", fake_get)
    monkeypatch.setattr("app.repositories.firebase_client.firebase_service.async_set", fake_set)
    monkeypatch.setattr("app.repositories.firebase_client.firebase_service.async_merge", fake_merge)

    async def fake_sync_profile(self: Any, req: Any) -> ResumeCombineResponse:
        current = req.current_profile or ResumeSchema()
        updates = req.profile_updates
        job_tag = updates.get("job_tag", "unknown")
        execution_order.append(f"start_{job_tag}")
        # Yield to event loop to simulate work & check serialization
        await asyncio.sleep(0.05)
        new_skills = list(set((current.skills or []) + updates.get("skills", [])))
        new_exp = list(current.experience or [])
        if "new_company" in updates:
            new_exp.append(ResumeSchema.model_validate({"experience": [{"company": updates["new_company"], "title": "Eng"}]}).experience[0])
        execution_order.append(f"end_{job_tag}")
        return ResumeCombineResponse(
            profile=ResumeSchema(
                full_name=current.full_name or "Alice Smith",
                skills=new_skills,
                experience=new_exp,
            ),
            model_used="gemini",
        )

    monkeypatch.setattr(
        "app.controllers.resume_controller.ResumeController.sync_profile", fake_sync_profile
    )

    payload1 = {
        "uid": uid,
        "mode": "additive",
        "source_kind": "resume",
        "profile_updates": {
            "job_tag": "job1",
            "skills": ["Go"],
            "new_company": "Backend Corp",
        },
    }
    payload2 = {
        "uid": uid,
        "mode": "additive",
        "source_kind": "resume",
        "profile_updates": {
            "job_tag": "job2",
            "skills": ["TypeScript"],
            "new_company": "Frontend Corp",
        },
    }

    # Dispatch both jobs concurrently
    task1 = asyncio.create_task(run_resume_sync_job("job-1", payload1))
    task2 = asyncio.create_task(run_resume_sync_job("job-2", payload2))

    res1, res2 = await asyncio.gather(task1, task2)

    assert res1["success"] is True
    assert res2["success"] is True

    # Verification 1: Strict sequential execution (job1 finishes completely before job2 starts)
    assert execution_order == ["start_job1", "end_job1", "start_job2", "end_job2"]

    # Verification 2: Final saved profile in Firestore retains all data from BOTH jobs
    final_profile = store[f"profile/{uid}"]
    skills = final_profile["skills"]
    assert "Python" in skills
    assert "Go" in skills
    assert "TypeScript" in skills

    companies = [e["company"] for e in final_profile["experience"]]
    assert "Initial Corp" in companies
    assert "Backend Corp" in companies
    assert "Frontend Corp" in companies

