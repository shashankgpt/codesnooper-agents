from collections.abc import Iterator
from copy import deepcopy
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from firebase_admin import auth

from app.config import Settings
from app.controllers.resume_controller import ResumeController
from app.core.middleware import TokenAuthMiddleware
from app.core.resume_jobs import (
    _run_local_resume_parse,  # pyright: ignore[reportPrivateUsage]
    persist_profile_and_complete_job,
    run_resume_combine_job,
)
from app.models.resume import ResumeCombineResponse, ResumeSchema
from app.repositories.firebase_client import firebase_service
from app.views.resume import router

HEADERS = {"X-API-Key": "test-gateway", "Authorization": "Bearer owner-token"}


@pytest.fixture
def database(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}

    def get_document(path: str) -> Any:
        return deepcopy(documents.get(path))

    def set_document(path: str, data: dict[str, Any]) -> None:
        documents[path] = deepcopy(data)

    def merge_document(path: str, data: dict[str, Any]) -> None:
        documents.setdefault(path, {}).update(deepcopy(data))

    def delete_document(path: str) -> None:
        documents.pop(path, None)

    monkeypatch.setattr(firebase_service, "async_get", AsyncMock(side_effect=get_document))
    monkeypatch.setattr(
        firebase_service,
        "async_set",
        AsyncMock(side_effect=set_document),
    )
    monkeypatch.setattr(
        firebase_service,
        "async_merge",
        AsyncMock(side_effect=merge_document),
    )
    monkeypatch.setattr(firebase_service, "async_delete", AsyncMock(side_effect=delete_document))
    return documents


@pytest.fixture
def verifier(monkeypatch: pytest.MonkeyPatch) -> Mock:
    verify = Mock(return_value={"uid": "owner"})
    monkeypatch.setattr(auth, "verify_id_token", verify)
    return verify


@pytest.fixture
def resume_client(
    monkeypatch: pytest.MonkeyPatch, database: dict[str, Any], verifier: Mock
) -> Iterator[TestClient]:
    monkeypatch.setattr(
        "app.core.middleware.get_settings", lambda: Settings(api_gateway_token="test-gateway")
    )
    app = FastAPI()
    app.add_middleware(TokenAuthMiddleware)
    app.include_router(router, prefix="/api/v1")
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize(
    "method, path",
    [("post", "/parse"), ("post", "/combine"), ("get", "/jobs/job"), ("delete", "/jobs/job")],
)
def test_gateway_token_alone_cannot_access_resume_resources(
    resume_client: TestClient, method: str, path: str
) -> None:
    response = resume_client.request(
        method, f"/api/v1/resume{path}", headers={"X-API-Key": "test-gateway"}
    )
    assert response.status_code == 401


@pytest.mark.parametrize("path", ["parse", "combine"])
def test_uid_mismatch_is_rejected_before_user_data_access(
    resume_client: TestClient, path: str
) -> None:
    response = resume_client.post(
        f"/api/v1/resume/{path}",
        headers=HEADERS,
        json={"uid": "victim", "blob_id": "career-profiles/victim/resume.pdf"},
    )
    assert response.status_code == 403
    cast("AsyncMock", firebase_service.async_get).assert_not_awaited()
    cast("AsyncMock", firebase_service.async_set).assert_not_awaited()


@pytest.mark.parametrize(
    "failure",
    [
        auth.InvalidIdTokenError("invalid"),
        auth.ExpiredIdTokenError("expired", None),
        auth.RevokedIdTokenError("revoked"),
        auth.UserDisabledError("disabled"),
        auth.UserNotFoundError("deleted"),
    ],
)
def test_invalid_user_authentication_is_rejected(
    resume_client: TestClient, verifier: Mock, failure: Exception
) -> None:
    verifier.side_effect = failure
    response = resume_client.get("/api/v1/resume/jobs/job", headers=HEADERS)
    assert response.status_code == 401
    cast("AsyncMock", firebase_service.async_get).assert_not_awaited()


@pytest.mark.parametrize("method", ["get", "delete"])
@pytest.mark.parametrize("owner", ["owner", "other-user", None])
def test_job_results_are_accessible_only_to_verified_owner(
    resume_client: TestClient, database: dict[str, Any], method: str, owner: str | None
) -> None:
    database["resume_parse_jobs/job"] = {"uid": owner, "status": "queued"}
    response = resume_client.request(method, "/api/v1/resume/jobs/job", headers=HEADERS)
    assert response.status_code == (
        {"get": 200, "delete": 204}[method] if owner == "owner" else 404
    )
    if owner != "owner":
        cast("AsyncMock", firebase_service.async_delete).assert_not_awaited()


def test_parse_queues_verified_owner_and_blob_path_without_bytes(
    resume_client: TestClient, database: dict[str, Any], verifier: Mock
) -> None:
    with (
        patch(
            "app.views.resume.fetch_resume_blob", AsyncMock(return_value=(b"PDF", "pdf"))
        ) as blob,
        patch("app.views.resume.spawn_resume_parse", AsyncMock()) as spawn,
    ):
        response = resume_client.post(
            "/api/v1/resume/parse",
            headers=HEADERS,
            json={
                "uid": "owner",
                "blob_id": "career-profiles/owner/resume.pdf",
                "source_text": "untrusted client text",
                "source_profile": {"name": "injected"},
            },
        )
    assert response.status_code == 202
    verifier.assert_called_once_with("owner-token", check_revoked=True)
    assert blob.await_args is not None and spawn.await_args is not None
    assert blob.await_args.args[1] == "owner"
    job_id, payload = spawn.await_args.args
    assert database[f"resume_parse_jobs/{job_id}"]["uid"] == payload["uid"] == "owner"
    assert payload["blob_id"] == "career-profiles/owner/resume.pdf"
    assert {"blob_content", "source_text", "source_profile"}.isdisjoint(payload)


@pytest.mark.parametrize("source", ["linkedin", "linkedin-ai", "html", "blob"])
async def test_combine_reuses_preflight_through_worker_and_controller(
    resume_client: TestClient, database: dict[str, Any], source: str
) -> None:
    database["profile/owner"] = {"full_name": "Saved Candidate", "skills": ["Python"]}
    url = (
        "https://linkedin.com/in/candidate"
        if source.startswith("linkedin")
        else "https://example.com/cv"
    )
    request = {"uid": "owner", "model": "gemini" if source == "linkedin-ai" else "auto"}
    if source == "blob":
        request["blob_id"] = "career-profiles/owner/resume.pdf"
    else:
        request["url"] = url
    profile = {"name": "Imported Candidate", "skills": ["FastAPI"]}
    source_type = "linkedin" if source == "linkedin-ai" else "pdf" if source == "blob" else "html"
    with (
        patch(
            "app.views.resume.fetch_linkedin_raw_data", AsyncMock(return_value=profile)
        ) as preflight_profile,
        patch(
            "app.views.resume.fetch_source_text",
            AsyncMock(return_value=("Preflight text", source_type)),
        ) as preflight_text,
        patch(
            "app.views.resume.fetch_resume_blob", AsyncMock(return_value=(b"PDF", "pdf"))
        ) as preflight_blob,
        patch("app.views.resume.spawn_resume_parse", AsyncMock()) as spawn,
    ):
        response = resume_client.post("/api/v1/resume/combine", headers=HEADERS, json=request)
    assert response.status_code == 202
    assert spawn.await_args is not None
    job_id, payload = spawn.await_args.args
    assert spawn.await_args.kwargs == {"agent_name": "resume_combine"}
    if source == "linkedin":
        preflight_profile.assert_awaited_once_with(url)
        preflight_text.assert_not_awaited()
        assert payload["source_profile"] == profile and payload["source_text"] is None
    else:
        preflight_text.assert_awaited_once()
        assert payload["source_text"] == "Preflight text" and payload["source_type"] == source_type
    assert preflight_blob.await_count == (1 if source == "blob" else 0)

    controller = ResumeController(Settings(langsmith_tracing=False))
    parser_graph = controller._graph  # pyright: ignore[reportPrivateUsage]
    with (
        patch("app.controllers.resume_controller.ResumeController", return_value=controller),
        patch("app.controllers.resume_controller.fetch_resume_blob", AsyncMock()) as worker_blob,
        patch("app.controllers.resume_controller.fetch_source_text", AsyncMock()) as worker_text,
        patch(
            "app.controllers.resume_controller.fetch_linkedin_raw_data", AsyncMock()
        ) as worker_profile,
        patch.object(
            parser_graph,
            "ainvoke",
            AsyncMock(
                return_value={
                    "resume": ResumeSchema(skills=["FastAPI"]),
                    "model_used": "gemini",
                    "error": None,
                }
            ),
        ) as graph,
        patch.object(
            controller,
            "combine_profiles",
            AsyncMock(
                return_value=ResumeCombineResponse(
                    profile=ResumeSchema(full_name="Combined Candidate"),
                    model_used="gemini",
                )
            ),
        ),
    ):
        result = await run_resume_combine_job(job_id, payload)
    assert result["success"] is True
    worker_blob.assert_not_awaited()
    worker_text.assert_not_awaited()
    worker_profile.assert_not_awaited()
    if source == "linkedin":
        graph.assert_not_awaited()
        assert result["model_used"] == "python-mapping+combine:gemini"
    else:
        graph.assert_awaited_once()
    assert database["profile/owner"]["full_name"] == "Combined Candidate"
    assert database["resume_parse_jobs/" + job_id]["status"] == "completed"


@pytest.mark.parametrize("worker", ["local", "combine", "modal", "persist"])
async def test_workers_reject_payload_owner_mismatch_before_reads_or_writes(
    database: dict[str, Any], worker: str
) -> None:
    database["resume_parse_jobs/job"] = {"uid": "owner", "status": "queued"}
    payload = {"uid": "victim", "url": "https://example.com/cv"}
    with pytest.raises(ValueError, match="owner mismatch"):
        if worker == "local":
            await _run_local_resume_parse("job", payload)
        elif worker == "combine":
            await run_resume_combine_job("job", payload)
        elif worker == "persist":
            await persist_profile_and_complete_job(
                "job", "victim", {"resume": {"full_name": "Injected"}}
            )
        else:
            from modal_app import run_agent_task

            with patch.object(firebase_service, "initialize"):
                await run_agent_task.local("job", "resume_parse", payload)
    cast("AsyncMock", firebase_service.async_get).assert_awaited_once_with("resume_parse_jobs/job")
    cast("AsyncMock", firebase_service.async_set).assert_not_awaited()
    cast("AsyncMock", firebase_service.async_merge).assert_not_awaited()


@pytest.mark.parametrize("worker", ["local", "modal"])
async def test_parse_worker_persists_only_for_verified_job_owner(
    database: dict[str, Any], worker: str
) -> None:
    database["resume_parse_jobs/job"] = {"uid": "owner", "status": "queued"}
    payload = {"uid": "owner", "url": "https://linkedin.com/in/candidate"}
    with patch(
        "app.controllers.resume_controller.fetch_linkedin_raw_data",
        AsyncMock(
            return_value={
                "name": "Imported Candidate",
            }
        ),
    ):
        if worker == "local":
            await _run_local_resume_parse("job", payload)
        else:
            from modal_app import run_agent_task

            with patch.object(firebase_service, "initialize"):
                await run_agent_task.local("job", "resume_parse", payload)
    assert database["profile/owner"]["uid"] == "owner"
    assert database["profile/owner"]["full_name"] == "Imported Candidate"
    assert database["resume_parse_jobs/job"]["status"] == "completed"
