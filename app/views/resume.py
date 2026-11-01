import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from app.core.resume_jobs import spawn_resume_parse
from app.models.resume import (
    ResumeCombineJobRequest,
    ResumeParseRequest,
    ResumeParseResponse,
    ResumeSchema,
)
from app.repositories.firebase_client import firebase_service

router = APIRouter(tags=["resume"])
logger = logging.getLogger(__name__)


class ResumeParseJobResponse(BaseModel):
    job_id: str
    status: str


class ResumeParseJobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: ResumeParseResponse | dict[str, Any] | None = None


def _request_uid(request: Request) -> str:
    uid = request.headers.get("x-user-id", "").strip()
    if not uid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing X-User-Id header")
    return uid


async def _create_or_reuse_job(
    request: Request, job_data: dict[str, Any]
) -> tuple[str, bool]:
    idempotency_key = request.headers.get("idempotency-key", "").strip() or None
    if idempotency_key and (len(idempotency_key) > 128 or any(char.isspace() for char in idempotency_key)):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Idempotency-Key header")
    return await firebase_service.async_create_or_get_resume_job(
        uuid4().hex, idempotency_key, job_data
    )


@router.post(
    "/resume/combine",
    response_model=ResumeParseJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start an asynchronous source import and profile combine job",
)
async def combine_resume_profiles(
    request: ResumeCombineJobRequest, raw_request: Request
) -> ResumeParseJobResponse:
    """Validate profile state, then queue source parsing and profile combination."""
    if _request_uid(raw_request) != request.uid:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User ID does not match request")
    try:
        current_profile = await firebase_service.async_get(f"profile/{request.uid}")
        if not current_profile:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No saved career profile was found to combine",
            )
        ResumeSchema.model_validate(current_profile)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Could not load existing profile for combine")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The saved career profile is not valid for combining",
        ) from exc

    payload = {
        "uid": request.uid,
        "url": request.url,
        "blob_id": request.blob_id,
        "model": request.model,
        "combine_model": request.combine_model,
    }
    job_id, created = await _create_or_reuse_job(
        raw_request,
        {
            "uid": request.uid,
            "status": "queued",
            "result": None,
            "job_type": "resume_combine",
            "updated_at": datetime.now(UTC).isoformat(),
            "idempotency_key": raw_request.headers.get("idempotency-key"),
        },
    )
    if not created:
        return ResumeParseJobResponse(job_id=job_id, status="queued")
    try:
        await spawn_resume_parse(job_id, payload, agent_name="resume_combine")
    except Exception as exc:
        logger.exception("Could not dispatch profile combine job %s", job_id)
        await firebase_service.async_merge(
            f"resume_parse_jobs/{job_id}", {"status": "failed", "result": {"error": str(exc)}}
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not start profile combination",
        ) from exc
    return ResumeParseJobResponse(job_id=job_id, status="queued")


@router.post(
    "/resume/parse",
    response_model=ResumeParseJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a resume parse job",
    description=(
        "Queues parsing from a public LinkedIn/resume URL or a private uploaded PDF, DOC, or DOCX. "
        "Poll GET /api/v1/resume/jobs/{job_id} for the structured result."
    ),
)
async def parse_resume(
    request: ResumeParseRequest,
    raw_request: Request,
) -> ResumeParseJobResponse:
    """Queue a resume parse from a public URL or a private uploaded Blob."""
    if _request_uid(raw_request) != request.uid:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User ID does not match request")
    payload = request.model_dump(mode="json")
    job_id, created = await _create_or_reuse_job(
        raw_request,
        {
            "uid": request.uid,
            "status": "queued",
            "result": None,
            "updated_at": datetime.now(UTC).isoformat(),
            "idempotency_key": raw_request.headers.get("idempotency-key"),
        },
    )
    if not created:
        return ResumeParseJobResponse(job_id=job_id, status="queued")
    try:
        await spawn_resume_parse(job_id, payload)
    except Exception as exc:
        logger.exception("Could not dispatch resume parse job %s", job_id)
        await firebase_service.async_merge(
            f"resume_parse_jobs/{job_id}", {"status": "failed", "result": {"error": str(exc)}}
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not start resume parsing",
        ) from exc
    return ResumeParseJobResponse(job_id=job_id, status="queued")


@router.get("/resume/jobs/{job_id}", response_model=ResumeParseJobStatusResponse)
async def get_resume_parse_job(job_id: str, request: Request) -> ResumeParseJobStatusResponse:
    """Return the current status and final result for a queued parse."""
    job = await firebase_service.async_get(f"resume_parse_jobs/{job_id}")
    if not job or job.get("uid") != _request_uid(request):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resume parse job not found",
        )
    return ResumeParseJobStatusResponse(
        job_id=job_id,
        status=job.get("status", "queued"),
        result=job.get("result"),
    )


@router.delete("/resume/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_resume_parse_job(job_id: str, request: Request) -> None:
    """Remove the pollable job result after the client has consumed it."""
    job = await firebase_service.async_get(f"resume_parse_jobs/{job_id}")
    if not job or job.get("uid") != _request_uid(request):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resume parse job not found",
        )
    await firebase_service.async_delete(f"resume_parse_jobs/{job_id}")
