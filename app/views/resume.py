import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.agents.resume_parser_agent import fetch_source_text
from app.agents.vercel_blob_source import fetch_resume_blob
from app.config import get_settings
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


@router.post(
    "/resume/combine",
    response_model=ResumeParseJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start an asynchronous source import and profile combine job",
)
async def combine_resume_profiles(request: ResumeCombineJobRequest) -> ResumeParseJobResponse:
    """Check source access, then queue parsing and AI combination of profile JSON."""
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

    source_url = request.url or f"blob://{request.blob_id}"
    try:
        if request.blob_id:
            source_bytes, _ = await fetch_resume_blob(request.blob_id, request.uid, get_settings())
            suffix = request.blob_id.rsplit(".", maxsplit=1)[-1].lower()
            if suffix not in {"pdf", "doc", "docx"}:
                raise ValueError("Uploaded resume must be a PDF, DOC, or DOCX file")
            source_type = suffix
            await fetch_source_text(
                source_url, source_type=source_type, source_bytes=source_bytes
            )
        else:
            await fetch_source_text(source_url)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Could not access source for profile combine")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not access or read the new source: {exc}",
        ) from exc

    job_id = uuid4().hex
    job_path = f"resume_parse_jobs/{job_id}"
    payload = {
        "uid": request.uid,
        "url": request.url,
        "blob_id": request.blob_id,
        "model": request.model,
        "combine_model": request.combine_model,
    }
    await firebase_service.async_set(
        job_path,
        {
            "uid": request.uid,
            "status": "queued",
            "result": None,
            "job_type": "resume_combine",
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )
    try:
        await spawn_resume_parse(job_id, payload, agent_name="resume_combine")
    except Exception as exc:
        logger.exception("Could not dispatch profile combine job %s", job_id)
        await firebase_service.async_merge(
            job_path, {"status": "failed", "result": {"error": str(exc)}}
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
) -> ResumeParseJobResponse:
    """Queue a resume parse from a public URL or a private uploaded Blob."""
    job_id = uuid4().hex
    job_path = f"resume_parse_jobs/{job_id}"
    payload = request.model_dump(mode="json")
    if request.blob_id:
        try:
            _blob_content, _ = await fetch_resume_blob(
                request.blob_id, request.uid, get_settings()
            )
        except Exception as exc:
            logger.exception("Could not prefetch uploaded resume Blob %s", request.blob_id)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Could not read uploaded resume Blob '{request.blob_id}': {exc}",
            ) from exc
        # Blob validated; worker will re-fetch from Vercel Blob on its own

    await firebase_service.async_set(
        job_path,
        {
            "uid": request.uid,
            "status": "queued",
            "result": None,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )
    try:
        await spawn_resume_parse(job_id, payload)
    except Exception as exc:
        logger.exception("Could not dispatch resume parse job %s", job_id)
        await firebase_service.async_merge(
            job_path, {"status": "failed", "result": {"error": str(exc)}}
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not start resume parsing",
        ) from exc
    return ResumeParseJobResponse(job_id=job_id, status="queued")


@router.get("/resume/jobs/{job_id}", response_model=ResumeParseJobStatusResponse)
async def get_resume_parse_job(job_id: str) -> ResumeParseJobStatusResponse:
    """Return the current status and final result for a queued parse."""
    job = await firebase_service.async_get(f"resume_parse_jobs/{job_id}")
    if not job:
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
async def delete_resume_parse_job(job_id: str) -> None:
    """Remove the pollable job result after the client has consumed it."""
    await firebase_service.async_delete(f"resume_parse_jobs/{job_id}")
