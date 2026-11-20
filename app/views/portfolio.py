import logging
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.config import get_settings
from app.controllers.portfolio_controller import PortfolioController
from app.core.resume_jobs import spawn_resume_parse
from app.models.portfolio import (
    PortfolioEnhanceRequest,
    PortfolioEnhanceResponse,
    PortfolioGenerateJobRequest,
)
from app.repositories.firebase_client import firebase_service

router = APIRouter(prefix="/portfolio", tags=["portfolio"])
logger = logging.getLogger(__name__)


class PortfolioJobResponse(BaseModel):
    """Initial async response when a portfolio generation task is queued."""

    job_id: str = Field(description="Unique identifier for polling job completion")
    status: str = Field(default="queued", description="Initial job status")


class PortfolioJobStatusResponse(BaseModel):
    """Pollable status payload for a portfolio generation job."""

    job_id: str
    status: str
    result: dict[str, Any] | None = None


def _request_uid(request: Request) -> str:
    """Extract and validate the authenticated user ID from request headers."""
    uid = request.headers.get("x-user-id", "").strip()
    if not uid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing X-User-Id header",
        )
    return uid


async def _create_or_reuse_portfolio_job(
    request: Request,
    job_data: dict[str, Any],
) -> tuple[str, bool]:
    """Support Idempotency-Key header for duplicate-safe job creation."""
    idempotency_key = request.headers.get("idempotency-key", "").strip() or None
    if idempotency_key and (
        len(idempotency_key) > 128 or any(char.isspace() for char in idempotency_key)
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Idempotency-Key header",
        )
    return await firebase_service.async_create_or_get_resume_job(
        uuid4().hex, idempotency_key, job_data
    )


@router.post(
    "/generate",
    response_model=PortfolioJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue asynchronous developer portfolio curation and draft generation",
)
async def generate_portfolio(
    request: PortfolioGenerateJobRequest,
    raw_request: Request,
) -> PortfolioJobResponse:
    """Validate user context, queue portfolio curation job, and return job ID for polling."""
    caller_uid = _request_uid(raw_request)
    if caller_uid != request.uid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User ID does not match request payload",
        )

    # Pre-flight check: Career profile must exist
    profile = await firebase_service.async_get(f"profile/{request.uid}")
    if not profile or not isinstance(profile, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No career profile found. Please create or import your career profile first.",
        )

    job_data = {
        "uid": request.uid,
        "status": "queued",
        "result": None,
        "job_type": "portfolio_generate",
        "updated_at": datetime.now(UTC).isoformat(),
        "idempotency_key": raw_request.headers.get("idempotency-key"),
    }

    job_id, created = await _create_or_reuse_portfolio_job(raw_request, job_data)
    if not created:
        return PortfolioJobResponse(job_id=job_id, status="queued")

    payload = {
        "uid": request.uid,
        "model": request.model,
        "preserve_custom_avatar": request.preserve_custom_avatar,
    }

    try:
        await spawn_resume_parse(job_id, payload, agent_name="portfolio_generate")
    except Exception as exc:
        logger.exception("Could not spawn portfolio generation job %s", job_id)
        await firebase_service.async_merge(
            f"resume_parse_jobs/{job_id}",
            {"status": "failed", "result": {"error": str(exc)}},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not initiate background portfolio generation",
        ) from exc

    return PortfolioJobResponse(job_id=job_id, status="queued")


@router.post(
    "/enhance",
    response_model=PortfolioEnhanceResponse,
    status_code=status.HTTP_200_OK,
    summary="Synchronously polish a portfolio section with LLM",
)
async def enhance_section(
    request: PortfolioEnhanceRequest,
    raw_request: Request,
) -> PortfolioEnhanceResponse:
    """Refine headline, bio, project, or experience text while protecting PII."""
    caller_uid = _request_uid(raw_request)
    if caller_uid != request.uid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User ID does not match request payload",
        )

    controller = PortfolioController(get_settings())
    try:
        return await controller.enhance_section(request)
    except Exception as exc:
        logger.exception("Error enhancing portfolio section '%s'", request.section)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enhance portfolio section: {exc}",
        ) from exc


@router.get(
    "/jobs/{job_id}",
    response_model=PortfolioJobStatusResponse,
    summary="Check status of a portfolio generation job",
)
async def get_portfolio_job_status(
    job_id: str,
    raw_request: Request,
) -> PortfolioJobStatusResponse:
    """Poll job status directly from Firestore with ownership verification."""
    caller_uid = _request_uid(raw_request)
    job_doc = await firebase_service.async_get(f"resume_parse_jobs/{job_id}")

    if not job_doc or not isinstance(job_doc, dict):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio job '{job_id}' not found",
        )

    if job_doc.get("uid") != caller_uid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to access this job",
        )

    raw_result = job_doc.get("result")
    raw_status = job_doc.get("status")
    result_dict: dict[str, Any] | None = (
        cast("dict[str, Any]", raw_result) if isinstance(raw_result, dict) else None
    )
    status_str: str = raw_status if isinstance(raw_status, str) else "unknown"
    return PortfolioJobStatusResponse(
        job_id=job_id,
        status=status_str,
        result=result_dict,
    )


@router.delete(
    "/jobs/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a consumed portfolio generation job",
)
async def delete_portfolio_job(
    job_id: str,
    raw_request: Request,
) -> None:
    """Clean up job record after client has polled the result."""
    caller_uid = _request_uid(raw_request)
    job_doc = await firebase_service.async_get(f"resume_parse_jobs/{job_id}")

    if not job_doc or not isinstance(job_doc, dict):
        return

    if job_doc.get("uid") != caller_uid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to delete this job",
        )

    await firebase_service.async_delete(f"resume_parse_jobs/{job_id}")
