"""Dispatch resume jobs to Modal, with an in-process path for local development."""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)
_modal_job_function: Any = None
_local_tasks: set[asyncio.Task[None]] = set()


def configure_modal_job_function(function: Any) -> None:
    global _modal_job_function
    _modal_job_function = function


async def require_resume_job_owner(job_id: str, uid: Any) -> str:
    """Match worker identity to the owner persisted by the authenticated API."""
    from app.repositories.firebase_client import firebase_service

    job = await firebase_service.async_get(f"resume_parse_jobs/{job_id}")
    if (
        not isinstance(uid, str)
        or not uid.strip()
        or uid != uid.strip()
        or "/" in uid
        or not job
        or job.get("uid") != uid
    ):
        raise ValueError("Resume parse job owner mismatch or job not found")
    return uid


async def persist_profile_and_complete_job(
    job_id: str,
    uid: str,
    result: dict[str, Any],
    persist_profile: bool = True,
) -> None:
    """Optionally persist a successful parse before marking its job complete."""
    from app.repositories.firebase_client import firebase_service

    uid = await require_resume_job_owner(job_id, uid)
    now = datetime.now(UTC).isoformat()
    profile = result.get("resume")
    if not isinstance(profile, dict):
        raise ValueError("Successful resume parse did not include profile data")

    if persist_profile:
        profile_path = f"profile/{uid}"
        current_profile = await firebase_service.async_get(profile_path) or {}
        await firebase_service.async_set(
            profile_path,
            {
                **current_profile,
                **profile,
                "uid": uid,
                "lastImportJobId": job_id,
                "createdAt": current_profile.get("createdAt", now),
                "updatedAt": now,
            },
        )
    await firebase_service.async_merge(
        f"resume_parse_jobs/{job_id}",
        {
            "status": "completed",
            "result": result,
            "updated_at": now,
        },
    )


async def spawn_resume_parse(
    job_id: str,
    payload: dict[str, Any],
    agent_name: str = "resume_parse",
) -> None:
    if _modal_job_function is None:
        from app.config import get_settings

        if get_settings().is_production:
            raise RuntimeError("Modal resume worker is unavailable")

        task = asyncio.create_task(_run_local_resume_parse(job_id, payload, agent_name))
        _local_tasks.add(task)
        task.add_done_callback(_local_tasks.discard)
        return

    await _modal_job_function.spawn.aio(job_id, agent_name, payload)


async def run_resume_combine_job(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Parse preflighted source text, AI-combine it with current JSON, and persist the result."""
    from app.config import get_settings
    from app.controllers.resume_controller import ResumeController
    from app.models.resume import (
        ResumeCombineJobRequest,
        ResumeCombineRequest,
        ResumeParseRequest,
        ResumeSchema,
    )
    from app.repositories.firebase_client import firebase_service

    uid = await require_resume_job_owner(job_id, payload.get("uid"))
    request = ResumeCombineJobRequest.model_validate(payload)
    current_profile_data = await firebase_service.async_get(f"profile/{uid}")
    if not current_profile_data:
        raise ValueError("No saved career profile was found to combine")
    current_profile = ResumeSchema.model_validate(current_profile_data)
    controller = ResumeController(get_settings())
    parsed = await controller.parse_resume(
        ResumeParseRequest(
            uid=uid,
            url=request.url,
            blob_id=request.blob_id,
            model=request.model,
            persist_profile=False,
            source_text=payload.get("source_text"),
            source_type=payload.get("source_type"),
            source_profile=payload.get("source_profile"),
        )
    )
    if not parsed.success or parsed.resume is None:
        raise ValueError(parsed.error or "Could not parse the new source")

    combined = await controller.combine_profiles(
        ResumeCombineRequest(
            current_profile=current_profile,
            new_profile=parsed.resume,
            model=request.combine_model,
        )
    )
    result = parsed.model_dump(mode="json")
    result["resume"] = combined.profile.model_dump(mode="json")
    result["model_used"] = f"{parsed.model_used}+combine:{combined.model_used}"
    await persist_profile_and_complete_job(job_id, uid, result)
    return result


async def _run_local_resume_parse(
    job_id: str,
    payload: dict[str, Any],
    agent_name: str = "resume_parse",
) -> None:
    """Run a resume job in the API process when developing without Modal."""
    from app.config import get_settings
    from app.controllers.resume_controller import ResumeController
    from app.models.resume import ResumeParseRequest
    from app.repositories.firebase_client import firebase_service

    path = f"resume_parse_jobs/{job_id}"
    uid = await require_resume_job_owner(job_id, payload.get("uid"))
    now = datetime.now(UTC).isoformat()
    try:
        await firebase_service.async_merge(path, {"status": "processing", "updated_at": now})
        if agent_name == "resume_combine":
            await run_resume_combine_job(job_id, payload)
            return
        if agent_name != "resume_parse":
            raise ValueError(f"Unsupported background agent: {agent_name}")
        request = ResumeParseRequest.model_validate(payload)
        response = await ResumeController(get_settings()).parse_resume(request)
        result = response.model_dump(mode="json")
        if response.success:
            await persist_profile_and_complete_job(
                job_id, request.uid, result, persist_profile=request.persist_profile
            )
            return
        await require_resume_job_owner(job_id, uid)
        await firebase_service.async_merge(
            path,
            {
                "status": "failed",
                "result": result,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
    except Exception as exc:
        logger.exception("Local resume parse job %s failed", job_id)
        try:
            await require_resume_job_owner(job_id, uid)
            await firebase_service.async_merge(
                path,
                {
                    "status": "failed",
                    "result": {"success": False, "error": str(exc)},
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
        except Exception:
            logger.exception("Could not save failure status for resume parse job %s", job_id)
