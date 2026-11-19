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


async def persist_profile_and_complete_job(
    job_id: str,
    uid: str,
    result: dict[str, Any],
    persist_profile: bool = True,
) -> None:
    """Optionally persist a successful parse before marking its job complete."""
    from app.repositories.firebase_client import firebase_service

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

    request = ResumeCombineJobRequest.model_validate(payload)
    current_profile_data = await firebase_service.async_get(f"profile/{request.uid}")
    if not current_profile_data:
        raise ValueError("No saved career profile was found to combine")
    current_profile = ResumeSchema.model_validate(current_profile_data)
    controller = ResumeController(get_settings())
    parsed = await controller.parse_resume(
        ResumeParseRequest(
            uid=request.uid,
            url=request.url,
            blob_id=request.blob_id,
            model=request.model,
            persist_profile=False,
            source_text=payload.get("source_text"),
            source_type=payload.get("source_type"),
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
    result["raw_text"] = combined.profile.raw_text
    result["model_used"] = f"{parsed.model_used}+combine:{combined.model_used}"
    await persist_profile_and_complete_job(job_id, request.uid, result)
    return result


async def persist_synced_profile_and_complete_job(
    job_id: str,
    uid: str,
    mode: str,
    result: dict[str, Any],
) -> None:
    """Safely persist synced career profile without clobbering lastImportJobId or newer edits."""
    from app.models.resume import ResumeSchema
    from app.repositories.firebase_client import firebase_service

    now = datetime.now(UTC).isoformat()
    profile_data = result.get("resume")
    if not isinstance(profile_data, dict):
        raise ValueError("Successful resume sync did not include profile data")

    profile_path = f"profile/{uid}"
    latest_doc = await firebase_service.async_get(profile_path) or {}

    # Never overwrite lastImportJobId with a sync job ID (it belongs to source imports)
    last_import_job_id = latest_doc.get("lastImportJobId")

    if mode == "authoritative":
        if not latest_doc:
            result["skipped"] = "profile_not_found"
        else:
            # Merge only narrative_context onto latest structured fields in Firestore
            new_narrative = profile_data.get("narrative_context")
            merged_doc = {**latest_doc, "narrative_context": new_narrative}
            # Re-derive raw_text against freshest structured fields to prevent stale overwrites
            recomputed = ResumeSchema.model_validate(merged_doc).refresh_raw_text()
            merged_doc["raw_text"] = recomputed.raw_text
            merged_doc["updatedAt"] = now
            merged_doc["uid"] = uid
            if last_import_job_id:
                merged_doc["lastImportJobId"] = last_import_job_id
            await firebase_service.async_set(profile_path, merged_doc)
            result["resume"] = recomputed.model_dump(mode="json")
            result["raw_text"] = recomputed.raw_text
    else:
        # Additive mode: strip None and empty lists, merge with latest doc, and refresh raw_text
        non_empty = {k: v for k, v in profile_data.items() if v is not None and v != []}
        merged_doc = {
            **latest_doc,
            **non_empty,
            "uid": uid,
            "updatedAt": now,
            "createdAt": latest_doc.get("createdAt", now),
        }
        if last_import_job_id:
            merged_doc["lastImportJobId"] = last_import_job_id
        recomputed = ResumeSchema.model_validate(merged_doc).refresh_raw_text()
        merged_doc["raw_text"] = recomputed.raw_text
        await firebase_service.async_set(profile_path, merged_doc)
        result["resume"] = recomputed.model_dump(mode="json")
        result["raw_text"] = recomputed.raw_text

    job_summary = {
        "success": True,
        "mode": mode,
        "source_kind": result.get("source_kind"),
        "model_used": result.get("model_used"),
        **({"skipped": result["skipped"]} if "skipped" in result else {}),
    }
    await firebase_service.async_merge(
        f"resume_parse_jobs/{job_id}",
        {
            "status": "completed",
            "result": job_summary,
            "updated_at": now,
        },
    )


async def run_resume_sync_job(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Execute authoritative or additive profile synchronization and persist the result."""
    from app.config import get_settings
    from app.controllers.resume_controller import ResumeController
    from app.models.resume import ResumeSchema, ResumeSyncJobRequest, ResumeSyncRequest
    from app.repositories.firebase_client import firebase_service

    request = ResumeSyncJobRequest.model_validate(payload)
    current_data = await firebase_service.async_get(f"profile/{request.uid}")
    current_profile = ResumeSchema.model_validate(current_data) if current_data else None

    controller = ResumeController(get_settings())
    sync_resp = await controller.sync_profile(
        ResumeSyncRequest(
            current_profile=current_profile,
            profile_updates=request.profile_updates,
            mode=request.mode,
            source_kind=request.source_kind,
            model=request.model,
        )
    )
    result = {
        "success": True,
        "resume": sync_resp.profile.model_dump(mode="json"),
        "raw_text": sync_resp.profile.raw_text,
        "model_used": sync_resp.model_used,
        "mode": request.mode,
        "source_kind": request.source_kind,
    }
    await persist_synced_profile_and_complete_job(job_id, request.uid, request.mode, result)
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
    now = datetime.now(UTC).isoformat()
    try:
        await firebase_service.async_merge(path, {"status": "processing", "updated_at": now})
        if agent_name == "resume_combine":
            await run_resume_combine_job(job_id, payload)
            return
        if agent_name == "resume_sync":
            await run_resume_sync_job(job_id, payload)
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
