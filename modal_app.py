from datetime import UTC, datetime, timedelta
from typing import Any

import modal

# Modal cloud image with all runtime dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("antiword")
    .pip_install(
        "fastapi[standard]>=0.115.0",
        "pydantic>=2.9.0",
        "pydantic-settings>=2.6.0",
        "firebase-admin>=6.6.0",
        "langchain>=0.3.0",
        "langchain-google-genai>=2.0.0",
        "langgraph>=0.2.0",
        "sentry-sdk[fastapi]>=2.14.0",
        "langchain-mistralai>=0.2.0",
        "langsmith>=0.3.0",
        "pypdf>=4.0.0",
        "python-docx>=1.1.2",
        "beautifulsoup4>=4.12.0",
        "httpx>=0.27.0",
        "vercel>=0.5.0",
    )
    .add_local_python_source("app")
)

app = modal.App(
    name="codesnooper-agents",
    image=image,
    secrets=[modal.Secret.from_name("codesnooper-secrets")],
)


@app.function()
@modal.asgi_app()
def fastapi_app() -> Any:
    """Serve the FastAPI ASGI application on Modal."""
    from app.core.resume_jobs import configure_modal_job_function
    from app.main import app as web_app

    configure_modal_job_function(run_agent_task)
    return web_app


@app.function(timeout=3600)
async def run_agent_task(task_id: str, agent_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Execute a resume parser request and persist its pollable result."""
    from app.config import get_settings
    from app.controllers.resume_controller import ResumeController
    from app.core.resume_jobs import persist_profile_and_complete_job, run_resume_combine_job
    from app.models.resume import ResumeParseRequest
    from app.repositories.firebase_client import firebase_service

    firebase_service.initialize()
    path = f"resume_parse_jobs/{task_id}"
    await firebase_service.async_merge(
        path,
        {"status": "processing", "updated_at": datetime.now(UTC).isoformat()},
    )
    try:
        if agent_name == "resume_combine":
            result = await run_resume_combine_job(task_id, payload)
        elif agent_name == "resume_parse":
            request = ResumeParseRequest.model_validate(payload)
            response = await ResumeController(get_settings()).parse_resume(request)
            result = response.model_dump(mode="json")
            if response.success:
                await persist_profile_and_complete_job(
                    task_id, request.uid, result, persist_profile=request.persist_profile
                )
            else:
                await firebase_service.async_merge(
                    path,
                    {
                        "status": "failed",
                        "result": result,
                        "updated_at": datetime.now(UTC).isoformat(),
                    },
                )
        else:
            raise ValueError(f"Unsupported background agent: {agent_name}")
        return result
    except Exception as exc:
        result = {"success": False, "error": str(exc)}
        await firebase_service.async_merge(
            path,
            {
                "status": "failed",
                "result": result,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        return result


@app.function(schedule=modal.Cron("0 * * * *"))
def scheduled_agent_check() -> None:
    """Remove stale parse results if a client never polls and consumes them."""
    import logging

    from google.cloud.firestore_v1.base_query import FieldFilter

    from app.repositories.firebase_client import firebase_service

    firebase_service.initialize()
    db = firebase_service.firestore
    if db is None:
        return
    cutoff_iso = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
    stale_query = (
        db.collection("resume_parse_jobs")
        .where(filter=FieldFilter("updated_at", "<", cutoff_iso))
        .limit(500)
    )
    batch = db.batch()
    deleted = 0
    for doc in stale_query.stream():
        batch.delete(doc.reference)
        deleted += 1
    if deleted:
        batch.commit()
        logging.getLogger(__name__).info("Cleaned up %d stale resume parse jobs", deleted)
