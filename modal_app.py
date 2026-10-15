from typing import Any

import modal

# Modal cloud image with all runtime dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
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
        "beautifulsoup4>=4.12.0",
        "httpx>=0.27.0",
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
    from app.main import app as web_app

    return web_app


@app.function(timeout=3600)
def run_agent_task(task_id: str, agent_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Execute long-running agent workflows asynchronously in the background.

    Triggered via .spawn() from API handlers or worker events.
    """
    from app.repositories.firebase_client import firebase_service

    firebase_service.initialize()

    # Background agent dispatch logic will be implemented here
    return {
        "task_id": task_id,
        "agent_name": agent_name,
        "status": "completed",
        "received_payload": payload,
    }


@app.function(schedule=modal.Cron("0 * * * *"))
def scheduled_agent_check() -> None:
    """Cron-triggered function for periodic agent executions."""
    # Periodic polling and agent dispatch logic will be implemented here
    pass
