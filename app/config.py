import os
from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Modal sets MODAL_ENVIRONMENT inside all containers.
# When running on Modal, secrets are injected as OS env vars directly —
# we skip loading .env so there's no confusion with stale local values.
_RUNNING_ON_MODAL = "MODAL_ENVIRONMENT" in os.environ


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    - Local development: values are read from the .env file.
    - Modal cloud: values are injected by the `codesnooper-secrets` Modal
      Secret and read directly from the OS environment (no .env file).
    """

    model_config = SettingsConfigDict(
        env_file=None if _RUNNING_ON_MODAL else ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_env: str = Field(default="development")
    app_debug: bool = Field(default=False)

    # CORS
    cors_origins: list[str] = Field(
        default=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]
    )

    # Firebase
    firebase_project_id: str = Field(default="")
    firebase_client_email: str = Field(default="")
    firebase_private_key: str = Field(default="")
    firebase_database_url: str = Field(default="")
    google_application_credentials: str = Field(default="")
    google_application_credentials_json: str = Field(default="")

    # Google Gemini
    google_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-3.5-flash-lite")

    # Mistral AI
    chat_mistral_api_key: str = Field(default="")
    mistral_api_key: str = Field(default="")
    mistral_model: str = Field(default="ministral-8b-latest")

    # LangSmith Tracing
    langsmith_api_key: str = Field(default="")
    langsmith_project: str = Field(default="codesnooper-agent-local")
    langsmith_agent_id: str = Field(default="")
    langsmith_agent_environment: str = Field(default="")
    langsmith_tracing: bool = Field(default=True)
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com")

    # Sentry
    sentry_dsn: str = Field(default="")
    sentry_traces_sample_rate: float = Field(default=1.0)

    # Bright Data
    brightdata_api_key: str = Field(default="")

    # API Gateway Token
    api_gateway_token: str = Field(
        default="",
        validation_alias=AliasChoices(
            "API_GATEWAY_TOKEN",
            "AUTH_TOKEN",
            "API_KEY",
            "API_SECRET_TOKEN",
        ),
    )

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    @property
    def langsmith_uses_agent_address(self) -> bool:
        """Use the agent/environment destination for configured production traces."""
        return bool(
            self.is_production
            and self.langsmith_agent_id.strip()
            and self.langsmith_agent_environment.strip()
        )


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton for dependency injection."""
    return Settings()


def configure_tracing(settings: Settings | None = None) -> None:
    """Configure LangSmith / LangChain tracing and resolve destination conflicts.

    In LangSmith (v0.3+), having both a Project (LANGSMITH_PROJECT) and an Agent Address
    (LANGSMITH_AGENT_ID / LANGSMITH_AGENT_ENVIRONMENT) in the environment causes an
    EnvAddressError collision, dropping all trace events.

    This function sets standard LangSmith/LangChain environment flags and cleans up
    conflicting agent address variables when project-based tracing is configured.
    """
    if settings is None:
        settings = get_settings()

    if not (settings.langsmith_api_key and settings.langsmith_tracing):
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key

    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint

    # Project and agent/environment destinations are mutually exclusive.
    if settings.langsmith_uses_agent_address:
        os.environ["LANGSMITH_AGENT_ID"] = settings.langsmith_agent_id
        os.environ["LANGSMITH_AGENT_ENVIRONMENT"] = settings.langsmith_agent_environment
        os.environ.pop("LANGSMITH_PROJECT", None)
        os.environ.pop("LANGCHAIN_PROJECT", None)
    elif settings.langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        os.environ.pop("LANGSMITH_AGENT_ID", None)
        os.environ.pop("LANGSMITH_AGENT_ENVIRONMENT", None)
    else:
        os.environ.pop("LANGSMITH_PROJECT", None)
        os.environ.pop("LANGCHAIN_PROJECT", None)
        os.environ.pop("LANGSMITH_AGENT_ID", None)
        os.environ.pop("LANGSMITH_AGENT_ENVIRONMENT", None)


def flush_tracing() -> None:
    """Flush pending LangSmith trace events before returning responses in serverless runtimes.

    In serverless platforms like Modal, containers freeze immediately after an ASGI response
    completes. Explicitly flushing the client guarantees that asynchronous trace batches
    are transmitted before container suspension.
    """
    try:
        from langchain_core.tracers.langchain import get_client

        client = get_client()
        client.flush()
    except Exception:
        # Flushing should never crash request fulfillment
        pass
