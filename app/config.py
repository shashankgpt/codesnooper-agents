from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
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
    gemini_model: str = Field(default="gemini-3.8-flash")

    # Sentry
    sentry_dsn: str = Field(default="")
    sentry_traces_sample_rate: float = Field(default=1.0)

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton for dependency injection."""
    return Settings()
