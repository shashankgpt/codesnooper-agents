from pydantic import BaseModel, Field


class HelloResponse(BaseModel):
    """Response for the hello endpoint."""

    message: str = Field(description="Welcome message")
    environment: str = Field(description="Active runtime environment")
    version: str = Field(description="Application version")


class HealthResponse(BaseModel):
    """Response for the health check endpoint."""

    status: str = Field(description="Overall service health status")
    firebase_connected: bool = Field(description="Whether Firebase/Firestore is accessible")
    gemini_connected: bool = Field(description="Whether Google Gemini API is accessible")
    langsmith_connected: bool = Field(description="Whether LangSmith API is accessible")
    gemini_model: str | None = Field(default=None, description="Active Gemini model used")
