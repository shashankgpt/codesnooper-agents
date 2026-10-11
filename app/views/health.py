from typing import Annotated

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.controllers.health_controller import HealthController
from app.models.common import HealthResponse, HelloResponse

router = APIRouter(tags=["health"])

SettingsDependency = Annotated[Settings, Depends(get_settings)]


@router.get("/hello", response_model=HelloResponse)
async def hello(settings: SettingsDependency) -> HelloResponse:
    """Return welcome greeting and system metadata."""
    controller = HealthController(settings)
    return controller.get_hello()


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: SettingsDependency) -> HealthResponse:
    """Check connectivity and operational health."""
    controller = HealthController(settings)
    return await controller.get_health()
