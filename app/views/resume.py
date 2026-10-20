from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.config import Settings, get_settings
from app.controllers.resume_controller import ResumeController
from app.models.resume import ResumeParseRequest, ResumeParseResponse

router = APIRouter(tags=["resume"])

SettingsDependency = Annotated[Settings, Depends(get_settings)]


@router.post(
    "/resume/parse",
    response_model=ResumeParseResponse,
    status_code=status.HTTP_200_OK,
    summary="Parse resume from URL",
    description=(
        "Downloads a resume from a URL (PDF direct link, LinkedIn profile, or public webpage) "
        "and extracts structured JSON containing candidate details, work history, education, "
        "skills, certifications, and links using Gemini or Mistral models."
    ),
)
async def parse_resume(
    request: ResumeParseRequest,
    settings: SettingsDependency,
) -> ResumeParseResponse:
    """Parse resume from URL into structured data."""
    controller = ResumeController(settings)
    return await controller.parse_resume(request)
