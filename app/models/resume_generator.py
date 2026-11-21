import re
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel


def default_id() -> str:
    """Generate short unique identifier for resume sub-items."""
    return str(uuid.uuid4())[:8]


def format_html_bullets(text: str) -> str:
    """Ensure bullet text is formatted as valid, clean HTML <ul><li>...</li></ul>."""
    if not text or not text.strip():
        return ""
    stripped = text.strip()
    if stripped.startswith("<ul>") and stripped.endswith("</ul>"):
        return stripped

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    cleaned_items: list[str] = []
    for line in lines:
        cleaned = re.sub(r"^[-*•\d.]+\s*", "", line).strip()
        # Strip potential lingering <li> tags
        cleaned = re.sub(r"^<\/?li>", "", cleaned).strip()
        if cleaned:
            cleaned_items.append(f"<li>{cleaned}</li>")

    if cleaned_items:
        return f"<ul>{''.join(cleaned_items)}</ul>"
    return f"<ul><li>{stripped}</li></ul>"


class ResumeGenerateJobRequest(BaseModel):
    """Payload to trigger background ATS resume generation."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    uid: str = Field(description="Firebase Auth UID (reads profile/{uid} & userPreferences/{uid})")
    user_email: str = Field(
        description="Session email (written as resumes.userId in Firestore for Next.js compatibility)"
    )
    resume_id: str | None = Field(
        default=None,
        description="Optional existing resume doc ID to populate or regenerate in-place",
    )
    template_id: str = Field(default="clean-ats", description="Target TemplateId")
    target_role: str | None = Field(
        default=None, description="Optional role tailoring hint from preferences or modal"
    )
    model: str = Field(default="gemini", description="AI provider ('gemini')")


class ResumePersonalInfo(BaseModel):
    """Candidate contact info, header links, and ATS summary."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    full_name: str = Field(default="", description="Candidate's full name")
    email: str = Field(default="", description="Candidate's email address")
    phone: str = Field(default="", description="Candidate's phone number")
    location: str | None = Field(default=None, description="Location (City, State / Country)")
    linkedin: str | None = Field(default=None, description="LinkedIn profile URL")
    github: str | None = Field(default=None, description="GitHub profile URL")
    website: str | None = Field(default=None, description="Personal site or portfolio URL")
    summary: str = Field(default="", description="Crisp 2-3 sentence third-person ATS summary")
    job_title: str | None = Field(default=None, description="Target or primary engineering title")

    @field_validator("phone", "email", "full_name", "summary", mode="before")
    @classmethod
    def coerce_none_to_str(cls, v: object) -> str:
        return str(v).strip() if v is not None else ""


class ResumeExperience(BaseModel):
    """Work experience entry with HTML action-metric-impact bullets."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    id: str = Field(default_factory=default_id, description="Unique experience identifier")
    company: str = Field(description="Company or employer name")
    role: str = Field(description="Job title / engineering role")
    start_date: str = Field(description="Start date (e.g. 'Jan 2022')")
    end_date: str | None = Field(
        default=None, description="End date (e.g. 'Present' or 'Dec 2023')"
    )
    current: bool = Field(default=False, description="Whether this is the current job")
    description: str = Field(
        description="HTML formatted bullets <ul><li>...</li></ul> with action verbs and metrics"
    )
    tech_stack: list[str] = Field(
        default_factory=list, description="Technologies and frameworks used"
    )

    @field_validator("description", mode="before")
    @classmethod
    def ensure_html_bullets(cls, v: object) -> str:
        return format_html_bullets(str(v) if v is not None else "")


class ResumeSkillCategory(BaseModel):
    """Categorized technical skills group."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    category: str = Field(
        description="Category name (e.g. 'Languages', 'Frameworks & Libraries', 'Backend & Cloud')"
    )
    items: list[str] = Field(default_factory=list, description="List of technical skills")


class ResumeProject(BaseModel):
    """High-signal engineering project entry."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    id: str = Field(default_factory=default_id, description="Unique project identifier")
    name: str = Field(description="Project name")
    description: str = Field(
        description="HTML formatted bullets <ul><li>...</li></ul> detailing architecture and impact"
    )
    link: str | None = Field(default=None, description="GitHub repository or live demo URL")
    tech_stack: list[str] = Field(
        default_factory=list, description="Technologies and frameworks used"
    )

    @field_validator("description", mode="before")
    @classmethod
    def ensure_html_bullets(cls, v: object) -> str:
        return format_html_bullets(str(v) if v is not None else "")


class ResumeEducation(BaseModel):
    """Academic history entry."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    id: str = Field(default_factory=default_id, description="Unique education identifier")
    institution: str = Field(description="College or university name")
    degree: str = Field(description="Degree and major (e.g. 'B.S. in Computer Science')")
    start_date: str = Field(default="", description="Start year or date")
    end_date: str = Field(default="", description="Graduation year or date")
    grade: str | None = Field(default=None, description="GPA or academic honors if notable")

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def coerce_none_to_str(cls, v: object) -> str:
        return str(v).strip() if v is not None else ""


class ResumeCustomSection(BaseModel):
    """Dynamic custom section, e.g. Certifications & Awards."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    id: str = Field(default_factory=default_id, description="Unique section identifier")
    title: str = Field(description="Section heading, e.g. 'Certifications & Awards'")
    content: str = Field(description="Content formatted as HTML or text")
    position: Literal["main", "sidebar"] = Field(default="main", description="Layout position")


class ResumeGenerationOutput(BaseModel):
    """Structured LLM output for ATS resume generation matching Next.js Resume structure."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    title: str = Field(description="Resume document title, e.g. '<Name> — <Role> Resume'")
    personal_info: ResumePersonalInfo = Field(
        description="Candidate contact info, header links, and summary"
    )
    experience: list[ResumeExperience] = Field(
        default_factory=list,
        description="Work experience reverse-chronological with HTML <ul><li> bullets",
    )
    skills: list[ResumeSkillCategory] = Field(
        default_factory=list,
        description="3-5 categorized technical skill groups",
    )
    projects: list[ResumeProject] = Field(
        default_factory=list,
        description="Up to 4 high-signal engineering projects with HTML <ul><li> bullets",
    )
    education: list[ResumeEducation] = Field(
        default_factory=list,
        description="Education history reverse-chronological",
    )
    custom_sections: list[ResumeCustomSection] = Field(
        default_factory=list,
        description="Dynamic sections e.g. Certifications & Awards",
    )
    ats_tags: list[str] = Field(
        default_factory=list,
        description="8-12 core technical keywords grounded strictly in candidate profile",
    )
