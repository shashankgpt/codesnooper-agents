import re
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


def default_id() -> str:
    """Generate short unique identifier for portfolio sub-items."""
    return str(uuid.uuid4())[:8]


def slugify(name: str) -> str:
    """Convert a name into a clean, URL-safe slug."""
    clean = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return clean or "developer"


class PortfolioSocials(BaseModel):
    """Candidate's external links and developer profiles."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    github_url: str | None = Field(default=None, description="GitHub profile URL")
    linkedin_url: str | None = Field(default=None, description="LinkedIn profile URL")
    website_url: str | None = Field(default=None, description="Personal site URL")


class PortfolioPrivacy(BaseModel):
    """Privacy controls for the public portfolio."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    email_visibility: Literal["hidden", "cta", "public"] = Field(
        default="cta",
        description="'hidden' (no email CTA), 'cta' (form proxy), or 'public' (raw email)",
    )
    public_email: str | None = Field(
        default=None,
        description="Public email address if email_visibility is 'public'",
    )


class PortfolioHighlight(BaseModel):
    """Key technical impact metric displayed in hero bento grid."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    id: str = Field(default_factory=default_id, description="Unique highlight identifier")
    metric: str = Field(description="Exact metric string, e.g. '99.99%', '25k RPS', '8+ yrs'")
    label: str = Field(description="Context label, e.g. 'Production Uptime'")
    detail: str | None = Field(default=None, description="Optional 1-line detail or context")


class PortfolioProject(BaseModel):
    """Curated technical project."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    id: str = Field(default_factory=default_id, description="Unique project identifier")
    title: str = Field(description="Project title")
    tagline: str | None = Field(default=None, description="Short punchy 1-line tagline")
    description: str = Field(description="Concise technical description of architecture")
    highlights: list[str] = Field(
        default_factory=list, description="Key technical achievements"
    )
    tech_stack: list[str] = Field(
        default_factory=list, description="Technologies, languages, and frameworks"
    )
    live_url: str | None = Field(default=None, description="Live production URL if available")
    repo_url: str | None = Field(default=None, description="Source repository URL")
    featured: bool = Field(default=True, description="Whether to highlight prominently")


class PortfolioExperience(BaseModel):
    """Work experience entry with impact bullets."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    id: str = Field(default_factory=default_id, description="Unique experience identifier")
    company: str = Field(description="Company or organization name")
    role: str = Field(description="Job title / role")
    location: str | None = Field(default=None, description="Location or 'Remote'")
    start_date: str = Field(description="Start date (e.g. '2021-01' or '2021')")
    end_date: str | None = Field(
        default=None, description="End date (e.g. '2023-05' or 'Present')"
    )
    is_current: bool = Field(default=False, description="Whether this is the current role")
    summary: str | None = Field(default=None, description="Optional brief role summary")
    bullets: list[str] = Field(
        default_factory=list, description="Impact-focused accomplishment bullets"
    )
    technologies: list[str] = Field(
        default_factory=list, description="Key technologies used in role"
    )


class PortfolioEducation(BaseModel):
    """Education history."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    id: str = Field(default_factory=default_id, description="Unique education identifier")
    institution: str = Field(description="University, college, or school name")
    degree: str = Field(description="Degree name (e.g. 'B.S. in Computer Science')")
    field_of_study: str | None = Field(default=None, description="Field of study / major")
    start_date: str | None = Field(default=None, description="Start year or date")
    end_date: str | None = Field(default=None, description="Graduation year or date")
    honors: str | None = Field(default=None, description="Honors, GPA, or distinctions")


class PortfolioCertification(BaseModel):
    """Professional certification or license."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    id: str = Field(default_factory=default_id, description="Unique certification identifier")
    name: str = Field(description="Certification title")
    issuer: str = Field(description="Issuing organization (e.g. 'AWS', 'Google Cloud')")
    issue_date: str | None = Field(default=None, description="Date earned or year")
    credential_url: str | None = Field(default=None, description="Verification URL")


class PortfolioSkillGroup(BaseModel):
    """Categorized technical skills."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    id: str = Field(default_factory=default_id, description="Unique skill group identifier")
    category: str = Field(description="Category name (e.g. 'Languages', 'Cloud & Infra')")
    skills: list[str] = Field(default_factory=list, description="List of technical skills")


class PortfolioContent(BaseModel):
    """Canonical content payload of a developer portfolio (Schema Version 2)."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    schema_version: int = Field(default=2, description="Portfolio schema version")
    name: str = Field(description="Full name")
    headline: str = Field(description="Sharp technical headline, max 90 chars")
    bio: str = Field(description="Editorial 1st-person bio (100-160 words, 2 paragraphs)")
    location: str | None = Field(default=None, description="City, State, or Country")
    image_url: str | None = Field(default=None, description="Profile avatar URL")
    open_to_work: bool = Field(default=True, description="Open to opportunities badge")
    socials: PortfolioSocials = Field(default_factory=PortfolioSocials)
    privacy: PortfolioPrivacy = Field(default_factory=PortfolioPrivacy)
    highlights: list[PortfolioHighlight] = Field(default_factory=list)
    projects: list[PortfolioProject] = Field(default_factory=list)
    experience: list[PortfolioExperience] = Field(default_factory=list)
    skill_groups: list[PortfolioSkillGroup] = Field(default_factory=list)
    education: list[PortfolioEducation] = Field(default_factory=list)
    certifications: list[PortfolioCertification] = Field(default_factory=list)


class PortfolioGenerationOutput(BaseModel):
    """Structured LLM output for portfolio generation."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    name: str = Field(description="Full name of candidate")
    headline: str = Field(description="Sharp technical headline (max 90 chars)")
    bio: str = Field(description="Two editorial 1st-person paragraphs (100-160 words)")
    location: str | None = Field(default=None, description="Candidate location")
    open_to_work: bool = Field(default=True, description="Whether open to opportunities")
    socials: PortfolioSocials = Field(default_factory=PortfolioSocials)
    highlights: list[PortfolioHighlight] = Field(
        default_factory=list,
        description="3-4 concrete impact metrics grounded strictly in resume data",
    )
    projects: list[PortfolioProject] = Field(
        default_factory=list,
        description="Featured technical projects curated from experience",
    )
    experience: list[PortfolioExperience] = Field(
        default_factory=list,
        description="Career history with concise impact bullets (max 4 per role)",
    )
    skill_groups: list[PortfolioSkillGroup] = Field(
        default_factory=list,
        description="3-6 logical skill categories with specific technical skills",
    )
    education: list[PortfolioEducation] = Field(default_factory=list)
    certifications: list[PortfolioCertification] = Field(default_factory=list)


class PortfolioGenerateJobRequest(BaseModel):
    """Payload to trigger background portfolio generation."""

    uid: str = Field(description="Firebase user UID")
    model: str = Field(default="gemini", description="AI provider ('gemini' or 'mistral')")
    preserve_custom_avatar: bool = Field(
        default=True, description="Keep existing avatar URL if present"
    )


class PortfolioEnhanceRequest(BaseModel):
    """Payload to enhance/polish a specific portfolio section."""

    uid: str = Field(description="Firebase user UID")
    section: Literal["headline", "bio", "project_description", "experience_bullets"] = Field(
        description="Target portfolio section"
    )
    current_text: str = Field(description="Existing text or bullet points")
    instruction: str | None = Field(
        default=None, description="Optional user prompt or refinement instruction"
    )
    model: str = Field(default="gemini", description="AI provider ('gemini' or 'mistral')")


class PortfolioEnhanceOutput(BaseModel):
    """Structured LLM output for section enhancement."""

    enhanced_text: str = Field(description="Polished narrative text or headline")
    bullets: list[str] = Field(
        default_factory=list, description="Refined bullet points if applicable"
    )


class PortfolioEnhanceResponse(BaseModel):
    """API response for section enhancement."""

    success: bool = True
    section: str
    enhanced_text: str
    bullets: list[str] = Field(default_factory=list)
