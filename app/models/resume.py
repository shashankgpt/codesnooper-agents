import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Experience(BaseModel):
    """Work experience entry."""

    company: str | None = Field(default=None, description="Company or organization name")
    title: str | None = Field(default=None, description="Job title or role")
    location: str | None = Field(
        default=None, description="Location of the job (city, state, country or Remote)"
    )
    start_date: str | None = Field(
        default=None, description="Start date (e.g., 'Jan 2021', '2021-01')"
    )
    end_date: str | None = Field(default=None, description="End date or 'Present'")
    is_current: bool = Field(
        default=False, description="Whether this is the candidate's current role"
    )
    description: str | None = Field(
        default=None, description="Summary of responsibilities and achievements"
    )
    highlights: list[str] = Field(
        default_factory=list, description="Key accomplishment bullet points"
    )

    @field_validator("highlights", mode="before")
    @classmethod
    def normalize_highlights(cls, v: Any) -> list[str]:
        """Ensure highlights are always a list of clean strings, converting any nested dicts."""
        if not v:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            cleaned: list[str] = []
            for item in v:
                if isinstance(item, str):
                    cleaned.append(item)
                elif isinstance(item, dict):
                    # Join key-values: e.g. {"project": "SA Dashboard"} -> "project: SA Dashboard"
                    cleaned.append(", ".join(f"{k}: {val}" for k, val in item.items() if val))
                elif item is not None:
                    cleaned.append(str(item))
            return cleaned
        return [str(v)]


class Education(BaseModel):
    """Education history entry."""

    institution: str | None = Field(default=None, description="University, college, or school name")
    degree: str | None = Field(
        default=None, description="Degree obtained (e.g., 'Bachelor of Science', 'Master of Arts')"
    )
    field_of_study: str | None = Field(
        default=None, description="Major, concentration, or field of study"
    )
    start_date: str | None = Field(default=None, description="Start date")
    end_date: str | None = Field(default=None, description="End date or expected graduation year")
    grade_or_gpa: str | None = Field(default=None, description="GPA or grade honors if mentioned")


class Certificate(BaseModel):
    """Certification or license entry."""

    name: str | None = Field(default=None, description="Certification or course name")
    issuing_organization: str | None = Field(
        default=None, description="Organization issuing the certificate"
    )
    issue_date: str | None = Field(default=None, description="Date issued")
    expiration_date: str | None = Field(default=None, description="Expiration date if applicable")
    credential_id: str | None = Field(default=None, description="Credential ID or license number")
    credential_url: str | None = Field(default=None, description="Verification URL")


class Project(BaseModel):
    """Notable project entry."""

    name: str | None = Field(default=None, description="Project name")
    description: str | None = Field(default=None, description="Brief project overview")
    technologies: list[str] = Field(default_factory=list, description="Technologies or tools used")
    url: str | None = Field(default=None, description="Project or repository link")


class ResumeSchema(BaseModel):
    """Structured output schema extracted from a resume or profile."""

    full_name: str | None = Field(default=None, description="Candidate's full name")
    email: str | None = Field(default=None, description="Primary email address")
    phone: str | None = Field(default=None, description="Phone number with country/area code")
    location: str | None = Field(
        default=None, description="Current location (City, State, Country)"
    )
    summary: str | None = Field(
        default=None, description="Executive summary, career objective, or bio"
    )
    skills: list[str] = Field(
        default_factory=list, description="List of technical, analytical, and domain skills"
    )
    experience: list[Experience] = Field(
        default_factory=list,
        description=(
            "Exhaustive list of ALL past and current employment positions/companies throughout "
            "the candidate's career. Include every single company, employer, and role mentioned. "
            "Do NOT return only the most recent company."
        ),
    )
    education: list[Education] = Field(
        default_factory=list,
        description="All degrees, universities, and educational institutions attended",
    )
    certificates: list[Certificate] = Field(
        default_factory=list,
        description="All professional certifications, licenses, and credentials",
    )
    projects: list[Project] = Field(
        default_factory=list, description="Key projects or portfolio pieces"
    )
    languages: list[str] = Field(default_factory=list, description="Languages spoken")
    linkedin_url: str | None = Field(default=None, description="LinkedIn profile URL")
    github_url: str | None = Field(default=None, description="GitHub profile URL")
    portfolio_url: str | None = Field(default=None, description="Personal website or portfolio URL")

    @field_validator("phone", mode="after")
    @classmethod
    def validate_phone(cls, v: str | None) -> str | None:
        """Discard obvious dummy phone numbers like +91-9999999999 or repeating digits."""
        if not v:
            return None
        clean = v.strip()
        digits = re.sub(r"\D", "", clean)
        # Discard if at least 8 digits and all digits are identical (e.g. 9999999999 or 0000000000)
        if len(digits) >= 8:
            last_10 = digits[-10:] if len(digits) >= 10 else digits
            if len(set(last_10)) == 1:
                return None
        return clean


class ResumeParseRequest(BaseModel):
    """Request payload for resume parsing."""

    url: str = Field(
        description="URL to the resume (direct PDF link, LinkedIn profile, or resume webpage)"
    )
    model: str = Field(
        default="gemini",
        description="LLM provider/model to use for extraction: 'gemini' (default, using gemini-3.5-flash-lite) or 'mistral'",
    )


class ResumeParseResponse(BaseModel):
    """Standardized response payload for resume parsing."""

    success: bool = Field(description="Whether parsing succeeded")
    url: str = Field(description="Original URL submitted")
    url_type: str = Field(description="Detected URL type: 'pdf', 'linkedin', or 'html'")
    model_used: str = Field(description="Model used for extraction")
    response_time: str | None = Field(
        default=None,
        description="Total elapsed execution time formatted (e.g. '3.25s')",
    )
    latency_ms: float | None = Field(
        default=None,
        description="Total agent execution duration in milliseconds",
    )
    resume: ResumeSchema | None = Field(default=None, description="Parsed structured resume object")
    error: str | None = Field(default=None, description="Error message if parsing failed")
