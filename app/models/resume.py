import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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
    """Request payload for resume parsing from a URL or uploaded Blob."""

    uid: str = Field(
        min_length=1,
        description="Firebase user UID that owns the parsed career profile",
    )
    url: str | None = Field(
        default=None,
        description="LinkedIn profile URL or public resume/document URL",
    )
    blob_id: str | None = Field(
        default=None,
        description="Vercel Blob pathname for an uploaded private resume",
    )
    blob_content: bytes | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Internal pre-fetched Blob bytes passed from the API process to its parse job",
    )
    source_text: str | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Source text preflighted by the API before an asynchronous combine job",
    )
    source_type: Literal["pdf", "doc", "docx", "linkedin", "html"] | None = Field(default=None, exclude=True, repr=False)
    model: str = Field(
        default="auto",
        description=(
            "Extraction model/method: 'auto' (uses python-mapping for LinkedIn, gemini for PDF/HTML), "
            "'python-mapping' (instant deterministic mapping, zero-LLM, no PII vault), "
            "'gemini' (AI extraction with gemini-3.5-flash-lite + PIIVault protection), "
            "or 'mistral' (AI extraction + PIIVault protection)"
        ),
    )
    persist_profile: bool = Field(
        default=True,
        description="Whether the parse worker should directly save the result as the user's profile",
    )

    @model_validator(mode="after")
    def validate_source(self) -> "ResumeParseRequest":
        self.uid = self.uid.strip()
        if not self.uid or "/" in self.uid:
            raise ValueError("uid must be a non-empty Firebase user ID")
        if self.blob_id:
            if self.url:
                raise ValueError("Provide either url or blob_id, not both")
        elif not self.url:
            raise ValueError("Provide a public url or a private Blob source")
        return self


class ResumeCombineRequest(BaseModel):
    """Two structured profiles supplied to the AI combine step."""

    current_profile: ResumeSchema
    new_profile: ResumeSchema
    model: str = Field(default="gemini", description="Configured AI model provider")


class ResumeCombineJobRequest(BaseModel):
    """Source and current profile for an asynchronous import-and-combine job."""

    uid: str = Field(min_length=1, description="Firebase user UID that owns the profile")
    url: str | None = Field(default=None, description="Public resume or profile URL")
    blob_id: str | None = Field(default=None, description="Private Vercel Blob pathname")
    model: str = Field(default="auto", description="Model used to parse the new source")
    combine_model: str = Field(default="gemini", description="Model used to combine profiles")

    @model_validator(mode="after")
    def validate_source(self) -> "ResumeCombineJobRequest":
        self.uid = self.uid.strip()
        if not self.uid or "/" in self.uid:
            raise ValueError("uid must be a non-empty Firebase user ID")
        if bool(self.url) == bool(self.blob_id):
            raise ValueError("Provide exactly one of url or blob_id")
        if self.url and not self.url.startswith(("https://", "http://")):
            raise ValueError("Resume URL must use HTTP or HTTPS")
        return self


class ResumeCombineResponse(BaseModel):
    """Schema-validated combined career profile."""

    success: bool = True
    profile: ResumeSchema
    model_used: str


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
