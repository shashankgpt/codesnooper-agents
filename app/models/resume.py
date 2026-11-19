import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

TRUNCATED_PATTERN = re.compile(
    r"(?:\.{2,}|…|&hellip;)\s*$|"
    r"\b(into|in|to|with|and|or|for|of|at|by|from|as|the|a|an)\s*(?:\.{2,}|…|&hellip;)?\s*$",
    re.IGNORECASE,
)


def is_incomplete_text(text: str | None) -> bool:
    """Return True if the text is truncated, cut off, or ends with an ellipsis."""
    if not text:
        return False
    clean = text.strip()
    if not clean:
        return False
    return bool(TRUNCATED_PATTERN.search(clean))


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

    @field_validator("description", mode="after")
    @classmethod
    def validate_description(cls, v: str | None) -> str | None:
        """Discard incomplete or cut-off descriptions ending with ellipsis or cut off."""
        if v and is_incomplete_text(v):
            return None
        return v

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


def generate_resume_raw_text(
    *,
    full_name: str | None = None,
    email: str | None = None,
    phone: str | None = None,
    location: str | None = None,
    linkedin_url: str | None = None,
    github_url: str | None = None,
    portfolio_url: str | None = None,
    summary: str | None = None,
    narrative_context: str | None = None,
    skills: list[str] | None = None,
    experience: list[Any] | None = None,
    education: list[Any] | None = None,
    certificates: list[Any] | None = None,
    projects: list[Any] | None = None,
    languages: list[str] | None = None,
) -> str:
    """Generate a comprehensive plain-text document of all user details for future referencing."""
    sections: list[str] = []

    # 1. Header & Contact Details
    header_lines: list[str] = []
    if full_name and full_name.strip():
        header_lines.append(f"# {full_name.strip()}")
    contact_parts: list[str] = []
    if email and email.strip():
        contact_parts.append(f"Email: {email.strip()}")
    if phone and phone.strip():
        contact_parts.append(f"Phone: {phone.strip()}")
    if location and location.strip():
        contact_parts.append(f"Location: {location.strip()}")
    if linkedin_url and linkedin_url.strip():
        contact_parts.append(f"LinkedIn: {linkedin_url.strip()}")
    if github_url and github_url.strip():
        contact_parts.append(f"GitHub: {github_url.strip()}")
    if portfolio_url and portfolio_url.strip():
        contact_parts.append(f"Portfolio: {portfolio_url.strip()}")
    if contact_parts:
        header_lines.append(" | ".join(contact_parts))
    if header_lines:
        sections.append("\n".join(header_lines))

    # 2. Professional Summary
    if summary and summary.strip():
        sections.append(f"## Professional Summary\n{summary.strip()}")

    # 2b. Additional Context & Highlights (Narrative Context)
    if narrative_context and narrative_context.strip():
        sections.append(f"## Additional Context & Highlights\n{narrative_context.strip()}")

    # 3. Technical & Professional Skills
    if skills:
        cleaned_skills = [str(s).strip() for s in skills if s and str(s).strip()]
        if cleaned_skills:
            sections.append(f"## Skills\n{', '.join(cleaned_skills)}")

    # 4. Work Experience
    if experience:
        exp_lines = ["## Work Experience"]
        for exp in experience:
            exp_dict = (
                exp.model_dump()
                if hasattr(exp, "model_dump")
                else (exp if isinstance(exp, dict) else {})
            )
            title = (exp_dict.get("title") or "Position").strip()
            company = (exp_dict.get("company") or "Company").strip()
            exp_loc = (exp_dict.get("location") or "").strip()
            start_date = (exp_dict.get("start_date") or "").strip()
            end_date = (
                exp_dict.get("end_date") or ("Present" if exp_dict.get("is_current") else "")
            ).strip()

            date_str = ""
            if start_date and end_date:
                date_str = f" ({start_date} - {end_date})"
            elif start_date or end_date:
                date_str = f" ({start_date or end_date})"

            loc_str = f" | {exp_loc}" if exp_loc else ""
            exp_lines.append(f"\n### {title} at {company}{date_str}{loc_str}")

            desc = (exp_dict.get("description") or "").strip()
            if desc:
                exp_lines.append(desc)

            highlights = exp_dict.get("highlights") or []
            if isinstance(highlights, list):
                for h in highlights:
                    if h and str(h).strip():
                        exp_lines.append(f"- {str(h).strip()}")
        if len(exp_lines) > 1:
            sections.append("\n".join(exp_lines))

    # 5. Education
    if education:
        edu_lines = ["## Education"]
        for edu in education:
            edu_dict = (
                edu.model_dump()
                if hasattr(edu, "model_dump")
                else (edu if isinstance(edu, dict) else {})
            )
            degree = (edu_dict.get("degree") or "").strip()
            field = (edu_dict.get("field_of_study") or edu_dict.get("field") or "").strip()
            institution = (
                edu_dict.get("institution") or edu_dict.get("school") or "Institution"
            ).strip()
            start_date = (edu_dict.get("start_date") or "").strip()
            end_date = (edu_dict.get("end_date") or "").strip()

            date_str = ""
            if start_date and end_date:
                date_str = f" ({start_date} - {end_date})"
            elif start_date or end_date:
                date_str = f" ({start_date or end_date})"

            if degree and field:
                deg_field = f"{degree} in {field}"
            elif degree:
                deg_field = degree
            elif field:
                deg_field = field
            else:
                deg_field = ""
            header = f"\n### {deg_field} - {institution}" if deg_field else f"\n### {institution}"
            edu_lines.append(f"{header}{date_str}")

            grade = (edu_dict.get("grade_or_gpa") or "").strip()
            if grade:
                edu_lines.append(f"Grade/Honors: {grade}")

            desc = (edu_dict.get("description") or "").strip()
            if desc:
                edu_lines.append(desc)
        if len(edu_lines) > 1:
            sections.append("\n".join(edu_lines))

    # 6. Projects
    if projects:
        proj_lines = ["## Projects"]
        for proj in projects:
            proj_dict = (
                proj.model_dump()
                if hasattr(proj, "model_dump")
                else (proj if isinstance(proj, dict) else {})
            )
            name = (proj_dict.get("name") or "Project").strip()
            url = (proj_dict.get("url") or "").strip()
            url_str = f" ({url})" if url else ""
            proj_lines.append(f"\n### {name}{url_str}")

            desc = (proj_dict.get("description") or "").strip()
            if desc:
                proj_lines.append(desc)

            techs = proj_dict.get("technologies") or []
            if techs:
                cleaned_techs = [str(t).strip() for t in techs if t and str(t).strip()]
                if cleaned_techs:
                    proj_lines.append(f"Technologies: {', '.join(cleaned_techs)}")
        if len(proj_lines) > 1:
            sections.append("\n".join(proj_lines))

    # 7. Certifications
    if certificates:
        cert_lines = ["## Certifications"]
        for cert in certificates:
            cert_dict = (
                cert.model_dump()
                if hasattr(cert, "model_dump")
                else (cert if isinstance(cert, dict) else {})
            )
            name = (cert_dict.get("name") or "Certification").strip()
            issuer = (
                cert_dict.get("issuing_organization") or cert_dict.get("issuer") or ""
            ).strip()
            date = (cert_dict.get("issue_date") or "").strip()
            url = (cert_dict.get("credential_url") or "").strip()

            parts = [name]
            if issuer:
                parts.append(f"Issued by {issuer}")
            if date:
                parts.append(f"Date: {date}")
            if url:
                parts.append(f"Credential: {url}")
            cert_lines.append(f"- {' | '.join(parts)}")
        if len(cert_lines) > 1:
            sections.append("\n".join(cert_lines))

    # 8. Languages
    if languages:
        cleaned_langs = [str(lang).strip() for lang in languages if lang and str(lang).strip()]
        if cleaned_langs:
            sections.append(f"## Languages\n{', '.join(cleaned_langs)}")

    return "\n\n".join(sections).strip()


def clean_narrative_context(v: str | None) -> str | None:
    """Filter out truncated lines while preserving valid multi-line narrative highlights."""
    if v is None:
        return None
    kept_lines: list[str] = []
    for raw_line in v.splitlines():
        stripped = raw_line.strip()
        if stripped and is_incomplete_text(stripped):
            continue
        kept_lines.append(raw_line.rstrip())
    # Drop trailing or consecutive orphan ### headings whose bullets were all filtered out
    pruned: list[str] = []
    for line in kept_lines:
        if line.lstrip().startswith("#"):
            while pruned and pruned[-1] == "":
                pruned.pop()
            if pruned and pruned[-1].lstrip().startswith("#"):
                pruned.pop()
        pruned.append(line)
    while pruned and (pruned[-1] == "" or pruned[-1].lstrip().startswith("#")):
        pruned.pop()
    cleaned = "\n".join(pruned).strip()
    return cleaned or None


class ReconciledNarrativeOutput(BaseModel):
    """Lightweight structured output for authoritative narrative_context reconciliation."""

    narrative_context: str | None = Field(
        default=None,
        description=(
            "Updated Markdown highlights keeping valid unmapped context while removing "
            "any facts contradicted, deleted, or already duplicated in structured fields. "
            "Return null if nothing unmapped remains."
        ),
    )

    @field_validator("narrative_context", mode="after")
    @classmethod
    def validate_narrative_context(cls, v: str | None) -> str | None:
        """Filter out truncated lines while preserving valid multi-line narrative highlights."""
        return clean_narrative_context(v)


class ResumeExtractionSchema(BaseModel):
    """Schema passed to llm.with_structured_output() — excludes raw_text to save output tokens."""

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
    narrative_context: str | None = Field(
        default=None,
        description=(
            "Unstructured or narrative highlights, recommendations, publications, honors, "
            "or freeform bio context that do not map cleanly to standard schema entities"
        ),
    )

    def to_resume_schema(self) -> "ResumeSchema":
        """Convert extraction schema to ResumeSchema and refresh raw text."""
        return ResumeSchema.model_validate(self.model_dump()).refresh_raw_text()

    @field_validator("summary", mode="after")
    @classmethod
    def validate_summary(cls, v: str | None) -> str | None:
        """Discard incomplete or cut-off summary ending with ellipsis or cut off."""
        if v and is_incomplete_text(v):
            return None
        return v

    @field_validator("narrative_context", mode="after")
    @classmethod
    def validate_narrative_context(cls, v: str | None) -> str | None:
        """Filter out truncated lines while preserving valid multi-line narrative highlights."""
        return clean_narrative_context(v)

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


class ResumeSchema(ResumeExtractionSchema):
    """Structured output schema extracted from a resume or profile, including raw_text."""

    raw_text: str | None = Field(
        default=None,
        description="Plain text summary of all user details for future AI referencing",
    )

    def generate_raw_text(self) -> str:
        """Construct a comprehensive plain text representation of all user details."""
        return generate_resume_raw_text(
            full_name=self.full_name,
            email=self.email,
            phone=self.phone,
            location=self.location,
            linkedin_url=self.linkedin_url,
            github_url=self.github_url,
            portfolio_url=self.portfolio_url,
            summary=self.summary,
            narrative_context=self.narrative_context,
            skills=self.skills,
            experience=self.experience,
            education=self.education,
            certificates=self.certificates,
            projects=self.projects,
            languages=self.languages,
        )

    def refresh_raw_text(self) -> "ResumeSchema":
        """Recompute raw_text from structured fields and return self."""
        self.raw_text = self.generate_raw_text() or None
        return self

    @model_validator(mode="after")
    def populate_raw_text(self) -> "ResumeSchema":
        """Auto-populate raw_text if not explicitly provided."""
        if not self.raw_text:
            generated = self.generate_raw_text()
            if generated:
                self.raw_text = generated
        return self


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
    source_type: Literal["pdf", "doc", "docx", "linkedin", "html"] | None = Field(
        default=None, exclude=True, repr=False
    )
    model: str = Field(
        default="auto",
        description=(
            "Extraction model/method: 'auto' (uses python-mapping for LinkedIn, gemini for "
            "PDF/HTML), 'python-mapping' (instant deterministic mapping, zero-LLM, no PII vault), "
            "'gemini' (AI extraction with gemini-3.5-flash-lite + PIIVault protection), "
            "or 'mistral' (AI extraction + PIIVault protection)"
        ),
    )
    persist_profile: bool = Field(
        default=True,
        description="Whether parse worker should directly save result as user's profile",
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


SyncMode = Literal["authoritative", "additive"]
SyncSourceKind = Literal["career_profile", "resume", "portfolio"]


class ResumeSyncRequest(BaseModel):
    """Controller-level sync payload (pure domain input, no Firestore dependency)."""

    current_profile: ResumeSchema | None = None
    profile_updates: dict[str, object] = Field(
        description="Sanitized JSON updates from Career Profile, Resume Builder, or Portfolio"
    )
    mode: SyncMode = Field(default="additive")
    source_kind: SyncSourceKind = Field(default="career_profile")
    model: str = Field(default="gemini")


class ResumeSyncJobRequest(BaseModel):
    """HTTP / background job payload for POST /api/v1/resume/sync."""

    uid: str = Field(min_length=1, description="Firebase user UID that owns the profile")
    mode: SyncMode = Field(default="additive")
    source_kind: SyncSourceKind = Field(default="career_profile")
    profile_updates: dict[str, object] = Field(
        description="Sanitized JSON updates from Career Profile, Resume Builder, or Portfolio"
    )
    model: str = Field(default="gemini")

    @model_validator(mode="after")
    def validate_request(self) -> "ResumeSyncJobRequest":
        self.uid = self.uid.strip()
        if not self.uid or "/" in self.uid:
            raise ValueError("uid must be a non-empty Firebase user ID")
        if not self.profile_updates:
            raise ValueError("profile_updates must not be empty")
        return self


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
    raw_text: str | None = Field(
        default=None,
        description="Plain text summary of all user details for future AI referencing",
    )
    error: str | None = Field(default=None, description="Error message if parsing failed")
