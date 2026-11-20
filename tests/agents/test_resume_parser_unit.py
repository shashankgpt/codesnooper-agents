from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.agents.resume_parser_agent import (
    ResumeParserState,
    build_resume_parser_graph,
    classify_url,
    detect_source_node,
    route_after_fetch,
)
from app.config import get_settings
from app.models.resume import (
    Certificate,
    Education,
    Experience,
    Project,
    ResumeParseRequest,
    ResumeSchema,
)


def test_classify_url() -> None:
    """Verify URL classification correctly categorizes PDF, LinkedIn, and HTML."""
    assert classify_url("https://example.com/resumes/johndoe.pdf") == "pdf"
    assert classify_url("https://site.org/downloads/cv.pdf?download=true") == "pdf"
    assert classify_url("https://www.linkedin.com/in/jane-doe-12345/") == "linkedin"
    assert classify_url("https://linkedin.com/in/alex/") == "linkedin"
    assert classify_url("https://myportfolio.dev/resume") == "html"
    assert classify_url("https://medium.com/@coder") == "html"


def test_linkedin_formatter_strips_extraneous_sections() -> None:
    """Verify format_profile_as_text strips activity, people_also_viewed, and social bloat."""
    from app.agents.linkedin_scraper import format_profile_as_text

    mock_profile = {
        "name": "Jane Doe",
        "position": "Staff Engineer",
        "location": "Bengaluru",
        "experience": [
            {"title": "Staff Engineer", "company": "Tech Corp", "dates_employed": "2020-Present"}
        ],
        "activity": [
            {"title": "Check out my new post on AI", "link": "https://linkedin.com/post/123"}
        ],
        "people_also_viewed": [
            {"name": "Other Person", "profile_link": "https://linkedin.com/in/other"}
        ],
        "similar_profiles": [{"name": "Similar Person"}],
        "articles": [{"title": "My Article"}],
        "posts": [{"title": "My Post"}],
    }

    result = format_profile_as_text(mock_profile)

    assert "Jane Doe" in result
    assert "Staff Engineer" in result
    assert "activity" not in result.lower()
    assert "people_also_viewed" not in result
    assert "Other Person" not in result
    assert "similar_profiles" not in result
    assert "Similar Person" not in result


def test_resume_schema_structure() -> None:
    """Verify ResumeSchema validates all required resume fields."""
    resume = ResumeSchema(
        full_name="Sarah Connor",
        email="sarah.connor@example.com",
        phone="+1 555-0199",
        location="Los Angeles, CA",
        summary=(
            "Experienced Systems Engineer with 8+ years designing fault-tolerant architectures."
        ),
        skills=["Python", "FastAPI", "Kubernetes", "Docker", "GCP"],
        experience=[
            Experience(
                company="Cyberdyne Systems",
                title="Lead AI Engineer",
                location="San Francisco, CA",
                start_date="2021-03",
                end_date="Present",
                is_current=True,
                description="Spearheaded distributed inference systems.",
                highlights=["Reduced latency by 45%", "Built automated rollback pipelines"],
            )
        ],
        education=[
            Education(
                institution="MIT",
                degree="Bachelor of Science",
                field_of_study="Computer Science",
                end_date="2016",
            )
        ],
        certificates=[
            Certificate(
                name="Google Cloud Professional Architect",
                issuing_organization="Google Cloud",
                issue_date="2022",
            )
        ],
        projects=[
            Project(
                name="CodeSnooper Agents",
                description="Autonomous multi-agent workflows",
                technologies=["Python", "LangChain", "LangGraph"],
            )
        ],
        languages=["English", "Spanish"],
        linkedin_url="https://linkedin.com/in/sarahconnor",
        github_url="https://github.com/sarahconnor",
    )

    assert resume.full_name == "Sarah Connor"
    assert resume.email == "sarah.connor@example.com"
    assert len(resume.skills) == 5
    assert len(resume.experience) == 1
    assert resume.experience[0].company == "Cyberdyne Systems"
    assert len(resume.certificates) == 1
    assert resume.certificates[0].name == "Google Cloud Professional Architect"


def test_detect_source_node() -> None:
    """Verify graph detect_source node sets url_type properly."""
    state: ResumeParserState = {"url": "https://company.org/cv.pdf"}
    updated = detect_source_node(state)
    assert updated.get("url_type") == "pdf"


def test_route_after_fetch() -> None:
    """Verify conditional edge routing."""
    # Succeeded fetch
    valid_state: ResumeParserState = {
        "url": "https://example.com/cv.pdf",
        "raw_content": "Some resume text",
        "error": None,
    }
    assert route_after_fetch(valid_state) == "parse_resume"

    # Failed fetch
    failed_state: ResumeParserState = {
        "url": "https://example.com/cv.pdf",
        "raw_content": None,
        "error": "404 Not Found",
    }
    assert route_after_fetch(failed_state) == "__end__"

    empty_state: ResumeParserState = {
        "url": "https://example.com/cv.pdf",
        "raw_content": "",
        "error": None,
    }
    assert route_after_fetch(empty_state) == "__end__"


def test_build_resume_parser_graph() -> None:
    """Verify LangGraph compiles without error."""
    graph = build_resume_parser_graph()
    assert graph is not None


def test_resume_parse_api_endpoint_mocked(client: TestClient) -> None:
    """Verify POST /api/v1/resume/parse handles requests and returns ResumeParseJobResponse."""
    with (
        patch(
            "app.views.resume.firebase_service.async_create_or_get_resume_job",
            new_callable=AsyncMock,
            return_value=("test_job_123", True),
        ),
        patch(
            "app.views.resume.firebase_service.async_set",
            new_callable=AsyncMock,
        ),
        patch(
            "app.views.resume.spawn_resume_parse",
            new_callable=AsyncMock,
        ) as mock_spawn,
        patch("app.core.middleware.get_settings") as mock_get_settings,
    ):
        mock_settings = MagicMock()
        mock_settings.m2m_signing_secrets_for_verification = ()
        mock_settings.gateway_tokens = ("test_secret_gateway_token_12345",)
        mock_settings.api_gateway_token = "test_secret_gateway_token_12345"
        mock_get_settings.return_value = mock_settings
        response = client.post(
            "/api/v1/resume/parse",
            headers={
                "Authorization": "Bearer test_secret_gateway_token_12345",
                "X-User-Id": "test_uid_123",
            },
            json={
                "uid": "test_uid_123",
                "url": "https://example.com/alex_resume.pdf",
                "model": "gemini",
            },
        )

        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        mock_spawn.assert_awaited_once()


def test_pii_vault_restores_naked_and_bracketed_placeholders() -> None:
    """Verify PIIVault restores naked placeholders, single braces, and double braces."""
    from app.core.pii_anonymizer import PIIVault

    vault = PIIVault()
    vault.register("EMAIL", "alex.morgan@example.com")
    vault.register("PHONE", "+1 415-555-2671")
    vault.register("NAME", "Alex Morgan")

    # Simulate LLM output where email and phone have brackets stripped or leading +
    raw_resume = ResumeSchema(
        full_name="{{PII_NAME_3}}",
        email="PII_EMAIL_1",
        phone="PII_PHONE_2",
    )

    restored = vault.restore_resume(raw_resume)
    assert restored.full_name == "Alex Morgan"
    assert restored.email == "alex.morgan@example.com"
    assert restored.phone == "+1 415-555-2671"


def test_pii_vault_fails_on_unrestored_residual_placeholder() -> None:
    """Verify PIIVault raises ValueError and find_residuals detects unrestored placeholders."""
    import pytest

    from app.core.pii_anonymizer import PIIVault

    vault = PIIVault()
    # Vault only knows EMAIL_1
    vault.register("EMAIL", "alex.morgan@example.com")

    # LLM hallucinated or returned an unrestorable placeholder PII_EMAIL_99
    unrestored_resume = ResumeSchema(
        full_name="Alex Morgan",
        email="PII_EMAIL_99",
        phone="PII_PHONE_88",
    )

    # find_residuals should detect both
    residuals = PIIVault.find_residuals(unrestored_resume)
    assert "PII_EMAIL_99" in residuals
    assert "PII_PHONE_88" in residuals

    # restore_resume must raise ValueError instead of leaking PII placeholders
    with pytest.raises(ValueError, match="residual placeholders"):
        vault.restore_resume(unrestored_resume)


def test_resume_schema_normalizes_dummy_phone_numbers() -> None:
    """Verify ResumeSchema discards dummy placeholder phone numbers like +91-9999999999."""
    resume_dummy_9s = ResumeSchema(
        full_name="Alex Morgan",
        email="alex.morgan@example.com",
        phone="+91-9999999999",
    )
    assert resume_dummy_9s.phone is None

    resume_dummy_0s = ResumeSchema(
        full_name="Alex Morgan",
        email="alex.morgan@example.com",
        phone="+91-0000000000",
    )
    assert resume_dummy_0s.phone is None

    resume_valid = ResumeSchema(
        full_name="Alex Morgan",
        email="alex.morgan@example.com",
        phone="+1 415-555-2671",
    )
    assert resume_valid.phone == "+1 415-555-2671"


def test_resume_schema_discards_incomplete_summary_and_description() -> None:
    """Verify ResumeSchema sets incomplete summaries and descriptions to None."""
    from app.models.resume import Experience, ResumeSchema, is_incomplete_text

    incomplete_summary = (
        "I'm a Lead Engineer with 10+ years of experience modernizing legacy systems into…"
    )
    assert is_incomplete_text(incomplete_summary) is True
    assert is_incomplete_text("Experienced engineer specializing in cloud systems.") is False

    resume = ResumeSchema(
        full_name="Alex Morgan",
        summary=incomplete_summary,
        experience=[
            Experience(
                company="TechCorp",
                title="Lead Engineer",
                description="Modernizing legacy systems into…",
            )
        ],
    )
    assert resume.summary is None
    assert resume.experience[0].description is None

    # Ellipsis with dots
    resume_dots = ResumeSchema(
        full_name="Alex Morgan",
        summary="Working on distributed systems and...",
    )
    assert resume_dots.summary is None

    # Complete summary is preserved
    complete_summary = "Lead Engineer with 10+ years of experience modernizing legacy systems."
    resume_complete = ResumeSchema(
        full_name="Alex Morgan",
        summary=complete_summary,
    )
    assert resume_complete.summary == complete_summary


def test_map_linkedin_dict_to_resume_schema() -> None:
    """Verify map_linkedin_dict_to_resume_schema accurately maps Bright Data JSON."""
    from app.agents.linkedin_scraper import map_linkedin_dict_to_resume_schema

    mock_profile = {
        "name": "Alex Morgan",
        "position": "Lead Software Engineer",
        "city": "Bengaluru, Karnataka, India",
        "about": "Lead Engineer with 10+ years experience. Contact: alex.morgan@example.com",
        "current_company": {"name": "TechCorp Solutions"},
        "experience": [
            {
                "title": "Lead Software Engineer",
                "company": "TechCorp Solutions",
                "location": "Bengaluru",
                "start_date": "2021-03",
                "end_date": "Present",
                "description": "- Led backend architecture\n- Reduced latency",
            },
            {
                "title": "Senior Engineer",
                "company": "Legacy Systems Ltd",
                "start_date": "2018-01",
                "end_date": "2021-02",
                "description": "API development",
            },
        ],
        "education": [
            {
                "title": "State University of Technology",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": "2012",
                "end_year": "2016",
            }
        ],
        "certifications": [
            {
                "title": "AWS Solutions Architect",
                "subtitle": "Amazon Web Services",
                "credential_url": "https://credly.com/badge/123",
            }
        ],
        "projects": [
            {
                "title": "Platform API",
                "description": "Multi-tenant auth service",
                "url": "https://github.com/example/api",
            }
        ],
        "skills": ["Python", "FastAPI", "React", "AWS"],
        "languages": [{"title": "English"}, {"title": "Hindi"}],
        "bio_links": [{"title": "Portfolio", "link": "https://alexmorgan.dev"}],
        "url": "https://www.linkedin.com/in/alexmorgan",
    }

    resume = map_linkedin_dict_to_resume_schema(mock_profile)

    assert resume.full_name == "Alex Morgan"
    assert resume.email == "alex.morgan@example.com"
    assert resume.location == "Bengaluru, Karnataka, India"
    assert len(resume.experience) == 2
    assert resume.experience[0].company == "TechCorp Solutions"
    assert resume.experience[0].is_current is True
    assert len(resume.education) == 1
    assert "State University" in (resume.education[0].institution or "")
    assert len(resume.certificates) == 1
    assert "AWS Solutions Architect" in (resume.certificates[0].name or "")
    assert len(resume.projects) == 1
    assert "Python" in resume.skills
    assert "English" in resume.languages
    assert resume.portfolio_url == "https://alexmorgan.dev"
    assert resume.linkedin_url == "https://www.linkedin.com/in/alexmorgan"
    assert resume.raw_text is not None
    assert "# Alex Morgan" in resume.raw_text
    assert "TechCorp Solutions" in resume.raw_text
    assert "Legacy Systems Ltd" in resume.raw_text

    # Incomplete about ending with ellipsis into… should not be used as summary
    mock_profile_incomplete = dict(mock_profile)
    mock_profile_incomplete["about"] = (
        "I'm a Lead Engineer with 10+ years of experience modernizing legacy systems into…"
    )
    mock_profile_incomplete["headline"] = ""
    mock_profile_incomplete["summary"] = ""
    mock_profile_incomplete["position"] = ""
    resume_incomplete = map_linkedin_dict_to_resume_schema(mock_profile_incomplete)
    assert resume_incomplete.summary is None


@pytest.mark.asyncio
async def test_resume_controller_routes_linkedin_to_python_mapping() -> None:
    """Verify ResumeController routes LinkedIn URLs to python-mapping without LLM."""
    from unittest.mock import AsyncMock, patch

    from app.controllers.resume_controller import ResumeController

    mock_profile = {
        "name": "Jane Developer",
        "city": "Austin, TX",
        "skills": ["Python", "Docker"],
        "experience": [{"title": "Engineer", "company": "Acme Inc"}],
    }

    controller = ResumeController(get_settings())
    req = ResumeParseRequest(uid="test_uid_123", url="https://www.linkedin.com/in/janedev/")

    with patch(
        "app.controllers.resume_controller.fetch_linkedin_raw_data",
        new_callable=AsyncMock,
    ) as mock_fetch:
        mock_fetch.return_value = mock_profile

        response = await controller.parse_resume(req)

        assert response.success is True
        assert response.model_used == "python-mapping"
        assert response.url_type == "linkedin"
        assert response.resume is not None
        assert response.resume.full_name == "Jane Developer"
        assert response.response_time is not None


@pytest.mark.asyncio
async def test_resume_controller_routes_linkedin_to_ai_with_pii_vault() -> None:
    """Verify ResumeController routes to AI agent and uses PIIVault when model='gemini'."""
    from unittest.mock import AsyncMock, patch

    from app.controllers.resume_controller import ResumeController

    controller = ResumeController(get_settings())
    req = ResumeParseRequest(
        uid="test_uid_123",
        url="https://www.linkedin.com/in/janedev/",
        model="gemini",
    )

    mock_raw_text = "Name: Jane Developer\nEmail: jane@example.com\nExperience: Acme Inc"
    mock_ai_resume = ResumeSchema(
        full_name="{{PII_NAME_2}}",
        email="{{PII_EMAIL_1}}",
        experience=[Experience(company="Acme Inc", title="Staff Engineer")],
    )

    with (
        patch(
            "app.controllers.resume_controller.fetch_source_text",
            new_callable=AsyncMock,
        ) as mock_fetch,
        patch.object(
            controller.graph,
            "ainvoke",
            new_callable=AsyncMock,
        ) as mock_invoke,
    ):
        mock_fetch.return_value = (mock_raw_text, "linkedin")
        mock_invoke.return_value = {
            "resume": mock_ai_resume,
            "error": None,
            "model_used": "gemini",
        }

        response = await controller.parse_resume(req)

        assert response.success is True
        assert response.model_used == "gemini"
        assert response.url_type == "linkedin"
        assert response.resume is not None
        # Assert PIIVault properly restored real values
        assert response.resume.full_name == "Jane Developer"
        assert response.resume.email == "jane@example.com"


@pytest.mark.asyncio
async def test_resume_controller_rejects_python_mapping_for_non_linkedin_urls() -> None:
    """Verify ResumeController rejects model='python-mapping' for PDF and HTML URLs."""
    from app.controllers.resume_controller import ResumeController

    controller = ResumeController(get_settings())
    req = ResumeParseRequest(
        uid="test_uid_123",
        url="https://example.com/resume.pdf",
        model="python-mapping",
    )

    response = await controller.parse_resume(req)

    assert response.success is False
    assert "Python mapping is only available for structured LinkedIn URLs" in (response.error or "")


@pytest.mark.asyncio
async def test_resume_controller_refetches_blob_when_blob_content_is_none() -> None:
    """Verify worker fetches from Vercel Blob when blob_content is not shipped in the payload."""
    from unittest.mock import AsyncMock, patch

    from app.controllers.resume_controller import ResumeController

    controller = ResumeController(get_settings())
    req = ResumeParseRequest(
        uid="test_user_blob",
        blob_id="career-profiles/test_user_blob/sample_resume.pdf",
        model="gemini",
    )

    mock_raw_text = "Name: Bob Worker\nEmail: bob@example.com\nSkills: Python"
    mock_ai_resume = ResumeSchema(
        full_name="{{PII_NAME_2}}",
        email="{{PII_EMAIL_1}}",
        skills=["Python"],
    )

    with (
        patch(
            "app.controllers.resume_controller.fetch_resume_blob",
            new_callable=AsyncMock,
        ) as mock_fetch_blob,
        patch(
            "app.controllers.resume_controller.fetch_source_text",
            new_callable=AsyncMock,
        ) as mock_fetch_text,
        patch.object(
            controller.graph,
            "ainvoke",
            new_callable=AsyncMock,
        ) as mock_invoke,
    ):
        mock_fetch_blob.return_value = (b"%PDF-1.4 dummy bytes", "pdf")
        mock_fetch_text.return_value = (mock_raw_text, "pdf")
        mock_invoke.return_value = {
            "resume": mock_ai_resume,
            "error": None,
            "model_used": "gemini",
        }

        response = await controller.parse_resume(req)

        assert response.success is True
        mock_fetch_blob.assert_awaited_once_with(
            "career-profiles/test_user_blob/sample_resume.pdf",
            "test_user_blob",
            controller.settings,
        )
        assert response.resume is not None
        assert response.resume.full_name == "Bob Worker"


@pytest.mark.asyncio
async def test_resume_controller_handles_base64_blob_content() -> None:
    """Verify worker handles base64-encoded blob_content string if passed in payload."""
    import base64
    from unittest.mock import AsyncMock, patch

    from app.controllers.resume_controller import ResumeController

    controller = ResumeController(get_settings())
    raw_bytes = b"%PDF-1.4 test bytes"
    b64_str = base64.b64encode(raw_bytes).decode("ascii")

    req = ResumeParseRequest(
        uid="test_user_b64",
        blob_id="career-profiles/test_user_b64/resume.pdf",
        model="gemini",
    )
    # Simulate deserialized payload with base64 string
    req.blob_content = b64_str  # type: ignore[assignment]

    mock_raw_text = "Name: Base64 Candidate\nEmail: b64@example.com"
    mock_ai_resume = ResumeSchema(
        full_name="{{PII_NAME_2}}",
        email="{{PII_EMAIL_1}}",
    )

    with (
        patch(
            "app.controllers.resume_controller.fetch_source_text",
            new_callable=AsyncMock,
        ) as mock_fetch_text,
        patch.object(
            controller.graph,
            "ainvoke",
            new_callable=AsyncMock,
        ) as mock_invoke,
    ):
        mock_fetch_text.return_value = (mock_raw_text, "pdf")
        mock_invoke.return_value = {
            "resume": mock_ai_resume,
            "error": None,
            "model_used": "gemini",
        }

        response = await controller.parse_resume(req)

        assert response.success is True
        mock_fetch_text.assert_awaited_once_with(
            "blob://career-profiles/test_user_b64/resume.pdf",
            "pdf",
            source_bytes=raw_bytes,
        )


def test_resume_schema_auto_generates_raw_text() -> None:
    """Verify ResumeSchema auto-generates raw_text containing all user details."""
    from app.models.resume import Certificate, Education, Experience, Project, ResumeSchema

    resume = ResumeSchema(
        full_name="Sarah Connor",
        email="sarah.connor@example.com",
        phone="+1 555-0144",
        location="Los Angeles, CA",
        summary="Principal systems engineer with deep distributed systems experience.",
        skills=["Python", "Go", "Kubernetes", "AWS"],
        experience=[
            Experience(
                company="Cyberdyne Systems",
                title="Lead Security Architect",
                location="Sunnyvale, CA",
                start_date="2020-01",
                end_date="Present",
                is_current=True,
                description="Engineered mission-critical security controls and network isolation.",
                highlights=["Reduced vulnerability exposure by 80%"],
            )
        ],
        education=[
            Education(
                institution="UCLA",
                degree="Bachelor of Science",
                field_of_study="Computer Science",
                start_date="2012",
                end_date="2016",
                grade_or_gpa="Honors in distributed computing",
            )
        ],
        certificates=[
            Certificate(
                name="AWS Certified Solutions Architect",
                issuing_organization="Amazon Web Services",
                issue_date="2021",
                credential_url="https://aws.amazon.com/verify/123",
            )
        ],
        projects=[
            Project(
                name="Resistance Net",
                description="Encrypted P2P mesh network for distributed resilient communication.",
                technologies=["Go", "libp2p", "WireGuard"],
                url="https://github.com/resistance/net",
            )
        ],
        languages=["English", "Spanish"],
        linkedin_url="https://linkedin.com/in/sarahconnor",
        github_url="https://github.com/sarahconnor",
        portfolio_url="https://sarahconnor.tech",
    )

    assert resume.raw_text is not None
    raw = resume.raw_text

    # Verify all critical sections exist in raw_text
    assert "# Sarah Connor" in raw
    assert "Email: sarah.connor@example.com" in raw
    assert "Phone: +1 555-0144" in raw
    assert "Location: Los Angeles, CA" in raw
    assert "LinkedIn: https://linkedin.com/in/sarahconnor" in raw
    assert "GitHub: https://github.com/sarahconnor" in raw
    assert "Portfolio: https://sarahconnor.tech" in raw

    assert "## Professional Summary" in raw
    assert "Principal systems engineer with deep distributed systems experience." in raw

    assert "## Skills" in raw
    assert "Python, Go, Kubernetes, AWS" in raw

    assert "## Work Experience" in raw
    assert (
        "### Lead Security Architect at Cyberdyne Systems (2020-01 - Present) | Sunnyvale, CA"
        in raw
    )
    assert "Engineered mission-critical security controls and network isolation." in raw
    assert "- Reduced vulnerability exposure by 80%" in raw

    assert "## Education" in raw
    assert "### Bachelor of Science in Computer Science - UCLA (2012 - 2016)" in raw
    assert "Grade/Honors: Honors in distributed computing" in raw

    assert "## Certifications" in raw
    assert (
        "AWS Certified Solutions Architect | Issued by Amazon Web Services | "
        "Date: 2021 | Credential: https://aws.amazon.com/verify/123"
    ) in raw

    assert "## Projects" in raw
    assert "### Resistance Net (https://github.com/resistance/net)" in raw
    assert "Technologies: Go, libp2p, WireGuard" in raw

    assert "## Languages" in raw
    assert "English, Spanish" in raw


def test_resume_schema_preserves_custom_raw_text() -> None:
    """Verify ResumeSchema preserves an explicit raw_text if provided."""
    from app.models.resume import ResumeSchema

    custom_text = "Custom plain text source of truth provided directly."
    resume = ResumeSchema(
        full_name="Jane Doe",
        email="jane@example.com",
        raw_text=custom_text,
    )
    assert resume.raw_text == custom_text


def test_resume_schema_narrative_context_rendering_and_refresh() -> None:
    """Verify narrative_context renders after Professional Summary and refresh_raw_text works."""
    from app.models.resume import ResumeSchema

    resume = ResumeSchema(
        full_name="Jane Doe",
        summary="Expert software engineer with 10 years experience.",
        narrative_context="Winner of Global Hackathon 2024.\nKeynote speaker at PyCon.",
        skills=["Python", "FastAPI"],
    )

    assert resume.raw_text is not None
    raw = resume.raw_text
    summary_idx = raw.find("## Professional Summary")
    narrative_idx = raw.find("## Additional Context & Highlights")
    skills_idx = raw.find("## Skills")

    assert summary_idx != -1
    assert narrative_idx != -1
    assert skills_idx != -1
    assert summary_idx < narrative_idx < skills_idx
    assert "Winner of Global Hackathon 2024." in raw

    # Test refresh_raw_text
    resume.narrative_context = "Updated narrative: Published research paper on AI agents."
    refreshed = resume.refresh_raw_text()
    assert refreshed is resume
    assert "Updated narrative: Published research paper on AI agents." in resume.raw_text
    assert "Winner of Global Hackathon 2024." not in resume.raw_text


def test_map_linkedin_dict_extracts_narrative_context() -> None:
    """Verify map_linkedin_dict extracts headline, recs, awards into narrative_context."""
    from app.agents.linkedin_scraper import map_linkedin_dict_to_resume_schema

    profile_dict = {
        "name": "Jane Doe",
        "headline": "Distinguished Architect & Fellow",
        "about": "Passionate about building large scale systems.",
        "recommendations": [
            {"recommender": "Alice Smith", "text": "Jane is an exceptional engineer."},
        ],
        "honors_and_awards": [
            {"title": "Best Innovator", "issuer": "Tech Corp", "description": "Outstanding impact"},
        ],
        "publications": [
            {
                "name": "Distributed Systems at Scale",
                "publisher": "ACM",
                "url": "https://doi.org/123",
            },
        ],
        "volunteer_experience": [
            {"role": "Mentor", "organization": "Girls Who Code", "cause": "Education"},
        ],
        "courses": [
            {"name": "Advanced Algorithms", "number": "CS401"},
        ],
    }

    schema = map_linkedin_dict_to_resume_schema(profile_dict, "https://linkedin.com/in/janedoe")
    assert schema.narrative_context is not None
    nc = schema.narrative_context
    assert "### Headline" in nc
    assert "Distinguished Architect & Fellow" in nc
    assert "### Recommendations" in nc
    assert "Alice Smith" in nc
    assert "Jane is an exceptional engineer." in nc
    assert "### Honors & Awards" in nc
    assert "Best Innovator" in nc
    assert "### Publications" in nc
    assert "Distributed Systems at Scale" in nc
    assert "### Volunteer Experience" in nc
    assert "Girls Who Code" in nc
    assert "### Courses" in nc
    assert "Advanced Algorithms" in nc

    # And verify narrative_context rendered in raw_text
    assert schema.raw_text is not None
    assert "## Additional Context & Highlights" in schema.raw_text


def test_cap_recommendation_text() -> None:
    """Verify cap_recommendation_text handles short, Dr. abbreviations, and long texts cleanly."""
    from app.agents.linkedin_scraper import cap_recommendation_text

    # 1. Short text unchanged
    assert cap_recommendation_text("Great engineer.") == "Great engineer."

    # 2. Text with Dr. abbreviation at beginning does not truncate prematurely
    dr_text = (
        "Dr. Smith worked closely with Jane on high-scale distributed systems and data pipelines. "
        "Her analytical capabilities and architectural insights significantly improved our team. "
        "She consistently demonstrated exemplary technical leadership throughout the multi-year "
        "engagement and delivered exceptional results across every complex milestone."
    )
    long_dr_text = dr_text + " " + dr_text  # > 500 chars
    capped = cap_recommendation_text(long_dr_text, limit=300)
    assert not capped.endswith("Dr.")
    assert capped.endswith(".")
    assert len(capped) <= 300
    assert "Dr. Smith" in capped

    # 3. Text with no period cuts cleanly at last word and appends period
    no_period_text = "word " * 120
    capped_no_period = cap_recommendation_text(no_period_text, limit=100)
    assert len(capped_no_period) <= 101
    assert capped_no_period.endswith(".")
    assert not capped_no_period.endswith(" .")


def test_validate_narrative_context_line_by_line_and_orphan_headings() -> None:
    """Verify line-by-line truncation filtering and orphan heading pruning."""
    from app.models.resume import ReconciledNarrativeOutput, ResumeSchema

    # Line 2 is truncated with ..., but line 1 and 3 are intact
    raw_nc = (
        "### Recommendations\n"
        "- Alice: Great engineer and leader.\n"
        "- Bob: Working on distributed sys...\n"
        "- Charlie: Exceptional problem solver.\n\n"
        "### Honors & Awards\n"
        "- Truncated award that cuts off..."
    )
    schema = ResumeSchema(full_name="Jane Doe", narrative_context=raw_nc)
    assert schema.narrative_context is not None
    nc = schema.narrative_context
    assert "- Alice: Great engineer and leader." in nc
    assert "- Charlie: Exceptional problem solver." in nc
    assert "Bob" not in nc
    # The Honors & Awards section only had a truncated bullet, so its header should be pruned
    assert "### Honors & Awards" not in nc

    # Verify ReconciledNarrativeOutput applies the same cleaner
    rec_out = ReconciledNarrativeOutput(narrative_context=raw_nc)
    assert rec_out.narrative_context == nc
