from unittest.mock import AsyncMock, patch

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
    ResumeParseResponse,
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
    """Verify _format_profile_as_text strips activity, people_also_viewed, and social bloat."""
    from app.agents.linkedin_scraper import _format_profile_as_text

    mock_profile = {
        "name": "Jane Doe",
        "position": "Staff Engineer",
        "location": "Bengaluru",
        "experience": [{"title": "Staff Engineer", "company": "Tech Corp", "dates_employed": "2020-Present"}],
        "activity": [{"title": "Check out my new post on AI", "link": "https://linkedin.com/post/123"}],
        "people_also_viewed": [{"name": "Other Person", "profile_link": "https://linkedin.com/in/other"}],
        "similar_profiles": [{"name": "Similar Person"}],
        "articles": [{"title": "My Article"}],
        "posts": [{"title": "My Post"}],
    }

    result = _format_profile_as_text(mock_profile)

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
    """Verify POST /api/v1/resume/parse handles requests and returns ResumeParseResponse."""
    mock_resume = ResumeSchema(
        full_name="Alex Mercer",
        email="alex@example.com",
        skills=["Python", "Machine Learning"],
        summary="Senior ML Engineer",
        experience=[],
        education=[],
        certificates=[],
    )

    with patch(
        "app.controllers.resume_controller.ResumeController.parse_resume",
        new_callable=AsyncMock,
    ) as mock_parse:
        mock_parse.return_value = ResumeParseResponse(
            success=True,
            url="https://example.com/alex_resume.pdf",
            url_type="pdf",
            model_used="gemini",
            resume=mock_resume,
            error=None,
        )

        token = get_settings().api_gateway_token
        response = client.post(
            "/api/v1/resume/parse",
            headers={"Authorization": f"Bearer {token}"},
            json={"url": "https://example.com/alex_resume.pdf", "model": "gemini"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["url_type"] == "pdf"
        assert data["resume"]["full_name"] == "Alex Mercer"
        assert "Python" in data["resume"]["skills"]


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


def test_map_linkedin_dict_to_resume_schema() -> None:
    """Verify map_linkedin_dict_to_resume_schema accurately maps Bright Data JSON to ResumeSchema."""
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
    req = ResumeParseRequest(url="https://www.linkedin.com/in/janedev/")

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
    """Verify ResumeController routes to AI agent and uses PIIVault when model='gemini' is requested."""
    from unittest.mock import AsyncMock, patch
    from app.controllers.resume_controller import ResumeController

    controller = ResumeController(get_settings())
    req = ResumeParseRequest(url="https://www.linkedin.com/in/janedev/", model="gemini")

    mock_raw_text = "Name: Jane Developer\nEmail: jane@example.com\nExperience: Acme Inc"
    mock_ai_resume = ResumeSchema(
        full_name="{{PII_NAME_2}}",
        email="{{PII_EMAIL_1}}",
        experience=[Experience(company="Acme Inc", title="Staff Engineer")],
    )

    with patch(
        "app.controllers.resume_controller.fetch_source_text",
        new_callable=AsyncMock,
    ) as mock_fetch, patch.object(
        controller._graph,
        "ainvoke",
        new_callable=AsyncMock,
    ) as mock_invoke:
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
    req = ResumeParseRequest(url="https://example.com/resume.pdf", model="python-mapping")

    response = await controller.parse_resume(req)

    assert response.success is False
    assert "Python mapping is only available for structured LinkedIn URLs" in (response.error or "")




