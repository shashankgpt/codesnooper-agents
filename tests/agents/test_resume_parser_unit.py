from unittest.mock import AsyncMock, patch

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
    vault.register("EMAIL", "shashank@example.com")
    vault.register("PHONE", "+91 9876543210")
    vault.register("NAME", "Shashank Gupta")

    # Simulate LLM output where email and phone have brackets stripped or leading +
    raw_resume = ResumeSchema(
        full_name="{{PII_NAME_3}}",
        email="PII_EMAIL_1",
        phone="PII_PHONE_2",
    )

    restored = vault.restore_resume(raw_resume)
    assert restored.full_name == "Shashank Gupta"
    assert restored.email == "shashank@example.com"
    assert restored.phone == "+91 9876543210"


def test_pii_vault_fails_on_unrestored_residual_placeholder() -> None:
    """Verify PIIVault raises ValueError and find_residuals detects unrestored placeholders."""
    import pytest
    from app.core.pii_anonymizer import PIIVault

    vault = PIIVault()
    # Vault only knows EMAIL_1
    vault.register("EMAIL", "shashank@example.com")

    # LLM hallucinated or returned an unrestorable placeholder PII_EMAIL_99
    unrestored_resume = ResumeSchema(
        full_name="Shashank Gupta",
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
        full_name="Shashank Gupta",
        email="shashankgpt270@gmail.com",
        phone="+91-9999999999",
    )
    assert resume_dummy_9s.phone is None

    resume_dummy_0s = ResumeSchema(
        full_name="Shashank Gupta",
        email="shashankgpt270@gmail.com",
        phone="+91-0000000000",
    )
    assert resume_dummy_0s.phone is None

    resume_valid = ResumeSchema(
        full_name="Shashank Gupta",
        email="shashankgpt270@gmail.com",
        phone="+91-9876543210",
    )
    assert resume_valid.phone == "+91-9876543210"



