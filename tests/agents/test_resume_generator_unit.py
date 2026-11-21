import re
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from app.config import Settings
from app.controllers.resume_generator_controller import ResumeGeneratorController
from app.core.pii_anonymizer import PIIVault
from app.models.resume_generator import (
    ResumeCustomSection,
    ResumeEducation,
    ResumeExperience,
    ResumeGenerateJobRequest,
    ResumeGenerationOutput,
    ResumePersonalInfo,
    ResumeProject,
    ResumeSkillCategory,
    format_html_bullets,
)
from app.prompts.resume_generator.v1 import build_resume_generator_prompts


def test_format_html_bullets_utility() -> None:
    """Verify format_html_bullets transforms plain text and bullet lists into clean HTML."""
    assert format_html_bullets("") == ""
    assert (
        format_html_bullets("<ul><li>Item 1</li><li>Item 2</li></ul>")
        == "<ul><li>Item 1</li><li>Item 2</li></ul>"
    )
    plain_text = "- Designed scalable Kafka consumer\n* Improved throughput by 40%\n• Reduced p99 latency to 12ms"
    formatted = format_html_bullets(plain_text)
    assert formatted.startswith("<ul>") and formatted.endswith("</ul>")
    assert "<li>Designed scalable Kafka consumer</li>" in formatted
    assert "<li>Improved throughput by 40%</li>" in formatted
    assert "<li>Reduced p99 latency to 12ms</li>" in formatted


def test_resume_models_schema_and_camelcase() -> None:
    """Verify ResumeGenerationOutput serializes to camelCase for Next.js BFF."""
    output = ResumeGenerationOutput(
        title="Alex Rivera — Senior Software Engineer Resume",
        personal_info=ResumePersonalInfo(
            full_name="Alex Rivera",
            email="alex.rivera@example.com",
            phone="+1 555-0199",
            location="San Francisco, CA",
            linkedin="https://linkedin.com/in/alexrivera",
            github="https://github.com/alexrivera",
            website="https://alexrivera.dev",
            summary="Senior Software Engineer with 8+ years experience designing distributed systems.",
            job_title="Senior Software Engineer",
        ),
        experience=[
            ResumeExperience(
                company="TechCorp",
                role="Senior Backend Engineer",
                start_date="Jan 2021",
                end_date="Present",
                current=True,
                description="- Architected event stream processing\n- Reduced costs by 30%",
                tech_stack=["Go", "Kafka", "PostgreSQL"],
            )
        ],
        skills=[
            ResumeSkillCategory(
                category="Languages",
                items=["Go", "Python", "TypeScript"],
            ),
            ResumeSkillCategory(
                category="Backend & Cloud",
                items=["Kubernetes", "AWS", "gRPC"],
            ),
        ],
        projects=[
            ResumeProject(
                name="Raft Lock Engine",
                description="- Zero-dependency distributed lock manager\n- 5ms election time",
                link="https://github.com/alexrivera/raft-lock",
                tech_stack=["Go", "Raft"],
            )
        ],
        education=[
            ResumeEducation(
                institution="UC Berkeley",
                degree="B.S. in Computer Science",
                start_date="2013",
                end_date="2017",
                grade="3.8 GPA",
            )
        ],
        custom_sections=[
            ResumeCustomSection(
                title="Certifications & Awards",
                content="<ul><li>AWS Certified Solutions Architect</li></ul>",
                position="main",
            )
        ],
        ats_tags=["Go", "Kafka", "Kubernetes", "Distributed Systems", "PostgreSQL"],
    )

    data = output.model_dump(by_alias=True, mode="json")
    assert data["title"] == "Alex Rivera — Senior Software Engineer Resume"
    assert data["personalInfo"]["fullName"] == "Alex Rivera"
    assert data["personalInfo"]["jobTitle"] == "Senior Software Engineer"
    assert data["experience"][0]["techStack"] == ["Go", "Kafka", "PostgreSQL"]
    assert data["experience"][0]["startDate"] == "Jan 2021"
    assert data["experience"][0]["current"] is True
    assert data["experience"][0]["description"].startswith("<ul>")
    assert data["skills"][0]["category"] == "Languages"
    assert data["projects"][0]["name"] == "Raft Lock Engine"
    assert data["projects"][0]["description"].startswith("<ul>")
    assert data["customSections"][0]["title"] == "Certifications & Awards"
    assert data["atsTags"] == ["Go", "Kafka", "Kubernetes", "Distributed Systems", "PostgreSQL"]


def test_pii_vault_anonymizes_and_restores_resume_json() -> None:
    """Verify PIIVault masks name, email, and phone, and restores them with zero residual tokens."""
    vault = PIIVault()
    name_ph = vault.register("NAME", "Alex Rivera")
    email_ph = vault.register("EMAIL", "alex.rivera@example.com")
    phone_ph = vault.register("PHONE", "+1 (555) 019-9234")

    sample_output = ResumeGenerationOutput(
        title=f"{name_ph} — Staff Engineer Resume",
        personal_info=ResumePersonalInfo(
            full_name=name_ph,
            email=email_ph,
            phone=phone_ph,
            summary=f"Contact {name_ph} at {email_ph} or {phone_ph}.",
        ),
    )

    raw_json = sample_output.model_dump_json(by_alias=True)
    assert name_ph in raw_json
    assert email_ph in raw_json
    assert phone_ph in raw_json
    assert "Alex Rivera" not in raw_json
    assert "alex.rivera@example.com" not in raw_json
    assert "+1 (555) 019-9234" not in raw_json

    restored_json = vault.restore_raw_json(raw_json)
    assert "Alex Rivera" in restored_json
    assert "alex.rivera@example.com" in restored_json
    assert "+1 (555) 019-9234" in restored_json
    assert "{{PII_" not in restored_json
    assert len(PIIVault.find_residuals_in_text(restored_json)) == 0


def test_build_resume_generator_prompts() -> None:
    """Verify prompt builder sets up system prompt and structured human context."""
    profile = {"full_name": "Sarah Connor", "email": "sarah@example.com"}
    messages = build_resume_generator_prompts(
        profile_data=profile,
        preferences_data={"targetRole": "Staff SRE"},
        raw_text="Extensive Kubernetes experience",
        target_role="Staff SRE",
    )
    assert len(messages) == 2
    assert "ATS optimization specialist" in str(messages[0].content)
    assert "Staff SRE" in str(messages[1].content)
    assert "Sarah Connor" in str(messages[1].content)


@pytest.mark.asyncio
async def test_resume_generator_controller_fails_if_profile_missing() -> None:
    """Verify generation aborts immediately if career profile does not exist in Firestore."""
    settings = Settings()
    controller = ResumeGeneratorController(settings)

    with patch(
        "app.controllers.resume_generator_controller.firebase_service.async_get",
        new_callable=AsyncMock,
    ) as mock_get:
        mock_get.return_value = None

        request = ResumeGenerateJobRequest(
            uid="user-no-profile",
            user_email="user@example.com",
        )
        with pytest.raises(ValueError, match="Career profile not found"):
            await controller.generate_resume(request, "job-123")


@pytest.mark.asyncio
async def test_resume_generator_controller_rich_profile_generation() -> None:
    """Verify complete ATS resume generation with rich profile, PII restoration and persistence."""
    settings = Settings()
    controller = ResumeGeneratorController(settings)

    mock_profile: dict[str, Any] = {
        "full_name": "Jordan Smith",
        "email": "jordan.smith@example.com",
        "phone": "+1 (555) 234-5678",
        "jobTitle": "Staff Backend Engineer",
        "raw_text": "Lead architect at CloudCorp with 10 years experience",
        "experience": [
            {
                "company": "CloudCorp",
                "title": "Staff Backend Engineer",
                "start_date": "Jan 2021",
                "end_date": "Present",
                "description": "Architected multi-region event platform handling 50k RPS.",
            }
        ],
        "skills": ["Go", "Python", "Kubernetes", "Kafka", "PostgreSQL"],
        "certificates": [{"name": "AWS Solutions Architect", "issuer": "AWS", "year": "2023"}],
    }

    async def _mock_generate(
        profile_data: dict[str, Any],
        preferences_data: dict[str, Any] | None = None,
        raw_text: str | None = None,
        target_role: str | None = None,
        model: str = "gemini",
    ) -> ResumeGenerationOutput:
        anon_name = str(profile_data.get("full_name") or "Anon Name")
        anon_email = str(profile_data.get("email") or "anon@example.com")
        anon_phone = str(profile_data.get("phone") or "555-0199")
        return ResumeGenerationOutput(
            title=f"{anon_name} — Staff Backend Engineer Resume",
            personal_info=ResumePersonalInfo(
                full_name=anon_name,
                email=anon_email,
                phone=anon_phone,
                location="San Francisco, CA",
                summary="Staff Backend Engineer specializing in high-throughput distributed systems.",
                job_title="Staff Backend Engineer",
            ),
            experience=[
                ResumeExperience(
                    company="CloudCorp",
                    role="Staff Backend Engineer",
                    start_date="Jan 2021",
                    end_date="Present",
                    current=True,
                    description="<ul><li>Architected multi-region event platform handling 50k RPS.</li></ul>",
                    tech_stack=["Go", "Kafka", "PostgreSQL"],
                )
            ],
            skills=[
                ResumeSkillCategory(category="Languages", items=["Go", "Python"]),
                ResumeSkillCategory(
                    category="Infrastructure & Cloud", items=["Kubernetes", "Kafka"]
                ),
            ],
            projects=[
                ResumeProject(
                    name="Event Broker",
                    description="<ul><li>Sub-millisecond event dispatch engine.</li></ul>",
                    tech_stack=["Go", "gRPC"],
                )
            ],
            education=[
                ResumeEducation(
                    institution="Stanford University",
                    degree="B.S. in Computer Science",
                    start_date="2012",
                    end_date="2016",
                )
            ],
            custom_sections=[],
            ats_tags=["Go", "Kafka", "Kubernetes", "Distributed Systems", "PostgreSQL"],
        )

    with (
        patch(
            "app.controllers.resume_generator_controller.firebase_service.async_get",
            new_callable=AsyncMock,
        ) as mock_get,
        patch(
            "app.controllers.resume_generator_controller.firebase_service.async_set",
            new_callable=AsyncMock,
        ) as mock_set,
        patch(
            "app.controllers.resume_generator_controller.firebase_service.async_merge",
            new_callable=AsyncMock,
        ) as mock_merge,
        patch.object(
            controller.agent,
            "generate_resume",
            new_callable=AsyncMock,
            side_effect=_mock_generate,
        ) as mock_agent_gen,
    ):

        def _get_side_effect(path: str) -> dict[str, Any] | None:
            if "profile/" in path:
                return mock_profile
            return {}

        mock_get.side_effect = _get_side_effect

        request = ResumeGenerateJobRequest(
            uid="uid-rich-user",
            user_email="jordan.smith@example.com",
            template_id="clean-ats",
            target_role="Staff Backend Engineer",
        )
        result = await controller.generate_resume(request, "job-rich-123")

        assert result["success"] is True
        assert result["resume_id"] is not None
        assert result["resume"]["userId"] == "jordan.smith@example.com"
        assert result["resume"]["uid"] == "uid-rich-user"
        assert result["resume"]["templateId"] == "clean-ats"
        assert result["resume"]["source"] == "ai_generated"

        # Verify phone & email restored in personalInfo
        p_info = result["resume"]["personalInfo"]
        assert p_info["fullName"] == "Jordan Smith"
        assert p_info["email"] == "jordan.smith@example.com"
        assert p_info["phone"] == "+1 (555) 234-5678"
        assert "{{PII_" not in str(result["resume"])

        # Verify agent was called with anonymized data
        passed_kwargs = mock_agent_gen.call_args[1]
        passed_profile = passed_kwargs["profile_data"]
        assert "Jordan Smith" not in str(passed_profile)
        assert "{{PII_NAME_" in str(passed_profile)

        # Verify Firestore persistence
        mock_set.assert_awaited_once()
        set_path, set_doc = mock_set.call_args[0]
        assert set_path.startswith("resumes/")
        assert set_doc["userId"] == "jordan.smith@example.com"
        assert set_doc["uid"] == "uid-rich-user"

        # Verify job marked completed
        mock_merge.assert_awaited_once()
        job_path, job_doc = mock_merge.call_args[0]
        assert job_path == "resume_parse_jobs/job-rich-123"
        assert job_doc["status"] == "completed"


@pytest.mark.asyncio
async def test_resume_generator_controller_sparse_profile() -> None:
    """Verify resume generation succeeds with minimal/sparse candidate profile."""
    settings = Settings()
    controller = ResumeGeneratorController(settings)

    mock_profile: dict[str, Any] = {
        "full_name": "Dev Minimal",
    }

    async def _mock_generate(
        profile_data: dict[str, Any],
        preferences_data: dict[str, Any] | None = None,
        raw_text: str | None = None,
        target_role: str | None = None,
        model: str = "gemini",
    ) -> ResumeGenerationOutput:
        anon_name = str(profile_data.get("full_name") or "Dev Minimal")
        return ResumeGenerationOutput(
            title=f"{anon_name} — Software Engineer Resume",
            personal_info=ResumePersonalInfo(
                full_name=anon_name,
                email="dev@example.com",
                phone="",
                summary="Software Engineer focused on backend development.",
            ),
            skills=[
                ResumeSkillCategory(category="Core", items=["Python"]),
            ],
        )

    with (
        patch(
            "app.controllers.resume_generator_controller.firebase_service.async_get",
            new_callable=AsyncMock,
        ) as mock_get,
        patch(
            "app.controllers.resume_generator_controller.firebase_service.async_set",
            new_callable=AsyncMock,
        ),
        patch(
            "app.controllers.resume_generator_controller.firebase_service.async_merge",
            new_callable=AsyncMock,
        ),
        patch.object(
            controller.agent,
            "generate_resume",
            new_callable=AsyncMock,
            side_effect=_mock_generate,
        ),
    ):
        mock_get.return_value = mock_profile

        request = ResumeGenerateJobRequest(
            uid="uid-sparse",
            user_email="dev@example.com",
        )
        result = await controller.generate_resume(request, "job-sparse-123")

        assert result["success"] is True
        assert result["resume"]["personalInfo"]["fullName"] == "Dev Minimal"
        assert result["resume"]["personalInfo"]["email"] == "dev@example.com"


@pytest.mark.asyncio
async def test_resume_generator_controller_residual_guard_raises_error() -> None:
    """Verify residual PII placeholders in LLM output trigger ValueError."""
    settings = Settings()
    controller = ResumeGeneratorController(settings)

    mock_profile: dict[str, Any] = {
        "full_name": "Alex Rivera",
        "email": "alex@example.com",
    }

    # Leaked/unknown placeholder that vault cannot resolve
    mock_leaked_output = ResumeGenerationOutput(
        title="{{PII_UNKNOWN_99}} Resume",
        personal_info=ResumePersonalInfo(
            full_name="{{PII_UNKNOWN_99}}",
            email="alex@example.com",
            phone="",
            summary="Leaked token test",
        ),
    )

    with (
        patch(
            "app.controllers.resume_generator_controller.firebase_service.async_get",
            new_callable=AsyncMock,
        ) as mock_get,
        patch.object(
            controller.agent,
            "generate_resume",
            new_callable=AsyncMock,
            return_value=mock_leaked_output,
        ),
    ):
        mock_get.return_value = mock_profile

        request = ResumeGenerateJobRequest(
            uid="uid-leak",
            user_email="alex@example.com",
        )
        with pytest.raises(ValueError, match="Restoration failed: residual placeholders remaining"):
            await controller.generate_resume(request, "job-leak-123")


@pytest.mark.asyncio
async def test_resume_generator_controller_existing_resume_update() -> None:
    """Verify regenerating into an existing resume preserves createdAt and updates doc."""
    settings = Settings()
    controller = ResumeGeneratorController(settings)

    mock_profile = {"full_name": "Taylor Swift", "email": "taylor@example.com"}
    existing_created_at = "2024-01-01T00:00:00Z"
    mock_existing_resume = {
        "id": "resume-existing-456",
        "userId": "taylor@example.com",
        "uid": "uid-taylor",
        "templateId": "modern-tech",
        "createdAt": existing_created_at,
    }

    mock_output = ResumeGenerationOutput(
        title="Taylor Swift — Staff Engineer Resume",
        personal_info=ResumePersonalInfo(
            full_name="Taylor Swift",
            email="taylor@example.com",
            phone="",
            summary="Lead software engineer.",
        ),
    )

    with (
        patch(
            "app.controllers.resume_generator_controller.firebase_service.async_get",
            new_callable=AsyncMock,
        ) as mock_get,
        patch(
            "app.controllers.resume_generator_controller.firebase_service.async_merge",
            new_callable=AsyncMock,
        ) as mock_merge,
        patch.object(
            controller.agent,
            "generate_resume",
            new_callable=AsyncMock,
            return_value=mock_output,
        ),
    ):

        def _get_impl(path: str) -> dict[str, Any] | None:
            if "profile/" in path:
                return mock_profile
            if "resumes/" in path:
                return mock_existing_resume
            return {}

        mock_get.side_effect = _get_impl

        request = ResumeGenerateJobRequest(
            uid="uid-taylor",
            user_email="taylor@example.com",
            resume_id="resume-existing-456",
            template_id="modern-tech",
        )
        result = await controller.generate_resume(request, "job-update-123")

        assert result["success"] is True
        assert result["resume_id"] == "resume-existing-456"
        assert result["resume"]["createdAt"] == existing_created_at
        assert result["resume"]["templateId"] == "modern-tech"
        assert mock_merge.await_count >= 1


@pytest.mark.asyncio
async def test_resume_generator_controller_forbidden_on_other_user_resume() -> None:
    """Verify attempting to regenerate another user's resume raises ValueError."""
    settings = Settings()
    controller = ResumeGeneratorController(settings)

    mock_profile = {"full_name": "Attacker", "email": "attacker@example.com"}
    mock_existing_resume = {
        "id": "victim-resume",
        "userId": "victim@example.com",
        "uid": "uid-victim",
    }

    with patch(
        "app.controllers.resume_generator_controller.firebase_service.async_get",
        new_callable=AsyncMock,
    ) as mock_get:

        def _get_impl(path: str) -> dict[str, Any] | None:
            if "profile/" in path:
                return mock_profile
            if "resumes/" in path:
                return mock_existing_resume
            return {}

        mock_get.side_effect = _get_impl

        request = ResumeGenerateJobRequest(
            uid="uid-attacker",
            user_email="attacker@example.com",
            resume_id="victim-resume",
        )
        with pytest.raises(ValueError, match="Forbidden"):
            await controller.generate_resume(request, "job-forbidden")


@pytest.mark.asyncio
async def test_resume_generator_controller_legacy_resume_ownership_rejected() -> None:
    """Verify legacy resume with only userId and no uid blocks unauthorized modification."""
    settings = Settings()
    controller = ResumeGeneratorController(settings)

    mock_profile = {"full_name": "Attacker", "email": "attacker@example.com"}
    mock_existing_resume = {
        "id": "legacy-victim-resume",
        "userId": "other@example.com",
        "uid": None,
    }

    with patch(
        "app.controllers.resume_generator_controller.firebase_service.async_get",
        new_callable=AsyncMock,
    ) as mock_get:

        def _get_impl(path: str) -> dict[str, Any] | None:
            if "profile/" in path:
                return mock_profile
            if "resumes/" in path:
                return mock_existing_resume
            return {}

        mock_get.side_effect = _get_impl

        request = ResumeGenerateJobRequest(
            uid="uid-attacker",
            user_email="attacker@example.com",
            resume_id="legacy-victim-resume",
        )
        with pytest.raises(
            ValueError,
            match=re.escape("Forbidden: You do not have permission to modify this resume."),
        ):
            await controller.generate_resume(request, "job-forbidden-legacy")


@pytest.mark.asyncio
async def test_resume_generator_controller_nonexistent_resume_rejected() -> None:
    """Verify attempting to update a non-existent resume raises Resume not found."""
    settings = Settings()
    controller = ResumeGeneratorController(settings)

    mock_profile = {"full_name": "User", "email": "user@example.com"}

    with patch(
        "app.controllers.resume_generator_controller.firebase_service.async_get",
        new_callable=AsyncMock,
    ) as mock_get:

        def _get_impl(path: str) -> dict[str, Any] | None:
            if "profile/" in path:
                return mock_profile
            if "resumes/" in path:
                return None
            return {}

        mock_get.side_effect = _get_impl

        request = ResumeGenerateJobRequest(
            uid="uid-user",
            user_email="user@example.com",
            resume_id="nonexistent-resume-id",
        )
        with pytest.raises(ValueError, match=re.escape("Resume not found.")):
            await controller.generate_resume(request, "job-not-found")


def test_resume_generation_output_null_contact_fields() -> None:
    """Verify ResumeGenerationOutput coerces null phone, email, summary, and date fields to empty strings."""
    raw_payload = {
        "title": "Jane Doe — Engineer",
        "personalInfo": {
            "fullName": None,
            "email": None,
            "phone": None,
            "summary": None,
        },
        "education": [
            {
                "institution": "Tech University",
                "degree": "B.S. in CS",
                "startDate": None,
                "endDate": None,
            }
        ],
    }
    output = ResumeGenerationOutput.model_validate(raw_payload)
    assert output.personal_info.full_name == ""
    assert output.personal_info.email == ""
    assert output.personal_info.phone == ""
    assert output.personal_info.summary == ""
    assert output.education[0].start_date == ""
    assert output.education[0].end_date == ""


@pytest.mark.asyncio
async def test_resume_generator_controller_preserves_narrative_context_with_raw_text() -> None:
    """Verify narrative_context is appended to raw_text when both are present."""
    settings = Settings()
    controller = ResumeGeneratorController(settings)

    mock_profile: dict[str, Any] = {
        "full_name": "Morgan Taylor",
        "email": "morgan@example.com",
        "raw_text": "Base resume text with work history.",
        "narrative_context": "Won Outstanding Innovation Award 2024 and 2 patents.",
    }

    captured_raw_text: str | None = None

    async def _mock_generate(
        profile_data: dict[str, Any],
        preferences_data: dict[str, Any] | None = None,
        raw_text: str | None = None,
        target_role: str | None = None,
        model: str = "gemini",
    ) -> ResumeGenerationOutput:
        nonlocal captured_raw_text
        captured_raw_text = raw_text
        return ResumeGenerationOutput(
            title="Morgan Taylor — Resume",
            personal_info=ResumePersonalInfo(
                full_name="Morgan Taylor",
                email="morgan@example.com",
                phone="",
                summary="Engineer summary.",
            ),
        )

    with (
        patch(
            "app.controllers.resume_generator_controller.firebase_service.async_get",
            new_callable=AsyncMock,
        ) as mock_get,
        patch(
            "app.controllers.resume_generator_controller.firebase_service.async_set",
            new_callable=AsyncMock,
        ),
        patch(
            "app.controllers.resume_generator_controller.firebase_service.async_merge",
            new_callable=AsyncMock,
        ),
        patch.object(
            controller.agent,
            "generate_resume",
            new_callable=AsyncMock,
            side_effect=_mock_generate,
        ),
    ):
        mock_get.return_value = mock_profile

        request = ResumeGenerateJobRequest(
            uid="uid-context",
            user_email="morgan@example.com",
        )
        result = await controller.generate_resume(request, "job-context-123")
        assert result["success"] is True
        assert captured_raw_text is not None
        assert "Base resume text with work history." in captured_raw_text
        assert "## Additional Context & Highlights" in captured_raw_text
        assert "Won Outstanding Innovation Award 2024 and 2 patents." in captured_raw_text
