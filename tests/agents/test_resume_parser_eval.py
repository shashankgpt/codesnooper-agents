from typing import Any

import pytest

from app.config import get_settings
from app.models.resume import (
    Certificate,
    Education,
    Experience,
    ResumeSchema,
)

CRITICAL_FIELDS = ["full_name", "email", "summary", "skills", "experience"]


def evaluate_schema_completeness(resume: ResumeSchema) -> dict[str, Any]:
    """Evaluate a parsed ResumeSchema against completeness benchmarks.

    Returns:
        dict containing:
            - completeness_score: float (0.0 to 1.0)
            - passed: bool (true if score >= threshold)
            - present_fields: list[str]
            - missing_fields: list[str]
            - field_scores: dict[str, bool]
    """
    field_checks: dict[str, bool] = {
        "full_name": bool(resume.full_name and resume.full_name.strip()),
        "email": bool(resume.email and "@" in resume.email),
        "summary": bool(resume.summary and len(resume.summary.strip()) > 10),
        "skills": bool(len(resume.skills) > 0),
        "experience": bool(len(resume.experience) > 0),
        "education": bool(len(resume.education) > 0),
        "certificates": bool(len(resume.certificates) > 0),
    }

    present = [field for field, ok in field_checks.items() if ok]
    missing = [field for field, ok in field_checks.items() if not ok]

    # Calculate score based on all standard fields
    score = sum(1 for ok in field_checks.values() if ok) / len(field_checks)

    # Core required check: full_name, email, summary, skills, experience
    core_passed = all(field_checks[field] for field in CRITICAL_FIELDS)

    return {
        "completeness_score": round(score, 2),
        "core_passed": core_passed,
        "present_fields": present,
        "missing_fields": missing,
        "field_checks": field_checks,
    }


def test_schema_completeness_eval_perfect_resume() -> None:
    """Eval test: 100% complete resume fixture passes all completeness criteria."""
    perfect_resume = ResumeSchema(
        full_name="Jordan Lee",
        email="jordan.lee@example.com",
        phone="+1 202-555-0143",
        location="Austin, TX",
        summary=(
            "Principal Cloud Architect with 10+ years specializing in "
            "distributed multi-agent systems."
        ),
        skills=["Python", "LangChain", "LangGraph", "FastAPI", "Terraform", "GCP"],
        experience=[
            Experience(
                company="OpenCloud Technologies",
                title="Principal Architect",
                start_date="2020-01",
                end_date="Present",
                is_current=True,
                description="Designed high-throughput agent streaming pipeline.",
            )
        ],
        education=[
            Education(
                institution="University of Texas at Austin",
                degree="B.S. in Computer Science",
                end_date="2014",
            )
        ],
        certificates=[
            Certificate(
                name="Google Cloud Professional Cloud Architect",
                issuing_organization="Google",
                issue_date="2022",
            )
        ],
    )

    eval_result = evaluate_schema_completeness(perfect_resume)
    assert eval_result["completeness_score"] == 1.0
    assert eval_result["core_passed"] is True
    assert len(eval_result["missing_fields"]) == 0


def test_schema_completeness_eval_detects_missing_fields() -> None:
    """Eval test: correctly flags missing critical fields and calculates accurate penalty."""
    incomplete_resume = ResumeSchema(
        full_name=None,  # Missing!
        email="test@example.com",
        summary=None,  # Missing!
        skills=["Python"],
        experience=[],  # Missing!
        education=[],
        certificates=[],
    )

    eval_result = evaluate_schema_completeness(incomplete_resume)
    assert eval_result["core_passed"] is False
    assert "full_name" in eval_result["missing_fields"]
    assert "summary" in eval_result["missing_fields"]
    assert "experience" in eval_result["missing_fields"]
    assert eval_result["completeness_score"] < 0.5


@pytest.mark.asyncio
async def test_live_agent_structured_output_eval() -> None:
    """Live LLM extraction eval testing structured output extraction directly from sample text."""
    settings = get_settings()
    if not settings.google_api_key:
        pytest.skip("GOOGLE_API_KEY not configured; skipping live LLM extraction eval")

    from langchain_core.messages import HumanMessage, SystemMessage

    from app.agents.base import create_gemini_llm
    from app.prompts.resume_parser import get_resume_prompts

    sample_resume_text = """
    DAVID CHOI
    Email: david.choi@cloudagents.dev | Phone: +1 (415) 555-8291 | San Francisco, CA
    LinkedIn: linkedin.com/in/davidchoi-dev | GitHub: github.com/davidchoi

    PROFESSIONAL SUMMARY
    Results-driven Lead AI Engineer with 7 years of production experience building scalable
    agentic systems and RAG workflows. Deep expertise in LangChain, LangGraph, Python,
    and cloud infrastructure.

    TECHNICAL SKILLS
    Languages & Frameworks: Python, TypeScript, FastAPI, LangChain, LangGraph, PyTorch
    Cloud & DevOps: Docker, Kubernetes, Google Cloud Platform (GCP), GitHub Actions
    Databases: Firestore, PostgreSQL, Vector Databases

    WORK EXPERIENCE
    Lead AI Engineer | SynthLabs AI (2022 - Present) | San Francisco, CA
    - Architected multi-agent orchestration service handling 50k requests daily.
    - Reduced end-to-end token latency by 35% using streaming graph checkpoints.

    Software Engineer | DataCorp (2018 - 2022) | Seattle, WA
    - Developed backend REST APIs with FastAPI and PostgreSQL.

    Junior Developer | Pioneer Systems (2016 - 2018) | Portland, OR
    - Maintained legacy microservices and built automated unit testing suites.

    EDUCATION
    B.S. in Software Engineering, University of Washington (2014 - 2018)

    CERTIFICATIONS
    - AWS Certified Solutions Architect - Associate (2023)
    - Google Cloud Professional Data Engineer (2021)
    """

    llm = create_gemini_llm()
    assert llm is not None, "Failed to create Gemini LLM"

    structured_llm = llm.with_structured_output(ResumeSchema)
    prompts = get_resume_prompts("v1")

    messages = [
        SystemMessage(content=prompts["system_prompt"]),
        HumanMessage(content=prompts["user_prompt_template"].format(content=sample_resume_text)),
    ]

    result: Any = await structured_llm.ainvoke(messages)
    assert isinstance(result, ResumeSchema)

    # Evaluate completeness
    eval_result = evaluate_schema_completeness(result)

    assert eval_result["core_passed"] is True, f"Failed core completeness: {eval_result}"
    assert "david choi" in (result.full_name or "").lower()
    assert result.email == "david.choi@cloudagents.dev"
    assert len(result.skills) >= 3

    # Verify ALL companies are extracted (not just the last or latest one)
    assert len(result.experience) >= 3, f"Expected at least 3 companies, got: {result.experience}"
    company_names = [exp.company.lower() for exp in result.experience if exp.company]
    assert any("synthlabs" in c for c in company_names), f"Missing SynthLabs in {company_names}"
    assert any("datacorp" in c for c in company_names), f"Missing DataCorp in {company_names}"
    assert any("pioneer" in c for c in company_names), f"Missing Pioneer in {company_names}"

    assert len(result.education) >= 1
    assert len(result.certificates) >= 1
    assert eval_result["completeness_score"] >= 0.85
