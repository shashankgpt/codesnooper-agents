from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from app.config import Settings
from app.controllers.portfolio_controller import PortfolioController
from app.core.pii_anonymizer import PIIVault
from app.models.portfolio import (
    PortfolioEnhanceOutput,
    PortfolioEnhanceRequest,
    PortfolioGenerateJobRequest,
    PortfolioGenerationOutput,
    PortfolioHighlight,
    PortfolioProject,
    PortfolioSkillGroup,
    PortfolioSocials,
    slugify,
)
from app.repositories.firebase_client import firebase_service


def test_slugify_transforms_names() -> None:
    """Verify slug generation creates clean URL-safe slugs."""
    assert slugify("Alex Rivera") == "alex-rivera"
    assert slugify("Sarah O'Connor, Ph.D.") == "sarah-o-connor-ph-d"
    assert slugify("   Jane   Doe   ") == "jane-doe"
    assert slugify("!@#$%^") == "developer"


def test_portfolio_models_schema_and_camelcase() -> None:
    """Verify PortfolioGenerationOutput serializes to camelCase for Next.js BFF."""
    output = PortfolioGenerationOutput(
        name="Alex Rivera",
        headline="Staff Backend Engineer · Distributed Systems",
        bio="Lead Systems Engineer with 8+ years experience.\n\nFocused on Go.",
        open_to_work=True,
        socials=PortfolioSocials(
            github_url="https://github.com/alexrivera",
            linkedin_url="https://linkedin.com/in/alexrivera",
            website_url="https://alexrivera.dev",
        ),
        highlights=[
            PortfolioHighlight(metric="99.99%", label="Uptime"),
            PortfolioHighlight(metric="25k RPS", label="Throughput"),
        ],
        projects=[
            PortfolioProject(
                title="Distributed Lock Manager",
                description="Zero-dependency Raft lock engine in Go",
                highlights=["Sub-5ms leader election"],
                tech_stack=["Go", "Raft", "gRPC"],
                live_url=None,
                repo_url="https://github.com/alexrivera/raft-lock",
                featured=True,
            )
        ],
        skill_groups=[
            PortfolioSkillGroup(
                category="Languages",
                skills=["Go", "TypeScript", "SQL"],
            )
        ],
    )

    data = output.model_dump(by_alias=True, mode="json")
    assert data["openToWork"] is True
    assert data["socials"]["githubUrl"] == "https://github.com/alexrivera"
    assert data["highlights"][0]["metric"] == "99.99%"
    assert data["projects"][0]["techStack"] == ["Go", "Raft", "gRPC"]
    assert data["skillGroups"][0]["category"] == "Languages"


def test_pii_vault_anonymizes_and_restores_portfolio_json() -> None:
    """Verify PIIVault masks name and email, and restores them with zero residual tokens."""
    vault = PIIVault()
    name_ph = vault.register("NAME", "Alex Rivera")
    email_ph = vault.register("EMAIL", "alex.rivera@example.com")

    sample_output = PortfolioGenerationOutput(
        name=name_ph,
        headline="Staff Backend Engineer",
        bio=f"My name is {name_ph} and I can be contacted at {email_ph}.",
        open_to_work=True,
    )

    raw_json = sample_output.model_dump_json(by_alias=True)
    assert name_ph in raw_json
    assert "Alex Rivera" not in raw_json

    restored_json = vault.restore_raw_json(raw_json)
    assert "Alex Rivera" in restored_json
    assert "alex.rivera@example.com" in restored_json
    assert "{{PII_" not in restored_json
    assert len(PIIVault.find_residuals_in_text(restored_json)) == 0


@pytest.mark.asyncio
async def test_portfolio_controller_fails_if_career_profile_missing() -> None:
    """Verify generation aborts immediately if career profile does not exist in Firestore."""
    settings = Settings()
    controller = PortfolioController(settings)

    with patch(
        "app.controllers.portfolio_controller.firebase_service.async_get",
        new_callable=AsyncMock,
    ) as mock_get:
        mock_get.return_value = None

        request = PortfolioGenerateJobRequest(uid="user-no-profile")
        with pytest.raises(ValueError, match="Career profile not found"):
            await controller.generate_portfolio(request, "job-123")


@pytest.mark.asyncio
async def test_portfolio_controller_creates_draft_when_no_portfolio_exists() -> None:
    """Verify first-time generation creates a status: 2 (Draft) document."""
    settings = Settings()
    controller = PortfolioController(settings)

    mock_profile: dict[str, Any] = {
        "full_name": "Jordan Smith",
        "email": "jordan@example.com",
        "phone": "+1 555-0199",  # Should be purged!
        "raw_text": "Staff Engineer with 10 years experience",
    }

    mock_generated_output = PortfolioGenerationOutput(
        name="Jordan Smith",
        headline="Staff Engineer · Distributed Systems",
        bio="Paragraph 1 about systems.\n\nParagraph 2 about architecture.",
        open_to_work=True,
    )

    mock_firestore = MagicMock()
    mock_coll = MagicMock()
    mock_coll.where.return_value = mock_coll
    mock_coll.limit.return_value = mock_coll
    mock_coll.get.return_value = []
    mock_firestore.collection.return_value = mock_coll

    with (
        patch(
            "app.controllers.portfolio_controller.firebase_service.async_get",
            new_callable=AsyncMock,
        ) as mock_get,
        patch(
            "app.controllers.portfolio_controller.firebase_service.async_set",
            new_callable=AsyncMock,
        ) as mock_set,
        patch(
            "app.controllers.portfolio_controller.firebase_service.async_merge",
            new_callable=AsyncMock,
        ),
        patch.object(
            type(firebase_service),
            "firestore",
            new_callable=PropertyMock,
            return_value=mock_firestore,
        ),
        patch.object(
            controller.agent,
            "generate_portfolio",
            new_callable=AsyncMock,
        ) as mock_agent_gen,
    ):
        def _get_profile_1(path: str) -> dict[str, Any]:
            return mock_profile if "profile/" in path else {}

        mock_get.side_effect = _get_profile_1
        mock_agent_gen.return_value = mock_generated_output

        request = PortfolioGenerateJobRequest(uid="uid-first-time")
        result = await controller.generate_portfolio(request, "job-123")

        assert result["success"] is True
        assert result["is_draft"] is True
        assert result["slug_name"] == "jordan-smith"

        mock_set.assert_called_once()
        set_path, set_data = mock_set.call_args[0]
        assert set_path.startswith("portfolios/")
        assert set_data["status"] == 2
        assert set_data["source"] == "ai_generated"
        assert set_data["hasUnpublishedChanges"] is True
        assert set_data["draftData"] is None
        assert set_data["slugName"] == "jordan-smith"
        assert "phone" not in set_data


@pytest.mark.asyncio
async def test_portfolio_controller_stages_into_draft_data_if_published_exists() -> None:
    """Verify regenerating a live (status: 1) portfolio stages into draftData."""
    settings = Settings()
    controller = PortfolioController(settings)

    mock_profile: dict[str, Any] = {
        "full_name": "Jordan Smith",
        "email": "jordan@example.com",
    }

    mock_generated_output = PortfolioGenerationOutput(
        name="Jordan Smith",
        headline="Updated Headline · Go & Kubernetes",
        bio="Updated bio paragraph 1.\n\nUpdated bio paragraph 2.",
        open_to_work=False,
    )

    mock_existing_doc = {
        "id": "port-doc-123",
        "createdBy": "uid-published",
        "slugName": "jordan-smith-custom",
        "status": 1,
        "imageUrl": "https://example.com/avatar.jpg",
        "privacy": {"emailVisibility": "hidden", "publicEmail": None},
    }

    mock_firestore = MagicMock()
    mock_doc_snap = MagicMock()
    mock_doc_snap.id = "port-doc-123"
    mock_doc_snap.to_dict.return_value = mock_existing_doc

    mock_coll = MagicMock()
    mock_coll.where.return_value = mock_coll
    mock_coll.limit.return_value = mock_coll
    mock_coll.get.return_value = [mock_doc_snap]
    mock_firestore.collection.return_value = mock_coll

    with (
        patch(
            "app.controllers.portfolio_controller.firebase_service.async_get",
            new_callable=AsyncMock,
        ) as mock_get,
        patch(
            "app.controllers.portfolio_controller.firebase_service.async_merge",
            new_callable=AsyncMock,
        ) as mock_merge,
        patch.object(
            type(firebase_service),
            "firestore",
            new_callable=PropertyMock,
            return_value=mock_firestore,
        ),
        patch.object(
            controller.agent,
            "generate_portfolio",
            new_callable=AsyncMock,
        ) as mock_agent_gen,
    ):
        def _get_profile_1(path: str) -> dict[str, Any]:
            return mock_profile if "profile/" in path else {}

        mock_get.side_effect = _get_profile_1
        mock_agent_gen.return_value = mock_generated_output

        request = PortfolioGenerateJobRequest(uid="uid-published")
        result = await controller.generate_portfolio(request, "job-published-456")

        assert result["success"] is True
        assert result["is_draft"] is True
        assert result["slug_name"] == "jordan-smith-custom"

        portfolio_merge_call = next(
            c for c in mock_merge.call_args_list if c[0][0] == "portfolios/port-doc-123"
        )
        merge_data = portfolio_merge_call[0][1]

        assert merge_data["hasUnpublishedChanges"] is True
        assert merge_data["draftData"]["headline"] == "Updated Headline · Go & Kubernetes"
        assert merge_data["draftData"]["imageUrl"] == "https://example.com/avatar.jpg"
        assert merge_data["draftData"]["privacy"]["emailVisibility"] == "hidden"


@pytest.mark.asyncio
async def test_portfolio_controller_enhance_section() -> None:
    """Verify section enhancement returns polished text and bullets."""
    settings = Settings()
    controller = PortfolioController(settings)

    mock_enhance_output = PortfolioEnhanceOutput(
        enhanced_text="Senior Systems Engineer with 8+ years building distributed infrastructure.",
        bullets=["Reduced p99 latency by 45%", "Scaled Kafka cluster to 50k msgs/sec"],
    )

    with patch.object(
        controller.agent,
        "enhance_section",
        new_callable=AsyncMock,
    ) as mock_enhance:
        mock_enhance.return_value = mock_enhance_output

        request = PortfolioEnhanceRequest(
            uid="user-123",
            section="bio",
            current_text="I work on backend systems.",
            instruction="Make it sound like senior systems engineering.",
        )
        response = await controller.enhance_section(request)

        assert response.success is True
        assert response.section == "bio"
        assert "Senior Systems Engineer" in response.enhanced_text
        assert len(response.bullets) == 2
