import asyncio
import copy
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any, cast

from langsmith import tracing_context

from app.agents.portfolio_generator_agent import PortfolioGeneratorAgent
from app.config import Settings, flush_tracing
from app.core.pii_anonymizer import PIIVault
from app.models.portfolio import (
    PortfolioEnhanceOutput,
    PortfolioEnhanceRequest,
    PortfolioEnhanceResponse,
    PortfolioGenerateJobRequest,
    PortfolioGenerationOutput,
    slugify,
)
from app.repositories.firebase_client import firebase_service

logger = logging.getLogger(__name__)


class PortfolioController:
    """Orchestrates portfolio generation, section enhancements, and Firestore persistence."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.agent = PortfolioGeneratorAgent(model_name=settings.gemini_model)

    async def generate_portfolio(
        self,
        request: PortfolioGenerateJobRequest,
        job_id: str,
    ) -> dict[str, Any]:
        """Execute async portfolio generation job and persist draft in Firestore."""
        logger.info("Starting portfolio generation job %s for uid %s", job_id, request.uid)

        # 1. Fetch user's profile and preferences
        raw_profile = await firebase_service.async_get(f"profile/{request.uid}")
        if not raw_profile or not isinstance(raw_profile, dict):
            raise ValueError(
                "Career profile not found for this user. "
                "Please import or complete career profile first."
            )
        profile_data: dict[str, Any] = cast("dict[str, Any]", raw_profile)

        raw_prefs = await firebase_service.async_get(f"userPreferences/{request.uid}")
        preferences_data: dict[str, Any] = (
            cast("dict[str, Any]", raw_prefs) if isinstance(raw_prefs, dict) else {}
        )

        # 2. Hard Privacy Gate: strictly purge phone numbers from memory before LLM
        safe_profile: dict[str, Any] = copy.deepcopy(profile_data)
        safe_profile.pop("phone", None)
        basic_info = safe_profile.get("basicInfo")
        if isinstance(basic_info, dict):
            basic_info.pop("phone", None)

        # 3. Anonymize PII with PIIVault
        vault = PIIVault()
        raw_text_val = safe_profile.pop("raw_text", None)
        narrative_val = safe_profile.pop("narrative_context", None)
        raw_val = raw_text_val or narrative_val
        raw_text = str(raw_val) if raw_val else ""

        basic_dict = basic_info if isinstance(basic_info, dict) else {}
        name_val = (
            safe_profile.get("full_name") or safe_profile.get("name") or basic_dict.get("name")
        )
        name_hint = (
            str(name_val).strip() if isinstance(name_val, str) and name_val.strip() else None
        )

        anonymized_raw_text = (
            vault.anonymize_text(raw_text, candidate_name=name_hint) if raw_text else None
        )

        profile_json = json.dumps(safe_profile, ensure_ascii=False)
        anonymized_profile_json = vault.anonymize_text(profile_json, candidate_name=name_hint)
        anonymized_profile_data: dict[str, Any] = json.loads(anonymized_profile_json)

        # 4. Invoke LangGraph curation agent
        project_name = (
            None if self.settings.langsmith_uses_agent_address else self.settings.langsmith_project
        )
        try:
            with tracing_context(
                enabled=bool(self.settings.langsmith_api_key and self.settings.langsmith_tracing),
                project_name=project_name,
            ):
                curated_output = await self.agent.generate_portfolio(
                    profile_data=anonymized_profile_data,
                    preferences_data=preferences_data,
                    raw_text=anonymized_raw_text,
                    model=request.model or "gemini",
                )
        finally:
            if self.settings.langsmith_api_key and self.settings.langsmith_tracing:
                flush_tracing()

        # 5. Restore real PII into curated output
        raw_json = curated_output.model_dump_json(by_alias=True)
        restored_json = vault.restore_raw_json(raw_json)
        restored_output = PortfolioGenerationOutput.model_validate_json(restored_json)

        # Verify zero residual tokens
        residuals = PIIVault.find_residuals_in_text(restored_json)
        if residuals:
            raise ValueError(f"Restoration failed: residual placeholders remaining: {residuals}")

        # 6. Check existing portfolios in Firestore
        existing_doc_id: str | None = None
        existing_doc: dict[str, Any] | None = None
        db = firebase_service.firestore

        if db:

            def _find_user_portfolio() -> tuple[str | None, dict[str, Any] | None]:
                docs = (
                    db.collection("portfolios")
                    .where("createdBy", "==", request.uid)
                    .where("status", "in", [1, 2])
                    .limit(1)
                    .get()
                )
                for d in docs:
                    return d.id, d.to_dict()
                return None, None

            existing_doc_id, existing_doc = await asyncio.to_thread(_find_user_portfolio)

        now_iso = datetime.now(UTC).isoformat()
        content_dict = restored_output.model_dump(by_alias=True, mode="json")
        content_dict["schemaVersion"] = 2

        # Retain avatar if requested
        if request.preserve_custom_avatar and existing_doc and existing_doc.get("imageUrl"):
            content_dict["imageUrl"] = existing_doc["imageUrl"]
        elif safe_profile.get("image_url") or safe_profile.get("avatar_url"):
            content_dict["imageUrl"] = safe_profile.get("image_url") or safe_profile.get(
                "avatar_url"
            )
        else:
            content_dict["imageUrl"] = None

        # Retain privacy settings if already configured
        if existing_doc and existing_doc.get("privacy"):
            content_dict["privacy"] = existing_doc["privacy"]
        else:
            content_dict["privacy"] = {
                "emailVisibility": "cta",
                "publicEmail": None,
            }

        # 7. Draft-First Persistence Logic
        if existing_doc_id and existing_doc:
            target_id = existing_doc_id
            slug_name = existing_doc.get("slugName") or slugify(restored_output.name)
            is_live = existing_doc.get("status") == 1

            if is_live:
                # Stage into draftData — live portfolio remains untouched
                update_payload = {
                    "hasUnpublishedChanges": True,
                    "draftData": content_dict,
                    "lastGeneratedJobId": job_id,
                    "updatedAt": now_iso,
                }
            else:
                # Top-level is already draft (status: 2) — update directly
                update_payload = {
                    **content_dict,
                    "status": 2,
                    "source": "ai_generated",
                    "hasUnpublishedChanges": True,
                    "draftData": None,
                    "lastGeneratedJobId": job_id,
                    "updatedAt": now_iso,
                }

            await firebase_service.async_merge(f"portfolios/{target_id}", update_payload)
            is_draft = True
        else:
            # First-time portfolio creation -> generate unique slug
            base_slug = slugify(restored_output.name)
            slug_name = base_slug

            if db:

                def _find_available_slug(candidate: str) -> str:
                    current_slug = candidate
                    counter = 2
                    while True:
                        docs = (
                            db.collection("portfolios")
                            .where("slugName", "==", current_slug)
                            .where("status", "in", [1, 2])
                            .limit(1)
                            .get()
                        )
                        if not docs:
                            return current_slug
                        current_slug = f"{candidate}-{counter}"
                        counter += 1

                slug_name = await asyncio.to_thread(_find_available_slug, base_slug)

            target_id = str(uuid.uuid4())
            new_portfolio_doc = {
                **content_dict,
                "id": target_id,
                "createdBy": request.uid,
                "slugName": slug_name,
                "status": 2,  # Draft
                "source": "ai_generated",
                "hasUnpublishedChanges": True,
                "draftData": None,
                "lastGeneratedJobId": job_id,
                "publishedAt": None,
                "createdAt": now_iso,
                "updatedAt": now_iso,
            }
            await firebase_service.async_set(f"portfolios/{target_id}", new_portfolio_doc)
            is_draft = True

        result_payload = {
            "success": True,
            "portfolio_id": target_id,
            "slug_name": slug_name,
            "is_draft": is_draft,
            "portfolio": content_dict,
        }

        # 8. Complete job status
        await firebase_service.async_merge(
            f"resume_parse_jobs/{job_id}",
            {
                "status": "completed",
                "result": result_payload,
                "updated_at": now_iso,
            },
        )
        logger.info("Portfolio generation job %s completed successfully", job_id)
        return result_payload

    async def enhance_section(
        self,
        request: PortfolioEnhanceRequest,
    ) -> PortfolioEnhanceResponse:
        """Polish a specific portfolio section with PII protection."""
        name_hint: str | None = None
        if request.uid:
            try:
                raw_profile = await firebase_service.async_get(f"profile/{request.uid}")
                if isinstance(raw_profile, dict):
                    basic_info = raw_profile.get("basicInfo")
                    basic_dict = basic_info if isinstance(basic_info, dict) else {}
                    name_val = (
                        raw_profile.get("full_name")
                        or raw_profile.get("name")
                        or basic_dict.get("name")
                    )
                    if isinstance(name_val, str) and name_val.strip():
                        name_hint = name_val.strip()
            except Exception as e:
                logger.warning("Could not fetch profile for name_hint: %s", e)

        vault = PIIVault()
        anonymized_text = vault.anonymize_text(request.current_text, candidate_name=name_hint)
        anonymized_instruction = (
            vault.anonymize_text(request.instruction, candidate_name=name_hint)
            if request.instruction
            else None
        )

        project_name = (
            None if self.settings.langsmith_uses_agent_address else self.settings.langsmith_project
        )
        try:
            with tracing_context(
                enabled=bool(self.settings.langsmith_api_key and self.settings.langsmith_tracing),
                project_name=project_name,
            ):
                output: PortfolioEnhanceOutput = await self.agent.enhance_section(
                    section=request.section,
                    current_text=anonymized_text,
                    instruction=anonymized_instruction,
                    model=request.model or "gemini",
                )
        finally:
            if self.settings.langsmith_api_key and self.settings.langsmith_tracing:
                flush_tracing()

        restored_text = vault.restore_text(output.enhanced_text)
        restored_bullets = [vault.restore_text(b) for b in output.bullets]

        # Verify zero residual tokens
        text_residuals = PIIVault.find_residuals_in_text(restored_text)
        bullet_residuals = [
            res for b in restored_bullets for res in PIIVault.find_residuals_in_text(b)
        ]
        all_residuals = sorted(set(text_residuals + bullet_residuals))
        if all_residuals:
            raise ValueError(
                f"Restoration failed: residual placeholders remaining: {all_residuals}"
            )

        return PortfolioEnhanceResponse(
            success=True,
            section=request.section,
            enhanced_text=restored_text,
            bullets=restored_bullets,
        )
