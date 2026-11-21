import copy
import json
import logging
import uuid
from datetime import UTC, date, datetime
from typing import Any, cast

from langsmith import tracing_context

from app.agents.resume_generator_agent import ResumeGeneratorAgent
from app.config import Settings, flush_tracing
from app.core.pii_anonymizer import PIIVault
from app.models.resume_generator import (
    ResumeCustomSection,
    ResumeGenerateJobRequest,
    ResumeGenerationOutput,
)
from app.repositories.firebase_client import firebase_service

logger = logging.getLogger(__name__)


def sanitize_timestamps(obj: Any) -> Any:
    """Recursively convert datetime, date, or Firestore timestamp objects to ISO strings."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if hasattr(obj, "isoformat") and callable(obj.isoformat):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: sanitize_timestamps(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_timestamps(v) for v in obj]
    return obj


class ResumeGeneratorController:
    """Orchestrates ATS resume curation, PII anonymization & restoration, and Firestore persistence."""

    sanitize_timestamps = staticmethod(sanitize_timestamps)

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.agent = ResumeGeneratorAgent(model_name=settings.gemini_model)

    async def generate_resume(
        self,
        request: ResumeGenerateJobRequest,
        job_id: str,
    ) -> dict[str, Any]:
        """Execute async ATS resume generation job and persist resume doc in Firestore."""
        logger.info("Starting resume generation job %s for uid %s", job_id, request.uid)

        # 1. Fetch user's profile and preferences
        raw_profile = await firebase_service.async_get(f"profile/{request.uid}")
        if not raw_profile or not isinstance(raw_profile, dict):
            raise ValueError(
                "Career profile not found for this user. "
                "Please import or complete career profile first."
            )
        profile_data: dict[str, Any] = sanitize_timestamps(cast("dict[str, Any]", raw_profile))

        raw_prefs = await firebase_service.async_get(f"userPreferences/{request.uid}")
        preferences_data: dict[str, Any] = (
            sanitize_timestamps(cast("dict[str, Any]", raw_prefs))
            if isinstance(raw_prefs, dict)
            else {}
        )

        # Check existing resume ownership upfront if resume_id is provided
        target_id = request.resume_id or str(uuid.uuid4())
        existing_doc: dict[str, Any] | None = None
        if request.resume_id:
            raw_existing = await firebase_service.async_get(f"resumes/{target_id}")
            if raw_existing and isinstance(raw_existing, dict):
                sanitized = sanitize_timestamps(raw_existing)
                if isinstance(sanitized, dict):
                    existing_doc = cast("dict[str, Any]", sanitized)
            if not existing_doc:
                raise ValueError("Resume not found.")

            doc_user = existing_doc.get("userId")
            doc_uid = existing_doc.get("uid")
            if (not doc_user and not doc_uid) or (
                (doc_user and doc_user != request.user_email)
                or (doc_uid and doc_uid != request.uid)
            ):
                raise ValueError("Forbidden: You do not have permission to modify this resume.")

        # 2. Extract candidate contact info and hints (CRITICAL: retain phone and email!)
        safe_profile: dict[str, Any] = copy.deepcopy(profile_data)
        basic_info = safe_profile.get("basicInfo")
        basic_dict = basic_info if isinstance(basic_info, dict) else {}

        name_val = (
            safe_profile.get("full_name") or safe_profile.get("name") or basic_dict.get("name")
        )
        name_hint = (
            str(name_val).strip() if isinstance(name_val, str) and name_val.strip() else None
        )

        email_val = safe_profile.get("email") or basic_dict.get("email") or request.user_email
        email_clean = (
            str(email_val).strip() if isinstance(email_val, str) and email_val.strip() else ""
        )

        phone_val = safe_profile.get("phone") or basic_dict.get("phone")
        phone_clean = (
            str(phone_val).strip() if isinstance(phone_val, str) and phone_val.strip() else ""
        )

        # 3. Anonymize PII with PIIVault
        vault = PIIVault()
        raw_text_val = safe_profile.pop("raw_text", None)
        narrative_val = safe_profile.pop("narrative_context", None)
        if raw_text_val and narrative_val:
            if str(narrative_val).strip() and str(narrative_val).strip() not in str(raw_text_val):
                raw_text = f"{raw_text_val}\n\n## Additional Context & Highlights\n{narrative_val}"
            else:
                raw_text = str(raw_text_val)
        elif raw_text_val:
            raw_text = str(raw_text_val)
        elif narrative_val:
            raw_text = str(narrative_val)
        else:
            raw_text = ""

        anonymized_raw_text = (
            vault.anonymize_text(raw_text, candidate_name=name_hint) if raw_text else None
        )

        profile_json = json.dumps(safe_profile, ensure_ascii=False, default=str)
        anonymized_profile_json = vault.anonymize_text(profile_json, candidate_name=name_hint)
        anonymized_profile_data: dict[str, Any] = json.loads(anonymized_profile_json)

        # Ensure contact placeholders exist in vault even if not masked in JSON
        if email_clean and not any(
            v.lower() == email_clean.lower() for v in vault.vault_map.values()
        ):
            vault.register("EMAIL", email_clean)
        if phone_clean and not any(
            v.lower() == phone_clean.lower() for v in vault.vault_map.values()
        ):
            vault.register("PHONE", phone_clean)
        if name_hint and not any(v.lower() == name_hint.lower() for v in vault.vault_map.values()):
            vault.register("NAME", name_hint)

        target_role = (
            request.target_role
            or preferences_data.get("targetRole")
            or safe_profile.get("target_role")
            or safe_profile.get("jobTitle")
        )

        # 4. Invoke LangGraph curation agent
        project_name = (
            None if self.settings.langsmith_uses_agent_address else self.settings.langsmith_project
        )
        try:
            with tracing_context(
                enabled=bool(self.settings.langsmith_api_key and self.settings.langsmith_tracing),
                project_name=project_name,
            ):
                curated_output = await self.agent.generate_resume(
                    profile_data=anonymized_profile_data,
                    preferences_data=preferences_data,
                    raw_text=anonymized_raw_text,
                    target_role=target_role,
                    model=request.model or "gemini",
                )
        finally:
            if self.settings.langsmith_api_key and self.settings.langsmith_tracing:
                flush_tracing()

        # 5. Restore real PII into curated output
        raw_json = curated_output.model_dump_json(by_alias=True)
        try:
            restored_json = vault.restore_raw_json(raw_json)
        except ValueError as exc:
            if "residual placeholders" in str(exc):
                raise ValueError(
                    f"Restoration failed: residual placeholders remaining: {exc}"
                ) from exc
            raise

        restored_output = ResumeGenerationOutput.model_validate_json(restored_json)

        # Verify zero residual tokens
        residuals = PIIVault.find_residuals_in_text(restored_json)
        if residuals:
            raise ValueError(f"Restoration failed: residual placeholders remaining: {residuals}")

        # Ensure contact info is preserved
        if (
            not restored_output.personal_info.email
            or not restored_output.personal_info.email.strip()
            or "@" not in restored_output.personal_info.email
        ):
            restored_output.personal_info.email = email_clean or request.user_email
        if phone_clean and (
            not restored_output.personal_info.phone
            or not restored_output.personal_info.phone.strip()
        ):
            restored_output.personal_info.phone = phone_clean
        if name_hint and (
            not restored_output.personal_info.full_name
            or not restored_output.personal_info.full_name.strip()
        ):
            restored_output.personal_info.full_name = name_hint

        # If custom_sections is empty but profile has certificates, backfill a section
        certs = profile_data.get("certificates") or profile_data.get("certifications")
        if not restored_output.custom_sections and certs and isinstance(certs, list):
            cert_lines: list[str] = []
            for cert in certs:
                if isinstance(cert, dict):
                    name = cert.get("name") or cert.get("title") or ""
                    issuer = cert.get("issuer") or cert.get("issuing_organization") or ""
                    year = cert.get("date") or cert.get("year") or cert.get("issue_date") or ""
                    line = f"{name} — {issuer} ({year})" if year else f"{name} — {issuer}"
                    if line.strip(" —"):
                        cert_lines.append(f"<li>{line.strip(' —')}</li>")
                elif isinstance(cert, str) and cert.strip():
                    cert_lines.append(f"<li>{cert.strip()}</li>")
            if cert_lines:
                restored_output.custom_sections.append(
                    ResumeCustomSection(
                        id=str(uuid.uuid4())[:8],
                        title="Certifications & Awards",
                        content=f"<ul>{''.join(cert_lines)}</ul>",
                        position="main",
                    )
                )

        # 6. Persistence logic
        now_iso = datetime.now(UTC).isoformat()
        content_dict = restored_output.model_dump(by_alias=True, mode="json")

        created_at = existing_doc.get("createdAt", now_iso) if existing_doc else now_iso

        full_resume_doc: dict[str, Any] = {
            **content_dict,
            "id": target_id,
            "userId": request.user_email,
            "uid": request.uid,
            "templateId": request.template_id
            or (existing_doc.get("templateId") if existing_doc else "clean-ats")
            or "clean-ats",
            "source": "ai_generated",
            "createdAt": created_at,
            "updatedAt": now_iso,
        }

        if request.resume_id and existing_doc:
            await firebase_service.async_merge(f"resumes/{target_id}", full_resume_doc)
        else:
            await firebase_service.async_set(f"resumes/{target_id}", full_resume_doc)

        result_payload = {
            "success": True,
            "resume_id": target_id,
            "resumeId": target_id,
            "resume": full_resume_doc,
        }

        # 7. Complete job status in resume_parse_jobs/{job_id}
        await firebase_service.async_merge(
            f"resume_parse_jobs/{job_id}",
            {
                "status": "completed",
                "result": result_payload,
                "updated_at": now_iso,
            },
        )
        logger.info(
            "Resume generation job %s completed successfully for resume %s", job_id, target_id
        )
        return result_payload
