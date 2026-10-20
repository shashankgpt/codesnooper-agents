"""PII Anonymization and De-anonymization Vault.

Comprehensively scrubs personally identifiable information (PII) before data
enters LangGraph workflows, ensuring zero PII is leaked to LLM providers or
LangSmith tracing logs.

PII scrubbed:
- Email addresses
- Phone numbers
- Avatar & profile image CDN URLs
- Full Names, First Names, Last Names
- LinkedIn handles & usernames
"""

import json
import logging
import re
from typing import Any

from app.models.resume import ResumeSchema

logger = logging.getLogger(__name__)

# Standard RFC-compliant email pattern
EMAIL_REGEX = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# Date range pattern to protect years like 2020-2024 from being identified as phones
DATE_RANGE_REGEX = re.compile(
    r"\b(?:19|20)\d{2}\s*[-–—/]\s*(?:(?:19|20)\d{2}|present|current)\b",
    re.IGNORECASE,
)

# Phone number regex (international or standard national formats)
PHONE_REGEX = re.compile(
    r"(?:(?:\+|00)\d{1,3}[\s.-]?)?"      # Optional country prefix: +91, +1, etc.
    r"(?:\(?\d{2,5}\)?[\s.-]?)?"          # Optional area code: (555), 022
    r"\d{3,5}[\s.-]?\d{3,5}"              # Main number body
)

# LinkedIn profile URL / handle regex
LINKEDIN_URL_REGEX = re.compile(
    r"(?:https?://)?(?:[a-zA-Z]{2,3}\.)?linkedin\.com/in/([a-zA-Z0-9_\-]+)/?",
    re.IGNORECASE,
)

# Image / avatar CDN URLs (e.g. LinkedIn media CDN)
MEDIA_CDN_REGEX = re.compile(
    r"https?://media\.licdn\.com/[^\s\"',]+",
    re.IGNORECASE,
)


class PIIVault:
    """Stores placeholder-to-PII mappings and performs bidirectional anonymization."""

    def __init__(self, vault_map: dict[str, str] | None = None) -> None:
        self.vault_map: dict[str, str] = vault_map or {}
        self._reverse_map: dict[str, str] = {v.lower(): k for k, v in self.vault_map.items()}

    def register(self, pii_type: str, real_value: str) -> str:
        """Register a real PII value and return its assigned placeholder."""
        clean_val = real_value.strip()
        lower_val = clean_val.lower()

        # Reuse existing placeholder if this value was already registered
        if lower_val in self._reverse_map:
            return self._reverse_map[lower_val]

        placeholder = f"{{{{PII_{pii_type.upper()}_{len(self.vault_map) + 1}}}}}"
        self.vault_map[placeholder] = clean_val
        self._reverse_map[lower_val] = placeholder
        return placeholder

    def anonymize_text(self, text: str, candidate_name: str | None = None) -> str:
        """Mask all identifiable PII in text with placeholders in optimal precedence order."""
        if not text:
            return text

        masked = text

        # 1. Mask Email addresses FIRST (so handle masking doesn't corrupt email addresses)
        def replace_email(match: re.Match[str]) -> str:
            val = match.group(0).strip()
            return self.register("EMAIL", val)

        masked = EMAIL_REGEX.sub(replace_email, masked)

        # 2. Mask Phone numbers (protecting employment date ranges like 2020-2024 first)
        date_placeholders: dict[str, str] = {}

        def hide_date(match: re.Match[str]) -> str:
            ph = f"__TEMPDATE_{len(date_placeholders)}__"
            date_placeholders[ph] = match.group(0)
            return ph

        temp_masked = DATE_RANGE_REGEX.sub(hide_date, masked)

        def replace_phone(match: re.Match[str]) -> str:
            candidate = match.group(0).strip()
            digits_only = re.sub(r"\D", "", candidate)
            if 10 <= len(digits_only) <= 15:
                # Do not mask obvious dummy repeating digits (e.g. 9999999999 or 0000000000)
                if len(set(digits_only[-10:])) == 1:
                    return candidate
                return self.register("PHONE", candidate)
            return candidate

        temp_masked = PHONE_REGEX.sub(replace_phone, temp_masked)

        for ph, real_date in date_placeholders.items():
            temp_masked = temp_masked.replace(ph, real_date)

        masked = temp_masked

        # 3. Strip raw media / avatar / banner CDN URLs completely
        masked = MEDIA_CDN_REGEX.sub("[IMAGE_REDACTED]", masked)

        # 4. Detect and mask candidate name (full name, first name, last name)
        name_to_mask = candidate_name
        if not name_to_mask:
            name_header_match = re.search(r"(?:^|\n)Name:\s*([^\n\r]+)", masked)
            if name_header_match:
                name_candidate = name_header_match.group(1).strip()
                if 3 <= len(name_candidate) <= 60 and " " in name_candidate:
                    name_to_mask = name_candidate

        if name_to_mask and len(name_to_mask.strip()) >= 3:
            name_clean = name_to_mask.strip()
            name_ph = self.register("NAME", name_clean)
            masked = re.sub(re.escape(name_clean), name_ph, masked, flags=re.IGNORECASE)

            # Mask individual first and last name components if >= 3 characters
            parts = [p.strip() for p in name_clean.split() if len(p.strip()) >= 3]
            for part in parts:
                part_pattern = re.compile(rf"\b{re.escape(part)}\b", re.IGNORECASE)
                part_ph = self.register("NAMEPART", part)
                masked = part_pattern.sub(part_ph, masked)

        # 5. Extract and mask LinkedIn handles LAST
        for handle_match in LINKEDIN_URL_REGEX.finditer(text):
            handle = handle_match.group(1).strip()
            if handle and len(handle) >= 3 and handle.lower() not in {"in", "feed", "jobs"}:
                ph = self.register("HANDLE", handle)
                masked = re.sub(re.escape(handle), ph, masked, flags=re.IGNORECASE)

        logger.info(
            "Anonymized text: masked %d PII items before LLM transmission",
            len(self.vault_map),
        )
        return masked

    def restore_resume(self, resume: ResumeSchema) -> ResumeSchema:
        """Restore all original PII values back into the structured ResumeSchema object."""
        if not self.vault_map:
            return resume

        raw_json = resume.model_dump_json()

        # Sort placeholders by length descending to avoid partial overlap replacements
        sorted_items = sorted(
            self.vault_map.items(),
            key=lambda item: len(item[0].strip("{}")),
            reverse=True,
        )

        for placeholder, original_value in sorted_items:
            tag = placeholder.strip("{}")
            escaped_val = json.dumps(original_value)[1:-1]

            # 1. Full double braces {{TAG}} (with optional whitespace or appended dummy domain)
            raw_json = re.sub(
                rf"\{{\{{\s*{re.escape(tag)}\s*\}}\}}(?:@[A-Za-z0-9.-]+\.[A-Za-z]{{2,}})?",
                lambda _: escaped_val,
                raw_json,
                flags=re.IGNORECASE,
            )

            # 2. Single braces {TAG}
            raw_json = re.sub(
                rf"\{{\s*{re.escape(tag)}\s*\}}(?:@[A-Za-z0-9.-]+\.[A-Za-z]{{2,}})?",
                lambda _: escaped_val,
                raw_json,
                flags=re.IGNORECASE,
            )

            # 3. Naked tag \bTAG\b with optional leading + and optional trailing dummy domain
            raw_json = re.sub(
                rf"\+?\b{re.escape(tag)}\b(?:@[A-Za-z0-9.-]+\.[A-Za-z]{{2,}})?",
                lambda _: escaped_val,
                raw_json,
                flags=re.IGNORECASE,
            )

        # Strict validation: verify NO residual placeholders leaked through into the final JSON
        residual_pattern = re.compile(
            r"\{\{PII_[A-Za-z0-9_]+\}\}|(?:\b|\b\+)PII_(?:EMAIL|PHONE|NAME|NAMEPART|HANDLE)_[A-Za-z0-9_]+\b",
            re.IGNORECASE,
        )
        residuals = residual_pattern.findall(raw_json)
        if residuals:
            unique_residuals = sorted(set(residuals))
            raise ValueError(
                f"PII de-anonymization failed: residual placeholders ({', '.join(unique_residuals)}) "
                "could not be restored to original data."
            )

        restored = ResumeSchema.model_validate_json(raw_json)
        logger.info("Restored PII values for %d placeholders in parsed resume", len(self.vault_map))
        return restored

    @staticmethod
    def find_residuals(resume: ResumeSchema) -> list[str]:
        """Check a ResumeSchema object for any unrestored PII placeholders."""
        raw_json = resume.model_dump_json()
        matches = re.findall(
            r"\{\{PII_[A-Za-z0-9_]+\}\}|(?:\b|\b\+)PII_(?:EMAIL|PHONE|NAME|NAMEPART|HANDLE)_[A-Za-z0-9_]+\b",
            raw_json,
            flags=re.IGNORECASE,
        )
        return sorted(set(matches))


def anonymize_content(text: str, candidate_name: str | None = None) -> tuple[str, PIIVault]:
    """Helper creating a PIIVault and anonymizing the input content."""
    vault = PIIVault()
    anonymized_text = vault.anonymize_text(text, candidate_name=candidate_name)
    return anonymized_text, vault
