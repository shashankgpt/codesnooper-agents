"""LinkedIn profile scraper using Bright Data Web Scraper Dataset API.

This approach:
- Completely avoids authwalls, bot detection, and IP blocks.
- Requires NO browser / Playwright / Chromium binaries on Modal or local machines.
- Does not risk personal LinkedIn accounts or expire like li_at cookies.
- Directly returns structured profile data (Experience, Education, Skills, Bio).
"""

import json
import logging
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

BRIGHTDATA_DATASET_ID = "gd_l1viktl72bvl7bjuj0"
BRIGHTDATA_SCRAPE_URL = (
    f"https://api.brightdata.com/datasets/v3/scrape"
    f"?dataset_id={BRIGHTDATA_DATASET_ID}&notify=false&include_errors=true"
)


EXCLUDED_PROFILE_KEYS = {
    "activity",
    "people_also_viewed",
    "similar_profiles",
    "articles",
    "posts",
    "avatar",
    "banner_image",
    "default_avatar",
    "memorialized_account",
    "followers",
    "connections",
    "input",
    "input_url",
    "timestamp",
}


def _format_profile_as_text(profile: dict[str, Any]) -> str:
    """Format Bright Data's LinkedIn JSON into structured text for LLM consumption."""
    # Filter out bulky, irrelevant fields (activity feed, posts, social networks) before prompt generation
    clean_profile = {k: v for k, v in profile.items() if k not in EXCLUDED_PROFILE_KEYS}
    sections: list[str] = []

    # Name and Headline
    name = clean_profile.get("name") or f"{clean_profile.get('first_name', '')} {clean_profile.get('last_name', '')}".strip()
    if name:
        sections.append(f"Name: {name}")

    position = clean_profile.get("position") or clean_profile.get("headline")
    if position:
        sections.append(f"Current Position / Headline: {position}")

    location = clean_profile.get("location") or clean_profile.get("city")
    if location:
        sections.append(f"Location: {location}")

    about = clean_profile.get("about")
    if about:
        sections.append(f"\n--- About / Summary ---\n{about}")

    # Experience
    experience_list = clean_profile.get("experience") or []
    if experience_list:
        sections.append("\n--- Experience ---")
        for idx, exp in enumerate(experience_list, start=1):
            title = exp.get("title") or exp.get("position") or "Position"
            company = exp.get("company") or exp.get("company_name") or "Company"
            duration = exp.get("duration") or exp.get("dates_employed") or ""
            desc = exp.get("description") or ""
            exp_text = f"{idx}. {title} at {company} ({duration})"
            if desc:
                exp_text += f"\n   Details: {desc}"
            sections.append(exp_text)

    # Education
    education_list = clean_profile.get("education") or clean_profile.get("educations_details") or []
    if education_list:
        sections.append("\n--- Education ---")
        if isinstance(education_list, list):
            for idx, edu in enumerate(education_list, start=1):
                if isinstance(edu, dict):
                    school = edu.get("title") or edu.get("school") or edu.get("institute") or ""
                    degree = edu.get("degree") or edu.get("field_of_study") or ""
                    period = edu.get("period") or edu.get("dates") or ""
                    sections.append(f"{idx}. {school} - {degree} ({period})")
                elif isinstance(edu, str):
                    sections.append(f"{idx}. {edu}")
        elif isinstance(education_list, str):
            sections.append(education_list)

    # Skills
    skills = clean_profile.get("skills") or []
    if skills:
        sections.append(f"\n--- Skills ---\n{', '.join(str(s) for s in skills)}")

    # Certifications
    certifications = clean_profile.get("certifications") or []
    if certifications:
        sections.append("\n--- Certifications ---")
        for cert in certifications:
            if isinstance(cert, dict):
                sections.append(f"- {cert.get('title', cert)}")
            else:
                sections.append(f"- {cert}")

    # Projects
    projects = clean_profile.get("projects") or []
    if projects:
        sections.append("\n--- Projects ---")
        for proj in projects:
            if isinstance(proj, dict):
                title = proj.get("title") or proj.get("name") or "Project"
                date = proj.get("start_date") or ""
                desc = proj.get("description") or ""
                proj_text = f"- {title}"
                if date:
                    proj_text += f" ({date})"
                if desc:
                    proj_text += f": {desc}"
                sections.append(proj_text)
            elif isinstance(proj, str):
                sections.append(f"- {proj}")

    # Recommendations
    recommendations = clean_profile.get("recommendations") or []
    if recommendations:
        sections.append("\n--- Recommendations ---")
        for rec in recommendations:
            sections.append(f"- {rec}")

    # Append structured raw profile data without activity or bulky social metadata
    sections.append("\n\n--- STRUCTURED RAW PROFILE DATA (JSON) ---")
    sections.append(json.dumps(clean_profile, indent=2, ensure_ascii=False))

    return "\n".join(sections).strip()


async def fetch_linkedin_profile(url: str, timeout_seconds: float = 90.0) -> str:
    """Fetch structured LinkedIn profile using Bright Data's Web Scraper Dataset API.

    Args:
        url: Full LinkedIn profile URL.
        timeout_seconds: Request timeout (dataset extraction typically takes 15-30s).

    Returns:
        Formatted profile text including raw structured JSON.

    Raises:
        ValueError: If Bright Data API key is missing or request fails.
    """
    settings = get_settings()
    api_key = settings.brightdata_api_key

    if not api_key:
        raise ValueError(
            "BRIGHTDATA_API_KEY environment variable is not configured. "
            "Please add BRIGHTDATA_API_KEY to your .env or deployment secrets."
        )

    logger.info("Triggering Bright Data LinkedIn dataset scrape for URL: %s", url)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": [{"url": url}],
        "limit_per_input": None,
    }

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            BRIGHTDATA_SCRAPE_URL,
            headers=headers,
            json=payload,
        )

        if response.status_code != 200:
            error_detail = response.text
            logger.error("Bright Data API returned HTTP %s: %s", response.status_code, error_detail)
            raise ValueError(
                f"Bright Data API failed with HTTP {response.status_code}: {error_detail}"
            )

        data = response.json()

    # The dataset endpoint can return a list or a single profile dictionary
    if isinstance(data, list):
        if not data:
            raise ValueError(f"Bright Data returned an empty profile list for URL: {url}")
        profile_data = data[0]
    elif isinstance(data, dict):
        profile_data = data
    else:
        raise ValueError(f"Unexpected response format from Bright Data: {type(data)}")

    # Check for errors in the payload
    if profile_data.get("error"):
        raise ValueError(f"Bright Data scraping error: {profile_data.get('error')}")

    # Explicitly purge bulky, non-resume keys before formatting
    for key in EXCLUDED_PROFILE_KEYS:
        profile_data.pop(key, None)

    cleaned_text = _format_profile_as_text(profile_data)
    logger.info(
        "Successfully retrieved profile from Bright Data for %s (%d chars formatted)",
        url,
        len(cleaned_text),
    )
    return cleaned_text
