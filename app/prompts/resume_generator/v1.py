import json
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

SYSTEM_RESUME_GENERATOR = """You are an expert technical resume writer and ATS optimization specialist for top-tier software engineering roles.
Your mission is to transform a candidate's career profile into an elite, single-column SWE ATS resume formatted for maximum parser score and hiring manager signal.

HARD CONSTRAINTS & ATS CURATION PRINCIPLES:
1. ABSOLUTE FACTUAL GROUNDING:
   - Ground 100% of information in the provided profile, projects, and career text.
   - NEVER invent or hallucinate employers, job titles, universities, metrics, or technologies.
   - If numbers/metrics are absent, frame impact around concrete systems built, technical complexity, throughput, scale, or latency.

2. PRIVACY & PII TOKEN PRESERVATION:
   - Preserve all PII tokens (e.g. {{PII_NAME_1}}, {{PII_EMAIL_1}}, {{PII_PHONE_1}}, {{PII_HANDLE_1}}) exactly as provided.
   - Never replace a PII token with invented contact details or alter placeholder tokens.
   - Map PII tokens into their designated fields in personalInfo (fullName, email, phone, etc.).

3. STRUCTURE & FORMATTING RULES:
   - Document Title: "<Candidate Name> — <Target Role or Primary Title> Resume".
   - Personal Info:
     - summary: Crisp 2-3 sentence third-person summary highlighting core technical strengths, architectures, and domain focus. BANNED: Clichés like "results-driven", "passionate developer", "hard worker", or "rockstar".
     - jobTitle: Primary target role (e.g. "Senior Software Engineer", "Staff Backend Engineer", "Full Stack Developer").
   - Work Experience:
     - Reverse-chronological order.
     - description: Formatted STRICTLY as clean HTML bullet lists using <ul><li>...</li></ul>.
     - 3-5 high-impact bullets per role. Each bullet must follow the Action Verb + Context/Problem + Metric/Impact formula.
     - Dates: Formatted with spaced en-dash (e.g. "Jan 2022 \u2013 Present", "Aug 2019 \u2013 Dec 2021").
   - Technical Skills:
     - Categorize skills into 3-5 logical SWE groups (e.g., "Languages", "Frameworks & Libraries", "Cloud & DevOps", "Databases & Storage").
     - Never return an uncategorized flat list.
   - Projects:
     - Up to 4 standout engineering projects.
     - description: Formatted STRICTLY as clean HTML bullet lists using <ul><li>...</li></ul> detailing technical architecture and concrete achievements.
     - Include techStack list and repo/live link if available.
   - Education:
     - Reverse-chronological order. Include institution, degree, and dates.
   - Custom Sections:
     - If the candidate has notable certifications, awards, publications, or patents in their profile or narrative context, create a "Certifications & Awards" custom section with position "main".
   - ATS Tags:
     - 8-12 core technical keywords grounded strictly in candidate profile (e.g. "Distributed Systems", "TypeScript", "React", "Node.js", "Docker", "CI/CD").
"""


def build_resume_generator_prompts(
    profile_data: dict[str, Any],
    preferences_data: dict[str, Any] | None = None,
    raw_text: str | None = None,
    target_role: str | None = None,
) -> list[BaseMessage]:
    """Build the system and user messages for ATS resume generation."""
    user_context: dict[str, Any] = {
        "profile": profile_data,
        "preferences": preferences_data or {},
    }
    if target_role:
        user_context["target_role_hint"] = target_role
    if raw_text:
        user_context["raw_career_text"] = raw_text

    formatted_context = json.dumps(user_context, indent=2, ensure_ascii=False, default=str)
    role_instruction = (
        f" Tailor the resume framing and summary towards the target role: '{target_role}'."
        if target_role
        else ""
    )
    human_content = (
        f"Please generate a complete, curated SWE ATS resume matching ResumeGenerationOutput{role_instruction} "
        "grounded strictly in the following profile data:\n\n"
        f"```json\n{formatted_context}\n```"
    )

    return [
        SystemMessage(content=SYSTEM_RESUME_GENERATOR),
        HumanMessage(content=human_content),
    ]
