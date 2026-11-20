import json
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

SYSTEM_PORTFOLIO_GENERATOR = """You are an expert technical editor and portfolio curator.
Your mission is to transform a candidate's career profile into an editorial developer monograph.

HARD CONSTRAINTS & EDITORIAL PRINCIPLES:
1. ABSOLUTE FACTUAL GROUNDING:
   - Ground 100% of information in the provided profile and experience text.
   - NEVER invent employers, job titles, institutions, projects, metrics, or technologies.
   - If numbers are missing, frame impact around the concrete systems built and scale.

2. PRIVACY & SECURITY:
   - Never output phone numbers or physical street addresses under any circumstances.
   - Preserve all PII tokens (e.g. {{PII_NAME_1}}, {{PII_EMAIL_1}}) exactly as provided.

3. EDITORIAL VOICE & TONE:
   - Voice: Confident, crisp engineering leadership.
   - BANNED CLICHES: Never use "passionate software engineer", "rockstar", or empty buzzwords.
   - Headline: Max 90 characters. Format: "<Title> · <Specialization/Tech>".
   - Bio: Exactly 2 paragraphs (100-160 words total in first person 'I').
     Paragraph 1 focuses on core systems architecture and domain expertise.
     Paragraph 2 covers technical philosophy and current focus.
   - Highlights: 3-4 punchy bento metric items (metric + label).
     Metric must be an exact string (e.g. "8+ yrs", "25k RPS", "99.99%").
     Label must be crisp (e.g. "Production Uptime", "Peak Throughput").

4. PROJECTS & WORK EXPERIENCE:
   - Curate up to 4 standout projects. Classify repo vs live URLs appropriately.
   - Experience: Up to 4 crisp impact bullets per role with concrete action and result.
   - Skill Groups: Group skills logically into 3-6 categories (e.g. "Languages", "Cloud").
     Never create a single flat unsorted list.
"""

SYSTEM_PORTFOLIO_ENHANCE = """You are an expert technical copywriter and portfolio editor.
Your objective is to polish and refine the developer portfolio section.

RULES:
- Preserve all facts, technologies, and metrics from the original text. Do not hallucinate.
- If user instructions are provided, prioritize them while keeping the tone architectural.
- For headline: return a single line under 90 characters.
- For bio: return concise, first-person paragraphs (no buzzwords).
- For project description: focus on problem, architecture, and impact.
- For experience bullets: return concise, impact-focused bullets beginning with strong verbs.
"""


def build_portfolio_generator_prompts(
    profile_data: dict[str, Any],
    preferences_data: dict[str, Any] | None = None,
    raw_text: str | None = None,
) -> list[BaseMessage]:
    """Build the system and user messages for portfolio generation."""
    user_context: dict[str, Any] = {
        "profile": profile_data,
        "preferences": preferences_data or {},
    }
    if raw_text:
        user_context["raw_career_text"] = raw_text

    formatted_context = json.dumps(user_context, indent=2, ensure_ascii=False)
    human_content = (
        "Please generate a complete, curated developer portfolio output "
        "grounded strictly in the following profile data:\n\n"
        f"```json\n{formatted_context}\n```"
    )

    return [
        SystemMessage(content=SYSTEM_PORTFOLIO_GENERATOR),
        HumanMessage(content=human_content),
    ]


def build_enhance_prompt(
    section: str,
    current_text: str,
    instruction: str | None = None,
    context_hint: str | None = None,
) -> list[BaseMessage]:
    """Build system and user messages for polishing a specific section."""
    prompt_lines = [
        f"Target Section: {section}",
        f"Current Content:\n{current_text}",
    ]
    if instruction:
        prompt_lines.append(f"User Refinement Instruction: {instruction}")
    if context_hint:
        prompt_lines.append(f"Additional Context:\n{context_hint}")

    prompt_lines.append("\nPlease provide the polished output:")
    human_content = "\n\n".join(prompt_lines)

    return [
        SystemMessage(content=SYSTEM_PORTFOLIO_ENHANCE),
        HumanMessage(content=human_content),
    ]
