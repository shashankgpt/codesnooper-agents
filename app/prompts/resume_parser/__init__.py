"""Resume parser prompt module registry."""

from typing import Literal

from app.prompts.resume_parser import v1

DEFAULT_PROMPT_VERSION = "v1"

PROMPT_REGISTRY = {
    "v1": v1,
}

PromptVersion = Literal["v1"]


def get_resume_prompts(version: str = DEFAULT_PROMPT_VERSION):
    """Retrieve prompt set for the given version string.

    Falls back to default version if not found.
    """
    module = PROMPT_REGISTRY.get(version, v1)
    return {
        "version": module.VERSION,
        "system_prompt": module.SYSTEM_PROMPT,
        "user_prompt_template": module.USER_PROMPT_TEMPLATE,
    }
