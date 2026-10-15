from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph

from app.config import get_settings


class AgentState(TypedDict, total=False):
    """Base state representation for agent workflows."""

    messages: list[BaseMessage]
    task_id: str
    context: dict[str, Any]
    result: str | None


def create_gemini_llm(model_name: str | None = None) -> ChatGoogleGenerativeAI | None:
    """Create a configured Gemini LLM instance.

    Returns None if GOOGLE_API_KEY is not configured.
    """
    settings = get_settings()
    if not settings.google_api_key:
        return None

    resolved_model = model_name or settings.gemini_model
    return ChatGoogleGenerativeAI(
        model=resolved_model,
        google_api_key=settings.google_api_key,
        temperature=0.1,
    )


def create_mistral_llm(model_name: str | None = None) -> ChatMistralAI | None:
    """Create a configured Mistral LLM instance.

    Returns None if Chat_Mistral_API_KEY / MISTRAL_API_KEY is not configured.
    """
    settings = get_settings()
    api_key = settings.chat_mistral_api_key or settings.mistral_api_key
    if not api_key:
        return None

    resolved_model = model_name or settings.mistral_model
    params: dict[str, Any] = {
        "model": resolved_model,
        "api_key": api_key,
        "temperature": 0.1,
    }
    return ChatMistralAI(**params)


def create_llm(model_name: str | None = None) -> ChatGoogleGenerativeAI | None:
    """Create a configured Gemini LLM instance for backward compatibility."""
    return create_gemini_llm(model_name)


def create_llm_for_provider(
    provider: str = "gemini",
    model_name: str | None = None,
) -> BaseChatModel:
    """Factory creating an LLM for the chosen provider ('gemini' or 'mistral').

    Raises ValueError if provider is unsupported or credentials are missing.
    """
    normalized = provider.strip().lower()
    if normalized == "gemini" or normalized.startswith("gemini"):
        resolved_model = model_name or (normalized if normalized != "gemini" else None)
        llm = create_gemini_llm(resolved_model)
        if not llm:
            raise ValueError(
                "Google API key is not configured. Please set GOOGLE_API_KEY in your environment."
            )
        return llm

    if normalized in {"mistral", "ministral"}:
        llm = create_mistral_llm(model_name)
        if not llm:
            raise ValueError(
                "Mistral API key is not configured. Please set MISTRAL_API_KEY in your environment."
            )
        return llm

    raise ValueError(f"Unsupported model provider: '{provider}'. Supported: 'gemini', 'mistral'")


def create_base_graph() -> StateGraph[AgentState]:
    """Create a minimal LangGraph state graph scaffold.

    Concrete agents will extend this by defining nodes, conditional edges,
    and checkpointing strategies.
    """
    return StateGraph(AgentState)
