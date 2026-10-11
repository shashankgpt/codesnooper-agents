from typing import Any, TypedDict

from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph

from app.config import get_settings


class AgentState(TypedDict, total=False):
    """Base state representation for agent workflows."""

    messages: list[BaseMessage]
    task_id: str
    context: dict[str, Any]
    result: str | None


def create_llm(model_name: str | None = None) -> ChatGoogleGenerativeAI | None:
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
    )


def create_base_graph() -> StateGraph[AgentState]:
    """Create a minimal LangGraph state graph scaffold.

    Concrete agents will extend this by defining nodes, conditional edges,
    and checkpointing strategies.
    """
    return StateGraph(AgentState)
