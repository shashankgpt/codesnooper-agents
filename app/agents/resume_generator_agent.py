import asyncio
import json
import logging
import re
from typing import Any, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.base import create_llm_for_provider
from app.models.resume_generator import ResumeGenerationOutput
from app.prompts.resume_generator.v1 import build_resume_generator_prompts

logger = logging.getLogger(__name__)


class ResumeAgentState(TypedDict):
    """Execution state for the resume generator LangGraph agent."""

    profile_data: dict[str, Any]
    preferences_data: NotRequired[dict[str, Any]]
    raw_text: NotRequired[str | None]
    target_role: NotRequired[str | None]
    model: NotRequired[str]
    resume_output: NotRequired[ResumeGenerationOutput | None]
    error: NotRequired[str | None]
    model_used: NotRequired[str]


class ResumeGeneratorAgent:
    """Agent that curates high-signal SWE ATS resumes from candidate profile data."""

    def __init__(self, model_name: str = "gemini") -> None:
        self.default_model = model_name
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        workflow = StateGraph(ResumeAgentState)
        workflow.add_node("curate_resume", self._curate_resume_node)
        workflow.add_edge(START, "curate_resume")
        workflow.add_edge("curate_resume", END)
        return workflow.compile()

    async def _curate_resume_node(self, state: ResumeAgentState) -> ResumeAgentState:
        """Call LLM with structured output to generate curated ATS resume."""
        provider = state.get("model") or self.default_model
        messages = build_resume_generator_prompts(
            profile_data=state["profile_data"],
            preferences_data=state.get("preferences_data"),
            raw_text=state.get("raw_text"),
            target_role=state.get("target_role"),
        )

        llm = create_llm_for_provider(provider)
        structured_llm = llm.with_structured_output(ResumeGenerationOutput, include_raw=True)

        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                result = await structured_llm.ainvoke(messages)
                output: ResumeGenerationOutput | None = None

                if isinstance(result, ResumeGenerationOutput):
                    output = result
                elif isinstance(result, dict):
                    parsed = result.get("parsed")
                    if isinstance(parsed, ResumeGenerationOutput):
                        output = parsed
                    elif isinstance(parsed, dict):
                        output = ResumeGenerationOutput.model_validate(parsed)

                    if output is None:
                        raw = result.get("raw")
                        tool_calls = getattr(raw, "tool_calls", None) if raw is not None else None
                        content = getattr(raw, "content", None) if raw is not None else None
                        if tool_calls:
                            args = tool_calls[0].get("args", {})
                            if isinstance(args, str):
                                args = json.loads(args)
                            output = ResumeGenerationOutput.model_validate(args)
                        elif content:
                            raw_str = content if isinstance(content, str) else str(content)
                            match = re.search(r"\{.*\}", raw_str, re.DOTALL)
                            if match:
                                output = ResumeGenerationOutput.model_validate_json(match.group(0))

                if output is None:
                    raise ValueError("LLM returned empty or unparseable resume output")

                return {
                    **state,
                    "resume_output": output,
                    "model_used": provider,
                    "error": None,
                }
            except Exception as exc:
                last_error = exc
                err_str = str(exc)
                is_transient = any(
                    k in err_str
                    for k in ["503", "429", "UNAVAILABLE", "NoneType", "unparseable", "empty"]
                )
                if is_transient and attempt < 3:
                    await asyncio.sleep(attempt * 2)
                else:
                    break

        logger.error("Failed to curate resume with provider '%s': %s", provider, last_error)
        return {
            **state,
            "resume_output": None,
            "error": str(last_error),
            "model_used": provider,
        }

    async def generate_resume(
        self,
        profile_data: dict[str, Any],
        preferences_data: dict[str, Any] | None = None,
        raw_text: str | None = None,
        target_role: str | None = None,
        model: str = "gemini",
    ) -> ResumeGenerationOutput:
        """Execute resume curation workflow and return structured output."""
        initial_state: ResumeAgentState = {
            "profile_data": profile_data,
            "preferences_data": preferences_data or {},
            "raw_text": raw_text,
            "target_role": target_role,
            "model": model,
        }
        final_state: ResumeAgentState = await self._graph.ainvoke(initial_state)

        resume_output = final_state.get("resume_output")
        if final_state.get("error") or resume_output is None:
            error_msg = final_state.get("error") or "Unknown resume generation error"
            raise RuntimeError(f"Resume curation failed: {error_msg}")

        return resume_output
