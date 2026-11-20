import asyncio
import json
import logging
import re
from typing import Any, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.base import create_llm_for_provider
from app.models.portfolio import (
    PortfolioEnhanceOutput,
    PortfolioGenerationOutput,
)
from app.prompts.portfolio_generator.v1 import (
    build_enhance_prompt,
    build_portfolio_generator_prompts,
)

logger = logging.getLogger(__name__)


class PortfolioAgentState(TypedDict):
    """Execution state for the portfolio generator LangGraph agent."""

    profile_data: dict[str, Any]
    preferences_data: NotRequired[dict[str, Any]]
    raw_text: NotRequired[str | None]
    model: NotRequired[str]
    portfolio_output: NotRequired[PortfolioGenerationOutput | None]
    error: NotRequired[str | None]
    model_used: NotRequired[str]


class PortfolioGeneratorAgent:
    """Agent that curates high-trust developer portfolios and refines sections."""

    def __init__(self, model_name: str = "gemini") -> None:
        self.default_model = model_name
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        workflow = StateGraph(PortfolioAgentState)
        workflow.add_node("curate_portfolio", self._curate_portfolio_node)
        workflow.add_edge(START, "curate_portfolio")
        workflow.add_edge("curate_portfolio", END)
        return workflow.compile()

    async def _curate_portfolio_node(self, state: PortfolioAgentState) -> PortfolioAgentState:
        """Call LLM with structured output to generate curated portfolio monograph."""
        provider = state.get("model") or self.default_model
        messages = build_portfolio_generator_prompts(
            profile_data=state["profile_data"],
            preferences_data=state.get("preferences_data"),
            raw_text=state.get("raw_text"),
        )

        llm = create_llm_for_provider(provider)
        structured_llm = llm.with_structured_output(PortfolioGenerationOutput, include_raw=True)

        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                result = await structured_llm.ainvoke(messages)
                output: PortfolioGenerationOutput | None = None

                if isinstance(result, PortfolioGenerationOutput):
                    output = result
                elif isinstance(result, dict):
                    parsed = result.get("parsed")
                    if isinstance(parsed, PortfolioGenerationOutput):
                        output = parsed
                    elif isinstance(parsed, dict):
                        output = PortfolioGenerationOutput.model_validate(parsed)

                    if output is None:
                        raw = result.get("raw")
                        tool_calls = getattr(raw, "tool_calls", None) if raw is not None else None
                        content = getattr(raw, "content", None) if raw is not None else None
                        if tool_calls:
                            args = tool_calls[0].get("args", {})
                            if isinstance(args, str):
                                args = json.loads(args)
                            output = PortfolioGenerationOutput.model_validate(args)
                        elif content:
                            raw_str = (
                                content if isinstance(content, str) else str(content)
                            )
                            match = re.search(r"\{.*\}", raw_str, re.DOTALL)
                            if match:
                                output = PortfolioGenerationOutput.model_validate_json(
                                    match.group(0)
                                )

                if output is None:
                    raise ValueError("LLM returned empty or unparseable portfolio output")

                return {
                    **state,
                    "portfolio_output": output,
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

        logger.error("Failed to curate portfolio with provider '%s': %s", provider, last_error)
        return {
            **state,
            "portfolio_output": None,
            "error": str(last_error),
            "model_used": provider,
        }

    async def generate_portfolio(
        self,
        profile_data: dict[str, Any],
        preferences_data: dict[str, Any] | None = None,
        raw_text: str | None = None,
        model: str = "gemini",
    ) -> PortfolioGenerationOutput:
        """Execute portfolio curation workflow and return structured output."""
        initial_state: PortfolioAgentState = {
            "profile_data": profile_data,
            "preferences_data": preferences_data or {},
            "raw_text": raw_text,
            "model": model,
        }
        final_state: PortfolioAgentState = await self._graph.ainvoke(initial_state)

        portfolio_output = final_state.get("portfolio_output")
        if final_state.get("error") or portfolio_output is None:
            error_msg = final_state.get("error") or "Unknown portfolio generation error"
            raise RuntimeError(f"Portfolio curation failed: {error_msg}")

        return portfolio_output

    async def enhance_section(
        self,
        section: str,
        current_text: str,
        instruction: str | None = None,
        context_hint: str | None = None,
        model: str = "gemini",
    ) -> PortfolioEnhanceOutput:
        """Polish a specific portfolio section with LLM."""
        messages = build_enhance_prompt(
            section=section,
            current_text=current_text,
            instruction=instruction,
            context_hint=context_hint,
        )
        llm = create_llm_for_provider(model)
        structured_llm = llm.with_structured_output(PortfolioEnhanceOutput, include_raw=True)

        result = await structured_llm.ainvoke(messages)
        output: PortfolioEnhanceOutput | None = None

        if isinstance(result, PortfolioEnhanceOutput):
            output = result
        elif isinstance(result, dict):
            parsed = result.get("parsed")
            if isinstance(parsed, PortfolioEnhanceOutput):
                output = parsed
            elif isinstance(parsed, dict):
                output = PortfolioEnhanceOutput.model_validate(parsed)
            if output is None:
                raw = result.get("raw")
                tool_calls = getattr(raw, "tool_calls", None) if raw is not None else None
                if tool_calls:
                    args = tool_calls[0].get("args", {})
                    if isinstance(args, str):
                        args = json.loads(args)
                    output = PortfolioEnhanceOutput.model_validate(args)

        if output is None:
            raise ValueError(f"Failed to enhance portfolio section '{section}'")

        return output
