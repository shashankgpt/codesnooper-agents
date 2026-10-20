import logging
import re
import time

from langsmith import tracing_context

from app.agents.linkedin_scraper import (
    fetch_linkedin_raw_data,
    map_linkedin_dict_to_resume_schema,
)
from app.agents.resume_parser_agent import (
    ResumeParserState,
    build_resume_parser_graph,
    classify_url,
    fetch_source_text,
)
from app.config import Settings, flush_tracing
from app.core.pii_anonymizer import PIIVault, anonymize_content
from app.models.resume import (
    ResumeParseRequest,
    ResumeParseResponse,
    ResumeSchema,
)

logger = logging.getLogger(__name__)


def _sanitize_url_for_logging(url: str) -> str:
    """Mask personal usernames or handles from URLs for privacy in logs and tracing."""
    return re.sub(
        r"(linkedin\.com/in/)[a-zA-Z0-9_\-]+",
        r"\1[REDACTED_HANDLE]",
        url,
        flags=re.IGNORECASE,
    )


class ResumeController:
    """Controller orchestrating resume parsing requests and agent workflows."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._graph = build_resume_parser_graph()

    async def parse_resume(self, request: ResumeParseRequest) -> ResumeParseResponse:
        """Execute the resume parsing agent on the provided URL."""
        start_time = time.perf_counter()
        sanitized_url = _sanitize_url_for_logging(str(request.url))
        logger.info(
            "Initiating resume parse for URL '%s' with model provider '%s'",
            sanitized_url,
            request.model,
        )

        try:
            url_str = str(request.url)
            url_type = classify_url(url_str)
            requested_model = (request.model or "auto").lower().strip()

            # Determine whether to use direct Python mapping or AI pipeline
            use_python_mapping = (
                (requested_model in ("auto", "python", "python-mapping", "mapping") and url_type == "linkedin")
                or (requested_model in ("python", "python-mapping", "mapping"))
            )

            if use_python_mapping:
                if url_type != "linkedin":
                    raise ValueError(
                        f"Python mapping is only available for structured LinkedIn URLs, but detected '{url_type}'. "
                        "Please specify model='gemini' or model='mistral' to use AI parsing with PII protection."
                    )

                # Direct Python mapping for LinkedIn profiles (instant, 100% accurate, zero-LLM, NO PIIVault)
                logger.info("Processing LinkedIn profile via deterministic Python mapping (zero-LLM, NO PIIVault)")
                scrape_start = time.perf_counter()
                profile_dict = await fetch_linkedin_raw_data(url_str)
                scrape_elapsed = time.perf_counter() - scrape_start

                map_start = time.perf_counter()
                resume = map_linkedin_dict_to_resume_schema(profile_dict, url_str)
                map_elapsed = time.perf_counter() - map_start

                logger.info(
                    "LinkedIn profile fetched in %.2fs (Bright Data network scrape) and mapped in %.4fs (Python in-memory mapping)",
                    scrape_elapsed,
                    map_elapsed,
                )
                return self._create_response(
                    start_time=start_time,
                    success=True,
                    url=url_str,
                    url_type="linkedin",
                    model_used="python-mapping",
                    resume=resume,
                    error=None,
                )

            # --- AI Pipeline with PIIVault ---
            ai_model = "gemini" if requested_model == "auto" else request.model
            logger.info("Processing resume via AI agent ('%s') with PIIVault protection", ai_model)

            # 1. Fetch raw text before entering LangGraph for PDFs, HTML, or AI-requested LinkedIn profiles
            raw_text, url_type = await fetch_source_text(url_str)

            # 2. Scrub PII immediately - vault is held STRICTLY in controller memory
            anonymized_text, vault = anonymize_content(raw_text)

            # 3. LangGraph state ONLY receives anonymized content and sanitized URL
            initial_state: ResumeParserState = {
                "url": sanitized_url,
                "url_type": url_type,
                "anonymized_content": anonymized_text,
                "model": ai_model,
                "prompt_version": "v1",
            }

            # Run configuration with sanitized metadata so no PII leaks to LangSmith
            run_config = {
                "run_name": f"ResumeParserWorkflow-{ai_model}",
                "tags": ["resume-parser", ai_model, self._settings.app_env],
                "metadata": {
                    "url_type": url_type,
                    "model": ai_model,
                },
            }
            try:
                trace_destination = (
                    {}
                    if self._settings.langsmith_uses_agent_address
                    else {"project_name": self._settings.langsmith_project}
                )
                with tracing_context(
                    enabled=bool(self._settings.langsmith_api_key and self._settings.langsmith_tracing),
                    **trace_destination,
                ):
                    result = await self._graph.ainvoke(initial_state, config=run_config)
            finally:
                if self._settings.langsmith_api_key and self._settings.langsmith_tracing:
                    flush_tracing()

            error = result.get("error")
            anonymized_resume = result.get("resume")

            # 4. De-anonymize in controller memory AFTER LangSmith trace has completed
            final_resume = None
            if anonymized_resume and vault:
                try:
                    final_resume = vault.restore_resume(anonymized_resume)
                except ValueError as pii_err:
                    logger.error("De-anonymization failed: %s", pii_err)
                    return self._create_response(
                        start_time=start_time,
                        success=False,
                        url=str(request.url),
                        url_type=url_type,
                        model_used=result.get("model_used", request.model),
                        error=str(pii_err),
                    )
            elif anonymized_resume:
                final_resume = anonymized_resume

            # Secondary guard: ensure final_resume never contains any residual PII placeholders
            if final_resume:
                residuals = PIIVault.find_residuals(final_resume)
                if residuals:
                    logger.error("Residual PII placeholders detected in final resume: %s", residuals)
                    return self._create_response(
                        start_time=start_time,
                        success=False,
                        url=str(request.url),
                        url_type=url_type,
                        model_used=result.get("model_used", request.model),
                        error=(
                            f"PII de-anonymization failed: residual placeholders ({', '.join(residuals)}) "
                            "detected in response."
                        ),
                    )

            success = error is None and final_resume is not None

            return self._create_response(
                start_time=start_time,
                success=success,
                url=str(request.url),
                url_type=url_type,
                model_used=result.get("model_used", request.model),
                resume=final_resume,
                error=error,
            )

        except Exception as exc:
            logger.exception("Unexpected error during resume processing: %s", exc)
            return self._create_response(
                start_time=start_time,
                success=False,
                url=str(request.url),
                url_type="html",
                model_used=request.model,
                error=f"Internal agent error: {exc}",
            )

    @staticmethod
    def _create_response(
        start_time: float,
        success: bool,
        url: str,
        url_type: str,
        model_used: str,
        resume: ResumeSchema | None = None,
        error: str | None = None,
    ) -> ResumeParseResponse:
        """Construct ResumeParseResponse with calculated duration."""
        elapsed_seconds = time.perf_counter() - start_time
        elapsed_ms = round(elapsed_seconds * 1000, 2)
        return ResumeParseResponse(
            success=success,
            url=url,
            url_type=url_type,
            model_used=model_used,
            response_time=f"{elapsed_seconds:.2f}s",
            latency_ms=elapsed_ms,
            resume=resume if success else None,
            error=error,
        )
