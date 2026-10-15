import logging
import re
import time

from app.agents.resume_parser_agent import (
    ResumeParserState,
    build_resume_parser_graph,
    fetch_source_text,
)
from app.config import Settings
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
            # 1. Fetch raw text before entering LangGraph
            raw_text, url_type = await fetch_source_text(str(request.url))

            # 2. Scrub PII immediately - vault is held STRICTLY in controller memory
            anonymized_text, vault = anonymize_content(raw_text)

            # 3. LangGraph state ONLY receives anonymized content and sanitized URL
            initial_state: ResumeParserState = {
                "url": sanitized_url,
                "url_type": url_type,
                "anonymized_content": anonymized_text,
                "model": request.model,
                "prompt_version": "v1",
            }

            # Run configuration with sanitized metadata so no PII leaks to LangSmith
            run_config = {
                "run_name": f"ResumeParserWorkflow-{request.model}",
                "tags": ["resume-parser", request.model, self._settings.app_env],
                "metadata": {
                    "url_type": url_type,
                    "model": request.model,
                },
            }
            result = await self._graph.ainvoke(initial_state, config=run_config)

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
