import asyncio
import io
import json
import logging
import re
import subprocess
import tempfile
from typing import Any, Literal, NotRequired, TypedDict
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup
from docx import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pypdf import PdfReader

from app.agents.base import create_llm_for_provider
from app.agents.linkedin_scraper import fetch_linkedin_profile
from app.core.pii_anonymizer import PIIVault, anonymize_content
from app.models.resume import ResumeSchema
from app.prompts.resume_parser import DEFAULT_PROMPT_VERSION, get_resume_prompts

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


class ResumeParserState(TypedDict):
    """Execution state for the resume parser agent (strictly PII-free)."""

    url: str
    url_type: NotRequired[Literal["pdf", "doc", "docx", "linkedin", "html"]]
    raw_content: NotRequired[str | None]
    anonymized_content: NotRequired[str | None]
    model: NotRequired[str]
    prompt_version: NotRequired[str]
    resume: NotRequired[ResumeSchema | None]
    error: NotRequired[str | None]
    model_used: NotRequired[str]


def resolve_direct_download_url(url: str) -> str:
    """Transform Google Drive preview/viewer links into direct download stream URLs."""
    parsed = urlparse(url.strip())
    domain = parsed.netloc.lower()

    if "drive.google.com" in domain or "docs.google.com" in domain:
        # Match /file/d/<file_id>/view or /file/d/<file_id>
        match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
        if match:
            file_id = match.group(1)
            return f"https://drive.usercontent.google.com/download?id={file_id}&export=download"

        query_params = parse_qs(parsed.query)
        if "id" in query_params:
            file_id = query_params["id"][0]
            return f"https://drive.usercontent.google.com/download?id={file_id}&export=download"

    return url


def classify_url(url: str) -> Literal["pdf", "doc", "docx", "linkedin", "html"]:
    """Determine whether the target URL is a PDF, LinkedIn profile, or general HTML."""
    parsed = urlparse(url.strip())
    domain = parsed.netloc.lower()
    path = parsed.path.lower()

    # Google Drive document / file preview links
    if "drive.google.com" in domain or "docs.google.com" in domain:
        if "/file/d/" in path or "id=" in parsed.query:
            return "pdf"

    if path.endswith(".docx"):
        return "docx"
    if path.endswith(".doc"):
        return "doc"
    if path.endswith(".pdf") or "pdf" in path.split("/"):
        return "pdf"
    if "linkedin.com" in domain:
        return "linkedin"
    return "html"


async def fetch_pdf_text(
    url: str, timeout: float = 30.0, extra_headers: dict[str, str] | None = None
) -> str:
    """Download a remote PDF and extract text from all readable pages."""
    target_url = resolve_direct_download_url(url)
    headers = {"User-Agent": USER_AGENT, **(extra_headers or {})}
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
        response = await client.get(target_url)
        response.raise_for_status()

    return await asyncio.to_thread(extract_pdf_text, response.content)


def extract_pdf_text(content: bytes) -> str:
    """Extract text from downloaded PDF bytes."""
    reader = PdfReader(io.BytesIO(content))
    extracted_pages: list[str] = []
    for page_idx, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            extracted_pages.append(f"--- Page {page_idx + 1} ---\n{page_text.strip()}")

    full_text = "\n\n".join(extracted_pages).strip()
    if not full_text:
        raise ValueError(
            "Could not extract readable text from PDF (it might be scanned/image-only)."
        )
    return full_text


async def fetch_word_text(
    url: str,
    document_type: Literal["doc", "docx"],
    timeout: float = 30.0,
    extra_headers: dict[str, str] | None = None,
) -> str:
    """Fetch and extract text from a DOC or DOCX document."""
    headers = {"User-Agent": USER_AGENT, **(extra_headers or {})}
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
        response = await client.get(url)
        response.raise_for_status()

    return await extract_word_text(response.content, document_type)


async def extract_word_text(content: bytes, document_type: Literal["doc", "docx"]) -> str:
    """Extract text from downloaded DOC or DOCX bytes."""
    return await asyncio.to_thread(_extract_word_text, content, document_type)


def _extract_word_text(content: bytes, document_type: Literal["doc", "docx"]) -> str:
    """Perform blocking document parsing outside the event loop."""
    if document_type == "docx":
        document = Document(io.BytesIO(content))
        paragraphs = [
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]
        for table in document.tables:
            paragraphs.extend(
                " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                for row in table.rows
            )
        text = "\n".join(paragraph for paragraph in paragraphs if paragraph)
    else:
        with tempfile.NamedTemporaryFile(suffix=".doc") as document_file:
            document_file.write(content)
            document_file.flush()
            try:
                result = subprocess.run(
                    ["antiword", document_file.name], capture_output=True, check=False, timeout=20
                )
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                raise ValueError(
                    "Could not read this DOC file. Please try saving it as DOCX or PDF."
                ) from exc
        if result.returncode != 0:
            raise ValueError("Could not read this DOC file. Please try saving it as DOCX or PDF.")
        text = result.stdout.decode("utf-8", errors="replace").strip()

    if not text:
        raise ValueError("Could not extract readable text from the document.")
    return text


async def fetch_html_text(url: str, timeout: float = 30.0) -> str:
    """Fetch HTML page and extract clean body text."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
        response = await client.get(url)
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract any structured JSON-LD (common in LinkedIn and portfolio sites) before cleaning
    json_ld_blocks: list[str] = []
    for script_tag in soup.find_all("script", type="application/ld+json"):
        if script_tag.string:
            json_ld_blocks.append(script_tag.string.strip())

    # Remove non-content tags
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
        tag.decompose()

    # Extract text and collapse excessive whitespace
    text = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned_text = "\n".join(lines)

    if json_ld_blocks:
        cleaned_text = (
            f"{cleaned_text}\n\n--- STRUCTURED PROFILE METADATA ---\n"
            + "\n".join(json_ld_blocks)
        )

    if not cleaned_text:
        raise ValueError("No visible text found on the webpage.")

    # Guard against LinkedIn login walls
    if "linkedin.com" in url.lower() and (
        "authwall" in response.text.lower() or "join linkedin" in cleaned_text.lower()
    ):
        logger.warning("LinkedIn profile returned login wall/authwall for URL: %s", url)

    return cleaned_text


# --- Graph Nodes ---


def detect_source_node(state: ResumeParserState) -> ResumeParserState:
    """Classify the incoming URL type."""
    detected = classify_url(state["url"])
    return {
        **state,
        "url_type": detected,
    }


async def fetch_source_text(
    url: str,
    source_type: Literal["pdf", "doc", "docx", "linkedin", "html"] | None = None,
    extra_headers: dict[str, str] | None = None,
    source_bytes: bytes | None = None,
) -> tuple[str, Literal["pdf", "doc", "docx", "linkedin", "html"]]:
    """Retrieve raw text from the source URL based on classified type."""
    url_type = source_type or classify_url(url)
    if source_bytes is not None and url_type == "pdf":
        raw_text = await asyncio.to_thread(extract_pdf_text, source_bytes)
    elif source_bytes is not None and url_type in ("doc", "docx"):
        raw_text = await extract_word_text(source_bytes, url_type)
    elif source_bytes is not None:
        raise ValueError("Direct Blob downloads are supported only for PDF, DOC, and DOCX files")
    elif url_type == "pdf":
        raw_text = await fetch_pdf_text(url, extra_headers=extra_headers)
    elif url_type in ("doc", "docx"):
        raw_text = await fetch_word_text(url, url_type, extra_headers=extra_headers)
    elif url_type == "linkedin":
        raw_text = await fetch_linkedin_profile(url)
    else:
        raw_text = await fetch_html_text(url)
    return raw_text, url_type


async def fetch_content_node(state: ResumeParserState) -> ResumeParserState:
    """Retrieve raw text from the source URL and immediately scrub all PII."""
    if state.get("anonymized_content"):
        return state

    url = state["url"]
    try:
        raw_text, detected_type = await fetch_source_text(url)
        anonymized_text, _ = anonymize_content(raw_text)

        return {
            **state,
            "url_type": detected_type,
            "raw_content": None,
            "anonymized_content": anonymized_text,
        }
    except Exception as exc:
        logger.error("Failed to fetch content from %s: %s", url, exc)
        return {
            **state,
            "raw_content": None,
            "anonymized_content": None,
            "error": f"Failed to retrieve resume from URL: {exc}",
        }




async def parse_resume_node(state: ResumeParserState) -> ResumeParserState:
    """Run LLM with structured output to parse text into ResumeSchema asynchronously."""
    content = state.get("anonymized_content") or state.get("raw_content")
    if not content:
        return {
            **state,
            "error": state.get("error") or "No content available to parse.",
        }

    model_name = state.get("model") or "gemini"
    prompt_version = state.get("prompt_version", DEFAULT_PROMPT_VERSION)
    prompts = get_resume_prompts(prompt_version)

    llm = create_llm_for_provider(provider=model_name)
    structured_llm = llm.with_structured_output(ResumeSchema, include_raw=True)

    # Allow full multi-page resumes (up to 150,000 characters) without cutting off earlier jobs
    truncated_content = content[:150000]

    messages = [
        SystemMessage(content=prompts["system_prompt"]),
        HumanMessage(content=prompts["user_prompt_template"].format(content=truncated_content)),
    ]

    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            result: Any = await structured_llm.ainvoke(messages)

            resume_obj: ResumeSchema | None = None

            # 1. Try directly parsed output
            if isinstance(result, ResumeSchema):
                resume_obj = result
            elif isinstance(result, dict):
                parsed = result.get("parsed")
                if isinstance(parsed, ResumeSchema):
                    resume_obj = parsed
                elif isinstance(parsed, dict):
                    resume_obj = ResumeSchema.model_validate(parsed)

                # 2. Fallback: inspect raw message for tool_calls or JSON content if parsed was None
                if resume_obj is None:
                    raw = result.get("raw")
                    if hasattr(raw, "tool_calls") and raw.tool_calls:
                        args = raw.tool_calls[0].get("args", {})
                        if isinstance(args, str):
                            args = json.loads(args)
                        resume_obj = ResumeSchema.model_validate(args)
                    elif hasattr(raw, "content") and raw.content:
                        raw_text = raw.content if isinstance(raw.content, str) else str(raw.content)
                        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                        if match:
                            resume_obj = ResumeSchema.model_validate_json(match.group(0))

            if resume_obj is None:
                raise ValueError("LLM returned empty or unparseable structured output")

            return {
                **state,
                "resume": resume_obj,
                "model_used": model_name,
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
                wait_time = attempt * 2
                logger.warning(
                    "LLM structured output failed on attempt %d (%s). Retrying in %ds...",
                    attempt,
                    err_str[:120],
                    wait_time,
                )
                await asyncio.sleep(wait_time)
            else:
                break

    logger.error("Error during structured extraction with model '%s': %s", model_name, last_error)
    return {
        **state,
        "resume": None,
        "model_used": model_name,
        "error": f"LLM parsing failed: {last_error}",
    }




def route_after_fetch(state: ResumeParserState) -> str:
    """Route to parse_resume if content exists, otherwise terminate with error."""
    has_content = bool(state.get("anonymized_content") or state.get("raw_content"))
    if has_content and not state.get("error"):
        return "parse_resume"
    return END


def build_resume_parser_graph():
    """Assemble and compile the LangGraph resume parsing workflow.

    The graph operates EXCLUSIVELY on anonymized data with placeholders.
    De-anonymization happens in the controller to ensure zero PII is logged in LangSmith.
    """
    workflow = StateGraph(ResumeParserState)

    workflow.add_node("detect_source", detect_source_node)
    workflow.add_node("fetch_content", fetch_content_node)
    workflow.add_node("parse_resume", parse_resume_node)

    workflow.add_edge(START, "detect_source")
    workflow.add_edge("detect_source", "fetch_content")
    workflow.add_conditional_edges(
        "fetch_content",
        route_after_fetch,
        {
            "parse_resume": "parse_resume",
            END: END,
        },
    )
    workflow.add_edge("parse_resume", END)

    return workflow.compile()

