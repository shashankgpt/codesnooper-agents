# AGENTS.md

> **Single source of truth** for all AI agents working on `codesnooper-agents`.
> `GEMINI.md` and `CLAUDE.md` just point here. Update only this file.

---

## Principles

- **Simple code.** No over-engineering. No defensive coding. Fewer lines > more lines.
- **Reuse.** Extract shared logic into base classes, utils, or services. Don't duplicate.
- **Small modules.** One file = one responsibility.
- **Don't assume.** If it's a business decision or ambiguous — stop and ask the user.
- **Parallelise.** If tasks are independent, run them in parallel.

---

## Repo Boundaries

| What | Where |
|:---|:---|
| AI agent code, LangGraph, Python pipelines | **This repo** (`codesnooper-agents`) |
| Frontend, UI, API routes, in-app AI | **`codesnooper-nextjs`** (separate repo) |

All AI/ML agent code belongs here. Don't put it in the frontend repo.

Frontend context: [`docs/FRONTEND_CONTEXT.md`](./docs/FRONTEND_CONTEXT.md)

---

## Before You Commit

> [!CAUTION]
> **All of these must pass. No exceptions.**

```bash
uv run ruff check .              # Lint
uv run ruff format --check .     # Format check
uv run pyright                   # Type check (strict)
uv run pytest                    # Tests
```

If lint or tests fail, fix them first.

---

## Testing

> [!IMPORTANT]
> **⚠️ ALWAYS USE GEMINI FLASH AGENT FOR ALL TESTING. NO THINKING MODELS. NO EXCEPTIONS.**
> Unit tests, integration tests, browser testing — ALL of it.

- pytest + pytest-asyncio
- Tests go in `tests/`
- Mock external APIs. Never hit live endpoints in tests.

---

## Rules

- **No secrets in git.** No `.env`, API keys, or tokens.
- **Zero lint errors.** Ruff is strict.
- **No `Any` types.** Pyright strict mode.
- **Pydantic** for all request/response models.
- **`app/config.py`** for env vars. No hardcoded credentials.
- **Structured logging.** No raw `print()`.
- **Async-first.** Use `async def` for routes and services.

---

## Workflow

1. **Context** — Check branch, git status, relevant models/controllers.
2. **Test first** — Write/update tests before implementation.
3. **Implement** — MVC + Repository pattern. Small files. Reuse existing code.
4. **Verify** — Run lint, type-check, tests (see above).
5. **Report** — Summarise changes with `file://` links.

---

## Tech Stack

FastAPI (async) · Python 3.12+ · LangChain + LangGraph
Google Gemini · Firebase Firestore · Modal.com (serverless)
UV · Pyright (strict) · Ruff · Sentry · pytest

---

## Commands

```bash
uv run fastapi dev app/main.py   # Dev server
uv run ruff check .              # Lint
uv run ruff format .             # Format
uv run pyright                   # Type check
uv run pytest                    # Tests
uv run modal serve modal_app.py  # Test deploy
uv run modal deploy modal_app.py # Prod deploy
```
