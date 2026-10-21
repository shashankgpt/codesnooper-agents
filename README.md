# CodeSnooper Agents 🤖

Agent workflows platform powered by LangChain, LangGraph, and FastAPI — deployed on [Modal.com](https://modal.com).

## Tech Stack

| Layer | Technology | Description |
|:---|:---|:---|
| **API** | FastAPI (async, Python 3.12) | High-performance asynchronous REST endpoints |
| **Agents** | LangChain + LangGraph | Graph-based stateful agent workflows |
| **LLM** | Google Gemini (`langchain-google-genai`) | Multimodal LLM reasoning |
| **Database** | Firebase Firestore (`firebase-admin`) | Persistent state and task store |
| **Deployment** | Modal.com (serverless Python SDK) | Ephemeral API containers and long-running background tasks |
| **Dependency Mgmt** | UV | Ultra-fast Python package installer & resolver |
| **Type Checking** | Pyright (strict) | Strict static typing across codebase |
| **Linting & Formatting** | Ruff | Modern all-in-one linter & code formatter |
| **Observability** | Sentry (`sentry-sdk[fastapi]`) | Automated error tracking and performance transactions |
| **Testing** | pytest + pytest-asyncio | Unit and integration test runner |

---

## Prerequisites

- **Python 3.12+** — [python.org](https://www.python.org/downloads/)
- **UV** — Fast Python package manager:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Firebase Service Account** — Download from [Firebase Console](https://console.firebase.google.com) → Project Settings → Service Accounts → Generate New Private Key
- **Google Gemini API Key** — Obtain from [Google AI Studio](https://aistudio.google.com/apikey)
- **Modal Account** (for deployment) — Sign up at [modal.com](https://modal.com)
- **Sentry DSN** (optional for local, recommended for production) — Obtain from [sentry.io](https://sentry.io)

---

## Local Setup

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/shashankgpt/codesnooper-agents.git
cd codesnooper-agents

# Synchronize virtual environment with all runtime and development packages
uv sync --all-extras
```

### 2. Configure Environment

```bash
# Create local .env from example template
cp .env.example .env
```

Update `.env` with your credentials:

```env
# Firebase Admin credentials for Firestore
GOOGLE_APPLICATION_CREDENTIALS=path/to/serviceAccountKey.json

# Google Gemini
GOOGLE_API_KEY=your-gemini-api-key

# Sentry
SENTRY_DSN=https://your-key@o123456.ingest.us.sentry.io/your-project-id
SENTRY_TRACES_SAMPLE_RATE=1.0

# Application
APP_ENV=development
APP_DEBUG=true

# Private Vercel Blob access for uploaded career-profile resumes
CAREER_PROFILE_BLOB_STORE_ID=store_id_from_vercel
CAREER_PROFILE_BLOB_READ_WRITE_TOKEN=vercel_blob_read_write_token
```

### 3. Start Development Server

```bash
uv run fastapi dev app/main.py
```

In development, resume parsing runs as a background task in the local API process. The deployed app dispatches those jobs to Modal.

The application runs at **http://localhost:8000**:
- **Interactive Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **OpenAPI Schema:** [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

### 4. Verify Endpoints

```bash
# Hello endpoint
curl http://localhost:8000/api/v1/hello
# Output: {"message":"Hello from CodeSnooper Agents! 🚀","environment":"development","version":"0.1.0"}

# Health check
curl http://localhost:8000/api/v1/health
# Output: {"status":"healthy","firebase_connected":true}
```

---

## Development & Quality Checks

Run the verification suite using `uv`:

```bash
# Run unit & integration tests
uv run pytest -v

# Run strict static type checking
uv run pyright

# Check code for linting issues
uv run ruff check .

# Automatically fix linting issues
uv run ruff check . --fix

# Verify code formatting
uv run ruff format --check .

# Apply code formatting
uv run ruff format .

# One-liner to run all verification steps
uv run ruff format --check . && uv run ruff check . && uv run pyright && uv run pytest -v
```

---

## Deployment to Modal.com

### 1. Authenticate with Modal

```bash
uv run modal setup
```

This launches a browser prompt to connect your local environment with your Modal account.

### 2. Configure Modal Secrets

Create a secret named **`codesnooper-secrets`** in Modal. You can do this via the [Modal Web Dashboard](https://modal.com/secrets) or through the CLI:

```bash
modal secret create codesnooper-secrets \
  GOOGLE_APPLICATION_CREDENTIALS_JSON='{"type":"service_account",...}' \
  GOOGLE_API_KEY="your-gemini-api-key" \
  LANGSMITH_API_KEY="your-langsmith-api-key" \
  LANGSMITH_TRACING="true" \
  LANGSMITH_AGENT_ID="codesnooper-agent" \
  LANGSMITH_AGENT_ENVIRONMENT="production" \
  LANGSMITH_ENDPOINT="https://api.smith.langchain.com" \
  SENTRY_DSN="https://your-key@o123456.ingest.us.sentry.io/your-project-id" \
  SENTRY_TRACES_SAMPLE_RATE="1.0" \
  CAREER_PROFILE_BLOB_STORE_ID="store_id_from_vercel" \
  CAREER_PROFILE_BLOB_READ_WRITE_TOKEN="vercel_blob_read_write_token" \
  APP_ENV="production"
```

Use the same Blob store ID and read-write token configured for the Next.js app.
Uploaded resume parse requests contain the Blob pathname. The API verifies that
the private file is readable before returning `202 Accepted`. The background
worker downloads it again via Vercel Blob; the API does not pass file bytes to the
worker. Both the API and worker functions require the Blob credentials and share
the `codesnooper-secrets` secret. Combine jobs extract and forward the source text
during preflight so their workers reuse it without downloading the file again.

Resume parse, combine, and job read/delete endpoints require a Firebase ID token
in `Authorization: Bearer <id-token>` and the gateway token in `X-API-Key` (or
`X-Gateway-Token`). The request `uid` must match the verified Firebase user; only
that user can access the job and its uploaded Blob.

LangSmith settings in the local `.env` file are not available in Modal. The
`codesnooper-secrets` secret must contain `LANGSMITH_API_KEY`; tracing is only
enabled when this key is present. Set `LANGSMITH_TRACING` to `true` and keep
`LANGSMITH_ENDPOINT` set to the endpoint for your LangSmith region. In
production, set `LANGSMITH_AGENT_ID` and `LANGSMITH_AGENT_ENVIRONMENT` to route
traces to that agent environment. Do not set `LANGSMITH_PROJECT` at the same
time. Update the secret and redeploy after changing these values.

> [!TIP]
> In cloud serverless platforms like Modal, providing `GOOGLE_APPLICATION_CREDENTIALS_JSON` containing the raw JSON content is preferred over committing a JSON file to disk.

### 3. Deploy Application

```bash
# Ephemeral test deployment (interactive live reload)
uv run modal serve modal_app.py

# Production persistent deployment
uv run modal deploy modal_app.py
```

Upon deployment, Modal provides a public HTTPS web endpoint URL:

```text
✓ Deployed app 'codesnooper-agents' in ...
✓ Created web endpoint → https://shashankgpt--codesnooper-agents-fastapi-app.modal.run
```

Test your deployed endpoint:

```bash
curl https://shashankgpt--codesnooper-agents-fastapi-app.modal.run/api/v1/hello
```

---

## Architecture & Code Structure

The project implements an adapted **MVC + Repository** architectural pattern tailored for async microservices:

```
codesnooper-agents/
├── app/
│   ├── main.py                     # FastAPI application factory & lifespan
│   ├── config.py                   # Pydantic BaseSettings configuration
│   ├── models/                     # Models: Pydantic request & response DTOs
│   │   ├── __init__.py
│   │   └── common.py
│   ├── views/                      # Views: FastAPI route handlers (endpoints)
│   │   ├── __init__.py
│   │   └── health.py
│   ├── controllers/                # Controllers: Business logic & service orchestration
│   │   ├── __init__.py
│   │   └── health_controller.py
│   ├── repositories/               # Repositories: Data access (Firebase Firestore async wrappers)
│   │   ├── __init__.py
│   │   └── firebase_client.py
│   ├── agents/                     # Agent workflows (LangChain & LangGraph graphs)
│   │   ├── __init__.py
│   │   └── base.py
│   └── core/                       # Core infrastructure & cross-cutting utilities
│       ├── __init__.py
│       ├── dependencies.py
│       ├── exceptions.py
│       └── sentry.py
├── tests/                          # Automated tests
│   ├── __init__.py
│   ├── conftest.py
│   └── views/
│       ├── __init__.py
│       └── test_health.py
├── modal_app.py                    # Modal cloud serverless deployment entrypoint
├── pyproject.toml                  # UV dependencies & build configuration
├── pyrightconfig.json              # Pyright strict type checking configuration
├── .python-version                 # Python version pin (3.12)
├── .env.example                    # Environment variable template
└── README.md                       # Documentation
```

### Background Execution Model

- **REST Endpoints (`app/views/`):** Trigger quick validations, return HTTP `202 Accepted` with a job ID, or handle synchronous queries.
- **Background Worker (`modal_app.py:run_agent_task`):** Modal functions invoked via `.spawn()` handle long-running agent graph executions asynchronously without timing out HTTP connections.
- **Scheduled Agents (`modal_app.py:scheduled_agent_check`):** Trigger periodic monitoring or ingestion agents on a cron schedule (`modal.Cron`).
- **State Persistence:** Results and status updates are persisted to Firestore.

---

## API Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/api/v1/hello` | Returns greetings and system runtime information |
| `GET` | `/api/v1/health` | Returns health status and Firebase connection state |

---

## License

MIT
