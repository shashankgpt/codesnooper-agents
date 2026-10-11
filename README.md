# CodeSnooper Agents 🤖

Agent workflows platform powered by LangChain, LangGraph, and FastAPI — deployed on [Modal.com](https://modal.com).

## Tech Stack

| Layer | Technology | Description |
|:---|:---|:---|
| **API** | FastAPI (async, Python 3.12) | High-performance asynchronous REST endpoints |
| **Agents** | LangChain + LangGraph | Graph-based stateful agent workflows |
| **LLM** | Google Gemini (`langchain-google-genai`) | Multimodal LLM reasoning |
| **Database** | Firebase Realtime Database (`firebase-admin`) | Persistent real-time state and task store |
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
# Firebase Realtime Database
FIREBASE_DATABASE_URL=https://your-project.firebaseio.com
GOOGLE_APPLICATION_CREDENTIALS=path/to/serviceAccountKey.json

# Google Gemini
GOOGLE_API_KEY=your-gemini-api-key

# Sentry
SENTRY_DSN=https://your-key@o123456.ingest.us.sentry.io/your-project-id
SENTRY_TRACES_SAMPLE_RATE=1.0

# Application
APP_ENV=development
APP_DEBUG=true
```

### 3. Start Development Server

```bash
uv run fastapi dev app/main.py
```

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
  FIREBASE_DATABASE_URL="https://your-project.firebaseio.com" \
  GOOGLE_APPLICATION_CREDENTIALS_JSON='{"type":"service_account",...}' \
  GOOGLE_API_KEY="your-gemini-api-key" \
  SENTRY_DSN="https://your-key@o123456.ingest.us.sentry.io/your-project-id" \
  SENTRY_TRACES_SAMPLE_RATE="1.0" \
  APP_ENV="production"
```

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
│   ├── repositories/               # Repositories: Data access (Firebase RTDB async wrappers)
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
- **State Persistence:** Results and status updates are persisted back to Firebase Realtime Database.

---

## API Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/api/v1/hello` | Returns greetings and system runtime information |
| `GET` | `/api/v1/health` | Returns health status and Firebase connection state |

---

## License

MIT