# CodeSnooper Frontend — Integration Guide for Agents Backend

> This document gives the **codesnooper-agents** backend team full context on the
> companion **codesnooper-nextjs** frontend application: what it does, its
> architecture, what data it manages, and how the agents backend fits in.

---

## 1. What Is codesnooper-nextjs?

`codesnooper-nextjs` is the **user-facing web application** for CodeSnooper — a
tailored resume platform for multi-hat developers. It is a full-stack Next.js
application that handles:

- User authentication & profiles
- Resume building (9 templates) + PDF export
- AI-powered cover letter generation
- Job search & ATS matchability scoring
- Job application tracking (Kanban pipeline)
- Guest resume builder (no login required)

| Concern | Technology |
|:---|:---|
| Framework | Next.js 14 (Pages Router) |
| Language | TypeScript |
| Styling | Tailwind CSS |
| Auth | NextAuth (Google OAuth + Email/Password via Firebase Auth) |
| Database (User Data) | Firebase Firestore |
| File Storage | Vercel Blob |
| AI (In-Process) | LangChain.js + Google Gemini / Mistral |
| Deployment | Vercel |
| Observability | Sentry (`@sentry/nextjs`) |
| Testing | Jest + Playwright |

**Repository:** `codesnooper-nextjs/` (sibling directory to this project)

---

## 2. Application Architecture

```
codesnooper-nextjs/
├── src/
│   ├── pages/                    # Next.js pages (routes)
│   │   ├── index.tsx             # Landing page
│   │   ├── resume/               # Resume builder & viewer
│   │   ├── cover-letter/         # AI cover letter builder
│   │   ├── jobs/                 # Job search & listings
│   │   ├── my-profile/           # User profile management
│   │   ├── ai/                   # AI features pages
│   │   ├── about-me/             # About / portfolio page
│   │   ├── blogs/                # Blog platform
│   │   └── api/                  # Server-side API routes
│   │       ├── ai/               # AI streaming endpoints (cover letter, bio rewrite)
│   │       ├── resume/           # Resume CRUD operations
│   │       ├── job-match/        # JD parsing, ATS scoring, optimization
│   │       ├── jobs/             # Job search (Adzuna, JSearch)
│   │       ├── job-tracker/      # Application pipeline CRUD
│   │       ├── auth/             # NextAuth handlers
│   │       ├── upload/           # File upload (Vercel Blob)
│   │       └── settings/         # User & site settings
│   ├── features/                 # Feature modules
│   │   ├── auth/                 # Auth logic, schemas, hooks
│   │   ├── resume/               # Resume templates, state, components
│   │   ├── job-match/            # Job matching algorithms & services
│   │   ├── job-tracker/          # Application tracking pipeline
│   │   └── portfolio/            # Portfolio builder
│   ├── components/               # Shared UI components
│   ├── lib/                      # Core utilities
│   │   ├── firebase.ts           # Firebase client SDK initialization
│   │   ├── firebaseAdmin.ts      # Firebase Admin SDK (server-side)
│   │   ├── logger.ts             # Structured logging
│   │   └── api/                  # API utilities (rate limiting, etc.)
│   ├── shared/                   # Shared constants, helpers
│   ├── styles/                   # Global CSS / Tailwind config
│   └── middleware.ts             # Route protection middleware
├── modules/                      # Server-side business logic
│   ├── ai/                       # LangChain.js AI services
│   ├── user/                     # User data access
│   ├── blogs/                    # Blog data access
│   ├── portfolio/                # Portfolio data access
│   └── settings/                 # Site settings management
└── public/                       # Static assets
```

---

## 3. Auth & Session Model

### Authentication Flow

```
User → Google OAuth / Email+Password
  ↓
Firebase Authentication (identity provider)
  ↓
NextAuth (session management — JWT tokens)
  ↓
Next.js Middleware (route protection)
```

### Session Token

All authenticated API routes receive user identity via Next.js middleware headers:

```
x-user-id: <firebase_uid>
x-user-email: <user_email>
```

### Route Protection Rules

| Route Type | Protected? | Notes |
|:---|:---|:---|
| Public pages (`/`, `/jobs`, `/blogs/*`) | ❌ | Freely accessible |
| Guest builder (`/resume/builder/guest`) | ❌ | Freemium flow, no login |
| Dashboard routes (`/resume/dashboard`) | ✅ | Requires auth |
| All `/api/*` (except auth, blogs, jobs) | ✅ | Requires session token |

---

## 4. Data Models & Shared State

### Firebase Firestore Collections (Frontend-Managed)

| Collection | Description | Used By |
|:---|:---|:---|
| `users` | User profiles, preferences | Frontend only |
| `resumes` | Saved resume data (per user) | Frontend only |
| `applications` | Job tracker pipeline entries | Frontend only |
| `blogs` | Blog posts and drafts | Frontend only |
| `portfolios` | Public portfolio pages | Frontend only |
| `site_settings` | Global app configuration | Frontend only |

### Firebase Firestore (Agents-Managed)

| Path | Description | Used By |
|:---|:---|:---|
| `resume_parse_jobs/{job_id}` | Background resume parse job status and result | Agents backend, polled by frontend |

> **Note:** Both the frontend and agents backend use **Firestore** in the same
> Firebase project. The agents store temporary resume job state in the
> `resume_parse_jobs` collection.

---

## 5. Frontend AI Features (In-Process)

The frontend runs some AI features directly within Next.js API routes using
LangChain.js. These are **separate from the agents backend**:

### `/api/ai` — AI Content Generation

| Action | Input | Output | LLM |
|:---|:---|:---|:---|
| `cover-letter` | Job title, company, job description, resume highlights | Streaming cover letter text | Gemini / Mistral |
| `rewrite-bio` | Existing bio text, target tone | Streaming improved bio | Gemini / Mistral |
| `improve-text` | Raw text, improvement instructions | Streaming improved text | Gemini / Mistral |

### `/api/job-match/*` — Resume ↔ Job Matching

| Endpoint | Purpose |
|:---|:---|
| `POST /api/job-match/parse` | Parse a job description into structured format |
| `POST /api/job-match/analyze` | Deep analysis of resume vs. job description |
| `POST /api/job-match/score-single` | Calculate ATS matchability score (single JD) |
| `POST /api/job-match/score-all` | Score resume against multiple saved applications |
| `POST /api/job-match/optimize` | Suggest resume bullet point improvements |

All of the above run LangChain.js **in-process** in Vercel serverless functions.
They do **not** call the agents backend.

---

## 6. Where the Agents Backend Fits In

### Current Integration Points

| Feature | Handled By | Rationale |
|:---|:---|:---|
| Cover letter generation | Frontend (LangChain.js) | Simple streaming, low latency |
| Bio rewrite / text improvement | Frontend (LangChain.js) | Simple streaming, low latency |
| Job description parsing | Frontend (LangChain.js) | In-context with resume data |
| ATS matchability scoring | Frontend (LangChain.js) | In-context with resume data |
| **Career profile parsing** | **Agents backend** | Public resume/LinkedIn URLs and private uploaded PDF, DOC, or DOCX files; PII protection and multi-step parsing |

### Why Some Features Need the Agents Backend

1. **Long-running operations** — document download, LinkedIn API calls, and multi-step
   LLM workflows can take 10-90 seconds, exceeding Vercel's function timeout.
2. **Python-only dependencies** — `pypdf`, `beautifulsoup4`, `playwright` for
   content extraction; Bright Data SDK for LinkedIn.
3. **PII protection** — The PIIVault pipeline (anonymize → LLM → restore) is
   implemented in Python and must run server-side.
4. **Background job execution** — Modal's `.spawn()` enables async agent tasks
   that don't block HTTP connections.

### Future Integration Points

The agents backend should be used for any feature that:
- Requires >30 seconds of processing
- Needs web scraping or external data fetching
- Involves multi-step agent workflows (LangGraph)
- Requires PII-safe LLM processing

---

## 7. API Keys & Environment Variables

### Shared Between Both Projects

| Variable | Frontend Name | Backend Name | Source |
|:---|:---|:---|:---|
| Firebase Project ID | `FIREBASE_PROJECT_ID` | `FIREBASE_PROJECT_ID` | Firebase Console |
| Firebase Client Email | `FIREBASE_CLIENT_EMAIL` | `FIREBASE_CLIENT_EMAIL` | Firebase Console |
| Firebase Private Key | `FIREBASE_PRIVATE_KEY` | `FIREBASE_PRIVATE_KEY` | Firebase Console |
| Google Gemini API Key | `GOOGLE_API_KEY` | `GOOGLE_API_KEY` | Google AI Studio |
| Mistral API Key | `Chat_Mistral_API_KEY` | `CHAT_MISTRAL_API_KEY` | Mistral Console |
| Sentry DSN | `NEXT_PUBLIC_SENTRY_DSN` | `SENTRY_DSN` | Sentry (separate DSNs) |

### Frontend-Only Variables

| Variable | Purpose |
|:---|:---|
| `NEXTAUTH_SECRET` | JWT signing secret |
| `NEXT_PUBLIC_FIREBASE_*` | Firebase client SDK config (public) |
| `BLOB_READ_WRITE_TOKEN` | Vercel Blob storage |
| `ADZUNA_APP_ID` / `ADZUNA_APP_KEY` | Job search API |
| `JSEARCH_API_KEY` | Job search API |
| `SEND_MAIL_API_KEY` | Email sending |

### Backend-Only Variables

| Variable | Purpose |
|:---|:---|
| `API_GATEWAY_TOKEN` | Token auth for agent API endpoints |
| `GOOGLE_APPLICATION_CREDENTIALS` | Firebase service account JSON file used by the backend |
| `BRIGHTDATA_API_KEY` | LinkedIn profile scraping via Bright Data |
| `LANGSMITH_API_KEY` | LangSmith tracing |
| `LANGSMITH_PROJECT` | LangSmith project name |

---

## 8. Frontend Deployment (Vercel)

| Concern | Detail |
|:---|:---|
| Platform | Vercel |
| Framework | Next.js 14 (Pages Router) |
| Build | `next build` |
| Runtime | Node.js 18+ (Vercel Edge + Serverless) |
| Static Assets | Served by Vercel CDN |
| Domain | TBD (custom domain pending) |
| Max Function Duration | 60s (Vercel Pro), 10s (Vercel Hobby) |

---

## 9. Frontend Feature Status

| Category | Features | Status |
|:---|:---|:---|
| **Auth** | Google OAuth + Email/Password, email verification, password reset | ✅ Production-ready |
| **Resume Builder** | 9 templates (Modern, Professional, Minimal, Creative, Executive, Technical, etc.) | ✅ Production-ready |
| **Guest Builder** | Frictionless builder without mandatory login | ✅ Working |
| **AI Cover Letter** | Streaming LLM generation via LangChain.js | ✅ Working |
| **Job Search** | Adzuna & JSearch integrations | ✅ Working |
| **Job Match** | ATS matchability score, JD parser, resume optimization | ✅ Working |
| **Application Tracker** | Kanban pipeline (Applied → Offer → Accepted/Rejected) | ✅ Working |
| **Blogs** | Blog writing and reading platform | ✅ Working |
| **Portfolio** | Public portfolio page builder | ✅ Working |

---

## 10. Monetization & Business Context

Understanding the business context helps the agents team prioritize features:

- **ICP:** Mid-to-senior engineers wearing multiple hats (Frontend, Backend, DevOps)
- **Core Value Proposition:** 1-click tailored resume with zero-hallucination guardrails
- **Pricing:** Free taste (1 export) → $15/mo Pro or $35 one-time "Get Hired" pass
- **Current Status:** 0 paying customers — validating willingness to pay
- **Scope Freeze:** Gamification, blog platform expansion, portfolio custom subdomains,
  and autonomous web crawlers are frozen until 50 paying customers

### What This Means for Agent Development

1. **Resume parsing is the highest-priority agent** — it directly enables the
   core "1-click tailor" flow.
2. **Don't build autonomous agents yet** — they are frozen per business strategy.
3. **Keep latency low** — the "60-second" promise means the entire parse + tailor
   pipeline must complete within 60 seconds.
4. **Zero hallucination is critical** — agents must extract only what exists in
   the source material, never fabricate skills or experience.
