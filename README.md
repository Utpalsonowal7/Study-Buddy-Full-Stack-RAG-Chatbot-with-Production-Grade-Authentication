# Study Buddy — Document RAG & Authentication API

A FastAPI backend that turns study documents into a searchable knowledge base. Users upload notes, ask questions grounded in their documents, receive answers with source citations, and revisit saved conversations. Gemini generates answers through the official Google Gen AI SDK; PostgreSQL and pgvector handle retrieval; Cloudinary stores the original files privately.

The repository contains the backend. Frontend integration is documented in [RAG_API.md](RAG_API.md).

## What the project demonstrates

- **Document retrieval:** PDF, TXT, and Markdown extraction, overlapping chunks, batched 3072-dimensional embeddings, and cosine similarity search in PostgreSQL.
- **Incremental chat:** actual Gemini text deltas delivered over Server-Sent Events, with heartbeat comments and browser cancellation support.
- **Persistent conversations:** follow-up questions use recent history; answers retain source excerpts, page references, and similarity scores.
- **User isolation:** document and conversation access is checked against the authenticated user before retrieval or streaming starts.
- **Private file storage:** authenticated Cloudinary assets and short-lived download URLs rather than public document links.
- **Authentication:** email OTP, Google/GitHub OAuth routes, HttpOnly JWT cookies, hashed refresh tokens, session rotation, and logout.
- **Failure handling:** unfinished chat writes roll back on generation failure or early disconnect; failed indexing attempts to remove its uploaded Cloudinary asset.
- **Operational checks:** Redis rate limits and a public readiness endpoint for PostgreSQL and Redis.

## Architecture

```mermaid
flowchart LR
    frontend["Frontend"] --> api["FastAPI routes"]
    api --> auth["Cookie authentication and ownership checks"]
    auth --> rag["RAG service"]
    api --> redisStore[("Redis: OTP and rate limits")]
    rag --> extraction["Extract and chunk documents"]
    extraction --> embedding["Gemini embedding SDK"]
    embedding --> vectorStore[("PostgreSQL and pgvector")]
    rag --> fileStore["Private Cloudinary files"]
    rag --> vectorStore
    vectorStore --> context["Retrieved excerpts and history"]
    context --> generation["Gemini generation SDK"]
    generation --> stream["SSE text deltas"]
    stream --> frontend
    rag --> historyStore[("Messages and citation snapshots")]
```

### Document ingestion

The authenticated upload endpoint validates the file, extracts its text, and creates overlapping chunks. Gemini embeds the chunks using `RETRIEVAL_DOCUMENT`. The service validates vector dimensions and values, stores the original as an authenticated Cloudinary raw asset, and persists document metadata and chunks. Upload requests wait for indexing to complete.

### Question answering

The service selects only the user's ready documents, embeds the question using `RETRIEVAL_QUERY`, and ranks chunks by cosine distance. Recent user questions help retrieve context for follow-ups. Up to six passages above the configured similarity threshold are supplied to Gemini, together with recent conversation history and instructions for document-grounded answers and numbered citations.

When retrieval finds insufficient context, the service returns:

> I couldn't find the answer in the provided document.

This fallback skips chat generation, although query embedding is still required. Grounding instructions reduce unsupported answers; answer quality should be evaluated with real documents.

### Streaming and persistence

`POST /api/v1/rag/chat/stream` forwards SDK text increments as SSE. Retrieval and ownership validation finish before the stream begins. A `meta` event carries citations and a provisional conversation ID, `delta` events carry text, and `done` confirms that the completed answer and citations have been committed. Provider/storage failures emit `error`; unfinished messages are rolled back. A disconnect after commit cannot undo an already saved answer.

## Technology stack

| Component | Technology |
|---|---|
| API and validation | FastAPI, Pydantic |
| Database access | SQLAlchemy async, asyncpg |
| Vector retrieval | PostgreSQL, pgvector `Vector(3072)` |
| AI provider | Gemini via `google-genai` async SDK |
| Document parsing | pypdf and UTF-8 text parsing |
| Original file storage | Cloudinary authenticated raw assets |
| Rate limits and OTP state | Redis |
| Authentication | PyJWT, email OTP, Google/GitHub OAuth |
| Email delivery | Brevo |
| Dependency management | uv and committed `uv.lock` |
| Tests | Python unittest, mocked providers, PostgreSQL integration tests |

## Run locally

### 1. Install dependencies

Requirements: **Python 3.14+**, [uv](https://docs.astral.sh/uv/), a **TLS-enabled PostgreSQL database with pgvector installed**, and Redis.

```bash
git clone https://github.com/Utpalsonowal7/Study-Buddy-Full-Stack-RAG-Chatbot-with-Production-Grade-Authentication.git
cd Study-Buddy-Full-Stack-RAG-Chatbot-with-Production-Grade-Authentication
uv sync --frozen
```

`--frozen` installs the versions in the committed lockfile without updating it.

### 2. Configure the environment

Create a `.env` in the repository root, or inject these variables through deployment settings. Preserve an existing `.env`. The repository currently has no `.env.example`; use the template below. Placeholder credentials must be replaced locally and must never be committed.

```dotenv
DATABASE_URL=postgresql+asyncpg://USER:PASSWORD@HOST:5432/DATABASE
REDIS_URL=redis://localhost:6379/0

JWT_ACCESS_TOKEN_SECRECT=replace_with_a_random_signing_secret
JWT_REFRESH_TOKEN_SECRET=replace_with_a_different_random_signing_secret
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=30
JWT_ALGORITHM=HS256
COOKIE_SECURE=false

RAG_API_KEY=replace_with_your_gemini_key
RAG_EMBEDDING_MODEL=gemini-embedding-001
RAG_CHAT_MODEL=gemini-3.1-flash-lite

CLOUDNARY_CLOUD_NAME=replace_with_your_cloud_name
CLOUDNARY_API_KEY=replace_with_your_cloudinary_key
CLOUDNARY_API_SECRET=replace_with_your_cloudinary_secret

BRAVO_API_KEY=replace_with_your_brevo_key
FRONTEND_URL=http://localhost:5173/

# Set empty strings when GitHub OAuth is not configured.
GITHUB_CLIENT_ID=
GITHUB_CLIENT_SECRET=
GITHUB_REDIRECT_URI=
```

**Use the exact spellings shown:** the current code reads `CLOUDNARY_*`, `BRAVO_API_KEY`, and `JWT_ACCESS_TOKEN_SECRECT`. `GEMINI_API_KEY` is an alternative to `RAG_API_KEY`; `RAG_API_KEY` takes precedence. No Gemini base URL variable is used. Generate two separate random JWT secrets of at least 32 characters, for example by running this twice:

```bash
uv run python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Create a Gemini key through [Google AI Studio](https://aistudio.google.com/app/apikey). The configured models must be available to your key. Changing embedding models requires re-indexing existing documents; matching dimensions alone does not make different embedding spaces compatible. The current embedding integration uses the `gemini-embedding-001` retrieval-task contract.

PostgreSQL connections require TLS in `app/db/database.py`. Install pgvector on the database server first. Startup enables it with `CREATE EXTENSION IF NOT EXISTS vector`; the database role needs permission, or an administrator must enable it beforehand. Redis requires a Redis protocol URL (`redis://` or `rediss://`), not an Upstash REST endpoint.

### 3. Optional OAuth configuration

Register callback URLs with the providers and configure these variables:

| Provider | Variables |
|---|---|
| Google | `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI`, `GOOGLE_AUTH_URI`, `GOOGLE_TOKEN_URI`, `GOOGLE_PROVIDER_URI` |
| GitHub | `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`, `GITHUB_REDIRECT_URI`, `GITHUB_AUTH_URI`, `GITHUB_TOKEN_URI`, `GITHUB_USER_URI`, `GITHUB_USER_EMAILS_URI` |

Local callback paths are `/api/v1/auth/google/callback` and `/api/v1/auth/github/callback`. Google endpoints are `https://accounts.google.com/o/oauth2/v2/auth`, `https://oauth2.googleapis.com/token`, and `https://openidconnect.googleapis.com/v1/userinfo`. GitHub endpoints are `https://github.com/login/oauth/authorize`, `https://github.com/login/oauth/access_token`, `https://api.github.com/user`, and `https://api.github.com/user/emails`.

GitHub's OAuth state cookie is currently always secure; test that flow over HTTPS. Email login uses Brevo and requires working email delivery configuration.

### 4. Start the server

```bash
uv run fastapi dev app/main.py
```

For a deployment process:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Interactive API documentation: `http://localhost:8000/docs`. OpenAPI schema: `/openapi.json`.

Startup creates missing tables: `users`, `sessions`, `docs`, `docs_chunks`, `conversations`, `messages`, and `message_sources`. It does **not** alter existing tables. Fresh databases using the latest models do not need the compatibility migration. Older RAG schemas can be upgraded using:

```bash
uv run --frozen python -m app.db.migrate_rag
```

That script adds metadata/citation fields and repairs the chunk foreign key without dropping records. It refuses unreconciled source references. Alembic migrations are a planned improvement, not currently implemented.

### 5. Check readiness

```bash
curl http://localhost:8000/api/v1/health
```

Healthy response:

```json
{"status":"ok","checks":{"database":"ok","redis":"ok"}}
```

HTTP 503 means PostgreSQL or Redis failed its check or exceeded its three-second timeout. Check connectivity, TLS, credentials, and database availability. The endpoint hides internal errors and does not validate Gemini, Cloudinary, email, or OAuth credentials.

## API guide

Use `http://localhost:8000/api/v1` as the API base. Authenticated browser requests must include `credentials: "include"`. Use consistent hostnames locally and configure allowed frontend origins in `app/main.py`.

### Authentication

| Method | Path | Input / purpose |
|---|---|---|
| POST | `/auth/register` | `{ "email": "…", "full_name": "…" }` |
| POST | `/auth/send-otp` | `{ "email": "…" }` |
| POST | `/auth/verify-otp` | `{ "email": "…", "otp": "…" }` |
| POST | `/auth/login` | Email; initiates OTP login |
| POST | `/auth/login/verify-otp` | Email and OTP; completes login |
| GET | `/auth/me` | Authenticated user |
| POST | `/auth/refresh` | Rotates refresh session and issues cookies |
| POST | `/auth/logout` | Invalidates session and clears cookies |
| GET | `/auth/google` | Browser redirect to Google |
| GET | `/auth/google/callback` | Provider callback |
| GET | `/auth/github` | Browser redirect to GitHub |
| GET | `/auth/github/callback` | Provider callback |

Registration creates an unverified account and session. Successful email OTP login marks the email verified. JWT cookies are HttpOnly and SameSite=Lax; `COOKIE_SECURE=false` is for local HTTP only. Refresh-token hashes are stored in the database, and refresh operations rotate session tokens. Navigate the browser to OAuth login routes; providers invoke the callbacks.

### Documents and conversations

| Method | Path | Purpose |
|---|---|---|
| POST | `/rag/documents` | Multipart `file`; returns indexed document (201) |
| GET | `/rag/documents` | Paginated document list |
| GET | `/rag/documents/{id}` | Document metadata |
| GET | `/rag/documents/{id}/download` | Private download URL valid for 60 seconds |
| DELETE | `/rag/documents/{id}` | Delete file and indexed chunks (204) |
| POST | `/rag/chat` | Complete answer as JSON |
| POST | `/rag/chat/stream` | Incremental SSE answer |
| GET | `/rag/conversations` | Paginated conversation list |
| GET | `/rag/conversations/{id}/messages` | Message history |
| DELETE | `/rag/conversations/{id}` | Delete conversation and messages (204) |

Chat request:

```json
{
  "question": "Explain photosynthesis in simple terms.",
  "document_ids": [1],
  "conversation_id": null
}
```

IDs are numbers. Omit `document_ids` to search all your ready documents. Include the selected IDs on every follow-up request to keep the same retrieval scope; a conversation ID alone does not remember document selection. JSON answers contain `conversation_id`, `answer`, and `sources`.

SSE events are `meta`, `delta`, `done`, and `error`. Append each `delta.text` immediately; treat `done` as confirmation of persistence. Use streamed `fetch` for this JSON POST, since browser `EventSource` cannot send it. Heartbeats arrive every 15 seconds while waiting. Configure reverse proxies to avoid SSE buffering and allow long requests.

**[RAG_API.md](RAG_API.md) provides complete request/response types, upload code, streaming parsing, cancellation, pagination, and frontend examples.**

## Limits and engineering tradeoffs

| Setting | Current value |
|---|---|
| Supported input | Text PDF, UTF-8 TXT, Markdown |
| File size / PDF pages | 10 MiB / 200 pages |
| Chunking | Approximately 1800 characters, 200-character overlap |
| Chunk quota | 200 per document, 2000 per user |
| Retrieved passages | Up to 6, minimum cosine score 0.25 |
| Question length / document selection | 2000 characters / up to 20 IDs |
| Upload rate limit | 10 requests/hour per client IP |
| Chat rate limit | 30 requests/minute per endpoint per client IP |
| List page size | Maximum 100 records |

- **Uploads are synchronous.** The code does not implement a durable worker queue or upload-progress SSE. FastAPI background tasks are used for OTP emails.
- **Retrieval uses exact cosine ranking.** Larger libraries need workload measurements and an appropriate vector-index strategy; no throughput or latency benchmark is claimed.
- **Citation snapshots survive document deletion.** Deleting a file removes its asset and retrieval chunks, but saved conversations still contain source excerpts. Delete conversations separately to remove that history.
- **Generation is grounded by prompts and retrieval.** No automated citation entailment verifier or measured hallucination rate is included.
- **Storage and database writes are separate systems.** Failed indexing attempts compensating Cloudinary deletion; cleanup failures require reconciliation.

## Testing

```bash
uv run --frozen python -m unittest discover -s tests -v
```

Three integration tests are skipped unless `TEST_DATABASE_URL` identifies a dedicated database whose name starts with `study_buddy_test`. Never use a production database. Provision that database on your TLS-enabled pgvector server, then run:

```bash
# Bash
TEST_DATABASE_URL=postgresql+asyncpg://USER:PASSWORD@HOST:5432/study_buddy_test_rag \
  uv run --frozen python -m unittest discover -s tests -v
```

```powershell
# PowerShell
$env:TEST_DATABASE_URL = "postgresql+asyncpg://USER:PASSWORD@HOST:5432/study_buddy_test_rag"
uv run --frozen python -m unittest discover -s tests -v
Remove-Item Env:TEST_DATABASE_URL
```

The latest verified full run passed **19 tests**, including the three database integration tests. Coverage includes extraction/rejection, SDK request formats, embedding task types and batching, invalid provider responses, private file options, ownership isolation, document/chat/history lifecycle, refresh/logout, migration repeatability, and failure rollback. A real local HTTP streaming test verifies that the first text arrives while generation is still paused and checks provider/database failure and disconnect behavior. External AI/storage services are mocked in the suite. Health was additionally checked against real PostgreSQL/Redis and simulated dependency failures.

Live Gemini output, Cloudinary uploads, email, and OAuth must also be tested with real credentials. Passing mocked tests does not establish production provider access or answer quality.

## Project structure

```text
app/
├── main.py                  # App, CORS, router registration, lifespan
├── config.py                # Environment settings
├── core/                    # Redis and rate limiter
├── db/                      # Async database engine and compatibility migration
├── dependencies/            # Auth, sessions, rate limiting, providers
├── models/
│   ├── auth/                # Users and sessions
│   └── rag/                 # Documents, chunks, conversations, messages, sources
├── routes/                  # Auth, RAG, health endpoints
├── schemas/                 # Request validation and response contracts
├── services/                # Authentication, RAG, AI, Cloudinary
└── utils/                   # JWT, cookies, parsing, SSE, email helpers
 tests/test_rag.py            # Unit and database/HTTP integration tests
 RAG_API.md                  # Frontend integration contract
 pyproject.toml              # Dependencies and Python requirement
 uv.lock                     # Reproducible dependency versions
```

## Deployment checklist and next steps

See [DEPLOYMENT.md](DEPLOYMENT.md) for the Docker workflow and Render Blueprint instructions. `render.yaml` configures a web service using existing PostgreSQL and Redis services.

Use TLS for browser traffic, stable signing secrets, secure cookies, authenticated persistent PostgreSQL/Redis, and a private secret store. Configure trusted proxy handling, CORS, request-size limits, and SSE timeouts for your deployment. Review cross-site cookie requirements if frontend and backend are on unrelated sites. Validate provider access and backups before release.

Planned improvements include Alembic schema migrations, durable background ingestion, OCR, retrieval-quality evaluation, vector-index benchmarking, and structured operational telemetry. This repository implements the core backend; a frontend, live deployment evidence, and performance benchmarks are not included.

For technical discussion, useful design topics are retrieval task separation, overlapping chunk tradeoffs, ownership checks before vector search, streaming transaction boundaries, private asset delivery, and citation retention after deletion. The implementation and tests provide concrete examples of those decisions.
