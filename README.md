# Study Buddy backend

FastAPI authentication and a document RAG API. Signed-in users can upload study material, ask questions with source citations, keep conversations, and delete their documents. Original files are stored as **authenticated raw assets in Cloudinary**; extracted text, document metadata, conversations, and message sources use your existing PostgreSQL models. Embeddings use the existing pgvector `Vector(3072)` column. Redis supplies rate limiting and OTP storage.

The project follows its original folders:

```text
app/routes/rag.py            # HTTP endpoints
app/services/rag.py          # Indexing, retrieval, and conversations
app/services/ai.py           # Embeddings and answers
app/services/cloudinary.py   # Private file storage
app/models/rag/              # Existing Document, DocumentChunk, Conversation,
                             # Message, and MessageSource models
app/schemas/rag.py           # Request and response validation
app/dependencies/rag.py      # Provider dependency injection
app/utils/documents.py       # File parsing and chunking
app/config.py                # Auth and RAG configuration
```

There is no separate `app/rag` package or duplicate RAG model set. The existing model fields and integer document/conversation IDs are used throughout. The incorrect message-source foreign key is corrected to `docs_chunks.id`. Message sources retain citation snapshots after their source document is deleted.

This checkout is the backend. It does not include a frontend or OCR. Provider credentials are intentionally placeholders, not working keys or simulated answers.

## Setup

Requires Python 3.14+, `uv`, PostgreSQL with TLS and the pgvector extension, and Redis.

```bash
uv sync --frozen --python 3.14
cp .env.example .env  # only if you do not already have a .env
```

Replace `GEMINI_API_KEY`, `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, and `CLOUDINARY_API_SECRET` in `.env` or secure deployment environment variables. Set separate random JWT signing keys, retaining the existing spelling `JWT_ACCESS_TOKEN_SECRECT`. Do not commit credentials. Missing or placeholder RAG/Cloudinary credentials return HTTP 503; no upload or model call is attempted with them.

The AI service uses the **native Gemini Developer API**. Create your key in [Google AI Studio](https://aistudio.google.com/app/apikey) and configure:

```dotenv
GEMINI_API_KEY=replace_me_with_your_gemini_key
GEMINI_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta
RAG_EMBEDDING_MODEL=gemini-embedding-001
RAG_CHAT_MODEL=gemini-3.1-flash-lite
```

`gemini-embedding-001` supports the existing 3072-dimensional pgvector column. Uploads use `RETRIEVAL_DOCUMENT`; questions use `RETRIEVAL_QUERY`. Embedding requests are batched, and all returned vectors are checked for the correct dimensions and finite, nonzero values. The JSON chat service calls `generateContent`, while SSE chat calls `streamGenerateContent?alt=sse`, with source excerpts, system instructions, and conversation history mapped to Gemini's `user`/`model` roles. Blocked requests return 422; incomplete or invalid model responses return 502. Thought parts are excluded from saved answers.

The default chat model is `gemini-3.1-flash-lite`, as requested. Set `RAG_CHAT_MODEL` to another Gemini text model available to your key if needed; live model availability has not been verified without credentials. The embedding client supports models with the `gemini-embedding-001` retrieval task contract; switching to `gemini-embedding-2` requires adapting its task instructions, not just changing the model name.

If you already copied the older `.env.example`, update your existing `.env` entries manually. `RAG_API_KEY` and `RAG_API_BASE_URL` from the OpenAI version are no longer used. Old `text-embedding-3-large`/`gpt-4o-mini` model values must also be replaced. Documents indexed with OpenAI embeddings must be deleted and re-uploaded to generate Gemini embeddings; the API rejects mismatched embedding models rather than mixing vector spaces. Existing tables, Cloudinary settings, and folder structure are unchanged by this provider switch.

Request formats were checked against Google's [official API definitions](https://github.com/googleapis/googleapis/blob/master/google/ai/generativelanguage/v1beta/generative_service.proto) and [embedding cookbook](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Embeddings.ipynb). Real calls still require a valid key with access and quota for the configured models.

If your database already has the original `docs` or `message_sources` tables, run the additive schema migration before starting the updated API:

```bash
uv run --frozen python -m app.db.migrate_rag
```

It adds document embedding-model/chunk-count metadata and citation snapshots, and fixes the message-source foreign key without dropping records. Existing untagged documents receive `legacy_unknown` and must be re-indexed before chat. It stops and rolls back if existing message sources reference missing chunks. PostgreSQL must have pgvector installed and the database user must be able to enable it (or an administrator must enable it first).

```bash
uv run --frozen uvicorn app.main:app --host 127.0.0.1 --port 8000
```

On this Codex cloud environment, prepared helpers live outside the checkout:

```bash
bash /workspace/study-buddy-setup/start.sh
# In another terminal, from the repository:
.venv/bin/python /workspace/study-buddy-setup/check.py
```

The helper starts disposable loopback pgvector/PostgreSQL (port 5433) and Redis Docker containers, enables PostgreSQL TLS, and starts the backend with temporary development signing keys. Production needs persistent storage, secure database authentication, stable signing keys, HTTPS, and `COOKIE_SECURE=true`. Cookies default to secure and SameSite=Lax; the local helper opts into HTTP cookies. Placeholder JWT keys are rejected before registration writes to the database. Local helper keys change on restart; old tokens will then be invalid. The application registers `app/models/rag` and creates the original `docs`, `docs_chunks`, `conversations`, `messages`, and `message_sources` tables using the existing `Base.metadata.create_all` workflow. The vector extension is enabled before table creation. Existing table alterations require the migration command above. Any `rag_*` tables created by the previous implementation are left untouched; their UUID-based records are not read by this API. Export those records before migrating any data and re-upload documents to the original model schema; no old tables or assets are automatically deleted.

## Document and chat API

All routes below start with `/api/v1/rag` and require the existing `access_token` cookie. Files and conversations are always scoped to the authenticated user; foreign IDs return 404. The upload endpoint takes multipart form field `file`.

| Method | Route | Behavior |
| --- | --- | --- |
| POST | `/documents` | Extract, embed, upload to Cloudinary, and index a PDF/TXT/MD file |
| GET | `/documents` | List documents (`offset`, `limit`) |
| GET | `/documents/{id}` | Read document metadata |
| GET | `/documents/{id}/download` | Get an authenticated Cloudinary download URL valid for 60 seconds |
| DELETE | `/documents/{id}` | Delete the Cloudinary asset and indexed chunks |
| POST | `/chat` | Return a complete answer as JSON (existing endpoint) |
| POST | `/chat/stream` | Stream Gemini text as SSE with the same question/document/conversation inputs |
| GET | `/conversations` | List conversations (`offset`, `limit`) |
| GET | `/conversations/{id}/messages` | Read messages (`after_id`, `limit`) |
| DELETE | `/conversations/{id}` | Delete a conversation and its messages |

Example chat request:

```json
{
  "question": "Explain photosynthesis in simple terms",
  "document_ids": [1],
  "conversation_id": null
}
```

Omit `document_ids` to search the current user's whole library. Pass the returned `conversation_id` for follow-up questions. Each response contains `answer`, `conversation_id`, and `sources`, including citation number, document ID, filename, page (for PDFs), chunk position, text excerpt, and similarity score. Recent user questions help retrieve context for follow-ups. The chat model receives retrieved passages and recent conversation history with instructions to treat their contents as untrusted data. If nothing meets the retrieval threshold, the API returns an insufficient-information answer without invoking the chat model.

### Streaming chat

Use **POST `/api/v1/rag/chat/stream`** for incremental output. This forwards Gemini's actual text deltas as they arrive; it does not generate a complete answer first and split it into fake chunks. Retrieval finishes before streaming begins. The endpoint sends `Content-Type: text/event-stream`, cache/proxy buffering headers, and heartbeat comments every 15 seconds while waiting for generation.

Events:

```text
event: meta
data: {"conversation_id": 7, "sources": [...]}

event: delta
data: {"text": "Photosynthesis "}

event: delta
data: {"text": "converts light into chemical energy [1]."}

event: done
data: {"conversation_id": 7, "message_id": 24}
```

`meta` includes citations and a provisional conversation ID. Append each `delta.text` immediately to the displayed assistant message. `done` is sent only after the full answer and its citations have been committed to the database. If Gemini fails or saving fails after streaming starts, an `error` event contains `{ "status": 502, "message": "..." }` (500 for storage failures) and no `done` is sent. Ownership, request validation, and retrieval failures before the stream starts use ordinary HTTP error responses. Failed or disconnected generation rolls back the unfinished messages; a new provisional conversation will not exist unless completed. Disconnecting after commit cannot undo an already saved answer.

The stream checks for disconnects while waiting for the next token and closes the upstream connection. Explanations follow the user's requested length, retain document citations, avoid the phrase "based on the provided context", and use the exact insufficient-information message: "I couldn't find the answer in the provided document."

Since this endpoint is a POST with JSON and authentication cookies, use streamed `fetch`, not plain browser `EventSource` or `response.json()`:

```javascript
async function streamAnswer(payload, { onText, onMeta, onDone }, signal) {
  const response = await fetch('/api/v1/rag/chat/stream', {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok) throw new Error(await response.text());
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) throw new Error('Stream ended without a completion event');
      buffer += value;
      let boundary;
      while ((boundary = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const lines = frame.split('\n');
        const event = lines.find(line => line.startsWith('event:'))?.slice(6).trim();
        const data = lines.filter(line => line.startsWith('data:')).map(line => line.slice(5).trim()).join('\n');
        if (!data) continue; // heartbeat comment
        const message = JSON.parse(data);
        if (event === 'meta') onMeta?.(message);
        else if (event === 'delta') onText(message.text);
        else if (event === 'error') throw new Error(message.message);
        else if (event === 'done') {
          onDone?.(message);
          return;
        }
      }
    }
  } finally {
    await reader.cancel();
    reader.releaseLock();
  }
}

// onText should append text to the current assistant bubble as each delta arrives.
// Pass an AbortController signal to cancel generation when the user presses Stop.
```

Configure your deployment proxy to avoid response buffering/compression for this route and allow sufficiently long requests. The existing JSON endpoint remains available. File uploads still use the synchronous document endpoint; this change adds chat SSE, not an upload-processing queue.

For a local development session, the existing registration endpoint creates a session cookie:

```bash
curl -c cookies.txt -H 'Content-Type: application/json' \
  -d '{"email":"student@example.com","full_name":"Student"}' \
  http://127.0.0.1:8000/api/v1/auth/register
curl -b cookies.txt -F 'file=@notes.pdf' \
  http://127.0.0.1:8000/api/v1/rag/documents
curl -b cookies.txt -H 'Content-Type: application/json' \
  -d '{"question":"Summarize my notes"}' \
  http://127.0.0.1:8000/api/v1/rag/chat
```

Registration creates an unverified account; it no longer incorrectly marks an unverified email as verified. Email OTP login marks the email verified after successful OTP validation. Brevo and OAuth configuration are separate from RAG credentials.

## Limits and failure handling

- UTF-8 TXT/Markdown and text-based PDF files, up to 10 MiB and 200 PDF pages. Encrypted PDFs and scanned PDFs without text are rejected.
- Overlapping chunks of about 1,800 characters; maximum 200 chunks per document and 2,000 per user. Delete documents to reclaim capacity.
- Embeddings are persisted in the existing pgvector column and ranked in PostgreSQL with cosine distance. Only the selected user's ready documents are searched. The per-user indexing quota remains 2,000 chunks; tune quotas and add vector indexes after evaluating larger workloads.
- At most six passages are sent to the model. The minimum cosine score is 0.25. Relevance and generated answers still need evaluation with your actual study material and chosen models.
- Uploads are synchronous and rate limited to 10 per hour per client IP; chat is limited to 30 per minute. Configure trusted proxy headers at deployment and protect request body sizes at the reverse proxy.
- Provider failures return 502. Unsupported files return 415, size/quota errors 413, empty/scanned documents 422, and incompatible embedding models 409.
- A failed database write after upload attempts Cloudinary cleanup. Cleanup failures are logged with the document ID for reconciliation. A storage deletion failure leaves metadata available for retry.
- Chat history contains excerpts and prior answers. Deleting a document removes its original asset and search index, but does not erase excerpts already present in a saved conversation. Delete the conversation separately.
- Refresh tokens are now stored as hashes and rotated with unique token IDs. Existing sessions storing raw refresh tokens must log in again.

## Tests

The standard-library test suite needs no additional test dependencies:

```bash
uv run --frozen python -m unittest discover -s tests -v
```

The three database integration tests are skipped unless `TEST_DATABASE_URL` names a dedicated PostgreSQL database starting with `study_buddy_test`. They exercise the original models, real pgvector queries, migration repeatability, and JWT authentication while mocking AI, Cloudinary, and the rate-limit call. Streaming delivery is also tested over a real local HTTP connection, with the first chunk arriving before generation is released to finish. They create and clean up their own users, records, and temporary migration schema; never point it at production.

```bash
# With the prepared local PostgreSQL container:
docker exec study-buddy-vector-postgres createdb -U postgres study_buddy_test_rag
TEST_DATABASE_URL=postgresql+asyncpg://postgres@127.0.0.1:5433/study_buddy_test_rag \
  uv run --frozen python -m unittest discover -s tests -v
```

Coverage includes extraction and file rejection, native Gemini request/response formats, retrieval task types, batching, and blocked/incomplete replies, private Cloudinary options, expiring downloads, authenticated upload/chat/history/deletion, source citations, cross-user isolation, insufficient retrieval, provider failures, registration/refresh/logout, repeatable schema migration without losing existing records, actual incremental SSE delivery, disconnect rollback, and truncated streams. Real Cloudinary uploads and real model answers require credentials and are not validated by mocked tests. Optional network destinations include `generativelanguage.googleapis.com` and `api.cloudinary.com`.
