# Study Buddy backend

FastAPI authentication and a document RAG API. Signed-in users can upload study material, ask questions with source citations, keep conversations, and delete their documents. Original files are stored as **authenticated raw assets in Cloudinary**; extracted text, embeddings, document metadata, and conversations are stored in PostgreSQL. Redis supplies rate limiting and OTP storage.

The API registers the models in `app/rag/models.py` (`rag_*` tables). Earlier model definitions in `app/models/rag/` are retained for reference but are not registered at startup; they require separate schema reconciliation before use.

This checkout is the backend. It does not include a frontend or OCR. Provider credentials are intentionally placeholders, not working keys or simulated answers.

## Setup

Requires Python 3.14+, `uv`, PostgreSQL with TLS, and Redis.

```bash
uv sync --frozen --python 3.14
cp .env.example .env  # only if you do not already have a .env
```

Replace `RAG_API_KEY`, `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, and `CLOUDINARY_API_SECRET` in `.env` or secure deployment environment variables. Set separate random JWT signing keys, retaining the existing spelling `JWT_ACCESS_TOKEN_SECRECT`. Do not commit credentials. Missing or placeholder RAG/Cloudinary credentials return HTTP 503; no upload or model call is attempted with them.

The default AI provider is OpenAI. Other OpenAI-compatible providers must support both `/embeddings` and `/chat/completions`; configure `RAG_API_BASE_URL`, `RAG_EMBEDDING_MODEL`, and `RAG_CHAT_MODEL` together. Do not change the embedding provider/model for an existing library without deleting and re-uploading its documents.

```bash
uv run --frozen uvicorn app.main:app --host 127.0.0.1 --port 8000
```

On this Codex cloud environment, prepared helpers live outside the checkout:

```bash
bash /workspace/study-buddy-setup/start.sh
# In another terminal, from the repository:
.venv/bin/python /workspace/study-buddy-setup/check.py
```

The helper starts disposable loopback PostgreSQL and Redis Docker containers, enables PostgreSQL TLS, and starts the backend with temporary development signing keys. Production needs persistent storage, secure database authentication, stable signing keys, HTTPS, and `COOKIE_SECURE=true`. Cookies default to secure and SameSite=Lax; the local helper opts into HTTP cookies. Placeholder JWT keys are rejected before registration writes to the database. Local helper keys change on restart; old tokens will then be invalid. The application creates the new RAG tables during startup using the existing `Base.metadata.create_all` workflow. Existing tables are not migrated by that command.

## Document and chat API

All routes below start with `/api/v1/rag` and require the existing `access_token` cookie. Files and conversations are always scoped to the authenticated user; foreign IDs return 404. The upload endpoint takes multipart form field `file`.

| Method | Route | Behavior |
| --- | --- | --- |
| POST | `/documents` | Extract, embed, upload to Cloudinary, and index a PDF/TXT/MD file |
| GET | `/documents` | List documents (`offset`, `limit`) |
| GET | `/documents/{id}` | Read document metadata |
| GET | `/documents/{id}/download` | Get an authenticated Cloudinary download URL valid for 60 seconds |
| DELETE | `/documents/{id}` | Delete the Cloudinary asset and indexed chunks |
| POST | `/chat` | Ask a question; optionally supply document IDs or a conversation ID |
| GET | `/conversations` | List conversations (`offset`, `limit`) |
| GET | `/conversations/{id}/messages` | Read messages (`after_id`, `limit`) |
| DELETE | `/conversations/{id}` | Delete a conversation and its messages |

Example chat request:

```json
{
  "question": "Explain photosynthesis in simple terms",
  "document_ids": ["a-document-uuid"],
  "conversation_id": null
}
```

Omit `document_ids` to search the current user's whole library. Pass the returned `conversation_id` for follow-up questions. Each response contains `answer`, `conversation_id`, and `sources`, including citation number, document ID, filename, page (for PDFs), chunk position, text excerpt, and similarity score. Recent user questions help retrieve context for follow-ups. The chat model receives retrieved passages and recent conversation history with instructions to treat their contents as untrusted data. If nothing meets the retrieval threshold, the API returns an insufficient-information answer without invoking the chat model.

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
- Embeddings are persisted as JSON and ranked with cosine similarity for this bounded library. Move retrieval to an indexed vector database for larger workloads; this implementation does not claim unlimited-scale search.
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

The database integration test is skipped unless `TEST_DATABASE_URL` names a dedicated PostgreSQL database starting with `study_buddy_test`. It exercises real database tables and JWT authentication while mocking AI, Cloudinary, and the rate-limit call. It creates and cleans up its own users and records; never point it at production.

```bash
# With the prepared local PostgreSQL container:
docker exec study-buddy-postgres createdb -U postgres study_buddy_test_rag
TEST_DATABASE_URL=postgresql+asyncpg://postgres@127.0.0.1:5432/study_buddy_test_rag \
  uv run --frozen python -m unittest discover -s tests -v
```

Coverage includes extraction and file rejection, provider request/response formats, private Cloudinary options, expiring downloads, authenticated upload/chat/history/deletion, source citations, cross-user isolation, insufficient retrieval, provider failures, and registration/refresh/logout. Real Cloudinary uploads and real model answers require credentials and are not validated by mocked tests. Optional network destinations include `api.openai.com` (or your configured AI hostname) and `api.cloudinary.com`.
