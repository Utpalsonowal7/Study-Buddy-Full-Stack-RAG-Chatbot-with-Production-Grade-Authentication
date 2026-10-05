# Study Buddy RAG API: frontend integration reference

This document describes the implemented backend contract for a frontend developer or another project chat. Authentication endpoints are intentionally omitted. Use the existing login flow first; all endpoints below require the browser's `access_token` HttpOnly cookie. Send `credentials: "include"`; do not read cookies in JavaScript or invent a bearer-token flow.

## Connection and local setup

```js
const API = "http://localhost:8000/api/v1";
```

Use `localhost` consistently for frontend and backend during local testing. Backend `.env` needs `COOKIE_SECURE=false` for local HTTP; restart after changing it. Production uses HTTPS and secure cookies. Allowed local frontend origins include `http://localhost:3000`, `http://localhost:5173`, and `http://127.0.0.1:5173`. Add other origins to CORS in `app/main.py`. A frontend on an unrelated production site needs cookie/CORS configuration appropriate to that deployment.

The backend requires PostgreSQL with pgvector, Redis, a real Gemini API key (`RAG_API_KEY` or `GEMINI_API_KEY`), and Cloudinary credentials (`CLOUDNARY_CLOUD_NAME`, `CLOUDNARY_API_KEY`, `CLOUDNARY_API_SECRET`, matching the current configuration spelling). Gemini uses the official `google-genai` SDK; no API base URL setting is needed. Default models are `gemini-embedding-001` and `gemini-3.1-flash-lite`. Never include provider credentials in frontend code.

## Endpoint list

Public health endpoint: **GET `/api/v1/health`**, without authentication. It checks PostgreSQL (`SELECT 1`) and Redis (`PING`) concurrently, with a three-second timeout per service. Healthy response: HTTP 200, `{"status":"ok","checks":{"database":"ok","redis":"ok"}}`. An unavailable dependency returns HTTP 503 with `status: "degraded"` and the affected check set to `"unavailable"`. Responses are not cached. This checks backend readiness; it does not call Gemini or Cloudinary or validate their credentials.

Paths are relative to `API`. All document and conversation IDs are positive integers. Responses are direct objects/arrays, without a `data` wrapper.

| Method | Path | Input | Success response |
|---|---|---|---|
| POST | `/rag/documents` | Multipart field `file` | 201, Document |
| GET | `/rag/documents?offset=0&limit=50` | Pagination | 200, Document[] |
| GET | `/rag/documents/{document_id}` | Document ID | 200, Document |
| GET | `/rag/documents/{document_id}/download` | Document ID | 200, `{ "url": "...", "expires_in": 60 }` |
| DELETE | `/rag/documents/{document_id}` | Document ID | 204, empty body |
| POST | `/rag/chat` | ChatRequest JSON | 200, ChatResponse |
| POST | `/rag/chat/stream` | ChatRequest JSON | 200, `text/event-stream` |
| GET | `/rag/conversations?offset=0&limit=50` | Pagination | 200, Conversation[] |
| GET | `/rag/conversations/{conversation_id}/messages?after_id=0&limit=100` | Conversation ID, message pagination | 200, Message[] |
| DELETE | `/rag/conversations/{conversation_id}` | Conversation ID | 204, empty body |

Document/conversation list pagination: `offset >= 0`, `limit` 1–100. Messages: `after_id >= 0`, `limit` 1–100; results arrive in ascending message ID order. Use the last returned message ID as the next `after_id`. All lists return arrays, without a total count.

## Request and response types

```ts
type Document = {
  id: number;
  filename: string;
  content_type: string;
  size_bytes: number;
  chunk_count: number;
  created_at: string; // ISO timestamp
};

type ChatRequest = {
  question: string; // nonblank, up to 2000 characters
  document_ids?: number[] | null; // 1–20 IDs if supplied; omit to search all ready documents
  conversation_id?: number | null; // omit for a new conversation
};

type Source = {
  citation: number; // number used in answer citations such as [1]
  document_id: number;
  filename: string;
  chunk: number;
  page: number | null;
  text: string;
  score: number;
};

type ChatResponse = {
  conversation_id: number;
  answer: string;
  sources: Source[];
};

type Conversation = {
  id: number;
  title: string | null;
  created_at: string;
};

type Message = {
  id: number;
  role: "user" | "assistant";
  content: string;
  sources: Source[];
  created_at: string;
};
```

Conversation history does not automatically restrict retrieval to previously selected documents. Include `document_ids` on each chat request when the frontend should retain that selection.

## JSON requests

```js
async function api(path, method = "GET", body) {
  const response = await fetch(`${API}${path}`, {
    method,
    credentials: "include",
    ...(body !== undefined && {
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  });
  if (response.status === 204) return null;
  const data = await response.json();
  if (!response.ok) throw new Error(JSON.stringify(data.detail ?? data));
  return data;
}

const documents = await api("/rag/documents");
const selectedDocumentId = documents[0].id; // handle an empty list in the UI
const result = await api("/rag/chat", "POST", {
  question: "Explain the main ideas.",
  document_ids: [selectedDocumentId],
});

const followUp = await api("/rag/chat", "POST", {
  question: "Give me a shorter explanation.",
  document_ids: [selectedDocumentId],
  conversation_id: result.conversation_id,
});

const conversations = await api("/rag/conversations");
const messages = await api(`/rag/conversations/${result.conversation_id}/messages`);
```

## Upload and download

Uploads accept UTF-8 `.txt`, `.md`, and text-based `.pdf` files, up to 10 MiB. Scanned PDFs need OCR, which is not implemented. Parsing limits include 200 pages and 200 chunks per document; the user library limit is 2000 chunks.

Uploads run synchronously: the request waits for extraction, Gemini embeddings, Cloudinary storage, and database indexing. Show a loading state until it completes. No background upload queue or upload-progress SSE endpoint exists. The returned Document does not expose a status field or a public Cloudinary URL.

```js
async function uploadDocument(file) {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API}/rag/documents`, {
    method: "POST",
    credentials: "include",
    body: form,
  });
  // Do not manually set Content-Type: the browser supplies the multipart boundary.
  const data = await response.json();
  if (!response.ok) throw new Error(JSON.stringify(data.detail ?? data));
  return data;
}

const uploaded = await uploadDocument(fileInput.files[0]);
const download = await api(`/rag/documents/${uploaded.id}/download`);
window.open(download.url, "_blank", "noopener,noreferrer");

// DELETE returns no JSON. Obtain confirmation in your UI before deletion.
await api(`/rag/documents/${uploaded.id}`, "DELETE");
```

Download URLs expire after 60 seconds; request a fresh URL when the user clicks Download. Deleting a document preserves citation snapshots in existing chat history. Deleting a conversation removes its stored messages.

## Incremental SSE chat

Use streamed `fetch` for POST `/rag/chat/stream`; browser `EventSource` cannot send this JSON POST. The server forwards actual Gemini text increments. It does not wait for a full answer and split it into fake chunks.

| Event | JSON data | Frontend behavior |
|---|---|---|
| `meta` | `{ conversation_id, sources }` | Show citations; conversation ID is provisional |
| `delta` | `{ text }` | Append text immediately to the assistant answer |
| `done` | `{ conversation_id, message_id }` | Mark complete; answer and citations are saved |
| `error` | `{ status, message }` | Show error; partial answer was not saved |

Heartbeat comments arrive every 15 seconds while waiting. Ignore comments. Retrieval/validation errors before streaming starts are ordinary HTTP errors. Errors after headers are SSE `error` events even though HTTP status is already 200. A connection ending without `done` is not confirmed success. Keep a new conversation ID only after `done`: a provisional conversation is rolled back if generation fails or disconnects before commit. Disconnecting after commit cannot undo a saved answer.

```js
async function streamChat(body, onEvent, signal) {
  const response = await fetch(`${API}/rag/chat/stream`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) throw new Error(await response.text());
  if (!response.body) throw new Error("Streaming is unavailable.");

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  let completed = false;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        let event = "message";
        const data = [];
        for (const line of frame.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
        }
        if (!data.length) continue; // heartbeat
        const payload = JSON.parse(data.join("\n"));
        if (event === "error") throw new Error(payload.message);
        if (event === "done") completed = true;
        onEvent(event, payload);
      }
    }
    if (!completed) throw new Error("The answer stream ended before completion.");
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

const controller = new AbortController();
let answer = "";
let savedConversationId;
await streamChat(
  { question: "Explain this document.", document_ids: [selectedDocumentId] },
  (event, data) => {
    if (event === "meta") console.log("Sources:", data.sources);
    if (event === "delta") {
      answer += data.text;
      // React: setAnswer(previous => previous + data.text);
    }
    if (event === "done") savedConversationId = data.conversation_id;
  },
  controller.signal,
);
// Stop button: controller.abort(); handle AbortError as cancellation.
```

Render model output and source excerpts safely as text, or use sanitized Markdown. Avoid unsanitized HTML injection. Production proxies must allow long requests and avoid SSE buffering.

## Errors, limits, and test sequence

Ordinary HTTP errors use `{ "detail": "message" }`; validation errors may have an array in `detail`.

| Status | Meaning |
|---|---|
| 400 | Empty/invalid file or no document available for chat |
| 401 | Missing/expired authentication; use existing auth flow |
| 404 | Document/conversation unavailable to this user; selected document must be ready |
| 409 | Stored embedding model differs from configured model; re-upload affected documents |
| 413 | File/chunk/library limit exceeded |
| 415 | Unsupported file extension |
| 422 | Validation error, unreadable/scanned/encrypted PDF, or blocked Gemini response |
| 429 | Rate limit reached |
| 502 | Provider failure or invalid/incomplete model response |
| 503 | Provider credentials are missing or still placeholders |

Upload rate limit: 10 requests/hour. Each chat endpoint: 30 requests/minute. These are enforced by the existing rate limiter; avoid rapid automatic retries. JSON chat returns the exact fallback `I couldn't find the answer in the provided document.` when retrieval does not find enough supporting context; streaming emits that fallback as a delta and then saves it normally.

Recommended frontend test sequence: complete existing login → upload a small text file → list documents → ask a relevant question using JSON chat → continue with its conversation ID → test incremental SSE → load conversation messages → download the original → test deletions. Real provider calls require working credentials; mocked backend tests do not prove live provider access.

Backend sources: `app/routes/rag.py`, `app/schemas/rag.py`, `app/services/rag.py`, and `app/services/ai.py`. If the API changes, update this reference to match these files.
