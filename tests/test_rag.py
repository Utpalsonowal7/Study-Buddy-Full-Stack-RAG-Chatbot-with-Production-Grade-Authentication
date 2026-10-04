import json
import os
import secrets
import unittest
from dataclasses import replace
from io import BytesIO
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

# Configure the isolated test process before importing app.config.
if os.getenv("TEST_DATABASE_URL"):
    from sqlalchemy.engine import make_url
    test_database = make_url(os.environ["TEST_DATABASE_URL"]).database
    if not test_database or not test_database.startswith("study_buddy_test"):
        raise ValueError("TEST_DATABASE_URL must name an isolated study_buddy_test* database.")
    os.environ.update(DATABASE_URL=os.environ["TEST_DATABASE_URL"], REDIS_URL="redis://127.0.0.1:6379/15",
                      JWT_ACCESS_TOKEN_SECRECT=secrets.token_urlsafe(48),
                      JWT_REFRESH_TOKEN_SECRET=secrets.token_urlsafe(48), JWT_ALGORITHM="HS256",
                      JWT_ACCESS_TOKEN_EXPIRE_MINUTES="15", JWT_REFRESH_TOKEN_EXPIRE_DAYS="30")

from app.utils.documents import extract_chunks, safe_filename
from app.services.ai import AIProvider
from app.services.cloudinary import CloudinaryStorage
from app.config import RagSettings as Settings


def vector(index=0):
    return [1.0 if i == index else 0.0 for i in range(3072)]


def settings():
    return Settings("test-key", "https://generativelanguage.googleapis.com/v1beta", "test-embedding", "test-chat",
                    "test-cloud", "test-cloud-key", "test-cloud-secret")


class ExtractionTests(unittest.TestCase):
    def test_pdf_text_and_page_number(self):
        writer = PdfWriter()
        page = writer.add_blank_page(200, 200)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 10 100 Td (Photosynthesis uses sunlight.) Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
        output = BytesIO()
        writer.write(output)
        content_type, chunks = extract_chunks(output.getvalue(), "biology.pdf")
        self.assertEqual(content_type, "application/pdf")
        self.assertEqual(chunks[0].page, 1)
        self.assertIn("Photosynthesis uses sunlight", chunks[0].text)

    def test_chunk_overlap_and_text_validation(self):
        text = "Photosynthesis turns light into chemical energy. " * 100
        content_type, chunks = extract_chunks(text.encode(), "notes.txt")
        self.assertEqual(content_type, "text/plain")
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].text[-200:], chunks[1].text[:200])
        self.assertEqual(safe_filename("../../notes.txt"), "notes.txt")
        for data, name, status in [(b"", "notes.txt", 422), (b"\xff", "notes.txt", 400),
                                   (b"binary\x00", "notes.md", 400), (b"hello", "file.exe", 415),
                                   (b"fake", "notes.pdf", 400)]:
            with self.subTest(name=name, data=data):
                with self.assertRaises(HTTPException) as raised:
                    extract_chunks(data, name)
                self.assertEqual(raised.exception.status_code, status)
        with self.assertRaises(HTTPException) as raised:
            extract_chunks(text.encode(), "notes.txt", limit=1)
        self.assertEqual(raised.exception.status_code, 413)

    def test_scanned_and_encrypted_pdf(self):
        writer = PdfWriter()
        writer.add_blank_page(200, 200)
        output = BytesIO()
        writer.write(output)
        with self.assertRaises(HTTPException) as raised:
            extract_chunks(output.getvalue(), "blank.pdf")
        self.assertEqual(raised.exception.status_code, 422)
        writer.encrypt("test")
        output = BytesIO()
        writer.write(output)
        with self.assertRaises(HTTPException) as raised:
            extract_chunks(output.getvalue(), "encrypted.pdf")
        self.assertEqual(raised.exception.status_code, 400)


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_gemini_wire_format_task_types_and_conversation(self):
        requests = []
        def handler(request):
            payload = json.loads(request.content)
            requests.append((request.url.path, payload))
            self.assertEqual(request.headers["x-goog-api-key"], "test-key")
            self.assertNotIn("Authorization", request.headers)
            self.assertEqual(request.url.query, b"")
            if request.url.path.endswith(":batchEmbedContents"):
                return httpx.Response(200, json={"embeddings": [{"values": vector(index)} for index in range(len(payload["requests"]))]})
            return httpx.Response(200, json={"candidates": [{"finishReason": "STOP", "content": {"parts": [
                {"text": "Internal reasoning", "thought": True},
                {"text": "Light becomes chemical energy "}, {"text": "[1]."}]}}]})
        real_client = httpx.AsyncClient
        with patch("app.services.ai.httpx.AsyncClient", side_effect=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)):
            ai = AIProvider(settings())
            self.assertEqual(await ai.embed(["first", "second"]), [vector(), vector(1)])
            self.assertEqual(await ai.embed(["question"], task_type="RETRIEVAL_QUERY"), [vector()])
            answer = await ai.answer("What happens?", [{"citation": 1, "filename": "notes.txt", "page": None, "text": "Light becomes chemical energy."}],
                                     [{"role": "user", "content": "Explain light"}, {"role": "assistant", "content": "Earlier answer [1]."}])
            self.assertEqual(answer, "Light becomes chemical energy [1].")
        self.assertEqual(requests[0][0], "/v1beta/models/test-embedding:batchEmbedContents")
        for item in requests[0][1]["requests"]:
            self.assertEqual(item["model"], "models/test-embedding")
            self.assertEqual(item["outputDimensionality"], 3072)
            self.assertEqual(item["taskType"], "RETRIEVAL_DOCUMENT")
        self.assertEqual(requests[1][1]["requests"][0]["taskType"], "RETRIEVAL_QUERY")
        self.assertEqual(requests[2][0], "/v1beta/models/test-chat:generateContent")
        self.assertIn("untrusted", requests[2][1]["systemInstruction"]["parts"][0]["text"])
        self.assertEqual([content["role"] for content in requests[2][1]["contents"]], ["user", "model", "user"])
        self.assertIn("[1] notes.txt", requests[2][1]["contents"][-1]["parts"][0]["text"])

    async def test_gemini_embedding_batches_preserve_order(self):
        ai = AIProvider(settings())
        async def response(endpoint, payload):
            return {"embeddings": [{"values": vector(int(item["content"]["parts"][0]["text"]))} for item in payload["requests"]]}
        ai._request = AsyncMock(side_effect=response)
        vectors = await ai.embed([str(index) for index in range(33)])
        self.assertEqual(vectors, [vector(index) for index in range(33)])
        self.assertEqual([len(call.args[1]["requests"]) for call in ai._request.call_args_list], [32, 1])

    async def test_gemini_blocked_incomplete_and_invalid_answers(self):
        for result, status in [
            ({"promptFeedback": {"blockReason": "SAFETY"}}, 422),
            ({"candidates": [{"finishReason": "SAFETY"}]}, 422),
            ({"candidates": [{"finishReason": "MAX_TOKENS"}]}, 502),
            ({"candidates": []}, 502),
            ({"promptFeedback": "invalid", "candidates": []}, 502),
            ({"candidates": ["invalid"]}, 502),
            ({"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "Hidden", "thought": True}]}}]}, 502),
        ]:
            with self.subTest(result=result):
                ai = AIProvider(settings())
                ai._request = AsyncMock(return_value=result)
                with self.assertRaises(HTTPException) as raised:
                    await ai.answer("What happens?", [], [])
                self.assertEqual(raised.exception.status_code, status)

    def test_gemini_defaults_and_existing_model_resource_names(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-key"}):
            for name in ["GEMINI_API_BASE_URL", "RAG_EMBEDDING_MODEL", "RAG_CHAT_MODEL"]:
                os.environ.pop(name, None)
            configured = Settings.from_env()
        self.assertEqual(configured.api_key, "test-gemini-key")
        self.assertEqual(configured.api_url, "https://generativelanguage.googleapis.com/v1beta")
        self.assertEqual(configured.embedding_model, "gemini-embedding-001")
        self.assertEqual(configured.chat_model, "gemini-flash-latest")
        self.assertEqual(AIProvider.model_path("models/gemini-embedding-001"), "models/gemini-embedding-001")
        with self.assertRaises(HTTPException):
            AIProvider.model_path("../invalid/model")

    async def test_placeholder_and_invalid_provider_outputs(self):
        with self.assertRaises(HTTPException) as raised:
            await AIProvider(replace(settings(), api_key="replace-me")).embed(["hello"])
        self.assertEqual(raised.exception.status_code, 503)
        for response in [{"embeddings": []}, {"embeddings": [{"values": [float('nan')] * 3072}]},
                         {"embeddings": [{"values": [0] * 3072}]}, {"embeddings": [{"values": [1, 0]}]}]:
            ai = AIProvider(settings())
            ai._request = AsyncMock(return_value=response)
            with self.assertRaises(HTTPException) as raised:
                await ai.embed(["hello"])
            self.assertEqual(raised.exception.status_code, 502)

    async def test_upstream_errors_do_not_expose_credentials(self):
        real_client = httpx.AsyncClient
        with patch("app.services.ai.httpx.AsyncClient", side_effect=lambda **kw: real_client(
                transport=httpx.MockTransport(lambda request: httpx.Response(401, json={"error": "private upstream details"})), **kw)):
            with self.assertRaises(HTTPException) as raised:
                await AIProvider(settings()).embed(["hello"])
        self.assertEqual(raised.exception.status_code, 502)
        self.assertNotIn("test-key", raised.exception.detail)
        self.assertNotIn("private upstream", raised.exception.detail)

    async def test_cloudinary_private_upload_delete_and_expiring_download(self):
        storage = CloudinaryStorage(settings())
        with patch("cloudinary.uploader.upload", return_value={"public_id": "study-buddy/1/test.txt", "secure_url": "https://res.cloudinary.com/test-cloud/raw/authenticated/test.txt"}) as upload:
            await storage.upload(b"hello", "study-buddy/1/test.txt")
            self.assertEqual(upload.call_args.kwargs["type"], "authenticated")
            self.assertEqual(upload.call_args.kwargs["resource_type"], "raw")
            self.assertFalse(upload.call_args.kwargs["overwrite"])
        url = storage.download_url("study-buddy/1/test.txt")
        self.assertIn("expires_at=", url)
        self.assertIn("type=authenticated", url)
        self.assertNotIn("test-cloud-secret", url)
        with patch("cloudinary.uploader.destroy", return_value={"result": "ok"}) as delete:
            await storage.delete("study-buddy/1/test.txt")
            self.assertEqual(delete.call_args.kwargs["type"], "authenticated")


@unittest.skipUnless(os.getenv("TEST_DATABASE_URL"), "Set TEST_DATABASE_URL to an isolated PostgreSQL test database")
class RagIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_schema_migration_is_repeatable_and_preserves_records(self):
        from sqlalchemy import text
        from app.db.database import engine
        from app.db.migrate_rag import migrate

        schema = "rag_migration_test_" + secrets.token_hex(8)
        try:
            async with engine.begin() as connection:
                await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
                await connection.execute(text(f'SET LOCAL search_path TO "{schema}", public'))
                await connection.execute(text('CREATE TABLE docs (id INTEGER PRIMARY KEY, name TEXT)'))
                await connection.execute(text('CREATE TABLE docs_chunks (id INTEGER PRIMARY KEY)'))
                await connection.execute(text('CREATE TABLE message_sources (id INTEGER PRIMARY KEY, "chunkId" INTEGER NOT NULL REFERENCES docs_chunks(id) ON DELETE CASCADE)'))
                await connection.execute(text("INSERT INTO docs VALUES (1, 'existing document')"))
                await connection.execute(text('INSERT INTO docs_chunks VALUES (1)'))
                await connection.execute(text('INSERT INTO message_sources VALUES (1, 1)'))
                await migrate(connection)
                await migrate(connection)
                row = (await connection.execute(text('SELECT name, "embeddingModel", "chunkCount" FROM docs WHERE id=1'))).one()
                self.assertEqual(tuple(row), ('existing document', 'legacy_unknown', 0))
                await connection.execute(text('DELETE FROM docs_chunks WHERE id=1'))
                row = (await connection.execute(text('SELECT "chunkId", snapshot FROM message_sources WHERE id=1'))).one()
                self.assertIsNone(row[0])
                self.assertEqual(row[1], {})
                await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        finally:
            await engine.dispose()

    async def test_authenticated_document_chat_lifecycle_and_isolation(self):
        from sqlalchemy.engine import make_url
        url = os.environ["TEST_DATABASE_URL"]
        self.assertTrue(make_url(url).database.startswith("study_buddy_test"), "Use an isolated study_buddy_test* database")
        os.environ.update(DATABASE_URL=url, REDIS_URL="redis://127.0.0.1:6379/15",
                          JWT_ACCESS_TOKEN_SECRECT=secrets.token_urlsafe(48),
                          JWT_REFRESH_TOKEN_SECRET=secrets.token_urlsafe(48), JWT_ALGORITHM="HS256",
                          JWT_ACCESS_TOKEN_EXPIRE_MINUTES="15", JWT_REFRESH_TOKEN_EXPIRE_DAYS="30")
        from app.main import app
        from app.db.database import engine, LocalSession
        from app.models.rag import DocumentChunk, Conversation, Document, Message, MessageSource
        from app.models.auth.user import User
        from app.dependencies.rag import get_ai, get_settings, get_storage
        from sqlalchemy import delete, select, func
        from sqlalchemy.ext.asyncio import AsyncSession

        ai = AIProvider(settings())
        ai.embed = AsyncMock(side_effect=lambda texts, **kwargs: [vector() for _ in texts])
        ai.answer = AsyncMock(return_value="Photosynthesis converts light into chemical energy [1].")
        storage = CloudinaryStorage(settings())
        storage.upload = AsyncMock(return_value="https://res.cloudinary.com/test-cloud/raw/authenticated/test.txt")
        storage.delete = AsyncMock()
        app.dependency_overrides.update({get_ai: lambda: ai, get_settings: settings, get_storage: lambda: storage})
        emails = [f"rag-{secrets.token_hex(8)}@example.com" for _ in range(2)]
        try:
            async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test") as client:
                with patch("app.dependencies.rate_limit.rate_limit", new=AsyncMock()):
                    response = await client.get("/api/v1/rag/documents")
                    self.assertEqual(response.status_code, 401)
                    response = await client.post("/api/v1/auth/register", json={"email": emails[0], "full_name": "Student"})
                    self.assertEqual(response.status_code, 200, response.text)
                    owner_cookies = httpx.Cookies(client.cookies)
                    response = await client.post("/api/v1/rag/documents", files={"file": ("biology.txt", b"Photosynthesis converts light into chemical energy.")})
                    self.assertEqual(response.status_code, 201, response.text)
                    document_id = response.json()["id"]
                    self.assertEqual(response.json()["chunk_count"], 1)
                    self.assertNotIn("public_id", response.json())
                    response = await client.get("/api/v1/rag/documents")
                    self.assertEqual([doc["id"] for doc in response.json()], [document_id])
                    app.dependency_overrides[get_settings] = lambda: replace(settings(), max_chunks_per_user=1)
                    response = await client.post("/api/v1/rag/documents", files={"file": ("extra.txt", b"More notes")})
                    self.assertEqual(response.status_code, 413)
                    app.dependency_overrides[get_settings] = settings
                    response = await client.post("/api/v1/rag/chat", json={"question": "What is photosynthesis?", "document_ids": [document_id]})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(ai.embed.call_args.kwargs["task_type"], "RETRIEVAL_QUERY")
                    conversation_id = response.json()["conversation_id"]
                    self.assertEqual(response.json()["sources"][0]["document_id"], document_id)
                    response = await client.post("/api/v1/rag/chat", json={"question": "Explain it simply", "conversation_id": conversation_id})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(len(ai.answer.call_args.args[2]), 2)
                    response = await client.get(f"/api/v1/rag/conversations/{conversation_id}/messages")
                    self.assertEqual([message["role"] for message in response.json()], ["user", "assistant", "user", "assistant"])
                    response = await client.get(f"/api/v1/rag/documents/{document_id}/download")
                    self.assertEqual(response.status_code, 200)
                    self.assertIn("expires_at", response.json()["url"])
                    ai.embed.side_effect = lambda texts, **kwargs: [vector(1) for _ in texts]
                    response = await client.post("/api/v1/rag/chat", json={"question": "Unrelated question"})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["sources"], [])
                    self.assertEqual(ai.answer.await_count, 2)
                    app.dependency_overrides[get_settings] = lambda: replace(settings(), embedding_model="different-model")
                    response = await client.post("/api/v1/rag/chat", json={"question": "Explain"})
                    self.assertEqual(response.status_code, 409)
                    app.dependency_overrides[get_settings] = settings

                    client.cookies.clear()
                    response = await client.post("/api/v1/auth/register", json={"email": emails[1], "full_name": "Other student"})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual((await client.get("/api/v1/rag/documents")).json(), [])
                    for method, path, payload in [
                        ("GET", f"/documents/{document_id}", None),
                        ("GET", f"/documents/{document_id}/download", None),
                        ("DELETE", f"/documents/{document_id}", None),
                        ("GET", f"/conversations/{conversation_id}/messages", None),
                        ("DELETE", f"/conversations/{conversation_id}", None),
                        ("POST", "/chat", {"question": "Get private data", "document_ids": [document_id]}),
                        ("POST", "/chat", {"question": "Get private data", "conversation_id": conversation_id}),
                    ]:
                        response = await client.request(method, "/api/v1/rag" + path, json=payload)
                        self.assertEqual(response.status_code, 404, (path, response.text))

                    client.cookies.clear()
                    client.cookies.update(owner_cookies)
                    response = await client.post("/api/v1/rag/documents", files={"file": ("bad.exe", b"bad")})
                    self.assertEqual(response.status_code, 415)
                    ai.embed.side_effect = HTTPException(502, "Provider down")
                    response = await client.post("/api/v1/rag/documents", files={"file": ("notes.txt", b"some notes")})
                    self.assertEqual(response.status_code, 502)
                    self.assertEqual(storage.upload.await_count, 1)
                    ai.embed.side_effect = lambda texts, **kwargs: [vector() for _ in texts]
                    storage.upload.side_effect = HTTPException(502, "Storage down")
                    response = await client.post("/api/v1/rag/documents", files={"file": ("notes.txt", b"some notes")})
                    self.assertEqual(response.status_code, 502)
                    self.assertEqual(len((await client.get("/api/v1/rag/documents")).json()), 1)
                    storage.upload.side_effect = None
                    with patch.object(AsyncSession, "commit", new=AsyncMock(side_effect=RuntimeError("Simulated database commit failure"))):
                        with self.assertRaises(RuntimeError):
                            await client.post("/api/v1/rag/documents", files={"file": ("rollback.txt", b"Rollback notes")})
                    self.assertEqual(storage.delete.await_count, 1)
                    self.assertEqual(len((await client.get("/api/v1/rag/documents")).json()), 1)
                    storage.delete.side_effect = HTTPException(502, "Storage down")
                    response = await client.delete(f"/api/v1/rag/documents/{document_id}")
                    self.assertEqual(response.status_code, 502)
                    self.assertEqual((await client.get(f"/api/v1/rag/documents/{document_id}")).status_code, 200)
                    storage.delete.side_effect = None
                    response = await client.delete(f"/api/v1/rag/documents/{document_id}")
                    self.assertEqual(response.status_code, 204)
                    retained = await client.get(f"/api/v1/rag/conversations/{conversation_id}/messages")
                    self.assertEqual(retained.json()[1]["sources"][0]["filename"], "biology.txt")
                    async with LocalSession() as db:
                        self.assertEqual(await db.scalar(select(func.count(DocumentChunk.id)).where(DocumentChunk.documentId == document_id)), 0)
                    response = await client.delete(f"/api/v1/rag/conversations/{conversation_id}")
                    self.assertEqual(response.status_code, 204)
                    async with LocalSession() as db:
                        self.assertEqual(await db.scalar(select(func.count(Message.id)).where(Message.conversationId == conversation_id)), 0)
                    response = await client.post("/api/v1/auth/refresh")
                    self.assertEqual(response.status_code, 200, response.text)
                    response = await client.post("/api/v1/auth/refresh")
                    self.assertEqual(response.status_code, 200, response.text)
                    response = await client.post("/api/v1/auth/logout")
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual((await client.get("/api/v1/rag/documents")).status_code, 401)
        finally:
            app.dependency_overrides.clear()
            async with LocalSession() as db:
                await db.execute(delete(User).where(User.email.in_(emails)))
                await db.commit()
            await engine.dispose()


if __name__ == "__main__":
    unittest.main()
