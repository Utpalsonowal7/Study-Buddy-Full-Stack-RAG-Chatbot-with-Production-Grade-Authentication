import logging
import math
from pathlib import PurePath
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.auth.user import User
from app.rag.models import Chunk, Conversation, Document, Message
from app.rag.providers import AIProvider, CloudinaryStorage
from app.rag.schemas import ChatRequest
from app.rag.settings import Settings

logger = logging.getLogger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise HTTPException(409, "Embedding dimensions changed. Re-upload documents using the current model.")
    denominator = math.hypot(*a) * math.hypot(*b)
    return sum(x * y for x, y in zip(a, b)) / denominator if denominator else 0.0


async def owned_document(db: AsyncSession, user_id: int, document_id: str) -> Document:
    document = await db.scalar(select(Document).where(Document.id == document_id, Document.user_id == user_id))
    if document is None:
        raise HTTPException(404, "Document not found.")
    return document


async def owned_conversation(db: AsyncSession, user_id: int, conversation_id: str, lock: bool = False) -> Conversation:
    query = select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user_id)
    conversation = await db.scalar(query.with_for_update() if lock else query)
    if conversation is None:
        raise HTTPException(404, "Conversation not found.")
    return conversation


async def index_document(db, user, filename, content_type, data, chunks, settings, ai, storage):
    # Serialize quota checks for concurrent uploads belonging to the same user.
    await db.scalar(select(User).where(User.id == user.id).with_for_update())
    count = await db.scalar(select(func.count(Chunk.id)).join(Document).where(Document.user_id == user.id))
    if count + len(chunks) > settings.max_chunks_per_user:
        raise HTTPException(413, "Your document library is full. Delete documents before uploading more.")
    # Check storage configuration before spending money on embeddings.
    storage.options()
    vectors = await ai.embed([chunk.text for chunk in chunks])
    document_id = str(uuid4())
    public_id = f"study-buddy/{user.id}/{document_id}{PurePath(filename).suffix.lower()}"
    await storage.upload(data, public_id)
    try:
        document = Document(id=document_id, user_id=user.id, filename=filename,
                            content_type=content_type, size_bytes=len(data), public_id=public_id,
                            embedding_model=settings.embedding_model, chunk_count=len(chunks))
        db.add(document)
        await db.flush()
        db.add_all([Chunk(document_id=document_id, position=i, page=chunk.page,
                          text=chunk.text, embedding=vector)
                    for i, (chunk, vector) in enumerate(zip(chunks, vectors))])
        await db.commit()
        return document
    except Exception:
        await db.rollback()
        try:
            await storage.delete(public_id)
        except Exception:
            logger.error("Cloudinary cleanup failed for document %s; storage reconciliation is required.", document_id)
        raise


async def chat(db: AsyncSession, user: User, request: ChatRequest, settings: Settings, ai: AIProvider):
    conversation = None
    history = []
    if request.conversation_id:
        conversation = await owned_conversation(db, user.id, str(request.conversation_id), lock=True)
        messages = (await db.scalars(select(Message).where(Message.conversation_id == conversation.id)
                                    .order_by(Message.id.desc()).limit(10))).all()
        history = [{"role": message.role, "content": message.content[:4000]} for message in reversed(messages)]

    documents_query = select(Document).where(Document.user_id == user.id)
    if request.document_ids:
        ids = {str(value) for value in request.document_ids}
        documents_query = documents_query.where(Document.id.in_(ids))
    documents = (await db.scalars(documents_query)).all()
    if request.document_ids and len(documents) != len(ids):
        raise HTTPException(404, "One or more documents were not found.")
    if not documents:
        raise HTTPException(400, "Upload a document before asking a question.")
    if any(document.embedding_model != settings.embedding_model for document in documents):
        raise HTTPException(409, "The embedding model changed. Re-upload documents using the current model.")

    # Include recent user questions so follow-up questions can retrieve the right passages.
    previous_questions = [message["content"] for message in history if message["role"] == "user"][-2:]
    retrieval_question = "\n".join([*previous_questions, request.question])
    query_vector = (await ai.embed([retrieval_question]))[0]
    chunks = (await db.scalars(select(Chunk).where(Chunk.document_id.in_([d.id for d in documents]))
                              .order_by(Chunk.id).limit(settings.max_chunks_per_user + 1))).all()
    if len(chunks) > settings.max_chunks_per_user:
        raise HTTPException(413, "Document library exceeds the retrieval limit.")
    ranked = sorted(((cosine_similarity(query_vector, chunk.embedding), chunk) for chunk in chunks),
                    key=lambda item: item[0], reverse=True)
    by_id = {document.id: document for document in documents}
    sources = []
    for score, chunk in ranked[:settings.top_k]:
        if score < settings.min_score:
            continue
        sources.append({"citation": len(sources) + 1, "document_id": chunk.document_id,
                        "filename": by_id[chunk.document_id].filename, "chunk": chunk.position,
                        "page": chunk.page, "text": chunk.text, "score": round(score, 6)})
    answer = await ai.answer(request.question, sources, history) if sources else (
        "I couldn't find relevant information in your documents. Try a more specific question or upload another document."
    )
    if conversation is None:
        conversation = Conversation(user_id=user.id, title=request.question[:100])
        db.add(conversation)
        await db.flush()
    db.add_all([
        Message(conversation_id=conversation.id, role="user", content=request.question, sources=[]),
        Message(conversation_id=conversation.id, role="assistant", content=answer, sources=sources),
    ])
    conversation_id = conversation.id
    await db.commit()
    return {"conversation_id": conversation_id, "answer": answer, "sources": sources}
