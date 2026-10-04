import logging
import math
from pathlib import PurePath
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import RagSettings
from app.models.auth.user import User
from app.models.rag import Conversation, Document, DocumentChunk, DocumentStatus, Message, MessageRole, MessageSource
from app.schemas.rag import ChatRequest
from app.services.ai import AIProvider

logger = logging.getLogger(__name__)


async def owned_document(db: AsyncSession, user_id: int, document_id: int) -> Document:
    document = await db.scalar(select(Document).where(Document.id == document_id, Document.userId == user_id))
    if document is None:
        raise HTTPException(404, "Document not found.")
    return document


async def owned_conversation(db: AsyncSession, user_id: int, conversation_id: int, lock: bool = False) -> Conversation:
    query = select(Conversation).where(Conversation.id == conversation_id, Conversation.userId == user_id)
    conversation = await db.scalar(query.with_for_update() if lock else query)
    if conversation is None:
        raise HTTPException(404, "Conversation not found.")
    return conversation


async def index_document(db, user, filename, content_type, data, chunks, settings, ai, storage):
    # Serialize quota checks for concurrent uploads belonging to the same user.
    await db.scalar(select(User).where(User.id == user.id).with_for_update())
    count = await db.scalar(select(func.count(DocumentChunk.id)).join(Document).where(Document.userId == user.id))
    if count + len(chunks) > settings.max_chunks_per_user:
        raise HTTPException(413, "Your document library is full. Delete documents before uploading more.")
    storage.options()
    vectors = await ai.embed([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks) or any(len(vector) != 3072 for vector in vectors):
        raise HTTPException(502, "The embedding provider must return 3072 dimensions for each chunk.")
    public_id = f"study-buddy/{user.id}/{uuid4()}{PurePath(filename).suffix.lower()}"
    file_url = await storage.upload(data, public_id)
    try:
        document = Document(userId=user.id, name=filename, originalName=filename,
                            fileType=content_type, fileSize=len(data), fileUrl=file_url,
                            cloudinaryPublicId=public_id, resourceType="raw", status=DocumentStatus.READY,
                            pageCount=max((chunk.page or 0 for chunk in chunks), default=0) or None,
                            embeddingModel=settings.embedding_model, chunkCount=len(chunks))
        db.add(document)
        await db.flush()
        db.add_all([DocumentChunk(documentId=document.id, chunkIndex=i, pageNumber=chunk.page,
                                  content=chunk.text, embedding=vector)
                    for i, (chunk, vector) in enumerate(zip(chunks, vectors))])
        await db.commit()
        return document
    except Exception:
        await db.rollback()
        try:
            await storage.delete(public_id)
        except Exception:
            logger.error("Cloudinary cleanup failed for asset %s; storage reconciliation is required.", public_id)
        raise


async def conversation_messages(db: AsyncSession, conversation_id: int, after_id: int, limit: int):
    messages = (await db.scalars(select(Message).where(Message.conversationId == conversation_id, Message.id > after_id)
                                .options(selectinload(Message.sources)).order_by(Message.id).limit(limit))).all()
    return [{"id": message.id, "role": message.role.value.lower(), "content": message.content,
             "sources": sorted([source.snapshot for source in message.sources if source.snapshot],
                               key=lambda source: source["citation"]), "created_at": message.createdAt}
            for message in messages]


async def chat(db: AsyncSession, user: User, request: ChatRequest, settings: RagSettings, ai: AIProvider):
    conversation = None
    history = []
    if request.conversation_id:
        conversation = await owned_conversation(db, user.id, request.conversation_id, lock=True)
        messages = (await db.scalars(select(Message).where(Message.conversationId == conversation.id)
                                    .order_by(Message.id.desc()).limit(10))).all()
        history = [{"role": message.role.value.lower(), "content": message.content[:4000]} for message in reversed(messages)]

    documents_query = select(Document).where(Document.userId == user.id, Document.status == DocumentStatus.READY)
    if request.document_ids:
        ids = set(request.document_ids)
        documents_query = documents_query.where(Document.id.in_(ids))
    documents = (await db.scalars(documents_query)).all()
    if request.document_ids and len(documents) != len(ids):
        raise HTTPException(404, "One or more ready documents were not found.")
    if not documents:
        raise HTTPException(400, "Upload a document before asking a question.")
    if any(document.embeddingModel != settings.embedding_model for document in documents):
        raise HTTPException(409, "The embedding model changed. Re-upload documents using the current model.")

    previous_questions = [message["content"] for message in history if message["role"] == "user"][-2:]
    query_vector = (await ai.embed(["\n".join([*previous_questions, request.question])], task_type="RETRIEVAL_QUERY"))[0]
    if len(query_vector) != 3072 or not all(math.isfinite(value) for value in query_vector) or not math.hypot(*query_vector):
        raise HTTPException(502, "The embedding provider must return a valid 3072-dimensional query vector.")
    # Rank vectors in PostgreSQL using the project's existing pgvector column.
    distance = DocumentChunk.embedding.cosine_distance(query_vector)
    ranked = (await db.execute(select(DocumentChunk, distance.label("distance"))
                              .where(DocumentChunk.documentId.in_([d.id for d in documents]))
                              .order_by(distance, DocumentChunk.id).limit(settings.top_k))).all()
    by_id = {document.id: document for document in documents}
    sources, source_rows = [], []
    for chunk, distance_value in ranked:
        score = 1.0 - float(distance_value)
        if not math.isfinite(score) or score < settings.min_score:
            continue
        source = {"citation": len(sources) + 1, "document_id": chunk.documentId,
                  "filename": by_id[chunk.documentId].originalName, "chunk": chunk.chunkIndex,
                  "page": chunk.pageNumber, "text": chunk.content, "score": round(score, 6)}
        sources.append(source)
        source_rows.append((chunk.id, source))
    answer = await ai.answer(request.question, sources, history) if sources else (
        "I couldn't find relevant information in your documents. Try a more specific question or upload another document."
    )
    if conversation is None:
        conversation = Conversation(userId=user.id, title=request.question[:100])
        db.add(conversation)
        await db.flush()
    question_message = Message(conversationId=conversation.id, role=MessageRole.USER, content=request.question)
    answer_message = Message(conversationId=conversation.id, role=MessageRole.ASSISTANT, content=answer)
    db.add_all([question_message, answer_message])
    await db.flush()
    db.add_all([MessageSource(messageId=answer_message.id, chunkId=chunk_id,
                              similarity=source["score"], snapshot=source)
                for chunk_id, source in source_rows])
    conversation_id = conversation.id
    await db.commit()
    return {"conversation_id": conversation_id, "answer": answer, "sources": sources}
