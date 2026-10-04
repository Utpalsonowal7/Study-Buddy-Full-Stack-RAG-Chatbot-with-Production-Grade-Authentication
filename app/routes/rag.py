from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.dependencies.auth import get_current_user
from app.dependencies.rate_limit import rate_limit_dependency
from app.dependencies.session import get_session
from app.models.auth.user import User
from app.utils.documents import extract_chunks, safe_filename
from app.models.rag import Conversation, Document
from app.services.ai import AIProvider
from app.services.cloudinary import CloudinaryStorage
from app.schemas.rag import ChatRequest, ChatResponse, ConversationOut, DocumentOut, MessageOut
from app.services.rag import chat, index_document, owned_conversation, owned_document, conversation_messages, prepare_chat, chat_stream
from app.config import RagSettings
from app.dependencies.rag import get_settings, get_ai, get_storage
from app.utils.sse import with_heartbeats

router = APIRouter(prefix="/rag", tags=["Study documents and chat"])


@router.post("/documents", response_model=DocumentOut, status_code=201,
             dependencies=[Depends(rate_limit_dependency(10, 3600))])
async def upload_document(file: UploadFile = File(...), user: User = Depends(get_current_user),
                          db: AsyncSession = Depends(get_session), settings: RagSettings = Depends(get_settings),
                          ai: AIProvider = Depends(get_ai), storage: CloudinaryStorage = Depends(get_storage)):
    try:
        data = await file.read(settings.max_file_bytes + 1)
        if len(data) > settings.max_file_bytes:
            raise HTTPException(413, "Files may be at most 10 MiB.")
        if not data:
            raise HTTPException(400, "The file is empty.")
        filename = safe_filename(file.filename)
        content_type, chunks = await run_in_threadpool(extract_chunks, data, filename, settings.max_chunks_per_document)
        return await index_document(db, user, filename, content_type, data, chunks, settings, ai, storage)
    finally:
        await file.close()


@router.get("/documents", response_model=list[DocumentOut])
async def list_documents(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_session),
                         offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100)):
    return (await db.scalars(select(Document).where(Document.userId == user.id)
                            .order_by(Document.createdAt.desc(), Document.id).offset(offset).limit(limit))).all()


@router.get("/documents/{document_id}", response_model=DocumentOut)
async def get_document(document_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_session)):
    return await owned_document(db, user.id, document_id)


@router.get("/documents/{document_id}/download")
async def download_document(document_id: int, user: User = Depends(get_current_user),
                            db: AsyncSession = Depends(get_session), storage: CloudinaryStorage = Depends(get_storage)):
    document = await owned_document(db, user.id, document_id)
    return {"url": storage.download_url(document.cloudinaryPublicId), "expires_in": 60}


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(document_id: int, user: User = Depends(get_current_user),
                          db: AsyncSession = Depends(get_session), storage: CloudinaryStorage = Depends(get_storage)):
    document = await owned_document(db, user.id, document_id)
    await storage.delete(document.cloudinaryPublicId)
    await db.delete(document)
    await db.commit()
    return Response(status_code=204)


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(rate_limit_dependency(30, 60))])
async def ask(request: ChatRequest, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_session),
              settings: RagSettings = Depends(get_settings), ai: AIProvider = Depends(get_ai)):
    return await chat(db, user, request, settings, ai)


@router.post("/chat/stream", response_class=StreamingResponse,
             responses={200: {"content": {"text/event-stream": {}}}},
             dependencies=[Depends(rate_limit_dependency(30, 60))])
async def ask_stream(request: ChatRequest, http_request: Request, user: User = Depends(get_current_user),
                     db: AsyncSession = Depends(get_session, scope="request"),
                     settings: RagSettings = Depends(get_settings), ai: AIProvider = Depends(get_ai)):
    # Validate ownership and retrieve context before committing HTTP 200 headers.
    context = await prepare_chat(db, user, request, settings, ai)
    return StreamingResponse(
        with_heartbeats(chat_stream(db, context, ai, http_request), http_request), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_session),
                             offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100)):
    return (await db.scalars(select(Conversation).where(Conversation.userId == user.id)
                            .order_by(Conversation.createdAt.desc(), Conversation.id).offset(offset).limit(limit))).all()


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut])
async def list_messages(conversation_id: int, user: User = Depends(get_current_user),
                        db: AsyncSession = Depends(get_session), after_id: int = Query(0, ge=0),
                        limit: int = Query(100, ge=1, le=100)):
    conversation = await owned_conversation(db, user.id, conversation_id)
    return await conversation_messages(db, conversation.id, after_id, limit)


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_session)):
    conversation = await owned_conversation(db, user.id, conversation_id, lock=True)
    await db.delete(conversation)
    await db.commit()
    return Response(status_code=204)
