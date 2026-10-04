from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

PositiveId = Annotated[StrictInt, Field(gt=0)]


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    filename: str = Field(validation_alias="originalName")
    content_type: str = Field(validation_alias="fileType")
    size_bytes: int = Field(validation_alias="fileSize")
    chunk_count: int = Field(validation_alias="chunkCount")
    created_at: datetime = Field(validation_alias="createdAt")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: PositiveId | None = None
    document_ids: list[PositiveId] | None = Field(default=None, min_length=1, max_length=20)

    @field_validator("question")
    @classmethod
    def meaningful_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be blank")
        return value.strip()


class Source(BaseModel):
    citation: int
    document_id: int
    filename: str
    chunk: int
    page: int | None
    text: str
    score: float


class ChatResponse(BaseModel):
    conversation_id: int
    answer: str
    sources: list[Source]


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str | None
    created_at: datetime = Field(validation_alias="createdAt")


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    sources: list[Source]
    created_at: datetime
