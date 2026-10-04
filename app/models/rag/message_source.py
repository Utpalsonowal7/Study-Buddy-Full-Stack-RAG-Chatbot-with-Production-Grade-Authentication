from datetime import datetime

from sqlalchemy import (
    Integer,
    JSON,
    Float,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class MessageSource(Base):
    __tablename__ = "message_sources"

    __table_args__ = (
        UniqueConstraint(
            "messageId",
            "chunkId",
            name="uq_message_source_chunk",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    messageId: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    chunkId: Mapped[int | None] = mapped_column(
        ForeignKey("docs_chunks.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    similarity: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    # Retain citations in chat history even after the original document is deleted.
    snapshot: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict, server_default="{}")

    createdAt: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    message = relationship(
        "Message",
        back_populates="sources",
    )

    chunk = relationship(
        "DocumentChunk",
        back_populates="sources",
    )
