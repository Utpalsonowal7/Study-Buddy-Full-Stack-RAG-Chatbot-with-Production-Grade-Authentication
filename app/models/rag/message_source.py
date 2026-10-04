from datetime import datetime

from sqlalchemy import (
    Integer,
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

    chunkId: Mapped[int] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    similarity: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

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
