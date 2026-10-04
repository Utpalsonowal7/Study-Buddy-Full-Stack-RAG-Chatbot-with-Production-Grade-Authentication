from datetime import datetime

from sqlalchemy import (
    Integer,
    Text,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from app.db.database import Base


class DocumentChunk(Base):
    __tablename__ = "docs_chunks"

    __table_args__ = (
        UniqueConstraint(
            "documentId",
            "chunkIndex",
            name="uq_document_chunk_index",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    documentId: Mapped[int] = mapped_column(
        ForeignKey("docs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    chunkIndex: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    pageNumber: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    embedding: Mapped[list[float]] = mapped_column(
        Vector(3072),
        nullable=False,
    )

    createdAt: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    document = relationship(
        "Document",
        back_populates="chunks",
    )

    sources = relationship(
        "MessageSource",
        back_populates="chunk",
    )
