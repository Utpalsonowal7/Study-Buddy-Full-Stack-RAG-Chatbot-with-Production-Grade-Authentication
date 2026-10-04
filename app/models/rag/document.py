from datetime import datetime
from enum import Enum

from sqlalchemy import (
    String,
    Integer,
    BigInteger,
    DateTime,
    ForeignKey,
    Enum as SQLEnum,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class DocumentStatus(str, Enum):
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"


class Document(Base):
    __tablename__ = "docs"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    userId: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    originalName: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    fileUrl: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    cloudinaryPublicId: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    resourceType: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="raw",
    )

    fileType: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    fileSize: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    pageCount: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    status: Mapped[DocumentStatus] = mapped_column(
        SQLEnum(DocumentStatus, name="document_status"),
        default=DocumentStatus.PROCESSING,
        nullable=False,
    )

    embeddingModel: Mapped[str] = mapped_column(
        String(255), nullable=False, server_default="legacy_unknown",
    )

    chunkCount: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0",
    )

    createdAt: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updatedAt: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )
