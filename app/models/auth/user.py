from enum import Enum
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SQLEnum,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Role(str, Enum):
    USER = "USER"
    ADMIN = "ADMIN"


class Provider(str, Enum):
    EMAIL = "EMAIL"
    GOOGLE = "GOOGLE"
    GITHUB = "GITHUB"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    name: Mapped[str | None] = mapped_column(String(100))

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )

    password: Mapped[str | None] = mapped_column(Text)

    is_email_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    provider: Mapped[Provider] = mapped_column(
        SQLEnum(Provider, name="provider"),
        default=Provider.EMAIL,
        nullable=False,
    )

    provider_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    role: Mapped[Role] = mapped_column(
        SQLEnum(Role, name="role"),
        default=Role.USER,
        nullable=False,
    )

    avatar: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    sessions: Mapped[list["Session"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint(
            "provider",
            "provider_id",
            name="uq_user_provider_provider_id",
        ),
    )
