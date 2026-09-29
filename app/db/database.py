from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.config import DB

engine = create_async_engine(
    DB,
    connect_args={"ssl": "require"},
)

LocalSession = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
     pass
