from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.db.database import engine, Base

from app.models.auth.session import Session
from app.models.auth.user import User


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return {"hello world"}

