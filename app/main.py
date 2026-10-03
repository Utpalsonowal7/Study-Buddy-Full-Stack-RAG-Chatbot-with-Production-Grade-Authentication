from fastapi import FastAPI, APIRouter
from contextlib import asynccontextmanager

from app.db.database import engine, Base

from app.models.auth.session import Session
from app.models.auth.user import User


from app.routes.auth import router as auth_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield


app = FastAPI(lifespan=lifespan)


api_roter = APIRouter(prefix="/api/v1")

api_roter.include_router(auth_router)


@app.get("/")
def root():
    return {"hello world"}

