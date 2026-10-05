from fastapi import FastAPI, APIRouter
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.db.database import engine, Base

from app.models.auth.session import Session
from app.models.auth.user import User
import app.models.rag



from app.routes.auth import router as auth_router
from app.routes.rag import router as rag_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    yield


app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:3000",
    "https://yourdomain.com",
    "http://127.0.0.1:5500",
    "http://localhost:5173",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://thu-phi.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


api_roter = APIRouter(prefix="/api/v1")

api_roter.include_router(auth_router)
api_roter.include_router(rag_router)

app.include_router(api_roter)


@app.get("/")
def root():
    return {"hello world"}
