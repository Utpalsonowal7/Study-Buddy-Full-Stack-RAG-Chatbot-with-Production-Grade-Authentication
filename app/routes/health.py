import asyncio

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.redis import redis
from app.db.database import engine

router = APIRouter(tags=["Health"])


async def database_ready():
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))


async def check(operation):
    try:
        async with asyncio.timeout(3):
            await operation()
        return "ok"
    except Exception:
        # Do not expose connection URLs, credentials, or provider error details.
        return "unavailable"


@router.get("/health", responses={503: {"description": "Required service unavailable"}})
async def health():
    database, cache = await asyncio.gather(check(database_ready), check(redis.ping))
    ready = database == cache == "ok"
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ok" if ready else "degraded",
                 "checks": {"database": database, "redis": cache}},
        headers={"Cache-Control": "no-store"},
    )
