import time
import uuid

from fastapi import HTTPException, status

from app.core.redis import redis


async def rate_limit(
    key: str,
    limit: int,
    window: int,
    message: str = "Rate limit exceeded. Try again later.",
):
    now = time.time()
    window_start = now - window

    await redis.zremrangebyscore(
        key,
        0,
        window_start,
    )

    count = await redis.zcard(key)

    if count >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=message,
        )

    await redis.zadd(
        key,
        {str(uuid.uuid4()): now},
    )

    await redis.expire(key, window)
