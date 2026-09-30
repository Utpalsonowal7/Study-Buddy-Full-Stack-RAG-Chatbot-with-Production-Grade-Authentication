from fastapi import Depends, Request
from app.core.rate_limit import rate_limit
from app.utils.client import get_client_ip


def rate_limit_dependency(
    limit: int,
    window: int,
):
    async def dependency(request: Request):

        client_ip = get_client_ip(request)

        await rate_limit(
            key=f"rate_limit:{request.url.path}:{client_ip}",
            limit=limit,
            window=window,
        )

    return dependency
