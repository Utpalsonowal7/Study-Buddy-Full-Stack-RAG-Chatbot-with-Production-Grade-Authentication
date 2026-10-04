"""SSE framing and decoding shared by Gemini and the chat endpoint."""
import json
import asyncio
from collections.abc import AsyncIterator

import anyio
from fastapi import HTTPException


def encode_event(event: str, data: dict) -> str:
    # JSON escapes embedded newlines so text cannot inject additional SSE events.
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def decode_json_events(lines: AsyncIterator[str]) -> AsyncIterator[dict]:
    data = []
    size = 0
    async for line in lines:
        if line == "":
            if data:
                payload = "\n".join(data)
                data, size = [], 0
                if payload == "[DONE]":
                    return
                yield decode_payload(payload)
        elif line.startswith("data:"):
            value = line[5:]
            if value.startswith(" "):
                value = value[1:]
            data.append(value)
            size += len(value)
            if size > 1024 * 1024:
                raise HTTPException(502, "Gemini returned an oversized stream event.")
    if data:
        payload = "\n".join(data)
        if payload != "[DONE]":
            yield decode_payload(payload)


def decode_payload(payload: str) -> dict:
    try:
        result = json.loads(payload)
        if not isinstance(result, dict):
            raise ValueError("Invalid event")
        return result
    except (TypeError, ValueError) as exc:
        raise HTTPException(502, "Gemini returned an invalid stream event.") from exc


async def with_heartbeats(stream: AsyncIterator[str], request) -> AsyncIterator[str]:
    """Monitor disconnects even while Gemini is waiting to produce its next token."""
    pending = None
    last_send = asyncio.get_running_loop().time()
    try:
        while True:
            if pending is None:
                pending = asyncio.create_task(anext(stream))
            done, _ = await asyncio.wait({pending}, timeout=1)
            if await request.is_disconnected():
                return
            if done:
                try:
                    event = pending.result()
                except StopAsyncIteration:
                    return
                pending = None
                yield event
                last_send = asyncio.get_running_loop().time()
            elif asyncio.get_running_loop().time() - last_send >= 15:
                yield ": keep-alive\n\n"
                last_send = asyncio.get_running_loop().time()
    finally:
        with anyio.CancelScope(shield=True):
            if pending is not None:
                pending.cancel()
                try:
                    await pending
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass
            await stream.aclose()
