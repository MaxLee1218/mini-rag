from __future__ import annotations

import asyncio
from typing import Any

import httpx


def asgi_request(app: Any, method: str, path: str, **kwargs: Any) -> httpx.Response:
    """Send one request without relying on the unavailable sync thread portal."""

    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())
