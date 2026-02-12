from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def httpx_client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as client:
        yield client


@pytest.fixture
def msg_nodes() -> dict[int, object]:
    return {}


@pytest.fixture
def curr_model_lock() -> asyncio.Lock:
    return asyncio.Lock()
