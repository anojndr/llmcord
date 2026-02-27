from __future__ import annotations

import httpx
import pytest

from llmcord.services import http as http_mod


@pytest.mark.asyncio
async def test_wait_before_retry_caps_retry_after_to_default_max_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_delays: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        observed_delays.append(delay)

    monkeypatch.setattr(http_mod.asyncio, "sleep", _fake_sleep)

    response = httpx.Response(
        http_mod.HTTP_TOO_MANY_REQUESTS,
        headers={"retry-after": "120"},
    )
    await http_mod.wait_before_retry(
        0,
        response=response,
    )

    assert observed_delays == [10.0]


@pytest.mark.asyncio
async def test_wait_before_retry_ignores_non_finite_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_delays: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        observed_delays.append(delay)

    class _FakeRandom:
        @staticmethod
        def random() -> float:
            return 0.0

    monkeypatch.setattr(http_mod.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(http_mod, "_JITTER_RANDOM", _FakeRandom())

    response = httpx.Response(
        http_mod.HTTP_TOO_MANY_REQUESTS,
        headers={"retry-after": "inf"},
    )
    await http_mod.wait_before_retry(
        0,
        response=response,
        max_backoff_seconds=30.0,
    )

    assert observed_delays == [2.0]
