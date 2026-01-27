"""Tests for config helpers."""

# ruff: noqa: S101, FBT001

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import pathlib

    import httpx

import config


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gemini-3-flash-preview", True),
        ("gemma-3-27b-it", False),
        ("GEMINI-PRO", True),
    ],
)
def test_is_gemini_model(model: str, expected: bool) -> None:
    """Validate Gemini model detection."""
    assert config.is_gemini_model(model) is expected


def test_ensure_list() -> None:
    """Ensure list normalization behaves as expected."""
    assert config.ensure_list(None) == []
    assert config.ensure_list("key123") == ["key123"]
    assert config.ensure_list(["key1", "key2"]) == ["key1", "key2"]


@pytest.mark.asyncio
async def test_get_or_create_httpx_client_reuses_instance() -> None:
    """Ensure the HTTPX client instance is reused and configured."""
    holder: list[httpx.AsyncClient | None] = []
    client1 = config.get_or_create_httpx_client(holder, headers={"X-Test": "1"})
    client2 = config.get_or_create_httpx_client(holder)

    try:
        assert client1 is client2
        assert client1.headers.get("X-Test") == "1"
        assert "User-Agent" in client1.headers
    finally:
        await client1.aclose()


def test_get_config_cache_reload(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reload configuration when mtime changes."""
    config.clear_config_cache()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("key: value\n", encoding="utf-8")

    times = [1000.0, 1007.0]

    def fake_time() -> float:
        return times.pop(0)

    monkeypatch.setattr(time, "time", fake_time)

    first = config.get_config(str(config_path))
    assert first["key"] == "value"

    config_path.write_text("key: new_value\n", encoding="utf-8")
    new_mtime = config_path.stat().st_mtime + 10
    os.utime(config_path, (new_mtime, new_mtime))
    second = config.get_config(str(config_path))
    assert second["key"] == "new_value"
