from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from llmcord import config as cfg


def test_is_gemini_model() -> None:
    assert cfg.is_gemini_model("gemini-3-flash-preview") is True
    assert cfg.is_gemini_model("Gemini-2.5-Pro") is True
    assert cfg.is_gemini_model("gemma-3-27b-it") is False
    assert cfg.is_gemini_model("gpt-4o") is False


@pytest.mark.asyncio
async def test_get_or_create_httpx_client_reuse_and_recreate() -> None:
    holder: list[httpx.AsyncClient | None] = []

    client1 = cfg.get_or_create_httpx_client(holder, headers={"X-Test": "1"})
    client2 = cfg.get_or_create_httpx_client(holder, headers={"X-Test": "2"})

    assert client1 is client2
    assert client1.headers["X-Test"] == "1"

    await client1.aclose()

    client3 = cfg.get_or_create_httpx_client(holder, headers={"X-Test": "3"})
    assert client3 is not client1
    assert client3.headers["X-Test"] == "3"

    await client3.aclose()

    holder_with_none: list[httpx.AsyncClient | None] = [None]
    client4 = cfg.get_or_create_httpx_client(holder_with_none)
    assert holder_with_none[0] is client4
    await client4.aclose()


def test_resolve_config_path_missing(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"
    with pytest.raises(cfg.ConfigFileNotFoundError) as excinfo:
        cfg._resolve_config_path(str(missing_path))
    assert str(missing_path) in str(excinfo.value)
    assert "Config file" in str(excinfo.value)


def test_resolve_config_path_existing(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("value: 1\n", encoding="utf-8")

    resolved = cfg._resolve_config_path(str(config_file))
    assert resolved == config_file


def test_get_config_cache_and_reload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg.clear_config_cache()

    config_file = tmp_path / "config.yaml"
    config_file.write_text("value: 1\n", encoding="utf-8")

    monkeypatch.setattr(cfg, "_resolve_config_path", lambda filename: config_file)

    now = 1000.0
    monkeypatch.setattr(cfg.time, "time", lambda: now)

    first = cfg.get_config("config.yaml")
    assert first["value"] == 1

    config_file.write_text("value: 2\n", encoding="utf-8")
    now += cfg.CONFIG_CACHE_TTL + 1

    second = cfg.get_config("config.yaml")
    assert second["value"] == 2


def test_get_config_empty_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg.clear_config_cache()

    config_file = tmp_path / "config.yaml"
    config_file.write_text("", encoding="utf-8")

    monkeypatch.setattr(cfg, "_resolve_config_path", lambda filename: config_file)
    monkeypatch.setattr(cfg.time, "time", lambda: 2000.0)

    with pytest.raises(cfg.ConfigFileEmptyError):
        cfg.get_config("config.yaml")


def test_config_error_messages(tmp_path: Path) -> None:
    not_found = cfg.ConfigFileNotFoundError("missing.yaml")
    assert "missing.yaml" in str(not_found)

    empty_error = cfg.ConfigFileEmptyError(tmp_path / "config.yaml")
    assert "corrupted" in str(empty_error)


def test_ensure_list() -> None:
    assert cfg.ensure_list(None) == []
    assert cfg.ensure_list("token") == ["token"]
    assert cfg.ensure_list(["a", "b"]) == ["a", "b"]
