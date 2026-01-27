"""Tests for web search helpers."""

# ruff: noqa: S101

from __future__ import annotations

import re

import pytest

import bad_keys
import web_search


def test_get_current_datetime_strings_format() -> None:
    """Validate current datetime string formats."""
    date_str, time_str = web_search.get_current_datetime_strings()
    assert re.match(r"^[A-Z][a-z]+ \d{2} \d{4}$", date_str)
    assert re.match(r"^\d{2}:\d{2}:\d{2} .+", time_str)


def test_convert_messages_to_openai_format() -> None:
    """Convert mixed message formats to OpenAI format."""
    messages = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "https://x"}},
                {"type": "audio", "audio": {"url": "https://y"}},
            ],
        },
    ]

    result = web_search.convert_messages_to_openai_format(
        messages,
        system_prompt="system",
        reverse=False,
        include_analysis_prompt=True,
    )

    assert result[0]["role"] == "system"
    assert result[-1]["role"] == "user"
    assert result[-1]["content"].startswith("Based on the conversation")
    assert result[2]["role"] == "assistant"
    expected_content_len = 2

    assert len(result[2]["content"]) == expected_content_len


def test_parse_exa_text_format() -> None:
    """Parse Exa text results into structured records."""
    text = (
        "Title: Result One\nURL: https://example.com/1\n"
        "Text: First content\n\n"
        "Title: Result Two\nURL: https://example.com/2\n"
        "Text: Second content"
    )

    results = web_search.parse_exa_text_format(text)
    assert results == [
        {
            "title": "Result One",
            "url": "https://example.com/1",
            "content": "First content",
        },
        {
            "title": "Result Two",
            "url": "https://example.com/2",
            "content": "Second content",
        },
    ]


@pytest.mark.asyncio
async def test_perform_web_search_tavily(monkeypatch: pytest.MonkeyPatch) -> None:
    """Handle Tavily search results and bad keys."""
    temp_db = bad_keys.BadKeysDB(local_db_path=":memory:")
    monkeypatch.setattr(web_search, "get_bad_keys_db", lambda: temp_db)

    async def fake_tavily_search(
        query: str,
        api_key: str,
        _max_results: int,
        _depth: str,
    ) -> dict:
        if api_key == "bad":
            return {"error": "bad key", "query": query}
        return {
            "results": [
                {
                    "title": "Title",
                    "url": "https://example.com",
                    "raw_content": "content",
                    "score": 0.9,
                },
            ],
            "query": query,
        }

    monkeypatch.setattr(web_search, "tavily_search", fake_tavily_search)

    formatted, metadata = await web_search.perform_web_search(
        queries=["query"],
        api_keys=["bad", "good"],
        web_search_provider="tavily",
        search_depth="basic",
    )

    assert "Title" in formatted
    assert metadata["provider"] == "tavily"
    assert metadata["queries"] == ["query"]
    assert metadata["urls"][0]["url"] == "https://example.com"
    assert temp_db.is_key_bad_synced("tavily", "bad") is True


@pytest.mark.asyncio
async def test_perform_web_search_exa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Handle Exa search results."""

    async def fake_exa_search(query: str, _exa_mcp_url: str, _max_results: int) -> dict:
        return {
            "results": [
                {
                    "title": "Exa Title",
                    "url": "https://exa.example.com",
                    "content": "exa content",
                },
            ],
            "query": query,
        }

    monkeypatch.setattr(web_search, "exa_search", fake_exa_search)

    formatted, metadata = await web_search.perform_web_search(
        queries=["query"],
        api_keys=None,
        web_search_provider="exa",
    )

    assert "Exa Title" in formatted
    assert metadata["provider"] == "exa"
