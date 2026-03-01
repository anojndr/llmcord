from __future__ import annotations

import pytest

from llmcord.services.search.core import (
    WebSearchOptions,
    _deduplicate_results,
    perform_web_search,
)


@pytest.mark.asyncio
async def test_exa_error_falls_back_to_tavily_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exa_calls = 0
    tavily_calls = 0

    async def _fake_run_exa_searches(
        _queries: list[str],
        _exa_mcp_url: str,
        _max_results_per_query: int,
        _tool: str = "web_search_exa",
    ) -> list[dict]:
        nonlocal exa_calls
        exa_calls += 1
        return [{"error": "Exa MCP unavailable", "query": "latest news"}]

    async def _fake_run_tavily_searches(
        _queries: list[str],
        _api_keys: list[str],
        _search_depth: str,
        _max_results_per_query: int,
    ) -> list[dict]:
        nonlocal tavily_calls
        tavily_calls += 1
        return [
            {
                "results": [
                    {
                        "title": "Headline",
                        "url": "https://example.com",
                        "content": "Result body",
                    },
                ],
                "query": "latest news",
            },
        ]

    monkeypatch.setattr(
        "llmcord.services.search.core._run_exa_searches",
        _fake_run_exa_searches,
    )
    monkeypatch.setattr(
        "llmcord.services.search.core._run_tavily_searches",
        _fake_run_tavily_searches,
    )

    _, metadata = await perform_web_search(
        ["latest news"],
        api_keys=["tvly-key"],
        options=WebSearchOptions(web_search_provider="exa"),
    )

    assert exa_calls == 1
    assert tavily_calls == 1
    assert metadata["provider"] == "tavily"


@pytest.mark.asyncio
async def test_exa_empty_results_fall_back_to_tavily_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exa_calls = 0
    tavily_calls = 0

    async def _fake_run_exa_searches(
        _queries: list[str],
        _exa_mcp_url: str,
        _max_results_per_query: int,
        _tool: str = "web_search_exa",
    ) -> list[dict]:
        nonlocal exa_calls
        exa_calls += 1
        return [{"results": [], "query": "latest news"}]

    async def _fake_run_tavily_searches(
        _queries: list[str],
        _api_keys: list[str],
        _search_depth: str,
        _max_results_per_query: int,
    ) -> list[dict]:
        nonlocal tavily_calls
        tavily_calls += 1
        return [
            {
                "results": [
                    {
                        "title": "Headline",
                        "url": "https://example.com",
                        "content": "Result body",
                    },
                ],
                "query": "latest news",
            },
        ]

    monkeypatch.setattr(
        "llmcord.services.search.core._run_exa_searches",
        _fake_run_exa_searches,
    )
    monkeypatch.setattr(
        "llmcord.services.search.core._run_tavily_searches",
        _fake_run_tavily_searches,
    )

    _, metadata = await perform_web_search(
        ["latest news"],
        api_keys=["tvly-key"],
        options=WebSearchOptions(web_search_provider="exa"),
    )

    assert exa_calls == 1
    assert tavily_calls == 1
    assert metadata["provider"] == "tavily"


def test_deduplicate_results_removes_duplicate_urls() -> None:
    """Duplicate URLs across queries are removed, keeping the first."""
    results = [
        {
            "results": [
                {"title": "A", "url": "https://a.com", "content": "aaa"},
                {"title": "B", "url": "https://b.com", "content": "bbb"},
            ],
            "query": "q1",
        },
        {
            "results": [
                {"title": "A dup", "url": "https://a.com", "content": "aaa2"},
                {"title": "C", "url": "https://c.com", "content": "ccc"},
            ],
            "query": "q2",
        },
    ]
    deduped = _deduplicate_results(results, ["q1", "q2"])

    assert len(deduped[0]["results"]) == 2
    assert len(deduped[1]["results"]) == 1
    assert deduped[1]["results"][0]["url"] == "https://c.com"


def test_deduplicate_results_preserves_error_dicts() -> None:
    """Error dicts pass through untouched."""
    results: list[dict] = [
        {"error": "boom", "query": "q1"},
        {
            "results": [
                {"title": "A", "url": "https://a.com", "content": "a"},
            ],
            "query": "q2",
        },
    ]
    deduped = _deduplicate_results(results, ["q1", "q2"])

    assert "error" in deduped[0]
    assert len(deduped[1]["results"]) == 1


def test_deduplicate_results_keeps_items_without_url() -> None:
    """Items with no URL are never considered duplicates."""
    results = [
        {
            "results": [
                {"title": "No URL 1", "url": "", "content": "x"},
                {"title": "No URL 2", "url": "", "content": "y"},
            ],
            "query": "q1",
        },
    ]
    deduped = _deduplicate_results(results, ["q1"])

    assert len(deduped[0]["results"]) == 2


def test_deduplicate_results_no_duplicates_unchanged() -> None:
    """When there are no duplicates, all results are preserved."""
    results = [
        {
            "results": [
                {"title": "A", "url": "https://a.com", "content": "a"},
            ],
            "query": "q1",
        },
        {
            "results": [
                {"title": "B", "url": "https://b.com", "content": "b"},
            ],
            "query": "q2",
        },
    ]
    deduped = _deduplicate_results(results, ["q1", "q2"])

    assert len(deduped[0]["results"]) == 1
    assert len(deduped[1]["results"]) == 1


@pytest.mark.asyncio
async def test_perform_web_search_deduplicates_exa_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: duplicate URLs across Exa queries are deduplicated."""

    async def _fake_run_exa_searches(
        _queries: list[str],
        _exa_mcp_url: str,
        _max_results_per_query: int,
        _tool: str = "web_search_exa",
    ) -> list[dict]:
        return [
            {
                "results": [
                    {
                        "title": "Shared",
                        "url": "https://shared.com",
                        "content": "body",
                    },
                    {
                        "title": "Only Q1",
                        "url": "https://only-q1.com",
                        "content": "q1",
                    },
                ],
                "query": "query1",
            },
            {
                "results": [
                    {
                        "title": "Shared dup",
                        "url": "https://shared.com",
                        "content": "body2",
                    },
                    {
                        "title": "Only Q2",
                        "url": "https://only-q2.com",
                        "content": "q2",
                    },
                ],
                "query": "query2",
            },
        ]

    monkeypatch.setattr(
        "llmcord.services.search.core._run_exa_searches",
        _fake_run_exa_searches,
    )

    _text, metadata = await perform_web_search(
        ["query1", "query2"],
        options=WebSearchOptions(web_search_provider="exa"),
    )

    urls = [u["url"] for u in metadata["urls"]]
    assert urls.count("https://shared.com") == 1
    assert "https://only-q1.com" in urls
    assert "https://only-q2.com" in urls
    assert len(urls) == 3


@pytest.mark.asyncio
async def test_perform_web_search_deduplicates_tavily_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: duplicate URLs across Tavily queries are deduplicated."""

    async def _fake_run_tavily_searches(
        _queries: list[str],
        _api_keys: list[str],
        _search_depth: str,
        _max_results_per_query: int,
    ) -> list[dict]:
        return [
            {
                "results": [
                    {
                        "title": "Shared Page",
                        "url": "https://shared.com/article",
                        "content": "first hit",
                        "score": 0.95,
                    },
                    {
                        "title": "Only Q1",
                        "url": "https://only-q1.com",
                        "content": "q1 content",
                        "score": 0.8,
                    },
                ],
                "query": "tavily query1",
            },
            {
                "results": [
                    {
                        "title": "Shared Page Again",
                        "url": "https://shared.com/article",
                        "content": "duplicate hit",
                        "score": 0.9,
                    },
                    {
                        "title": "Only Q2",
                        "url": "https://only-q2.com",
                        "content": "q2 content",
                        "score": 0.85,
                    },
                ],
                "query": "tavily query2",
            },
        ]

    monkeypatch.setattr(
        "llmcord.services.search.core._run_tavily_searches",
        _fake_run_tavily_searches,
    )

    _text, metadata = await perform_web_search(
        ["tavily query1", "tavily query2"],
        api_keys=["tvly-key"],
        options=WebSearchOptions(web_search_provider="tavily"),
    )

    urls = [u["url"] for u in metadata["urls"]]
    assert urls.count("https://shared.com/article") == 1
    assert "https://only-q1.com" in urls
    assert "https://only-q2.com" in urls
    assert len(urls) == 3
