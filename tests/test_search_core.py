from __future__ import annotations

import pytest

from llmcord.services.search.core import WebSearchOptions, perform_web_search


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
