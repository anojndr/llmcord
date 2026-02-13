"""Core search service orchestration."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from llmcord.services.database import get_bad_keys_db
from llmcord.services.search.config import EXA_MCP_URL
from llmcord.services.search.exa import exa_search
from llmcord.services.search.tavily import tavily_search

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from llmcord.services.database import BadKeysDB


@dataclass(frozen=True, slots=True)
class WebSearchOptions:
    """Options for web search orchestration."""

    max_results_per_query: int = 5
    max_chars_per_url: int = 4000
    min_score: float = 0.3
    search_depth: str = "advanced"
    web_search_provider: str = "tavily"
    exa_mcp_url: str = EXA_MCP_URL


async def _search_single_query_tavily(
    query: str,
    depth: str,
    keys: list[str],
    max_results_per_query: int,
    db: BadKeysDB,
) -> dict:
    """Search with Tavily using retry logic and key rotation."""
    for key in keys:
        result = await tavily_search(query, key, max_results_per_query, depth)

        if "error" not in result:
            return result

        error_msg = result.get("error", "Unknown error")[:200]
        db.mark_key_bad_synced("tavily", key, error_msg)

    logger.error("All Tavily API keys failed for query '%s'", query)
    return {"error": "All API keys exhausted", "query": query}


async def _run_tavily_searches(
    queries: list[str],
    api_keys: list[str],
    search_depth: str,
    max_results_per_query: int,
    db: BadKeysDB,
) -> list[dict]:
    """Execute Tavily searches with key rotation."""
    good_keys = db.get_good_keys_synced("tavily", api_keys)
    if not good_keys:
        db.reset_provider_keys_synced("tavily")
        good_keys = api_keys.copy()

    search_tasks = [
        _search_single_query_tavily(
            query,
            search_depth,
            good_keys,
            max_results_per_query,
            db,
        )
        for query in queries
    ]
    return await asyncio.gather(*search_tasks)


async def _run_exa_searches(
    queries: list[str],
    exa_mcp_url: str,
    max_results_per_query: int,
) -> list[dict]:
    """Execute Exa MCP searches concurrently."""
    search_tasks = [
        exa_search(query, exa_mcp_url, max_results_per_query) for query in queries
    ]
    return await asyncio.gather(*search_tasks)


def _count_total_results(results: list[dict]) -> int:
    """Count total result items across all query responses."""
    total = 0
    for result in results:
        items = result.get("results", []) if isinstance(result, dict) else []
        if isinstance(items, list):
            total += len(items)
    return total


def _format_result_item(
    item: dict,
    min_score: float,
    max_chars_per_url: int,
) -> tuple[str, dict | None] | None:
    """Format a single search result item and return text and URL entry."""
    score = item.get("score")
    if score is not None and score < min_score:
        return None

    title = item.get("title", "No title")
    url = item.get("url", "")
    url_entry = {"title": title, "url": url, "score": score} if url else None

    raw_content = item.get("raw_content", "")
    content = item.get("content", "") or item.get("text", "")
    page_content = raw_content if raw_content else content
    page_content = page_content[:max_chars_per_url] if page_content else ""

    score_str = f" (relevance: {score:.2f})" if score is not None else ""
    result_text = f"\n**{title}**{score_str}\n{url}\n{page_content}\n"
    return result_text, url_entry


def _format_query_results(
    query: str,
    result: dict,
    min_score: float,
    max_chars_per_url: int,
    provider_name: str,
) -> tuple[list[str], list[dict]]:
    """Format results for a single query and collect URL metadata."""
    if "error" in result:
        logger.warning(
            "%s search error for query '%s': %s",
            provider_name,
            query,
            result.get("error"),
        )
        return [], []

    result_items = result.get("results", [])
    logger.info(
        "Query '%s' returned %s items",
        query,
        len(result_items),
    )

    query_results: list[str] = []
    urls: list[dict] = []
    for item in result_items:
        formatted = _format_result_item(item, min_score, max_chars_per_url)
        if not formatted:
            continue
        result_text, url_entry = formatted
        query_results.append(result_text)
        if url_entry:
            urls.append(url_entry)

    return query_results, urls


def _format_search_results(
    results: list[dict],
    queries: list[str],
    min_score: float,
    max_chars_per_url: int,
    provider_name: str,
) -> tuple[list[str], list[dict]]:
    """Format all search results and aggregate URLs."""
    formatted: list[str] = []
    urls: list[dict] = []

    logger.info(
        "%s search returned %s result sets for %s queries",
        provider_name,
        len(results),
        len(queries),
    )

    for query, result in zip(queries, results, strict=True):
        query_results, query_urls = _format_query_results(
            query,
            result,
            min_score,
            max_chars_per_url,
            provider_name,
        )
        if query_results:
            formatted.append(f"\n### Search Results for: {query}")
            formatted.extend(query_results)
        urls.extend(query_urls)

    logger.info("Total URLs collected: %s", len(urls))
    return formatted, urls


async def perform_web_search(
    queries: list[str],
    api_keys: list[str] | None = None,
    *,
    options: WebSearchOptions | None = None,
) -> tuple[str, dict]:
    """Perform concurrent web searches for multiple queries.

    Supports both Tavily and Exa MCP as search providers. Returns a tuple of
    (formatted_results, metadata).

    Best practices applied:
    - Concurrent requests with asyncio.gather()
        - KeyRotator for synced bad key tracking with database persistence
            (Tavily only)
        - Configurable search depth (Tavily: "basic", "advanced", "fast",
            "ultra-fast")

    Args:
        queries: List of search queries
        api_keys: List of API keys for rotation (required for Tavily, optional
            for Exa)
        options: Web search options for provider and formatting.

    Returns:
        tuple: (formatted_results_string, {"queries": [...], "urls": [{...},
            ...], "provider": "..."})

    """
    if not queries:
        return "", {}

    opts = options or WebSearchOptions()
    db = get_bad_keys_db()
    provider_name = opts.web_search_provider.capitalize()

    if opts.web_search_provider == "tavily":
        if not api_keys:
            logger.warning("Tavily requires API keys but none provided")
            return "", {}

        results = await _run_tavily_searches(
            queries,
            api_keys,
            opts.search_depth,
            opts.max_results_per_query,
            db,
        )
    else:
        results = await _run_exa_searches(
            queries,
            opts.exa_mcp_url,
            opts.max_results_per_query,
        )
        total_results = _count_total_results(results)
        retries_remaining = 3
        while total_results == 0 and retries_remaining > 0:
            logger.info(
                "Exa returned 0 results; retrying (%s remaining)",
                retries_remaining,
            )
            results = await _run_exa_searches(
                queries,
                opts.exa_mcp_url,
                opts.max_results_per_query,
            )
            total_results = _count_total_results(results)
            retries_remaining -= 1

        if total_results == 0 and api_keys:
            logger.info("Exa still returned 0 results; falling back to Tavily")
            provider_name = "Tavily"
            results = await _run_tavily_searches(
                queries,
                api_keys,
                opts.search_depth,
                opts.max_results_per_query,
                db,
            )
            opts = WebSearchOptions(
                max_results_per_query=opts.max_results_per_query,
                max_chars_per_url=opts.max_chars_per_url,
                min_score=opts.min_score,
                search_depth=opts.search_depth,
                web_search_provider="tavily",
                exa_mcp_url=opts.exa_mcp_url,
            )

    formatted_results, all_urls = _format_search_results(
        results,
        queries,
        opts.min_score,
        opts.max_chars_per_url,
        provider_name,
    )

    metadata = {
        "queries": queries,
        "urls": all_urls,
        "provider": opts.web_search_provider,
    }

    if formatted_results:
        results_text = "".join(formatted_results)
        return (
            (
                "\n\n---\nHere are the web search results in case the user "
                "asked you to search the net or something:\n\n"
                "**Web Search Results:**"
                f"{results_text}"
            ),
            metadata,
        )
    return "", metadata
