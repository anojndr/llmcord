"""Core search service orchestration."""
import asyncio
import logging

from llmcord.services.database import get_bad_keys_db
from llmcord.services.search.config import EXA_MCP_URL
from llmcord.services.search.exa import exa_search
from llmcord.services.search.tavily import tavily_search

logger = logging.getLogger(__name__)


async def perform_web_search(  # noqa: C901, PLR0913, PLR0915
    queries: list[str],
    api_keys: list[str] | None = None,
    max_results_per_query: int = 5,
    max_chars_per_url: int = 2000,
    min_score: float = 0.3,
    search_depth: str = "advanced",
    web_search_provider: str = "tavily",
    exa_mcp_url: str = EXA_MCP_URL,
) -> tuple[str, dict]:
    """Perform concurrent web searches for multiple queries.

    Supports both Tavily and Exa MCP as search providers. Returns a tuple of
    (formatted_results, metadata).

    Best practices applied:
    - Concurrent requests with asyncio.gather()
        - KeyRotator for synced bad key tracking with database persistence
            (Tavily only)
    - Configurable search depth (Tavily: "basic", "advanced", "fast", "ultra-fast")

    Args:
        queries: List of search queries
        api_keys: List of API keys for rotation (required for Tavily, optional
            for Exa)
        max_results_per_query: Maximum number of URLs per query (default: 5)
        max_chars_per_url: Maximum characters per URL content (default: 2000)
        min_score: Minimum relevance score to include a result (0.0-1.0, default:
            0.3)
        search_depth: Tavily search depth - "basic", "advanced", "fast", or
            "ultra-fast" (default: "advanced")
        web_search_provider: Which provider to use - "tavily" or "exa" (default:
            "tavily")
        exa_mcp_url: The Exa MCP endpoint URL (default: https://mcp.exa.ai/mcp)

    Returns:
        tuple: (formatted_results_string, {"queries": [...], "urls": [{...}, ...],
            "provider": "..."})

    """
    if not queries:
        return "", {}

    # Validate provider and requirements
    if web_search_provider == "tavily" and not api_keys:
        logger.warning("Tavily requires API keys but none provided")
        return "", {}

    db = get_bad_keys_db()

    # Provider-specific search functions
    async def search_single_query_tavily(
        query: str,
        depth: str,
        keys: list[str],
    ) -> dict:
        """Search with Tavily using retry logic and key rotation."""
        for key in keys:
            result = await tavily_search(query, key, max_results_per_query, depth)

            if "error" not in result:
                return result

            # Mark the key as bad
            error_msg = result.get("error", "Unknown error")[:200]
            db.mark_key_bad_synced("tavily", key, error_msg)

        # All keys failed
        logger.error("All Tavily API keys failed for query '%s'", query)
        return {"error": "All API keys exhausted", "query": query}

    async def search_single_query_exa(query: str) -> dict:
        """Search with Exa MCP."""
        return await exa_search(query, exa_mcp_url, max_results_per_query)

    async def execute_searches() -> tuple[list, list]:
        """Execute searches with the configured provider and return results and URLs."""
        if web_search_provider == "tavily":
            # Pre-fetch good keys for Tavily
            good_keys = db.get_good_keys_synced("tavily", api_keys)
            if not good_keys:
                db.reset_provider_keys_synced("tavily")
                good_keys = api_keys.copy()

            search_tasks = [
                search_single_query_tavily(query, search_depth, good_keys)
                for query in queries
            ]
        else:  # exa
            search_tasks = [
                search_single_query_exa(query)
                for query in queries
            ]

        results = await asyncio.gather(*search_tasks)

        formatted = []
        urls = []

        provider_name = web_search_provider.capitalize()
        logger.info(
            "%s search returned %s result sets for %s queries",
            provider_name,
            len(results),
            len(queries),
        )

        for i, result in enumerate(results):
            if "error" in result:
                logger.warning(
                    "%s search error for query '%s': %s",
                    provider_name,
                    queries[i],
                    result.get("error"),
                )
                continue

            query = queries[i]
            query_results = []

            result_items = result.get("results", [])
            logger.info(
                "Query '%s' returned %s items",
                query,
                len(result_items),
            )

            for item in result_items:
                # Handle both Tavily and Exa result formats
                score = item.get("score")
                if score is not None and score < min_score:
                    continue

                title = item.get("title", "No title")
                url = item.get("url", "")

                if url:
                    urls.append({"title": title, "url": url, "score": score})

                # Prefer raw_content (full page content) over content (snippet)
                # Tavily uses raw_content, Exa might use text or content
                raw_content = item.get("raw_content", "")
                content = item.get("content", "") or item.get("text", "")

                page_content = raw_content if raw_content else content
                page_content = (
                    page_content[:max_chars_per_url]
                    if page_content
                    else ""
                )

                # Format score display - handle missing scores gracefully
                score_str = f" (relevance: {score:.2f})" if score else ""
                result_text = f"\n**{title}**{score_str}\n{url}\n{page_content}\n"
                query_results.append(result_text)

            if query_results:
                formatted.append(f"\n### Search Results for: {query}")
                formatted.extend(query_results)

        logger.info("Total URLs collected: %s", len(urls))
        return formatted, urls

    # Execute the searches
    formatted_results, all_urls = await execute_searches()

    metadata = {
        "queries": queries,
        "urls": all_urls,
        "provider": web_search_provider,
    }

    if formatted_results:
        return (
            "\n\n---\nHere are the web search results in case the user asked you "
            "to search the net or something:\n\n**Web Search Results:**"
            + "".join(formatted_results),
            metadata,
        )
    return "", metadata
