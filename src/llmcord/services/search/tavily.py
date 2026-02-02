"""Tavily search provider integration."""
import asyncio
import importlib
import json
import logging
import time

import httpx

from llmcord.core.config import HttpxClientOptions, get_or_create_httpx_client
from llmcord.services.database import get_bad_keys_db
from llmcord.services.search.config import MAX_ERROR_CHARS, MAX_LOG_CHARS

logger = logging.getLogger(__name__)


# Shared httpx client for Tavily API calls - uses DRY factory pattern
_tavily_client_holder: list = []


def _get_tavily_client() -> httpx.AsyncClient:
    """Get or create the shared Tavily httpx client using the DRY factory pattern."""
    return get_or_create_httpx_client(
        _tavily_client_holder,
        options=HttpxClientOptions(
            timeout=30.0,
            connect_timeout=10.0,
            max_connections=20,
            max_keepalive=10,
            follow_redirects=True,
            headers={},  # Tavily doesn't need browser headers, just defaults
        ),
    )


def _get_client_from_package() -> httpx.AsyncClient:
    search_module = importlib.import_module("llmcord.services.search")
    get_client = search_module.get_tavily_client
    return get_client()


async def tavily_search(
    query: str,
    tavily_api_key: str,
    max_results: int = 5,
    search_depth: str = "basic",
) -> dict:
    """Execute a single Tavily search query.

    Returns the search results with page content or an error dict.

    Best practices applied:
    - search_depth configurable (basic/advanced/fast/ultra-fast)
    - Reuses shared httpx client for connection pooling

    Args:
        query: Search query (keep under 400 characters)
        tavily_api_key: Tavily API key
        max_results: Maximum results to return (1-20)
        search_depth: "basic", "advanced", "fast", or "ultra-fast"

    """
    try:
        client = _get_client_from_package()

        # Build request payload
        payload = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": False,
            "include_raw_content": "markdown",
        }

        # Increase timeout for advanced depth which takes longer
        timeout = 45.0 if search_depth == "advanced" else 30.0

        logger.info(
            "Tavily API request for query '%s': depth=%s, max_results=%s",
            query,
            search_depth,
            max_results,
        )

        response = await client.post(
            "https://api.tavily.com/search",
            json=payload,
            headers={
                "Authorization": f"Bearer {tavily_api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
        logger.info("Tavily API response status: %s", response.status_code)

        # Log raw response for debugging (first 1000 chars)
        raw_text = (
            response.text[:MAX_LOG_CHARS]
            if len(response.text) > MAX_LOG_CHARS
            else response.text
        )
        logger.debug("Tavily API raw response: %s", raw_text)

        response.raise_for_status()
        result = response.json()
        logger.info(
            "Tavily API response for query '%s': %s results",
            query,
            len(result.get("results", [])),
        )
        if not result.get("results"):
            logger.warning(
                "Tavily returned empty results for query '%s'. Full response: %s",
                query,
                result,
            )
        else:
            return result
    except httpx.HTTPStatusError as exc:
        logger.exception(
            "Tavily HTTP error for query '%s': %s - %s",
            query,
            exc.response.status_code,
            exc.response.text[:MAX_ERROR_CHARS],
        )
        return {
            "error": f"HTTP {exc.response.status_code}: {exc.response.text[:200]}",
            "query": query,
        }
    except httpx.TimeoutException as exc:
        logger.exception("Tavily timeout for query '%s'", query)
        return {"error": f"Timeout: {exc}", "query": query}
    except httpx.RequestError as exc:
        logger.exception("Tavily connection error for query '%s'", query)
        return {"error": f"Connection error: {exc}", "query": query}
    except Exception as exc:
        logger.exception("Tavily search error for query '%s'", query)
        return {"error": str(exc), "query": query}


async def tavily_research_create(
    input_text: str,
    model: str,
    tavily_api_key: str,
) -> dict:
    """Create a Tavily research task and return the API response.

    Args:
        input_text: The research task or question.
        model: Tavily research model ("mini", "pro", or "auto").
        tavily_api_key: Tavily API key.

    Returns:
        Response dict or error dict with "error" key.

    """
    try:
        client = _get_client_from_package()
        payload = {
            "input": input_text,
            "model": model,
            "stream": False,
        }

        response = await client.post(
            "https://api.tavily.com/research",
            json=payload,
            headers={
                "Authorization": f"Bearer {tavily_api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
    except httpx.TimeoutException as exc:
        logger.exception("Tavily research create timeout")
        return {"error": f"Timeout: {exc}"}
    except httpx.RequestError as exc:
        logger.exception("Tavily research create connection error")
        return {"error": f"Connection error: {exc}"}
    except Exception as exc:
        logger.exception("Tavily research create error")
        return {"error": str(exc)}
    else:
        if response.status_code in (200, 201):
            return response.json()

        error_detail = response.text[:MAX_ERROR_CHARS]
        logger.warning(
            "Tavily research create error: %s - %s",
            response.status_code,
            error_detail,
        )
        return {
            "error": f"HTTP {response.status_code}",
            "status_code": response.status_code,
            "detail": error_detail,
        }


async def tavily_research_get(
    request_id: str,
    tavily_api_key: str,
) -> dict:
    """Get Tavily research task status or results.

    Args:
        request_id: Tavily research request ID.
        tavily_api_key: Tavily API key.

    Returns:
        Response dict or error dict with "error" key.

    """
    try:
        client = _get_tavily_client()
        response = await client.get(
            f"https://api.tavily.com/research/{request_id}",
            headers={
                "Authorization": f"Bearer {tavily_api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
    except httpx.TimeoutException as exc:
        logger.exception("Tavily research get timeout")
        return {"error": f"Timeout: {exc}"}
    except httpx.RequestError as exc:
        logger.exception("Tavily research get connection error")
        return {"error": f"Connection error: {exc}"}
    except Exception as exc:
        logger.exception("Tavily research get error")
        return {"error": str(exc)}
    else:
        if response.status_code in (200, 202):
            return response.json()

        error_detail = response.text[:MAX_ERROR_CHARS]
        logger.warning(
            "Tavily research get error: %s - %s",
            response.status_code,
            error_detail,
        )
        return {
            "error": f"HTTP {response.status_code}",
            "status_code": response.status_code,
            "detail": error_detail,
        }


def _format_research_content(content: object) -> str:
    """Format Tavily research content for prompt injection."""
    if isinstance(content, str):
        return content.strip()
    try:
        return json.dumps(content, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(content)


def _extract_research_urls(sources: list[object]) -> list[dict[str, str]]:
    """Extract normalized URL metadata from Tavily research sources."""
    urls: list[dict[str, str]] = []
    for source in sources:
        if not isinstance(source, dict) or not source.get("url"):
            continue
        urls.append(
            {
                "title": source.get("title", "No title"),
                "url": source.get("url", ""),
                "favicon": source.get("favicon", ""),
            },
        )
    return urls


def _build_research_metadata(
    query: str,
    model: str,
    request_id: str,
    sources: list[object],
) -> dict:
    """Build Tavily research metadata payload for source views."""
    return {
        "queries": [query],
        "urls": _extract_research_urls(sources),
        "provider": "tavily",
        "mode": "research",
        "model": model,
        "request_id": request_id,
    }


def _handle_research_status(
    status_result: dict,
    query: str,
    model: str,
    request_id: str,
) -> tuple[str, dict] | None:
    """Process a completed research response into output and metadata."""
    status = status_result.get("status")
    if status == "failed":
        logger.warning("Tavily research failed for request %s", request_id)
        return None
    if status != "completed":
        logger.warning("Unexpected Tavily research status: %s", status)
        return None

    content = _format_research_content(status_result.get("content", ""))
    sources = status_result.get("sources", [])
    metadata = _build_research_metadata(query, model, request_id, sources)

    if content:
        formatted = "\n\n---\nHere is the Tavily research report:\n\n"
        return formatted + content, metadata

    return "", metadata


async def _poll_tavily_research(
    request_id: str,
    api_key: str,
    poll_interval: float,
    timeout_seconds: float,
) -> dict:
    """Poll Tavily research status until completion or timeout."""
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout_seconds:
        status_result = await tavily_research_get(request_id, api_key)
        if "error" in status_result:
            return status_result

        status = status_result.get("status")
        if status in ("pending", "in_progress"):
            await asyncio.sleep(poll_interval)
            continue
        return status_result

    return {"error": "timeout"}


async def perform_tavily_research(
    query: str,
    api_keys: list[str] | None,
    model: str,
    poll_interval: float = 1.0,
    timeout_seconds: float = 3600.0,
) -> tuple[str, dict]:
    """Perform a Tavily research task and return formatted content + metadata.

    Args:
        query: The research task or question.
        api_keys: List of Tavily API keys for rotation.
        model: Tavily research model ("mini" or "pro").
        poll_interval: Seconds between status polls.
        timeout_seconds: Maximum seconds to wait for completion.

    Returns:
        tuple: (formatted_report, metadata)

    """
    if not query or not api_keys:
        return "", {}

    def _build_exhausted_metadata() -> dict:
        return {
            "queries": [query],
            "provider": "tavily",
            "mode": "research",
            "model": model,
            "keys_exhausted": True,
        }

    db = get_bad_keys_db()
    good_keys = db.get_good_keys_synced("tavily", api_keys)
    if not good_keys:
        db.reset_provider_keys_synced("tavily")
        good_keys = api_keys.copy()

    for key in good_keys:
        create_result = await tavily_research_create(query, model, key)
        if "error" in create_result:
            error_msg = str(
                create_result.get("detail") or create_result["error"],
            )[:200]
            db.mark_key_bad_synced("tavily", key, error_msg)
            continue

        request_id = create_result.get("request_id")
        if not request_id:
            logger.warning(
                "Tavily research create missing request_id: %s",
                create_result,
            )
            continue

        status_result = await _poll_tavily_research(
            str(request_id),
            key,
            poll_interval,
            timeout_seconds,
        )

        if "error" in status_result:
            if status_result.get("status_code"):
                error_msg = str(
                    status_result.get("detail") or status_result["error"],
                )[:200]
                db.mark_key_bad_synced("tavily", key, error_msg)
            else:
                logger.warning(
                    "Tavily research error for request %s: %s",
                    request_id,
                    status_result.get("error"),
                )
            continue

        handled = _handle_research_status(
            status_result,
            query,
            model,
            str(request_id),
        )
        if handled is not None:
            return handled

    return "", _build_exhausted_metadata()
