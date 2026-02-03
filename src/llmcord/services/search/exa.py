"""Exa MCP search provider integration."""
import json
import logging

import httpx

from llmcord.core.config import (
    HttpxClientOptions,
    get_config,
    get_or_create_httpx_client,
)
from llmcord.services.http import DEFAULT_RETRYABLE_STATUSES, wait_with_backoff
from llmcord.services.search.config import EXA_MCP_URL, HTTP_OK, MAX_ERROR_CHARS

logger = logging.getLogger(__name__)

# Shared httpx client for Exa MCP API calls - uses DRY factory pattern
_exa_client_holder: list = []


def _get_exa_client() -> httpx.AsyncClient:
    """Get or create the shared Exa MCP httpx client using the DRY factory pattern."""
    config = get_config()
    proxy_url = config.get("proxy_url") or None
    return get_or_create_httpx_client(
        _exa_client_holder,
        options=HttpxClientOptions(
            timeout=60.0,  # Exa can take longer for deep searches
            connect_timeout=15.0,
            max_connections=20,
            max_keepalive=10,
            follow_redirects=True,
            headers={},
            proxy_url=proxy_url,
        ),
    )


def _normalize_exa_block(block: str, index: int) -> str | None:
    """Normalize a text block so it starts with a Title line."""
    if index == 0:
        if not block.strip().startswith("Title:"):
            return None
        return block.strip()
    return "Title:" + block


def _parse_exa_block(block: str) -> dict | None:
    """Parse a single Exa text block into a result dict."""
    title = ""
    url = ""
    text_started = False
    text_lines: list[str] = []

    for line in block.split("\n"):
        if line.startswith("Title:"):
            title = line[6:].strip()
        elif line.startswith("URL:"):
            url = line[4:].strip()
        elif line.startswith("Text:"):
            text_started = True
            text_lines.append(line[5:].strip())
        elif text_started:
            text_lines.append(line)

    content = "\n".join(text_lines).strip()
    if not (title or url):
        return None

    return {
        "title": title or "Untitled",
        "url": url,
        "content": content,
    }


def parse_exa_text_format(text_content: str) -> list[dict]:
    """Parse Exa MCP's structured text response format into a list of result dicts.

    Exa returns results in this format (multiple results separated by blank lines):
    Title: ...
    Author: ...
    Published Date: ...
    URL: ...
    Text: ...

    Returns a list of dicts with 'title', 'url', and 'content' keys.
    """
    if not text_content:
        return []

    results: list[dict] = []

    # Split by double newlines to separate individual results
    # Each result block starts with "Title:"
    blocks = text_content.split("\n\nTitle:")

    for i, block in enumerate(blocks):
        current_block = _normalize_exa_block(block, i)
        if not current_block:
            continue
        parsed = _parse_exa_block(current_block)
        if parsed:
            results.append(parsed)

    return results


def _build_exa_payload(query: str, max_results: int) -> dict:
    """Build the JSON-RPC payload for Exa MCP tool calls."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "web_search_exa",
            "arguments": {
                "query": query,
                "numResults": max_results,
            },
        },
    }


async def _collect_sse_body(response: httpx.Response) -> str:
    """Collect SSE event data into a single JSON string."""
    full_response = ""
    current_event_data: list[str] = []

    async for raw_line in response.aiter_lines():
        line = raw_line.rstrip("\n").rstrip("\r")

        if not line:
            if current_event_data:
                event_body = "".join(current_event_data)
                logger.debug(
                    "Exa MCP SSE event body length: %s",
                    len(event_body),
                )
                if event_body.strip().startswith("{"):
                    full_response = event_body
                current_event_data = []
            continue

        if line.startswith("data:"):
            data = line[5:]
            current_event_data.append(data.removeprefix(" "))
            continue

        if line.startswith("event:"):
            event_type = line[6:].strip()
            logger.debug("Exa MCP SSE event: %s", event_type)

    if current_event_data and not full_response:
        full_response = "".join(current_event_data)

    return full_response


async def _parse_sse_response(response: httpx.Response, query: str) -> dict:
    """Parse SSE response body into JSON."""
    full_response = await _collect_sse_body(response)
    if not full_response:
        logger.warning(
            "Exa MCP returned empty SSE stream for query '%s'",
            query,
        )
        return {"results": [], "query": query}

    try:
        return json.loads(full_response)
    except json.JSONDecodeError as exc:
        logger.exception(
            "Exa MCP SSE JSON parse error. Data: %s",
            full_response[:MAX_ERROR_CHARS],
        )
        return {"error": f"JSON parse error: {exc}", "query": query}


async def _parse_json_response(response: httpx.Response, query: str) -> dict:
    """Parse JSON response body into a dict."""
    response_text = await response.aread()
    response_text = response_text.decode("utf-8")

    if not response_text:
        logger.warning(
            "Exa MCP returned empty response for query '%s'",
            query,
        )
        return {"results": [], "query": query}

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as exc:
        logger.exception(
            "Exa MCP JSON parse error. Response: %s",
            response_text[:MAX_ERROR_CHARS],
        )
        return {"error": f"JSON parse error: {exc}", "query": query}


async def _parse_exa_http_response(response: httpx.Response, query: str) -> dict:
    """Parse the HTTP response from Exa MCP into a JSON dict."""
    logger.info("Exa MCP response status: %s", response.status_code)

    if response.status_code != HTTP_OK:
        error_text = await response.aread()
        error_text = error_text.decode("utf-8", errors="replace")
        error_text = error_text[:MAX_ERROR_CHARS]
        logger.error(
            "Exa MCP HTTP error for query '%s': %s - %s",
            query,
            response.status_code,
            error_text,
        )
        return {
            "error": f"HTTP {response.status_code}: {error_text[:200]}",
            "query": query,
        }

    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        return await _parse_sse_response(response, query)

    return await _parse_json_response(response, query)


def _extract_exa_results(result: dict, query: str) -> dict:
    """Extract Exa results from the MCP JSON-RPC response."""
    if "error" in result:
        error = result.get("error")
        if isinstance(error, dict):
            error_msg = error.get("message", "Unknown MCP error")
        else:
            error_msg = str(error)
        logger.error("Exa MCP error for query '%s': %s", query, error_msg)
        return {"error": error_msg, "query": query}

    logger.debug(
        "Exa MCP full result keys: %s",
        result.keys() if isinstance(result, dict) else type(result),
    )

    mcp_result = result.get("result", {})
    logger.debug(
        "Exa MCP mcp_result keys: %s",
        mcp_result.keys() if isinstance(mcp_result, dict) else type(mcp_result),
    )

    content = mcp_result.get("content", [])
    logger.debug(
        "Exa MCP content type: %s, length: %s",
        type(content),
        len(content) if isinstance(content, list) else "N/A",
    )

    if not (content and isinstance(content, list) and len(content) > 0):
        logger.warning("Exa MCP returned empty content for query '%s'", query)
        return {"results": [], "query": query}

    first_content = content[0]
    first_content_keys = (
        first_content.keys()
        if isinstance(first_content, dict)
        else type(first_content)
    )
    logger.debug("Exa MCP first content item: %s", first_content_keys)

    if isinstance(first_content, dict):
        text_content = first_content.get("text", "")
    else:
        text_content = str(first_content)

    preview = text_content[:MAX_ERROR_CHARS] if text_content else "empty"
    logger.info("Exa MCP text_content preview: %s...", preview)

    try:
        search_data = json.loads(text_content) if text_content else {}
    except json.JSONDecodeError:
        logger.info(
            "Exa MCP returned text format, parsing structured text for query '%s'",
            query,
        )
        results = parse_exa_text_format(text_content)
        if results:
            logger.info("Exa MCP parsed %s results from text format", len(results))
            return {"results": results, "query": query}

        logger.warning(
            "Could not parse Exa text format, using as single result",
        )
        return {
            "results": [
                {
                    "title": "Search Result",
                    "url": "",
                    "content": text_content,
                },
            ],
            "query": query,
        }

    if isinstance(search_data, list):
        results = search_data
    else:
        results = search_data.get("results", [])

    logger.info(
        "Exa MCP response for query '%s': %s results",
        query,
        len(results),
    )
    return {"results": results, "query": query}


async def exa_search(
    query: str,
    exa_mcp_url: str = EXA_MCP_URL,
    max_results: int = 5,
) -> dict:
    """Execute a single Exa MCP web search query.

    Uses the Exa MCP HTTP endpoint for web search. Handles SSE (Server-Sent
    Events) streaming responses. Returns the search results with page content or
    an error dict.

    Args:
        query: Search query
        exa_mcp_url: The Exa MCP endpoint URL (default: https://mcp.exa.ai/mcp)
        max_results: Maximum results to return

    """
    client = _get_exa_client()
    payload = _build_exa_payload(query, max_results)

    logger.info(
        "Exa MCP request for query '%s': max_results=%s",
        query,
        max_results,
    )

    retries = 2
    for attempt in range(retries + 1):
        try:
            async with client.stream(
                "POST",
                exa_mcp_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                timeout=60.0,
            ) as response:
                if (
                    response.status_code in DEFAULT_RETRYABLE_STATUSES
                    and attempt < retries
                ):
                    logger.warning(
                        "Exa MCP transient HTTP %s for query '%s', retrying (%s/%s)",
                        response.status_code,
                        query,
                        attempt + 1,
                        retries,
                    )
                    await wait_with_backoff(attempt)
                    continue

                result = await _parse_exa_http_response(response, query)
                return _extract_exa_results(result, query)

        except httpx.TimeoutException as exc:
            if attempt < retries:
                logger.warning(
                    "Exa MCP timeout for query '%s', retrying (%s/%s): %s",
                    query,
                    attempt + 1,
                    retries,
                    exc,
                )
                await wait_with_backoff(attempt)
                continue
            logger.exception("Exa MCP timeout for query '%s'", query)
            return {"error": f"Timeout: {exc}", "query": query}
        except httpx.RequestError as exc:
            if attempt < retries:
                logger.warning(
                    "Exa MCP connection error for query '%s', retrying (%s/%s): %s",
                    query,
                    attempt + 1,
                    retries,
                    exc,
                )
                await wait_with_backoff(attempt)
                continue
            logger.exception("Exa MCP connection error for query '%s'", query)
            return {"error": f"Connection error: {exc}", "query": query}
        except Exception as exc:
            logger.exception("Exa MCP search error for query '%s'", query)
            return {"error": str(exc), "query": query}

    return {"error": "Retry attempts exhausted", "query": query}
