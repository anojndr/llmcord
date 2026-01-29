"""Exa MCP search provider integration."""
import json
import logging

import httpx

from llmcord.config import get_or_create_httpx_client
from llmcord.services.search.config import EXA_MCP_URL, HTTP_OK, MAX_ERROR_CHARS

logger = logging.getLogger(__name__)

# Shared httpx client for Exa MCP API calls - uses DRY factory pattern
_exa_client_holder: list = []


def _get_exa_client() -> httpx.AsyncClient:
    """Get or create the shared Exa MCP httpx client using the DRY factory pattern."""
    return get_or_create_httpx_client(
        _exa_client_holder,
        timeout=60.0,  # Exa can take longer for deep searches
        connect_timeout=15.0,
        max_connections=20,
        max_keepalive=10,
        follow_redirects=True,
        headers={},
    )


def parse_exa_text_format(text_content: str) -> list[dict]:  # noqa: C901
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

    results = []

    # Split by double newlines to separate individual results
    # Each result block starts with "Title:"
    blocks = text_content.split("\n\nTitle:")

    for i, block in enumerate(blocks):
        # Add back "Title:" prefix for all blocks except the first if it starts
        # with "Title:".
        if i == 0:
            if not block.strip().startswith("Title:"):
                continue  # Skip if first block doesn't start with Title
            current_block = block.strip()
        else:
            current_block = "Title:" + block

        # Parse the block
        title = ""
        url = ""
        content = ""

        lines = current_block.split("\n")
        text_started = False
        text_lines = []

        for line in lines:
            if line.startswith("Title:"):
                title = line[6:].strip()
            elif line.startswith("URL:"):
                url = line[4:].strip()
            elif line.startswith("Text:"):
                text_started = True
                text_lines.append(line[5:].strip())
            elif text_started:
                text_lines.append(line)
            # Skip Author and Published Date as they're not needed

        content = "\n".join(text_lines).strip()

        if title or url:  # Only add if we have at least a title or URL
            results.append({
                "title": title or "Untitled",
                "url": url,
                "content": content,
            })

    return results


async def exa_search(  # noqa: C901, PLR0911, PLR0912, PLR0915
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
    try:
        client = _get_exa_client()

        # MCP uses JSON-RPC 2.0 format for tool calls
        # The web_search_exa tool is enabled by default on Exa MCP
        payload = {
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

        logger.info(
            "Exa MCP request for query '%s': max_results=%s",
            query,
            max_results,
        )

        # Use streaming request to handle SSE responses
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

            # Check content type to determine how to parse
            content_type = response.headers.get("content-type", "")

            if "text/event-stream" in content_type:
                # Handle SSE stream - collect events and find the result
                full_response = ""
                current_event_data = []

                async for raw_line in response.aiter_lines():
                    line = raw_line.rstrip("\n").rstrip("\r")

                    if not line:
                        # End of event (blank line)
                        if current_event_data:
                            # Join without newlines to reconstruct split JSON
                            # without injecting invalid control chars.
                            event_body = "".join(current_event_data)
                            logger.debug(
                                "Exa MCP SSE event body length: %s",
                                len(event_body),
                            )

                            # We are looking for the JSON result. It should start with {
                            if event_body.strip().startswith("{"):
                                full_response = event_body

                            current_event_data = []
                    elif line.startswith("data:"):
                        # Extract data, handling optional space
                        data = line[5:]
                        data = data.removeprefix(" ")
                        current_event_data.append(data)
                    elif line.startswith("event:"):
                        # Log event type for debugging
                        event_type = line[6:].strip()
                        logger.debug("Exa MCP SSE event: %s", event_type)

                # Check if there's any remaining data after the loop finishes
                if current_event_data and not full_response:
                    full_response = "".join(current_event_data)

                if not full_response:
                    logger.warning(
                        "Exa MCP returned empty SSE stream for query '%s'",
                        query,
                    )
                    return {"results": [], "query": query}

                try:
                    result = json.loads(full_response)
                except json.JSONDecodeError as exc:
                    logger.exception(
                        "Exa MCP SSE JSON parse error. Data: %s",
                        full_response[:MAX_ERROR_CHARS],
                    )
                    return {"error": f"JSON parse error: {exc}", "query": query}
            else:
                # Regular JSON response
                response_text = await response.aread()
                response_text = response_text.decode("utf-8")

                if not response_text:
                    logger.warning(
                        "Exa MCP returned empty response for query '%s'",
                        query,
                    )
                    return {"results": [], "query": query}

                try:
                    result = json.loads(response_text)
                except json.JSONDecodeError as exc:
                    logger.exception(
                        "Exa MCP JSON parse error. Response: %s",
                        response_text[:MAX_ERROR_CHARS],
                    )
                    return {"error": f"JSON parse error: {exc}", "query": query}

        # MCP JSON-RPC response format
        if "error" in result:
            error_msg = result.get("error", {}).get("message", "Unknown MCP error")
            logger.error("Exa MCP error for query '%s': %s", query, error_msg)
            return {"error": error_msg, "query": query}

        # Log the full result structure for debugging
        logger.debug(
            "Exa MCP full result keys: %s",
            result.keys() if isinstance(result, dict) else type(result),
        )

        # Extract results from the MCP response
        # The result is in result.result.content[0].text as JSON
        mcp_result = result.get("result", {})
        logger.debug(
            "Exa MCP mcp_result keys: %s",
            mcp_result.keys()
            if isinstance(mcp_result, dict)
            else type(mcp_result),
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

        # Log first content item structure
        first_content = content[0]
        logger.debug(
            "Exa MCP first content item: %s",
            first_content.keys()
            if isinstance(first_content, dict)
            else type(first_content),
        )

        # Parse the text content which contains the actual search results
        if isinstance(first_content, dict):
            text_content = first_content.get("text", "")
        else:
            text_content = str(first_content)
        preview = text_content[:MAX_ERROR_CHARS] if text_content else "empty"
        logger.info("Exa MCP text_content preview: %s...", preview)

        try:
            search_data = json.loads(text_content) if text_content else {}
            logger.debug(
                "Exa MCP search_data keys: %s",
                search_data.keys()
                if isinstance(search_data, dict)
                else type(search_data),
            )

            # Normalize to match Tavily response format
            # Exa might return results directly or under a different key
            results = search_data.get("results", [])
            if not results and isinstance(search_data, list):
                # If search_data is a list directly, use it as results
                results = search_data

            logger.info(
                "Exa MCP response for query '%s': %s results",
                query,
                len(results),
            )
        except json.JSONDecodeError:
            # Exa MCP returns a structured text format, not JSON
            # Parse the structured text format returned by Exa MCP.
            logger.info(
                "Exa MCP returned text format, parsing structured text for "
                "query '%s'",
                query,
            )
            results = parse_exa_text_format(text_content)
            if results:
                logger.info(
                    "Exa MCP parsed %s results from text format",
                    len(results),
                )
                return {"results": results, "query": query}
            # Fallback: treat the entire text as a single result
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
        else:
            return {"results": results, "query": query}

    except httpx.TimeoutException as exc:
        logger.exception("Exa MCP timeout for query '%s'", query)
        return {"error": f"Timeout: {exc}", "query": query}
    except httpx.RequestError as exc:
        logger.exception("Exa MCP connection error for query '%s'", query)
        return {"error": f"Connection error: {exc}", "query": query}
    except Exception as exc:
        logger.exception("Exa MCP search error for query '%s'", query)
        return {"error": str(exc), "query": query}
