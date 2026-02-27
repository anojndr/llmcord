"""Exa MCP search provider integration.

This module integrates with Exa via the Exa MCP HTTP endpoint.

Exa can return errors in multiple shapes:
- HTTP non-200 responses with a JSON body containing `requestId`, `error`, `tag`
- Rate limit (429) responses with a simplified `{ "error": "..." }` body
- JSON-RPC errors (`{"error": {"message": ..., "data": ...}}`)
- Successful responses that still contain per-URL failures in a `statuses` field
"""

import json
import logging
from typing import cast

import httpx

from llmcord.core.config import (
    HttpxClientOptions,
    get_or_create_httpx_client,
)
from llmcord.core.error_handling import log_exception
from llmcord.services.http import DEFAULT_RETRYABLE_STATUSES, wait_before_retry
from llmcord.services.search.config import EXA_MCP_URL, HTTP_OK, MAX_ERROR_CHARS

logger = logging.getLogger(__name__)

# Shared httpx client for Exa MCP API calls - uses DRY factory pattern
_exa_client_holder: list = []


def _get_exa_client() -> httpx.AsyncClient:
    """Get or create the shared Exa MCP httpx client.

    Uses the DRY factory pattern.
    """
    return get_or_create_httpx_client(
        _exa_client_holder,
        options=HttpxClientOptions(
            timeout=60.0,  # Exa can take longer for deep searches
            connect_timeout=15.0,
            max_connections=20,
            max_keepalive=10,
            follow_redirects=True,
            headers={},
        ),
    )


def _truncate_error_text(text: str, *, limit: int = MAX_ERROR_CHARS) -> str:
    if not text:
        return ""
    return text[:limit]


def _safe_json_loads(text: str) -> object | None:
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _as_str(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _extract_exa_api_error_shape(
    data: dict[str, object],
) -> tuple[str | None, str | None, str | None]:
    request_id = _as_str(data.get("requestId"))
    tag = _as_str(data.get("tag"))
    if not (request_id or tag):
        return None, None, None

    message: str | None = None
    raw_error = data.get("error")
    if isinstance(raw_error, str):
        message = raw_error
    elif isinstance(raw_error, dict):
        raw_error_dict = cast("dict[str, object]", raw_error)
        message = _as_str(raw_error_dict.get("message")) or _as_str(
            raw_error_dict.get("error"),
        )

    return message, tag, request_id


def _extract_exa_simple_error_shape(
    data: dict[str, object],
) -> tuple[str | None, str | None, str | None]:
    raw_error = data.get("error")
    if isinstance(raw_error, str):
        return _as_str(raw_error), None, None
    return None, None, None


def _extract_exa_jsonrpc_error_shape(
    data: dict[str, object],
) -> tuple[str | None, str | None, str | None]:
    raw_error = data.get("error")
    if not isinstance(raw_error, dict):
        return None, None, None

    error_dict = cast("dict[str, object]", raw_error)
    nested = error_dict.get("data")
    if nested is not None:
        nested_info = _extract_exa_error_info(nested)
        if any(nested_info):
            return nested_info

    return _as_str(error_dict.get("message")), None, None


def _extract_exa_result_nested_shape(
    data: dict[str, object],
) -> tuple[str | None, str | None, str | None]:
    if "result" not in data:
        return None, None, None
    return _extract_exa_error_info(data.get("result"))


def _extract_exa_error_info(obj: object) -> tuple[str | None, str | None, str | None]:
    """Return (message, tag, request_id) if an Exa-style error payload is found."""
    if not isinstance(obj, dict):
        return None, None, None

    data = cast("dict[str, object]", obj)
    extractors = (
        _extract_exa_api_error_shape,
        _extract_exa_simple_error_shape,
        _extract_exa_jsonrpc_error_shape,
        _extract_exa_result_nested_shape,
    )
    for extractor in extractors:
        message, tag, request_id = extractor(data)
        if message or tag or request_id:
            return message, tag, request_id

    return None, None, None


def _exa_tag_hint(tag: str) -> str | None:
    hints: dict[str, str] = {
        "INVALID_API_KEY": "Check your Exa API key / authentication.",
        "NO_MORE_CREDITS": "Exa credits are exhausted.",
        "API_KEY_BUDGET_EXCEEDED": "Exa API key budget exceeded.",
        "ACCESS_DENIED": "Access denied for this feature or plan.",
        "FEATURE_DISABLED": "Feature disabled for this plan.",
        "ROBOTS_FILTER_FAILED": "All URLs blocked by robots.txt.",
        "PROHIBITED_CONTENT": "Blocked by content safety moderation.",
        "CONTENT_FILTER_ERROR": "Blocked by content safety policy.",
        "INVALID_REQUEST_BODY": "Request body validation failed.",
        "INVALID_REQUEST": "Conflicting or invalid request parameters.",
        "INVALID_URLS": "One or more URLs/IDs are invalid.",
        "FETCH_DOCUMENT_ERROR": "A URL could not be processed.",
        "UNABLE_TO_GENERATE_RESPONSE": "Unable to generate a response for this query.",
        "DEFAULT_ERROR": "Exa server error; retry later.",
        "INTERNAL_ERROR": "Exa internal error; retry later.",
    }
    return hints.get(tag)


def _extract_exa_error_fields(
    *,
    body_text: str | None,
    body_json: object | None,
    fallback_message: str | None,
) -> tuple[str, str | None, str | None]:
    message, tag, request_id = _extract_exa_error_info(body_json)

    if not message and body_text:
        body_candidate = _safe_json_loads(body_text)
        message, tag, request_id = _extract_exa_error_info(body_candidate)

    if not message:
        message = fallback_message or (body_text or "Unknown Exa error")

    return _truncate_error_text(str(message)), tag, request_id


def _compose_exa_error_message(
    *,
    http_status: int | None,
    message: str,
    tag: str | None,
    request_id: str | None,
) -> str:
    prefix_parts: list[str] = []
    if http_status is not None:
        prefix_parts.append(f"HTTP {http_status}")
    if tag is not None:
        prefix_parts.append(tag)
    prefix = " ".join(prefix_parts)
    full_message = f"{prefix}: {message}" if prefix else message

    if tag:
        hint = _exa_tag_hint(tag)
        if hint:
            full_message = f"{full_message} ({hint})"

    if request_id:
        full_message = f"{full_message} (requestId={request_id})"

    return full_message


def _format_exa_error(
    *,
    query: str,
    http_status: int | None,
    body_text: str | None = None,
    body_json: object | None = None,
    fallback_message: str | None = None,
) -> dict:
    message, tag, request_id = _extract_exa_error_fields(
        body_text=body_text,
        body_json=body_json,
        fallback_message=fallback_message,
    )

    full_message = _compose_exa_error_message(
        http_status=http_status,
        message=message,
        tag=tag,
        request_id=request_id,
    )

    error_dict: dict[str, object] = {"error": full_message, "query": query}
    details: dict[str, object] = {}
    if http_status is not None:
        details["http_status"] = http_status
    if tag is not None:
        details["tag"] = tag
    if request_id is not None:
        details["request_id"] = request_id
    if details:
        error_dict["exa_error"] = details
    return error_dict


def _collect_exa_status_errors(statuses: object) -> list[dict[str, object]]:
    if not isinstance(statuses, list):
        return []

    errors: list[dict[str, object]] = []
    for status_entry in statuses:
        if not isinstance(status_entry, dict):
            continue
        status_dict = cast("dict[str, object]", status_entry)
        if status_dict.get("status") != "error":
            continue
        error_obj = status_dict.get("error")
        if not isinstance(error_obj, dict):
            continue
        error_dict = cast("dict[str, object]", error_obj)
        tag = _as_str(error_dict.get("tag"))
        http_status_code = error_dict.get("httpStatusCode")
        url_id = _as_str(status_dict.get("id"))
        errors.append(
            {
                "id": url_id,
                "tag": tag,
                "httpStatusCode": http_status_code,
            },
        )
    return errors


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
            title = line[len("Title:") :].strip()
        elif line.startswith("URL:"):
            url = line[len("URL:") :].strip()
        elif line.startswith("Text:"):
            text_started = True
            text_lines.append(line[len("Text:") :].strip())
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


def _parse_markdown_block(block: str) -> dict | None:
    """Parse a markdown-style Exa block (e.g. from get_code_context_exa)."""
    lines = block.strip().split("\n")
    if not lines:
        return None

    title = lines[0].strip()
    url = ""
    content_start = 1

    if len(lines) > 1:
        second_line = lines[1].strip()
        if second_line.startswith("http"):
            url = second_line
            content_start = 2

    content = "\n".join(lines[content_start:]).strip()
    if not (title or url):
        return None

    return {
        "title": title,
        "url": url,
        "content": content,
    }


def _parse_exa_text_blocks(text_content: str) -> list[dict]:
    """Parse "Title:" format text blocks."""
    results = []
    blocks = text_content.split("\n\nTitle:")
    for i, block in enumerate(blocks):
        current_block = _normalize_exa_block(block, i)
        if not current_block:
            continue
        parsed = _parse_exa_block(current_block)
        if parsed:
            results.append(parsed)
    return results


def _parse_markdown_text_blocks(text_content: str) -> list[dict]:
    """Parse markdown-style "##" text blocks."""
    results = []
    blocks = text_content.split("\n\n## ")
    for i, block in enumerate(blocks):
        clean_block = block.strip()
        if i == 0 and clean_block.startswith("## "):
            clean_block = clean_block[3:]
        parsed = _parse_markdown_block(clean_block)
        if parsed:
            results.append(parsed)
    return results


def parse_exa_text_format(text_content: str) -> list[dict]:
    """Parse Exa MCP text response format into a list of result dicts.

    Supports both "Title:" format and Markdown "##" format.
    """
    if not text_content:
        return []

    # 1. Try splitting by "\n\nTitle:"
    if "\n\nTitle:" in text_content or text_content.strip().startswith("Title:"):
        results = _parse_exa_text_blocks(text_content)
        if results:
            return results

    # 2. Try splitting by "\n\n## "
    if "\n\n## " in text_content or text_content.strip().startswith("## "):
        results = _parse_markdown_text_blocks(text_content)
        if results:
            return results

    return []


def _build_exa_payload(
    query: str,
    max_results: int,
    tool: str = "web_search_exa",
) -> dict:
    """Build the JSON-RPC payload for Exa MCP tool calls."""
    arguments: dict = {
        "query": query,
    }

    if tool == "company_research_exa":
        arguments = {"companyName": query, "numResults": max_results}
    elif tool == "get_code_context_exa":
        arguments = {"query": query, "tokensNum": 5000}
    elif tool == "people_search_exa":
        arguments = {"query": query, "numResults": max_results}
    else:
        arguments["numResults"] = max_results

    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool,
            "arguments": arguments,
        },
    }


async def _collect_sse_body(response: httpx.Response) -> str:
    """Collect SSE event data into a single JSON string."""
    full_response = ""
    current_event_data: list[str] = []
    current_event_maybe_json = False

    async for raw_line in response.aiter_lines():
        line = raw_line.rstrip("\n").rstrip("\r")

        if not line:
            if current_event_data:
                if current_event_maybe_json:
                    event_body = "".join(current_event_data)
                    logger.debug(
                        "Exa MCP SSE event body length: %s",
                        len(event_body),
                    )
                    if event_body.lstrip().startswith("{"):
                        full_response = event_body
                current_event_data = []
                current_event_maybe_json = False
            continue

        if line.startswith("data:"):
            data = line[5:]
            payload = data.removeprefix(" ")
            if not current_event_data:
                current_event_maybe_json = payload.lstrip().startswith("{")
            current_event_data.append(payload)
            continue

        if line.startswith("event:"):
            event_type = line[6:].strip()
            logger.debug("Exa MCP SSE event: %s", event_type)

    if current_event_data and not full_response and current_event_maybe_json:
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
        log_exception(
            logger=logger,
            message="Exa MCP SSE JSON parse error",
            error=exc,
            context={
                "query": query,
                "data_preview": full_response[:MAX_ERROR_CHARS],
            },
        )
        return {"error": f"JSON parse error: {exc}", "query": query}


async def _parse_json_response(response: httpx.Response, query: str) -> dict:
    """Parse JSON response body into a dict."""
    response_bytes = await response.aread()
    response_text = response_bytes.decode("utf-8")

    if not response_text:
        logger.warning(
            "Exa MCP returned empty response for query '%s'",
            query,
        )
        return {"results": [], "query": query}

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as exc:
        log_exception(
            logger=logger,
            message="Exa MCP JSON parse error",
            error=exc,
            context={
                "query": query,
                "response_preview": response_text[:MAX_ERROR_CHARS],
            },
        )
        return {"error": f"JSON parse error: {exc}", "query": query}


async def _parse_exa_http_response(
    response: httpx.Response,
    query: str,
) -> dict:
    """Parse the HTTP response from Exa MCP into a JSON dict."""
    logger.info("Exa MCP response status: %s", response.status_code)

    if response.status_code != HTTP_OK:
        error_bytes = await response.aread()
        error_text = error_bytes.decode("utf-8", errors="replace")
        error_text = _truncate_error_text(error_text)
        error_json = _safe_json_loads(error_text)
        logger.error(
            "Exa MCP HTTP error for query '%s': %s - %s",
            query,
            response.status_code,
            error_text,
        )
        return _format_exa_error(
            query=query,
            http_status=response.status_code,
            body_text=error_text,
            body_json=error_json,
        )

    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        return await _parse_sse_response(response, query)

    return await _parse_json_response(response, query)


def _extract_text_content(mcp_result: dict, query: str) -> str:
    """Extract text content from MCP result."""
    content = mcp_result.get("content", [])
    logger.debug(
        "Exa MCP content type: %s, length: %s",
        type(content),
        len(content) if isinstance(content, list) else "N/A",
    )

    if not (content and isinstance(content, list) and len(content) > 0):
        logger.warning("Exa MCP returned empty content for query '%s'", query)
        return ""

    first_content = content[0]
    first_content_keys = (
        first_content.keys() if isinstance(first_content, dict) else type(first_content)
    )
    logger.debug("Exa MCP first content item: %s", first_content_keys)

    if isinstance(first_content, dict):
        return first_content.get("text", "")
    return str(first_content)


def _handle_json_data(search_data: list | dict, query: str) -> dict:
    """Handle parsed JSON search data."""
    if isinstance(search_data, list):
        results: list = search_data
        statuses: object | None = None
        error_payload: object | None = None
    else:
        # Exa can return: {"results": [...], "statuses": [...]} or
        # error payloads with: {"requestId": ..., "error": ..., "tag": ...}
        error_payload = search_data if "error" in search_data else None
        results = search_data.get("results", [])
        statuses = search_data.get("statuses")

    if error_payload is not None and (not results):
        return _format_exa_error(
            query=query,
            http_status=None,
            body_json=error_payload,
            body_text=None,
        )

    status_errors = _collect_exa_status_errors(statuses)
    if status_errors and not results:
        # If every URL failed content fetching, surface a provider error.
        tag_set: set[str] = set()
        for err in status_errors:
            tag = err.get("tag")
            if isinstance(tag, str) and tag:
                tag_set.add(tag)

        tags = sorted(tag_set)
        tag_summary = ", ".join(tags) if tags else "unknown"
        return {
            "error": f"Exa content fetch failed: {tag_summary}",
            "query": query,
            "exa_status_errors": status_errors,
        }

    logger.info(
        "Exa MCP response for query '%s': %s results",
        query,
        len(results),
    )
    response_dict: dict[str, object] = {"results": results, "query": query}
    if status_errors:
        response_dict["exa_status_errors"] = status_errors
    return response_dict


def _handle_text_format(text_content: str, query: str) -> dict:
    """Handle text format search data."""
    logger.info(
        ("Exa MCP returned text format, parsing structured text for query '%s'"),
        query,
    )
    results = parse_exa_text_format(text_content)
    if results:
        logger.info(
            "Exa MCP parsed %s results from text format",
            len(results),
        )
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


def _extract_exa_results(result: dict, query: str) -> dict:
    """Extract Exa results from the MCP JSON-RPC response."""
    if "error" in result:
        error = result.get("error")
        error_msg = _as_str(error) if isinstance(error, str) else None
        if isinstance(error, dict):
            error_msg = _as_str(error.get("message")) or _as_str(error.get("error"))
            formatted = _format_exa_error(
                query=query,
                http_status=None,
                body_json=error,
                fallback_message=error_msg or "Unknown MCP error",
            )
            logger.error(
                "Exa MCP error for query '%s': %s",
                query,
                formatted.get("error"),
            )
            return formatted

        logger.error("Exa MCP error for query '%s': %s", query, error_msg)
        return {"error": error_msg or "Unknown MCP error", "query": query}

    mcp_result = result.get("result", {})
    if mcp_result.get("isError"):
        error_msg = "Unknown MCP tool error"
        content = mcp_result.get("content", [])
        if content and isinstance(content, list) and len(content) > 0:
            first_content = content[0]
            if isinstance(first_content, dict):
                error_msg = first_content.get("text", error_msg)

        # The tool error text may itself be an Exa API error payload.
        error_json = _safe_json_loads(str(error_msg))
        formatted = _format_exa_error(
            query=query,
            http_status=None,
            body_text=str(error_msg),
            body_json=error_json,
            fallback_message=str(error_msg),
        )
        logger.error("Exa MCP tool error for query '%s': %s", query, error_msg)
        return formatted

    text_content = _extract_text_content(mcp_result, query)
    if not text_content:
        return {"results": [], "query": query}

    preview = text_content[:MAX_ERROR_CHARS]
    logger.info("Exa MCP text_content preview: %s...", preview)

    try:
        search_data = json.loads(text_content)
        return _handle_json_data(search_data, query)
    except json.JSONDecodeError:
        return _handle_text_format(text_content, query)


async def _do_exa_request(
    client: httpx.AsyncClient,
    client_type: str,
    exa_mcp_url: str,
    payload: dict,
    query: str,
) -> dict | None:
    """Execute Exa MCP search with retries for a specific client."""
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
                        "Exa MCP transient HTTP %s (%s) for query '%s', "
                        "retrying (%s/%s)",
                        response.status_code,
                        client_type,
                        query,
                        attempt + 1,
                        retries,
                    )
                    await wait_before_retry(attempt, response=response)
                    continue

                result = await _parse_exa_http_response(response, query)
                return _extract_exa_results(result, query)

        except (httpx.TimeoutException, httpx.RequestError) as exc:
            if attempt < retries:
                logger.warning(
                    "Exa MCP %s error (%s) for query '%s', retrying (%s/%s)",
                    type(exc).__name__,
                    client_type,
                    query,
                    attempt + 1,
                    retries,
                )
                await wait_before_retry(attempt)
                continue
            raise
    return None


async def exa_search(
    query: str,
    exa_mcp_url: str = EXA_MCP_URL,
    max_results: int = 5,
    tool: str = "web_search_exa",
) -> dict:
    """Execute a single Exa MCP web search query.

    Uses the Exa MCP HTTP endpoint for web search. Handles SSE (Server-Sent
    Events) streaming responses. Returns the search results with page content or
    an error dict.

    Args:
        query: Search query
        exa_mcp_url: The Exa MCP endpoint URL (default: https://mcp.exa.ai/mcp)
        max_results: Maximum results to return
        tool: The Exa MCP tool to call (default: web_search_exa)

    """
    client = _get_exa_client()
    payload = _build_exa_payload(query, max_results, tool)

    logger.info(
        "Exa MCP request (%s) for query '%s': max_results=%s",
        tool,
        query,
        max_results,
    )

    try:
        result = await _do_exa_request(
            client,
            "direct",
            exa_mcp_url,
            payload,
            query,
        )
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        log_exception(
            logger=logger,
            message="Exa MCP search error (direct)",
            error=exc,
            context={"query": query, "tool": tool},
        )
        return {"error": f"{type(exc).__name__}: {exc}", "query": query}
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        log_exception(
            logger=logger,
            message="Exa MCP search error",
            error=exc,
            context={"query": query, "tool": tool},
        )
        return {"error": str(exc), "query": query}

    return result or {"error": "Retry attempts exhausted", "query": query}
