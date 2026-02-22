"""Tavily search provider integration."""

import asyncio
import importlib
import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import httpx

from llmcord.core.config import (
    HttpxClientOptions,
    get_or_create_httpx_client,
)
from llmcord.core.error_handling import log_exception
from llmcord.services.http import request_with_retries
from llmcord.services.search.config import HTTP_OK, MAX_ERROR_CHARS, MAX_LOG_CHARS

logger = logging.getLogger(__name__)


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
        return stripped if stripped else None
    return None


def _extract_tavily_request_id(obj: object) -> str | None:
    if not isinstance(obj, dict):
        return None
    data = cast("dict[str, object]", obj)
    return _as_str(data.get("request_id")) or _as_str(data.get("requestId"))


def _extract_tavily_error_message(obj: object) -> str | None:
    """Extract a human-readable error message from a Tavily error payload.

    Tavily's OpenAPI error schema is typically:
    {"detail": {"error": "..."}}
    But we also defensively handle other shapes.
    """
    if not isinstance(obj, dict):
        return None

    data = cast("dict[str, object]", obj)

    raw_detail = data.get("detail")
    if isinstance(raw_detail, dict):
        detail = cast("dict[str, object]", raw_detail)
        message = _as_str(detail.get("error")) or _as_str(detail.get("message"))
        if message:
            return message

    raw_error = data.get("error")
    if isinstance(raw_error, str):
        return _as_str(raw_error)
    if isinstance(raw_error, dict):
        error_dict = cast("dict[str, object]", raw_error)
        message = _as_str(error_dict.get("error")) or _as_str(error_dict.get("message"))
        if message:
            return message

    return _as_str(data.get("message"))


def _tavily_error_kind_from_http_status(http_status: int | None) -> str | None:
    if http_status is None:
        return None

    kinds: dict[int, str] = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        429: "rate_limited",
        432: "plan_limit_exceeded",
        433: "paygo_limit_exceeded",
        500: "server_error",
    }
    return kinds.get(http_status, "http_error")


def _tavily_kind_hint(kind: str) -> str | None:
    hints: dict[str, str] = {
        "unauthorized": "Check your Tavily API key (missing/invalid).",
        "rate_limited": "Rate limit exceeded; retry later or rotate keys.",
        "plan_limit_exceeded": (
            "Plan/key usage limit exceeded; upgrade plan or rotate keys."
        ),
        "paygo_limit_exceeded": (
            "Pay-as-you-go limit exceeded; raise PayGo limit in dashboard."
        ),
        "server_error": "Tavily server error; retry later.",
        "timeout": "Tavily request timed out; retry later.",
        "connection_error": "Network/connectivity issue; retry later.",
        "unexpected_response": (
            "Unexpected response from Tavily; check payload/response format."
        ),
    }
    return hints.get(kind)


def _compose_tavily_error_message(
    *,
    http_status: int | None,
    kind: str | None,
    message: str,
    request_id: str | None,
) -> str:
    prefix_parts: list[str] = []
    if http_status is not None:
        prefix_parts.append(f"HTTP {http_status}")
    if kind is not None:
        prefix_parts.append(kind)
    prefix = " ".join(prefix_parts)
    full_message = f"{prefix}: {message}" if prefix else message

    if kind:
        hint = _tavily_kind_hint(kind)
        if hint:
            full_message = f"{full_message} ({hint})"

    if request_id:
        full_message = f"{full_message} (request_id={request_id})"
    return full_message


@dataclass(frozen=True, slots=True)
class _TavilyParsedResponse:
    body_text: str
    body_json: object | None
    request_id: str | None


@dataclass(frozen=True, slots=True)
class _TavilyErrorContext:
    query: str | None
    http_status: int | None
    kind_override: str | None = None
    body_text: str | None = None
    body_json: object | None = None
    fallback_message: str | None = None
    request_id: str | None = None
    retry_after: str | None = None


async def _read_tavily_response(response: httpx.Response) -> _TavilyParsedResponse:
    body_bytes = await response.aread()
    body_text = body_bytes.decode("utf-8", errors="replace")
    body_json = _safe_json_loads(body_text)
    request_id = _extract_tavily_request_id(body_json)
    return _TavilyParsedResponse(
        body_text=body_text,
        body_json=body_json,
        request_id=request_id,
    )


def _format_tavily_error(context: _TavilyErrorContext) -> dict:
    message = _extract_tavily_error_message(context.body_json)
    if not message and context.body_text:
        message = _extract_tavily_error_message(_safe_json_loads(context.body_text))
    if not message:
        message = context.fallback_message or (
            context.body_text or "Unknown Tavily error"
        )

    message = _truncate_error_text(str(message))
    kind = context.kind_override or _tavily_error_kind_from_http_status(
        context.http_status,
    )
    full_message = _compose_tavily_error_message(
        http_status=context.http_status,
        kind=kind,
        message=message,
        request_id=context.request_id,
    )

    error_dict: dict[str, object] = {"error": full_message}
    if context.query is not None:
        error_dict["query"] = context.query
    if context.http_status is not None:
        error_dict["status_code"] = context.http_status

    details: dict[str, object] = {}
    if context.http_status is not None:
        details["http_status"] = context.http_status
    if kind is not None:
        details["kind"] = kind
    if context.request_id is not None:
        details["request_id"] = context.request_id
    if context.retry_after is not None:
        details["retry_after"] = context.retry_after
    if details:
        error_dict["tavily_error"] = details
    return error_dict


async def _parse_tavily_json_response(
    *,
    response: httpx.Response,
    query: str | None,
    ok_statuses: set[int],
) -> dict:
    parsed = await _read_tavily_response(response)
    retry_after = response.headers.get("retry-after")

    body_preview = _truncate_error_text(parsed.body_text, limit=MAX_LOG_CHARS)
    logger.debug("Tavily API raw response: %s", body_preview)

    if response.status_code not in ok_statuses:
        return _format_tavily_error(
            _TavilyErrorContext(
                query=query,
                http_status=response.status_code,
                body_text=parsed.body_text,
                body_json=parsed.body_json,
                request_id=parsed.request_id,
                retry_after=retry_after,
            ),
        )

    if parsed.body_json is None:
        return _format_tavily_error(
            _TavilyErrorContext(
                query=query,
                http_status=response.status_code,
                kind_override="unexpected_response",
                body_text=parsed.body_text,
                fallback_message="Invalid JSON in Tavily response",
                request_id=parsed.request_id,
            ),
        )

    if not isinstance(parsed.body_json, dict):
        return _format_tavily_error(
            _TavilyErrorContext(
                query=query,
                http_status=response.status_code,
                kind_override="unexpected_response",
                body_text=parsed.body_text,
                body_json=parsed.body_json,
                fallback_message="Unexpected Tavily response shape",
                request_id=parsed.request_id,
            ),
        )

    # Defensive: some services return 200 with an error payload.
    maybe_message = _extract_tavily_error_message(parsed.body_json)
    if maybe_message:
        return _format_tavily_error(
            _TavilyErrorContext(
                query=query,
                http_status=response.status_code,
                kind_override="unexpected_response",
                body_text=parsed.body_text,
                body_json=parsed.body_json,
                request_id=parsed.request_id,
            ),
        )

    return cast("dict", parsed.body_json)


# Shared httpx client for Tavily API calls - uses DRY factory pattern
_tavily_client_holder: list = []


def _get_tavily_client() -> httpx.AsyncClient:
    """Get or create the shared Tavily httpx client.

    Uses the DRY factory pattern.
    """
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
    return cast("httpx.AsyncClient", get_client())


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

        response = await request_with_retries(
            lambda: client.post(
                "https://api.tavily.com/search",
                json=payload,
                headers={
                    "Authorization": f"Bearer {tavily_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=timeout,
            ),
            log_context=f"Tavily search '{query}'",
        )
        logger.info("Tavily API response status: %s", response.status_code)

        result = await _parse_tavily_json_response(
            response=response,
            query=query,
            ok_statuses={HTTP_OK},
        )
        if "error" in result:
            return result

        logger.info(
            "Tavily API response for query '%s': %s results",
            query,
            len(result.get("results", [])),
        )
        if not result.get("results"):
            logger.warning(
                ("Tavily returned empty results for query '%s'. Full response: %s"),
                query,
                result,
            )
            return {"results": [], "query": query}
    except httpx.TimeoutException as exc:
        log_exception(
            logger=logger,
            message="Tavily timeout",
            error=exc,
            context={"query": query},
        )
        return _format_tavily_error(
            _TavilyErrorContext(
                query=query,
                http_status=None,
                kind_override="timeout",
                fallback_message=f"Timeout: {exc}",
            ),
        )
    except httpx.RequestError as exc:
        log_exception(
            logger=logger,
            message="Tavily connection error",
            error=exc,
            context={"query": query},
        )
        return _format_tavily_error(
            _TavilyErrorContext(
                query=query,
                http_status=None,
                kind_override="connection_error",
                fallback_message=f"Connection error: {exc}",
            ),
        )
    except (
        httpx.HTTPError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        log_exception(
            logger=logger,
            message="Tavily search error",
            error=exc,
            context={"query": query},
        )
        return _format_tavily_error(
            _TavilyErrorContext(
                query=query,
                http_status=None,
                kind_override="unexpected_response",
                fallback_message=str(exc),
            ),
        )
    else:
        return result


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

        response = await request_with_retries(
            lambda: client.post(
                "https://api.tavily.com/research",
                json=payload,
                headers={
                    "Authorization": f"Bearer {tavily_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            ),
            log_context="Tavily research create",
        )
    except httpx.TimeoutException as exc:
        log_exception(
            logger=logger,
            message="Tavily research create timeout",
            error=exc,
            context={"model": model},
        )
        return _format_tavily_error(
            _TavilyErrorContext(
                query=None,
                http_status=None,
                kind_override="timeout",
                fallback_message=f"Timeout: {exc}",
            ),
        )
    except httpx.RequestError as exc:
        log_exception(
            logger=logger,
            message="Tavily research create connection error",
            error=exc,
            context={"model": model},
        )
        return _format_tavily_error(
            _TavilyErrorContext(
                query=None,
                http_status=None,
                kind_override="connection_error",
                fallback_message=f"Connection error: {exc}",
            ),
        )
    except (
        httpx.HTTPError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        log_exception(
            logger=logger,
            message="Tavily research create error",
            error=exc,
            context={"model": model},
        )
        return _format_tavily_error(
            _TavilyErrorContext(
                query=None,
                http_status=None,
                kind_override="unexpected_response",
                fallback_message=str(exc),
            ),
        )
    else:
        return await _parse_tavily_json_response(
            response=response,
            query=None,
            ok_statuses={HTTP_OK, 201},
        )


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
        client = _get_client_from_package()
        response = await request_with_retries(
            lambda: client.get(
                f"https://api.tavily.com/research/{request_id}",
                headers={
                    "Authorization": f"Bearer {tavily_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            ),
            log_context=f"Tavily research get '{request_id}'",
        )
    except httpx.TimeoutException as exc:
        log_exception(
            logger=logger,
            message="Tavily research get timeout",
            error=exc,
            context={"request_id": request_id},
        )
        return _format_tavily_error(
            _TavilyErrorContext(
                query=None,
                http_status=None,
                kind_override="timeout",
                fallback_message=f"Timeout: {exc}",
            ),
        )
    except httpx.RequestError as exc:
        log_exception(
            logger=logger,
            message="Tavily research get connection error",
            error=exc,
            context={"request_id": request_id},
        )
        return _format_tavily_error(
            _TavilyErrorContext(
                query=None,
                http_status=None,
                kind_override="connection_error",
                fallback_message=f"Connection error: {exc}",
            ),
        )
    except (
        httpx.HTTPError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        log_exception(
            logger=logger,
            message="Tavily research get error",
            error=exc,
            context={"request_id": request_id},
        )
        return _format_tavily_error(
            _TavilyErrorContext(
                query=None,
                http_status=None,
                kind_override="unexpected_response",
                fallback_message=str(exc),
            ),
        )
    else:
        return await _parse_tavily_json_response(
            response=response,
            query=None,
            ok_statuses={HTTP_OK, 202},
        )


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
        if not isinstance(source, Mapping):
            continue
        mapping = cast("Mapping[str, object]", source)
        url = mapping.get("url")
        if not url:
            continue
        urls.append(
            {
                "title": str(mapping.get("title", "No title")),
                "url": str(url),
                "favicon": str(mapping.get("favicon", "")),
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

    for key in api_keys:
        create_result = await tavily_research_create(query, model, key)
        if "error" in create_result:
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
            if not status_result.get("status_code"):
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
