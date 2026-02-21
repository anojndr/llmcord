"""HTTP helper utilities for resilient requests."""

from __future__ import annotations

import asyncio
import email.utils
import logging
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

DEFAULT_RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
HTTP_TOO_MANY_REQUESTS = 429
_JITTER_RANDOM = secrets.SystemRandom()


def _parse_retry_after_seconds(value: str) -> float | None:
    stripped = value.strip()
    result: float | None = None
    if not stripped:
        return result

    try:
        seconds = float(stripped)
    except ValueError:
        seconds = None

    if seconds is not None:
        if seconds >= 0:
            result = seconds
        return result

    try:
        parsed = email.utils.parsedate_to_datetime(stripped)
    except (TypeError, ValueError, OSError):
        parsed = None
    if parsed is None:
        return result

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)

    now = datetime.now(UTC)
    delta = (parsed - now).total_seconds()
    return max(delta, 0.0)


async def wait_before_retry(
    attempt: int,
    *,
    response: httpx.Response | None = None,
    max_backoff_seconds: float = 30.0,
    unused_delay_1: float = 0.0,
    unused_delay_2: float = 0.0,
) -> None:
    """Wait before retrying, using exponential backoff.

    If a 429 response includes a `Retry-After` header, respect it.
    Delay parameters are retained for legacy compatibility.
    """
    _ = (unused_delay_1, unused_delay_2)

    retry_after_seconds: float | None = None
    if response is not None and response.status_code == HTTP_TOO_MANY_REQUESTS:
        retry_after_header = response.headers.get("retry-after")
        if isinstance(retry_after_header, str):
            retry_after_seconds = _parse_retry_after_seconds(retry_after_header)

    if retry_after_seconds is not None:
        delay = retry_after_seconds
    else:
        base = 2 ** (attempt + 1)
        jitter = _JITTER_RANDOM.random()
        delay = min(max_backoff_seconds, base + jitter)

    if delay > 0:
        await asyncio.sleep(delay)


@dataclass(frozen=True, slots=True)
class RetryOptions:
    """Configuration for HTTP retry behavior.

    All retries are immediate.
    Delay options are retained for legacy compatibility but unused.
    """

    retries: int = 2
    unused_delay_1: float = 0.0
    unused_delay_2: float = 0.0
    retryable_statuses: set[int] | frozenset[int] | None = None


async def request_with_retries(
    request_factory: Callable[[], Awaitable[httpx.Response]],
    *,
    options: RetryOptions | None = None,
    log_context: str = "",
) -> httpx.Response:
    """Run a request with bounded retries for transient failures."""
    retry_options = options or RetryOptions()
    statuses = retry_options.retryable_statuses or DEFAULT_RETRYABLE_STATUSES

    for attempt in range(retry_options.retries + 1):
        try:
            response = await request_factory()
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            if attempt < retry_options.retries:
                context_suffix = f" for {log_context}" if log_context else ""
                logger.warning(
                    "Transient HTTP error%s, retrying (%s/%s): %s",
                    context_suffix,
                    attempt + 1,
                    retry_options.retries,
                    exc,
                )
                await wait_before_retry(
                    attempt,
                    unused_delay_1=retry_options.unused_delay_1,
                    unused_delay_2=retry_options.unused_delay_2,
                )
                continue
            raise

        if response.status_code in statuses and attempt < retry_options.retries:
            context_suffix = f" for {log_context}" if log_context else ""
            logger.warning(
                "Transient HTTP %s%s, retrying (%s/%s)",
                response.status_code,
                context_suffix,
                attempt + 1,
                retry_options.retries,
            )
            await response.aclose()
            await wait_before_retry(
                attempt,
                response=response,
                unused_delay_1=retry_options.unused_delay_1,
                unused_delay_2=retry_options.unused_delay_2,
            )
            continue

        return response

    msg = "request_with_retries exhausted without a response"
    raise RuntimeError(msg)
