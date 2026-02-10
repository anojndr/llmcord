"""HTTP helper utilities for resilient requests."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

DEFAULT_RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


async def wait_with_backoff(
    attempt: int,
    *,
    base_delay: float = 0.5,
    max_delay: float = 4.0,
) -> None:
    """Wait with exponential backoff based on the retry attempt."""
    delay = min(max_delay, base_delay * (2**attempt))
    await asyncio.sleep(delay)


@dataclass(frozen=True, slots=True)
class RetryOptions:
    """Configuration for HTTP retry behavior."""

    retries: int = 2
    base_delay: float = 0.5
    max_delay: float = 4.0
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
                await wait_with_backoff(
                    attempt,
                    base_delay=retry_options.base_delay,
                    max_delay=retry_options.max_delay,
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
            await wait_with_backoff(
                attempt,
                base_delay=retry_options.base_delay,
                max_delay=retry_options.max_delay,
            )
            continue

        return response

    msg = "request_with_retries exhausted without a response"
    raise RuntimeError(msg)
