"""HTTP helper utilities for resilient requests."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

DEFAULT_RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


async def wait_before_retry(
    attempt: int,
    *,
    unused_delay_1: float = 0.0,
    unused_delay_2: float = 0.0,
) -> None:
    """Retry immediately without delay.

    Delay parameters are unused and retained for legacy compatibility.
    """
    _ = (attempt, unused_delay_1, unused_delay_2)


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
                unused_delay_1=retry_options.unused_delay_1,
                unused_delay_2=retry_options.unused_delay_2,
            )
            continue

        return response

    msg = "request_with_retries exhausted without a response"
    raise RuntimeError(msg)
