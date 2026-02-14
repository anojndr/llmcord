"""HTTP client configuration and factory."""

from dataclasses import dataclass

import httpx

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/133.0.0.0 Safari/537.36"
)

FALLBACK_USER_AGENT = "llmcord (https://github.com/mariozechner/llmcord)"


# Browser-like headers for web scraping/HTTP requests
@dataclass(frozen=True, slots=True)
class HttpxClientOptions:
    """Options for configuring an httpx.AsyncClient."""

    timeout: float = 30.0
    connect_timeout: float = 10.0
    max_connections: int = 20
    max_keepalive: int = 10
    headers: dict[str, str] | None = None
    follow_redirects: bool = True


BROWSER_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
        "image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "max-age=0",
    "Sec-Ch-Ua": '"Not(A:Brand";v="24", "Chromium";v="133", "Google Chrome";v="133"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}


def get_or_create_httpx_client(
    client_holder: list[httpx.AsyncClient | None],
    *,
    options: HttpxClientOptions | None = None,
) -> httpx.AsyncClient:
    """Get or create a shared httpx.AsyncClient with lazy initialization.

    This factory function provides a consistent pattern for creating httpx
    clients across the codebase, avoiding duplication of client configuration.

    Args:
        client_holder: A mutable list containing the client instance (or empty).
            Used as a container so the client can be stored globally.
        options: Optional configuration overrides for the httpx client.

    Returns:
        httpx.AsyncClient instance.

    Example:
        _my_client = []  # Container for lazy init
        def get_my_client():
            return get_or_create_httpx_client(
                _my_client,
                options=HttpxClientOptions(timeout=30.0),
            )

    """
    # Check if client exists and is not closed
    if (
        client_holder
        and client_holder[0] is not None
        and not client_holder[0].is_closed
    ):
        return client_holder[0]

    # Create new client
    effective_options = options or HttpxClientOptions()
    final_headers = {**BROWSER_HEADERS, **(effective_options.headers or {})}

    client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            effective_options.timeout,
            connect=effective_options.connect_timeout,
        ),
        limits=httpx.Limits(
            max_connections=effective_options.max_connections,
            max_keepalive_connections=effective_options.max_keepalive,
        ),
        headers=final_headers,
        follow_redirects=effective_options.follow_redirects,
    )

    # Store in holder
    if client_holder is not None:
        if len(client_holder) == 0:
            client_holder.append(client)
        else:
            client_holder[0] = client

    return client
