"""HTTP client helpers for configuration."""

import httpx

from .app_constants import BROWSER_HEADERS


def get_or_create_httpx_client(  # noqa: PLR0913
    client_holder: list[httpx.AsyncClient | None],
    *,
    timeout: float = 30.0,
    connect_timeout: float = 10.0,
    max_connections: int = 20,
    max_keepalive: int = 10,
    headers: dict[str, str] | None = None,
    proxy_url: str | None = None,
    follow_redirects: bool = True,
) -> httpx.AsyncClient:
    """Get or create a shared httpx.AsyncClient with lazy initialization.

    This factory function provides a consistent pattern for creating httpx clients
    across the codebase, avoiding duplication of client configuration.

    Args:
        client_holder: A mutable list containing the client instance (or empty).
            Used as a container so the client can be stored globally.
        timeout: Total request timeout in seconds.
        connect_timeout: Connection timeout in seconds.
        max_connections: Maximum number of connections.
        max_keepalive: Maximum number of keepalive connections.
        headers: Optional headers dict (merged with `BROWSER_HEADERS` if provided).
        proxy_url: Optional proxy URL.
        follow_redirects: Whether to follow redirects.

    Returns:
        httpx.AsyncClient instance.

    Example:
        _my_client = []  # Container for lazy init
        def get_my_client():
            return get_or_create_httpx_client(_my_client, timeout=30.0)

    """
    # Check if client exists and is not closed
    if (
        client_holder
        and client_holder[0] is not None
        and not client_holder[0].is_closed
    ):
        return client_holder[0]

    # Create new client
    final_headers = {**BROWSER_HEADERS, **(headers or {})}

    client = httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=connect_timeout),
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive,
        ),
        headers=final_headers,
        proxy=proxy_url,
        follow_redirects=follow_redirects,
    )

    # Store in holder
    if client_holder is not None:
        if len(client_holder) == 0:
            client_holder.append(client)
        else:
            client_holder[0] = client

    return client
