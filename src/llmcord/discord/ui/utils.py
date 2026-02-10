"""Utility functions for UI components."""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import discord
import httpx
from bs4 import BeautifulSoup

from llmcord.core.config import (
    EMBED_COLOR_INCOMPLETE,
    HttpxClientOptions,
    get_config,
    get_or_create_httpx_client,
)
from llmcord.discord.ui.constants import HTTP_OK
from llmcord.services.database import get_bad_keys_db

LOGGER = logging.getLogger(__name__)

# Shared httpx client for rentry.co uploads - uses factory pattern for DRY
_rentry_client_holder: list[httpx.AsyncClient | None] = []

RetryHandler = Callable[[discord.Interaction, int, int], Awaitable[None]]
_retry_handler_holder: list[RetryHandler | None] = [None]


def set_retry_handler(handler: RetryHandler | None) -> None:
    """Set the global retry handler used by persistent buttons."""
    _retry_handler_holder[0] = handler


def get_retry_handler() -> RetryHandler | None:
    """Get the global retry handler."""
    return _retry_handler_holder[0]


def build_error_embed(
    description: str,
    *,
    title: str = "Something went wrong",
) -> discord.Embed:
    """Build a standardized error embed for user-facing failures."""
    return discord.Embed(
        title=title,
        description=description,
        color=EMBED_COLOR_INCOMPLETE,
    )


@dataclass(slots=True)
class ResponseData:
    """Container for response-related metadata."""

    full_response: str | None
    grounding_metadata: object | None
    tavily_metadata: dict[str, object] | None
    request_message_id: int | None
    request_user_id: int | None


def get_response_data(message_id: int) -> ResponseData:
    """Get response data from the database."""
    db = get_bad_keys_db()
    (
        full_response,
        grounding_metadata,
        tavily_metadata,
        request_message_id,
        request_user_id,
    ) = db.get_message_response_data(str(message_id))

    return ResponseData(
        full_response=full_response,
        grounding_metadata=grounding_metadata,
        tavily_metadata=tavily_metadata,
        request_message_id=int(request_message_id)
        if request_message_id and str(request_message_id).isdigit()
        else None,
        request_user_id=int(request_user_id)
        if request_user_id and str(request_user_id).isdigit()
        else None,
    )


def _get_rentry_client() -> httpx.AsyncClient:
    """Get or create the shared rentry.co httpx client.

    Uses the DRY factory pattern.
    """
    config = get_config()
    proxy_url = config.get("proxy_url") or None
    return get_or_create_httpx_client(
        _rentry_client_holder,
        options=HttpxClientOptions(
            timeout=30.0,
            connect_timeout=10.0,
            max_connections=10,
            max_keepalive=5,
            proxy_url=proxy_url,
            follow_redirects=True,  # Now following redirects for robustness
        ),
    )


async def _get_csrf_token(
    client: httpx.AsyncClient,
) -> tuple[str | None, httpx.Response]:
    """Fetch the CSRF token from the rentry.co home page."""
    # Step 1: GET the page. Following redirects is important.
    response = await client.get("https://rentry.co/", timeout=30)

    # Step 2: Extract CSRF token
    # We try multiple ways to find the token as sites often change their structure.
    soup = BeautifulSoup(response.text, "html.parser")
    csrf_input = (
        soup.find("input", {"name": "csrfmiddlewaretoken"})
        or soup.find("input", {"name": "csrf_token"})
        or soup.find("input", {"name": "csrf"})
    )

    csrf_token = None
    if csrf_input:
        csrf_token = str(csrf_input.get("value", ""))

    # Check cookies as a fallback (Django often sets csrftoken cookie)
    if not csrf_token:
        csrf_token = response.cookies.get("csrftoken")

    return csrf_token, response


async def upload_to_rentry(text: str) -> str | None:
    """Upload text to rentry.co and return the paste URL.

    Returns None if upload fails.
    """
    try:
        client = _get_rentry_client()
        csrf_token, response = await _get_csrf_token(client)

        if not csrf_token:
            # Debug: log a snippet of the response to help diagnose
            LOGGER.error("Could not find CSRF token on rentry.co")
            LOGGER.debug(
                "Response status: %s, Content preview: %s, URL: %s",
                response.status_code,
                response.text[:500],
                response.url,
            )
            return None

        # POST the text content
        # Note: We still use 'csrfmiddlewaretoken' as the key.
        form_data = {
            "csrfmiddlewaretoken": csrf_token,
            "text": text,
        }

        headers = {
            "Referer": str(response.url),
            "Origin": "https://rentry.co",
        }

        # We allow the client to follow redirects as it's more robust.
        post_response = await client.post(
            "https://rentry.co/",
            data=form_data,
            headers=headers,
            timeout=30,
        )

        # Handle various success scenarios
        # 1. Redirect to the new paste URL
        if post_response.status_code in (301, 302, 303, 307, 308):
            paste_url = post_response.headers.get("Location")
            if paste_url:
                if paste_url.startswith("/"):
                    paste_url = f"https://rentry.co{paste_url}"
                return paste_url

        # 2. Success with 200 OK (final page after redirect or direct response)
        if post_response.status_code == HTTP_OK:
            final_url = str(post_response.url)
            # Ensure we're not just back on the home page
            if final_url.rstrip("/") != "https://rentry.co":
                return final_url

        LOGGER.error(
            "Unexpected response from rentry.co: %s (URL: %s)",
            post_response.status_code,
            post_response.url,
        )
    except Exception:
        LOGGER.exception("Error uploading to rentry.co")

    return None
