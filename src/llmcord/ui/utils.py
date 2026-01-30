"""Utility functions for UI components."""
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import discord
import httpx
from bs4 import BeautifulSoup

from llmcord.config import get_or_create_httpx_client
from llmcord.services.database import get_bad_keys_db
from llmcord.ui.constants import HTTP_OK

LOGGER = logging.getLogger(__name__)

# Shared httpx client for text.is uploads - uses factory pattern for DRY
_textis_client_holder: list[httpx.AsyncClient] = []

RetryHandler = Callable[[discord.Interaction, int, int], Awaitable[None]]
_retry_handler: RetryHandler | None = None


def set_retry_handler(handler: RetryHandler | None) -> None:
    """Set the global retry handler used by persistent buttons."""
    global _retry_handler
    _retry_handler = handler


def get_retry_handler() -> RetryHandler | None:
    """Get the global retry handler."""
    return _retry_handler


@dataclass(slots=True)
class ResponseData:
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


def _get_textis_client() -> httpx.AsyncClient:
    """Get or create the shared text.is httpx client using the DRY factory pattern."""
    return get_or_create_httpx_client(
        _textis_client_holder,
        timeout=30.0,
        connect_timeout=10.0,
        max_connections=10,
        max_keepalive=5,
        proxy_url=None,
        follow_redirects=False,  # text.is needs redirect handling
    )


async def upload_to_textis(text: str) -> str | None:
    """Upload text to text.is and return the paste URL.

    Returns None if upload fails.
    """
    try:
        client = _get_textis_client()

        # Get the CSRF token from the main page
        response = await client.get("https://text.is/", timeout=30)

        # Extract CSRF token from the form
        soup = BeautifulSoup(response.text, "lxml")  # lxml is faster than html.parser
        csrf_input = soup.find("input", {"name": "csrfmiddlewaretoken"})
        if not csrf_input:
            # Debug: log a snippet of the response to help diagnose
            LOGGER.error("Could not find CSRF token on text.is")
            LOGGER.debug(
                "Response status: %s, Content preview: %s",
                response.status_code,
                response.text[:500],
            )
            return None

        csrf_token = csrf_input.get("value")

        # Get cookies from the response
        cookies = response.cookies

        # POST the text content
        form_data = {
            "csrfmiddlewaretoken": csrf_token,
            "text": text,
        }

        headers = {
            "Referer": "https://text.is/",
            "Origin": "https://text.is",
        }

        post_response = await client.post(
            "https://text.is/",
            data=form_data,
            headers=headers,
            cookies=cookies,
            timeout=30,
        )

        # The response should be a redirect (302) to the paste URL
        if post_response.status_code in (301, 302, 303, 307, 308):
            paste_url = post_response.headers.get("Location")
            if paste_url:
                # Handle relative URLs
                if paste_url.startswith("/"):
                    paste_url = f"https://text.is{paste_url}"
                return paste_url

        # If we got a 200, the paste might have been created and we're on the page
        if post_response.status_code == HTTP_OK:
            # Check if the URL changed (we might be on the paste page)
            final_url = str(post_response.url)
            if final_url != "https://text.is/" and "text.is/" in final_url:
                return final_url

        LOGGER.error(
            "Unexpected response from text.is: %s",
            post_response.status_code,
        )
    except Exception:
        LOGGER.exception("Error uploading to text.is")
    else:
        return None

    return None
