"""Custom exceptions for llmcord logic."""

from typing import NoReturn

from llmcord.core.config import get_config

EMPTY_RESPONSE_MESSAGE = "Response stream ended with no content"
FIRST_RESPONSE_TIMEOUT_MESSAGE = "No first token received within timeout window"
FIRST_TOKEN_TIMEOUT_SECONDS = 30
LITELLM_TIMEOUT_SECONDS = 60


def get_first_token_timeout_seconds() -> int:
    """Return configured first-token timeout seconds with safe fallback."""
    raw_value = get_config().get(
        "first_token_timeout_seconds",
        FIRST_TOKEN_TIMEOUT_SECONDS,
    )

    if isinstance(raw_value, bool):
        return FIRST_TOKEN_TIMEOUT_SECONDS

    try:
        timeout_seconds = int(raw_value)
    except (TypeError, ValueError):
        return FIRST_TOKEN_TIMEOUT_SECONDS

    if timeout_seconds <= 0:
        return FIRST_TOKEN_TIMEOUT_SECONDS
    return timeout_seconds


class EmptyResponseError(RuntimeError):
    """Raised when the response stream ends without content."""


class FirstTokenTimeoutError(RuntimeError):
    """Raised when no first token arrives within the timeout window."""

    def __init__(
        self,
        message: str | None = None,
        *,
        timeout_seconds: int | None = None,
    ) -> None:
        """Initialize the timeout error with a default message."""
        self.timeout_seconds = timeout_seconds
        if message is not None:
            resolved_message = message
        elif timeout_seconds is not None:
            resolved_message = (
                f"No first token received within {timeout_seconds} seconds"
            )
        else:
            resolved_message = FIRST_RESPONSE_TIMEOUT_MESSAGE
        super().__init__(resolved_message)


def _raise_empty_response() -> NoReturn:
    """Raise EmptyResponseError."""
    raise EmptyResponseError(EMPTY_RESPONSE_MESSAGE)
