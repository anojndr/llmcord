"""Custom exceptions for llmcord logic."""

from typing import NoReturn

EMPTY_RESPONSE_MESSAGE = "Response stream ended with no content"
FIRST_RESPONSE_TIMEOUT_MESSAGE = "No first token received within timeout window"
FIRST_TOKEN_TIMEOUT_SECONDS = 60
LITELLM_TIMEOUT_SECONDS = 60


class EmptyResponseError(RuntimeError):
    """Raised when the response stream ends without content."""


class FirstTokenTimeoutError(RuntimeError):
    """Raised when no first token arrives within the timeout window."""

    def __init__(self, message: str | None = None) -> None:
        """Initialize the timeout error with a default message."""
        super().__init__(message or FIRST_RESPONSE_TIMEOUT_MESSAGE)


def _raise_empty_response() -> NoReturn:
    """Raise EmptyResponseError."""
    raise EmptyResponseError(EMPTY_RESPONSE_MESSAGE)
