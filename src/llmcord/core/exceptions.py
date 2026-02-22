"""Custom exceptions for llmcord logic."""

from typing import NoReturn

EMPTY_RESPONSE_MESSAGE = "Response stream ended with no content"
LITELLM_TIMEOUT_SECONDS = 60


class EmptyResponseError(RuntimeError):
    """Raised when the response stream ends without content."""


def _raise_empty_response() -> NoReturn:
    """Raise EmptyResponseError."""
    raise EmptyResponseError(EMPTY_RESPONSE_MESSAGE)
