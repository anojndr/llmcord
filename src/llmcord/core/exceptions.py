"""Custom exceptions for llmcord logic."""

from typing import NoReturn

EMPTY_RESPONSE_MESSAGE = "Response stream ended with no content"
FIRST_RESPONSE_TIMEOUT_MESSAGE = "No first token received within timeout window"
FIRST_TOKEN_TIMEOUT_SECONDS = 60
GOOGLE_GEMINI_CLI_FIRST_TOKEN_TIMEOUT_SECONDS = 10
LITELLM_TIMEOUT_SECONDS = 60


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
