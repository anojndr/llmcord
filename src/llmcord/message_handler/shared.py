"""Shared helpers and protocols for message handling."""
# ruff: noqa: E501

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn, Protocol

import discord
import tiktoken
from discord.ui import LayoutView, TextDisplay

from llmcord.config import EMBED_COLOR_INCOMPLETE, PROCESSING_MESSAGE

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

EMPTY_RESPONSE_MESSAGE = "Response stream ended with no content"
FIRST_TOKEN_TIMEOUT_MESSAGE = "No first token received within timeout window"  # noqa: S105


class EmptyResponseError(RuntimeError):
    """Raised when the response stream ends without content."""


class FirstTokenTimeoutError(RuntimeError):
    """Raised when no first token arrives within the timeout window."""

    def __init__(self, message: str | None = None) -> None:
        """Initialize the timeout error with a default message."""
        super().__init__(message or FIRST_TOKEN_TIMEOUT_MESSAGE)


class TweetUserProtocol(Protocol):
    """Protocol for tweet user objects."""

    username: str | None


class TweetProtocol(Protocol):
    """Protocol for tweet objects returned by the Twitter API wrapper."""

    user: TweetUserProtocol | None
    rawContent: str | None  # noqa: N815


class TwitterApiProtocol(Protocol):
    """Protocol for the Twitter API wrapper used by the bot."""

    async def tweet_details(self, tweet_id: int) -> TweetProtocol | None:
        """Return tweet details for a given tweet ID."""
        ...

    def tweet_replies(
        self,
        tweet_id: int,
        limit: int,
    ) -> AsyncIterator[TweetProtocol]:
        """Return an async stream of replies for a tweet."""
        ...


class TextDisplayComponentProtocol(Protocol):
    """Protocol for text display components."""

    type: discord.ComponentType
    content: str | None


# Pre-load tiktoken encoding at module load time to avoid first-message delay
# This shifts the ~1-2s loading cost from first message to bot startup
_tiktoken_encoding = tiktoken.get_encoding("o200k_base")


def _get_tiktoken_encoding() -> tiktoken.Encoding:
    """Get the pre-loaded tiktoken encoding."""
    return _tiktoken_encoding


def raise_empty_response() -> NoReturn:
    """Raise a standardized empty-response error."""
    raise EmptyResponseError(EMPTY_RESPONSE_MESSAGE)


def count_conversation_tokens(messages: list[dict[str, object]]) -> int:
    """Count tokens in the entire conversation using tiktoken."""
    try:
        # Use cached encoding for performance
        enc = _get_tiktoken_encoding()
        total_tokens = 0

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                # For multimodal content, count tokens in text parts only
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total_tokens += len(enc.encode(part.get("text", "")))
            elif isinstance(content, str):
                total_tokens += len(enc.encode(content))

            # Count role tokens (approximation)
            total_tokens += len(enc.encode(msg.get("role", "")))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return 0
    else:
        return total_tokens


def count_text_tokens(text: str) -> int:
    """Count tokens in a text string using tiktoken."""
    try:
        enc = _get_tiktoken_encoding()
        return len(enc.encode(text))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return 0


def _strip_trigger_prefix(content: str, bot_mention: str) -> str:
    """Remove bot mention and "at ai" prefix for command detection."""
    stripped = content.strip()
    if bot_mention and stripped.lower().startswith(bot_mention.lower()):
        stripped = stripped[len(bot_mention):].strip()
    if stripped.lower().startswith("at ai"):
        stripped = stripped[5:].strip()
    return stripped


def extract_research_command(
    content: str,
    bot_mention: str,
) -> tuple[str | None, str]:
    """Extract Tavily research command and query from user content."""
    stripped = _strip_trigger_prefix(content, bot_mention)
    lowered = stripped.lower()
    if lowered.startswith("researchpro"):
        return "pro", stripped[len("researchpro"):].strip()
    if lowered.startswith("researchmini"):
        return "mini", stripped[len("researchmini"):].strip()
    return None, stripped


def append_search_to_content(
    content: str | list[dict[str, object]],
    search_results: str,
) -> str | list[dict[str, object]]:
    """Append search results to message content, handling both string and multimodal formats.

    Args:
        content: Either a string or a list of content parts (for multimodal messages)
        search_results: The search results text to append

    Returns:
        The modified content with search results appended

    """
    if not search_results:
        return content

    if isinstance(content, list):
        # For multimodal content, append to the text part
        if text_part := _find_text_part(content):
            text_part["text"] += "\n\n" + search_results
        return content
    if content:
        return str(content) + "\n\n" + search_results
    return content


def replace_content_text(
    content: str | list[dict[str, object]],
    new_text: str,
) -> str | list[dict[str, object]]:
    """Replace text content in a message while preserving multimodal structure."""
    if isinstance(content, list):
        if text_part := _find_text_part(content):
            text_part["text"] = new_text
            return content
        content.append({"type": "text", "text": new_text})
        return content
    return new_text


def _find_text_part(content: list[dict[str, object]]) -> dict[str, object] | None:
    """Return the first text part in a multimodal content list, if any."""
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            return part
    return None


def strip_bot_mention(content: str, bot_mention: str) -> str:
    """Strip a leading bot mention from content if present."""
    if bot_mention and content.lower().startswith(bot_mention.lower()):
        return content[len(bot_mention):]
    return content


def is_gemini_file_type(content_type: str) -> bool:
    """Return True if the content type should be sent as a Gemini file part."""
    return content_type.startswith(("audio", "video")) or content_type == "application/pdf"


async def create_processing_message(
    new_msg: discord.Message,
    *,
    use_plain_responses: bool,
) -> discord.Message:
    """Create and return the initial processing message."""
    if use_plain_responses:
        return await new_msg.reply(
            view=LayoutView().add_item(TextDisplay(content=PROCESSING_MESSAGE)),
        )

    processing_embed = discord.Embed(description=PROCESSING_MESSAGE, color=EMBED_COLOR_INCOMPLETE)
    return await new_msg.reply(embed=processing_embed, silent=True)


def build_warning_fields(warnings: Iterable[str]) -> list[dict[str, object]]:
    """Convert warning strings into embed field dictionaries."""
    return [
        {"name": warning, "value": "", "inline": False}
        for warning in warnings
    ]


def safe_lower(text: str | None) -> str:
    """Lowercase text safely when it may be None."""
    return text.lower() if text else ""


def strip_prefix(text: str, prefix: str) -> str:
    """Strip a prefix from text if present."""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def ensure_str(value: object | None) -> str:
    """Return a string representation, defaulting to empty for None."""
    return "" if value is None else str(value)


DEFAULT_EMPTY_TEXT = ""


def extract_grounding_metadata(
    response_obj: object,
    choice_obj: object | None = None,
) -> object | None:
    """Extract grounding metadata from multiple response shapes."""
    grounding_metadata = None

    if hasattr(response_obj, "model_extra") and response_obj.model_extra:
        grounding_metadata = (
            response_obj.model_extra.get("vertex_ai_grounding_metadata")
            or response_obj.model_extra.get("google_grounding_metadata")
            or response_obj.model_extra.get("grounding_metadata")
            or response_obj.model_extra.get("groundingMetadata")
        )

    if not grounding_metadata and hasattr(response_obj, "grounding_metadata"):
        grounding_metadata = response_obj.grounding_metadata

    hidden_params = getattr(response_obj, "_hidden_params", None)
    if not grounding_metadata and hidden_params:
        grounding_metadata = (
            hidden_params.get("grounding_metadata")
            or hidden_params.get("google_grounding_metadata")
            or hidden_params.get("groundingMetadata")
        )

    if not grounding_metadata and choice_obj and hasattr(choice_obj, "grounding_metadata"):
        grounding_metadata = choice_obj.grounding_metadata

    return grounding_metadata
