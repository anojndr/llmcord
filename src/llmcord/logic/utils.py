"""Merged utility functions and patches."""

import json
import logging
import re
import threading
from collections.abc import Iterable, Iterator
from typing import Protocol

import discord
import tiktoken
from twscrape import xclid  # type: ignore[import-untyped]

try:
    import pymupdf.layout as pymupdf_layout  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional dependency
    pymupdf_layout = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# =============================================================================
# Twscrape Patch (formerly logic/utils.py)
# =============================================================================


def script_url(k: str, v: str) -> str:
    """Generate Twitter script URL."""
    return f"https://abs.twimg.com/responsive-web/client-web/{k}.{v}.js"


def patched_get_scripts_list(text: str) -> Iterator[str]:
    """Patched function for twscrape to handle script parsing."""
    try:
        scripts = text.split('e=>e+"."+')[1].split('[e]+"a.js"')[0]
    except IndexError:
        logger.warning("Failed to extract scripts from Twitter page.")
        return

    try:
        for k, v in json.loads(scripts).items():
            yield script_url(k, f"{v}a")
    except json.decoder.JSONDecodeError:
        try:
            fixed_scripts = re.sub(
                r"([,\{])(\s*)([\w]+_[\w_]+)(\s*):",
                r'\1\2"\3"\4:',
                scripts,
            )
            for k, v in json.loads(fixed_scripts).items():
                yield script_url(k, f"{v}a")
        except json.JSONDecodeError:
            logger.warning("Failed to parse fixed scripts JSON.")


# Apply the patch
xclid.get_scripts_list = patched_get_scripts_list


# =============================================================================
# Message Processing Helpers (formerly logic/helpers.py)
# =============================================================================

# Pre-load tiktoken encoding at module load time to avoid first-message delay
# This shifts the ~1-2s loading cost from first message to bot startup
_tiktoken_encoding: tiktoken.Encoding | None
try:
    _tiktoken_encoding = tiktoken.get_encoding("o200k_base")
except (KeyError, RuntimeError, ValueError):
    logger.exception("Failed to load tiktoken encoding.")
    _tiktoken_encoding = None


def _get_tiktoken_encoding() -> tiktoken.Encoding | None:
    """Get the pre-loaded tiktoken encoding."""
    return _tiktoken_encoding


_pymupdf_layout_activation_lock = threading.Lock()
_pymupdf_layout_state: dict[str, bool | None] = {"activated": None}


def _ensure_pymupdf_layout_activated() -> bool:
    """Activate PyMuPDF Layout if available, returning activation state."""
    cached = _pymupdf_layout_state["activated"]
    if cached is not None:
        return cached

    with _pymupdf_layout_activation_lock:
        cached = _pymupdf_layout_state["activated"]
        if cached is not None:
            return cached

        if pymupdf_layout is None:
            _pymupdf_layout_state["activated"] = False
            return False
        try:
            pymupdf_layout.activate()
        except (AttributeError, RuntimeError) as exc:
            _pymupdf_layout_state["activated"] = False
            logger.debug("PyMuPDF Layout not available: %s", exc)
            return False

        _pymupdf_layout_state["activated"] = True
        return True


def _get_embed_text(embed: discord.Embed) -> str:
    """Safely extract text content from a Discord embed, handling None values.

    Note: Footer text is intentionally excluded as it contains metadata
    (model name, token count) that should not be sent to the LLM.
    """
    parts = [embed.title, embed.description]
    return "\n".join(filter(None, parts))


class TextDisplayComponentProtocol(Protocol):
    """Protocol for text display components."""

    type: discord.ComponentType
    content: str | None


def build_node_text_parts(
    cleaned_content: str,
    embeds: Iterable[discord.Embed],
    components: Iterable[TextDisplayComponentProtocol],
    text_attachments: list[str] | None = None,
    extra_parts: list[str] | None = None,
) -> str:
    """Build node text from multiple content sources.

    DRY: Consolidates the duplicated text joining pattern.

    Args:
        cleaned_content: The cleaned message content
        embeds: List of Discord embeds
        components: List of Discord components
        text_attachments: Optional list of text attachment contents
        extra_parts: Optional list of additional text parts (transcripts,
            tweets, etc.)

    Returns:
        Joined text content

    """
    parts = []

    if cleaned_content:
        parts.append(cleaned_content)

    # Add embed text
    for embed in embeds:
        embed_text = _get_embed_text(embed)
        if embed_text:
            parts.append(embed_text)

    # Add text display components
    parts.extend(
        component.content
        for component in components
        if component.type == discord.ComponentType.text_display and component.content
    )

    # Add text attachments
    if text_attachments:
        parts.extend(text_attachments)

    # Add extra parts (transcripts, tweets, reddit posts, etc.)
    if extra_parts:
        parts.extend(extra_parts)

    return "\n".join(parts)


def append_search_to_content(
    content: str | list[dict[str, object]],
    search_results: str,
) -> str | list[dict[str, object]]:
    """Append search results to message content.

    Handles both string and multimodal formats.

    Args:
        content: Either a string or a list of content parts (for multimodal
            messages)
        search_results: The search results text to append

    Returns:
        The modified content with search results appended

    """
    if not search_results:
        return content

    if isinstance(content, list):
        # For multimodal content, append to the text part
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                new_text = str(part.get("text", ""))
                part["text"] = f"{new_text}\n\n{search_results}"
                break
        return content
    if content:
        return str(content) + "\n\n" + search_results
    return content


def replace_content_text(
    content: str | list[dict[str, object]],
    new_text: str,
) -> str | list[dict[str, object]]:
    """Replace text content while preserving multimodal structure."""
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = new_text
                return content
        content.append({"type": "text", "text": new_text})
        return content
    return new_text


def _strip_trigger_prefix(content: str, bot_mention: str) -> str:
    """Remove bot mention and "at ai" prefix for command detection."""
    stripped = content.strip()
    if bot_mention and stripped.lower().startswith(bot_mention.lower()):
        stripped = stripped[len(bot_mention) :].strip()
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
        return "pro", stripped[len("researchpro") :].strip()
    if lowered.startswith("researchmini"):
        return "mini", stripped[len("researchmini") :].strip()
    return None, stripped


def _count_msg_part_tokens(part: object, enc: tiktoken.Encoding) -> int:
    """Count tokens in a single message part."""
    if isinstance(part, dict) and part.get("type") == "text":
        text = part.get("text", "")
        if isinstance(text, str):
            return len(enc.encode(text))
    return 0


def count_conversation_tokens(messages: list[dict[str, object]]) -> int:
    """Count tokens in the entire conversation using tiktoken."""
    try:
        # Use cached encoding for performance
        enc = _get_tiktoken_encoding()
        if not enc:
            return 0
        total_tokens = 0

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                # For multimodal content, count tokens in text parts only
                for part in content:
                    total_tokens += _count_msg_part_tokens(part, enc)
            elif isinstance(content, str):
                total_tokens += len(enc.encode(content))

            # Count role tokens (approximation)
            role = msg.get("role", "")
            if isinstance(role, str):
                total_tokens += len(enc.encode(role))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return 0
    else:
        return total_tokens


def count_text_tokens(text: str) -> int:
    """Count tokens in a text string using tiktoken."""
    try:
        enc = _get_tiktoken_encoding()
        if not enc:
            return 0
        return len(enc.encode(text))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return 0
