"""Attachment processing helpers for message handling."""
# ruff: noqa: E501

from __future__ import annotations

import asyncio
import io
import logging
import threading
from typing import TYPE_CHECKING

import discord
import pymupdf4llm
from PIL import Image
from twscrape import gather

if TYPE_CHECKING:
    from collections.abc import Iterable

    import httpx

    from .shared import TextDisplayComponentProtocol, TwitterApiProtocol

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - optional dependency
    fitz = None  # type: ignore[assignment]

try:
    import pymupdf.layout as pymupdf_layout
except ImportError:  # pragma: no cover - optional dependency
    pymupdf_layout = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


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
        except Exception as exc:  # noqa: BLE001
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


async def extract_pdf_text(pdf_content: bytes) -> str | None:
    """Extract text content from a PDF file using pymupdf4llm.

    This is used for non-Gemini models since they don't natively support PDF attachments.
    Runs in a thread pool since PyMuPDF operations are CPU-bound.

    Args:
        pdf_content: The raw PDF file bytes

    Returns:
        Extracted markdown text from the PDF, or None if extraction failed

    """
    def _extract() -> str | None:
        if fitz is None:
            return None
        try:
            # Open PDF from bytes (in-memory)
            doc = fitz.open(stream=pdf_content, filetype="pdf")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to open PDF: %s", exc)
            return None
        try:
            _ensure_pymupdf_layout_activated()
            md_text = pymupdf4llm.to_markdown(doc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to extract PDF text: %s", exc)
            return None
        else:
            return md_text
        finally:
            doc.close()

    try:
        # Run in thread pool with timeout to avoid blocking
        return await asyncio.wait_for(
            asyncio.to_thread(_extract),
            timeout=30,  # 30 second timeout for large PDFs
        )
    except asyncio.TimeoutError:
        logger.warning("PDF extraction timed out")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("PDF extraction error: %s", exc)
        return None


async def fetch_tweet_with_replies(
    twitter_api: TwitterApiProtocol,
    tweet_id: int,
    max_replies: int = 0,
    *,
    include_url: bool = False,
    tweet_url: str = "",
) -> str | None:
    """Fetch a tweet and optionally its replies, returning formatted text.

    DRY: Consolidates the duplicated tweet fetching logic that appeared in:
    - Google Lens Twitter extraction
    - Main tweet URL processing

    Args:
        twitter_api: The twscrape API instance
        tweet_id: The tweet's ID
        max_replies: Maximum number of replies to fetch (0 = no replies)
        include_url: Whether to include the tweet URL in the output
        tweet_url: The tweet URL (used if include_url=True)

    Returns:
        Formatted tweet text or None if fetch failed

    """
    try:
        tweet = await asyncio.wait_for(twitter_api.tweet_details(tweet_id), timeout=10)

        # Handle edge case where tweet or user is None
        if not tweet or not tweet.user:
            return None

        username = tweet.user.username or "unknown"

        if include_url and tweet_url:
            tweet_text = f"\n--- Tweet from @{username} ({tweet_url}) ---\n{tweet.rawContent or ''}"
        else:
            tweet_text = f"Tweet from @{username}:\n{tweet.rawContent or ''}"

        if max_replies > 0:
            replies = await asyncio.wait_for(
                gather(twitter_api.tweet_replies(tweet_id, limit=max_replies)),
                timeout=10,
            )
            if replies:
                tweet_text += "\n\nReplies:" if include_url else "\nReplies:"
                for reply in replies:
                    if reply and reply.user:
                        reply_username = reply.user.username or "unknown"
                        tweet_text += f"\n- @{reply_username}: {reply.rawContent or ''}"

    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to fetch tweet %s: %s", tweet_id, exc)
        return None

    return tweet_text


async def download_attachments(
    httpx_client: httpx.AsyncClient,
    attachments: list[discord.Attachment],
) -> tuple[list[discord.Attachment], list[httpx.Response]]:
    """Download attachments and return successful pairs."""
    async def download_attachment(
        att: discord.Attachment,
    ) -> httpx.Response | None:
        try:
            return await httpx_client.get(att.url, timeout=60)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to download attachment %s: %s",
                att.filename,
                exc,
            )
            return None

    attachment_responses = await asyncio.gather(
        *[download_attachment(att) for att in attachments],
    )

    # Filter out failed downloads
    successful_pairs = [
        (att, resp)
        for att, resp in zip(
            attachments,
            attachment_responses,
            strict=False,
        )
        if resp is not None
    ]

    good_attachments = [pair[0] for pair in successful_pairs]
    good_responses = [pair[1] for pair in successful_pairs]
    return good_attachments, good_responses


def normalize_attachments(
    good_attachments: list[discord.Attachment],
    attachment_responses: list[httpx.Response],
) -> list[dict[str, object]]:
    """Normalize attachment content (e.g., convert GIFs) and extract text."""
    processed_attachments = []
    for att, resp in zip(
        good_attachments,
        attachment_responses,
        strict=False,
    ):
        content = resp.content
        content_type = att.content_type

        if content_type == "image/gif":
            try:
                with Image.open(io.BytesIO(content)) as img:
                    output = io.BytesIO()
                    img.save(output, format="PNG")
                    content = output.getvalue()
                    content_type = "image/png"
            except Exception:
                logger.exception("Error converting GIF to PNG")

        processed_attachments.append(
            {
                "content_type": content_type,
                "content": content,
                "text": (
                    resp.text
                    if content_type.startswith("text")
                    else None
                ),
            },
        )

    return processed_attachments


def build_node_text_parts(
    cleaned_content: str,
    embeds: Iterable[discord.Embed],
    components: Iterable[TextDisplayComponentProtocol],
    text_attachments: list[str] | None = None,
    extra_parts: list[str] | None = None,
) -> str:
    """Build node text from multiple content sources.

    DRY: Consolidates the duplicated text joining pattern that appeared twice
    in process_message for building curr_node.text.

    Args:
        cleaned_content: The cleaned message content
        embeds: List of Discord embeds
        components: List of Discord components
        text_attachments: Optional list of text attachment contents
        extra_parts: Optional list of additional text parts (transcripts, tweets, etc.)

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
