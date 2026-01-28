"""Message handling logic for llmcord Discord bot.

Uses LiteLLM for unified LLM API access via shared litellm_utils.
"""
# ruff: noqa: E501
import asyncio
import io
import logging
import re
import threading
from base64 import b64encode
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from datetime import datetime, timezone
from typing import NoReturn, Protocol

import asyncpraw
import discord
import httpx
import litellm
import pymupdf4llm
import tiktoken
from bs4 import BeautifulSoup
from discord.ui import LayoutView, TextDisplay
from PIL import Image
from twscrape import gather
from youtube_transcript_api import YouTubeTranscriptApi

from bad_keys import get_bad_keys_db
from config import (
    BROWSER_HEADERS,
    EDIT_DELAY_SECONDS,
    EMBED_COLOR_COMPLETE,
    EMBED_COLOR_INCOMPLETE,
    MAX_MESSAGE_NODES,
    PROCESSING_MESSAGE,
    PROVIDERS_SUPPORTING_USERNAMES,
    STREAMING_INDICATOR,
    VISION_MODEL_TAGS,
    ensure_list,
    get_config,
    is_gemini_model,
)
from litellm_utils import LiteLLMOptions, prepare_litellm_kwargs
from models import MsgNode
from views import (
    ResponseView,
    SourceButton,
    SourceView,
    TavilySourceButton,
    _has_grounding_data,
)
from web_search import (
    decide_web_search,
    get_current_datetime_strings,
    perform_tavily_research,
    perform_web_search,
)

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - optional dependency
    fitz = None  # type: ignore[assignment]

try:
    import pymupdf.layout as pymupdf_layout
except ImportError:  # pragma: no cover - optional dependency
    pymupdf_layout = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

EMPTY_RESPONSE_MESSAGE = "Response stream ended with no content"
FIRST_TOKEN_TIMEOUT_MESSAGE = "No first token received within timeout window"  # noqa: S105
MAX_OVERLOADED_ERRORS = 3
LITELLM_TIMEOUT_SECONDS = 60
FIRST_TOKEN_TIMEOUT_SECONDS = 60


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


def _raise_empty_response() -> NoReturn:
    raise EmptyResponseError(EMPTY_RESPONSE_MESSAGE)

# Pre-load tiktoken encoding at module load time to avoid first-message delay
# This shifts the ~1-2s loading cost from first message to bot startup
_tiktoken_encoding = tiktoken.get_encoding("o200k_base")

def _get_tiktoken_encoding() -> tiktoken.Encoding:
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


# =============================================================================
# DRY Helper Functions for Message Processing
# =============================================================================

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
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] += "\n\n" + search_results
                break
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


async def process_message(  # noqa: C901, PLR0912, PLR0913, PLR0915
    new_msg: discord.Message,
    discord_bot: discord.Client,
    httpx_client: httpx.AsyncClient,
    twitter_api: TwitterApiProtocol,
    reddit_client: asyncpraw.Reddit | None,
    msg_nodes: dict[int, MsgNode],
    curr_model_lock: asyncio.Lock,
    curr_model_ref: list[str],
) -> None:
    """Process a message."""
    # Per-request edit timing to avoid interference between concurrent requests
    last_edit_time = 0

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (
        not is_dm
        and discord_bot.user not in new_msg.mentions
        and "at ai" not in new_msg.content.lower()
    ) or new_msg.author.bot:
        return

    role_ids = {role.id for role in getattr(new_msg.author, "roles", ())}
    channel_ids = {
        channel_id
        for channel_id in (
            new_msg.channel.id,
            getattr(new_msg.channel, "parent_id", None),
            getattr(new_msg.channel, "category_id", None),
        )
        if channel_id is not None
    }

    config = get_config()  # Now cached, no need for to_thread

    allow_dms = config.get("allow_dms", True)

    permissions = config["permissions"]

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    # Pre-convert to sets once for efficient lookups
    allowed_user_ids = set(permissions["users"]["allowed_ids"])
    blocked_user_ids = set(permissions["users"]["blocked_ids"])
    allowed_role_ids = set(permissions["roles"]["allowed_ids"])
    blocked_role_ids = set(permissions["roles"]["blocked_ids"])
    allowed_channel_ids = set(permissions["channels"]["allowed_ids"])
    blocked_channel_ids = set(permissions["channels"]["blocked_ids"])

    allow_all_users = (
        not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    )
    is_good_user = (
        user_is_admin
        or allow_all_users
        or new_msg.author.id in allowed_user_ids
        or bool(role_ids & allowed_role_ids)
    )
    is_bad_user = (
        not is_good_user
        or new_msg.author.id in blocked_user_ids
        or bool(role_ids & blocked_role_ids)
    )

    allow_all_channels = not allowed_channel_ids
    is_good_channel = (
        user_is_admin or allow_dms
        if is_dm
        else allow_all_channels or bool(channel_ids & allowed_channel_ids)
    )
    is_bad_channel = not is_good_channel or bool(channel_ids & blocked_channel_ids)

    if is_bad_user or is_bad_channel:
        return

    # Send processing message immediately after confirming bot should respond
    use_plain_responses = config.get("use_plain_responses", False)

    if use_plain_responses:
        processing_msg = await new_msg.reply(
            view=LayoutView().add_item(TextDisplay(content=PROCESSING_MESSAGE)),
        )
    else:
        processing_embed = discord.Embed(description=PROCESSING_MESSAGE, color=EMBED_COLOR_INCOMPLETE)
        processing_msg = await new_msg.reply(embed=processing_embed, silent=True)

    # Thread-safe read of current model
    async with curr_model_lock:
        provider_slash_model = curr_model_ref[0]

    # Validate provider/model format
    try:
        provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)
    except ValueError:
        logger.exception(
            "Invalid model format: %s. Expected 'provider/model'.",
            provider_slash_model,
        )
        embed = discord.Embed(
            description=(
                "❌ Invalid model configuration: "
                f"'{provider_slash_model}'. Expected format: 'provider/model'.\n"
                "Please contact an administrator."
            ),
            color=EMBED_COLOR_INCOMPLETE,
        )
        await processing_msg.edit(embed=embed)
        return

    # Validate provider exists in config
    providers = config.get("providers", {})
    if provider not in providers:
        logger.error("Provider '%s' not found in config.", provider)
        embed = discord.Embed(
            description=(
                f"❌ Provider '{provider}' is not configured. "
                "Please contact an administrator."
            ),
            color=EMBED_COLOR_INCOMPLETE,
        )
        await processing_msg.edit(embed=embed)
        return

    provider_config = providers[provider]

    base_url = provider_config.get("base_url")
    api_keys = ensure_list(provider_config.get("api_key")) or ["sk-no-key-required"]

    model_parameters = config["models"].get(provider_slash_model, None)

    # Support model aliasing: if config specifies a different actual model name, use it
    # This allows e.g. "gemini-3-flash-minimal" to map to "gemini-3-flash-preview"
    actual_model = model_parameters.get("model", model) if model_parameters else model

    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (
        (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None
    )

    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(
        provider_slash_model.lower().startswith(x)
        for x in PROVIDERS_SUPPORTING_USERNAMES
    )

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)
    max_tweet_replies = config.get("max_tweet_replies", 50)

    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg is not None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text is None:
                cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()
                cleaned_content = re.sub(r"\bat ai\b", "", cleaned_content, flags=re.IGNORECASE).lstrip()

                if cleaned_content.lower().startswith("googlelens"):
                    cleaned_content = cleaned_content[10:].strip()

                    # Check for cached lens results first
                    _, _, cached_lens_results = get_bad_keys_db().get_message_search_data(curr_msg.id)
                    if cached_lens_results:
                        # Use cached lens results
                        cleaned_content = cleaned_content + cached_lens_results
                        curr_node.lens_results = cached_lens_results
                        logger.debug(
                            "Using cached lens results for message %s",
                            curr_msg.id,
                        )
                    elif image_url := next(
                        (
                            att.url
                            for att in curr_msg.attachments
                            if att.content_type
                            and att.content_type.startswith("image")
                        ),
                        None,
                    ):
                        try:
                            # Use shared browser headers (DRY)
                            params = {
                                "rpt": "imageview",
                                "url": image_url,
                            }

                            yandex_resp = await httpx_client.get(
                                "https://yandex.com/images/search",
                                params=params,
                                headers=BROWSER_HEADERS,
                                follow_redirects=True,
                                timeout=60,
                            )
                            soup = BeautifulSoup(
                                yandex_resp.text,
                                "lxml",
                            )  # lxml is faster than html.parser

                            lens_results = []
                            twitter_urls_found = []
                            sites_items = soup.select(".CbirSites-Item")

                            if sites_items:
                                for item in sites_items:
                                    title_el = item.select_one(".CbirSites-ItemTitle a")
                                    domain_el = item.select_one(".CbirSites-ItemDomain")
                                    desc_el = item.select_one(
                                        ".CbirSites-ItemDescription",
                                    )

                                    title = (
                                        title_el.get_text(strip=True)
                                        if title_el
                                        else "N/A"
                                    )
                                    link = title_el["href"] if title_el else "#"
                                    domain = (
                                        domain_el.get_text(strip=True)
                                        if domain_el
                                        else ""
                                    )
                                    desc = (
                                        desc_el.get_text(strip=True)
                                        if desc_el
                                        else ""
                                    )

                                    lens_results.append(
                                        f"- [{title}]({link}) ({domain}) - {desc}",
                                    )

                                    # Check if the link is a Twitter/X URL and extract for later processing
                                    if re.search(
                                        r"(?:twitter\.com|x\.com)/[a-zA-Z0-9_]+/status/[0-9]+",
                                        link,
                                    ):
                                        twitter_urls_found.append(link)

                            # Extract tweet content from Twitter/X URLs found in Yandex results (using DRY helper)
                            twitter_content = []
                            if twitter_urls_found:
                                for twitter_url in twitter_urls_found:
                                    tweet_id_match = re.search(
                                        r"(?:twitter\.com|x\.com)/[a-zA-Z0-9_]+/status/([0-9]+)",
                                        twitter_url,
                                    )
                                    if tweet_id_match:
                                        tweet_text = await fetch_tweet_with_replies(
                                            twitter_api,
                                            int(tweet_id_match.group(1)),
                                            max_replies=max_tweet_replies,
                                            include_url=True,
                                            tweet_url=twitter_url,
                                        )
                                        if tweet_text:
                                            twitter_content.append(tweet_text)

                            if lens_results:
                                result_text = (
                                    "\n\nanswer the user's query based on the yandex reverse image results:\n"
                                    + "\n".join(lens_results)
                                )
                                if twitter_content:
                                    result_text += (
                                        "\n\n--- Extracted Twitter/X Content ---"
                                        + "".join(twitter_content)
                                    )
                                cleaned_content += result_text

                                # Store lens results for persistence
                                curr_node.lens_results = result_text

                                # Save lens results to database for persistence in chat history
                                get_bad_keys_db().save_message_search_data(
                                    curr_msg.id,
                                    lens_results=result_text,
                                )
                                logger.info(
                                    "Saved lens results for message %s",
                                    curr_msg.id,
                                )
                        except Exception:
                            logger.exception("Error fetching Yandex results")

                allowed_types = (
                    "text",
                    "image",
                    "application/pdf",
                )  # PDF always allowed - Gemini uses native, others get text extraction
                if is_gemini_model(actual_model):
                    allowed_types += ("audio", "video")

                good_attachments = [
                    att
                    for att in curr_msg.attachments
                    if att.content_type
                    and any(att.content_type.startswith(x) for x in allowed_types)
                ]

                # Download attachments with timeout and error handling
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
                    *[download_attachment(att) for att in good_attachments],
                )

                # Filter out failed downloads
                successful_pairs = [
                    (att, resp)
                    for att, resp in zip(
                        good_attachments,
                        attachment_responses,
                        strict=False,
                    )
                    if resp is not None
                ]
                good_attachments = [pair[0] for pair in successful_pairs]
                attachment_responses = [pair[1] for pair in successful_pairs]

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

                # Initial text building before async content fetches (using DRY helper)
                curr_node.text = build_node_text_parts(
                    cleaned_content,
                    curr_msg.embeds,
                    curr_msg.components,
                    text_attachments=[
                        att["text"]
                        for att in processed_attachments
                        if att["content_type"].startswith("text") and att["text"]
                    ],
                )

                curr_node.images = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{att['content_type']};base64,"
                                f"{b64encode(att['content']).decode('utf-8')}"
                            ),
                        },
                    }
                    for att in processed_attachments
                    if att["content_type"].startswith("image")
                ]

                curr_node.raw_attachments = [
                    {"content_type": att["content_type"], "content": att["content"]}
                    for att in processed_attachments
                ]

                # Fetch YouTube transcripts in parallel for better performance
                async def fetch_yt_transcript(video_id: str) -> str | None:
                    try:
                        ytt_api = YouTubeTranscriptApi()
                        transcript_obj = await asyncio.to_thread(ytt_api.fetch, video_id)
                        transcript = transcript_obj.to_raw_data()

                        response = await httpx_client.get(f"https://www.youtube.com/watch?v={video_id}", follow_redirects=True, timeout=30)
                        html = response.text
                        title_match = re.search(r'<meta name="title" content="(.*?)">', html)
                        title = title_match.group(1) if title_match else "Unknown Title"
                        channel_match = re.search(r'<link itemprop="name" content="(.*?)">', html)
                        channel = channel_match.group(1) if channel_match else "Unknown Channel"

                        return f"YouTube Video ID: {video_id}\nTitle: {title}\nChannel: {channel}\nTranscript:\n" + " ".join(x["text"] for x in transcript)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Failed to fetch YouTube transcript: %s", exc)
                        return None

                video_ids = re.findall(
                    r"(?:https?:\/\/)?(?:[a-zA-Z0-9-]+\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})",
                    cleaned_content,
                )
                if video_ids:
                    yt_results = await asyncio.gather(
                        *[fetch_yt_transcript(vid) for vid in video_ids],
                    )
                    yt_transcripts = [t for t in yt_results if t is not None]
                else:
                    yt_transcripts = []

                # Fetch tweets using DRY helper function
                tweets = []
                for tweet_id in re.findall(
                    r"(?:https?:\/\/)?(?:[a-zA-Z0-9-]+\.)?(?:twitter\.com|x\.com)\/[a-zA-Z0-9_]+\/status\/([0-9]+)",
                    cleaned_content,
                ):
                    tweet_text = await fetch_tweet_with_replies(
                        twitter_api,
                        int(tweet_id),
                        max_replies=max_tweet_replies,
                        include_url=False,
                    )
                    if tweet_text:
                        tweets.append(tweet_text)

                reddit_posts = []
                if reddit_client:
                    for post_url in re.findall(
                        r"(https?:\/\/(?:[a-zA-Z0-9-]+\.)?(?:reddit\.com\/r\/[a-zA-Z0-9_]+\/comments\/[a-zA-Z0-9_]+(?:[\w\-\.\/\?\=\&%]*)|redd\.it\/[a-zA-Z0-9_]+))",
                        cleaned_content,
                    ):
                        try:
                            submission = await reddit_client.submission(url=post_url)

                            # Handle edge case where submission is missing key attributes
                            if not submission:
                                continue

                            # Safely access attributes with defaults
                            title = getattr(submission, "title", "Untitled")
                            subreddit_name = (
                                submission.subreddit.display_name
                                if submission.subreddit
                                else "unknown"
                            )
                            author_name = (
                                submission.author.name
                                if submission.author
                                else "[deleted]"
                            )
                            selftext = getattr(submission, "selftext", "") or ""

                            post_text = (
                                "Reddit Post: "
                                f"{title}\nSubreddit: r/{subreddit_name}\n"
                                f"Author: u/{author_name}\n\n{selftext}"
                            )

                            if not getattr(submission, "is_self", True) and getattr(submission, "url", None):
                                post_text += f"\nLink: {submission.url}"

                            submission.comment_sort = "top"
                            await submission.comments()
                            comments_list = (
                                submission.comments.list()
                                if submission.comments
                                else []
                            )
                            top_comments = comments_list[:5]

                            if top_comments:
                                post_text += "\n\nTop Comments:"
                                for comment in top_comments:
                                    if isinstance(comment, asyncpraw.models.MoreComments):
                                        continue
                                    comment_author = (
                                        comment.author.name
                                        if comment.author
                                        else "[deleted]"
                                    )
                                    comment_body = getattr(comment, "body", "") or ""
                                    post_text += f"\n- u/{comment_author}: {comment_body}"

                            reddit_posts.append(post_text)
                        except Exception as exc:  # noqa: BLE001
                            logger.debug(
                                "Failed to fetch Reddit content for %s: %s",
                                post_url,
                                exc,
                            )

                # PDF text extraction for non-Gemini models
                # Genuine Gemini models handle PDFs natively, but Gemma and other providers need text extraction
                pdf_texts = []
                if not is_gemini_model(actual_model):
                    pdf_attachments = [
                        att for att in processed_attachments
                        if att["content_type"] == "application/pdf"
                    ]
                    if pdf_attachments:
                        # Extract text from PDFs in parallel
                        pdf_extraction_tasks = [
                            extract_pdf_text(att["content"])
                            for att in pdf_attachments
                        ]
                        pdf_results = await asyncio.gather(*pdf_extraction_tasks)

                        for i, pdf_text in enumerate(pdf_results):
                            if pdf_text:
                                # Format PDF content nicely for the LLM
                                pdf_texts.append(
                                    f"--- PDF Attachment {i + 1} Content ---\n{pdf_text}",
                                )
                                logger.info(
                                    "Extracted text from PDF attachment (%s chars)",
                                    len(pdf_text),
                                )
                            else:
                                logger.warning(
                                    "Could not extract text from PDF attachment %s",
                                    i + 1,
                                )

                # Final text building with all async content (using DRY helper)
                curr_node.text = build_node_text_parts(
                    cleaned_content,
                    curr_msg.embeds,
                    curr_msg.components,
                    text_attachments=[
                        resp.text
                        for att, resp in zip(
                            good_attachments,
                            attachment_responses,
                            strict=False,
                        )
                        if att.content_type.startswith("text") and resp.text
                    ],
                    extra_parts=yt_transcripts + tweets + reddit_posts + pdf_texts,
                )

                if not curr_node.text and curr_node.images:
                    curr_node.text = "What is in this image?"

                if (
                    not curr_node.text
                    and not curr_node.images
                    and not curr_node.raw_attachments
                    and curr_msg == new_msg
                ):
                    curr_node.text = "Hello"

                curr_node.role = (
                    "assistant"
                    if curr_msg.author == discord_bot.user
                    else "user"
                )

                curr_node.user_id = (
                    curr_msg.author.id if curr_node.role == "user" else None
                )

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(
                    good_attachments,
                )

                try:
                    if (
                        curr_msg.reference is None
                        and discord_bot.user.mention not in curr_msg.content
                        and "at ai" not in curr_msg.content.lower()
                        and (
                            prev_msg_in_channel := (
                                [
                                    m
                                    async for m in curr_msg.channel.history(
                                        before=curr_msg,
                                        limit=1,
                                    )
                                ]
                                or [None]
                            )[0]
                        )
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author
                        == (
                            discord_bot.user
                            if curr_msg.channel.type == discord.ChannelType.private
                            else curr_msg.author
                        )
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = (
                            is_public_thread
                            and curr_msg.reference is None
                            and curr_msg.channel.parent.type == discord.ChannelType.text
                        )

                        if parent_msg_id := (
                            curr_msg.channel.id
                            if parent_is_thread_start
                            else getattr(curr_msg.reference, "message_id", None)
                        ):
                            if parent_is_thread_start:
                                curr_node.parent_msg = (
                                    curr_msg.channel.starter_message
                                    or await curr_msg.channel.parent.fetch_message(
                                        parent_msg_id,
                                    )
                                )
                            else:
                                curr_node.parent_msg = (
                                    curr_msg.reference.cached_message
                                    or await curr_msg.channel.fetch_message(
                                        parent_msg_id,
                                    )
                                )

                except (discord.NotFound, discord.HTTPException):
                    logger.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            # Load stored search data from database if not already cached
            # This is outside the text==None block so it runs even for cached nodes
            if curr_node.search_results is None or curr_node.lens_results is None:
                (
                    stored_search_results,
                    stored_tavily_metadata,
                    stored_lens_results,
                ) = get_bad_keys_db().get_message_search_data(curr_msg.id)
                if stored_search_results and curr_node.search_results is None:
                    curr_node.search_results = stored_search_results
                    curr_node.tavily_metadata = stored_tavily_metadata
                    logger.info(
                        "Loaded stored search data for message %s",
                        curr_msg.id,
                    )
                if stored_lens_results and curr_node.lens_results is None:
                    curr_node.lens_results = stored_lens_results
                    logger.info(
                        "Loaded stored lens results for message %s",
                        curr_msg.id,
                    )

            # Build message content (now unified for all providers via LiteLLM)
            # Check if there are Gemini-specific file attachments (audio, video, PDF)
            gemini_file_attachments = []
            if is_gemini_model(actual_model) and curr_node.raw_attachments:
                for att in curr_node.raw_attachments:
                    if att["content_type"].startswith(
                        ("audio", "video"),
                    ) or att["content_type"] == "application/pdf":
                        # LiteLLM supports inline data format for Gemini
                        encoded_data = b64encode(att["content"]).decode("utf-8")
                        gemini_file_attachments.append(
                            {
                                "type": "file",
                                "file": {
                                    "file_data": (
                                        f"data:{att['content_type']};base64,{encoded_data}"
                                    ),
                                },
                            },
                        )

            # Determine if we need multimodal content format
            has_images = bool(curr_node.images[:max_images])
            has_gemini_files = bool(gemini_file_attachments)

            if has_images or has_gemini_files:
                # Build multimodal content array
                content = []

                # Add text part if present
                if curr_node.text[:max_text]:
                    content.append(
                        {"type": "text", "text": curr_node.text[:max_text]},
                    )

                # Add images
                content.extend(curr_node.images[:max_images])

                # Add Gemini file attachments (audio, video, PDF)
                content.extend(gemini_file_attachments)
                if gemini_file_attachments:
                    logger.info(
                        "Added %s Gemini file attachment(s) (audio/video/PDF) to message",
                        len(gemini_file_attachments),
                    )

                # Ensure we have at least some content
                if not content:
                    content = [{"type": "text", "text": "What is in this file?"}]
            else:
                content = curr_node.text[:max_text]

            # Include stored search results from history messages
            if curr_node.search_results and curr_node.role == "user":
                content = append_search_to_content(content, curr_node.search_results)

            if content:
                message = {"content": content, "role": curr_node.role}
                if accept_usernames and curr_node.user_id is not None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)

            if curr_node.text and len(curr_node.text) > max_text:
                user_warnings.add(
                    f"⚠️ Max {max_text:,} characters per message",
                )
            if len(curr_node.images) > max_images:
                user_warnings.add(
                    (
                        f"⚠️ Max {max_images} image"
                        f"{'' if max_images == 1 else 's'} per message"
                    )
                    if max_images > 0
                    else "⚠️ Can't see images",
                )
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg is not None and len(messages) == max_messages):
                user_warnings.add(
                    f"⚠️ Only using last {len(messages)} message"
                    f"{'' if len(messages) == 1 else 's'}",
                )

            curr_msg = curr_node.parent_msg

    logger.info(
        "Message received (user ID: %s, attachments: %s, conversation length: %s):\n%s",
        new_msg.author.id,
        len(new_msg.attachments),
        len(messages),
        new_msg.content,
    )

    # Handle edge case: no valid messages could be built
    if not messages:
        logger.warning("No valid messages could be built from the conversation.")
        embed = discord.Embed(
            description="❌ Could not process your message. Please try again.",
            color=EMBED_COLOR_INCOMPLETE,
        )
        await processing_msg.edit(embed=embed)
        return

    system_prompt = config.get("system_prompt")

    if system_prompt:
        date_str, time_str = get_current_datetime_strings()

        system_prompt = (
            system_prompt.replace("{date}", date_str)
            .replace("{time}", time_str)
            .strip()
        )
        if accept_usernames:
            system_prompt += (
                "\n\nUser's names are their Discord IDs and should be typed as '<@ID>'."
            )

        messages.append({"role": "system", "content": system_prompt})

    # Web Search Integration for non-Gemini models and Gemini preview models
    # Supports both Tavily and Exa MCP as search providers
    tavily_api_keys = ensure_list(config.get("tavily_api_key"))
    exa_mcp_url = config.get("exa_mcp_url", "")
    web_search_provider = config.get(
        "web_search_provider",
        "tavily",
    )  # Default to Tavily

    # Determine if web search is available based on provider config
    web_search_available = False
    if (web_search_provider == "tavily" and tavily_api_keys) or (
        web_search_provider == "exa" and exa_mcp_url
    ):
        web_search_available = True
    elif web_search_provider == "auto":
        # Auto-detect: prefer Tavily if keys are available, otherwise try Exa
        if tavily_api_keys:
            web_search_provider = "tavily"
            web_search_available = True
        elif exa_mcp_url:
            web_search_provider = "exa"
            web_search_available = True

    is_preview_model = "preview" in actual_model.lower()
    is_non_gemini = not is_gemini_model(actual_model)
    # Check for googlelens query - strip both mention and "at ai" prefix
    content_for_lens_check = (
        new_msg.content.lower()
        .removeprefix(discord_bot.user.mention.lower())
        .strip()
    )
    if content_for_lens_check.startswith("at ai"):
        content_for_lens_check = content_for_lens_check[5:].strip()
    is_googlelens_query = content_for_lens_check.startswith("googlelens")

    search_metadata = None
    research_model, research_query = extract_research_command(
        new_msg.content,
        discord_bot.user.mention if discord_bot.user else "",
    )

    # Check for existing search results to handle retries correctly
    if new_msg.id in msg_nodes and msg_nodes[new_msg.id].search_results:
        search_metadata = msg_nodes[new_msg.id].tavily_metadata
        has_existing_search = True
    else:
        has_existing_search = False

    if research_model and not has_existing_search:
        if not research_query:
            user_warnings.add(
                "⚠️ Provide a research query after researchpro/researchmini",
            )
        elif not tavily_api_keys:
            user_warnings.add("⚠️ Tavily API key missing for research")
        else:
            for msg in messages:
                if msg.get("role") == "user":
                    msg["content"] = replace_content_text(
                        msg.get("content", ""),
                        research_query,
                    )
                    break

            research_results, search_metadata = await perform_tavily_research(
                query=research_query,
                api_keys=tavily_api_keys,
                model=research_model,
            )

            if research_results:
                for msg in messages:
                    if msg.get("role") == "user":
                        msg["content"] = append_search_to_content(
                            msg.get("content", ""),
                            research_results,
                        )

                        logger.info(
                            "Tavily research results appended to user message",
                        )

                        get_bad_keys_db().save_message_search_data(
                            new_msg.id,
                            research_results,
                            search_metadata,
                        )

                        if new_msg.id in msg_nodes:
                            msg_nodes[new_msg.id].search_results = research_results
                            msg_nodes[new_msg.id].tavily_metadata = search_metadata
                        break

    if (
        not research_model
        and web_search_available
        and (is_non_gemini or is_preview_model)
        and not is_googlelens_query
        and not has_existing_search
    ):
        # Get web search decider model - first check user preference, then config default
        db = get_bad_keys_db()
        user_id = str(new_msg.author.id)
        user_decider_model = db.get_user_search_decider_model(user_id)
        default_decider = config.get(
            "web_search_decider_model",
            "gemini/gemini-3-flash-preview",
        )

        # Use user preference if set and valid, otherwise use config default
        if user_decider_model and user_decider_model in config.get("models", {}):
            decider_model_str = user_decider_model
        else:
            decider_model_str = default_decider

        decider_provider, decider_model = (
            decider_model_str.split("/", 1)
            if "/" in decider_model_str
            else ("gemini", decider_model_str)
        )

        # Get provider config for the decider
        decider_provider_config = config.get("providers", {}).get(decider_provider, {})
        decider_api_keys = ensure_list(decider_provider_config.get("api_key"))

        decider_config = {
            "provider": decider_provider,
            "model": decider_model,
            "api_keys": decider_api_keys,
            "base_url": decider_provider_config.get("base_url"),
        }

        if decider_api_keys:
            # Run the search decider with chat history (uses all keys with rotation)
            search_decision = await decide_web_search(messages, decider_config)

            if search_decision.get("needs_search") and search_decision.get("queries"):
                queries = search_decision["queries"]
                logger.info(
                    "Web search triggered with %s. Queries: %s",
                    web_search_provider,
                    queries,
                )

                # Perform web search with the configured provider
                search_depth = config.get("tavily_search_depth", "advanced")
                search_results, search_metadata = await perform_web_search(
                    queries,
                    api_keys=tavily_api_keys,
                    max_results_per_query=5,
                    max_chars_per_url=2000,
                    search_depth=search_depth,
                    web_search_provider=web_search_provider,
                    exa_mcp_url=exa_mcp_url or "https://mcp.exa.ai/mcp?tools=web_search_exa,web_search_advanced_exa,get_code_context_exa,deep_search_exa,crawling_exa,company_research_exa,linkedin_search_exa,deep_researcher_start,deep_researcher_check",
                )

                if search_results:
                    # Append search results to the first (most recent) user message
                    for msg in messages:
                        if msg.get("role") == "user":
                            msg["content"] = append_search_to_content(
                                msg.get("content", ""),
                                search_results,
                            )

                            logger.info(
                                "Web search results appended to user message",
                            )

                            # Save search results to database for persistence in chat history
                            get_bad_keys_db().save_message_search_data(
                                new_msg.id,
                                search_results,
                                search_metadata,
                            )

                            # Also update the cached MsgNode so follow-up requests in the same session get the data
                            if new_msg.id in msg_nodes:
                                msg_nodes[new_msg.id].search_results = search_results
                                msg_nodes[new_msg.id].tavily_metadata = search_metadata
                            break

    # Continue with response generation
    async def retry_callback() -> None:
        await process_message(
            new_msg=new_msg,
            discord_bot=discord_bot,
            httpx_client=httpx_client,
            twitter_api=twitter_api,
            reddit_client=reddit_client,
            msg_nodes=msg_nodes,
            curr_model_lock=curr_model_lock,
            curr_model_ref=curr_model_ref,
        )

    await generate_response(
        new_msg=new_msg,
        _discord_bot=discord_bot,
        msg_nodes=msg_nodes,
        messages=messages,
        user_warnings=user_warnings,
        provider=provider,
        _model=model,
        actual_model=actual_model,
        provider_slash_model=provider_slash_model,
        base_url=base_url,
        api_keys=api_keys,
        model_parameters=model_parameters,
        extra_headers=extra_headers,
        _extra_query=extra_query,
        _extra_body=extra_body,
        _system_prompt=system_prompt,
        config=config,
        _max_text=max_text,
        tavily_metadata=search_metadata,
        last_edit_time=last_edit_time,
        processing_msg=processing_msg,
        retry_callback=retry_callback,
    )


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


async def generate_response(  # noqa: C901, PLR0912, PLR0913, PLR0915
    new_msg: discord.Message,
    _discord_bot: discord.Client,
    msg_nodes: dict[int, MsgNode],
    messages: list[dict[str, object]],
    user_warnings: set[str],
    provider: str,
    _model: str,
    actual_model: str,
    provider_slash_model: str,
    base_url: str | None,
    api_keys: list[str],
    model_parameters: dict[str, object] | None,
    extra_headers: dict[str, str] | None,
    _extra_query: dict[str, object] | None,
    _extra_body: dict[str, object] | None,
    _system_prompt: str | None,
    config: dict[str, object],
    _max_text: int,
    tavily_metadata: dict[str, object] | None,
    last_edit_time: float,
    processing_msg: discord.Message,
    retry_callback: Callable[[], Awaitable[None]],
) -> None:
    """Generate and stream the LLM response using LiteLLM."""
    curr_content = finish_reason = None
    # Initialize with the pre-created processing message
    response_msgs = [processing_msg]
    msg_nodes[processing_msg.id] = MsgNode(parent_msg=new_msg)
    await msg_nodes[processing_msg.id].lock.acquire()
    response_contents = []

    # Count input tokens (chat history + latest query)
    input_tokens = count_conversation_tokens(messages)

    if use_plain_responses := config.get("use_plain_responses", False):
        max_message_length = 4000
    else:
        max_message_length = 4096 - len(STREAMING_INDICATOR)
        embed = discord.Embed.from_dict(
            {
                "fields": [
                    {"name": warning, "value": "", "inline": False}
                    for warning in sorted(user_warnings)
                ],
            },
        )
        embed.set_footer(text=f"{provider_slash_model} | total tokens: {input_tokens:,}")

    async def reply_helper(**reply_kwargs: object) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)

        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()

    def _extract_grounding_metadata(
        response_obj: object,
        choice_obj: object | None = None,
    ) -> object | None:
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

    def _split_response_content(text: str, max_length: int) -> list[str]:
        if not text:
            return []
        return [text[i : i + max_length] for i in range(0, len(text), max_length)]

    async def get_stream(
        api_key: str,
    ) -> AsyncIterator[tuple[str, object | None, object | None]]:
        """Get streaming response from LiteLLM using shared configuration."""
        # Check if Gemini grounding should be enabled (no URL in message content)
        enable_grounding = not re.search(r"https?://", new_msg.content)

        # Use shared utility to prepare kwargs with all provider-specific config
        litellm_kwargs = prepare_litellm_kwargs(
            provider=provider,
            model=actual_model,
            messages=messages[::-1],  # Reverse to get chronological order
            api_key=api_key,
            options=LiteLLMOptions(
                base_url=base_url,
                extra_headers=extra_headers,
                stream=True,
                model_parameters=model_parameters,
                enable_grounding=enable_grounding,
            ),
        )

        # Add timeout to prevent indefinite hangs (60 seconds for streaming)
        litellm_kwargs["timeout"] = LITELLM_TIMEOUT_SECONDS

        # Make the streaming call
        stream = await litellm.acompletion(**litellm_kwargs)

        async def _iter_stream_with_first_chunk(
            stream_iter: AsyncIterator[object],
        ) -> AsyncIterator[object]:
            try:
                first_chunk = await asyncio.wait_for(
                    stream_iter.__anext__(),
                    timeout=FIRST_TOKEN_TIMEOUT_SECONDS,
                )
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError as exc:
                raise FirstTokenTimeoutError from exc

            yield first_chunk
            async for chunk in stream_iter:
                yield chunk

        async for chunk in _iter_stream_with_first_chunk(stream):
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta_content = choice.delta.content or ""
            chunk_finish_reason = choice.finish_reason

            grounding_metadata = _extract_grounding_metadata(chunk, choice)

            # Log the chunk attributes on finish to help debug
            if chunk_finish_reason and is_gemini_model(actual_model):
                chunk_attrs = [attr for attr in dir(chunk) if not attr.startswith("_")]
                logger.debug("Gemini chunk finish - attributes: %s", chunk_attrs)
                if hasattr(chunk, "model_extra") and chunk.model_extra:
                    logger.info(
                        "Gemini chunk model_extra keys: %s",
                        list(chunk.model_extra.keys()),
                    )
                hidden_params = getattr(chunk, "_hidden_params", None)
                if hidden_params:
                    logger.info(
                        "Gemini chunk _hidden_params keys: %s",
                        list(hidden_params.keys()),
                    )

            yield delta_content, chunk_finish_reason, grounding_metadata

    async def get_completion(
        api_key: str,
    ) -> tuple[str | None, object | None]:
        """Get non-streaming response from LiteLLM using shared configuration."""
        enable_grounding = not re.search(r"https?://", new_msg.content)

        litellm_kwargs = prepare_litellm_kwargs(
            provider=provider,
            model=actual_model,
            messages=messages[::-1],
            api_key=api_key,
            options=LiteLLMOptions(
                base_url=base_url,
                extra_headers=extra_headers,
                stream=False,
                model_parameters=model_parameters,
                enable_grounding=enable_grounding,
            ),
        )

        litellm_kwargs["timeout"] = LITELLM_TIMEOUT_SECONDS
        response = await litellm.acompletion(**litellm_kwargs)

        if not response.choices:
            return None, _extract_grounding_metadata(response)

        choice = response.choices[0]
        message = getattr(choice, "message", None)
        content = None
        if message is not None:
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")

        grounding_metadata = _extract_grounding_metadata(response, choice)
        return content, grounding_metadata

    grounding_metadata = None
    attempt_count = 0
    fallback_level = 0  # 0 = original, 1 = mistral, 2 = gemma
    original_provider = provider  # Store original for logging
    original_model = actual_model
    last_error_msg = None

    # Determine if the original model is already mistral (skip to gemma fallback)
    is_original_mistral = (
        original_provider == "mistral" and "mistral" in original_model.lower()
    )

    # Get good keys (filter out known bad ones - synced with search decider)
    try:
        good_keys = get_bad_keys_db().get_good_keys_synced(provider, api_keys)
    except Exception:
        logger.exception("Failed to get good keys, falling back to all keys")
        good_keys = api_keys.copy()

    # If all keys are bad, reset and try again with all keys
    if not good_keys:
        logger.warning(
            "All API keys for provider '%s' (synced) are marked as bad. Resetting...",
            provider,
        )
        try:
            get_bad_keys_db().reset_provider_keys_synced(provider)
        except Exception:
            logger.exception("Failed to reset provider keys")
        good_keys = api_keys.copy()

    initial_key_count = len(good_keys)

    while True:
        curr_content = finish_reason = None
        response_contents = []
        attempt_count += 1

        # Get the next good key to try
        if not good_keys:
            # All good keys exhausted, try next fallback level
            next_fallback = None

            if fallback_level == 0:
                # First fallback: mistral (unless original was already mistral)
                if is_original_mistral:
                    # Skip mistral, go directly to gemma
                    fallback_level = 2
                    next_fallback = ("gemini", "gemma-3-27b-it", "gemini/gemma-3-27b-it")
                    logger.warning(
                        "All %s keys exhausted for mistral/%s. Falling back to gemini/gemma-3-27b-it...",
                        initial_key_count,
                        original_model,
                    )
                else:
                    fallback_level = 1
                    next_fallback = ("mistral", "mistral-large-latest", "mistral/mistral-large-latest")
                    logger.warning(
                        "All %s keys exhausted for provider '%s'. Falling back to mistral/mistral-large-latest...",
                        initial_key_count,
                        original_provider,
                    )
            elif fallback_level == 1:
                # Second fallback: gemma
                fallback_level = 2
                next_fallback = ("gemini", "gemma-3-27b-it", "gemini/gemma-3-27b-it")
                logger.warning(
                    "Mistral fallback also failed. Falling back to gemini/gemma-3-27b-it...",
                )

            if next_fallback:
                new_provider, new_model, new_provider_slash_model = next_fallback

                # Switch to fallback provider
                provider = new_provider
                actual_model = new_model
                provider_slash_model = new_provider_slash_model

                # Get fallback provider configuration
                fallback_provider_config = config.get("providers", {}).get(provider, {})
                base_url = fallback_provider_config.get("base_url")
                fallback_api_keys = ensure_list(fallback_provider_config.get("api_key"))

                if fallback_api_keys:
                    api_keys = fallback_api_keys
                    good_keys = fallback_api_keys.copy()
                    initial_key_count = len(good_keys)
                    attempt_count = 0  # Reset attempt count for new provider
                    continue  # Try with the new provider
                logger.error(
                    "No API keys available for fallback provider '%s'",
                    provider,
                )
                # Continue to try the next fallback level
                good_keys = []
                continue
            logger.error("All fallback options exhausted (mistral and gemma)")
            error_text = (
                "❌ All API keys (including all fallbacks) exhausted. Please try again later."
            )
            if last_error_msg:
                error_text += f"\nLast error: {last_error_msg}"
            if use_plain_responses:
                layout = LayoutView().add_item(TextDisplay(content=error_text))
                if response_msgs:
                    await response_msgs[-1].edit(view=layout)
                else:
                    await reply_helper(view=layout)
                response_contents = [error_text]
            else:
                embed.description = error_text
                embed.color = EMBED_COLOR_INCOMPLETE
                if response_msgs:
                    await response_msgs[-1].edit(embed=embed, view=None)
                else:
                    await reply_helper(embed=embed)
                response_contents = [error_text]
            break

        current_api_key = good_keys[(attempt_count - 1) % len(good_keys)]

        try:
            async with new_msg.channel.typing():
                async for delta_content, new_finish_reason, new_grounding_metadata in get_stream(current_api_key):
                    if new_grounding_metadata:
                        grounding_metadata = new_grounding_metadata
                        logger.info(
                            "Captured grounding metadata from stream: %s",
                            type(grounding_metadata),
                        )

                    if finish_reason is not None:
                        break

                    finish_reason = new_finish_reason

                    prev_content = curr_content or ""
                    curr_content = delta_content

                    new_content = (
                        prev_content
                        if finish_reason is None
                        else (prev_content + curr_content)
                    )

                    if response_contents == [] and new_content == "":
                        continue

                    if start_next_msg := response_contents == [] or (
                        len(response_contents[-1] + new_content) > max_message_length
                    ):
                        response_contents.append("")

                    response_contents[-1] += new_content

                    if not use_plain_responses:
                        time_delta = datetime.now(timezone.utc).timestamp() - last_edit_time

                        ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                        msg_split_incoming = finish_reason is None and (
                            len(response_contents[-1] + curr_content)
                            > max_message_length
                        )
                        is_final_edit = finish_reason is not None or msg_split_incoming
                        is_good_finish = finish_reason is not None and any(
                            x in str(finish_reason).lower()
                            for x in ("stop", "end_turn")
                        )

                        if start_next_msg or ready_to_edit or is_final_edit:
                            embed.description = (
                                response_contents[-1]
                                if is_final_edit
                                else (response_contents[-1] + STREAMING_INDICATOR)
                            )
                            embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                            view = (
                                SourceView(grounding_metadata)
                                if is_final_edit
                                and _has_grounding_data(grounding_metadata)
                                else None
                            )

                            msg_index = len(response_contents) - 1
                            if start_next_msg:
                                if msg_index < len(response_msgs):
                                    await response_msgs[msg_index].edit(
                                        embed=embed,
                                        view=view,
                                    )
                                else:
                                    await reply_helper(
                                        embed=embed,
                                        silent=True,
                                        view=view,
                                    )
                            else:
                                await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
                                await response_msgs[msg_index].edit(
                                    embed=embed,
                                    view=view,
                                )

                            last_edit_time = datetime.now(timezone.utc).timestamp()

            fallback_used = False
            if (
                finish_reason is None
                and provider == "gemini"
                and is_gemini_model(actual_model)
            ):
                logger.warning(
                    "Gemini stream ended without finish reason. Falling back to non-streaming request.",
                )
                non_stream_content, non_stream_grounding = await get_completion(
                    current_api_key,
                )
                if non_stream_grounding:
                    grounding_metadata = non_stream_grounding
                if non_stream_content:
                    response_contents = _split_response_content(
                        non_stream_content,
                        max_message_length,
                    )
                    fallback_used = True
                elif not response_contents:
                    _raise_empty_response()
                else:
                    logger.warning(
                        "Gemini non-streaming fallback returned empty content; keeping partial stream output.",
                    )

            if not response_contents:
                if provider == "gemini" and is_gemini_model(actual_model):
                    logger.warning(
                        "Empty streaming response from Gemini. Falling back to non-streaming request.",
                    )
                    non_stream_content, non_stream_grounding = await get_completion(
                        current_api_key,
                    )
                    if non_stream_grounding:
                        grounding_metadata = non_stream_grounding
                    if non_stream_content:
                        response_contents = _split_response_content(
                            non_stream_content,
                            max_message_length,
                        )
                    else:
                        _raise_empty_response()
                else:
                    _raise_empty_response()

            if fallback_used and not use_plain_responses:
                for i, content in enumerate(response_contents):
                    embed.description = content
                    embed.color = EMBED_COLOR_COMPLETE
                    if i < len(response_msgs):
                        await response_msgs[i].edit(embed=embed, view=None)
                    else:
                        await reply_helper(embed=embed, silent=True, view=None)

            if use_plain_responses:
                for i, content in enumerate(response_contents):
                    # Build the LayoutView with text content
                    layout = LayoutView().add_item(TextDisplay(content=content))

                    # Add buttons only to the last message
                    if i == len(response_contents) - 1:
                        # Add Gemini grounding sources button if available
                        if _has_grounding_data(grounding_metadata):
                            layout.add_item(SourceButton(grounding_metadata))
                        # Add Tavily sources button if available
                        if tavily_metadata and (
                            tavily_metadata.get("urls")
                            or tavily_metadata.get("queries")
                        ):
                            layout.add_item(TavilySourceButton(tavily_metadata))

                    if i < len(response_msgs):
                        # Edit existing message (first one is the processing message)
                        await response_msgs[i].edit(view=layout)
                    else:
                        # Create new message for overflow content
                        await reply_helper(view=layout)

            break

        except Exception as e:
            last_error_msg = str(e)
            logger.exception("Error while generating response")

            # Determine if this is an API key error vs other error (timeout, network, etc.)
            is_key_error = False
            is_first_token_timeout = isinstance(e, FirstTokenTimeoutError)
            is_timeout_error = (
                isinstance(e, asyncio.TimeoutError) or "timeout" in str(e).lower()
            )
            is_empty_response = "no content" in str(e).lower()

            if is_first_token_timeout:
                logger.warning(
                    "No first token within %s seconds for provider '%s'; forcing fallback model.",
                    FIRST_TOKEN_TIMEOUT_SECONDS,
                    provider,
                )
                good_keys = []
                continue

            # Check for typical API key/auth error patterns
            error_str = str(e).lower()

            key_error_patterns = [
                "unauthorized", "invalid_api_key", "invalid key", "api key",
                "authentication", "forbidden", "401", "403", "quota", "rate limit",
                "billing", "insufficient_quota", "expired",
            ]
            for pattern in key_error_patterns:
                if pattern in error_str:
                    is_key_error = True
                    break

            # Only mark the key as bad for actual key-related errors
            if is_key_error:
                error_msg = str(e)[:200] if e else "Unknown error"
                try:
                    get_bad_keys_db().mark_key_bad_synced(
                        provider,
                        current_api_key,
                        error_msg,
                    )
                except Exception:
                    logger.exception("Failed to mark key as bad")

                # Remove the bad key from good_keys list for this session
                if current_api_key in good_keys:
                    good_keys.remove(current_api_key)
                    logger.info(
                        "Removed bad key for '%s', %s keys remaining",
                        provider,
                        len(good_keys),
                    )
            elif is_timeout_error or is_empty_response:
                # For timeouts and empty responses, don't mark key as bad but still retry
                # Remove from good_keys for this session to try next key
                if current_api_key in good_keys:
                    good_keys.remove(current_api_key)
                logger.warning(
                    "Non-key error for provider '%s': %s, %s keys remaining",
                    provider,
                    "timeout" if is_timeout_error else "empty response",
                    len(good_keys),
                )
            else:
                # For other errors, also try the next key
                if current_api_key in good_keys:
                    good_keys.remove(current_api_key)
                logger.warning(
                    "Unknown error for provider '%s', %s keys remaining",
                    provider,
                    len(good_keys),
                )

            # If no keys left, the main loop will handle fallback to mistral
            # Continue to the next iteration

    if not use_plain_responses and len(response_msgs) > len(response_contents):
        for msg in response_msgs[len(response_contents):]:
            await msg.delete()
            if msg.id in msg_nodes:
                msg_nodes[msg.id].lock.release()
                del msg_nodes[msg.id]
        response_msgs = response_msgs[:len(response_contents)]

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    # Update the last message with ResponseView for "View Response Better" button
    if not use_plain_responses and response_msgs and response_contents:
        full_response = "".join(response_contents)
        response_view = ResponseView(
            full_response,
            grounding_metadata,
            tavily_metadata,
            retry_callback,
            new_msg.author.id,
        )

        # Count output tokens and update footer with total
        output_tokens = count_text_tokens(full_response)
        total_tokens = input_tokens + output_tokens

        # Update the last message with the final view
        last_msg_index = len(response_msgs) - 1
        if last_msg_index < len(response_contents):
            embed.description = response_contents[last_msg_index]
            embed.color = EMBED_COLOR_COMPLETE
            embed.set_footer(
                text=f"{provider_slash_model} | total tokens: {total_tokens:,}",
            )
            await response_msgs[last_msg_index].edit(embed=embed, view=response_view)

    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        # Get keys to remove (oldest first based on insertion order)
        keys_to_remove = sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]
        for msg_id in keys_to_remove:
            node = msg_nodes.get(msg_id)
            if node is not None:
                async with node.lock:
                    msg_nodes.pop(msg_id, None)
