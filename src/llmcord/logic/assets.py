"""Asset extraction and handling (PDFs, Tweets, YouTube, Reddit)."""
import asyncio
import io
import logging
import re
from typing import AsyncIterator, Protocol

import discord
import httpx
import pymupdf4llm
from PIL import Image
from bs4 import BeautifulSoup
from twscrape import gather
from youtube_transcript_api import YouTubeTranscriptApi
import asyncpraw

from llmcord.services.database import get_bad_keys_db
from llmcord.config import BROWSER_HEADERS
from llmcord.logic.helpers import _ensure_pymupdf_layout_activated

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - optional dependency
    fitz = None  # type: ignore[assignment]


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


async def fetch_tweet_with_replies(
    twitter_api: TwitterApiProtocol,
    tweet_id: int,
    max_replies: int = 0,
    *,
    include_url: bool = False,
    tweet_url: str = "",
) -> str | None:
    """Fetch a tweet and optionally its replies, returning formatted text.

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
            tweet_text = (
                f"\n--- Tweet from @{username} ({tweet_url}) ---\n"
                f"{tweet.rawContent or ''}"
            )
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
    attachments: list[discord.Attachment],
    httpx_client: httpx.AsyncClient,
) -> list[tuple[discord.Attachment, httpx.Response]]:
    """Download attachments with timeout."""

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

    responses = await asyncio.gather(
        *[download_attachment(att) for att in attachments],
    )

    return [
        (att, resp)
        for att, resp in zip(attachments, responses, strict=False)
        if resp is not None
    ]


async def process_attachments(
    successful_pairs: list[tuple[discord.Attachment, httpx.Response]],
) -> list[dict[str, bytes | str | None]]:
    """Process downloaded attachments (handle GIFs, extract text)."""
    processed = []
    for att, resp in successful_pairs:
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

        processed.append(
            {
                "content_type": content_type,
                "content": content,
                "text": (
                    resp.text if content_type and content_type.startswith("text") else None
                ),
            },
        )
    return processed


async def perform_yandex_lookup(
    image_url: str,
    httpx_client: httpx.AsyncClient,
    twitter_api: TwitterApiProtocol,
    max_tweet_replies: int,
) -> tuple[list[str], list[str]]:
    """Perform Yandex reverse image search and extract results."""
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

            title = title_el.get_text(strip=True) if title_el else "N/A"
            link = title_el["href"] if title_el else "#"
            domain = domain_el.get_text(strip=True) if domain_el else ""
            desc = desc_el.get_text(strip=True) if desc_el else ""

            lens_results.append(
                f"- [{title}]({link}) ({domain}) - {desc}",
            )

            # Check if the link is a Twitter/X URL and extract for later processing
            if re.search(
                r"(?:twitter\.com|x\.com)/[a-zA-Z0-9_]+/status/[0-9]+",
                link,
            ):
                twitter_urls_found.append(link)

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

    return lens_results, twitter_content


async def extract_youtube_transcript(
    video_id: str,
    httpx_client: httpx.AsyncClient,
) -> str | None:
    """Fetch YouTube transcript and metadata."""
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_obj = await asyncio.to_thread(ytt_api.fetch, video_id)
        transcript = transcript_obj.to_raw_data()

        response = await httpx_client.get(
            f"https://www.youtube.com/watch?v={video_id}",
            follow_redirects=True,
            timeout=30,
        )
        html = response.text
        title_match = re.search(r'<meta name="title" content="(.*?)">', html)
        title = title_match.group(1) if title_match else "Unknown Title"
        channel_match = re.search(r'<link itemprop="name" content="(.*?)">', html)
        channel = channel_match.group(1) if channel_match else "Unknown Channel"

        return (
            f"YouTube Video ID: {video_id}\nTitle: {title}\nChannel: {channel}\n"
            f"Transcript:\n" + " ".join(x["text"] for x in transcript)
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to fetch YouTube transcript: %s", exc)
        return None


async def extract_reddit_post(
    post_url: str,
    reddit_client: asyncpraw.Reddit,
) -> str | None:
    """Extract Reddit post content."""
    try:
        submission = await reddit_client.submission(url=post_url)

        # Handle edge case where submission is missing key attributes
        if not submission:
            return None

        # Safely access attributes with defaults
        title = getattr(submission, "title", "Untitled")
        subreddit_name = (
            submission.subreddit.display_name if submission.subreddit else "unknown"
        )
        author_name = submission.author.name if submission.author else "[deleted]"
        selftext = getattr(submission, "selftext", "") or ""

        post_text = (
            f"Reddit Post: {title}\nSubreddit: r/{subreddit_name}\n"
            f"Author: u/{author_name}\n\n{selftext}"
        )

        if not getattr(submission, "is_self", True) and getattr(
            submission, "url", None
        ):
            post_text += f"\nLink: {submission.url}"

        submission.comment_sort = "top"
        await submission.comments()
        comments_list = submission.comments.list() if submission.comments else []
        top_comments = comments_list[:5]

        if top_comments:
            post_text += "\n\nTop Comments:"
            for comment in top_comments:
                if isinstance(comment, asyncpraw.models.MoreComments):
                    continue
                comment_author = comment.author.name if comment.author else "[deleted]"
                comment_body = getattr(comment, "body", "") or ""
                post_text += f"\n- u/{comment_author}: {comment_body}"

        return post_text
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to fetch Reddit content for %s: %s", post_url, exc)
        return None
