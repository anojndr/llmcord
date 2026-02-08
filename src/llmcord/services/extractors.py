"""Asset extraction and handling (PDFs, Tweets, YouTube, Reddit)."""
import asyncio
import io
import logging
import re
from collections.abc import AsyncIterator
from typing import Protocol

import asyncpraw
import asyncprawcore
import discord
import httpx
import pymupdf4llm
from bs4 import BeautifulSoup
from PIL import Image
from twscrape import gather
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    YouTubeTranscriptApiException,
)
from youtube_transcript_api.proxies import GenericProxyConfig

from llmcord.core.config import BROWSER_HEADERS
from llmcord.logic.utils import _ensure_pymupdf_layout_activated
from llmcord.services.http import request_with_retries

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - optional dependency
    fitz = None  # type: ignore[assignment]


async def extract_pdf_text(pdf_content: bytes) -> str | None:
    """Extract text content from a PDF file using pymupdf4llm.

    This is used for non-Gemini models since they don't natively support
    PDF attachments.
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
        except (RuntimeError, ValueError) as exc:
            logger.warning("Failed to open PDF: %s", exc)
            return None
        try:
            _ensure_pymupdf_layout_activated()
            md_text = pymupdf4llm.to_markdown(doc)
        except (RuntimeError, ValueError) as exc:
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
    except (RuntimeError, ValueError, OSError) as exc:
        logger.warning("PDF extraction error: %s", exc)
        return None


async def extract_pdf_images(
    pdf_content: bytes,
) -> list[tuple[str, bytes]]:
    """Extract embedded images from a PDF file using PyMuPDF.

    This is used for non-Gemini models so they can view images
    contained in PDF attachments.
    Runs in a thread pool since PyMuPDF operations are CPU-bound.

    Args:
        pdf_content: The raw PDF file bytes

    Returns:
        List of (content_type, image_bytes) tuples for each extracted image

    """

    def _extract() -> list[tuple[str, bytes]]:
        if fitz is None:
            return []
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
        except (RuntimeError, ValueError) as exc:
            logger.warning("Failed to open PDF for image extraction: %s", exc)
            return []

        images: list[tuple[str, bytes]] = []
        seen_xrefs: set[int] = set()
        try:
            for page in doc:
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    if xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)
                    result = _extract_single_pdf_image(doc, xref)
                    if result is not None:
                        images.append(result)
        finally:
            doc.close()

        return images

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_extract),
            timeout=30,
        )
    except asyncio.TimeoutError:
        logger.warning("PDF image extraction timed out")
        return []
    except (RuntimeError, ValueError, OSError) as exc:
        logger.warning("PDF image extraction error: %s", exc)
        return []


_PDF_IMAGE_MIME_MAP: dict[str, str] = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "gif": "image/png",
    "tiff": "image/tiff",
    "tif": "image/tiff",
}


def _extract_single_pdf_image(
    doc: object,
    xref: int,
) -> tuple[str, bytes] | None:
    """Extract a single image from a PDF document by xref."""
    try:
        extracted = doc.extract_image(xref)  # type: ignore[union-attr]
    except (RuntimeError, ValueError):
        return None
    if not extracted or not extracted.get("image"):
        return None

    ext = extracted.get("ext", "png")
    content_type = _PDF_IMAGE_MIME_MAP.get(ext, f"image/{ext}")
    img_bytes: bytes = extracted["image"]

    if ext == "gif":
        try:
            with Image.open(io.BytesIO(img_bytes)) as img:
                output = io.BytesIO()
                img.save(output, format="PNG")
                img_bytes = output.getvalue()
        except OSError:
            logger.debug("Failed to convert PDF GIF image")
            return None

    return content_type, img_bytes


class TweetUserProtocol(Protocol):
    """Protocol for tweet user objects."""

    username: str | None


class TweetProtocol(Protocol):
    """Protocol for tweet objects returned by the Twitter API wrapper."""

    user: TweetUserProtocol | None
    raw_content: str | None


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


def _get_tweet_text(tweet: TweetProtocol) -> str:
    """Return tweet text while handling rawContent/raw_content attributes."""
    raw_content = getattr(tweet, "rawContent", None)
    if raw_content is None:
        raw_content = getattr(tweet, "raw_content", None)
    return raw_content or ""


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
        tweet = await asyncio.wait_for(
            twitter_api.tweet_details(tweet_id),
            timeout=10,
        )

        # Handle edge case where tweet or user is None
        if not tweet or not tweet.user:
            return None

        username = tweet.user.username or "unknown"

        if include_url and tweet_url:
            tweet_text = (
                f"\n--- Tweet from @{username} ({tweet_url}) ---\n"
                f"{_get_tweet_text(tweet)}"
            )
        else:
            tweet_text = f"Tweet from @{username}:\n{_get_tweet_text(tweet)}"

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
                        tweet_text += (
                            f"\n- @{reply_username}: {_get_tweet_text(reply)}"
                        )
    except (asyncio.TimeoutError, RuntimeError, ValueError) as exc:
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
            return await request_with_retries(
                lambda: httpx_client.get(att.url, timeout=60),
                log_context=f"attachment {att.filename}",
            )
        except httpx.HTTPError as exc:
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
            except OSError:
                logger.exception("Error converting GIF to PNG")

        processed.append(
            {
                "content_type": content_type,
                "content": content,
                "text": (
                    resp.text
                    if content_type and content_type.startswith("text")
                    else None
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

    try:
        yandex_resp = await request_with_retries(
            lambda: httpx_client.get(
                "https://yandex.com/images/search",
                params=params,
                headers=BROWSER_HEADERS,
                follow_redirects=True,
                timeout=60,
            ),
            log_context="Yandex reverse image search",
        )
    except httpx.HTTPError as exc:
        logger.warning("Yandex lookup failed for %s: %s", image_url, exc)
        return [], []

    if yandex_resp.status_code != httpx.codes.OK:
        logger.warning(
            "Yandex lookup returned status %s for %s",
            yandex_resp.status_code,
            image_url,
        )
        return [], []
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

            # Check if the link is a Twitter/X URL and extract for later
            # processing.
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


def _build_youtube_transcript_api(
    proxy_url: str | None,
) -> YouTubeTranscriptApi:
    if proxy_url:
        proxy_config = GenericProxyConfig(
            http_url=proxy_url,
            https_url=proxy_url,
        )
        return YouTubeTranscriptApi(proxy_config=proxy_config)
    return YouTubeTranscriptApi()


async def extract_youtube_transcript(
    video_id: str,
    httpx_client: httpx.AsyncClient,
    *,
    proxy_url: str | None = None,
) -> str | None:
    """Fetch YouTube transcript and metadata."""
    try:
        ytt_api = _build_youtube_transcript_api(proxy_url)
        transcript_obj = await asyncio.to_thread(ytt_api.fetch, video_id)
        transcript = transcript_obj.to_raw_data()

        response = await request_with_retries(
            lambda: httpx_client.get(
                f"https://www.youtube.com/watch?v={video_id}",
                follow_redirects=True,
                timeout=30,
            ),
            log_context=f"YouTube metadata {video_id}",
        )
        response.raise_for_status()
        html = response.text
        title_match = re.search(r'<meta name="title" content="(.*?)">', html)
        title = title_match.group(1) if title_match else "Unknown Title"
        channel_match = re.search(
            r'<link itemprop="name" content="(.*?)">',
            html,
        )
        channel = channel_match.group(1) if channel_match else "Unknown Channel"

        transcript_text = " ".join(x["text"] for x in transcript)
    except (
        YouTubeTranscriptApiException,
        httpx.HTTPError,
        RuntimeError,
        ValueError,
        KeyError,
    ) as exc:
        logger.debug(
            "Failed to fetch YouTube transcript for %s: %s",
            video_id,
            exc,
        )
        return None
    else:
        return (
            f"YouTube Video ID: {video_id}\n"
            f"Title: {title}\n"
            f"Channel: {channel}\n"
            f"Transcript:\n{transcript_text}"
        )


async def _resolve_reddit_share_url(
    post_url: str,
    httpx_client: httpx.AsyncClient,
    proxy_url: str | None,
) -> str:
    """Resolve Reddit share URL to canonical URL."""
    if not re.search(r"/r/\w+/s/", post_url):
        return post_url

    if proxy_url:
        async with httpx.AsyncClient(
            proxy=proxy_url,
            follow_redirects=True,
        ) as proxy_client:
            resolve_resp = await proxy_client.head(
                post_url,
                headers=BROWSER_HEADERS,
                timeout=30,
            )
            resolved_url = str(resolve_resp.url)
    else:
        resolve_resp = await httpx_client.head(
            post_url,
            headers=BROWSER_HEADERS,
            follow_redirects=True,
            timeout=30,
        )
        resolved_url = str(resolve_resp.url)

    # Strip query params added by Reddit share redirect
    if "?" in resolved_url:
        resolved_url = resolved_url.split("?")[0]
    return resolved_url


def _build_reddit_json_url(post_url: str) -> str:
    """Convert a Reddit post URL to its JSON endpoint."""
    if post_url.endswith(".json"):
        return post_url
    if "?" in post_url:
        base_url, query = post_url.split("?", 1)
        return f"{base_url.rstrip('/')}.json?{query}"
    return f"{post_url.rstrip('/')}.json"


async def _fetch_reddit_json(
    json_url: str,
    httpx_client: httpx.AsyncClient,
    proxy_url: str | None,
) -> dict:
    """Fetch Reddit JSON data from the given URL."""
    if proxy_url:
        async with httpx.AsyncClient(
            proxy=proxy_url,
            follow_redirects=True,
        ) as proxy_client:
            response = await proxy_client.get(
                json_url,
                headers=BROWSER_HEADERS,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()

    response = await httpx_client.get(
        json_url,
        headers=BROWSER_HEADERS,
        follow_redirects=True,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _format_reddit_comments(
    comments_listing: list[dict],
    max_comments: int | None,
) -> str:
    """Format Reddit comments into a string."""
    top_comments = (
        comments_listing if max_comments is None else comments_listing[:max_comments]
    )
    if not top_comments:
        return ""

    result = "\n\nTop Comments:"
    for comment in top_comments:
        c_data = comment.get("data", {})
        if comment.get("kind") == "more":
            continue
        comment_author = c_data.get("author", "[deleted]")
        comment_body = c_data.get("body", "")
        if comment_body:
            result += f"\n- u/{comment_author}: {comment_body}"
    return result


async def extract_reddit_post_json(
    post_url: str,
    httpx_client: httpx.AsyncClient,
    *,
    proxy_url: str | None = None,
    max_comments: int | None = None,
) -> str | None:
    """Extract Reddit post content using JSON endpoints.

    Args:
        post_url: The Reddit post URL to extract content from
        httpx_client: The HTTP client to use for requests
        proxy_url: Optional proxy URL to use for requests
        max_comments: Maximum number of comments to extract. None = unlimited.

    Returns:
        Formatted post text with comments, or None if extraction failed

    """
    try:
        post_url = await _resolve_reddit_share_url(post_url, httpx_client, proxy_url)
        json_url = _build_reddit_json_url(post_url)
        data = await _fetch_reddit_json(json_url, httpx_client, proxy_url)

        post_listing = data[0]["data"]["children"][0]["data"]
        comments_listing = data[1]["data"]["children"]

        title = post_listing.get("title", "Untitled")
        subreddit_name = post_listing.get("subreddit", "unknown")
        author_name = post_listing.get("author", "unknown")
        selftext = post_listing.get("selftext", "")
        url = post_listing.get("url", "")
        is_self = post_listing.get("is_self", True)

        post_text = (
            f"Reddit Post: {title}\nSubreddit: r/{subreddit_name}\n"
            f"Author: u/{author_name}\n\n{selftext}"
        )

        if not is_self and url:
            post_text += f"\nLink: {url}"

        post_text += _format_reddit_comments(comments_listing, max_comments)

    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        logger.debug(
            "Failed to fetch Reddit content (JSON) for %s: %s",
            post_url,
            exc,
        )
        return None
    else:
        return post_text


async def extract_reddit_post_praw(
    post_url: str,
    reddit_client: asyncpraw.Reddit,
    *,
    max_comments: int | None = None,
) -> str | None:
    """Extract Reddit post content using AsyncPRAW.

    Args:
        post_url: The Reddit post URL to extract content from
        reddit_client: The AsyncPRAW Reddit client instance
        max_comments: Maximum number of comments to extract. None = unlimited.

    Returns:
        Formatted post text with comments, or None if extraction failed

    """
    try:
        submission = await reddit_client.submission(url=post_url)

        # Handle edge case where submission is missing key attributes
        if not submission:
            return None

        # Safely access attributes with defaults
        title = getattr(submission, "title", "Untitled")
        subreddit_name = (
            submission.subreddit.display_name
            if submission.subreddit
            else "unknown"
        )
        author_name = (
            submission.author.name if submission.author else "[deleted]"
        )
        selftext = getattr(submission, "selftext", "") or ""

        post_text = (
            f"Reddit Post: {title}\nSubreddit: r/{subreddit_name}\n"
            f"Author: u/{author_name}\n\n{selftext}"
        )

        if not getattr(submission, "is_self", True) and getattr(
            submission, "url", None,
        ):
            post_text += f"\nLink: {submission.url}"

        submission.comment_sort = "top"
        await submission.comments()
        comments_list = (
            submission.comments.list() if submission.comments else []
        )
        top_comments = (
            comments_list
            if max_comments is None
            else comments_list[:max_comments]
        )

        if top_comments:
            post_text += "\n\nTop Comments:"
            for comment in top_comments:
                if isinstance(comment, asyncpraw.models.MoreComments):
                    continue
                comment_author = (
                    comment.author.name if comment.author else "[deleted]"
                )
                comment_body = getattr(comment, "body", "") or ""
                post_text += f"\n- u/{comment_author}: {comment_body}"

    except (
        asyncprawcore.exceptions.AsyncPrawcoreException,
        asyncpraw.exceptions.RedditAPIException,
        AttributeError,
        ValueError,
    ) as exc:
        logger.debug(
            "Failed to fetch Reddit content (PRAW) for %s: %s",
            post_url,
            exc,
        )
        return None
    else:
        return post_text


async def extract_reddit_post(
    post_url: str,
    httpx_client: httpx.AsyncClient,
    reddit_client: asyncpraw.Reddit | None = None,
    *,
    proxy_url: str | None = None,
    max_comments: int | None = None,
) -> str | None:
    """Extract Reddit post content using the configured method.

    Args:
        post_url: The Reddit post URL to extract content from
        httpx_client: The HTTP client to use for requests
        reddit_client: Optional AsyncPRAW Reddit client instance
        proxy_url: Optional proxy URL to use for requests
        max_comments: Maximum number of comments to extract. None = unlimited.

    Returns:
        Formatted post text with comments, or None if extraction failed

    """
    # Resolve Reddit share URLs first (format: /r/.../s/...)
    # Share links need to be resolved to canonical URLs for both methods
    if re.search(r"/r/\w+/s/", post_url):
        try:
            if proxy_url:
                async with httpx.AsyncClient(
                    proxy=proxy_url,
                    follow_redirects=True,
                ) as proxy_client:
                    resolve_resp = await proxy_client.head(
                        post_url,
                        headers=BROWSER_HEADERS,
                        timeout=30,
                    )
                    post_url = str(resolve_resp.url)
            else:
                resolve_resp = await httpx_client.head(
                    post_url,
                    headers=BROWSER_HEADERS,
                    follow_redirects=True,
                    timeout=30,
                )
                post_url = str(resolve_resp.url)
            # Strip query params added by Reddit share redirect
            if "?" in post_url:
                post_url = post_url.split("?")[0]
        except httpx.HTTPError as exc:
            logger.debug("Failed to resolve Reddit share URL %s: %s", post_url, exc)
            return None

    if reddit_client:
        return await extract_reddit_post_praw(
            post_url,
            reddit_client,
            max_comments=max_comments,
        )

    return await extract_reddit_post_json(
        post_url,
        httpx_client,
        proxy_url=proxy_url,
        max_comments=max_comments,
    )

