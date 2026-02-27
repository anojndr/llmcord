"""Asset extraction and handling (PDFs, Tweets, YouTube, Reddit)."""

import asyncio
import html
import importlib
import io
import logging
import re
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, Protocol
from urllib.parse import urlparse
from uuid import uuid4

import asyncpraw
import asyncprawcore
import brotli
import discord
import httpx
import pymupdf4llm
import trafilatura
from asyncpraw import exceptions as asyncpraw_exceptions
from asyncpraw import models as asyncpraw_models
from bs4 import BeautifulSoup, Tag
from PIL import Image
from twscrape import gather
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    YouTubeTranscriptApiException,
)

from llmcord.core.config import (
    BROWSER_HEADERS,
    FALLBACK_USER_AGENT,
)
from llmcord.core.error_handling import log_exception
from llmcord.logic.utils import _ensure_pymupdf_layout_activated
from llmcord.services.http import RetryOptions, request_with_retries

logger = logging.getLogger(__name__)

_REVERSE_IMAGE_URL_EXTRACTION_TIMEOUT_SECONDS = 5.0
_REVERSE_IMAGE_URL_EXTRACTION_RETRIES = 0

fitz_module: Any
try:
    fitz_module = importlib.import_module("fitz")
except ImportError:  # pragma: no cover - optional dependency
    fitz_module = None
fitz: Any = fitz_module


def _decode_brotli_if_needed(
    response: httpx.Response,
    content: bytes,
) -> bytes:
    encoding_header = response.headers.get("content-encoding", "").lower()
    if "br" not in encoding_header:
        return content
    try:
        return brotli.decompress(content)
    except brotli.error as exc:
        logger.debug(
            "Failed to decompress Brotli response for %s: %s",
            response.request.url if response.request else "response",
            exc,
        )
        return content


def _decode_text_content(
    content: bytes,
    response: httpx.Response,
) -> str:
    encoding = response.encoding or "utf-8"
    try:
        return content.decode(encoding, errors="replace")
    except LookupError:
        return content.decode("utf-8", errors="replace")


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
    except TimeoutError:
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
    except TimeoutError:
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

_NOTEGPT_SUCCESS_CODE = 100000


def _extract_single_pdf_image(
    doc: Any,  # noqa: ANN401
    xref: int,
) -> tuple[str, bytes] | None:
    """Extract a single image from a PDF document by xref."""
    try:
        extracted = doc.extract_image(xref)
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


async def _iter_tweet_replies(
    twitter_api: TwitterApiProtocol,
    tweet_id: int,
    max_replies: int,
) -> AsyncGenerator[TweetProtocol, None]:
    async for reply in twitter_api.tweet_replies(tweet_id, limit=max_replies):
        yield reply


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
                gather(_iter_tweet_replies(twitter_api, tweet_id, max_replies)),
                timeout=10,
            )
            if replies:
                tweet_text += "\n\nReplies:" if include_url else "\nReplies:"
                for reply in replies:
                    if reply and reply.user:
                        reply_username = reply.user.username or "unknown"
                        tweet_text += f"\n- @{reply_username}: {_get_tweet_text(reply)}"
    except (TimeoutError, RuntimeError, ValueError) as exc:
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
        content = _decode_brotli_if_needed(resp, resp.content)
        content_type = att.content_type

        if content_type == "image/gif":
            try:
                with Image.open(io.BytesIO(content)) as img:
                    output = io.BytesIO()
                    img.save(output, format="PNG")
                    content = output.getvalue()
                    content_type = "image/png"
            except OSError as exc:
                log_exception(
                    logger=logger,
                    message="Error converting GIF to PNG",
                    error=exc,
                    context={"filename": att.filename},
                )

        text_content = None
        if content_type and content_type.startswith("text"):
            text_content = _decode_text_content(content, resp)

        processed.append(
            {
                "content_type": content_type,
                "content": content,
                "text": text_content,
            },
        )
    return processed


async def extract_url_content(
    url: str,
    httpx_client: httpx.AsyncClient,
    *,
    timeout_seconds: float = 20,
    retries: int = 2,
) -> str | None:
    """Extract text content from a URL."""
    # Skip specialized domains that we handle elsewhere or that don't scrape well
    if any(
        domain in url.lower()
        for domain in ["twitter.com", "x.com", "youtube.com", "youtu.be"]
    ):
        return None

    try:
        response = await request_with_retries(
            lambda: httpx_client.get(
                url,
                headers=BROWSER_HEADERS,
                follow_redirects=True,
                timeout=timeout_seconds,
            ),
            options=RetryOptions(retries=retries),
            log_context=f"URL content extraction {url}",
        )
        if response.status_code != httpx.codes.OK:
            # Fallback for sites that block browser headers (like Wikipedia)
            # using a descriptive bot user agent.
            response = await request_with_retries(
                lambda: httpx_client.get(
                    url,
                    headers={
                        "User-Agent": FALLBACK_USER_AGENT,
                    },
                    follow_redirects=True,
                    timeout=timeout_seconds,
                ),
                options=RetryOptions(retries=retries),
                log_context=f"URL content extraction fallback {url}",
            )
            if response.status_code != httpx.codes.OK:
                return None

        # Use trafilatura for better content extraction
        # Run in thread pool since extraction can be CPU-bound
        text = await asyncio.to_thread(
            trafilatura.extract,
            response.content,
            url=url,
            output_format="markdown",
            with_metadata=True,
        )
        if not text:
            return None
    except (httpx.HTTPError, RuntimeError, ValueError):
        return None
    else:
        return text


async def _fetch_twitter_results(
    twitter_urls: list[str],
    twitter_api: TwitterApiProtocol,
    max_tweet_replies: int,
) -> list[str]:
    """Fetch content for a list of Twitter URLs."""
    twitter_content = []
    for twitter_url in twitter_urls:
        tweet_id = _extract_tweet_id_from_url(twitter_url)
        if tweet_id is not None:
            tweet_text = await fetch_tweet_with_replies(
                twitter_api,
                tweet_id,
                max_replies=max_tweet_replies,
                include_url=True,
                tweet_url=twitter_url,
            )
            if tweet_text:
                twitter_content.append(tweet_text)
    return twitter_content


def _extract_tweet_id_from_url(url: str) -> int | None:
    match = re.search(
        r"(?:twitter\.com|x\.com)/[a-zA-Z0-9_]+/status/([0-9]+)",
        url,
    )
    if not match:
        return None

    tweet_id_str = match.group(1)
    return int(tweet_id_str)


def _parse_yandex_sites_item(item: Tag) -> dict[str, str | None]:
    """Parse a single Yandex site item into a result dict."""
    title_el = item.select_one(".CbirSites-ItemTitle a")
    domain_el = item.select_one(".CbirSites-ItemDomain")
    desc_el = item.select_one(".CbirSites-ItemDescription")

    title = title_el.get_text(strip=True) if title_el else "N/A"
    link = str(title_el["href"]) if title_el and title_el.has_attr("href") else None
    domain = domain_el.get_text(strip=True) if domain_el else ""
    desc = desc_el.get_text(strip=True) if desc_el else ""

    return {
        "title": title,
        "link": link,
        "domain": domain,
        "desc": desc,
    }


def _parse_google_lens_item(item: dict[str, Any]) -> dict[str, str | None]:
    title = str(item.get("title") or "N/A")
    link = item.get("link")
    source = str(item.get("source") or "")

    parsed_domain = ""
    if isinstance(link, str):
        parsed = urlparse(link)
        parsed_domain = parsed.netloc

    domain = source or parsed_domain

    details: list[str] = []
    if "price" in item and isinstance(item["price"], dict):
        price_value = item["price"].get("value")
        if price_value:
            details.append(f"Price: {price_value}")
    if item.get("condition"):
        details.append(f"Condition: {item['condition']}")
    if "in_stock" in item:
        details.append("In stock" if bool(item["in_stock"]) else "Out of stock")

    return {
        "title": title,
        "link": link if isinstance(link, str) else None,
        "domain": domain,
        "desc": " | ".join(details),
    }


async def _process_yandex_results(
    sites_items: list[Tag],
    httpx_client: httpx.AsyncClient,
) -> tuple[list[str], list[str]]:
    """Process Yandex sites items into formatted results and Twitter URLs."""
    lens_results = []
    twitter_urls_found = []

    # Limit to the first 10 items as requested.
    items_to_process = sites_items[:10]
    parsed_items = [_parse_yandex_sites_item(item) for item in items_to_process]

    extraction_tasks = [
        extract_url_content(
            data["link"],
            httpx_client,
            timeout_seconds=_REVERSE_IMAGE_URL_EXTRACTION_TIMEOUT_SECONDS,
            retries=_REVERSE_IMAGE_URL_EXTRACTION_RETRIES,
        )
        if data["link"]
        else asyncio.sleep(0, result=None)
        for data in parsed_items
    ]

    extracted_contents = await asyncio.gather(*extraction_tasks)

    for data, content in zip(parsed_items, extracted_contents, strict=False):
        link = data["link"] or "#"
        result_line = f"- [{data['title']}]({link}) ({data['domain']}) - {data['desc']}"
        if content:
            result_line += f"\n  Content: {content}"
        lens_results.append(result_line)

        if link and _extract_tweet_id_from_url(link) is not None:
            twitter_urls_found.append(link)

    return lens_results, twitter_urls_found


async def _process_google_lens_results(
    visual_matches: list[dict[str, Any]],
    httpx_client: httpx.AsyncClient,
) -> tuple[list[str], list[str]]:
    lens_results = []
    twitter_urls_found = []

    items_to_process = visual_matches[:10]
    parsed_items = [_parse_google_lens_item(item) for item in items_to_process]

    extraction_tasks = [
        extract_url_content(
            data["link"],
            httpx_client,
            timeout_seconds=_REVERSE_IMAGE_URL_EXTRACTION_TIMEOUT_SECONDS,
            retries=_REVERSE_IMAGE_URL_EXTRACTION_RETRIES,
        )
        if data["link"]
        else asyncio.sleep(0, result=None)
        for data in parsed_items
    ]
    extracted_contents = await asyncio.gather(*extraction_tasks)

    for data, content in zip(parsed_items, extracted_contents, strict=False):
        link = data["link"] or "#"
        result_line = f"- [{data['title']}]({link}) ({data['domain']})"
        if data["desc"]:
            result_line += f" - {data['desc']}"
        if content:
            result_line += f"\n  Content: {content}"
        lens_results.append(result_line)

        if link and _extract_tweet_id_from_url(link) is not None:
            twitter_urls_found.append(link)

    return lens_results, twitter_urls_found


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
        "cbir_page": "sites",
    }

    async def fetch_yandex(client: httpx.AsyncClient, base_url: str) -> httpx.Response:
        resp = await client.get(
            base_url,
            params=params,
            headers=BROWSER_HEADERS,
            follow_redirects=True,
            timeout=60,
        )
        resp.raise_for_status()
        if any(
            phrase in resp.text.lower()
            for phrase in ["captcha", "not a robot", "confirm that you are not a robot"]
        ):
            msg = "Yandex captcha detected"
            raise httpx.HTTPStatusError(
                msg,
                request=resp.request,
                response=resp,
            )
        return resp

    yandex_resp = None
    sites_items: list[Tag] = []
    for domain in [
        "https://yandex.com/images/search",
        "https://yandex.ru/images/search",
    ]:

        async def _yandex_req(
            client: httpx.AsyncClient,
            d: str = domain,
        ) -> httpx.Response:
            return await fetch_yandex(client, d)

        try:
            yandex_resp = await request_with_retries(
                lambda: _yandex_req(httpx_client),
                log_context=f"Yandex reverse image search ({domain})",
            )

            soup = BeautifulSoup(yandex_resp.text, "lxml")
            sites_items = soup.select(".CbirSites-Item, .CbirSitesInfiniteList-Item")
            if sites_items:
                break
            logger.debug("No results found for %s, trying next domain", domain)
        except httpx.HTTPError as exc:
            logger.debug("Yandex lookup failed for %s: %s", domain, exc)
            continue

    if not yandex_resp:
        return [], []

    lens_results, twitter_urls_found = await _process_yandex_results(
        sites_items,
        httpx_client,
    )

    twitter_content = await _fetch_twitter_results(
        twitter_urls_found,
        twitter_api,
        max_tweet_replies,
    )

    return lens_results, twitter_content


async def perform_google_lens_lookup(
    image_url: str,
    serpapi_api_key: str,
    httpx_client: httpx.AsyncClient,
    twitter_api: TwitterApiProtocol,
    max_tweet_replies: int,
) -> tuple[list[str], list[str]]:
    """Perform Google Lens reverse image search via SerpApi and extract results."""
    if not serpapi_api_key:
        return [], []

    params = {
        "engine": "google_lens",
        "url": image_url,
        "type": "visual_matches",
        "api_key": serpapi_api_key,
    }

    async def fetch_google_lens(client: httpx.AsyncClient) -> httpx.Response:
        resp = await client.get(
            "https://serpapi.com/search.json",
            params=params,
            timeout=60,
        )
        resp.raise_for_status()
        return resp

    try:
        response = await request_with_retries(
            lambda: fetch_google_lens(httpx_client),
            log_context="Google Lens reverse image search",
        )
    except httpx.HTTPError as exc:
        logger.debug("Google Lens lookup failed: %s", exc)
        return [], []

    payload: dict[str, Any] = response.json()
    visual_matches = payload.get("visual_matches", [])
    if not isinstance(visual_matches, list):
        return [], []

    lens_results, twitter_urls_found = await _process_google_lens_results(
        visual_matches,
        httpx_client,
    )
    twitter_content = await _fetch_twitter_results(
        twitter_urls_found,
        twitter_api,
        max_tweet_replies,
    )
    return lens_results, twitter_content


def _build_youtube_transcript_api() -> YouTubeTranscriptApi:
    return YouTubeTranscriptApi()


def _summarize_youtube_failure(exc: Exception) -> str:
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    if not lines:
        return exc.__class__.__name__

    for index, line in enumerate(lines):
        if "most likely caused by:" in line.lower() and index + 1 < len(lines):
            return lines[index + 1].rstrip(".")

    for line in lines:
        if not line.lower().startswith(
            "could not retrieve a transcript for the video",
        ):
            return line.rstrip(".")

    return lines[0].rstrip(".")


async def _extract_youtube_transcript_notegpt(
    video_id: str,
    httpx_client: httpx.AsyncClient,
) -> tuple[str, str, str, str | None]:
    """Fetch YouTube transcript and metadata using NoteGPT."""
    title, channel = "Unknown Title", "Unknown Channel"
    transcript_text = "[Transcript not available]"
    failure_reason: str | None = None
    try:
        anonymous_user_id = str(uuid4())
        response = await request_with_retries(
            lambda: httpx_client.get(
                f"https://notegpt.io/api/v2/video-transcript?platform=youtube&video_id={video_id}",
                headers={
                    "Cookie": f"anonymous_user_id={anonymous_user_id}",
                    "User-Agent": FALLBACK_USER_AGENT,
                },
                timeout=30,
            ),
            log_context=f"NoteGPT YouTube transcript {video_id}",
        )
        if response.status_code == httpx.codes.OK:
            data = response.json()
            response_code = data.get("code")
            if response_code == _NOTEGPT_SUCCESS_CODE:
                inner_data = data.get("data", {})
                video_info = inner_data.get("videoInfo", {})
                title = video_info.get("name", title)
                channel = video_info.get("author", channel)

                transcripts = inner_data.get("transcripts", {})
                transcript_found = False
                if transcripts:
                    # Try to find English first, otherwise use first available
                    langs = list(transcripts.keys())
                    target_lang = (
                        "en" if "en" in langs else (langs[0] if langs else None)
                    )
                    if target_lang:
                        lang_data = transcripts[target_lang]
                        # Prefer 'default', then 'auto', then 'custom'
                        transcript_list = (
                            lang_data.get("default")
                            or lang_data.get("auto")
                            or lang_data.get("custom")
                        )
                        if transcript_list:
                            transcript_text = " ".join(
                                item["text"] for item in transcript_list
                            )
                            transcript_found = True
                if not transcript_found:
                    failure_reason = "No transcript segments returned by NoteGPT"
            else:
                failure_reason = f"NoteGPT returned code {response_code}"
        else:
            failure_reason = f"NoteGPT HTTP {response.status_code}"
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        failure_reason = _summarize_youtube_failure(exc)
        logger.info(
            "Failed to fetch YouTube transcript from NoteGPT for %s: %s",
            video_id,
            failure_reason,
        )
    return title, channel, transcript_text, failure_reason


async def _extract_youtube_transcript_api(
    video_id: str,
    httpx_client: httpx.AsyncClient,
) -> tuple[str, str, str, str | None]:
    """Fetch YouTube transcript and metadata using YouTubeTranscriptApi."""
    title, channel = "Unknown Title", "Unknown Channel"
    try:
        resp = await request_with_retries(
            lambda: httpx_client.get(
                f"https://www.youtube.com/watch?v={video_id}",
                headers=BROWSER_HEADERS,
                follow_redirects=True,
                timeout=30,
            ),
            log_context=f"YouTube metadata {video_id}",
        )
        if resp.status_code != httpx.codes.OK:
            resp = await request_with_retries(
                lambda: httpx_client.get(
                    f"https://www.youtube.com/watch?v={video_id}",
                    headers={"User-Agent": FALLBACK_USER_AGENT},
                    follow_redirects=True,
                    timeout=30,
                ),
                log_context=f"YouTube metadata fallback {video_id}",
            )

        if resp.status_code == httpx.codes.OK:
            resp_text = resp.text
            title_match = re.search(
                r'<meta name="title" content="(.*?)">',
                resp_text,
            )
            if title_match:
                title = html.unescape(title_match.group(1))

            channel_match = re.search(
                r'<link itemprop="name" content="(.*?)">',
                resp_text,
            )
            if channel_match:
                channel = html.unescape(channel_match.group(1))
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        logger.debug("Failed to fetch YouTube metadata for %s: %s", video_id, exc)

    transcript_text = "[Transcript not available]"
    failure_reason: str | None = None
    try:
        ytt_api = _build_youtube_transcript_api()
        transcript_list = await asyncio.to_thread(ytt_api.list, video_id)
        try:
            # Prefer English, but fall back to many other languages if needed
            transcript = transcript_list.find_transcript(
                [
                    "en",
                    "es",
                    "fr",
                    "de",
                    "it",
                    "pt",
                    "ru",
                    "ja",
                    "ko",
                    "zh-Hans",
                    "zh-Hant",
                ],
            )
        except YouTubeTranscriptApiException:
            # Fall back to the first available transcript
            try:
                transcript = next(iter(transcript_list))
            except StopIteration:
                transcript = None

        if transcript:
            fetched_transcript = await asyncio.to_thread(transcript.fetch)
            transcript_data = fetched_transcript.to_raw_data()
            transcript_text = " ".join(x["text"] for x in transcript_data)
    except (
        YouTubeTranscriptApiException,
        httpx.HTTPError,
        RuntimeError,
        ValueError,
    ) as exc:
        failure_reason = _summarize_youtube_failure(exc)
        logger.info(
            "Failed to fetch YouTube transcript for %s: %s",
            video_id,
            failure_reason,
        )
    if transcript_text == "[Transcript not available]" and failure_reason is None:
        failure_reason = "No transcripts available for this video"
    return title, channel, transcript_text, failure_reason


async def extract_youtube_transcript_with_reason(
    video_id: str,
    httpx_client: httpx.AsyncClient,
    *,
    method: str = "youtube-transcript-api",
) -> tuple[str | None, str | None]:
    """Fetch YouTube transcript and metadata with failure reason."""
    if method == "notegpt":
        (
            title,
            channel,
            transcript_text,
            failure_reason,
        ) = await _extract_youtube_transcript_notegpt(
            video_id,
            httpx_client,
        )
    else:
        (
            title,
            channel,
            transcript_text,
            failure_reason,
        ) = await _extract_youtube_transcript_api(
            video_id,
            httpx_client,
        )

    if (
        title == "Unknown Title"
        and channel == "Unknown Channel"
        and transcript_text == "[Transcript not available]"
    ):
        return (
            None,
            failure_reason or "Unable to fetch metadata and transcript",
        )

    return (
        (
            f"YouTube Video ID: {video_id}\n"
            f"Title: {title}\n"
            f"Channel: {channel}\n"
            f"Transcript:\n{transcript_text}"
        ),
        failure_reason,
    )


async def extract_youtube_transcript(
    video_id: str,
    httpx_client: httpx.AsyncClient,
    *,
    method: str = "youtube-transcript-api",
) -> str | None:
    """Fetch YouTube transcript and metadata."""
    transcript, _failure_reason = await extract_youtube_transcript_with_reason(
        video_id,
        httpx_client,
        method=method,
    )
    return transcript


async def _resolve_reddit_share_url(
    post_url: str,
    httpx_client: httpx.AsyncClient,
) -> str:
    """Resolve Reddit share URL to canonical URL."""
    if not re.search(r"/r/\w+/s/", post_url):
        return post_url

    try:
        resolve_resp = await request_with_retries(
            lambda: httpx_client.head(
                post_url,
                headers=BROWSER_HEADERS,
                timeout=30,
                follow_redirects=True,
            ),
            log_context=f"Reddit share URL resolution {post_url}",
        )
        resolved_url = str(resolve_resp.url)
    except httpx.HTTPError:
        logger.debug("Failed to resolve Reddit share URL %s", post_url)
        return post_url

    # Strip query params added by Reddit share redirect
    if "?" in resolved_url:
        resolved_url = resolved_url.split("?", maxsplit=1)[0]
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
) -> tuple[dict, str]:
    """Fetch Reddit JSON data from the given URL.

    Returns:
        Tuple of (json_data, resolved_url) where resolved_url is the final URL
        after following redirects (useful for share URL resolution).

    """
    response = await request_with_retries(
        lambda: httpx_client.get(
            json_url,
            headers=BROWSER_HEADERS,
            timeout=30,
            follow_redirects=True,
        ),
        log_context=f"Reddit JSON fetch {json_url}",
    )
    if response.status_code != httpx.codes.OK:
        response = await request_with_retries(
            lambda: httpx_client.get(
                json_url,
                headers={"User-Agent": FALLBACK_USER_AGENT},
                timeout=30,
                follow_redirects=True,
            ),
            log_context=f"Reddit JSON fetch fallback {json_url}",
        )
    response.raise_for_status()
    return response.json(), str(response.url)


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


async def _extract_reddit_json_direct(
    post_url: str,
    httpx_client: httpx.AsyncClient,  # noqa: ARG001 - unused, kept for signature
    max_comments: int | None,
) -> str:
    """Extract Reddit post using a dedicated direct connection.

    Args:
        post_url: The Reddit post URL (may be share URL)
        httpx_client: Unused, kept for signature compatibility
        max_comments: Maximum number of comments to extract

    Returns:
        Formatted post text

    Raises:
        httpx.HTTPError: If request fails
        KeyError, TypeError, ValueError: If parsing fails

    """
    # Create a fresh client for direct Reddit JSON access
    async with httpx.AsyncClient(
        follow_redirects=True,
        headers=BROWSER_HEADERS,
    ) as client:
        # Resolve share URLs first if needed
        if re.search(r"/r/\w+/s/", post_url):
            resolve_resp = await client.head(
                post_url,
                headers=BROWSER_HEADERS,
                timeout=30,
            )
            if resolve_resp.status_code != httpx.codes.OK:
                resolve_resp = await client.head(
                    post_url,
                    headers={"User-Agent": FALLBACK_USER_AGENT},
                    timeout=30,
                )
            post_url = str(resolve_resp.url)
            if "?" in post_url:
                post_url = post_url.split("?")[0]

        json_url = _build_reddit_json_url(post_url)
        response = await client.get(
            json_url,
            headers=BROWSER_HEADERS,
            timeout=30,
        )
        if response.status_code != httpx.codes.OK:
            response = await client.get(
                json_url,
                headers={"User-Agent": FALLBACK_USER_AGENT},
                timeout=30,
            )
        response.raise_for_status()
        data = response.json()

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
    return post_text


async def extract_reddit_post_json(
    post_url: str,
    httpx_client: httpx.AsyncClient,
    *,
    max_comments: int | None = None,
) -> str | None:
    """Extract Reddit post content using JSON endpoints.

    Args:
        post_url: The Reddit post URL to extract content from
        httpx_client: The HTTP client to use for requests
        max_comments: Maximum number of comments to extract. None = unlimited.

    Returns:
        Formatted post text with comments, or None if extraction failed

    """
    original_url = post_url

    # Try direct connection first
    try:
        return await _extract_reddit_json_direct(
            original_url,
            httpx_client,
            max_comments,
        )
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        logger.debug(
            "Failed to fetch Reddit content (JSON) for %s: %s",
            original_url,
            exc,
        )
        return None


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
            submission.subreddit.display_name if submission.subreddit else "unknown"
        )
        author_name = submission.author.name if submission.author else "[deleted]"
        selftext = getattr(submission, "selftext", "") or ""

        post_text = (
            f"Reddit Post: {title}\nSubreddit: r/{subreddit_name}\n"
            f"Author: u/{author_name}\n\n{selftext}"
        )

        if not getattr(submission, "is_self", True) and getattr(
            submission,
            "url",
            None,
        ):
            post_text += f"\nLink: {submission.url}"

        submission.comment_sort = "top"
        await submission.comments()
        comments_list = submission.comments.list() if submission.comments else []
        top_comments = (
            comments_list if max_comments is None else comments_list[:max_comments]
        )

        if top_comments:
            post_text += "\n\nTop Comments:"
            for comment in top_comments:
                if isinstance(comment, asyncpraw_models.MoreComments):
                    continue
                comment_author = comment.author.name if comment.author else "[deleted]"
                comment_body = getattr(comment, "body", "") or ""
                post_text += f"\n- u/{comment_author}: {comment_body}"

    except (
        asyncprawcore.exceptions.AsyncPrawcoreException,
        asyncpraw_exceptions.RedditAPIException,
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
    max_comments: int | None = None,
) -> str | None:
    """Extract Reddit post content using the configured method.

    Args:
        post_url: The Reddit post URL to extract content from
        httpx_client: The HTTP client to use for requests
        reddit_client: Optional AsyncPRAW Reddit client instance
        max_comments: Maximum number of comments to extract. None = unlimited.

    Returns:
        Formatted post text with comments, or None if extraction failed

    """
    if reddit_client:
        # For PRAW, we need to resolve share URLs first since PRAW requires
        # canonical URLs. Share links (format: /r/.../s/...) need to be
        # resolved via HTTP redirect.
        if re.search(r"/r/\w+/s/", post_url):
            post_url = await _resolve_reddit_share_url(
                post_url,
                httpx_client,
            )

        return await extract_reddit_post_praw(
            post_url,
            reddit_client,
            max_comments=max_comments,
        )

    # For JSON extraction, share URL resolution is handled automatically
    # via follow_redirects during the GET request.
    return await extract_reddit_post_json(
        post_url,
        httpx_client,
        max_comments=max_comments,
    )
