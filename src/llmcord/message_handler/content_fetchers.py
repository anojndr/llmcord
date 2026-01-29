"""Fetch external content referenced in messages."""
# ruff: noqa: E501

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import asyncpraw
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi

from llmcord.bad_keys import get_bad_keys_db
from llmcord.config import BROWSER_HEADERS, is_gemini_model

from .attachment_utils import extract_pdf_text, fetch_tweet_with_replies

if TYPE_CHECKING:
    import discord
    import httpx

    from llmcord.models import MsgNode

    from .shared import TwitterApiProtocol

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LensRequest:
    """Inputs required to fetch Google Lens results."""

    cleaned_content: str
    curr_msg: discord.Message
    curr_node: MsgNode
    httpx_client: httpx.AsyncClient
    twitter_api: TwitterApiProtocol
    max_tweet_replies: int


async def apply_google_lens_results(request: LensRequest) -> str:
    """Append Google Lens (Yandex) results to the cleaned content if applicable."""
    cached = _get_cached_lens_results(request.curr_msg.id, request.curr_node)
    if cached:
        return request.cleaned_content + cached

    image_url = _get_first_image_url(request.curr_msg)
    if not image_url:
        return request.cleaned_content

    try:
        soup = await _fetch_yandex_soup(request.httpx_client, image_url)
    except Exception:
        logger.exception("Error fetching Yandex results")
        return request.cleaned_content

    lens_results, twitter_urls_found = _parse_lens_results(soup)
    if not lens_results:
        return request.cleaned_content

    twitter_content = await _fetch_twitter_content(
        twitter_urls_found,
        request.twitter_api,
        request.max_tweet_replies,
    )
    result_text = _format_lens_results(lens_results, twitter_content)

    request.curr_node.lens_results = result_text
    get_bad_keys_db().save_message_search_data(
        request.curr_msg.id,
        lens_results=result_text,
    )
    logger.info(
        "Saved lens results for message %s",
        request.curr_msg.id,
    )

    return request.cleaned_content + result_text


def _get_cached_lens_results(message_id: int, curr_node: MsgNode) -> str | None:
    """Return cached lens results for a message if present."""
    _, _, cached_lens_results = get_bad_keys_db().get_message_search_data(message_id)
    if not cached_lens_results:
        return None

    curr_node.lens_results = cached_lens_results
    logger.debug(
        "Using cached lens results for message %s",
        message_id,
    )
    return cached_lens_results


def _get_first_image_url(curr_msg: discord.Message) -> str | None:
    """Return the first image attachment URL, if any."""
    return next(
        (
            att.url
            for att in curr_msg.attachments
            if att.content_type
            and att.content_type.startswith("image")
        ),
        None,
    )


async def _fetch_yandex_soup(
    httpx_client: httpx.AsyncClient,
    image_url: str,
) -> BeautifulSoup:
    """Fetch Yandex image search results for the given image URL."""
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
    return BeautifulSoup(
        yandex_resp.text,
        "lxml",
    )


def _parse_lens_results(soup: BeautifulSoup) -> tuple[list[str], list[str]]:
    """Parse Yandex results into formatted items and Twitter URLs."""
    lens_results = []
    twitter_urls_found = []
    sites_items = soup.select(".CbirSites-Item")

    if not sites_items:
        return lens_results, twitter_urls_found

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

        if re.search(
            r"(?:twitter\.com|x\.com)/[a-zA-Z0-9_]+/status/[0-9]+",
            link,
        ):
            twitter_urls_found.append(link)

    return lens_results, twitter_urls_found


async def _fetch_twitter_content(
    twitter_urls_found: list[str],
    twitter_api: TwitterApiProtocol,
    max_tweet_replies: int,
) -> list[str]:
    """Fetch tweet content for a list of Twitter/X URLs."""
    twitter_content = []
    for twitter_url in twitter_urls_found:
        tweet_id_match = re.search(
            r"(?:twitter\.com|x\.com)/[a-zA-Z0-9_]+/status/([0-9]+)",
            twitter_url,
        )
        if not tweet_id_match:
            continue

        tweet_text = await fetch_tweet_with_replies(
            twitter_api,
            int(tweet_id_match.group(1)),
            max_replies=max_tweet_replies,
            include_url=True,
            tweet_url=twitter_url,
        )
        if tweet_text:
            twitter_content.append(tweet_text)

    return twitter_content


def _format_lens_results(lens_results: list[str], twitter_content: list[str]) -> str:
    """Format lens results and twitter content into a single string."""
    result_text = (
        "\n\nanswer the user's query based on the yandex reverse image results:\n"
        + "\n".join(lens_results)
    )
    if twitter_content:
        result_text += (
            "\n\n--- Extracted Twitter/X Content ---"
            + "".join(twitter_content)
        )
    return result_text


async def fetch_youtube_transcripts(
    cleaned_content: str,
    httpx_client: httpx.AsyncClient,
) -> list[str]:
    """Fetch YouTube transcripts referenced in the content."""
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
    if not video_ids:
        return []

    yt_results = await asyncio.gather(
        *[fetch_yt_transcript(vid) for vid in video_ids],
    )
    return [t for t in yt_results if t is not None]


async def fetch_tweets(
    cleaned_content: str,
    twitter_api: TwitterApiProtocol,
    max_tweet_replies: int,
) -> list[str]:
    """Fetch tweet content referenced in the message text."""
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

    return tweets


async def fetch_reddit_posts(
    cleaned_content: str,
    reddit_client: asyncpraw.Reddit | None,
) -> list[str]:
    """Fetch Reddit content referenced in the message text."""
    if not reddit_client:
        return []

    reddit_posts = []
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

    return reddit_posts


async def extract_pdf_texts(
    processed_attachments: list[dict[str, object]],
    actual_model: str,
) -> list[str]:
    """Extract text from PDF attachments for non-Gemini models."""
    # Genuine Gemini models handle PDFs natively, but Gemma and other providers need text extraction
    if is_gemini_model(actual_model):
        return []

    pdf_attachments = [
        att for att in processed_attachments
        if att["content_type"] == "application/pdf"
    ]
    if not pdf_attachments:
        return []

    pdf_extraction_tasks = [
        extract_pdf_text(att["content"])
        for att in pdf_attachments
    ]
    pdf_results = await asyncio.gather(*pdf_extraction_tasks)

    pdf_texts = []
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

    return pdf_texts
