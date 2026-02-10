"""Content extraction and attachment processing logic."""

import asyncio
import logging
import re
from base64 import b64encode
from dataclasses import dataclass

import discord
import httpx

from llmcord.core.config import is_gemini_model
from llmcord.core.models import MsgNode
from llmcord.globals import reddit_client
from llmcord.services.database import get_bad_keys_db
from llmcord.services.extractors import (
    TwitterApiProtocol,
    download_attachments,
    extract_pdf_images,
    extract_pdf_text,
    extract_reddit_post,
    extract_youtube_transcript,
    fetch_tweet_with_replies,
    perform_yandex_lookup,
    process_attachments,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GoogleLensContext:
    """Inputs for Google Lens enrichment."""

    cleaned_content: str
    curr_msg: discord.Message
    curr_node: MsgNode
    httpx_client: httpx.AsyncClient
    twitter_api: TwitterApiProtocol
    max_tweet_replies: int


@dataclass(slots=True)
class ExternalContentContext:
    """Inputs for external content extraction."""

    cleaned_content: str
    httpx_client: httpx.AsyncClient
    twitter_api: TwitterApiProtocol
    max_tweet_replies: int
    processed_attachments: list[dict[str, bytes | str | None]]
    actual_model: str
    enable_youtube_transcripts: bool
    youtube_transcript_proxy: str | None
    reddit_proxy: str | None


async def apply_googlelens(context: GoogleLensContext) -> str:
    """Apply Google Lens enrichment to content if requested."""
    cleaned_content = context.cleaned_content
    if not cleaned_content.lower().startswith("googlelens"):
        return cleaned_content

    cleaned_content = cleaned_content[10:].strip()
    _, _, cached_lens_results = get_bad_keys_db().get_message_search_data(
        str(context.curr_msg.id),
    )
    if cached_lens_results:
        cleaned_content = cleaned_content + cached_lens_results
        context.curr_node.lens_results = cached_lens_results
        logger.debug(
            "Using cached lens results for message %s",
            context.curr_msg.id,
        )
        return cleaned_content

    image_url = next(
        (
            att.url
            for att in context.curr_msg.attachments
            if att.content_type and att.content_type.startswith("image")
        ),
        None,
    )
    if not image_url:
        return cleaned_content

    try:
        lens_results, twitter_content = await perform_yandex_lookup(
            image_url,
            context.httpx_client,
            context.twitter_api,
            context.max_tweet_replies,
        )

        if lens_results:
            result_text = (
                "\n\nanswer the user's query based on the yandex "
                "reverse image results:\n" + "\n".join(lens_results)
            )
            if twitter_content:
                result_text += "\n\n--- Extracted Twitter/X Content ---" + "".join(
                    twitter_content,
                )
            cleaned_content += result_text

            context.curr_node.lens_results = result_text
            get_bad_keys_db().save_message_search_data(
                str(context.curr_msg.id),
                lens_results=result_text,
            )
            logger.info(
                "Saved lens results for message %s",
                context.curr_msg.id,
            )
    except Exception:
        logger.exception("Error fetching Yandex results")

    return cleaned_content


def get_allowed_attachment_types(actual_model: str) -> tuple[str, ...]:
    """Get allowed attachment MIME types for the model."""
    allowed_types: tuple[str, ...] = (
        "text",
        "image",
        "application/pdf",
    )
    if is_gemini_model(actual_model):
        allowed_types += ("audio", "video")
    return allowed_types


async def download_and_process_attachments(
    *,
    attachments: list[discord.Attachment],
    httpx_client: httpx.AsyncClient,
) -> tuple[
    list[discord.Attachment],
    list[httpx.Response],
    list[dict[str, bytes | str | None]],
]:
    """Download and process message attachments."""
    successful_pairs = await download_attachments(attachments, httpx_client)
    good_attachments = [pair[0] for pair in successful_pairs]
    attachment_responses = [pair[1] for pair in successful_pairs]
    processed_attachments = await process_attachments(successful_pairs)
    return good_attachments, attachment_responses, processed_attachments


async def _extract_pdf_texts(
    *,
    processed_attachments: list[dict[str, bytes | str | None]],
    actual_model: str,
) -> list[str]:
    """Extract text from PDF attachments for non-Gemini models."""
    if is_gemini_model(actual_model):
        return []

    pdf_attachments = [
        att for att in processed_attachments if att["content_type"] == "application/pdf"
    ]
    if not pdf_attachments:
        return []

    pdf_extraction_tasks = [
        extract_pdf_text(att["content"])
        for att in pdf_attachments
        if isinstance(att["content"], bytes)
    ]
    pdf_results = await asyncio.gather(*pdf_extraction_tasks)

    pdf_texts: list[str] = []
    for i, pdf_text in enumerate(pdf_results):
        if pdf_text:
            pdf_label = f"--- PDF Attachment {i + 1} Content ---"
            pdf_texts.append(f"{pdf_label}\n{pdf_text}")
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


async def extract_pdf_images_for_model(
    *,
    processed_attachments: list[dict[str, bytes | str | None]],
    actual_model: str,
) -> list[dict[str, object]]:
    """Extract images from PDF attachments for non-Gemini models."""
    if is_gemini_model(actual_model):
        return []

    pdf_attachments = [
        att for att in processed_attachments if att["content_type"] == "application/pdf"
    ]
    if not pdf_attachments:
        return []

    extraction_tasks = [
        extract_pdf_images(att["content"])
        for att in pdf_attachments
        if isinstance(att["content"], bytes)
    ]
    all_results = await asyncio.gather(*extraction_tasks)

    image_dicts: list[dict[str, object]] = []
    for pdf_images in all_results:
        for content_type, img_bytes in pdf_images:
            image_dicts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:{content_type};base64,"
                            f"{b64encode(img_bytes).decode('utf-8')}"
                        ),
                    },
                },
            )

    if image_dicts:
        logger.info(
            "Extracted %s image(s) from PDF attachment(s)",
            len(image_dicts),
        )

    return image_dicts


async def collect_external_content(
    context: ExternalContentContext,
) -> list[str]:
    """Collect content from external sources (YouTube, Twitter, Reddit, PDFs)."""
    video_ids = re.findall(
        (
            r"(?:https?:\/\/)?(?:[a-zA-Z0-9-]+\.)?"
            r"(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})"
        ),
        context.cleaned_content,
    )
    if video_ids and context.enable_youtube_transcripts:
        yt_results = await asyncio.gather(
            *[
                extract_youtube_transcript(
                    vid,
                    context.httpx_client,
                    proxy_url=context.youtube_transcript_proxy,
                )
                for vid in video_ids
            ],
        )
        yt_transcripts = [t for t in yt_results if t is not None]
    else:
        yt_transcripts = []

    tweets: list[str] = []
    for tweet_id in re.findall(
        (
            r"(?:https?:\/\/)?(?:[a-zA-Z0-9-]+\.)?"
            r"(?:twitter\.com|x\.com)\/[a-zA-Z0-9_]+\/status\/([0-9]+)"
        ),
        context.cleaned_content,
    ):
        tweet_text = await fetch_tweet_with_replies(
            context.twitter_api,
            int(tweet_id),
            max_replies=context.max_tweet_replies,
            include_url=False,
        )
        if tweet_text:
            tweets.append(tweet_text)

    reddit_posts: list[str] = []
    for post_url in re.findall(
        (
            r"(https?:\/\/(?:[a-zA-Z0-9-]+\.)?"
            r"(?:reddit\.com\/r\/[a-zA-Z0-9_]+\/(?:comments|s)\/"
            r"[a-zA-Z0-9_]+(?:[\w\-\.\/\?\=\&%]*)"
            r"|redd\.it\/[a-zA-Z0-9_]+))"
        ),
        context.cleaned_content,
    ):
        post_text = await extract_reddit_post(
            post_url,
            context.httpx_client,
            reddit_client,
            proxy_url=context.reddit_proxy,
        )
        if post_text:
            reddit_posts.append(post_text)

    pdf_texts = await _extract_pdf_texts(
        processed_attachments=context.processed_attachments,
        actual_model=context.actual_model,
    )

    return yt_transcripts + tweets + reddit_posts + pdf_texts


def is_googlelens_query(
    new_msg: discord.Message,
    discord_bot: discord.Client,
) -> bool:
    """Check if the message is a Google Lens query."""
    if not discord_bot.user:
        return False
    content_for_lens_check = (
        new_msg.content.lower().removeprefix(discord_bot.user.mention.lower()).strip()
    )
    if content_for_lens_check.startswith("at ai"):
        content_for_lens_check = content_for_lens_check[5:].strip()
    return content_for_lens_check.startswith("googlelens")
