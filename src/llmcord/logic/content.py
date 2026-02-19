"""Content extraction and attachment processing logic."""

from __future__ import annotations

import asyncio
import logging
import re
from base64 import b64encode
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

from llmcord.core.config import get_config, is_gemini_model
from llmcord.core.error_handling import log_exception
from llmcord.globals import reddit_client
from llmcord.services.database import get_db
from llmcord.services.extractors import (
    TwitterApiProtocol,
    download_attachments,
    extract_pdf_images,
    extract_pdf_text,
    extract_reddit_post,
    extract_url_content,
    extract_youtube_transcript_with_reason,
    fetch_tweet_with_replies,
    perform_google_lens_lookup,
    perform_yandex_lookup,
    process_attachments,
)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import discord
    import httpx

    from llmcord.core.models import MsgNode


_YOUTUBE_ID_RE = re.compile(
    r"(?:https?://)?(?:[a-zA-Z0-9-]+\.)?"
    r"(?:youtube\.com/(?:watch\?[^#\s]*?v=|embed/|v/|shorts/|live/)|youtu\.be/)"
    r"([a-zA-Z0-9_-]{11})",
)
_TWEET_ID_RE = re.compile(
    (
        r"(?:https?:\/\/)?(?:[a-zA-Z0-9-]+\.)?"
        r"(?:twitter\.com|x\.com)\/[a-zA-Z0-9_]+\/status\/([0-9]+)"
    ),
)
_REDDIT_URL_RE = re.compile(
    (
        r"(https?:\/\/(?:[a-zA-Z0-9-]+\.)?"
        r"(?:reddit\.com\/r\/[a-zA-Z0-9_]+\/(?:comments|s)\/"
        r"[a-zA-Z0-9_]+(?:[\w\-\.\/\?\=\&%]*)"
        r"|redd\.it\/[a-zA-Z0-9_]+))"
    ),
)
_GENERIC_URL_RE = re.compile(r"https?://[^\s<>{}\[\]]+")


def _format_youtube_failure(video_id: str, failure_reason: str | None) -> str:
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    if not failure_reason:
        return youtube_url

    normalized_reason = " ".join(failure_reason.split())
    max_reason_len = 120
    if len(normalized_reason) > max_reason_len:
        normalized_reason = normalized_reason[: max_reason_len - 3] + "..."

    return f"{youtube_url} ({normalized_reason})"


def _extract_result_url(result_line: str) -> str | None:
    markdown_match = re.search(r"\]\((https?://[^)\s]+)\)", result_line)
    if markdown_match:
        return markdown_match.group(1)

    plain_match = re.search(r"https?://[^\s)]+", result_line)
    if plain_match:
        return plain_match.group(0)
    return None


def _normalize_result_url(url: str) -> str:
    parsed = urlsplit(url)
    normalized_path = parsed.path.rstrip("/")
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            normalized_path,
            "",
            "",
        ),
    )


def _merge_reverse_image_results(
    yandex_results: list[str],
    google_results: list[str],
    *,
    prefer_overlapping_matches: bool = False,
) -> list[str]:
    interleaved: list[str] = []
    max_len = max(len(yandex_results), len(google_results))
    for index in range(max_len):
        if index < len(yandex_results):
            interleaved.append(f"[Yandex] {yandex_results[index]}")
        if index < len(google_results):
            interleaved.append(f"[Google Lens] {google_results[index]}")

    deduped: OrderedDict[str, str] = OrderedDict()
    for line in interleaved:
        extracted_url = _extract_result_url(line)
        if extracted_url:
            dedupe_key = _normalize_result_url(extracted_url)
        else:
            dedupe_key = line.strip().lower()

        if dedupe_key not in deduped:
            deduped[dedupe_key] = line

    if not prefer_overlapping_matches:
        return list(deduped.values())

    yandex_keys = {
        _normalize_result_url(url)
        for line in yandex_results
        if (url := _extract_result_url(line))
    }
    google_keys = {
        _normalize_result_url(url)
        for line in google_results
        if (url := _extract_result_url(line))
    }
    overlap_keys = yandex_keys & google_keys

    overlap_lines: list[str] = []
    single_lines: list[str] = []
    for key, line in deduped.items():
        if key in overlap_keys:
            overlap_lines.append(line)
        else:
            single_lines.append(line)

    return [*overlap_lines, *single_lines]


async def _collect_youtube_transcripts(
    context: ExternalContentContext,
) -> tuple[list[str], list[str]]:
    video_ids = _YOUTUBE_ID_RE.findall(context.cleaned_content)
    if not video_ids or not context.enable_youtube_transcripts:
        return [], []

    # Deduplicate while preserving order
    unique_video_ids = list(OrderedDict.fromkeys(video_ids))

    results = await asyncio.gather(
        *[
            extract_youtube_transcript_with_reason(
                vid,
                context.httpx_client,
                method=context.youtube_transcript_method,
            )
            for vid in unique_video_ids
        ],
    )

    extracted: list[str] = []
    failed_extractions: list[str] = []
    for vid, (res, failure_reason) in zip(unique_video_ids, results, strict=False):
        if res is not None:
            extracted.append(res)
            if "[Transcript not available]" in res:
                failed_extractions.append(
                    _format_youtube_failure(
                        vid,
                        failure_reason or "transcript unavailable",
                    ),
                )
        else:
            failed_extractions.append(
                _format_youtube_failure(vid, failure_reason),
            )
    return extracted, failed_extractions


async def _collect_tweets(context: ExternalContentContext) -> list[str]:
    tweet_ids = _TWEET_ID_RE.findall(context.cleaned_content)
    if not tweet_ids:
        return []

    tweets = await asyncio.gather(
        *[
            fetch_tweet_with_replies(
                context.twitter_api,
                int(tweet_id),
                max_replies=context.max_tweet_replies,
                include_url=False,
            )
            for tweet_id in tweet_ids
        ],
    )
    return [tweet for tweet in tweets if tweet]


async def _collect_reddit_posts(context: ExternalContentContext) -> list[str]:
    post_urls = _REDDIT_URL_RE.findall(context.cleaned_content)
    if not post_urls:
        return []

    reddit_posts = await asyncio.gather(
        *[
            extract_reddit_post(
                post_url,
                context.httpx_client,
                reddit_client,
            )
            for post_url in post_urls
        ],
    )
    return [post for post in reddit_posts if post]


def _normalize_generic_urls(text: str) -> list[str]:
    generic_urls = _GENERIC_URL_RE.findall(text)
    normalized: list[str] = []
    for raw_url in generic_urls:
        cleaned_url = raw_url.rstrip('.,;:!?)"')
        if not cleaned_url:
            continue

        lowered = cleaned_url.lower()
        if any(
            domain in lowered
            for domain in [
                "twitter.com",
                "x.com",
                "youtube.com",
                "youtu.be",
                "reddit.com",
                "redd.it",
                "tiktok.com",
                "vm.tiktok.com",
                "m.tiktok.com",
                "vt.tiktok.com",
                "facebook.com",
                "m.facebook.com",
                "fb.watch",
            ]
        ):
            continue

        if cleaned_url not in normalized:
            normalized.append(cleaned_url)
    return normalized


async def _collect_generic_url_contents(
    context: ExternalContentContext,
) -> tuple[list[str], list[str]]:
    urls = _normalize_generic_urls(context.cleaned_content)
    if not urls:
        return [], []

    extracted = await asyncio.gather(
        *[
            extract_url_content(
                url,
                context.httpx_client,
            )
            for url in urls
        ],
    )

    results: list[str] = []
    failed_extractions: list[str] = []
    for url, text in zip(urls, extracted, strict=False):
        if text:
            results.append(f"--- URL Content: {url} ---\n{text}")
        else:
            failed_extractions.append(url)
    return results, failed_extractions


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
    youtube_transcript_method: str
    curr_node: MsgNode


async def apply_googlelens(context: GoogleLensContext) -> str:
    """Apply Google Lens enrichment to content if requested."""
    cleaned_content = context.cleaned_content
    if not cleaned_content.lower().startswith("googlelens"):
        return cleaned_content

    cleaned_content = cleaned_content[10:].strip()
    _, _, cached_lens_results = get_db().get_message_search_data(
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

    image_attachments = [
        att
        for att in context.curr_msg.attachments
        if att.content_type and att.content_type.startswith("image")
    ]
    if not image_attachments:
        return cleaned_content

    try:
        config = get_config()
        serpapi_api_key = str(config.get("serpapi_api_key") or "").strip()
        prefer_overlapping_matches = bool(
            config.get("googlelens_prefer_overlapping_matches", True),
        )

        async def _lookup_single_image(
            index: int,
            att: discord.Attachment,
        ) -> str:
            image_url = att.url
            yandex_task = perform_yandex_lookup(
                image_url,
                context.httpx_client,
                context.twitter_api,
                context.max_tweet_replies,
            )
            google_lens_task = perform_google_lens_lookup(
                image_url,
                serpapi_api_key,
                context.httpx_client,
                context.twitter_api,
                context.max_tweet_replies,
            )
            (
                (yandex_results, yandex_twitter),
                (google_results, google_twitter),
            ) = await asyncio.gather(yandex_task, google_lens_task)
            lens_results = _merge_reverse_image_results(
                yandex_results,
                google_results,
                prefer_overlapping_matches=prefer_overlapping_matches,
            )
            twitter_content = list(
                OrderedDict.fromkeys([*yandex_twitter, *google_twitter]),
            )

            if not lens_results:
                return ""

            image_name = f"Image {index + 1}"
            if att.filename:
                image_name += f" ({att.filename})"

            result_text = f"\n\n{image_name}\nResults for {image_name}:\n" + "\n".join(
                lens_results,
            )
            if twitter_content:
                result_text += (
                    f"\n\n--- Extracted Twitter/X Content for {image_name} ---"
                    + "".join(twitter_content)
                )
            return result_text

        results = await asyncio.gather(
            *(_lookup_single_image(i, att) for i, att in enumerate(image_attachments)),
        )
        all_results_text = "".join(results)

        if all_results_text:
            instruction_text = (
                "\n\nAnswer the user's query based on the above reverse image "
                "results (Google Lens + Yandex). Base your answers on frequency, "
                "as that is most likely the correct answer. For example, if the "
                "query is 'what anime?' and one title appears more frequently in "
                "the results for an image, that title is most likely the answer. "
                "In your responses, provide the top 3 most frequently appearing "
                "results for each image, clearly labeling which image you are "
                "referring to. For each result, include a confidence level based "
                "on how often it appears and its relevance, following this format:\n"
                "1. [Title] - [X]% confidence - [Short reasoning]\n"
                "Example:\n"
                "1. Towa no Yuugure (Dusk Beyond the End of the World) - 100% "
                "confidence - most likely the answer\n"
                "2. SPY x FAMILY - 10% confidence - very unlikely to be the answer"
            )
            cleaned_content += all_results_text + instruction_text

            context.curr_node.lens_results = all_results_text
            get_db().save_message_search_data(
                str(context.curr_msg.id),
                lens_results=all_results_text,
            )
            logger.info(
                "Saved lens results for message %s",
                context.curr_msg.id,
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        log_exception(
            logger=logger,
            message="Error fetching reverse image search results",
            error=exc,
            context={
                "message_id": context.curr_msg.id,
                "image_attachments": len(image_attachments),
            },
        )

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
    successful_pairs = await download_attachments(
        attachments,
        httpx_client,
    )
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
    (
        youtube_result,
        tweets,
        reddit_posts,
        generic_result,
        pdf_texts,
    ) = await asyncio.gather(
        _collect_youtube_transcripts(context),
        _collect_tweets(context),
        _collect_reddit_posts(context),
        _collect_generic_url_contents(context),
        _extract_pdf_texts(
            processed_attachments=context.processed_attachments,
            actual_model=context.actual_model,
        ),
    )

    yt_transcripts, yt_failed_extractions = youtube_result
    generic_url_contents, generic_failed_extractions = generic_result
    context.curr_node.failed_extractions.extend(yt_failed_extractions)
    context.curr_node.failed_extractions.extend(generic_failed_extractions)

    return yt_transcripts + tweets + reddit_posts + generic_url_contents + pdf_texts


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
