"""Message building and processing logic."""

import asyncio
import logging
import re
from base64 import b64encode
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import cast

import discord
import httpx

from llmcord.core.config import (
    VISION_MODEL_TAGS,
    is_gemini_model,
)
from llmcord.core.error_handling import log_exception
from llmcord.core.models import MsgNode
from llmcord.logic.content import (
    ExternalContentContext,
    GoogleLensContext,
    apply_googlelens,
    collect_external_content,
    download_and_process_attachments,
    extract_pdf_images_for_model,
    get_allowed_attachment_types,
    is_googlelens_query,
)
from llmcord.logic.media_preprocessing import (
    preprocess_media_attachments_with_gemini,
)
from llmcord.logic.utils import (
    append_search_to_content,
    build_node_text_parts,
)
from llmcord.services.database import get_db
from llmcord.services.extractors import TwitterApiProtocol
from llmcord.services.facebook import maybe_download_facebook_videos_with_failures
from llmcord.services.tiktok import maybe_download_tiktok_videos_with_failures

logger = logging.getLogger(__name__)
_SOCIAL_VIDEO_URL_TOKENS = (
    "tiktok.com",
    "vt.tiktok.com",
    "facebook.com",
    "fb.watch",
)


@dataclass(slots=True)
class MessageBuildContext:
    """Inputs for building conversation messages."""

    new_msg: discord.Message
    discord_bot: discord.Client
    httpx_client: httpx.AsyncClient
    twitter_api: TwitterApiProtocol
    msg_nodes: dict[int, MsgNode]
    actual_model: str
    accept_usernames: bool
    max_text: int
    max_images: int
    max_messages: int
    max_tweet_replies: int
    enable_youtube_transcripts: bool
    youtube_transcript_method: str
    provider_slash_model: str | None = None
    status_callback: Callable[[str], Awaitable[None]] | None = None


@dataclass(slots=True)
class MessageBuildResult:
    """Result for conversation message building."""

    messages: list[dict[str, object]]
    user_warnings: set[str]
    failed_extractions: list[str]


async def build_messages(
    *,
    context: MessageBuildContext,
) -> MessageBuildResult:
    """Build the list of messages for the LLM."""
    messages: list[dict[str, object]] = []
    user_warnings: set[str] = set()
    failed_extractions: list[str] = []
    curr_msg: discord.Message | None = context.new_msg

    while curr_msg is not None and len(messages) < context.max_messages:
        curr_node = context.msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            await _populate_node_if_needed(
                curr_msg=curr_msg,
                curr_node=curr_node,
                context=context,
            )
            await _load_cached_search_data(curr_msg, curr_node)

            content = _build_content_payload(
                curr_node=curr_node,
                actual_model=context.actual_model,
                max_text=context.max_text,
                max_images=context.max_images,
            )
            if curr_node.search_results and curr_node.role == "user":
                content = append_search_to_content(
                    cast("str | list[dict[str, object]]", content),
                    curr_node.search_results,
                )

            if content:
                message: dict[str, object] = {
                    "content": content,
                    "role": curr_node.role,
                }
                if context.accept_usernames and curr_node.user_id is not None:
                    message["name"] = str(curr_node.user_id)
                messages.append(message)

            _update_user_warnings(
                curr_node=curr_node,
                user_warnings=user_warnings,
                context=context,
                messages_len=len(messages),
            )
            failed_extractions.extend(curr_node.failed_extractions)

            curr_msg = curr_node.parent_msg

    return MessageBuildResult(
        messages=messages,
        user_warnings=user_warnings,
        failed_extractions=list(dict.fromkeys(failed_extractions)),
    )


async def _populate_node_if_needed(
    *,
    curr_msg: discord.Message,
    curr_node: MsgNode,
    context: MessageBuildContext,
) -> None:
    if curr_node.text is not None:
        return

    cleaned_content = _clean_message_content(curr_msg, context.discord_bot)
    cleaned_content = await apply_googlelens(
        GoogleLensContext(
            cleaned_content=cleaned_content,
            curr_msg=curr_msg,
            curr_node=curr_node,
            httpx_client=context.httpx_client,
            twitter_api=context.twitter_api,
            max_tweet_replies=context.max_tweet_replies,
        ),
    )
    omit_image_inputs = _should_omit_googlelens_image_inputs(
        curr_msg=curr_msg,
        curr_node=curr_node,
        context=context,
    )
    (
        cached_media_preprocessing_parts,
        cached_media_preprocessing_failed,
    ) = await _load_cached_media_preprocessing_data_if_needed(
        cleaned_content=cleaned_content,
        curr_msg=curr_msg,
        context=context,
    )
    has_cached_media_preprocessing = (
        cached_media_preprocessing_parts is not None
        or cached_media_preprocessing_failed is not None
    )

    allowed_types = get_allowed_attachment_types(
        context.actual_model,
        include_audio_video_for_non_gemini=not has_cached_media_preprocessing,
    )
    good_attachments = [
        att
        for att in curr_msg.attachments
        if att.content_type
        and any(att.content_type.startswith(x) for x in allowed_types)
    ]

    if good_attachments and context.status_callback is not None:
        await context.status_callback(
            f"Downloading {len(good_attachments)} attachment(s)...",
        )

    (
        good_attachments,
        attachment_responses,
        processed_attachments,
    ) = await download_and_process_attachments(
        attachments=good_attachments,
        httpx_client=context.httpx_client,
    )

    if not has_cached_media_preprocessing:
        await _add_social_video_attachments(
            cleaned_content=cleaned_content,
            context=context,
            curr_node=curr_node,
            processed_attachments=processed_attachments,
        )

    if has_cached_media_preprocessing:
        media_preprocessing_parts = cached_media_preprocessing_parts or []
        curr_node.has_failed_media_preprocessing = bool(
            cached_media_preprocessing_failed,
        )
    else:
        (
            media_preprocessing_parts,
            curr_node.has_failed_media_preprocessing,
        ) = await preprocess_media_attachments_with_gemini(
            actual_model=context.actual_model,
            processed_attachments=processed_attachments,
            status_callback=context.status_callback,
        )
        await _save_cached_media_preprocessing_data_if_needed(
            curr_msg=curr_msg,
            media_preprocessing_parts=media_preprocessing_parts,
            media_preprocessing_failed=curr_node.has_failed_media_preprocessing,
        )

    curr_node.text = _build_initial_text(
        cleaned_content=cleaned_content,
        curr_msg=curr_msg,
        processed_attachments=processed_attachments,
    )

    curr_node.images, omitted_image_count = _extract_image_parts(
        processed_attachments=processed_attachments,
        omit_image_inputs=omit_image_inputs,
    )
    if omitted_image_count:
        logger.info(
            "Skipped %s image attachment(s) because Google Lens results were added",
            omitted_image_count,
        )

    curr_node.raw_attachments = [
        {"content_type": att["content_type"], "content": att["content"]}
        for att in processed_attachments
    ]

    extra_parts = await collect_external_content(
        ExternalContentContext(
            cleaned_content=cleaned_content,
            httpx_client=context.httpx_client,
            twitter_api=context.twitter_api,
            max_tweet_replies=context.max_tweet_replies,
            processed_attachments=processed_attachments,
            actual_model=context.actual_model,
            enable_youtube_transcripts=context.enable_youtube_transcripts,
            youtube_transcript_method=context.youtube_transcript_method,
            curr_node=curr_node,
        ),
    )

    curr_node.text = _build_final_text(
        cleaned_content=cleaned_content,
        curr_msg=curr_msg,
        good_attachments=good_attachments,
        attachment_responses=attachment_responses,
        extra_parts=[*media_preprocessing_parts, *extra_parts],
    )

    pdf_images = await extract_pdf_images_for_model(
        processed_attachments=processed_attachments,
        actual_model=context.actual_model,
    )
    if pdf_images:
        curr_node.images.extend(pdf_images)

    if not curr_node.text and curr_node.images:
        curr_node.text = "What is in this image?"

    if (
        not curr_node.text
        and not curr_node.images
        and not curr_node.raw_attachments
        and curr_msg == context.new_msg
    ):
        curr_node.text = "Hello"

    curr_node.role = (
        "assistant" if curr_msg.author == context.discord_bot.user else "user"
    )

    curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None
    curr_node.has_bad_attachments = len(curr_msg.attachments) > len(
        good_attachments,
    ) + _count_skipped_cached_media_attachments(
        curr_msg=curr_msg,
        context=context,
        has_cached_media_preprocessing=has_cached_media_preprocessing,
    )

    await _set_parent_message(
        curr_msg=curr_msg,
        curr_node=curr_node,
        discord_bot=context.discord_bot,
    )


def _should_omit_googlelens_image_inputs(
    *,
    curr_msg: discord.Message,
    curr_node: MsgNode,
    context: MessageBuildContext,
) -> bool:
    if not curr_node.lens_results:
        return False
    if not is_googlelens_query(curr_msg, context.discord_bot):
        return False
    return not _supports_provider_model_image_input(context=context)


async def _add_social_video_attachments(
    *,
    cleaned_content: str,
    context: MessageBuildContext,
    curr_node: MsgNode,
    processed_attachments: list[dict[str, bytes | str | None]],
) -> None:
    cleaned_lower = cleaned_content.lower()

    if context.status_callback is not None and any(
        token in cleaned_lower
        for token in (
            "tiktok.com",
            "vt.tiktok.com",
        )
    ):
        await context.status_callback("Downloading linked TikTok video(s)...")

    tiktok_result = await maybe_download_tiktok_videos_with_failures(
        cleaned_content=cleaned_content,
        actual_model=context.actual_model,
        httpx_client=context.httpx_client,
        force_download=True,
    )
    if tiktok_result.failed_urls:
        curr_node.failed_extractions.extend(tiktok_result.failed_urls)
    if tiktok_result.videos:
        processed_attachments.extend(
            {
                "content_type": tiktok_video.content_type,
                "content": tiktok_video.content,
                "text": None,
            }
            for tiktok_video in tiktok_result.videos
        )
        logger.info(
            "Added %s downloaded TikTok video attachment(s) for Gemini processing",
            len(tiktok_result.videos),
        )

    if context.status_callback is not None and any(
        token in cleaned_lower
        for token in (
            "facebook.com",
            "fb.watch",
        )
    ):
        await context.status_callback("Downloading linked Facebook video(s)...")

    facebook_result = await maybe_download_facebook_videos_with_failures(
        cleaned_content=cleaned_content,
        actual_model=context.actual_model,
        httpx_client=context.httpx_client,
        force_download=True,
    )
    if facebook_result.failed_urls:
        curr_node.failed_extractions.extend(facebook_result.failed_urls)
    if facebook_result.videos:
        processed_attachments.extend(
            {
                "content_type": facebook_video.content_type,
                "content": facebook_video.content,
                "text": None,
            }
            for facebook_video in facebook_result.videos
        )
        logger.info(
            "Added %s downloaded Facebook video attachment(s) for Gemini processing",
            len(facebook_result.videos),
        )


def _message_has_direct_media_attachments(curr_msg: discord.Message) -> bool:
    return any(
        isinstance(att.content_type, str)
        and att.content_type.startswith(("audio/", "video/"))
        for att in curr_msg.attachments
    )


def _cleaned_content_has_social_video_urls(cleaned_content: str) -> bool:
    cleaned_lower = cleaned_content.lower()
    return any(token in cleaned_lower for token in _SOCIAL_VIDEO_URL_TOKENS)


def _message_may_use_media_preprocessing(
    *,
    curr_msg: discord.Message,
    cleaned_content: str,
    context: MessageBuildContext,
) -> bool:
    return not is_gemini_model(context.actual_model) and (
        _message_has_direct_media_attachments(curr_msg)
        or _cleaned_content_has_social_video_urls(cleaned_content)
    )


async def _load_cached_media_preprocessing_data_if_needed(
    *,
    cleaned_content: str,
    curr_msg: discord.Message,
    context: MessageBuildContext,
) -> tuple[list[str] | None, bool | None]:
    if not _message_may_use_media_preprocessing(
        curr_msg=curr_msg,
        cleaned_content=cleaned_content,
        context=context,
    ):
        return None, None

    db = get_db()
    async_get_data = getattr(db, "aget_message_media_preprocessing_data", None)
    if callable(async_get_data):
        return await async_get_data(str(curr_msg.id))

    sync_get_data = getattr(db, "get_message_media_preprocessing_data", None)
    if callable(sync_get_data):
        return await asyncio.to_thread(sync_get_data, str(curr_msg.id))

    return None, None


async def _save_cached_media_preprocessing_data_if_needed(
    *,
    curr_msg: discord.Message,
    media_preprocessing_parts: list[str],
    media_preprocessing_failed: bool,
) -> None:
    if not media_preprocessing_parts and not media_preprocessing_failed:
        return

    db = get_db()
    async_save_data = getattr(db, "asave_message_media_preprocessing_data", None)
    if callable(async_save_data):
        await async_save_data(
            str(curr_msg.id),
            media_preprocessing_results=media_preprocessing_parts,
            media_preprocessing_failed=media_preprocessing_failed,
        )
        return

    sync_save_data = getattr(db, "save_message_media_preprocessing_data", None)
    if callable(sync_save_data):
        await asyncio.to_thread(
            sync_save_data,
            str(curr_msg.id),
            media_preprocessing_results=media_preprocessing_parts,
            media_preprocessing_failed=media_preprocessing_failed,
        )


def _count_skipped_cached_media_attachments(
    *,
    curr_msg: discord.Message,
    context: MessageBuildContext,
    has_cached_media_preprocessing: bool,
) -> int:
    if not has_cached_media_preprocessing or is_gemini_model(context.actual_model):
        return 0

    return sum(
        1
        for att in curr_msg.attachments
        if isinstance(att.content_type, str)
        and att.content_type.startswith(("audio/", "video/"))
    )


def _supports_provider_model_image_input(*, context: MessageBuildContext) -> bool:
    if context.max_images <= 0:
        return False

    normalized_targets = {
        context.actual_model.lower(),
    }
    if isinstance(context.provider_slash_model, str):
        normalized_targets.add(context.provider_slash_model.lower())

    return any(
        tag in target for target in normalized_targets for tag in VISION_MODEL_TAGS
    )


def _extract_image_parts(
    *,
    processed_attachments: list[dict[str, bytes | str | None]],
    omit_image_inputs: bool,
) -> tuple[list[dict[str, object]], int]:
    image_parts: list[dict[str, object]] = []
    omitted_image_count = 0

    for attachment in processed_attachments:
        content_type = attachment["content_type"]
        if not content_type or not str(content_type).startswith("image"):
            continue
        if omit_image_inputs:
            omitted_image_count += 1
            continue

        content = (
            attachment["content"] if isinstance(attachment["content"], bytes) else b""
        )
        encoded = b64encode(content).decode("utf-8")
        image_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{content_type!s};base64,{encoded}"},
            },
        )

    return image_parts, omitted_image_count


def _build_content_payload(
    *,
    curr_node: MsgNode,
    actual_model: str,
    max_text: int,
    max_images: int,
) -> object:
    gemini_file_attachments: list[dict[str, object]] = []
    if is_gemini_model(actual_model) and curr_node.raw_attachments:
        for att in curr_node.raw_attachments:
            if str(att["content_type"]).startswith(("audio", "video")) or (
                att["content_type"] == "application/pdf"
            ):
                content_bytes = att["content"]
                if not isinstance(content_bytes, bytes):
                    continue
                encoded_data = b64encode(content_bytes).decode("utf-8")
                file_data = f"data:{att['content_type']};base64,{encoded_data}"
                gemini_file_attachments.append(
                    {"type": "file", "file": {"file_data": file_data}},
                )

    has_images = bool(curr_node.images[:max_images])
    has_gemini_files = bool(gemini_file_attachments)

    if has_images or has_gemini_files:
        content: list[dict[str, object]] = []
        if (curr_node.text or "")[:max_text]:
            content.append({"type": "text", "text": (curr_node.text or "")[:max_text]})
        content.extend(curr_node.images[:max_images])
        content.extend(gemini_file_attachments)
        if gemini_file_attachments:
            logger.info(
                ("Added %s Gemini file attachment(s) (audio/video/PDF) to message"),
                len(gemini_file_attachments),
            )
        if not content:
            return [{"type": "text", "text": "What is in this file?"}]
        return content

    return (curr_node.text or "")[:max_text]


def _build_initial_text(
    *,
    cleaned_content: str,
    curr_msg: discord.Message,
    processed_attachments: list[dict[str, bytes | str | None]],
) -> str:
    return build_node_text_parts(
        cleaned_content,
        curr_msg.embeds,
        text_attachments=[
            str(att["text"])
            for att in processed_attachments
            if att["content_type"]
            and str(att["content_type"]).startswith("text")
            and att["text"]
        ],
    )


def _build_final_text(
    *,
    cleaned_content: str,
    curr_msg: discord.Message,
    good_attachments: list[discord.Attachment],
    attachment_responses: list[httpx.Response],
    extra_parts: list[str],
) -> str:
    return build_node_text_parts(
        cleaned_content,
        curr_msg.embeds,
        text_attachments=[
            str(resp.text)
            for att, resp in zip(
                good_attachments,
                attachment_responses,
                strict=False,
            )
            if att.content_type and att.content_type.startswith("text") and resp.text
        ],
        extra_parts=extra_parts,
    )


async def _set_parent_message(
    *,
    curr_msg: discord.Message,
    curr_node: MsgNode,
    discord_bot: discord.Client,
) -> None:
    try:
        if not discord_bot.user:
            return
        prev_msg_in_channel = None
        if (
            curr_msg.reference is None
            and discord_bot.user.mention not in curr_msg.content
            and "at ai" not in curr_msg.content.lower()
        ):
            history = [
                message
                async for message in curr_msg.channel.history(
                    before=curr_msg,
                    limit=1,
                )
            ]
            prev_msg_in_channel = history[0] if history else None

        is_dm = curr_msg.channel.type == discord.ChannelType.private
        is_thread = isinstance(curr_msg.channel, discord.Thread)

        if (
            prev_msg_in_channel
            and prev_msg_in_channel.type
            in (
                discord.MessageType.default,
                discord.MessageType.reply,
            )
            and prev_msg_in_channel.author
            == (discord_bot.user if (is_dm or is_thread) else curr_msg.author)
        ):
            curr_node.parent_msg = prev_msg_in_channel
            return

        parent_is_thread_start = False
        if is_thread and curr_msg.reference is None:
            thread_channel = cast("discord.Thread", curr_msg.channel)
            if thread_channel.parent is not None:
                parent_is_thread_start = True

        parent_msg_id = (
            curr_msg.channel.id
            if parent_is_thread_start
            else getattr(curr_msg.reference, "message_id", None)
        )
        if not parent_msg_id:
            return

        if parent_is_thread_start and is_thread:
            thread_channel = cast("discord.Thread", curr_msg.channel)
            curr_node.parent_msg = (
                thread_channel.starter_message
                or await thread_channel.parent.fetch_message(parent_msg_id)  # type: ignore[union-attr]
            )
        elif curr_msg.reference:
            curr_node.parent_msg = (
                curr_msg.reference.cached_message
                or await curr_msg.channel.fetch_message(parent_msg_id)
            )
    except (discord.NotFound, discord.HTTPException) as exc:
        log_exception(
            logger=logger,
            message="Error fetching next message in the chain",
            error=exc,
            context={"message_id": curr_msg.id},
        )
        curr_node.fetch_parent_failed = True


def _clean_message_content(
    curr_msg: discord.Message,
    discord_bot: discord.Client,
) -> str:
    if not discord_bot.user:
        return curr_msg.content

    # Discord can emit mentions as either <@ID> or <@!ID>. Strip either if the
    # message starts with a bot mention.
    bot_id = getattr(discord_bot.user, "id", None)
    if bot_id is None:
        return curr_msg.content

    cleaned_content = re.sub(
        rf"^\s*<@!?{bot_id}>\s*",
        "",
        curr_msg.content,
    )

    cleaned_content = re.sub(
        r"\bat ai\b",
        "",
        cleaned_content,
        flags=re.IGNORECASE,
    ).lstrip()

    # If the message was *only* a trigger ("at ai" and/or a bot mention) with no
    # other content (in any order), use '.' as a minimal continuation prompt.
    if not curr_msg.embeds:
        trigger_present = bool(
            re.search(
                rf"<@!?{bot_id}>|\bat ai\b",
                curr_msg.content,
                flags=re.IGNORECASE,
            ),
        )
        content_without_triggers = re.sub(
            rf"<@!?{bot_id}>",
            "",
            curr_msg.content,
            flags=re.IGNORECASE,
        )
        content_without_triggers = re.sub(
            r"\bat ai\b",
            "",
            content_without_triggers,
            flags=re.IGNORECASE,
        )
        if trigger_present and not content_without_triggers.strip():
            return "."

    return cleaned_content


async def _load_cached_search_data(
    curr_msg: discord.Message,
    curr_node: MsgNode,
) -> None:
    if curr_node.search_results is not None and curr_node.lens_results is not None:
        return

    db = get_db()
    async_get_search_data = getattr(db, "aget_message_search_data", None)
    if callable(async_get_search_data):
        (
            stored_search_results,
            stored_tavily_metadata,
            stored_lens_results,
        ) = await async_get_search_data(str(curr_msg.id))
    else:
        sync_get_search_data = getattr(db, "get_message_search_data", None)
        (
            stored_search_results,
            stored_tavily_metadata,
            stored_lens_results,
        ) = (
            await asyncio.to_thread(sync_get_search_data, str(curr_msg.id))
            if callable(sync_get_search_data)
            else (None, None, None)
        )
    if stored_search_results and curr_node.search_results is None:
        curr_node.search_results = stored_search_results
        curr_node.tavily_metadata = stored_tavily_metadata
        logger.info("Loaded stored search data for message %s", curr_msg.id)
    if stored_lens_results and curr_node.lens_results is None:
        curr_node.lens_results = stored_lens_results
        logger.info("Loaded stored lens results for message %s", curr_msg.id)


def _update_user_warnings(
    *,
    curr_node: MsgNode,
    user_warnings: set[str],
    context: MessageBuildContext,
    messages_len: int,
) -> None:
    if curr_node.text and len(curr_node.text) > context.max_text:
        user_warnings.add(
            f"⚠️ Max {context.max_text:,} characters per message",
        )
    if len(curr_node.images) > context.max_images:
        user_warnings.add(
            (
                f"⚠️ Max {context.max_images} image"
                f"{'' if context.max_images == 1 else 's'} per message"
            )
            if context.max_images > 0
            else "⚠️ Can't see images",
        )
    if curr_node.has_bad_attachments:
        user_warnings.add("⚠️ Unsupported attachments")
    if curr_node.has_failed_media_preprocessing:
        user_warnings.add("⚠️ Some audio/video attachments could not be analyzed")
    if curr_node.fetch_parent_failed or (
        curr_node.parent_msg is not None and messages_len == context.max_messages
    ):
        user_warnings.add(
            f"⚠️ Only using last {messages_len} message"
            f"{'' if messages_len == 1 else 's'}",
        )

    if curr_node.failed_extractions:
        user_warnings.add(
            '⚠️ failed to extract from some urls. click "failed urls" to see '
            "which urls.",
        )
