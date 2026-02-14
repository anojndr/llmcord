"""Message building and processing logic."""

import logging
import re
from base64 import b64encode
from dataclasses import dataclass
from typing import cast

import discord
import httpx

from llmcord.core.config import EMBED_FIELD_NAME_LIMIT, is_gemini_model
from llmcord.core.models import MsgNode
from llmcord.logic.content import (
    ExternalContentContext,
    GoogleLensContext,
    apply_googlelens,
    collect_external_content,
    download_and_process_attachments,
    extract_pdf_images_for_model,
    get_allowed_attachment_types,
)
from llmcord.logic.utils import (
    TextDisplayComponentProtocol,
    append_search_to_content,
    build_node_text_parts,
)
from llmcord.services.database import get_bad_keys_db
from llmcord.services.extractors import TwitterApiProtocol
from llmcord.services.tiktok import maybe_download_tiktok_video

logger = logging.getLogger(__name__)


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


@dataclass(slots=True)
class MessageBuildResult:
    """Result for conversation message building."""

    messages: list[dict[str, object]]
    user_warnings: set[str]


async def build_messages(
    *,
    context: MessageBuildContext,
) -> MessageBuildResult:
    """Build the list of messages for the LLM."""
    messages: list[dict[str, object]] = []
    user_warnings: set[str] = set()
    curr_msg: discord.Message | None = context.new_msg

    while curr_msg is not None and len(messages) < context.max_messages:
        curr_node = context.msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            await _populate_node_if_needed(
                curr_msg=curr_msg,
                curr_node=curr_node,
                context=context,
            )
            _load_cached_search_data(curr_msg, curr_node)

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

            curr_msg = curr_node.parent_msg

    return MessageBuildResult(messages=messages, user_warnings=user_warnings)


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

    allowed_types = get_allowed_attachment_types(context.actual_model)
    good_attachments = [
        att
        for att in curr_msg.attachments
        if att.content_type
        and any(att.content_type.startswith(x) for x in allowed_types)
    ]

    (
        good_attachments,
        attachment_responses,
        processed_attachments,
    ) = await download_and_process_attachments(
        attachments=good_attachments,
        httpx_client=context.httpx_client,
    )

    if is_gemini_model(context.actual_model):
        tiktok_video = await maybe_download_tiktok_video(
            cleaned_content=cleaned_content,
            actual_model=context.actual_model,
            httpx_client=context.httpx_client,
        )
        if tiktok_video is not None:
            processed_attachments.append(
                {
                    "content_type": tiktok_video.content_type,
                    "content": tiktok_video.content,
                    "text": None,
                },
            )
            logger.info(
                "Added downloaded TikTok video attachment for Gemini processing",
            )

    curr_node.text = _build_initial_text(
        cleaned_content=cleaned_content,
        curr_msg=curr_msg,
        processed_attachments=processed_attachments,
    )

    curr_node.images = []
    for att in processed_attachments:
        if att["content_type"] and str(att["content_type"]).startswith("image"):
            content = att["content"] if isinstance(att["content"], bytes) else b""
            encoded = b64encode(content).decode("utf-8")
            curr_node.images.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{att['content_type']!s};base64,{encoded}",
                    },
                },
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
        extra_parts=extra_parts,
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
    )

    await _set_parent_message(
        curr_msg=curr_msg,
        curr_node=curr_node,
        discord_bot=context.discord_bot,
    )


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
        cast("list[TextDisplayComponentProtocol]", curr_msg.components),
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
        cast("list[TextDisplayComponentProtocol]", curr_msg.components),
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
    except (discord.NotFound, discord.HTTPException):
        logger.exception("Error fetching next message in the chain")
        curr_node.fetch_parent_failed = True


def _clean_message_content(
    curr_msg: discord.Message,
    discord_bot: discord.Client,
) -> str:
    if not discord_bot.user:
        return curr_msg.content
    cleaned_content = curr_msg.content.removeprefix(
        discord_bot.user.mention,
    ).lstrip()
    return re.sub(
        r"\bat ai\b",
        "",
        cleaned_content,
        flags=re.IGNORECASE,
    ).lstrip()


def _load_cached_search_data(
    curr_msg: discord.Message,
    curr_node: MsgNode,
) -> None:
    if curr_node.search_results is not None and curr_node.lens_results is not None:
        return

    (
        stored_search_results,
        stored_tavily_metadata,
        stored_lens_results,
    ) = get_bad_keys_db().get_message_search_data(str(curr_msg.id))
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
    if curr_node.fetch_parent_failed or (
        curr_node.parent_msg is not None and messages_len == context.max_messages
    ):
        user_warnings.add(
            f"⚠️ Only using last {messages_len} message"
            f"{'' if messages_len == 1 else 's'}",
        )

    if curr_node.failed_extractions:
        failed_list = "\n- " + "\n- ".join(curr_node.failed_extractions)
        warning = f"⚠️ Failed to extract from: {failed_list}"
        if len(warning) > EMBED_FIELD_NAME_LIMIT:
            warning = warning[: EMBED_FIELD_NAME_LIMIT - 3] + "..."
        user_warnings.add(warning)
