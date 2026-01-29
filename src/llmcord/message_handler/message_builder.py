"""Build message history and attachments for LLM requests."""
# ruff: noqa: E501

from __future__ import annotations

import logging
import re
from base64 import b64encode
from typing import TYPE_CHECKING

import discord

from llmcord.bad_keys import get_bad_keys_db
from llmcord.config import is_gemini_model
from llmcord.models import MsgNode

from .attachment_utils import (
    build_node_text_parts,
    download_attachments,
    normalize_attachments,
)
from .content_fetchers import (
    LensRequest,
    apply_google_lens_results,
    extract_pdf_texts,
    fetch_reddit_posts,
    fetch_tweets,
    fetch_youtube_transcripts,
)
from .shared import append_search_to_content, is_gemini_file_type, strip_bot_mention

if TYPE_CHECKING:
    import asyncpraw
    import httpx

    from .shared import TwitterApiProtocol

logger = logging.getLogger(__name__)


async def build_messages_from_chain(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    new_msg: discord.Message,
    discord_bot: discord.Client,
    httpx_client: httpx.AsyncClient,
    twitter_api: TwitterApiProtocol,
    reddit_client: asyncpraw.Reddit | None,
    msg_nodes: dict[int, MsgNode],
    actual_model: str,
    max_text: int,
    max_images: int,
    max_messages: int,
    max_tweet_replies: int,
    accept_usernames: bool,
) -> tuple[list[dict[str, object]], set[str]]:
    """Build messages and warnings from the conversation chain."""
    messages: list[dict[str, object]] = []
    user_warnings: set[str] = set()
    curr_msg = new_msg

    while curr_msg is not None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text is None:
                cleaned_content = strip_bot_mention(
                    curr_msg.content,
                    discord_bot.user.mention,
                ).lstrip()
                cleaned_content = re.sub(r"\bat ai\b", "", cleaned_content, flags=re.IGNORECASE).lstrip()

                if cleaned_content.lower().startswith("googlelens"):
                    cleaned_content = cleaned_content[10:].strip()
                    cleaned_content = await apply_google_lens_results(
                        LensRequest(
                            cleaned_content=cleaned_content,
                            curr_msg=curr_msg,
                            curr_node=curr_node,
                            httpx_client=httpx_client,
                            twitter_api=twitter_api,
                            max_tweet_replies=max_tweet_replies,
                        ),
                    )

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

                good_attachments, attachment_responses = await download_attachments(
                    httpx_client,
                    good_attachments,
                )

                processed_attachments = normalize_attachments(
                    good_attachments,
                    attachment_responses,
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

                yt_transcripts = await fetch_youtube_transcripts(
                    cleaned_content,
                    httpx_client,
                )

                tweets = await fetch_tweets(
                    cleaned_content,
                    twitter_api,
                    max_tweet_replies,
                )

                reddit_posts = await fetch_reddit_posts(
                    cleaned_content,
                    reddit_client,
                )

                pdf_texts = await extract_pdf_texts(
                    processed_attachments,
                    actual_model,
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
                    if is_gemini_file_type(att["content_type"]):
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

    return messages, user_warnings
