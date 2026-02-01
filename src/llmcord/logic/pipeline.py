"""Message processing orchestration."""
import asyncio
import logging
import re
from base64 import b64encode
from dataclasses import dataclass

import discord
import httpx

from llmcord.core.config import (
    EMBED_COLOR_INCOMPLETE,
    PROVIDERS_SUPPORTING_USERNAMES,
    VISION_MODEL_TAGS,
    ensure_list,
    get_config,
    is_gemini_model,
)
from llmcord.core.models import MsgNode
from llmcord.globals import reddit_client
from llmcord.logic.generation import GenerationContext, generate_response
from llmcord.logic.permissions import should_process_message
from llmcord.logic.utils import (
    append_search_to_content,
    build_node_text_parts,
    extract_research_command,
    replace_content_text,
)
from llmcord.services.database import get_bad_keys_db
from llmcord.services.extractors import (
    TwitterApiProtocol,
    download_attachments,
    extract_pdf_text,
    extract_reddit_post,
    extract_youtube_transcript,
    fetch_tweet_with_replies,
    perform_yandex_lookup,
    process_attachments,
)
from llmcord.services.search import (
    WebSearchOptions,
    decide_web_search,
    get_current_datetime_strings,
    perform_tavily_research,
    perform_web_search,
)
from llmcord.services.search.config import EXA_MCP_URL

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProcessContext:
    """Dependencies and options for message processing."""

    discord_bot: discord.Client
    httpx_client: httpx.AsyncClient
    twitter_api: TwitterApiProtocol
    msg_nodes: dict[int, MsgNode]
    curr_model_lock: asyncio.Lock
    curr_model_ref: list[str]
    override_provider_slash_model: str | None = None
    fallback_chain: list[tuple[str, str, str]] | None = None


@dataclass(slots=True)
class MessageBuildResult:
    """Result for conversation message building."""

    messages: list[dict[str, object]]
    user_warnings: set[str]


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


@dataclass(slots=True)
class ResearchCommandContext:
    """Inputs for handling research commands."""

    new_msg: discord.Message
    messages: list[dict[str, object]]
    msg_nodes: dict[int, MsgNode]
    research_model: str | None
    research_query: str | None
    tavily_api_keys: list[str]
    user_warnings: set[str]
    has_existing_search: bool


@dataclass(slots=True)
class WebSearchContext:
    """Inputs for web search execution."""

    new_msg: discord.Message
    messages: list[dict[str, object]]
    msg_nodes: dict[int, MsgNode]
    config: dict[str, object]
    web_search_provider: str
    tavily_api_keys: list[str]
    exa_mcp_url: str
    is_googlelens_query: bool
    actual_model: str
    has_existing_search: bool
    search_metadata: dict[str, object] | None


@dataclass(slots=True)
class ProviderSettings:
    """Resolved provider/model settings."""

    provider: str
    model: str
    provider_slash_model: str
    base_url: str | None
    api_keys: list[str]
    model_parameters: dict[str, object] | None
    extra_headers: dict[str, str] | None
    extra_query: dict[str, object] | None
    extra_body: dict[str, object] | None
    actual_model: str


@dataclass(slots=True)
class SearchResolutionContext:
    """Inputs for resolving search metadata."""

    new_msg: discord.Message
    discord_bot: discord.Client
    msg_nodes: dict[int, MsgNode]
    messages: list[dict[str, object]]
    user_warnings: set[str]
    tavily_api_keys: list[str]
    config: dict[str, object]
    web_search_available: bool
    web_search_provider: str
    exa_mcp_url: str
    actual_model: str


async def _resolve_provider_settings(
    *,
    processing_msg: discord.Message,
    curr_model_lock: asyncio.Lock,
    curr_model_ref: list[str],
    override_provider_slash_model: str | None,
    config: dict[str, object],
) -> ProviderSettings | None:
    provider_slash_model = await _get_provider_slash_model(
        override_provider_slash_model=override_provider_slash_model,
        curr_model_lock=curr_model_lock,
        curr_model_ref=curr_model_ref,
    )

    parsed_provider = _parse_provider_slash_model(provider_slash_model)
    if parsed_provider is None:
        logger.exception(
            "Invalid model format: %s. Expected 'provider/model'.",
            provider_slash_model,
        )
        await _send_processing_error(
            processing_msg,
            (
                "❌ Invalid model configuration: "
                f"'{provider_slash_model}'. Expected format: 'provider/model'.\n"
                "Please contact an administrator."
            ),
        )
        return None

    provider, model = parsed_provider
    providers = config.get("providers", {})
    provider_config = providers.get(provider)
    if provider_config is None:
        logger.error("Provider '%s' not found in config.", provider)
        await _send_processing_error(
            processing_msg,
            (
                f"❌ Provider '{provider}' is not configured. "
                "Please contact an administrator."
            ),
        )
        return None

    base_url = provider_config.get("base_url")
    api_keys = ensure_list(provider_config.get("api_key")) or ["sk-no-key-required"]
    model_parameters = config["models"].get(provider_slash_model, None)
    override_model = None
    if model_parameters:
        override_model = (
            model_parameters.get("model")
            or model_parameters.get("model_name")
            or model_parameters.get("modelName")
        )

    if isinstance(override_model, str) and "/" in override_model:
        _, override_model = override_model.split("/", 1)

    actual_model = override_model or model
    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {})
    extra_body = extra_body or None

    return ProviderSettings(
        provider=provider,
        model=model,
        provider_slash_model=provider_slash_model,
        base_url=base_url,
        api_keys=api_keys,
        model_parameters=model_parameters,
        extra_headers=extra_headers,
        extra_query=extra_query,
        extra_body=extra_body,
        actual_model=actual_model,
    )


def _resolve_web_search_provider(
    config: dict[str, object],
    tavily_api_keys: list[str],
    exa_mcp_url: str,
) -> tuple[str, bool]:
    web_search_provider = config.get("web_search_provider", "tavily")
    web_search_available = False
    if (web_search_provider == "tavily" and tavily_api_keys) or (
        web_search_provider == "exa" and exa_mcp_url
    ):
        web_search_available = True
    elif web_search_provider == "auto":
        if tavily_api_keys:
            web_search_provider = "tavily"
            web_search_available = True
        elif exa_mcp_url:
            web_search_provider = "exa"
            web_search_available = True

    return web_search_provider, web_search_available


async def _resolve_search_metadata(
    context: SearchResolutionContext,
) -> dict[str, object] | None:
    is_googlelens_query = _is_googlelens_query(
        context.new_msg,
        context.discord_bot,
    )
    research_model, research_query = extract_research_command(
        context.new_msg.content,
        context.discord_bot.user.mention if context.discord_bot.user else "",
    )

    if (
        context.new_msg.id in context.msg_nodes
        and context.msg_nodes[context.new_msg.id].search_results
    ):
        search_metadata = context.msg_nodes[context.new_msg.id].tavily_metadata
        has_existing_search = True
    else:
        search_metadata = None
        has_existing_search = False

    search_metadata = (
        await _run_research_command(
            ResearchCommandContext(
                new_msg=context.new_msg,
                messages=context.messages,
                msg_nodes=context.msg_nodes,
                research_model=research_model,
                research_query=research_query,
                tavily_api_keys=context.tavily_api_keys,
                user_warnings=context.user_warnings,
                has_existing_search=has_existing_search,
            ),
        )
        or search_metadata
    )

    if not research_model and context.web_search_available:
        search_metadata = await _maybe_run_web_search(
            WebSearchContext(
                new_msg=context.new_msg,
                messages=context.messages,
                msg_nodes=context.msg_nodes,
                config=context.config,
                web_search_provider=context.web_search_provider,
                tavily_api_keys=context.tavily_api_keys,
                exa_mcp_url=context.exa_mcp_url,
                is_googlelens_query=is_googlelens_query,
                actual_model=context.actual_model,
                has_existing_search=has_existing_search,
                search_metadata=search_metadata,
            ),
        )

    return search_metadata


async def _send_processing_error(
    processing_msg: discord.Message,
    description: str,
) -> None:
    embed = discord.Embed(
        description=description,
        color=EMBED_COLOR_INCOMPLETE,
    )
    await processing_msg.edit(embed=embed)


async def _get_provider_slash_model(
    *,
    override_provider_slash_model: str | None,
    curr_model_lock: asyncio.Lock,
    curr_model_ref: list[str],
) -> str:
    if override_provider_slash_model:
        return override_provider_slash_model

    async with curr_model_lock:
        return curr_model_ref[0]


def _parse_provider_slash_model(
    provider_slash_model: str,
) -> tuple[str, str] | None:
    try:
        return provider_slash_model.removesuffix(":vision").split("/", 1)
    except ValueError:
        return None


def _clean_message_content(
    curr_msg: discord.Message,
    discord_bot: discord.Client,
) -> str:
    cleaned_content = curr_msg.content.removeprefix(
        discord_bot.user.mention,
    ).lstrip()
    return re.sub(
        r"\bat ai\b",
        "",
        cleaned_content,
        flags=re.IGNORECASE,
    ).lstrip()


async def _apply_googlelens(context: GoogleLensContext) -> str:
    cleaned_content = context.cleaned_content
    if not cleaned_content.lower().startswith("googlelens"):
        return cleaned_content

    cleaned_content = cleaned_content[10:].strip()
    _, _, cached_lens_results = get_bad_keys_db().get_message_search_data(
        context.curr_msg.id,
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
                "reverse image results:\n"
                + "\n".join(lens_results)
            )
            if twitter_content:
                result_text += (
                    "\n\n--- Extracted Twitter/X Content ---"
                    + "".join(twitter_content)
                )
            cleaned_content += result_text

            context.curr_node.lens_results = result_text
            get_bad_keys_db().save_message_search_data(
                context.curr_msg.id,
                lens_results=result_text,
            )
            logger.info("Saved lens results for message %s", context.curr_msg.id)
    except Exception:
        logger.exception("Error fetching Yandex results")

    return cleaned_content


def _get_allowed_attachment_types(actual_model: str) -> tuple[str, ...]:
    allowed_types: tuple[str, ...] = (
        "text",
        "image",
        "application/pdf",
    )
    if is_gemini_model(actual_model):
        allowed_types += ("audio", "video")
    return allowed_types


async def _download_and_process_attachments(
    *,
    attachments: list[discord.Attachment],
    httpx_client: httpx.AsyncClient,
) -> tuple[
    list[discord.Attachment],
    list[httpx.Response],
    list[dict[str, bytes | str | None]],
]:
    successful_pairs = await download_attachments(attachments, httpx_client)
    good_attachments = [pair[0] for pair in successful_pairs]
    attachment_responses = [pair[1] for pair in successful_pairs]
    processed_attachments = await process_attachments(successful_pairs)
    return good_attachments, attachment_responses, processed_attachments


def _build_initial_text(
    *,
    cleaned_content: str,
    curr_msg: discord.Message,
    processed_attachments: list[dict[str, bytes | str | None]],
) -> str:
    return build_node_text_parts(
        cleaned_content,
        curr_msg.embeds,
        curr_msg.components,
        text_attachments=[
            att["text"]
            for att in processed_attachments
            if att["content_type"]
            and att["content_type"].startswith("text")
            and att["text"]
        ],
    )


async def _extract_pdf_texts(
    *,
    processed_attachments: list[dict[str, bytes | str | None]],
    actual_model: str,
) -> list[str]:
    if is_gemini_model(actual_model):
        return []

    pdf_attachments = [
        att
        for att in processed_attachments
        if att["content_type"] == "application/pdf"
    ]
    if not pdf_attachments:
        return []

    pdf_extraction_tasks = [
        extract_pdf_text(att["content"]) for att in pdf_attachments
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


async def _collect_external_content(context: ExternalContentContext) -> list[str]:
    video_ids = re.findall(
        (
            r"(?:https?:\/\/)?(?:[a-zA-Z0-9-]+\.)?"
            r"(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})"
        ),
        context.cleaned_content,
    )
    if video_ids:
        yt_results = await asyncio.gather(
            *[
                extract_youtube_transcript(vid, context.httpx_client)
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
            r"(?:reddit\.com\/r\/[a-zA-Z0-9_]+\/comments\/"
            r"[a-zA-Z0-9_]+(?:[\w\-\.\/\?\=\&%]*)"
            r"|redd\.it\/[a-zA-Z0-9_]+))"
        ),
        context.cleaned_content,
    ):
        post_text = await extract_reddit_post(
            post_url,
            context.httpx_client,
            reddit_client,
        )
        if post_text:
            reddit_posts.append(post_text)

    pdf_texts = await _extract_pdf_texts(
        processed_attachments=context.processed_attachments,
        actual_model=context.actual_model,
    )

    return yt_transcripts + tweets + reddit_posts + pdf_texts


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
        extra_parts=extra_parts,
    )


def _is_googlelens_query(
    new_msg: discord.Message,
    discord_bot: discord.Client,
) -> bool:
    content_for_lens_check = (
        new_msg.content.lower()
        .removeprefix(discord_bot.user.mention.lower())
        .strip()
    )
    if content_for_lens_check.startswith("at ai"):
        content_for_lens_check = content_for_lens_check[5:].strip()
    return content_for_lens_check.startswith("googlelens")


async def _set_parent_message(
    *,
    curr_msg: discord.Message,
    curr_node: MsgNode,
    discord_bot: discord.Client,
) -> None:
    try:
        prev_msg_in_channel = None
        if (
            curr_msg.reference is None
            and discord_bot.user.mention not in curr_msg.content
            and "at ai" not in curr_msg.content.lower()
        ):
            history = [
                m async for m in curr_msg.channel.history(before=curr_msg, limit=1)
            ]
            prev_msg_in_channel = history[0] if history else None

        if prev_msg_in_channel and prev_msg_in_channel.type in (
            discord.MessageType.default,
            discord.MessageType.reply,
        ) and prev_msg_in_channel.author == (
            discord_bot.user
            if curr_msg.channel.type == discord.ChannelType.private
            else curr_msg.author
        ):
            curr_node.parent_msg = prev_msg_in_channel
            return

        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
        parent_is_thread_start = (
            is_public_thread
            and curr_msg.reference is None
            and curr_msg.channel.parent.type == discord.ChannelType.text
        )
        parent_msg_id = (
            curr_msg.channel.id
            if parent_is_thread_start
            else getattr(curr_msg.reference, "message_id", None)
        )
        if not parent_msg_id:
            return

        if parent_is_thread_start:
            curr_node.parent_msg = (
                curr_msg.channel.starter_message
                or await curr_msg.channel.parent.fetch_message(parent_msg_id)
            )
        else:
            curr_node.parent_msg = (
                curr_msg.reference.cached_message
                or await curr_msg.channel.fetch_message(parent_msg_id)
            )
    except (discord.NotFound, discord.HTTPException):
        logger.exception("Error fetching next message in the chain")
        curr_node.fetch_parent_failed = True


async def _populate_node_if_needed(
    *,
    curr_msg: discord.Message,
    curr_node: MsgNode,
    context: MessageBuildContext,
) -> None:
    if curr_node.text is not None:
        return

    cleaned_content = _clean_message_content(curr_msg, context.discord_bot)
    cleaned_content = await _apply_googlelens(
        GoogleLensContext(
            cleaned_content=cleaned_content,
            curr_msg=curr_msg,
            curr_node=curr_node,
            httpx_client=context.httpx_client,
            twitter_api=context.twitter_api,
            max_tweet_replies=context.max_tweet_replies,
        ),
    )

    allowed_types = _get_allowed_attachment_types(context.actual_model)
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
    ) = await _download_and_process_attachments(
        attachments=good_attachments,
        httpx_client=context.httpx_client,
    )

    curr_node.text = _build_initial_text(
        cleaned_content=cleaned_content,
        curr_msg=curr_msg,
        processed_attachments=processed_attachments,
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
        if att["content_type"] and att["content_type"].startswith("image")
    ]

    curr_node.raw_attachments = [
        {"content_type": att["content_type"], "content": att["content"]}
        for att in processed_attachments
    ]

    extra_parts = await _collect_external_content(
        ExternalContentContext(
            cleaned_content=cleaned_content,
            httpx_client=context.httpx_client,
            twitter_api=context.twitter_api,
            max_tweet_replies=context.max_tweet_replies,
            processed_attachments=processed_attachments,
            actual_model=context.actual_model,
        ),
    )

    curr_node.text = _build_final_text(
        cleaned_content=cleaned_content,
        curr_msg=curr_msg,
        good_attachments=good_attachments,
        attachment_responses=attachment_responses,
        extra_parts=extra_parts,
    )

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
        "assistant"
        if curr_msg.author == context.discord_bot.user
        else "user"
    )

    curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None
    curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

    await _set_parent_message(
        curr_msg=curr_msg,
        curr_node=curr_node,
        discord_bot=context.discord_bot,
    )


def _load_cached_search_data(curr_msg: discord.Message, curr_node: MsgNode) -> None:
    if curr_node.search_results is not None and curr_node.lens_results is not None:
        return

    (
        stored_search_results,
        stored_tavily_metadata,
        stored_lens_results,
    ) = get_bad_keys_db().get_message_search_data(curr_msg.id)
    if stored_search_results and curr_node.search_results is None:
        curr_node.search_results = stored_search_results
        curr_node.tavily_metadata = stored_tavily_metadata
        logger.info("Loaded stored search data for message %s", curr_msg.id)
    if stored_lens_results and curr_node.lens_results is None:
        curr_node.lens_results = stored_lens_results
        logger.info("Loaded stored lens results for message %s", curr_msg.id)


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
            if att["content_type"].startswith(("audio", "video")) or (
                att["content_type"] == "application/pdf"
            ):
                encoded_data = b64encode(att["content"]).decode("utf-8")
                file_data = f"data:{att['content_type']};base64,{encoded_data}"
                gemini_file_attachments.append(
                    {"type": "file", "file": {"file_data": file_data}},
                )

    has_images = bool(curr_node.images[:max_images])
    has_gemini_files = bool(gemini_file_attachments)

    if has_images or has_gemini_files:
        content: list[dict[str, object]] = []
        if curr_node.text[:max_text]:
            content.append({"type": "text", "text": curr_node.text[:max_text]})
        content.extend(curr_node.images[:max_images])
        content.extend(gemini_file_attachments)
        if gemini_file_attachments:
            logger.info(
                "Added %s Gemini file attachment(s) (audio/video/PDF) to message",
                len(gemini_file_attachments),
            )
        if not content:
            return [{"type": "text", "text": "What is in this file?"}]
        return content

    return curr_node.text[:max_text]


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


async def _build_messages(
    *,
    context: MessageBuildContext,
) -> MessageBuildResult:
    messages: list[dict[str, object]] = []
    user_warnings: set[str] = set()
    curr_msg = context.new_msg

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
                content = append_search_to_content(content, curr_node.search_results)

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


def _apply_system_prompt(
    *,
    messages: list[dict[str, object]],
    system_prompt: str | None,
    accept_usernames: bool,
) -> None:
    if not system_prompt:
        return

    date_str, time_str = get_current_datetime_strings()
    formatted_prompt = (
        system_prompt.replace("{date}", date_str)
        .replace("{time}", time_str)
        .strip()
    )
    if accept_usernames:
        formatted_prompt += (
            "\n\nUser's names are their Discord IDs and should be typed as '<@ID>'."
        )

    messages.append({"role": "system", "content": formatted_prompt})


async def _run_research_command(
    context: ResearchCommandContext,
) -> dict[str, object] | None:
    if not context.research_model or context.has_existing_search:
        return None

    if not context.research_query:
        context.user_warnings.add(
            "⚠️ Provide a research query after researchpro/researchmini",
        )
        return None

    if not context.tavily_api_keys:
        context.user_warnings.add("⚠️ Tavily API key missing for research")
        return None

    for msg in context.messages:
        if msg.get("role") == "user":
            msg["content"] = replace_content_text(
                msg.get("content", ""),
                context.research_query,
            )
            break

    research_results, search_metadata = await perform_tavily_research(
        query=context.research_query,
        api_keys=context.tavily_api_keys,
        model=context.research_model,
    )

    if research_results:
        for msg in context.messages:
            if msg.get("role") == "user":
                msg["content"] = append_search_to_content(
                    msg.get("content", ""),
                    research_results,
                )

                logger.info("Tavily research results appended to user message")

                get_bad_keys_db().save_message_search_data(
                    context.new_msg.id,
                    research_results,
                    search_metadata,
                )

                if context.new_msg.id in context.msg_nodes:
                    node = context.msg_nodes[context.new_msg.id]
                    node.search_results = research_results
                    node.tavily_metadata = search_metadata
                break

    return search_metadata


async def _maybe_run_web_search(
    context: WebSearchContext,
) -> dict[str, object] | None:
    search_metadata = context.search_metadata
    is_preview_model = "preview" in context.actual_model.lower()
    is_non_gemini = not is_gemini_model(context.actual_model)
    if (
        context.web_search_provider
        and (is_non_gemini or is_preview_model)
        and not context.is_googlelens_query
        and not context.has_existing_search
    ):
        db = get_bad_keys_db()
        user_id = str(context.new_msg.author.id)
        user_decider_model = db.get_user_search_decider_model(user_id)
        default_decider = context.config.get(
            "web_search_decider_model",
            "gemini/gemini-3-flash-preview",
        )

        if user_decider_model and user_decider_model in context.config.get(
            "models",
            {},
        ):
            decider_model_str = user_decider_model
        else:
            decider_model_str = default_decider

        decider_provider, decider_model = (
            decider_model_str.split("/", 1)
            if "/" in decider_model_str
            else ("gemini", decider_model_str)
        )

        decider_provider_config = context.config.get("providers", {}).get(
            decider_provider,
            {},
        )
        decider_api_keys = ensure_list(decider_provider_config.get("api_key"))

        decider_config = {
            "provider": decider_provider,
            "model": decider_model,
            "api_keys": decider_api_keys,
            "base_url": decider_provider_config.get("base_url"),
        }

        if decider_api_keys:
            search_decision = await decide_web_search(context.messages, decider_config)

            if search_decision.get("needs_search") and search_decision.get("queries"):
                queries = search_decision["queries"]
                logger.info(
                    "Web search triggered with %s. Queries: %s",
                    context.web_search_provider,
                    queries,
                )

                search_depth = context.config.get("tavily_search_depth", "advanced")
                effective_exa_mcp_url = context.exa_mcp_url or EXA_MCP_URL
                search_options = WebSearchOptions(
                    max_results_per_query=5,
                    max_chars_per_url=2000,
                    search_depth=search_depth,
                    web_search_provider=context.web_search_provider,
                    exa_mcp_url=effective_exa_mcp_url,
                )
                search_results, search_metadata = await perform_web_search(
                    queries,
                    api_keys=context.tavily_api_keys,
                    options=search_options,
                )

                if search_results:
                    for msg in context.messages:
                        if msg.get("role") == "user":
                            msg["content"] = append_search_to_content(
                                msg.get("content", ""),
                                search_results,
                            )

                            logger.info(
                                "Web search results appended to user message",
                            )

                            get_bad_keys_db().save_message_search_data(
                                context.new_msg.id,
                                search_results,
                                search_metadata,
                            )

                            if context.new_msg.id in context.msg_nodes:
                                node = context.msg_nodes[context.new_msg.id]
                                node.search_results = search_results
                                node.tavily_metadata = search_metadata
                            break

    context.search_metadata = search_metadata
    return search_metadata


async def process_message(
    new_msg: discord.Message,
    context: ProcessContext,
) -> None:
    """Process a message."""
    discord_bot = context.discord_bot
    httpx_client = context.httpx_client
    twitter_api = context.twitter_api
    msg_nodes = context.msg_nodes
    curr_model_lock = context.curr_model_lock
    curr_model_ref = context.curr_model_ref
    override_provider_slash_model = context.override_provider_slash_model
    fallback_chain = context.fallback_chain
    # Per-request edit timing to avoid interference between concurrent requests
    last_edit_time = 0

    should_process, processing_msg_or_none = await should_process_message(
        new_msg, discord_bot,
    )
    if not should_process:
        return

    # should_process guarantees processing_msg is not None if it returns True
    if processing_msg_or_none is None:
        return
    processing_msg = processing_msg_or_none

    config = get_config()  # Now cached, no need for to_thread

    provider_settings = await _resolve_provider_settings(
        processing_msg=processing_msg,
        curr_model_lock=curr_model_lock,
        curr_model_ref=curr_model_ref,
        override_provider_slash_model=override_provider_slash_model,
        config=config,
    )
    if provider_settings is None:
        return

    accept_images = any(
        x in provider_settings.provider_slash_model.lower()
        for x in VISION_MODEL_TAGS
    )
    accept_usernames = any(
        provider_settings.provider_slash_model.lower().startswith(x)
        for x in PROVIDERS_SUPPORTING_USERNAMES
    )

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)
    max_tweet_replies = config.get("max_tweet_replies", 50)
    build_result = await _build_messages(
        context=MessageBuildContext(
            new_msg=new_msg,
            discord_bot=discord_bot,
            httpx_client=httpx_client,
            twitter_api=twitter_api,
            msg_nodes=msg_nodes,
            actual_model=provider_settings.actual_model,
            accept_usernames=accept_usernames,
            max_text=max_text,
            max_images=max_images,
            max_messages=max_messages,
            max_tweet_replies=max_tweet_replies,
        ),
    )
    messages = build_result.messages
    user_warnings = build_result.user_warnings

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
    _apply_system_prompt(
        messages=messages,
        system_prompt=system_prompt,
        accept_usernames=accept_usernames,
    )

    # Web Search Integration for non-Gemini models and Gemini preview models
    # Supports both Tavily and Exa MCP as search providers
    tavily_api_keys = ensure_list(config.get("tavily_api_key"))
    exa_mcp_url = config.get("exa_mcp_url", "")
    web_search_provider, web_search_available = _resolve_web_search_provider(
        config,
        tavily_api_keys,
        exa_mcp_url,
    )
    search_metadata = await _resolve_search_metadata(
        SearchResolutionContext(
            new_msg=new_msg,
            discord_bot=discord_bot,
            msg_nodes=msg_nodes,
            messages=messages,
            user_warnings=user_warnings,
            tavily_api_keys=tavily_api_keys,
            config=config,
            web_search_available=web_search_available,
            web_search_provider=web_search_provider,
            exa_mcp_url=exa_mcp_url,
            actual_model=provider_settings.actual_model,
        ),
    )

    # Continue with response generation
    async def retry_callback() -> None:
        retry_context = ProcessContext(
            discord_bot=discord_bot,
            httpx_client=httpx_client,
            twitter_api=twitter_api,
            msg_nodes=msg_nodes,
            curr_model_lock=curr_model_lock,
            curr_model_ref=curr_model_ref,
            override_provider_slash_model="gemini/gemma-3-27b-it",
            fallback_chain=[
                (
                    "mistral",
                    "mistral-large-latest",
                    "mistral/mistral-large-latest",
                ),
            ],
        )
        await process_message(new_msg=new_msg, context=retry_context)

    tavily_metadata = search_metadata
    generation_context = GenerationContext(
        new_msg=new_msg,
        discord_bot=discord_bot,
        msg_nodes=msg_nodes,
        messages=messages,
        user_warnings=user_warnings,
        provider=provider_settings.provider,
        model=provider_settings.model,
        actual_model=provider_settings.actual_model,
        provider_slash_model=provider_settings.provider_slash_model,
        base_url=provider_settings.base_url,
        api_keys=provider_settings.api_keys,
        model_parameters=provider_settings.model_parameters,
        extra_headers=provider_settings.extra_headers,
        extra_query=provider_settings.extra_query,
        extra_body=provider_settings.extra_body,
        system_prompt=system_prompt,
        config=config,
        max_text=max_text,
        tavily_metadata=tavily_metadata,
        last_edit_time=last_edit_time,
        processing_msg=processing_msg,
        retry_callback=retry_callback,
        fallback_chain=fallback_chain,
    )
    await generate_response(generation_context)
