"""Top-level message processing workflow."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import discord

from llmcord.config import (
    EMBED_COLOR_INCOMPLETE,
    PROVIDERS_SUPPORTING_USERNAMES,
    VISION_MODEL_TAGS,
    ensure_list,
    get_config,
)

from .message_builder import build_messages_from_chain
from .response_generation import generate_response
from .search_handlers import append_system_prompt, handle_search_workflow
from .shared import TwitterApiProtocol, create_processing_message

if TYPE_CHECKING:
    import asyncio

    import asyncpraw
    import httpx

    from llmcord.models import MsgNode

logger = logging.getLogger(__name__)


def _is_message_allowed(new_msg: discord.Message, discord_bot: discord.Client) -> bool:
    """Return True if the message passes permission checks."""
    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (
        not is_dm
        and discord_bot.user not in new_msg.mentions
        and "at ai" not in new_msg.content.lower()
    ) or new_msg.author.bot:
        return False

    role_ids = {role.id for role in getattr(new_msg.author, "roles", ())}
    channel_ids = {
        channel_id
        for channel_id in (
            new_msg.channel.id,
            getattr(new_msg.channel, "parent_id", None),
            getattr(new_msg.channel, "category_id", None),
        )
        if channel_id is not None
    }

    config = get_config()  # Now cached, no need for to_thread

    allow_dms = config.get("allow_dms", True)

    permissions = config["permissions"]

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    # Pre-convert to sets once for efficient lookups
    allowed_user_ids = set(permissions["users"]["allowed_ids"])
    blocked_user_ids = set(permissions["users"]["blocked_ids"])
    allowed_role_ids = set(permissions["roles"]["allowed_ids"])
    blocked_role_ids = set(permissions["roles"]["blocked_ids"])
    allowed_channel_ids = set(permissions["channels"]["allowed_ids"])
    blocked_channel_ids = set(permissions["channels"]["blocked_ids"])

    allow_all_users = (
        not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    )
    is_good_user = (
        user_is_admin
        or allow_all_users
        or new_msg.author.id in allowed_user_ids
        or bool(role_ids & allowed_role_ids)
    )
    is_bad_user = (
        not is_good_user
        or new_msg.author.id in blocked_user_ids
        or bool(role_ids & blocked_role_ids)
    )

    allow_all_channels = not allowed_channel_ids
    is_good_channel = (
        user_is_admin or allow_dms
        if is_dm
        else allow_all_channels or bool(channel_ids & allowed_channel_ids)
    )
    is_bad_channel = not is_good_channel or bool(channel_ids & blocked_channel_ids)

    return not (is_bad_user or is_bad_channel)


async def process_message(  # noqa: PLR0913
    new_msg: discord.Message,
    discord_bot: discord.Client,
    httpx_client: httpx.AsyncClient,
    twitter_api: TwitterApiProtocol,
    reddit_client: asyncpraw.Reddit | None,
    msg_nodes: dict[int, MsgNode],
    curr_model_lock: asyncio.Lock,
    curr_model_ref: list[str],
) -> None:
    """Process a message."""
    # Per-request edit timing to avoid interference between concurrent requests
    last_edit_time = 0

    if not _is_message_allowed(new_msg, discord_bot):
        return

    config = get_config()
    use_plain_responses = config.get("use_plain_responses", False)

    # Send processing message immediately after confirming bot should respond
    processing_msg = await create_processing_message(
        new_msg,
        use_plain_responses=use_plain_responses,
    )

    # Thread-safe read of current model
    async with curr_model_lock:
        provider_slash_model = curr_model_ref[0]

    # Validate provider/model format
    try:
        provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)
    except ValueError:
        logger.exception(
            "Invalid model format: %s. Expected 'provider/model'.",
            provider_slash_model,
        )
        embed = discord.Embed(
            description=(
                "❌ Invalid model configuration: "
                f"'{provider_slash_model}'. Expected format: 'provider/model'.\n"
                "Please contact an administrator."
            ),
            color=EMBED_COLOR_INCOMPLETE,
        )
        await processing_msg.edit(embed=embed)
        return

    # Validate provider exists in config
    providers = config.get("providers", {})
    if provider not in providers:
        logger.error("Provider '%s' not found in config.", provider)
        embed = discord.Embed(
            description=(
                f"❌ Provider '{provider}' is not configured. "
                "Please contact an administrator."
            ),
            color=EMBED_COLOR_INCOMPLETE,
        )
        await processing_msg.edit(embed=embed)
        return

    provider_config = providers[provider]

    base_url = provider_config.get("base_url")
    api_keys = ensure_list(provider_config.get("api_key")) or ["sk-no-key-required"]

    model_parameters = config["models"].get(provider_slash_model, None)

    # Support model aliasing: if config specifies a different actual model name, use it
    # This allows e.g. "gemini-3-flash-minimal" to map to "gemini-3-flash-preview"
    actual_model = model_parameters.get("model", model) if model_parameters else model

    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (
        (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None
    )

    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(
        provider_slash_model.lower().startswith(x)
        for x in PROVIDERS_SUPPORTING_USERNAMES
    )

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)
    max_tweet_replies = config.get("max_tweet_replies", 50)

    messages, user_warnings = await build_messages_from_chain(
        new_msg=new_msg,
        discord_bot=discord_bot,
        httpx_client=httpx_client,
        twitter_api=twitter_api,
        reddit_client=reddit_client,
        msg_nodes=msg_nodes,
        actual_model=actual_model,
        max_text=max_text,
        max_images=max_images,
        max_messages=max_messages,
        max_tweet_replies=max_tweet_replies,
        accept_usernames=accept_usernames,
    )

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
    system_prompt = append_system_prompt(
        messages,
        system_prompt,
        accept_usernames=accept_usernames,
    )

    search_metadata = await handle_search_workflow(
        new_msg=new_msg,
        discord_bot=discord_bot,
        messages=messages,
        msg_nodes=msg_nodes,
        user_warnings=user_warnings,
        actual_model=actual_model,
    )

    # Continue with response generation
    async def retry_callback() -> None:
        await process_message(
            new_msg=new_msg,
            discord_bot=discord_bot,
            httpx_client=httpx_client,
            twitter_api=twitter_api,
            reddit_client=reddit_client,
            msg_nodes=msg_nodes,
            curr_model_lock=curr_model_lock,
            curr_model_ref=curr_model_ref,
        )

    await generate_response(
        new_msg=new_msg,
        _discord_bot=discord_bot,
        msg_nodes=msg_nodes,
        messages=messages,
        user_warnings=user_warnings,
        provider=provider,
        _model=model,
        actual_model=actual_model,
        provider_slash_model=provider_slash_model,
        base_url=base_url,
        api_keys=api_keys,
        model_parameters=model_parameters,
        extra_headers=extra_headers,
        _extra_query=extra_query,
        _extra_body=extra_body,
        _system_prompt=system_prompt,
        config=config,
        _max_text=max_text,
        tavily_metadata=search_metadata,
        last_edit_time=last_edit_time,
        processing_msg=processing_msg,
        retry_callback=retry_callback,
    )
