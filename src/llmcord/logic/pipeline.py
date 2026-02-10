"""Message processing orchestration."""

import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, cast

import discord
import httpx

from llmcord.core.config import (
    EMBED_COLOR_INCOMPLETE,
    PROVIDERS_SUPPORTING_USERNAMES,
    VISION_MODEL_TAGS,
    ensure_list,
    get_config,
)
from llmcord.core.models import MsgNode
from llmcord.discord.ui.utils import build_error_embed
from llmcord.logic.content import is_googlelens_query
from llmcord.logic.generation import GenerationContext, generate_response
from llmcord.logic.messages import MessageBuildContext, build_messages
from llmcord.logic.permissions import should_process_message
from llmcord.logic.providers import (
    ProviderSettings,
    resolve_provider_settings,
)
from llmcord.logic.search_logic import (
    SearchResolutionContext,
    resolve_search_metadata,
    resolve_web_search_provider,
)
from llmcord.services.extractors import TwitterApiProtocol
from llmcord.services.search import get_current_datetime_strings

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


async def _send_processing_error(
    processing_msg: discord.Message,
    description: str,
) -> None:
    embed = discord.Embed(
        description=description,
        color=EMBED_COLOR_INCOMPLETE,
    )
    await processing_msg.edit(embed=embed)


def _is_system_prompt_disabled(
    *,
    provider_settings: ProviderSettings,
    config: dict[str, object],
) -> bool:
    model_parameters = provider_settings.model_parameters or {}
    disable_override = model_parameters.get("disable_system_prompt")
    if isinstance(disable_override, bool):
        return disable_override

    disabled_models = ensure_list(cast(Any, config.get("disable_system_prompt_models")))
    normalized_targets = {
        provider_settings.provider_slash_model.lower(),
        f"{provider_settings.provider}/{provider_settings.actual_model}".lower(),
        provider_settings.actual_model.lower(),
        provider_settings.model.lower(),
    }
    for model_name in disabled_models:
        if not isinstance(model_name, str):
            continue
        model_name_lower = model_name.strip().lower()
        if model_name_lower and model_name_lower in normalized_targets:
            return True

    return False


def _apply_system_prompt(
    *,
    messages: list[dict[str, object]],
    system_prompt: str | None,
    accept_usernames: bool,
    apply_system_prompt: bool,
) -> None:
    if not system_prompt:
        return

    if not apply_system_prompt:
        return

    date_str, time_str = get_current_datetime_strings()
    formatted_prompt = (
        system_prompt.replace("{date}", date_str).replace("{time}", time_str).strip()
    )
    if accept_usernames:
        formatted_prompt += (
            "\n\nUser's names are their Discord IDs and should be typed as '<@ID>'."
        )
    messages.append({"role": "system", "content": formatted_prompt})


def _get_message_limits(
    config: dict,
    *,
    accept_images: bool,
) -> tuple[int, int, int, int, bool, str | None, str | None]:
    """Extract message limits from config."""
    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)
    max_tweet_replies = config.get("max_tweet_replies", 50)
    enable_youtube_transcripts = config.get("enable_youtube_transcripts", True)
    youtube_transcript_proxy = config.get(
        "youtube_transcript_proxy",
    ) or config.get("proxy_url")
    reddit_proxy = config.get("reddit_proxy") or config.get("proxy_url")
    return (
        max_text,
        max_images,
        max_messages,
        max_tweet_replies,
        enable_youtube_transcripts,
        youtube_transcript_proxy,
        reddit_proxy,
    )


from collections.abc import Awaitable, Callable
...
def _make_retry_callback(
    new_msg: discord.Message,
    config: dict,
    context: ProcessContext,
) -> Callable[[], Awaitable[None]]:
    """Create a retry callback for when the primary model fails."""

    async def retry_callback() -> None:
        retry_model = config.get("retry_stable_model")
        if not isinstance(retry_model, str) or not retry_model.strip():
            retry_model = "gemini/gemma-3-27b-it"

        retry_context = ProcessContext(
            discord_bot=context.discord_bot,
            httpx_client=context.httpx_client,
            twitter_api=context.twitter_api,
            msg_nodes=context.msg_nodes,
            curr_model_lock=context.curr_model_lock,
            curr_model_ref=context.curr_model_ref,
            override_provider_slash_model=retry_model,
            fallback_chain=[
                (
                    "mistral",
                    "mistral-large-latest",
                    "mistral/mistral-large-latest",
                ),
            ],
        )
        await process_message(new_msg=new_msg, context=retry_context)

    return retry_callback


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
    last_edit_time = 0

    should_process, processing_msg_or_none = await should_process_message(
        new_msg,
        discord_bot,
    )
    if not should_process:
        return
    if processing_msg_or_none is None:
        return
    processing_msg = processing_msg_or_none

    config = get_config()
    use_plain_responses = config.get("use_plain_responses", False)

    try:
        provider_settings = await resolve_provider_settings(
            processing_msg=processing_msg,
            curr_model_lock=curr_model_lock,
            curr_model_ref=curr_model_ref,
            override_provider_slash_model=override_provider_slash_model,
            send_error_func=_send_processing_error,
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

        (
            max_text,
            max_images,
            max_messages,
            max_tweet_replies,
            enable_youtube_transcripts,
            youtube_transcript_proxy,
            reddit_proxy,
        ) = _get_message_limits(config, accept_images=accept_images)

        build_result = await build_messages(
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
                enable_youtube_transcripts=enable_youtube_transcripts,
                youtube_transcript_proxy=youtube_transcript_proxy,
                reddit_proxy=reddit_proxy,
            ),
        )
        messages = build_result.messages
        user_warnings = build_result.user_warnings

        logger.info(
            "Message received (user ID: %s, attachments: %s, "
            "conversation length: %s):\n%s",
            new_msg.author.id,
            len(new_msg.attachments),
            len(messages),
            new_msg.content,
        )

        if not messages:
            logger.warning(
                "No valid messages could be built from the conversation.",
            )
            embed = discord.Embed(
                description="❌ Could not process your message. Please try again.",
                color=EMBED_COLOR_INCOMPLETE,
            )
            await processing_msg.edit(embed=embed)
            return

        system_prompt = config.get("system_prompt")
        apply_system_prompt = not _is_system_prompt_disabled(
            provider_settings=provider_settings,
            config=config,
        )
        _apply_system_prompt(
            messages=messages,
            system_prompt=system_prompt,
            accept_usernames=accept_usernames,
            apply_system_prompt=apply_system_prompt,
        )

        tavily_api_keys = ensure_list(config.get("tavily_api_key"))
        exa_mcp_url = config.get("exa_mcp_url", "")
        web_search_provider, web_search_available = resolve_web_search_provider(
            config,
            tavily_api_keys,
            exa_mcp_url,
        )
        search_metadata = await resolve_search_metadata(
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
            is_googlelens_query_func=is_googlelens_query,
        )

        retry_callback = _make_retry_callback(new_msg, config, context)

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
            tavily_metadata=search_metadata,
            last_edit_time=last_edit_time,
            processing_msg=processing_msg,
            retry_callback=retry_callback,
            fallback_chain=fallback_chain,
        )
        await generate_response(generation_context)
    except Exception:
        logger.exception("Error processing message")
        with suppress(Exception):
            error_message = (
                "An internal error occurred while processing your request. "
                "Please try again later."
            )
            if use_plain_responses:
                await processing_msg.edit(
                    content=error_message,
                    embed=None,
                    view=None,
                )
            else:
                await processing_msg.edit(
                    embed=build_error_embed(error_message),
                    view=None,
                )
        return
