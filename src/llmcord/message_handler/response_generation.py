"""Response generation and streaming logic."""
# ruff: noqa: E501

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import discord
import litellm
from discord.ui import LayoutView, TextDisplay

from llmcord.bad_keys import get_bad_keys_db
from llmcord.config import (
    EDIT_DELAY_SECONDS,
    EMBED_COLOR_COMPLETE,
    EMBED_COLOR_INCOMPLETE,
    MAX_MESSAGE_NODES,
    STREAMING_INDICATOR,
    ensure_list,
    is_gemini_model,
)
from llmcord.litellm_utils import LiteLLMOptions, prepare_litellm_kwargs
from llmcord.models import MsgNode
from llmcord.views import (
    ResponseView,
    SourceButton,
    SourceView,
    TavilySourceButton,
    _has_grounding_data,
)

from .shared import (
    FirstTokenTimeoutError,
    build_warning_fields,
    count_conversation_tokens,
    count_text_tokens,
    extract_grounding_metadata,
    raise_empty_response,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

logger = logging.getLogger(__name__)

MAX_OVERLOADED_ERRORS = 3
LITELLM_TIMEOUT_SECONDS = 60
FIRST_TOKEN_TIMEOUT_SECONDS = 60


async def generate_response(  # noqa: C901, PLR0912, PLR0913, PLR0915
    new_msg: discord.Message,
    _discord_bot: discord.Client,
    msg_nodes: dict[int, MsgNode],
    messages: list[dict[str, object]],
    user_warnings: set[str],
    provider: str,
    _model: str,
    actual_model: str,
    provider_slash_model: str,
    base_url: str | None,
    api_keys: list[str],
    model_parameters: dict[str, object] | None,
    extra_headers: dict[str, str] | None,
    _extra_query: dict[str, object] | None,
    _extra_body: dict[str, object] | None,
    _system_prompt: str | None,
    config: dict[str, object],
    _max_text: int,
    tavily_metadata: dict[str, object] | None,
    last_edit_time: float,
    processing_msg: discord.Message,
    retry_callback: Callable[[], Awaitable[None]],
) -> None:
    """Generate and stream the LLM response using LiteLLM."""
    curr_content = finish_reason = None
    # Initialize with the pre-created processing message
    response_msgs = [processing_msg]
    msg_nodes[processing_msg.id] = MsgNode(parent_msg=new_msg)
    await msg_nodes[processing_msg.id].lock.acquire()
    response_contents = []

    # Count input tokens (chat history + latest query)
    input_tokens = count_conversation_tokens(messages)

    if use_plain_responses := config.get("use_plain_responses", False):
        max_message_length = 4000
    else:
        max_message_length = 4096 - len(STREAMING_INDICATOR)
        embed = discord.Embed.from_dict(
            {
                "fields": build_warning_fields(sorted(user_warnings)),
            },
        )
        embed.set_footer(text=f"{provider_slash_model} | total tokens: {input_tokens:,}")

    async def reply_helper(**reply_kwargs: object) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)

        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()

    async def get_stream(
        api_key: str,
    ) -> AsyncIterator[tuple[str, object | None, object | None]]:
        """Get streaming response from LiteLLM using shared configuration."""
        # Check if Gemini grounding should be enabled (no URL in message content)
        enable_grounding = not re.search(r"https?://", new_msg.content)

        # Use shared utility to prepare kwargs with all provider-specific config
        litellm_kwargs = prepare_litellm_kwargs(
            provider=provider,
            model=actual_model,
            messages=messages[::-1],  # Reverse to get chronological order
            api_key=api_key,
            options=LiteLLMOptions(
                base_url=base_url,
                extra_headers=extra_headers,
                stream=True,
                model_parameters=model_parameters,
                enable_grounding=enable_grounding,
            ),
        )

        # Add timeout to prevent indefinite hangs (60 seconds for streaming)
        litellm_kwargs["timeout"] = LITELLM_TIMEOUT_SECONDS

        # Make the streaming call
        stream = await litellm.acompletion(**litellm_kwargs)

        async def _iter_stream_with_first_chunk(
            stream_iter: AsyncIterator[object],
        ) -> AsyncIterator[object]:
            try:
                first_chunk = await asyncio.wait_for(
                    stream_iter.__anext__(),
                    timeout=FIRST_TOKEN_TIMEOUT_SECONDS,
                )
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError as exc:
                raise FirstTokenTimeoutError from exc

            yield first_chunk
            async for chunk in stream_iter:
                yield chunk

        async for chunk in _iter_stream_with_first_chunk(stream):
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta_content = choice.delta.content or ""
            chunk_finish_reason = choice.finish_reason

            grounding_metadata = extract_grounding_metadata(chunk, choice)

            # Log the chunk attributes on finish to help debug
            if chunk_finish_reason and is_gemini_model(actual_model):
                chunk_attrs = [attr for attr in dir(chunk) if not attr.startswith("_")]
                logger.debug("Gemini chunk finish - attributes: %s", chunk_attrs)
                if hasattr(chunk, "model_extra") and chunk.model_extra:
                    logger.info(
                        "Gemini chunk model_extra keys: %s",
                        list(chunk.model_extra.keys()),
                    )
                hidden_params = getattr(chunk, "_hidden_params", None)
                if hidden_params:
                    logger.info(
                        "Gemini chunk _hidden_params keys: %s",
                        list(hidden_params.keys()),
                    )

            yield delta_content, chunk_finish_reason, grounding_metadata

    grounding_metadata = None
    attempt_count = 0
    fallback_level = 0  # 0 = original, 1 = mistral, 2 = gemma
    original_provider = provider  # Store original for logging
    original_model = actual_model
    last_error_msg = None

    # Determine if the original model is already mistral (skip to gemma fallback)
    is_original_mistral = (
        original_provider == "mistral" and "mistral" in original_model.lower()
    )

    # Get good keys (filter out known bad ones - synced with search decider)
    try:
        good_keys = get_bad_keys_db().get_good_keys_synced(provider, api_keys)
    except Exception:
        logger.exception("Failed to get good keys, falling back to all keys")
        good_keys = api_keys.copy()

    # If all keys are bad, reset and try again with all keys
    if not good_keys:
        logger.warning(
            "All API keys for provider '%s' (synced) are marked as bad. Resetting...",
            provider,
        )
        try:
            get_bad_keys_db().reset_provider_keys_synced(provider)
        except Exception:
            logger.exception("Failed to reset provider keys")
        good_keys = api_keys.copy()

    initial_key_count = len(good_keys)

    while True:
        curr_content = finish_reason = None
        response_contents = []
        attempt_count += 1

        # Get the next good key to try
        if not good_keys:
            # All good keys exhausted, try next fallback level
            next_fallback = None

            if fallback_level == 0:
                # First fallback: mistral (unless original was already mistral)
                if is_original_mistral:
                    # Skip mistral, go directly to gemma
                    fallback_level = 2
                    next_fallback = ("gemini", "gemma-3-27b-it", "gemini/gemma-3-27b-it")
                    logger.warning(
                        "All %s keys exhausted for mistral/%s. Falling back to gemini/gemma-3-27b-it...",
                        initial_key_count,
                        original_model,
                    )
                else:
                    fallback_level = 1
                    next_fallback = ("mistral", "mistral-large-latest", "mistral/mistral-large-latest")
                    logger.warning(
                        "All %s keys exhausted for provider '%s'. Falling back to mistral/mistral-large-latest...",
                        initial_key_count,
                        original_provider,
                    )
            elif fallback_level == 1:
                # Second fallback: gemma
                fallback_level = 2
                next_fallback = ("gemini", "gemma-3-27b-it", "gemini/gemma-3-27b-it")
                logger.warning(
                    "Mistral fallback also failed. Falling back to gemini/gemma-3-27b-it...",
                )

            if next_fallback:
                new_provider, new_model, new_provider_slash_model = next_fallback

                # Switch to fallback provider
                provider = new_provider
                actual_model = new_model
                provider_slash_model = new_provider_slash_model

                # Get fallback provider configuration
                fallback_provider_config = config.get("providers", {}).get(provider, {})
                base_url = fallback_provider_config.get("base_url")
                fallback_api_keys = ensure_list(fallback_provider_config.get("api_key"))

                if fallback_api_keys:
                    api_keys = fallback_api_keys
                    good_keys = fallback_api_keys.copy()
                    initial_key_count = len(good_keys)
                    attempt_count = 0  # Reset attempt count for new provider
                    continue  # Try with the new provider
                logger.error(
                    "No API keys available for fallback provider '%s'",
                    provider,
                )
                # Continue to try the next fallback level
                good_keys = []
                continue
            logger.error("All fallback options exhausted (mistral and gemma)")
            error_text = (
                "❌ All API keys (including all fallbacks) exhausted. Please try again later."
            )
            if last_error_msg:
                error_text += f"\nLast error: {last_error_msg}"
            if use_plain_responses:
                layout = LayoutView().add_item(TextDisplay(content=error_text))
                if response_msgs:
                    await response_msgs[-1].edit(view=layout)
                else:
                    await reply_helper(view=layout)
                response_contents = [error_text]
            else:
                embed.description = error_text
                embed.color = EMBED_COLOR_INCOMPLETE
                if response_msgs:
                    await response_msgs[-1].edit(embed=embed, view=None)
                else:
                    await reply_helper(embed=embed)
                response_contents = [error_text]
            break

        current_api_key = good_keys[(attempt_count - 1) % len(good_keys)]

        try:
            async with new_msg.channel.typing():
                async for delta_content, new_finish_reason, new_grounding_metadata in get_stream(current_api_key):
                    if new_grounding_metadata:
                        grounding_metadata = new_grounding_metadata
                        logger.info(
                            "Captured grounding metadata from stream: %s",
                            type(grounding_metadata),
                        )

                    if finish_reason is not None:
                        break

                    finish_reason = new_finish_reason

                    prev_content = curr_content or ""
                    curr_content = delta_content

                    new_content = (
                        prev_content
                        if finish_reason is None
                        else (prev_content + curr_content)
                    )

                    if response_contents == [] and new_content == "":
                        continue

                    if start_next_msg := response_contents == [] or (
                        len(response_contents[-1] + new_content) > max_message_length
                    ):
                        response_contents.append("")

                    response_contents[-1] += new_content

                    if not use_plain_responses:
                        time_delta = datetime.now(timezone.utc).timestamp() - last_edit_time

                        ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                        msg_split_incoming = finish_reason is None and (
                            len(response_contents[-1] + curr_content)
                            > max_message_length
                        )
                        is_final_edit = finish_reason is not None or msg_split_incoming
                        is_good_finish = finish_reason is not None and any(
                            x in str(finish_reason).lower()
                            for x in ("stop", "end_turn")
                        )

                        if start_next_msg or ready_to_edit or is_final_edit:
                            embed.description = (
                                response_contents[-1]
                                if is_final_edit
                                else (response_contents[-1] + STREAMING_INDICATOR)
                            )
                            embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                            view = (
                                SourceView(grounding_metadata)
                                if is_final_edit
                                and _has_grounding_data(grounding_metadata)
                                else None
                            )

                            msg_index = len(response_contents) - 1
                            if start_next_msg:
                                if msg_index < len(response_msgs):
                                    await response_msgs[msg_index].edit(
                                        embed=embed,
                                        view=view,
                                    )
                                else:
                                    await reply_helper(
                                        embed=embed,
                                        silent=True,
                                        view=view,
                                    )
                            else:
                                await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
                                await response_msgs[msg_index].edit(
                                    embed=embed,
                                    view=view,
                                )

                            last_edit_time = datetime.now(timezone.utc).timestamp()

            if not response_contents:
                raise_empty_response()

            if use_plain_responses:
                for i, content in enumerate(response_contents):
                    # Build the LayoutView with text content
                    layout = LayoutView().add_item(TextDisplay(content=content))

                    # Add buttons only to the last message
                    if i == len(response_contents) - 1:
                        # Add Gemini grounding sources button if available
                        if _has_grounding_data(grounding_metadata):
                            layout.add_item(SourceButton(grounding_metadata))

                        # Add Tavily sources button if available
                        if tavily_metadata and (
                            tavily_metadata.get("urls")
                            or tavily_metadata.get("queries")
                        ):
                            layout.add_item(TavilySourceButton(tavily_metadata))

                    if i < len(response_msgs):
                        # Edit existing message (first one is the processing message)
                        await response_msgs[i].edit(view=layout)
                    else:
                        # Create new message for overflow content
                        await reply_helper(view=layout)

            break

        except Exception as e:
            last_error_msg = str(e)
            logger.exception("Error while generating response")

            # Determine if this is an API key error vs other error (timeout, network, etc.)
            is_key_error = False
            is_first_token_timeout = isinstance(e, FirstTokenTimeoutError)
            is_timeout_error = (
                isinstance(e, asyncio.TimeoutError) or "timeout" in str(e).lower()
            )
            is_empty_response = "no content" in str(e).lower()

            if is_first_token_timeout:
                logger.warning(
                    "No first token within %s seconds for provider '%s'; forcing fallback model.",
                    FIRST_TOKEN_TIMEOUT_SECONDS,
                    provider,
                )
                good_keys = []
                continue

            if is_empty_response:
                logger.warning(
                    "Empty response for provider '%s'; forcing fallback model.",
                    provider,
                )
                good_keys = []
                continue

            # Check for typical API key/auth error patterns
            error_str = str(e).lower()

            key_error_patterns = [
                "unauthorized", "invalid_api_key", "invalid key", "api key",
                "authentication", "forbidden", "401", "403", "quota", "rate limit",
                "billing", "insufficient_quota", "expired",
            ]
            for pattern in key_error_patterns:
                if pattern in error_str:
                    is_key_error = True
                    break

            # Only mark the key as bad for actual key-related errors
            if is_key_error:
                error_msg = str(e)[:200] if e else "Unknown error"
                try:
                    get_bad_keys_db().mark_key_bad_synced(
                        provider,
                        current_api_key,
                        error_msg,
                    )
                except Exception:
                    logger.exception("Failed to mark key as bad")

                # Remove the bad key from good_keys list for this session
                if current_api_key in good_keys:
                    good_keys.remove(current_api_key)
                    logger.info(
                        "Removed bad key for '%s', %s keys remaining",
                        provider,
                        len(good_keys),
                    )
            elif is_timeout_error or is_empty_response:
                # For timeouts and empty responses, don't mark key as bad but still retry
                # Remove from good_keys for this session to try next key
                if current_api_key in good_keys:
                    good_keys.remove(current_api_key)
                logger.warning(
                    "Non-key error for provider '%s': %s, %s keys remaining",
                    provider,
                    "timeout" if is_timeout_error else "empty response",
                    len(good_keys),
                )
            else:
                # For other errors, also try the next key
                if current_api_key in good_keys:
                    good_keys.remove(current_api_key)
                logger.warning(
                    "Unknown error for provider '%s', %s keys remaining",
                    provider,
                    len(good_keys),
                )

            # If no keys left, the main loop will handle fallback to mistral
            # Continue to the next iteration

    if not use_plain_responses and len(response_msgs) > len(response_contents):
        for msg in response_msgs[len(response_contents):]:
            await msg.delete()
            if msg.id in msg_nodes:
                msg_nodes[msg.id].lock.release()
                del msg_nodes[msg.id]
        response_msgs = response_msgs[:len(response_contents)]

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    full_response = "".join(response_contents)

    # Update the last message with ResponseView for "View Response Better" button
    if not use_plain_responses and response_msgs and response_contents:
        response_view = ResponseView(
            full_response,
            grounding_metadata,
            tavily_metadata,
            retry_callback,
            new_msg.author.id,
        )

        # Count output tokens and update footer with total
        output_tokens = count_text_tokens(full_response)
        total_tokens = input_tokens + output_tokens

        # Update the last message with the final view
        last_msg_index = len(response_msgs) - 1
        if last_msg_index < len(response_contents):
            embed.description = response_contents[last_msg_index]
            embed.color = EMBED_COLOR_COMPLETE
            embed.set_footer(
                text=f"{provider_slash_model} | total tokens: {total_tokens:,}",
            )
            last_response_msg = response_msgs[last_msg_index]
            await last_response_msg.edit(embed=embed, view=response_view)
            try:
                get_bad_keys_db().save_message_response_data(
                    response_message_id=str(last_response_msg.id),
                    user_message_id=str(new_msg.id),
                    channel_id=str(new_msg.channel.id),
                    user_id=str(new_msg.author.id),
                    full_response=full_response,
                    grounding_metadata=grounding_metadata,
                    tavily_metadata=tavily_metadata,
                )
            except Exception:
                logger.exception(
                    "Failed to persist response data for message %s",
                    last_response_msg.id,
                )

    if use_plain_responses and response_msgs and response_contents:
        last_response_msg = response_msgs[-1]
        try:
            get_bad_keys_db().save_message_response_data(
                response_message_id=str(last_response_msg.id),
                user_message_id=str(new_msg.id),
                channel_id=str(new_msg.channel.id),
                user_id=str(new_msg.author.id),
                full_response=full_response,
                grounding_metadata=grounding_metadata,
                tavily_metadata=tavily_metadata,
            )
        except Exception:
            logger.exception(
                "Failed to persist response data for message %s",
                last_response_msg.id,
            )

    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        # Get keys to remove (oldest first based on insertion order)
        keys_to_remove = sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]
        for msg_id in keys_to_remove:
            node = msg_nodes.get(msg_id)
            if node is not None:
                async with node.lock:
                    msg_nodes.pop(msg_id, None)
