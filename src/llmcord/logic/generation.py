"""LLM response generation logic."""

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, cast

import discord
import httpx
import litellm

from llmcord.core.config import (
    MAX_MESSAGE_NODES,
    STREAMING_INDICATOR,
    is_gemini_model,
)
from llmcord.core.error_handling import log_exception
from llmcord.core.exceptions import (
    FIRST_TOKEN_TIMEOUT_SECONDS,
    LITELLM_TIMEOUT_SECONDS,
    FirstTokenTimeoutError,
    _raise_empty_response,
)
from llmcord.core.models import MsgNode
from llmcord.discord.ui import metadata as ui_metadata
from llmcord.discord.ui.embed_limits import sanitize_embed_kwargs
from llmcord.logic.discord_ui import (
    maybe_edit_stream_message,
    render_exhausted_response,
    update_response_view,
)
from llmcord.logic.fallbacks import apply_fallback_config, get_next_fallback
from llmcord.logic.generation_types import (
    FallbackState,
    GeneratedImage,
    GenerationContext,
    GenerationLoopState,
    GenerationState,
    StreamConfig,
    StreamEditDecision,
    StreamLoopState,
)
from llmcord.logic.images import (
    append_generated_images,
    extract_gemini_images_from_chunk,
    send_generated_images,
)
from llmcord.logic.utils import (
    count_conversation_tokens,
)
from llmcord.services.database import get_bad_keys_db
from llmcord.services.database.messages import MessageResponsePayload
from llmcord.services.llm import LiteLLMOptions, prepare_litellm_kwargs
from llmcord.services.llm.providers.gemini_cli import stream_google_gemini_cli

logger = logging.getLogger(__name__)


def _collect_litellm_exceptions() -> tuple[type[Exception], ...]:
    return tuple(
        dict.fromkeys(
            exception_type
            for exception_type in vars(litellm.exceptions).values()
            if isinstance(exception_type, type)
            and issubclass(exception_type, Exception)
        ),
    )


LITELLM_RETRYABLE_ERRORS = _collect_litellm_exceptions()

GENERATION_EXCEPTIONS = (
    FirstTokenTimeoutError,
    asyncio.TimeoutError,
    httpx.HTTPError,
    OSError,
    RuntimeError,
    ValueError,
    *LITELLM_RETRYABLE_ERRORS,
)


def _get_good_keys(provider: str, api_keys: list[str]) -> list[str]:
    try:
        return get_bad_keys_db().get_good_keys_synced(provider, api_keys)
    except (OSError, RuntimeError, ValueError) as exc:
        log_exception(
            logger=logger,
            message="Failed to get good keys, falling back to all keys",
            error=exc,
            context={"provider": provider, "configured_keys": len(api_keys)},
        )
        return api_keys.copy()


def _reset_provider_keys(provider: str, api_keys: list[str]) -> list[str]:
    logger.warning(
        ("All API keys for provider '%s' (synced) are marked as bad. Resetting..."),
        provider,
    )
    try:
        get_bad_keys_db().reset_provider_keys_synced(provider)
    except (OSError, RuntimeError, ValueError) as exc:
        log_exception(
            logger=logger,
            message="Failed to reset provider keys",
            error=exc,
            context={"provider": provider, "configured_keys": len(api_keys)},
        )
    return api_keys.copy()


def _format_display_model(provider: str, actual_model: str) -> str:
    if actual_model.startswith(f"{provider}/"):
        return actual_model
    return f"{provider}/{actual_model}"


async def _initialize_generation_state(
    *,
    context: GenerationContext,
) -> GenerationState:
    response_msgs = [context.processing_msg]
    processing_msg_id = context.processing_msg.id
    context.msg_nodes[processing_msg_id] = MsgNode(parent_msg=context.new_msg)
    await context.msg_nodes[processing_msg_id].lock.acquire()

    input_tokens = count_conversation_tokens(context.messages)
    max_message_length = 4096 - len(STREAMING_INDICATOR)
    embed = discord.Embed.from_dict(
        {
            "fields": [
                {"name": warning, "value": "", "inline": False}
                for warning in sorted(context.user_warnings)
            ],
        },
    )
    embed.set_footer(
        text=_format_display_model(
            context.provider,
            context.actual_model,
        ),
    )

    display_model = _format_display_model(
        context.provider,
        context.actual_model,
    )

    return GenerationState(
        response_msgs=response_msgs,
        response_contents=[],
        input_tokens=input_tokens,
        max_message_length=max_message_length,
        embed=embed,
        grounding_metadata=None,
        last_edit_time=context.last_edit_time,
        generated_images=[],
        generated_image_hashes=set(),
        display_model=display_model,
    )


async def _prune_response_messages(
    *,
    context: GenerationContext,
    state: GenerationState,
) -> None:
    if len(state.response_msgs) <= len(state.response_contents):
        return

    for msg in state.response_msgs[len(state.response_contents) :]:
        await msg.delete()
        if msg.id in context.msg_nodes:
            context.msg_nodes[msg.id].lock.release()
            del context.msg_nodes[msg.id]

    state.response_msgs = state.response_msgs[: len(state.response_contents)]


def _release_response_locks(
    *,
    context: GenerationContext,
    state: GenerationState,
) -> str:
    full_response = "".join(state.response_contents) if state.response_contents else ""
    history_response = state.full_history_response or full_response
    for response_msg in state.response_msgs:
        context.msg_nodes[response_msg.id].text = history_response
        context.msg_nodes[response_msg.id].lock.release()
    return full_response


def _build_grounding_payload(
    grounding_metadata: object | None,
) -> dict[str, object] | None:
    if not grounding_metadata or not ui_metadata.has_grounding_data(
        grounding_metadata,
    ):
        return None

    return {
        "web_search_queries": ui_metadata.get_grounding_queries(
            grounding_metadata,
        ),
        "grounding_chunks": [
            {
                "web": {
                    "title": chunk.get("title", ""),
                    "uri": chunk.get("uri", ""),
                },
            }
            for chunk in ui_metadata.get_grounding_chunks(grounding_metadata)
        ],
    }


async def _persist_response_payload(
    *,
    context: GenerationContext,
    state: GenerationState,
    full_response: str,
    grounding_payload: dict[str, object] | None,
) -> None:
    if not (state.response_msgs and state.response_contents):
        return

    last_msg_index = len(state.response_msgs) - 1
    if last_msg_index >= len(state.response_msgs):
        return

    try:
        payload = MessageResponsePayload(
            request_message_id=str(context.new_msg.id),
            request_user_id=str(context.new_msg.author.id),
            full_response=full_response,
            thought_process=state.thought_process or None,
            grounding_metadata=grounding_payload,
            tavily_metadata=context.tavily_metadata,
            failed_extractions=context.failed_extractions or None,
        )
        get_bad_keys_db().save_message_response_data(
            message_id=str(state.response_msgs[last_msg_index].id),
            payload=payload,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        log_exception(
            logger=logger,
            message="Failed to persist response data",
            error=exc,
            context={
                "request_message_id": context.new_msg.id,
                "response_message_id": state.response_msgs[last_msg_index].id,
            },
        )


async def _trim_message_nodes(context: GenerationContext) -> None:
    if (num_nodes := len(context.msg_nodes)) <= MAX_MESSAGE_NODES:
        return

    keys_to_remove = sorted(context.msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]
    for msg_id in keys_to_remove:
        node = context.msg_nodes.get(msg_id)
        if node is not None:
            async with node.lock:
                context.msg_nodes.pop(msg_id, None)


async def _finalize_response(
    *,
    context: GenerationContext,
    state: GenerationState,
    grounding_metadata: object | None,
) -> None:
    await _prune_response_messages(context=context, state=state)
    full_response = _release_response_locks(context=context, state=state)
    grounding_payload = _build_grounding_payload(grounding_metadata)

    await _persist_response_payload(
        context=context,
        state=state,
        full_response=full_response,
        grounding_payload=grounding_payload,
    )
    await update_response_view(
        context=context,
        state=state,
        full_response=full_response,
        grounding_metadata=grounding_metadata,
    )
    await send_generated_images(context=context, state=state)
    await _trim_message_nodes(context)


def _extract_grounding_metadata(
    response_obj: object,
    choice_obj: object | None = None,
) -> object | None:
    grounding_metadata = None

    model_extra = getattr(response_obj, "model_extra", None)
    if isinstance(model_extra, Mapping) and model_extra:
        grounding_metadata = (
            model_extra.get("vertex_ai_grounding_metadata")
            or model_extra.get("google_grounding_metadata")
            or model_extra.get("grounding_metadata")
            or model_extra.get("groundingMetadata")
        )

    if not grounding_metadata and hasattr(response_obj, "grounding_metadata"):
        grounding_metadata = response_obj.grounding_metadata

    hidden_params = getattr(response_obj, "_hidden_params", None)
    if not grounding_metadata and isinstance(hidden_params, Mapping):
        grounding_metadata = (
            hidden_params.get("grounding_metadata")
            or hidden_params.get("google_grounding_metadata")
            or hidden_params.get("groundingMetadata")
        )

    if (
        not grounding_metadata
        and choice_obj
        and hasattr(choice_obj, "grounding_metadata")
    ):
        grounding_metadata = choice_obj.grounding_metadata

    return grounding_metadata


async def _iter_stream_with_first_chunk(
    stream_iter: AsyncIterator[object],
    *,
    timeout_seconds: int,
) -> AsyncIterator[object]:
    try:
        first_chunk = await asyncio.wait_for(
            stream_iter.__anext__(),
            timeout=timeout_seconds,
        )
    except StopAsyncIteration:
        return
    except TimeoutError as exc:
        raise FirstTokenTimeoutError(timeout_seconds=timeout_seconds) from exc

    yield first_chunk
    async for chunk in stream_iter:
        yield chunk


async def _get_stream(
    *,
    context: GenerationContext,
    stream_config: StreamConfig,
) -> AsyncIterator[
    tuple[str, object | None, object | None, list[GeneratedImage], bool]
]:
    """Yield stream chunks from LiteLLM with grounding metadata."""
    if stream_config.provider == "google-gemini-cli":
        stream = stream_google_gemini_cli(
            model=stream_config.actual_model,
            messages=context.messages[::-1],
            api_key=stream_config.api_key,
            base_url=stream_config.base_url,
            extra_headers=stream_config.extra_headers,
            model_parameters=stream_config.model_parameters,
        )
        async for chunk in _iter_stream_with_first_chunk(
            stream,
            timeout_seconds=FIRST_TOKEN_TIMEOUT_SECONDS,
        ):
            delta_content, chunk_finish_reason, is_thinking = cast(
                "tuple[str, object | None, bool]",
                chunk,
            )
            yield (
                delta_content,
                chunk_finish_reason,
                None,
                [],
                is_thinking,
            )
        return

    enable_grounding = not re.search(r"https?://", context.new_msg.content)

    if not is_gemini_model(stream_config.actual_model):
        removed_audio_video_files = _remove_audio_video_file_parts_from_messages(
            context.messages,
        )
        if removed_audio_video_files:
            logger.info(
                "Removed audio/video file parts from messages for non-Gemini model %s",
                stream_config.actual_model,
            )

    litellm_kwargs = prepare_litellm_kwargs(
        provider=stream_config.provider,
        model=stream_config.actual_model,
        messages=context.messages[::-1],
        api_key=stream_config.api_key,
        options=LiteLLMOptions(
            base_url=stream_config.base_url,
            extra_headers=stream_config.extra_headers,
            stream=True,
            model_parameters=stream_config.model_parameters,
            enable_grounding=enable_grounding,
        ),
    )

    litellm_kwargs["timeout"] = LITELLM_TIMEOUT_SECONDS

    logger.debug("\n--- LLM REQUEST ---")
    logger.debug("Model: %s", litellm_kwargs.get("model"))
    logger.debug(
        "Messages:\n%s",
        json.dumps(litellm_kwargs.get("messages"), indent=2),
    )
    logger.debug("-------------------\n")

    stream = await litellm.acompletion(**litellm_kwargs)

    async for chunk in _iter_stream_with_first_chunk(
        stream,
        timeout_seconds=FIRST_TOKEN_TIMEOUT_SECONDS,
    ):
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue

        choice = choices[0]
        delta_content = getattr(choice.delta, "content", "") or ""
        chunk_finish_reason = getattr(choice, "finish_reason", None)
        grounding_metadata = _extract_grounding_metadata(chunk, choice)
        image_payloads = extract_gemini_images_from_chunk(
            chunk,
            choice,
            delta_content,
        )

        if chunk_finish_reason and is_gemini_model(stream_config.actual_model):
            chunk_attrs = [attr for attr in dir(chunk) if not attr.startswith("_")]
            logger.debug("Gemini chunk finish - attributes: %s", chunk_attrs)
            chunk_model_extra = getattr(chunk, "model_extra", None)
            if isinstance(chunk_model_extra, Mapping) and chunk_model_extra:
                logger.info(
                    "Gemini chunk model_extra keys: %s",
                    list(chunk_model_extra.keys()),
                )
            hidden_params = getattr(chunk, "_hidden_params", None)
            if isinstance(hidden_params, Mapping):
                logger.info(
                    "Gemini chunk _hidden_params keys: %s",
                    list(hidden_params.keys()),
                )

        yield (
            delta_content,
            chunk_finish_reason,
            grounding_metadata,
            image_payloads,
            False,
        )


def _append_stream_content(
    *,
    response_contents: list[str],
    prev_content: str | None,
    finish_reason: object | None,
    delta_content: str,
    max_message_length: int,
) -> StreamEditDecision | None:
    previous = prev_content or ""
    new_content = previous if finish_reason is None else previous + delta_content

    if response_contents == [] and new_content == "":
        return None

    start_next_msg = response_contents == [] or (
        len(response_contents[-1] + new_content) > max_message_length
    )
    if start_next_msg:
        response_contents.append("")

    response_contents[-1] += new_content

    msg_split_incoming = finish_reason is None and (
        len(response_contents[-1] + delta_content) > max_message_length
    )
    is_final_edit = finish_reason is not None or msg_split_incoming
    is_good_finish = finish_reason is not None and any(
        x in str(finish_reason).lower() for x in ("stop", "end_turn")
    )

    return StreamEditDecision(
        start_next_msg=start_next_msg,
        msg_split_incoming=msg_split_incoming,
        is_final_edit=is_final_edit,
        is_good_finish=is_good_finish,
    )


def _append_history_content(
    *,
    history_parts: list[str],
    delta_content: str,
    is_thinking: bool,
    in_thinking_block: bool,
) -> bool:
    if is_thinking and delta_content:
        if not in_thinking_block:
            history_parts.append("\n<thinking>\n")
            in_thinking_block = True
        history_parts.append(delta_content)
        return in_thinking_block

    if in_thinking_block:
        history_parts.append("\n</thinking>\n")
        in_thinking_block = False
    if delta_content:
        history_parts.append(delta_content)
    return in_thinking_block


def _is_developer_instruction_error(error: Exception) -> bool:
    error_str = str(error).lower()
    return "developer instruction" in error_str and "not enabled" in error_str


def _is_image_input_error(error: Exception) -> bool:
    error_str = str(error).lower()
    image_error_patterns = (
        "image input",
        "support image",
        "no endpoints found",
        "image_url",
        "unsupported image",
    )
    return any(pattern in error_str for pattern in image_error_patterns)


def _remove_system_messages(messages: list[dict[str, object]]) -> bool:
    original_len = len(messages)
    messages[:] = [
        message
        for message in messages
        if str(message.get("role") or "").lower() != "system"
    ]
    return len(messages) != original_len


def _remove_images_from_messages(messages: list[dict[str, object]]) -> bool:
    images_removed = False

    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue

        filtered_content: list[object] = []
        for part in content:
            if not isinstance(part, dict):
                filtered_content.append(part)
                continue

            part_dict = cast("dict[str, object]", part)

            if part_dict.get("type") == "image_url":
                images_removed = True
                continue

            filtered_content.append(part_dict)

        if filtered_content:
            message["content"] = filtered_content
            continue

        message["content"] = [
            {
                "type": "text",
                "text": "Image input was removed due to provider error.",
            },
        ]

    return images_removed


def _is_audio_or_video_file_part(part: dict[str, object]) -> bool:
    if part.get("type") != "file":
        return False

    file_info_obj = part.get("file")
    if not isinstance(file_info_obj, dict):
        return False
    file_info = cast("dict[str, object]", file_info_obj)

    file_data = file_info.get("file_data")
    if not isinstance(file_data, str):
        return False

    return file_data.startswith(("data:audio/", "data:video/"))


def _remove_audio_video_file_parts_from_messages(
    messages: list[dict[str, object]],
) -> bool:
    removed_any = False

    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue

        filtered_content: list[object] = []
        for part in content:
            if not isinstance(part, dict):
                filtered_content.append(part)
                continue

            part_dict = cast("dict[str, object]", part)
            if _is_audio_or_video_file_part(part_dict):
                removed_any = True
                continue

            filtered_content.append(part_dict)

        if filtered_content:
            message["content"] = filtered_content
            continue

        message["content"] = [
            {
                "type": "text",
                "text": (
                    "Audio/video attachment omitted because this model does not "
                    "support native file input."
                ),
            },
        ]

    return removed_any


def _handle_generation_exception(
    *,
    error: Exception,
    provider: str,
    current_api_key: str,
    good_keys: list[str],
) -> tuple[list[str], str]:
    last_error_msg = str(error)
    log_exception(
        logger=logger,
        message="Error while generating response",
        error=error,
        context={"provider": provider},
    )

    is_first_token_timeout = isinstance(error, FirstTokenTimeoutError)
    is_timeout_error = isinstance(error, asyncio.TimeoutError) or (
        "timeout" in str(error).lower()
    )
    is_empty_response = "no content" in str(error).lower()

    special_case_message: str | None = None
    if is_first_token_timeout:
        special_case_message = (
            "No first token within %s seconds for provider '%s'; "
            "removing key and trying remaining keys."
        )
    elif is_empty_response:
        special_case_message = (
            "Empty response for provider '%s'; removing key and trying remaining keys."
        )

    error_str = str(error).lower()
    is_developer_instruction_error = _is_developer_instruction_error(error)
    key_error_patterns = [
        "unauthorized",
        "invalid_api_key",
        "invalid key",
        "api key",
        "authentication",
        "forbidden",
        "401",
        "403",
        "quota",
        "rate limit",
        "billing",
        "insufficient_quota",
        "expired",
    ]
    is_key_error = any(pattern in error_str for pattern in key_error_patterns)

    if is_developer_instruction_error:
        special_case_message = (
            "Developer instructions unsupported for provider '%s'; "
            "removing key and trying remaining keys."
        )

    if special_case_message:
        if is_first_token_timeout:
            timeout_seconds = getattr(error, "timeout_seconds", None)
            if not isinstance(timeout_seconds, int):
                timeout_seconds = FIRST_TOKEN_TIMEOUT_SECONDS
            logger.warning(
                special_case_message,
                timeout_seconds,
                provider,
            )
        else:
            logger.warning(special_case_message, provider)
        _remove_key(good_keys, current_api_key)
        return good_keys, last_error_msg

    _handle_key_rotation_error(
        error=error,
        provider=provider,
        current_api_key=current_api_key,
        good_keys=good_keys,
        error_flags=(is_key_error, is_timeout_error),
    )

    return good_keys, last_error_msg


def _remove_key(good_keys: list[str], current_api_key: str) -> None:
    if current_api_key in good_keys:
        good_keys.remove(current_api_key)


def _handle_key_rotation_error(
    *,
    error: Exception,
    provider: str,
    current_api_key: str,
    good_keys: list[str],
    error_flags: tuple[bool, bool],
) -> None:
    is_key_error, is_timeout_error = error_flags
    if is_key_error:
        error_msg = str(error)[:200] if error else "Unknown error"
        try:
            get_bad_keys_db().mark_key_bad_synced(
                provider,
                current_api_key,
                error_msg,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            log_exception(
                logger=logger,
                message="Failed to mark key as bad",
                error=exc,
                context={"provider": provider},
            )

        _remove_key(good_keys, current_api_key)
        logger.info(
            "Removed bad key for '%s', %s keys remaining",
            provider,
            len(good_keys),
        )
        return

    if is_timeout_error:
        _remove_key(good_keys, current_api_key)
        logger.warning(
            "Non-key error for provider '%s': %s, %s keys remaining",
            provider,
            "timeout",
            len(good_keys),
        )
        return

    _remove_key(good_keys, current_api_key)
    logger.warning(
        "Unknown error for provider '%s', %s keys remaining",
        provider,
        len(good_keys),
    )


def _initialize_loop_state(context: GenerationContext) -> GenerationLoopState:
    provider = context.provider
    actual_model = context.actual_model
    fallback_chain = context.fallback_chain or []
    use_custom_fallbacks = context.fallback_chain is not None

    fallback_state = FallbackState(
        fallback_level=0,
        fallback_index=0,
        use_custom_fallbacks=use_custom_fallbacks,
        original_provider=provider,
        original_model=actual_model,
    )

    good_keys = _get_good_keys(provider, context.api_keys)
    if not good_keys:
        good_keys = _reset_provider_keys(provider, context.api_keys)

    return GenerationLoopState(
        provider=provider,
        actual_model=actual_model,
        base_url=context.base_url,
        api_keys=context.api_keys,
        good_keys=good_keys,
        initial_key_count=len(good_keys),
        attempt_count=0,
        last_error_msg=None,
        fallback_state=fallback_state,
        fallback_chain=fallback_chain,
    )


async def _handle_exhausted_keys(
    *,
    context: GenerationContext,
    state: GenerationState,
    loop_state: GenerationLoopState,
    reply_helper: Callable[..., Awaitable[None]],
) -> bool:
    next_fallback = get_next_fallback(
        state=loop_state.fallback_state,
        fallback_chain=loop_state.fallback_chain,
        provider=loop_state.provider,
        initial_key_count=loop_state.initial_key_count,
    )

    if next_fallback:
        previous_display_model = state.display_model
        (
            loop_state.provider,
            loop_state.actual_model,
            _provider_slash_model,
            loop_state.base_url,
            fallback_api_keys,
        ) = apply_fallback_config(
            next_fallback=next_fallback,
            config=context.config,
        )

        state.display_model = _format_display_model(
            loop_state.provider,
            loop_state.actual_model,
        )
        if state.display_model != previous_display_model:
            state.fallback_warning = (
                f"⚠️ Switched to {state.display_model} because of errors"
            )
        if state.embed is not None:
            state.embed.set_footer(text=state.display_model)

        if fallback_api_keys:
            loop_state.api_keys = fallback_api_keys
            loop_state.good_keys = fallback_api_keys.copy()
            loop_state.initial_key_count = len(loop_state.good_keys)
            loop_state.attempt_count = 0
            return True

        logger.error(
            "No API keys available for fallback provider '%s'",
            loop_state.provider,
        )
        loop_state.good_keys = []
        return True

    state.response_contents = await render_exhausted_response(
        state=state,
        reply_helper=reply_helper,
        last_error_msg=loop_state.last_error_msg,
        fallback_state=loop_state.fallback_state,
    )
    return False


async def _stream_response(
    *,
    context: GenerationContext,
    state: GenerationState,
    stream_config: StreamConfig,
    reply_helper: Callable[..., Awaitable[None]],
) -> None:
    response_contents = state.response_contents
    max_message_length = state.max_message_length
    grounding_metadata = state.grounding_metadata
    loop_state = StreamLoopState(curr_content=None, finish_reason=None)
    history_parts: list[str] = []
    in_thinking_block = False

    async with context.new_msg.channel.typing():
        async for (
            delta_content,
            new_finish_reason,
            new_grounding_metadata,
            image_payloads,
            is_thinking,
        ) in _get_stream(
            context=context,
            stream_config=stream_config,
        ):
            if new_grounding_metadata:
                grounding_metadata = new_grounding_metadata
                logger.info(
                    "Captured grounding metadata from stream: %s",
                    type(grounding_metadata),
                )

            if image_payloads:
                append_generated_images(state, image_payloads)

            if loop_state.finish_reason is not None:
                break

            in_thinking_block = _append_history_content(
                history_parts=history_parts,
                delta_content=delta_content,
                is_thinking=is_thinking,
                in_thinking_block=in_thinking_block,
            )
            if is_thinking and delta_content:
                state.thought_process += delta_content

            loop_state.finish_reason = new_finish_reason

            if is_thinking:
                continue

            decision = _append_stream_content(
                response_contents=response_contents,
                prev_content=loop_state.curr_content,
                finish_reason=loop_state.finish_reason,
                delta_content=delta_content,
                max_message_length=max_message_length,
            )
            loop_state.curr_content = delta_content

            if decision is None:
                continue

            await maybe_edit_stream_message(
                context=context,
                state=state,
                reply_helper=reply_helper,
                decision=decision,
                grounding_metadata=grounding_metadata,
            )

    if in_thinking_block:
        history_parts.append("\n</thinking>\n")
    state.full_history_response = "".join(history_parts)

    if not response_contents:
        _raise_empty_response()

    state.grounding_metadata = grounding_metadata


async def _run_generation_loop(
    *,
    context: GenerationContext,
    state: GenerationState,
) -> None:
    loop_state = _initialize_loop_state(context)
    response_msgs = state.response_msgs

    async def reply_helper(**reply_kwargs: Any) -> None:  # noqa: ANN401
        reply_target = context.new_msg if not response_msgs else response_msgs[-1]
        sanitized_reply_kwargs = cast(
            "dict[str, Any]",
            sanitize_embed_kwargs(reply_kwargs),
        )
        response_msg = await reply_target.reply(
            **sanitized_reply_kwargs,
        )
        response_msgs.append(response_msg)

        context.msg_nodes[response_msg.id] = MsgNode(parent_msg=context.new_msg)
        await context.msg_nodes[response_msg.id].lock.acquire()

    while True:
        state.response_contents = []
        state.thought_process = ""
        state.full_history_response = ""
        loop_state.attempt_count += 1

        if not loop_state.good_keys:
            should_continue = await _handle_exhausted_keys(
                context=context,
                state=state,
                loop_state=loop_state,
                reply_helper=reply_helper,
            )
            if should_continue:
                continue
            break

        current_api_key = loop_state.good_keys[
            (loop_state.attempt_count - 1) % len(loop_state.good_keys)
        ]

        try:
            stream_config = StreamConfig(
                provider=loop_state.provider,
                actual_model=loop_state.actual_model,
                api_key=current_api_key,
                base_url=loop_state.base_url,
                extra_headers=context.extra_headers,
                model_parameters=context.model_parameters,
            )
            await _stream_response(
                context=context,
                state=state,
                stream_config=stream_config,
                reply_helper=reply_helper,
            )
            break
        except GENERATION_EXCEPTIONS as exc:
            if (
                not loop_state.developer_instruction_removed
                and _is_developer_instruction_error(cast("Exception", exc))
                and _remove_system_messages(context.messages)
            ):
                loop_state.developer_instruction_removed = True
                loop_state.last_error_msg = str(exc)
                loop_state.attempt_count -= 1
                logger.warning(
                    "Retrying without system prompt after developer instruction "
                    "provider error: %s",
                    exc,
                )
                continue

            if (
                not loop_state.image_input_removed
                and _is_image_input_error(cast("Exception", exc))
                and _remove_images_from_messages(context.messages)
            ):
                loop_state.image_input_removed = True
                loop_state.last_error_msg = str(exc)
                state.image_removal_warning = (
                    "⚠️ Image removed from input due to provider error."
                )
                loop_state.attempt_count -= 1
                logger.warning(
                    "Retrying without image input after provider error: %s",
                    exc,
                )
                continue

            (
                loop_state.good_keys,
                loop_state.last_error_msg,
            ) = _handle_generation_exception(
                error=cast("Exception", exc),
                provider=loop_state.provider,
                current_api_key=current_api_key,
                good_keys=loop_state.good_keys,
            )


async def generate_response(
    context: GenerationContext,
) -> None:
    """Generate and stream the LLM response using LiteLLM."""
    state = await _initialize_generation_state(context=context)
    await _run_generation_loop(context=context, state=state)

    await _finalize_response(
        context=context,
        state=state,
        grounding_metadata=state.grounding_metadata,
    )
