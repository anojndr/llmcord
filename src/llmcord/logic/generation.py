"""LLM response generation logic."""
import asyncio
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone

import discord
import litellm

from llmcord.core.config import (
    EDIT_DELAY_SECONDS,
    EMBED_COLOR_COMPLETE,
    EMBED_COLOR_INCOMPLETE,
    MAX_MESSAGE_NODES,
    STREAMING_INDICATOR,
    ensure_list,
    is_gemini_model,
)
from llmcord.core.exceptions import (
    FIRST_TOKEN_TIMEOUT_SECONDS,
    LITELLM_TIMEOUT_SECONDS,
    FirstTokenTimeoutError,
    _raise_empty_response,
)
from llmcord.core.models import MsgNode
from llmcord.discord.ui.metadata import (
    get_grounding_chunks,
    get_grounding_queries,
    has_grounding_data,
)
from llmcord.discord.ui.response_view import LayoutView, ResponseView, TextDisplay
from llmcord.discord.ui.sources_view import SourceButton, SourceView, TavilySourceButton
from llmcord.logic.utils import (
    count_conversation_tokens,
    count_text_tokens,
)
from llmcord.services.database import get_bad_keys_db
from llmcord.services.database.messages import MessageResponsePayload
from llmcord.services.llm import LiteLLMOptions, prepare_litellm_kwargs

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GenerationContext:
    """Inputs required to generate an LLM response."""

    new_msg: discord.Message
    discord_bot: discord.Client
    msg_nodes: dict[int, MsgNode]
    messages: list[dict[str, object]]
    user_warnings: set[str]
    provider: str
    model: str
    actual_model: str
    provider_slash_model: str
    base_url: str | None
    api_keys: list[str]
    model_parameters: dict[str, object] | None
    extra_headers: dict[str, str] | None
    extra_query: dict[str, object] | None
    extra_body: dict[str, object] | None
    system_prompt: str | None
    config: dict[str, object]
    max_text: int
    tavily_metadata: dict[str, object] | None
    last_edit_time: float
    processing_msg: discord.Message
    retry_callback: Callable[[], Awaitable[None]]
    fallback_chain: list[tuple[str, str, str]] | None = None


@dataclass(slots=True)
class GenerationState:
    """Mutable state for response generation."""

    response_msgs: list[discord.Message]
    response_contents: list[str]
    input_tokens: int
    max_message_length: int
    embed: discord.Embed | None
    use_plain_responses: bool
    grounding_metadata: object | None
    last_edit_time: float


@dataclass(slots=True)
class FallbackState:
    """Track fallback selection state."""

    fallback_level: int
    fallback_index: int
    use_custom_fallbacks: bool
    is_original_mistral: bool
    original_provider: str
    original_model: str


@dataclass(slots=True)
class StreamConfig:
    """Configuration for a streaming LLM request."""

    provider: str
    actual_model: str
    api_key: str
    base_url: str | None
    extra_headers: dict[str, str] | None
    model_parameters: dict[str, object] | None


@dataclass(slots=True)
class StreamLoopState:
    """Mutable state for stream processing."""

    curr_content: str | None
    finish_reason: object | None


@dataclass(slots=True)
class StreamEditDecision:
    """Decisions for streaming edits."""

    start_next_msg: bool
    msg_split_incoming: bool
    is_final_edit: bool
    is_good_finish: bool


@dataclass(slots=True)
class GenerationLoopState:
    """Mutable state for generation loop."""

    provider: str
    actual_model: str
    base_url: str | None
    api_keys: list[str]
    good_keys: list[str]
    initial_key_count: int
    attempt_count: int
    last_error_msg: str | None
    fallback_state: FallbackState
    fallback_chain: list[tuple[str, str, str]]


def _get_good_keys(provider: str, api_keys: list[str]) -> list[str]:
    try:
        return get_bad_keys_db().get_good_keys_synced(provider, api_keys)
    except (OSError, RuntimeError, ValueError):
        logger.exception("Failed to get good keys, falling back to all keys")
        return api_keys.copy()


def _reset_provider_keys(provider: str, api_keys: list[str]) -> list[str]:
    logger.warning(
        "All API keys for provider '%s' (synced) are marked as bad. Resetting...",
        provider,
    )
    try:
        get_bad_keys_db().reset_provider_keys_synced(provider)
    except (OSError, RuntimeError, ValueError):
        logger.exception("Failed to reset provider keys")
    return api_keys.copy()


def _get_next_fallback(
    *,
    state: FallbackState,
    fallback_chain: list[tuple[str, str, str]],
    provider: str,
    initial_key_count: int,
) -> tuple[str, str, str] | None:
    if state.use_custom_fallbacks:
        if state.fallback_index < len(fallback_chain):
            next_fallback = fallback_chain[state.fallback_index]
            state.fallback_index += 1
            logger.warning(
                "All %s keys exhausted for provider '%s'. Falling back to %s...",
                initial_key_count,
                provider,
                next_fallback[2],
            )
            return next_fallback
        return None

    if state.fallback_level == 0:
        if state.is_original_mistral:
            state.fallback_level = 2
            next_fallback = (
                "gemini",
                "gemma-3-27b-it",
                "gemini/gemma-3-27b-it",
            )
            logger.warning(
                "All %s keys exhausted for mistral/%s. Falling back to gemini/"
                "gemma-3-27b-it...",
                initial_key_count,
                state.original_model,
            )
            return next_fallback

        state.fallback_level = 1
        next_fallback = (
            "mistral",
            "mistral-large-latest",
            "mistral/mistral-large-latest",
        )
        logger.warning(
            "All %s keys exhausted for provider '%s'. Falling back to "
            "mistral/mistral-large-latest...",
            initial_key_count,
            state.original_provider,
        )
        return next_fallback

    if state.fallback_level == 1:
        state.fallback_level = 2
        next_fallback = (
            "gemini",
            "gemma-3-27b-it",
            "gemini/gemma-3-27b-it",
        )
        logger.warning(
            "Mistral fallback also failed. Falling back to gemini/gemma-3-27b-it...",
        )
        return next_fallback

    return None


def _apply_fallback_config(
    *,
    next_fallback: tuple[str, str, str],
    config: dict[str, object],
) -> tuple[str, str, str, str | None, list[str]]:
    provider, model, provider_slash_model = next_fallback
    provider_config = config.get("providers", {}).get(provider, {})
    base_url = provider_config.get("base_url")
    api_keys = ensure_list(provider_config.get("api_key"))
    return provider, model, provider_slash_model, base_url, api_keys


async def _render_exhausted_response(
    *,
    state: GenerationState,
    reply_helper: Callable[..., Awaitable[None]],
    last_error_msg: str | None,
    fallback_state: FallbackState,
) -> list[str]:
    if fallback_state.use_custom_fallbacks:
        logger.error("All custom fallback options exhausted")
    else:
        logger.error("All fallback options exhausted (mistral and gemma)")

    error_text = (
        "❌ All API keys (including all fallbacks) exhausted. "
        "Please try again later."
    )
    if last_error_msg:
        error_text += f"\nLast error: {last_error_msg}"

    if state.use_plain_responses:
        layout = LayoutView().add_item(TextDisplay(content=error_text))
        if state.response_msgs:
            await state.response_msgs[-1].edit(view=layout)
        else:
            await reply_helper(view=layout)
        return [error_text]

    if state.embed is None:
        return [error_text]

    state.embed.description = error_text
    state.embed.color = EMBED_COLOR_INCOMPLETE
    if state.response_msgs:
        await state.response_msgs[-1].edit(embed=state.embed, view=None)
    else:
        await reply_helper(embed=state.embed)
    return [error_text]


async def _initialize_generation_state(
    *,
    context: GenerationContext,
) -> GenerationState:
    response_msgs = [context.processing_msg]
    context.msg_nodes[context.processing_msg.id] = MsgNode(parent_msg=context.new_msg)
    await context.msg_nodes[context.processing_msg.id].lock.acquire()

    input_tokens = count_conversation_tokens(context.messages)
    use_plain_responses = context.config.get("use_plain_responses", False)

    if use_plain_responses:
        max_message_length = 4000
        embed = None
    else:
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
            text=(
                f"{context.provider_slash_model} | total tokens: {input_tokens:,}"
            ),
        )

    return GenerationState(
        response_msgs=response_msgs,
        response_contents=[],
        input_tokens=input_tokens,
        max_message_length=max_message_length,
        embed=embed,
        use_plain_responses=use_plain_responses,
        grounding_metadata=None,
        last_edit_time=context.last_edit_time,
    )


async def _prune_response_messages(
    *,
    context: GenerationContext,
    state: GenerationState,
) -> None:
    if state.use_plain_responses:
        return

    if len(state.response_msgs) <= len(state.response_contents):
        return

    for msg in state.response_msgs[len(state.response_contents):]:
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
    for response_msg in state.response_msgs:
        context.msg_nodes[response_msg.id].text = full_response
        context.msg_nodes[response_msg.id].lock.release()
    return full_response


def _build_grounding_payload(
    grounding_metadata: object | None,
) -> dict[str, object] | None:
    if not grounding_metadata or not has_grounding_data(grounding_metadata):
        return None

    return {
        "web_search_queries": get_grounding_queries(grounding_metadata),
        "grounding_chunks": [
            {"web": {"title": chunk.get("title", ""), "uri": chunk.get("uri", "")}}
            for chunk in get_grounding_chunks(grounding_metadata)
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
            grounding_metadata=grounding_payload,
            tavily_metadata=context.tavily_metadata,
        )
        get_bad_keys_db().save_message_response_data(
            message_id=str(state.response_msgs[last_msg_index].id),
            payload=payload,
        )
    except (OSError, RuntimeError, ValueError):
        logger.exception("Failed to persist response data")


async def _update_response_view(
    *,
    context: GenerationContext,
    state: GenerationState,
    full_response: str,
    grounding_metadata: object | None,
) -> None:
    if (
        state.use_plain_responses
        or not state.response_msgs
        or not state.response_contents
    ):
        return

    response_view = ResponseView(
        full_response,
        grounding_metadata,
        context.tavily_metadata,
        context.retry_callback,
        context.new_msg.author.id,
    )

    output_tokens = count_text_tokens(full_response)
    total_tokens = state.input_tokens + output_tokens
    last_msg_index = len(state.response_msgs) - 1
    if last_msg_index < len(state.response_contents) and state.embed:
        state.embed.description = state.response_contents[last_msg_index]
        state.embed.color = EMBED_COLOR_COMPLETE
        state.embed.set_footer(
            text=(
                f"{context.provider_slash_model} | total tokens: {total_tokens:,}"
            ),
        )
        await state.response_msgs[last_msg_index].edit(
            embed=state.embed,
            view=response_view,
        )


async def _trim_message_nodes(context: GenerationContext) -> None:
    if (num_nodes := len(context.msg_nodes)) <= MAX_MESSAGE_NODES:
        return

    keys_to_remove = sorted(context.msg_nodes.keys())[
        : num_nodes - MAX_MESSAGE_NODES
    ]
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
    await _update_response_view(
        context=context,
        state=state,
        full_response=full_response,
        grounding_metadata=grounding_metadata,
    )
    await _trim_message_nodes(context)


def _extract_grounding_metadata(
    response_obj: object,
    choice_obj: object | None = None,
) -> object | None:
    grounding_metadata = None

    if hasattr(response_obj, "model_extra") and response_obj.model_extra:
        grounding_metadata = (
            response_obj.model_extra.get("vertex_ai_grounding_metadata")
            or response_obj.model_extra.get("google_grounding_metadata")
            or response_obj.model_extra.get("grounding_metadata")
            or response_obj.model_extra.get("groundingMetadata")
        )

    if not grounding_metadata and hasattr(response_obj, "grounding_metadata"):
        grounding_metadata = response_obj.grounding_metadata

    hidden_params = getattr(response_obj, "_hidden_params", None)
    if not grounding_metadata and hidden_params:
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


def _split_response_content(text: str, max_length: int) -> list[str]:
    if not text:
        return []
    return [text[i : i + max_length] for i in range(0, len(text), max_length)]


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
    except asyncio.TimeoutError as exc:
        raise FirstTokenTimeoutError from exc

    yield first_chunk
    async for chunk in stream_iter:
        yield chunk


async def _get_stream(
    *,
    context: GenerationContext,
    stream_config: StreamConfig,
) -> AsyncIterator[tuple[str, object | None, object | None]]:
    """Yield stream chunks from LiteLLM with grounding metadata."""
    enable_grounding = not re.search(r"https?://", context.new_msg.content)

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
    stream = await litellm.acompletion(**litellm_kwargs)

    async for chunk in _iter_stream_with_first_chunk(
        stream,
        timeout_seconds=FIRST_TOKEN_TIMEOUT_SECONDS,
    ):
        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        delta_content = choice.delta.content or ""
        chunk_finish_reason = choice.finish_reason
        grounding_metadata = _extract_grounding_metadata(chunk, choice)

        if chunk_finish_reason and is_gemini_model(stream_config.actual_model):
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


def _append_stream_content(
    *,
    response_contents: list[str],
    prev_content: str | None,
    finish_reason: object | None,
    delta_content: str,
    max_message_length: int,
) -> StreamEditDecision | None:
    previous = prev_content or ""
    new_content = previous if finish_reason is None else (previous + delta_content)

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


async def _maybe_edit_stream_message(
    *,
    state: GenerationState,
    reply_helper: Callable[..., Awaitable[None]],
    decision: StreamEditDecision,
    grounding_metadata: object | None,
) -> None:
    response_contents = state.response_contents
    response_msgs = state.response_msgs
    embed = state.embed
    last_edit_time = state.last_edit_time

    if embed is None:
        return

    time_delta = datetime.now(timezone.utc).timestamp() - last_edit_time
    ready_to_edit = time_delta >= EDIT_DELAY_SECONDS

    if not (decision.start_next_msg or ready_to_edit or decision.is_final_edit):
        return

    embed.description = (
        response_contents[-1]
        if decision.is_final_edit
        else (response_contents[-1] + STREAMING_INDICATOR)
    )
    embed.color = (
        EMBED_COLOR_COMPLETE
        if decision.msg_split_incoming or decision.is_good_finish
        else EMBED_COLOR_INCOMPLETE
    )

    view = (
        SourceView(grounding_metadata)
        if decision.is_final_edit and has_grounding_data(grounding_metadata)
        else None
    )

    msg_index = len(response_contents) - 1
    if decision.start_next_msg:
        if msg_index < len(response_msgs):
            await response_msgs[msg_index].edit(embed=embed, view=view)
        else:
            await reply_helper(embed=embed, silent=True, view=view)
    else:
        await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
        await response_msgs[msg_index].edit(embed=embed, view=view)
    state.last_edit_time = datetime.now(timezone.utc).timestamp()


async def _render_plain_responses(
    *,
    response_contents: list[str],
    response_msgs: list[discord.Message],
    reply_helper: Callable[..., Awaitable[None]],
    grounding_metadata: object | None,
    tavily_metadata: dict[str, object] | None,
) -> None:
    for i, content in enumerate(response_contents):
        layout = LayoutView().add_item(TextDisplay(content=content))

        if i == len(response_contents) - 1:
            if has_grounding_data(grounding_metadata):
                layout.add_item(SourceButton(grounding_metadata))
            if tavily_metadata and (
                tavily_metadata.get("urls") or tavily_metadata.get("queries")
            ):
                layout.add_item(TavilySourceButton(tavily_metadata))

        if i < len(response_msgs):
            await response_msgs[i].edit(view=layout)
        else:
            await reply_helper(view=layout)


def _handle_generation_exception(
    *,
    error: Exception,
    provider: str,
    current_api_key: str,
    good_keys: list[str],
) -> tuple[list[str], str]:
    last_error_msg = str(error)
    logger.exception("Error while generating response")

    is_first_token_timeout = isinstance(error, FirstTokenTimeoutError)
    is_timeout_error = (
        isinstance(error, asyncio.TimeoutError) or "timeout" in str(error).lower()
    )
    is_empty_response = "no content" in str(error).lower()

    if is_first_token_timeout:
        logger.warning(
            "No first token within %s seconds for provider '%s'; "
            "forcing fallback model.",
            FIRST_TOKEN_TIMEOUT_SECONDS,
            provider,
        )
        return [], last_error_msg

    if is_empty_response:
        logger.warning(
            "Empty response for provider '%s'; forcing fallback model.",
            provider,
        )
        return [], last_error_msg

    error_str = str(error).lower()
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

    if is_key_error:
        error_msg = str(error)[:200] if error else "Unknown error"
        try:
            get_bad_keys_db().mark_key_bad_synced(
                provider,
                current_api_key,
                error_msg,
            )
        except Exception:
            logger.exception("Failed to mark key as bad")

        if current_api_key in good_keys:
            good_keys.remove(current_api_key)
            logger.info(
                "Removed bad key for '%s', %s keys remaining",
                provider,
                len(good_keys),
            )
    elif is_timeout_error:
        if current_api_key in good_keys:
            good_keys.remove(current_api_key)
        logger.warning(
            "Non-key error for provider '%s': %s, %s keys remaining",
            provider,
            "timeout",
            len(good_keys),
        )
    else:
        if current_api_key in good_keys:
            good_keys.remove(current_api_key)
        logger.warning(
            "Unknown error for provider '%s', %s keys remaining",
            provider,
            len(good_keys),
        )

    return good_keys, last_error_msg


def _initialize_loop_state(context: GenerationContext) -> GenerationLoopState:
    provider = context.provider
    actual_model = context.actual_model
    fallback_chain = context.fallback_chain or []
    use_custom_fallbacks = context.fallback_chain is not None
    is_original_mistral = provider == "mistral" and "mistral" in actual_model.lower()

    fallback_state = FallbackState(
        fallback_level=0,
        fallback_index=0,
        use_custom_fallbacks=use_custom_fallbacks,
        is_original_mistral=is_original_mistral,
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
    next_fallback = _get_next_fallback(
        state=loop_state.fallback_state,
        fallback_chain=loop_state.fallback_chain,
        provider=loop_state.provider,
        initial_key_count=loop_state.initial_key_count,
    )

    if next_fallback:
        (
            loop_state.provider,
            loop_state.actual_model,
            _provider_slash_model,
            loop_state.base_url,
            fallback_api_keys,
        ) = _apply_fallback_config(
            next_fallback=next_fallback,
            config=context.config,
        )

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

    state.response_contents = await _render_exhausted_response(
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
    use_plain_responses = state.use_plain_responses
    grounding_metadata = state.grounding_metadata
    loop_state = StreamLoopState(curr_content=None, finish_reason=None)

    async with context.new_msg.channel.typing():
        async for (
            delta_content,
            new_finish_reason,
            new_grounding_metadata,
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

            if loop_state.finish_reason is not None:
                break

            loop_state.finish_reason = new_finish_reason

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

            if use_plain_responses:
                continue

            await _maybe_edit_stream_message(
                state=state,
                reply_helper=reply_helper,
                decision=decision,
                grounding_metadata=grounding_metadata,
            )

    if not response_contents:
        _raise_empty_response()

    if use_plain_responses:
        await _render_plain_responses(
            response_contents=response_contents,
            response_msgs=state.response_msgs,
            reply_helper=reply_helper,
            grounding_metadata=grounding_metadata,
            tavily_metadata=context.tavily_metadata,
        )

    state.grounding_metadata = grounding_metadata


async def _run_generation_loop(
    *,
    context: GenerationContext,
    state: GenerationState,
) -> None:
    loop_state = _initialize_loop_state(context)
    response_msgs = state.response_msgs

    async def reply_helper(**reply_kwargs: object) -> None:
        reply_target = context.new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)

        context.msg_nodes[response_msg.id] = MsgNode(parent_msg=context.new_msg)
        await context.msg_nodes[response_msg.id].lock.acquire()

    while True:
        state.response_contents = []
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
        except (
            FirstTokenTimeoutError,
            asyncio.TimeoutError,
            litellm.exceptions.OpenAIError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            (
                loop_state.good_keys,
                loop_state.last_error_msg,
            ) = _handle_generation_exception(
                error=exc,
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
