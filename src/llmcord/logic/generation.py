"""LLM response generation logic."""
import asyncio
import base64
import binascii
import hashlib
import io
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
from llmcord.discord.ui import metadata as ui_metadata
from llmcord.discord.ui import response_view, sources_view
from llmcord.logic.utils import (
    count_conversation_tokens,
    count_text_tokens,
)
from llmcord.services.database import get_bad_keys_db
from llmcord.services.database.messages import MessageResponsePayload
from llmcord.services.llm import LiteLLMOptions, prepare_litellm_kwargs

logger = logging.getLogger(__name__)

DATA_URL_PATTERN = re.compile(
    r"data:(image/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=]+)",
)

LITELLM_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    litellm.exceptions.OpenAIError,
)
for _exception_name in (
    "RateLimitError",
    "APIError",
    "APIConnectionError",
    "ServiceUnavailableError",
):
    _exception_type = getattr(litellm.exceptions, _exception_name, None)
    if _exception_type is not None:
        LITELLM_RETRYABLE_ERRORS += (_exception_type,)

GENERATION_EXCEPTIONS = (
    FirstTokenTimeoutError,
    asyncio.TimeoutError,
    OSError,
    RuntimeError,
    ValueError,
    *LITELLM_RETRYABLE_ERRORS,
)


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
    generated_images: list["GeneratedImage"]
    generated_image_hashes: set[str]


@dataclass(slots=True)
class FallbackState:
    """Track fallback selection state."""

    fallback_level: int
    fallback_index: int
    use_custom_fallbacks: bool
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
class GeneratedImage:
    """Generated image payload from Gemini responses."""

    data: bytes
    mime_type: str
    filename: str


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
        (
            "All API keys for provider '%s' (synced) are marked as bad. "
            "Resetting..."
        ),
        provider,
    )
    try:
        get_bad_keys_db().reset_provider_keys_synced(provider)
    except (OSError, RuntimeError, ValueError):
        logger.exception("Failed to reset provider keys")
    return api_keys.copy()


def _get_default_fallback_chain(
    original_provider: str,
    original_model: str,
) -> list[tuple[str, str, str]]:
    openrouter_fallback = (
        "openrouter",
        "openrouter/free",
        "openrouter/openrouter/free",
    )
    mistral_fallback = (
        "mistral",
        "mistral-large-latest",
        "mistral/mistral-large-latest",
    )
    gemini_fallback = (
        "gemini",
        "gemma-3-27b-it",
        "gemini/gemma-3-27b-it",
    )

    if (
        original_provider == "openrouter"
        and original_model == "openrouter/free"
    ):
        return [mistral_fallback, gemini_fallback]
    if (
        original_provider == "mistral"
        and original_model == "mistral-large-latest"
    ):
        return [openrouter_fallback, gemini_fallback]
    if (
        original_provider == "gemini"
        and original_model == "gemma-3-27b-it"
    ):
        return [openrouter_fallback, mistral_fallback]

    return [openrouter_fallback, mistral_fallback, gemini_fallback]


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
                (
                    "All %s keys exhausted for provider '%s'. "
                    "Falling back to %s..."
                ),
                initial_key_count,
                provider,
                next_fallback[2],
            )
            return next_fallback
        return None

    default_fallbacks = _get_default_fallback_chain(
        state.original_provider,
        state.original_model,
    )
    if state.fallback_level < len(default_fallbacks):
        next_fallback = default_fallbacks[state.fallback_level]
        state.fallback_level += 1
        if state.fallback_level == 1:
            logger.warning(
                (
                    "All %s keys exhausted for provider '%s'. "
                    "Falling back to %s..."
                ),
                initial_key_count,
                provider,
                next_fallback[2],
            )
        else:
            logger.warning(
                "Fallback also failed. Falling back to %s...",
                next_fallback[2],
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
        logger.error("All fallback options exhausted")

    error_text = (
        "❌ All API keys are currently unavailable. Please try again later."
    )
    if last_error_msg:
        logger.info("Last fallback error summary: %s", last_error_msg)

    if state.embed is None:
        state.embed = discord.Embed(
            description=error_text,
            color=EMBED_COLOR_INCOMPLETE,
        )
    else:
        state.embed.description = error_text
        state.embed.color = EMBED_COLOR_INCOMPLETE

    if state.response_msgs:
        await state.response_msgs[-1].edit(embed=state.embed, view=None)
    else:
        await reply_helper(embed=state.embed)
    return [error_text]


def _extension_from_mime(mime_type: str) -> str:
    extension = mime_type.split("/", maxsplit=1)[-1].split(";", maxsplit=1)[0]
    return extension or "png"


def _build_generated_image(data: bytes, mime_type: str) -> GeneratedImage:
    digest = hashlib.sha256(data).hexdigest()[:12]
    extension = _extension_from_mime(mime_type)
    filename = f"gemini-output-{digest}.{extension}"
    return GeneratedImage(data=data, mime_type=mime_type, filename=filename)


def _coerce_payload(obj: object) -> object:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict") and callable(obj.dict):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return obj


def _extract_images_from_string(text: str) -> list[GeneratedImage]:
    images: list[GeneratedImage] = []
    for mime_type, b64_data in DATA_URL_PATTERN.findall(text):
        try:
            data = base64.b64decode(b64_data)
        except (ValueError, binascii.Error):
            continue
        images.append(_build_generated_image(data, mime_type))
    return images


def _extract_images_from_inline_data(inline_data: dict) -> list[GeneratedImage]:
    mime_type = inline_data.get("mime_type") or inline_data.get("mimeType")
    data = inline_data.get("data")
    if (
        not isinstance(mime_type, str)
        or not mime_type.startswith("image/")
        or not isinstance(data, str)
    ):
        return []
    try:
        decoded = base64.b64decode(data)
    except (ValueError, binascii.Error):
        return []
    return [_build_generated_image(decoded, mime_type)]


def _extract_images_from_image_url(image_url: object) -> list[GeneratedImage]:
    if isinstance(image_url, dict):
        image_url = image_url.get("url")
    if not isinstance(image_url, str):
        return []
    return _extract_images_from_string(image_url)


def _extract_images_from_mime_data(
    data: object,
    mime_type: object,
) -> list[GeneratedImage]:
    if (
        not isinstance(mime_type, str)
        or not mime_type.startswith("image/")
        or not isinstance(data, str)
    ):
        return []
    try:
        decoded = base64.b64decode(data)
    except (ValueError, binascii.Error):
        return []
    return [_build_generated_image(decoded, mime_type)]


def _extract_images_from_dict(obj: dict) -> list[GeneratedImage]:
    images: list[GeneratedImage] = []

    inline_data = obj.get("inline_data") or obj.get("inlineData")
    if isinstance(inline_data, dict):
        images.extend(_extract_images_from_inline_data(inline_data))

    image_url = obj.get("image_url") or obj.get("imageUrl")
    if image_url:
        images.extend(_extract_images_from_image_url(image_url))

    images.extend(
        _extract_images_from_mime_data(obj.get("data"), obj.get("mime_type")),
    )
    images.extend(
        _extract_images_from_mime_data(obj.get("data"), obj.get("mimeType")),
    )
    return images


def _extract_generated_images(value: object) -> list[GeneratedImage]:
    images: list[GeneratedImage] = []
    seen_ids: set[int] = set()

    stack: list[object] = [value]
    while stack:
        obj = _coerce_payload(stack.pop())
        if obj is None:
            continue

        obj_id = id(obj)
        if obj_id in seen_ids:
            continue
        seen_ids.add(obj_id)

        if isinstance(obj, str):
            images.extend(_extract_images_from_string(obj))
            continue
        if isinstance(obj, dict):
            images.extend(_extract_images_from_dict(obj))
            stack.extend(obj.values())
            continue
        if isinstance(obj, (list, tuple, set)):
            stack.extend(obj)

    return images


def _extract_gemini_images_from_chunk(
    chunk: object,
    choice: object,
    delta_content: str,
) -> list[GeneratedImage]:
    images: list[GeneratedImage] = []
    sources = [
        delta_content,
        getattr(choice, "delta", None),
        getattr(choice, "message", None),
        getattr(chunk, "model_extra", None),
        getattr(chunk, "_hidden_params", None),
    ]
    for source in sources:
        if source is None:
            continue
        images.extend(_extract_generated_images(source))
    return images


def _append_generated_images(
    state: GenerationState,
    images: list[GeneratedImage],
) -> None:
    for image in images:
        digest = hashlib.sha256(image.data).hexdigest()
        if digest in state.generated_image_hashes:
            continue
        state.generated_image_hashes.add(digest)
        state.generated_images.append(image)


async def _initialize_generation_state(
    *,
    context: GenerationContext,
) -> GenerationState:
    response_msgs = [context.processing_msg]
    processing_msg_id = context.processing_msg.id
    context.msg_nodes[processing_msg_id] = MsgNode(parent_msg=context.new_msg)
    await context.msg_nodes[processing_msg_id].lock.acquire()

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
        footer_text = (
            f"{context.provider_slash_model} | total tokens: {input_tokens:,}"
        )
        embed.set_footer(text=footer_text)

    return GenerationState(
        response_msgs=response_msgs,
        response_contents=[],
        input_tokens=input_tokens,
        max_message_length=max_message_length,
        embed=embed,
        use_plain_responses=use_plain_responses,
        grounding_metadata=None,
        last_edit_time=context.last_edit_time,
        generated_images=[],
        generated_image_hashes=set(),
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

    response_view_instance = response_view.ResponseView(
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
        footer_text = (
            f"{context.provider_slash_model} | total tokens: {total_tokens:,}"
        )
        state.embed.set_footer(text=footer_text)
        await state.response_msgs[last_msg_index].edit(
            embed=state.embed,
            view=response_view_instance,
        )


async def _send_generated_images(
    *,
    context: GenerationContext,
    state: GenerationState,
) -> None:
    if not state.generated_images:
        logger.debug(
            "No Gemini-generated images to send for message %s",
            context.new_msg.id,
        )
        return

    logger.info(
        "Sending %s Gemini-generated image(s) for message %s",
        len(state.generated_images),
        context.new_msg.id,
    )
    reply_target = (
        state.response_msgs[-1] if state.response_msgs else context.new_msg
    )
    batch_size = 10
    for index in range(0, len(state.generated_images), batch_size):
        batch = state.generated_images[index : index + batch_size]
        try:
            files = [
                discord.File(io.BytesIO(image.data), filename=image.filename)
                for image in batch
            ]
            total_bytes = sum(len(image.data) for image in batch)
            logger.debug(
                "Prepared %s image(s) (%s bytes) for batch %s-%s",
                len(batch),
                total_bytes,
                index,
                index + len(batch) - 1,
            )
            content = "Generated image(s):" if index == 0 else None
            await reply_target.reply(content=content, files=files)
            logger.info(
                "Sent Gemini-generated image batch %s-%s for message %s",
                index,
                index + len(batch) - 1,
                context.new_msg.id,
            )
        except Exception:
            logger.exception(
                (
                    "Failed to send Gemini-generated image batch %s-%s for "
                    "message %s"
                ),
                index,
                index + len(batch) - 1,
                context.new_msg.id,
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
    await _send_generated_images(context=context, state=state)
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
) -> AsyncIterator[
    tuple[str, object | None, object | None, list[GeneratedImage]]
]:
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
        image_payloads = _extract_gemini_images_from_chunk(
            chunk,
            choice,
            delta_content,
        )

        if chunk_finish_reason and is_gemini_model(stream_config.actual_model):
            chunk_attrs = [
                attr for attr in dir(chunk) if not attr.startswith("_")
            ]
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

        yield (
            delta_content,
            chunk_finish_reason,
            grounding_metadata,
            image_payloads,
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
        sources_view.SourceView(grounding_metadata)
        if decision.is_final_edit
        and ui_metadata.has_grounding_data(grounding_metadata)
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
        layout = response_view.LayoutView().add_item(
            response_view.TextDisplay(content=content),
        )

        if i == len(response_contents) - 1:
            if ui_metadata.has_grounding_data(grounding_metadata):
                layout.add_item(
                    sources_view.SourceButton(grounding_metadata),
                )
            if tavily_metadata and (
                tavily_metadata.get("urls") or tavily_metadata.get("queries")
            ):
                layout.add_item(
                    sources_view.TavilySourceButton(tavily_metadata),
                )

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
            "Empty response for provider '%s'; removing key and trying "
            "remaining keys."
        )

    error_str = str(error).lower()
    is_developer_instruction_error = (
        "developer instruction" in error_str
        and "not enabled" in error_str
    )
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
            logger.warning(
                special_case_message,
                FIRST_TOKEN_TIMEOUT_SECONDS,
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
        except Exception:
            logger.exception("Failed to mark key as bad")

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
            image_payloads,
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
                _append_generated_images(state, image_payloads)

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
        reply_target = (
            context.new_msg if not response_msgs else response_msgs[-1]
        )
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
        except GENERATION_EXCEPTIONS as exc:
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
