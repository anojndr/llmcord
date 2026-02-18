"""Web search decision logic."""

import asyncio
import importlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

import httpx
import litellm

from llmcord.core.config import get_config
from llmcord.core.config.utils import is_gemini_model
from llmcord.core.error_handling import log_exception
from llmcord.core.exceptions import (
    FIRST_TOKEN_TIMEOUT_SECONDS,
    FirstTokenTimeoutError,
)
from llmcord.logic.fallbacks import (
    FallbackModel,
    apply_fallback_config,
    get_next_fallback,
)
from llmcord.logic.generation_types import FallbackState
from llmcord.services.database import KeyRotator, get_bad_keys_db
from llmcord.services.llm import LiteLLMOptions, prepare_litellm_kwargs
from llmcord.services.llm.providers.gemini_cli import stream_google_gemini_cli
from llmcord.services.search.config import (
    MIN_DECIDER_MESSAGES,
    SEARCH_DECIDER_SYSTEM_PROMPT,
)
from llmcord.services.search.utils import (
    convert_messages_to_openai_format,
    get_current_datetime_strings,
)

logger = logging.getLogger(__name__)

_FALLBACK_MODEL_PARTS = 3


def _collect_litellm_exceptions() -> tuple[type[Exception], ...]:
    return tuple(
        dict.fromkeys(
            exception_type
            for exception_type in vars(litellm.exceptions).values()
            if isinstance(exception_type, type)
            and issubclass(exception_type, Exception)
        ),
    )


DECIDER_RETRYABLE_EXCEPTIONS = _collect_litellm_exceptions()


@dataclass(slots=True)
class DeciderRunConfig:
    """Configuration for the search decider runner."""

    provider: str
    model: str
    api_keys: list[str]
    base_url: str | None
    extra_headers: dict[str, str] | None
    model_parameters: dict[str, object] | None


def _get_decider_runner() -> Callable[
    [list[Any], DeciderRunConfig],
    Awaitable[tuple[dict[str, Any] | None, bool]],
]:
    search_module = importlib.import_module("llmcord.services.search")
    return search_module.run_decider_once


async def _iter_stream_with_first_chunk(
    stream_iter: AsyncIterator[object],
    *,
    timeout_seconds: int,
    chunk_has_token: Callable[[object], bool] | None = None,
) -> AsyncIterator[object]:
    start_time = asyncio.get_running_loop().time()
    buffered_chunks: list[object] = []

    try:
        while True:
            elapsed = asyncio.get_running_loop().time() - start_time
            remaining_timeout = max(timeout_seconds - elapsed, 0.0)
            chunk = await asyncio.wait_for(
                stream_iter.__anext__(),
                timeout=remaining_timeout,
            )
            buffered_chunks.append(chunk)
            if chunk_has_token is None or chunk_has_token(chunk):
                break
    except StopAsyncIteration:
        for buffered_chunk in buffered_chunks:
            yield buffered_chunk
        return
    except TimeoutError as exc:
        raise FirstTokenTimeoutError(timeout_seconds=timeout_seconds) from exc

    for buffered_chunk in buffered_chunks:
        yield buffered_chunk
    async for chunk in stream_iter:
        yield chunk


def _google_gemini_cli_chunk_has_token(chunk: object) -> bool:
    delta_content, _chunk_finish_reason, _is_thinking = cast(
        "tuple[str, object | None, bool]",
        chunk,
    )
    return bool(delta_content)


async def _get_decider_response_text(
    *,
    run_config: DeciderRunConfig,
    current_api_key: str,
    litellm_messages: list[dict[str, object]],
) -> str:
    if run_config.provider == "google-gemini-cli":
        response_chunks: list[str] = []
        stream = stream_google_gemini_cli(
            model=run_config.model,
            messages=litellm_messages,
            api_key=current_api_key,
            base_url=run_config.base_url,
            extra_headers=run_config.extra_headers,
            model_parameters=run_config.model_parameters,
        )
        async for chunk in _iter_stream_with_first_chunk(
            stream,
            timeout_seconds=FIRST_TOKEN_TIMEOUT_SECONDS,
            chunk_has_token=_google_gemini_cli_chunk_has_token,
        ):
            delta_content, _chunk_finish_reason, is_thinking = cast(
                "tuple[str, object | None, bool]",
                chunk,
            )
            if delta_content and not is_thinking:
                response_chunks.append(delta_content)
        return "".join(response_chunks).strip()

    litellm_kwargs = prepare_litellm_kwargs(
        provider=run_config.provider,
        model=run_config.model,
        messages=litellm_messages,
        api_key=current_api_key,
        options=LiteLLMOptions(
            base_url=run_config.base_url,
        ),
    )
    response = await litellm.acompletion(**litellm_kwargs)
    return (response.choices[0].message.content or "").strip()


async def _run_decider_once(
    messages: list,
    run_config: DeciderRunConfig,
) -> tuple[dict | None, bool]:
    if not run_config.api_keys:
        return None, True

    # Use KeyRotator for consistent key rotation with synced bad key tracking
    rotator = KeyRotator(
        run_config.provider,
        run_config.api_keys,
        db=get_bad_keys_db(),
    )

    exhausted_keys = True

    async for current_api_key in rotator.get_keys_async():
        try:
            date_str, time_str = get_current_datetime_strings()
            system_prompt_with_date = (
                f"{SEARCH_DECIDER_SYSTEM_PROMPT}\n\nCurrent date: {date_str}. "
                f"Current time: {time_str}."
            )

            decider_is_gemini = is_gemini_model(run_config.model)

            # Convert messages to OpenAI format (LiteLLM uses OpenAI format)
            litellm_messages = convert_messages_to_openai_format(
                messages,
                system_prompt=system_prompt_with_date,
                reverse=True,
                include_analysis_prompt=True,
                is_gemini=decider_is_gemini,
            )

            if len(litellm_messages) <= MIN_DECIDER_MESSAGES:
                exhausted_keys = False
                break

            logger.debug("\n--- SEARCH DECIDER REQUEST ---")
            logger.debug("Provider: %s", run_config.provider)
            logger.debug("Model: %s", run_config.model)
            logger.debug(
                "Messages:\n%s",
                json.dumps(litellm_messages, indent=2),
            )
            logger.debug("------------------------------\n")

            response_text = await _get_decider_response_text(
                run_config=run_config,
                current_api_key=current_api_key,
                litellm_messages=litellm_messages,
            )

            # Parse response
            if response_text.startswith("```"):
                # Remove markdown code blocks if present
                response_text = response_text.split("```")[1]
                response_text = response_text.removeprefix("json")
                response_text = response_text.strip()

            try:
                result = json.loads(response_text)
                # Validate response structure
                if isinstance(result, dict):
                    return result, False
                logger.warning(
                    "Web search decider returned non-dict response: %s",
                    response_text[:100],
                )
                exhausted_keys = False
                break
            except json.JSONDecodeError as json_err:
                logger.warning(
                    "Failed to parse JSON response from search decider: %s. "
                    "Response: %s",
                    json_err,
                    response_text[:200],
                )
                # Attempt to extract needs_search from malformed response
                if (
                    '"needs_search": false' in response_text.lower()
                    or '"needs_search":false' in response_text.lower()
                ):
                    exhausted_keys = False
                    break
                exhausted_keys = False
                break
        except (
            TimeoutError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            httpx.HTTPError,
            *DECIDER_RETRYABLE_EXCEPTIONS,
        ) as exc:
            if isinstance(exc, FirstTokenTimeoutError):
                logger.warning(
                    "Search decider first-token timeout exceeded %ss "
                    "for provider '%s'; "
                    "continuing key rotation and falling "
                    "back to fallback chain if keys are exhausted.",
                    exc.timeout_seconds,
                    run_config.provider,
                )
            log_exception(
                logger=logger,
                message="Error in web search decider",
                error=exc,
                context={
                    "provider": run_config.provider,
                    "model": run_config.model,
                },
            )
            rotator.mark_current_bad(str(exc))
            continue

        exhausted_keys = False
        break

    return None, exhausted_keys


def _normalize_fallback_chain(raw_chain: object) -> list[FallbackModel]:
    if not isinstance(raw_chain, list):
        return []

    normalized: list[FallbackModel] = []
    for item in raw_chain:
        if isinstance(item, tuple) and len(item) == _FALLBACK_MODEL_PARTS:
            provider = item[0]
            model = item[1]
            provider_slash_model = item[2]
            if not (
                isinstance(provider, str)
                and isinstance(model, str)
                and isinstance(provider_slash_model, str)
            ):
                continue
            normalized.append((provider, model, provider_slash_model))
    return normalized


async def decide_web_search(messages: list, decider_config: dict) -> dict:
    """Decide whether web search is needed and generate optimized queries.

    Uses LiteLLM for unified API access across all providers. Uses KeyRotator
    for consistent key rotation and bad key tracking.

    Returns: {"needs_search": bool, "queries": list[str]} or
    {"needs_search": False}.

    decider_config should contain:
        - provider: "gemini" or other (OpenAI-compatible)
        - model: model name
        - api_keys: list of API keys
        - base_url: (optional) for OpenAI-compatible providers
    """
    provider = decider_config.get("provider", "gemini")
    model = decider_config.get("model", "gemini-3-flash-preview")
    api_keys = decider_config.get("api_keys", [])
    base_url = decider_config.get("base_url")
    extra_headers = decider_config.get("extra_headers")
    model_parameters = decider_config.get("model_parameters")

    default_result = {"needs_search": False}
    config = get_config()
    models_config = config.get("models")
    if not isinstance(models_config, dict):
        models_config = {}

    fallback_chain = _normalize_fallback_chain(decider_config.get("fallback_chain"))
    fallback_state = FallbackState(
        fallback_level=0,
        fallback_index=0,
        use_custom_fallbacks=bool(fallback_chain),
        original_provider=provider,
        original_model=model,
    )

    while True:
        runner = _get_decider_runner()
        result, exhausted_keys = await runner(
            messages,
            DeciderRunConfig(
                provider=provider,
                model=model,
                api_keys=api_keys,
                base_url=base_url,
                extra_headers=extra_headers,
                model_parameters=model_parameters,
            ),
        )
        if result is not None:
            return result
        if not exhausted_keys:
            return default_result

        next_fallback = get_next_fallback(
            state=fallback_state,
            fallback_chain=fallback_chain,
            provider=provider,
            initial_key_count=len(api_keys),
        )
        if not next_fallback:
            logger.error(
                ("Search decider fallback options exhausted, skipping web search"),
            )
            return default_result

        provider, model, provider_slash_model, base_url, api_keys = (
            apply_fallback_config(
                next_fallback=next_fallback,
                config=config,
            )
        )
        fallback_provider_config = config.get("providers", {}).get(provider, {})
        extra_headers = fallback_provider_config.get("extra_headers")
        model_parameters_raw = models_config.get(provider_slash_model)
        model_parameters = (
            model_parameters_raw if isinstance(model_parameters_raw, dict) else None
        )

        if api_keys:
            continue

        logger.error(
            "No API keys available for search decider fallback provider '%s'",
            provider,
        )
