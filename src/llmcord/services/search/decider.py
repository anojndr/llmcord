"""Web search decision logic."""

import importlib
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import litellm

from llmcord.core.config import ensure_list, get_config
from llmcord.core.config.utils import is_gemini_model
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


async def _get_decider_response_text(
    *,
    run_config: DeciderRunConfig,
    current_api_key: str,
    litellm_messages: list[dict[str, object]],
) -> str:
    if run_config.provider == "google-gemini-cli":
        response_chunks: list[str] = []
        async for delta_content, _chunk_finish_reason in stream_google_gemini_cli(
            model=run_config.model,
            messages=litellm_messages,
            api_key=current_api_key,
            base_url=run_config.base_url,
            extra_headers=run_config.extra_headers,
            model_parameters=run_config.model_parameters,
        ):
            if delta_content:
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
        except Exception as exc:
            logger.exception("Error in web search decider")
            rotator.mark_current_bad(str(exc))
            continue

        exhausted_keys = False
        break

    return None, exhausted_keys


def _get_next_decider_fallback(
    fallback_level: int,
    *,
    original_provider: str,
    original_model: str,
) -> tuple[int, tuple[str, str, str] | None, str | None]:
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

    ordered_fallbacks = [openrouter_fallback, mistral_fallback, gemini_fallback]
    original = (original_provider, original_model)
    fallback_chain = [
        fallback
        for fallback in ordered_fallbacks
        if (fallback[0], fallback[1]) != original
    ]

    if fallback_level < len(fallback_chain):
        next_fallback = fallback_chain[fallback_level]
        next_level = fallback_level + 1
        if fallback_level == 0:
            log_message = (
                "Search decider exhausted all keys for provider "
                f"'{original_provider}'. Falling back to {next_fallback[2]}..."
            )
        else:
            log_message = (
                f"Search decider fallback failed. Falling back to {next_fallback[2]}..."
            )
        return next_level, next_fallback, log_message

    return fallback_level, None, None


async def decide_web_search(messages: list, decider_config: dict) -> dict:
    """Decide whether web search is needed and generate optimized queries.

    Uses LiteLLM for unified API access across all providers. Uses KeyRotator
    for consistent key rotation and bad key tracking.

    Returns: {"needs_search": bool, "queries": list[str]} or
    {"needs_search": False}.

    decider_config should contain:
        - provider: "gemini", "github_copilot", or other (OpenAI-compatible)
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

    fallback_level = 0
    original_provider = provider
    original_model = model

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

        fallback_level, next_fallback, log_message = _get_next_decider_fallback(
            fallback_level,
            original_provider=original_provider,
            original_model=original_model,
        )
        if log_message:
            logger.warning(log_message)
        if not next_fallback:
            logger.error(
                ("Search decider fallback options exhausted, skipping web search"),
            )
            return default_result

        new_provider, new_model, _ = next_fallback
        provider = new_provider
        model = new_model

        fallback_provider_config = config.get("providers", {}).get(provider, {})
        base_url = fallback_provider_config.get("base_url")
        api_keys = ensure_list(fallback_provider_config.get("api_key"))
        extra_headers = fallback_provider_config.get("extra_headers")
        model_parameters = None

        if api_keys:
            continue

        logger.error(
            "No API keys available for search decider fallback provider '%s'",
            provider,
        )
