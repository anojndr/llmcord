"""Web search decision logic."""
import importlib
import json
import logging
from dataclasses import dataclass

import litellm

from llmcord.core.config import ensure_list, get_config
from llmcord.services.database import KeyRotator, get_bad_keys_db
from llmcord.services.llm import LiteLLMOptions, prepare_litellm_kwargs
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
    disable_system_prompt: bool


def _get_decider_runner() -> object:
    search_module = importlib.import_module("llmcord.services.search")
    return search_module.run_decider_once


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

            if run_config.disable_system_prompt:
                litellm_messages = convert_messages_to_openai_format(
                    messages,
                    system_prompt=None,
                    reverse=True,
                    include_analysis_prompt=False,
                )
                litellm_messages.append(
                    {"role": "user", "content": system_prompt_with_date},
                )
                litellm_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Based on the conversation above, analyze the "
                            "last user query and respond with your JSON "
                            "decision."
                        ),
                    },
                )
            else:
                # Convert messages to OpenAI format (LiteLLM uses OpenAI format)
                litellm_messages = convert_messages_to_openai_format(
                    messages,
                    system_prompt=system_prompt_with_date,
                    reverse=True,
                    include_analysis_prompt=True,
                )

            if len(litellm_messages) <= MIN_DECIDER_MESSAGES:
                exhausted_keys = False
                break

            # Use shared utility to prepare kwargs with all provider-specific
            # config.
            litellm_kwargs = prepare_litellm_kwargs(
                provider=run_config.provider,
                model=run_config.model,
                messages=litellm_messages,
                api_key=current_api_key,
                options=LiteLLMOptions(
                    base_url=run_config.base_url,
                    temperature=0.1,
                ),
            )

            # Make the LiteLLM call
            response = await litellm.acompletion(**litellm_kwargs)

            response_text = (response.choices[0].message.content or "").strip()

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

    if (
        original_provider == "openrouter"
        and original_model == "openrouter/free"
    ):
        fallback_chain = [mistral_fallback, gemini_fallback]
    elif (
        original_provider == "mistral"
        and original_model == "mistral-large-latest"
    ):
        fallback_chain = [openrouter_fallback, gemini_fallback]
    elif (
        original_provider == "gemini"
        and original_model == "gemma-3-27b-it"
    ):
        fallback_chain = [openrouter_fallback, mistral_fallback]
    else:
        fallback_chain = [openrouter_fallback, mistral_fallback, gemini_fallback]

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
                "Search decider fallback failed. "
                f"Falling back to {next_fallback[2]}..."
            )
        return next_level, next_fallback, log_message

    return fallback_level, None, None


def _is_decider_system_prompt_disabled(
    *,
    config: dict,
    provider: str,
    model: str,
) -> bool:
    model_key = f"{provider}/{model}"
    model_parameters = config.get("models", {}).get(model_key, {})
    disable_override = model_parameters.get("disable_system_prompt")
    if isinstance(disable_override, bool):
        return disable_override

    disabled_models = ensure_list(config.get("disable_system_prompt_models"))
    normalized_targets = {model_key.lower(), model.lower()}
    for model_name in disabled_models:
        if not isinstance(model_name, str):
            continue
        model_name_lower = model_name.strip().lower()
        if model_name_lower and model_name_lower in normalized_targets:
            return True

    return provider == "openrouter" and model == "free"


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

    default_result = {"needs_search": False}
    config = get_config()

    fallback_level = 0
    original_provider = provider
    original_model = model
    disable_system_prompt = _is_decider_system_prompt_disabled(
        config=config,
        provider=provider,
        model=model,
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
                disable_system_prompt=disable_system_prompt,
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
                (
                    "Search decider fallback options exhausted, skipping web "
                    "search"
                ),
            )
            return default_result

        new_provider, new_model, _ = next_fallback
        provider = new_provider
        model = new_model
        disable_system_prompt = _is_decider_system_prompt_disabled(
            config=config,
            provider=provider,
            model=model,
        )

        fallback_provider_config = config.get("providers", {}).get(provider, {})
        base_url = fallback_provider_config.get("base_url")
        api_keys = ensure_list(fallback_provider_config.get("api_key"))

        if api_keys:
            continue

        logger.error(
            "No API keys available for search decider fallback provider '%s'",
            provider,
        )
