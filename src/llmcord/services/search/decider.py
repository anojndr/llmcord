"""Web search decision logic."""
import json
import logging

import litellm

from llmcord.services.database import KeyRotator
from llmcord.config import ensure_list, get_config
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


async def _run_decider_once(
    messages: list,
    provider: str,
    model: str,
    api_keys: list[str],
    base_url: str | None,
) -> tuple[dict | None, bool]:
    if not api_keys:
        return None, True

    # Use KeyRotator for consistent key rotation with synced bad key tracking
    rotator = KeyRotator(provider, api_keys)

    exhausted_keys = True

    async for current_api_key in rotator.get_keys_async():
        try:
            date_str, time_str = get_current_datetime_strings()
            system_prompt_with_date = (
                f"{SEARCH_DECIDER_SYSTEM_PROMPT}\n\nCurrent date: {date_str}. "
                f"Current time: {time_str}."
            )

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

            # Use shared utility to prepare kwargs with all provider-specific config
            litellm_kwargs = prepare_litellm_kwargs(
                provider=provider,
                model=model,
                messages=litellm_messages,
                api_key=current_api_key,
                options=LiteLLMOptions(
                    base_url=base_url,
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
    is_original_mistral: bool,
    original_provider: str,
    original_model: str,
) -> tuple[int, tuple[str, str, str] | None, str | None]:
    if fallback_level == 0:
        # First fallback: mistral (unless original was already mistral)
        if is_original_mistral:
            return (
                2,
                ("gemini", "gemma-3-27b-it", "gemini/gemma-3-27b-it"),
                (
                    "Search decider exhausted all keys for mistral/"
                    f"{original_model}. Falling back to gemini/gemma-3-27b-it..."
                ),
            )
        return (
            1,
            ("mistral", "mistral-large-latest", "mistral/mistral-large-latest"),
            (
                "Search decider exhausted all keys for provider "
                f"'{original_provider}'. "
                "Falling back to mistral/mistral-large-latest..."
            ),
        )

    if fallback_level == 1:
        # Second fallback: gemma
        return (
            2,
            ("gemini", "gemma-3-27b-it", "gemini/gemma-3-27b-it"),
            (
                "Search decider mistral fallback failed. "
                "Falling back to gemini/gemma-3-27b-it..."
            ),
        )

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

    default_result = {"needs_search": False}
    config = get_config()

    fallback_level = 0  # 0 = original, 1 = mistral, 2 = gemma
    original_provider = provider
    original_model = model
    is_original_mistral = (
        original_provider == "mistral" and "mistral" in original_model.lower()
    )

    while True:
        result, exhausted_keys = await _run_decider_once(
            messages,
            provider,
            model,
            api_keys,
            base_url,
        )
        if result is not None:
            return result
        if not exhausted_keys:
            return default_result

        fallback_level, next_fallback, log_message = _get_next_decider_fallback(
            fallback_level,
            is_original_mistral=is_original_mistral,
            original_provider=original_provider,
            original_model=original_model,
        )
        if log_message:
            logger.warning(log_message)
        if not next_fallback:
            logger.error(
                "Search decider fallback options exhausted (mistral and gemma), "
                "skipping web search",
            )
            return default_result

        new_provider, new_model, _ = next_fallback
        provider = new_provider
        model = new_model

        fallback_provider_config = config.get("providers", {}).get(provider, {})
        base_url = fallback_provider_config.get("base_url")
        api_keys = ensure_list(fallback_provider_config.get("api_key"))

        if api_keys:
            continue

        logger.error(
            "No API keys available for search decider fallback provider '%s'",
            provider,
        )
