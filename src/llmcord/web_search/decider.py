"""Web search decision logic and prompt utilities."""

from __future__ import annotations

import json
import logging
import textwrap
from datetime import datetime

import litellm

from llmcord.bad_keys import KeyRotator
from llmcord.config import ensure_list, get_config
from llmcord.litellm_utils import LiteLLMOptions, prepare_litellm_kwargs

logger = logging.getLogger(__name__)

MIN_DECIDER_MESSAGES = 2

SEARCH_DECIDER_SYSTEM_PROMPT = textwrap.dedent(
        """
        You are a web search query optimizer. Your job is to analyze user queries
        and determine if they need web search for up-to-date or factual information.

        RESPOND ONLY with a valid JSON object. No other text, no markdown.

        If web search is NOT needed (e.g., creative writing, opinions, general
        knowledge, math, coding):
        {"needs_search": false}

        If web search IS needed (e.g., current events, recent news, product specs,
        prices, real-time info):
        {"needs_search": true, "queries": ["query1", "query2", ...]}

        RULES for generating queries:
        1. Keep queries concise—under 400 characters. Think of it as a query for an
             agent performing web search, not long-form prompts.
        2. EXTRACT CONCRETE ENTITIES, NOT META-DESCRIPTIONS. Your queries must be
             actual searchable terms, NOT descriptions of what to search for.
             - NEVER output queries like "events mentioned by X", "topics in the
                 conversation", "things the user asked about", etc.
             - ALWAYS extract the ACTUAL entities, names, events, or topics from the
                 conversation and use those as queries.
             - If the user says "search for the events John mentioned" and John
                 mentioned "Russia Ukraine war" and "Trump policies", your queries
                 should be ["Russia Ukraine war", "Trump policies"], NOT
                 ["events mentioned by John"].
        3. SINGLE ENTITY = SINGLE QUERY. If the user asks about ONE entity, output
             exactly ONE search query. Do NOT split into multiple queries.
             Example: "latest news" → ["latest news today"] (ONE query only)
             Example: "iPhone 16 price" → ["iPhone 16 price"] (ONE query only)
        4. MULTIPLE ENTITIES = MULTIPLE QUERIES. Only if the user asks about
             multiple entities, create separate queries for EACH entity PLUS a query
             containing all entities.
             Example: "which is the best? B&K 5128 Diffuse Field Target, VDSF 5128 Demo
             Target Response On-Ear, VDSF 5128 Demo Target Response In-Ear, 5128 Harman
             In-Ear 2024 Beta, or 4128/4195 VDSF Target Response?" →
             ["B&K 5128 Diffuse Field Target", "VDSF 5128 Demo Target Response On-Ear",
             "VDSF 5128 Demo Target Response In-Ear", "5128 Harman In-Ear 2024 Beta",
             "4128/4195 VDSF Target Response", "B&K 5128 Diffuse Field Target vs VDSF
             5128 Demo Target Response On-Ear vs VDSF 5128 Demo Target Response In-Ear
             vs 5128 Harman In-Ear 2024 Beta vs 4128/4195 VDSF Target Response"]
        5. Make queries search-engine friendly
        6. Preserve the user's original intent

        BAD QUERIES (never output these):
        - "events mentioned by Joeii in their reply" ❌ (meta-description, not
            searchable)
        - "topics discussed in the conversation" ❌ (vague, not extracting actual
            content)
        - "things the user wants to know about" ❌ (self-referential)
        - "information from the image" ❌ (not extracting actual content from image)

        GOOD QUERIES (extract actual content):
        - If someone mentions "Russia invading Ukraine" → ["Russia Ukraine war 2024"]
        - If someone mentions "Trump's policies" → ["Trump policies"]
        - If someone mentions "China and Taiwan conflict" → ["China Taiwan relations"]

        Examples:
        - "What's the weather today?" →
            {"needs_search": true, "queries": ["weather today"]}
        - "Who won the 2024 Super Bowl?" →
            {"needs_search": true, "queries": ["2024 Super Bowl winner"]}
        - "latest news" →
            {"needs_search": true, "queries": ["latest news today"]}
        - "Write me a poem about cats" → {"needs_search": false}
        - "Compare RTX 4090 and RTX 4080" →
            {"needs_search": true,
            "queries": ["RTX 4090", "RTX 4080", "RTX 4090 vs RTX 4080"]}
        - User shares image with text about "Biden's economic plan" and says
            "search for this" →
            {"needs_search": true, "queries": ["Biden economic plan"]}
        - Conversation mentions "Greenland invasion by Russia" and user says
            "look that up" →
            {"needs_search": true, "queries": ["Russia Greenland invasion"]}
        """,
).strip()


def get_current_datetime_strings() -> tuple[str, str]:
    """Get current date and time strings for system prompts.

    Returns a tuple of (date_str, time_str).

    Date format: "January 21 2026"
    Time format: "20:00:00 +0800"
    """
    now = datetime.now().astimezone()
    date_str = now.strftime("%B %d %Y")
    time_str = now.strftime("%H:%M:%S %Z%z")
    return date_str, time_str


def convert_messages_to_openai_format(
    messages: list,
    system_prompt: str | None = None,
    *,
    reverse: bool = True,
    include_analysis_prompt: bool = False,
) -> list[dict]:
    """Convert internal message format to OpenAI-compatible message format.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        system_prompt: Optional system prompt to prepend
        reverse: Whether to reverse the message order (default True for chronological)
        include_analysis_prompt: Whether to append the analysis instruction prompt

    Returns:
        List of OpenAI-compatible message dicts

    """
    openai_messages = []

    if system_prompt:
        openai_messages.append({"role": "system", "content": system_prompt})

    message_list = messages[::-1] if reverse else messages

    for msg in message_list:
        role = msg.get("role", "user")
        if role == "system":
            continue  # Skip system messages from chat history

        content = msg.get("content", "")
        if isinstance(content, list):
            # Filter to only include types supported by OpenAI-compatible APIs
            # GitHub Copilot and others only accept 'text' and 'image_url' types
            filtered_content = [
                part
                for part in content
                if isinstance(part, dict)
                and part.get("type") in ("text", "image_url")
            ]
            if filtered_content:
                openai_messages.append({"role": role, "content": filtered_content})
        elif content:
            openai_messages.append({"role": role, "content": str(content)})

    if include_analysis_prompt:
        openai_messages.append(
            {
                "role": "user",
                "content": (
                    "Based on the conversation above, analyze the last user "
                    "query and respond with your JSON decision."
                ),
            },
        )

    return openai_messages


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
