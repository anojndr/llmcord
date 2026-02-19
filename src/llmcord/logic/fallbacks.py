"""Fallback logic for LLM generation."""

import logging
from typing import Any

from llmcord.core.config import ensure_list
from llmcord.logic.generation_types import FallbackState

logger = logging.getLogger(__name__)


FallbackModel = tuple[str, str, str]
DEFAULT_FALLBACK_MODELS: tuple[FallbackModel, ...] = (
    ("openrouter", "openrouter/free", "openrouter/openrouter/free"),
    ("mistral", "mistral-large-latest", "mistral/mistral-large-latest"),
    ("gemini", "gemma-3-27b-it", "gemini/gemma-3-27b-it"),
)

GOOGLE_GEMINI_CLI_FALLBACKS: dict[tuple[str, str], FallbackModel] = {
    (
        "google-gemini-cli",
        "gemini-3-flash-preview-low",
    ): (
        "gemini",
        "gemini-3-flash-preview-low",
        "gemini/gemini-3-flash-preview-low",
    ),
    (
        "google-gemini-cli",
        "gemini-3-flash-preview-minimal",
    ): (
        "gemini",
        "gemini-3-flash-preview-minimal",
        "gemini/gemini-3-flash-preview-minimal",
    ),
    (
        "google-gemini-cli",
        "gemini-3-flash-preview-high",
    ): (
        "gemini",
        "gemini-3-flash-preview-high",
        "gemini/gemini-3-flash-preview-high",
    ),
    (
        "google-antigravity",
        "gemini-3-flash-preview-low",
    ): (
        "gemini",
        "gemini-3-flash-preview-low",
        "gemini/gemini-3-flash-preview-low",
    ),
    (
        "google-antigravity",
        "gemini-3-flash-preview-minimal",
    ): (
        "gemini",
        "gemini-3-flash-preview-minimal",
        "gemini/gemini-3-flash-preview-minimal",
    ),
    (
        "google-antigravity",
        "gemini-3-flash-preview-high",
    ): (
        "gemini",
        "gemini-3-flash-preview-high",
        "gemini/gemini-3-flash-preview-high",
    ),
}


def build_default_fallback_chain(
    original_provider: str,
    original_model: str,
) -> list[FallbackModel]:
    """Build ordered default fallback models excluding the original pair."""
    original = (original_provider, original_model)

    fallback_chain = [
        fallback
        for fallback in DEFAULT_FALLBACK_MODELS
        if (fallback[0], fallback[1]) != original
    ]

    preferred_first_fallback = GOOGLE_GEMINI_CLI_FALLBACKS.get(original)
    if (
        preferred_first_fallback is not None
        and preferred_first_fallback not in fallback_chain
    ):
        fallback_chain.insert(0, preferred_first_fallback)

    return fallback_chain


def get_next_fallback(
    *,
    state: FallbackState,
    fallback_chain: list[FallbackModel],
    provider: str,
    initial_key_count: int,
) -> FallbackModel | None:
    """Determine the next fallback provider and model to try."""
    if state.use_custom_fallbacks:
        if state.fallback_index < len(fallback_chain):
            next_fallback = fallback_chain[state.fallback_index]
            state.fallback_index += 1
            logger.warning(
                ("All %s keys exhausted for provider '%s'. Falling back to %s..."),
                initial_key_count,
                provider,
                next_fallback[2],
            )
            return next_fallback
        return None

    default_fallbacks = build_default_fallback_chain(
        state.original_provider,
        state.original_model,
    )
    if state.fallback_level < len(default_fallbacks):
        next_fallback = default_fallbacks[state.fallback_level]
        state.fallback_level += 1
        if state.fallback_level == 1:
            logger.warning(
                ("All %s keys exhausted for provider '%s'. Falling back to %s..."),
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


def apply_fallback_config(
    *,
    next_fallback: FallbackModel,
    config: dict[str, Any],
) -> tuple[str, str, str, str | None, list[str]]:
    """Extract configuration values for a fallback provider."""
    provider, model, provider_slash_model = next_fallback
    providers = config.get("providers")
    if not isinstance(providers, dict):
        providers = {}
    provider_config = providers.get(provider)
    if not isinstance(provider_config, dict):
        provider_config = {}
    base_url = provider_config.get("base_url")
    api_keys = ensure_list(provider_config.get("api_key"))
    return provider, model, provider_slash_model, base_url, api_keys
