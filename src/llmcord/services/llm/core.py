"""Core LLM service operations."""

from typing import Any

from llmcord.core.config import is_gemini_model
from llmcord.services.llm.providers.gemini import configure_gemini_kwargs
from llmcord.services.llm.providers.github import (
    configure_github_copilot_kwargs,
)
from llmcord.services.llm.types import LiteLLMOptions


def build_litellm_model_name(provider: str, model: str) -> str:
    """Build the LiteLLM model name with proper provider prefix.

    Args:
        provider: Provider name (e.g., "gemini", "openai", "github_copilot")
        model: Model name

    Returns:
        LiteLLM-compatible model string (e.g., "gemini/gemini-3-flash-preview")

    """
    if provider == "gemini":
        return f"gemini/{model}"
    if provider == "github_copilot":
        return f"github_copilot/{model}"
    if provider == "mistral":
        return f"mistral/{model}"
    if provider == "openrouter":
        return f"openrouter/{model}"
    # For OpenAI-compatible providers, just use the model name
    # LiteLLM will use base_url if provided
    return model


def prepare_litellm_kwargs(
    provider: str,
    model: str,
    messages: list,
    api_key: str,
    *,
    options: LiteLLMOptions | None = None,
) -> dict[str, Any]:
    """Prepare kwargs for LiteLLM acompletion() with provider configuration.

    This is the main entry point for creating consistent LiteLLM calls across
    both the main model handler and search decider.

    Args:
        provider: Provider name (e.g., "gemini", "openai", "github_copilot")
        model: Model name (actual model, after any aliasing)
        messages: List of message dicts
        api_key: API key to use
        options: Optional configuration bundle for provider-specific settings

    Returns:
        Dict of kwargs ready to pass to litellm.acompletion()

    """
    options = options or LiteLLMOptions()

    # Build the model name with proper prefix
    litellm_model = build_litellm_model_name(provider, model)

    # Base kwargs
    kwargs: dict[str, Any] = {
        "model": litellm_model,
        "messages": messages,
        "api_key": api_key,
    }

    if options.stream:
        kwargs["stream"] = True

    if options.temperature is not None:
        kwargs["temperature"] = options.temperature

    # Add base_url for OpenAI-compatible providers (not Gemini or GitHub
    # Copilot).
    if options.base_url and provider not in ("gemini", "github_copilot"):
        kwargs["base_url"] = options.base_url

    # Provider-specific configuration
    # Only apply Gemini-specific config to actual Gemini models (not Gemma)
    if provider == "gemini" and is_gemini_model(model):
        configure_gemini_kwargs(
            kwargs,
            model,
            options.model_parameters,
            enable_grounding=options.enable_grounding,
        )
    elif provider == "github_copilot":
        configure_github_copilot_kwargs(kwargs, api_key, options.extra_headers)

    # Merge extra headers if provided (after provider-specific headers)
    if options.extra_headers and "extra_headers" not in kwargs:
        kwargs["extra_headers"] = options.extra_headers
    elif options.extra_headers and "extra_headers" in kwargs:
        kwargs["extra_headers"] = {
            **kwargs["extra_headers"],
            **options.extra_headers,
        }

    return kwargs
