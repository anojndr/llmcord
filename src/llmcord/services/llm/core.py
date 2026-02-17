"""Core LLM service operations."""

from typing import Any, cast

from llmcord.core.config import is_gemini_model
from llmcord.services.llm.providers.gemini import configure_gemini_kwargs
from llmcord.services.llm.types import LiteLLMOptions


def _has_audio_or_video_file_inputs(messages: list[object]) -> bool:
    for message in messages:
        if not isinstance(message, dict):
            continue
        message_dict = cast("dict[str, object]", message)
        content = message_dict.get("content")
        if not isinstance(content, list):
            continue

        for part in content:
            if not isinstance(part, dict):
                continue
            part_dict = cast("dict[str, object]", part)
            if part_dict.get("type") != "file":
                continue
            file_obj = part_dict.get("file")
            if not isinstance(file_obj, dict):
                continue
            file_data = cast("dict[str, object]", file_obj).get("file_data")
            if isinstance(file_data, str) and file_data.startswith(
                ("data:audio/", "data:video/"),
            ):
                return True
    return False


def build_litellm_model_name(provider: str, model: str) -> str:
    """Build the LiteLLM model name with proper provider prefix.

    Args:
        provider: Provider name (e.g., "gemini", "openai")
        model: Model name

    Returns:
        LiteLLM-compatible model string (e.g., "gemini/gemini-3-flash-preview")

    """
    if provider == "gemini":
        return f"gemini/{model}"
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
        provider: Provider name (e.g., "gemini", "openai")
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

    # Add base_url for OpenAI-compatible providers (not Gemini).
    if options.base_url and provider != "gemini":
        kwargs["base_url"] = options.base_url

    # Provider-specific configuration
    # Only apply Gemini-specific config to actual Gemini models (not Gemma)
    if provider == "gemini" and is_gemini_model(model):
        has_media_inputs = _has_audio_or_video_file_inputs(messages)
        configure_gemini_kwargs(
            kwargs,
            model,
            options.model_parameters,
            enable_grounding=options.enable_grounding,
            contains_audio_or_video_input=has_media_inputs,
        )

    # Merge extra headers if provided (after provider-specific headers)
    if options.extra_headers and "extra_headers" not in kwargs:
        kwargs["extra_headers"] = options.extra_headers
    elif options.extra_headers and "extra_headers" in kwargs:
        kwargs["extra_headers"] = {
            **kwargs["extra_headers"],
            **options.extra_headers,
        }

    return kwargs
