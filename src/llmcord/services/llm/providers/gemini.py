"""Gemini provider implementation."""

from typing import Any


def configure_gemini_kwargs(
    kwargs: dict[str, Any],
    model: str,
    model_parameters: dict | None = None,
    *,
    enable_grounding: bool = False,
) -> None:
    """Configure Gemini-specific kwargs."""
    is_gemini_3 = "gemini-3" in model
    is_preview = "preview" in model

    if model_parameters:
        safety_settings = model_parameters.get("safety_settings")
        if safety_settings is None:
            safety_settings = model_parameters.get("safetySettings")
        if safety_settings is not None:
            kwargs["safety_settings"] = safety_settings

    # Add thinking config for Gemini 3 models
    thinking_level = (
        model_parameters.get("thinking_level") if model_parameters else None
    )

    if not thinking_level:
        if "gemini-3-flash" in model:
            thinking_level = "MINIMAL"
        elif "gemini-3-pro" in model:
            thinking_level = "LOW"

    if thinking_level:
        # Map thinking levels to LiteLLM reasoning_effort
        thinking_map = {
            "MINIMAL": "minimal",
            "LOW": "low",
            "MEDIUM": "medium",
            "HIGH": "high",
        }
        kwargs["reasoning_effort"] = thinking_map.get(
            thinking_level,
            thinking_level.lower(),
        )

    # Add Google Search and URL Context tools for non-preview models
    # when grounding is enabled.
    if enable_grounding and not is_preview:
        kwargs["tools"] = [{"googleSearch": {}}, {"urlContext": {}}]

    # Set temperature for non-Gemini 3 models
    if (
        model_parameters
        and not is_gemini_3
        and "temperature" in model_parameters
        and "temperature" not in kwargs
    ):
        kwargs["temperature"] = model_parameters["temperature"]
