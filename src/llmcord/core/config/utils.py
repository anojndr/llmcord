"""Configuration helper functions."""

import json
from collections.abc import Iterable, Mapping


def is_gemini_model(model: str) -> bool:
    """Check if a model is an actual Gemini model.

    Gemini models have special capabilities like native PDF handling and
    audio/video support, plus grounding tools that Gemma models don't have
    even though they're served via the same Gemini provider.

    Args:
        model: Model name (e.g., "gemini-3-flash-preview", "gemma-3-27b-it")

    Returns:
        True if this is a genuine Gemini model, False for Gemma and other
        models.

    """
    model_lower = model.lower()
    # Gemma models contain "gemma" in their name
    if "gemma" in model_lower:
        return False
    # Gemini models contain "gemini" in their name
    return "gemini" in model_lower


def normalize_api_keys(raw_api_keys: object) -> list[str]:
    """Normalize provider ``api_key`` config into a list of strings."""
    if raw_api_keys is None:
        return []

    if isinstance(raw_api_keys, str):
        return [raw_api_keys]

    if isinstance(raw_api_keys, Mapping):
        return [json.dumps(raw_api_keys, separators=(",", ":"))]

    if not isinstance(raw_api_keys, Iterable):
        return [str(raw_api_keys)]

    normalized: list[str] = []
    for value in raw_api_keys:
        if isinstance(value, str):
            normalized.append(value)
            continue
        if isinstance(value, Mapping):
            normalized.append(json.dumps(value, separators=(",", ":")))
            continue
        normalized.append(str(value))
    return normalized
