"""Model helper functions."""


def is_gemini_model(model: str) -> bool:
    """Check if a model is an actual Gemini model.

    Gemini models have special capabilities like native PDF handling and audio/video
    support, plus grounding tools that Gemma models don't have even though they're
    served via the same Gemini provider.

    Args:
        model: Model name (e.g., "gemini-3-flash-preview", "gemma-3-27b-it")

    Returns:
        True if this is a genuine Gemini model, False for Gemma and other models.

    """
    model_lower = model.lower()
    # Gemma models contain "gemma" in their name
    if "gemma" in model_lower:
        return False
    # Gemini models contain "gemini" in their name
    return "gemini" in model_lower
