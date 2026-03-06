"""Helpers for model aliases that encode request-level execution settings."""

GEMINI_THINKING_LEVEL_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("-minimal", "MINIMAL"),
    ("-low", "LOW"),
    ("-medium", "MEDIUM"),
    ("-high", "HIGH"),
)

OPENAI_REASONING_EFFORT_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("-none", "none"),
    ("-minimal", "minimal"),
    ("-low", "low"),
    ("-medium", "medium"),
    ("-high", "high"),
    ("-xhigh", "xhigh"),
)


def extract_model_suffix_alias(
    model: str,
    suffix_aliases: tuple[tuple[str, str], ...],
) -> tuple[str, str | None]:
    """Return the base model name and any execution-setting alias suffix."""
    for suffix, alias in suffix_aliases:
        if model.endswith(suffix):
            return model.removesuffix(suffix), alias
    return model, None


def strip_model_suffix_alias(
    model: str,
    suffix_aliases: tuple[tuple[str, str], ...],
) -> str:
    """Remove a recognized alias suffix from a model name."""
    clean_model, _alias = extract_model_suffix_alias(model, suffix_aliases)
    return clean_model


def extract_suffix_alias(
    model: str,
    suffix_aliases: tuple[tuple[str, str], ...],
) -> str | None:
    """Return only the alias value encoded in a model suffix, if any."""
    _clean_model, alias = extract_model_suffix_alias(model, suffix_aliases)
    return alias
