from __future__ import annotations

import pytest

from llmcord.logic.fallbacks import build_default_fallback_chain


@pytest.mark.parametrize(
    ("original_provider", "original_model", "expected_first_fallback"),
    [
        (
            "google-gemini-cli",
            "gemini-3-flash-preview-low",
            "gemini/gemini-3-flash-preview-low",
        ),
        (
            "google-gemini-cli",
            "gemini-3-flash-preview-minimal",
            "gemini/gemini-3-flash-preview-minimal",
        ),
        (
            "google-gemini-cli",
            "gemini-3-flash-preview-high",
            "gemini/gemini-3-flash-preview-high",
        ),
    ],
)
def test_google_gemini_cli_preview_models_use_matching_gemini_first_fallback(
    original_provider: str,
    original_model: str,
    expected_first_fallback: str,
) -> None:
    fallback_chain = build_default_fallback_chain(original_provider, original_model)

    assert fallback_chain[0][2] == expected_first_fallback


def test_non_cli_models_keep_default_first_fallback() -> None:
    fallback_chain = build_default_fallback_chain("gemini", "gemini-2.5-pro")

    assert fallback_chain[0][2] == "openrouter/openrouter/free"
