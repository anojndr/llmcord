from __future__ import annotations

import pytest

from llmcord.logic.fallbacks import apply_fallback_config, build_default_fallback_chain

EXPECTED_KEY_COUNT = 2


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
        (
            "google-antigravity",
            "gemini-3-flash-preview-low",
            "gemini/gemini-3-flash-preview-low",
        ),
        (
            "google-antigravity",
            "gemini-3-flash-preview-minimal",
            "gemini/gemini-3-flash-preview-minimal",
        ),
        (
            "google-antigravity",
            "gemini-3-flash-preview-high",
            "gemini/gemini-3-flash-preview-high",
        ),
    ],
)
def test_google_cloud_code_assist_preview_models_use_matching_gemini_first_fallback(
    original_provider: str,
    original_model: str,
    expected_first_fallback: str,
) -> None:
    fallback_chain = build_default_fallback_chain(original_provider, original_model)

    assert fallback_chain[0][2] == expected_first_fallback


def test_non_cli_models_keep_default_first_fallback() -> None:
    fallback_chain = build_default_fallback_chain("gemini", "gemini-2.5-pro")

    assert fallback_chain[0][2] == "openrouter/openrouter/free"


@pytest.mark.parametrize("provider", ["google-gemini-cli", "google-antigravity"])
def test_apply_fallback_config_normalizes_multiple_mapping_api_keys(
    provider: str,
) -> None:
    provider_name, model, provider_model, _base_url, api_keys = apply_fallback_config(
        next_fallback=(provider, "gemini-3-flash-preview-minimal", "unused/model"),
        config={
            "providers": {
                provider: {
                    "api_key": [
                        {
                            "refresh": "refresh-a",
                            "projectId": "project-a",
                        },
                        {
                            "refresh": "refresh-b",
                            "projectId": "project-b",
                        },
                    ],
                },
            },
        },
    )

    assert provider_name == provider
    assert model == "gemini-3-flash-preview-minimal"
    assert provider_model == "unused/model"
    assert len(api_keys) == EXPECTED_KEY_COUNT
    assert all(isinstance(key, str) for key in api_keys)
    assert '"refresh":"refresh-a"' in api_keys[0]
    assert '"refresh":"refresh-b"' in api_keys[1]
