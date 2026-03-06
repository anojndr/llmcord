from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from llmcord.logic.media_preprocessing import (
    _build_media_request,
    resolve_media_preprocessor_settings,
)


def test_media_request_sends_prompt_and_file_together_for_video() -> None:
    request = _build_media_request(
        content_type="video/mp4",
        content_bytes=b"video-bytes",
    )

    assert len(request) == 1
    message = request[0]
    assert message["role"] == "user"

    content = message["content"]
    assert isinstance(content, list)
    parts = cast("list[dict[str, object]]", content)
    assert len(parts) == 2
    assert parts[0]["type"] == "text"
    assert "Describe this video per timestamp" in str(parts[0]["text"])
    assert parts[1]["type"] == "file"


def test_media_preprocessor_prefers_gemini_flash_lite_preview(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "llmcord.logic.media_preprocessing.get_config",
        lambda: {
            "web_search_decider_model": "gemini/gemini-2.5-pro",
            "models": {
                "gemini/gemini-2.5-pro": {},
                "gemini/gemini-3-flash-preview": {},
            },
        },
    )

    captured_candidates: list[str] = []

    def _fake_resolve_provider_settings_for_model(
        provider_slash_model: str,
        *,
        allow_missing_api_keys: bool = True,
    ) -> object:
        _ = allow_missing_api_keys
        captured_candidates.append(provider_slash_model)
        return SimpleNamespace(
            provider="gemini",
            actual_model=provider_slash_model.split("/", 1)[1],
        )

    monkeypatch.setattr(
        "llmcord.logic.media_preprocessing.resolve_provider_settings_for_model",
        _fake_resolve_provider_settings_for_model,
    )

    settings = resolve_media_preprocessor_settings()

    assert settings is not None
    assert captured_candidates[0] == "gemini/gemini-3.1-flash-lite-preview"
    assert settings.actual_model == "gemini-3.1-flash-lite-preview"
