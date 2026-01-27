from __future__ import annotations

import litellm_utils


def test_build_litellm_model_name() -> None:
    assert (
        litellm_utils.build_litellm_model_name("gemini", "gemini-3-flash-preview")
        == "gemini/gemini-3-flash-preview"
    )
    assert (
        litellm_utils.build_litellm_model_name("github_copilot", "gpt-4")
        == "github_copilot/gpt-4"
    )
    assert (
        litellm_utils.build_litellm_model_name("mistral", "mistral-small")
        == "mistral/mistral-small"
    )
    assert (
        litellm_utils.build_litellm_model_name("openai", "gpt-4.1")
        == "gpt-4.1"
    )


def test_prepare_litellm_kwargs_gemini_grounding() -> None:
    options = litellm_utils.LiteLLMOptions(
        enable_grounding=True,
        model_parameters={"temperature": 0.2},
    )
    kwargs = litellm_utils.prepare_litellm_kwargs(
        provider="gemini",
        model="gemini-1.5-pro",
        messages=[{"role": "user", "content": "hi"}],
        api_key="key",
        options=options,
    )

    assert kwargs["model"] == "gemini/gemini-1.5-pro"
    assert "tools" in kwargs


def test_prepare_litellm_kwargs_openai_base_url() -> None:
    options = litellm_utils.LiteLLMOptions(base_url="https://example.com")
    kwargs = litellm_utils.prepare_litellm_kwargs(
        provider="openai",
        model="gpt-4.1",
        messages=[{"role": "user", "content": "hi"}],
        api_key="key",
        options=options,
    )

    assert kwargs["base_url"] == "https://example.com"


def test_prepare_litellm_kwargs_copilot_headers(monkeypatch) -> None:
    def fake_configure_token(_: str) -> None:
        return None

    monkeypatch.setattr(litellm_utils, "configure_github_copilot_token", fake_configure_token)

    options = litellm_utils.LiteLLMOptions(extra_headers={"X-Test": "1"})
    kwargs = litellm_utils.prepare_litellm_kwargs(
        provider="github_copilot",
        model="gpt-4",
        messages=[{"role": "user", "content": "hi"}],
        api_key="token",
        options=options,
    )

    assert "extra_headers" in kwargs
    assert kwargs["extra_headers"]["X-Test"] == "1"
    assert "Copilot-Integration-Id" in kwargs["extra_headers"]
