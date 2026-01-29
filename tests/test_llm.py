from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from llmcord.services import llm


class _Response:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict:
        return self._payload


def test_build_litellm_model_name() -> None:
    assert llm.build_litellm_model_name("gemini", "gemini-3") == "gemini/gemini-3"
    assert (
        llm.build_litellm_model_name("github_copilot", "gpt-4.1")
        == "github_copilot/gpt-4.1"
    )
    assert (
        llm.build_litellm_model_name("mistral", "mistral-large")
        == "mistral/mistral-large"
    )
    assert llm.build_litellm_model_name("openai", "gpt-4o") == "gpt-4o"


def test_prepare_litellm_kwargs_for_gemini() -> None:
    options = llm.LiteLLMOptions(
        stream=True,
        enable_grounding=True,
        temperature=0.4,
        model_parameters={
            "safety_settings": [{"category": "test", "threshold": "NONE"}],
            "thinking_level": "HIGH",
            "temperature": 0.1,
        },
    )

    kwargs = llm.prepare_litellm_kwargs(
        provider="gemini",
        model="gemini-3-flash-preview",
        messages=[{"role": "user", "content": "hi"}],
        api_key="key",
        options=options,
    )

    assert kwargs["model"] == "gemini/gemini-3-flash-preview"
    assert kwargs["stream"] is True
    assert kwargs["temperature"] == 0.4
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["safety_settings"][0]["category"] == "test"
    assert "tools" not in kwargs  # preview models should not add tools


def test_prepare_litellm_kwargs_for_openai() -> None:
    options = llm.LiteLLMOptions(base_url="https://example", extra_headers={"X": "1"})

    kwargs = llm.prepare_litellm_kwargs(
        provider="openai",
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        api_key="key",
        options=options,
    )

    assert kwargs["model"] == "gpt-4o"
    assert kwargs["base_url"] == "https://example"
    assert kwargs["extra_headers"]["X"] == "1"


def test_prepare_litellm_kwargs_gemini_defaults() -> None:
    options = llm.LiteLLMOptions(
        enable_grounding=True,
        model_parameters={"temperature": 0.2},
    )

    kwargs = llm.prepare_litellm_kwargs(
        provider="gemini",
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        api_key="key",
        options=options,
    )

    assert kwargs["temperature"] == 0.2
    assert kwargs["tools"] == [{"googleSearch": {}}, {"urlContext": {}}]


def test_prepare_litellm_kwargs_gemini_pro_defaults() -> None:
    options = llm.LiteLLMOptions(
        model_parameters={
            "safetySettings": [{"category": "safe", "threshold": "LOW"}],
        },
    )

    kwargs = llm.prepare_litellm_kwargs(
        provider="gemini",
        model="gemini-3-pro",
        messages=[{"role": "user", "content": "hi"}],
        api_key="key",
        options=options,
    )

    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["safety_settings"][0]["category"] == "safe"


def test_prepare_litellm_kwargs_custom_thinking_level() -> None:
    options = llm.LiteLLMOptions(
        model_parameters={"thinking_level": "CUSTOM"},
    )

    kwargs = llm.prepare_litellm_kwargs(
        provider="gemini",
        model="gemini-3-flash-preview",
        messages=[{"role": "user", "content": "hi"}],
        api_key="key",
        options=options,
    )

    assert kwargs["reasoning_effort"] == "custom"


def test_prepare_litellm_kwargs_default_thinking_level() -> None:
    kwargs = llm.prepare_litellm_kwargs(
        provider="gemini",
        model="gemini-3-flash-preview",
        messages=[{"role": "user", "content": "hi"}],
        api_key="key",
        options=llm.LiteLLMOptions(),
    )

    assert kwargs["reasoning_effort"] == "minimal"


def test_prepare_litellm_kwargs_for_github_copilot(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, str] = {}

    def fake_configure(token: str) -> None:
        called["token"] = token

    monkeypatch.setattr(llm, "configure_github_copilot_token", fake_configure)

    kwargs = llm.prepare_litellm_kwargs(
        provider="github_copilot",
        model="gpt-4.1",
        messages=[{"role": "user", "content": "hi"}],
        api_key="ghu_123",
        options=llm.LiteLLMOptions(extra_headers={"X": "1"}),
    )

    assert called["token"] == "ghu_123"
    assert kwargs["extra_headers"]["X"] == "1"
    for key in llm.GITHUB_COPILOT_HEADERS:
        assert key in kwargs["extra_headers"]


def test_configure_github_copilot_token_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(tmp_path))
    monkeypatch.setenv("GITHUB_COPILOT_ACCESS_TOKEN_FILE", "token.txt")

    def fake_get(*_args, **_kwargs):
        return _Response(
            200,
            payload={"token": "abc", "expires_at": 123, "endpoints": {"x": "y"}},
        )

    monkeypatch.setattr(requests, "get", fake_get)

    llm.configure_github_copilot_token("ghu_token")

    token_file = tmp_path / "token.txt"
    api_file = tmp_path / "api-key.json"

    assert token_file.exists()
    assert token_file.read_text(encoding="utf-8") == "ghu_token"

    payload = json.loads(api_file.read_text(encoding="utf-8"))
    assert payload["token"] == "abc"
    assert payload["endpoints"]["x"] == "y"


def test_configure_github_copilot_token_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(tmp_path))

    def fake_get(*_args, **_kwargs):
        return _Response(403, payload={}, text="denied")

    monkeypatch.setattr(requests, "get", fake_get)

    llm.configure_github_copilot_token("ghu_token")

    token_file = tmp_path / "access-token"
    assert token_file.exists()
    assert not (tmp_path / "api-key.json").exists()


def test_configure_github_copilot_token_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(tmp_path))

    def fake_get(*_args, **_kwargs):
        raise requests.RequestException("boom")

    monkeypatch.setattr(requests, "get", fake_get)

    llm.configure_github_copilot_token("ghu_token")
    assert (tmp_path / "access-token").exists()
