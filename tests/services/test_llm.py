"""Tests for LLM helpers and LiteLLM kwargs preparation."""

from __future__ import annotations

from llmcord.services.llm import (
    LiteLLMOptions,
    build_litellm_model_name,
    prepare_litellm_kwargs,
)


def assert_true(*, condition: bool, message: str) -> None:
    """Raise an AssertionError when a condition is false."""
    if not condition:
        raise AssertionError(message)


def test_build_litellm_model_name() -> None:
    """Model name mapping should match provider conventions."""
    assert_true(
        condition=build_litellm_model_name("gemini", "gemini-1.5-pro")
        == "gemini/gemini-1.5-pro",
        message="Expected gemini provider prefix",
    )
    assert_true(
        condition=build_litellm_model_name("openai", "gpt-4") == "gpt-4",
        message="Expected openai model passthrough",
    )
    assert_true(
        condition=build_litellm_model_name("github_copilot", "gpt-4")
        == "github_copilot/gpt-4",
        message="Expected github_copilot provider prefix",
    )
    assert_true(
        condition=build_litellm_model_name("mistral", "large") == "mistral/large",
        message="Expected mistral provider prefix",
    )

def test_prepare_litellm_kwargs_basic() -> None:
    """Basic kwargs should include model, messages, and api_key."""
    kwargs = prepare_litellm_kwargs(
        provider="openai",
        model="gpt-4",
        messages=[{"role": "user", "content": "hello"}],
        api_key="sk-test",
    )
    assert_true(condition=kwargs["model"] == "gpt-4", message="Expected model")
    assert_true(
        condition=kwargs["messages"] == [{"role": "user", "content": "hello"}],
        message="Expected messages passthrough",
    )
    assert_true(
        condition=kwargs["api_key"] == "sk-test",
        message="Expected api_key passthrough",
    )

def test_prepare_litellm_kwargs_gemini_grounding() -> None:
    """Gemini grounding option should add googleSearch tool."""
    kwargs = prepare_litellm_kwargs(
        provider="gemini",
        model="gemini-1.5-pro",
        messages=[],
        api_key="test",
        options=LiteLLMOptions(enable_grounding=True),
    )
    assert_true(
        condition=kwargs["model"] == "gemini/gemini-1.5-pro",
        message="Expected gemini model prefix",
    )
    # Tools should include googleSearch
    assert_true(
        condition=any("googleSearch" in tool for tool in kwargs.get("tools", [])),
        message="Expected googleSearch tool",
    )

def test_prepare_litellm_kwargs_gemini_thinking() -> None:
    """Gemini thinking level should map to reasoning_effort."""
    kwargs = prepare_litellm_kwargs(
        provider="gemini",
        model="gemini-3-flash-preview",
        messages=[],
        api_key="test",
        options=LiteLLMOptions(model_parameters={"thinking_level": "HIGH"}),
    )
    assert_true(
        condition=kwargs["reasoning_effort"] == "high",
        message="Expected reasoning effort to be high",
    )

def test_prepare_litellm_kwargs_base_url() -> None:
    """Base URL should be passed for OpenAI providers."""
    kwargs = prepare_litellm_kwargs(
        provider="openai",
        model="gpt-4",
        messages=[],
        api_key="test",
        options=LiteLLMOptions(base_url="https://api.custom.com"),
    )
    assert_true(
        condition=kwargs["base_url"] == "https://api.custom.com",
        message="Expected base_url passthrough",
    )

def test_prepare_litellm_kwargs_base_url_ignored_for_gemini() -> None:
    """Gemini providers should ignore base_url settings."""
    # Gemini uses specific API endpoints; base_url is ignored for gemini.
    kwargs = prepare_litellm_kwargs(
        provider="gemini",
        model="gemini-1.5-pro",
        messages=[],
        api_key="test",
        options=LiteLLMOptions(base_url="https://api.custom.com"),
    )
    assert_true(
        condition="base_url" not in kwargs,
        message="Expected base_url to be omitted for gemini",
    )
