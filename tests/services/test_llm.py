import pytest
from llmcord.services.llm import (
    prepare_litellm_kwargs,
    build_litellm_model_name,
    LiteLLMOptions,
)

def test_build_litellm_model_name():
    assert build_litellm_model_name("gemini", "gemini-1.5-pro") == "gemini/gemini-1.5-pro"
    assert build_litellm_model_name("openai", "gpt-4") == "gpt-4"
    assert build_litellm_model_name("github_copilot", "gpt-4") == "github_copilot/gpt-4"
    assert build_litellm_model_name("mistral", "large") == "mistral/large"

def test_prepare_litellm_kwargs_basic():
    kwargs = prepare_litellm_kwargs(
        provider="openai",
        model="gpt-4",
        messages=[{"role": "user", "content": "hello"}],
        api_key="sk-test"
    )
    assert kwargs["model"] == "gpt-4"
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert kwargs["api_key"] == "sk-test"

def test_prepare_litellm_kwargs_gemini_grounding():
    kwargs = prepare_litellm_kwargs(
        provider="gemini",
        model="gemini-1.5-pro",
        messages=[],
        api_key="test",
        options=LiteLLMOptions(enable_grounding=True)
    )
    assert kwargs["model"] == "gemini/gemini-1.5-pro"
    # Tools should include googleSearch
    assert any("googleSearch" in tool for tool in kwargs.get("tools", []))

def test_prepare_litellm_kwargs_gemini_thinking():
    kwargs = prepare_litellm_kwargs(
        provider="gemini",
        model="gemini-3-flash-preview",
        messages=[],
        api_key="test",
        options=LiteLLMOptions(model_parameters={"thinking_level": "HIGH"})
    )
    assert kwargs["reasoning_effort"] == "high"

def test_prepare_litellm_kwargs_base_url():
    kwargs = prepare_litellm_kwargs(
        provider="openai",
        model="gpt-4",
        messages=[],
        api_key="test",
        options=LiteLLMOptions(base_url="https://api.custom.com")
    )
    assert kwargs["base_url"] == "https://api.custom.com"

def test_prepare_litellm_kwargs_base_url_ignored_for_gemini():
    # Gemini uses specific API endpoints, so base_url is typically ignored or handled differently
    # The code explicitly excludes base_url for gemini
    kwargs = prepare_litellm_kwargs(
        provider="gemini",
        model="gemini-1.5-pro",
        messages=[],
        api_key="test",
        options=LiteLLMOptions(base_url="https://api.custom.com")
    )
    assert "base_url" not in kwargs
