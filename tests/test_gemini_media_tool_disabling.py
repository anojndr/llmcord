from llmcord.services.llm.core import prepare_litellm_kwargs
from llmcord.services.llm.types import LiteLLMOptions


def test_gemini_tools_enabled_without_audio_video_inputs() -> None:
    kwargs = prepare_litellm_kwargs(
        provider="gemini",
        model="gemini-2.0-flash",
        messages=[{"role": "user", "content": "hello"}],
        api_key="test-key",
        options=LiteLLMOptions(enable_grounding=True),
    )

    assert isinstance(kwargs.get("tools"), list)


def test_gemini_tools_disabled_with_video_file_input() -> None:
    kwargs = prepare_litellm_kwargs(
        provider="gemini",
        model="gemini-2.0-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "summarize this video"},
                    {
                        "type": "file",
                        "file": {"file_data": "data:video/mp4;base64,AAAA"},
                    },
                ],
            },
        ],
        api_key="test-key",
        options=LiteLLMOptions(enable_grounding=True),
    )

    assert "tools" not in kwargs


def test_gemini_reasoning_alias_strips_suffix_for_hyphenated_model() -> None:
    kwargs = prepare_litellm_kwargs(
        provider="gemini",
        model="gemini-3.1-flash-lite-preview-minimal",
        messages=[{"role": "user", "content": "hello"}],
        api_key="test-key",
        options=LiteLLMOptions(),
    )

    assert kwargs["model"] == "gemini/gemini-3.1-flash-lite-preview"
    assert kwargs["reasoning_effort"] == "minimal"
