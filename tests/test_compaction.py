from __future__ import annotations

from typing import cast

import pytest

from llmcord.logic.compaction import (
    SUMMARIZATION_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    UPDATE_SUMMARIZATION_PROMPT,
    ModelRunConfig,
    build_token_budget,
    compact_messages,
    resolve_context_window,
)
from llmcord.logic.utils import count_conversation_tokens
from llmcord.services.search.decider import DeciderRunConfig, _run_decider_once


def test_resolve_context_window_uses_known_model_limits() -> None:
    assert resolve_context_window(provider="openai-codex", model="gpt-5.4") == 257000
    assert (
        resolve_context_window(
            provider="gemini",
            model="gemini-3.1-flash-lite-preview-low",
        )
        == 249000
    )
    assert resolve_context_window(provider="openrouter", model="free") == 199000
    assert resolve_context_window(provider="gemini", model="gemma-3-27b-it") == 14000
    assert (
        resolve_context_window(
            provider="mistral",
            model="mistral-large-latest",
        )
        == 255000
    )


def test_resolve_context_window_applies_base_limit_to_aliases() -> None:
    configured_limits = {
        "gpt-5.4": 260000,
        "gemini-3.1-flash-preview": 240000,
    }

    assert (
        resolve_context_window(
            provider="openai-codex",
            model="gpt-5.4-xhigh",
            configured_token_limits=configured_limits,
        )
        == 260000
    )
    assert (
        resolve_context_window(
            provider="gemini",
            model="gemini-3.1-flash-preview-high",
            configured_token_limits=configured_limits,
        )
        == 240000
    )


def test_resolve_context_window_supports_future_models_from_config() -> None:
    configured_limits = {"acme/my-future-model": 64000}

    assert (
        resolve_context_window(
            provider="acme",
            model="my-future-model",
            provider_slash_model="acme/my-future-model",
            configured_token_limits=configured_limits,
        )
        == 64000
    )


def test_resolve_context_window_prefers_model_override() -> None:
    assert (
        resolve_context_window(
            provider="mistral",
            model="mistral-large-latest",
            model_parameters={"context_window_tokens": 32000},
        )
        == 32000
    )


def test_compaction_prompts_are_domain_agnostic() -> None:
    assert "any domain" in SUMMARIZATION_SYSTEM_PROMPT
    assert "coding, research, planning, writing, support, or general Q&A" in (
        SUMMARIZATION_PROMPT
    )
    normalized_prompt = " ".join(SUMMARIZATION_PROMPT.split())
    normalized_update_prompt = " ".join(UPDATE_SUMMARIZATION_PROMPT.split())
    expected_fragment = (
        "Preserve exact names, identifiers, URLs, file paths, commands, "
        "function names, error messages, dates, numbers, and user "
        "preferences when they matter."
    )
    assert expected_fragment in normalized_prompt
    assert expected_fragment in normalized_update_prompt


@pytest.mark.asyncio
async def test_compact_messages_inserts_summary_and_fits_budget() -> None:
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "Follow the user's instructions."},
        {"role": "user", "content": "A" * 12000},
        {"role": "assistant", "content": "B" * 12000},
        {"role": "user", "content": "Latest request: keep the recent context."},
    ]

    result = await compact_messages(
        messages,
        config=ModelRunConfig(
            provider="gemini",
            model="gemma-3-27b-it",
            api_keys=["dummy-key"],
            base_url=None,
            extra_headers=None,
            model_parameters={"context_window_tokens": 4000},
            configured_token_limits={"gemma-3-27b-it": 8000},
        ),
        use_model_summary=False,
    )

    target_budget = build_token_budget(
        resolve_context_window(
            provider="gemini",
            model="gemma-3-27b-it",
            model_parameters={"context_window_tokens": 4000},
        ),
    ).target_input_tokens

    assert result.was_compacted is True
    assert result.final_tokens <= target_budget
    assert "⚠️ Older conversation context was compacted to fit the model." in (
        result.warnings
    )
    assert any(
        message.get("role") == "assistant"
        and "Earlier conversation summary (auto-compacted):"
        in str(message.get("content", ""))
        for message in result.messages
    )


@pytest.mark.asyncio
async def test_compact_messages_uses_model_summary_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_summarize_messages_with_model(
        _messages: list[dict[str, object]],
        **_kwargs: object,
    ) -> str:
        return "## Goal\nKeep working.\n\n## Constraints & Preferences\n- (none)"

    monkeypatch.setattr(
        "llmcord.logic.compaction._summarize_messages_with_model",
        _fake_summarize_messages_with_model,
    )

    messages: list[dict[str, object]] = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "A" * 12000},
        {"role": "assistant", "content": "B" * 12000},
        {"role": "user", "content": "Latest request: respond normally."},
    ]

    result = await compact_messages(
        messages,
        config=ModelRunConfig(
            provider="openai-codex",
            model="gpt-5.4",
            api_keys=["codex-token"],
            base_url="https://chatgpt.com/backend-api",
            extra_headers=None,
            model_parameters={"context_window_tokens": 4000},
            configured_token_limits={"gpt-5.4": 12000},
        ),
    )

    assert result.used_model_summary is True
    assert any(
        "## Goal\nKeep working." in str(message.get("content", ""))
        for message in result.messages
    )


@pytest.mark.asyncio
async def test_compact_messages_trims_oversized_single_message() -> None:
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "X" * 80000},
    ]

    result = await compact_messages(
        messages,
        config=ModelRunConfig(
            provider="gemini",
            model="gemma-3-27b-it",
            api_keys=["dummy-key"],
            base_url=None,
            extra_headers=None,
            model_parameters=None,
            configured_token_limits=None,
        ),
        use_model_summary=False,
    )

    target_budget = build_token_budget(
        resolve_context_window(provider="gemini", model="gemma-3-27b-it"),
    ).target_input_tokens

    assert result.final_tokens <= target_budget
    assert result.trimmed_content is True
    assert "⚠️ Some message content was trimmed to fit the model." in result.warnings


@pytest.mark.asyncio
async def test_search_decider_compacts_messages_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_messages: list[dict[str, object]] = []

    async def _fake_get_decider_response_text(**kwargs: object) -> str:
        nonlocal captured_messages
        captured_messages = list(
            cast("list[dict[str, object]]", kwargs["litellm_messages"]),
        )
        return '{"needs_search":false}'

    monkeypatch.setattr(
        "llmcord.services.search.decider._get_decider_response_text",
        _fake_get_decider_response_text,
    )

    result, exhausted = await _run_decider_once(
        [
            {"role": "user", "content": "Latest question about the current topic."},
            {"role": "assistant", "content": "B" * 24000},
            {"role": "user", "content": "A" * 24000},
        ],
        DeciderRunConfig(
            provider="openrouter",
            model="free",
            api_keys=["key"],
            base_url="https://openrouter.ai/api/v1",
            extra_headers=None,
            model_parameters=None,
        ),
    )

    decider_budget = build_token_budget(
        resolve_context_window(provider="openrouter", model="free"),
    ).target_input_tokens

    assert exhausted is False
    assert result == {"needs_search": False}
    assert captured_messages
    assert count_conversation_tokens(captured_messages) <= decider_budget
