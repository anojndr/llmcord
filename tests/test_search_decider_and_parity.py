from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import cast

import httpx
import pytest

from llmcord.core.exceptions import (
    FIRST_TOKEN_TIMEOUT_SECONDS,
    FirstTokenTimeoutError,
)
from llmcord.logic.search_logic import SearchResolutionContext, resolve_search_metadata
from llmcord.services.search.decider import (
    DeciderRunConfig,
    _get_decider_response_text,
    _google_gemini_cli_chunk_has_token,
    _iter_stream_with_first_chunk,
    _run_decider_once,
    decide_web_search,
)
from llmcord.services.search.utils import convert_messages_to_openai_format

from ._fakes import FakeMessage, FakeUser

EXPECTED_WEB_SEARCH_MAX_CHARS = 4000


def _assert_with_trailing_newline_tolerance(actual: str, expected: str) -> None:
    """Assert exact formatting while allowing only trailing newline variance."""
    normalized = actual.rstrip("\n")
    assert normalized == expected
    trailing = actual[len(normalized) :]
    assert set(trailing).issubset({"\n"})


class _FakeDB:
    def get_message_search_data(self, _message_id: str) -> tuple[None, None, None]:
        return None, None, None

    def get_user_search_decider_model(self, _user_id: str) -> None:
        return None

    def save_message_search_data(self, *_args: object, **_kwargs: object) -> None:
        return None


class _DummyBot:
    def __init__(self, user_id: int = 999) -> None:
        self.user = FakeUser(user_id)


@pytest.mark.asyncio
async def test_web_search_decider_appends_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_decide_web_search(
        messages: list[dict[str, object]],
        _decider_config: dict,
    ) -> dict[str, object]:
        assert messages, "Expected messages passed to decider"
        return {"needs_search": True, "queries": ["latest news"]}

    async def _fake_perform_web_search(
        queries: list[str],
        *args: object,
        **kwargs: object,
    ):
        assert queries == ["latest news"]
        options = kwargs.get("options")
        assert options is not None
        assert (
            getattr(options, "max_chars_per_url", None) == EXPECTED_WEB_SEARCH_MAX_CHARS
        )
        return "--- Search Results ---\nHeadline: Example", {"provider": "mock"}

    def _fake_is_googlelens_query(*args: object, **kwargs: object) -> bool:
        return False

    monkeypatch.setattr(
        "llmcord.logic.search_logic.decide_web_search",
        _fake_decide_web_search,
    )
    monkeypatch.setattr(
        "llmcord.logic.search_logic.perform_web_search",
        _fake_perform_web_search,
    )
    monkeypatch.setattr("llmcord.logic.search_logic.get_bad_keys_db", lambda: _FakeDB())

    bot = _DummyBot()
    msg = FakeMessage(id=20, content="at ai latest news", author=FakeUser(1234))
    messages: list[dict[str, object]] = [{"role": "user", "content": "latest news"}]

    search_metadata = await resolve_search_metadata(
        SearchResolutionContext(
            new_msg=msg,  # type: ignore[arg-type]
            discord_bot=bot,  # type: ignore[arg-type]
            msg_nodes={},
            messages=messages,
            user_warnings=set(),
            tavily_api_keys=["tvly-TEST"],
            config={
                "web_search_provider": "tavily",
                "web_search_max_chars_per_url": EXPECTED_WEB_SEARCH_MAX_CHARS,
                "web_search_decider_model": "gemini/gemini-3-flash-preview",
                "providers": {"gemini": {"api_key": ["k"]}},
                "models": {},
            },
            web_search_available=True,
            web_search_provider="tavily",
            actual_model="gpt-4o",
        ),
        is_googlelens_query_func=_fake_is_googlelens_query,
    )

    assert search_metadata == {"provider": "mock"}
    _assert_with_trailing_newline_tolerance(
        str(messages[0]["content"]),
        "latest news\n\n--- Search Results ---\nHeadline: Example",
    )


def test_search_formatting_tolerates_only_trailing_newline() -> None:
    expected = "latest news\n\n--- Search Results ---\nHeadline: Example"

    # Allowed: exact output or output with one or more trailing newlines.
    _assert_with_trailing_newline_tolerance(expected, expected)
    _assert_with_trailing_newline_tolerance(f"{expected}\n", expected)
    _assert_with_trailing_newline_tolerance(f"{expected}\n\n", expected)

    # Not allowed: whitespace mutation that changes formatting semantics.
    with pytest.raises(AssertionError):
        _assert_with_trailing_newline_tolerance(
            "latest news\n\n--- Search Results ---\nHeadline:  Example",
            expected,
        )


def test_decider_message_format_preserves_or_describes_files() -> None:
    # Non-Gemini decider: file parts are converted to a text description.
    base_messages: list[dict[str, object]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "summarize this"},
                {
                    "type": "file",
                    "file": {
                        "file_data": "data:application/pdf;base64,AA==",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="},
                },
            ],
        },
    ]

    openai_non_gemini = convert_messages_to_openai_format(
        base_messages,
        system_prompt=None,
        reverse=False,
        is_gemini=False,
    )
    assert openai_non_gemini
    content = openai_non_gemini[0]["content"]
    assert isinstance(content, list)
    assert any(
        part.get("type") == "text" and "pdf" in str(part.get("text", "")).lower()
        for part in content
    )

    # Gemini decider: file parts remain as file parts.
    openai_gemini = convert_messages_to_openai_format(
        base_messages,
        system_prompt=None,
        reverse=False,
        is_gemini=True,
    )
    assert openai_gemini
    gemini_content = openai_gemini[0]["content"]
    assert isinstance(gemini_content, list)
    assert any(part.get("type") == "file" for part in gemini_content)


@pytest.mark.asyncio
async def test_web_search_decider_normalizes_mapping_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_config: dict[str, object] = {}

    async def _fake_decide_web_search(
        _messages: list[dict[str, object]],
        decider_config: dict,
    ) -> dict[str, object]:
        captured_config.update(decider_config)
        return {"needs_search": False}

    def _fake_is_googlelens_query(*args: object, **kwargs: object) -> bool:
        return False

    monkeypatch.setattr(
        "llmcord.logic.search_logic.decide_web_search",
        _fake_decide_web_search,
    )
    monkeypatch.setattr("llmcord.logic.search_logic.get_bad_keys_db", lambda: _FakeDB())

    bot = _DummyBot()
    msg = FakeMessage(id=21, content="at ai hi", author=FakeUser(1234))
    messages: list[dict[str, object]] = [{"role": "user", "content": "hi"}]

    await resolve_search_metadata(
        SearchResolutionContext(
            new_msg=msg,  # type: ignore[arg-type]
            discord_bot=bot,  # type: ignore[arg-type]
            msg_nodes={},
            messages=messages,
            user_warnings=set(),
            tavily_api_keys=["tvly-TEST"],
            config={
                "web_search_provider": "tavily",
                "web_search_decider_model": (
                    "google-gemini-cli/gemini-3-flash-preview-minimal"
                ),
                "providers": {
                    "google-gemini-cli": {
                        "api_key": {
                            "refresh": "refresh-token",
                            "projectId": "project-123",
                        },
                    },
                },
                "models": {
                    "google-gemini-cli/gemini-3-flash-preview-minimal": {},
                },
            },
            web_search_available=True,
            web_search_provider="tavily",
            actual_model="gpt-4o",
        ),
        is_googlelens_query_func=_fake_is_googlelens_query,
    )

    api_keys = captured_config.get("api_keys")
    assert isinstance(api_keys, list)
    assert len(api_keys) == 1
    assert '"refresh":"refresh-token"' in str(api_keys[0])


@pytest.mark.asyncio
async def test_decider_google_gemini_cli_uses_native_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_stream_google_gemini_cli(**kwargs: object):
        assert kwargs.get("model") == "gemini-3-flash-preview-minimal"
        yield '{"needs_search":false}', None, False

    async def _fail_litellm_call(**_kwargs: object) -> object:
        msg = "litellm path should not be used for google-gemini-cli decider"
        raise AssertionError(msg)

    monkeypatch.setattr(
        "llmcord.services.search.decider.stream_google_gemini_cli",
        _fake_stream_google_gemini_cli,
    )
    monkeypatch.setattr(
        "llmcord.services.search.decider.litellm.acompletion",
        _fail_litellm_call,
    )

    result, exhausted = await _run_decider_once(
        [{"role": "user", "content": "hello"}],
        DeciderRunConfig(
            provider="google-gemini-cli",
            model="gemini-3-flash-preview-minimal",
            api_keys=["refresh-token"],
            base_url="https://cloudcode-pa.googleapis.com",
            extra_headers=None,
            model_parameters=None,
        ),
    )

    assert exhausted is False
    assert result == {"needs_search": False}


@pytest.mark.asyncio
async def test_decider_google_antigravity_uses_native_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_stream_google_gemini_cli(**kwargs: object):
        assert kwargs.get("provider_id") == "google-antigravity"
        assert kwargs.get("model") == "gemini-3-pro"
        yield '{"needs_search":false}', None, False

    async def _fail_litellm_call(**_kwargs: object) -> object:
        msg = "litellm path should not be used for google-antigravity decider"
        raise AssertionError(msg)

    monkeypatch.setattr(
        "llmcord.services.search.decider.stream_google_gemini_cli",
        _fake_stream_google_gemini_cli,
    )
    monkeypatch.setattr(
        "llmcord.services.search.decider.litellm.acompletion",
        _fail_litellm_call,
    )

    result, exhausted = await _run_decider_once(
        [{"role": "user", "content": "hello"}],
        DeciderRunConfig(
            provider="google-antigravity",
            model="gemini-3-pro",
            api_keys=["refresh-token"],
            base_url=None,
            extra_headers=None,
            model_parameters=None,
        ),
    )

    assert exhausted is False
    assert result == {"needs_search": False}


@pytest.mark.asyncio
async def test_decider_google_gemini_cli_uses_first_token_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_timeout: int | None = None

    async def _fake_stream_google_gemini_cli(
        **_kwargs: object,
    ):
        yield '{"needs_search":false}', None, False

    async def _fake_iter_stream_with_first_chunk(
        stream_iter: AsyncIterator[tuple[str, object | None, bool]],
        *,
        timeout_seconds: int,
        chunk_has_token: object | None = None,
    ) -> AsyncIterator[tuple[str, object | None, bool]]:
        nonlocal captured_timeout
        captured_timeout = timeout_seconds
        assert chunk_has_token is not None
        async for chunk in stream_iter:
            yield chunk

    monkeypatch.setattr(
        "llmcord.services.search.decider.stream_google_gemini_cli",
        _fake_stream_google_gemini_cli,
    )
    monkeypatch.setattr(
        "llmcord.services.search.decider._iter_stream_with_first_chunk",
        _fake_iter_stream_with_first_chunk,
    )

    response = await _get_decider_response_text(
        run_config=DeciderRunConfig(
            provider="google-gemini-cli",
            model="gemini-3-flash-preview-minimal",
            api_keys=["refresh-token"],
            base_url="https://cloudcode-pa.googleapis.com",
            extra_headers=None,
            model_parameters=None,
        ),
        current_api_key="refresh-token",
        litellm_messages=[{"role": "user", "content": "hello"}],
    )

    assert response == '{"needs_search":false}'
    assert captured_timeout == FIRST_TOKEN_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_decider_iter_stream_timeout_accepts_first_thinking_token() -> None:
    async def _stream() -> AsyncIterator[tuple[str, object | None, bool]]:
        await asyncio.sleep(0.01)
        yield '{"needs_search":', None, True
        await asyncio.sleep(1.1)
        yield "false}", None, False

    received_chunks: list[tuple[str, object | None, bool]] = []
    async for chunk in _iter_stream_with_first_chunk(
        _stream(),
        timeout_seconds=1,
        chunk_has_token=_google_gemini_cli_chunk_has_token,
    ):
        received_chunks.append(cast("tuple[str, object | None, bool]", chunk))
        break

    assert received_chunks == [('{"needs_search":', None, True)]


@pytest.mark.asyncio
async def test_decider_httpx_timeout_marks_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeout_error = httpx.ReadTimeout("timed out")

    async def _fake_get_decider_response_text(**_kwargs: object) -> str:
        raise timeout_error

    monkeypatch.setattr(
        "llmcord.services.search.decider._get_decider_response_text",
        _fake_get_decider_response_text,
    )

    result, exhausted = await _run_decider_once(
        [{"role": "user", "content": "hello"}],
        DeciderRunConfig(
            provider="google-gemini-cli",
            model="gemini-3-flash-preview-minimal",
            api_keys=["refresh-token"],
            base_url="https://cloudcode-pa.googleapis.com",
            extra_headers=None,
            model_parameters=None,
        ),
    )

    assert result is None
    assert exhausted is True


@pytest.mark.asyncio
async def test_decider_retryable_litellm_error_marks_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRateLimitError(Exception):
        def __init__(self) -> None:
            super().__init__("rate limited")

    async def _fake_get_decider_response_text(**_kwargs: object) -> str:
        raise _FakeRateLimitError

    monkeypatch.setattr(
        "llmcord.services.search.decider._get_decider_response_text",
        _fake_get_decider_response_text,
    )
    monkeypatch.setattr(
        "llmcord.services.search.decider.DECIDER_RETRYABLE_EXCEPTIONS",
        (_FakeRateLimitError,),
    )

    result, exhausted = await _run_decider_once(
        [{"role": "user", "content": "hello"}],
        DeciderRunConfig(
            provider="openrouter",
            model="free",
            api_keys=["key"],
            base_url="https://openrouter.ai/api/v1",
            extra_headers=None,
            model_parameters=None,
        ),
    )

    assert result is None
    assert exhausted is True


@pytest.mark.asyncio
async def test_decider_uses_custom_fallback_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted_models: list[str] = []

    async def _fake_runner(
        _messages: list[object],
        run_config: DeciderRunConfig,
    ) -> tuple[dict[str, object] | None, bool]:
        attempted_models.append(f"{run_config.provider}/{run_config.model}")
        if len(attempted_models) == 1:
            return None, True
        return {"needs_search": False}, False

    monkeypatch.setattr(
        "llmcord.services.search.decider._get_decider_runner",
        lambda: _fake_runner,
    )
    monkeypatch.setattr(
        "llmcord.services.search.decider.get_config",
        lambda: {
            "providers": {
                "mistral": {
                    "api_key": ["mistral-key"],
                    "base_url": "https://api.mistral.ai/v1",
                },
            },
            "models": {},
        },
    )

    result = await decide_web_search(
        [{"role": "user", "content": "hello"}],
        {
            "provider": "gemini",
            "model": "gemini-3-flash-preview",
            "api_keys": ["gemini-key"],
            "base_url": None,
            "extra_headers": None,
            "model_parameters": None,
            "fallback_chain": [
                (
                    "mistral",
                    "mistral-large-latest",
                    "mistral/mistral-large-latest",
                ),
            ],
        },
    )

    assert result == {"needs_search": False}
    assert attempted_models == [
        "gemini/gemini-3-flash-preview",
        "mistral/mistral-large-latest",
    ]


@pytest.mark.asyncio
async def test_decider_first_token_timeout_triggers_fallback_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_get_decider_response_text(
        *,
        run_config: DeciderRunConfig,
        **_kwargs: object,
    ) -> str:
        if run_config.provider == "google-gemini-cli":
            raise FirstTokenTimeoutError(
                timeout_seconds=FIRST_TOKEN_TIMEOUT_SECONDS,
            )
        return '{"needs_search":false}'

    monkeypatch.setattr(
        "llmcord.services.search.decider._get_decider_response_text",
        _fake_get_decider_response_text,
    )
    monkeypatch.setattr(
        "llmcord.services.search.decider._get_decider_runner",
        lambda: _run_decider_once,
    )
    monkeypatch.setattr(
        "llmcord.services.search.decider.get_config",
        lambda: {
            "providers": {
                "mistral": {
                    "api_key": ["mistral-key"],
                    "base_url": "https://api.mistral.ai/v1",
                },
            },
            "models": {},
        },
    )

    result = await decide_web_search(
        [{"role": "user", "content": "hello"}],
        {
            "provider": "google-gemini-cli",
            "model": "gemini-3-flash-preview-minimal",
            "api_keys": ["refresh-token"],
            "base_url": "https://cloudcode-pa.googleapis.com",
            "extra_headers": None,
            "model_parameters": None,
            "fallback_chain": [
                (
                    "mistral",
                    "mistral-large-latest",
                    "mistral/mistral-large-latest",
                ),
            ],
        },
    )

    assert result == {"needs_search": False}
