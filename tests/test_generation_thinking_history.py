from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace, TracebackType
from typing import cast

import pytest

from llmcord.core.exceptions import FIRST_TOKEN_TIMEOUT_SECONDS
from llmcord.logic.generation import _get_stream, _stream_response
from llmcord.logic.generation_types import (
    GenerationContext,
    GenerationState,
    StreamConfig,
)


class _TypingContext:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> bool:
        return False


class _DummyChannel:
    def typing(self) -> _TypingContext:
        return _TypingContext()


@pytest.mark.asyncio
async def test_thinking_chunks_hidden_but_preserved_in_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_get_stream(
        **_kwargs: object,
    ) -> AsyncIterator[tuple[str, object | None, object | None, list[object], bool]]:
        yield "private-thought-1", None, None, [], True
        yield "answer ", None, None, [], False
        yield "private-thought-2", None, None, [], True
        yield "done", "stop", None, [], False

    monkeypatch.setattr("llmcord.logic.generation._get_stream", _fake_get_stream)

    async def _reply_helper(**_kwargs: object) -> None:
        return None

    context = cast(
        "GenerationContext",
        SimpleNamespace(
            new_msg=SimpleNamespace(channel=_DummyChannel()),
            tavily_metadata=None,
        ),
    )
    state = GenerationState(
        response_msgs=[],
        response_contents=[],
        input_tokens=0,
        max_message_length=4096,
        embed=None,
        grounding_metadata=None,
        last_edit_time=0.0,
        generated_images=[],
        generated_image_hashes=set(),
        display_model="google-gemini-cli/gemini-3-flash-preview",
    )

    await _stream_response(
        context=context,
        state=state,
        stream_config=StreamConfig(
            provider="google-gemini-cli",
            actual_model="gemini-3-flash-preview",
            api_key="dummy-key",
            base_url=None,
            extra_headers=None,
            model_parameters=None,
        ),
        reply_helper=_reply_helper,
    )

    visible_response = "".join(state.response_contents)
    assert "private-thought-1" not in visible_response
    assert "private-thought-2" not in visible_response
    assert visible_response == "answer done"

    assert state.thought_process == "private-thought-1private-thought-2"
    assert "<thinking>" in state.full_history_response
    assert "private-thought-1" in state.full_history_response
    assert "private-thought-2" in state.full_history_response
    assert "answer " in state.full_history_response
    assert "done" in state.full_history_response


@pytest.mark.asyncio
async def test_google_gemini_cli_stream_uses_first_token_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_timeout: int | None = None

    async def _fake_google_stream(
        **_kwargs: object,
    ) -> AsyncIterator[tuple[str, object | None, bool]]:
        yield "hello", "stop", False

    async def _fake_iter_stream_with_first_chunk(
        stream_iter: AsyncIterator[tuple[str, object | None, bool]],
        *,
        timeout_seconds: int,
    ) -> AsyncIterator[tuple[str, object | None, bool]]:
        nonlocal captured_timeout
        captured_timeout = timeout_seconds
        async for chunk in stream_iter:
            yield chunk

    monkeypatch.setattr(
        "llmcord.logic.generation.stream_google_gemini_cli",
        _fake_google_stream,
    )
    monkeypatch.setattr(
        "llmcord.logic.generation._iter_stream_with_first_chunk",
        _fake_iter_stream_with_first_chunk,
    )

    context = cast(
        "GenerationContext",
        SimpleNamespace(
            messages=[{"role": "user", "content": "hello"}],
            new_msg=SimpleNamespace(content="hello"),
        ),
    )

    stream = _get_stream(
        context=context,
        stream_config=StreamConfig(
            provider="google-gemini-cli",
            actual_model="gemini-3-flash-preview",
            api_key="dummy-key",
            base_url=None,
            extra_headers=None,
            model_parameters=None,
        ),
    )
    chunk = await anext(stream)

    assert chunk[0] == "hello"
    assert chunk[1] == "stop"
    assert chunk[4] is False
    assert captured_timeout == FIRST_TOKEN_TIMEOUT_SECONDS
