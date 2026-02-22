from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace, TracebackType
from typing import cast

import pytest

from llmcord.logic.generation import _stream_response
from llmcord.logic.generation_types import (
    GenerationContext,
    GenerationState,
    StreamConfig,
)
from llmcord.services.llm.providers.gemini_errors import (
    HTTP_BAD_REQUEST,
    HTTP_TOO_MANY_REQUESTS,
    classify_gemini_error,
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


def test_classify_gemini_invalid_argument_skips_provider() -> None:
    error = Exception(
        'GeminiError: {"error": {"code": 400, "message": "bad field", '
        '"status": "INVALID_ARGUMENT"}}',
    )
    classification = classify_gemini_error(error)
    assert classification is not None
    assert classification.http_status == HTTP_BAD_REQUEST
    assert classification.api_status == "INVALID_ARGUMENT"
    assert classification.action == "skip_provider"


def test_classify_gemini_leaked_key_removes_key() -> None:
    error = Exception(
        "Your API key was reported as leaked. Please use another API key.",
    )
    classification = classify_gemini_error(error)
    assert classification is not None
    assert classification.action == "remove_key"
    assert "reported as leaked" in classification.message.lower()


def test_classify_gemini_resource_exhausted_removes_key() -> None:
    error = Exception(
        'GeminiError: {"error": {"code": 429, "message": "rate limit", '
        '"status": "RESOURCE_EXHAUSTED"}}',
    )
    classification = classify_gemini_error(error)
    assert classification is not None
    assert classification.http_status == HTTP_TOO_MANY_REQUESTS
    assert classification.api_status == "RESOURCE_EXHAUSTED"
    assert classification.action == "remove_key"


@pytest.mark.asyncio
async def test_gemini_recitation_finish_reason_without_content_is_handled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_get_stream(
        **_kwargs: object,
    ) -> AsyncIterator[tuple[str, object | None, object | None, list[object], bool]]:
        yield "", "RECITATION", None, [], False

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
        display_model="gemini/gemini-3-flash-preview",
    )

    await _stream_response(
        context=context,
        state=state,
        stream_config=StreamConfig(
            provider="gemini",
            actual_model="gemini-3-flash-preview",
            api_key="dummy",
            base_url=None,
            extra_headers=None,
            model_parameters=None,
        ),
        reply_helper=_reply_helper,
    )

    assert state.response_contents
    assert "recitation" in "".join(state.response_contents).lower()
