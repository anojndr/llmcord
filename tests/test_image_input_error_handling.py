from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import litellm
import pytest

from llmcord.logic.discord_ui import _build_footer_text
from llmcord.logic.generation import (
    GENERATION_EXCEPTIONS,
    _get_stream,
    _is_image_input_error,
    _remove_images_from_messages,
)
from llmcord.logic.generation_types import (
    GenerationContext,
    GenerationState,
    StreamConfig,
)

from ._fakes import FakeMessage, FakeUser


def _build_context_with_history_file_parts() -> GenerationContext:
    return cast(
        "GenerationContext",
        SimpleNamespace(
            new_msg=FakeMessage(id=999, content="latest", author=FakeUser(1)),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {"file_data": "data:audio/mp3;base64,AAAA"},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "prior context"},
                        {
                            "type": "file",
                            "file": {"file_data": "data:video/mp4;base64,BBBB"},
                        },
                    ],
                },
                {"role": "user", "content": "latest query"},
            ],
        ),
    )


def _build_context_with_mixed_pdf_audio_video_history() -> GenerationContext:
    return cast(
        "GenerationContext",
        SimpleNamespace(
            new_msg=FakeMessage(id=1001, content="latest", author=FakeUser(1)),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {
                                "file_data": "data:application/pdf;base64,UEZERkQ=",
                            },
                        },
                        {
                            "type": "file",
                            "file": {"file_data": "data:audio/mp3;base64,AAAA"},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "file",
                            "file": {"file_data": "data:video/mp4;base64,BBBB"},
                        },
                    ],
                },
                {"role": "user", "content": "latest query"},
            ],
        ),
    )


def _assert_no_audio_video_file_parts(messages: list[dict[str, object]]) -> None:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_dict = cast("dict[str, object]", part)
            if part_dict.get("type") != "file":
                continue
            file_obj = part_dict.get("file")
            if not isinstance(file_obj, dict):
                continue
            file_data = cast("dict[str, object]", file_obj).get("file_data")
            assert not (
                isinstance(file_data, str)
                and file_data.startswith(("data:audio/", "data:video/"))
            )


def _collect_pdf_file_parts(messages: list[dict[str, object]]) -> list[str]:
    pdf_file_parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_dict = cast("dict[str, object]", part)
            if part_dict.get("type") != "file":
                continue
            file_obj = part_dict.get("file")
            if not isinstance(file_obj, dict):
                continue
            file_data = cast("dict[str, object]", file_obj).get("file_data")
            if isinstance(file_data, str) and file_data.startswith(
                "data:application/pdf;",
            ):
                pdf_file_parts.append(file_data)
    return pdf_file_parts


async def _capture_stream_messages_for_non_gemini(
    monkeypatch: pytest.MonkeyPatch,
    context: GenerationContext,
) -> list[dict[str, object]]:
    captured_messages: list[dict[str, object]] = []

    class _FakeDelta:
        content = "ok"

    class _FakeChoice:
        delta = _FakeDelta()
        finish_reason = "stop"

    class _FakeChunk:
        choices = (_FakeChoice(),)

    async def _fake_stream():
        yield _FakeChunk()

    async def _fake_acompletion(**kwargs: object):
        messages_obj = kwargs.get("messages")
        if isinstance(messages_obj, list):
            captured_messages.extend(cast("list[dict[str, object]]", messages_obj))
        return _fake_stream()

    def _fake_prepare_litellm_kwargs(**kwargs: object) -> dict[str, object]:
        return {
            "model": "openai/gpt-4o",
            "messages": kwargs["messages"],
            "stream": True,
        }

    monkeypatch.setattr(
        "llmcord.logic.generation.litellm.acompletion",
        _fake_acompletion,
    )
    monkeypatch.setattr(
        "llmcord.logic.generation.prepare_litellm_kwargs",
        _fake_prepare_litellm_kwargs,
    )

    stream_config = StreamConfig(
        provider="openai",
        actual_model="gpt-4o",
        api_key="key",
        base_url=None,
        extra_headers=None,
        model_parameters=None,
    )

    async for _ in _get_stream(
        context=context,
        stream_config=stream_config,
    ):
        break

    return captured_messages


def test_detects_openrouter_image_input_error() -> None:
    error_message = (
        "NotFoundError: OpenrouterException - "
        '{"error":{"message":"No endpoints found that support image input",'
        '"code":404}}'
    )
    error = Exception(
        error_message,
    )
    assert _is_image_input_error(error) is True


def test_removes_images_from_message_content() -> None:
    messages: list[dict[str, object]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                },
            ],
        },
    ]

    removed = _remove_images_from_messages(messages)

    assert removed is True
    assert isinstance(messages[0]["content"], list)
    assert messages[0]["content"] == [{"type": "text", "text": "Describe this"}]


def test_footer_includes_image_removed_warning() -> None:
    state = GenerationState(
        response_msgs=[],
        response_contents=["ok"],
        input_tokens=100,
        max_message_length=4096,
        embed=None,
        use_plain_responses=False,
        grounding_metadata=None,
        last_edit_time=0.0,
        generated_images=[],
        generated_image_hashes=set(),
        display_model="openrouter/openrouter/free",
        fallback_warning=None,
        image_removal_warning="⚠️ Image removed from input due to provider error.",
    )

    footer_text = _build_footer_text(state=state, total_tokens=150)

    assert "openrouter/openrouter/free | total tokens: 150" in footer_text
    assert "Image removed from input due to provider error" in footer_text


def test_not_found_error_is_retryable_generation_exception() -> None:
    assert litellm.exceptions.NotFoundError in GENERATION_EXCEPTIONS


@pytest.mark.asyncio
async def test_non_gemini_stream_strips_audio_video_file_parts_from_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _build_context_with_history_file_parts()
    captured_messages = await _capture_stream_messages_for_non_gemini(
        monkeypatch,
        context,
    )

    assert captured_messages
    _assert_no_audio_video_file_parts(captured_messages)

    first_message_content = captured_messages[-1].get("content")
    assert isinstance(first_message_content, list)
    assert first_message_content == [
        {
            "type": "text",
            "text": (
                "Audio/video attachment omitted because this model does not "
                "support native file input."
            ),
        },
    ]


@pytest.mark.asyncio
async def test_non_gemini_stream_keeps_pdf_while_stripping_audio_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _build_context_with_mixed_pdf_audio_video_history()
    captured_messages = await _capture_stream_messages_for_non_gemini(
        monkeypatch,
        context,
    )

    assert captured_messages
    _assert_no_audio_video_file_parts(captured_messages)

    pdf_file_parts = _collect_pdf_file_parts(captured_messages)
    assert pdf_file_parts
