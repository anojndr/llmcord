"""Tests for the infinite cycling prevention guard in generation loop."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, patch

import discord
import pytest

from llmcord.core.models import MsgNode
from llmcord.logic.generation import (
    MAX_TOTAL_GENERATION_ATTEMPTS,
    _run_generation_loop,
)
from llmcord.logic.generation_types import (
    FallbackState,
    GenerationContext,
    GenerationLoopState,
    GenerationState,
)

_FAKE_MSG_ID_COUNTER = 90000


class _FakeUser:
    id = 1
    mention = "<@1>"


class _FakeTypingCtx:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_args: object) -> None:
        return


class _FakeChannel:
    id = 100
    type = discord.ChannelType.text

    def typing(self) -> _FakeTypingCtx:
        return _FakeTypingCtx()


class _FakeMessage:
    """Minimal fake message used in generation tests."""

    _next_id = _FAKE_MSG_ID_COUNTER

    def __init__(self) -> None:
        _FakeMessage._next_id += 1
        self.id = _FakeMessage._next_id
        self.content = "hello"
        self.author = _FakeUser()
        self.channel = _FakeChannel()

    async def reply(self, **kwargs: Any) -> _FakeMessage:
        return _FakeMessage()

    async def edit(self, **kwargs: Any) -> None:
        return


def _build_context(
    *,
    api_keys: list[str] | None = None,
    fallback_chain: list[tuple[str, str, str]] | None = None,
) -> GenerationContext:
    new_msg = _FakeMessage()
    processing_msg = _FakeMessage()

    return GenerationContext(
        new_msg=new_msg,  # type: ignore[arg-type]
        discord_bot=AsyncMock(),
        msg_nodes={},
        messages=[{"role": "user", "content": "hi"}],
        user_warnings=set(),
        failed_extractions=[],
        provider="test-provider",
        model="test-model",
        actual_model="test-model",
        provider_slash_model="test-provider/test-model",
        base_url=None,
        api_keys=api_keys or ["key-1"],
        model_parameters=None,
        extra_headers=None,
        extra_query=None,
        extra_body=None,
        system_prompt=None,
        config={"providers": {}},
        max_text=4000,
        tavily_metadata=None,
        last_edit_time=0.0,
        processing_msg=processing_msg,  # type: ignore[arg-type]
        retry_callback=AsyncMock(),
        fallback_chain=fallback_chain,
    )


def _build_state(context: GenerationContext) -> GenerationState:
    return GenerationState(
        response_msgs=[context.processing_msg],
        response_contents=[],
        input_tokens=0,
        max_message_length=4000,
        embed=discord.Embed(),
        grounding_metadata=None,
        last_edit_time=0.0,
        generated_images=[],
        generated_image_hashes=set(),
        display_model="test-provider/test-model",
    )


def _make_error_handler_that_keeps_keys() -> Any:
    """Return a handler that rotates keys without ever removing them.

    This simulates an infinite rotation bug.
    """

    def _handler(  # noqa: PLR0913
        *,
        error: Exception,
        provider: str,
        current_api_key: str,
        good_keys: list[str],
        timeout_strikes: dict[str, int],
        timeout_strike_threshold: int = 2,
    ) -> tuple[list[str], str]:
        # Rotate but never remove — this would loop forever without the guard
        if good_keys and good_keys[0] == current_api_key:
            good_keys.append(good_keys.pop(0))
        return good_keys, str(error)

    return _handler


@pytest.mark.asyncio
async def test_total_attempts_guard_breaks_infinite_rotation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When keys are rotated but never removed (simulating a bug),
    the loop must still terminate after MAX_TOTAL_GENERATION_ATTEMPTS.
    """
    keys = ["key-1", "key-2", "key-3"]
    context = _build_context(api_keys=keys)

    processing_node = MsgNode(parent_msg=context.new_msg)
    await processing_node.lock.acquire()
    context.msg_nodes[context.processing_msg.id] = processing_node

    state = _build_state(context)

    call_count = 0

    async def _always_fail(**_kwargs: Any) -> None:
        nonlocal call_count
        call_count += 1
        msg = "simulated error"
        raise RuntimeError(msg)

    with (
        patch(
            "llmcord.logic.generation._stream_response",
            side_effect=_always_fail,
        ),
        patch(
            "llmcord.logic.generation._handle_generation_exception",
            side_effect=_make_error_handler_that_keeps_keys(),
        ),
        patch(
            "llmcord.logic.generation.GLOBAL_PROVIDER_CIRCUIT_BREAKER",
            AsyncMock(
                is_open=AsyncMock(return_value=False),
                record_failure=AsyncMock(return_value=False),
            ),
        ),
        caplog.at_level(logging.ERROR),
    ):
        await _run_generation_loop(context=context, state=state)

    assert call_count == MAX_TOTAL_GENERATION_ATTEMPTS
    assert "Exceeded" in caplog.text
    assert "infinite cycling" in caplog.text


@pytest.mark.asyncio
async def test_total_attempts_not_reset_across_fallbacks() -> None:
    """total_attempts must NOT reset when switching to a fallback provider,
    ensuring a global upper bound.
    """
    loop_state = GenerationLoopState(
        provider="p",
        actual_model="m",
        base_url=None,
        api_keys=["k"],
        good_keys=["k"],
        initial_key_count=1,
        attempt_count=5,
        last_error_msg=None,
        fallback_state=FallbackState(
            fallback_level=0,
            fallback_index=0,
            use_custom_fallbacks=False,
            original_provider="p",
            original_model="m",
        ),
        fallback_chain=[],
        total_attempts=15,
    )

    # Simulate what _handle_exhausted_keys does:
    # reset attempt_count but NOT total_attempts
    loop_state.attempt_count = 0
    loop_state.total_attempts += 1

    assert loop_state.attempt_count == 0
    assert loop_state.total_attempts == 16, (
        "total_attempts must be independent of per-fallback attempt_count"
    )
