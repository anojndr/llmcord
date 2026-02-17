from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import cast

import discord
import pytest
from discord import app_commands

from llmcord.core.error_handling import (
    log_discord_event_error,
    register_asyncio_exception_handler,
)
from llmcord.discord.error_handling import (
    edit_processing_message_error,
    handle_app_command_error,
    send_interaction_error,
    send_message_processing_error,
)

EXPECTED_EDIT_CALLS = 2


def _raise_runtime_error() -> None:
    raise RuntimeError


def _raise_value_error() -> None:
    raise ValueError


class _FakeResponse:
    def __init__(self, *, done: bool) -> None:
        self._done = done
        self.calls: list[dict[str, object]] = []

    def is_done(self) -> bool:
        return self._done

    async def send_message(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


class _FakeFollowup:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def send(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


class _FakeInteraction:
    def __init__(self, *, done: bool) -> None:
        self.response = _FakeResponse(done=done)
        self.followup = _FakeFollowup()
        self.command = SimpleNamespace(qualified_name="model")
        self.user = SimpleNamespace(id=1234)
        self.channel_id = 4567
        self.guild_id = 8910


class _FakeMessage:
    def __init__(self) -> None:
        self.reply_calls: list[dict[str, object]] = []
        self.edit_calls: list[dict[str, object]] = []

    async def reply(self, **kwargs: object) -> None:
        self.reply_calls.append(kwargs)

    async def edit(self, **kwargs: object) -> None:
        self.edit_calls.append(kwargs)


@pytest.mark.asyncio
async def test_send_interaction_error_uses_response_before_defer() -> None:
    interaction = _FakeInteraction(done=False)

    await send_interaction_error(cast("discord.Interaction", interaction))

    assert len(interaction.response.calls) == 1
    assert interaction.response.calls[0]["ephemeral"] is True
    assert interaction.followup.calls == []


@pytest.mark.asyncio
async def test_send_interaction_error_uses_followup_after_defer() -> None:
    interaction = _FakeInteraction(done=True)

    await send_interaction_error(cast("discord.Interaction", interaction))

    assert len(interaction.followup.calls) == 1
    assert interaction.followup.calls[0]["ephemeral"] is True
    assert interaction.response.calls == []


@pytest.mark.asyncio
async def test_message_processing_error_helpers_write_expected_payloads() -> None:
    message = _FakeMessage()

    await send_message_processing_error(cast("discord.Message", message))
    await edit_processing_message_error(
        cast("discord.Message", message),
        use_plain_responses=True,
    )
    await edit_processing_message_error(
        cast("discord.Message", message),
        use_plain_responses=False,
    )

    assert len(message.reply_calls) == 1
    assert len(message.edit_calls) == EXPECTED_EDIT_CALLS
    assert message.edit_calls[0]["embed"] is None
    assert message.edit_calls[0]["view"] is None
    assert isinstance(message.edit_calls[1]["embed"], discord.Embed)
    assert message.edit_calls[1]["view"] is None


@pytest.mark.asyncio
async def test_handle_app_command_error_logs_and_responds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    interaction = _FakeInteraction(done=False)
    logger = logging.getLogger("tests.error_handling")

    with caplog.at_level(logging.ERROR, logger=logger.name):
        await handle_app_command_error(
            cast("discord.Interaction", interaction),
            app_commands.AppCommandError("boom"),
            logger=logger,
        )

    assert "Unhandled slash command error" in caplog.text
    assert len(interaction.response.calls) == 1


def test_log_discord_event_error_logs_active_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("tests.discord_events")

    with caplog.at_level(logging.ERROR, logger=logger.name):
        try:
            _raise_runtime_error()
        except RuntimeError:
            log_discord_event_error(
                logger=logger,
                event_name="on_message",
                args=(),
                kwargs={},
            )

    assert "Unhandled Discord event error" in caplog.text


@pytest.mark.asyncio
async def test_register_asyncio_exception_handler_logs_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    logger = logging.getLogger("tests.asyncio_handler")
    register_asyncio_exception_handler(loop, logger=logger)
    handler = loop.get_exception_handler()
    assert handler is not None

    try:
        with caplog.at_level(logging.ERROR, logger=logger.name):
            try:
                _raise_value_error()
            except ValueError as exc:
                handler(loop, {"message": "loop context", "exception": exc})
    finally:
        loop.set_exception_handler(previous_handler)

    assert "loop context" in caplog.text
