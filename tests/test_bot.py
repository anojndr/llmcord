"""Tests for bot command handlers and processing utilities."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, TypeVar
from unittest.mock import AsyncMock, Mock, patch

import pytest

from llmcord.commands import (
    model_autocomplete,
    model_command,
    reset_all_preferences_command,
    search_decider_model_command,
)
from llmcord.processing import (
    _handle_retry_request,
    _process_user_message,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable

pytestmark = pytest.mark.usefixtures("mock_dependencies")

T = TypeVar("T")


def run_async(coro: Awaitable[T]) -> T:
    """Run an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def assert_true(*, condition: bool, message: str) -> None:
    """Raise an AssertionError when a condition is false."""
    if not condition:
        raise AssertionError(message)


def test_model_command_valid(
    mock_interaction: Mock,
) -> None:
    """Model command should report current model when valid."""
    run_async(model_command.callback(mock_interaction, model="gpt-4"))

    # Since gpt-4 is default and only model, it says "Your current model"
    mock_interaction.followup.send.assert_called_with("Your current model: `gpt-4`")


def test_model_command_invalid(
    mock_interaction: Mock,
) -> None:
    """Invalid model should report not configured message."""
    run_async(model_command.callback(mock_interaction, model="invalid-model"))

    args, _kwargs = mock_interaction.followup.send.call_args
    assert_true(
        condition="is not configured" in args[0],
        message="Expected not configured message",
    )


def test_model_autocomplete(
    mock_interaction: Mock,
) -> None:
    """Autocomplete should return choices with current model first."""
    choices = run_async(model_autocomplete(mock_interaction, curr_str="gpt"))
    assert_true(
        condition=len(choices) > 0,
        message="Expected at least one choice",
    )
    # gpt-4 is current, so it starts with ◉
    assert_true(
        condition=choices[0].name.startswith("◉ gpt"),
        message="Expected current model indicator",
    )


def test_search_decider_model_command(
    mock_interaction: Mock,
) -> None:
    """Search decider model command should confirm update message."""
    run_async(search_decider_model_command.callback(mock_interaction, model="gpt-4"))
    args, _kwargs = mock_interaction.followup.send.call_args
    # It might say "Your search decider model switched to..." or similar
    assert_true(
        condition="search decider model" in args[0],
        message="Expected search decider model message",
    )


def test_reset_all_preferences_command_not_owner(mock_interaction: Mock) -> None:
    """Non-owner should be blocked from reset-all command."""
    mock_interaction.user.id = 99999  # Not owner
    run_async(reset_all_preferences_command.callback(mock_interaction))
    mock_interaction.response.send_message.assert_called_with(
        "❌ This command can only be used by the bot owner.",
        ephemeral=True,
    )


def test_retry_request_different_user(mock_interaction: Mock) -> None:
    """Retry request should be rejected for different user."""
    mock_interaction.user.id = 123
    run_async(
        _handle_retry_request(
            mock_interaction,
            111,
            456,
        ),
    )

    args, _ = mock_interaction.followup.send.call_args
    assert_true(
        condition="cannot retry this message" in args[0],
        message="Expected retry rejection message",
    )


def test_process_user_message_success(
    mock_message: Mock,
) -> None:
    """Process user message should call processing pipeline."""
    with patch(
        "llmcord.processing.process_message",
        new_callable=AsyncMock,
    ) as mock_process:
        run_async(_process_user_message(mock_message))
        mock_process.assert_called_once()
