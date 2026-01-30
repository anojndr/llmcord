import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import discord
from llmcord.commands import (
    model_command,
    search_decider_model_command,
    model_autocomplete,
    search_decider_model_autocomplete,
    reset_all_preferences_command,
)
from llmcord.processing import (
    _handle_retry_request,
    _process_user_message,
)

def run_async(coro):
    return asyncio.run(coro)

def test_model_command_valid(mock_interaction, mock_dependencies):
    run_async(model_command.callback(mock_interaction, model="gpt-4"))
    
    # Since gpt-4 is default and only model, it says "Your current model"
    mock_interaction.followup.send.assert_called_with("Your current model: `gpt-4`")

def test_model_command_invalid(mock_interaction, mock_dependencies):
    run_async(model_command.callback(mock_interaction, model="invalid-model"))
    
    args, kwargs = mock_interaction.followup.send.call_args
    assert "is not configured" in args[0]

def test_model_autocomplete(mock_interaction, mock_dependencies):
    choices = run_async(model_autocomplete(mock_interaction, curr_str="gpt"))
    assert len(choices) > 0
    # gpt-4 is current, so it starts with ◉
    assert choices[0].name.startswith("◉ gpt")

def test_search_decider_model_command(mock_interaction, mock_dependencies):
    run_async(search_decider_model_command.callback(mock_interaction, model="gpt-4"))
    args, kwargs = mock_interaction.followup.send.call_args
    # It might say "Your search decider model switched to..." or similar
    assert "search decider model" in args[0]

def test_reset_all_preferences_command_not_owner(mock_interaction):
    mock_interaction.user.id = 99999 # Not owner
    run_async(reset_all_preferences_command.callback(mock_interaction))
    mock_interaction.response.send_message.assert_called_with(
        "❌ This command can only be used by the bot owner.",
        ephemeral=True,
    )

def test_retry_request_different_user(mock_interaction):
    mock_interaction.user.id = 123
    run_async(_handle_retry_request(mock_interaction, 111, 456)) # request_user_id=456
    
    args, _ = mock_interaction.followup.send.call_args
    assert "cannot retry this message" in args[0]

def test_process_user_message_success(mock_message, mock_dependencies):
    with patch("llmcord.processing.process_message", new_callable=AsyncMock) as mock_process:
        run_async(_process_user_message(mock_message))
        mock_process.assert_called_once()
