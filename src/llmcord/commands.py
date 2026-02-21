"""Compatibility re-exports for command handlers."""

from llmcord.core.config import get_config
from llmcord.discord.commands import (
    humanize_command,
    model_autocomplete,
    model_command,
    reset_all_preferences_command,
    search_decider_model_command,
)

config = get_config()

__all__ = [
    "config",
    "humanize_command",
    "model_autocomplete",
    "model_command",
    "reset_all_preferences_command",
    "search_decider_model_command",
]
