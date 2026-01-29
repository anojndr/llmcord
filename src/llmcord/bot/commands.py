"""Slash commands and related helpers for the bot."""
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import discord
from discord.app_commands import Choice

from llmcord.bad_keys import get_bad_keys_db
from llmcord.bot.app import config, discord_bot, get_channel_locked_model
from llmcord.config import get_config

logger = logging.getLogger(__name__)

# =============================================================================
# DRY Helper Functions for Slash Commands
# =============================================================================

GetModelFn = Callable[[str], str | None]
SetModelFn = Callable[[str, str], None]
GetDefaultFn = Callable[[], str | None]


@dataclass(frozen=True)
class ModelSwitchHandlers:
    """Group model switch handlers for a command."""

    get_current: GetModelFn
    set_model: SetModelFn
    get_default: GetDefaultFn


@dataclass(frozen=True)
class ModelAutocompleteHandlers:
    """Group model autocomplete handlers for a command."""

    get_current: GetModelFn
    get_default: GetDefaultFn


async def _handle_model_switch(
    interaction: discord.Interaction,
    model: str,
    handlers: ModelSwitchHandlers,
    model_type_label: str = "model",
) -> None:
    """Handle model switching for slash commands.

    DRY: Consolidate the shared logic between /model and /searchdecidermodel commands.

    Args:
        interaction: Discord interaction object.
        model: The model to switch to.
        handlers: Handler group for getting/setting model values.
        model_type_label: Label for logging/messages (e.g., "model").

    """
    user_id = str(interaction.user.id)

    if model not in config["models"]:
        await interaction.followup.send(
            f"❌ Model `{model}` is not configured in `config.yaml`.",
            ephemeral=True,
        )
        return

    # Get user's current model preference (or default)
    current_user_model = handlers.get_current(user_id)
    default_model = handlers.get_default()

    if current_user_model is None:
        current_user_model = default_model

    if current_user_model is None:
        await interaction.followup.send(
            (
                f"❌ No valid {model_type_label} configured. "
                "Please ask an administrator to check `config.yaml`."
            ),
            ephemeral=True,
        )
        return

    if model == current_user_model:
        output = f"Your current {model_type_label}: `{current_user_model}`"
    else:
        handlers.set_model(user_id, model)
        output = f"Your {model_type_label} switched to: `{model}`"
        logger.info(
            "User %s switched %s to: %s",
            user_id,
            model_type_label,
            model,
        )

    await interaction.followup.send(output)


def _build_model_autocomplete(
    curr_str: str,
    handlers: ModelAutocompleteHandlers,
    user_id: str,
    config_data: Mapping[str, Any],
) -> list[Choice[str]]:
    """Build model autocomplete choices.

    DRY: Consolidate the shared autocomplete logic between commands.

    Args:
        curr_str: Current search string from user input.
        handlers: Handler group for getting default/current models.
        user_id: User ID string.
        config_data: Configuration mapping to source models from.

    Returns:
        List of discord.app_commands.Choice objects.

    """
    user_model = handlers.get_current(user_id)
    default_model = handlers.get_default()

    if user_model is None:
        user_model = default_model

    # Validate that user's saved model still exists in config
    if not user_model or user_model not in config_data.get("models", {}):
        user_model = default_model or next(iter(config_data.get("models", {})), "")

    if not user_model:
        return []

    choices = (
        [Choice(name=f"◉ {user_model} (current)", value=user_model)]
        if curr_str.lower() in user_model.lower()
        else []
    )
    choices += [
        Choice(name=f"○ {m}", value=m)
        for m in config_data["models"]
        if m != user_model and curr_str.lower() in m.lower()
    ]

    return choices[:25]


# =============================================================================
# Slash Commands
# =============================================================================

@discord_bot.tree.command(
    name="model",
    description="View or switch your current model",
)
async def model_command(interaction: discord.Interaction, model: str) -> None:
    """Handle the /model command."""
    await interaction.response.defer(
        ephemeral=(interaction.channel.type == discord.ChannelType.private),
    )

    # Check if this channel has a locked model
    channel_id = interaction.channel_id
    locked_model = get_channel_locked_model(channel_id)
    if locked_model:
        await interaction.followup.send(
            f"❌ This channel is locked to model `{locked_model}`. "
            f"The /model command is disabled here.",
            ephemeral=True,
        )
        return

    db = get_bad_keys_db()

    def get_default() -> str | None:
        return next(iter(config.get("models", {})), None)

    handlers = ModelSwitchHandlers(
        get_current=db.get_user_model,
        set_model=db.set_user_model,
        get_default=get_default,
    )
    await _handle_model_switch(
        interaction=interaction,
        model=model,
        handlers=handlers,
        model_type_label="model",
    )


@model_command.autocomplete("model")
async def model_autocomplete(
    interaction: discord.Interaction,
    curr_str: str,
) -> list[Choice[str]]:
    """Provide autocomplete for /model."""
    config_data = get_config() if not curr_str else config

    db = get_bad_keys_db()
    user_id = str(interaction.user.id)

    def get_default() -> str | None:
        return next(iter(config_data.get("models", {})), None)

    handlers = ModelAutocompleteHandlers(
        get_current=db.get_user_model,
        get_default=get_default,
    )
    return _build_model_autocomplete(curr_str, handlers, user_id, config_data)


@discord_bot.tree.command(
    name="searchdecidermodel",
    description="View or switch your search decider model",
)
async def search_decider_model_command(
    interaction: discord.Interaction,
    model: str,
) -> None:
    """Handle the /searchdecidermodel command."""
    await interaction.response.defer(
        ephemeral=(interaction.channel.type == discord.ChannelType.private),
    )

    db = get_bad_keys_db()

    def get_default() -> str | None:
        default = config.get(
            "web_search_decider_model",
            "gemini/gemini-3-flash-preview",
        )
        return default if default in config.get("models", {}) else None

    handlers = ModelSwitchHandlers(
        get_current=db.get_user_search_decider_model,
        set_model=db.set_user_search_decider_model,
        get_default=get_default,
    )
    await _handle_model_switch(
        interaction=interaction,
        model=model,
        handlers=handlers,
        model_type_label="search decider model",
    )


@search_decider_model_command.autocomplete("model")
async def search_decider_model_autocomplete(
    interaction: discord.Interaction,
    curr_str: str,
) -> list[Choice[str]]:
    """Provide autocomplete for /searchdecidermodel."""
    config_data = get_config() if curr_str == "" else config

    db = get_bad_keys_db()
    user_id = str(interaction.user.id)

    def get_default() -> str | None:
        default = config_data.get(
            "web_search_decider_model",
            "gemini/gemini-3-flash-preview",
        )
        if default in config_data.get("models", {}):
            return default
        return next(iter(config_data.get("models", {})), "") or None

    handlers = ModelAutocompleteHandlers(
        get_current=db.get_user_search_decider_model,
        get_default=get_default,
    )
    return _build_model_autocomplete(curr_str, handlers, user_id, config_data)


@discord_bot.tree.command(
    name="resetallpreferences",
    description="[Owner] Reset all users' model preferences",
)
async def reset_all_preferences_command(interaction: discord.Interaction) -> None:
    """Handle the /resetallpreferences command."""
    # Only allow the bot owner to use this command
    owner_user_id = 676735636656357396
    if interaction.user.id != owner_user_id:
        await interaction.response.send_message(
            "❌ This command can only be used by the bot owner.",
            ephemeral=True,
        )
        return

    # Defer the response since database operations may take time
    await interaction.response.defer(ephemeral=True)

    db = get_bad_keys_db()

    # Reset both preferences
    model_count = db.reset_all_user_model_preferences()
    decider_count = db.reset_all_user_search_decider_preferences()

    await interaction.followup.send(
        f"✅ Successfully reset all user preferences:\n"
        f"• **Main model preferences**: {model_count} user(s) reset\n"
        f"• **Search decider model preferences**: {decider_count} user(s) reset\n\n"
        f"All users will now use the default models.",
    )
    logger.info(
        "Owner %s reset all user preferences (models: %s, deciders: %s)",
        interaction.user.id,
        model_count,
        decider_count,
    )
