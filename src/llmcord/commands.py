import logging

import discord
from discord.app_commands import Choice

from llmcord.config import get_config
from llmcord.globals import config, discord_bot
from llmcord.helpers import (
    ModelAutocompleteHandlers,
    ModelSwitchHandlers,
    _build_model_autocomplete,
    _handle_model_switch,
    get_channel_locked_model,
)
from llmcord.services.database import get_bad_keys_db

logger = logging.getLogger(__name__)


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
