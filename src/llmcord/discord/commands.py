"""Discord slash commands for llmcord."""

import logging
from pathlib import Path

import discord
from discord.app_commands import Choice

from llmcord import config as config_module
from llmcord.discord.ui.utils import build_error_embed
from llmcord.globals import discord_bot
from llmcord.services.database import get_bad_keys_db
from llmcord.services.ytmp3 import Ytmp3Service
from llmcord.utils.common import (
    ModelAutocompleteHandlers,
    ModelSwitchHandlers,
    _build_model_autocomplete,
    _handle_model_switch,
    get_channel_locked_model,
)

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
    ephemeral = False
    if interaction.channel and interaction.channel.type == discord.ChannelType.private:
        ephemeral = True
    await interaction.response.defer(ephemeral=ephemeral)

    # Check if this channel has a locked model
    channel_id = interaction.channel_id
    if channel_id is None:
        return
    locked_model = get_channel_locked_model(channel_id)
    if locked_model:
        await interaction.followup.send(
            embed=build_error_embed(
                (
                    f"This channel is locked to model `{locked_model}`. "
                    "The /model command is disabled here."
                ),
            ),
            ephemeral=True,
        )
        return

    db = get_bad_keys_db()
    config_data = config_module.get_config()

    def get_default() -> str | None:
        return next(iter(config_data.get("models", {})), None)

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
    config_data = config_module.get_config()

    db = get_bad_keys_db()
    config_data = config_module.get_config()
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
    ephemeral = False
    if interaction.channel and interaction.channel.type == discord.ChannelType.private:
        ephemeral = True
    await interaction.response.defer(ephemeral=ephemeral)

    db = get_bad_keys_db()
    config_data = config_module.get_config()

    def get_default() -> str | None:
        default = config_data.get(
            "web_search_decider_model",
            "gemini/gemini-3-flash-preview",
        )
        return default if default in config_data.get("models", {}) else None

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
    config_data = config_module.get_config()

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
async def reset_all_preferences_command(
    interaction: discord.Interaction,
) -> None:
    """Handle the /resetallpreferences command."""
    # Only allow the bot owner to use this command
    owner_user_id = 676735636656357396
    if interaction.user.id != owner_user_id:
        await interaction.response.send_message(
            embed=build_error_embed(
                "This command can only be used by the bot owner.",
            ),
            ephemeral=True,
        )
        return

    # Defer the response since database operations may take time
    if not interaction.response.is_done():
        await interaction.response.defer(ephemeral=True)

    db = get_bad_keys_db()

    # Reset both preferences
    model_count = db.reset_all_user_model_preferences()
    decider_count = db.reset_all_user_search_decider_preferences()

    message_lines = [
        "✅ Successfully reset all user preferences:",
        f"• **Main model preferences**: {model_count} user(s) reset",
        (f"• **Search decider model preferences**: {decider_count} user(s) reset"),
        "",
        "All users will now use the default models.",
    ]
    await interaction.followup.send("\n".join(message_lines))
    logger.info(
        "Owner %s reset all user preferences (models: %s, deciders: %s)",
        interaction.user.id,
        model_count,
        decider_count,
    )


@discord_bot.tree.command(
    name="ytmp3",
    description="Convert a YouTube video to MP3",
)
async def ytmp3_command(interaction: discord.Interaction, url: str) -> None:
    """Handle the /ytmp3 command."""
    await interaction.response.defer(ephemeral=False)

    # Basic URL validation
    if "youtube.com" not in url and "youtu.be" not in url:
        await interaction.followup.send(
            embed=build_error_embed("Please provide a valid YouTube URL."),
        )
        return

    try:
        file_path = await Ytmp3Service.download_audio(url)

        if file_path:
            await interaction.followup.send(
                content=f"✅ Converted: {url}",
                file=discord.File(file_path),
            )
            # Clean up the file after sending
            try:
                Path(file_path).unlink()
            except OSError:
                logger.exception("Failed to remove temp file %s", file_path)
        else:
            await interaction.followup.send(
                embed=build_error_embed(
                    "Could not convert the video. Please try again later.",
                ),
            )

    except (OSError, RuntimeError, ValueError):
        logger.exception("Error in /ytmp3 command")
        await interaction.followup.send(
            embed=build_error_embed(
                (
                    "An error occurred while converting the video. "
                    "Please try again later."
                ),
            ),
        )
