"""Discord slash commands for llmcord."""

import io
import logging
from functools import partial

import discord
from discord import app_commands
from discord.app_commands import Choice

from llmcord import config as config_module
from llmcord.discord.ui.embed_limits import call_with_embed_limits
from llmcord.discord.ui.utils import build_error_embed
from llmcord.globals import discord_bot
from llmcord.services.database import get_db
from llmcord.services.humanizer import (
    QuillBotHumanizerError,
    humanize_text_with_quillbot,
)
from llmcord.utils.common import (
    ModelAutocompleteHandlers,
    ModelSwitchHandlers,
    _build_model_autocomplete,
    _handle_model_switch,
    get_channel_locked_model,
    get_default_model,
)

logger = logging.getLogger(__name__)

_DISCORD_MESSAGE_CHAR_LIMIT = 2_000


async def _defer_for_channel_visibility(interaction: discord.Interaction) -> None:
    """Defer command responses as ephemeral in DMs and public elsewhere."""
    is_private_channel = (
        interaction.channel is not None
        and interaction.channel.type == discord.ChannelType.private
    )
    await interaction.response.defer(ephemeral=is_private_channel)


def _get_default_search_decider_model(config_data: dict[str, object]) -> str | None:
    """Return the configured decider default, falling back to the first model."""
    models = config_data.get("models")
    default = config_data.get(
        "web_search_decider_model",
        "gemini/gemini-3-flash-preview",
    )
    if isinstance(default, str) and isinstance(models, dict) and default in models:
        return default
    return get_default_model(config_data)


# =============================================================================
# Slash Commands
# =============================================================================


@discord_bot.tree.command(
    name="model",
    description="View or switch your current model",
)
async def model_command(interaction: discord.Interaction, model: str) -> None:
    """Handle the /model command."""
    await _defer_for_channel_visibility(interaction)

    # Check if this channel has a locked model
    channel_id = interaction.channel_id
    if channel_id is None:
        return
    locked_model = get_channel_locked_model(channel_id)
    if locked_model:
        await call_with_embed_limits(
            interaction.followup.send,
            embed=build_error_embed(
                (
                    f"This channel is locked to model `{locked_model}`. "
                    "The /model command is disabled here."
                ),
            ),
            ephemeral=True,
        )
        return

    db = get_db()
    config_data = config_module.get_config()

    handlers = ModelSwitchHandlers(
        get_current=db.get_user_model,
        set_model=db.set_user_model,
        get_default=partial(get_default_model, config_data),
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

    db = get_db()
    user_id = str(interaction.user.id)

    handlers = ModelAutocompleteHandlers(
        get_current=db.get_user_model,
        get_default=partial(get_default_model, config_data),
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
    await _defer_for_channel_visibility(interaction)

    db = get_db()
    config_data = config_module.get_config()

    handlers = ModelSwitchHandlers(
        get_current=db.get_user_search_decider_model,
        set_model=db.set_user_search_decider_model,
        get_default=partial(_get_default_search_decider_model, config_data),
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

    db = get_db()
    user_id = str(interaction.user.id)

    handlers = ModelAutocompleteHandlers(
        get_current=db.get_user_search_decider_model,
        get_default=partial(_get_default_search_decider_model, config_data),
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
        await call_with_embed_limits(
            interaction.response.send_message,
            embed=build_error_embed(
                "This command can only be used by the bot owner.",
            ),
            ephemeral=True,
        )
        return

    # Defer the response since database operations may take time
    if not interaction.response.is_done():
        await interaction.response.defer(ephemeral=True)

    db = get_db()

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
    name="humanize",
    description="Humanize text with QuillBot",
)
@app_commands.describe(text="Text to humanize")
async def humanize_command(
    interaction: discord.Interaction,
    text: app_commands.Range[str, 1, 4_000],
) -> None:
    """Handle the /humanize command."""
    await _defer_for_channel_visibility(interaction)

    try:
        result = await humanize_text_with_quillbot(text)
    except QuillBotHumanizerError as exc:
        await call_with_embed_limits(
            interaction.followup.send,
            embed=build_error_embed(f"Humanize failed: {exc}"),
            ephemeral=True,
        )
        return

    metadata = (
        "QuillBot mode: "
        f"`{result.mode}` | word limit: `{result.word_limit}`"
        f" | segments: `{result.segment_count}`"
    )

    if len(result.text) + len(metadata) + 2 <= _DISCORD_MESSAGE_CHAR_LIMIT:
        await interaction.followup.send(f"{result.text}\n\n{metadata}")
        return

    payload = io.BytesIO(result.text.encode("utf-8"))
    await interaction.followup.send(
        content=metadata,
        file=discord.File(payload, filename="humanized.txt"),
    )
