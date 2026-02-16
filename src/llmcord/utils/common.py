"""Shared utility helpers for Discord command handling."""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import discord
from discord.app_commands import Choice

from llmcord import config as config_module
from llmcord.discord.ui.embed_limits import call_with_embed_limits
from llmcord.discord.ui.utils import build_error_embed

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


def get_channel_locked_model(channel_id: int) -> str | None:
    """Check whether a channel has a locked model override.

    Args:
        channel_id: The Discord channel ID to check.

    Returns:
        The model name if the channel has a locked model, or None otherwise.

    """
    config_data = config_module.get_config()
    overrides = config_data.get("channel_model_overrides", {})
    # Convert channel_id to string for comparison since YAML may parse keys as
    # ints or strings.
    # or strings.
    return overrides.get(channel_id) or overrides.get(str(channel_id))


async def _handle_model_switch(
    interaction: discord.Interaction,
    model: str,
    handlers: ModelSwitchHandlers,
    model_type_label: str = "model",
) -> None:
    """Handle model switching for slash commands.

    DRY: Consolidate the shared logic between /model and /searchdecidermodel
    commands.

    Args:
        interaction: Discord interaction object.
        model: The model to switch to.
        handlers: Handler group for getting/setting model values.
        model_type_label: Label for logging/messages (e.g., "model").

    """
    user_id = str(interaction.user.id)

    config_data = config_module.get_config()
    if model not in config_data["models"]:
        await call_with_embed_limits(
            interaction.followup.send,
            embed=build_error_embed(
                (f"Model `{model}` is not available. Please choose another model."),
            ),
            ephemeral=True,
        )
        return

    # Get user's current model preference (or default)
    current_user_model = handlers.get_current(user_id)
    default_model = handlers.get_default()

    if current_user_model is None:
        current_user_model = default_model

    if current_user_model is None:
        await call_with_embed_limits(
            interaction.followup.send,
            embed=build_error_embed(
                (
                    f"No valid {model_type_label} is configured. "
                    "Please contact an administrator."
                ),
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
        user_model = default_model or next(
            iter(config_data.get("models", {})),
            "",
        )

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
