"""Centralized error handling for Discord interactions and callbacks."""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import TYPE_CHECKING

import discord
import httpx
from discord import app_commands

from llmcord.core.error_handling import COMMON_HANDLER_EXCEPTIONS, log_exception
from llmcord.discord.ui.embed_limits import call_with_embed_limits
from llmcord.discord.ui.utils import build_error_embed

if TYPE_CHECKING:
    from collections.abc import Mapping

LOGGER = logging.getLogger(__name__)

INTERNAL_ERROR_MESSAGE = (
    "An internal error occurred while processing your request. Please try again later."
)
MESSAGE_PROCESSING_EXCEPTIONS = (
    ImportError,
    *COMMON_HANDLER_EXCEPTIONS,
    discord.DiscordException,
    httpx.HTTPError,
)


def _build_interaction_context(interaction: discord.Interaction) -> dict[str, object]:
    """Build structured context fields for interaction logs."""
    command_name = (
        interaction.command.qualified_name if interaction.command is not None else None
    )
    user_id = interaction.user.id if interaction.user is not None else None
    return {
        "command": command_name,
        "user_id": user_id,
        "channel_id": interaction.channel_id,
        "guild_id": interaction.guild_id,
    }


def build_message_context(message: discord.Message) -> dict[str, object]:
    """Build structured context fields for message-processing logs."""
    return {
        "message_id": message.id,
        "author_id": message.author.id,
        "channel_id": message.channel.id,
    }


def unwrap_app_command_error(error: app_commands.AppCommandError) -> Exception:
    """Unwrap command invocation errors to their root cause."""
    if isinstance(error, app_commands.CommandInvokeError):
        original_error = error.original
        if isinstance(original_error, Exception):
            return original_error
    return error


async def send_interaction_error(
    interaction: discord.Interaction,
    *,
    description: str = INTERNAL_ERROR_MESSAGE,
    ephemeral: bool = True,
) -> None:
    """Send a standardized interaction error response when possible."""
    embed = build_error_embed(description)
    with suppress(Exception):
        if interaction.response.is_done():
            await call_with_embed_limits(
                interaction.followup.send,
                embed=embed,
                ephemeral=ephemeral,
            )
            return
        await call_with_embed_limits(
            interaction.response.send_message,
            embed=embed,
            ephemeral=ephemeral,
        )


async def send_message_processing_error(
    message: discord.Message,
    *,
    description: str = INTERNAL_ERROR_MESSAGE,
) -> None:
    """Reply to a user message with a standardized processing error."""
    with suppress(Exception):
        await call_with_embed_limits(
            message.reply,
            embed=build_error_embed(description),
        )


async def edit_processing_message_error(
    processing_msg: discord.Message,
    *,
    description: str = INTERNAL_ERROR_MESSAGE,
) -> None:
    """Update the processing placeholder message with a standardized error."""
    with suppress(Exception):
        await call_with_embed_limits(
            processing_msg.edit,
            embed=build_error_embed(description),
            view=None,
        )


async def handle_app_command_error(
    interaction: discord.Interaction,
    error: app_commands.AppCommandError,
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Log and respond to uncaught slash-command errors."""
    target_logger = logger or LOGGER
    root_error = unwrap_app_command_error(error)
    log_exception(
        logger=target_logger,
        message="Unhandled slash command error",
        error=root_error,
        context=_build_interaction_context(interaction),
    )
    await send_interaction_error(interaction)


async def handle_ui_callback_error(
    *,
    interaction: discord.Interaction,
    error: Exception,
    surface: str,
    logger: logging.Logger | None = None,
    context: Mapping[str, object] | None = None,
) -> None:
    """Log and respond to uncaught Discord UI callback errors."""
    target_logger = logger or LOGGER
    interaction_context = _build_interaction_context(interaction)
    interaction_context["surface"] = surface
    if context:
        interaction_context.update(context)
    log_exception(
        logger=target_logger,
        message="Unhandled Discord UI callback error",
        error=error,
        context=interaction_context,
    )
    await send_interaction_error(interaction)
