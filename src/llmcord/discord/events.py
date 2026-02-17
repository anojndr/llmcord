"""Discord event handlers for llmcord."""

import logging

import discord
from discord import app_commands

from llmcord.core.error_handling import log_discord_event_error
from llmcord.discord.error_handling import handle_app_command_error
from llmcord.discord.processing import (
    _handle_retry_request,
    _process_user_message,
)
from llmcord.discord.ui.response_view import PersistentResponseView
from llmcord.discord.ui.utils import set_retry_handler
from llmcord.globals import config, discord_bot, twitter_api

logger = logging.getLogger(__name__)


# =============================================================================
# Event Handlers
# =============================================================================


@discord_bot.event
async def on_ready() -> None:
    """Log readiness and sync slash commands."""
    if not discord_bot.user:
        return

    # Generate bot invite link using the bot's application ID
    client_id = discord_bot.user.id
    invite_url = (
        "https://discord.com/oauth2/authorize?client_id="
        f"{client_id}&permissions=412317191168&scope=bot"
    )
    logger.info("\n\nBOT INVITE URL:\n%s\n", invite_url)

    await discord_bot.tree.sync()

    # Register persistent views for response buttons
    discord_bot.add_view(PersistentResponseView())

    # Register retry handler for persistent buttons
    set_retry_handler(_handle_retry_request)

    if twitter_accounts := config.get("twitter_accounts"):
        for acc in twitter_accounts:
            if await twitter_api.pool.get_account(acc["username"]):
                continue
            await twitter_api.pool.add_account(
                acc["username"],
                acc["password"],
                acc["email"],
                acc["email_password"],
                cookies=acc.get("cookies"),
            )
        await twitter_api.pool.login_all()


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    """Handle inbound Discord messages."""
    await _process_user_message(new_msg)


@discord_bot.tree.error
async def on_app_command_error(
    interaction: discord.Interaction,
    error: app_commands.AppCommandError,
) -> None:
    """Handle uncaught slash-command exceptions in one place."""
    await handle_app_command_error(interaction, error, logger=logger)


@discord_bot.event
async def on_error(event_method: str, *args: object, **kwargs: object) -> None:
    """Handle uncaught Discord event exceptions in one place."""
    log_discord_event_error(
        logger=logger,
        event_name=event_method,
        args=args,
        kwargs=kwargs,
    )
