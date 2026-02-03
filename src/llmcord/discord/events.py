"""Discord event handlers for llmcord."""

import logging

import discord

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
