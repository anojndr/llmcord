"""Discord event handlers for llmcord."""

import asyncio
import contextlib
import logging

import discord
from discord import app_commands

from llmcord.core.error_handling import (
    COMMON_HANDLER_EXCEPTIONS,
    log_discord_event_error,
    log_exception,
)
from llmcord.discord.error_handling import handle_app_command_error
from llmcord.discord.processing import (
    _handle_retry_request,
    _process_user_message,
)
from llmcord.discord.ui.response_view import PersistentResponseView
from llmcord.discord.ui.utils import set_retry_handler
from llmcord.globals import config, discord_bot, twitter_api

logger = logging.getLogger(__name__)

_TWITTER_INIT_TASK_ATTR = "_llmcord_twitter_init_task"


async def _init_twitter_accounts() -> None:
    twitter_accounts = config.get("twitter_accounts") or []
    if not twitter_accounts:
        return

    timeout_seconds_raw = config.get("twitter_login_timeout_seconds")
    timeout_seconds = 120.0
    if timeout_seconds_raw is not None:
        with contextlib.suppress(TypeError, ValueError):
            timeout_seconds = float(timeout_seconds_raw)

    logger.info(
        "Initializing %s Twitter/X account(s) in background (timeout=%.1fs)",
        len(twitter_accounts),
        timeout_seconds,
    )

    try:
        for acc in twitter_accounts:
            username = acc.get("username")
            if not username:
                continue

            existing = await asyncio.wait_for(
                twitter_api.pool.get_account(username),
                timeout=10,
            )
            if existing:
                continue

            await asyncio.wait_for(
                twitter_api.pool.add_account(
                    username,
                    acc["password"],
                    acc["email"],
                    acc["email_password"],
                    cookies=acc.get("cookies"),
                ),
                timeout=30,
            )

        await asyncio.wait_for(
            twitter_api.pool.login_all(),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        logger.warning(
            "Twitter/X login timed out after %.1fs; continuing without Twitter "
            "scraping until login succeeds",
            timeout_seconds,
        )
    except COMMON_HANDLER_EXCEPTIONS as exc:
        log_exception(
            logger=logger,
            message=(
                "Twitter/X account initialization failed; continuing without "
                "Twitter scraping"
            ),
            error=exc,
        )
    else:
        logger.info("Twitter/X account initialization completed")


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

    twitter_accounts = config.get("twitter_accounts")
    existing_task = getattr(discord_bot, _TWITTER_INIT_TASK_ATTR, None)
    if twitter_accounts and (existing_task is None or existing_task.done()):
        setattr(
            discord_bot,
            _TWITTER_INIT_TASK_ATTR,
            asyncio.create_task(
                _init_twitter_accounts(),
                name="llmcord-twitter-init",
            ),
        )


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
