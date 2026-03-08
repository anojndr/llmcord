"""Discord event handlers for llmcord."""

import asyncio
import contextlib
import logging
import time

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
_STARTUP_STARTED_ATTR = "_llmcord_startup_started_at"
_BOT_INVITE_PERMISSIONS = 412317191168
_ONE_MILLISECOND_SECONDS = 0.001
_TEN_MILLISECONDS_SECONDS = 0.01


def _format_elapsed(seconds: float) -> str:
    """Format elapsed time for human-readable startup logs."""
    if seconds < _ONE_MILLISECOND_SECONDS:
        return "<1ms"
    if seconds < _TEN_MILLISECONDS_SECONDS:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.2f}s"


def _get_twitter_accounts() -> list[dict[str, object]]:
    """Return configured Twitter/X accounts in a predictable shape."""
    configured = config.get("twitter_accounts") or []
    return [account for account in configured if isinstance(account, dict)]


def _count_usable_twitter_accounts(
    twitter_accounts: list[dict[str, object]],
) -> int:
    """Count configured Twitter/X accounts that include a username."""
    return sum(
        1
        for account in twitter_accounts
        if _get_twitter_account_str(account, "username")
    )


def _get_twitter_account_str(
    account: dict[str, object],
    key: str,
) -> str | None:
    """Return a non-empty Twitter/X account config string value."""
    value = account.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped


def _get_twitter_login_timeout() -> float:
    """Return the configured Twitter/X login timeout in seconds."""
    timeout_seconds_raw = config.get("twitter_login_timeout_seconds")
    timeout_seconds = 120.0
    if timeout_seconds_raw is not None:
        with contextlib.suppress(TypeError, ValueError):
            timeout_seconds = float(timeout_seconds_raw)
    return timeout_seconds


def _consume_startup_seconds() -> float | None:
    """Return startup elapsed seconds once, if startup timing is available."""
    started_at = getattr(discord_bot, _STARTUP_STARTED_ATTR, None)
    if not isinstance(started_at, float):
        return None
    setattr(discord_bot, _STARTUP_STARTED_ATTR, None)
    return max(0.0, time.perf_counter() - started_at)


async def _init_twitter_accounts() -> None:
    twitter_accounts = _get_twitter_accounts()
    if not twitter_accounts:
        return

    timeout_seconds = _get_twitter_login_timeout()
    configured_count = len(twitter_accounts)
    usable_count = _count_usable_twitter_accounts(twitter_accounts)
    skipped_missing_username = configured_count - usable_count
    if usable_count == 0:
        logger.warning(
            "Skipping Twitter/X account initialization because no configured "
            "account has a username (configured=%s)",
            configured_count,
        )
        return

    started_at = time.perf_counter()
    logger.info(
        "Starting Twitter/X account initialization "
        "(configured=%s, usable=%s, skipped_missing_username=%s, "
        "timeout=%.1fs)",
        configured_count,
        usable_count,
        skipped_missing_username,
        timeout_seconds,
    )

    already_loaded_count = 0
    added_count = 0
    skipped_incomplete_credentials = 0
    try:
        for acc in twitter_accounts:
            username = _get_twitter_account_str(acc, "username")
            if username is None:
                continue
            password = _get_twitter_account_str(acc, "password")
            email = _get_twitter_account_str(acc, "email")
            email_password = _get_twitter_account_str(acc, "email_password")
            cookies = _get_twitter_account_str(acc, "cookies")
            if password is None or email is None or email_password is None:
                skipped_incomplete_credentials += 1
                logger.warning(
                    "Skipping Twitter/X account %s because password, email, "
                    "or email_password is missing",
                    username,
                )
                continue

            existing = await asyncio.wait_for(
                twitter_api.pool.get_account(username),
                timeout=10,
            )
            if existing:
                already_loaded_count += 1
                continue

            await asyncio.wait_for(
                twitter_api.pool.add_account(
                    username,
                    password,
                    email,
                    email_password,
                    cookies=cookies,
                ),
                timeout=30,
            )
            added_count += 1

        await asyncio.wait_for(
            twitter_api.pool.login_all(),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        logger.warning(
            "Twitter/X login timed out after %.1fs; continuing without Twitter "
            "scraping until login succeeds "
            "(configured=%s, usable=%s, skipped_missing_username=%s, "
            "already_loaded=%s, added=%s, skipped_incomplete_credentials=%s)",
            timeout_seconds,
            configured_count,
            usable_count,
            skipped_missing_username,
            already_loaded_count,
            added_count,
            skipped_incomplete_credentials,
        )
    except COMMON_HANDLER_EXCEPTIONS as exc:
        log_exception(
            logger=logger,
            message=(
                "Twitter/X account initialization failed; continuing without "
                "Twitter scraping"
            ),
            error=exc,
            context={
                "configured": configured_count,
                "usable": usable_count,
                "skipped_missing_username": skipped_missing_username,
                "already_loaded": already_loaded_count,
                "added": added_count,
                "skipped_incomplete_credentials": skipped_incomplete_credentials,
            },
        )
    else:
        logger.info(
            "Twitter/X account initialization finished in %s "
            "(configured=%s, usable=%s, skipped_missing_username=%s, "
            "already_loaded=%s, added=%s, skipped_incomplete_credentials=%s)",
            _format_elapsed(time.perf_counter() - started_at),
            configured_count,
            usable_count,
            skipped_missing_username,
            already_loaded_count,
            added_count,
            skipped_incomplete_credentials,
        )


# =============================================================================
# Event Handlers
# =============================================================================


@discord_bot.event
async def on_ready() -> None:
    """Log readiness and sync slash commands."""
    if not discord_bot.user:
        return

    startup_seconds = _consume_startup_seconds()
    startup_suffix = ""
    if startup_seconds is not None:
        startup_suffix = f" after {_format_elapsed(startup_seconds)} of startup"
    logger.info(
        "Discord client ready as %s (%s); connected to %s guild(s)%s. "
        "Syncing application commands.",
        discord_bot.user,
        discord_bot.user.id,
        len(discord_bot.guilds),
        startup_suffix,
    )

    ready_started_at = time.perf_counter()
    synced_commands = await discord_bot.tree.sync()

    # Register persistent views for response buttons
    discord_bot.add_view(PersistentResponseView())

    # Register retry handler for persistent buttons
    set_retry_handler(_handle_retry_request)
    logger.info(
        "Discord startup tasks completed in %s: synced %s application "
        "command(s), registered persistent response controls, and installed "
        "the retry handler",
        _format_elapsed(time.perf_counter() - ready_started_at),
        len(synced_commands),
    )

    client_id = discord_bot.application_id or discord_bot.user.id
    invite_url = (
        "https://discord.com/oauth2/authorize?client_id="
        f"{client_id}&permissions={_BOT_INVITE_PERMISSIONS}&scope=bot"
    )
    logger.info(
        "Bot invite URL (application_id=%s, permissions=%s): %s",
        client_id,
        _BOT_INVITE_PERMISSIONS,
        invite_url,
    )

    twitter_accounts = _get_twitter_accounts()
    existing_task = getattr(discord_bot, _TWITTER_INIT_TASK_ATTR, None)
    if twitter_accounts and (existing_task is None or existing_task.done()):
        timeout_seconds = _get_twitter_login_timeout()
        logger.info(
            "Queued background Twitter/X initialization for %s configured "
            "account(s) (usable=%s, login timeout=%.1fs)",
            len(twitter_accounts),
            _count_usable_twitter_accounts(twitter_accounts),
            timeout_seconds,
        )
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
