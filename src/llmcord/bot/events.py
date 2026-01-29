"""Discord event handlers for the bot."""
import logging
from contextlib import suppress

import discord

from llmcord.bad_keys import get_bad_keys_db
from llmcord.bot.app import (
    config,
    curr_model_lock,
    discord_bot,
    get_channel_locked_model,
    httpx_client,
    msg_nodes,
    reddit_client,
    twitter_api,
)
from llmcord.message_handler import process_message
from llmcord.views import ResponseView

logger = logging.getLogger(__name__)
PERSISTENT_VIEWS_LOADED = False


def _coerce_int(value: object | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_user_model(new_msg: discord.Message) -> str | None:
    """Resolve the model selection for a message using channel lock or user prefs."""
    channel_id = new_msg.channel.id
    locked_model = get_channel_locked_model(channel_id)

    if locked_model:
        if locked_model not in config.get("models", {}):
            logger.error(
                "Channel %s has locked model '%s' but it's not in config.yaml models",
                channel_id,
                locked_model,
            )
            return None
        return locked_model

    user_id = str(new_msg.author.id)
    db = get_bad_keys_db()
    user_model = db.get_user_model(user_id)

    default_model = next(iter(config.get("models", {})), None)
    if not default_model:
        logger.error("No models configured in config.yaml")
        return None

    if user_model is None or user_model not in config.get("models", {}):
        return default_model
    return user_model


async def _dispatch_message(new_msg: discord.Message) -> None:
    """Dispatch a message through the core processing pipeline."""
    user_model = _resolve_user_model(new_msg)
    if not user_model:
        return

    curr_model_ref = [user_model]
    await process_message(
        new_msg=new_msg,
        discord_bot=discord_bot,
        httpx_client=httpx_client,
        twitter_api=twitter_api,
        reddit_client=reddit_client,
        msg_nodes=msg_nodes,
        curr_model_lock=curr_model_lock,
        curr_model_ref=curr_model_ref,
    )


async def _retry_message(channel_id: int, user_message_id: int) -> None:
    """Retry a prior user message by fetching it and re-dispatching."""
    channel = discord_bot.get_channel(channel_id)
    if channel is None:
        channel = await discord_bot.fetch_channel(channel_id)

    message = await channel.fetch_message(user_message_id)
    await _dispatch_message(message)


async def _load_persistent_response_views() -> None:
    """Re-register persistent response views so buttons work after restarts."""
    global PERSISTENT_VIEWS_LOADED
    if PERSISTENT_VIEWS_LOADED:
        return

    db = get_bad_keys_db()
    try:
        records = db.list_message_response_data()
    except Exception:
        logger.exception("Failed to load persistent response views")
        return

    loaded = 0
    for record in records:
        response_message_id = _coerce_int(record.get("response_message_id"))
        if response_message_id is None:
            continue

        user_message_id = _coerce_int(record.get("user_message_id"))
        channel_id = _coerce_int(record.get("channel_id"))
        user_id = _coerce_int(record.get("user_id"))
        full_response = str(record.get("full_response") or "")
        grounding_metadata = record.get("grounding_metadata")
        tavily_metadata = record.get("tavily_metadata")

        retry_callback = None
        if user_message_id and channel_id and user_id:

            async def _retry_callback(
                channel_id=channel_id,
                user_message_id=user_message_id,
            ) -> None:
                await _retry_message(channel_id, user_message_id)

            retry_callback = _retry_callback

        view = ResponseView(
            full_response,
            grounding_metadata,
            tavily_metadata,
            retry_callback,
            user_id,
        )
        discord_bot.add_view(view, message_id=response_message_id)
        loaded += 1

    PERSISTENT_VIEWS_LOADED = True
    logger.info("Loaded %d persistent response views", loaded)


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

    await _load_persistent_response_views()

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
    try:
        await _dispatch_message(new_msg)
    except Exception as exc:
        logger.exception("Error processing message")
        # Try to notify the user about the error
        with suppress(Exception):
            await new_msg.reply(
                "❌ An internal error occurred while processing your request.\n"
                f"Error: {exc}",
            )
