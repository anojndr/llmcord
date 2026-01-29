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

logger = logging.getLogger(__name__)


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
    # Check if this channel has a locked model override
    channel_id = new_msg.channel.id
    locked_model = get_channel_locked_model(channel_id)

    if locked_model:
        # Use the channel's locked model (ignore user preference)
        if locked_model not in config.get("models", {}):
            logger.error(
                "Channel %s has locked model '%s' but it's not in config.yaml models",
                channel_id,
                locked_model,
            )
            return
        user_model = locked_model
    else:
        # Get user's model preference from database (or use default)
        user_id = str(new_msg.author.id)
        db = get_bad_keys_db()
        user_model = db.get_user_model(user_id)

        # Fall back to default model if user hasn't set a preference
        # or if their saved model is no longer valid.
        default_model = next(iter(config.get("models", {})), None)
        if not default_model:
            logger.error("No models configured in config.yaml")
            return

        if user_model is None or user_model not in config.get("models", {}):
            user_model = default_model

    # Create a reference list to pass user's model by reference
    curr_model_ref = [user_model]

    try:
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
    except Exception as exc:
        logger.exception("Error processing message")
        # Try to notify the user about the error
        with suppress(Exception):
            await new_msg.reply(
                "❌ An internal error occurred while processing your request.\n"
                f"Error: {exc}",
            )
