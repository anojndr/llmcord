import logging
from contextlib import suppress

import discord

from llmcord.globals import (
    config,
    curr_model_lock,
    discord_bot,
    httpx_client,
    msg_nodes,
    reddit_client,
    twitter_api,
)
from llmcord.helpers import get_channel_locked_model
from llmcord.logic.message import process_message
from llmcord.services.database import get_bad_keys_db

logger = logging.getLogger(__name__)


async def _process_user_message(new_msg: discord.Message) -> None:
    """Shared handler for normal messages and retries."""
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


async def _handle_retry_request(
    interaction: discord.Interaction,
    request_message_id: int,
    request_user_id: int,
) -> None:
    """Retry a previous prompt using its original message."""
    if interaction.user.id != request_user_id:
        await interaction.followup.send(
            "❌ You cannot retry this message.",
            ephemeral=True,
        )
        return

    channel = interaction.channel
    if channel is None or not hasattr(channel, "fetch_message"):
        await interaction.followup.send(
            "❌ Unable to locate the original channel for this message.",
            ephemeral=True,
        )
        return

    try:
        request_msg = await channel.fetch_message(request_message_id)
    except (discord.NotFound, discord.Forbidden, discord.HTTPException):
        await interaction.followup.send(
            "❌ Unable to fetch the original message for retry.",
            ephemeral=True,
        )
        return

    await _process_user_message(request_msg)
