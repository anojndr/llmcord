"""Discord message processing helpers."""

import importlib
import logging
from contextlib import suppress
from typing import TYPE_CHECKING, cast

import discord

from llmcord import config as config_module
from llmcord import globals as app_globals
from llmcord.discord.ui.embed_limits import call_with_embed_limits
from llmcord.discord.ui.utils import build_error_embed
from llmcord.logic.pipeline import ProcessContext
from llmcord.services.database import get_bad_keys_db
from llmcord.utils.common import get_channel_locked_model

if TYPE_CHECKING:
    from llmcord.services.extractors import TwitterApiProtocol

logger = logging.getLogger(__name__)


async def _process_user_message(new_msg: discord.Message) -> None:
    """Shared handler for normal messages and retries."""
    config_data = config_module.get_config()

    # Check if this channel has a locked model override
    channel_id = new_msg.channel.id
    locked_model = get_channel_locked_model(channel_id)

    user_model: str | None
    if locked_model:
        # Use the channel's locked model (ignore user preference)
        if locked_model not in config_data.get("models", {}):
            logger.error(
                ("Channel %s has locked model '%s' but it's not in config.yaml models"),
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
        default_model = next(iter(config_data.get("models", {})), "")
        if not default_model:
            logger.error("No models configured in config.yaml")
            return

        if user_model is None or user_model not in config_data.get(
            "models",
            {},
        ):
            user_model = default_model

    # Create a reference list to pass user's model by reference
    curr_model_ref = [user_model]

    try:
        context = ProcessContext(
            discord_bot=app_globals.discord_bot,
            httpx_client=app_globals.httpx_client,
            twitter_api=cast("TwitterApiProtocol", app_globals.twitter_api),
            msg_nodes=app_globals.msg_nodes,
            curr_model_lock=app_globals.curr_model_lock,
            curr_model_ref=curr_model_ref,
        )
        processing_module = importlib.import_module("llmcord.processing")
        await processing_module.process_message(
            new_msg=new_msg,
            context=context,
        )
    except Exception:
        logger.exception("Error processing message")
        # Try to notify the user about the error
        with suppress(Exception):
            await call_with_embed_limits(
                new_msg.reply,
                embed=build_error_embed(
                    "An internal error occurred while processing your request. "
                    "Please try again later.",
                ),
            )


async def _handle_retry_request(
    interaction: discord.Interaction,
    request_message_id: int,
    request_user_id: int,
) -> None:
    """Retry a previous prompt using its original message."""
    if interaction.user.id != request_user_id:
        await call_with_embed_limits(
            interaction.followup.send,
            embed=build_error_embed("You can only retry your own message."),
            ephemeral=True,
        )
        return

    channel = interaction.channel
    fetch_message = getattr(channel, "fetch_message", None) if channel else None
    if not callable(fetch_message):
        await call_with_embed_limits(
            interaction.followup.send,
            embed=build_error_embed(
                "Unable to locate the original channel for this message.",
            ),
            ephemeral=True,
        )
        return

    try:
        request_msg = await fetch_message(request_message_id)
    except (discord.NotFound, discord.Forbidden, discord.HTTPException):
        await call_with_embed_limits(
            interaction.followup.send,
            embed=build_error_embed(
                "Unable to fetch the original message for retry.",
            ),
            ephemeral=True,
        )
        return

    await _process_user_message(request_msg)
