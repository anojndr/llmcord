"""Permissions handling for llmcord."""

import discord

from llmcord.core.config import (
    EMBED_COLOR_INCOMPLETE,
    PROCESSING_MESSAGE,
    get_config,
)
from llmcord.discord.ui.embed_limits import call_with_embed_limits


async def should_process_message(
    new_msg: discord.Message,
    discord_bot: discord.Client,
) -> tuple[bool, discord.Message | None]:
    """Check if the message should be processed and send a response.

    Returns:
        tuple: (should_process, processing_msg_or_none)

    """
    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (
        not is_dm
        and discord_bot.user not in new_msg.mentions
        and "at ai" not in new_msg.content.lower()
    ) or new_msg.author.bot:
        return False, None

    role_ids = {role.id for role in getattr(new_msg.author, "roles", ())}
    channel_ids = {
        channel_id
        for channel_id in (
            new_msg.channel.id,
            getattr(new_msg.channel, "parent_id", None),
            getattr(new_msg.channel, "category_id", None),
        )
        if channel_id is not None
    }

    config = get_config()  # Now cached, no need for to_thread

    allow_dms = config.get("allow_dms", True)
    permissions = config["permissions"]

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    # Pre-convert to sets once for efficient lookups
    allowed_user_ids = set(permissions["users"]["allowed_ids"])
    blocked_user_ids = set(permissions["users"]["blocked_ids"])
    allowed_role_ids = set(permissions["roles"]["allowed_ids"])
    blocked_role_ids = set(permissions["roles"]["blocked_ids"])
    allowed_channel_ids = set(permissions["channels"]["allowed_ids"])
    blocked_channel_ids = set(permissions["channels"]["blocked_ids"])

    if is_dm:
        allow_all_users = not allowed_user_ids
    else:
        allow_all_users = not allowed_user_ids and not allowed_role_ids
    is_good_user = (
        user_is_admin
        or allow_all_users
        or new_msg.author.id in allowed_user_ids
        or bool(role_ids & allowed_role_ids)
    )
    is_bad_user = (
        not is_good_user
        or new_msg.author.id in blocked_user_ids
        or bool(role_ids & blocked_role_ids)
    )

    allow_all_channels = not allowed_channel_ids
    is_good_channel = (
        user_is_admin or allow_dms
        if is_dm
        else allow_all_channels or bool(channel_ids & allowed_channel_ids)
    )
    is_bad_channel = not is_good_channel or bool(
        channel_ids & blocked_channel_ids,
    )

    if is_bad_user or is_bad_channel:
        return False, None

    # Send processing message immediately after confirming bot should respond
    processing_embed = discord.Embed(
        description=PROCESSING_MESSAGE,
        color=EMBED_COLOR_INCOMPLETE,
    )
    processing_msg = await call_with_embed_limits(
        new_msg.reply,
        embed=processing_embed,
        silent=True,
    )

    return True, processing_msg
