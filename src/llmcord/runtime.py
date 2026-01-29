"""Main entry point for the bot runtime."""
import asyncio

import llmcord.bot.commands as _commands  # noqa: F401
import llmcord.bot.events as _events  # noqa: F401
from llmcord.bad_keys import init_bad_keys_db
from llmcord.bot.app import config, discord_bot
from llmcord.bot.server import start_server
from llmcord.config import get_bot_token


async def main() -> None:
    """Initialize dependencies and start background services."""
    # Initialize Turso database with credentials from config
    turso_url = config.get("turso_database_url")
    turso_token = config.get("turso_auth_token")
    init_bad_keys_db(db_url=turso_url, auth_token=turso_token)

    await asyncio.gather(start_server(), discord_bot.start(get_bot_token(config)))
