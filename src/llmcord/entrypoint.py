import asyncio

# Import utils to apply the twscrape patch (side-effect import)
from llmcord.logic import utils  # noqa: F401
# Ensure commands and events are registered
# pylint: disable=unused-import
import llmcord.discord.commands  # noqa: F401
import llmcord.discord.events  # noqa: F401
from llmcord.globals import config, discord_bot
from llmcord.server import start_server
from llmcord.services.database import init_bad_keys_db


async def main() -> None:
    """Initialize dependencies and start background services."""
    # Initialize Turso database with credentials from config
    turso_url = config.get("turso_database_url")
    turso_token = config.get("turso_auth_token")
    init_bad_keys_db(db_url=turso_url, auth_token=turso_token)

    await asyncio.gather(start_server(), discord_bot.start(config["bot_token"]))
