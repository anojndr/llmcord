"""Entrypoint module for initializing services."""

import asyncio
import importlib

from llmcord.globals import config, discord_bot
from llmcord.server import start_server
from llmcord.services.database import init_bad_keys_db


async def main() -> None:
    """Initialize dependencies and start background services."""
    # Register Discord events and slash commands
    importlib.import_module("llmcord.discord.commands")
    importlib.import_module("llmcord.discord.events")

    # Initialize Turso database with credentials from config
    turso_url = config.get("turso_database_url")
    turso_token = config.get("turso_auth_token")
    init_bad_keys_db(db_url=turso_url, auth_token=turso_token)

    await asyncio.gather(start_server(), discord_bot.start(config["bot_token"]))
