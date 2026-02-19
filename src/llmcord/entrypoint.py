"""Entrypoint module for initializing services."""

import asyncio
import importlib

from llmcord.globals import config, discord_bot
from llmcord.server import start_server
from llmcord.services.database import init_db


async def main() -> None:
    """Initialize dependencies and start background services."""
    # Register Discord events and slash commands
    importlib.import_module("llmcord.discord.commands")
    importlib.import_module("llmcord.discord.events")

    # Initialize SQLite database
    init_db()

    await asyncio.gather(start_server(), discord_bot.start(config["bot_token"]))
