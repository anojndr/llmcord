"""Entrypoint module for initializing services."""

import asyncio
import contextlib
import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from llmcord.globals import config, discord_bot, httpx_client
from llmcord.server import start_server
from llmcord.services.database import init_db

if TYPE_CHECKING:
    from aiohttp.web import AppRunner

    from llmcord.services.database import AppDB


@dataclass(slots=True)
class _EntrypointState:
    server_runner: "AppRunner | None" = None
    db_instance: "AppDB | None" = None


_STATE = _EntrypointState()


async def shutdown() -> None:
    """Best-effort shutdown of long-lived resources.

    This is safe to call multiple times.
    """
    if not discord_bot.is_closed():
        with contextlib.suppress(Exception):
            await discord_bot.close()
            # Wait for discord.py keep-alive threads to exit cleanly before the
            # event loop is closed to prevent "Event loop is closed" errors.
            await asyncio.sleep(0.25)

    if httpx_client is not None:
        with contextlib.suppress(Exception):
            await httpx_client.aclose()

    if _STATE.db_instance is not None:
        with contextlib.suppress(Exception):
            await _STATE.db_instance.aclose()
        _STATE.db_instance = None

    if _STATE.server_runner is not None:
        with contextlib.suppress(Exception):
            await _STATE.server_runner.cleanup()
        _STATE.server_runner = None


async def main() -> None:
    """Initialize dependencies and start background services."""
    # Register Discord events and slash commands
    importlib.import_module("llmcord.discord.commands")
    importlib.import_module("llmcord.discord.events")

    # Initialize SQLite database
    _STATE.db_instance = await init_db()

    _STATE.server_runner = await start_server()
    try:
        await discord_bot.start(config["bot_token"])
    finally:
        # Ctrl+C typically cancels the main task; shield shutdown so Discord closes
        # before the event loop is closed.
        with contextlib.suppress(Exception):
            await asyncio.shield(shutdown())
