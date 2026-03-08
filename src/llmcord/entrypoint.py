"""Entrypoint module for initializing services."""

import asyncio
import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from llmcord import processing as processing_module
from llmcord.discord import commands as discord_commands
from llmcord.discord import events as discord_events
from llmcord.globals import config, discord_bot, httpx_client, reddit_client
from llmcord.server import start_server
from llmcord.services.database import init_db
from llmcord.services.search import decider as search_decider

if TYPE_CHECKING:
    from aiohttp.web import AppRunner

    from llmcord.services.database import AppDB


@dataclass(slots=True)
class _EntrypointState:
    server_runner: "AppRunner | None" = None
    db_instance: "AppDB | None" = None


_STATE = _EntrypointState()


def preload_runtime_dependencies() -> None:
    """Import runtime modules eagerly so the first request does not lazy-load."""
    _ = (discord_commands, discord_events)
    processing_module.preload_runtime_dependencies()
    search_decider.preload_runtime_dependencies()


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

    if reddit_client is not None:
        with contextlib.suppress(Exception):
            await reddit_client.close()

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
    preload_runtime_dependencies()

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
