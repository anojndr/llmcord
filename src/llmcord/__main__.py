"""llmcord - A Discord bot for LLM interactions.

Main entry point.
"""
import asyncio
import contextlib

from .entrypoint import main

with contextlib.suppress(KeyboardInterrupt):
    asyncio.run(main())
