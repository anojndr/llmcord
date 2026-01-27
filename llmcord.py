"""llmcord - A Discord bot for LLM interactions.
Main entry point.
"""
import asyncio

from bot import main

try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
