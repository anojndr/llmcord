import asyncio
import logging
from typing import Any

import asyncpraw
import discord
import httpx
from discord.ext import commands
from twscrape import API

from llmcord.core.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Global state
config = get_config()
curr_model_lock = asyncio.Lock()  # Lock for thread-safe model operations

msg_nodes: dict[Any, Any] = {}
msg_nodes_lock = asyncio.Lock()

# Initialize clients
intents = discord.Intents.default()
intents.message_content = True
status_message = (
    config.get("status_message") or "github.com/jakobdylanc/llmcord"
)[:128]
activity = discord.CustomActivity(name=status_message)
discord_bot = commands.Bot(
    intents=intents,
    activity=activity,
    command_prefix=None,
    allowed_mentions=discord.AllowedMentions(replied_user=False),
)

httpx_client = httpx.AsyncClient()
twitter_api = API(proxy=config.get("twitter_proxy"))

if config.get("reddit_client_id") and config.get("reddit_client_secret"):
    reddit_client = asyncpraw.Reddit(
        client_id=config.get("reddit_client_id"),
        client_secret=config.get("reddit_client_secret"),
        user_agent=config.get("reddit_user_agent", "llmcord:v1.0 (by /u/llmcord)"),
    )
else:
    reddit_client = None
