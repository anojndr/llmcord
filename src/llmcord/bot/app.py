"""Shared bot application state and client initialization."""
import asyncio
import logging

import asyncpraw
import discord
import httpx
from discord.ext import commands
from twscrape import API

# Import utils to apply the twscrape patch
from llmcord import utils  # noqa: F401
from llmcord.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Global state
config = get_config()
curr_model_lock = asyncio.Lock()  # Lock for thread-safe model operations

msg_nodes = {}
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


def get_channel_locked_model(channel_id: int) -> str | None:
    """Check whether a channel has a locked model override.

    Args:
        channel_id: The Discord channel ID to check.

    Returns:
        The model name if the channel has a locked model, or None otherwise.

    """
    overrides = config.get("channel_model_overrides", {})
    # Convert channel_id to string for comparison since YAML may parse keys as ints
    # or strings.
    return overrides.get(channel_id) or overrides.get(str(channel_id))
