"""Global state and shared clients."""

import asyncio
import logging
from typing import Any

import asyncpraw
import discord
import httpx
from discord.ext import commands
from twscrape import API

from llmcord.core.config import (
    HttpxClientOptions,
    get_config,
    get_or_create_httpx_client,
)

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
status_message = (config.get("status_message") or "github.com/jakobdylanc/llmcord")[
    :128
]
activity = discord.CustomActivity(name=status_message)
command_prefix = config.get("command_prefix") or "!"
discord_bot = commands.Bot(
    intents=intents,
    activity=activity,
    command_prefix=command_prefix,
    allowed_mentions=discord.AllowedMentions(replied_user=False),
)

_httpx_client_holder: list[httpx.AsyncClient | None] = []
proxy_url = config.get("proxy_url") or None
httpx_client = get_or_create_httpx_client(
    _httpx_client_holder,
    options=HttpxClientOptions(proxy_url=proxy_url),
)
twitter_proxy = config.get("twitter_proxy") or proxy_url
twitter_api = API(proxy=twitter_proxy)


if (
    config.get("reddit_mode", "json") == "praw"
    and config.get("reddit_client_id")
    and config.get("reddit_client_secret")
):
    user_agent = config.get(
        "reddit_user_agent",
        "llmcord:v1.0 (by /u/llmcord)",
    )
    reddit_client = asyncpraw.Reddit(
        client_id=config.get("reddit_client_id"),
        client_secret=config.get("reddit_client_secret"),
        user_agent=user_agent,
    )
else:
    reddit_client = None
