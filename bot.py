"""
Discord bot setup, commands, and event handlers for llmcord.
"""
import asyncio
from base64 import b64encode
from datetime import datetime
import io
import logging
import os
import re

from aiohttp import web
import asyncpraw
from bs4 import BeautifulSoup
import discord
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
from google import genai
from google.genai import types
import httpx
from openai import AsyncOpenAI
from PIL import Image
from twscrape import API, gather
from youtube_transcript_api import YouTubeTranscriptApi

from bad_keys import get_bad_keys_db, init_bad_keys_db
from config import (
    get_config,
    VISION_MODEL_TAGS,
    PROVIDERS_SUPPORTING_USERNAMES,
    EMBED_COLOR_COMPLETE,
    EMBED_COLOR_INCOMPLETE,
    STREAMING_INDICATOR,
    EDIT_DELAY_SECONDS,
    MAX_MESSAGE_NODES,
)
from models import MsgNode
from views import ResponseView, SourceView
from web_search import decide_web_search, perform_web_search

# Import utils to apply the twscrape patch
import utils  # noqa: F401

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# Global state
config = get_config()
curr_model_lock = asyncio.Lock()  # Lock for thread-safe model operations

msg_nodes = {}
msg_nodes_lock = asyncio.Lock()

# Initialize clients
intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config.get("status_message") or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None, allowed_mentions=discord.AllowedMentions(replied_user=False))

httpx_client = httpx.AsyncClient()
twitter_api = API(proxy=config.get("twitter_proxy"))

if config.get("reddit_client_id") and config.get("reddit_client_secret"):
    reddit_client = asyncpraw.Reddit(
        client_id=config.get("reddit_client_id"),
        client_secret=config.get("reddit_client_secret"),
        user_agent=config.get("reddit_user_agent", "llmcord:v1.0 (by /u/llmcord)")
    )
else:
    reddit_client = None


@discord_bot.tree.command(name="model", description="View or switch your current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    user_id = str(interaction.user.id)
    db = get_bad_keys_db()

    if model not in config["models"]:
        await interaction.response.send_message(f"Model `{model}` is not a valid model.", ephemeral=True)
        return

    # Get user's current model preference (or default)
    current_user_model = db.get_user_model(user_id)
    if current_user_model is None:
        current_user_model = next(iter(config["models"]))  # Default model

    if model == current_user_model:
        output = f"Your current model: `{current_user_model}`"
    else:
        db.set_user_model(user_id, model)
        output = f"Your model switched to: `{model}`"
        logging.info(f"User {user_id} switched model to: {model}")

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config

    # Refresh config from cache (will reload if file changed)
    if curr_str == "":
        config = get_config()

    # Get user's current model preference (or default)
    user_id = str(interaction.user.id)
    db = get_bad_keys_db()
    user_model = db.get_user_model(user_id)
    if user_model is None:
        user_model = next(iter(config["models"]))  # Default model
    
    # Validate that user's saved model still exists in config
    if user_model not in config["models"]:
        user_model = next(iter(config["models"]))

    choices = [Choice(name=f"◉ {user_model} (current)", value=user_model)] if curr_str.lower() in user_model.lower() else []
    choices += [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != user_model and curr_str.lower() in model.lower()]

    return choices[:25]


@discord_bot.event
async def on_ready() -> None:
    # Generate bot invite link using the bot's application ID
    client_id = discord_bot.user.id
    invite_url = f"https://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot"
    logging.info(f"\n\nBOT INVITE URL:\n{invite_url}\n")

    await discord_bot.tree.sync()

    if twitter_accounts := config.get("twitter_accounts"):
        for acc in twitter_accounts:
            if await twitter_api.pool.get_account(acc["username"]):
                continue
            await twitter_api.pool.add_account(
                acc["username"],
                acc["password"],
                acc["email"],
                acc["email_password"],
                cookies=acc.get("cookies"),
            )
        await twitter_api.pool.login_all()


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    from message_handler import process_message
    
    # Get user's model preference from database (or use default)
    user_id = str(new_msg.author.id)
    db = get_bad_keys_db()
    user_model = db.get_user_model(user_id)
    
    # Fall back to default model if user hasn't set a preference or if their saved model is no longer valid
    default_model = next(iter(config["models"]))
    if user_model is None or user_model not in config["models"]:
        user_model = default_model
    
    # Create a reference list to pass user's model by reference
    curr_model_ref = [user_model]
    
    await process_message(
        new_msg=new_msg,
        discord_bot=discord_bot,
        httpx_client=httpx_client,
        twitter_api=twitter_api,
        reddit_client=reddit_client,
        msg_nodes=msg_nodes,
        curr_model_lock=curr_model_lock,
        curr_model_ref=curr_model_ref,
    )


async def health_check(request):
    return web.Response(text="I'm alive")


async def start_server():
    app = web.Application()
    app.add_routes([web.get('/', health_check)])
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", 8000))
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()


async def main() -> None:
    # Initialize Turso database with credentials from config
    turso_url = config.get("turso_database_url")
    turso_token = config.get("turso_auth_token")
    init_bad_keys_db(db_url=turso_url, auth_token=turso_token)
    
    await asyncio.gather(start_server(), discord_bot.start(config["bot_token"]))

