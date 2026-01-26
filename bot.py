"""
Discord bot setup, commands, and event handlers for llmcord.
"""
import asyncio
import logging
import os

from aiohttp import web
import asyncpraw
import discord
from discord.app_commands import Choice
from discord.ext import commands
import httpx
from twscrape import API

from bad_keys import get_bad_keys_db, init_bad_keys_db
from config import get_config
from message_handler import process_message

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


# =============================================================================
# DRY Helper Functions for Slash Commands
# =============================================================================

def get_channel_locked_model(channel_id: int) -> str | None:
    """
    Check if a channel has a locked model override.
    
    Args:
        channel_id: The Discord channel ID to check
        
    Returns:
        The model name if the channel has a locked model, None otherwise
    """
    overrides = config.get("channel_model_overrides", {})
    # Convert channel_id to string for comparison since YAML may parse keys as ints or strings
    return overrides.get(channel_id) or overrides.get(str(channel_id))

async def _handle_model_switch(
    interaction: discord.Interaction,
    model: str,
    get_current_model_fn,
    set_model_fn,
    get_default_fn,
    model_type_label: str = "model"
) -> None:
    """
    Generic handler for model switching slash commands.
    
    DRY: Consolidates the shared logic between /model and /searchdecidermodel commands.
    
    Args:
        interaction: Discord interaction object
        model: The model to switch to
        get_current_model_fn: Function to get current model for user (db method)
        set_model_fn: Function to set model for user (db method)
        get_default_fn: Function to get default model (returns model string)
        model_type_label: Label for logging/messages (e.g., "model" or "search decider model")
    """
    user_id = str(interaction.user.id)

    if model not in config["models"]:
        await interaction.followup.send(f"❌ Model `{model}` is not configured in `config.yaml`.", ephemeral=True)
        return

    # Get user's current model preference (or default)
    current_user_model = get_current_model_fn(user_id)
    default_model = get_default_fn()
    
    if current_user_model is None:
        current_user_model = default_model
    
    if current_user_model is None:
        await interaction.followup.send(f"❌ No valid {model_type_label} configured. Please ask an administrator to check `config.yaml`.", ephemeral=True)
        return

    if model == current_user_model:
        output = f"Your current {model_type_label}: `{current_user_model}`"
    else:
        set_model_fn(user_id, model)
        output = f"Your {model_type_label} switched to: `{model}`"
        logging.info(f"User {user_id} switched {model_type_label} to: {model}")

    await interaction.followup.send(output)


def _build_model_autocomplete(
    curr_str: str,
    get_current_model_fn,
    get_default_fn,
    user_id: str
) -> list[Choice[str]]:
    """
    Generic builder for model autocomplete choices.
    
    DRY: Consolidates the shared autocomplete logic between commands.
    
    Args:
        curr_str: Current search string from user input
        get_current_model_fn: Function to get current model for user (db method)
        get_default_fn: Function to get default model (returns model string)
        user_id: User ID string
    
    Returns:
        List of discord.app_commands.Choice objects
    """
    user_model = get_current_model_fn(user_id)
    default_model = get_default_fn()
    
    if user_model is None:
        user_model = default_model
    
    # Validate that user's saved model still exists in config
    if not user_model or user_model not in config.get("models", {}):
        user_model = default_model or next(iter(config.get("models", {})), "")
    
    if not user_model:
        return []

    choices = [Choice(name=f"◉ {user_model} (current)", value=user_model)] if curr_str.lower() in user_model.lower() else []
    choices += [Choice(name=f"○ {m}", value=m) for m in config["models"] if m != user_model and curr_str.lower() in m.lower()]

    return choices[:25]


# =============================================================================
# Slash Commands
# =============================================================================

@discord_bot.tree.command(name="model", description="View or switch your current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    await interaction.response.defer(ephemeral=(interaction.channel.type == discord.ChannelType.private))
    
    # Check if this channel has a locked model
    channel_id = interaction.channel_id
    locked_model = get_channel_locked_model(channel_id)
    if locked_model:
        await interaction.followup.send(
            f"❌ This channel is locked to model `{locked_model}`. "
            f"The /model command is disabled here.",
            ephemeral=True
        )
        return
    
    db = get_bad_keys_db()
    
    def get_default():
        return next(iter(config.get("models", {})), None)
    
    await _handle_model_switch(
        interaction=interaction,
        model=model,
        get_current_model_fn=db.get_user_model,
        set_model_fn=db.set_user_model,
        get_default_fn=get_default,
        model_type_label="model"
    )


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config
    if not curr_str:
        config = get_config()

    db = get_bad_keys_db()
    user_id = str(interaction.user.id)
    
    def get_default():
        return next(iter(config.get("models", {})), None)
    
    return _build_model_autocomplete(curr_str, db.get_user_model, get_default, user_id)


@discord_bot.tree.command(name="searchdecidermodel", description="View or switch your search decider model")
async def search_decider_model_command(interaction: discord.Interaction, model: str) -> None:
    await interaction.response.defer(ephemeral=(interaction.channel.type == discord.ChannelType.private))
    
    db = get_bad_keys_db()
    
    def get_default():
        default = config.get("web_search_decider_model", "gemini/gemini-3-flash-preview")
        return default if default in config.get("models", {}) else None
    
    await _handle_model_switch(
        interaction=interaction,
        model=model,
        get_current_model_fn=db.get_user_search_decider_model,
        set_model_fn=db.set_user_search_decider_model,
        get_default_fn=get_default,
        model_type_label="search decider model"
    )


@search_decider_model_command.autocomplete("model")
async def search_decider_model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config
    if curr_str == "":
        config = get_config()

    db = get_bad_keys_db()
    user_id = str(interaction.user.id)
    
    def get_default():
        default = config.get("web_search_decider_model", "gemini/gemini-3-flash-preview")
        return default if default in config.get("models", {}) else next(iter(config.get("models", {})), "")
    
    return _build_model_autocomplete(curr_str, db.get_user_search_decider_model, get_default, user_id)


@discord_bot.tree.command(name="resetallpreferences", description="[Owner] Reset all users' model preferences")
async def reset_all_preferences_command(interaction: discord.Interaction) -> None:
    # Only allow the bot owner to use this command
    OWNER_USER_ID = 676735636656357396
    if interaction.user.id != OWNER_USER_ID:
        await interaction.response.send_message(
            "❌ This command can only be used by the bot owner.", 
            ephemeral=True
        )
        return
    
    # Defer the response since database operations may take time
    await interaction.response.defer(ephemeral=True)
    
    db = get_bad_keys_db()
    
    # Reset both preferences
    model_count = db.reset_all_user_model_preferences()
    decider_count = db.reset_all_user_search_decider_preferences()
    
    await interaction.followup.send(
        f"✅ Successfully reset all user preferences:\n"
        f"• **Main model preferences**: {model_count} user(s) reset\n"
        f"• **Search decider model preferences**: {decider_count} user(s) reset\n\n"
        f"All users will now use the default models."
    )
    logging.info(f"Owner {interaction.user.id} reset all user preferences (models: {model_count}, deciders: {decider_count})")


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
    # Check if this channel has a locked model override
    channel_id = new_msg.channel.id
    locked_model = get_channel_locked_model(channel_id)
    
    if locked_model:
        # Use the channel's locked model (ignore user preference)
        if locked_model not in config.get("models", {}):
            logging.error(f"Channel {channel_id} has locked model '{locked_model}' but it's not in config.yaml models")
            return
        user_model = locked_model
    else:
        # Get user's model preference from database (or use default)
        user_id = str(new_msg.author.id)
        db = get_bad_keys_db()
        user_model = db.get_user_model(user_id)
        
        # Fall back to default model if user hasn't set a preference or if their saved model is no longer valid
        default_model = next(iter(config.get("models", {})), None)
        if not default_model:
            logging.error("No models configured in config.yaml")
            return
        
        if user_model is None or user_model not in config.get("models", {}):
            user_model = default_model
    
    # Create a reference list to pass user's model by reference
    curr_model_ref = [user_model]
    
    try:
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
    except Exception as e:
        logging.exception("Error processing message")
        # Try to notify the user about the error
        try:
            await new_msg.reply(f"❌ An internal error occurred while processing your request.\nError: {e}")
        except Exception:
            pass


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

