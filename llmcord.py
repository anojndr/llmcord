import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import os
import re
import logging
import sqlite3
from typing import Any, Literal, Optional
import io
from PIL import Image

from aiohttp import web
import discord
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
from google import genai
from google.genai import types
import httpx
from openai import AsyncOpenAI
from twscrape import xclid
import json
from bs4 import BeautifulSoup
import asyncpraw

def script_url(k: str, v: str):
    return f"https://abs.twimg.com/responsive-web/client-web/{k}.{v}.js"

def patched_get_scripts_list(text: str):
    scripts = text.split('e=>e+"."+')[1].split('[e]+"a.js"')[0]

    try:
        for k, v in json.loads(scripts).items():
            yield script_url(k, f"{v}a")
    except json.decoder.JSONDecodeError:
        fixed_scripts = re.sub(
            r'([,\{])(\s*)([\w]+_[\w_]+)(\s*):',
            r'\1\2"\3"\4:',
            scripts
        )
        for k, v in json.loads(fixed_scripts).items():
            yield script_url(k, f"{v}a")

xclid.get_scripts_list = patched_get_scripts_list

from twscrape import API, gather
from youtube_transcript_api import YouTubeTranscriptApi
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)


class BadKeysDB:
    """SQLite-based tracking of bad API keys to avoid wasting retries."""
    
    def __init__(self, db_path: str = "bad_keys.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the database table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bad_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(provider, key_hash)
                )
            """)
            conn.commit()
        finally:
            conn.close()
    
    def _hash_key(self, api_key: str) -> str:
        """Create a hash of the API key to avoid storing sensitive data."""
        import hashlib
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]
    
    def is_key_bad(self, provider: str, api_key: str) -> bool:
        """Check if an API key is marked as bad."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM bad_keys WHERE provider = ? AND key_hash = ?",
                (provider, self._hash_key(api_key))
            )
            return cursor.fetchone() is not None
        finally:
            conn.close()
    
    def mark_key_bad(self, provider: str, api_key: str, error_message: str = None):
        """Mark an API key as bad."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO bad_keys (provider, key_hash, error_message) VALUES (?, ?, ?)",
                (provider, self._hash_key(api_key), error_message)
            )
            conn.commit()
            logging.info(f"Marked API key as bad for provider '{provider}' (hash: {self._hash_key(api_key)[:8]}...)")
        finally:
            conn.close()
    
    def get_good_keys(self, provider: str, all_keys: list[str]) -> list[str]:
        """Filter out bad keys from a list of API keys."""
        return [key for key in all_keys if not self.is_key_bad(provider, key)]
    
    def get_bad_key_count(self, provider: str) -> int:
        """Get the count of bad keys for a provider."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM bad_keys WHERE provider = ?",
                (provider,)
            )
            return cursor.fetchone()[0]
        finally:
            conn.close()
    
    def reset_provider_keys(self, provider: str):
        """Reset all bad keys for a specific provider."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM bad_keys WHERE provider = ?", (provider,))
            conn.commit()
            logging.info(f"Reset all bad keys for provider '{provider}'")
        finally:
            conn.close()
    
    def reset_all(self):
        """Reset all bad keys for all providers."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM bad_keys")
            conn.commit()
            logging.info("Reset all bad keys database")
        finally:
            conn.close()


# Global instance of the bad keys database
bad_keys_db = BadKeysDB()

# Web Search Decider and Tavily Search Functions
SEARCH_DECIDER_SYSTEM_PROMPT = """You are a web search query optimizer. Your job is to analyze user queries and determine if they need web search for up-to-date or factual information.

RESPOND ONLY with a valid JSON object. No other text, no markdown.

If web search is NOT needed (e.g., creative writing, opinions, general knowledge, math, coding):
{"needs_search": false}

If web search IS needed (e.g., current events, recent news, product specs, prices, real-time info):
{"needs_search": true, "queries": ["query1", "query2", ...]}

RULES for generating queries:
1. SINGLE SUBJECT = SINGLE QUERY. If the user asks about ONE thing, output exactly ONE search query. Do NOT split into multiple queries.
   Example: "latest news" → ["latest news today"] (ONE query only)
   Example: "iPhone 16 price" → ["iPhone 16 price"] (ONE query only)
2. MULTIPLE SUBJECTS = MULTIPLE QUERIES. Only if the user asks about multiple distinct subjects/entities, create separate queries for EACH subject PLUS a comparison query.
   Example: "iPhone 16 vs Samsung S25 specs" → ["iPhone 16 specs", "Samsung S25 specs", "iPhone 16 vs Samsung S25 specs comparison"]
3. Make queries search-engine friendly (clear, specific keywords)
4. Keep the total number of queries reasonable (1-5 queries max)
5. Preserve the user's original intent

Examples:
- "What's the weather today?" → {"needs_search": true, "queries": ["weather today"]}
- "Who won the 2024 Super Bowl?" → {"needs_search": true, "queries": ["2024 Super Bowl winner"]}
- "latest news" → {"needs_search": true, "queries": ["latest news today"]}
- "Write me a poem about cats" → {"needs_search": false}
- "Compare RTX 4090 and RTX 4080 performance" → {"needs_search": true, "queries": ["RTX 4090 specs performance", "RTX 4080 specs performance", "RTX 4090 vs RTX 4080 comparison"]}
"""


async def decide_web_search(messages: list, decider_config: dict) -> dict:
    """
    Uses a configurable model to decide if web search is needed and generates optimized queries.
    Supports both Gemini and OpenAI-compatible APIs.
    Implements retry mechanism with API key rotation and bad key tracking.
    Returns: {"needs_search": bool, "queries": list[str]} or {"needs_search": False}
    
    decider_config should contain:
        - provider: "gemini" or other (OpenAI-compatible)
        - model: model name
        - api_keys: list of API keys
        - base_url: (optional) for OpenAI-compatible providers
    """
    provider = decider_config.get("provider", "gemini")
    model = decider_config.get("model", "gemini-3-flash-preview")
    api_keys = decider_config.get("api_keys", [])
    base_url = decider_config.get("base_url")
    
    if not api_keys:
        return {"needs_search": False}
    
    # Use a prefixed provider name to separate decider keys from main model keys
    decider_provider = f"decider_{provider}"
    
    # Get good keys (filter out known bad ones)
    good_keys = bad_keys_db.get_good_keys(decider_provider, api_keys)
    
    # If all keys are bad, reset and try again with all keys
    if not good_keys:
        logging.warning(f"All API keys for '{decider_provider}' are marked as bad. Resetting...")
        bad_keys_db.reset_provider_keys(decider_provider)
        good_keys = api_keys.copy()
    
    attempt_count = 0
    
    while good_keys:
        attempt_count += 1
        current_api_key = good_keys[(attempt_count - 1) % len(good_keys)]
        
        try:
            if provider == "gemini":
                # Build Gemini-format contents
                search_decider_contents = []
                for msg in messages[::-1]:  # Reverse to get chronological order
                    role = "model" if msg.get("role") == "assistant" else "user"
                    content = msg.get("content", "")
                    parts = []
                    
                    if isinstance(content, list):
                        # Handle multimodal content (text + images)
                        for part in content:
                            if isinstance(part, dict):
                                if part.get("type") == "text" and part.get("text"):
                                    parts.append(types.Part.from_text(text=str(part["text"])[:5000]))
                                elif part.get("type") == "image_url" and part.get("image_url", {}).get("url"):
                                    # Extract base64 image data
                                    image_url = part["image_url"]["url"]
                                    if image_url.startswith("data:"):
                                        try:
                                            header, b64_data = image_url.split(",", 1)
                                            mime_type = header.split(":")[1].split(";")[0]
                                            import base64
                                            image_bytes = base64.b64decode(b64_data)
                                            parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
                                        except Exception:
                                            pass
                    elif content:
                        parts.append(types.Part.from_text(text=str(content)[:5000]))
                    
                    if parts:
                        search_decider_contents.append(types.Content(role=role, parts=parts))
                
                if not search_decider_contents:
                    return {"needs_search": False}
                
                # Add instruction to analyze the conversation
                search_decider_contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Based on the conversation above, analyze the last user query and respond with your JSON decision.")]
                ))
                
                config_kwargs = dict(
                    system_instruction=SEARCH_DECIDER_SYSTEM_PROMPT,
                    temperature=0.1,
                )
                
                # Add thinking config for Gemini 3 models
                if "gemini-3" in model:
                    config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="MINIMAL")
                
                gemini_config = types.GenerateContentConfig(**config_kwargs)
                
                client = genai.Client(api_key=current_api_key, http_options=dict(api_version="v1beta"))
                
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=search_decider_contents,
                    config=gemini_config
                )
                
                response_text = response.text.strip()
            else:
                # OpenAI-compatible API
                openai_messages = [{"role": "system", "content": SEARCH_DECIDER_SYSTEM_PROMPT}]
                
                for msg in messages[::-1]:  # Reverse to get chronological order
                    role = msg.get("role", "user")
                    if role == "system":
                        continue  # Skip system messages from chat history
                    
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        # Keep multimodal format for OpenAI
                        openai_messages.append({"role": role, "content": content})
                    elif content:
                        openai_messages.append({"role": role, "content": str(content)[:5000]})
                
                if len(openai_messages) <= 1:  # Only system prompt
                    return {"needs_search": False}
                
                # Add instruction to analyze the conversation
                openai_messages.append({
                    "role": "user",
                    "content": "Based on the conversation above, analyze the last user query and respond with your JSON decision."
                })
                
                openai_client = AsyncOpenAI(base_url=base_url, api_key=current_api_key)
                
                response = await openai_client.chat.completions.create(
                    model=model,
                    messages=openai_messages,
                    temperature=0.1,
                )
                
                response_text = response.choices[0].message.content.strip()
            
            # Parse response (common for both providers)
            if response_text.startswith("```"):
                # Remove markdown code blocks if present
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            result = json.loads(response_text)
            return result
        except Exception as e:
            logging.exception(f"Error in web search decider (attempt {attempt_count}): {e}")
            
            # Mark the current key as bad
            error_msg = str(e)[:200] if e else "Unknown error"
            bad_keys_db.mark_key_bad(decider_provider, current_api_key, error_msg)
            
            # Remove the bad key from good_keys list for this session
            if current_api_key in good_keys:
                good_keys.remove(current_api_key)
            
            # If all keys are exhausted, try resetting once
            if not good_keys:
                if attempt_count >= len(api_keys) * 2:
                    logging.error("Web search decider failed after exhausting all keys, skipping web search")
                    return {"needs_search": False}
                else:
                    logging.warning(f"All decider keys exhausted. Resetting for retry...")
                    bad_keys_db.reset_provider_keys(decider_provider)
                    good_keys = api_keys.copy()
    
    return {"needs_search": False}


async def tavily_search(query: str, tavily_api_key: str, max_results: int = 5) -> dict:
    """
    Execute a single Tavily search query.
    Returns the search results with page content or an error dict.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                    "include_answer": False,
                    "include_raw_content": "markdown",  # Get full page content in markdown format
                },
                headers={
                    "Authorization": f"Bearer {tavily_api_key}",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logging.exception(f"Tavily search error for query '{query}': {e}")
        return {"error": str(e), "query": query}


async def perform_web_search(queries: list[str], tavily_api_keys: list[str], max_results_per_query: int = 5, max_chars_per_url: int = 2000) -> str:
    """
    Perform concurrent web searches for multiple queries using Tavily.
    Rotates through multiple API keys for load distribution.
    Returns formatted search results as a string to be appended to the user's message.
    
    Args:
        queries: List of search queries
        tavily_api_keys: List of Tavily API keys for rotation
        max_results_per_query: Maximum number of URLs per query (default: 5)
        max_chars_per_url: Maximum characters per URL content (default: 2000)
    """
    if not queries or not tavily_api_keys:
        return ""
    
    # Execute all searches concurrently, rotating through API keys
    search_tasks = [
        tavily_search(query, tavily_api_keys[i % len(tavily_api_keys)], max_results_per_query) 
        for i, query in enumerate(queries)
    ]
    results = await asyncio.gather(*search_tasks)
    
    formatted_results = []
    
    for i, result in enumerate(results):
        if "error" in result:
            continue
        
        query = queries[i]
        formatted_results.append(f"\n### Search Results for: {query}")
        
        for item in result.get("results", []):
            title = item.get("title", "No title")
            url = item.get("url", "")
            
            # Prefer raw_content (full page content) over content (snippet)
            raw_content = item.get("raw_content", "")
            content = item.get("content", "")
            
            # Use raw_content if available, otherwise fall back to content snippet
            page_content = raw_content if raw_content else content
            # Limit to max_chars_per_url (default 2000 chars per URL)
            page_content = page_content[:max_chars_per_url] if page_content else ""
            
            result_text = f"\n**{title}**\n{url}\n{page_content}\n"
            formatted_results.append(result_text)
    
    if formatted_results:
        return "\n\n---\n**Web Search Results:**" + "".join(formatted_results)
    return ""


VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500



def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    try:
        with open(filename, encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        with open(os.path.join("/etc/secrets", filename), encoding="utf-8") as file:
            return yaml.safe_load(file)


config = get_config()
curr_model = next(iter(config["models"]))
curr_model_lock = asyncio.Lock()  # Lock for thread-safe model switching

msg_nodes = {}
msg_nodes_lock = asyncio.Lock()  # Lock for thread-safe msg_nodes access

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


@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)
    raw_attachments: list[dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None
    thought_signature: Optional[str] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model not in config["models"]:
        await interaction.response.send_message(f"Model `{model}` is not a valid model.", ephemeral=True)
        return

    async with curr_model_lock:
        if model == curr_model:
            output = f"Current model: `{curr_model}`"
        else:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config

    if curr_str == "":
        config = await asyncio.to_thread(get_config)

    choices = [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []
    choices += [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()]

    return choices[:25]


@discord_bot.event
async def on_ready() -> None:
    if client_id := config.get("client_id"):
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")

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
    # Per-request edit timing to avoid interference between concurrent requests
    last_edit_time = 0

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (not is_dm and discord_bot.user not in new_msg.mentions and "at ai" not in new_msg.content.lower()) or new_msg.author.bot:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    config = await asyncio.to_thread(get_config)

    allow_dms = config.get("allow_dms", True)

    permissions = config["permissions"]

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    # Thread-safe read of current model
    async with curr_model_lock:
        provider_slash_model = curr_model
    provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)

    provider_config = config["providers"][provider]

    base_url = provider_config.get("base_url")
    api_keys = provider_config.get("api_key", "sk-no-key-required")
    if isinstance(api_keys, str):
        api_keys = [api_keys]

    model_parameters = config["models"].get(provider_slash_model, None)

    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(provider_slash_model.lower().startswith(x) for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)
    max_tweet_replies = config.get("max_tweet_replies", 50)

    messages = []
    gemini_contents = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()
                cleaned_content = re.sub(r"\bat ai\b", "", cleaned_content, flags=re.IGNORECASE).lstrip()

                if cleaned_content.lower().startswith("googlelens"):
                    cleaned_content = cleaned_content[10:].strip()

                    if image_url := next((att.url for att in curr_msg.attachments if att.content_type and att.content_type.startswith("image")), None):
                        try:
                            headers = {
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                            }
                            params = {
                                "rpt": "imageview",
                                "url": image_url
                            }
                            
                            yandex_resp = await httpx_client.get("https://yandex.com/images/search", params=params, headers=headers, follow_redirects=True, timeout=60)
                            soup = BeautifulSoup(yandex_resp.text, "html.parser")
                            
                            lens_results = []
                            sites_items = soup.select(".CbirSites-Item")
                            
                            if sites_items:
                                for item in sites_items:
                                    title_el = item.select_one(".CbirSites-ItemTitle a")
                                    domain_el = item.select_one(".CbirSites-ItemDomain")
                                    desc_el = item.select_one(".CbirSites-ItemDescription")
                                    
                                    title = title_el.get_text(strip=True) if title_el else "N/A"
                                    link = title_el["href"] if title_el else "#"
                                    domain = domain_el.get_text(strip=True) if domain_el else ""
                                    desc = desc_el.get_text(strip=True) if desc_el else ""
                                    
                                    lens_results.append(f"- [{title}]({link}) ({domain}) - {desc}")
                            
                            if lens_results:
                                cleaned_content = (cleaned_content + "\n\nanswer the user's query based on the yandex reverse image results:\n" + "\n".join(lens_results))
                        except Exception:
                            logging.exception("Error fetching Yandex results")

                allowed_types = ("text", "image")
                if provider == "gemini":
                    allowed_types += ("audio", "video", "application/pdf")

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in allowed_types)]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                processed_attachments = []
                for att, resp in zip(good_attachments, attachment_responses):
                    content = resp.content
                    content_type = att.content_type

                    if content_type == "image/gif":
                        try:
                            with Image.open(io.BytesIO(content)) as img:
                                output = io.BytesIO()
                                img.save(output, format="PNG")
                                content = output.getvalue()
                                content_type = "image/png"
                        except Exception:
                            logging.exception("Error converting GIF to PNG")

                    processed_attachments.append(dict(content_type=content_type, content=content, text=resp.text if content_type.startswith("text") else None))

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [component.content for component in curr_msg.components if component.type == discord.ComponentType.text_display]
                    + [att["text"] for att in processed_attachments if att["content_type"].startswith("text")]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att['content_type']};base64,{b64encode(att['content']).decode('utf-8')}"))
                    for att in processed_attachments
                    if att["content_type"].startswith("image")
                ]

                curr_node.raw_attachments = [
                    dict(content_type=att["content_type"], content=att["content"])
                    for att in processed_attachments
                ]

                yt_transcripts = []
                for video_id in re.findall(r"(?:https?:\/\/)?(?:[a-zA-Z0-9-]+\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})", cleaned_content):
                    try:
                        ytt_api = YouTubeTranscriptApi()
                        transcript_obj = await asyncio.to_thread(ytt_api.fetch, video_id)
                        transcript = transcript_obj.to_raw_data()

                        response = await httpx_client.get(f"https://www.youtube.com/watch?v={video_id}", follow_redirects=True)
                        html = response.text
                        title_match = re.search(r'<meta name="title" content="(.*?)">', html)
                        title = title_match.group(1) if title_match else "Unknown Title"
                        channel_match = re.search(r'<link itemprop="name" content="(.*?)">', html)
                        channel = channel_match.group(1) if channel_match else "Unknown Channel"

                        yt_transcripts.append(f"YouTube Video ID: {video_id}\nTitle: {title}\nChannel: {channel}\nTranscript:\n" + " ".join(x["text"] for x in transcript))
                    except Exception:
                        pass

                tweets = []
                for tweet_id in re.findall(r"(?:https?:\/\/)?(?:[a-zA-Z0-9-]+\.)?(?:twitter\.com|x\.com)\/[a-zA-Z0-9_]+\/status\/([0-9]+)", cleaned_content):
                    try:
                        tweet = await asyncio.wait_for(twitter_api.tweet_details(int(tweet_id)), timeout=10)
                        tweets.append(f"Tweet from @{tweet.user.username}:\n{tweet.rawContent}")

                        if max_tweet_replies > 0:
                            replies = await asyncio.wait_for(gather(twitter_api.tweet_replies(int(tweet_id), limit=max_tweet_replies)), timeout=10)
                            if replies:
                                tweets.append("\nReplies:")
                                for reply in replies:
                                    tweets.append(f"- @{reply.user.username}: {reply.rawContent}")
                    except Exception:
                        pass

                reddit_posts = []
                if reddit_client:
                    for post_url in re.findall(r"(https?:\/\/(?:[a-zA-Z0-9-]+\.)?(?:reddit\.com\/r\/[a-zA-Z0-9_]+\/comments\/[a-zA-Z0-9_]+(?:[\w\-\.\/\?=&%]*)|redd\.it\/[a-zA-Z0-9_]+))", cleaned_content):
                        try:
                            submission = await reddit_client.submission(url=post_url)
                            
                            post_text = f"Reddit Post: {submission.title}\nSubreddit: r/{submission.subreddit.display_name}\nAuthor: u/{submission.author.name if submission.author else '[deleted]'}\n\n{submission.selftext}"
                            
                            if not submission.is_self and submission.url:
                                post_text += f"\nLink: {submission.url}"

                            submission.comment_sort = 'top'
                            await submission.comments()
                            top_comments = submission.comments.list()[:5]
                            
                            if top_comments:
                                post_text += "\n\nTop Comments:"
                                for comment in top_comments:
                                    if isinstance(comment, asyncpraw.models.MoreComments):
                                        continue
                                    post_text += f"\n- u/{comment.author.name if comment.author else '[deleted]'}: {comment.body}"
                            
                            reddit_posts.append(post_text)
                        except Exception:
                            pass

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [component.content for component in curr_msg.components if component.type == discord.ComponentType.text_display]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                    + yt_transcripts
                    + tweets
                    + reddit_posts
                )

                if not curr_node.text and curr_node.images:
                    curr_node.text = "What is in this image?"

                if not curr_node.text and not curr_node.images and not curr_node.raw_attachments and curr_msg == new_msg:
                    curr_node.text = "Hello"

                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"

                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    if (
                        curr_msg.reference == None
                        and discord_bot.user.mention not in curr_msg.content
                        and "at ai" not in curr_msg.content.lower()
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            if provider == "gemini":
                parts = []
                if curr_node.text:
                    part = types.Part.from_text(text=curr_node.text[:max_text])
                    if curr_node.thought_signature:
                        part.thought_signature = curr_node.thought_signature
                    parts.append(part)

                for att in curr_node.raw_attachments:
                    if att["content_type"].startswith("text"):
                        continue

                    part = types.Part.from_bytes(data=att["content"], mime_type=att["content_type"])
                    if model_parameters and (media_res := model_parameters.get("media_resolution")):
                        part.media_resolution = {"level": media_res}
                    parts.append(part)

                if parts:
                    role = "model" if curr_node.role == "assistant" else "user"
                    gemini_contents.append(types.Content(role=role, parts=parts))

            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id != None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    system_prompt = config.get("system_prompt")

    if system_prompt:
        now = datetime.now().astimezone()

        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        if accept_usernames:
            system_prompt += "\n\nUser's names are their Discord IDs and should be typed as '<@ID>'."


        messages.append(dict(role="system", content=system_prompt))

    # Web Search Integration for non-Gemini models and Gemini preview models
    tavily_api_keys = config.get("tavily_api_key")
    if isinstance(tavily_api_keys, str):
        tavily_api_keys = [tavily_api_keys]
    elif not tavily_api_keys:
        tavily_api_keys = []
    
    is_preview_model = "preview" in model.lower()
    is_non_gemini = provider != "gemini"
    is_googlelens_query = new_msg.content.lower().removeprefix(discord_bot.user.mention).strip().lower().startswith("googlelens")
    
    if tavily_api_keys and (is_non_gemini or is_preview_model) and not is_googlelens_query:
        # Get web search decider model from config (default: gemini/gemini-3-flash-preview)
        decider_model_str = config.get("web_search_decider_model", "gemini/gemini-3-flash-preview")
        decider_provider, decider_model = decider_model_str.split("/", 1) if "/" in decider_model_str else ("gemini", decider_model_str)
        
        # Get provider config for the decider
        decider_provider_config = config.get("providers", {}).get(decider_provider, {})
        decider_api_keys = decider_provider_config.get("api_key", [])
        if isinstance(decider_api_keys, str):
            decider_api_keys = [decider_api_keys]
        
        decider_config = {
            "provider": decider_provider,
            "model": decider_model,
            "api_keys": decider_api_keys,
            "base_url": decider_provider_config.get("base_url"),
        }
        
        if decider_api_keys:
            # Run the search decider with chat history (uses all keys with rotation)
            search_decision = await decide_web_search(messages, decider_config)
            
            if search_decision.get("needs_search") and search_decision.get("queries"):
                queries = search_decision["queries"]
                logging.info(f"Web search triggered. Queries: {queries}")
                
                # Perform concurrent Tavily searches (5 URLs per query, 2k chars per URL)
                search_results = await perform_web_search(queries, tavily_api_keys, max_results_per_query=5, max_chars_per_url=2000)
                
                if search_results:
                    # Append search results to the first (most recent) user message
                    for i, msg in enumerate(messages):
                        if msg.get("role") == "user":
                            original_content = msg.get("content", "")
                            if isinstance(original_content, list):
                                # For multimodal content, append to the text part
                                for part in original_content:
                                    if isinstance(part, dict) and part.get("type") == "text":
                                        part["text"] = part["text"] + "\n\n" + search_results
                                        break
                            else:
                                msg["content"] = str(original_content) + "\n\n" + search_results
                            
                            # Also update gemini_contents if it's a Gemini preview model
                            if provider == "gemini" and gemini_contents:
                                # Find the corresponding user content and update it  
                                for j, gc in enumerate(gemini_contents):
                                    if gc.role == "user":
                                        if gc.parts and gc.parts[0].text:
                                            # Create new parts with updated text
                                            new_text = gc.parts[0].text + "\n\n" + search_results
                                            new_parts = [types.Part.from_text(text=new_text)] + list(gc.parts[1:])
                                            gemini_contents[j] = types.Content(role="user", parts=new_parts)
                                        break
                            
                            logging.info(f"Web search results appended to user message")
                            break

    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []

    openai_kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)

    if use_plain_responses := config.get("use_plain_responses", False):
        max_message_length = 4000
    else:
        max_message_length = 4096 - len(STREAMING_INDICATOR)
        embed = discord.Embed.from_dict(dict(fields=[dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)]))
        embed.set_footer(text=f"Model: {provider_slash_model}")

    async def reply_helper(**reply_kwargs) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)

        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()

    async def get_stream(api_key):
        if provider == "gemini":
            is_gemini_3 = "gemini-3" in model
            is_preview = "preview" in model

            http_options = dict(api_version="v1beta")
            if model_parameters and model_parameters.get("media_resolution"):
                http_options["api_version"] = "v1alpha"

            client = genai.Client(api_key=api_key, http_options=http_options)
            
            has_url = re.search(r"https?://", new_msg.content)

            if is_preview:
                tools = []
            else:
                if has_url:
                    tools = []
                else:
                    tools = [types.Tool(google_search=types.GoogleSearch()), types.Tool(url_context=types.UrlContext())]

            config_kwargs = dict(
                system_instruction=system_prompt,
                tools=tools,
            )

            if model_parameters and not is_gemini_3:
                config_kwargs["temperature"] = model_parameters.get("temperature")

            thinking_level = model_parameters.get("thinking_level") if model_parameters else None

            if not thinking_level:
                if "gemini-3-flash" in model:
                    thinking_level = "MINIMAL"
                elif "gemini-3-pro" in model:
                    thinking_level = "LOW"

            if thinking_level:
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)

            active_contents = gemini_contents[::-1]

            while True:
                gemini_config = types.GenerateContentConfig(**config_kwargs)
                
                full_text = ""
                function_calls_to_execute = []
                captured_thought_signature = None
                finish_reason = None
                grounding_metadata = None

                async for chunk in await client.aio.models.generate_content_stream(
                    model=model,
                    contents=active_contents,
                    config=gemini_config
                ):
                    text = ""
                    if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                        for part in chunk.candidates[0].content.parts:
                            if part.text:
                                text += part.text
                    
                    full_text += text
                    
                    reason = chunk.candidates[0].finish_reason if chunk.candidates else None
                    if reason:
                        finish_reason = reason
                        
                    grounding_metadata = chunk.candidates[0].grounding_metadata if chunk.candidates else None

                    thought_signature = None
                    if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                        for part in chunk.candidates[0].content.parts:
                            if part.function_call:
                                function_calls_to_execute.append(part.function_call)
                            if hasattr(part, "thought_signature") and part.thought_signature:
                                thought_signature = part.thought_signature
                                captured_thought_signature = thought_signature

                    yield_reason = str(reason) if reason else None
                    if function_calls_to_execute:
                        yield_reason = None

                    yield text, yield_reason, grounding_metadata, thought_signature

                if function_calls_to_execute:
                    parts = []
                    if full_text:
                        parts.append(types.Part.from_text(text=full_text))
                    
                    for fc in function_calls_to_execute:
                        parts.append(types.Part(function_call=fc))
                    
                    if captured_thought_signature and parts:
                        parts[0].thought_signature = captured_thought_signature
                        
                    active_contents.append(types.Content(role="model", parts=parts))
                    
                    continue
                
                break
        else:
            openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            
            current_kwargs = openai_kwargs.copy()
            
            messages_list = list(current_kwargs["messages"])
            current_kwargs["messages"] = messages_list

            while True:
                tool_calls = {}
                finish_reason = None
                
                async for chunk in await openai_client.chat.completions.create(**current_kwargs):
                    if not (choice := chunk.choices[0] if chunk.choices else None):
                        continue
                    
                    if choice.delta.tool_calls:
                        for tool_call_chunk in choice.delta.tool_calls:
                            index = tool_call_chunk.index
                            if index not in tool_calls:
                                tool_calls[index] = {
                                    "id": tool_call_chunk.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_call_chunk.function.name,
                                        "arguments": ""
                                    }
                                }
                            
                            if tool_call_chunk.function.name:
                                tool_calls[index]["function"]["name"] = tool_call_chunk.function.name
                            
                            if tool_call_chunk.function.arguments:
                                tool_calls[index]["function"]["arguments"] += tool_call_chunk.function.arguments
                    
                    if choice.delta.content or (choice.finish_reason and choice.finish_reason != "tool_calls"):
                        yield choice.delta.content or "", choice.finish_reason, None, None
                    
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

                if finish_reason == "tool_calls":
                    break
                else:
                    break

    grounding_metadata = None
    attempt_count = 0
    
    # Get good keys (filter out known bad ones)
    good_keys = bad_keys_db.get_good_keys(provider, api_keys)
    
    # If all keys are bad, reset and try again with all keys
    if not good_keys:
        logging.warning(f"All API keys for provider '{provider}' are marked as bad. Resetting...")
        bad_keys_db.reset_provider_keys(provider)
        good_keys = api_keys.copy()

    while True:
        curr_content = finish_reason = None
        response_contents = []
        attempt_count += 1
        
        # Get the next good key to try
        if not good_keys:
            # All good keys exhausted in this session, reset and try again
            logging.warning(f"All available keys exhausted for provider '{provider}'. Resetting bad keys database...")
            bad_keys_db.reset_provider_keys(provider)
            good_keys = api_keys.copy()
        
        current_api_key = good_keys[(attempt_count - 1) % len(good_keys)]

        try:
            async with new_msg.channel.typing():
                thought_signature = None
                async for delta_content, new_finish_reason, new_grounding_metadata, new_thought_signature in get_stream(current_api_key):
                    if new_grounding_metadata:
                        grounding_metadata = new_grounding_metadata

                    if new_thought_signature:
                        thought_signature = new_thought_signature

                    if finish_reason != None:
                        break

                    finish_reason = new_finish_reason

                    prev_content = curr_content or ""
                    curr_content = delta_content

                    new_content = prev_content if finish_reason == None else (prev_content + curr_content)

                    if response_contents == [] and new_content == "":
                        continue

                    if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                        response_contents.append("")

                    response_contents[-1] += new_content

                    if not use_plain_responses:
                        time_delta = datetime.now().timestamp() - last_edit_time

                        ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                        msg_split_incoming = finish_reason == None and len(response_contents[-1] + curr_content) > max_message_length
                        is_final_edit = finish_reason != None or msg_split_incoming
                        is_good_finish = finish_reason != None and any(x in str(finish_reason).lower() for x in ("stop", "end_turn"))

                        if start_next_msg or ready_to_edit or is_final_edit:
                            embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                            embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                            view = SourceView(grounding_metadata) if is_final_edit and grounding_metadata and grounding_metadata.web_search_queries else None

                            msg_index = len(response_contents) - 1
                            if start_next_msg:
                                if msg_index < len(response_msgs):
                                    await response_msgs[msg_index].edit(embed=embed, view=view)
                                else:
                                    await reply_helper(embed=embed, silent=True, view=view)
                            else:
                                await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
                                await response_msgs[msg_index].edit(embed=embed, view=view)

                            last_edit_time = datetime.now().timestamp()

            if not response_contents:
                raise Exception("Response stream ended with no content")

            if use_plain_responses:
                for content in response_contents:
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))

            break

        except Exception as e:
            logging.exception("Error while generating response")
            
            # Mark the current key as bad
            error_msg = str(e)[:200] if e else "Unknown error"
            bad_keys_db.mark_key_bad(provider, current_api_key, error_msg)
            
            # Remove the bad key from good_keys list for this session
            if current_api_key in good_keys:
                good_keys.remove(current_api_key)
            
            # Check if we've exhausted all keys after reset
            if not good_keys:
                # Check if we've already tried resetting once in this run
                if attempt_count >= len(api_keys) * 2:  # Tried all keys at least twice
                    error_text = "I encountered too many errors with all available API keys. Please try again later."

                    if use_plain_responses:
                        await reply_helper(content=error_text)
                        response_contents = [error_text]
                    else:
                        embed.description = error_text
                        embed.color = EMBED_COLOR_INCOMPLETE
                        if response_msgs:
                            await response_msgs[-1].edit(embed=embed, view=None)
                        else:
                            await reply_helper(embed=embed)
                        response_contents = [error_text]
                    break
                else:
                    # Reset and try again with all keys
                    logging.warning(f"All keys exhausted for provider '{provider}'. Resetting for retry...")
                    bad_keys_db.reset_provider_keys(provider)
                    good_keys = api_keys.copy()

    if not use_plain_responses and len(response_msgs) > len(response_contents):
        for msg in response_msgs[len(response_contents):]:
            await msg.delete()
            if msg.id in msg_nodes:
                msg_nodes[msg.id].lock.release()
                del msg_nodes[msg.id]
        response_msgs = response_msgs[:len(response_contents)]

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        if thought_signature:
            msg_nodes[response_msg.id].thought_signature = thought_signature
        msg_nodes[response_msg.id].lock.release()

    # Update the last message with ResponseView for "View Response Better" button
    if not use_plain_responses and response_msgs and response_contents:
        full_response = "".join(response_contents)
        response_view = ResponseView(full_response, grounding_metadata)
        
        # Update the last message with the final view
        last_msg_index = len(response_msgs) - 1
        if last_msg_index < len(response_contents):
            embed.description = response_contents[last_msg_index]
            embed.color = EMBED_COLOR_COMPLETE
            await response_msgs[last_msg_index].edit(embed=embed, view=response_view)

    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def upload_to_textis(text: str) -> Optional[str]:
    """
    Upload text to text.is and return the paste URL.
    Returns None if upload fails.
    """
    try:
        async with httpx.AsyncClient(follow_redirects=False) as client:
            # Get the CSRF token from the main page
            response = await client.get("https://text.is/", timeout=30)
            
            # Extract CSRF token from the form
            soup = BeautifulSoup(response.text, "html.parser")
            csrf_input = soup.find("input", {"name": "csrfmiddlewaretoken"})
            if not csrf_input:
                logging.error("Could not find CSRF token on text.is")
                return None
            
            csrf_token = csrf_input.get("value")
            
            # Get cookies from the response
            cookies = response.cookies
            
            # POST the text content
            form_data = {
                "csrfmiddlewaretoken": csrf_token,
                "text": text,
            }
            
            headers = {
                "Referer": "https://text.is/",
                "Origin": "https://text.is",
            }
            
            post_response = await client.post(
                "https://text.is/",
                data=form_data,
                headers=headers,
                cookies=cookies,
                timeout=30
            )
            
            # The response should be a redirect (302) to the paste URL
            if post_response.status_code in (301, 302, 303, 307, 308):
                paste_url = post_response.headers.get("Location")
                if paste_url:
                    # Handle relative URLs
                    if paste_url.startswith("/"):
                        paste_url = f"https://text.is{paste_url}"
                    return paste_url
            
            # If we got a 200, the paste might have been created and we're on the page
            if post_response.status_code == 200:
                # Check if the URL changed (we might be on the paste page)
                final_url = str(post_response.url)
                if final_url != "https://text.is/" and "text.is/" in final_url:
                    return final_url
            
            logging.error(f"Unexpected response from text.is: {post_response.status_code}")
            return None
            
    except Exception as e:
        logging.exception(f"Error uploading to text.is: {e}")
        return None


class ResponseView(discord.ui.View):
    """View with 'View Response Better' button that uploads to text.is"""
    
    def __init__(self, full_response: str, metadata: Optional[types.GroundingMetadata] = None):
        super().__init__(timeout=None)
        self.full_response = full_response
        self.metadata = metadata
        
        # Only add sources button if we have metadata with search queries
        if metadata and metadata.web_search_queries:
            self.add_item(SourceButton(metadata))

    @discord.ui.button(label="View Response Better", style=discord.ButtonStyle.secondary, emoji="📄")
    async def view_response_better(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True)
        
        paste_url = await upload_to_textis(self.full_response)
        
        if paste_url:
            embed = discord.Embed(
                title="View Response",
                description=f"Your response has been uploaded for better viewing:\n\n**[Click here to view]({paste_url})**",
                color=discord.Color.green()
            )
            embed.set_footer(text="Powered by text.is")
            await interaction.followup.send(embed=embed, ephemeral=True)
        else:
            await interaction.followup.send(
                "❌ Failed to upload response to text.is. Please try again later.",
                ephemeral=True
            )


class SourceButton(discord.ui.Button):
    """Button to show sources from grounding metadata"""
    
    def __init__(self, metadata: types.GroundingMetadata):
        super().__init__(label="Show Sources", style=discord.ButtonStyle.secondary)
        self.metadata = metadata
    
    async def callback(self, interaction: discord.Interaction):
        embed = discord.Embed(title="Sources", color=discord.Color.blue())

        if self.metadata.web_search_queries:
            embed.add_field(name="Search Queries", value="\n".join(f"• {q}" for q in self.metadata.web_search_queries), inline=False)

        if self.metadata.grounding_chunks:
            sources = []
            for i, chunk in enumerate(self.metadata.grounding_chunks):
                if chunk.web:
                    sources.append(f"{i+1}. [{chunk.web.title}]({chunk.web.uri})")

            if sources:
                current_chunk = ""
                field_count = 1
                for source in sources:
                    if len(current_chunk) + len(source) + 1 > 1024:
                        embed.add_field(name=f"Search Results ({field_count})" if field_count > 1 else "Search Results", value=current_chunk, inline=False)
                        current_chunk = source
                        field_count += 1
                    else:
                        current_chunk = (current_chunk + "\n" + source) if current_chunk else source

                if current_chunk:
                    embed.add_field(name=f"Search Results ({field_count})" if field_count > 1 else "Search Results", value=current_chunk, inline=False)

        await interaction.response.send_message(embed=embed, ephemeral=True)


class SourceView(discord.ui.View):
    """Legacy view for backwards compatibility - now using ResponseView instead"""
    def __init__(self, metadata: types.GroundingMetadata):
        super().__init__(timeout=None)
        self.metadata = metadata

    @discord.ui.button(label="Show Sources", style=discord.ButtonStyle.secondary)
    async def show_sources(self, interaction: discord.Interaction, button: discord.ui.Button):
        embed = discord.Embed(title="Sources", color=discord.Color.blue())

        if self.metadata.web_search_queries:
            embed.add_field(name="Search Queries", value="\n".join(f"• {q}" for q in self.metadata.web_search_queries), inline=False)

        if self.metadata.grounding_chunks:
            sources = []
            for i, chunk in enumerate(self.metadata.grounding_chunks):
                if chunk.web:
                    sources.append(f"{i+1}. [{chunk.web.title}]({chunk.web.uri})")

            if sources:
                current_chunk = ""
                field_count = 1
                for source in sources:
                    if len(current_chunk) + len(source) + 1 > 1024:
                        embed.add_field(name=f"Search Results ({field_count})" if field_count > 1 else "Search Results", value=current_chunk, inline=False)
                        current_chunk = source
                        field_count += 1
                    else:
                        current_chunk = (current_chunk + "\n" + source) if current_chunk else source

                if current_chunk:
                    embed.add_field(name=f"Search Results ({field_count})" if field_count > 1 else "Search Results", value=current_chunk, inline=False)

        await interaction.response.send_message(embed=embed, ephemeral=True)


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
    await asyncio.gather(start_server(), discord_bot.start(config["bot_token"]))


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
