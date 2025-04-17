import asyncio
from base64 import b64encode
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime as dt, timedelta, timezone
import logging
import random
import re
from typing import Literal, Optional, List, Tuple, Dict, Any, Set
import os
import io # Import io for file handling
import sqlite3
import time
from urllib.parse import urlparse, parse_qs
import math # Import math for ceiling division

import discord # type: ignore
from discord import app_commands
import discord.ui
import httpx
import yaml
from bs4 import BeautifulSoup
import asyncpraw
from serpapi import GoogleSearch as SerpApiSearch

# Conditional imports based on provider
from openai import AsyncOpenAI, RateLimitError as OpenAIRateLimitError, APIError as OpenAIAPIError, APIConnectionError as OpenAIConnectionError, AuthenticationError as OpenAIAuthenticationError
from google import genai as google_genai
from google.genai import types as google_genai_types
from google.genai import errors as google_genai_errors
from google.api_core import exceptions as google_api_core_exceptions
from asyncprawcore import exceptions as asyncprawcore_exceptions

# YouTube specific imports
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from googleapiclient.discovery import build as build_google_api_client
from googleapiclient.errors import HttpError as GoogleApiHttpError

logging.basicConfig(
    level=logging.DEBUG, # Set to DEBUG to capture detailed logs
    format="%(asctime)s %(levelname)s: %(message)s",
)

# --- Constants ---
VISION_MODEL_TAGS = ("gpt-4", "claude-3", "gemini", "gemma", "llama", "pixtral", "mistral-small", "vision", "vl", "flash", "pro")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1.3 # Slightly increased delay

# URL Regex patterns
YOUTUBE_URL_PATTERN = re.compile(
    r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/|v\/|shorts\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
)
REDDIT_URL_PATTERN = re.compile(
    r'(https?://(?:www\.)?reddit\.com/r/[a-zA-Z0-9_]+/comments/[a-zA-Z0-9_]+(?:/[^/ ]*)?)|(https?://(?:www\.)?reddit\.com/r/[a-zA-Z0-9_]+/s/[a-zA-Z0-9_]+)'
)
GENERAL_URL_PATTERN = re.compile(r'https?://\S+')

# Cache and DB settings
MAX_MESSAGE_NODES = 250 # Increased cache size slightly
RATE_LIMIT_DB_DIR = "ratelimit_db"
RATE_LIMIT_COOLDOWN_SECONDS = 24 * 60 * 60 # 24 hours
DB_CLEANUP_INTERVAL_SECONDS = 60 * 60 # Check every hour

# --- Configuration Loading ---

# User-specific model selections (in-memory for now)
# Stores {user_id: "provider/model_name"}
user_model_preferences: Dict[int, str] = {}

# Define allowed models per provider for the slash command
ALLOWED_MODELS = {
    "google": ["gemini-2.0-flash", "gemini-2.5-pro-exp-03-25"],
    "openai": ["gpt-4.1", "o3-mini"],
}

def get_config(filename="config.yaml"):
    try:
        with open(filename, "r", encoding='utf-8') as file:
            config = yaml.safe_load(file)
            # --- Validate and Structure API Keys ---
            if 'providers' in config:
                for provider_name, provider_cfg in config['providers'].items():
                    if 'api_key' in provider_cfg and isinstance(provider_cfg['api_key'], str):
                        # Convert single key to list for consistency
                        provider_cfg['api_keys'] = [provider_cfg['api_key']]
                        del provider_cfg['api_key'] # Remove old single key field
                    elif 'api_keys' not in provider_cfg or not isinstance(provider_cfg['api_keys'], list):
                         # Ensure api_keys list exists, even if empty (for providers like ollama)
                         if provider_name != 'google': # Google requires keys
                             provider_cfg['api_keys'] = []
                         elif 'api_keys' not in provider_cfg:
                              logging.warning(f"Provider '{provider_name}' requires 'api_keys' list in config.yaml.")
                              provider_cfg['api_keys'] = [] # Initialize empty to prevent errors, but log warning

            # Handle SerpAPI keys
            if 'serpapi_api_key' in config and isinstance(config['serpapi_api_key'], str):
                config['serpapi_api_keys'] = [config['serpapi_api_key']]
                del config['serpapi_api_key']
            elif 'serpapi_api_keys' not in config or not isinstance(config['serpapi_api_keys'], list):
                config['serpapi_api_keys'] = []

            return config
    except FileNotFoundError:
        logging.error(f"CRITICAL: {filename} not found. Please copy config-example.yaml to {filename} and configure it.")
        exit(1)
    except yaml.YAMLError as e:
        logging.error(f"CRITICAL: Error parsing {filename}: {e}")
        exit(1)
    except Exception as e:
        logging.error(f"CRITICAL: Unexpected error loading config: {e}")
        exit(1)

cfg = get_config()

# --- Rate Limit Database Manager ---
os.makedirs(RATE_LIMIT_DB_DIR, exist_ok=True)

class RateLimitManager:
    def __init__(self, service_name: str):
        self.db_path = os.path.join(RATE_LIMIT_DB_DIR, f"{service_name}_ratelimit.db")
        self._lock = asyncio.Lock()
        self._conn = None
        self._cursor = None
        # Store the full list of keys for this service
        self.all_keys = self._get_keys_from_config(service_name)
        # Shuffle keys once at initialization for random rotation order
        random.shuffle(self.all_keys)
        logging.info(f"Initialized RateLimitManager for {service_name} with {len(self.all_keys)} keys.")

    def _get_keys_from_config(self, service_name: str) -> List[str]:
        """Helper to get the list of keys for a service from the global config."""
        if service_name == "serpapi":
            return cfg.get("serpapi_api_keys", [])
        elif service_name in cfg.get("providers", {}):
            return cfg["providers"][service_name].get("api_keys", [])
        else:
            return []

    async def _connect(self):
        """Establish SQLite connection."""
        if self._conn is None:
            self._conn = await asyncio.to_thread(sqlite3.connect, self.db_path, check_same_thread=False)
            self._cursor = self._conn.cursor()
            await asyncio.to_thread(self._cursor.execute, '''
                CREATE TABLE IF NOT EXISTS rate_limited_keys (
                    key TEXT PRIMARY KEY,
                    timestamp REAL
                )
            ''')
            await asyncio.to_thread(self._conn.commit)

    async def close(self):
        """Close the database connection."""
        async with self._lock:
            if self._conn:
                await asyncio.to_thread(self._conn.close)
                self._conn = None
                self._cursor = None

    async def add_key(self, key: str):
        """Mark a key as rate-limited with the current timestamp."""
        async with self._lock:
            await self._connect()
            current_time = time.time()
            try:
                await asyncio.to_thread(
                    self._cursor.execute,
                    "INSERT OR REPLACE INTO rate_limited_keys (key, timestamp) VALUES (?, ?)",
                    (key, current_time)
                )
                await asyncio.to_thread(self._conn.commit)
                logging.warning(f"Key starting with '{key[:4]}...' for service {os.path.basename(self.db_path).split('_')[0]} marked as rate-limited.")
            except sqlite3.Error as e:
                logging.error(f"SQLite error adding key {key[:4]}...: {e}")

    async def is_rate_limited(self, key: str) -> bool:
        """Check if a key is currently rate-limited (within the cooldown period)."""
        async with self._lock:
            await self._connect()
            try:
                await asyncio.to_thread(
                    self._cursor.execute,
                    "SELECT timestamp FROM rate_limited_keys WHERE key = ?",
                    (key,)
                )
                result = await asyncio.to_thread(self._cursor.fetchone)
                if result:
                    timestamp = result[0]
                    if time.time() - timestamp < RATE_LIMIT_COOLDOWN_SECONDS:
                        return True
                    else:
                        # Cooldown expired, remove the key from the DB
                        await asyncio.to_thread(
                            self._cursor.execute,
                            "DELETE FROM rate_limited_keys WHERE key = ?",
                            (key,)
                        )
                        await asyncio.to_thread(self._conn.commit)
                        logging.info(f"Rate limit cooldown expired for key starting with '{key[:4]}...'. Removed from DB.")
                        return False
                return False
            except sqlite3.Error as e:
                logging.error(f"SQLite error checking key {key[:4]}...: {e}")
                return False # Assume not rate-limited if DB error occurs

    async def get_available_keys(self) -> List[str]:
        """Get a list of keys that are not currently rate-limited, maintaining shuffled order."""
        available_keys = []
        for key in self.all_keys: # Iterate through the pre-shuffled list
            if not await self.is_rate_limited(key):
                available_keys.append(key)
        return available_keys

    async def get_all_limited_keys_in_db(self) -> Set[str]:
        """Returns a set of all keys currently present in the rate-limit DB."""
        async with self._lock:
            await self._connect()
            try:
                await asyncio.to_thread(
                    self._cursor.execute,
                    "SELECT key FROM rate_limited_keys"
                )
                results = await asyncio.to_thread(self._cursor.fetchall)
                return {row[0] for row in results}
            except sqlite3.Error as e:
                logging.error(f"SQLite error getting all limited keys: {e}")
                return set()

    async def reset_if_all_limited(self) -> bool:
        """Checks if all keys are rate-limited and resets the DB if true."""
        if not self.all_keys: # No keys configured
            return False

        limited_keys_in_db = await self.get_all_limited_keys_in_db()
        all_keys_set = set(self.all_keys)

        # Check if the set of keys in the DB is exactly the same as all configured keys
        if limited_keys_in_db == all_keys_set:
            logging.warning(f"All {len(self.all_keys)} keys for service {os.path.basename(self.db_path).split('_')[0]} are rate-limited. Resetting database.")
            await self.reset_db()
            return True
        return False

    async def reset_db(self):
        """Delete all entries from the rate-limited keys table."""
        async with self._lock:
            await self._connect()
            try:
                await asyncio.to_thread(
                    self._cursor.execute,
                    "DELETE FROM rate_limited_keys"
                )
                await asyncio.to_thread(self._conn.commit)
                logging.info(f"Rate limit database reset for {os.path.basename(self.db_path).split('_')[0]}.")
            except sqlite3.Error as e:
                logging.error(f"SQLite error resetting database: {e}")

    async def cleanup_expired_keys(self):
        """Remove keys whose cooldown period has expired."""
        async with self._lock:
            await self._connect()
            cutoff_time = time.time() - RATE_LIMIT_COOLDOWN_SECONDS
            try:
                await asyncio.to_thread(
                    self._cursor.execute,
                    "DELETE FROM rate_limited_keys WHERE timestamp < ?",
                    (cutoff_time,)
                )
                deleted_count = self._cursor.rowcount
                await asyncio.to_thread(self._conn.commit)
                if deleted_count > 0:
                    logging.info(f"Cleaned up {deleted_count} expired rate-limited keys for {os.path.basename(self.db_path).split('_')[0]}.")
            except sqlite3.Error as e:
                logging.error(f"SQLite error during cleanup: {e}")

# --- Initialize Rate Limit Managers ---
rate_limit_managers: Dict[str, RateLimitManager] = {}
# Initialize for LLM providers listed in config
for provider_name in cfg.get("providers", {}):
    rate_limit_managers[provider_name] = RateLimitManager(provider_name)
# Initialize for SerpAPI
rate_limit_managers["serpapi"] = RateLimitManager("serpapi")

# --- Background Cleanup Task ---
async def cleanup_task():
    while True:
        await asyncio.sleep(DB_CLEANUP_INTERVAL_SECONDS)
        logging.debug("Running periodic rate limit cleanup...")
        for manager in rate_limit_managers.values():
            await manager.cleanup_expired_keys()

# --- Discord Client Setup ---
if client_id := cfg.get("client_id"): # Use .get for safety
    logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/api/oauth2/authorize?client_id={client_id}&permissions=412317273088&scope=bot\n")

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(cfg.get("status_message") or "github.com/jakobdylanc/llmcord")[:128])
discord_client = discord.Client(intents=intents, activity=activity)
tree = app_commands.CommandTree(discord_client)

# --- Global Clients and State ---
httpx_client = httpx.AsyncClient()
msg_nodes: Dict[int, 'MsgNode'] = {}
active_tasks: Dict[int, asyncio.Lock] = {} # Store active generation tasks
youtube_transcript_api_client = YouTubeTranscriptApi() # Initialize transcript client once
youtube_data_service = None # Initialize later if key exists
youtube_api_key = cfg.get("youtube_api_key") # Store key for potential re-init
reddit_client = None # Initialize later if config exists

# --- Data Classes ---
@dataclass
class MsgNode:
    # Content fields
    text: Optional[str] = None
    images: list = field(default_factory=list) # Stores tuples of (mime_type, bytes)
    youtube_content: Optional[str] = None
    reddit_content: Optional[str] = None
    generic_url_content: Optional[str] = None
    google_lens_content: Optional[str] = None

    # Role/User info
    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    # Metadata fields
    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False
    grounding_metadata: Optional[google_genai_types.GroundingMetadata] = None

    # Structure fields
    parent_msg: Optional[discord.Message] = None
    next_segment_msg_id: Optional[int] = None # Added field to link split messages

    # Concurrency control
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

class ResponseActionsView(discord.ui.View):
    """A View to show buttons for interacting with the bot's response."""
    def __init__(self, message_id: int, grounding_metadata: Optional[google_genai_types.GroundingMetadata], timeout=None):
        super().__init__(timeout=timeout)
        # IMPORTANT: message_id MUST be updated if the view is attached to a new message
        # This is handled in update_response_messages
        self.message_id = message_id
        self.grounding_metadata = grounding_metadata

        # Conditionally add the "Show Sources" button if grounding metadata exists
        # Check for actual content in grounding metadata
        has_queries = bool(getattr(self.grounding_metadata, 'web_search_queries', []))
        # Ensure grounding_chunks is iterable before checking its contents
        grounding_chunks = getattr(self.grounding_metadata, 'grounding_chunks', None) or [] # FIX: Ensure iterable
        has_web_chunks = any(hasattr(chunk, 'web') for chunk in grounding_chunks)

        if self.grounding_metadata and (has_queries or has_web_chunks):
            # Create and add the ShowSourcesButton instance
            self.add_item(self.ShowSourcesButton(parent_view=self)) # Pass reference to parent view

        # Always add the "Get Output File" button
        self.add_item(self.GetOutputFileButton(parent_view=self)) # Pass reference to parent view

    # --- Button Definitions as Inner Classes ---
    class ShowSourcesButton(discord.ui.Button):
        def __init__(self, parent_view: 'ResponseActionsView'):
            # Use a simpler, unique ID based on the parent view's initial message_id
            # The actual message_id used in the callback comes from parent_view.message_id
            super().__init__(label="Show Sources", style=discord.ButtonStyle.secondary, custom_id=f"show_sources_{parent_view.message_id}")
            self.parent_view = parent_view

        async def callback(self, interaction: discord.Interaction):
            await interaction.response.defer(thinking=True)
            # Use the potentially updated message_id from the parent view
            message_id = self.parent_view.message_id
            node = msg_nodes.get(message_id)

            # Use grounding_metadata stored in the parent view
            metadata = self.parent_view.grounding_metadata

            if not node or not metadata:
                # Check the node directly as well, in case the view's metadata is stale (less likely but possible)
                if node and node.grounding_metadata:
                    metadata = node.grounding_metadata
                else:
                    await interaction.followup.send("Sorry, I couldn't find the grounding sources for this message.")
                    return

            embed = discord.Embed(title="Grounding Information", color=discord.Color.blue())

            # Display Search Queries
            if queries := getattr(metadata, 'web_search_queries', []):
                query_text = "\n".join(f"- `{q}`" for q in queries)
                if len(query_text) > 1024:
                    query_text = query_text[:1021] + "..."
                embed.add_field(name="Search Queries Used", value=query_text, inline=False)

            # Display Grounding Sources (URIs and Titles)
            sources_text = []
            # Ensure grounding_chunks is iterable before checking its contents
            chunks = getattr(metadata, 'grounding_chunks', None) or [] # FIX: Ensure iterable
            if chunks:
                for i, chunk in enumerate(chunks):
                    if web_info := getattr(chunk, 'web', None):
                        title = getattr(web_info, 'title', 'Unknown Title')
                        uri = getattr(web_info, 'uri', None)
                        if uri:
                            title_display = title[:100] + ('...' if len(title) > 100 else '')
                            sources_text.append(f"{i+1}. [{title_display}]({uri})")

            if sources_text:
                source_value = "\n".join(sources_text)
                if len(source_value) > 1024:
                    source_value = source_value[:1021] + "..."
                embed.add_field(name="Sources Found", value=source_value, inline=False)

            if not embed.fields:
                embed.description = "No grounding information (queries or sources) available."

            await interaction.followup.send(embed=embed)

    class GetOutputFileButton(discord.ui.Button):
        def __init__(self, parent_view: 'ResponseActionsView'):
            # Use a simpler, unique ID based on the parent view's initial message_id
            super().__init__(label="Get Output as a Text File", style=discord.ButtonStyle.secondary, custom_id=f"get_output_file_{parent_view.message_id}")
            self.parent_view = parent_view

        async def callback(self, interaction: discord.Interaction):
            """Sends the full response text as a text file."""
            await interaction.response.defer(thinking=True)

            # Use the potentially updated message_id from the parent view
            message_id = self.parent_view.message_id
            last_segment_node = msg_nodes.get(message_id)

            if not last_segment_node or not last_segment_node.parent_msg:
                await interaction.followup.send("Sorry, couldn't link this response to its origin.")
                return

            original_user_msg_id = last_segment_node.parent_msg.id

            # --- Find the first segment node ---
            response_segment_nodes = {} # Store node_id: node
            next_segment_ids = set()
            for node_id, node in msg_nodes.items():
                # Check if it's an assistant response to the correct user message
                if node.role == "assistant" and node.parent_msg and node.parent_msg.id == original_user_msg_id:
                    response_segment_nodes[node_id] = node
                    if node.next_segment_msg_id:
                        next_segment_ids.add(node.next_segment_msg_id)

            first_segment_node = None
            # Find the node whose ID is not pointed to by any other node in this chain
            for node_id, node in response_segment_nodes.items():
                if node_id not in next_segment_ids:
                    first_segment_node = node
                    break # Found the head

            if not first_segment_node:
                # Fallback: If only one segment exists, it must be the one the button is on
                if message_id in response_segment_nodes:
                    first_segment_node = response_segment_nodes[message_id]
                else:
                    await interaction.followup.send("Sorry, couldn't determine the start of the response chain.")
                    return
            # --- End Find the first segment node ---


            # --- Traverse forward from the first segment node ---
            full_response_text_parts = []
            curr_node = first_segment_node
            processed_segment_ids_traversal = set()

            while curr_node is not None:
                # Find the message ID associated with the current node object
                current_node_id = None
                for msg_id, node_obj in response_segment_nodes.items():
                    if node_obj == curr_node:
                        current_node_id = msg_id
                        break

                # Safety check for loops or missing nodes during traversal
                if current_node_id is None or current_node_id in processed_segment_ids_traversal:
                    logging.warning(f"Traversal loop detected or node ID not found for response chain starting from user msg {original_user_msg_id}")
                    break

                processed_segment_ids_traversal.add(current_node_id)

                node_text = curr_node.text
                if node_text:
                    full_response_text_parts.append(node_text)

                next_segment_id = curr_node.next_segment_msg_id
                # Ensure the next node exists in the global cache before proceeding
                if next_segment_id and next_segment_id in msg_nodes:
                    curr_node = msg_nodes.get(next_segment_id)
                else:
                    break # End of chain
            # --- End Traverse forward ---

            if not full_response_text_parts:
                 await interaction.followup.send("The response content appears to be empty after processing.")
                 return

            # Join the collected parts WITHOUT extra newlines
            full_response_text = "".join(full_response_text_parts) # FIX: Use empty string join

            # Create a text file in memory
            file_content = io.BytesIO(full_response_text.encode('utf-8'))
            discord_file = discord.File(fp=file_content, filename="llm_response.txt")

            await interaction.followup.send("Here is the full response:", file=discord_file)


# --- Helper Functions ---

def extract_video_id(url: str) -> Optional[str]:
    """Extracts the YouTube video ID from a URL."""
    match = YOUTUBE_URL_PATTERN.search(url)
    return match.group(1) if match else None

async def fetch_youtube_transcript(video_id: str) -> tuple[Optional[str], Optional[str]]:
    """Fetches transcript asynchronously. Returns (transcript_text, error_message)."""
    try:
        transcript_list = await asyncio.to_thread(youtube_transcript_api_client.list_transcripts, video_id)
        fetched_transcript_obj = None
        try:
            transcript = transcript_list.find_manually_created_transcript(['en'])
            fetched_transcript_obj = await asyncio.to_thread(transcript.fetch)
        except NoTranscriptFound:
            try:
                transcript = transcript_list.find_generated_transcript(['en'])
                fetched_transcript_obj = await asyncio.to_thread(transcript.fetch)
            except NoTranscriptFound:
                found_transcripts = list(transcript_list)
                if found_transcripts:
                    original_transcript = found_transcripts[0]
                    if original_transcript.is_translatable:
                        try:
                            translated_transcript_obj = await asyncio.to_thread(original_transcript.translate, 'en')
                            fetched_transcript_obj = await asyncio.to_thread(translated_transcript_obj.fetch)
                        except Exception as translate_e:
                            logging.warning(f"Failed to translate transcript for {video_id} from {original_transcript.language}: {translate_e}")
                            fetched_transcript_obj = await asyncio.to_thread(original_transcript.fetch)
                    else:
                         fetched_transcript_obj = await asyncio.to_thread(original_transcript.fetch)

        if fetched_transcript_obj:
            # youtube-transcript-api returns a list of dicts
            full_transcript = " ".join([snippet['text'] for snippet in fetched_transcript_obj])
            return full_transcript, None
        else:
            return None, "No suitable transcript found."

    except TranscriptsDisabled:
        return None, "Transcripts are disabled for this video."
    except VideoUnavailable:
         return None, "Video is unavailable (possibly private or deleted)."
    except Exception as e:
        if "VideoUnavailable" in str(e): # Heuristic check
             return None, "Video is unavailable (possibly private or deleted)."
        logging.error(f"Error fetching transcript for {video_id}: {e}")
        return None, f"An error occurred fetching transcript: {type(e).__name__}"

async def fetch_youtube_details_and_comments(video_id: str) -> tuple[Optional[dict], Optional[str]]:
    """Fetches video details and comments using YouTube Data API. Returns (details_dict, error_message)."""
    if not youtube_data_service:
        return None, "YouTube Data API service not initialized (missing API key?)."
    try:
        video_response = await asyncio.to_thread(
            youtube_data_service.videos().list(part='snippet,statistics', id=video_id).execute
        )
        if not video_response.get('items'):
            return None, "Video not found via YouTube Data API."

        comment_response = await asyncio.to_thread(
            youtube_data_service.commentThreads().list(part='snippet', videoId=video_id, order='relevance', maxResults=50, textFormat='plainText').execute
        )
        return {'video': video_response, 'comments': comment_response}, None
    except GoogleApiHttpError as e:
        error_content = e.content.decode('utf-8') if e.content else str(e)
        logging.error(f"YouTube Data API error for {video_id}: {error_content}")
        if 'commentsDisabled' in error_content:
            try:
                video_response_only = await asyncio.to_thread(
                    youtube_data_service.videos().list(part='snippet,statistics', id=video_id).execute
                )
                if not video_response_only.get('items'):
                     return None, "Video not found via YouTube Data API (checked after comment error)."
                return {'video': video_response_only, 'comments': None}, "Comments are disabled for this video."
            except Exception as e_vid:
                 logging.error(f"YouTube Data API error fetching video details after comment error for {video_id}: {e_vid}")
                 if isinstance(e_vid, GoogleApiHttpError):
                     return None, f"YouTube Data API error (fetching details): {e_vid.resp.status} {e_vid._get_reason()}"
                 else:
                     return None, f"Unexpected error fetching video details after comment error: {type(e_vid).__name__}"
        elif e.resp.status == 403:
             reason = "Unknown Forbidden"
             try:
                 # Attempt to parse error details if available
                 error_details = e.error_details
                 if error_details and isinstance(error_details, list) and len(error_details) > 0:
                     reason = error_details[0].get('reason', reason)
             except AttributeError: # Handle cases where error_details might not exist
                 pass
             except Exception as parse_e:
                 logging.warning(f"Could not parse YouTube API error details: {parse_e}")

             if reason == 'quotaExceeded':
                 return None, "YouTube Data API quota exceeded."
             else:
                 return None, f"YouTube Data API request forbidden ({reason}). Check API key permissions."
        elif e.resp.status == 404:
             # This might indicate the video ID itself is wrong, even if the API call is valid
             return None, "Video not found via YouTube Data API."
        else:
            return None, f"YouTube Data API error: {e.resp.status} {e._get_reason()}"
    except Exception as e:
        logging.error(f"Unexpected error fetching YouTube details for {video_id}: {e}")
        return None, f"An unexpected error occurred fetching video details: {type(e).__name__}"

async def fetch_reddit_content(url: str) -> tuple[Optional[str], Optional[str]]:
    """Fetches content from a Reddit URL asynchronously. Returns (formatted_content, error_message)."""
    global reddit_client
    if not reddit_client:
        return None, "Reddit client not initialized (missing config?)."

    try:
        # Use url=url for asyncpraw to handle redirects (including share links)
        submission = await reddit_client.submission(url=url)
        # Eagerly load basic attributes
        await submission.load()

        # Check if submission is valid (not removed, deleted, etc.)
        if not submission or getattr(submission, 'removed_by_category', None) or not hasattr(submission, 'title'):
             # Check if it's just a redirect to a subreddit (sometimes happens with invalid comment links)
             if isinstance(submission, asyncpraw.models.Subreddit):
                 return None, "Link points to a subreddit, not a specific submission."
             return None, "Submission not found or removed."

        content_parts = []
        content_parts.append(f"Title: {submission.title}")
        if submission.selftext:
            # Truncate selftext if long
            content_parts.append(f"Body: {(submission.selftext[:1500] + '...') if len(submission.selftext) > 1500 else submission.selftext}")

        # Fetch top comments (limit to e.g., 5 top-level comments)
        content_parts.append("Top Comments:")
        comment_count = 0
        # Ensure comments are loaded before iterating
        await submission.comments.replace_more(limit=0) # Replace top-level MoreComments objects
        async for top_level_comment in submission.comments:
            if isinstance(top_level_comment, asyncpraw.models.MoreComments):
                continue # Skip MoreComments objects if any remain
            if top_level_comment.stickied: # Skip stickied comments
                continue
            if comment_count >= 5:
                break
            # Truncate comment
            comment_text = top_level_comment.body
            truncated_comment = (comment_text[:250] + '...') if len(comment_text) > 250 else comment_text
            content_parts.append(f"- {truncated_comment}")
            comment_count += 1

        return "\n".join(content_parts), None

    except asyncprawcore_exceptions.Redirect:
        return None, "Could not follow Reddit redirect (possibly invalid share link)."
    except (asyncprawcore_exceptions.NotFound, asyncprawcore_exceptions.Forbidden) as e: # Handle 404 Not Found and 403 Forbidden (private subreddit)
        return None, f"Could not access Reddit content ({type(e).__name__}). It might be private, quarantined, or deleted."
    except Exception as e:
        logging.error(f"Error fetching Reddit content for {url}: {e}")
        return None, f"An unexpected error occurred fetching Reddit content: {type(e).__name__}"

async def fetch_generic_url_content(url: str) -> tuple[Optional[str], Optional[str]]:
    """Fetches and extracts text content from a generic URL asynchronously. Returns (text_content, error_message)."""
    MAX_CONTENT_LENGTH = 3000 # Limit extracted text length per URL
    try:
        # Use a reasonable timeout
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status() # Raise exception for non-2xx status codes

        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            return None, f"Skipped: Content type is '{content_type}', not HTML."

        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text from main content areas or paragraphs, clean it up
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        if main_content:
            # Remove script and style elements
            for script_or_style in main_content(['script', 'style']):
                script_or_style.decompose()
            text_content = main_content.get_text(separator=' ', strip=True)
        else:
            text_content = soup.get_text(separator=' ', strip=True) # Fallback

        return text_content[:MAX_CONTENT_LENGTH] + ('...' if len(text_content) > MAX_CONTENT_LENGTH else ''), None

    except httpx.RequestError as e:
        logging.warning(f"HTTP error fetching generic URL {url}: {e}")
        return None, f"Network error accessing URL: {type(e).__name__}"
    except Exception as e:
        logging.error(f"Error processing generic URL {url}: {e}")
        return None, f"Failed to process URL content: {type(e).__name__}"

async def fetch_google_lens_results(image_url: str, api_key: str) -> tuple[Optional[str], Optional[str]]:
    """Fetches Google Lens results for an image URL using SerpApi. Returns (formatted_results, error_message)."""
    params = {
        "engine": "google_lens",
        "url": image_url,
        "api_key": api_key,
        "safe": "off", # Optional: Adjust safety filter if needed
    }

    try:
        # SerpApi library is synchronous, run it in a thread
        search = SerpApiSearch(params)
        results = await asyncio.to_thread(search.get_dict)

        if error := results.get("error"):
            # Check for specific rate limit / quota errors from SerpApi response
            if "Your account has run out of searches" in error or "exceeds the hourly throughput limit" in error:
                raise httpx.HTTPStatusError(error, request=None, response=httpx.Response(429)) # Simulate a 429 error
            logging.warning(f"SerpApi Google Lens error for {image_url}: {error}")
            return None, f"SerpApi Error: {error}"

        if results.get("search_metadata", {}).get("status") != "Success":
            status = results.get("search_metadata", {}).get("status", "Unknown Status")
            logging.warning(f"SerpApi Google Lens search failed for {image_url}. Status: {status}")
            # Treat certain statuses as potentially retryable or indicative of issues
            if status == "Error": # Generic error might warrant retry with different key
                 raise httpx.HTTPStatusError(f"SerpApi Search Failed (Status: {status})", request=None, response=httpx.Response(503)) # Simulate 503
            return None, f"SerpApi Search Failed (Status: {status})"

        visual_matches = results.get("visual_matches", [])
        formatted_results = "\n".join([f"- {match.get('title', 'N/A')} ({match.get('source', 'N/A')}) Link: <{match.get('link', '#')}>" for match in visual_matches[:5]]) # Limit to top 5 matches
        return formatted_results if formatted_results else "No visual matches found.", None

    except httpx.HTTPStatusError as e: # Catch simulated rate limit error
        raise e # Re-raise to be caught by the main retry loop
    except Exception as e:
        logging.error(f"Exception during SerpApi Google Lens call for {image_url}: {e}")
        # Treat unexpected errors as potentially retryable server issues
        raise httpx.HTTPStatusError(f"Unexpected SerpApi Error: {type(e).__name__}", request=None, response=httpx.Response(500)) # Simulate 500

def format_youtube_data(video_id: str, transcript: Optional[str], details: Optional[dict], errors: list[str]) -> str:
    """Formats extracted YouTube data into a string for the LLM."""
    lines = []
    video_info = details.get('video', {}).get('items', [{}])[0] if details and details.get('video') else {}
    snippet = video_info.get('snippet', {})
    comments_info = details.get('comments', {}) if details else {}
    # Ensure comments_info is not None before accessing 'items'
    comment_items = comments_info.get('items', []) if comments_info else []

    # Title
    title = snippet.get('title', 'N/A')
    lines.append(f"Title: {title}")

    # Channel Name
    channel_title = snippet.get('channelTitle', 'N/A')
    lines.append(f"Channel: {channel_title}")

    # Description (truncate if long)
    description = snippet.get('description', 'N/A')
    if description and description != 'N/A':
        lines.append(f"Description: {(description[:500] + '...') if len(description) > 500 else description}")
    else:
        lines.append("Description: N/A")

    # Transcript (truncate if long)
    if transcript:
        lines.append(f"Transcript: {(transcript[:2000] + '...') if len(transcript) > 2000 else transcript}")
    else:
        # Add transcript error if transcript is None and not already in errors
        if not any("Transcript:" in e for e in errors):
             errors.append("- Transcript: Could not be retrieved.")


    # Comments
    if comment_items:
        lines.append("Top Comments:")
        for i, item in enumerate(comment_items):
            # Safely navigate the comment structure
            top_level_comment = item.get('snippet', {}).get('topLevelComment', {})
            comment_snippet = top_level_comment.get('snippet', {})
            comment_text = comment_snippet.get('textDisplay', 'N/A') # Use textDisplay for formatted text
            if comment_text == 'N/A':
                 comment_text = comment_snippet.get('textOriginal', 'N/A') # Fallback to original

            # Truncate comment
            truncated_comment = (comment_text[:200] + '...') if len(comment_text) > 200 else comment_text
            lines.append(f"- {truncated_comment}")
    elif details and 'comments' in details and details['comments'] is None and not any("Comments are disabled" in e for e in errors):
         # Explicitly state if comments are disabled and not already in errors
         errors.append("- Comments: Comments are disabled for this video.")
    elif not any("Comments:" in e for e in errors): # Add generic comment error if no items and no specific error
         errors.append("- Comments: Could not retrieve comments.")


    if errors:
        lines.append("\nExtraction Errors/Notes:")
        lines.extend(errors)

    return "\n".join(lines)

async def update_response_messages(
    initial_user_msg: discord.Message, # Renamed parameter for clarity
    response_msgs: list[discord.Message],
    full_response_text: str,
    base_embed: discord.Embed, # Pass the base embed with warnings
    is_final_chunk: bool,
    msg_nodes: dict,
    max_length: int,
    model_name_display: str, # Added model name for footer
    final_view: Optional[discord.ui.View] = None
):
    """Handles sending/editing multiple messages for a potentially long response."""
    # Handle empty response case explicitly
    if not full_response_text and is_final_chunk:
        num_messages_needed = 1
        full_response_text = "(empty response)" # Display placeholder
    else:
        num_messages_needed = max(1, math.ceil(len(full_response_text) / max_length))


    for i in range(num_messages_needed):
        segment_start = i * max_length
        segment_end = (i + 1) * max_length
        text_segment = full_response_text[segment_start:segment_end]

        is_last_segment_for_this_update = (i == num_messages_needed - 1)
        is_absolute_final_segment = is_final_chunk and is_last_segment_for_this_update

        # Create a new embed for this segment
        segment_embed = base_embed.copy() # Start with the base embed (contains warnings)
        segment_embed.description = text_segment + ("" if is_absolute_final_segment else STREAMING_INDICATOR)
        segment_embed.color = EMBED_COLOR_COMPLETE if is_absolute_final_segment else EMBED_COLOR_INCOMPLETE
        segment_embed.set_footer(text=f"Model: {model_name_display}")

        # Determine the view (only for the absolute final segment)
        current_segment_view = None
        if is_absolute_final_segment and isinstance(final_view, ResponseActionsView):
            # If the intended final view is ResponseActionsView, ensure its message_id is set correctly
            # If we are editing, use the existing message's ID. If sending new, we need the new ID.
            if i < len(response_msgs):
                # Ensure the view instance targets the correct message ID
                final_view.message_id = response_msgs[i].id
                current_segment_view = final_view
            else:
                # We can't set the correct message_id before sending, so we'll attach it *after* sending.
                # Send without the view first.
                pass # current_segment_view remains None for now
        elif is_absolute_final_segment and final_view is None:
             # If it's the final segment but no view is needed (e.g., no grounding)
             current_segment_view = None


        try:
            previous_msg_id = response_msgs[i-1].id if i > 0 and i <= len(response_msgs) else None

            if i < len(response_msgs):
                # Edit existing message
                target_msg = response_msgs[i]
                # Update the view's message_id if it's the final segment being edited
                if is_absolute_final_segment and isinstance(final_view, ResponseActionsView):
                    final_view.message_id = target_msg.id
                    current_segment_view = final_view # Use the updated view
                await target_msg.edit(embed=segment_embed, view=current_segment_view)
                # Ensure link from previous segment if it exists
                if previous_msg_id and previous_msg_id in msg_nodes:
                     msg_nodes[previous_msg_id].next_segment_msg_id = target_msg.id

            else:
                # Send new message
                reply_target = response_msgs[-1] if response_msgs else initial_user_msg # Use the renamed parameter
                # Send without view first if view needs ID set later
                view_to_send = None if (is_absolute_final_segment and isinstance(final_view, ResponseActionsView)) else current_segment_view
                new_msg_sent = await reply_target.reply(embed=segment_embed, mention_author = False, view=view_to_send)
                response_msgs.append(new_msg_sent)

                # Create and lock node for the new message
                if new_msg_sent.id not in msg_nodes:
                    # Parent should be the original user message that triggered this whole response sequence
                    msg_nodes[new_msg_sent.id] = MsgNode(parent_msg=initial_user_msg)
                    await msg_nodes[new_msg_sent.id].lock.acquire()
                    logging.debug(f"Created and locked MsgNode for new segment message {new_msg_sent.id}")

                # Link previous segment to this new one
                if previous_msg_id and previous_msg_id in msg_nodes:
                     msg_nodes[previous_msg_id].next_segment_msg_id = new_msg_sent.id

                # Attach ResponseActionsView if needed (after getting the message ID)
                if is_absolute_final_segment and isinstance(final_view, ResponseActionsView) and current_segment_view is None:
                    # Update the view's internal message_id before attaching
                    final_view.message_id = new_msg_sent.id
                    await new_msg_sent.edit(view=final_view)

        except (discord.NotFound, discord.HTTPException) as e:
             logging.warning(f"Failed to send or edit message segment {i}: {e}")
             # If editing failed, try sending a new message for this segment
             if i < len(response_msgs):
                 failed_msg_ref = response_msgs.pop(i) # Remove the failed message reference
                 logging.info(f"Edit failed for segment {i} (msg {failed_msg_ref.id}), attempting to send as new reply.")
                 # Retry sending this segment
                 try:
                     reply_target = response_msgs[-1] if response_msgs else initial_user_msg # Use renamed parameter
                     # Determine view again for the new message
                     view_to_send_retry = None # Default to no view initially
                     if is_absolute_final_segment and isinstance(final_view, ResponseActionsView):
                          # Send without view first, attach after
                          pass
                     else:
                          view_to_send_retry = current_segment_view # Use original view intent

                     new_msg_sent_retry = await reply_target.reply(embed=segment_embed, mention_author = False, view=view_to_send_retry)
                     # Insert the new message at the correct position
                     response_msgs.insert(i, new_msg_sent_retry)

                     if new_msg_sent_retry.id not in msg_nodes:
                          msg_nodes[new_msg_sent_retry.id] = MsgNode(parent_msg=initial_user_msg) # Use renamed parameter
                          await msg_nodes[new_msg_sent_retry.id].lock.acquire()
                          logging.debug(f"Created and locked MsgNode for replacement segment message {new_msg_sent_retry.id}")

                     # Link previous segment if possible after retry
                     previous_msg_id_retry = response_msgs[i-1].id if i > 0 and i <= len(response_msgs) else None
                     if previous_msg_id_retry and previous_msg_id_retry in msg_nodes:
                          msg_nodes[previous_msg_id_retry].next_segment_msg_id = new_msg_sent_retry.id

                     # Attach ResponseActionsView if needed after sending retry
                     if is_absolute_final_segment and isinstance(final_view, ResponseActionsView) and view_to_send_retry is None:
                          # Update the view's internal message_id before attaching
                          final_view.message_id = new_msg_sent_retry.id
                          await new_msg_sent_retry.edit(view=final_view)

                 except Exception as send_e:
                     logging.error(f"Failed to send new message segment {i} after edit failed: {send_e}")
             # If sending the initial message failed, we can't do much more here

    # Clean up any extra messages if the response shrank
    while len(response_msgs) > num_messages_needed:
         msg_to_delete = response_msgs.pop()
         try:
             await msg_to_delete.delete()
             if msg_to_delete.id in msg_nodes:
                 # Release lock if held before deleting node
                 if msg_nodes[msg_to_delete.id].lock.locked():
                     msg_nodes[msg_to_delete.id].lock.release()
                 del msg_nodes[msg_to_delete.id]
         except (discord.NotFound, discord.HTTPException) as e:
             logging.warning(f"Failed to delete superfluous message {msg_to_delete.id}: {e}")


# --- Slash Command Definition ---

async def provider_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    """Autocompletes the provider parameter."""
    providers = list(ALLOWED_MODELS.keys())
    return [
        app_commands.Choice(name=provider, value=provider)
        for provider in providers if current.lower() in provider.lower()
    ]

async def model_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    """Autocompletes the model parameter based on the selected provider."""
    provider = interaction.data['options'][0].get('value') # Get the value of the 'provider' option
    if not provider or provider not in ALLOWED_MODELS:
        return []

    models = ALLOWED_MODELS[provider]
    return [
        app_commands.Choice(name=model, value=model)
        for model in models if current.lower() in model.lower()
    ]

@tree.command(name="model", description="Set your preferred LLM provider and model.")
@app_commands.describe(provider="The LLM provider (e.g., google, openai).", model="The specific model name.")
@app_commands.autocomplete(provider=provider_autocomplete)
@app_commands.autocomplete(model=model_autocomplete)
async def set_model(interaction: discord.Interaction, provider: str, model: str):
    """Allows users to set their preferred LLM model."""
    user_id = interaction.user.id

    # Validate provider and model
    if provider not in ALLOWED_MODELS:
        await interaction.response.send_message(f"Invalid provider '{provider}'. Allowed providers: {', '.join(ALLOWED_MODELS.keys())}")
        return
    if model not in ALLOWED_MODELS[provider]:
        await interaction.response.send_message(f"Invalid model '{model}' for provider '{provider}'. Allowed models: {', '.join(ALLOWED_MODELS[provider])}")
        return

    # Store the user's preference
    model_selection = f"{provider}/{model}"
    user_model_preferences[user_id] = model_selection
    logging.info(f"User {user_id} set model preference to {model_selection}")

    await interaction.response.send_message(f"Your LLM model has been set to `{model_selection}`.")


# --- Main Event Handler ---
@discord_client.event
async def on_message(new_msg): # new_msg is the initial message from the user
    global msg_nodes, youtube_data_service, reddit_client

    # --- Basic Checks ---
    if new_msg.author.bot or new_msg.id in active_tasks:
        return

    is_dm = new_msg.channel.type == discord.ChannelType.private
    is_mention = discord_client.user in new_msg.mentions
    content_lower_stripped = new_msg.content.strip().lower()
    is_at_ai_mention = content_lower_stripped.startswith("at ai")
    is_at_ai_mention_orig = new_msg.content.lower().startswith("at ai") # Check original for trigger logic

    if not is_dm and not is_mention and not is_at_ai_mention_orig:
        return

    # --- Reload Config and Check Permissions ---
    global cfg
    cfg = get_config() # Reload config

    allow_dms = cfg.get("allow_dms", True)
    permissions = cfg.get("permissions", {})
    user_perms = permissions.get("users", {"allowed_ids": [], "blocked_ids": []})
    role_perms = permissions.get("roles", {"allowed_ids": [], "blocked_ids": []})
    channel_perms = permissions.get("channels", {"allowed_ids": [], "blocked_ids": []})

    role_ids = {role.id for role in getattr(new_msg.author, "roles", [])}
    channel_ids = {new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None)} - {None}

    allow_all_users = not user_perms["allowed_ids"] if is_dm else not user_perms["allowed_ids"] and not role_perms["allowed_ids"]
    is_good_user = allow_all_users or new_msg.author.id in user_perms["allowed_ids"] or any(id in role_perms["allowed_ids"] for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in user_perms["blocked_ids"] or any(id in role_perms["blocked_ids"] for id in role_ids)

    allow_all_channels = not channel_perms["allowed_ids"]
    is_good_channel = allow_dms if is_dm else allow_all_channels or any(id in channel_perms["allowed_ids"] for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in channel_perms["blocked_ids"] for id in channel_ids)

    if is_bad_user or is_bad_channel:
        logging.debug(f"Message {new_msg.id} blocked due to permissions.")
        return

    # --- Initialize External Services (if not already done) ---
    # Re-check and initialize YouTube service if key exists and service is None
    if youtube_api_key and not youtube_data_service:
        try:
            youtube_data_service = await asyncio.to_thread(
                build_google_api_client, 'youtube', 'v3', developerKey=youtube_api_key
            )
            logging.info("YouTube Data API client initialized successfully.")
        except Exception as e: # Catch potential HttpError or others during build
            logging.error(f"Failed to initialize YouTube Data API client: {e}")
            youtube_data_service = None

    if cfg.get("reddit") and not reddit_client:
        reddit_cfg = cfg["reddit"]
        if all(k in reddit_cfg for k in ['client_id', 'client_secret', 'user_agent']):
            try:
                reddit_client = asyncpraw.Reddit(
                    client_id=reddit_cfg['client_id'],
                    client_secret=reddit_cfg['client_secret'],
                    user_agent=reddit_cfg['user_agent'],
                )
                logging.info("Async PRAW Reddit client initialized successfully (read-only).")
            except Exception as e:
                logging.error(f"Failed to initialize asyncpraw Reddit client: {e}")
                reddit_client = None
        else:
            logging.warning("Reddit config found but incomplete. Reddit processing disabled.")
            reddit_client = None

    # --- Task Locking ---
    task_lock = asyncio.Lock()
    active_tasks[new_msg.id] = task_lock
    async with task_lock: # Acquire lock for this message ID

        # --- Determine LLM Provider and Model (User Preference or Default) ---
        user_id = new_msg.author.id
        provider_slash_model = user_model_preferences.get(user_id, cfg.get("model", "openai/gpt-4.1"))
        try:
            # Validate the selected model (either user's or default) against config
            provider, model_name = provider_slash_model.split("/", 1)
            if provider not in cfg.get("providers", {}):
                 logging.error(f"Provider '{provider}' not found in config.yaml providers list.")
                 await new_msg.reply(f"⚠️ Configured LLM provider '{provider}' not found.", mention_author = False)
                 active_tasks.pop(new_msg.id, None)
                 return
        except ValueError:
            logging.error(f"Invalid model format determined: '{provider_slash_model}'. Should be 'provider/model_name'.")
            await new_msg.reply("⚠️ Invalid model format in configuration.", mention_author = False)
            active_tasks.pop(new_msg.id, None)
            return

        is_google_provider = provider == "google"
        llm_rate_manager = rate_limit_managers.get(provider)
        serpapi_rate_manager = rate_limit_managers.get("serpapi")

        # --- Configuration Values ---
        accept_images = any(tag in model_name.lower() for tag in VISION_MODEL_TAGS)
        accept_usernames = provider in PROVIDERS_SUPPORTING_USERNAMES

        max_text = cfg.get("max_text", 100000)
        max_images = cfg.get("max_images", 5) if accept_images else 0
        max_messages = cfg.get("max_messages", 25)
        use_plain_responses = cfg.get("use_plain_responses", False)
        # Use a slightly smaller max length for embeds to be safe with potential variations
        max_message_length = 1990 if use_plain_responses else 4000

        # --- Build Message Chain ---
        api_history = []
        user_warnings = set()
        youtube_context_string = ""
        reddit_context_string = ""
        generic_url_context_string = ""
        google_lens_context_string = ""
        curr_msg = new_msg
        processed_ids = set() # Prevent infinite loops

        while curr_msg is not None and len(api_history) < max_messages and curr_msg.id not in processed_ids:
            processed_ids.add(curr_msg.id)
            curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

            async with curr_node.lock:
                # --- Process current node (fetch content if needed) ---
                if curr_node.text is None:
                    # Fetch raw message content
                    raw_content = curr_msg.content
                    cleaned_content = raw_content.strip() # Start with basic stripping
                    is_lens_command = False

                    # Clean mention/trigger only for the *initial* message being processed in this chain walk
                    # This check needs refinement if we combine segments later. Let's assume initial message is new_msg for now.
                    if curr_msg.id == new_msg.id:
                        if is_mention:
                            cleaned_content = raw_content.replace(f'<@{discord_client.user.id}>', '', 1).lstrip()
                            if cleaned_content == raw_content.lstrip() and new_msg.guild and new_msg.guild.me.display_name:
                                cleaned_content = raw_content.replace(f'@{new_msg.guild.me.display_name}', '', 1).lstrip()
                        elif is_at_ai_mention_orig:
                            if raw_content.lower().startswith("at ai ") and len(raw_content) >= 6: cleaned_content = raw_content[6:].lstrip()
                            elif raw_content.lower() == "at ai": cleaned_content = ""
                            elif raw_content.lower().startswith("at ai") and len(raw_content) >= 5: cleaned_content = raw_content[5:].lstrip()
                            else: cleaned_content = raw_content # Fallback

                        # Check for 'lens' command *after* cleaning trigger
                        if cleaned_content.lower().startswith("lens "):
                            is_lens_command = True
                            cleaned_content = cleaned_content[5:].lstrip()

                    # --- URL and Attachment Processing (only for initial message) ---
                    if curr_msg.id == new_msg.id:
                        # YouTube
                        youtube_urls_found = YOUTUBE_URL_PATTERN.findall(cleaned_content)
                        unique_video_ids = list(dict.fromkeys(youtube_urls_found))
                        if unique_video_ids:
                            logging.info(f"Found YouTube video IDs: {unique_video_ids}")
                            youtube_data_tasks = [asyncio.gather(fetch_youtube_transcript(vid), fetch_youtube_details_and_comments(vid), return_exceptions=True) for vid in unique_video_ids]
                            results = await asyncio.gather(*youtube_data_tasks)
                            youtube_content_parts = []
                            for i, (vid, result_pair) in enumerate(zip(unique_video_ids, results)):
                                url = f"https://www.youtube.com/watch?v={vid}"
                                content_part = f"youtube url {i+1}: {url}\nyoutube url {i+1} content:\n"
                                errors = []
                                transcript_res, details_res = result_pair
                                transcript, t_err = transcript_res if not isinstance(transcript_res, BaseException) else (None, str(transcript_res))
                                details, d_err = details_res if not isinstance(details_res, BaseException) else (None, str(details_res))
                                if t_err: errors.append(f"- Transcript: {t_err}")
                                if d_err: errors.append(f"- Details/Comments: {d_err}")
                                content_part += format_youtube_data(vid, transcript, details, errors)
                                youtube_content_parts.append(content_part)
                            youtube_context_string = "\n\n".join(youtube_content_parts)
                            curr_node.youtube_content = youtube_context_string

                        # Reddit
                        if reddit_client:
                            reddit_url_matches = list(REDDIT_URL_PATTERN.finditer(cleaned_content))
                            unique_reddit_urls = list(dict.fromkeys([match.group(0) for match in reddit_url_matches]))
                            if unique_reddit_urls:
                                logging.info(f"Found Reddit URLs: {unique_reddit_urls}")
                                reddit_tasks = [fetch_reddit_content(url) for url in unique_reddit_urls]
                                results = await asyncio.gather(*reddit_tasks, return_exceptions=True)
                                reddit_content_parts = []
                                for i, (url, res) in enumerate(zip(unique_reddit_urls, results)):
                                    content_part = f"reddit url {i+1}: {url}\nreddit url {i+1} content:\n"
                                    text, err = res if not isinstance(res, BaseException) else (None, str(res))
                                    content_part += text if text else f"Could not extract content: {err or 'No text found.'}"
                                    reddit_content_parts.append(content_part)
                                reddit_context_string = "\n\n".join(reddit_content_parts)
                                curr_node.reddit_content = reddit_context_string

                        # Generic URLs
                        all_urls = GENERAL_URL_PATTERN.findall(cleaned_content)
                        yt_urls = {m.group(0) for m in YOUTUBE_URL_PATTERN.finditer(cleaned_content)}
                        rd_urls = {m.group(0) for m in REDDIT_URL_PATTERN.finditer(cleaned_content)}
                        generic_urls = []
                        for url in all_urls:
                            url = url.rstrip('.,;!?')
                            if url not in yt_urls and url not in rd_urls:
                                try:
                                    p = urlparse(url)
                                    if p.scheme in ['http', 'https'] and '.' in p.netloc and len(url) < 500 and not url.startswith('data:'):
                                        generic_urls.append(url)
                                except ValueError: continue
                        unique_generic_urls = list(dict.fromkeys(generic_urls))
                        if unique_generic_urls:
                            logging.info(f"Found generic URLs: {unique_generic_urls}")
                            generic_tasks = [fetch_generic_url_content(url) for url in unique_generic_urls]
                            results = await asyncio.gather(*generic_tasks, return_exceptions=True)
                            generic_content_parts = []
                            for i, (url, res) in enumerate(zip(unique_generic_urls, results)):
                                content_part = f"url {i+1}: {url}\nurl {i+1} content:\n"
                                text, err = res if not isinstance(res, BaseException) else (None, str(res))
                                content_part += text if text else f"Could not extract content: {err or 'No text/error.'}"
                                generic_content_parts.append(content_part)
                            generic_url_context_string = "\n\n".join(generic_content_parts)
                            curr_node.generic_url_content = generic_url_context_string

                        # Google Lens (SerpAPI)
                        if is_lens_command and serpapi_rate_manager and serpapi_rate_manager.all_keys:
                            image_attachments = [att for att in curr_msg.attachments if att.content_type and att.content_type.startswith("image")]
                            if image_attachments:
                                if len(image_attachments) > max_images:
                                    user_warnings.add(f"⚠️ Too many images ({len(image_attachments)} > {max_images}). Using first {max_images} for Lens.")
                                    image_attachments = image_attachments[:max_images]

                                logging.info(f"Processing {len(image_attachments)} image(s) with Google Lens...")
                                lens_content_parts = []
                                lens_errors = []
                                for i, attachment in enumerate(image_attachments):
                                    lens_result_text = lens_error = None
                                    last_lens_error = None
                                    available_serp_keys = await serpapi_rate_manager.get_available_keys()
                                    if not available_serp_keys:
                                        lens_error = "All SerpApi keys are currently rate-limited."
                                        await serpapi_rate_manager.reset_if_all_limited() # Check if reset is needed
                                    else:
                                        for key_index, key in enumerate(available_serp_keys):
                                            try:
                                                lens_result_text, lens_error = await fetch_google_lens_results(attachment.url, key)
                                                if lens_result_text is not None: # Success
                                                    break # Stop trying keys for this image
                                                else: # Error occurred, but maybe not rate limit
                                                    last_lens_error = lens_error or "Unknown SerpApi error"
                                                    # Continue to next key if error wasn't fatal
                                            except httpx.HTTPStatusError as http_err:
                                                last_lens_error = str(http_err)
                                                if http_err.response.status_code in [429, 500, 503]: # Rate limit or server error
                                                    await serpapi_rate_manager.add_key(key)
                                                    if await serpapi_rate_manager.reset_if_all_limited():
                                                         # If reset happened, get fresh list for next image
                                                         available_serp_keys = await serpapi_rate_manager.get_available_keys()
                                                         # Need to restart the key loop for *this* image with fresh keys
                                                         # This is complex, for now just log and fail this image if reset happens mid-loop
                                                         logging.warning("SerpApi DB reset during Lens processing. May need to retry message.")
                                                         lens_error = "SerpApi keys reset during processing."
                                                         break # Break inner key loop for this image
                                                    if key_index == len(available_serp_keys) - 1: # Last key also failed
                                                        lens_error = f"All available SerpApi keys failed. Last error: {last_lens_error}"
                                                else: # Other HTTP error (e.g., 401, 403) - likely key issue, but don't mark rate limit
                                                    logging.error(f"Non-rate-limit HTTP error {http_err.response.status_code} with SerpApi key {key[:4]}...")
                                                    if key_index == len(available_serp_keys) - 1:
                                                         lens_error = f"All available SerpApi keys failed. Last error: {last_lens_error}"
                                            except Exception as e: # Catch other unexpected errors
                                                last_lens_error = f"Unexpected error: {type(e).__name__}"
                                                logging.error(f"Unexpected error during SerpApi call: {e}")
                                                if key_index == len(available_serp_keys) - 1:
                                                     lens_error = f"All available SerpApi keys failed. Last error: {last_lens_error}"

                                    # Append result or error for this image
                                    content_part = f"SerpAPI's Google Lens API results for image {i+1}:\n"
                                    content_part += lens_result_text if lens_result_text else f"Could not get results: {lens_error or 'Unknown error.'}"
                                    lens_content_parts.append(content_part)
                                    if lens_error and "rate-limited" in lens_error: # Add warning if all keys were limited
                                        user_warnings.add("⚠️ SerpApi rate limit hit.")

                                google_lens_context_string = "\n\n".join(lens_content_parts)
                                curr_node.google_lens_content = google_lens_context_string
                            elif not image_attachments:
                                user_warnings.add("⚠️ 'lens' command used, but no images attached.")
                        elif is_lens_command and (not serpapi_rate_manager or not serpapi_rate_manager.all_keys):
                             user_warnings.add("⚠️ 'lens' command used, but no SerpApi API key configured.")


                    # Process Attachments (Text and Image)
                    good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]
                    attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments], return_exceptions=True)

                    # Combine text content
                    embed_texts = ["\n".join(filter(None, (e.title, e.description, getattr(e.footer, 'text', None)))) for e in curr_msg.embeds if e.type == 'rich']
                    text_attachments = [resp.text for att, resp in zip(good_attachments, attachment_responses) if isinstance(resp, httpx.Response) and att.content_type.startswith("text")]

                    # Start with cleaned content (mention/trigger/lens removed if applicable)
                    combined_text_parts = [cleaned_content] if cleaned_content else []
                    combined_text_parts.extend(embed_texts)
                    combined_text_parts.extend(text_attachments)

                    # Append context strings
                    if curr_node.youtube_content: combined_text_parts.append(curr_node.youtube_content)
                    if curr_node.reddit_content: combined_text_parts.append(curr_node.reddit_content)
                    if curr_node.generic_url_content: combined_text_parts.append(curr_node.generic_url_content)
                    if curr_node.google_lens_content: combined_text_parts.append(curr_node.google_lens_content)

                    curr_node.text = "\n\n".join(filter(None, combined_text_parts)) # Join with double newline for clarity

                    # Store images
                    curr_node.images = [(att.content_type, resp.content) for att, resp in zip(good_attachments, attachment_responses) if isinstance(resp, httpx.Response) and att.content_type.startswith("image")]

                    curr_node.role = "assistant" if curr_msg.author == discord_client.user else "user"
                    curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None
                    curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments) or any(isinstance(r, BaseException) for r in attachment_responses)

                # --- Fetch Parent Message ---
                # (Existing logic for finding parent_msg remains largely the same)
                try:
                    parent_msg_to_fetch_id = None
                    is_reply = curr_msg.reference is not None
                    is_thread_start_reply = (
                        curr_msg.channel.type == discord.ChannelType.public_thread
                        and not is_reply
                        and hasattr(curr_msg.channel, 'parent')
                        and curr_msg.channel.parent.type == discord.ChannelType.text
                    )

                    if is_reply:
                        parent_msg_to_fetch_id = curr_msg.reference.message_id
                        curr_node.parent_msg = curr_msg.reference.cached_message
                        if not curr_node.parent_msg and parent_msg_to_fetch_id:
                            try: curr_node.parent_msg = await curr_msg.channel.fetch_message(parent_msg_to_fetch_id)
                            except (discord.NotFound, discord.HTTPException): curr_node.fetch_parent_failed = True; logging.warning(f"Failed to fetch replied-to message {parent_msg_to_fetch_id}")
                    elif is_thread_start_reply:
                        parent_msg_to_fetch_id = curr_msg.channel.id
                        curr_node.parent_msg = curr_msg.channel.starter_message
                        if not curr_node.parent_msg and parent_msg_to_fetch_id:
                             try: curr_node.parent_msg = await curr_msg.channel.parent.fetch_message(parent_msg_to_fetch_id)
                             except (discord.NotFound, discord.HTTPException, AttributeError): curr_node.fetch_parent_failed = True; logging.warning(f"Failed to fetch thread starter message {parent_msg_to_fetch_id}")
                    elif not is_dm and not is_mention and not is_at_ai_mention_orig: # Auto-chain previous in guilds
                         prev_msg = ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0]
                         if prev_msg and prev_msg.author == curr_msg.author and prev_msg.type in (discord.MessageType.default, discord.MessageType.reply):
                             curr_node.parent_msg = prev_msg
                    elif is_dm: # Auto-chain in DMs
                         prev_msg = ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0]
                         if prev_msg and prev_msg.type in (discord.MessageType.default, discord.MessageType.reply):
                              prev_is_mention = discord_client.user in prev_msg.mentions
                              prev_is_at_ai = prev_msg.content.lower().startswith("at ai")
                              if not prev_is_mention and not prev_is_at_ai:
                                   curr_node.parent_msg = prev_msg
                except Exception as e:
                    logging.exception(f"Unexpected error fetching parent message for {curr_msg.id}: {e}")
                    curr_node.fetch_parent_failed = True

            # --- Combine segments if it's an assistant message ---
            combined_text_content = curr_node.text
            combined_images = list(curr_node.images) # Start with current node's images
            combined_grounding_metadata = curr_node.grounding_metadata # Use metadata from the first segment initially

            if curr_node.role == "assistant":
                next_segment_id = curr_node.next_segment_msg_id
                while next_segment_id is not None and next_segment_id in msg_nodes and next_segment_id not in processed_ids:
                    processed_ids.add(next_segment_id) # Mark segment as processed
                    next_node = msg_nodes[next_segment_id]
                    async with next_node.lock: # Lock the next segment node
                        if next_node.text is not None: # Ensure node was processed
                            # Append text with a newline separator if both parts have text
                            if combined_text_content and next_node.text:
                                combined_text_content += "\n\n" + next_node.text
                            elif next_node.text:
                                combined_text_content = next_node.text

                            combined_images.extend(next_node.images)
                            # Keep grounding metadata from the *last* segment if available
                            if next_node.grounding_metadata:
                                combined_grounding_metadata = next_node.grounding_metadata
                        else:
                            # This shouldn't happen if locking is correct, but log if it does
                            logging.warning(f"Encountered unprocessed segment node {next_segment_id} while combining.")
                    next_segment_id = next_node.next_segment_msg_id # Move to the next segment

            # --- Construct API message format using combined content ---
            api_parts = []
            # Use combined_text_content, truncated if necessary
            final_text_content = combined_text_content[:max_text] if combined_text_content else ""

            if final_text_content:
                if is_google_provider:
                    api_parts.append(google_genai_types.Part(text=final_text_content))
                else:
                    api_parts.append({"type": "text", "text": final_text_content})

            # Use combined_images, truncated if necessary
            processed_images = 0
            for mime_type, img_bytes in combined_images:
                if processed_images < max_images:
                    if is_google_provider:
                        api_parts.append(google_genai_types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
                    else:
                        b64_img = b64encode(img_bytes).decode('utf-8')
                        api_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}})
                    processed_images += 1
                else: break

            if api_parts:
                if is_google_provider:
                    api_role = 'model' if curr_node.role == 'assistant' else 'user'
                    # Store combined grounding metadata with the combined assistant message
                    content_obj = google_genai_types.Content(parts=api_parts, role=api_role)
                    # Grounding metadata is stored on the node, view logic uses the node.
                    api_history.append(content_obj)
                else:
                    message_dict = {"role": curr_node.role, "content": api_parts}
                    if accept_usernames and curr_node.user_id is not None and curr_node.role == 'user':
                         message_dict["name"] = str(curr_node.user_id)
                    api_history.append(message_dict)

            # --- Add User Warnings (based on the *first* node of the segment) ---
            # Check original node's text length before combination for truncation warning
            if len(curr_node.text or "") > max_text: user_warnings.add(f"⚠️ Max {max_text:,} chars/msg")
            # Check original node's image count before combination
            if len(curr_node.images) > max_images: user_warnings.add(f"⚠️ Max {max_images} image(s)" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments: user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg is not None and len(api_history) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(api_history)} message(s)")

            # --- Move to Parent ---
            # Crucially, only move up the chain via parent_msg, not next_segment_msg_id
            curr_msg = curr_node.parent_msg

        # --- Prepare API Call ---
        api_history.reverse() # Correct order
        logging.info(f"Processing message {new_msg.id} (user: {new_msg.author.id}, history: {len(api_history)}, provider: {provider}, model: {model_name})")

        system_instruction = None
        if system_prompt := cfg.get("system_prompt"):
            system_prompt_extras = [f"Today's date: {dt.now().strftime('%B %d %Y')}."]
            if not is_google_provider and accept_usernames:
                system_prompt_extras.append("User's names are their Discord IDs and should be typed as '<@ID>'.")
            system_instruction = "\n".join([system_prompt] + system_prompt_extras)

        if system_instruction and not is_google_provider:
            api_history.insert(0, {"role": "system", "content": system_instruction})

        # --- Generate and Send Response (with Key Rotation and Retry) ---
        response_msgs = []
        final_error = None
        llm_call_succeeded = False
        response_contents = [] # Initialize here

        available_keys = await llm_rate_manager.get_available_keys()
        if not available_keys:
            final_error = f"All {provider} API keys are currently rate-limited."
            await llm_rate_manager.reset_if_all_limited() # Check if reset needed

        async with new_msg.channel.typing():
            for key_index, current_api_key in enumerate(available_keys):
                logging.info(f"Attempting LLM call with {provider} key index {key_index} (starts with {current_api_key[:4]}...)")
                response_contents = [] # Reset for each key attempt
                final_response_obj = None
                final_grounding_metadata = None
                final_prompt_feedback = None
                final_view_to_attach = None # Reset final view for each attempt
                stream_error = None # Track errors specifically during streaming

                # Create the base embed with warnings *before* the loop
                base_embed = discord.Embed()
                for warning in sorted(user_warnings):
                    base_embed.add_field(name=warning, value="", inline=False)

                # --- START DEBUG LOGGING ---
                logging.debug(f"--- Sending API Request (Key Index: {key_index}) ---")
                logging.debug(f"Provider: {provider}, Model: {model_name}")
                # Use repr for potentially large history to avoid overly long logs
                logging.debug(f"API History (repr): {repr(api_history)}")
                logging.debug(f"Extra Body: {cfg.get('extra_api_parameters', {})}")
                logging.debug(f"-------------------------------------------------")
                # --- END DEBUG LOGGING ---

                try:
                    # --- Initialize Client with Current Key ---
                    genai_client = None
                    openai_client = None
                    if is_google_provider:
                        genai_client = google_genai.Client(api_key=current_api_key)
                    else:
                        openai_client = AsyncOpenAI(
                            base_url=cfg["providers"][provider].get("base_url"),
                            api_key=current_api_key or "sk-no-key-required" # Handle optional key for local models
                        )

                    # --- API Call ---
                    if is_google_provider:
                        safety_settings = [
                            google_genai_types.SafetySetting(
                                category=google_genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                threshold=google_genai_types.HarmBlockThreshold.BLOCK_NONE),
                            google_genai_types.SafetySetting(
                                category=google_genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                threshold=google_genai_types.HarmBlockThreshold.BLOCK_NONE),
                            google_genai_types.SafetySetting(
                                category=google_genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                threshold=google_genai_types.HarmBlockThreshold.BLOCK_NONE),
                            google_genai_types.SafetySetting(
                                category=google_genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                threshold=google_genai_types.HarmBlockThreshold.BLOCK_NONE),
                            google_genai_types.SafetySetting(
                                category=google_genai_types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                                threshold=google_genai_types.HarmBlockThreshold.BLOCK_NONE),
                        ]
                        tools = [google_genai_types.Tool(google_search=google_genai_types.GoogleSearch())]
                        generation_config_dict = {k: v for k, v in cfg.get("extra_api_parameters", {}).items() if k in ['temperature', 'top_p', 'top_k', 'max_output_tokens', 'stop_sequences']}
                        config = google_genai_types.GenerateContentConfig(
                            tools=tools,
                            safety_settings=safety_settings,
                            system_instruction=system_instruction,
                            **generation_config_dict
                        )
                        stream = await genai_client.aio.models.generate_content_stream(
                            model=f"models/{model_name}",
                            contents=api_history,
                            config=config,
                        )
                    else: # OpenAI compatible
                        stream = await openai_client.chat.completions.create(
                            model=model_name,
                            messages=api_history,
                            stream=True,
                            extra_body=cfg.get("extra_api_parameters", {})
                        )

                    # --- Process Stream ---
                    last_chunk_time = 0
                    async for chunk in stream:
                        final_response_obj = chunk # Store last chunk for metadata
                        delta_text = ""

                        if is_google_provider:
                            if hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback:
                                final_prompt_feedback = chunk.prompt_feedback
                                # Check for immediate prompt blocking
                                if final_prompt_feedback.block_reason != google_genai_types.BlockReason.BLOCK_REASON_UNSPECIFIED:
                                     block_reason_str = final_prompt_feedback.block_reason.name
                                     logging.warning(f"Google GenAI prompt blocked with key {current_api_key[:4]}... Reason: {block_reason_str}")
                                     # Treat prompt blocking as a failure for this key, but not necessarily a rate limit
                                     stream_error = google_api_core_exceptions.InvalidArgument(f"Prompt blocked by safety filter: {block_reason_str}")
                                     break # Stop processing this stream

                            delta_text = getattr(chunk, 'text', None) or ""
                            # Check for response blocking mid-stream (less common but possible)
                            if hasattr(chunk, 'candidates') and chunk.candidates:
                                candidate = chunk.candidates[0]
                                finish_reason_enum = getattr(candidate, 'finish_reason', None)
                                if finish_reason_enum == google_genai_types.FinishReason.SAFETY:
                                    logging.warning(f"Google GenAI response blocked mid-stream due to safety with key {current_api_key[:4]}...")
                                    stream_error = google_api_core_exceptions.InvalidArgument("Response blocked by safety filter mid-stream.")
                                    break # Stop processing this stream
                        else: # OpenAI
                            delta_text = chunk.choices[0].delta.content or ""
                            finish_reason = chunk.choices[0].finish_reason
                            if finish_reason: # Check for finish reason (e.g., 'content_filter')
                                if finish_reason == 'content_filter':
                                     logging.warning(f"OpenAI response blocked mid-stream due to content filter with key {current_api_key[:4]}...")
                                     stream_error = OpenAIAPIError("Response blocked by content filter mid-stream.", request=None, body=None) # Simulate error
                                     break
                                elif finish_reason == 'length':
                                     # Add warning only if not already present
                                     if "⚠️ Max output tokens reached." not in user_warnings:
                                         user_warnings.add("⚠️ Max output tokens reached.")
                                         # Update base embed immediately
                                         base_embed.clear_fields() # Clear old warnings
                                         for warning in sorted(user_warnings): # Add updated warnings
                                             base_embed.add_field(name=warning, value="", inline=False)
                                # No need to break for 'stop' or 'tool_calls' here, just process delta

                        if delta_text:
                            response_contents.append(delta_text)

                        # Edit logic (only if no stream error yet)
                        if not stream_error:
                            current_time = dt.now().timestamp()
                            if not response_msgs or (current_time - last_chunk_time >= EDIT_DELAY_SECONDS):
                                if use_plain_responses:
                                    # Handle plain text streaming (send new message if needed)
                                    full_response = "".join(response_contents)
                                    num_plain_needed = max(1, math.ceil(len(full_response) / max_message_length))
                                    for i in range(num_plain_needed):
                                        text_segment = full_response[i * max_message_length : (i + 1) * max_message_length]
                                        is_last_segment = (i == num_plain_needed - 1)
                                        display_content = text_segment + ("" if is_last_segment else "...") # Indicate continuation
                                        previous_msg_id_plain = response_msgs[i-1].id if i > 0 and i <= len(response_msgs) else None

                                        if i < len(response_msgs):
                                            await response_msgs[i].edit(content=display_content, suppress_embeds=True, view=None)
                                            # Link previous segment
                                            if previous_msg_id_plain and previous_msg_id_plain in msg_nodes:
                                                msg_nodes[previous_msg_id_plain].next_segment_msg_id = response_msgs[i].id
                                        else:
                                            reply_target = response_msgs[-1] if response_msgs else new_msg
                                            new_msg_sent = await reply_target.reply(content=display_content, suppress_embeds=True, mention_author = False, view=None)
                                            response_msgs.append(new_msg_sent)
                                            if new_msg_sent.id not in msg_nodes:
                                                msg_nodes[new_msg_sent.id] = MsgNode(parent_msg=new_msg)
                                                await msg_nodes[new_msg_sent.id].lock.acquire()
                                            # Link previous segment
                                            if previous_msg_id_plain and previous_msg_id_plain in msg_nodes:
                                                msg_nodes[previous_msg_id_plain].next_segment_msg_id = new_msg_sent.id
                                else:
                                    # Handle embed streaming (calls the updated function)
                                    await update_response_messages(
                                        new_msg, response_msgs, "".join(response_contents),
                                        base_embed, False, msg_nodes, max_message_length, # Pass model name here
                                         provider_slash_model
                                    )
                                last_chunk_time = current_time

                    # --- Post-Stream Check (if stream finished without error) ---
                    if stream_error:
                        raise stream_error # Raise the error caught during streaming

                    # --- Final Checks and View Creation ---
                    if is_google_provider:
                        finish_reason_enum = None
                        if final_response_obj and hasattr(final_response_obj, 'candidates') and final_response_obj.candidates:
                            candidate = final_response_obj.candidates[0]
                            finish_reason_enum = getattr(candidate, 'finish_reason', None)
                            if finish_reason_enum == google_genai_types.FinishReason.SAFETY:
                                logging.warning(f"Google GenAI response blocked post-stream due to safety with key {current_api_key[:4]}...")
                                raise google_api_core_exceptions.InvalidArgument("Response blocked by safety filter post-stream.")
                            elif finish_reason_enum == google_genai_types.FinishReason.RECITATION:
                                 if "⚠️ Response potentially contains recited content." not in user_warnings:
                                     user_warnings.add("⚠️ Response potentially contains recited content.")
                                     base_embed.clear_fields(); [base_embed.add_field(name=w, value="", inline=False) for w in sorted(user_warnings)]
                            elif finish_reason_enum == google_genai_types.FinishReason.MAX_TOKENS:
                                 if "⚠️ Max output tokens reached." not in user_warnings:
                                     user_warnings.add("⚠️ Max output tokens reached.")
                                     base_embed.clear_fields(); [base_embed.add_field(name=w, value="", inline=False) for w in sorted(user_warnings)]

                            # Extract grounding metadata if response wasn't blocked
                            if hasattr(candidate, 'grounding_metadata'):
                                final_grounding_metadata = candidate.grounding_metadata
                                # Create the view instance here, message_id will be set later
                                # Pass grounding metadata to the view constructor
                                final_view_to_attach = ResponseActionsView(message_id=0, grounding_metadata=final_grounding_metadata) # Placeholder ID
                    else: # OpenAI or other providers
                         # Always create the view for non-Google providers (for the file button)
                         final_view_to_attach = ResponseActionsView(message_id=0, grounding_metadata=None) # Placeholder ID, no grounding


                    # --- Final Update/Send ---
                    full_response_text = "".join(response_contents)

                    if use_plain_responses:
                        # Final update for plain text
                        num_plain_needed = max(1, math.ceil(len(full_response_text) / max_message_length))
                        if not full_response_text and num_plain_needed == 1: # Handle truly empty response
                             text_segment = "(empty response)" # Placeholder for empty
                             # Ensure final_view_to_attach has the correct ID if needed (even for plain text, though view won't show)
                             if isinstance(final_view_to_attach, ResponseActionsView):
                                 if response_msgs: final_view_to_attach.message_id = response_msgs[0].id
                                 else: # Need to send first to get ID
                                     temp_msg = await new_msg.reply(content=text_segment, suppress_embeds=True, mention_author = False, view=None)
                                     final_view_to_attach.message_id = temp_msg.id
                                     # Don't attach view to plain text message
                                     response_msgs.append(temp_msg)
                                     # Skip the loop below as we've handled the only message
                                     num_plain_needed = 0 # Prevent loop execution
                             else: # No view needed
                                 if response_msgs: await response_msgs[0].edit(content=text_segment, suppress_embeds=True, view=None)
                                 else: response_msgs.append(await new_msg.reply(content=text_segment, suppress_embeds=True, mention_author = False, view=None))

                        for i in range(num_plain_needed):
                            text_segment = full_response_text[i * max_message_length : (i + 1) * max_message_length]
                            is_last_segment = (i == num_plain_needed - 1)
                            current_segment_view = None
                            previous_msg_id_final_plain = response_msgs[i-1].id if i > 0 and i <= len(response_msgs) else None
                            # Views are not attached to plain text, but we still need to manage the view object's state if it exists
                            if is_last_segment and isinstance(final_view_to_attach, ResponseActionsView):
                                # If we are editing the last message, set the ID now
                                if i < len(response_msgs):
                                    final_view_to_attach.message_id = response_msgs[i].id
                                    # current_segment_view remains None for plain text
                                # If sending new, view will be attached after sending

                            if i < len(response_msgs):
                                await response_msgs[i].edit(content=text_segment, suppress_embeds=True, view=current_segment_view)
                                # Link previous segment
                                if previous_msg_id_final_plain and previous_msg_id_final_plain in msg_nodes:
                                    msg_nodes[previous_msg_id_final_plain].next_segment_msg_id = response_msgs[i].id
                            else:
                                reply_target = response_msgs[-1] if response_msgs else new_msg
                                # Send without view first if it needs ID set later
                                view_to_send = None # Always None for plain text
                                new_msg_sent = await reply_target.reply(content=text_segment, suppress_embeds=True, mention_author = False, view=view_to_send)
                                response_msgs.append(new_msg_sent)
                                if new_msg_sent.id not in msg_nodes:
                                    msg_nodes[new_msg_sent.id] = MsgNode(parent_msg=new_msg)
                                    await msg_nodes[new_msg_sent.id].lock.acquire()
                                # Link previous segment
                                if previous_msg_id_final_plain and previous_msg_id_final_plain in msg_nodes:
                                    msg_nodes[previous_msg_id_final_plain].next_segment_msg_id = new_msg_sent.id
                                # Set message ID on the view object if it exists, even though not attached
                                if is_last_segment and isinstance(final_view_to_attach, ResponseActionsView):
                                    final_view_to_attach.message_id = new_msg_sent.id

                        # Clean up extra plain messages
                        while len(response_msgs) > num_plain_needed:
                            msg_to_delete = response_msgs.pop()
                            try: await msg_to_delete.delete()
                            except: pass
                            if msg_to_delete.id in msg_nodes:
                                if msg_nodes[msg_to_delete.id].lock.locked(): msg_nodes[msg_to_delete.id].lock.release()
                                del msg_nodes[msg_to_delete.id]

                    else:
                        # Final update for embeds
                        # Use the final_view_to_attach created earlier
                        await update_response_messages(
                            new_msg, response_msgs, full_response_text,
                            base_embed, True, msg_nodes, max_message_length,
                            provider_slash_model, final_view=final_view_to_attach # Pass the combined view
                        )

                    llm_call_succeeded = True
                    logging.info(f"LLM call successful with {provider} key index {key_index}.")
                    break # Exit key rotation loop on success

                # --- Exception Handling for the Current Key ---
                except (OpenAIRateLimitError, google_api_core_exceptions.ResourceExhausted) as e:
                    logging.warning(f"Rate limit error with {provider} key index {key_index} ({current_api_key[:4]}...): {e}")
                    await llm_rate_manager.add_key(current_api_key)
                    final_error = e
                    if await llm_rate_manager.reset_if_all_limited():
                        logging.warning(f"All {provider} keys rate-limited, DB reset. Stopping retries for this request.")
                        final_error = Exception(f"All {provider} API keys became rate-limited during this request.")
                        break # Stop trying keys for this request
                    # Continue to next key
                except (OpenAIConnectionError, google_api_core_exceptions.ServiceUnavailable, google_api_core_exceptions.DeadlineExceeded) as e:
                    logging.warning(f"Connection/Service error with {provider} key index {key_index} ({current_api_key[:4]}...): {e}. Retrying with next key.")
                    final_error = e
                    await asyncio.sleep(1) # Small delay before next key
                    # Continue to next key
                except (OpenAIAuthenticationError, google_api_core_exceptions.PermissionDenied) as e:
                     logging.error(f"Authentication error with {provider} key index {key_index} ({current_api_key[:4]}...): {e}. Skipping this key.")
                     # Don't mark as rate limited, but skip for this request cycle
                     final_error = e
                     # Continue to next key
                except (OpenAIAPIError, google_api_core_exceptions.InvalidArgument, google_api_core_exceptions.FailedPrecondition, google_genai_errors.ClientError) as e: # Added ClientError
                     # Handle errors like prompt/response blocking, invalid args etc.
                     logging.error(f"API Error (non-rate-limit) with {provider} key index {key_index} ({current_api_key[:4]}...): {e}")
                     final_error = e
                     # Decide if retryable. For safety blocks or bad args, maybe not.
                     # For now, let's retry with the next key unless it's clearly a permanent issue with the prompt itself.
                     error_str = str(e).lower()
                     if "safety filter" in error_str or "prompt" in error_str or "invalid" in error_str or "match is not a function" in error_str: # Added the specific error message check
                          logging.warning("Error seems related to prompt/safety/invalid arg/server bug, stopping retries for this request.")
                          break # Stop trying keys
                     # Continue to next key otherwise
                except Exception as e:
                    logging.exception(f"Unexpected error during API call with {provider} key index {key_index} ({current_api_key[:4]}...)")
                    final_error = e
                    # Decide if retryable. Let's retry for unexpected errors.
                    # Continue to next key

            # --- After Key Rotation Loop ---
            if not llm_call_succeeded:
                error_message = f"⚠️ LLM call failed after trying all available keys for provider {provider}." # Simplified initial message
                if final_error:
                    error_message += f" Last error: {type(final_error).__name__}"
                    # Provide more specific feedback for common errors
                    if isinstance(final_error, (OpenAIAuthenticationError, google_api_core_exceptions.PermissionDenied)):
                         error_message += " (Check API keys)"
                    elif isinstance(final_error, (OpenAIRateLimitError, google_api_core_exceptions.ResourceExhausted)):
                         error_message += " (Rate limit or quota exceeded)"
                    elif "safety filter" in str(final_error).lower():
                         error_message += " (Content blocked by safety filters)"
                    elif "invalid" in str(final_error).lower() or "match is not a function" in str(final_error).lower(): # Added check here too
                         error_message += " (Invalid request/argument or server error)"
                    # Add the raw error message if it's an APIError for more context
                    # FIX: Check for OpenAIAPIError and google_api_core_exceptions.GoogleAPICallError
                    if isinstance(final_error, (OpenAIAPIError, google_api_core_exceptions.GoogleAPICallError)):
                         try:
                             # Attempt to get the message from the error object
                             detail_message = getattr(final_error, 'message', str(final_error))
                             if detail_message and len(detail_message) < 100: # Keep it concise
                                 error_message += f" - {detail_message}"
                         except Exception:
                             pass # Ignore if message extraction fails


                logging.error(f"LLM call failed after trying all available keys for provider {provider}. Last error: {final_error}")
                try:
                    # If some partial message was sent during a failed stream attempt, edit it. Otherwise, reply.
                    if response_msgs:
                        if use_plain_responses:
                             # Append error to the last plain text message
                             last_msg_content = response_msgs[-1].content.replace("...", "") # Remove continuation indicator if present
                             await response_msgs[-1].edit(content=last_msg_content + f"\n\n{error_message}", view=None)
                        else:
                             # Get the last embed, update its description and color
                             last_embed = response_msgs[-1].embeds[0].copy()
                             last_embed.description = (last_embed.description or "").replace(STREAMING_INDICATOR, "") + f"\n\n{error_message}"
                             last_embed.color = discord.Color.red()
                             last_embed.set_footer(text=f"Model: {provider_slash_model}") # Add footer even on error
                             await response_msgs[-1].edit(embed=last_embed, view=None) # Remove view on error
                    else:
                         await new_msg.reply(error_message, mention_author = False)
                except discord.HTTPException as e:
                    logging.error(f"Failed to send final error message to Discord: {e}")

    # --- Finalize and Cache ---
    full_response_text = "".join(response_contents)
    for i, response_msg in enumerate(response_msgs):
        if response_msg.id in msg_nodes:
            node = msg_nodes[response_msg.id]
            # Store the correct text segment in each node
            segment_start = i * max_message_length
            segment_end = (i + 1) * max_message_length
            node.text = full_response_text[segment_start:segment_end]

            # Store grounding metadata only on the node for the *last* message segment
            if response_msg == response_msgs[-1]:
                 node.grounding_metadata = final_grounding_metadata

            # Release lock if held
            if node.lock.locked():
                node.lock.release()
                logging.debug(f"Released lock for MsgNode {response_msg.id}")

    # --- Cache Cleanup ---
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        ids_to_remove = sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]
        for msg_id in ids_to_remove:
            node_to_remove = msg_nodes.pop(msg_id, None)
            # Ensure lock is released if node is removed mid-processing (shouldn't happen often)
            if node_to_remove and node_to_remove.lock.locked():
                node_to_remove.lock.release()
                logging.warning(f"Released lock for MsgNode {msg_id} during cache cleanup.")

    # Remove task from active tasks *after* releasing lock
    active_tasks.pop(new_msg.id, None)


@discord_client.event
async def on_ready():
    logging.info(f"Logged in as {discord_client.user.name} (ID: {discord_client.user.id})")
    logging.info(f"discord.py version: {discord.__version__}")
    logging.info(f"Loaded config: {cfg}")
    # Start the background cleanup task
    await tree.sync() # Sync slash commands
    discord_client.loop.create_task(cleanup_task())
    logging.info("Rate limit cleanup task started.")

async def main():
    global cfg # Ensure main uses the validated config
    # Initial checks before starting
    if not cfg.get("bot_token"):
        logging.error("CRITICAL: bot_token is missing in config.yaml")
        exit(1)
    if not cfg.get("model"):
        logging.warning("LLM model not specified in config.yaml, defaulting might occur or errors.")
    if not cfg.get("providers"):
        logging.error("CRITICAL: 'providers' section is missing in config.yaml")
        exit(1)

    # Log warnings for missing optional keys
    if not cfg.get("youtube_api_key"): logging.warning("youtube_api_key not found. YouTube processing disabled.")
    if not cfg.get("reddit"): logging.warning("Reddit config not found. Reddit processing disabled.")
    if not cfg.get("serpapi_api_keys"): logging.warning("serpapi_api_keys not found. Google Lens disabled.")

    try:
        await discord_client.start(cfg["bot_token"])
    except discord.LoginFailure:
        logging.error("CRITICAL: Failed to log in. Check the bot_token in config.yaml.")
    except Exception as e:
        logging.error(f"CRITICAL: Error starting Discord client: {e}")
    finally:
        # Gracefully close DB connections on exit
        logging.info("Closing rate limit database connections...")
        await asyncio.gather(*(manager.close() for manager in rate_limit_managers.values()))
        logging.info("Database connections closed.")
        if httpx_client:
            await httpx_client.aclose()
        if reddit_client:
             await reddit_client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped manually.")
