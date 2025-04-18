import asyncio
from base64 import b64encode, b64decode
from dataclasses import dataclass, field
from datetime import datetime as dt
import functools
import io
import logging
import re
import os
from typing import Literal, Optional, List, Dict, Any, Tuple, Set
import textwrap
import sqlite3 # Import for specific exceptions if needed later

import discord
from bs4 import BeautifulSoup
from discord import ui, Interaction
from googleapiclient.discovery import build as build_google_api
from googleapiclient.errors import HttpError as GoogleApiHttpError
import httpx
# Import specific OpenAI errors
from openai import (
    AsyncOpenAI, APIError, RateLimitError as OpenAIRateLimitError,
    APIConnectionError, APITimeoutError, BadRequestError as OpenAIBadRequestError,
    AuthenticationError as OpenAIAuthenticationError, PermissionDeniedError as OpenAIPermissionDeniedError
)
import asyncpraw
import asyncprawcore
import yaml
from google import genai
from google.genai import types as genai_types
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# Import errors from the correct module
from google.genai import errors as genai_errors
# Import google-api-core exceptions for specific error handling like rate limits
from google.api_core import exceptions as core_exceptions


# Import custom managers
from api_key_manager import ApiKeyManager
from rate_limit_manager import COOLDOWN_PERIOD_HOURS # Import the constant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("gpt-4", "o3", "o4", "claude-3", "gemini", "gemma", "llama", "pixtral", "mistral-small", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai") # Gemini does not support the 'name' field
BOT_MENTION_KEYWORD = "at ai" # Case-insensitive keyword to trigger the bot

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()
EMBED_COLOR_ERROR = discord.Color.red()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1.3 # Slightly increased delay
MAX_FINAL_EDIT_RETRIES = 3 # Number of retries for the final message edit
FINAL_EDIT_RETRY_DELAY = 1.5 # Seconds between final edit retries

# General URL processing settings
GENERAL_URL_TIMEOUT = 10 # seconds
MAX_REDDIT_COMMENTS = 100 # Keep a limit on fetched comments for performance/API usage

MAX_MESSAGE_NODES = 1000 # Increased cache size
MAX_FULL_RESPONSE_CACHE = 500 # Limit cache size for full responses

# --- Configuration Loading ---
_config_cache = None
_config_lock = asyncio.Lock()

async def get_config(filename="config.yaml"):
    """Loads config with basic caching and async lock."""
    global _config_cache
    async with _config_lock:
        # For simplicity in this example, we'll just reload every time.
        # A more robust solution might check file modification time.
        try:
            with open(filename, "r") as file:
                _config_cache = yaml.safe_load(file)
                # --- API Key List Handling ---
                # Ensure api_key is always a list
                providers = _config_cache.get("providers", {})
                for provider_name, provider_config in providers.items():
                    api_key_value = provider_config.get("api_key")
                    if api_key_value and not isinstance(api_key_value, list):
                        # Convert single key string to list
                        provider_config["api_key"] = [api_key_value]
                    elif not api_key_value:
                         # Ensure it's an empty list if missing or None/empty string
                         provider_config["api_key"] = []
                # --- End API Key List Handling ---
                return _config_cache
        except FileNotFoundError:
            logging.error(f"CRITICAL: {filename} not found. Please create it based on config-example.yaml.")
            exit()
        except yaml.YAMLError as e:
            logging.error(f"CRITICAL: Error parsing {filename}: {e}")
            exit()
        except Exception as e:
            logging.error(f"CRITICAL: Unexpected error loading config {filename}: {e}")
            exit()

# --- Global Variables & Clients ---
cfg = asyncio.run(get_config()) # Initial load

if client_id := cfg.get("client_id"):
    logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/api/oauth2/authorize?client_id={client_id}&permissions=412317273088&scope=bot\n")

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(cfg.get("status_message") or "github.com/jakobdylanc/llmcord")[:128])
discord_client = discord.Client(intents=intents, activity=activity)

HTTPX_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
httpx_client = httpx.AsyncClient(headers=HTTPX_HEADERS, follow_redirects=True, timeout=GENERAL_URL_TIMEOUT)

msg_nodes: Dict[int, 'MsgNode'] = {}
last_task_time = 0
gemini_grounding_metadata: Dict[int, Any] = {}
full_response_texts: Dict[int, str] = {}
reddit_client: Optional[asyncpraw.Reddit] = None
api_key_managers: Dict[str, ApiKeyManager] = {} # Cache for ApiKeyManager instances
api_key_managers_lock = asyncio.Lock() # Lock for accessing/creating managers

# --- YouTube Data Fetching ---
@functools.lru_cache(maxsize=2)
def get_youtube_service(api_key):
    """Builds and returns a YouTube Data API service object, caching the result."""
    if not api_key:
        logging.warning("YouTube Data API key is missing. Metadata/comments won't be fetched.")
        return None
    try:
        # Disable cache discovery file for potentially faster startup in some environments
        return build_google_api('youtube', 'v3', developerKey=api_key, cache_discovery=False)
    except Exception as e:
        logging.error(f"Failed to build YouTube API client: {e}")
        return None

def fetch_youtube_transcript_sync(video_id):
    """Synchronously fetches and formats the transcript for a given video ID."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Prioritize manual, then generated, in English first, then any language
        transcript = None
        try: transcript = transcript_list.find_manually_created_transcript(['en'])
        except NoTranscriptFound:
            try: transcript = transcript_list.find_generated_transcript(['en'])
            except NoTranscriptFound:
                 available_langs = [t.language_code for t in transcript_list]
                 if available_langs:
                     try: transcript = transcript_list.find_manually_created_transcript(available_langs)
                     except NoTranscriptFound:
                         try: transcript = transcript_list.find_generated_transcript(available_langs)
                         except NoTranscriptFound: pass # No transcript found at all

        if transcript:
            fetched_transcript = transcript.fetch()
            # Ensure snippet is a dict-like object before accessing 'text'
            return "\n".join([entry['text'] for entry in fetched_transcript if isinstance(entry, dict) and 'text' in entry])
        else:
            logging.info(f"No suitable transcript found for video ID: {video_id}")
            return "[Transcript not found or unavailable]"
    except TranscriptsDisabled:
        logging.info(f"Transcripts disabled for video ID: {video_id}")
        return "[Transcripts disabled]"
    except Exception as e:
        # Log the actual exception type and message
        logging.error(f"Error fetching transcript for {video_id}: {type(e).__name__} - {e}")
        return "[Error fetching transcript]"

def fetch_youtube_metadata_and_comments_sync(video_id, api_key):
    """Synchronously fetches metadata and comments using YouTube Data API."""
    youtube = get_youtube_service(api_key)
    if not youtube: return None

    metadata = {'title': None, 'description': None, 'channel_name': None, 'comments_text': None, 'transcript_text': None} # Add transcript_text here
    try:
        # Get video details (snippet)
        video_response = youtube.videos().list(part="snippet", id=video_id).execute()
        if not video_response.get("items"):
            logging.warning(f"Video not found or unavailable via YouTube Data API: {video_id}")
            return None
        video_snippet = video_response["items"][0]["snippet"]
        metadata['title'] = video_snippet.get("title")
        metadata['description'] = video_snippet.get("description")
        channel_id = video_snippet.get("channelId")

        # Get channel details (snippet)
        if channel_id:
            try:
                channel_response = youtube.channels().list(part="snippet", id=channel_id).execute()
                if channel_response.get("items"):
                    metadata['channel_name'] = channel_response["items"][0]["snippet"].get("title")
            except GoogleApiHttpError as e:
                 logging.error(f"YouTube Data API error fetching channel {channel_id} for video {video_id}: {e}")
            except Exception as e:
                 logging.error(f"Unexpected error fetching channel {channel_id} for video {video_id}: {e}")

        # Get comments (snippet)
        try:
            comment_response = youtube.commentThreads().list(
                part="snippet", videoId=video_id, order="relevance", maxResults=50, textFormat="plainText"
            ).execute()
            comments = []
            for item in comment_response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]
                author = comment.get("authorDisplayName", "Unknown Author")
                text = comment.get("textDisplay", "")
                comments.append(f"{author}: {text}")
            metadata['comments_text'] = "\n".join(comments) if comments else "[No comments found or retrieved]"
        except GoogleApiHttpError as e:
            # Check if comments are disabled (specific error reason)
            if hasattr(e, 'resp') and e.resp.status == 403: # Comments likely disabled or region restricted
                 logging.info(f"Comments disabled or forbidden for video ID: {video_id}")
                 metadata['comments_text'] = "[Comments are disabled or unavailable]"
            else:
                 logging.error(f"YouTube Data API error fetching comments for {video_id}: {e}")
                 metadata['comments_text'] = "[Error fetching comments]"
        except Exception as e:
             logging.error(f"Unexpected error fetching comments for {video_id}: {e}")
             metadata['comments_text'] = "[Error fetching comments]"

    except GoogleApiHttpError as e:
        logging.error(f"YouTube Data API error for video {video_id}: {e}")
        return None # Indicate API error
    except Exception as e:
        logging.error(f"Unexpected error fetching metadata for {video_id}: {e}")
        return None

    return metadata

async def process_youtube_url(video_id, youtube_api_key, original_url):
    """Asynchronously fetches and formats transcript, metadata, and comments for a YouTube video."""
    # Fetch metadata and comments first (if API key exists)
    metadata = None
    if youtube_api_key:
        metadata_task = asyncio.to_thread(fetch_youtube_metadata_and_comments_sync, video_id, youtube_api_key)
        try:
            metadata = await metadata_task
        except Exception as e:
            logging.error(f"Exception in metadata fetch thread for {video_id}: {e}")
            metadata = None # Treat as if metadata fetch failed

    # Fetch transcript regardless of API key
    transcript_task = asyncio.to_thread(fetch_youtube_transcript_sync, video_id)
    try:
        transcript_text = await transcript_task
    except Exception as e:
        logging.error(f"Exception in transcript fetch thread for {video_id}: {e}")
        transcript_text = "[Error fetching transcript]" # Assign error string

    # Combine results
    if metadata:
        metadata['transcript_text'] = transcript_text # Add transcript to metadata dict
        return metadata
    elif transcript_text not in ("[Error fetching transcript]", "[Transcript not found or unavailable]", "[Transcripts disabled]"):
        # Return only transcript if metadata failed but transcript succeeded
        return {'transcript_text': transcript_text, 'title': None, 'description': None, 'channel_name': None, 'comments_text': None}
    else:
        # If both failed significantly or only transcript failed significantly
        logging.warning(f"Could not retrieve significant data for YouTube URL: {original_url}")
        return None # Nothing useful to append

# --- Reddit Data Fetching ---
async def get_reddit_client():
    """Initializes and returns the asyncpraw Reddit client instance."""
    global reddit_client
    if reddit_client:
        return reddit_client

    cfg = await get_config() # Reload config in case it changed
    client_id = cfg.get("reddit_client_id")
    client_secret = cfg.get("reddit_client_secret")
    user_agent = cfg.get("reddit_user_agent")

    if not all([client_id, client_secret, user_agent]):
        logging.warning("Reddit credentials (client_id, client_secret, user_agent) not found in config. Skipping Reddit client initialization.")
        return None

    try:
        reddit_client = asyncpraw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            # No username/password needed for read-only
        )
        # Quick check (optional but good) - fetch a known public subreddit
        # This check seems to cause issues sometimes (like the 404), let's make it less critical
        try:
            # Use fetch=True to ensure data is loaded, helps catch auth issues early
            await reddit_client.subreddit("all", fetch=True)
            logging.info("Async PRAW Reddit client initialized and validated successfully.")
        except asyncprawcore.exceptions.NotFound:
             logging.warning("Could not validate Reddit client by fetching /r/all (404 Not Found). Client might still work for specific subreddits.")
        # Catch other potential validation errors but don't stop initialization
        except asyncprawcore.exceptions.RequestException as validate_err:
             logging.warning(f"Validation check for Reddit client failed: {validate_err}. Client might still work.")

        return reddit_client
    except asyncprawcore.exceptions.ResponseException as e:
         status_code = getattr(e.response, 'status', 'Unknown') # Use getattr for safety
         response_text = "[No response text available]"
         if e.response:
             try:
                 # Need to await text() for aiohttp responses used by asyncprawcore
                 response_text = await e.response.text()
             except Exception as text_err:
                 logging.warning(f"Could not read response text during Reddit client init error: {text_err}")
         logging.error(f"Failed to initialize asyncpraw Reddit client (Response Error): {status_code} - {response_text}", exc_info=False) # Don't log full traceback for auth errors
         reddit_client = None
         return None
    except Exception as e:
        logging.error(f"Failed to initialize asyncpraw Reddit client: {e}", exc_info=True)
        reddit_client = None # Ensure client is None if init fails
        return None

async def fetch_reddit_content(reddit, submission_id):
    """Asynchronously fetches and formats content from a Reddit submission."""
    if not reddit: return "[Reddit client not available]"
    try:
        submission = await reddit.submission(id=submission_id)
        await submission.load() # Ensure submission data is loaded

        content_parts = []
        content_parts.append(f"Title: {submission.title}")
        author_name = getattr(submission.author, 'name', '[deleted]')
        content_parts.append(f"Author: u/{author_name}")
        content_parts.append(f"Score: {submission.score}")
        if submission.selftext:
            content_parts.append(f"Body:\n{submission.selftext}") # No truncation
        elif submission.url and not submission.is_self: # Only show URL for link posts
            content_parts.append(f"URL: {submission.url}")

        # Load top-level comments only, replace_more ensures we get Comment objects
        await submission.comments.replace_more(limit=0)
        comments_text = []
        comment_list = await submission.comments() # Get the CommentForest

        # Iterate through the top-level items in the CommentForest
        for i, top_level_comment in enumerate(comment_list):
             if i >= MAX_REDDIT_COMMENTS: break
             # Skip MoreComments objects if any remain (shouldn't with limit=0 but safety first)
             if isinstance(top_level_comment, asyncpraw.models.MoreComments): continue

             # Access attributes safely
             comment_author = getattr(top_level_comment.author, 'name', '[deleted]')
             comment_body = getattr(top_level_comment, 'body', '') # No truncation
             comments_text.append(f"u/{comment_author}: {comment_body}")


        if comments_text:
            content_parts.append(f"\nTop {len(comments_text)} Comments:\n" + "\n".join(comments_text))
        else:
            content_parts.append("\n[No comments found or retrieved or comments disabled]") # Clarify reason

        return "\n".join(content_parts)

    except asyncprawcore.exceptions.NotFound:
        logging.warning(f"Reddit submission not found: {submission_id}")
        return "[Reddit submission not found]"
    except asyncprawcore.exceptions.Forbidden:
         logging.warning(f"Access forbidden for Reddit submission: {submission_id} (possibly private or quarantined)")
         return "[Reddit submission access forbidden]"
    except Exception as e:
        logging.error(f"Error fetching Reddit content for submission {submission_id}: {type(e).__name__} - {e}", exc_info=True)
        return f"[Error fetching Reddit content: {type(e).__name__}]"

# --- General URL Content Extraction ---
async def fetch_and_extract_content(url):
    """Asynchronously fetches URL content and extracts text using Beautiful Soup."""
    try:
        # Use a context manager for the client to ensure proper cleanup
        async with httpx.AsyncClient(headers=HTTPX_HEADERS, follow_redirects=True, timeout=GENERAL_URL_TIMEOUT) as client:
            response = await client.get(url)
            response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)

            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                logging.info(f"Skipping non-HTML content type '{content_type}' for URL: {url}")
                return "[Unsupported content type]"

            # Use await response.aread() for async reading, then decode
            html_content = await response.aread()
            # Attempt to decode using detected encoding or fallback to utf-8
            decoded_content = None
            try:
                # Use response.encoding which httpx tries to determine
                if response.encoding:
                    decoded_content = html_content.decode(response.encoding, errors='replace')
                else:
                    # Fallback if encoding detection failed
                    decoded_content = html_content.decode('utf-8', errors='replace')
            except (LookupError, TypeError, UnicodeDecodeError) as decode_error:
                 logging.warning(f"Decoding error for {url} (encoding: {response.encoding}): {decode_error}. Falling back to utf-8.")
                 # Force decode with utf-8 replacement on error
                 decoded_content = html_content.decode('utf-8', errors='replace')

            if decoded_content is None: # Should not happen, but safety check
                logging.error(f"Failed to decode content for URL {url}")
                return "[Error decoding content]"

            soup = BeautifulSoup(decoded_content, 'lxml') # Use lxml parser

            # Remove script and style elements
            for script_or_style in soup(["script", "style", "nav", "footer", "aside"]): # Remove common non-content tags
                script_or_style.decompose()

            # Attempt to find main content areas, otherwise use body
            main_content = soup.find('main') or soup.find('article') or soup.body
            text = main_content.get_text(separator='\n', strip=True) if main_content else soup.get_text(separator='\n', strip=True)

            # Clean up excessive whitespace more aggressively
            text = re.sub(r'\s*\n\s*', '\n', text).strip() # Replace multiple newlines/spaces around newline with single newline
            text = re.sub(r'[ \t]{2,}', ' ', text) # Replace multiple spaces/tabs with single space

            if not text:
                logging.info(f"Extracted empty text content from URL: {url}")
                return "[No text content found]"

            return text # Return full text

    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        logging.warning(f"HTTP error fetching URL {url}: {type(e).__name__} - {e}")
        return f"[Error fetching URL: {type(e).__name__}]"
    except Exception as e:
        logging.error(f"Error extracting content from URL {url}: {type(e).__name__} - {e}", exc_info=True)
        return "[Error extracting content]"

# --- Data Classes and Views ---
@dataclass
class MsgNode:
    """Represents a message node in the conversation history cache."""
    text: Optional[str] = None
    images: list = field(default_factory=list) # Stores image data URLs or references
    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None # Discord User ID for 'user' role
    has_bad_attachments: bool = False # Flag if unsupported attachments were present
    fetch_parent_failed: bool = False # Flag if fetching the parent message failed
    parent_msg: Optional[discord.Message] = None # Reference to the parent discord.Message object
    lock: asyncio.Lock = field(default_factory=asyncio.Lock) # Lock for async access to node data

class ResponseActionsView(ui.View):
    """A persistent view containing buttons for response actions like showing sources or getting the text file."""
    def __init__(self, *, show_sources_button: bool):
        super().__init__(timeout=None) # Persistent view

        # Conditionally add the "Show Sources" button
        if show_sources_button:
            sources_button = ui.Button(label="Show Sources", style=discord.ButtonStyle.secondary, custom_id="show_sources_button")
            sources_button.callback = self.show_sources_callback # Assign callback dynamically
            self.add_item(sources_button)

        # Always add the "Get Response File" button
        file_button = ui.Button(label="Get Response File", style=discord.ButtonStyle.secondary, custom_id="get_response_as_file_button") # Renamed label
        file_button.callback = self.get_response_file_callback
        self.add_item(file_button)

    async def show_sources_callback(self, interaction: Interaction):
        """Callback for the 'Show Sources' button."""
        # Ensure interaction message exists
        if not interaction.message:
            await interaction.response.send_message("Error: Could not find the original message.")
            return

        metadata = gemini_grounding_metadata.get(interaction.message.id)

        if not metadata:
            await interaction.response.send_message("Sorry, I couldn't find the sources for this message.")
            # Optionally disable the button if metadata is missing
            # button = discord.utils.get(self.children, custom_id="show_sources_button")
            # if button:
            #     button.disabled = True
            #     await interaction.message.edit(view=self)
            return

        # --- Build the response string/parts ---
        response_parts = []
        # Safely access attributes using getattr
        if queries := getattr(metadata, 'web_search_queries', None):
            response_parts.append("**Search Queries:**")
            response_parts.extend([f"- `{q}`" for q in queries])
            response_parts.append("") # Add spacing

        source_lines = []
        if chunks := getattr(metadata, 'grounding_chunks', None):
            for chunk in chunks:
                # Check if chunk has 'web' attribute and it's not None
                if web_info := getattr(chunk, 'web', None):
                    title = getattr(web_info, 'title', 'Unknown Title')
                    uri = getattr(web_info, 'uri', None)
                    # Ensure URI is valid before creating markdown link
                    if uri and isinstance(uri, str) and uri.startswith('http'):
                        # Escape any markdown characters in the title
                        escaped_title = discord.utils.escape_markdown(title)
                        source_lines.append(f"- [{escaped_title}](<{uri}>)")
                    else:
                        source_lines.append(f"- {discord.utils.escape_markdown(title)}")

        if source_lines:
            response_parts.append("**Sources Used:**")
            response_parts.extend(source_lines)

        full_response_text = "\n".join(response_parts) or "No grounding information found."

        # --- Send potentially split response ---
        max_len = 2000 # Discord message limit
        messages_to_send = []
        current_message = ""

        for line in full_response_text.splitlines(keepends=True):
            if len(current_message) + len(line) <= max_len:
                current_message += line
            else:
                # Send the current message if it's not empty
                if current_message.strip():
                    messages_to_send.append(current_message)
                # Start a new message, handle case where single line exceeds limit
                if len(line) > max_len:
                     # Split the long line itself
                     for i in range(0, len(line), max_len):
                         messages_to_send.append(line[i:i+max_len])
                     current_message = "" # Reset after handling long line
                else:
                    current_message = line

        # Add the last message if it has content
        if current_message.strip():
            messages_to_send.append(current_message)

        # Send the messages
        if not messages_to_send: # Should not happen if full_response_text wasn't empty
             await interaction.response.send_message("No grounding information found.", suppress_embeds=True)
             return

        # Send the first message using the initial interaction response
        await interaction.response.send_message(messages_to_send[0], suppress_embeds=True)

        # Send subsequent messages using followup
        for msg_content in messages_to_send[1:]:
            await interaction.followup.send(msg_content, suppress_embeds=True)

    async def get_response_file_callback(self, interaction: Interaction):
        """Callback for the 'get the response as text file' button."""
        # Ensure interaction message exists
        if not interaction.message:
            await interaction.response.send_message("Error: Could not find the original message.")
            return

        full_text = full_response_texts.get(interaction.message.id)

        if not full_text:
            await interaction.response.send_message(
                "Sorry, the full response text for this message is no longer available (it might be too old or the bot restarted)."
            )
            # Optionally disable the button
            # button = discord.utils.get(self.children, custom_id="get_response_as_file_button")
            # if button:
            #     button.disabled = True
            #     await interaction.message.edit(view=self)
            return

        try:
            # Create a file-like object from the text, ensuring UTF-8 encoding
            file_content = io.BytesIO(full_text.encode('utf-8'))
            # Create a discord File object
            discord_file = discord.File(fp=file_content, filename="response.txt")
            await interaction.response.send_message(file=discord_file)
        except Exception as e:
            logging.error(f"Error creating or sending response file for message {interaction.message.id}: {e}", exc_info=True)
            await interaction.response.send_message("Sorry, an error occurred while creating the response file.")


# --- Regex Patterns ---
YOUTUBE_URL_PATTERN = re.compile(
    # Protocol optional, www optional, domains, path, query parameter v=, video ID, optional extra params
    r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/|youtube-nocookie\.com/embed/)([\w-]{11})(?:\S+)?'
)
REDDIT_URL_PATTERN = re.compile(
    # Protocol optional, www optional, domain, /r/, subreddit, /comments/, submission_id, optional slug/?, optional query/fragment
    r'(?:https?://)?(?:www\.)?reddit\.com/r/(\w+)/comments/([\w]+)(?:/[\w-]+/?)*(?:\S*)?'
)
# Improved regex to avoid matching markdown links like [text](url) or <url>
# It looks for URLs not immediately preceded by '](', '<', or followed by ')' (if preceded by '(')
# It also handles various TLDs better and requires http(s)://
GENERAL_URL_PATTERN = re.compile(
    r'(?<![<(])' # Negative lookbehind for '<' or '('
    r'(https?://' # Require http:// or https://
    r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}' # Domain name
    r'(?::\d+)?' # Optional port
    r'(?:/[^\s<>()"]*)?' # Optional path, avoid spaces and brackets/quotes
    r')'
    r'(?![^<>()"]*\))' # Negative lookahead to avoid matching URLs inside parentheses like (http://...)
)

def extract_youtube_video_ids_with_urls(text):
    """Extracts unique YouTube video IDs and their original URLs from text."""
    matches = YOUTUBE_URL_PATTERN.finditer(text)
    # Use a dict to store unique IDs and the first URL found for each
    unique_videos = {}
    for match in matches:
        video_id = match.group(1)
        full_url = match.group(0)
        if video_id not in unique_videos:
            # Ensure URL starts with http(s):// for clarity
            if not full_url.startswith(('http://', 'https://')):
                full_url = 'https://' + full_url
            unique_videos[video_id] = full_url
    # Return list of tuples (video_id, url)
    return list(unique_videos.items())

def extract_reddit_submission_ids_with_urls(text):
    """Extracts unique Reddit submission IDs and their original URLs from text."""
    unique_submissions = {}
    for match in REDDIT_URL_PATTERN.finditer(text):
        submission_id = match.group(2)
        full_url = match.group(0)
        if submission_id not in unique_submissions:
             # Ensure URL starts with http(s)://
            if not full_url.startswith(('http://', 'https://')):
                full_url = 'https://' + full_url
            unique_submissions[submission_id] = full_url
    return [(url, sub_id) for sub_id, url in unique_submissions.items()] # Return list of (url, id)

def extract_general_urls(text):
    """Extracts all HTTP/HTTPS URLs from text, avoiding markdown links."""
    return GENERAL_URL_PATTERN.findall(text)

# --- Helper Function for Gemini Image Conversion ---
def get_gemini_image_part(image_dict):
    """Converts an OpenAI-style image dict to a Gemini Part, handling potential errors."""
    data_url = image_dict.get("image_url", {}).get("url", "")
    match = re.match(r"data:(image/\w+);base64,(.*)", data_url)
    if match:
        mime_type, base64_data = match.groups()
        try:
            image_bytes = b64decode(base64_data)
            return genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        except Exception as e: # Catch specific decoding exception
            log_snippet = data_url[:50] + "..." + data_url[-20:] if len(data_url) > 70 else data_url
            logging.warning(f"Failed to decode base64 image data. Error: {type(e).__name__}: {e}. Data URL snippet: {log_snippet}")
    else:
         if data_url: # Only log if data_url wasn't empty
             logging.warning(f"Regex failed to match expected data URL format: {data_url[:100]}...")
    return None

# --- API Key Manager Initialization ---
async def get_api_key_manager(provider_name: str, config: Dict) -> Optional[ApiKeyManager]:
    """Gets or creates an ApiKeyManager instance for a provider."""
    global api_key_managers
    async with api_key_managers_lock:
        if provider_name in api_key_managers:
            return api_key_managers[provider_name]

        provider_config = config.get("providers", {}).get(provider_name)
        if not provider_config:
            logging.error(f"Provider '{provider_name}' not found in config.")
            return None

        api_keys = provider_config.get("api_key", []) # Expect list now
        # No need to check for empty list here, ApiKeyManager handles it

        try:
            manager = ApiKeyManager(provider_name, api_keys)
            api_key_managers[provider_name] = manager
            return manager
        except ValueError as e:
            logging.error(f"Failed to initialize ApiKeyManager for {provider_name}: {e}")
            return None
        except Exception as e:
             logging.error(f"Unexpected error initializing ApiKeyManager for {provider_name}: {e}", exc_info=True)
             return None

# --- Main Event Handler ---
@discord_client.event
async def on_message(new_msg):
    global msg_nodes, last_task_time, gemini_grounding_metadata, full_response_texts

    # --- Initial Checks and Permissions ---
    if new_msg.author.bot: return
    is_dm = isinstance(new_msg.channel, discord.DMChannel)
    # --- Refined Trigger Check ---
    # Use regex to check if message starts with keyword followed by whitespace or end of string
    trigger_pattern = re.compile(rf'^{re.escape(BOT_MENTION_KEYWORD)}(\s+.*|$)', re.IGNORECASE)
    triggered_by_keyword = bool(trigger_pattern.match(new_msg.content))

    if not is_dm and discord_client.user not in new_msg.mentions: # Check mentions first
        is_reply_to_bot = False
        if new_msg.reference and new_msg.reference.message_id:
             ref_node = msg_nodes.get(new_msg.reference.message_id)
             if ref_node and ref_node.role == "assistant": is_reply_to_bot = True
             else:
                 try:
                     # Fetch if not in cache or role unknown
                     ref_msg = new_msg.reference.cached_message or await new_msg.channel.fetch_message(new_msg.reference.message_id)
                     if ref_msg.author == discord_client.user:
                         is_reply_to_bot = True
                 except (discord.NotFound, discord.HTTPException):
                     pass # Ignore if reference message not found
        # Add keyword check here
        if not is_reply_to_bot and not triggered_by_keyword: # Check keyword trigger if not mention/reply
             return # Exit if not mentioned and not a direct reply to the bot

    cfg = await get_config() # Use async getter

    # Permission checks (simplified for brevity, logic remains the same)
    role_ids = set(role.id for role in getattr(new_msg.author, "roles", []))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))
    allow_dms = cfg.get("allow_dms", True)
    permissions = cfg.get("permissions", {})
    user_perms = permissions.get("users", {"allowed_ids": [], "blocked_ids": []})
    role_perms = permissions.get("roles", {"allowed_ids": [], "blocked_ids": []})
    channel_perms = permissions.get("channels", {"allowed_ids": [], "blocked_ids": []})
    (allowed_user_ids, blocked_user_ids) = (user_perms.get("allowed_ids", []), user_perms.get("blocked_ids", []))
    (allowed_role_ids, blocked_role_ids) = (role_perms.get("allowed_ids", []), role_perms.get("blocked_ids", []))
    (allowed_channel_ids, blocked_channel_ids) = (channel_perms.get("allowed_ids", []), channel_perms.get("blocked_ids", []))
    user_blocked = new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)
    user_allowed = not allowed_user_ids and not allowed_role_ids or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    if user_blocked or not user_allowed: logging.warning(f"User {new_msg.author.id} denied (Blocked: {user_blocked}, Allowed: {user_allowed})."); return
    if is_dm and not allow_dms: logging.warning(f"User {new_msg.author.id} denied DM (allow_dms: False)."); return
    elif not is_dm:
        channel_blocked = any(id in blocked_channel_ids for id in channel_ids)
        channel_allowed = not allowed_channel_ids or any(id in allowed_channel_ids for id in channel_ids)
        if channel_blocked or not channel_allowed: logging.warning(f"User {new_msg.author.id} denied in channel {new_msg.channel.id} (Blocked: {channel_blocked}, Allowed: {channel_allowed})."); return

    # --- Configuration and Client Initialization ---
    provider_slash_model = cfg.get("model", "google/gemini-2.0-flash")
    try:
        provider, model = provider_slash_model.split("/", 1)
        if provider not in cfg.get("providers", {}): raise ValueError(f"Provider '{provider}' not found.")
    except (ValueError, AttributeError) as e:
        logging.error(f"Invalid model format '{provider_slash_model}' or provider config missing: {e}")
        await new_msg.reply("Config error: Invalid model/provider.", mention_author = False); return

    key_manager = await get_api_key_manager(provider, cfg)
    if not key_manager:
        await new_msg.reply(f"Config error: Could not initialize API keys for provider '{provider}'. Check logs.", mention_author = False)
        return

    # --- Model Capabilities and Limits ---
    accept_images = any(tag in model.lower() for tag in VISION_MODEL_TAGS)
    accept_usernames = provider in PROVIDERS_SUPPORTING_USERNAMES
    max_text = cfg.get("max_text", 100000)
    max_images = cfg.get("max_images", 5) if accept_images else 0
    max_messages = cfg.get("max_messages", 25)
    use_plain_responses = cfg.get("use_plain_responses", False)
    max_embed_desc_length = 4090
    max_plain_msg_length = 2000
    max_message_length = max_plain_msg_length if use_plain_responses else max_embed_desc_length

    # --- Initial Message Content Processing ---
    original_content_stripped = new_msg.content.strip() # Store original stripped content
    is_reply = new_msg.reference is not None

    # Check if the original content was just a mention or keyword
    is_only_mention = original_content_stripped == f"<@!{discord_client.user.id}>" or original_content_stripped == f"<@{discord_client.user.id}>" # Check against stripped content
    # Check if it starts with keyword and has nothing significant after it
    is_only_keyword = original_content_stripped.lower() == BOT_MENTION_KEYWORD.lower() # Check against stripped content
    is_only_trigger = is_only_mention or is_only_keyword

    # Clean prefix/mention only for the initial message
    cleaned_content = new_msg.content # Start with original content
    if triggered_by_keyword:
        # More robust keyword removal at the start
        cleaned_content = re.sub(rf'^{re.escape(BOT_MENTION_KEYWORD)}\s*', '', cleaned_content, flags=re.IGNORECASE).strip()
    # Remove standard bot mention (<@...> or <@!...) anywhere in the string
    cleaned_content = re.sub(rf'<@!?{discord_client.user.id}>', '', cleaned_content).strip()


    # Process attachments for the initial message
    good_attachments = [att for att in new_msg.attachments if att.content_type and (att.content_type.startswith("text") or (accept_images and att.content_type.startswith("image")))]
    bad_attachments_initial = len(new_msg.attachments) > len(good_attachments)
    attachment_texts_initial = []
    attachment_images_initial = []

    if good_attachments:
        attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments], return_exceptions=True)
        for att, resp in zip(good_attachments, attachment_responses):
            if isinstance(resp, httpx.Response) and resp.status_code == 200:
                if att.content_type.startswith("text"):
                    try:
                        text_content = resp.content.decode(resp.encoding or 'utf-8', errors='replace')
                        attachment_texts_initial.append(text_content[:max_text])
                        if len(text_content) > max_text: bad_attachments_initial = True # Consider it bad if truncated
                    except Exception as e:
                        logging.warning(f"Failed to decode text attachment {att.filename}: {e}")
                        bad_attachments_initial = True
                elif att.content_type.startswith("image"):
                    try:
                        img_data = f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"
                        attachment_images_initial.append(dict(type="image_url", image_url=dict(url=img_data)))
                    except Exception as e:
                         logging.warning(f"Failed to encode image attachment {att.filename}: {e}")
                         bad_attachments_initial = True
            else:
                logging.warning(f"Failed to fetch attachment {att.url}: {resp}")
                bad_attachments_initial = True

    # --- Empty Query Check (Modified) ---
    is_content_effectively_empty = not cleaned_content.strip() and not attachment_texts_initial and not attachment_images_initial
    # Treat as empty ONLY if the content is empty AND it wasn't a reply consisting solely of a trigger mention/keyword
    should_treat_as_empty = is_content_effectively_empty and not (is_reply and is_only_trigger)

    if should_treat_as_empty:
        await new_msg.reply("Your query is empty. Please reply to a message to reference it or don't send an empty query.", mention_author=False)
        return

    # --- URL Extraction (from initial message only) ---
    text_content_for_extraction = new_msg.content # Use original content for URL extraction
    video_ids_with_urls = extract_youtube_video_ids_with_urls(text_content_for_extraction)
    reddit_urls_with_ids = extract_reddit_submission_ids_with_urls(text_content_for_extraction)
    general_urls = extract_general_urls(text_content_for_extraction)
    youtube_urls_set = {url for _, url in video_ids_with_urls}
    reddit_urls_set = {url for url, _ in reddit_urls_with_ids}
    filtered_general_urls = [url for url in general_urls if url not in youtube_urls_set and url not in reddit_urls_set]
    youtube_content_to_append = ""
    reddit_content_to_append = ""
    general_url_content_to_append = ""

    # --- Build Message Chain (Following Replies Only) ---
    messages = []
    user_warnings = set()
    curr_msg = new_msg
    processed_ids = set() # Prevent infinite loops

    while curr_msg is not None and len(messages) < max_messages and curr_msg.id not in processed_ids:
        processed_ids.add(curr_msg.id)
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            # Populate node if not already done
            if curr_node.text is None:
                # Use pre-processed content/attachments for the initial message
                if curr_msg.id == new_msg.id:
                    node_text_content = cleaned_content # Use already cleaned content
                    node_attachment_texts = attachment_texts_initial
                    node_attachment_images = attachment_images_initial
                    node_bad_attachments = bad_attachments_initial
                else:
                    # Process content and attachments for parent messages
                    node_text_content = curr_msg.content
                    # Remove bot mention from parent messages as well
                    node_text_content = re.sub(rf'<@!?{discord_client.user.id}>', '', node_text_content).strip()

                    node_good_attachments = [att for att in curr_msg.attachments if att.content_type and (att.content_type.startswith("text") or (accept_images and att.content_type.startswith("image")))]
                    node_bad_attachments = len(curr_msg.attachments) > len(node_good_attachments)
                    node_attachment_texts = []
                    node_attachment_images = []

                    if node_good_attachments:
                        node_attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in node_good_attachments], return_exceptions=True)
                        for att, resp in zip(node_good_attachments, node_attachment_responses):
                            if isinstance(resp, httpx.Response) and resp.status_code == 200:
                                if att.content_type.startswith("text"):
                                    try:
                                        text_content = resp.content.decode(resp.encoding or 'utf-8', errors='replace')
                                        node_attachment_texts.append(text_content[:max_text])
                                        if len(text_content) > max_text: node_bad_attachments = True
                                    except Exception as e:
                                        logging.warning(f"Failed to decode text attachment {att.filename} in parent: {e}")
                                        node_bad_attachments = True
                                elif att.content_type.startswith("image"):
                                    try:
                                        img_data = f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"
                                        node_attachment_images.append(dict(type="image_url", image_url=dict(url=img_data)))
                                    except Exception as e:
                                         logging.warning(f"Failed to encode image attachment {att.filename} in parent: {e}")
                                         node_bad_attachments = True
                            else:
                                logging.warning(f"Failed to fetch attachment {att.url} in parent: {resp}")
                                node_bad_attachments = True

                embed_texts = [
                    "\n".join(filter(None, (embed.title, embed.description, getattr(embed.footer, 'text', None))))
                    for embed in curr_msg.embeds if embed.type == 'rich' # Process only rich embeds
                ]

                curr_node.text = "\n".join(filter(None, [node_text_content] + embed_texts + node_attachment_texts))
                curr_node.images = node_attachment_images
                curr_node.role = "assistant" if curr_msg.author == discord_client.user else "user"
                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None
                curr_node.has_bad_attachments = node_bad_attachments

                # Determine parent message (Simplified logic - ONLY follow explicit replies or thread starters)
                parent_msg_obj = None
                try:
                    # 1. Direct Reply
                    if curr_msg.reference and curr_msg.reference.message_id:
                        # Avoid fetching self-reference in replies (shouldn't happen often but safety)
                        if curr_msg.reference.message_id != curr_msg.id:
                            parent_msg_obj = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(curr_msg.reference.message_id)
                        else:
                             logging.warning(f"Message {curr_msg.id} references itself, stopping chain.")

                    # 2. Thread Start (Only if it's the *first* user message in the thread)
                    elif isinstance(curr_msg.channel, discord.Thread) and curr_msg.id == new_msg.id: # Check if it's the trigger message
                         is_first_message = not [msg async for msg in curr_msg.channel.history(limit=1, before=curr_msg)]
                         if is_first_message:
                             # Try fetching the starter message directly
                             try:
                                 parent_msg_obj = curr_msg.channel.starter_message or await curr_msg.channel.fetch_starter_message()
                             except (discord.NotFound, discord.HTTPException, AttributeError):
                                 logging.warning(f"Could not fetch starter message for thread {curr_msg.channel.id}")
                                 # Fallback: Try fetching the message that created the thread (if applicable)
                                 try:
                                     parent_channel = curr_msg.channel.parent
                                     if parent_channel and curr_msg.channel.id != curr_msg.id: # Ensure it's not the thread message itself
                                         parent_msg_obj = await parent_channel.fetch_message(curr_msg.channel.id)
                                 except (discord.NotFound, discord.HTTPException, AttributeError):
                                     logging.warning(f"Could not fetch thread creation message for thread {curr_msg.channel.id}")
                                     pass # No parent found

                    # --- REMOVED Automatic Chaining Logic ---
                    # The logic that checked channel history for same author messages is removed.

                    curr_node.parent_msg = parent_msg_obj
                except (discord.NotFound, discord.HTTPException) as e: curr_node.fetch_parent_failed = True; logging.warning(f"Error fetching parent for {curr_msg.id}: {e}")
                except Exception as e: curr_node.fetch_parent_failed = True; logging.error(f"Unexpected error determining parent for {curr_msg.id}: {e}", exc_info=True)


            # --- Prepare message content for LLM ---
            content_for_llm = None
            images_for_llm = curr_node.images[:max_images] # Apply limit
            text_for_llm = curr_node.text[:max_text] if curr_node.text else ""
            if curr_node.text and len(curr_node.text) > max_text: user_warnings.add(f"⚠️ Max {max_text:,} chars/message")

            if provider == "google":
                parts = []
                if text_for_llm: parts.append(genai_types.Part.from_text(text=text_for_llm))
                if accept_images:
                    for img_dict in images_for_llm:
                        if gemini_part := get_gemini_image_part(img_dict): parts.append(gemini_part)
                        else: curr_node.has_bad_attachments = True # Mark if conversion failed
                content_for_llm = parts # List of Parts
            else: # OpenAI format
                if accept_images and images_for_llm:
                    content_for_llm = []
                    if text_for_llm: content_for_llm.append(dict(type="text", text=text_for_llm))
                    content_for_llm.extend(images_for_llm)
                elif text_for_llm: content_for_llm = text_for_llm # String only if no images
                # If neither text nor images, content_for_llm remains None

            # Add message to list if it has content OR if it's the initial message and was just a trigger
            is_initial_trigger_reply = curr_msg.id == new_msg.id and is_reply and is_only_trigger and not content_for_llm
            if content_for_llm or (curr_node.role == "assistant" and not content_for_llm) or is_initial_trigger_reply:
                message = dict(role=curr_node.role)
                if accept_usernames and curr_node.user_id is not None: message["name"] = str(curr_node.user_id)
                if provider == "google": message["parts"] = content_for_llm if content_for_llm else [] # Ensure parts is list
                else: message["content"] = content_for_llm if content_for_llm else "" # Ensure content is string or list
                messages.append(message)
            elif curr_node.role == "user": logging.warning(f"User message {curr_msg.id} resulted in empty content after processing.")

            # Add warnings based on node state
            if len(curr_node.images) > max_images: user_warnings.add(f"⚠️ Max {max_images} image{'s' if max_images != 1 else ''}" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments: user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed: user_warnings.add("⚠️ Couldn't fetch full history")
            elif curr_node.parent_msg is not None and len(messages) == max_messages: user_warnings.add(f"⚠️ Only using last {len(messages)} message{'s' if len(messages) != 1 else ''}")

            curr_msg = curr_node.parent_msg # Move to the next message in the chain (only if parent_msg was set by reply/thread logic)

    # --- Process External Content (URLs) ---
    if video_ids_with_urls:
        youtube_api_key = cfg.get("youtube_data_api_key")
        logging.info(f"Processing {len(video_ids_with_urls)} YouTube URL(s)...")
        tasks = [process_youtube_url(vid, youtube_api_key, url) for vid, url in video_ids_with_urls]
        youtube_results = await asyncio.gather(*tasks, return_exceptions=True)
        append_parts = []
        for i, result in enumerate(youtube_results):
            vid, url = video_ids_with_urls[i]
            if isinstance(result, Exception):
                logging.error(f"Error processing YouTube URL {url}: {result}", exc_info=result)
                append_parts.append(f"youtube url {i+1}: {url}\nyoutube url {i+1} content: [Error processing URL: {type(result).__name__}]")
            elif result:
                formatted_parts = [f"URL: {url}"] # Start with the URL itself
                if result.get('title'): formatted_parts.append(f"Title: {result['title']}")
                if result.get('channel_name'): formatted_parts.append(f"Channel: {result['channel_name']}") # No truncation
                if result.get('description'): formatted_parts.append(f"Description: {result['description']}") # No truncation
                if result.get('transcript_text'): formatted_parts.append(f"Transcript: {result['transcript_text']}") # No truncation
                if result.get('comments_text'): formatted_parts.append(f"Top Comments: {result['comments_text']}") # No truncation
                append_parts.append(f"--- YouTube Content {i+1} ---\n" + "\n".join(formatted_parts))
        if append_parts:
            youtube_content_to_append = "\n\n" + "\n\n".join(append_parts)
            logging.info("Finished processing YouTube URLs.")
        if not youtube_api_key:
             user_warnings.add("⚠️ YouTube URLs found, but processing limited (no API key).")

    if reddit_urls_with_ids:
        reddit = await get_reddit_client()
        if reddit:
            logging.info(f"Processing {len(reddit_urls_with_ids)} Reddit URL(s)...")
            tasks = [fetch_reddit_content(reddit, sub_id) for _, sub_id in reddit_urls_with_ids]
            reddit_results = await asyncio.gather(*tasks, return_exceptions=True)
            append_parts = []
            for i, result in enumerate(reddit_results):
                url, sub_id = reddit_urls_with_ids[i]
                if isinstance(result, Exception):
                    logging.error(f"Error processing Reddit URL {url}: {result}", exc_info=result)
                    append_parts.append(f"reddit url {i+1}: {url}\nreddit url {i+1} content: [Error processing URL: {type(result).__name__}]")
                else:
                    append_parts.append(f"--- Reddit Content {i+1} ---\nURL: {url}\n{result}")
            if append_parts:
                reddit_content_to_append = "\n\n" + "\n\n".join(append_parts)
                logging.info("Finished processing Reddit URLs.")
        else:
            logging.warning("Reddit URLs detected, but Reddit client failed to initialize. Skipping Reddit processing.")
            user_warnings.add("⚠️ Reddit URLs found, but processing disabled (check config/logs).")

    if filtered_general_urls:
        logging.info(f"Processing {len(filtered_general_urls)} general URL(s)...")
        tasks = [fetch_and_extract_content(url) for url in filtered_general_urls]
        general_results = await asyncio.gather(*tasks, return_exceptions=True)
        append_parts = []
        for i, result in enumerate(general_results):
            url = filtered_general_urls[i]
            if isinstance(result, Exception):
                logging.error(f"Error processing general URL {url}: {result}", exc_info=result)
                append_parts.append(f"url {i+1}: {url}\nurl {i+1} content: [Error processing URL: {type(result).__name__}]")
            elif result and not result.startswith("[Error") and result != "[Unsupported content type]" and result != "[No text content found]": # Append only if successful and has content
                append_parts.append(f"--- Web Content {i+1} ---\nURL: {url}\n{result}")
        if append_parts:
            general_url_content_to_append = "\n\n" + "\n\n".join(append_parts)
            logging.info("Finished processing general URLs.")
    elif general_urls:
        logging.info("General URLs found, but they were all YouTube or Reddit URLs.")


    # --- Prepare Final Messages for LLM ---
    system_instruction_text = None
    if system_prompt := cfg.get("system_prompt"):
        system_prompt_extras = [f"Today's date: {dt.now().strftime('%B %d %Y')}."]
        if accept_usernames:
            system_prompt_extras.append("User's names (if provided) are their Discord IDs and should be typed as '<@ID>'.")
        full_system_prompt = "\n".join([system_prompt] + system_prompt_extras)
        if provider == "google":
             system_instruction_text = full_system_prompt
        else:
            # Insert system message at the beginning if not already present
            if not messages or messages[0].get("role") != "system":
                 messages.insert(0, dict(role="system", content=full_system_prompt))
            else: # Update existing system message if needed (less common)
                 messages[0]["content"] = full_system_prompt


    # Reverse messages to chronological order for LLM
    final_messages_for_llm = messages[::-1]

    # Append external content to the *last user message*
    content_to_append = youtube_content_to_append + reddit_content_to_append + general_url_content_to_append
    if content_to_append:
        appended_to_message = False
        for i in range(len(final_messages_for_llm) - 1, -1, -1):
            if final_messages_for_llm[i]['role'] == 'user':
                target_message = final_messages_for_llm[i]
                append_text = "\n\n<context>\n" + content_to_append + "\n</context>"

                if provider == "google":
                    # Ensure 'parts' exists and is a list, initialize if necessary
                    if 'parts' not in target_message or not isinstance(target_message['parts'], list):
                         content_val = target_message.get('content', '') # Get existing content if any
                         target_message['parts'] = [genai_types.Part.from_text(text=str(content_val))] if content_val else []
                    # Append the new text part
                    target_message['parts'].append(genai_types.Part.from_text(text=append_text))
                    target_message.pop('content', None) # Remove old content field if it exists
                else: # OpenAI
                    if 'content' not in target_message: target_message['content'] = ""
                    if isinstance(target_message['content'], str):
                        target_message['content'] += append_text
                    elif isinstance(target_message['content'], list):
                        # Find the last text part and append, or add a new text part
                        found_text_part = False
                        for part in reversed(target_message['content']):
                            if part.get('type') == 'text':
                                part['text'] += append_text
                                found_text_part = True
                                break
                        if not found_text_part:
                            target_message['content'].append({"type": "text", "text": append_text})
                    else:
                        # Handle unexpected content type by converting to string and appending
                        logging.warning(f"Unexpected content type in OpenAI message: {type(target_message['content'])}. Converting.")
                        target_message['content'] = str(target_message['content']) + append_text
                appended_to_message = True
                logging.info("Appended external content to the last user message.")
                break
        if not appended_to_message:
             logging.warning("Could not find a user message to append external content to. Appending as a new user message.")
             # Append as a new user message if no suitable one found
             new_user_message = {"role": "user"}
             append_text = "<context>\n" + content_to_append + "\n</context>"
             if provider == "google":
                 new_user_message["parts"] = [genai_types.Part.from_text(text=append_text)]
             else:
                 new_user_message["content"] = append_text
             if accept_usernames: # Add user ID if possible
                 new_user_message["name"] = str(new_msg.author.id)
             final_messages_for_llm.append(new_user_message)


    logging.info(f"Prepared {len(final_messages_for_llm)} messages for LLM provider '{provider}'. Last message length (approx): {len(str(final_messages_for_llm[-1])) if final_messages_for_llm else 0}")

    # --- LLM Interaction with Retry Logic ---
    response_msgs = []
    edit_task = None
    last_task_time = 0
    # final_response_text is reset inside the loop
    embed = discord.Embed(description=STREAMING_INDICATOR, color=EMBED_COLOR_INCOMPLETE)
    embed.set_footer(text=f"Model: {provider_slash_model}") # Add footer initially
    initial_embed_sent = False
    last_exception = None
    tried_keys: Set[str] = set()
    max_retries = key_manager.get_total_keys()
    # If max_retries is 0 (e.g., local model with no keys), set to 1 to allow one attempt
    if max_retries == 0 and provider in ["ollama", "lmstudio", "vllm", "oobabooga", "jan"]:
        max_retries = 1


    async with new_msg.channel.typing(): # Keep typing indicator during retries
        for attempt in range(max_retries):
            # --- Reset state for retry ---
            current_message_content = ""
            final_response_text = "" # Reset accumulated text for the new attempt
            last_chunk = None
            finish_reason = None
            stream_error = None # Specific error encountered during streaming
            # Don't reset response_msgs here, we might edit the last one on error

            # --- Get Key and Initialize Client ---
            api_key = await key_manager.get_valid_key()
            # If api_key is None and it's a local provider, proceed without a key
            if api_key is None and provider not in ["ollama", "lmstudio", "vllm", "oobabooga", "jan"]:
                logging.error(f"Attempt {attempt+1}/{max_retries}: No valid API keys available for provider '{provider}'.")
                last_exception = Exception(f"No valid API keys available for provider '{provider}' after checking rate limits.")
                break # Exit retry loop if no keys are available for non-local

            if api_key: # Log key usage only if a key exists
                if api_key in tried_keys:
                     logging.warning(f"Attempt {attempt+1}/{max_retries}: Re-trying key ...{api_key[-4:]} for {provider} after DB reset.")
                     # Allow re-trying if DB was reset
                tried_keys.add(api_key)
                logging.info(f"Attempt {attempt+1}/{max_retries}: Using key ...{api_key[-4:]} for provider '{provider}'.")
            else:
                 logging.info(f"Attempt {attempt+1}/{max_retries}: Proceeding without API key for local provider '{provider}'.")


            try:
                # Initialize the specific client with the selected key (or None for local)
                if provider == "google":
                    if not api_key: # Should not happen based on previous check, but safety first
                         raise ValueError("Google provider requires an API key.")
                    genai_client = genai.Client(api_key=api_key)
                    # Prepare Gemini specific request parts (contents, config)
                    gemini_contents = []
                    for msg in final_messages_for_llm:
                        role = 'model' if msg['role'] == 'assistant' else msg['role']
                        if role == 'system': continue # Skip system for contents, handled in config
                        parts_data = msg.get('parts', [])
                        if isinstance(parts_data, list):
                            valid_parts = [p for p in parts_data if isinstance(p, genai_types.Part)]
                            if valid_parts:
                                gemini_contents.append(genai_types.Content(role=role, parts=valid_parts))
                            # else: logging.warning(f"No valid parts for Gemini message role {role}.") # Reduce noise
                        # else: logging.warning(f"Unexpected parts data type for Gemini: {type(parts_data)}") # Reduce noise

                    # Define safety settings and tools
                    safety_settings = [
                        genai_types.SafetySetting(category=cat, threshold=genai_types.HarmBlockThreshold.BLOCK_NONE)
                        for cat in [
                            genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT, genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            # Add CIVIC_INTEGRITY if needed/available for the model
                            # genai_types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY
                        ]
                    ]
                    # Conditionally add grounding tool based on model support (example)
                    tools = []
                    # Simple check based on common model names supporting grounding
                    if any(m in model for m in ["gemini-2.5", "gemini-2.0"]):
                        tools.append(genai_types.Tool(google_search=genai_types.GoogleSearch()))
                        logging.debug(f"Adding GoogleSearch tool for model {model}")
                    else:
                         logging.debug(f"Grounding tool skipped for model {model}")


                    generation_config_params = {"safety_settings": safety_settings}
                    if tools: generation_config_params["tools"] = tools
                    if system_instruction_text: generation_config_params["system_instruction"] = system_instruction_text

                    # Apply extra parameters from config
                    extra_params = cfg.get("extra_api_parameters", {})
                    for key, value in extra_params.items():
                         # Map common params, allow others if they exist in GenerateContentConfig
                         if key == "max_tokens": generation_config_params["max_output_tokens"] = value
                         elif key in ["temperature", "top_p", "top_k", "stop_sequences", "candidate_count", "seed"]: # Add seed here
                             generation_config_params[key] = value
                         else: logging.warning(f"Ignoring unsupported extra_api_parameter for Gemini: {key}")

                    generation_config = genai_types.GenerateContentConfig(**generation_config_params)

                    # --- Gemini Streaming ---
                    stream_iterator = await genai_client.aio.models.generate_content_stream(
                        model=model, contents=gemini_contents, config=generation_config
                    )
                    async for chunk in stream_iterator:
                        last_chunk = chunk # Store the last received chunk

                        # Check for safety blocking first
                        if chunk.candidates and chunk.candidates[0].finish_reason == genai_types.FinishReason.SAFETY:
                            logging.warning(f"Gemini response blocked (Safety). Key: ...{api_key[-4:] if api_key else 'N/A'}. Ratings: {chunk.candidates[0].safety_ratings}")
                            user_warnings.add("⚠️ Response blocked by safety filter")
                            finish_reason = "SAFETY" # Use uppercase enum name for consistency
                            current_message_content = "Response blocked by safety filter."
                            stream_error = genai_errors.BlockedPromptException("Response blocked by safety filter") # Use specific error type
                            break # Stop processing this stream

                        # Check for other finish reasons indicating an issue mid-stream
                        # Ensure finish_reason exists before accessing .name
                        chunk_finish_reason = chunk.candidates[0].finish_reason if chunk.candidates else None
                        if chunk_finish_reason and chunk_finish_reason not in (genai_types.FinishReason.FINISH_REASON_UNSPECIFIED, genai_types.FinishReason.STOP):
                             # e.g., RECITATION, OTHER
                             reason_name = chunk_finish_reason.name # Safe to access .name now
                             logging.warning(f"Gemini stream stopped prematurely. Key: ...{api_key[-4:] if api_key else 'N/A'}. Reason: {reason_name}")
                             finish_reason = reason_name # Store the reason name
                             # Treat OTHER and RECITATION as errors for retry purposes
                             if chunk_finish_reason in (genai_types.FinishReason.OTHER, genai_types.FinishReason.RECITATION):
                                 stream_error = genai_errors.StopCandidateException(f"Stream stopped due to: {reason_name}")
                             break # Stop processing stream

                        if new_text := getattr(chunk, 'text', None):
                            current_message_content += new_text
                            final_response_text += new_text # Accumulate full text for this attempt

                            # --- Message Splitting and Sending/Editing Logic (Common) ---
                            while len(current_message_content) > max_message_length:
                                split_point = current_message_content[:max_message_length].rfind('\n') if '\n' in current_message_content[:max_message_length] else max_message_length
                                text_to_send = current_message_content[:split_point]
                                current_message_content = current_message_content[split_point:].lstrip()
                                reply_to_msg = new_msg if not response_msgs else response_msgs[-1]

                                if use_plain_responses:
                                    response_msg = await reply_to_msg.reply(content=text_to_send or " ", suppress_embeds=True, mention_author = False)
                                else:
                                    if edit_task: await edit_task # Wait for previous edit to finish
                                    embed.description = text_to_send or " " # Ensure not empty
                                    embed.color = EMBED_COLOR_COMPLETE
                                    embed.clear_fields() # Clear fields on completed messages
                                    embed.set_footer(text=f"Model: {provider_slash_model}") # Ensure footer on split message
                                    if not response_msgs: # This is the first message being sent
                                         # Add warnings to the first message
                                         for warning in sorted(user_warnings): embed.add_field(name=warning, value="", inline=False)
                                         response_msg = await reply_to_msg.reply(embed=embed, mention_author = False)
                                         initial_embed_sent = True
                                    else: # Edit the *previous* message to finalize it before sending the new one
                                         # Finalize previous message (remove streaming indicator, set color, remove view)
                                         prev_embed = discord.Embed(description=text_to_send or " ", color=EMBED_COLOR_COMPLETE)
                                         prev_embed.set_footer(text=f"Model: {provider_slash_model}") # Ensure footer on finalized previous message
                                         edit_task = asyncio.create_task(response_msgs[-1].edit(embed=prev_embed, view=None)) # Remove view from finalized message
                                         await edit_task # Wait for final edit
                                         # Now send the new message as a reply to the finalized one
                                         embed.description = current_message_content + STREAMING_INDICATOR # Prepare embed for the *new* message
                                         embed.color = EMBED_COLOR_INCOMPLETE
                                         embed.set_footer(text=f"Model: {provider_slash_model}") # Ensure footer on new streaming message
                                         embed.clear_fields() # Clear fields for the new message
                                         response_msg = await response_msgs[-1].reply(embed=embed, mention_author = False) # Reply to the previous bot message
                                         edit_task = None # Reset edit task for the new message

                                if response_msg not in response_msgs:
                                    response_msgs.append(response_msg)
                                    msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                                last_task_time = dt.now().timestamp()

                            # --- Throttled Edit (Common) ---
                            if not use_plain_responses:
                                ready_to_edit = (edit_task is None or edit_task.done()) and dt.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS
                                if not response_msgs or not initial_embed_sent: # Send initial message if not split yet
                                    embed.description = current_message_content + STREAMING_INDICATOR
                                    embed.color = EMBED_COLOR_INCOMPLETE
                                    embed.clear_fields()
                                    embed.set_footer(text=f"Model: {provider_slash_model}") # Ensure footer on initial message
                                    for warning in sorted(user_warnings): embed.add_field(name=warning, value="", inline=False)
                                    reply_to_msg = new_msg if not response_msgs else response_msgs[-1]
                                    response_msg = await reply_to_msg.reply(embed=embed, mention_author = False)
                                    if response_msg not in response_msgs:
                                         response_msgs.append(response_msg)
                                         msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                                    initial_embed_sent = True
                                    last_task_time = dt.now().timestamp()
                                elif ready_to_edit and response_msgs: # Edit existing message
                                    if edit_task: await edit_task
                                    embed.description = current_message_content + STREAMING_INDICATOR
                                    embed.color = EMBED_COLOR_INCOMPLETE
                                    embed.clear_fields() # Keep fields clear during streaming edits
                                    embed.set_footer(text=f"Model: {provider_slash_model}") # Ensure footer during streaming edits
                                    try:
                                        edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                                    except discord.errors.HTTPException as e:
                                         # Handle potential 400 error during edit (e.g., if content somehow became too long between checks)
                                         logging.error(f"HTTPException during streaming edit (Message ID: {response_msgs[-1].id}): {e}")
                                         # If edit fails, we might lose the streaming indicator, but the final update should fix it.
                                         # Avoid crashing the whole process.
                                         edit_task = None # Ensure task is None so we don't await a failed task
                                    last_task_time = dt.now().timestamp()
                    # --- End Common Streaming Logic ---

                    # Check for stream error after processing chunk
                    if stream_error:
                         raise stream_error # Raise the error to be caught by the outer try/except

                    # Determine final finish reason *after* the loop
                    if finish_reason is None and last_chunk and last_chunk.candidates:
                        final_reason_enum = last_chunk.candidates[0].finish_reason
                        if final_reason_enum: # Check if it's not None
                            finish_reason = final_reason_enum.name
                        else:
                            # If finish_reason is still None after the loop, assume STOP if content was generated
                            finish_reason = "STOP" if final_response_text else "UNKNOWN"
                            logging.warning(f"Gemini stream finished without explicit reason. Assuming '{finish_reason}'.")


                else: # OpenAI compatible provider
                    provider_config = cfg["providers"][provider]
                    base_url = provider_config.get("base_url")
                    # Use the selected API key (or default for local)
                    openai_api_key = api_key if api_key else "sk-no-key-required"
                    openai_client = AsyncOpenAI(base_url=base_url, api_key=openai_api_key, timeout=300)

                    kwargs = dict(model=model, messages=final_messages_for_llm, stream=True, extra_body=cfg.get("extra_api_parameters", {}))

                    # --- OpenAI Streaming ---
                    stream = await openai_client.chat.completions.create(**kwargs)
                    async for curr_chunk in stream:
                        last_chunk = curr_chunk # Store last chunk for OpenAI as well
                        # Check finish reason first
                        current_finish_reason = curr_chunk.choices[0].finish_reason
                        if current_finish_reason:
                             finish_reason = current_finish_reason # Store the final reason
                             # Check if it's an error reason
                             if finish_reason not in ("stop", "length", "tool_calls"): # Consider length (max_tokens) as non-error finish
                                 logging.warning(f"OpenAI stream stopped prematurely. Key: ...{openai_api_key[-4:]}. Reason: {finish_reason}")
                                 # Treat content_filter as a specific error type for potential retry logic
                                 if finish_reason == "content_filter":
                                     stream_error = APIError("Response blocked by content filter", request=None, body=None) # Create a generic APIError
                                 else:
                                     # For other reasons like 'function_call_error', treat as unexpected stop for now
                                     stream_error = APIError(f"Stream stopped unexpectedly: {finish_reason}", request=None, body=None)
                                 break # Stop processing stream

                        delta_content = curr_chunk.choices[0].delta.content or ""

                        current_message_content += delta_content
                        final_response_text += delta_content # Accumulate full text for this attempt

                        # --- Message Splitting and Sending/Editing Logic (Common - Copied) ---
                        while len(current_message_content) > max_message_length:
                            split_point = current_message_content[:max_message_length].rfind('\n') if '\n' in current_message_content[:max_message_length] else max_message_length
                            text_to_send = current_message_content[:split_point]
                            current_message_content = current_message_content[split_point:].lstrip()
                            reply_to_msg = new_msg if not response_msgs else response_msgs[-1]

                            if use_plain_responses:
                                response_msg = await reply_to_msg.reply(content=text_to_send or " ", suppress_embeds=True, mention_author = False)
                            else:
                                if edit_task: await edit_task
                                embed.description = text_to_send or " "
                                embed.color = EMBED_COLOR_COMPLETE
                                embed.set_footer(text=f"Model: {provider_slash_model}") # Ensure footer on split message
                                embed.clear_fields()
                                if not response_msgs:
                                    for warning in sorted(user_warnings): embed.add_field(name=warning, value="", inline=False)
                                    response_msg = await reply_to_msg.reply(embed=embed, mention_author = False)
                                    initial_embed_sent = True
                                else:
                                    prev_embed = discord.Embed(description=text_to_send or " ", color=EMBED_COLOR_COMPLETE)
                                    prev_embed.set_footer(text=f"Model: {provider_slash_model}") # Ensure footer on finalized previous message
                                    edit_task = asyncio.create_task(response_msgs[-1].edit(embed=prev_embed, view=None))
                                    await edit_task
                                    embed.description = current_message_content + STREAMING_INDICATOR
                                    embed.color = EMBED_COLOR_INCOMPLETE
                                    embed.clear_fields()
                                    embed.set_footer(text=f"Model: {provider_slash_model}") # Ensure footer on new streaming message
                                    response_msg = await response_msgs[-1].reply(embed=embed, mention_author = False)
                                    edit_task = None

                            if response_msg not in response_msgs:
                                response_msgs.append(response_msg)
                                msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                            last_task_time = dt.now().timestamp()

                        # --- Throttled Edit (Common - Copied) ---
                        if not use_plain_responses:
                            ready_to_edit = (edit_task is None or edit_task.done()) and dt.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS
                            if not response_msgs or not initial_embed_sent:
                                embed.description = current_message_content + STREAMING_INDICATOR
                                embed.color = EMBED_COLOR_INCOMPLETE
                                embed.set_footer(text=f"Model: {provider_slash_model}") # Ensure footer on initial message
                                embed.clear_fields(); [embed.add_field(name=w, value="", inline=False) for w in sorted(user_warnings)]
                                reply_to_msg = new_msg if not response_msgs else response_msgs[-1]
                                response_msg = await reply_to_msg.reply(embed=embed, mention_author = False)
                                if response_msg not in response_msgs: response_msgs.append(response_msg); msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                                initial_embed_sent = True; last_task_time = dt.now().timestamp()
                            elif ready_to_edit and response_msgs:
                                if edit_task: await edit_task # Wait for previous edit if any
                                embed.description = current_message_content + STREAMING_INDICATOR
                                embed.color = EMBED_COLOR_INCOMPLETE
                                embed.clear_fields()
                                embed.set_footer(text=f"Model: {provider_slash_model}") # Ensure footer during streaming edits
                                try: edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                                except discord.errors.HTTPException as e: logging.error(f"HTTPException during streaming edit (Msg ID: {response_msgs[-1].id}): {e}"); edit_task = None
                                last_task_time = dt.now().timestamp()
                    # --- End Common Streaming Logic ---

                    # Check for stream error after processing chunk
                    if stream_error:
                         raise stream_error # Raise the error to be caught by the outer try/except

                # If we finished the stream without errors, break the retry loop
                logging.info(f"Successfully completed API call with key ...{api_key[-4:] if api_key else 'N/A'} for provider '{provider}'. Finish reason: {finish_reason}")
                last_exception = None # Clear last exception on success
                break # Exit retry loop

            # --- Exception Handling for the current attempt ---
            # Catch Google Rate Limit (ResourceExhausted) and OpenAI Rate Limit
            except (core_exceptions.ResourceExhausted, OpenAIRateLimitError) as e:
                logging.warning(f"Rate limit error for key ...{api_key[-4:] if api_key else 'N/A'} (Provider: {provider}). Attempt {attempt+1}/{max_retries}. Error: {e}")
                if api_key: # Only mark if a key was actually used
                    await key_manager.mark_rate_limited(api_key)
                last_exception = e
                await asyncio.sleep(0.5) # Small delay before trying next key
                continue # Go to the next attempt

            # Catch Temporary / Server-Side Errors
            except (genai_errors.InternalServerError, genai_errors.DeadlineExceededError, APITimeoutError, APIConnectionError, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.WriteTimeout, genai_errors.ServiceUnavailableError, genai_errors.UnknownError, core_exceptions.InternalServerError, core_exceptions.ServiceUnavailable) as e:
                logging.warning(f"Temporary API Error for key ...{api_key[-4:] if api_key else 'N/A'} (Provider: {provider}). Attempt {attempt+1}/{max_retries}. Error: {type(e).__name__}: {e}")
                # Don't mark key as rate limited for temporary errors
                last_exception = e
                await asyncio.sleep(1.0 + attempt * 0.5) # Slightly longer delay, increasing with attempts
                continue # Go to the next attempt

            # Catch Authentication / Permission Errors
            except (genai_errors.PermissionDeniedError, OpenAIAuthenticationError, OpenAIPermissionDeniedError, core_exceptions.PermissionDenied) as e:
                 logging.error(f"Authentication/Permission Error for key ...{api_key[-4:] if api_key else 'N/A'} (Provider: {provider}). Attempt {attempt+1}/{max_retries}. Error: {type(e).__name__}: {e}", exc_info=False)
                 if api_key: # Mark the failing key as rate-limited to prevent immediate reuse
                      await key_manager.mark_rate_limited(api_key)
                 last_exception = e
                 # Continue to try other keys unless it's the last attempt
                 if attempt + 1 >= max_retries:
                     break # Exit loop if last attempt failed
                 else:
                     await asyncio.sleep(0.5)
                     continue

            # Catch Bad Requests / Content Filters / Other API Errors
            except (genai_errors.APIError, APIError, OpenAIBadRequestError, ValueError, genai_errors.BlockedPromptException, genai_errors.StopCandidateException, core_exceptions.InvalidArgument, core_exceptions.FailedPrecondition) as e:
                 logging.error(f"Non-retryable API/Request Error for key ...{api_key[-4:] if api_key else 'N/A'} (Provider: {provider}). Attempt {attempt+1}/{max_retries}. Error: {type(e).__name__}: {e}", exc_info=False)
                 # Don't mark key as rate limited, could be a config/prompt issue
                 last_exception = e
                 break # Exit retry loop for non-retryable errors for this request

            # Catch any other unexpected errors
            except Exception as e:
                logging.exception(f"Unexpected error during API call/stream with key ...{api_key[-4:] if api_key else 'N/A'} (Provider: {provider}). Attempt {attempt+1}/{max_retries}.")
                last_exception = e
                # Assume most unexpected errors aren't key-specific or easily retryable
                break # Exit retry loop

        # --- After Retry Loop ---
        if last_exception:
            # All keys failed or a non-retryable error occurred
            error_type_name = type(last_exception).__name__
            error_message_detail = str(last_exception)

            # Customize error message based on type
            if isinstance(last_exception, (core_exceptions.ResourceExhausted, OpenAIRateLimitError)):
                 # Use the imported constant here
                 error_message = f"Sorry, all API keys for '{provider}' seem to be rate-limited. Please try again in {COOLDOWN_PERIOD_HOURS} hours or contact the bot owner."
            elif isinstance(last_exception, (genai_errors.InternalServerError, genai_errors.DeadlineExceededError, APITimeoutError, APIConnectionError, httpx.ReadTimeout, httpx.ConnectTimeout, genai_errors.ServiceUnavailableError, core_exceptions.InternalServerError, core_exceptions.ServiceUnavailable)):
                 error_message = f"Sorry, the AI service seems busy or unavailable ({error_type_name}). Please try again later."
            elif isinstance(last_exception, (genai_errors.PermissionDeniedError, OpenAIAuthenticationError, OpenAIPermissionDeniedError, core_exceptions.PermissionDenied)):
                 error_message = f"Sorry, an API authentication/permission error occurred ({error_type_name}). Please contact the bot owner to check the API keys."
            elif isinstance(last_exception, (genai_errors.APIError, APIError, OpenAIBadRequestError, ValueError, genai_errors.BlockedPromptException, genai_errors.StopCandidateException, core_exceptions.InvalidArgument, core_exceptions.FailedPrecondition)):
                 # Includes bad requests, content filters, other API issues
                 error_message = f"Sorry, the request failed ({error_type_name}). Please check your prompt/attachments or contact the bot owner if the issue persists.\n`{error_message_detail}`"
            else: # General unexpected error
                 error_message = f"Sorry, an unexpected error occurred: {error_type_name}\n`{error_message_detail}`"

            logging.error(f"All API key attempts failed for provider '{provider}'. Final error: {error_type_name}: {error_message_detail}")

            # Send error message to Discord
            if response_msgs and not use_plain_responses: # Edit last message if possible
                if edit_task:
                    try: await edit_task
                    except Exception as task_e: logging.error(f"Error awaiting final edit task before error message: {task_e}")
                embed.description = (current_message_content or "") + f"\n\n**Error:** {error_message}" # Include any partial text received
                # Truncate if necessary
                if len(embed.description) > max_embed_desc_length:
                    embed.description = embed.description[:max_embed_desc_length - len("... (error message truncated)")] + "... (error message truncated)"
                embed.color = EMBED_COLOR_ERROR
                embed.set_footer(text=f"Model: {provider_slash_model}") # Add footer even on error
                embed.clear_fields(); [embed.add_field(name=w, value="", inline=False) for w in sorted(user_warnings)] # Add warnings even on error
                try: await response_msgs[-1].edit(embed=embed, view=None)
                except discord.errors.HTTPException as edit_err: logging.error(f"HTTPException during final error edit: {edit_err}")
                except discord.errors.NotFound: logging.warning("Could not edit final error message (message not found).")
            else: # Reply to original message
                # Combine warnings with the error message for plain text
                warning_text = "\n".join(sorted(user_warnings))
                full_error_message = f"{error_message}\n\n{warning_text}".strip()
                # Truncate if necessary
                if len(full_error_message) > max_plain_msg_length:
                     full_error_message = full_error_message[:max_plain_msg_length - len("... (message truncated)")] + "... (message truncated)"
                await new_msg.reply(full_error_message, mention_author = False)
            return # Stop processing this message

    # --- Final Message Update (Successful Completion) ---
    final_view = None
    should_show_sources = False
    # Refined check for good finish reason (convert potential enum name to lowercase)
    is_good_finish = finish_reason is not None and finish_reason.lower() in ("stop", "end_turn", "tool_calls", "length", "max_tokens")

    final_part_content = current_message_content # Remaining content after loop

    # Check for grounding metadata (Gemini only)
    if provider == "google" and last_chunk and last_chunk.candidates:
        candidate = last_chunk.candidates[0]
        # Check if grounding_metadata exists and is not empty
        if metadata := getattr(candidate, 'grounding_metadata', None):
            has_queries = hasattr(metadata, 'web_search_queries') and metadata.web_search_queries
            has_chunks = hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks
            if (has_queries or has_chunks) and response_msgs:
                gemini_grounding_metadata[response_msgs[-1].id] = metadata
                if not use_plain_responses: should_show_sources = True; logging.info("Grounding metadata found, enabling sources button.")
                else: logging.info("Grounding metadata found, plain responses enabled, button skipped.")
            elif not response_msgs: logging.warning("Grounding metadata found, but no response message exists.")
            # else: logging.debug("Grounding metadata object present but empty.") # Debug level
        # else: logging.debug("No grounding metadata found in final chunk.") # Debug level

    # Create the final view only if needed and response was successful
    if not use_plain_responses and is_good_finish:
        final_view = ResponseActionsView(show_sources_button=should_show_sources)

    # Final update/send logic
    if use_plain_responses:
        # Send the last part if it has content and wasn't sent during splitting
        last_sent_content = response_msgs[-1].content if response_msgs else ""
        if final_part_content and final_part_content.strip() and final_part_content != last_sent_content: # Check strip()
            reply_to_msg = new_msg if not response_msgs else response_msgs[-1]
            # Plain responses don't get the button view
            try:
                response_msg = await reply_to_msg.reply(content=final_part_content or " ", suppress_embeds=True, mention_author = False)
                response_msgs.append(response_msg)
                msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
            except discord.errors.HTTPException as send_err:
                 logging.error(f"Failed to send final plain text part: {send_err}")

    elif response_msgs: # Final edit for the last embed message
        if edit_task:
            try: await edit_task
            except Exception as task_e: logging.error(f"Error awaiting final edit task before final update: {task_e}")

        embed.description = final_part_content or " " # Use the remaining content
        embed.color = EMBED_COLOR_COMPLETE if is_good_finish else EMBED_COLOR_ERROR
        embed.set_footer(text=f"Model: {provider_slash_model}") # Ensure footer on final update
        embed.clear_fields(); [embed.add_field(name=w, value="", inline=False) for w in sorted(user_warnings)]

        # Ensure description isn't too long for the final edit
        if len(embed.description) > max_embed_desc_length:
             logging.warning(f"Final embed description truncated (was {len(embed.description)} chars).")
             embed.description = embed.description[:max_embed_desc_length]

        # --- Retry logic for final edit ---
        last_edit_exception = None
        for attempt in range(MAX_FINAL_EDIT_RETRIES):
            try:
                await response_msgs[-1].edit(embed=embed, view=final_view) # Use the combined view
                logging.info(f"Final edit successful for message {response_msgs[-1].id} on attempt {attempt + 1}.")
                last_edit_exception = None # Clear exception on success
                break # Exit retry loop on success
            except discord.errors.HTTPException as e:
                last_edit_exception = e
                status_code = e.status if hasattr(e, 'status') else 'Unknown'
                error_text = e.text if hasattr(e, 'text') else 'No details'
                logging.warning(f"HTTPException during final edit attempt {attempt + 1}/{MAX_FINAL_EDIT_RETRIES} (Message ID: {response_msgs[-1].id}): {status_code} {error_text}")
                if attempt < MAX_FINAL_EDIT_RETRIES - 1:
                    await asyncio.sleep(FINAL_EDIT_RETRY_DELAY * (attempt + 1)) # Basic backoff
                else:
                    logging.error(f"Final edit failed after {MAX_FINAL_EDIT_RETRIES} attempts for message {response_msgs[-1].id}. Leaving message as is.")
            except discord.errors.NotFound:
                logging.warning(f"Could not perform final edit on message {response_msgs[-1].id} (message not found).")
                last_edit_exception = None # Don't treat NotFound as a retryable failure of the edit itself
                break # Exit loop, message is gone
            except Exception as e: # Catch other potential errors during edit
                 last_edit_exception = e
                 logging.error(f"Unexpected error during final edit attempt {attempt + 1} for message {response_msgs[-1].id}: {e}", exc_info=True)
                 break # Don't retry unexpected errors

        # If last_edit_exception is still set, all retries failed. The message is left in its last streamed state.
        # No need for the old fallback plain text message here.

    elif not response_msgs and final_part_content: # Handle case where no streaming occurred but there's a final response
         # Send the initial message now if it wasn't sent during streaming
         embed.description = final_part_content or " "
         # Ensure description isn't too long
         if len(embed.description) > max_embed_desc_length:
              logging.warning(f"Initial embed description truncated (was {len(embed.description)} chars).")
              embed.description = embed.description[:max_embed_desc_length]
         embed.color = EMBED_COLOR_COMPLETE if is_good_finish else EMBED_COLOR_ERROR
         embed.set_footer(text=f"Model: {provider_slash_model}") # Ensure footer on initial non-streamed message
         embed.clear_fields(); [embed.add_field(name=w, value="", inline=False) for w in sorted(user_warnings)]
         try:
             response_msg = await new_msg.reply(embed=embed, view=final_view, mention_author = False) # Use the combined view
             response_msgs.append(response_msg)
             msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
             if should_show_sources and provider == "google": # Need to store metadata if button added here
                  if last_chunk and last_chunk.candidates and hasattr(last_chunk.candidates[0], 'grounding_metadata'):
                       gemini_grounding_metadata[response_msg.id] = last_chunk.candidates[0].grounding_metadata
         except discord.errors.HTTPException as send_err:
              logging.error(f"Failed to send initial non-streamed response: {send_err}")
         except Exception as e:
              logging.error(f"Unexpected error sending initial non-streamed response: {e}", exc_info=True)


    # --- Update Caches ---
    if response_msgs and is_good_finish: # Only store if successful
        last_msg_id = response_msgs[-1].id
        # Store full text for file download, check cache size before adding
        if len(full_response_texts) < MAX_FULL_RESPONSE_CACHE:
            full_response_texts[last_msg_id] = final_response_text
        else:
            # Simple FIFO eviction if cache is full
            try:
                oldest_key = next(iter(full_response_texts))
                full_response_texts.pop(oldest_key, None)
                full_response_texts[last_msg_id] = final_response_text
                logging.debug(f"Full response cache full, evicted {oldest_key}")
            except StopIteration: # Cache was empty
                full_response_texts[last_msg_id] = final_response_text


        # Update node cache (optional, could store truncated text if needed)
        node = msg_nodes.setdefault(last_msg_id, MsgNode(parent_msg=new_msg))
        async with node.lock:
            # Store truncated text in node if desired, or leave as None
            # node.text = final_response_text[:1000] # Example: store first 1000 chars
            node.role = "assistant"

    # --- Cleanup Old Nodes and Full Responses ---
    # Cleanup msg_nodes (More efficient check)
    if len(msg_nodes) > MAX_MESSAGE_NODES:
        # Sort keys by message ID (assuming higher ID means newer)
        sorted_node_ids = sorted(msg_nodes.keys())
        num_to_remove = len(msg_nodes) - MAX_MESSAGE_NODES
        node_ids_to_remove = sorted_node_ids[:num_to_remove]
        for msg_id in node_ids_to_remove:
            msg_nodes.pop(msg_id, None)
            gemini_grounding_metadata.pop(msg_id, None) # Also remove associated grounding data
        logging.debug(f"Cleaned up {num_to_remove} old message nodes.")

    # Cleanup full_response_texts (Already handled by FIFO above)


@discord_client.event
async def on_ready():
    logging.info(f"Logged in as {discord_client.user}")
    # Register persistent view (show_sources_button=False initially, actual button added dynamically)
    # Need to pass a dummy value that evaluates to False initially.
    discord_client.add_view(ResponseActionsView(show_sources_button=False))
    await get_reddit_client() # Initialize Reddit client


@discord_client.event
async def on_disconnect():
    logging.warning("Discord client disconnected.")
    await httpx_client.aclose()
    logging.info("Closed httpx client.")
    global reddit_client
    if reddit_client:
        try:
            await reddit_client.close()
            logging.info("Closed asyncpraw Reddit client session.")
        except Exception as e:
            logging.error(f"Error closing asyncpraw Reddit client: {e}")
        finally:
             reddit_client = None


async def main():
    global cfg # Ensure cfg is accessible
    cfg = await get_config() # Load config async at start
    bot_token = cfg.get("bot_token")
    if not bot_token:
        logging.critical("CRITICAL: bot_token not found in config.yaml. Exiting.")
        return
    try:
        await discord_client.start(bot_token)
    except discord.LoginFailure:
        logging.critical("CRITICAL: Failed to log in. Please check your bot_token in config.yaml.")
    except Exception as e:
        logging.critical(f"CRITICAL: Discord client encountered an error during startup: {e}", exc_info=True)
    finally:
        if not discord_client.is_closed():
            await discord_client.close()
        # Ensure httpx client is closed if main loop exits unexpectedly
        if not httpx_client.is_closed:
             await httpx_client.aclose()
             logging.info("Closed httpx client during shutdown.")
        # Ensure Reddit client is closed
        global reddit_client
        if reddit_client:
            try:
                await reddit_client.close()
                logging.info("Closed asyncpraw Reddit client session during shutdown.")
            except Exception as e:
                logging.error(f"Error closing asyncpraw Reddit client during shutdown: {e}")
            finally:
                 reddit_client = None


if __name__ == "__main__":
    try:
        # Ensure database directory exists before running main loop
        os.makedirs("ratelimit_db", exist_ok=True)
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Shutdown requested by user.")
