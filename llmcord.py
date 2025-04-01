import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime as dt
import logging
from typing import Literal, Optional, Dict, List, Tuple, Any
import base64
import random
import re
import json
from urllib.parse import parse_qs, urlparse
import os

import discord
import discord.ui
from discord import app_commands
import httpx
from openai import AsyncOpenAI
import yaml
from ruamel.yaml import YAML
import asyncpraw
from youtube_transcript_api import YouTubeTranscriptApi

from google import genai
from google.genai import types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("gpt-4", "claude-3", "gemini", "gemma", "pixtral", "mistral-small", "llava", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

GEMINI_PROVIDERS = ("google")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 100

YOUTUBE_URL_REGEX = r'(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})'

REDDIT_URL_REGEX = r'(https?://)?(www\.)?(reddit\.com/r/[^/\s]+/comments/[^/\s]+/[^/\s]+/?|reddit\.com/r/[^/\s]+/s/[^/\s]+/?)'

GENERAL_URL_REGEX = r'https?://(?!(?:www\.)?(?:youtube\.com|youtu\.be|reddit\.com))[^\s<>"\'()]+(?<![\.,!?;:])'

provider_key_indices: Dict[str, int] = {}

PROVIDER_MODELS = {
    "openai": ["o3-mini", "o1", "gpt-4o-2024-11-20"],
    "google": ["gemini-2.5-pro-exp-03-25", "gemini-2.0-flash"],
    "anthropic": ["claude-3.7-sonnet-thought", "claude-3.7-sonnet"],
}

class ShowSourcesButton(discord.ui.View):
    def __init__(self, grounding_embed):
        super().__init__()
        self.grounding_embed = grounding_embed

    @discord.ui.button(label="Show Sources", style=discord.ButtonStyle.primary)
    async def show_sources(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message(embed=self.grounding_embed, ephemeral=False)

def get_config(filename="config.yaml"):
    with open(filename, "r") as file:
        return yaml.safe_load(file)

def get_next_api_key(provider_config, provider_name):
    """Get the next API key for the provider using round-robin rotation."""

    api_keys = provider_config.get("api_keys", [])

    if not api_keys:
        return provider_config.get("api_key", "sk-no-key-required")

    curr_index = provider_key_indices.get(provider_name, -1)

    next_index = (curr_index + 1) % len(api_keys)

    provider_key_indices[provider_name] = next_index

    return api_keys[next_index]

async def safe_gemini_grounding_retrieval(provider, provider_config, model, gemini_messages, generate_content_config):
    """
    Safely retrieve grounding metadata with API key rotation.
    Returns grounding metadata or None if all keys fail.
    """
    api_keys = provider_config.get("api_keys", [])

    if not api_keys:
        single_key = provider_config.get("api_key", "sk-no-key-required")
        api_keys = [single_key]

    start_index = provider_key_indices.get(provider, 0)

    key_indices = [(start_index + i) % len(api_keys) for i in range(len(api_keys))]

    for idx in key_indices:
        api_key = api_keys[idx]

        logging.info(f"Trying Gemini grounding retrieval with key index {idx} for provider {provider}")

        try:
            genai_client = genai.Client(api_key=api_key)
            grounding_response = await genai_client.aio.models.generate_content(
                model=model,
                contents=gemini_messages,
                config=generate_content_config
            )

            if (hasattr(grounding_response.candidates[0], 'grounding_metadata') and 
                grounding_response.candidates[0].grounding_metadata):

                logging.info(f"Grounding metadata retrieval succeeded with key index {idx}")

                provider_key_indices[provider] = (idx + 1) % len(api_keys)
                return grounding_response.candidates[0].grounding_metadata
            else:
                logging.warning(f"No grounding metadata available with key index {idx}")
        except Exception as e:
            logging.warning(f"Grounding metadata retrieval failed with key index {idx}: {str(e)}")

    logging.warning("All keys failed for grounding metadata retrieval")
    return None

async def safe_gemini_stream_processor(new_msg, provider, provider_config, model, gemini_messages, generate_content_config, embed, response_msgs, response_contents, edit_task, use_plain_responses, max_message_length, enable_grounding):
    """
    Safely process Gemini API calls with automatic retries using different keys.
    """
    api_keys = provider_config.get("api_keys", [])

    if not api_keys:
        single_key = provider_config.get("api_key", "sk-no-key-required")
        api_keys = [single_key]

    start_index = provider_key_indices.get(provider, 0)

    key_indices = [(start_index + i) % len(api_keys) for i in range(len(api_keys))]

    last_exception = None
    success = False
    curr_content = None
    last_task_time = 0

    for idx in key_indices:
        api_key = api_keys[idx]
        provider_key_indices[provider] = idx

        logging.info(f"Trying Gemini API with key index {idx} for provider {provider}")

        try:
            genai_client = genai.Client(api_key=api_key)
            stream = await genai_client.aio.models.generate_content_stream(
                model=model,
                contents=gemini_messages,
                config=generate_content_config
            )

            try:
                async for chunk in stream:
                    new_content = chunk.text

                    if new_content:
                        prev_content = curr_content or ""
                        curr_content = new_content

                        if response_contents == [] and new_content == "":
                            continue

                        if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                            response_contents.append("")

                        response_contents[-1] += new_content

                        if not use_plain_responses:
                            ready_to_edit = (edit_task == None or edit_task.done()) and dt.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS
                            msg_split_incoming = len(response_contents[-1] + new_content) > max_message_length

                            if start_next_msg or ready_to_edit or msg_split_incoming:
                                if edit_task != None:
                                    await edit_task

                                embed.description = response_contents[-1] + (STREAMING_INDICATOR if not msg_split_incoming else "")
                                embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming else EMBED_COLOR_INCOMPLETE

                                if start_next_msg:
                                    reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                                    response_msg = await reply_to_msg.reply(embed=embed, mention_author = False)
                                    response_msgs.append(response_msg)

                                    global msg_nodes
                                    msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                                    await msg_nodes[response_msg.id].lock.acquire()
                                else:
                                    edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))

                                last_task_time = dt.now().timestamp()

                success = True

                provider_key_indices[provider] = (idx + 1) % len(api_keys)

                if not use_plain_responses and response_msgs:
                    embed.description = response_contents[-1]
                    embed.color = EMBED_COLOR_COMPLETE

                    view = None
                    if enable_grounding:
                        try:

                            grounding_metadata = await safe_gemini_grounding_retrieval(
                                provider=provider,
                                provider_config=provider_config,
                                model=model,
                                gemini_messages=gemini_messages,
                                generate_content_config=generate_content_config
                            )

                            if grounding_metadata:
                                view = create_grounding_view(grounding_metadata)

                        except Exception as e:
                            logging.exception(f"Error preparing grounding metadata: {str(e)}")

                    await response_msgs[-1].edit(embed=embed, view=view)

                return True  

            except Exception as stream_error:

                logging.warning(f"Stream processing failed with key index {idx}: {str(stream_error)}")
                last_exception = stream_error
                continue

        except Exception as init_error:

            logging.warning(f"Initial API connection failed with key index {idx}: {str(init_error)}")
            last_exception = init_error
            continue

    if not success:
        logging.error(f"All Gemini API keys failed. Last error: {str(last_exception)}")
        error_embed = discord.Embed(
            title="Error",
            description="An error occurred with all API keys. Please try again later.",
            color=discord.Color.red()
        )
        await new_msg.reply(embed=error_embed)
        return False

def create_grounding_view(grounding_metadata):
    """Create a view with grounding metadata information."""
    grounding_embed = discord.Embed(
        title="📚 Sources",
        description="This response was enhanced with Google Search results.",
        color=discord.Color.blue()
    )

    if hasattr(grounding_metadata, 'web_search_queries') and grounding_metadata.web_search_queries:
        search_queries = []
        for query in grounding_metadata.web_search_queries:
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            search_queries.append(f"[{query}]({search_url})")

        if search_queries:
            grounding_embed.add_field(
                name="Search Queries",
                value="\n".join(search_queries),
                inline=False
            )

    if hasattr(grounding_metadata, 'grounding_chunks') and grounding_metadata.grounding_chunks:
        all_sources = []
        for i, chunk in enumerate(grounding_metadata.grounding_chunks[:10]):  
            if hasattr(chunk, 'web') and hasattr(chunk.web, 'uri') and hasattr(chunk.web, 'title'):
                title = chunk.web.title
                if len(title) > 100:
                    title = title[:97] + "..."
                all_sources.append(f"[{title}]({chunk.web.uri})")

        if all_sources:
            sources_batches = []
            current_batch = []
            current_length = 0

            for source in all_sources:
                if current_length + len(source) + 1 > 1000:  
                    if current_batch:  
                        sources_batches.append(current_batch)
                    current_batch = [source]
                    current_length = len(source)
                else:
                    current_batch.append(source)
                    current_length += len(source) + 1  

            if current_batch:  
                sources_batches.append(current_batch)

            for i, batch in enumerate(sources_batches):
                field_name = "Top Sources" if i == 0 else f"More Sources ({i+1})"
                grounding_embed.add_field(
                    name=field_name,
                    value="\n".join(batch),
                    inline=False
                )

    if grounding_embed.fields:
        return ShowSourcesButton(grounding_embed)
    return None

async def safe_openai_stream_processor(new_msg, provider, provider_config, base_url, model, messages, extra_params, embed, response_msgs, response_contents, edit_task, use_plain_responses, max_message_length):
    """
    Safely process OpenAI API calls with automatic retries using different keys.
    """
    api_keys = provider_config.get("api_keys", [])

    if not api_keys:
        single_key = provider_config.get("api_key", "sk-no-key-required")
        api_keys = [single_key]

    start_index = provider_key_indices.get(provider, 0)

    key_indices = [(start_index + i) % len(api_keys) for i in range(len(api_keys))]

    last_exception = None
    success = False
    curr_content = None
    finish_reason = None
    last_task_time = 0

    for idx in key_indices:
        api_key = api_keys[idx]
        provider_key_indices[provider] = idx

        logging.info(f"Trying OpenAI API with key index {idx} for provider {provider}")

        try:
            client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                extra_body=extra_params
            )

            try:
                async for curr_chunk in stream:
                    if finish_reason != None:
                        break

                    finish_reason = curr_chunk.choices[0].finish_reason

                    prev_content = curr_content or ""
                    curr_content = curr_chunk.choices[0].delta.content or ""

                    new_content = prev_content if finish_reason == None else (prev_content + curr_content)

                    if response_contents == [] and new_content == "":
                        continue

                    if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                        response_contents.append("")

                    response_contents[-1] += new_content

                    if not use_plain_responses:
                        ready_to_edit = (edit_task == None or edit_task.done()) and dt.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS
                        msg_split_incoming = finish_reason == None and len(response_contents[-1] + curr_content) > max_message_length
                        is_final_edit = finish_reason != None or msg_split_incoming
                        is_good_finish = finish_reason != None and finish_reason.lower() in ("stop", "end_turn")

                        if start_next_msg or ready_to_edit or is_final_edit:
                            if edit_task != None:
                                await edit_task

                            embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                            embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                            if start_next_msg:
                                reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                                response_msg = await reply_to_msg.reply(embed=embed, mention_author = False)
                                response_msgs.append(response_msg)

                                global msg_nodes
                                msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                                await msg_nodes[response_msg.id].lock.acquire()
                            else:
                                edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))

                            last_task_time = dt.now().timestamp()

                success = True

                provider_key_indices[provider] = (idx + 1) % len(api_keys)

                return True  

            except Exception as stream_error:

                logging.warning(f"Stream processing failed with key index {idx}: {str(stream_error)}")
                last_exception = stream_error
                continue

        except Exception as init_error:

            logging.warning(f"Initial API connection failed with key index {idx}: {str(init_error)}")
            last_exception = init_error
            continue

    if not success:
        logging.error(f"All OpenAI API keys failed. Last error: {str(last_exception)}")
        error_embed = discord.Embed(
            title="Error",
            description="An error occurred with all API keys. Please try again later.",
            color=discord.Color.red()
        )
        await new_msg.reply(embed=error_embed)
        return False

def extract_youtube_links(text):
    """Extract YouTube video IDs from text."""
    matches = re.finditer(YOUTUBE_URL_REGEX, text)
    youtube_links = []

    for match in matches:
        video_id = match.group(4)
        full_url = match.group(0)
        youtube_links.append((video_id, full_url))

    return youtube_links

def extract_reddit_links(text):
    """Extract Reddit URLs from text."""
    matches = re.finditer(REDDIT_URL_REGEX, text)
    reddit_links = []

    for match in matches:
        reddit_url = match.group(0)
        reddit_links.append(reddit_url)

    return reddit_links

def extract_general_urls(text):
    """Extract general URLs from text, excluding YouTube and Reddit URLs."""
    matches = re.finditer(GENERAL_URL_REGEX, text)
    general_urls = []

    for match in matches:
        url = match.group(0)
        general_urls.append(url)

    return general_urls

async def get_youtube_video_metadata(video_id: str, api_key: str, httpx_client) -> Dict[str, Any]:
    """Get the video title, description and channel name using YouTube Data API v3."""
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key={api_key}"

    try:
        response = await httpx_client.get(url)
        response.raise_for_status()
        data = response.json()

        if not data.get('items'):
            return {
                "title": "Video not found",
                "description": "No information available",
                "channel_name": "Unknown channel"
            }

        snippet = data['items'][0]['snippet']
        return {
            "title": snippet.get('title', 'No title'),
            "description": snippet.get('description', 'No description'),
            "channel_name": snippet.get('channelTitle', 'Unknown channel')
        }
    except Exception as e:
        logging.error(f"Error fetching YouTube video metadata: {str(e)}")
        return {
            "title": "Failed to fetch title",
            "description": "Failed to fetch description",
            "channel_name": "Failed to fetch channel name"
        }

async def get_youtube_transcript(video_id: str) -> str:
    """Get the transcript of a YouTube video using youtube-transcript-api."""
    if not YouTubeTranscriptApi:
        return "Transcription unavailable: youtube-transcript-api not installed"

    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)

        full_transcript = ""
        for snippet in transcript.snippets:
            full_transcript += f"{snippet.text}\n"

        return full_transcript if full_transcript else "No transcript available"
    except Exception as e:
        logging.error(f"Error fetching YouTube transcript for {video_id}: {str(e)}")
        return "Failed to fetch transcript"

async def get_youtube_comments(video_id: str, api_key: str, httpx_client, max_comments: int = 20) -> List[Dict[str, str]]:
    """Get top comments for a YouTube video using YouTube Data API v3."""
    url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&maxResults={max_comments}&key={api_key}"

    try:
        response = await httpx_client.get(url)
        response.raise_for_status()
        data = response.json()

        comments = []
        for item in data.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append({
                "author": comment.get('authorDisplayName', 'Unknown author'),
                "text": comment.get('textDisplay', 'No comment text'),
                "like_count": comment.get('likeCount', 0)
            })

        return comments
    except Exception as e:
        logging.error(f"Error fetching YouTube comments: {str(e)}")
        return []

async def process_youtube_url(video_id: str, full_url: str, api_key: str, httpx_client) -> str:
    """Process a YouTube URL to extract all relevant information."""
    tasks = [
        get_youtube_video_metadata(video_id, api_key, httpx_client),
        get_youtube_transcript(video_id),
        get_youtube_comments(video_id, api_key, httpx_client)
    ]

    metadata, transcript, comments = await asyncio.gather(*tasks)

    content = f"Title: {metadata['title']}\n"
    content += f"Channel: {metadata['channel_name']}\n\n"

    content += "Description:\n"
    content += f"{metadata['description']}\n\n"

    content += "Transcript:\n"
    content += f"{transcript}\n\n"

    if comments:
        content += f"Top {len(comments)} Comments:\n"
        for i, comment in enumerate(comments, 1):
            content += f"{i}. {comment['author']}: {comment['text']} (Likes: {comment['like_count']})\n"
    else:
        content += "No comments available\n"

    return content

async def process_all_youtube_urls(youtube_links: List[Tuple[str, str]], youtube_api_key: str, httpx_client) -> str:
    """Process multiple YouTube URLs concurrently."""
    if not youtube_links:
        return ""

    tasks = [
        process_youtube_url(video_id, full_url, youtube_api_key, httpx_client) 
        for video_id, full_url in youtube_links
    ]

    results = await asyncio.gather(*tasks)

    formatted_content = ""
    for i, ((video_id, full_url), content) in enumerate(zip(youtube_links, results), 1):
        formatted_content += f"youtube url {i}: {full_url}\n"
        formatted_content += f"youtube url {i} content: {content}\n\n"

    return formatted_content

async def resolve_reddit_url(reddit_url: str, httpx_client) -> str:
    """Resolve a Reddit URL, following redirects for shortened URLs."""
    try:

        if "/s/" in reddit_url:
            response = await httpx_client.get(reddit_url, follow_redirects=True)
            if response.status_code == 200:
                return str(response.url)
            else:
                logging.error(f"Failed to resolve Reddit URL: {reddit_url} - Status code: {response.status_code}")
                return reddit_url
        return reddit_url
    except Exception as e:
        logging.error(f"Error resolving Reddit URL {reddit_url}: {str(e)}")
        return reddit_url

async def process_reddit_url(reddit_url, client_id, client_secret, user_agent, httpx_client):
    """Process a Reddit URL to extract its content using asyncpraw."""
    try:

        resolved_url = await resolve_reddit_url(reddit_url, httpx_client)

        reddit = asyncpraw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )

        submission = await reddit.submission(url=resolved_url)

        await submission.load()
        title = submission.title
        author = submission.author.name if hasattr(submission.author, 'name') else "[deleted]"
        selftext = submission.selftext
        score = submission.score
        upvote_ratio = submission.upvote_ratio
        num_comments = submission.num_comments
        subreddit = submission.subreddit.display_name

        comments = await submission.comments()
        top_comments = []
        await comments.replace_more(limit=0)  

        comment_list = await comments.list()
        for comment in comment_list[:10]:  
            comment_author = comment.author.name if hasattr(comment.author, 'name') else "[deleted]"
            comment_text = comment.body
            comment_score = comment.score
            top_comments.append(f"Comment by {comment_author} (Score: {comment_score}):\n{comment_text}\n")

        content = f"Subreddit: r/{subreddit}\n"
        content += f"Title: {title}\n"
        content += f"Author: u/{author}\n"
        content += f"Score: {score} (Upvote ratio: {upvote_ratio})\n"
        content += f"Number of comments: {num_comments}\n\n"

        if selftext:
            content += f"Post content:\n{selftext}\n\n"

        if top_comments:
            content += f"Top comments:\n{''.join(top_comments)}"

        return content
    except Exception as e:
        logging.error(f"Error processing Reddit URL {reddit_url}: {str(e)}")
        return f"Failed to fetch content for {reddit_url}: {str(e)}"
    finally:

        if 'reddit' in locals():
            await reddit.close()

async def process_all_reddit_urls(reddit_links, client_id, client_secret, user_agent, httpx_client):
    """Process multiple Reddit URLs concurrently."""
    if not reddit_links:
        return ""

    tasks = [
        process_reddit_url(reddit_url, client_id, client_secret, user_agent, httpx_client) 
        for reddit_url in reddit_links
    ]
    results = await asyncio.gather(*tasks)

    formatted_content = ""
    for i, (reddit_url, content) in enumerate(zip(reddit_links, results), 1):
        formatted_content += f"reddit url {i}: {reddit_url}\n"
        formatted_content += f"reddit url {i} content: {content}\n\n"

    return formatted_content

async def extract_url_content(url, httpx_client):
    """Extract content from a URL."""
    try:
        response = await httpx_client.get(url, follow_redirects=True, timeout=10.0)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '')

        if 'text/html' in content_type:

            html_content = response.text

            html_content = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html_content, flags=re.DOTALL)

            html_content = re.sub(r'<[^>]*>', ' ', html_content)

            html_content = re.sub(r'\s+', ' ', html_content).strip()

            max_length = 10000  
            if len(html_content) > max_length:
                html_content = html_content[:max_length] + "... [content truncated]"

            return html_content
        elif 'text/plain' in content_type:
            text_content = response.text

            max_length = 10000  
            if len(text_content) > max_length:
                text_content = text_content[:max_length] + "... [content truncated]"
            return text_content
        else:
            return f"Unable to extract content: unsupported content type {content_type}"
    except Exception as e:
        logging.error(f"Error extracting content from URL {url}: {str(e)}")
        return f"Failed to extract content: {str(e)}"

async def process_all_general_urls(general_urls, httpx_client):
    """Process multiple general URLs concurrently."""
    if not general_urls:
        return ""

    tasks = [
        extract_url_content(url, httpx_client) 
        for url in general_urls
    ]

    results = await asyncio.gather(*tasks)

    formatted_content = ""
    for i, (url, content) in enumerate(zip(general_urls, results), 1):
        formatted_content += f"url {i}: {url}\n"
        formatted_content += f"url {i} content: {content}\n\n"

    return formatted_content

cfg = get_config()

if client_id := cfg["client_id"]:
    logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/api/oauth2/authorize?client_id={client_id}&permissions=412317273088&scope=bot\n")

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(cfg["status_message"] or "github.com/jakobdylanc/llmcord")[:128])
discord_client = discord.Client(intents=intents, activity=activity)
tree = app_commands.CommandTree(discord_client)

httpx_client = httpx.AsyncClient()

msg_nodes = {}
last_task_time = 0

@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list = field(default_factory=list)
    youtube_links: list = field(default_factory=list)
    youtube_content: Optional[str] = None
    reddit_links: list = field(default_factory=list)  
    reddit_content: Optional[str] = None
    general_urls: list = field(default_factory=list)  
    url_content: Optional[str] = None  

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

@tree.command(name="model", description="Change the model used by the bot")
@app_commands.describe(
    provider="The provider of the LLM",
    model="The model to use"
)
async def model_command(interaction: discord.Interaction, provider: str, model: str):

    cfg = get_config()

    is_dm = interaction.channel.type == discord.ChannelType.private
    role_ids = set(role.id for role in getattr(interaction.user, "roles", ()))
    channel_ids = set(filter(None, (interaction.channel.id, getattr(interaction.channel, "parent_id", None), getattr(interaction.channel, "category_id", None))))

    permissions = cfg["permissions"]
    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = allow_all_users or interaction.user.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or interaction.user.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = cfg["allow_dms"] if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        await interaction.response.send_message("You don't have permission to change the model.", ephemeral=True)
        return

    if provider not in cfg["providers"]:
        await interaction.response.send_message(f"Provider '{provider}' not found in config.", ephemeral=True)
        return

    if provider not in PROVIDER_MODELS:
        await interaction.response.send_message(f"Provider '{provider}' is not supported for model selection.", ephemeral=True)
        return

    if model not in PROVIDER_MODELS[provider]:
        await interaction.response.send_message(f"Model '{model}' is not available for provider '{provider}'.", ephemeral=True)
        return

    ryaml = YAML()
    ryaml.preserve_quotes = True

    with open("config.yaml", "r") as file:
        config = ryaml.load(file)

    config["model"] = f"{provider}/{model}"

    with open("config.yaml", "w") as file:
        ryaml.dump(config, file)

    await interaction.response.send_message(f"Model updated to {provider}/{model}", ephemeral=True)

@tree.command()
@app_commands.describe(provider="The provider to check models for")
async def models(interaction: discord.Interaction, provider: str):
    """List available models for a provider"""
    cfg = get_config()

    if provider not in PROVIDER_MODELS:
        await interaction.response.send_message(f"No predefined models found for provider '{provider}'.")
        return

    if provider not in cfg["providers"]:
        await interaction.response.send_message(f"Provider '{provider}' not found in your config.")
        return

    models_list = PROVIDER_MODELS[provider]
    await interaction.response.send_message(f"Available models for {provider}:\n" + "\n".join([f"- {model}" for model in models_list]))

async def provider_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    cfg = get_config()

    providers = [provider for provider in cfg["providers"].keys() if provider in PROVIDER_MODELS]
    return [
        app_commands.Choice(name=provider, value=provider)
        for provider in providers if current.lower() in provider.lower()
    ][:25]  

async def model_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:

    options = interaction.data.get("options", [])
    provider = None
    for option in options:
        if option["name"] == "provider":
            provider = option["value"]
            break

    if not provider or provider not in PROVIDER_MODELS:
        return []

    models = PROVIDER_MODELS.get(provider, [])
    return [
        app_commands.Choice(name=model, value=model)
        for model in models if current.lower() in model.lower()
    ][:25]  

model_command.autocomplete("provider")(provider_autocomplete)
model_command.autocomplete("model")(model_autocomplete)
models.autocomplete("provider")(provider_autocomplete)

@discord_client.event
async def on_ready():
    await tree.sync()
    logging.info(f"Logged in as {discord_client.user} (ID: {discord_client.user.id})")

@discord_client.event
async def on_message(new_msg):
    global msg_nodes, last_task_time

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if ((not is_dm and discord_client.user not in new_msg.mentions and "at ai" not in new_msg.content.lower()) or new_msg.author.bot):
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    cfg = get_config()

    allow_dms = cfg["allow_dms"]
    permissions = cfg["permissions"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    provider, model = cfg["model"].split("/", 1)
    is_gemini = provider.lower() in GEMINI_PROVIDERS

    youtube_config = cfg.get("youtube", {})
    youtube_api_key = youtube_config.get("api_key")
    enable_youtube_global = youtube_config.get("enable", True)

    reddit_config = cfg.get("reddit", {})
    reddit_client_id = reddit_config.get("client_id")
    reddit_client_secret = reddit_config.get("client_secret")
    reddit_user_agent = reddit_config.get("user_agent", "llmcord-bot/1.0")
    enable_reddit = reddit_config.get("enable", True) and asyncpraw and reddit_client_id and reddit_client_secret

    if is_gemini:
        provider_config = cfg["providers"][provider]
        enable_grounding = provider_config.get("enable_grounding", False)
        enable_youtube = provider_config.get("enable_youtube", enable_youtube_global)
    else:
        enable_youtube = enable_youtube_global and youtube_api_key is not None

    accept_images = any(x in model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(x in provider.lower() for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = cfg["max_text"]
    max_images = cfg["max_images"] if accept_images else 0
    max_messages = cfg["max_messages"]

    use_plain_responses = cfg["use_plain_responses"]
    max_message_length = 2000 if use_plain_responses else (4096 - len(STREAMING_INDICATOR))

    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = curr_msg.content
                if discord_client.user.mention in cleaned_content:
                    cleaned_content = cleaned_content.removeprefix(discord_client.user.mention).lstrip()
                elif cleaned_content.lower().startswith("at ai"):
                    cleaned_content = cleaned_content[5:].lstrip()  

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(type) for type in ("text", "image"))]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                youtube_links = extract_youtube_links(curr_node.text)
                curr_node.youtube_links = [(video_id, full_url) for video_id, full_url in youtube_links]

                reddit_links = extract_reddit_links(curr_node.text)
                curr_node.reddit_links = reddit_links

                general_urls = extract_general_urls(curr_node.text)
                curr_node.general_urls = general_urls

                if curr_msg.author != discord_client.user and enable_youtube and youtube_api_key and curr_node.youtube_links:
                    try:
                        curr_node.youtube_content = await process_all_youtube_urls(
                            curr_node.youtube_links, 
                            youtube_api_key, 
                            httpx_client
                        )
                        logging.info(f"Processed {len(curr_node.youtube_links)} YouTube links for message {curr_msg.id}")
                    except Exception as e:
                        logging.error(f"Error processing YouTube content: {str(e)}")
                        curr_node.youtube_content = ""

                if curr_msg.author != discord_client.user and enable_reddit and curr_node.reddit_links:
                    try:
                        curr_node.reddit_content = await process_all_reddit_urls(
                            curr_node.reddit_links,
                            reddit_client_id,
                            reddit_client_secret,
                            reddit_user_agent,
                            httpx_client
                        )
                        logging.info(f"Processed {len(curr_node.reddit_links)} Reddit links for message {curr_msg.id}")
                    except Exception as e:
                        logging.error(f"Error processing Reddit content: {str(e)}")
                        curr_node.reddit_content = ""

                if curr_msg.author != discord_client.user and curr_node.general_urls:
                    try:
                        curr_node.url_content = await process_all_general_urls(
                            curr_node.general_urls, 
                            httpx_client
                        )
                        logging.info(f"Processed {len(curr_node.general_urls)} general URLs for message {curr_msg.id}")
                    except Exception as e:
                        logging.error(f"Error processing general URL content: {str(e)}")
                        curr_node.url_content = ""

                curr_node.role = "assistant" if curr_msg.author == discord_client.user else "user"

                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    if (
                        curr_msg.reference == None
                        and discord_client.user.mention not in curr_msg.content
                        and "at ai" not in curr_msg.content.lower()
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_client.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
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

            has_youtube_gemini = is_gemini and enable_youtube and curr_node.youtube_links and curr_node.role == "user"
            has_youtube_content = not is_gemini and curr_node.youtube_content and curr_node.role == "user"
            has_reddit_content = curr_node.reddit_content and curr_node.role == "user"
            has_url_content = curr_node.url_content and curr_node.role == "user"  

            if has_youtube_gemini and len(curr_node.youtube_links) > 0:
                video_id, full_url = curr_node.youtube_links[0]

                text_without_url = curr_node.text.replace(full_url, "").strip()
                if not text_without_url:
                    text_without_url = "Explain this video"

                message = {
                    "role": curr_node.role,
                    "youtube_url": full_url,
                    "youtube_text": text_without_url
                }

                if accept_usernames and curr_node.user_id is not None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)
                logging.info(f"Added YouTube link {full_url} to the message (Gemini handling)")

            elif has_youtube_content or has_reddit_content or has_url_content:
                content = curr_node.text[:max_text]

                if has_youtube_content:
                    content = f"{content}\n\n{curr_node.youtube_content}"

                if has_reddit_content:
                    content = f"{content}\n\n{curr_node.reddit_content}"

                if has_url_content:
                    content = f"{content}\n\n{curr_node.url_content}"

                if content != "":
                    message = dict(content=content, role=curr_node.role)
                    if accept_usernames and curr_node.user_id is not None:
                        message["name"] = str(curr_node.user_id)

                    messages.append(message)

                    if has_youtube_content:
                        logging.info(f"Added message with YouTube content (non-Gemini handling)")

                    if has_reddit_content:
                        logging.info(f"Added message with Reddit content")

                    if has_url_content:
                        logging.info(f"Added message with general URL content")

            elif curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]

                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id != None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)
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

    if system_prompt := cfg["system_prompt"]:
        system_prompt_extras = [f"Today's date: {dt.now().strftime('%B %d %Y')}."]
        if accept_usernames:
            system_prompt_extras.append("User's names are their Discord IDs and should be typed as '<@ID>'.")

        full_system_prompt = "\n".join([system_prompt] + system_prompt_extras)
        messages.append(dict(role="system", content=full_system_prompt))

    curr_content = finish_reason = edit_task = None
    response_msgs = []
    response_contents = []

    embed = discord.Embed()
    for warning in sorted(user_warnings):
        embed.add_field(name=warning, value="", inline=False)

    model_info = f"{provider}/{model}"
    embed.set_footer(text=model_info)

    try:
        async with new_msg.channel.typing():
            if is_gemini:
                gemini_messages = []
                system_instruction = None

                for msg in messages[::-1]:
                    role = msg["role"]

                    if "youtube_url" in msg:
                        youtube_url = msg["youtube_url"]
                        youtube_text = msg["youtube_text"]

                        gemini_role = "user" if role == "user" else "model"

                        parts = [
                            types.Part.from_uri(
                                file_uri=youtube_url,
                                mime_type="video/*",
                            ),
                            types.Part.from_text(text=youtube_text),
                        ]

                        gemini_messages.append(types.Content(role=gemini_role, parts=parts))
                        continue

                    if role == "system":
                        system_instruction = msg["content"]
                        continue

                    gemini_role = "user" if role == "user" else "model"
                    content = msg["content"]

                    if isinstance(content, list):
                        parts = []
                        for item in content:
                            if item["type"] == "text":
                                parts.append(types.Part.from_text(text=item["text"]))
                            elif item["type"] == "image_url":
                                image_url = item["image_url"]["url"]
                                if image_url.startswith("data:"):
                                    mime_type = image_url.split(';')[0].split(':')[1]
                                    base64_data = image_url.split(',')[1]
                                    image_bytes = base64.b64decode(base64_data)
                                    parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

                        if parts:
                            gemini_messages.append(types.Content(role=gemini_role, parts=parts))
                    else:
                        gemini_messages.append(types.Content(role=gemini_role, parts=[types.Part.from_text(text=content)]))

                extra_params = cfg["extra_api_parameters"].copy()
                temperature = extra_params.pop("temperature", 1.0)
                max_output_tokens = extra_params.pop("max_tokens", 4096)

                safety_settings = [
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE
                    )
                ]

                generate_content_config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    safety_settings=safety_settings,
                    **extra_params
                )

                if system_instruction:
                    generate_content_config.system_instruction = system_instruction

                tools = []
                if enable_grounding:
                    google_search_tool = types.Tool(google_search=types.GoogleSearch())
                    tools.append(google_search_tool)

                if tools:
                    generate_content_config.tools = tools

                provider_config = cfg["providers"][provider]
                success = await safe_gemini_stream_processor(
                    new_msg=new_msg,
                    provider=provider, 
                    provider_config=provider_config,
                    model=model,
                    gemini_messages=gemini_messages,
                    generate_content_config=generate_content_config,
                    embed=embed,
                    response_msgs=response_msgs,
                    response_contents=response_contents,
                    edit_task=edit_task,
                    use_plain_responses=use_plain_responses,
                    max_message_length=max_message_length,
                    enable_grounding=enable_grounding
                )

                if not success:
                    return  

            else:

                provider_config = cfg["providers"][provider]
                base_url = provider_config["base_url"]

                success = await safe_openai_stream_processor(
                    new_msg=new_msg,
                    provider=provider,
                    provider_config=provider_config,
                    base_url=base_url,
                    model=model,
                    messages=messages[::-1],
                    extra_params=cfg["extra_api_parameters"],
                    embed=embed,
                    response_msgs=response_msgs,
                    response_contents=response_contents,
                    edit_task=edit_task,
                    use_plain_responses=use_plain_responses,
                    max_message_length=max_message_length
                )

                if not success:
                    return  

            if use_plain_responses:
                for content in response_contents:
                    reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                    response_msg = await reply_to_msg.reply(content=content, suppress_embeds=True)
                    response_msgs.append(response_msg)

                    msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                    await msg_nodes[response_msg.id].lock.acquire()

    except Exception as e:
        logging.exception(f"Error while generating response: {str(e)}")

        try:
            error_embed = discord.Embed(
                title="Error",
                description=f"An unexpected error occurred while processing your request. Please try again later.",
                color=discord.Color.red()
            )
            await new_msg.reply(embed=error_embed)
        except:
            logging.exception("Failed to send error message")

    for response_msg in response_msgs:
        if response_msg.id in msg_nodes:
            msg_nodes[response_msg.id].text = "".join(response_contents)
            msg_nodes[response_msg.id].lock.release()

    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)

async def main():
    await discord_client.start(cfg["bot_token"])

asyncio.run(main())