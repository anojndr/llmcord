"""Constant definitions for llmcord."""

import discord

# Model and provider constants
VISION_MODEL_TAGS = (
    "claude",
    "gemini",
    "gemma",
    "gpt-4",
    "gpt-5",
    "grok-4",
    "llama",
    "llava",
    "mistral",
    "o3",
    "o4",
    "openrouter/free",
    "vision",
    "vl",
)

PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

# Discord embed colors
EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

# Streaming and editing constants
STREAMING_INDICATOR = " ⚪"
PROCESSING_MESSAGE = "⏳ Working on your request (1/5): Starting up..."
EDIT_DELAY_SECONDS = 1

# Message node limits
MAX_MESSAGE_NODES = 500

# Discord limit constants
EMBED_FIELD_NAME_LIMIT = 256
