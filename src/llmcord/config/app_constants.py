"""Configuration constants for llmcord."""

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
    "vision",
    "vl",
)

PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

# Discord embed colors
EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

# Streaming and editing constants
STREAMING_INDICATOR = " ⚪"
PROCESSING_MESSAGE = "⏳ Processing request..."
EDIT_DELAY_SECONDS = 1

# Message node limits
MAX_MESSAGE_NODES = 500

# Browser-like headers for web scraping/HTTP requests
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
        "image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}
