"""
Configuration loading and constants for llmcord.
"""
import os
from typing import Any

import yaml
import discord


# Model and provider constants
VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

# Discord embed colors
EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

# Streaming and editing constants
STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

# Message node limits
MAX_MESSAGE_NODES = 500


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(filename, encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        with open(os.path.join("/etc/secrets", filename), encoding="utf-8") as file:
            return yaml.safe_load(file)
