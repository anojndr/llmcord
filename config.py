"""
Configuration loading and constants for llmcord.
"""
import os
import time
from functools import lru_cache
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
PROCESSING_MESSAGE = "⏳ Processing request..."
EDIT_DELAY_SECONDS = 1

# Message node limits
MAX_MESSAGE_NODES = 500

# Config caching - reload every 30 seconds at most
_config_cache: dict[str, Any] = {}
_config_mtime: float = 0
_config_check_time: float = 0
CONFIG_CACHE_TTL = 5  # Check file modification time every 5 seconds


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    """
    Load configuration from YAML file with caching.
    Only reloads if file has been modified (checked every CONFIG_CACHE_TTL seconds).
    """
    global _config_cache, _config_mtime, _config_check_time
    
    current_time = time.time()
    
    # Only check file mtime periodically to avoid stat() on every call
    if current_time - _config_check_time > CONFIG_CACHE_TTL or not _config_cache:
        _config_check_time = current_time
        
        try:
            filepath = filename
            if not os.path.exists(filepath):
                filepath = os.path.join("/etc/secrets", filename)
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Config file not found: {filename}")
            
            file_mtime = os.path.getmtime(filepath)
            
            # Only reload if file was modified
            if file_mtime != _config_mtime or not _config_cache:
                _config_mtime = file_mtime
                with open(filepath, encoding="utf-8") as file:
                    loaded_config = yaml.safe_load(file)
                    # Handle empty/corrupted YAML that returns None
                    if loaded_config is None:
                        raise ValueError(f"Config file is empty or corrupted: {filepath}")
                    _config_cache = loaded_config
        except FileNotFoundError as e:
            # Fallback for edge cases
            fallback_path = os.path.join("/etc/secrets", filename)
            try:
                with open(fallback_path, encoding="utf-8") as file:
                    loaded_config = yaml.safe_load(file)
                    if loaded_config is None:
                        raise ValueError(f"Config file is empty or corrupted: {fallback_path}")
                    _config_cache = loaded_config
            except FileNotFoundError:
                raise FileNotFoundError(f"Config file '{filename}' not found in current directory or /etc/secrets/") from e
    
    return _config_cache


def clear_config_cache() -> None:
    """Clear the config cache to force a reload on next get_config() call."""
    global _config_cache, _config_mtime, _config_check_time
    _config_cache = {}
    _config_mtime = 0
    _config_check_time = 0
