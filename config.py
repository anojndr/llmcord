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


def is_gemini_model(model: str) -> bool:
    """
    Check if a model is an actual Gemini model (not Gemma or other models on the Gemini provider).
    
    Gemini models have special capabilities like native PDF handling, audio/video support,
    and grounding tools that Gemma models don't have even though they're served via the
    same Gemini provider.
    
    Args:
        model: Model name (e.g., "gemini-3-flash-preview", "gemma-3-27b-it")
        
    Returns:
        True if this is a genuine Gemini model, False for Gemma and other models
    """
    model_lower = model.lower()
    # Gemma models contain "gemma" in their name
    if "gemma" in model_lower:
        return False
    # Gemini models contain "gemini" in their name
    return "gemini" in model_lower
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
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def get_or_create_httpx_client(
    client_holder: list,
    *,
    timeout: float = 30.0,
    connect_timeout: float = 10.0,
    max_connections: int = 20,
    max_keepalive: int = 10,
    headers: dict = None,
    proxy_url: str = None,
    follow_redirects: bool = True,
):
    """
    Get or create a shared httpx.AsyncClient with lazy initialization.
    
    This factory function provides a consistent pattern for creating httpx clients
    across the codebase, avoiding duplication of client configuration.
    
    Args:
        client_holder: A mutable list containing the client instance (or empty).
                      Used as a container so the client can be stored globally.
        timeout: Total request timeout in seconds
        connect_timeout: Connection timeout in seconds
        max_connections: Maximum number of connections
        max_keepalive: Maximum number of keepalive connections
        headers: Optional headers dict (merged with BROWSER_HEADERS if provided)
        proxy_url: Optional proxy URL
        follow_redirects: Whether to follow redirects
    
    Returns:
        httpx.AsyncClient instance
    
    Example:
        _my_client = []  # Container for lazy init
        def get_my_client():
            return get_or_create_httpx_client(_my_client, timeout=30.0)
    """
    import httpx
    
    # Check if client exists and is not closed
    if client_holder and client_holder[0] is not None and not client_holder[0].is_closed:
        return client_holder[0]
    
    # Create new client
    final_headers = {**BROWSER_HEADERS, **(headers or {})}
    
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=connect_timeout),
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_keepalive),
        headers=final_headers,
        proxy=proxy_url,
        follow_redirects=follow_redirects,
    )
    
    # Store in holder
    if client_holder:
        if len(client_holder) == 0:
            client_holder.append(client)
        else:
            client_holder[0] = client
    
    return client


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


def ensure_list(value: Any) -> list:
    """
    Convert a value to a list if it isn't one already.
    
    This is commonly needed for API keys which may be configured as either
    a single string or a list of strings.
    
    Args:
        value: A string, list, or None
        
    Returns:
        - If value is None: empty list
        - If value is a string: single-element list containing that string
        - If value is already a list: return as-is
    
    Examples:
        >>> ensure_list("key123")
        ['key123']
        >>> ensure_list(["key1", "key2"])
        ['key1', 'key2']
        >>> ensure_list(None)
        []
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)
