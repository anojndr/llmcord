"""Configuration loading and constants for llmcord.

This package exposes the split configuration modules as a single interface
to maintain backward compatibility.
"""

from llmcord.config.constants import (
    EDIT_DELAY_SECONDS,
    EMBED_COLOR_COMPLETE,
    EMBED_COLOR_INCOMPLETE,
    MAX_MESSAGE_NODES,
    PROCESSING_MESSAGE,
    PROVIDERS_SUPPORTING_USERNAMES,
    STREAMING_INDICATOR,
    VISION_MODEL_TAGS,
)
from llmcord.config.http import (
    BROWSER_HEADERS,
    get_or_create_httpx_client,
)
from llmcord.config.manager import (
    CONFIG_CACHE_TTL,
    ConfigFileEmptyError,
    ConfigFileNotFoundError,
    _CONFIG_STATE,
    _ConfigCacheState,
    _resolve_config_path,
    clear_config_cache,
    ensure_list,
    get_config,
)
from llmcord.config.models import is_gemini_model

__all__ = [
    "BROWSER_HEADERS",
    "CONFIG_CACHE_TTL",
    "EDIT_DELAY_SECONDS",
    "EMBED_COLOR_COMPLETE",
    "EMBED_COLOR_INCOMPLETE",
    "MAX_MESSAGE_NODES",
    "PROCESSING_MESSAGE",
    "PROVIDERS_SUPPORTING_USERNAMES",
    "STREAMING_INDICATOR",
    "VISION_MODEL_TAGS",
    "ConfigFileEmptyError",
    "ConfigFileNotFoundError",
    "_CONFIG_STATE",
    "_ConfigCacheState",
    "_resolve_config_path",
    "clear_config_cache",
    "ensure_list",
    "get_config",
    "get_or_create_httpx_client",
    "is_gemini_model",
]
