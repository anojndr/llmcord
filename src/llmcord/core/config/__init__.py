"""Configuration loading and constants for llmcord.

This package exposes the split configuration modules as a single interface
to maintain backward compatibility.
"""

from llmcord.core.config.constants import (
    EDIT_DELAY_SECONDS,
    EMBED_COLOR_COMPLETE,
    EMBED_COLOR_INCOMPLETE,
    EMBED_FIELD_NAME_LIMIT,
    MAX_MESSAGE_NODES,
    PROCESSING_MESSAGE,
    PROVIDERS_SUPPORTING_USERNAMES,
    STREAMING_INDICATOR,
    VISION_MODEL_TAGS,
)
from llmcord.core.config.http import (
    BROWSER_HEADERS,
    DEFAULT_USER_AGENT,
    HttpxClientOptions,
    get_or_create_httpx_client,
)
from llmcord.core.config.manager import (
    _CONFIG_STATE,
    CONFIG_CACHE_TTL,
    ConfigFileEmptyError,
    ConfigFileNotFoundError,
    ProfileConfigError,
    _ConfigCacheState,
    _resolve_config_path,
    clear_config_cache,
    ensure_list,
    get_config,
)
from llmcord.core.config.utils import is_gemini_model

__all__ = [
    "BROWSER_HEADERS",
    "DEFAULT_USER_AGENT",
    "CONFIG_CACHE_TTL",
    "EDIT_DELAY_SECONDS",
    "EMBED_COLOR_COMPLETE",
    "EMBED_COLOR_INCOMPLETE",
    "EMBED_FIELD_NAME_LIMIT",
    "MAX_MESSAGE_NODES",
    "PROCESSING_MESSAGE",
    "PROVIDERS_SUPPORTING_USERNAMES",
    "STREAMING_INDICATOR",
    "VISION_MODEL_TAGS",
    "_CONFIG_STATE",
    "ConfigFileEmptyError",
    "ConfigFileNotFoundError",
    "HttpxClientOptions",
    "ProfileConfigError",
    "_ConfigCacheState",
    "_resolve_config_path",
    "clear_config_cache",
    "ensure_list",
    "get_config",
    "get_or_create_httpx_client",
    "is_gemini_model",
]
