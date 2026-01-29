"""Configuration utilities and constants for llmcord."""

from .app_constants import (
    BROWSER_HEADERS,
    EDIT_DELAY_SECONDS,
    EMBED_COLOR_COMPLETE,
    EMBED_COLOR_INCOMPLETE,
    MAX_MESSAGE_NODES,
    PROCESSING_MESSAGE,
    PROVIDERS_SUPPORTING_USERNAMES,
    STREAMING_INDICATOR,
    VISION_MODEL_TAGS,
)
from .http import get_or_create_httpx_client
from .loader import (
    CONFIG_CACHE_TTL,
    ConfigFileEmptyError,
    ConfigFileNotFoundError,
    clear_config_cache,
    ensure_list,
    get_bot_profile,
    get_bot_token,
    get_config,
    get_health_check_port,
)
from .model_utils import is_gemini_model

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
    "clear_config_cache",
    "ensure_list",
    "get_bot_profile",
    "get_bot_token",
    "get_config",
    "get_health_check_port",
    "get_or_create_httpx_client",
    "is_gemini_model",
]
