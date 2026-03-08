"""Compatibility re-exports for processing helpers."""

from llmcord.core.config import get_config
from llmcord.discord.processing import (
    _handle_retry_request,
    _process_user_message,
    preload_runtime_dependencies,
)
from llmcord.logic.pipeline import process_message

config = get_config()

__all__ = [
    "_handle_retry_request",
    "_process_user_message",
    "config",
    "preload_runtime_dependencies",
    "process_message",
]
