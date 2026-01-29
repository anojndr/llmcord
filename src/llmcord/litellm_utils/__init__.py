"""Shared LiteLLM utilities for consistent provider configuration.

This package centralizes all provider-specific logic so that both the main model
handler and search decider use the same configuration, headers, and token handling.
"""

from .builder import build_litellm_model_name, prepare_litellm_kwargs
from .github_copilot import GITHUB_COPILOT_HEADERS, configure_github_copilot_token
from .options import LiteLLMOptions

__all__ = [
    "GITHUB_COPILOT_HEADERS",
    "LiteLLMOptions",
    "build_litellm_model_name",
    "configure_github_copilot_token",
    "prepare_litellm_kwargs",
]
