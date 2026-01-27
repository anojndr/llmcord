"""Shared LiteLLM utilities for consistent provider configuration.

This module centralizes all provider-specific logic so that both the main model
handler and search decider use the same configuration, headers, and token handling.
"""
import json
import logging
import os
import time
from typing import Any

import requests

from config import is_gemini_model


def build_litellm_model_name(provider: str, model: str) -> str:
    """Build the LiteLLM model name with proper provider prefix.
    
    Args:
        provider: Provider name (e.g., "gemini", "openai", "github_copilot")
        model: Model name
    
    Returns:
        LiteLLM-compatible model string (e.g., "gemini/gemini-3-flash-preview")

    """
    if provider == "gemini":
        return f"gemini/{model}"
    if provider == "github_copilot":
        return f"github_copilot/{model}"
    if provider == "mistral":
        return f"mistral/{model}"
    # For OpenAI-compatible providers, just use the model name
    # LiteLLM will use base_url if provided
    return model


def configure_github_copilot_token(access_token: str) -> None:
    """Configure a GitHub Copilot access token for LiteLLM.
    
    LiteLLM's GitHub Copilot provider expects tokens in a config file.
    This function writes the access token AND exchanges it for an API key.
    
    Args:
        access_token: GitHub access token (ghu_...)

    """
    # Get the token directory from env or use default
    token_dir = os.environ.get(
        "GITHUB_COPILOT_TOKEN_DIR",
        os.path.expanduser("~/.config/litellm/github_copilot"),
    )

    # Create directory if it doesn't exist
    os.makedirs(token_dir, exist_ok=True)

    # Write access token file (just the raw token)
    access_token_file = os.path.join(
        token_dir,
        os.environ.get("GITHUB_COPILOT_ACCESS_TOKEN_FILE", "access-token"),
    )
    with open(access_token_file, "w") as f:
        f.write(access_token)

    logging.debug(f"Configured GitHub Copilot access token at {access_token_file}")

    # Also exchange the access token for a Copilot API key
    # This replicates what LiteLLM's authenticator does
    try:
        headers = {
            "Authorization": f"token {access_token}",
            "Accept": "application/json",
            "Editor-Version": "vscode/1.85.1",
            "Editor-Plugin-Version": "copilot-chat/0.22.0",
        }

        response = requests.get(
            "https://api.github.com/copilot_internal/v2/token",
            headers=headers,
            timeout=30,
        )

        if response.status_code == 200:
            token_data = response.json()

            # Write the API key file in the format LiteLLM expects
            api_key_file = os.path.join(token_dir, "api-key.json")
            api_key_data = {
                "token": token_data.get("token"),
                "expires_at": token_data.get("expires_at", int(time.time()) + 3600),
                "endpoints": token_data.get("endpoints", {}),
            }

            with open(api_key_file, "w") as f:
                json.dump(api_key_data, f)

            logging.debug(f"Exchanged GitHub access token for Copilot API key at {api_key_file}")
        else:
            logging.warning(f"Failed to exchange GitHub token: {response.status_code} - {response.text[:200]}")
    except Exception as e:
        logging.warning(f"Error exchanging GitHub token for Copilot API key: {e}")


# GitHub Copilot required headers - used by all Copilot API calls
GITHUB_COPILOT_HEADERS = {
    "Editor-Version": "vscode/1.85.1",
    "Editor-Plugin-Version": "copilot-chat/0.22.0",
    "Copilot-Integration-Id": "vscode-chat",
    "Copilot-Vision-Request": "true",  # Required for vision/image requests
}


def prepare_litellm_kwargs(
    provider: str,
    model: str,
    messages: list,
    api_key: str,
    *,
    base_url: str | None = None,
    extra_headers: dict | None = None,
    stream: bool = False,
    model_parameters: dict | None = None,
    temperature: float | None = None,
    enable_grounding: bool = False,
) -> dict[str, Any]:
    """Prepare kwargs for LiteLLM acompletion() with all provider-specific configuration.
    
    This is the main entry point for creating consistent LiteLLM calls across
    both the main model handler and search decider.
    
    Args:
        provider: Provider name (e.g., "gemini", "openai", "github_copilot")
        model: Model name (actual model, after any aliasing)
        messages: List of message dicts
        api_key: API key to use
        base_url: Optional base URL for OpenAI-compatible providers
        extra_headers: Optional extra headers from config
        stream: Whether to enable streaming
        model_parameters: Optional model parameters from config
        temperature: Optional temperature override
        enable_grounding: Whether to enable Gemini grounding/search tools
    
    Returns:
        Dict of kwargs ready to pass to litellm.acompletion()

    """
    # Build the model name with proper prefix
    litellm_model = build_litellm_model_name(provider, model)

    # Base kwargs
    kwargs: dict[str, Any] = {
        "model": litellm_model,
        "messages": messages,
        "api_key": api_key,
    }

    if stream:
        kwargs["stream"] = True

    if temperature is not None:
        kwargs["temperature"] = temperature

    # Add base_url for OpenAI-compatible providers (not Gemini or GitHub Copilot)
    if base_url and provider not in ("gemini", "github_copilot"):
        kwargs["base_url"] = base_url

    # Provider-specific configuration
    # Only apply Gemini-specific config to actual Gemini models (not Gemma)
    if provider == "gemini" and is_gemini_model(model):
        _configure_gemini_kwargs(kwargs, model, model_parameters, enable_grounding)
    elif provider == "github_copilot":
        _configure_github_copilot_kwargs(kwargs, api_key, extra_headers)

    # Merge extra headers if provided (after provider-specific headers)
    if extra_headers and "extra_headers" not in kwargs:
        kwargs["extra_headers"] = extra_headers
    elif extra_headers and "extra_headers" in kwargs:
        kwargs["extra_headers"] = {**kwargs["extra_headers"], **extra_headers}

    return kwargs


def _configure_gemini_kwargs(
    kwargs: dict[str, Any],
    model: str,
    model_parameters: dict | None = None,
    enable_grounding: bool = False,
) -> None:
    """Configure Gemini-specific kwargs."""
    is_gemini_3 = "gemini-3" in model
    is_preview = "preview" in model

    # Add thinking config for Gemini 3 models
    thinking_level = model_parameters.get("thinking_level") if model_parameters else None

    if not thinking_level:
        if "gemini-3-flash" in model:
            thinking_level = "MINIMAL"
        elif "gemini-3-pro" in model:
            thinking_level = "LOW"

    if thinking_level:
        # Map thinking levels to LiteLLM reasoning_effort
        thinking_map = {
            "MINIMAL": "minimal",
            "LOW": "low",
            "MEDIUM": "medium",
            "HIGH": "high",
        }
        kwargs["reasoning_effort"] = thinking_map.get(thinking_level, thinking_level.lower())

    # Add Google Search and URL Context tools for non-preview models when grounding is enabled
    if enable_grounding and not is_preview:
        kwargs["tools"] = [{"googleSearch": {}}, {"urlContext": {}}]

    # Set temperature for non-Gemini 3 models
    if model_parameters and not is_gemini_3:
        if "temperature" in model_parameters and "temperature" not in kwargs:
            kwargs["temperature"] = model_parameters["temperature"]


def _configure_github_copilot_kwargs(
    kwargs: dict[str, Any],
    api_key: str,
    extra_headers: dict | None = None,
) -> None:
    """Configure GitHub Copilot-specific kwargs."""
    # Configure the token file before making the call
    configure_github_copilot_token(api_key)

    # Add required headers
    copilot_headers = GITHUB_COPILOT_HEADERS.copy()
    if extra_headers:
        copilot_headers = {**copilot_headers, **extra_headers}

    kwargs["extra_headers"] = copilot_headers

