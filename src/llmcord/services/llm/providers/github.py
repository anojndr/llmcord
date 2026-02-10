"""GitHub Copilot provider implementation."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests  # type: ignore[import-untyped]

HTTP_STATUS_OK = 200
logger = logging.getLogger(__name__)

# GitHub Copilot required headers - used by all Copilot API calls
GITHUB_COPILOT_HEADERS = {
    "Editor-Version": "vscode/1.85.1",
    "Editor-Plugin-Version": "copilot-chat/0.22.0",
    "Copilot-Integration-Id": "vscode-chat",
    "Copilot-Vision-Request": "true",  # Required for vision/image requests
}


def configure_github_copilot_token(access_token: str) -> None:
    """Configure a GitHub Copilot access token for LiteLLM.

    LiteLLM's GitHub Copilot provider expects tokens in a config file.
    This function writes the access token AND exchanges it for an API key.

    Args:
        access_token: GitHub access token (ghu_...)

    """
    # Get the token directory from env or use default
    token_dir = Path(
        os.environ.get(
            "GITHUB_COPILOT_TOKEN_DIR",
            "~/.config/litellm/github_copilot",
        ),
    ).expanduser()

    # Create directory if it doesn't exist
    token_dir.mkdir(parents=True, exist_ok=True)

    # Write access token file (just the raw token)
    access_token_file = token_dir / os.environ.get(
        "GITHUB_COPILOT_ACCESS_TOKEN_FILE",
        "access-token",
    )
    access_token_file.write_text(access_token, encoding="utf-8")

    logger.debug(
        "Configured GitHub Copilot access token at %s",
        access_token_file,
    )

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

        if response.status_code == HTTP_STATUS_OK:
            token_data = response.json()

            # Write the API key file in the format LiteLLM expects
            api_key_file = token_dir / "api-key.json"
            expires_at = token_data.get(
                "expires_at",
                int(time.time()) + 3600,
            )
            api_key_data = {
                "token": token_data.get("token"),
                "expires_at": expires_at,
                "endpoints": token_data.get("endpoints", {}),
            }

            with api_key_file.open("w", encoding="utf-8") as file_handle:
                json.dump(api_key_data, file_handle)

            logger.debug(
                "Exchanged GitHub access token for Copilot API key at %s",
                api_key_file,
            )
        else:
            logger.warning(
                "Failed to exchange GitHub token: %s - %s",
                response.status_code,
                response.text[:200],
            )
    except (requests.RequestException, ValueError, OSError) as exc:
        logger.warning(
            "Error exchanging GitHub token for Copilot API key: %s",
            exc,
        )


def configure_github_copilot_kwargs(
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
