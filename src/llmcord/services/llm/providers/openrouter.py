"""OpenRouter provider implementation."""

from __future__ import annotations

import os
from typing import Any

OPENROUTER_SITE_URL_ENV = "OR_SITE_URL"
OPENROUTER_APP_NAME_ENV = "OR_APP_NAME"


def _build_openrouter_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if site_url := os.getenv(OPENROUTER_SITE_URL_ENV):
        headers["HTTP-Referer"] = site_url
    if app_name := os.getenv(OPENROUTER_APP_NAME_ENV):
        headers["X-Title"] = app_name
    return headers


def configure_openrouter_kwargs(
    kwargs: dict[str, Any],
    extra_headers: dict[str, str] | None = None,
) -> None:
    """Configure OpenRouter-specific kwargs."""
    openrouter_headers = _build_openrouter_headers()
    if extra_headers:
        openrouter_headers = {**openrouter_headers, **extra_headers}

    if openrouter_headers:
        kwargs["extra_headers"] = openrouter_headers
