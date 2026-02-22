"""Google Gemini CLI (Cloud Code Assist) provider support."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import os
import re
import secrets
import time
import urllib.parse
import webbrowser
from collections.abc import AsyncIterator
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Event
from typing import Any, cast

import httpx

GOOGLE_GEMINI_CLI_PROVIDER = "google-gemini-cli"
GOOGLE_ANTIGRAVITY_PROVIDER = "google-antigravity"

GEMINI_CLI_CLIENT_ID = base64.b64decode(
    "NjgxMjU1ODA5Mzk1LW9vOGZ0Mm9wcmRybnA5ZTNhcWY2YXYzaG1kaWIxMzVqLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29t",
).decode("utf-8")
GEMINI_CLI_CLIENT_SECRET = base64.b64decode(
    "R09DU1BYLTR1SGdNUG0tMW83U2stZ2VWNkN1NWNsWEZzeGw=",
).decode("utf-8")
GEMINI_CLI_REDIRECT_URI = "http://localhost:8085/oauth2callback"
GEMINI_CLI_CALLBACK_HOST = "127.0.0.1"
GEMINI_CLI_CALLBACK_PORT = 8085
GEMINI_CLI_CALLBACK_PATH = "/oauth2callback"
GEMINI_CLI_SCOPES = (
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
)

ANTIGRAVITY_CLIENT_ID = base64.b64decode(
    "MTA3MTAwNjA2MDU5MS10bWhzc2luMmgyMWxjcmUyMzV2dG9sb2poNGc0MDNlcC5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbQ==",
).decode("utf-8")
ANTIGRAVITY_CLIENT_SECRET = base64.b64decode(
    "R09DU1BYLUs1OEZXUjQ4NkxkTEoxbUxCOHNYQzR6NnFEQWY=",
).decode("utf-8")
ANTIGRAVITY_REDIRECT_URI = "http://localhost:51121/oauth-callback"
ANTIGRAVITY_CALLBACK_HOST = "127.0.0.1"
ANTIGRAVITY_CALLBACK_PORT = 51121
ANTIGRAVITY_CALLBACK_PATH = "/oauth-callback"
ANTIGRAVITY_SCOPES = (
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
)

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com"
DEFAULT_ENDPOINT = "https://cloudcode-pa.googleapis.com"
ANTIGRAVITY_DAILY_ENDPOINT = "https://daily-cloudcode-pa.sandbox.googleapis.com"
ANTIGRAVITY_DEFAULT_PROJECT_ID = "rising-fact-p41fc"
TOKEN_EXPIRY_BUFFER_MS = 5 * 60 * 1000
TIER_FREE = "free-tier"
TIER_LEGACY = "legacy-tier"

GEMINI_CLI_HEADERS = {
    "User-Agent": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    "X-Goog-Api-Client": "gl-node/22.17.0",
    "Client-Metadata": json.dumps(
        {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        },
    ),
}

DEFAULT_ANTIGRAVITY_VERSION = "1.15.8"
ANTIGRAVITY_SYSTEM_INSTRUCTION = (
    "You are Antigravity, a powerful agentic AI coding assistant designed by the "
    "Google Deepmind team working on Advanced Agentic Coding."
    "You are pair programming with a USER to solve their coding task. The task may "
    "require creating a new codebase, modifying or debugging an existing codebase, "
    "or simply answering a question."
    "**Absolute paths only**"
    "**Proactiveness**"
)

DEFAULT_SAFETY_SETTINGS: list[dict[str, str]] = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "OFF",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "OFF",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "OFF",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "OFF",
    },
    {
        "category": "HARM_CATEGORY_CIVIC_INTEGRITY",
        "threshold": "OFF",
    },
]

IMAGE_TOOL_NAME = "generate_image"
IMAGE_TOOL_DEFAULT_MODEL = "gemini-3-pro-image"
IMAGE_TOOL_DEFAULT_ASPECT_RATIO = "1:1"
IMAGE_TOOL_READ_TIMEOUT_SECONDS = 180
IMAGE_TOOL_MAX_ATTEMPTS_PER_ENDPOINT = 2
IMAGE_TOOL_RETRY_BASE_DELAY_SECONDS = 1.0
IMAGE_TOOL_ASPECT_RATIOS = (
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
)
IMAGE_SYSTEM_INSTRUCTION = (
    "You are an AI image generator. Generate images based on user descriptions. "
    "Focus on creating high-quality, visually appealing images that match the "
    "user's request."
)
IMAGE_TOOL_DECLARATION: dict[str, object] = {
    "name": IMAGE_TOOL_NAME,
    "description": (
        "Generate an image via Google Antigravity image models and return the "
        "result as inline image data."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Image description.",
            },
            "model": {
                "type": "string",
                "description": ("Image model id (e.g., gemini-3-pro-image, imagen-3)."),
            },
            "aspectRatio": {
                "type": "string",
                "enum": list(IMAGE_TOOL_ASPECT_RATIOS),
                "description": "Aspect ratio for generated image.",
            },
        },
        "required": ["prompt"],
    },
}


_RETRYABLE_ERROR_PATTERN = re.compile(
    r"resource.?exhausted|rate.?limit|overloaded|service.?unavailable|other.?side.?closed",
    re.IGNORECASE,
)


def _is_retryable_error(status_code: int, error_text: str) -> bool:
    """Check if an error is retryable (rate limit, server error, etc.)."""
    if status_code in {429, 500, 502, 503, 504}:
        return True
    return bool(_RETRYABLE_ERROR_PATTERN.search(error_text))


def _extract_error_message(error_text: str) -> str:
    """Extract a clean, user-friendly error message from API error response.

    Parses JSON error responses and returns just the ``message`` field when
    available, otherwise falls back to the raw text.
    """
    try:
        parsed = json.loads(error_text)
        if isinstance(parsed, dict):
            error_obj = parsed.get("error")
            if isinstance(error_obj, dict):
                message = error_obj.get("message")
                if isinstance(message, str) and message:
                    return message
    except (json.JSONDecodeError, ValueError):
        pass
    return error_text


def _get_antigravity_headers() -> dict[str, str]:
    version = os.environ.get("PI_AI_ANTIGRAVITY_VERSION") or DEFAULT_ANTIGRAVITY_VERSION
    return {
        "User-Agent": f"antigravity/{version} darwin/arm64",
        "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
        "Client-Metadata": json.dumps(
            {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            },
        ),
    }


_CREDENTIAL_CACHE: dict[str, GeminiCliCredentials] = {}


@dataclass(slots=True)
class GeminiCliCredentials:
    """Credentials used by the Cloud Code Assist Gemini endpoint."""

    refresh: str | None
    access: str | None
    expires: int | None
    project_id: str | None
    email: str | None = None
    oauth_client_id: str | None = None
    oauth_client_secret: str | None = None
    oauth_token_url: str | None = None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _is_expired(expires_ms: int | None) -> bool:
    if expires_ms is None:
        return True
    return _now_ms() >= expires_ms - TOKEN_EXPIRY_BUFFER_MS


def parse_api_key_credentials(
    api_key: str,
    provider_id: str = GOOGLE_GEMINI_CLI_PROVIDER,
) -> GeminiCliCredentials:
    """Parse a provider api_key value into structured credentials.

    Supported formats:
    1. JSON string with fields like refresh/access/expires/projectId/email
    2. Raw refresh token string
    """
    value = api_key.strip()
    if not value:
        msg = f"{provider_id} provider requires api_key credentials"
        raise ValueError(msg)

    if not value.startswith("{"):
        return GeminiCliCredentials(
            refresh=value,
            access=None,
            expires=None,
            project_id=None,
            email=None,
        )

    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in {provider_id} api_key"
        raise ValueError(msg) from exc

    if not isinstance(payload, dict):
        msg = f"{provider_id} api_key JSON must be an object"
        raise ValueError(msg)

    expires_raw = payload.get("expires") or payload.get("expires_at")
    expires_ms: int | None = None
    if isinstance(expires_raw, (int, float)):
        expires_ms = int(expires_raw)

    return GeminiCliCredentials(
        refresh=(
            payload.get("refresh")
            or payload.get("refresh_token")
            or payload.get("refreshToken")
        ),
        access=payload.get("access")
        or payload.get("token")
        or payload.get("access_token"),
        expires=expires_ms,
        project_id=(
            payload.get("projectId")
            or payload.get("project_id")
            or payload.get("project")
        ),
        email=payload.get("email"),
        oauth_client_id=payload.get("client_id") or payload.get("oauth_client_id"),
        oauth_client_secret=payload.get("client_secret")
        or payload.get("oauth_client_secret"),
        oauth_token_url=payload.get("token_uri") or payload.get("oauth_token_url"),
    )


def credentials_to_api_key(credentials: GeminiCliCredentials) -> str:
    """Serialize credentials for config storage or logging guidance."""
    payload: dict[str, Any] = {
        "refresh": credentials.refresh,
        "access": credentials.access,
        "expires": credentials.expires,
        "projectId": credentials.project_id,
    }
    if credentials.email:
        payload["email"] = credentials.email
    if credentials.oauth_client_id:
        payload["client_id"] = credentials.oauth_client_id
    if credentials.oauth_client_secret:
        payload["client_secret"] = credentials.oauth_client_secret
    if credentials.oauth_token_url:
        payload["token_uri"] = credentials.oauth_token_url
    return json.dumps(payload, separators=(",", ":"))


def _parse_data_url(url: str) -> dict[str, str] | None:
    if not url.startswith("data:"):
        return None
    if ";base64," not in url:
        return None
    meta, encoded = url.split(";base64,", 1)
    mime_type = meta.removeprefix("data:")
    if not mime_type or not encoded:
        return None
    return {"mimeType": mime_type, "data": encoded}


def _convert_messages(
    messages: list[dict[str, object]],
) -> tuple[list[dict[str, object]], str | None]:
    contents: list[dict[str, object]] = []
    system_prompt: str | None = None

    for message in messages:
        role = str(message.get("role") or "user")
        raw_content = message.get("content")

        if role == "system":
            if isinstance(raw_content, str) and raw_content.strip():
                system_prompt = raw_content
            continue

        api_role = "model" if role == "assistant" else "user"
        parts: list[dict[str, object]] = []

        if isinstance(raw_content, str):
            if raw_content:
                parts.append({"text": raw_content})
        elif isinstance(raw_content, list):
            for part in raw_content:
                if not isinstance(part, dict):
                    continue
                part_dict = cast("dict[str, object]", part)
                part_type = part_dict.get("type")
                if part_type == "text":
                    text = part_dict.get("text")
                    if isinstance(text, str) and text:
                        parts.append({"text": text})
                    continue
                if part_type == "image_url":
                    image_obj = part_dict.get("image_url")
                    if not isinstance(image_obj, dict):
                        continue
                    image_obj_dict = cast("dict[str, object]", image_obj)
                    image_url = image_obj_dict.get("url")
                    if not isinstance(image_url, str):
                        continue
                    parsed = _parse_data_url(image_url)
                    if parsed is not None:
                        parts.append({"inlineData": parsed})
                    else:
                        parts.append({"text": f"Image URL: {image_url}"})
                    continue
                if part_type == "file":
                    file_obj = part_dict.get("file")
                    if not isinstance(file_obj, dict):
                        continue
                    file_obj_dict = cast("dict[str, object]", file_obj)
                    file_data = file_obj_dict.get("file_data")
                    if not isinstance(file_data, str):
                        continue
                    parsed = _parse_data_url(file_data)
                    if parsed is not None:
                        parts.append({"inlineData": parsed})
                    else:
                        parts.append(
                            {"text": "Attached file was omitted (unsupported format)."},
                        )

        if parts:
            contents.append(
                {
                    "role": api_role,
                    "parts": parts,
                },
            )

    return contents, system_prompt


def _extract_thinking_level(
    model: str,
    model_parameters: dict[str, object] | None,
) -> str | None:
    if model_parameters:
        level = model_parameters.get("thinking_level") or model_parameters.get(
            "thinkingLevel",
        )
        if isinstance(level, str) and level:
            return level.upper()

    suffix_map = {
        "-minimal": "MINIMAL",
        "-low": "LOW",
        "-medium": "MEDIUM",
        "-high": "HIGH",
    }
    for suffix, level in suffix_map.items():
        if model.endswith(suffix):
            return level

    if "gemini-3-flash" in model:
        return "MINIMAL"
    if "gemini-3-pro" in model:
        return "LOW"
    return None


def _strip_thinking_level_suffix(model: str) -> str:
    suffixes = ("-minimal", "-low", "-medium", "-high")
    for suffix in suffixes:
        if model.endswith(suffix):
            return model.removesuffix(suffix)
    return model


def _build_generation_config(
    model: str,
    model_parameters: dict[str, object] | None,
) -> dict[str, object] | None:
    config: dict[str, object] = {}
    level = _extract_thinking_level(model, model_parameters)
    budget_tokens: object | None = None
    if model_parameters:
        temperature = model_parameters.get("temperature")
        if isinstance(temperature, (float, int)):
            config["temperature"] = float(temperature)

        max_output_tokens = model_parameters.get(
            "max_output_tokens",
        ) or model_parameters.get("max_tokens")
        if isinstance(max_output_tokens, int):
            config["maxOutputTokens"] = max_output_tokens

        budget_tokens = model_parameters.get("thinking_budget") or model_parameters.get(
            "thinkingBudget",
        )

    if isinstance(budget_tokens, int) or level is not None:
        thinking_config: dict[str, object] = {"includeThoughts": True}
        if isinstance(budget_tokens, int):
            thinking_config["thinkingBudget"] = budget_tokens
        if level is not None:
            thinking_config["thinkingLevel"] = level
        config["thinkingConfig"] = thinking_config

    if not config:
        return None
    return config


async def _poll_operation(
    *,
    operation_name: str,
    headers: dict[str, str],
) -> dict[str, object]:
    async with httpx.AsyncClient(timeout=60) as client:
        while True:
            response = await client.get(
                f"{CODE_ASSIST_ENDPOINT}/v1internal/{operation_name}",
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and data.get("done"):
                return data
            await asyncio.sleep(5)


def _is_vpc_sc_error(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    payload_dict = cast("dict[str, object]", payload)
    error = payload_dict.get("error")
    if not isinstance(error, dict):
        return False
    error_dict = cast("dict[str, object]", error)
    details = error_dict.get("details")
    if not isinstance(details, list):
        return False
    for detail in details:
        if not isinstance(detail, dict):
            continue
        detail_dict = cast("dict[str, object]", detail)
        if detail_dict.get("reason") == "SECURITY_POLICY_VIOLATED":
            return True
    return False


def _get_default_tier(allowed_tiers: object) -> str:
    if not isinstance(allowed_tiers, list):
        return TIER_LEGACY
    for tier in allowed_tiers:
        if not isinstance(tier, dict):
            continue
        tier_dict = cast("dict[str, object]", tier)
        tier_id = tier_dict.get("id")
        if tier_dict.get("isDefault") and isinstance(tier_id, str):
            return tier_id
    return TIER_LEGACY


async def _discover_project_gemini_cli(access_token: str) -> str:
    """Discover or provision a Cloud Code Assist project id for Gemini CLI."""
    env_project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get(
        "GOOGLE_CLOUD_PROJECT_ID",
    )
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": "google-api-nodejs-client/9.15.1",
        "X-Goog-Api-Client": "gl-node/22.17.0",
    }

    load_body = {
        "cloudaicompanionProject": env_project,
        "metadata": {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
            "duetProject": env_project,
        },
    }

    async with httpx.AsyncClient(timeout=60) as client:
        load_response = await client.post(
            f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
            headers=headers,
            json=load_body,
        )

        load_data: dict[str, object]
        if not load_response.is_success:
            payload: object
            try:
                payload = load_response.json()
            except ValueError:
                payload = None
            if _is_vpc_sc_error(payload):
                load_data = {"currentTier": {"id": "standard-tier"}}
            else:
                load_response.raise_for_status()
                msg = "Unexpected loadCodeAssist failure"
                raise RuntimeError(msg)
        else:
            body = load_response.json()
            if not isinstance(body, dict):
                msg = "Invalid loadCodeAssist response"
                raise RuntimeError(msg)
            load_data = body

        current_tier = load_data.get("currentTier")
        managed_project = load_data.get("cloudaicompanionProject")
        if isinstance(current_tier, dict):
            if isinstance(managed_project, str) and managed_project:
                return managed_project
            if env_project:
                return env_project
            msg = (
                "This account requires GOOGLE_CLOUD_PROJECT or "
                "GOOGLE_CLOUD_PROJECT_ID to be set."
            )
            raise RuntimeError(msg)

        tier_id = _get_default_tier(load_data.get("allowedTiers"))
        if tier_id != TIER_FREE and not env_project:
            msg = (
                "This account requires GOOGLE_CLOUD_PROJECT or "
                "GOOGLE_CLOUD_PROJECT_ID to be set."
            )
            raise RuntimeError(msg)

        onboard_body: dict[str, object] = {
            "tierId": tier_id,
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            },
        }
        if tier_id != TIER_FREE and env_project:
            onboard_body["cloudaicompanionProject"] = env_project
            metadata = onboard_body["metadata"]
            if isinstance(metadata, dict):
                metadata["duetProject"] = env_project

        onboard_response = await client.post(
            f"{CODE_ASSIST_ENDPOINT}/v1internal:onboardUser",
            headers=headers,
            json=onboard_body,
        )
        onboard_response.raise_for_status()
        operation_data = onboard_response.json()
        if not isinstance(operation_data, dict):
            msg = "Invalid onboardUser response"
            raise RuntimeError(msg)

        if not operation_data.get("done") and isinstance(
            operation_data.get("name"),
            str,
        ):
            operation_data = await _poll_operation(
                operation_name=operation_data["name"],
                headers=headers,
            )

        response_data = operation_data.get("response")
        if isinstance(response_data, dict):
            response_dict = cast("dict[str, object]", response_data)
            companion = response_dict.get("cloudaicompanionProject")
            if isinstance(companion, dict):
                companion_dict = cast("dict[str, object]", companion)
                project_id = companion_dict.get("id")
                if isinstance(project_id, str) and project_id:
                    return project_id

        if env_project:
            return env_project

    msg = "Could not discover or provision a Google Cloud project"
    raise RuntimeError(msg)


async def _discover_project_antigravity(access_token: str) -> str:
    """Discover antigravity project id, falling back to a known default."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": "google-api-nodejs-client/9.15.1",
        "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
        "Client-Metadata": json.dumps(
            {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            },
        ),
    }
    load_body = {
        "metadata": {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        },
    }
    endpoints = (CODE_ASSIST_ENDPOINT, ANTIGRAVITY_DAILY_ENDPOINT)

    async with httpx.AsyncClient(timeout=60) as client:
        for endpoint in endpoints:
            try:
                load_response = await client.post(
                    f"{endpoint}/v1internal:loadCodeAssist",
                    headers=headers,
                    json=load_body,
                )
            except httpx.HTTPError:
                continue

            if not load_response.is_success:
                continue

            body = load_response.json()
            if not isinstance(body, dict):
                continue

            companion = body.get("cloudaicompanionProject")
            if isinstance(companion, str) and companion:
                return companion
            if isinstance(companion, dict):
                companion_dict = cast("dict[str, object]", companion)
                project_id = companion_dict.get("id")
                if isinstance(project_id, str) and project_id:
                    return project_id

    return ANTIGRAVITY_DEFAULT_PROJECT_ID


async def discover_project(
    access_token: str,
    *,
    provider_id: str = GOOGLE_GEMINI_CLI_PROVIDER,
) -> str:
    """Discover or provision a Cloud Code Assist project id."""
    if provider_id == GOOGLE_ANTIGRAVITY_PROVIDER:
        return await _discover_project_antigravity(access_token)
    return await _discover_project_gemini_cli(access_token)


async def _refresh_google_cloud_token(
    refresh_token: str,
    project_id: str | None,
    provider_id: str,
    oauth_client_id: str | None,
    oauth_client_secret: str | None,
    oauth_token_url: str | None,
) -> GeminiCliCredentials:
    if provider_id == GOOGLE_ANTIGRAVITY_PROVIDER:
        default_client_id = ANTIGRAVITY_CLIENT_ID
        default_client_secret = ANTIGRAVITY_CLIENT_SECRET
    else:
        default_client_id = GEMINI_CLI_CLIENT_ID
        default_client_secret = GEMINI_CLI_CLIENT_SECRET

    token_url = oauth_token_url or TOKEN_URL
    client_id = oauth_client_id or default_client_id
    client_secret = oauth_client_secret or default_client_secret
    request_data: dict[str, str] = {
        "client_id": client_id,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    if client_secret:
        request_data["client_secret"] = client_secret

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            token_url,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=request_data,
        )
        if not response.is_success:
            detail = response.text
            msg = (
                "Google OAuth token refresh failed "
                f"({response.status_code}) at {token_url}: {detail}"
            )
            raise RuntimeError(msg)
        payload = response.json()
        if not isinstance(payload, dict):
            msg = "Invalid token refresh response"
            raise RuntimeError(msg)

        access_token = payload.get("access_token")
        expires_in = payload.get("expires_in")
        if not isinstance(access_token, str) or not access_token:
            msg = "Token refresh did not return access_token"
            raise RuntimeError(msg)
        if not isinstance(expires_in, int):
            msg = "Token refresh did not return expires_in"
            raise RuntimeError(msg)

        resolved_project = project_id or await discover_project(
            access_token,
            provider_id=provider_id,
        )
        returned_refresh = payload.get("refresh_token")
        refresh_value = (
            returned_refresh if isinstance(returned_refresh, str) else refresh_token
        )
        return GeminiCliCredentials(
            refresh=refresh_value,
            access=access_token,
            expires=_now_ms() + (expires_in * 1000),
            project_id=resolved_project,
            email=None,
            oauth_client_id=oauth_client_id,
            oauth_client_secret=oauth_client_secret,
            oauth_token_url=oauth_token_url,
        )


async def get_valid_google_gemini_cli_credentials(api_key: str) -> GeminiCliCredentials:
    """Return valid access credentials for Cloud Code Assist requests."""
    cached = _CREDENTIAL_CACHE.get(api_key)
    credentials = cached or parse_api_key_credentials(api_key)

    has_access = isinstance(credentials.access, str) and bool(credentials.access)
    has_project = isinstance(credentials.project_id, str) and bool(
        credentials.project_id,
    )
    if has_access and has_project and not _is_expired(credentials.expires):
        _CREDENTIAL_CACHE[api_key] = credentials
        return credentials

    if not credentials.refresh:
        msg = (
            "google-gemini-cli api_key must include a refresh token if access token "
            "is missing or expired"
        )
        raise RuntimeError(msg)

    refreshed = await _refresh_google_cloud_token(
        credentials.refresh,
        credentials.project_id,
        GOOGLE_GEMINI_CLI_PROVIDER,
        credentials.oauth_client_id,
        credentials.oauth_client_secret,
        credentials.oauth_token_url,
    )
    if credentials.email and not refreshed.email:
        refreshed.email = credentials.email

    _CREDENTIAL_CACHE[api_key] = refreshed
    return refreshed


async def get_valid_cloud_code_assist_credentials(
    api_key: str,
    provider_id: str,
) -> GeminiCliCredentials:
    """Return valid access credentials for Cloud Code Assist providers."""
    cache_key = f"{provider_id}:{api_key}"
    cached = _CREDENTIAL_CACHE.get(cache_key)
    credentials = cached or parse_api_key_credentials(api_key, provider_id)

    has_access = isinstance(credentials.access, str) and bool(credentials.access)
    has_project = isinstance(credentials.project_id, str) and bool(
        credentials.project_id,
    )
    if has_access and has_project and not _is_expired(credentials.expires):
        _CREDENTIAL_CACHE[cache_key] = credentials
        return credentials

    if not credentials.refresh:
        msg = (
            f"{provider_id} api_key must include a refresh token if access token "
            "is missing or expired"
        )
        raise RuntimeError(msg)

    refreshed = await _refresh_google_cloud_token(
        credentials.refresh,
        credentials.project_id,
        provider_id,
        credentials.oauth_client_id,
        credentials.oauth_client_secret,
        credentials.oauth_token_url,
    )
    if credentials.email and not refreshed.email:
        refreshed.email = credentials.email

    _CREDENTIAL_CACHE[cache_key] = refreshed
    return refreshed


def _build_cloudcode_request(
    *,
    provider_id: str,
    model: str,
    project_id: str,
    messages: list[dict[str, object]],
    model_parameters: dict[str, object] | None,
) -> dict[str, object]:
    clean_model = _strip_thinking_level_suffix(model)
    contents, system_prompt = _convert_messages(messages)
    request_body: dict[str, object] = {
        "contents": contents,
    }
    if system_prompt:
        request_body["systemInstruction"] = {
            "parts": [{"text": system_prompt}],
        }

    if provider_id == GOOGLE_ANTIGRAVITY_PROVIDER:
        existing_parts = []
        existing_instruction = request_body.get("systemInstruction")
        if isinstance(existing_instruction, dict):
            instruction_dict = cast("dict[str, object]", existing_instruction)
            parts = instruction_dict.get("parts")
            if isinstance(parts, list):
                existing_parts = parts
        request_body["systemInstruction"] = {
            "role": "user",
            "parts": [
                {"text": ANTIGRAVITY_SYSTEM_INSTRUCTION},
                {
                    "text": (
                        "Please ignore following [ignore]"
                        f"{ANTIGRAVITY_SYSTEM_INSTRUCTION}"
                        "[/ignore]"
                    ),
                },
                *existing_parts,
            ],
        }

    generation_config = _build_generation_config(model, model_parameters)
    if generation_config:
        request_body["generationConfig"] = generation_config

    request_body["safetySettings"] = DEFAULT_SAFETY_SETTINGS

    if provider_id == GOOGLE_ANTIGRAVITY_PROVIDER:
        request_body["tools"] = [
            {
                "functionDeclarations": [IMAGE_TOOL_DECLARATION],
            },
        ]
        request_body["toolConfig"] = {
            "functionCallingConfig": {
                "mode": "AUTO",
            },
        }

    request_payload: dict[str, object] = {
        "project": project_id,
        "model": clean_model,
        "request": request_body,
        "userAgent": (
            "antigravity" if provider_id == GOOGLE_ANTIGRAVITY_PROVIDER else "llmcord"
        ),
        "requestId": (
            f"agent-{int(time.time() * 1000)}-{secrets.token_hex(5)}"
            if provider_id == GOOGLE_ANTIGRAVITY_PROVIDER
            else f"llmcord-{int(time.time() * 1000)}-{secrets.token_hex(5)}"
        ),
    }
    if provider_id == GOOGLE_ANTIGRAVITY_PROVIDER:
        request_payload["requestType"] = "agent"
    return request_payload


_RETRY_DELAY_BUFFER_MS = 1000
_MAX_RETRY_DELAY_MS = 60_000


def _normalize_retry_delay(ms: float) -> int | None:
    """Add a 1-second buffer and return the delay in milliseconds.

    Mirrors the reference ``normalizeDelay`` behaviour so that the caller
    always waits *slightly* longer than the server-advertised window.
    """
    if ms <= 0:
        return None
    return int(ms) + _RETRY_DELAY_BUFFER_MS


def _extract_retry_delay_ms(  # noqa: PLR0911
    response: httpx.Response,
    response_text: str,
) -> int | None:
    """Extract retry delay from error response (milliseconds).

    Checks response headers first (Retry-After, x-ratelimit-reset,
    x-ratelimit-reset-after), then falls back to body text patterns:
    - ``Your quota will reset after 18h31m10s``
    - ``Please retry in Xs`` / ``Please retry in Xms``
    - ``"retryDelay": "34.074824224s"`` (JSON field)
    """
    # --- Header: Retry-After (seconds or HTTP-date) ---
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            delay = _normalize_retry_delay(float(retry_after) * 1000)
            if delay is not None:
                return delay
        except ValueError:
            pass
        # Try as HTTP-date (e.g. "Fri, 31 Dec 1999 23:59:59 GMT")
        try:
            retry_dt = parsedate_to_datetime(retry_after)
            delay = _normalize_retry_delay(
                retry_dt.timestamp() * 1000 - _now_ms(),
            )
            if delay is not None:
                return delay
        except (ValueError, TypeError):
            pass

    # --- Header: x-ratelimit-reset (unix epoch seconds) ---
    ratelimit_reset = response.headers.get("x-ratelimit-reset")
    if ratelimit_reset:
        try:
            reset_seconds = int(ratelimit_reset)
            delay = _normalize_retry_delay(reset_seconds * 1000 - _now_ms())
            if delay is not None:
                return delay
        except ValueError:
            pass

    # --- Header: x-ratelimit-reset-after (seconds) ---
    ratelimit_reset_after = response.headers.get("x-ratelimit-reset-after")
    if ratelimit_reset_after:
        try:
            delay = _normalize_retry_delay(float(ratelimit_reset_after) * 1000)
            if delay is not None:
                return delay
        except ValueError:
            pass

    # --- Body pattern 1: "Your quota will reset after 18h31m10s" ---
    duration_match = re.search(
        r"reset after (?:(\d+)h)?(?:(\d+)m)?(\d+(?:\.\d+)?)s",
        response_text,
        re.IGNORECASE,
    )
    if duration_match:
        hours = int(duration_match.group(1)) if duration_match.group(1) else 0
        minutes = int(duration_match.group(2)) if duration_match.group(2) else 0
        seconds = float(duration_match.group(3))
        total_ms = ((hours * 60 + minutes) * 60 + seconds) * 1000
        delay = _normalize_retry_delay(total_ms)
        if delay is not None:
            return delay

    # --- Body pattern 2: "Please retry in Xs" / "Please retry in Xms" ---
    retry_in_match = re.search(
        r"Please retry in ([0-9.]+)(ms|s)",
        response_text,
        re.IGNORECASE,
    )
    if retry_in_match:
        value = float(retry_in_match.group(1))
        unit = retry_in_match.group(2).lower()
        if value > 0:
            ms = value if unit == "ms" else value * 1000
            delay = _normalize_retry_delay(ms)
            if delay is not None:
                return delay

    # --- Body pattern 3: "retryDelay": "34.074824224s" ---
    retry_delay_match = re.search(
        r'"retryDelay":\s*"([0-9.]+)(ms|s)"',
        response_text,
        re.IGNORECASE,
    )
    if retry_delay_match:
        value = float(retry_delay_match.group(1))
        unit = retry_delay_match.group(2).lower()
        if value > 0:
            ms = value if unit == "ms" else value * 1000
            delay = _normalize_retry_delay(ms)
            if delay is not None:
                return delay

    return None


def _normalize_aspect_ratio(value: object) -> str:
    if not isinstance(value, str):
        return IMAGE_TOOL_DEFAULT_ASPECT_RATIO
    normalized = value.strip()
    if normalized in IMAGE_TOOL_ASPECT_RATIOS:
        return normalized
    return IMAGE_TOOL_DEFAULT_ASPECT_RATIO


def _extract_image_parts_from_content(raw_content: object) -> list[dict[str, object]]:
    image_parts: list[dict[str, object]] = []
    if not isinstance(raw_content, list):
        return image_parts

    for part in raw_content:
        if not isinstance(part, dict):
            continue
        part_dict = cast("dict[str, object]", part)
        part_type = part_dict.get("type")

        if part_type == "image_url":
            image_obj = part_dict.get("image_url")
            if not isinstance(image_obj, dict):
                continue
            image_url = cast("dict[str, object]", image_obj).get("url")
            if not isinstance(image_url, str):
                continue
            parsed = _parse_data_url(image_url)
            if parsed is not None:
                image_parts.append({"inlineData": parsed})
            continue

        if part_type == "file":
            file_obj = part_dict.get("file")
            if not isinstance(file_obj, dict):
                continue
            file_data = cast("dict[str, object]", file_obj).get("file_data")
            if not isinstance(file_data, str):
                continue
            parsed = _parse_data_url(file_data)
            if parsed is None:
                continue
            mime_type = parsed.get("mimeType")
            if isinstance(mime_type, str) and mime_type.startswith("image/"):
                image_parts.append({"inlineData": parsed})

    return image_parts


def _build_antigravity_image_tool_contents(
    messages: list[dict[str, object]],
    fallback_prompt: object,
) -> list[dict[str, object]]:
    image_contents: list[dict[str, object]] = []

    for message in messages:
        role = str(message.get("role") or "user")
        if role == "system":
            continue
        raw_content = message.get("content")

        if role == "assistant":
            assistant_images = _extract_image_parts_from_content(raw_content)
            if assistant_images:
                image_contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": (
                                    "Previously generated image for follow-up "
                                    "editing context."
                                ),
                            },
                            *assistant_images,
                        ],
                    },
                )
            continue

        user_parts: list[dict[str, object]] = []
        if isinstance(raw_content, str):
            if raw_content.strip():
                user_parts.append({"text": raw_content})
        elif isinstance(raw_content, list):
            for part in raw_content:
                if not isinstance(part, dict):
                    continue
                part_dict = cast("dict[str, object]", part)
                part_type = part_dict.get("type")
                if part_type == "text":
                    text = part_dict.get("text")
                    if isinstance(text, str) and text:
                        user_parts.append({"text": text})
                    continue
                if part_type in {"image_url", "file"}:
                    user_parts.extend(_extract_image_parts_from_content([part_dict]))

        if user_parts:
            image_contents.append({"role": "user", "parts": user_parts})

    if image_contents:
        return image_contents
    if isinstance(fallback_prompt, str) and fallback_prompt.strip():
        return [{"role": "user", "parts": [{"text": fallback_prompt}]}]
    return [
        {
            "role": "user",
            "parts": [{"text": "Generate an image based on the latest request."}],
        },
    ]


def _build_antigravity_image_request(
    *,
    project_id: str,
    contents: list[dict[str, object]],
    model: str,
    aspect_ratio: str,
) -> dict[str, object]:
    return {
        "project": project_id,
        "model": model,
        "request": {
            "contents": contents,
            "systemInstruction": {
                "parts": [{"text": IMAGE_SYSTEM_INSTRUCTION}],
            },
            "generationConfig": {
                "imageConfig": {"aspectRatio": aspect_ratio},
                "candidateCount": 1,
            },
            "safetySettings": DEFAULT_SAFETY_SETTINGS,
        },
        "requestType": "agent",
        "userAgent": "antigravity",
        "requestId": f"agent-{int(time.time() * 1000)}-{secrets.token_hex(5)}",
    }


async def _request_antigravity_generated_image(
    *,
    access_token: str,
    project_id: str,
    contents: list[dict[str, object]],
    model: str,
    aspect_ratio: str,
    endpoint: str,
    headers: dict[str, str],
) -> tuple[str, str, str]:
    def _raise_image_request_error(message: str) -> None:
        raise RuntimeError(message)

    def _candidate_endpoints(initial: str) -> list[str]:
        normalized_initial = initial.rstrip("/")
        endpoint_candidates = [
            normalized_initial,
            ANTIGRAVITY_DAILY_ENDPOINT,
            DEFAULT_ENDPOINT,
        ]
        deduped: list[str] = []
        for candidate in endpoint_candidates:
            normalized = candidate.rstrip("/")
            if normalized not in deduped:
                deduped.append(normalized)
        return deduped

    def _is_retryable_status(status_code: int) -> bool:
        return status_code in {408, 429, 500, 502, 503, 504}

    request_payload = _build_antigravity_image_request(
        project_id=project_id,
        contents=contents,
        model=model,
        aspect_ratio=aspect_ratio,
    )
    request_headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        **headers,
    }
    last_error: Exception | None = None

    for current_endpoint in _candidate_endpoints(endpoint):
        image_endpoint = f"{current_endpoint}/v1internal:streamGenerateContent?alt=sse"
        for attempt in range(IMAGE_TOOL_MAX_ATTEMPTS_PER_ENDPOINT):
            try:
                async with (
                    httpx.AsyncClient(
                        timeout=httpx.Timeout(
                            connect=30,
                            read=IMAGE_TOOL_READ_TIMEOUT_SECONDS,
                            write=30,
                            pool=30,
                        ),
                    ) as client,
                    client.stream(
                        "POST",
                        image_endpoint,
                        headers=request_headers,
                        json=request_payload,
                    ) as response,
                ):
                    if not response.is_success:
                        error_bytes = await response.aread()
                        error_text = error_bytes.decode("utf-8", errors="replace")
                        if (
                            _is_retryable_status(response.status_code)
                            and attempt + 1 < IMAGE_TOOL_MAX_ATTEMPTS_PER_ENDPOINT
                        ):
                            await asyncio.sleep(
                                IMAGE_TOOL_RETRY_BASE_DELAY_SECONDS * (2**attempt),
                            )
                            continue
                        msg = (
                            "Antigravity image generation failed "
                            f"({response.status_code}): {error_text}"
                        )
                        _raise_image_request_error(msg)

                    model_notes: list[str] = []
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if not payload:
                            continue
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(chunk, dict):
                            continue

                        response_data = chunk.get("response")
                        if not isinstance(response_data, dict):
                            continue
                        candidates = response_data.get("candidates")
                        if not isinstance(candidates, list):
                            continue

                        for candidate in candidates:
                            if not isinstance(candidate, dict):
                                continue
                            content = candidate.get("content")
                            if not isinstance(content, dict):
                                continue
                            parts = content.get("parts")
                            if not isinstance(parts, list):
                                continue

                            for part in parts:
                                if not isinstance(part, dict):
                                    continue
                                text = part.get("text")
                                if isinstance(text, str) and text:
                                    model_notes.append(text)

                                inline_data = part.get("inlineData")
                                if not isinstance(inline_data, dict):
                                    continue
                                inline_data_dict = cast(
                                    "dict[str, object]",
                                    inline_data,
                                )
                                mime_type = inline_data_dict.get("mimeType")
                                encoded_data = inline_data_dict.get("data")
                                if (
                                    not isinstance(encoded_data, str)
                                    or not encoded_data
                                ):
                                    continue
                                if not isinstance(mime_type, str) or not mime_type:
                                    mime_type = "image/png"
                                return (
                                    encoded_data,
                                    mime_type,
                                    " ".join(model_notes).strip(),
                                )

                    msg = "Antigravity image generation returned no image data"
                    _raise_image_request_error(msg)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if attempt + 1 < IMAGE_TOOL_MAX_ATTEMPTS_PER_ENDPOINT:
                    await asyncio.sleep(
                        IMAGE_TOOL_RETRY_BASE_DELAY_SECONDS * (2**attempt),
                    )
                    continue
                break
            except RuntimeError as exc:
                last_error = exc
                break
            except httpx.HTTPError as exc:
                last_error = exc
                break

    if last_error is not None:
        msg = (
            "Antigravity image generation failed after retries across endpoints: "
            f"{str(last_error).strip() or type(last_error).__name__}"
        )
        raise RuntimeError(msg)

    msg = "Antigravity image generation failed after retries across endpoints"
    raise RuntimeError(msg)


async def _execute_antigravity_image_tool_call(
    *,
    endpoint: str,
    headers: dict[str, str],
    credentials: GeminiCliCredentials,
    messages: list[dict[str, object]],
    call_name: str,
    call_args: dict[str, object],
) -> str:
    if call_name != IMAGE_TOOL_NAME:
        return ""

    model = call_args.get("model")
    model_id = (
        model.strip()
        if isinstance(model, str) and model.strip()
        else IMAGE_TOOL_DEFAULT_MODEL
    )
    aspect_ratio = _normalize_aspect_ratio(call_args.get("aspectRatio"))

    if not credentials.access or not credentials.project_id:
        return "Image tool call failed: missing antigravity access credentials."

    fallback_prompt = call_args.get("prompt")
    image_contents = _build_antigravity_image_tool_contents(
        messages,
        fallback_prompt,
    )

    encoded_data, mime_type, notes = await _request_antigravity_generated_image(
        access_token=credentials.access,
        project_id=credentials.project_id,
        contents=image_contents,
        model=model_id,
        aspect_ratio=aspect_ratio,
        endpoint=endpoint,
        headers=headers,
    )

    summary = (
        f"Generated image via {GOOGLE_ANTIGRAVITY_PROVIDER}/{model_id}. "
        f"Aspect ratio: {aspect_ratio}."
    )
    if notes:
        summary = f"{summary} Model notes: {notes}"
    data_url = f"data:{mime_type};base64,{encoded_data}"
    return f"{summary}\n{data_url}"


def _raise_cloud_code_error(message: str) -> None:
    raise RuntimeError(message)


def _raise_image_tool_error(exc: Exception) -> None:
    msg = f"Image tool call failed: {str(exc).strip() or type(exc).__name__}"
    raise RuntimeError(msg) from exc


async def _run_image_tool(
    *,
    endpoint: str,
    headers: dict[str, str],
    credentials: GeminiCliCredentials,
    messages: list[dict[str, object]],
    name: str,
    args: dict[str, object],
) -> str:
    """Execute an image tool call, wrapping errors."""
    try:
        return await _execute_antigravity_image_tool_call(
            endpoint=endpoint,
            headers=headers,
            credentials=credentials,
            messages=messages,
            call_name=name,
            call_args=args,
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        httpx.HTTPError,
    ) as exc:
        _raise_image_tool_error(exc)
    return ""  # unreachable, keeps mypy happy


async def stream_google_gemini_cli(
    *,
    provider_id: str = GOOGLE_GEMINI_CLI_PROVIDER,
    model: str,
    messages: list[dict[str, object]],
    api_key: str,
    base_url: str | None,
    extra_headers: dict[str, str] | None,
    model_parameters: dict[str, object] | None,
) -> AsyncIterator[tuple[str, object | None, bool]]:
    """Stream text deltas from Cloud Code Assist SSE endpoint."""
    credentials = await get_valid_cloud_code_assist_credentials(api_key, provider_id)
    if not credentials.access or not credentials.project_id:
        msg = f"Missing access token or project id for {provider_id}"
        raise RuntimeError(msg)

    body = _build_cloudcode_request(
        provider_id=provider_id,
        model=model,
        project_id=credentials.project_id,
        messages=messages,
        model_parameters=model_parameters,
    )

    endpoints = [
        (base_url or DEFAULT_ENDPOINT).rstrip("/"),
    ]
    if provider_id == GOOGLE_ANTIGRAVITY_PROVIDER and not base_url:
        endpoints = [
            ANTIGRAVITY_DAILY_ENDPOINT,
            DEFAULT_ENDPOINT,
        ]

    provider_headers = (
        _get_antigravity_headers()
        if provider_id == GOOGLE_ANTIGRAVITY_PROVIDER
        else GEMINI_CLI_HEADERS
    )

    headers = {
        "Authorization": f"Bearer {credentials.access}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        **provider_headers,
    }
    if extra_headers:
        headers.update(extra_headers)

    # --- Retry loop (matches reference: MAX_RETRIES=3, attempt 0..3) ---
    max_retries = 3
    base_delay_ms = 1000
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        endpoint = endpoints[min(attempt, len(endpoints) - 1)]
        url = f"{endpoint}/v1internal:streamGenerateContent?alt=sse"

        try:
            async with (
                httpx.AsyncClient(timeout=30) as client,
                client.stream(
                    "POST",
                    url,
                    headers=headers,
                    json=body,
                ) as response,
            ):
                if not response.is_success:
                    error_bytes = await response.aread()
                    decoded_error = error_bytes.decode("utf-8", errors="replace")

                    if attempt < max_retries and _is_retryable_error(
                        response.status_code,
                        decoded_error,
                    ):
                        server_delay = _extract_retry_delay_ms(
                            response,
                            decoded_error,
                        )
                        delay_ms = server_delay or (base_delay_ms * (2**attempt))

                        # Cap server-provided delay at 60 seconds
                        if (
                            server_delay is not None
                            and server_delay > _MAX_RETRY_DELAY_MS
                        ):
                            delay_seconds = (server_delay + 999) // 1000
                            max_seconds = _MAX_RETRY_DELAY_MS // 1000
                            _raise_cloud_code_error(
                                f"Server requested {delay_seconds}s retry "
                                f"delay (max: {max_seconds}s). "
                                f"{_extract_error_message(decoded_error)}",
                            )

                        await asyncio.sleep(delay_ms / 1000)
                        continue

                    _raise_cloud_code_error(
                        "Cloud Code Assist API error "
                        f"({response.status_code}): "
                        f"{_extract_error_message(decoded_error)}",
                    )

                # --- Stream SSE response ---
                has_content = False
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload:
                        continue
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(chunk, dict):
                        continue

                    response_data = chunk.get("response")
                    if not isinstance(response_data, dict):
                        continue
                    candidates = response_data.get("candidates")
                    if not isinstance(candidates, list) or not candidates:
                        continue
                    candidate = candidates[0]
                    if not isinstance(candidate, dict):
                        continue

                    content = candidate.get("content")
                    if isinstance(content, dict):
                        parts = content.get("parts")
                        if isinstance(parts, list):
                            for part in parts:
                                if not isinstance(part, dict):
                                    continue
                                function_call = part.get("functionCall")
                                if isinstance(function_call, dict):
                                    function_call_dict = cast(
                                        "dict[str, object]",
                                        function_call,
                                    )
                                    function_name = function_call_dict.get("name")
                                    function_args = function_call_dict.get("args")
                                    call_args: dict[str, object] = {}
                                    if isinstance(function_args, dict):
                                        call_args = cast(
                                            "dict[str, object]",
                                            function_args,
                                        )

                                    if (
                                        provider_id == GOOGLE_ANTIGRAVITY_PROVIDER
                                        and isinstance(function_name, str)
                                    ):
                                        # Prevent upstream first-token timeout.
                                        yield "", None, False
                                        tool_result_text = await _run_image_tool(
                                            endpoint=endpoint,
                                            headers=provider_headers,
                                            credentials=credentials,
                                            messages=messages,
                                            name=function_name,
                                            args=call_args,
                                        )
                                        if tool_result_text:
                                            yield tool_result_text, None, False

                                text = part.get("text")
                                if isinstance(text, str) and text:
                                    has_content = True
                                    yield text, None, bool(part.get("thought"))

                    finish_reason = candidate.get("finishReason")
                    if finish_reason is not None:
                        has_content = True
                        yield "", finish_reason, False

                if has_content:
                    return

                # Empty stream — retry if we have attempts left
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue

                _raise_cloud_code_error(
                    "Cloud Code Assist API returned an empty response",
                )

        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_error = exc
            if attempt < max_retries:
                await asyncio.sleep(base_delay_ms * (2**attempt) / 1000)
                continue
            msg = f"Network error: {exc}"
            raise RuntimeError(msg) from exc
        except RuntimeError:
            raise
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt < max_retries:
                await asyncio.sleep(base_delay_ms * (2**attempt) / 1000)
                continue
            raise

    if last_error is not None:
        msg = f"Failed after {max_retries + 1} attempts: {last_error}"
        raise RuntimeError(msg) from last_error
    msg = "Failed to get response after retries"
    raise RuntimeError(msg)


def _generate_pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode("utf-8")).digest(),
        )
        .decode("utf-8")
        .rstrip("=")
    )
    return verifier, challenge


def _wait_for_auth_code(
    *,
    callback_host: str,
    callback_port: int,
    callback_path: str,
    timeout_seconds: int = 300,
) -> tuple[str, str]:
    code_holder: dict[str, str] = {}
    event = Event()

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != callback_path:
                self.send_response(404)
                self.end_headers()
                return

            params = urllib.parse.parse_qs(parsed.query)
            code = params.get("code", [""])[0]
            state = params.get("state", [""])[0]
            error = params.get("error", [""])[0]

            if error:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(
                    f"Authentication failed: {error}".encode(),
                )
                return

            if not code or not state:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing OAuth code or state")
                return

            code_holder["code"] = code
            code_holder["state"] = state
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                b"Authentication complete. You can close this window.",
            )
            event.set()

        def log_message(self, format: str, *args: Any) -> None:
            _ = (format, args)

    server = HTTPServer((callback_host, callback_port), CallbackHandler)

    async def run_server() -> None:
        await asyncio.to_thread(server.serve_forever)

    async def wait_event() -> None:
        await asyncio.to_thread(event.wait, timeout_seconds)

    async def wait_for_code() -> tuple[str, str]:
        server_task = asyncio.create_task(run_server())
        try:
            await wait_event()
            if not event.is_set():
                msg = "Timed out waiting for OAuth callback"
                raise TimeoutError(msg)
            return code_holder["code"], code_holder["state"]
        finally:
            server.shutdown()
            server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server_task

    return asyncio.run(wait_for_code())


async def _get_user_email(access_token: str) -> str | None:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            "https://www.googleapis.com/oauth2/v1/userinfo?alt=json",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if not response.is_success:
            return None
        payload = response.json()
        if isinstance(payload, dict) and isinstance(payload.get("email"), str):
            return payload["email"]
    return None


async def login_gemini_cli() -> GeminiCliCredentials:
    """Interactive browser login that returns reusable credentials."""
    return await _login_cloud_code_assist_provider(GOOGLE_GEMINI_CLI_PROVIDER)


async def login_antigravity() -> GeminiCliCredentials:
    """Interactive browser login that returns reusable antigravity credentials."""
    return await _login_cloud_code_assist_provider(GOOGLE_ANTIGRAVITY_PROVIDER)


def _oauth_provider_config(
    provider_id: str,
) -> tuple[str, str, str, tuple[str, ...], str, int, str]:
    if provider_id == GOOGLE_ANTIGRAVITY_PROVIDER:
        return (
            ANTIGRAVITY_CLIENT_ID,
            ANTIGRAVITY_CLIENT_SECRET,
            ANTIGRAVITY_REDIRECT_URI,
            ANTIGRAVITY_SCOPES,
            ANTIGRAVITY_CALLBACK_HOST,
            ANTIGRAVITY_CALLBACK_PORT,
            ANTIGRAVITY_CALLBACK_PATH,
        )
    return (
        GEMINI_CLI_CLIENT_ID,
        GEMINI_CLI_CLIENT_SECRET,
        GEMINI_CLI_REDIRECT_URI,
        GEMINI_CLI_SCOPES,
        GEMINI_CLI_CALLBACK_HOST,
        GEMINI_CLI_CALLBACK_PORT,
        GEMINI_CLI_CALLBACK_PATH,
    )


async def _login_cloud_code_assist_provider(provider_id: str) -> GeminiCliCredentials:
    """Interactive browser login for Google Cloud Code Assist providers."""
    (
        client_id,
        client_secret,
        redirect_uri,
        scopes,
        callback_host,
        callback_port,
        callback_path,
    ) = _oauth_provider_config(provider_id)

    verifier, challenge = _generate_pkce()
    params = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": " ".join(scopes),
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": verifier,
            "access_type": "offline",
            "prompt": "consent",
        },
    )
    auth_url = f"{AUTH_URL}?{params}"
    webbrowser.open(auth_url)

    code, state = await asyncio.to_thread(
        _wait_for_auth_code,
        callback_host=callback_host,
        callback_port=callback_port,
        callback_path=callback_path,
    )
    if state != verifier:
        msg = "OAuth state mismatch"
        raise RuntimeError(msg)

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
                "code_verifier": verifier,
            },
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            msg = "Invalid token exchange response"
            raise RuntimeError(msg)

        access_token = payload.get("access_token")
        refresh_token = payload.get("refresh_token")
        expires_in = payload.get("expires_in")
        if not isinstance(access_token, str) or not isinstance(refresh_token, str):
            msg = "OAuth token exchange did not return required tokens"
            raise RuntimeError(msg)
        if not isinstance(expires_in, int):
            msg = "OAuth token exchange did not return expires_in"
            raise RuntimeError(msg)

        email = await _get_user_email(access_token)
        project_id = await discover_project(access_token, provider_id=provider_id)
        return GeminiCliCredentials(
            refresh=refresh_token,
            access=access_token,
            expires=_now_ms() + (expires_in * 1000),
            project_id=project_id,
            email=email,
            oauth_client_id=client_id,
            oauth_client_secret=client_secret,
            oauth_token_url=TOKEN_URL,
        )


def cli_login_main() -> int:
    """CLI entrypoint for obtaining google-gemini-cli credentials JSON."""
    credentials = asyncio.run(login_gemini_cli())
    print("Login complete. Use this JSON as providers.google-gemini-cli.api_key:")
    print(credentials_to_api_key(credentials))
    return 0


def cli_login_antigravity_main() -> int:
    """CLI entrypoint for obtaining google-antigravity credentials JSON."""
    credentials = asyncio.run(login_antigravity())
    print("Login complete. Use this JSON as providers.google-antigravity.api_key:")
    print(credentials_to_api_key(credentials))
    return 0
