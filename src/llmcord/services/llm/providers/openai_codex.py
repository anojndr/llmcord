"""OpenAI Codex (ChatGPT OAuth) provider support."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import platform
import secrets
import time
import urllib.parse
import webbrowser
from collections.abc import AsyncIterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Event, Thread
from typing import Any, cast

import httpx

OPENAI_CODEX_PROVIDER = "openai-codex"

OPENAI_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_CODEX_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
OPENAI_CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
OPENAI_CODEX_REDIRECT_URI = "http://localhost:1455/auth/callback"
OPENAI_CODEX_CALLBACK_HOST = "127.0.0.1"
OPENAI_CODEX_CALLBACK_PORT = 1455
OPENAI_CODEX_CALLBACK_PATH = "/auth/callback"
OPENAI_CODEX_SCOPE = "openid profile email offline_access"
OPENAI_CODEX_DEFAULT_BASE_URL = "https://chatgpt.com/backend-api"
OPENAI_CODEX_JWT_CLAIM_PATH = "https://api.openai.com/auth"
OPENAI_CODEX_ORIGINATOR = "pi"
OPENAI_CODEX_TOKEN_EXPIRY_BUFFER_MS = 5 * 60 * 1000
OPENAI_CODEX_MAX_RETRIES = 3
OPENAI_CODEX_BASE_DELAY_SECONDS = 1.0
OPENAI_CODEX_AUTH_TIMEOUT_SECONDS = 300
OPENAI_CODEX_HTTP_TOO_MANY_REQUESTS = 429

_OPENAI_CODEX_SUCCESS_HTML = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Authentication successful</title>
</head>
<body>
  <p>Authentication successful. Return to your terminal to continue.</p>
</body>
</html>"""

_OPENAI_CODEX_RETRYABLE_ERROR_SNIPPETS = (
    "rate limit",
    "overloaded",
    "service unavailable",
    "upstream connect",
    "connection refused",
)

_CREDENTIAL_CACHE: dict[str, OpenAICodexCredentials] = {}


@dataclass(slots=True)
class OpenAICodexCredentials:
    """Credentials used by the OpenAI Codex ChatGPT endpoint."""

    refresh: str | None
    access: str | None
    expires: int | None
    account_id: str | None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _is_expired(expires_ms: int | None) -> bool:
    if expires_ms is None:
        return True
    return _now_ms() >= expires_ms - OPENAI_CODEX_TOKEN_EXPIRY_BUFFER_MS


def _base64url_decode(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(padded.encode("utf-8"))


def _decode_jwt_payload(token: str) -> dict[str, object] | None:
    parts = token.split(".")
    expected_parts = 3
    if len(parts) != expected_parts:
        return None

    payload_segment = parts[1]
    if not payload_segment:
        return None

    try:
        payload_json = _base64url_decode(payload_segment).decode("utf-8")
        parsed = json.loads(payload_json)
    except (ValueError, json.JSONDecodeError):
        return None

    return parsed if isinstance(parsed, dict) else None


def _extract_account_id_from_token(token: str) -> str | None:
    payload = _decode_jwt_payload(token)
    if not isinstance(payload, dict):
        return None

    auth_claim = payload.get(OPENAI_CODEX_JWT_CLAIM_PATH)
    if not isinstance(auth_claim, dict):
        return None
    auth_claim_dict = cast("dict[str, object]", auth_claim)

    account_id = auth_claim_dict.get("chatgpt_account_id")
    if isinstance(account_id, str) and account_id.strip():
        return account_id.strip()
    return None


def parse_api_key_credentials(
    api_key: str,
    provider_id: str = OPENAI_CODEX_PROVIDER,
) -> OpenAICodexCredentials:
    """Parse provider api_key values into structured OpenAI Codex credentials."""
    value = api_key.strip()
    if not value:
        msg = f"{provider_id} provider requires api_key credentials"
        raise ValueError(msg)

    if not value.startswith("{"):
        return OpenAICodexCredentials(
            refresh=None,
            access=value,
            expires=None,
            account_id=_extract_account_id_from_token(value),
        )

    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in {provider_id} api_key"
        raise ValueError(msg) from exc

    if not isinstance(payload, dict):
        msg = f"{provider_id} api_key JSON must be an object"
        raise TypeError(msg)

    expires_raw = payload.get("expires") or payload.get("expires_at")
    expires_ms: int | None = None
    if isinstance(expires_raw, int | float):
        expires_ms = int(expires_raw)

    access = (
        payload.get("access")
        or payload.get("access_token")
        or payload.get(
            "token",
        )
    )
    access_token = access if isinstance(access, str) and access.strip() else None

    account_id = payload.get("accountId") or payload.get("account_id")
    parsed_account_id: str | None = None
    if isinstance(account_id, str) and account_id.strip():
        parsed_account_id = account_id.strip()
    elif access_token is not None:
        parsed_account_id = _extract_account_id_from_token(access_token)

    refresh = payload.get("refresh") or payload.get("refresh_token")
    refresh_token = refresh if isinstance(refresh, str) and refresh.strip() else None

    return OpenAICodexCredentials(
        refresh=refresh_token,
        access=access_token,
        expires=expires_ms,
        account_id=parsed_account_id,
    )


def credentials_to_api_key(credentials: OpenAICodexCredentials) -> str:
    """Serialize OpenAI Codex credentials for config storage."""
    payload: dict[str, str | int | None] = {
        "refresh": credentials.refresh,
        "access": credentials.access,
        "expires": credentials.expires,
        "accountId": credentials.account_id,
    }
    return json.dumps(payload, separators=(",", ":"))


def _generate_pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("utf-8")).digest())
        .decode("utf-8")
        .rstrip("=")
    )
    return verifier, challenge


def _parse_authorization_input(value: str) -> tuple[str | None, str | None]:
    stripped = value.strip()
    if not stripped:
        return None, None

    try:
        parsed_url = urllib.parse.urlparse(stripped)
        if parsed_url.scheme and parsed_url.netloc:
            params = urllib.parse.parse_qs(parsed_url.query)
            code = cast("str | None", params.get("code", [None])[0])
            state = cast("str | None", params.get("state", [None])[0])
            if code:
                return code, state

            fragment_params = urllib.parse.parse_qs(parsed_url.fragment)
            fragment_code = cast("str | None", fragment_params.get("code", [None])[0])
            fragment_state = cast(
                "str | None",
                fragment_params.get("state", [None])[0],
            )
            if fragment_code:
                return fragment_code, fragment_state
    except ValueError:
        pass

    if "#" in stripped:
        code, state = stripped.split("#", 1)
        return code or None, state or None

    if "code=" in stripped:
        params = urllib.parse.parse_qs(stripped)
        return (
            cast("str | None", params.get("code", [None])[0]),
            cast("str | None", params.get("state", [None])[0]),
        )

    return stripped, None


def _wait_for_auth_code(expected_state: str) -> str | None:
    code_holder: dict[str, str] = {}
    callback_event = Event()

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed_url = urllib.parse.urlparse(self.path)
            if parsed_url.path != OPENAI_CODEX_CALLBACK_PATH:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found")
                return

            params = urllib.parse.parse_qs(parsed_url.query)
            callback_state = params.get("state", [""])[0]
            code = params.get("code", [""])[0]

            if callback_state != expected_state:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"State mismatch")
                return

            if not code:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing authorization code")
                return

            code_holder["code"] = code
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_OPENAI_CODEX_SUCCESS_HTML.encode("utf-8"))
            callback_event.set()

        def log_message(self, format: str, *args: Any) -> None:
            _ = (format, args)

    try:
        server = HTTPServer(
            (OPENAI_CODEX_CALLBACK_HOST, OPENAI_CODEX_CALLBACK_PORT),
            CallbackHandler,
        )
    except OSError as exc:
        error_code = getattr(exc, "errno", "unknown")
        msg = (
            "Failed to bind OpenAI Codex callback server on "
            f"http://{OPENAI_CODEX_CALLBACK_HOST}:{OPENAI_CODEX_CALLBACK_PORT} "
            f"({error_code}); falling back to manual code paste."
        )
        raise RuntimeError(msg) from exc

    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        callback_event.wait(OPENAI_CODEX_AUTH_TIMEOUT_SECONDS)
        return code_holder.get("code")
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1)


async def _exchange_authorization_code(
    code: str,
    verifier: str,
    redirect_uri: str = OPENAI_CODEX_REDIRECT_URI,
) -> OpenAICodexCredentials:
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            OPENAI_CODEX_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "authorization_code",
                "client_id": OPENAI_CODEX_CLIENT_ID,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
            },
        )

    if not response.is_success:
        message = _format_error_message(response.status_code, response.text)
        msg = f"OpenAI Codex code->token exchange failed: {message}"
        raise RuntimeError(msg)

    payload = response.json()
    if not isinstance(payload, dict):
        msg = "Invalid token exchange response from OpenAI Codex"
        raise TypeError(msg)

    access_token = payload.get("access_token")
    refresh_token = payload.get("refresh_token")
    expires_in = payload.get("expires_in")

    if not isinstance(access_token, str) or not access_token:
        msg = "OpenAI Codex token exchange did not return access_token"
        raise RuntimeError(msg)
    if not isinstance(refresh_token, str) or not refresh_token:
        msg = "OpenAI Codex token exchange did not return refresh_token"
        raise RuntimeError(msg)
    if not isinstance(expires_in, int):
        msg = "OpenAI Codex token exchange did not return expires_in"
        raise TypeError(msg)

    account_id = _extract_account_id_from_token(access_token)
    if not account_id:
        msg = "Failed to extract accountId from OpenAI Codex access token"
        raise RuntimeError(msg)

    return OpenAICodexCredentials(
        refresh=refresh_token,
        access=access_token,
        expires=_now_ms() + (expires_in * 1000),
        account_id=account_id,
    )


async def _refresh_access_token(refresh_token: str) -> OpenAICodexCredentials:
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            OPENAI_CODEX_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": OPENAI_CODEX_CLIENT_ID,
            },
        )

    if not response.is_success:
        message = _format_error_message(response.status_code, response.text)
        msg = f"OpenAI Codex token refresh failed: {message}"
        raise RuntimeError(msg)

    payload = response.json()
    if not isinstance(payload, dict):
        msg = "Invalid OpenAI Codex refresh response"
        raise TypeError(msg)

    access_token = payload.get("access_token")
    maybe_refresh = payload.get("refresh_token")
    expires_in = payload.get("expires_in")

    if not isinstance(access_token, str) or not access_token:
        msg = "OpenAI Codex refresh did not return access_token"
        raise RuntimeError(msg)
    if not isinstance(expires_in, int):
        msg = "OpenAI Codex refresh did not return expires_in"
        raise TypeError(msg)

    account_id = _extract_account_id_from_token(access_token)
    if not account_id:
        msg = "Failed to extract accountId from refreshed OpenAI Codex token"
        raise RuntimeError(msg)

    refreshed_token = maybe_refresh if isinstance(maybe_refresh, str) else refresh_token
    return OpenAICodexCredentials(
        refresh=refreshed_token,
        access=access_token,
        expires=_now_ms() + (expires_in * 1000),
        account_id=account_id,
    )


async def login_openai_codex(
    originator: str = OPENAI_CODEX_ORIGINATOR,
) -> OpenAICodexCredentials:
    """Run interactive OAuth login for OpenAI Codex and return credentials."""
    verifier, challenge = _generate_pkce()
    state = secrets.token_hex(16)

    authorize_url = urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": OPENAI_CODEX_CLIENT_ID,
            "redirect_uri": OPENAI_CODEX_REDIRECT_URI,
            "scope": OPENAI_CODEX_SCOPE,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "originator": originator,
        },
    )
    auth_url = f"{OPENAI_CODEX_AUTHORIZE_URL}?{authorize_url}"

    webbrowser.open(auth_url)

    code: str | None = None
    try:
        code = await asyncio.to_thread(_wait_for_auth_code, state)
    except RuntimeError:
        code = None

    if not code:
        prompt = "Paste the authorization code (or full redirect URL): "
        manual_input = await asyncio.to_thread(input, prompt)
        parsed_code, parsed_state = _parse_authorization_input(manual_input)
        if parsed_state and parsed_state != state:
            msg = "OAuth state mismatch"
            raise RuntimeError(msg)
        code = parsed_code

    if not code:
        msg = "Missing authorization code"
        raise RuntimeError(msg)

    return await _exchange_authorization_code(code=code, verifier=verifier)


async def get_valid_openai_codex_credentials(api_key: str) -> OpenAICodexCredentials:
    """Return valid access credentials for OpenAI Codex requests."""
    cached = _CREDENTIAL_CACHE.get(api_key)
    credentials = cached or parse_api_key_credentials(api_key, OPENAI_CODEX_PROVIDER)

    has_access = isinstance(credentials.access, str) and bool(credentials.access)
    has_account = isinstance(credentials.account_id, str) and bool(
        credentials.account_id,
    )
    if has_access and has_account and not _is_expired(credentials.expires):
        _CREDENTIAL_CACHE[api_key] = credentials
        return credentials

    if not credentials.refresh:
        msg = (
            "openai-codex api_key must include a refresh token if access token "
            "is missing, expired, or missing accountId"
        )
        raise RuntimeError(msg)

    refreshed = await _refresh_access_token(credentials.refresh)
    _CREDENTIAL_CACHE[api_key] = refreshed
    return refreshed


def _resolve_codex_url(base_url: str | None) -> str:
    raw_base_url = base_url if isinstance(base_url, str) and base_url.strip() else ""
    normalized = (raw_base_url or OPENAI_CODEX_DEFAULT_BASE_URL).rstrip("/")
    if normalized.endswith("/codex/responses"):
        return normalized
    if normalized.endswith("/codex"):
        return f"{normalized}/responses"
    return f"{normalized}/codex/responses"


def _build_headers(
    *,
    token: str,
    account_id: str,
    extra_headers: dict[str, str] | None,
) -> dict[str, str]:
    user_agent = (
        "llmcord "
        f"({platform.system().lower()} {platform.release()}; "
        f"{platform.machine().lower()})"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "originator": OPENAI_CODEX_ORIGINATOR,
        "User-Agent": user_agent,
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    return headers


def _convert_message_parts(
    *,
    role: str,
    raw_content: object,
) -> list[dict[str, str]]:
    parts: list[dict[str, str]] = []

    def append_text_part(text: str) -> None:
        if not text:
            return
        part_type = "output_text" if role == "assistant" else "input_text"
        parts.append({"type": part_type, "text": text})

    if isinstance(raw_content, str):
        append_text_part(raw_content)
        return parts

    if not isinstance(raw_content, list):
        return parts

    for item in raw_content:
        if not isinstance(item, dict):
            continue
        item_dict = cast("dict[str, object]", item)
        item_type = item_dict.get("type")

        if item_type == "text":
            text = item_dict.get("text")
            if isinstance(text, str):
                append_text_part(text)
            continue

        if item_type != "image_url" or role == "assistant":
            continue

        image_url_obj = item_dict.get("image_url")
        if not isinstance(image_url_obj, dict):
            continue
        image_url = cast("dict[str, object]", image_url_obj).get("url")
        if isinstance(image_url, str) and image_url:
            parts.append({"type": "input_image", "image_url": image_url})

    return parts


def _convert_messages_for_codex(
    messages: list[dict[str, object]],
) -> tuple[list[dict[str, object]], str | None]:
    converted: list[dict[str, object]] = []
    instructions: str | None = None

    for message in messages:
        role_value = message.get("role")
        role = role_value.lower() if isinstance(role_value, str) else "user"

        if role == "system":
            system_content = message.get("content")
            if isinstance(system_content, str) and system_content.strip():
                instructions = system_content.strip()
            continue

        api_role = "assistant" if role == "assistant" else "user"
        parts = _convert_message_parts(
            role=api_role,
            raw_content=message.get("content"),
        )
        if not parts:
            continue

        converted.append({"role": api_role, "content": parts})

    return converted, instructions


_REASONING_EFFORT_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("-none", "none"),
    ("-minimal", "minimal"),
    ("-low", "low"),
    ("-medium", "medium"),
    ("-high", "high"),
    ("-xhigh", "xhigh"),
)


def _normalize_reasoning_effort(effort: str) -> str | None:
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    normalized = effort.strip().lower()
    return normalized if normalized in allowed else None


def _extract_reasoning_effort_alias(model: str) -> str | None:
    for suffix, effort in _REASONING_EFFORT_SUFFIXES:
        if model.endswith(suffix):
            return effort
    return None


def _strip_reasoning_effort_suffix(model: str) -> str:
    for suffix, _effort in _REASONING_EFFORT_SUFFIXES:
        if model.endswith(suffix):
            return model.removesuffix(suffix)
    return model


def _extract_reasoning_effort(
    *,
    model: str,
    model_parameters: dict[str, object] | None,
) -> str | None:
    if model_parameters:
        raw_effort = model_parameters.get("reasoning_effort") or model_parameters.get(
            "reasoningEffort",
        )
        if isinstance(raw_effort, str):
            normalized = _normalize_reasoning_effort(raw_effort)
            if normalized is not None:
                return normalized

    return _extract_reasoning_effort_alias(model)


def _clamp_reasoning_effort(model: str, effort: str) -> str:
    model_id = model.split("/", 1)[1] if "/" in model else model
    if model_id.startswith(("gpt-5.2", "gpt-5.3")) and effort == "minimal":
        return "low"
    if model_id == "gpt-5.1" and effort == "xhigh":
        return "high"
    if model_id == "gpt-5.1-codex-mini":
        return "high" if effort in {"high", "xhigh"} else "medium"
    return effort


def _build_codex_request(
    *,
    model: str,
    messages: list[dict[str, object]],
    model_parameters: dict[str, object] | None,
    disable_tools: bool,
) -> dict[str, object]:
    input_messages, instructions = _convert_messages_for_codex(messages)
    clean_model = _strip_reasoning_effort_suffix(model)

    body: dict[str, object] = {
        "model": clean_model,
        "store": False,
        "stream": True,
        "input": input_messages,
        "text": {"verbosity": "medium"},
        "include": ["reasoning.encrypted_content"],
    }

    if instructions:
        body["instructions"] = instructions

    if not disable_tools:
        body["tool_choice"] = "auto"
        body["parallel_tool_calls"] = True

    temperature: object | None = None
    reasoning_summary_raw: object | None = None
    text_verbosity_raw: object | None = None

    if model_parameters:
        temperature = model_parameters.get("temperature")
        reasoning_summary_raw = model_parameters.get("reasoning_summary")
        text_verbosity_raw = model_parameters.get("text_verbosity")

    if isinstance(temperature, int | float):
        body["temperature"] = float(temperature)

    reasoning_effort = _extract_reasoning_effort(
        model=model,
        model_parameters=model_parameters,
    )
    reasoning_payload: dict[str, str] = {}
    if reasoning_effort is not None:
        reasoning_payload["effort"] = _clamp_reasoning_effort(
            clean_model,
            reasoning_effort,
        )

    if isinstance(reasoning_summary_raw, str) and reasoning_summary_raw.strip():
        reasoning_payload["summary"] = reasoning_summary_raw.strip().lower()
    else:
        # Request summary output by default so hidden thought process can be shown.
        reasoning_payload["summary"] = "auto"

    if reasoning_payload:
        body["reasoning"] = reasoning_payload

    if isinstance(text_verbosity_raw, str) and text_verbosity_raw.strip():
        text_verbosity = text_verbosity_raw.strip().lower()
        if text_verbosity in {"low", "medium", "high"}:
            body["text"] = {"verbosity": text_verbosity}

    return body


def _is_retryable_error(status_code: int, error_text: str) -> bool:
    if status_code in {429, 500, 502, 503, 504}:
        return True

    error_lower = error_text.lower()
    return any(token in error_lower for token in _OPENAI_CODEX_RETRYABLE_ERROR_SNIPPETS)


def _format_error_message(status_code: int, error_text: str) -> str:
    message = error_text.strip() or f"HTTP {status_code}"

    try:
        parsed = json.loads(error_text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        error_obj = parsed.get("error")
        if isinstance(error_obj, dict):
            parsed_message = error_obj.get("message")
            if isinstance(parsed_message, str) and parsed_message.strip():
                message = parsed_message.strip()
            code = error_obj.get("code") or error_obj.get("type")
            if isinstance(code, str):
                code_lower = code.lower()
                usage_error = {
                    "usage_limit_reached",
                    "usage_not_included",
                    "rate_limit_exceeded",
                }
                if (
                    code_lower in usage_error
                    or status_code == OPENAI_CODEX_HTTP_TOO_MANY_REQUESTS
                ):
                    return "You have hit your ChatGPT usage limit."

    if status_code == OPENAI_CODEX_HTTP_TOO_MANY_REQUESTS:
        return "You have hit your ChatGPT usage limit."

    return message


def _extract_error_from_event(event: dict[str, object]) -> str | None:
    event_type = event.get("type")
    if not isinstance(event_type, str):
        return None

    if event_type == "error":
        code = event.get("code")
        message = event.get("message")
        if isinstance(message, str) and message:
            return message
        if isinstance(code, str) and code:
            return code
        return "OpenAI Codex stream error"

    if event_type == "response.failed":
        response_obj = event.get("response")
        if isinstance(response_obj, dict):
            response_data = cast("dict[str, object]", response_obj)
            error_obj = response_data.get("error")
            if isinstance(error_obj, dict):
                error_data = cast("dict[str, object]", error_obj)
                message = error_data.get("message")
                if isinstance(message, str) and message:
                    return message
        return "OpenAI Codex response failed"

    return None


def _extract_finish_reason(event: dict[str, object]) -> str | None:
    event_type = event.get("type")
    if event_type not in {"response.completed", "response.done"}:
        return None

    response_obj = event.get("response")
    if not isinstance(response_obj, dict):
        return "stop"
    response_data = cast("dict[str, object]", response_obj)

    status = response_data.get("status")
    normalized = status.lower() if isinstance(status, str) else ""

    if normalized == "failed":
        error_obj = response_data.get("error")
        if isinstance(error_obj, dict):
            error_data = cast("dict[str, object]", error_obj)
            message = error_data.get("message")
            if isinstance(message, str) and message:
                raise RuntimeError(message)
        failed_message = "OpenAI Codex response failed"
        raise RuntimeError(failed_message)

    if normalized == "cancelled":
        cancelled_message = "OpenAI Codex response was cancelled"
        raise RuntimeError(cancelled_message)

    if normalized == "incomplete":
        details_obj = response_data.get("incomplete_details")
        if isinstance(details_obj, dict):
            details_data = cast("dict[str, object]", details_obj)
            reason = details_data.get("reason")
            if isinstance(reason, str) and "max_output_tokens" in reason.lower():
                return "length"
        return "stop"

    return "stop"


def _extract_reasoning_text(event: dict[str, object]) -> str | None:
    for key in ("delta", "text", "summary"):
        value = event.get(key)
        if isinstance(value, str):
            return value

    part = event.get("part")
    if isinstance(part, dict):
        part_dict = cast("dict[str, object]", part)
        for key in ("delta", "text", "summary"):
            value = part_dict.get(key)
            if isinstance(value, str):
                return value

    return None


def _extract_text_delta(
    event: dict[str, object],
    *,
    seen_output_items: set[str],
    seen_reasoning_items: set[str],
) -> tuple[str | None, bool]:
    event_type = event.get("type")
    if not isinstance(event_type, str):
        return None, False

    item_id = event.get("item_id")
    item_key = item_id if isinstance(item_id, str) else None

    if event_type == "response.output_text.delta":
        if item_key:
            seen_output_items.add(item_key)
        delta = event.get("delta")
        return (delta, False) if isinstance(delta, str) else (None, False)

    if event_type == "response.output_text.done":
        if item_key and item_key in seen_output_items:
            return None, False
        text = event.get("text")
        return (text, False) if isinstance(text, str) else (None, False)

    if "reasoning" in event_type and "encrypted_content" not in event_type:
        if item_key and event_type.endswith((".delta", ".added")):
            seen_reasoning_items.add(item_key)

        if (
            item_key
            and event_type.endswith(".done")
            and item_key in seen_reasoning_items
        ):
            return None, True

        reasoning_text = _extract_reasoning_text(event)
        if isinstance(reasoning_text, str):
            return reasoning_text, True
        return None, True

    return None, False


async def _iter_sse_events(
    response: httpx.Response,
) -> AsyncIterator[dict[str, object]]:
    buffer = ""

    async for chunk in response.aiter_text():
        if not chunk:
            continue
        buffer += chunk

        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            data_lines = [
                line[5:].strip()
                for line in block.splitlines()
                if line.startswith("data:")
            ]
            if not data_lines:
                continue

            payload = "\n".join(data_lines).strip()
            if not payload or payload == "[DONE]":
                continue

            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if isinstance(parsed, dict):
                yield cast("dict[str, object]", parsed)


async def stream_openai_codex(
    *,
    model: str,
    messages: list[dict[str, object]],
    api_key: str,
    base_url: str | None,
    extra_headers: dict[str, str] | None,
    model_parameters: dict[str, object] | None,
    disable_tools: bool = False,
) -> AsyncIterator[tuple[str, object | None, bool]]:
    """Stream text deltas from OpenAI Codex responses SSE endpoint."""
    credentials = await get_valid_openai_codex_credentials(api_key)
    if not credentials.access or not credentials.account_id:
        msg = "Missing OpenAI Codex access token or account id"
        raise RuntimeError(msg)

    body = _build_codex_request(
        model=model,
        messages=messages,
        model_parameters=model_parameters,
        disable_tools=disable_tools,
    )
    url = _resolve_codex_url(base_url)
    headers = _build_headers(
        token=credentials.access,
        account_id=credentials.account_id,
        extra_headers=extra_headers,
    )

    last_error: Exception | None = None

    for attempt in range(OPENAI_CODEX_MAX_RETRIES + 1):
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
                    error_text = error_bytes.decode("utf-8", errors="replace")
                    if attempt < OPENAI_CODEX_MAX_RETRIES and _is_retryable_error(
                        response.status_code,
                        error_text,
                    ):
                        delay = OPENAI_CODEX_BASE_DELAY_SECONDS * (2**attempt)
                        await asyncio.sleep(delay)
                        continue

                    message = _format_error_message(response.status_code, error_text)
                    msg = f"OpenAI Codex API error ({response.status_code}): {message}"
                    raise RuntimeError(msg)

                saw_content = False
                seen_output_items: set[str] = set()
                seen_reasoning_items: set[str] = set()

                async for event in _iter_sse_events(response):
                    maybe_error = _extract_error_from_event(event)
                    if maybe_error:
                        raise RuntimeError(maybe_error)

                    delta_text, is_thinking = _extract_text_delta(
                        event,
                        seen_output_items=seen_output_items,
                        seen_reasoning_items=seen_reasoning_items,
                    )
                    if isinstance(delta_text, str) and delta_text:
                        saw_content = True
                        yield delta_text, None, is_thinking
                        continue

                    finish_reason = _extract_finish_reason(event)
                    if finish_reason is not None:
                        saw_content = True
                        yield "", finish_reason, False
                        return

                if saw_content:
                    yield "", "stop", False
                    return

                if attempt < OPENAI_CODEX_MAX_RETRIES:
                    delay = OPENAI_CODEX_BASE_DELAY_SECONDS * (2**attempt)
                    await asyncio.sleep(delay)
                    continue

                msg = "OpenAI Codex API returned an empty response"
                raise RuntimeError(msg)

        except (httpx.NetworkError, httpx.TimeoutException) as exc:
            last_error = exc
            if attempt < OPENAI_CODEX_MAX_RETRIES:
                delay = OPENAI_CODEX_BASE_DELAY_SECONDS * (2**attempt)
                await asyncio.sleep(delay)
                continue
            msg = f"Network error: {exc}"
            raise RuntimeError(msg) from exc
        except RuntimeError:
            raise
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt < OPENAI_CODEX_MAX_RETRIES:
                delay = OPENAI_CODEX_BASE_DELAY_SECONDS * (2**attempt)
                await asyncio.sleep(delay)
                continue
            msg = f"HTTP error: {exc}"
            raise RuntimeError(msg) from exc

    if last_error is not None:
        msg = (
            f"OpenAI Codex failed after {OPENAI_CODEX_MAX_RETRIES + 1} "
            f"attempts: {last_error}"
        )
        raise RuntimeError(msg) from last_error

    msg = "OpenAI Codex failed after retries"
    raise RuntimeError(msg)


def cli_login_openai_codex_main() -> int:
    """CLI entrypoint for obtaining openai-codex credentials JSON."""
    credentials = asyncio.run(login_openai_codex())
    output_lines = [
        "Login complete. Use this JSON as providers.openai-codex.api_key:",
        credentials_to_api_key(credentials),
    ]
    for line in output_lines:
        print(line)  # noqa: T201
    return 0
