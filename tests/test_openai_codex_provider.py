from __future__ import annotations

import base64
import json
from typing import Self, cast

import pytest

from llmcord.services.llm.providers.openai_codex import (
    OPENAI_CODEX_PROVIDER,
    OpenAICodexCredentials,
    _build_codex_request,
    credentials_to_api_key,
    parse_api_key_credentials,
    stream_openai_codex,
)

EXPECTED_EXPIRES_MS = 1_700_000_000_000


def _to_base64url(value: dict[str, object]) -> str:
    encoded = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(encoded).decode("utf-8").rstrip("=")


def _build_jwt(account_id: str) -> str:
    header = _to_base64url({"alg": "none", "typ": "JWT"})
    payload = _to_base64url(
        {
            "https://api.openai.com/auth": {
                "chatgpt_account_id": account_id,
            },
        },
    )
    return f"{header}.{payload}.signature"


def _build_fake_async_client_class(  # noqa: C901
    *,
    sse_chunks: list[str],
    captured_request: dict[str, object],
) -> type[object]:
    class _FakeStreamResponse:
        def __init__(self) -> None:
            self.is_success = True
            self.status_code = 200

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def aread(self) -> bytes:
            return b""

        async def aiter_text(self):
            for chunk in sse_chunks:
                yield chunk

    class _FakeAsyncClient:
        def __init__(self, timeout: int) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        def stream(
            self,
            method: str,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, object],
        ) -> _FakeStreamResponse:
            captured_request.update(
                {
                    "method": method,
                    "url": url,
                    "headers": headers,
                    "json": json,
                },
            )
            return _FakeStreamResponse()

    return _FakeAsyncClient


def test_parse_openai_codex_api_key_json_roundtrip() -> None:
    access_token = _build_jwt("account-123")
    raw = json.dumps(
        {
            "refresh": "refresh-token",
            "access": access_token,
            "expires": EXPECTED_EXPIRES_MS,
            "accountId": "account-123",
        },
    )

    parsed = parse_api_key_credentials(raw, OPENAI_CODEX_PROVIDER)

    assert parsed.refresh == "refresh-token"
    assert parsed.access == access_token
    assert parsed.expires == EXPECTED_EXPIRES_MS
    assert parsed.account_id == "account-123"

    serialized = credentials_to_api_key(parsed)
    loaded = json.loads(serialized)
    assert loaded["refresh"] == "refresh-token"
    assert loaded["accountId"] == "account-123"


def test_parse_openai_codex_raw_access_token_extracts_account_id() -> None:
    access_token = _build_jwt("account-xyz")

    parsed = parse_api_key_credentials(access_token, OPENAI_CODEX_PROVIDER)

    assert parsed.refresh is None
    assert parsed.access == access_token
    assert parsed.account_id == "account-xyz"


def test_build_codex_request_converts_system_text_and_images() -> None:
    body = _build_codex_request(
        model="gpt-5.2",
        messages=[
            {
                "role": "system",
                "content": "system prompt",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,AAAB",
                        },
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "world",
            },
        ],
        model_parameters={"temperature": 0.7, "text_verbosity": "high"},
        disable_tools=False,
    )

    assert body["model"] == "gpt-5.2"
    assert body["instructions"] == "system prompt"
    assert body["temperature"] == 0.7
    assert body["tool_choice"] == "auto"

    input_messages = body["input"]
    assert isinstance(input_messages, list)
    first = cast("dict[str, object]", input_messages[0])
    assert first["role"] == "user"

    first_content = first["content"]
    assert isinstance(first_content, list)
    assert {"type": "input_text", "text": "hello"} in first_content
    assert {
        "type": "input_image",
        "image_url": "data:image/png;base64,AAAB",
    } in first_content


def test_build_codex_request_uses_reasoning_effort_alias() -> None:
    body = _build_codex_request(
        model="gpt-5.2-xhigh",
        messages=[{"role": "user", "content": "hello"}],
        model_parameters=None,
        disable_tools=False,
    )

    assert body["model"] == "gpt-5.2"
    reasoning = body.get("reasoning")
    assert isinstance(reasoning, dict)
    reasoning_dict = cast("dict[str, object]", reasoning)
    assert reasoning_dict.get("effort") == "xhigh"
    assert reasoning_dict.get("summary") == "auto"


def test_build_codex_request_uses_reasoning_effort_alias_for_gpt_5_4() -> None:
    body = _build_codex_request(
        model="gpt-5.4-xhigh",
        messages=[{"role": "user", "content": "hello"}],
        model_parameters=None,
        disable_tools=False,
    )

    assert body["model"] == "gpt-5.4"
    reasoning = body.get("reasoning")
    assert isinstance(reasoning, dict)
    reasoning_dict = cast("dict[str, object]", reasoning)
    assert reasoning_dict.get("effort") == "xhigh"
    assert reasoning_dict.get("summary") == "auto"


def test_build_codex_request_clamps_gpt_5_4_minimal_reasoning_to_low() -> None:
    body = _build_codex_request(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hello"}],
        model_parameters={"reasoning_effort": "minimal"},
        disable_tools=False,
    )

    reasoning = body.get("reasoning")
    assert isinstance(reasoning, dict)
    reasoning_dict = cast("dict[str, object]", reasoning)
    assert reasoning_dict.get("effort") == "low"
    assert reasoning_dict.get("summary") == "auto"


@pytest.mark.asyncio
async def test_stream_openai_codex_streams_sse_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_credentials(_api_key: str) -> OpenAICodexCredentials:
        return OpenAICodexCredentials(
            refresh="refresh-token",
            access=_build_jwt("account-123"),
            expires=EXPECTED_EXPIRES_MS,
            account_id="account-123",
        )

    captured_request: dict[str, object] = {}
    monkeypatch.setattr(
        "llmcord.services.llm.providers.openai_codex.get_valid_openai_codex_credentials",
        _fake_credentials,
    )
    monkeypatch.setattr(
        "llmcord.services.llm.providers.openai_codex.httpx.AsyncClient",
        _build_fake_async_client_class(
            sse_chunks=[
                (
                    'data: {"type":"response.output_text.delta",'
                    '"item_id":"item-1","delta":"hello "}\n\n'
                ),
                (
                    'data: {"type":"response.output_text.delta",'
                    '"item_id":"item-1","delta":"world"}\n\n'
                ),
                (
                    'data: {"type":"response.completed",'
                    '"response":{"status":"completed"}}\n\n'
                ),
            ],
            captured_request=captured_request,
        ),
    )

    chunks = [
        chunk
        async for chunk in stream_openai_codex(
            model="gpt-5.2",
            messages=[{"role": "user", "content": "hello"}],
            api_key="api-key",
            base_url="https://chatgpt.com/backend-api",
            extra_headers=None,
            model_parameters=None,
        )
    ]

    assert chunks[0] == ("hello ", None, False)
    assert chunks[1] == ("world", None, False)
    assert chunks[-1] == ("", "stop", False)

    assert captured_request["method"] == "POST"
    assert str(captured_request["url"]).endswith("/codex/responses")

    headers = cast("dict[str, str]", captured_request["headers"])
    assert headers["chatgpt-account-id"] == "account-123"

    body = cast("dict[str, object]", captured_request["json"])
    assert body["model"] == "gpt-5.2"


@pytest.mark.asyncio
async def test_stream_openai_codex_marks_reasoning_chunks_as_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_credentials(_api_key: str) -> OpenAICodexCredentials:
        return OpenAICodexCredentials(
            refresh="refresh-token",
            access=_build_jwt("account-123"),
            expires=EXPECTED_EXPIRES_MS,
            account_id="account-123",
        )

    monkeypatch.setattr(
        "llmcord.services.llm.providers.openai_codex.get_valid_openai_codex_credentials",
        _fake_credentials,
    )
    monkeypatch.setattr(
        "llmcord.services.llm.providers.openai_codex.httpx.AsyncClient",
        _build_fake_async_client_class(
            sse_chunks=[
                (
                    'data: {"type":"response.reasoning_summary.delta",'
                    '"item_id":"item-r1","delta":"thinking..."}\n\n'
                ),
                (
                    'data: {"type":"response.output_text.delta",'
                    '"item_id":"item-1","delta":"answer"}\n\n'
                ),
                (
                    'data: {"type":"response.completed",'
                    '"response":{"status":"completed"}}\n\n'
                ),
            ],
            captured_request={},
        ),
    )

    chunks = [
        chunk
        async for chunk in stream_openai_codex(
            model="gpt-5.2",
            messages=[{"role": "user", "content": "hello"}],
            api_key="api-key",
            base_url="https://chatgpt.com/backend-api",
            extra_headers=None,
            model_parameters=None,
        )
    ]

    assert chunks[0] == ("thinking...", None, True)
    assert chunks[1] == ("answer", None, False)
