import json
from typing import cast

import pytest

from llmcord.services.llm.providers.gemini_cli import (
    GOOGLE_ANTIGRAVITY_PROVIDER,
    GOOGLE_GEMINI_CLI_PROVIDER,
    GeminiCliCredentials,
    _build_cloudcode_request,
    _execute_antigravity_image_tool_call,
    credentials_to_api_key,
    parse_api_key_credentials,
)

EXPECTED_EXPIRES_MS = 1_700_000_000_000
EXPECTED_CONTENT_COUNT = 2
EXPECTED_IMAGE_CONTEXT_MESSAGES = 3


def test_parse_google_gemini_cli_api_key_json_roundtrip() -> None:
    raw = json.dumps(
        {
            "refresh": "refresh-token",
            "access": "access-token",
            "expires": EXPECTED_EXPIRES_MS,
            "projectId": "project-123",
            "email": "user@example.com",
        },
    )

    parsed = parse_api_key_credentials(raw, GOOGLE_GEMINI_CLI_PROVIDER)

    assert parsed.refresh == "refresh-token"
    assert parsed.access == "access-token"
    assert parsed.expires == EXPECTED_EXPIRES_MS
    assert parsed.project_id == "project-123"
    assert parsed.email == "user@example.com"

    serialized = credentials_to_api_key(parsed)
    loaded = json.loads(serialized)
    assert loaded["refresh"] == "refresh-token"
    assert loaded["projectId"] == "project-123"


def test_parse_google_gemini_cli_authorized_user_fields() -> None:
    raw = json.dumps(
        {
            "type": "authorized_user",
            "client_id": "oauth-client-id",
            "client_secret": "oauth-client-secret",
            "refresh_token": "refresh-token",
            "token_uri": "https://oauth2.googleapis.com/token",
        },
    )

    parsed = parse_api_key_credentials(raw, GOOGLE_GEMINI_CLI_PROVIDER)

    assert parsed.refresh == "refresh-token"
    assert parsed.oauth_client_id == "oauth-client-id"
    assert isinstance(parsed.oauth_client_secret, str)
    assert parsed.oauth_client_secret.endswith("secret")
    assert isinstance(parsed.oauth_token_url, str)
    assert parsed.oauth_token_url.startswith("https://")

    serialized = credentials_to_api_key(parsed)
    loaded = json.loads(serialized)
    assert loaded["client_id"] == "oauth-client-id"
    assert isinstance(loaded["client_secret"], str)
    assert str(loaded["client_secret"]).endswith("secret")
    assert isinstance(loaded["token_uri"], str)
    assert str(loaded["token_uri"]).startswith("https://")


def test_parse_google_gemini_cli_api_key_plain_refresh_token() -> None:
    parsed = parse_api_key_credentials(
        "refresh-only-token",
        GOOGLE_GEMINI_CLI_PROVIDER,
    )

    assert parsed.refresh == "refresh-only-token"
    assert parsed.access is None
    assert parsed.project_id is None


def test_build_cloudcode_request_converts_multipart_content() -> None:
    request = _build_cloudcode_request(
        provider_id=GOOGLE_GEMINI_CLI_PROVIDER,
        model="gemini-3-flash-preview",
        project_id="project-123",
        messages=[
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
            {
                "role": "system",
                "content": "system prompt",
            },
        ],
        model_parameters={"thinking_level": "minimal"},
    )

    assert request["project"] == "project-123"
    assert request["model"] == "gemini-3-flash-preview"

    payload = request["request"]
    assert isinstance(payload, dict)
    payload_dict = cast("dict[str, object]", payload)
    contents = payload_dict["contents"]
    assert isinstance(contents, list)
    assert len(contents) == EXPECTED_CONTENT_COUNT

    first_content = contents[0]
    assert isinstance(first_content, dict)
    first_content_dict = cast("dict[str, object]", first_content)
    first_parts = first_content_dict["parts"]
    assert isinstance(first_parts, list)
    assert {"text": "hello"} in first_parts
    assert {"inlineData": {"mimeType": "image/png", "data": "AAAB"}} in first_parts

    system_instruction = payload_dict["systemInstruction"]
    assert system_instruction == {"parts": [{"text": "system prompt"}]}


def test_build_cloudcode_request_strips_thinking_suffix_from_model() -> None:
    request = _build_cloudcode_request(
        provider_id=GOOGLE_GEMINI_CLI_PROVIDER,
        model="gemini-3-flash-preview-high",
        project_id="project-123",
        messages=[{"role": "user", "content": "hello"}],
        model_parameters=None,
    )

    assert request["model"] == "gemini-3-flash-preview"

    payload = request["request"]
    assert isinstance(payload, dict)
    payload_dict = cast("dict[str, object]", payload)
    generation_config = payload_dict["generationConfig"]
    assert isinstance(generation_config, dict)
    generation_config_dict = cast("dict[str, object]", generation_config)
    thinking_config = generation_config_dict["thinkingConfig"]
    assert isinstance(thinking_config, dict)
    thinking_config_dict = cast("dict[str, object]", thinking_config)
    assert thinking_config_dict["thinkingLevel"] == "HIGH"


def test_build_cloudcode_request_antigravity_adds_agent_fields() -> None:
    request = _build_cloudcode_request(
        provider_id=GOOGLE_ANTIGRAVITY_PROVIDER,
        model="gemini-3-pro",
        project_id="project-123",
        messages=[{"role": "user", "content": "hello"}],
        model_parameters=None,
    )

    assert request["requestType"] == "agent"
    assert request["userAgent"] == "antigravity"

    payload = request["request"]
    assert isinstance(payload, dict)
    payload_dict = cast("dict[str, object]", payload)
    system_instruction = payload_dict["systemInstruction"]
    assert isinstance(system_instruction, dict)
    instruction_dict = cast("dict[str, object]", system_instruction)
    assert instruction_dict.get("role") == "user"

    tools = payload_dict.get("tools")
    assert isinstance(tools, list)
    assert tools
    tool_entry = tools[0]
    assert isinstance(tool_entry, dict)
    tool_entry_dict = cast("dict[str, object]", tool_entry)
    declarations = tool_entry_dict.get("functionDeclarations")
    assert isinstance(declarations, list)
    assert declarations
    declaration = declarations[0]
    assert isinstance(declaration, dict)
    declaration_dict = cast("dict[str, object]", declaration)
    assert declaration_dict.get("name") == "generate_image"

    tool_config = payload_dict.get("toolConfig")
    assert isinstance(tool_config, dict)
    function_calling_config = cast("dict[str, object]", tool_config).get(
        "functionCallingConfig",
    )
    assert isinstance(function_calling_config, dict)
    assert cast("dict[str, object]", function_calling_config).get("mode") == "AUTO"


@pytest.mark.asyncio
async def test_antigravity_image_tool_uses_original_message_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_contents: list[dict[str, object]] = []

    async def _fake_request_antigravity_generated_image(
        **kwargs: object,
    ) -> tuple[str, str, str]:
        nonlocal captured_contents
        contents = kwargs.get("contents")
        assert isinstance(contents, list)
        captured_contents = cast("list[dict[str, object]]", contents)
        return "AAAB", "image/png", ""

    monkeypatch.setattr(
        "llmcord.services.llm.providers.gemini_cli._request_antigravity_generated_image",
        _fake_request_antigravity_generated_image,
    )

    credentials = GeminiCliCredentials(
        refresh="refresh",
        access="access",
        expires=0,
        project_id="project-id",
    )

    tool_result = await _execute_antigravity_image_tool_call(
        endpoint="https://daily-cloudcode-pa.sandbox.googleapis.com",
        headers={},
        credentials=credentials,
        messages=[
            {
                "role": "user",
                "content": "Put a hat on me",
            },
        ],
        call_name="generate_image",
        call_args={"prompt": "AI rewritten: portrait with stylish fedora"},
    )

    assert "data:image/png;base64,AAAB" in tool_result
    assert captured_contents
    first = captured_contents[0]
    assert first.get("role") == "user"
    parts = first.get("parts")
    assert isinstance(parts, list)
    assert {"text": "Put a hat on me"} in parts


@pytest.mark.asyncio
async def test_antigravity_image_tool_includes_prior_generated_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_contents: list[dict[str, object]] = []

    async def _fake_request_antigravity_generated_image(
        **kwargs: object,
    ) -> tuple[str, str, str]:
        nonlocal captured_contents
        contents = kwargs.get("contents")
        assert isinstance(contents, list)
        captured_contents = cast("list[dict[str, object]]", contents)
        return "AAAB", "image/png", ""

    monkeypatch.setattr(
        "llmcord.services.llm.providers.gemini_cli._request_antigravity_generated_image",
        _fake_request_antigravity_generated_image,
    )

    credentials = GeminiCliCredentials(
        refresh="refresh",
        access="access",
        expires=0,
        project_id="project-id",
    )

    await _execute_antigravity_image_tool_call(
        endpoint="https://daily-cloudcode-pa.sandbox.googleapis.com",
        headers={},
        credentials=credentials,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Make me sip tea"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,USERIMG"},
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Generated image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,GENIMG"},
                    },
                ],
            },
            {
                "role": "user",
                "content": "Put a hat on me",
            },
        ],
        call_name="generate_image",
        call_args={"prompt": "AI rewritten: person with hat"},
    )

    assert len(captured_contents) == EXPECTED_IMAGE_CONTEXT_MESSAGES
    assistant_message = captured_contents[1]
    assert assistant_message.get("role") == "user"
    assistant_parts = assistant_message.get("parts")
    assert isinstance(assistant_parts, list)
    assert {
        "text": "Previously generated image for follow-up editing context.",
    } in assistant_parts
    assert {
        "inlineData": {"mimeType": "image/png", "data": "GENIMG"},
    } in assistant_parts

    last_user = captured_contents[2]
    last_user_parts = last_user.get("parts")
    assert isinstance(last_user_parts, list)
    assert {"text": "Put a hat on me"} in last_user_parts
    assert all(content.get("role") == "user" for content in captured_contents)
