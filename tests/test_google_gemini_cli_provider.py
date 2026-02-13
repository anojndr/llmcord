import json
from typing import cast

from llmcord.services.llm.providers.gemini_cli import (
    _build_cloudcode_request,
    credentials_to_api_key,
    parse_api_key_credentials,
)

EXPECTED_EXPIRES_MS = 1_700_000_000_000
EXPECTED_CONTENT_COUNT = 2


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

    parsed = parse_api_key_credentials(raw)

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

    parsed = parse_api_key_credentials(raw)

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
    parsed = parse_api_key_credentials("refresh-only-token")

    assert parsed.refresh == "refresh-only-token"
    assert parsed.access is None
    assert parsed.project_id is None


def test_build_cloudcode_request_converts_multipart_content() -> None:
    request = _build_cloudcode_request(
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
