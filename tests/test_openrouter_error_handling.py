from __future__ import annotations

# ruff: noqa: SLF001
import pytest

from llmcord.logic import generation as generation_mod
from llmcord.services.llm.providers.openrouter_errors import (
    HTTP_FORBIDDEN,
    HTTP_TOO_MANY_REQUESTS,
    OpenRouterAPIError,
    classify_openrouter_error,
    extract_openrouter_error_details,
    raise_for_openrouter_payload_error,
)


def test_openrouter_standard_error_envelope_is_raised_and_classified_429() -> None:
    payload = {
        "error": {
            "code": HTTP_TOO_MANY_REQUESTS,
            "message": "You are being rate limited",
        },
    }

    with pytest.raises(OpenRouterAPIError) as excinfo:
        raise_for_openrouter_payload_error(payload_obj=payload)

    assert "429" in str(excinfo.value)
    assert "rate" in str(excinfo.value).lower()

    classification = classify_openrouter_error(excinfo.value)
    assert classification is not None
    assert classification.http_status == HTTP_TOO_MANY_REQUESTS
    assert classification.action == "remove_key"


def test_openrouter_moderation_metadata_in_message_and_skips_provider() -> None:
    payload = {
        "error": {
            "code": HTTP_FORBIDDEN,
            "message": "Input flagged",
            "metadata": {
                "reasons": ["sexual", "self_harm"],
                "flagged_input": "...",
                "provider_name": "SomeProvider",
                "model_slug": "some/model",
            },
        },
    }

    with pytest.raises(OpenRouterAPIError) as excinfo:
        raise_for_openrouter_payload_error(payload_obj=payload)

    msg = str(excinfo.value)
    assert "403" in msg
    assert "reasons=" in msg
    assert "sexual" in msg

    classification = classify_openrouter_error(excinfo.value)
    assert classification is not None
    assert classification.http_status == HTTP_FORBIDDEN
    assert classification.action == "skip_provider"


def test_openrouter_midstream_error_chunk_is_detected_and_classified() -> None:
    chunk = {
        "id": "cmpl-abc123",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "openrouter/some-model",
        "provider": "openrouter",
        "error": {
            "code": "server_error",
            "message": "Provider disconnected",
        },
        "choices": [
            {
                "index": 0,
                "delta": {"content": ""},
                "finish_reason": "error",
            },
        ],
    }

    details = extract_openrouter_error_details(chunk)
    assert details is not None
    assert details.code == "server_error"
    assert "disconnected" in details.message.lower()

    with pytest.raises(OpenRouterAPIError) as excinfo:
        raise_for_openrouter_payload_error(payload_obj=chunk)

    classification = classify_openrouter_error(excinfo.value)
    assert classification is not None
    assert classification.action == "skip_provider"


def test_openrouter_generation_exception_handler_removes_key_for_401() -> None:
    good_keys = ["k1", "k2"]
    timeout_strikes: dict[str, int] = {}
    error = OpenRouterAPIError(
        "OpenRouter HTTP 401: Invalid API key",
        status_code=401,
        code=401,
    )

    new_keys, last_error_msg = generation_mod._handle_generation_exception(
        error=error,
        provider="openrouter",
        current_api_key="k1",
        good_keys=good_keys,
        timeout_strikes=timeout_strikes,
    )

    assert new_keys == ["k2"]
    assert "401" in last_error_msg


def test_openrouter_generation_exception_handler_skips_provider_for_400() -> None:
    good_keys = ["k1", "k2"]
    timeout_strikes: dict[str, int] = {}
    error = OpenRouterAPIError(
        "OpenRouter HTTP 400: Bad Request",
        status_code=400,
        code=400,
    )

    new_keys, last_error_msg = generation_mod._handle_generation_exception(
        error=error,
        provider="openrouter",
        current_api_key="k1",
        good_keys=good_keys,
        timeout_strikes=timeout_strikes,
    )

    assert new_keys == []
    assert "400" in last_error_msg
