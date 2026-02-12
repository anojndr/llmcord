from __future__ import annotations

import litellm

from llmcord.logic.discord_ui import _build_footer_text
from llmcord.logic.generation import (
    GENERATION_EXCEPTIONS,
    _is_image_input_error,
    _remove_images_from_messages,
)
from llmcord.logic.generation_types import GenerationState


def test_detects_openrouter_image_input_error() -> None:
    error_message = (
        "NotFoundError: OpenrouterException - "
        '{"error":{"message":"No endpoints found that support image input",'
        '"code":404}}'
    )
    error = Exception(
        error_message,
    )
    assert _is_image_input_error(error) is True


def test_removes_images_from_message_content() -> None:
    messages: list[dict[str, object]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                },
            ],
        },
    ]

    removed = _remove_images_from_messages(messages)

    assert removed is True
    assert isinstance(messages[0]["content"], list)
    assert messages[0]["content"] == [{"type": "text", "text": "Describe this"}]


def test_footer_includes_image_removed_warning() -> None:
    state = GenerationState(
        response_msgs=[],
        response_contents=["ok"],
        input_tokens=100,
        max_message_length=4096,
        embed=None,
        use_plain_responses=False,
        grounding_metadata=None,
        last_edit_time=0.0,
        generated_images=[],
        generated_image_hashes=set(),
        display_model="openrouter/openrouter/free",
        fallback_warning=None,
        image_removal_warning="⚠️ Image removed from input due to provider error.",
    )

    footer_text = _build_footer_text(state=state, total_tokens=150)

    assert "openrouter/openrouter/free | total tokens: 150" in footer_text
    assert "Image removed from input due to provider error" in footer_text


def test_not_found_error_is_retryable_generation_exception() -> None:
    assert litellm.exceptions.NotFoundError in GENERATION_EXCEPTIONS
