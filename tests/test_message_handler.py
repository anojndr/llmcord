"""Tests for message handler helpers."""

# ruff: noqa: S101

from __future__ import annotations

import dataclasses

import discord
import pytest

import message_handler


@dataclasses.dataclass
class DummyTextDisplay:
    """Minimal text display stub for component tests."""

    type: discord.ComponentType
    content: str | None


def test_get_embed_text() -> None:
    """Extract text from an embed."""
    embed = discord.Embed(title="Title", description="Description")
    assert message_handler._get_embed_text(embed) == "Title\nDescription"  # noqa: SLF001


def test_build_node_text_parts() -> None:
    """Build a concatenated text payload from parts."""
    embed = discord.Embed(title="Title", description="Description")
    components = [
        DummyTextDisplay(type=discord.ComponentType.text_display, content="Component"),
    ]

    result = message_handler.build_node_text_parts(
        cleaned_content="Hello",
        embeds=[embed],
        components=components,
        text_attachments=["Attachment"],
        extra_parts=["Extra"],
    )

    assert "Hello" in result
    assert "Title" in result
    assert "Component" in result
    assert "Attachment" in result
    assert "Extra" in result


def test_append_search_to_content_string() -> None:
    """Append search text to string content."""
    content = message_handler.append_search_to_content("hello", "search")
    assert content == "hello\n\nsearch"


def test_append_search_to_content_list() -> None:
    """Append search text to list-based content payload."""
    content = [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "https://example.com"}},
    ]

    updated = message_handler.append_search_to_content(content, "search")
    assert updated[0]["text"].endswith("\n\nsearch")


def test_raise_empty_response_message() -> None:
    """Ensure empty response error uses the expected message."""
    with pytest.raises(message_handler.EmptyResponseError) as exc:
        message_handler._raise_empty_response()  # noqa: SLF001

    assert str(exc.value) == message_handler.EMPTY_RESPONSE_MESSAGE


def test_first_token_timeout_error_default_message() -> None:
    """Use the default first-token timeout message."""
    err = message_handler.FirstTokenTimeoutError()
    assert str(err) == message_handler.FIRST_TOKEN_TIMEOUT_MESSAGE
