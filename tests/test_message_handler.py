from __future__ import annotations

import dataclasses

import discord

import message_handler


@dataclasses.dataclass
class DummyTextDisplay:
    type: discord.ComponentType
    content: str | None


def test_get_embed_text() -> None:
    embed = discord.Embed(title="Title", description="Description")
    assert message_handler._get_embed_text(embed) == "Title\nDescription"


def test_build_node_text_parts() -> None:
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
    content = message_handler.append_search_to_content("hello", "search")
    assert content == "hello\n\nsearch"


def test_append_search_to_content_list() -> None:
    content = [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "https://example.com"}},
    ]

    updated = message_handler.append_search_to_content(content, "search")
    assert updated[0]["text"].endswith("\n\nsearch")
