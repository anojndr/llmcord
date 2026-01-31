"""Tests for message processing helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import discord

from llmcord.logic.helpers import (
    _strip_trigger_prefix,
    append_search_to_content,
    build_node_text_parts,
    extract_research_command,
    replace_content_text,
)


def assert_true(*, condition: bool, message: str) -> None:
    """Raise an AssertionError when a condition is false."""
    if not condition:
        raise AssertionError(message)


def test_build_node_text_parts() -> None:
    """Node text parts should include all content sources."""
    embed = MagicMock(spec=discord.Embed)
    embed.title = "Embed Title"
    embed.description = "Embed Description"

    component = MagicMock()
    component.type = discord.ComponentType.text_display
    component.content = "Component Content"

    components = [component]
    embeds = [embed]

    text = build_node_text_parts(
        cleaned_content="Main Content",
        embeds=embeds,
        components=components,
        text_attachments=["Attachment Text"],
        extra_parts=["Extra Part"],
    )

    assert_true(condition="Main Content" in text, message="Missing main content")
    assert_true(condition="Embed Title" in text, message="Missing embed title")
    assert_true(
        condition="Embed Description" in text,
        message="Missing embed description",
    )
    assert_true(
        condition="Component Content" in text,
        message="Missing component content",
    )
    assert_true(
        condition="Attachment Text" in text,
        message="Missing attachment text",
    )
    assert_true(condition="Extra Part" in text, message="Missing extra part")


def test_append_search_to_content_string() -> None:
    """Search results should append to string content."""
    content = "Original Content"
    result = append_search_to_content(content, "Search Results")
    assert_true(
        condition=result == "Original Content\n\nSearch Results",
        message="Expected appended search results",
    )


def test_append_search_to_content_multimodal() -> None:
    """Search results should append to multimodal content entries."""
    content = [{"type": "text", "text": "Original Content"}]
    result = append_search_to_content(content, "Search Results")
    assert_true(
        condition=result[0]["text"] == "Original Content\n\nSearch Results",
        message="Expected appended search results in multimodal content",
    )


def test_replace_content_text_string() -> None:
    """Replace text in string content."""
    content = "Old Text"
    result = replace_content_text(content, "New Text")
    assert_true(condition=result == "New Text", message="Expected new text")


def test_replace_content_text_multimodal() -> None:
    """Replace text in multimodal content items."""
    content = [{"type": "text", "text": "Old Text"}]
    result = replace_content_text(content, "New Text")
    assert_true(
        condition=result[0]["text"] == "New Text",
        message="Expected replaced text in multimodal content",
    )


def test_strip_trigger_prefix() -> None:
    """Strip trigger prefixes from content."""
    assert_true(
        condition=_strip_trigger_prefix("@Bot Hello", "@Bot") == "Hello",
        message="Expected trigger prefix removal",
    )
    assert_true(
        condition=_strip_trigger_prefix("at ai Hello", "@Bot") == "Hello",
        message="Expected trigger prefix removal",
    )
    assert_true(
        condition=_strip_trigger_prefix("@Bot at ai Hello", "@Bot") == "Hello",
        message="Expected trigger prefix removal",
    )


def test_extract_research_command() -> None:
    """Extract research command suffix and query from content."""
    cmd, query = extract_research_command("@Bot researchpro topic", "@Bot")
    assert_true(condition=cmd == "pro", message="Expected 'pro' command")
    assert_true(condition=query == "topic", message="Expected query")

    cmd, query = extract_research_command("researchmini topic", "")
    assert_true(condition=cmd == "mini", message="Expected 'mini' command")
    assert_true(condition=query == "topic", message="Expected query")

    cmd, query = extract_research_command("hello world", "")
    assert_true(condition=cmd is None, message="Expected no command")
    assert_true(condition=query == "hello world", message="Expected full query")
