import pytest
import discord
from unittest.mock import MagicMock
from llmcord.logic.helpers import (
    build_node_text_parts,
    append_search_to_content,
    replace_content_text,
    _strip_trigger_prefix,
    extract_research_command,
)

def test_build_node_text_parts():
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
        extra_parts=["Extra Part"]
    )
    
    assert "Main Content" in text
    assert "Embed Title" in text
    assert "Embed Description" in text
    assert "Component Content" in text
    assert "Attachment Text" in text
    assert "Extra Part" in text

def test_append_search_to_content_string():
    content = "Original Content"
    result = append_search_to_content(content, "Search Results")
    assert result == "Original Content\n\nSearch Results"

def test_append_search_to_content_multimodal():
    content = [{"type": "text", "text": "Original Content"}]
    result = append_search_to_content(content, "Search Results")
    assert result[0]["text"] == "Original Content\n\nSearch Results"

def test_replace_content_text_string():
    content = "Old Text"
    result = replace_content_text(content, "New Text")
    assert result == "New Text"

def test_replace_content_text_multimodal():
    content = [{"type": "text", "text": "Old Text"}]
    result = replace_content_text(content, "New Text")
    assert result[0]["text"] == "New Text"

def test_strip_trigger_prefix():
    assert _strip_trigger_prefix("@Bot Hello", "@Bot") == "Hello"
    assert _strip_trigger_prefix("at ai Hello", "@Bot") == "Hello"
    assert _strip_trigger_prefix("@Bot at ai Hello", "@Bot") == "Hello"

def test_extract_research_command():
    cmd, query = extract_research_command("@Bot researchpro topic", "@Bot")
    assert cmd == "pro"
    assert query == "topic"
    
    cmd, query = extract_research_command("researchmini topic", "")
    assert cmd == "mini"
    assert query == "topic"
    
    cmd, query = extract_research_command("hello world", "")
    assert cmd is None
    assert query == "hello world"
