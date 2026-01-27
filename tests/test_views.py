"""Tests for view helpers."""

# ruff: noqa: S101

from __future__ import annotations

import discord

import views


def test_get_grounding_queries_from_mapping() -> None:
    """Extract grounding queries from mapping metadata."""
    metadata = {"web_search_queries": ["q1", "q2"]}
    assert views._get_grounding_queries(metadata) == ["q1", "q2"]  # noqa: SLF001
    assert views._has_grounding_data(metadata) is True  # noqa: SLF001


def test_get_grounding_queries_from_list() -> None:
    """Extract grounding queries from list metadata."""
    metadata = [{"searchQueries": ["q1"]}]
    assert views._get_grounding_queries(metadata) == ["q1"]  # noqa: SLF001


def test_get_grounding_chunks() -> None:
    """Extract grounding chunks from metadata."""
    metadata = {
        "grounding_chunks": [
            {"title": "Title", "uri": "https://example.com"},
        ],
    }
    assert views._get_grounding_chunks(metadata) == [  # noqa: SLF001
        {"title": "Title", "uri": "https://example.com"},
    ]


def test_add_chunked_embed_field_splits() -> None:
    """Split long lists into multiple embed fields."""
    embed = discord.Embed()
    items = ["a" * 600, "b" * 600]

    views.add_chunked_embed_field(embed, items, "Sources", field_limit=1024)

    expected_fields = 2

    assert len(embed.fields) == expected_fields
    assert embed.fields[0].name == "Sources"
    assert embed.fields[1].name == "Sources (2)"
