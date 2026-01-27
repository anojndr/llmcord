from __future__ import annotations

import discord

import views


def test_get_grounding_queries_from_mapping() -> None:
    metadata = {"web_search_queries": ["q1", "q2"]}
    assert views._get_grounding_queries(metadata) == ["q1", "q2"]
    assert views._has_grounding_data(metadata) is True


def test_get_grounding_queries_from_list() -> None:
    metadata = [{"searchQueries": ["q1"]}]
    assert views._get_grounding_queries(metadata) == ["q1"]


def test_get_grounding_chunks() -> None:
    metadata = {
        "grounding_chunks": [
            {"title": "Title", "uri": "https://example.com"},
        ],
    }
    assert views._get_grounding_chunks(metadata) == [
        {"title": "Title", "uri": "https://example.com"},
    ]


def test_add_chunked_embed_field_splits() -> None:
    embed = discord.Embed()
    items = ["a" * 600, "b" * 600]

    views.add_chunked_embed_field(embed, items, "Sources", field_limit=1024)

    assert len(embed.fields) == 2
    assert embed.fields[0].name == "Sources"
    assert embed.fields[1].name == "Sources (2)"
