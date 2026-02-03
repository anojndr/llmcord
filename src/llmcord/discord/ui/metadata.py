"""Metadata and embedding utilities for UI components."""
from collections.abc import Mapping, Sequence

import discord


def _normalize_queries(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(query) for query in value if query]
    if value:
        return [str(value)]
    return []


def _extract_queries_from_mapping(mapping: Mapping[str, object]) -> list[str]:
    for key in ("web_search_queries", "searchQueries", "webSearchQueries"):
        value = mapping.get(key)
        if value:
            return _normalize_queries(value)
    return []


def get_grounding_queries(metadata: object | None) -> list[str]:
    """Extract web search queries from grounding metadata.

    Handles GenAI types.GroundingMetadata, LiteLLM dict formats, and list
    formats.
    """
    if metadata is None:
        return []

    # Handle list format (LiteLLM streaming returns list of grounding results)
    if isinstance(metadata, list):
        queries = []
        for item in metadata:
            if isinstance(item, Mapping):
                queries.extend(_extract_queries_from_mapping(item))
        return queries

    # Handle dict format (LiteLLM)
    if isinstance(metadata, Mapping):
        return _extract_queries_from_mapping(metadata)

    # Handle object format (GenAI GroundingMetadata)
    if hasattr(metadata, "web_search_queries"):
        return _normalize_queries(getattr(metadata, "web_search_queries", None))

    return []


def _extract_chunks_from_mapping(mapping: Mapping[str, object]) -> list[object]:
    for key in ("grounding_chunks", "groundingChunks", "chunks"):
        value = mapping.get(key)
        if value:
            if isinstance(value, list):
                return list(value)
            return [value]
    return []


def _normalize_chunk(chunk: object) -> dict[str, str] | None:
    if isinstance(chunk, Mapping):
        web = chunk.get("web")
        if isinstance(web, Mapping):
            title = str(web.get("title") or "")
            uri = str(web.get("uri") or "")
            if title or uri:
                return {"title": title, "uri": uri}
        title = str(chunk.get("title") or "")
        uri = str(chunk.get("uri") or "")
        if title or uri:
            return {"title": title, "uri": uri}
        return None

    web = getattr(chunk, "web", None)
    if web:
        title = str(getattr(web, "title", "") or "")
        uri = str(getattr(web, "uri", "") or "")
        if title or uri:
            return {"title": title, "uri": uri}
    return None


def _normalize_chunks(chunks: Sequence[object]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for chunk in chunks:
        result = _normalize_chunk(chunk)
        if result:
            normalized.append(result)
    return normalized


def get_grounding_chunks(metadata: object | None) -> list[dict[str, str]]:
    """Extract grounding chunks from grounding metadata.

    Returns list of dicts with 'title' and 'uri' keys.
    Handles GenAI types.GroundingMetadata, LiteLLM dict formats, and list
    formats.
    """
    if metadata is None:
        return []

    # Handle list format (LiteLLM streaming returns list of grounding results)
    if isinstance(metadata, list):
        result: list[dict[str, str]] = []
        for item in metadata:
            if isinstance(item, Mapping):
                chunks = _extract_chunks_from_mapping(item)
                result.extend(_normalize_chunks(chunks))
        return result

    # Handle dict format (LiteLLM)
    if isinstance(metadata, Mapping):
        chunks = _extract_chunks_from_mapping(metadata)
        return _normalize_chunks(chunks)

    # Handle object format (GenAI GroundingMetadata)
    if hasattr(metadata, "grounding_chunks"):
        chunks = getattr(metadata, "grounding_chunks", []) or []
        return _normalize_chunks(chunks)

    return []


def has_grounding_data(metadata: object | None) -> bool:
    """Check if metadata has any grounding data (queries or chunks).

    Used to determine if the Show Sources button should be displayed.
    Only returns True if actual grounding queries or chunks exist.
    """
    if metadata is None:
        return False

    return bool(get_grounding_queries(metadata)) or bool(
        get_grounding_chunks(metadata),
    )


def add_chunked_embed_field(
    embed: discord.Embed,
    items: list[str],
    base_name: str,
    field_limit: int = 1024,
) -> None:
    """Add items to embed fields, splitting into multiple fields if needed.

    Args:
        embed: The Discord embed to add fields to
        items: List of strings to add
        base_name: Base name for the field (e.g., "Sources", "Search Results")
        field_limit: Maximum characters per field (default: 1024)

    """
    if not items:
        return

    current_chunk = ""
    field_count = 1

    for item in items:
        if len(current_chunk) + len(item) + 1 > field_limit:
            field_name = (
                f"{base_name} ({field_count})" if field_count > 1 else base_name
            )
            embed.add_field(name=field_name, value=current_chunk, inline=False)
            current_chunk = item
            field_count += 1
        elif current_chunk:
            current_chunk = f"{current_chunk}\n{item}"
        else:
            current_chunk = item

    if current_chunk:
        field_name = (
            f"{base_name} ({field_count})" if field_count > 1 else base_name
        )
        embed.add_field(name=field_name, value=current_chunk, inline=False)


def build_grounding_sources_embed(metadata: object) -> discord.Embed:
    """Build a Discord embed showing sources from grounding metadata.

    Args:
        metadata: Grounding metadata (either GenAI GroundingMetadata or
            LiteLLM dict)

    Returns:
        A Discord Embed with the formatted sources

    """
    embed = discord.Embed(title="Sources", color=discord.Color.blue())

    queries = get_grounding_queries(metadata)
    if queries:
        embed.add_field(
            name="Search Queries",
            value="\n".join(f"• {q}" for q in queries),
            inline=False,
        )

    chunks = get_grounding_chunks(metadata)
    if chunks:
        sources = []
        for i, chunk in enumerate(chunks):
            if chunk.get("uri"):
                sources.append(f"{i+1}. [{chunk['title']}]({chunk['uri']})")
        add_chunked_embed_field(embed, sources, "Search Results")

    # Handle edge case where no content was added to the embed
    if not embed.fields:
        embed.add_field(
            name="Sources",
            value="No source information available.",
            inline=False,
        )

    return embed
