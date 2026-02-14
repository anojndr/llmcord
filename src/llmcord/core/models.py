"""Data models for llmcord."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal

import discord


@dataclass(slots=True)
class MsgNode:
    """Represents a message node in the conversation chain.

    Uses __slots__ for memory efficiency since many instances are created.
    """

    text: str | None = None
    images: list[dict[str, Any]] = field(default_factory=list)
    raw_attachments: list[dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: int | None = None
    thought_signature: str | None = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: discord.Message | None = None

    # Web search data for persisting in chat history
    search_results: str | None = None
    tavily_metadata: dict | None = None

    # Google Lens / Yandex reverse image search results
    lens_results: str | None = None

    failed_extractions: list[str] = field(default_factory=list)

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
