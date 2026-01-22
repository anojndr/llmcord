"""
Data models for llmcord.
"""
import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import discord


@dataclass(slots=True)
class MsgNode:
    """Represents a message node in the conversation chain.
    
    Uses __slots__ for memory efficiency since many instances are created.
    """
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)
    raw_attachments: list[dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None
    thought_signature: Optional[str] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None
    
    # Web search data for persisting in chat history
    search_results: Optional[str] = None
    tavily_metadata: Optional[dict] = None
    
    # Google Lens / Yandex reverse image search results
    lens_results: Optional[str] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

