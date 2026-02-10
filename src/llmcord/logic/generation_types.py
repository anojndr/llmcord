"""Data types for generation logic."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import discord

from llmcord.core.models import MsgNode


@dataclass(slots=True)
class GenerationContext:
    """Inputs required to generate an LLM response."""

    new_msg: discord.Message
    discord_bot: discord.Client
    msg_nodes: dict[int, MsgNode]
    messages: list[dict[str, object]]
    user_warnings: set[str]
    provider: str
    model: str
    actual_model: str
    provider_slash_model: str
    base_url: str | None
    api_keys: list[str]
    model_parameters: dict[str, object] | None
    extra_headers: dict[str, str] | None
    extra_query: dict[str, object] | None
    extra_body: dict[str, object] | None
    system_prompt: str | None
    config: dict[str, object]
    max_text: int
    tavily_metadata: dict[str, object] | None
    last_edit_time: float
    processing_msg: discord.Message
    retry_callback: Callable[[], Awaitable[None]]
    fallback_chain: list[tuple[str, str, str]] | None = None


@dataclass(slots=True)
class GenerationState:
    """Mutable state for response generation."""

    response_msgs: list[discord.Message]
    response_contents: list[str]
    input_tokens: int
    max_message_length: int
    embed: discord.Embed | None
    use_plain_responses: bool
    grounding_metadata: object | None
    last_edit_time: float
    generated_images: list["GeneratedImage"]
    generated_image_hashes: set[str]
    display_model: str


@dataclass(slots=True)
class FallbackState:
    """Track fallback selection state."""

    fallback_level: int
    fallback_index: int
    use_custom_fallbacks: bool
    original_provider: str
    original_model: str


@dataclass(slots=True)
class StreamConfig:
    """Configuration for a streaming LLM request."""

    provider: str
    actual_model: str
    api_key: str
    base_url: str | None
    extra_headers: dict[str, str] | None
    model_parameters: dict[str, object] | None


@dataclass(slots=True)
class StreamLoopState:
    """Mutable state for stream processing."""

    curr_content: str | None
    finish_reason: object | None


@dataclass(slots=True)
class GeneratedImage:
    """Generated image payload from Gemini responses."""

    data: bytes
    mime_type: str
    filename: str


@dataclass(slots=True)
class StreamEditDecision:
    """Decisions for streaming edits."""

    start_next_msg: bool
    msg_split_incoming: bool
    is_final_edit: bool
    is_good_finish: bool


@dataclass(slots=True)
class GenerationLoopState:
    """Mutable state for generation loop."""

    provider: str
    actual_model: str
    base_url: str | None
    api_keys: list[str]
    good_keys: list[str]
    initial_key_count: int
    attempt_count: int
    last_error_msg: str | None
    fallback_state: FallbackState
    fallback_chain: list[tuple[str, str, str]]
