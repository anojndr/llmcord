"""Search service module."""

from typing import TYPE_CHECKING, Any

from llmcord.services.search.core import WebSearchOptions, perform_web_search
from llmcord.services.search.decider import _run_decider_once, decide_web_search
from llmcord.services.search.tavily import (
    _get_tavily_client,
    perform_tavily_research,
    tavily_search,
)
from llmcord.services.search.utils import (
    convert_messages_to_openai_format,
    get_current_datetime_strings,
)


from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from llmcord.services.search.decider import DeciderRunConfig


def run_decider_once(
    messages: list[Any],
    run_config: "DeciderRunConfig",
) -> "Awaitable[tuple[dict[str, Any] | None, bool]]":
    """Public wrapper for the decider runner (test patch friendly)."""
    return _run_decider_once(messages, run_config)


def get_tavily_client() -> object:
    """Public wrapper for Tavily client creation (test patch friendly)."""
    return _get_tavily_client()


__all__ = [
    "WebSearchOptions",
    "_get_tavily_client",
    "_run_decider_once",
    "convert_messages_to_openai_format",
    "decide_web_search",
    "get_current_datetime_strings",
    "get_tavily_client",
    "perform_tavily_research",
    "perform_web_search",
    "run_decider_once",
    "tavily_search",
]
