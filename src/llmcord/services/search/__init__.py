"""Search service module."""

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


def run_decider_once(*args: object, **kwargs: object) -> object:
    """Public wrapper for the decider runner (test patch friendly)."""
    return _run_decider_once(*args, **kwargs)


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
