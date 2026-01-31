"""Compatibility re-exports for logic utilities."""
from llmcord.logic.utils import (
    _strip_trigger_prefix,
    append_search_to_content,
    build_node_text_parts,
    extract_research_command,
    replace_content_text,
)

__all__ = [
    "_strip_trigger_prefix",
    "append_search_to_content",
    "build_node_text_parts",
    "extract_research_command",
    "replace_content_text",
]
