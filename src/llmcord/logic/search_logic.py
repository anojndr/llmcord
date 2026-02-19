"""Web search and research command orchestration."""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import discord

from llmcord.core.config import is_gemini_model
from llmcord.core.config.utils import normalize_api_keys
from llmcord.core.models import MsgNode
from llmcord.logic.utils import (
    append_search_to_content,
    extract_research_command,
    replace_content_text,
)
from llmcord.services.database import get_db
from llmcord.services.search import (
    WebSearchOptions,
    decide_web_search,
    perform_tavily_research,
    perform_web_search,
)
from llmcord.services.search.config import EXA_MCP_URL

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ResearchCommandContext:
    """Inputs for handling research commands."""

    new_msg: discord.Message
    messages: list[dict[str, object]]
    msg_nodes: dict[int, MsgNode]
    research_model: str | None
    research_query: str | None
    tavily_api_keys: list[str]
    user_warnings: set[str]
    has_existing_search: bool


@dataclass(slots=True)
class WebSearchContext:
    """Inputs for web search execution."""

    new_msg: discord.Message
    messages: list[dict[str, object]]
    msg_nodes: dict[int, MsgNode]
    config: dict[str, object]
    web_search_provider: str
    tavily_api_keys: list[str]
    is_googlelens_query: bool
    actual_model: str
    actual_provider: str | None
    has_existing_search: bool
    search_metadata: dict[str, object] | None


@dataclass(slots=True)
class SearchResolutionContext:
    """Inputs for resolving search metadata."""

    new_msg: discord.Message
    discord_bot: discord.Client
    msg_nodes: dict[int, MsgNode]
    messages: list[dict[str, object]]
    user_warnings: set[str]
    tavily_api_keys: list[str]
    config: dict[str, object]
    web_search_available: bool
    web_search_provider: str
    actual_model: str
    actual_provider: str | None = None


def resolve_web_search_provider(
    config: dict[str, Any],
    tavily_api_keys: list[str],
) -> tuple[str, bool]:
    """Determine the web search provider and availability."""
    web_search_provider = str(config.get("web_search_provider", "tavily"))
    web_search_available = False
    if (web_search_provider == "tavily" and tavily_api_keys) or (
        web_search_provider == "exa"
    ):
        web_search_available = True
    elif web_search_provider == "auto":
        if tavily_api_keys:
            web_search_provider = "tavily"
            web_search_available = True
        else:
            web_search_provider = "exa"
            web_search_available = True

    return web_search_provider, web_search_available


def _get_web_search_max_chars_per_url(config: Mapping[str, object]) -> int:
    raw_value = config.get("web_search_max_chars_per_url", 4000)
    if isinstance(raw_value, bool):
        return 1

    if not isinstance(raw_value, int | str):
        return 4000

    try:
        max_chars = int(raw_value)
    except (TypeError, ValueError):
        return 4000

    return max(max_chars, 1)


async def resolve_search_metadata(
    context: SearchResolutionContext,
    is_googlelens_query_func: Callable[[discord.Message, discord.Client], bool],
) -> dict[str, object] | None:
    """Resolve search metadata and perform search if needed."""
    is_googlelens_query = is_googlelens_query_func(
        context.new_msg,
        context.discord_bot,
    )
    research_model, research_query = extract_research_command(
        context.new_msg.content,
        context.discord_bot.user.mention if context.discord_bot.user else "",
    )

    if (
        context.new_msg.id in context.msg_nodes
        and context.msg_nodes[context.new_msg.id].search_results
    ):
        search_metadata = context.msg_nodes[context.new_msg.id].tavily_metadata
        has_existing_search = True
    else:
        search_metadata = None
        has_existing_search = False

    search_metadata = (
        await run_research_command(
            ResearchCommandContext(
                new_msg=context.new_msg,
                messages=context.messages,
                msg_nodes=context.msg_nodes,
                research_model=research_model,
                research_query=research_query,
                tavily_api_keys=context.tavily_api_keys,
                user_warnings=context.user_warnings,
                has_existing_search=has_existing_search,
            ),
        )
        or search_metadata
    )

    if (
        research_model
        and context.web_search_available
        and search_metadata
        and search_metadata.get("keys_exhausted")
    ):
        search_metadata = await maybe_run_web_search(
            WebSearchContext(
                new_msg=context.new_msg,
                messages=context.messages,
                msg_nodes=context.msg_nodes,
                config=context.config,
                web_search_provider=context.web_search_provider,
                tavily_api_keys=context.tavily_api_keys,
                is_googlelens_query=is_googlelens_query,
                actual_model=context.actual_model,
                actual_provider=context.actual_provider,
                has_existing_search=has_existing_search,
                search_metadata=search_metadata,
            ),
        )

    if not research_model and context.web_search_available:
        search_metadata = await maybe_run_web_search(
            WebSearchContext(
                new_msg=context.new_msg,
                messages=context.messages,
                msg_nodes=context.msg_nodes,
                config=context.config,
                web_search_provider=context.web_search_provider,
                tavily_api_keys=context.tavily_api_keys,
                is_googlelens_query=is_googlelens_query,
                actual_model=context.actual_model,
                actual_provider=context.actual_provider,
                has_existing_search=has_existing_search,
                search_metadata=search_metadata,
            ),
        )

    return search_metadata


async def run_research_command(
    context: ResearchCommandContext,
) -> dict[str, object] | None:
    """Execute research command if present."""
    if not context.research_model or context.has_existing_search:
        return None

    if not context.research_query:
        context.user_warnings.add(
            "⚠️ Provide a research query after researchpro/researchmini",
        )
        return None

    if not context.tavily_api_keys:
        context.user_warnings.add("⚠️ Tavily API key missing for research")
        return None

    for msg in context.messages:
        if msg.get("role") == "user":
            msg["content"] = replace_content_text(
                cast("str | list[dict[str, object]]", msg.get("content", "")),
                context.research_query,
            )
            break

    research_results, search_metadata = await perform_tavily_research(
        query=context.research_query,
        api_keys=context.tavily_api_keys,
        model=context.research_model,
    )

    if research_results:
        for msg in context.messages:
            if msg.get("role") == "user":
                msg["content"] = append_search_to_content(
                    cast("str | list[dict[str, object]]", msg.get("content", "")),
                    research_results,
                )

                logger.info("Tavily research results appended to user message")

                get_db().save_message_search_data(
                    str(context.new_msg.id),
                    research_results,
                    search_metadata,
                )

                if context.new_msg.id in context.msg_nodes:
                    node = context.msg_nodes[context.new_msg.id]
                    node.search_results = research_results
                    node.tavily_metadata = search_metadata
                break

    return search_metadata


async def maybe_run_web_search(
    context: WebSearchContext,
) -> dict[str, object] | None:
    """Decide and execute web search."""
    search_metadata = context.search_metadata
    is_preview_model = "preview" in context.actual_model.lower()
    is_non_gemini = not is_gemini_model(context.actual_model)
    is_google_antigravity = context.actual_provider == "google-antigravity"
    if (
        context.web_search_provider
        and (is_non_gemini or is_preview_model or is_google_antigravity)
        and not context.is_googlelens_query
        and not context.has_existing_search
    ):
        db = get_db()
        user_id = str(context.new_msg.author.id)
        user_decider_model = db.get_user_search_decider_model(user_id)
        default_decider = str(
            context.config.get(
                "web_search_decider_model",
                "gemini/gemini-3-flash-preview",
            ),
        )

        models_config = cast("dict[str, Any]", context.config.get("models", {}))
        if user_decider_model and user_decider_model in models_config:
            decider_model_str = user_decider_model
        else:
            decider_model_str = default_decider

        decider_provider, decider_model = (
            decider_model_str.split("/", 1)
            if "/" in decider_model_str
            else ("gemini", decider_model_str)
        )

        decider_provider_config = cast(
            "dict[str, Any]",
            context.config.get("providers", {}),
        ).get(
            decider_provider,
            {},
        )
        decider_api_keys = normalize_api_keys(
            decider_provider_config.get("api_key"),
        )
        decider_model_parameters = cast(
            "dict[str, Any] | None",
            models_config.get(decider_model_str),
        )

        decider_config = {
            "provider": decider_provider,
            "model": decider_model,
            "api_keys": decider_api_keys,
            "base_url": decider_provider_config.get("base_url"),
            "extra_headers": decider_provider_config.get("extra_headers"),
            "model_parameters": decider_model_parameters,
        }

        if decider_api_keys:
            search_decision = await decide_web_search(
                context.messages,
                decider_config,
            )

            if search_decision.get("needs_search") and search_decision.get(
                "queries",
            ):
                queries = search_decision["queries"]
                logger.info(
                    "Web search triggered with %s. Queries: %s",
                    context.web_search_provider,
                    queries,
                )

                search_depth = str(
                    context.config.get(
                        "tavily_search_depth",
                        "advanced",
                    ),
                )
                max_chars_per_url = _get_web_search_max_chars_per_url(
                    context.config,
                )
                search_options = WebSearchOptions(
                    max_results_per_query=5,
                    max_chars_per_url=max_chars_per_url,
                    search_depth=search_depth,
                    web_search_provider=context.web_search_provider,
                    exa_mcp_url=EXA_MCP_URL,
                    tool=search_decision.get("tool", "web_search_exa"),
                )
                search_results, search_metadata = await perform_web_search(
                    queries,
                    api_keys=context.tavily_api_keys,
                    options=search_options,
                )

                if search_results:
                    for msg in context.messages:
                        if msg.get("role") == "user":
                            msg["content"] = append_search_to_content(
                                cast(
                                    "str | list[dict[str, object]]",
                                    msg.get("content", ""),
                                ),
                                search_results,
                            )

                            logger.info(
                                "Web search results appended to user message",
                            )

                            get_db().save_message_search_data(
                                str(context.new_msg.id),
                                search_results,
                                search_metadata,
                            )

                            if context.new_msg.id in context.msg_nodes:
                                node = context.msg_nodes[context.new_msg.id]
                                node.search_results = search_results
                                node.tavily_metadata = search_metadata
                            break

    return search_metadata
