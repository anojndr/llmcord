"""Search and system prompt handling for message processing."""
# ruff: noqa: E501

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llmcord.bad_keys import get_bad_keys_db
from llmcord.config import ensure_list, get_config, is_gemini_model
from llmcord.web_search import (
    decide_web_search,
    get_current_datetime_strings,
    perform_tavily_research,
    perform_web_search,
)

from .shared import (
    append_search_to_content,
    extract_research_command,
    replace_content_text,
    safe_lower,
    strip_prefix,
)

if TYPE_CHECKING:
    import discord

logger = logging.getLogger(__name__)


def append_system_prompt(
    messages: list[dict[str, object]],
    system_prompt: str | None,
    *,
    accept_usernames: bool,
) -> str | None:
    """Append a formatted system prompt to the message list."""
    if not system_prompt:
        return None

    date_str, time_str = get_current_datetime_strings()

    system_prompt = (
        system_prompt.replace("{date}", date_str)
        .replace("{time}", time_str)
        .strip()
    )
    if accept_usernames:
        system_prompt += (
            "\n\nUser's names are their Discord IDs and should be typed as '<@ID>'."
        )

    messages.append({"role": "system", "content": system_prompt})
    return system_prompt


async def handle_search_workflow(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    new_msg: discord.Message,
    discord_bot: discord.Client,
    messages: list[dict[str, object]],
    msg_nodes: dict[int, object],
    user_warnings: set[str],
    actual_model: str,
) -> dict[str, object] | None:
    """Handle research commands and web search, returning search metadata."""
    config = get_config()

    # Web Search Integration for non-Gemini models and Gemini preview models
    # Supports both Tavily and Exa MCP as search providers
    tavily_api_keys = ensure_list(config.get("tavily_api_key"))
    exa_mcp_url = config.get("exa_mcp_url", "")
    web_search_provider = config.get(
        "web_search_provider",
        "tavily",
    )  # Default to Tavily

    # Determine if web search is available based on provider config
    web_search_available = False
    if (web_search_provider == "tavily" and tavily_api_keys) or (
        web_search_provider == "exa" and exa_mcp_url
    ):
        web_search_available = True
    elif web_search_provider == "auto":
        # Auto-detect: prefer Tavily if keys are available, otherwise try Exa
        if tavily_api_keys:
            web_search_provider = "tavily"
            web_search_available = True
        elif exa_mcp_url:
            web_search_provider = "exa"
            web_search_available = True

    is_preview_model = "preview" in actual_model.lower()
    is_non_gemini = not is_gemini_model(actual_model)
    # Check for googlelens query - strip both mention and "at ai" prefix
    content_for_lens_check = strip_prefix(
        safe_lower(new_msg.content),
        safe_lower(discord_bot.user.mention),
    ).strip()
    if content_for_lens_check.startswith("at ai"):
        content_for_lens_check = content_for_lens_check[5:].strip()
    is_googlelens_query = content_for_lens_check.startswith("googlelens")

    search_metadata = None
    research_model, research_query = extract_research_command(
        new_msg.content,
        discord_bot.user.mention if discord_bot.user else "",
    )

    # Check for existing search results to handle retries correctly
    if new_msg.id in msg_nodes and msg_nodes[new_msg.id].search_results:
        search_metadata = msg_nodes[new_msg.id].tavily_metadata
        has_existing_search = True
    else:
        has_existing_search = False

    if research_model and not has_existing_search:
        if not research_query:
            user_warnings.add(
                "⚠️ Provide a research query after researchpro/researchmini",
            )
        elif not tavily_api_keys:
            user_warnings.add("⚠️ Tavily API key missing for research")
        else:
            for msg in messages:
                if msg.get("role") == "user":
                    msg["content"] = replace_content_text(
                        msg.get("content", ""),
                        research_query,
                    )
                    break

            research_results, search_metadata = await perform_tavily_research(
                query=research_query,
                api_keys=tavily_api_keys,
                model=research_model,
            )

            if research_results:
                for msg in messages:
                    if msg.get("role") == "user":
                        msg["content"] = append_search_to_content(
                            msg.get("content", ""),
                            research_results,
                        )

                        logger.info(
                            "Tavily research results appended to user message",
                        )

                        get_bad_keys_db().save_message_search_data(
                            new_msg.id,
                            research_results,
                            search_metadata,
                        )

                        if new_msg.id in msg_nodes:
                            msg_nodes[new_msg.id].search_results = research_results
                            msg_nodes[new_msg.id].tavily_metadata = search_metadata
                        break

    if (
        not research_model
        and web_search_available
        and (is_non_gemini or is_preview_model)
        and not is_googlelens_query
        and not has_existing_search
    ):
        # Get web search decider model - first check user preference, then config default
        db = get_bad_keys_db()
        user_id = str(new_msg.author.id)
        user_decider_model = db.get_user_search_decider_model(user_id)
        default_decider = config.get(
            "web_search_decider_model",
            "gemini/gemini-3-flash-preview",
        )

        # Use user preference if set and valid, otherwise use config default
        if user_decider_model and user_decider_model in config.get("models", {}):
            decider_model_str = user_decider_model
        else:
            decider_model_str = default_decider

        decider_provider, decider_model = (
            decider_model_str.split("/", 1)
            if "/" in decider_model_str
            else ("gemini", decider_model_str)
        )

        # Get provider config for the decider
        decider_provider_config = config.get("providers", {}).get(decider_provider, {})
        decider_api_keys = ensure_list(decider_provider_config.get("api_key"))

        decider_config = {
            "provider": decider_provider,
            "model": decider_model,
            "api_keys": decider_api_keys,
            "base_url": decider_provider_config.get("base_url"),
        }

        if decider_api_keys:
            # Run the search decider with chat history (uses all keys with rotation)
            search_decision = await decide_web_search(messages, decider_config)

            if search_decision.get("needs_search") and search_decision.get("queries"):
                queries = search_decision["queries"]
                logger.info(
                    "Web search triggered with %s. Queries: %s",
                    web_search_provider,
                    queries,
                )

                # Perform web search with the configured provider
                search_depth = config.get("tavily_search_depth", "advanced")
                search_results, search_metadata = await perform_web_search(
                    queries,
                    api_keys=tavily_api_keys,
                    max_results_per_query=5,
                    max_chars_per_url=2000,
                    search_depth=search_depth,
                    web_search_provider=web_search_provider,
                    exa_mcp_url=exa_mcp_url or "https://mcp.exa.ai/mcp?tools=web_search_exa,web_search_advanced_exa,get_code_context_exa,deep_search_exa,crawling_exa,company_research_exa,linkedin_search_exa,deep_researcher_start,deep_researcher_check",
                )

                if search_results:
                    # Append search results to the first (most recent) user message
                    for msg in messages:
                        if msg.get("role") == "user":
                            msg["content"] = append_search_to_content(
                                msg.get("content", ""),
                                search_results,
                            )

                            logger.info(
                                "Web search results appended to user message",
                            )

                            # Save search results to database for persistence in chat history
                            get_bad_keys_db().save_message_search_data(
                                new_msg.id,
                                search_results,
                                search_metadata,
                            )

                            # Also update the cached MsgNode so follow-up requests in the same session get the data
                            if new_msg.id in msg_nodes:
                                msg_nodes[new_msg.id].search_results = search_results
                                msg_nodes[new_msg.id].tavily_metadata = search_metadata
                            break

    return search_metadata
