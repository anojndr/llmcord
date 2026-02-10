"""Discord UI rendering logic for LLM responses."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

import discord

from llmcord.core.config import (
    EDIT_DELAY_SECONDS,
    EMBED_COLOR_COMPLETE,
    EMBED_COLOR_INCOMPLETE,
    STREAMING_INDICATOR,
)
from llmcord.discord.ui import metadata as ui_metadata
from llmcord.discord.ui import response_view, sources_view
from llmcord.logic.generation_types import (
    FallbackState,
    GenerationContext,
    GenerationState,
    StreamEditDecision,
)
from llmcord.logic.utils import count_text_tokens

logger = logging.getLogger(__name__)


async def render_exhausted_response(
    *,
    state: GenerationState,
    reply_helper: Callable[..., Awaitable[None]],
    last_error_msg: str | None,
    fallback_state: FallbackState,
) -> list[str]:
    """Render a response when all fallbacks are exhausted."""
    if fallback_state.use_custom_fallbacks:
        logger.error("All custom fallback options exhausted")
    else:
        logger.error("All fallback options exhausted")

    error_text = "❌ All API keys are currently unavailable. Please try again later."
    if last_error_msg:
        logger.info("Last fallback error summary: %s", last_error_msg)

    if state.embed is None:
        state.embed = discord.Embed(
            description=error_text,
            color=EMBED_COLOR_INCOMPLETE,
        )
    else:
        state.embed.description = error_text
        state.embed.color = EMBED_COLOR_INCOMPLETE

    if state.response_msgs:
        await state.response_msgs[-1].edit(embed=state.embed, view=None)
    else:
        await reply_helper(embed=state.embed)
    return [error_text]


async def update_response_view(
    *,
    context: GenerationContext,
    state: GenerationState,
    full_response: str,
    grounding_metadata: object | None,
) -> None:
    """Update the response view with buttons and footer."""
    if (
        state.use_plain_responses
        or not state.response_msgs
        or not state.response_contents
    ):
        return

    response_view_instance = response_view.ResponseView(
        full_response,
        grounding_metadata,
        context.tavily_metadata,
        context.retry_callback,
        context.new_msg.author.id,
    )

    output_tokens = count_text_tokens(full_response)
    total_tokens = state.input_tokens + output_tokens
    last_msg_index = len(state.response_msgs) - 1
    if last_msg_index < len(state.response_contents) and state.embed:
        state.embed.description = state.response_contents[last_msg_index]
        state.embed.color = EMBED_COLOR_COMPLETE
        footer_text = f"{state.display_model} | total tokens: {total_tokens:,}"
        state.embed.set_footer(text=footer_text)
        await state.response_msgs[last_msg_index].edit(
            embed=state.embed,
            view=response_view_instance,
        )


async def maybe_edit_stream_message(
    *,
    state: GenerationState,
    reply_helper: Callable[..., Awaitable[None]],
    decision: StreamEditDecision,
    grounding_metadata: object | None,
) -> None:
    """Decide whether to edit the current streaming message."""
    response_contents = state.response_contents
    response_msgs = state.response_msgs
    embed = state.embed
    last_edit_time = state.last_edit_time

    if embed is None:
        return

    time_delta = datetime.now(timezone.utc).timestamp() - last_edit_time
    ready_to_edit = time_delta >= EDIT_DELAY_SECONDS

    if not (decision.start_next_msg or ready_to_edit or decision.is_final_edit):
        return

    embed.description = (
        response_contents[-1]
        if decision.is_final_edit
        else (response_contents[-1] + STREAMING_INDICATOR)
    )
    embed.color = (
        EMBED_COLOR_COMPLETE
        if decision.msg_split_incoming or decision.is_good_finish
        else EMBED_COLOR_INCOMPLETE
    )

    view = (
        sources_view.SourceView(grounding_metadata)
        if decision.is_final_edit and ui_metadata.has_grounding_data(grounding_metadata)
        else None
    )

    msg_index = len(response_contents) - 1
    if decision.start_next_msg:
        if msg_index < len(response_msgs):
            await response_msgs[msg_index].edit(embed=embed, view=view)
        else:
            await reply_helper(embed=embed, silent=True, view=view)
    else:
        await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
        await response_msgs[msg_index].edit(embed=embed, view=view)
    state.last_edit_time = datetime.now(timezone.utc).timestamp()


async def render_plain_responses(
    *,
    response_contents: list[str],
    response_msgs: list[discord.Message],
    reply_helper: Callable[..., Awaitable[None]],
    grounding_metadata: object | None,
    tavily_metadata: dict[str, object] | None,
) -> None:
    """Render multiple response parts as plain messages with layout views."""
    for i, content in enumerate(response_contents):
        layout = response_view.LayoutView().add_item(
            response_view.TextDisplay(content=content),
        )

        if i == len(response_contents) - 1:
            if ui_metadata.has_grounding_data(grounding_metadata):
                layout.add_item(
                    sources_view.SourceButton(grounding_metadata),
                )
            if tavily_metadata and (
                tavily_metadata.get("urls") or tavily_metadata.get("queries")
            ):
                layout.add_item(
                    sources_view.TavilySourceButton(tavily_metadata),
                )

        if i < len(response_msgs):
            await response_msgs[i].edit(view=layout)
        else:
            await reply_helper(view=layout)
