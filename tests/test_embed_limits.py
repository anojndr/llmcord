from __future__ import annotations

from typing import cast

import discord
import pytest

from llmcord.discord.ui.embed_limits import (
    EMBED_AUTHOR_NAME_LIMIT,
    EMBED_DESCRIPTION_LIMIT,
    EMBED_FIELD_COUNT_LIMIT,
    EMBED_FIELD_NAME_LIMIT,
    EMBED_FIELD_VALUE_LIMIT,
    EMBED_FOOTER_TEXT_LIMIT,
    EMBED_TITLE_LIMIT,
    MESSAGE_EMBED_CHAR_LIMIT,
    MESSAGE_EMBED_COUNT_LIMIT,
    call_with_embed_limits,
    enforce_embed_limits,
    enforce_embeds_limits,
    sanitize_embed_kwargs,
)


def _embed_text_total(embed: discord.Embed) -> int:
    total = len(embed.title or "") + len(embed.description or "")
    total += len(getattr(embed.author, "name", "") or "")
    total += len(getattr(embed.footer, "text", "") or "")
    for field in embed.fields:
        total += len(getattr(field, "name", "") or "")
        total += len(getattr(field, "value", "") or "")
    return total


def _embeds_text_total(embeds: list[discord.Embed]) -> int:
    return sum(_embed_text_total(embed) for embed in embeds)


def test_enforce_embed_limits_clamps_all_component_limits() -> None:
    embed = discord.Embed(
        title="t" * (EMBED_TITLE_LIMIT + 50),
        description="d" * (EMBED_DESCRIPTION_LIMIT + 500),
    )
    embed.set_author(name="a" * (EMBED_AUTHOR_NAME_LIMIT + 25))
    embed.set_footer(text="f" * (EMBED_FOOTER_TEXT_LIMIT + 25))

    for _ in range(EMBED_FIELD_COUNT_LIMIT + 10):
        embed.add_field(
            name="n" * (EMBED_FIELD_NAME_LIMIT + 50),
            value="v" * (EMBED_FIELD_VALUE_LIMIT + 50),
            inline=False,
        )

    limited = enforce_embed_limits(embed)

    assert len(limited.title or "") <= EMBED_TITLE_LIMIT
    assert len(limited.description or "") <= EMBED_DESCRIPTION_LIMIT
    assert len(getattr(limited.author, "name", "") or "") <= EMBED_AUTHOR_NAME_LIMIT
    assert len(getattr(limited.footer, "text", "") or "") <= EMBED_FOOTER_TEXT_LIMIT
    assert len(limited.fields) <= EMBED_FIELD_COUNT_LIMIT
    assert all(
        len(getattr(field, "name", "") or "") <= EMBED_FIELD_NAME_LIMIT
        and len(getattr(field, "value", "") or "") <= EMBED_FIELD_VALUE_LIMIT
        for field in limited.fields
    )
    assert _embed_text_total(limited) <= MESSAGE_EMBED_CHAR_LIMIT


def test_enforce_embeds_limits_clamps_message_budget_and_count() -> None:
    embeds = [
        discord.Embed(title=f"Title {index}", description="x" * 2500)
        for index in range(MESSAGE_EMBED_COUNT_LIMIT + 3)
    ]

    limited = enforce_embeds_limits(embeds)

    assert len(limited) == MESSAGE_EMBED_COUNT_LIMIT
    assert _embeds_text_total(limited) <= MESSAGE_EMBED_CHAR_LIMIT


def test_sanitize_embed_kwargs_handles_embed_and_embeds() -> None:
    kwargs = {
        "embed": discord.Embed(description="x" * 5000),
        "embeds": [discord.Embed(description="y" * 5000) for _ in range(12)],
        "ephemeral": True,
    }

    sanitized = sanitize_embed_kwargs(kwargs)

    assert "embed" not in sanitized
    embeds_obj = sanitized.get("embeds")
    assert isinstance(embeds_obj, list)
    assert len(embeds_obj) == MESSAGE_EMBED_COUNT_LIMIT
    assert (
        _embeds_text_total(cast("list[discord.Embed]", embeds_obj))
        <= MESSAGE_EMBED_CHAR_LIMIT
    )
    assert sanitized["ephemeral"] is True


@pytest.mark.asyncio
async def test_call_with_embed_limits_sanitizes_before_invocation() -> None:
    captured_kwargs: dict[str, object] = {}

    async def _fake_send(**kwargs: object) -> dict[str, object]:
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        return kwargs

    await call_with_embed_limits(
        _fake_send,
        embeds=[discord.Embed(description="z" * 7000) for _ in range(20)],
    )

    embeds_obj = captured_kwargs.get("embeds")
    assert isinstance(embeds_obj, list)
    assert len(embeds_obj) == MESSAGE_EMBED_COUNT_LIMIT
    assert (
        _embeds_text_total(cast("list[discord.Embed]", embeds_obj))
        <= MESSAGE_EMBED_CHAR_LIMIT
    )
