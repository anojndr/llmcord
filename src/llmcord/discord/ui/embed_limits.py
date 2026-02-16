"""Embed limit enforcement helpers for Discord messages."""

from collections.abc import Awaitable, Callable, Mapping, Sequence

import discord

EMBED_TITLE_LIMIT = 256
EMBED_DESCRIPTION_LIMIT = 4096
EMBED_FIELD_COUNT_LIMIT = 25
EMBED_FIELD_NAME_LIMIT = 256
EMBED_FIELD_VALUE_LIMIT = 1024
EMBED_FOOTER_TEXT_LIMIT = 2048
EMBED_AUTHOR_NAME_LIMIT = 256
MESSAGE_EMBED_CHAR_LIMIT = 6000
MESSAGE_EMBED_COUNT_LIMIT = 10
_ELLIPSIS = "..."
_MIN_FIELD_TEXT = " "


def _truncate_text(text: str, limit: int) -> str:
    """Return text truncated to the provided character limit."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= len(_ELLIPSIS):
        return text[:limit]
    return f"{text[: limit - len(_ELLIPSIS)]}{_ELLIPSIS}"


def _coerce_optional_text(value: object) -> str | None:
    """Convert optional text-like values to strings."""
    if value is None:
        return None
    return str(value)


def _string_length(value: object) -> int:
    """Return string length for textual values, otherwise zero."""
    return len(value) if isinstance(value, str) else 0


def _embed_text_length(embed: discord.Embed) -> int:
    """Compute Discord's text-character budget usage for one embed."""
    total = _string_length(embed.title) + _string_length(embed.description)
    total += _string_length(getattr(embed.author, "name", None))
    total += _string_length(getattr(embed.footer, "text", None))
    for field in embed.fields:
        total += _string_length(getattr(field, "name", None))
        total += _string_length(getattr(field, "value", None))
    return total


def _limit_embed_title(embed: discord.Embed) -> None:
    """Clamp embed title length to Discord's documented limit."""
    if embed.title is None:
        return
    limited_title = _truncate_text(str(embed.title), EMBED_TITLE_LIMIT)
    embed.title = limited_title or None


def _limit_embed_description(embed: discord.Embed) -> None:
    """Clamp embed description length to Discord's documented limit."""
    if embed.description is None:
        return
    limited_description = _truncate_text(
        str(embed.description),
        EMBED_DESCRIPTION_LIMIT,
    )
    embed.description = limited_description or None


def _limit_embed_author(embed: discord.Embed) -> None:
    """Clamp embed author name length while preserving metadata."""
    author_name = _coerce_optional_text(getattr(embed.author, "name", None))
    if not author_name:
        return
    embed.set_author(
        name=_truncate_text(author_name, EMBED_AUTHOR_NAME_LIMIT) or _MIN_FIELD_TEXT,
        url=_coerce_optional_text(getattr(embed.author, "url", None)),
        icon_url=_coerce_optional_text(getattr(embed.author, "icon_url", None)),
    )


def _limit_embed_footer(embed: discord.Embed) -> None:
    """Clamp embed footer text length while preserving icon URL."""
    footer_text = _coerce_optional_text(getattr(embed.footer, "text", None))
    if not footer_text:
        return
    embed.set_footer(
        text=_truncate_text(footer_text, EMBED_FOOTER_TEXT_LIMIT) or _MIN_FIELD_TEXT,
        icon_url=_coerce_optional_text(getattr(embed.footer, "icon_url", None)),
    )


def _normalized_fields(embed: discord.Embed) -> list[tuple[str, str, bool]]:
    """Collect fields with Discord-compatible field count and text limits."""
    normalized: list[tuple[str, str, bool]] = []
    for field in list(embed.fields)[:EMBED_FIELD_COUNT_LIMIT]:
        name = _truncate_text(str(getattr(field, "name", "")), EMBED_FIELD_NAME_LIMIT)
        value = _truncate_text(
            str(getattr(field, "value", "")),
            EMBED_FIELD_VALUE_LIMIT,
        )
        normalized.append(
            (
                name or _MIN_FIELD_TEXT,
                value or _MIN_FIELD_TEXT,
                bool(getattr(field, "inline", False)),
            ),
        )
    return normalized


def _limit_embed_fields(embed: discord.Embed) -> None:
    """Clamp embed fields and rebuild to keep valid Discord payloads."""
    limited_fields = _normalized_fields(embed)
    embed.clear_fields()
    for name, value, inline in limited_fields:
        embed.add_field(name=name, value=value, inline=inline)


def _apply_per_embed_limits(embed: discord.Embed) -> None:
    """Apply all single-embed component limits."""
    _limit_embed_title(embed)
    _limit_embed_description(embed)
    _limit_embed_author(embed)
    _limit_embed_footer(embed)
    _limit_embed_fields(embed)


def _trim_text_for_overflow(text: str, overflow: int) -> str:
    """Trim text by overflow amount, preserving truncation marker when possible."""
    return _truncate_text(text, max(0, len(text) - overflow))


def _shrink_description(embed: discord.Embed, overflow: int) -> bool:
    """Shrink description to recover character budget."""
    description = _coerce_optional_text(embed.description)
    if not description:
        return False
    updated = _trim_text_for_overflow(description, overflow)
    if updated == description:
        return False
    embed.description = updated or None
    return True


def _rebuild_fields(fields: Sequence[object], embed: discord.Embed) -> None:
    """Replace embed fields from a generic field sequence."""
    embed.clear_fields()
    for field in fields:
        embed.add_field(
            name=str(getattr(field, "name", "")) or _MIN_FIELD_TEXT,
            value=str(getattr(field, "value", "")) or _MIN_FIELD_TEXT,
            inline=bool(getattr(field, "inline", False)),
        )


def _remove_last_field(embed: discord.Embed) -> bool:
    """Drop the last embed field to reduce text budget."""
    existing_fields = list(embed.fields)
    if not existing_fields:
        return False
    _rebuild_fields(existing_fields[:-1], embed)
    return True


def _shrink_footer(embed: discord.Embed, overflow: int) -> bool:
    """Shrink footer text to recover character budget."""
    footer_text = _coerce_optional_text(getattr(embed.footer, "text", None))
    if not footer_text:
        return False
    icon_url = _coerce_optional_text(getattr(embed.footer, "icon_url", None))
    updated = _trim_text_for_overflow(footer_text, overflow)
    if updated:
        embed.set_footer(text=updated, icon_url=icon_url)
    else:
        embed.remove_footer()
    return updated != footer_text


def _shrink_author(embed: discord.Embed, overflow: int) -> bool:
    """Shrink author name to recover character budget."""
    author_name = _coerce_optional_text(getattr(embed.author, "name", None))
    if not author_name:
        return False
    author_url = _coerce_optional_text(getattr(embed.author, "url", None))
    icon_url = _coerce_optional_text(getattr(embed.author, "icon_url", None))
    updated = _trim_text_for_overflow(author_name, overflow)
    if updated:
        embed.set_author(name=updated, url=author_url, icon_url=icon_url)
    else:
        embed.remove_author()
    return updated != author_name


def _shrink_title(embed: discord.Embed, overflow: int) -> bool:
    """Shrink title to recover character budget."""
    title = _coerce_optional_text(embed.title)
    if not title:
        return False
    updated = _trim_text_for_overflow(title, overflow)
    if updated == title:
        return False
    embed.title = updated or None
    return True


def _fit_embed_to_budget(embed: discord.Embed, budget: int) -> None:
    """Reduce embed text until it fits within a specific character budget."""
    target_budget = max(0, budget)
    while _embed_text_length(embed) > target_budget:
        overflow = _embed_text_length(embed) - target_budget
        if _shrink_description(embed, overflow):
            continue
        if _remove_last_field(embed):
            continue
        if _shrink_footer(embed, overflow):
            continue
        if _shrink_author(embed, overflow):
            continue
        if _shrink_title(embed, overflow):
            continue
        break


def enforce_embed_limits(
    embed: discord.Embed,
    *,
    budget: int = MESSAGE_EMBED_CHAR_LIMIT,
) -> discord.Embed:
    """Clamp one embed to Discord's per-embed and per-message text limits."""
    _apply_per_embed_limits(embed)
    _fit_embed_to_budget(embed, budget)
    return embed


def enforce_embeds_limits(embeds: Sequence[discord.Embed]) -> list[discord.Embed]:
    """Clamp embed collections to Discord message-level embed limits."""
    limited_embeds: list[discord.Embed] = []
    remaining_budget = MESSAGE_EMBED_CHAR_LIMIT
    for embed in list(embeds)[:MESSAGE_EMBED_COUNT_LIMIT]:
        enforce_embed_limits(embed, budget=remaining_budget)
        limited_embeds.append(embed)
        remaining_budget = max(0, remaining_budget - _embed_text_length(embed))
    return limited_embeds


def sanitize_embed_kwargs(kwargs: Mapping[str, object]) -> dict[str, object]:
    """Normalize `embed` and `embeds` kwargs to valid Discord limits."""
    sanitized = dict(kwargs)
    single_embed = sanitized.get("embed")
    many_embeds = sanitized.get("embeds")
    has_many_embeds = "embeds" in sanitized

    candidates: list[discord.Embed] = []
    if isinstance(single_embed, discord.Embed):
        candidates.append(single_embed)

    if isinstance(many_embeds, Sequence) and not isinstance(
        many_embeds,
        (bytes, bytearray, str),
    ):
        candidates.extend(
            embed for embed in many_embeds if isinstance(embed, discord.Embed)
        )

    if not candidates:
        return sanitized

    limited = enforce_embeds_limits(candidates)
    if has_many_embeds:
        sanitized["embeds"] = limited
        sanitized.pop("embed", None)
    else:
        sanitized["embed"] = limited[0]
    return sanitized


async def call_with_embed_limits[R](
    callback: Callable[..., Awaitable[R]],
    /,
    *args: object,
    **kwargs: object,
) -> R:
    """Call an async Discord API method with sanitized embed kwargs."""
    return await callback(*args, **sanitize_embed_kwargs(kwargs))
