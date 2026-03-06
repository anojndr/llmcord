"""Automatic prompt compaction to stay within model context windows."""

from __future__ import annotations

import copy
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import httpx
import litellm

from llmcord.logic.utils import (
    count_conversation_tokens,
    estimate_message_tokens,
    estimate_tokens_from_chars,
)
from llmcord.services.llm import LiteLLMOptions, prepare_litellm_kwargs
from llmcord.services.llm.providers.gemini_cli import stream_google_gemini_cli
from llmcord.services.llm.providers.model_aliases import (
    GEMINI_THINKING_LEVEL_SUFFIXES,
    OPENAI_REASONING_EFFORT_SUFFIXES,
    strip_model_suffix_alias,
)
from llmcord.services.llm.providers.openai_codex import (
    OPENAI_CODEX_PROVIDER,
    stream_openai_codex,
)
from llmcord.services.llm.providers.openrouter_errors import (
    raise_for_openrouter_payload_error,
)

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_WINDOW_TOKENS = 128000
DEFAULT_RESPONSE_RESERVE_TOKENS = 16384
MIN_RESPONSE_RESERVE_TOKENS = 1024
MIN_SUMMARY_TOKENS = 512
MAX_SUMMARY_TOKENS = 2048
MIN_HARD_FIT_MESSAGE_TOKENS = 64
COMPACTION_MARKER = "\n...[compacted]...\n"
SUMMARY_MESSAGE_HEADING = "Earlier conversation summary (auto-compacted):"
TRIMMED_CONTENT_WARNING = "⚠️ Some message content was trimmed to fit the model."
COMPACTED_CONTEXT_WARNING = (
    "⚠️ Older conversation context was compacted to fit the model."
)

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Summarize the conversation so "
    "another assistant can continue it accurately in any domain. Do not "
    "continue the conversation. Only return the requested summary."
)
SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize.
Create a structured context checkpoint summary that another assistant can use
to continue the interaction accurately, whether the conversation is about
coding, research, planning, writing, support, or general Q&A.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if needed.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by the user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks or answers]

### In Progress
- [ ] [Current work that is not finished]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Important references, examples, or facts needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact names, identifiers, URLs, file
paths, commands, function names, error messages, dates, numbers, and user
preferences when they matter."""
UPDATE_SUMMARIZATION_PROMPT = """The messages above are NEW conversation
messages to incorporate into the existing summary provided in
<previous-summary> tags.

Update the existing structured summary with new information. RULES:
- Preserve relevant details from the previous summary.
- Add new progress, decisions, and context from the new messages.
- Move completed items from "In Progress" to "Done" when appropriate.
- Update "Next Steps" to reflect the current state.
- Preserve exact names, identifiers, URLs, file paths, commands, function
  names, error messages, dates, numbers, and user preferences when they
  matter.

Use the exact same format as the previous summary instruction."""

DEFAULT_MODEL_TOKEN_LIMITS: dict[str, int] = {
    "gpt-5.4": 257000,
    "gemini-3-flash-preview": 249000,
    "gemini-3.1-flash-preview": 249000,
    "gemini-3.1-flash-lite-preview": 249000,
    "openrouter/free": 199000,
    "gemma-3-27b-it": 14000,
    "mistral-large-latest": 255000,
}
_GEMINI_ALIAS_PROVIDERS = {
    "gemini",
    "google-gemini-cli",
    "google-antigravity",
}


@dataclass(slots=True)
class TokenBudget:
    """Resolved context window and reserved token budgets for a model."""

    context_window: int
    response_reserve: int
    target_input_tokens: int
    summary_tokens: int
    recent_tokens: int
    summary_input_tokens: int


@dataclass(slots=True)
class CompactionResult:
    """Result from compacting a prompt."""

    messages: list[dict[str, object]]
    original_tokens: int
    final_tokens: int
    was_compacted: bool
    used_model_summary: bool
    trimmed_content: bool
    warnings: list[str]


@dataclass(slots=True)
class ModelRunConfig:
    """Provider/model configuration for compaction-side LLM calls."""

    provider: str
    model: str
    api_keys: Sequence[str]
    base_url: str | None
    extra_headers: dict[str, str] | None
    model_parameters: dict[str, object] | None
    provider_slash_model: str | None = None
    configured_token_limits: Mapping[str, object] | None = None


def _collect_litellm_exceptions() -> tuple[type[Exception], ...]:
    return tuple(
        dict.fromkeys(
            exception_type
            for exception_type in vars(litellm.exceptions).values()
            if isinstance(exception_type, type)
            and issubclass(exception_type, Exception)
        ),
    )


COMPACTION_RETRYABLE_EXCEPTIONS = _collect_litellm_exceptions()


def resolve_context_window(
    *,
    provider: str,
    model: str,
    provider_slash_model: str | None = None,
    model_parameters: Mapping[str, object] | None = None,
    configured_token_limits: Mapping[str, object] | None = None,
) -> int:
    """Resolve the model context window from config, overrides, or defaults."""
    configured_window = _extract_context_window_override(model_parameters)
    if configured_window is not None:
        return configured_window

    exact_candidates, base_candidates = _build_context_window_lookup_candidates(
        provider=provider,
        model=model,
        provider_slash_model=provider_slash_model,
    )

    configured_window = _lookup_context_window_from_mapping(
        configured_token_limits,
        exact_candidates=exact_candidates,
        base_candidates=base_candidates,
    )
    if configured_window is not None:
        return configured_window

    default_window = _lookup_context_window_from_mapping(
        DEFAULT_MODEL_TOKEN_LIMITS,
        exact_candidates=exact_candidates,
        base_candidates=base_candidates,
    )
    return default_window or DEFAULT_CONTEXT_WINDOW_TOKENS


def build_token_budget(context_window: int) -> TokenBudget:
    """Build conservative token budgets for prompting and summarization."""
    response_reserve = min(
        DEFAULT_RESPONSE_RESERVE_TOKENS,
        max(MIN_RESPONSE_RESERVE_TOKENS, context_window // 8),
    )
    target_input_tokens = max(
        MIN_HARD_FIT_MESSAGE_TOKENS,
        context_window - response_reserve,
    )
    summary_tokens = min(
        MAX_SUMMARY_TOKENS,
        max(MIN_SUMMARY_TOKENS, context_window // 16),
    )
    recent_tokens = max(
        MIN_HARD_FIT_MESSAGE_TOKENS,
        target_input_tokens - summary_tokens,
    )
    summary_input_tokens = max(
        MIN_HARD_FIT_MESSAGE_TOKENS,
        context_window - (summary_tokens + MIN_RESPONSE_RESERVE_TOKENS),
    )
    return TokenBudget(
        context_window=context_window,
        response_reserve=response_reserve,
        target_input_tokens=target_input_tokens,
        summary_tokens=summary_tokens,
        recent_tokens=recent_tokens,
        summary_input_tokens=summary_input_tokens,
    )


async def compact_messages(
    messages: list[dict[str, object]],
    *,
    config: ModelRunConfig,
    use_model_summary: bool = True,
) -> CompactionResult:
    """Compact OpenAI-style messages in chronological order."""
    context_window = resolve_context_window(
        provider=config.provider,
        model=config.model,
        provider_slash_model=config.provider_slash_model,
        model_parameters=config.model_parameters,
        configured_token_limits=config.configured_token_limits,
    )
    budget = build_token_budget(context_window)
    original_tokens = count_conversation_tokens(messages)
    working_messages = copy.deepcopy(messages)

    if original_tokens <= budget.target_input_tokens:
        return CompactionResult(
            messages=working_messages,
            original_tokens=original_tokens,
            final_tokens=original_tokens,
            was_compacted=False,
            used_model_summary=False,
            trimmed_content=False,
            warnings=[],
        )

    system_messages, conversation_messages = _split_system_messages(working_messages)
    summary_message: dict[str, object] | None = None
    used_model_summary = False

    if len(conversation_messages) > 1:
        summary_source, recent_messages = _split_messages_for_summary(
            conversation_messages,
            recent_token_budget=budget.recent_tokens,
        )
        if summary_source:
            summary_body, used_model_summary = await _build_summary_body(
                summary_source,
                config=config,
                budget=budget,
                use_model_summary=use_model_summary,
            )
            summary_message = {
                "role": "assistant",
                "content": (
                    f"{SUMMARY_MESSAGE_HEADING}\n\n"
                    f"{summary_body or _build_fallback_summary(summary_source, budget)}"
                ),
            }
            conversation_messages = recent_messages

    compacted_messages = [*system_messages]
    if summary_message is not None:
        compacted_messages.append(summary_message)
    compacted_messages.extend(conversation_messages)

    hard_fit_messages, trimmed_content = _hard_fit_messages(
        compacted_messages,
        target_input_tokens=budget.target_input_tokens,
    )
    final_tokens = count_conversation_tokens(hard_fit_messages)
    warnings: list[str] = []
    if summary_message is not None:
        warnings.append(COMPACTED_CONTEXT_WARNING)
    if trimmed_content or final_tokens > budget.target_input_tokens:
        warnings.append(TRIMMED_CONTENT_WARNING)

    return CompactionResult(
        messages=hard_fit_messages,
        original_tokens=original_tokens,
        final_tokens=final_tokens,
        was_compacted=True,
        used_model_summary=used_model_summary,
        trimmed_content=trimmed_content,
        warnings=warnings,
    )


def _extract_context_window_override(
    model_parameters: Mapping[str, object] | None,
) -> int | None:
    if model_parameters is None:
        return None

    for key in (
        "context_window_tokens",
        "context_window",
        "token_limit",
        "max_input_tokens",
    ):
        raw_value = model_parameters.get(key)
        if isinstance(raw_value, bool):
            continue
        if not isinstance(raw_value, int | str):
            continue
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _build_context_window_lookup_candidates(
    *,
    provider: str,
    model: str,
    provider_slash_model: str | None,
) -> tuple[list[str], list[str]]:
    provider_key = _normalize_identifier_key(provider)
    model_key = _normalize_identifier_key(model)
    base_model_key = _strip_alias_suffixes(model_key, provider=provider_key)

    exact_candidates: list[str] = []
    base_candidates: list[str] = []

    if provider_slash_model:
        provider_model_key = _normalize_identifier_key(provider_slash_model)
        exact_candidates.append(provider_model_key)
        base_candidates.append(
            _strip_alias_suffixes_from_identifier(
                provider_model_key,
                default_provider=provider_key,
            ),
        )

    exact_candidates.extend((f"{provider_key}/{model_key}", model_key))
    base_candidates.extend((f"{provider_key}/{base_model_key}", base_model_key))
    return _dedupe_preserving_order(exact_candidates), _dedupe_preserving_order(
        base_candidates,
    )


def _lookup_context_window_from_mapping(
    configured_token_limits: Mapping[str, object] | None,
    *,
    exact_candidates: Sequence[str],
    base_candidates: Sequence[str],
) -> int | None:
    if not isinstance(configured_token_limits, Mapping):
        return None

    parsed_limits: dict[str, int] = {}
    for raw_key, raw_value in configured_token_limits.items():
        if not isinstance(raw_key, str):
            continue
        parsed_value = _parse_positive_int(raw_value)
        if parsed_value is None:
            continue
        parsed_limits[_normalize_identifier_key(raw_key)] = parsed_value

    for candidate in (*exact_candidates, *base_candidates):
        if candidate in parsed_limits:
            return parsed_limits[candidate]
    return None


def _parse_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, int | str):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _normalize_identifier_key(identifier: str) -> str:
    return identifier.strip().lower().removesuffix(":vision")


def _strip_alias_suffixes_from_identifier(
    identifier: str,
    *,
    default_provider: str,
) -> str:
    normalized_identifier = _normalize_identifier_key(identifier)
    provider, separator, model_name = normalized_identifier.partition("/")
    if not separator:
        return _strip_alias_suffixes(
            normalized_identifier,
            provider=default_provider,
        )
    return f"{provider}/{_strip_alias_suffixes(model_name, provider=provider)}"


def _strip_alias_suffixes(model_name: str, *, provider: str) -> str:
    stripped_model = model_name
    if provider in _GEMINI_ALIAS_PROVIDERS:
        stripped_model = strip_model_suffix_alias(
            stripped_model,
            GEMINI_THINKING_LEVEL_SUFFIXES,
        )
    if provider == OPENAI_CODEX_PROVIDER:
        stripped_model = strip_model_suffix_alias(
            stripped_model,
            OPENAI_REASONING_EFFORT_SUFFIXES,
        )
    return stripped_model


def _dedupe_preserving_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _split_system_messages(
    messages: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    system_messages: list[dict[str, object]] = []
    conversation_messages: list[dict[str, object]] = []
    in_system_prefix = True

    for message in messages:
        if in_system_prefix and message.get("role") == "system":
            system_messages.append(message)
            continue
        in_system_prefix = False
        conversation_messages.append(message)

    return system_messages, conversation_messages


def _split_messages_for_summary(
    messages: list[dict[str, object]],
    *,
    recent_token_budget: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    kept_reverse: list[dict[str, object]] = []
    kept_tokens = 0

    for message in reversed(messages):
        message_tokens = estimate_message_tokens(message)
        if kept_reverse and kept_tokens + message_tokens > recent_token_budget:
            break
        kept_reverse.append(message)
        kept_tokens += message_tokens

    if not kept_reverse and messages:
        kept_reverse.append(messages[-1])

    recent_messages = list(reversed(kept_reverse))
    summary_count = max(0, len(messages) - len(recent_messages))
    return messages[:summary_count], recent_messages


async def _build_summary_body(
    messages: list[dict[str, object]],
    *,
    config: ModelRunConfig,
    budget: TokenBudget,
    use_model_summary: bool,
) -> tuple[str, bool]:
    if not use_model_summary or not config.api_keys:
        return _build_fallback_summary(messages, budget), False

    summary = await _summarize_messages_with_model(
        messages,
        config=config,
        budget=budget,
    )
    if summary:
        return summary, True
    return _build_fallback_summary(messages, budget), False


async def _summarize_messages_with_model(
    messages: list[dict[str, object]],
    *,
    config: ModelRunConfig,
    budget: TokenBudget,
) -> str | None:
    previous_summary: str | None = None
    message_chunks = _chunk_messages_for_summary(
        messages,
        chunk_token_budget=budget.summary_input_tokens,
    )

    for chunk in message_chunks:
        prompt_text = _build_summary_prompt(
            messages=chunk,
            previous_summary=previous_summary,
        )
        response_text = await _complete_text_prompt(
            config=config,
            prompt_messages=[
                {
                    "role": "system",
                    "content": SUMMARIZATION_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt_text,
                },
            ],
            max_tokens=budget.summary_tokens,
        )
        if not response_text:
            return None
        previous_summary = response_text.strip()

    return previous_summary


def _chunk_messages_for_summary(
    messages: list[dict[str, object]],
    *,
    chunk_token_budget: int,
) -> list[list[dict[str, object]]]:
    chunks: list[list[dict[str, object]]] = []
    current_chunk: list[dict[str, object]] = []
    current_tokens = 0

    for message in messages:
        message_copy = copy.deepcopy(message)
        message_tokens = estimate_message_tokens(message_copy)
        if message_tokens > chunk_token_budget:
            message_copy = _trim_message_to_budget(
                message_copy,
                token_budget=chunk_token_budget,
            )
            message_tokens = estimate_message_tokens(message_copy)

        if current_chunk and current_tokens + message_tokens > chunk_token_budget:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append(message_copy)
        current_tokens += message_tokens

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def _build_summary_prompt(
    *,
    messages: list[dict[str, object]],
    previous_summary: str | None,
) -> str:
    conversation_text = _serialize_messages_for_summary(messages)
    prompt_parts = [f"<conversation>\n{conversation_text}\n</conversation>"]
    if previous_summary:
        prompt_parts.append(
            f"<previous-summary>\n{previous_summary}\n</previous-summary>",
        )
        prompt_parts.append(UPDATE_SUMMARIZATION_PROMPT)
    else:
        prompt_parts.append(SUMMARIZATION_PROMPT)
    return "\n\n".join(prompt_parts)


def _serialize_messages_for_summary(messages: list[dict[str, object]]) -> str:
    parts: list[str] = []

    for message in messages:
        role = str(message.get("role", "user")).capitalize()
        content = _flatten_message_content(message.get("content", ""))
        if content:
            parts.append(f"[{role}]: {content}")

    return "\n\n".join(parts)


def _flatten_message_content(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if not isinstance(item, Mapping):
            continue
        part = cast("Mapping[str, object]", item)
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text", "")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        elif part_type == "image_url":
            parts.append("[image]")
        elif part_type == "file":
            parts.append("[file attachment]")
    return "\n".join(parts).strip()


async def _complete_text_prompt(
    *,
    config: ModelRunConfig,
    prompt_messages: list[dict[str, object]],
    max_tokens: int,
) -> str | None:
    for api_key in config.api_keys:
        try:
            if config.provider in {"google-gemini-cli", "google-antigravity"}:
                response_text = await _collect_google_stream_text(
                    config=config,
                    api_key=api_key,
                    model_parameters=_build_summary_model_parameters(
                        config.model_parameters,
                        max_tokens=max_tokens,
                    ),
                    prompt_messages=prompt_messages,
                )
            elif config.provider == OPENAI_CODEX_PROVIDER:
                response_text = await _collect_openai_codex_stream_text(
                    config=config,
                    api_key=api_key,
                    model_parameters=_build_summary_model_parameters(
                        config.model_parameters,
                        max_tokens=max_tokens,
                    ),
                    prompt_messages=prompt_messages,
                )
            else:
                response_text = await _collect_litellm_stream_text(
                    config=config,
                    api_key=api_key,
                    prompt_messages=prompt_messages,
                    max_tokens=max_tokens,
                )
        except (
            TimeoutError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            httpx.HTTPError,
            *COMPACTION_RETRYABLE_EXCEPTIONS,
        ) as exc:
            logger.warning(
                "Compaction summary request failed for %s/%s: %s",
                config.provider,
                config.model,
                exc,
            )
            continue

        if response_text.strip():
            return response_text.strip()

    return None


def _build_summary_model_parameters(
    model_parameters: dict[str, object] | None,
    *,
    max_tokens: int,
) -> dict[str, object]:
    summary_params = dict(model_parameters or {})
    summary_params["max_output_tokens"] = max_tokens
    summary_params["max_tokens"] = max_tokens
    summary_params.setdefault("temperature", 0)
    summary_params.setdefault("text_verbosity", "low")
    return summary_params


async def _collect_google_stream_text(
    *,
    config: ModelRunConfig,
    api_key: str,
    model_parameters: dict[str, object],
    prompt_messages: list[dict[str, object]],
) -> str:
    chunks: list[str] = []
    stream = stream_google_gemini_cli(
        provider_id=config.provider,
        model=config.model,
        messages=prompt_messages,
        api_key=api_key,
        base_url=config.base_url,
        extra_headers=config.extra_headers,
        model_parameters=model_parameters,
        disable_tools=True,
    )
    async for delta_content, finish_reason, is_thinking in stream:
        if delta_content and not is_thinking:
            chunks.append(delta_content)
        if finish_reason is not None:
            break
    return "".join(chunks)


async def _collect_openai_codex_stream_text(
    *,
    config: ModelRunConfig,
    api_key: str,
    model_parameters: dict[str, object],
    prompt_messages: list[dict[str, object]],
) -> str:
    chunks: list[str] = []
    stream = stream_openai_codex(
        model=config.model,
        messages=prompt_messages,
        api_key=api_key,
        base_url=config.base_url,
        extra_headers=config.extra_headers,
        model_parameters=model_parameters,
        disable_tools=True,
    )
    async for delta_content, finish_reason, is_thinking in stream:
        if delta_content and not is_thinking:
            chunks.append(delta_content)
        if finish_reason is not None:
            break
    return "".join(chunks)


async def _collect_litellm_stream_text(
    *,
    config: ModelRunConfig,
    api_key: str,
    prompt_messages: list[dict[str, object]],
    max_tokens: int,
) -> str:
    chunks: list[str] = []
    litellm_kwargs = prepare_litellm_kwargs(
        provider=config.provider,
        model=config.model,
        messages=prompt_messages,
        api_key=api_key,
        options=LiteLLMOptions(
            base_url=config.base_url,
            extra_headers=config.extra_headers,
            stream=True,
            temperature=0,
        ),
    )
    litellm_kwargs["max_tokens"] = max_tokens

    stream = await litellm.acompletion(**litellm_kwargs)
    async for chunk in cast("AsyncIterator[object]", stream):
        if config.provider == "openrouter":
            raise_for_openrouter_payload_error(payload_obj=chunk)

        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        choice = choices[0]
        delta_content = getattr(getattr(choice, "delta", None), "content", "") or ""
        if delta_content:
            chunks.append(delta_content)
        if getattr(choice, "finish_reason", None) is not None:
            break

    return "".join(chunks)


def _build_fallback_summary(
    messages: list[dict[str, object]],
    budget: TokenBudget,
) -> str:
    remaining_tokens = budget.summary_tokens
    summary_lines = [
        (
            "Model-generated summary was unavailable, so earlier context was "
            "condensed into message snippets."
        ),
    ]
    remaining_tokens -= estimate_tokens_from_chars(len(summary_lines[0]))

    for message in messages:
        preview = _build_message_preview(message)
        if not preview:
            continue
        line = f"- {preview}"
        line_tokens = estimate_tokens_from_chars(len(line))
        if remaining_tokens - line_tokens < 0:
            break
        summary_lines.append(line)
        remaining_tokens -= line_tokens

    fallback_summary = "\n".join(summary_lines)
    return _trim_text_to_budget(fallback_summary, token_budget=budget.summary_tokens)


def _build_message_preview(message: dict[str, object]) -> str:
    role = str(message.get("role", "user")).capitalize()
    content = _flatten_message_content(message.get("content", ""))
    if not content:
        return f"{role}: [non-text content]"
    normalized = " ".join(content.split())
    trimmed = normalized[:240].rstrip()
    if len(normalized) > len(trimmed):
        trimmed = f"{trimmed}..."
    return f"{role}: {trimmed}"


def _hard_fit_messages(
    messages: list[dict[str, object]],
    *,
    target_input_tokens: int,
) -> tuple[list[dict[str, object]], bool]:
    working_messages = copy.deepcopy(messages)
    if count_conversation_tokens(working_messages) <= target_input_tokens:
        return working_messages, False

    system_messages, conversation_messages = _split_system_messages(working_messages)
    fitted_system_messages, system_trimmed = _fit_system_messages(
        system_messages,
        target_input_tokens=target_input_tokens,
    )
    remaining_budget = max(
        0,
        target_input_tokens - count_conversation_tokens(fitted_system_messages),
    )
    fitted_conversation_messages, conversation_trimmed = _fit_recent_messages(
        conversation_messages,
        target_input_tokens=remaining_budget,
    )
    return (
        [*fitted_system_messages, *fitted_conversation_messages],
        system_trimmed or conversation_trimmed,
    )


def _fit_system_messages(
    messages: list[dict[str, object]],
    *,
    target_input_tokens: int,
) -> tuple[list[dict[str, object]], bool]:
    if not messages:
        return [], False
    if count_conversation_tokens(messages) <= target_input_tokens:
        return messages, False

    working_messages = cast(
        "list[dict[str, object]]",
        copy.deepcopy(messages),
    )
    trimmed = False
    remaining_budget = target_input_tokens
    for index, message in enumerate(working_messages):
        remaining_messages = working_messages[index + 1 :]
        reserved_tokens = count_conversation_tokens(remaining_messages)
        budget_for_message = max(
            MIN_HARD_FIT_MESSAGE_TOKENS,
            remaining_budget - reserved_tokens,
        )
        current_tokens = estimate_message_tokens(message)
        if current_tokens > budget_for_message:
            working_messages[index] = _trim_message_to_budget(
                message,
                token_budget=budget_for_message,
            )
            trimmed = True
        remaining_budget -= estimate_message_tokens(working_messages[index])

    return working_messages, trimmed


def _fit_recent_messages(
    messages: list[dict[str, object]],
    *,
    target_input_tokens: int,
) -> tuple[list[dict[str, object]], bool]:
    if not messages or target_input_tokens <= 0:
        if not messages:
            return [], False
        return (
            [
                _trim_message_to_budget(
                    messages[-1],
                    token_budget=MIN_HARD_FIT_MESSAGE_TOKENS,
                ),
            ],
            True,
        )

    kept_reverse: list[dict[str, object]] = []
    remaining_budget = target_input_tokens
    trimmed = False

    for message in reversed(messages):
        message_copy = cast(
            "dict[str, object]",
            copy.deepcopy(message),
        )
        message_tokens = estimate_message_tokens(message_copy)
        if message_tokens <= remaining_budget:
            kept_reverse.append(message_copy)
            remaining_budget -= message_tokens
            continue

        if not kept_reverse:
            kept_reverse.append(
                _trim_message_to_budget(
                    message_copy,
                    token_budget=max(
                        MIN_HARD_FIT_MESSAGE_TOKENS,
                        remaining_budget,
                    ),
                ),
            )
            trimmed = True
            break

        if remaining_budget >= MIN_HARD_FIT_MESSAGE_TOKENS:
            kept_reverse.append(
                _trim_message_to_budget(
                    message_copy,
                    token_budget=remaining_budget,
                ),
            )
            trimmed = True
        else:
            trimmed = True
        break

    return list(reversed(kept_reverse)), trimmed


def _trim_message_to_budget(
    message: dict[str, object],
    *,
    token_budget: int,
) -> dict[str, object]:
    trimmed_message = copy.deepcopy(message)
    token_budget = max(1, token_budget)
    content = trimmed_message.get("content", "")
    trimmed_message["content"] = _trim_content_to_budget(
        content,
        token_budget=token_budget,
    )
    return trimmed_message


def _trim_content_to_budget(
    content: object,
    *,
    token_budget: int,
) -> object:
    if isinstance(content, str):
        return _trim_text_to_budget(content, token_budget=token_budget)
    if not isinstance(content, list):
        return "[content compacted]"

    remaining_budget = max(1, token_budget)
    kept_parts: list[dict[str, object]] = []
    dropped_non_text_parts = 0

    for part in content:
        if not isinstance(part, Mapping):
            continue
        (
            fitted_parts,
            remaining_budget,
            should_stop,
            dropped_non_text_count,
        ) = _fit_content_part(
            cast("Mapping[str, object]", part),
            remaining_budget=remaining_budget,
        )
        kept_parts.extend(fitted_parts)
        dropped_non_text_parts += dropped_non_text_count
        if should_stop:
            break

    if dropped_non_text_parts > 0:
        note = (
            f"[{dropped_non_text_parts} non-text attachment"
            f"{'' if dropped_non_text_parts == 1 else 's'} omitted]"
        )
        note_tokens = estimate_tokens_from_chars(len(note))
        if note_tokens <= remaining_budget:
            kept_parts.append({"type": "text", "text": note})

    if kept_parts:
        return kept_parts
    return [{"type": "text", "text": "[content compacted]"}]


def _fit_content_part(
    part: Mapping[str, object],
    *,
    remaining_budget: int,
) -> tuple[list[dict[str, object]], int, bool, int]:
    part_copy = copy.deepcopy(dict(part))
    part_tokens = estimate_message_tokens({"role": "user", "content": [part_copy]})
    if part_tokens <= remaining_budget:
        return [part_copy], remaining_budget - part_tokens, False, 0

    if part_copy.get("type") != "text":
        return [], remaining_budget, False, 1

    text = part_copy.get("text", "")
    if not isinstance(text, str):
        return [], remaining_budget, True, 0

    trimmed_text = _trim_text_to_budget(
        text,
        token_budget=remaining_budget,
    )
    if not trimmed_text:
        return [], remaining_budget, True, 0

    part_copy["text"] = trimmed_text
    trimmed_tokens = estimate_message_tokens({"role": "user", "content": [part_copy]})
    next_budget = max(0, remaining_budget - trimmed_tokens)
    return [part_copy], next_budget, True, 0


def _trim_text_to_budget(text: str, *, token_budget: int) -> str:
    if token_budget <= 0:
        return "[compacted]"

    target_chars = max(16, token_budget * 4)
    normalized_text = text.strip()
    if len(normalized_text) <= target_chars:
        return normalized_text

    if target_chars <= len(COMPACTION_MARKER) + 8:
        return f"{normalized_text[: target_chars - 3].rstrip()}..."

    remaining_chars = target_chars - len(COMPACTION_MARKER)
    head_chars = remaining_chars // 2
    tail_chars = remaining_chars - head_chars
    start = normalized_text[:head_chars].rstrip()
    end = normalized_text[-tail_chars:].lstrip()
    return f"{start}{COMPACTION_MARKER}{end}"
