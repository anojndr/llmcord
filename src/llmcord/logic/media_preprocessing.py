"""Gemini preprocessing for audio and video attachments."""

import logging
from base64 import b64encode
from collections.abc import Awaitable, Callable
from typing import cast

import httpx
import litellm

from llmcord.core.config import get_config, is_gemini_model
from llmcord.core.error_handling import log_exception
from llmcord.core.exceptions import LITELLM_TIMEOUT_SECONDS
from llmcord.logic.providers import (
    ProviderSettings,
    resolve_provider_settings_for_model,
)
from llmcord.services.llm import LiteLLMOptions, prepare_litellm_kwargs
from llmcord.services.llm.providers.gemini_cli import stream_google_gemini_cli

logger = logging.getLogger(__name__)

_GEMINI_MEDIA_PROVIDERS = {"gemini", "google-gemini-cli", "google-antigravity"}
_MEDIA_PREPROCESSOR_MODEL_KEY = "media_preprocessor_model"
_DEFAULT_MEDIA_PREPROCESSOR_MODEL = "gemini/gemini-3.1-flash-lite-preview"
_VIDEO_PROMPT = """Describe this video per timestamp, then transcribe it per
timestamp. The output should look like this:

Example: 30-second video

<output>
Video description per timestamp:

0s to 10s: A cat jumps down from the cabinet
10s to 20s: The cat licks its toes
20s to 30s: The cat opens its mouth, probably meowing

Video transcription per timestamp:

0s to 10s: Come on, jump down, kitty kitty
11s to 20s: You are so cute while licking your toes
21s to 30s: Why are you meowing? What do you want?
</output>"""
_AUDIO_PROMPT = """Transcribe this audio per timestamp. The output should look
like this:

Example: 30-second audio

<output>
Audio transcription per timestamp:

0s to 10s: Come on, jump down, kitty kitty
11s to 20s: You are so cute while licking your toes
21s to 30s: Why are you meowing? What do you want?
</output>"""


def _collect_litellm_exceptions() -> tuple[type[Exception], ...]:
    return tuple(
        dict.fromkeys(
            exception_type
            for exception_type in vars(litellm.exceptions).values()
            if isinstance(exception_type, type)
            and issubclass(exception_type, Exception)
        ),
    )


_LITELLM_RETRYABLE_EXCEPTIONS = _collect_litellm_exceptions()


def _iter_candidate_models(config: dict[str, object]) -> list[str]:
    candidates: list[str] = [_DEFAULT_MEDIA_PREPROCESSOR_MODEL]

    for key in (
        _MEDIA_PREPROCESSOR_MODEL_KEY,
        "web_search_decider_model",
    ):
        candidate = config.get(key)
        if isinstance(candidate, str):
            candidate = candidate.strip()
            if candidate:
                candidates.append(candidate)

    models = config.get("models", {})
    if isinstance(models, dict):
        candidates.extend(
            candidate.strip()
            for candidate in models
            if isinstance(candidate, str) and candidate.strip()
        )

    return list(dict.fromkeys(candidates))


def resolve_media_preprocessor_settings() -> ProviderSettings | None:
    """Pick a configured Gemini model that can analyze media files."""
    config = get_config()

    for candidate in _iter_candidate_models(config):
        settings = resolve_provider_settings_for_model(
            candidate,
            allow_missing_api_keys=False,
        )
        if settings is None:
            continue
        if settings.provider not in _GEMINI_MEDIA_PROVIDERS:
            continue
        if not is_gemini_model(settings.actual_model):
            continue
        return settings

    return None


def _media_prompt_for_content_type(content_type: str) -> str:
    if content_type.startswith("video/"):
        return _VIDEO_PROMPT
    return _AUDIO_PROMPT


def _build_media_request(
    *,
    content_type: str,
    content_bytes: bytes,
) -> list[dict[str, object]]:
    encoded_data = b64encode(content_bytes).decode("utf-8")
    file_data = f"data:{content_type};base64,{encoded_data}"
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": _media_prompt_for_content_type(content_type),
                },
                {"type": "file", "file": {"file_data": file_data}},
            ],
        },
    ]


def _extract_response_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_dict = cast("dict[str, object]", part)
            part_text = part_dict.get("text")
            if isinstance(part_text, str):
                text_parts.append(part_text)
        return "\n".join(text_parts).strip()

    return str(content).strip()


async def _run_gemini_media_request(
    *,
    provider_settings: ProviderSettings,
    messages: list[dict[str, object]],
) -> str | None:
    for api_key in provider_settings.api_keys:
        try:
            if provider_settings.provider in {
                "google-gemini-cli",
                "google-antigravity",
            }:
                response_chunks: list[str] = []
                stream = stream_google_gemini_cli(
                    provider_id=provider_settings.provider,
                    model=provider_settings.actual_model,
                    messages=messages,
                    api_key=api_key,
                    base_url=provider_settings.base_url,
                    extra_headers=provider_settings.extra_headers,
                    model_parameters=provider_settings.model_parameters,
                    disable_tools=True,
                )
                async for chunk in stream:
                    delta_content, _chunk_finish_reason, is_thinking = chunk
                    if delta_content and not is_thinking:
                        response_chunks.append(delta_content)

                response_text = "".join(response_chunks).strip()
                if response_text:
                    return response_text
                continue

            litellm_kwargs = prepare_litellm_kwargs(
                provider=provider_settings.provider,
                model=provider_settings.actual_model,
                messages=messages,
                api_key=api_key,
                options=LiteLLMOptions(
                    base_url=provider_settings.base_url,
                    extra_headers=provider_settings.extra_headers,
                    model_parameters=provider_settings.model_parameters,
                ),
            )
            litellm_kwargs["timeout"] = LITELLM_TIMEOUT_SECONDS
            response = await litellm.acompletion(**litellm_kwargs)
            response_text = _extract_response_text(
                response.choices[0].message.content or "",
            )
            if response_text:
                return response_text
        except (
            TimeoutError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            httpx.HTTPError,
            *_LITELLM_RETRYABLE_EXCEPTIONS,
        ) as exc:
            log_exception(
                logger=logger,
                message="Gemini media preprocessing failed",
                error=exc,
                context={
                    "provider": provider_settings.provider,
                    "model": provider_settings.actual_model,
                },
            )

    return None


def _build_attachment_label(content_type: str, attachment_index: int) -> str:
    kind = "Video" if content_type.startswith("video/") else "Audio"
    return f"{kind} attachment {attachment_index}"


def _format_processed_output(
    *,
    content_type: str,
    attachment_index: int,
    output_text: str,
) -> str:
    label = _build_attachment_label(content_type, attachment_index)
    return f"--- Gemini preprocessing for {label} ---\n{output_text.strip()}"


def _format_failed_output(
    *,
    content_type: str,
    attachment_index: int,
    failure_reason: str,
) -> str:
    label = _build_attachment_label(content_type, attachment_index)
    return f"--- {label} ---\nGemini preprocessing failed: {failure_reason}"


async def preprocess_media_attachments_with_gemini(
    *,
    actual_model: str,
    processed_attachments: list[dict[str, bytes | str | None]],
    status_callback: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[list[str], bool]:
    """Turn audio/video attachments into text for non-Gemini target models."""
    if is_gemini_model(actual_model):
        return [], False

    media_attachments = [
        attachment
        for attachment in processed_attachments
        if isinstance(attachment.get("content"), bytes)
        and isinstance(attachment.get("content_type"), str)
        and cast("str", attachment["content_type"]).startswith(("audio/", "video/"))
    ]
    if not media_attachments:
        return [], False

    provider_settings = resolve_media_preprocessor_settings()
    if provider_settings is None:
        logger.warning(
            "No configured Gemini model is available for media preprocessing",
        )
        failed_outputs = [
            _format_failed_output(
                content_type=cast("str", attachment["content_type"]),
                attachment_index=index,
                failure_reason=("no Gemini media-preprocessing model is configured"),
            )
            for index, attachment in enumerate(media_attachments, start=1)
        ]
        return failed_outputs, True

    if status_callback is not None:
        await status_callback(
            (
                f"Analyzing {len(media_attachments)} audio/video "
                "attachment(s) with Gemini..."
            ),
        )

    media_outputs: list[str] = []
    had_failure = False

    for index, attachment in enumerate(media_attachments, start=1):
        content_type = cast("str", attachment["content_type"])
        content_bytes = cast("bytes", attachment["content"])
        response_text = await _run_gemini_media_request(
            provider_settings=provider_settings,
            messages=_build_media_request(
                content_type=content_type,
                content_bytes=content_bytes,
            ),
        )
        if response_text:
            media_outputs.append(
                _format_processed_output(
                    content_type=content_type,
                    attachment_index=index,
                    output_text=response_text,
                ),
            )
            continue

        had_failure = True
        media_outputs.append(
            _format_failed_output(
                content_type=content_type,
                attachment_index=index,
                failure_reason="Gemini returned no analysis",
            ),
        )

    return media_outputs, had_failure
