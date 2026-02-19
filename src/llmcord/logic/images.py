"""Image extraction logic for LLM responses."""

import base64
import binascii
import hashlib
import io
import logging
import re
from collections.abc import Mapping
from typing import cast

import discord

from llmcord.core.error_handling import log_exception
from llmcord.logic.generation_types import (
    GeneratedImage,
    GenerationContext,
    GenerationState,
)

logger = logging.getLogger(__name__)

DATA_URL_PATTERN = re.compile(
    r"data:(image/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=]+)",
)


def _extension_from_mime(mime_type: str) -> str:
    extension = mime_type.split("/", maxsplit=1)[-1].split(";", maxsplit=1)[0]
    return extension or "png"


def _build_generated_image(data: bytes, mime_type: str) -> GeneratedImage:
    digest = hashlib.sha256(data).hexdigest()
    short_digest = digest[:12]
    extension = _extension_from_mime(mime_type)
    filename = f"gemini-output-{short_digest}.{extension}"
    return GeneratedImage(
        data=data,
        mime_type=mime_type,
        filename=filename,
        digest=digest,
    )


def _coerce_payload(obj: object) -> object:
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    dict_method = getattr(obj, "dict", None)
    if callable(dict_method):
        return dict_method()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return obj


def _extract_images_from_string(text: str) -> list[GeneratedImage]:
    images: list[GeneratedImage] = []
    for mime_type, b64_data in DATA_URL_PATTERN.findall(text):
        try:
            data = base64.b64decode(b64_data)
        except (ValueError, binascii.Error):
            continue
        images.append(_build_generated_image(data, mime_type))
    return images


def _extract_images_from_inline_data(inline_data: dict) -> list[GeneratedImage]:
    mime_type = inline_data.get("mime_type") or inline_data.get("mimeType")
    data = inline_data.get("data")
    if (
        not isinstance(mime_type, str)
        or not mime_type.startswith("image/")
        or not isinstance(data, str)
    ):
        return []
    try:
        decoded = base64.b64decode(data)
    except (ValueError, binascii.Error):
        return []
    return [_build_generated_image(decoded, mime_type)]


def _extract_images_from_image_url(image_url: object) -> list[GeneratedImage]:
    if isinstance(image_url, Mapping):
        mapping = cast("Mapping[str, object]", image_url)
        image_url = mapping.get("url")
    if not isinstance(image_url, str):
        return []
    return _extract_images_from_string(image_url)


def _extract_images_from_mime_data(
    data: object,
    mime_type: object,
) -> list[GeneratedImage]:
    if (
        not isinstance(mime_type, str)
        or not mime_type.startswith("image/")
        or not isinstance(data, str)
    ):
        return []
    try:
        decoded = base64.b64decode(data)
    except (ValueError, binascii.Error):
        return []
    return [_build_generated_image(decoded, mime_type)]


def _extract_images_from_dict(obj: dict) -> list[GeneratedImage]:
    images: list[GeneratedImage] = []

    inline_data = obj.get("inline_data") or obj.get("inlineData")
    if isinstance(inline_data, dict):
        images.extend(_extract_images_from_inline_data(inline_data))

    image_url = obj.get("image_url") or obj.get("imageUrl")
    if image_url:
        images.extend(_extract_images_from_image_url(image_url))

    images.extend(
        _extract_images_from_mime_data(obj.get("data"), obj.get("mime_type")),
    )
    images.extend(
        _extract_images_from_mime_data(obj.get("data"), obj.get("mimeType")),
    )
    return images


def extract_generated_images(value: object) -> list[GeneratedImage]:
    """Recursively extract generated images from any object payload."""
    images: list[GeneratedImage] = []
    seen_ids: set[int] = set()

    stack: list[object] = [value]
    while stack:
        obj = _coerce_payload(stack.pop())
        if obj is None:
            continue

        obj_id = id(obj)
        if obj_id in seen_ids:
            continue
        seen_ids.add(obj_id)

        if isinstance(obj, str):
            images.extend(_extract_images_from_string(obj))
            continue
        if isinstance(obj, dict):
            images.extend(_extract_images_from_dict(obj))
            stack.extend(obj.values())
            continue
        if isinstance(obj, (list, tuple, set)):
            stack.extend(obj)

    return images


def extract_gemini_images_from_chunk(
    chunk: object,
    choice: object,
    delta_content: str,
) -> list[GeneratedImage]:
    """Extract images from Gemini-specific response chunk fields."""
    images: list[GeneratedImage] = []
    sources = [
        delta_content,
        getattr(choice, "delta", None),
        getattr(choice, "message", None),
        getattr(chunk, "model_extra", None),
        getattr(chunk, "_hidden_params", None),
    ]
    for source in sources:
        if source is None:
            continue
        images.extend(extract_generated_images(source))
    return images


def append_generated_images(
    state: GenerationState,
    images: list[GeneratedImage],
) -> None:
    """Add unique generated images to the state tracking."""
    for image in images:
        if image.digest in state.generated_image_hashes:
            continue
        state.generated_image_hashes.add(image.digest)
        state.generated_images.append(image)


async def send_generated_images(
    *,
    context: GenerationContext,
    state: GenerationState,
) -> None:
    """Send all tracked generated images as replies in batches."""
    if not state.generated_images:
        logger.debug(
            "No Gemini-generated images to send for message %s",
            context.new_msg.id,
        )
        return

    logger.info(
        "Sending %s Gemini-generated image(s) for message %s",
        len(state.generated_images),
        context.new_msg.id,
    )
    reply_target = state.response_msgs[-1] if state.response_msgs else context.new_msg
    batch_size = 10
    for index in range(0, len(state.generated_images), batch_size):
        batch = state.generated_images[index : index + batch_size]
        try:
            files = [
                discord.File(io.BytesIO(image.data), filename=image.filename)
                for image in batch
            ]
            total_bytes = sum(len(image.data) for image in batch)
            logger.debug(
                "Prepared %s image(s) (%s bytes) for batch %s-%s",
                len(batch),
                total_bytes,
                index,
                index + len(batch) - 1,
            )
            content = "Generated image(s):" if index == 0 else None
            await reply_target.reply(content=content, files=files)
            logger.info(
                "Sent Gemini-generated image batch %s-%s for message %s",
                index,
                index + len(batch) - 1,
                context.new_msg.id,
            )
        except (discord.HTTPException, OSError, RuntimeError, ValueError) as exc:
            log_exception(
                logger=logger,
                message="Failed to send Gemini-generated image batch",
                error=exc,
                context={
                    "start_index": index,
                    "end_index": index + len(batch) - 1,
                    "message_id": context.new_msg.id,
                },
            )
