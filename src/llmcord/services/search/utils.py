"""Utility functions for search service."""

import base64
import binascii
import logging
import re
from datetime import datetime
from typing import Any

try:
    import fitz  # type: ignore[import-untyped]
except ImportError:
    fitz = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def get_current_datetime_strings() -> tuple[str, str]:
    """Get current date and time strings for system prompts.

    Returns a tuple of (date_str, time_str).

    Date format: "January 21 2026"
    Time format: "20:00:00 +0800"
    """
    now = datetime.now().astimezone()
    date_str = now.strftime("%B %d %Y")
    time_str = now.strftime("%H:%M:%S %Z%z")
    return date_str, time_str


def _describe_mime_type(mime_type: str) -> str:
    """Return a human-readable description for a MIME type."""
    mime_descriptions: dict[str, str] = {
        "audio": "[Audio file attached]",
        "video": "[Video file attached]",
    }
    for prefix, description in mime_descriptions.items():
        if mime_type.startswith(prefix):
            return description
    return f"[File attached: {mime_type}]"


def _extract_pdf_text_for_decider(file_data: str) -> str:
    """Try to extract text from base64-encoded PDF data.

    Args:
        file_data: The full data URL string (data:application/pdf;base64,...).

    Returns:
        Extracted text with header, or a placeholder string.

    """
    if fitz is None:
        return "[PDF document attached]"

    b64_data = file_data.split(",", 1)[1] if "," in file_data else ""
    try:
        pdf_bytes = base64.b64decode(b64_data)
    except (ValueError, binascii.Error):
        return "[PDF document attached]"

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            text = "\n".join(page.get_text() for page in doc)
        finally:
            doc.close()
        if text.strip():
            return f"--- PDF Attachment Content ---\n{text.strip()}"
    except (RuntimeError, ValueError, OSError):
        logger.debug("Could not extract PDF text for search decider")
    return "[PDF document attached]"


def _extract_file_part_description(part: dict) -> str | None:
    """Extract a text description from a Gemini file part.

    For non-Gemini models, converts file parts (PDFs, audio, video) into
    text descriptions so the decider is aware of attached content.

    Args:
        part: A dict with type 'file' and nested file data.

    Returns:
        A text description string, or None if the part cannot be described.

    """
    file_info = part.get("file", {})
    file_data = file_info.get("file_data", "")
    if not file_data:
        return None

    # Parse the data URL to get the mime type
    match = re.match(r"data:([^;]+);base64,", file_data)
    if not match:
        return None

    mime_type = match.group(1)
    if mime_type == "application/pdf":
        return _extract_pdf_text_for_decider(file_data)
    return _describe_mime_type(mime_type)


def _filter_content_parts(
    content: list,
    *,
    is_gemini: bool,
) -> list[dict]:
    """Filter content parts based on the target model type.

    For Gemini models, all content types (text, image_url, file) are kept.
    For non-Gemini models, 'file' parts are converted to text descriptions
    so the decider is aware of any file content, URL content, or Google Lens
    content attached to the message.

    Args:
        content: List of content part dicts.
        is_gemini: Whether the target model is a Gemini model.

    Returns:
        Filtered list of content part dicts.

    """
    accepted_types = {"text", "image_url"}
    if is_gemini:
        accepted_types.add("file")

    filtered: list[dict] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in accepted_types:
            filtered.append(part)
        elif part_type == "file":
            # Non-Gemini model: convert file parts to text descriptions
            description = _extract_file_part_description(part)
            if description:
                filtered.append({"type": "text", "text": description})

    return filtered


def convert_messages_to_openai_format(
    messages: list,
    system_prompt: str | None = None,
    *,
    reverse: bool = True,
    include_analysis_prompt: bool = False,
    is_gemini: bool = False,
) -> list[dict]:
    """Convert internal message format to OpenAI-compatible message format.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        system_prompt: Optional system prompt to prepend
        reverse: Whether to reverse the message order (default True for
            chronological)
        include_analysis_prompt: Whether to append the analysis instruction
            prompt
        is_gemini: Whether the target model is a Gemini model. When True,
            'file' type content parts are preserved. When False, 'file'
            parts are converted to text descriptions.

    Returns:
        List of OpenAI-compatible message dicts

    """
    openai_messages: list[dict[str, Any]] = []

    if system_prompt:
        openai_messages.append({"role": "system", "content": system_prompt})

    message_list = messages[::-1] if reverse else messages

    for msg in message_list:
        role = msg.get("role", "user")
        if role == "system":
            continue  # Skip system messages from chat history

        content = msg.get("content", "")
        if isinstance(content, list):
            filtered_content = _filter_content_parts(
                content,
                is_gemini=is_gemini,
            )
            if filtered_content:
                openai_messages.append(
                    {"role": role, "content": filtered_content},
                )
        elif content:
            openai_messages.append({"role": role, "content": str(content)})

    if include_analysis_prompt:
        openai_messages.append(
            {
                "role": "user",
                "content": (
                    "Based on the conversation above, analyze the last user "
                    "query and respond with your JSON decision."
                ),
            },
        )

    return openai_messages
