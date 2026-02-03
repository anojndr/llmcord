"""Utility functions for search service."""
from datetime import datetime


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


def convert_messages_to_openai_format(
    messages: list,
    system_prompt: str | None = None,
    *,
    reverse: bool = True,
    include_analysis_prompt: bool = False,
) -> list[dict]:
    """Convert internal message format to OpenAI-compatible message format.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        system_prompt: Optional system prompt to prepend
        reverse: Whether to reverse the message order (default True for
            chronological)
        include_analysis_prompt: Whether to append the analysis instruction
            prompt

    Returns:
        List of OpenAI-compatible message dicts

    """
    openai_messages = []

    if system_prompt:
        openai_messages.append({"role": "system", "content": system_prompt})

    message_list = messages[::-1] if reverse else messages

    for msg in message_list:
        role = msg.get("role", "user")
        if role == "system":
            continue  # Skip system messages from chat history

        content = msg.get("content", "")
        if isinstance(content, list):
            # Filter to only include types supported by OpenAI-compatible APIs
            # GitHub Copilot and others only accept 'text' and 'image_url' types
            filtered_content = [
                part
                for part in content
                if isinstance(part, dict)
                and part.get("type") in ("text", "image_url")
            ]
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
