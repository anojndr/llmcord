from __future__ import annotations

from llmcord.logic.utils import count_conversation_tokens, count_text_tokens


def test_count_text_tokens_uses_compaction_heuristic() -> None:
    assert count_text_tokens("abcdefghij") == 3


def test_count_conversation_tokens_uses_multimodal_heuristic() -> None:
    messages: list[dict[str, object]] = [
        {"role": "user", "content": "abcd"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "abcde"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {"file_data": "data:application/pdf;base64,BBBB"},
                },
                {"text": "xyz"},
            ],
        },
    ]

    assert count_conversation_tokens(messages) == 1 + 1202 + 1201
