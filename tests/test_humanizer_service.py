from __future__ import annotations

from llmcord.services.humanizer import split_text_for_word_limit


def test_split_text_for_word_limit_no_split_needed() -> None:
    text = "one two three"
    assert split_text_for_word_limit(text, 125) == ["one two three"]


def test_split_text_for_word_limit_splits_in_order() -> None:
    words = [f"w{i}" for i in range(1, 136)]
    text = " ".join(words)

    segments = split_text_for_word_limit(text, 125)

    expected_segment_count = 2
    assert len(segments) == expected_segment_count
    assert segments[0].split() == words[:125]
    assert segments[1].split() == words[125:]


def test_split_text_for_word_limit_multiple_full_chunks_and_remainder() -> None:
    words = [f"w{i}" for i in range(1, 386)]
    text = " ".join(words)

    segments = split_text_for_word_limit(text, 125)

    assert [len(segment.split()) for segment in segments] == [125, 125, 125, 10]
    rebuilt = " ".join(segments)
    assert rebuilt.split() == words
