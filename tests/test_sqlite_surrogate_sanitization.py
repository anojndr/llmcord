from __future__ import annotations

import pathlib

import pytest

from llmcord.services.database import AppDB
from llmcord.services.database.messages import MessageResponsePayload

_SURROGATE_MIN = 0xD800
_SURROGATE_MAX = 0xDFFF


def _has_surrogates(text: str) -> bool:
    return any(_SURROGATE_MIN <= ord(ch) <= _SURROGATE_MAX for ch in text)


@pytest.mark.asyncio
async def test_save_message_search_data_replaces_unpaired_surrogates(
    tmp_path: pathlib.Path,
) -> None:
    db = AppDB(local_db_path=str(tmp_path / "test.db"))
    await db.init()

    # U+D83D is a high surrogate (commonly part of emoji) but unpaired here.
    bad_text = "hello \ud83d world"

    await db.asave_message_search_data(
        message_id="123",
        search_results=bad_text,
        tavily_metadata=None,
        lens_results=None,
    )

    (
        stored_search_results,
        _stored_metadata,
        _stored_lens,
    ) = await db.aget_message_search_data("123")
    assert stored_search_results is not None
    assert not _has_surrogates(stored_search_results)
    assert "\ufffd" in stored_search_results

    await db.aclose()


@pytest.mark.asyncio
async def test_save_message_response_data_replaces_unpaired_surrogates(
    tmp_path: pathlib.Path,
) -> None:
    db = AppDB(local_db_path=str(tmp_path / "test.db"))
    await db.init()

    bad_text = "response \ud83d here"
    payload = MessageResponsePayload(
        request_message_id="111",
        request_user_id="222",
        full_response=bad_text,
        thought_process=bad_text,
    )

    await db.asave_message_response_data("999", payload)

    full_response, thought_process, *_rest = await db.aget_message_response_data("999")
    assert full_response is not None
    assert thought_process is not None
    assert not _has_surrogates(full_response)
    assert not _has_surrogates(thought_process)
    assert "\ufffd" in full_response
    assert "\ufffd" in thought_process

    await db.aclose()
