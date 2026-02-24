from __future__ import annotations

from typing import Any

import pytest

from llmcord.logic.messages import MessageBuildContext, build_messages

from ._fakes import DummyTwitterApi, FakeAttachment, FakeMessage, FakeUser


class _FakeDB:
    def get_message_search_data(self, _message_id: str) -> tuple[None, None, None]:
        return None, None, None


class _DummyBot:
    def __init__(self, user_id: int = 999) -> None:
        self.user = FakeUser(user_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "at ai",
        "<@999>",
        "<@!999>",
        "<@999> at ai",
        "at ai <@999>",
        "  <@!999>  at ai  ",
    ],
)
async def test_trigger_only_becomes_dot(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: Any,
    msg_nodes: dict[int, object],
    content: str,
) -> None:
    async def _fake_download_and_process_attachments(**kwargs: object):
        return [], [], []

    async def _noop_set_parent_message(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "llmcord.logic.messages.download_and_process_attachments",
        _fake_download_and_process_attachments,
    )
    monkeypatch.setattr(
        "llmcord.logic.messages._set_parent_message",
        _noop_set_parent_message,
    )
    monkeypatch.setattr("llmcord.logic.messages.get_db", lambda: _FakeDB())

    bot = _DummyBot()
    msg = FakeMessage(
        id=1,
        content=content,
        author=FakeUser(1234),
    )

    result = await build_messages(
        context=MessageBuildContext(
            new_msg=msg,  # type: ignore[arg-type]
            discord_bot=bot,  # type: ignore[arg-type]
            httpx_client=httpx_client,
            twitter_api=DummyTwitterApi(),
            msg_nodes=msg_nodes,  # type: ignore[arg-type]
            actual_model="gpt-4o",
            accept_usernames=False,
            max_text=100000,
            max_images=0,
            max_messages=1,
            max_tweet_replies=0,
            enable_youtube_transcripts=False,
            youtube_transcript_method="youtube-transcript-api",
        ),
    )

    assert result.messages, "Expected at least one built message"
    assert str(result.messages[0]["content"]) == "."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "at ai",
        "<@999>",
        "<@!999>",
        "<@999> at ai",
        "at ai <@999>",
    ],
)
async def test_trigger_only_with_attachment_becomes_dot(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: Any,
    msg_nodes: dict[int, object],
    content: str,
) -> None:
    async def _fake_download_and_process_attachments(**kwargs: object):
        return [], [], []

    async def _noop_set_parent_message(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "llmcord.logic.messages.download_and_process_attachments",
        _fake_download_and_process_attachments,
    )
    monkeypatch.setattr(
        "llmcord.logic.messages._set_parent_message",
        _noop_set_parent_message,
    )
    monkeypatch.setattr("llmcord.logic.messages.get_db", lambda: _FakeDB())

    bot = _DummyBot()
    msg = FakeMessage(
        id=1,
        content=content,
        author=FakeUser(1234),
        attachments=[
            FakeAttachment(
                url="https://example.com/file.png",
                content_type="image/png",
            ),
        ],
    )

    result = await build_messages(
        context=MessageBuildContext(
            new_msg=msg,  # type: ignore[arg-type]
            discord_bot=bot,  # type: ignore[arg-type]
            httpx_client=httpx_client,
            twitter_api=DummyTwitterApi(),
            msg_nodes=msg_nodes,  # type: ignore[arg-type]
            actual_model="gpt-4o",
            accept_usernames=False,
            max_text=100000,
            max_images=0,
            max_messages=1,
            max_tweet_replies=0,
            enable_youtube_transcripts=False,
            youtube_transcript_method="youtube-transcript-api",
        ),
    )

    assert result.messages, "Expected at least one built message"
    assert str(result.messages[0]["content"]) == "."
