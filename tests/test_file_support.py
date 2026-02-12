from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, cast

import httpx
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
async def test_text_attachment_appended(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: Any,
    msg_nodes: dict[int, object],
) -> None:
    attachment = FakeAttachment(
        url="https://cdn.discordapp.com/attachments/1/2/note.txt",
        content_type="text/plain",
        filename="note.txt",
    )

    response = httpx.Response(200, text="hello from attachment")

    async def _fake_download_and_process_attachments(**kwargs: object):
        processed = [
            {
                "content_type": "text/plain",
                "content": b"hello",
                "text": "hello from attachment",
            },
        ]
        return [attachment], [response], processed

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
    monkeypatch.setattr("llmcord.logic.messages.get_bad_keys_db", lambda: _FakeDB())

    bot = _DummyBot()
    msg = FakeMessage(
        id=30,
        content="at ai summarize this file",
        author=FakeUser(1234),
        attachments=[attachment],
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
            max_tweet_replies=50,
            enable_youtube_transcripts=True,
            youtube_transcript_proxy=None,
            reddit_proxy=None,
            proxy_url=None,
        ),
    )

    user_content = str(result.messages[0]["content"])
    assert "hello from attachment" in user_content


@pytest.mark.asyncio
async def test_pdf_text_extracted_and_appended_for_non_gemini(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: Any,
    msg_nodes: dict[int, object],
) -> None:
    attachment = FakeAttachment(
        url="https://cdn.discordapp.com/attachments/1/2/doc.pdf",
        content_type="application/pdf",
        filename="doc.pdf",
    )

    response = httpx.Response(200, content=b"%PDF-1.4 dummy")

    async def _fake_download_and_process_attachments(**kwargs: object):
        processed = [
            {"content_type": "application/pdf", "content": b"%PDF-1.4", "text": None},
        ]
        return [attachment], [response], processed

    async def _fake_extract_pdf_text(_pdf_bytes: bytes) -> str:
        return "PDF TEXT CONTENT"

    async def _noop_set_parent_message(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "llmcord.logic.messages.download_and_process_attachments",
        _fake_download_and_process_attachments,
    )
    monkeypatch.setattr(
        "llmcord.logic.content.extract_pdf_text",
        _fake_extract_pdf_text,
    )
    monkeypatch.setattr(
        "llmcord.logic.messages._set_parent_message",
        _noop_set_parent_message,
    )
    monkeypatch.setattr("llmcord.logic.messages.get_bad_keys_db", lambda: _FakeDB())

    bot = _DummyBot()
    msg = FakeMessage(
        id=31,
        content="at ai summarize",
        author=FakeUser(1234),
        attachments=[attachment],
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
            max_tweet_replies=50,
            enable_youtube_transcripts=True,
            youtube_transcript_proxy=None,
            reddit_proxy=None,
            proxy_url=None,
        ),
    )

    user_content = str(result.messages[0]["content"])
    assert "PDF TEXT CONTENT" in user_content


@pytest.mark.asyncio
async def test_image_attachment_included(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: Any,
    msg_nodes: dict[int, object],
) -> None:
    asset_path = Path("image-for-google-lens-test/chrome_jHqK1D4oUn.png")
    img_bytes = asset_path.read_bytes()

    attachment = FakeAttachment(
        url="https://cdn.discordapp.com/attachments/1/2/chrome_jHqK1D4oUn.png",
        content_type="image/png",
        filename=asset_path.name,
    )
    response = httpx.Response(200, content=img_bytes)

    async def _fake_download_and_process_attachments(**kwargs: object):
        processed = [{"content_type": "image/png", "content": img_bytes, "text": None}]
        return [attachment], [response], processed

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
    monkeypatch.setattr("llmcord.logic.messages.get_bad_keys_db", lambda: _FakeDB())

    bot = _DummyBot()
    msg = FakeMessage(
        id=32,
        content="at ai what's in this?",
        author=FakeUser(1234),
        attachments=[attachment],
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
            max_images=5,
            max_messages=1,
            max_tweet_replies=50,
            enable_youtube_transcripts=True,
            youtube_transcript_proxy=None,
            reddit_proxy=None,
            proxy_url=None,
        ),
    )

    content = result.messages[0]["content"]
    assert isinstance(content, list)
    parts = cast("list[dict[str, object]]", content)
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    assert image_parts

    image_url_obj = image_parts[0].get("image_url")
    assert isinstance(image_url_obj, dict)
    image_url_map = cast("dict[str, object]", image_url_obj)
    url_obj = image_url_map.get("url")
    assert isinstance(url_obj, str)
    assert url_obj.startswith("data:image/png;base64,")

    payload = url_obj.split(",", 1)[1]
    assert base64.b64decode(payload)
