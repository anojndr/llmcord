from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from llmcord.logic.messages import MessageBuildContext, build_messages
from llmcord.services.database import AppDB
from llmcord.services.facebook import (
    DownloadedFacebookVideo,
    FacebookDownloadResult,
)
from llmcord.services.tiktok import DownloadedTikTokVideo, TikTokDownloadResult

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
    monkeypatch.setattr("llmcord.logic.messages.get_db", lambda: _FakeDB())

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
            youtube_transcript_method="youtube-transcript-api",
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
    monkeypatch.setattr("llmcord.logic.messages.get_db", lambda: _FakeDB())

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
            youtube_transcript_method="youtube-transcript-api",
        ),
    )

    user_content = str(result.messages[0]["content"])
    assert "PDF TEXT CONTENT" in user_content


@pytest.mark.asyncio
async def test_audio_attachment_preprocessed_with_gemini_for_non_gemini_model(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: Any,
    msg_nodes: dict[int, object],
) -> None:
    attachment = FakeAttachment(
        url="https://cdn.discordapp.com/attachments/1/2/audio.mp3",
        content_type="audio/mpeg",
        filename="audio.mp3",
    )
    response = httpx.Response(200, content=b"fake-audio")

    async def _fake_download_and_process_attachments(**kwargs: object):
        assert kwargs["attachments"] == [attachment]
        processed = [
            {
                "content_type": "audio/mpeg",
                "content": b"fake-audio",
                "text": None,
            },
        ]
        return [attachment], [response], processed

    async def _fake_preprocess_media_attachments_with_gemini(**kwargs: object):
        assert kwargs["actual_model"] == "gpt-5.4"
        return (
            [
                (
                    "--- Gemini preprocessing for Audio attachment 1 ---\n"
                    "Audio transcription per timestamp:\n\n"
                    "0s to 10s: hello there"
                ),
            ],
            False,
        )

    async def _noop_set_parent_message(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "llmcord.logic.messages.download_and_process_attachments",
        _fake_download_and_process_attachments,
    )
    monkeypatch.setattr(
        "llmcord.logic.messages.preprocess_media_attachments_with_gemini",
        _fake_preprocess_media_attachments_with_gemini,
    )
    monkeypatch.setattr(
        "llmcord.logic.messages._set_parent_message",
        _noop_set_parent_message,
    )
    monkeypatch.setattr("llmcord.logic.messages.get_db", lambda: _FakeDB())

    bot = _DummyBot()
    msg = FakeMessage(
        id=311,
        content="at ai summarize this audio",
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
            actual_model="gpt-5.4",
            accept_usernames=False,
            max_text=100000,
            max_images=0,
            max_messages=1,
            max_tweet_replies=50,
            enable_youtube_transcripts=True,
            youtube_transcript_method="youtube-transcript-api",
        ),
    )

    user_content = str(result.messages[0]["content"])
    assert "Audio transcription per timestamp" in user_content
    assert "0s to 10s: hello there" in user_content
    assert "⚠️ Some audio/video attachments could not be analyzed" not in (
        result.user_warnings
    )


@pytest.mark.asyncio
async def test_video_attachment_preprocessing_failure_adds_warning(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: Any,
    msg_nodes: dict[int, object],
) -> None:
    attachment = FakeAttachment(
        url="https://cdn.discordapp.com/attachments/1/2/video.mp4",
        content_type="video/mp4",
        filename="video.mp4",
    )
    response = httpx.Response(200, content=b"fake-video")

    async def _fake_download_and_process_attachments(**kwargs: object):
        assert kwargs["attachments"] == [attachment]
        processed = [
            {
                "content_type": "video/mp4",
                "content": b"fake-video",
                "text": None,
            },
        ]
        return [attachment], [response], processed

    async def _fake_preprocess_media_attachments_with_gemini(**kwargs: object):
        assert kwargs["actual_model"] == "gpt-5.4"
        return (
            [
                (
                    "--- Video attachment 1 ---\n"
                    "Gemini preprocessing failed: "
                    "no Gemini media-preprocessing model is configured"
                ),
            ],
            True,
        )

    async def _noop_set_parent_message(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "llmcord.logic.messages.download_and_process_attachments",
        _fake_download_and_process_attachments,
    )
    monkeypatch.setattr(
        "llmcord.logic.messages.preprocess_media_attachments_with_gemini",
        _fake_preprocess_media_attachments_with_gemini,
    )
    monkeypatch.setattr(
        "llmcord.logic.messages._set_parent_message",
        _noop_set_parent_message,
    )
    monkeypatch.setattr("llmcord.logic.messages.get_db", lambda: _FakeDB())

    bot = _DummyBot()
    msg = FakeMessage(
        id=312,
        content="at ai",
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
            actual_model="gpt-5.4",
            accept_usernames=False,
            max_text=100000,
            max_images=0,
            max_messages=1,
            max_tweet_replies=50,
            enable_youtube_transcripts=True,
            youtube_transcript_method="youtube-transcript-api",
        ),
    )

    user_content = str(result.messages[0]["content"])
    assert "Gemini preprocessing failed" in user_content
    assert "⚠️ Some audio/video attachments could not be analyzed" in (
        result.user_warnings
    )


@pytest.mark.asyncio
async def test_audio_preprocessing_cache_reused_after_restart(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: Any,
    msg_nodes: dict[int, object],
    tmp_path: Path,
) -> None:
    db = AppDB(local_db_path=str(tmp_path / "media-cache.db"))
    await db.init()

    attachment = FakeAttachment(
        url="https://cdn.discordapp.com/attachments/1/2/audio.mp3",
        content_type="audio/mpeg",
        filename="audio.mp3",
    )
    response = httpx.Response(200, content=b"fake-audio")
    cached_output = (
        "--- Gemini preprocessing for Audio attachment 1 ---\n"
        "Audio transcription per timestamp:\n\n"
        "0s to 10s: cached audio transcript"
    )
    download_attachment_counts: list[int] = []
    preprocess_call_count = 0

    async def _fake_download_and_process_attachments(**kwargs: object):
        attachments = cast("list[FakeAttachment]", kwargs["attachments"])
        download_attachment_counts.append(len(attachments))
        if not attachments:
            return [], [], []

        assert attachments == [attachment]
        processed = [
            {
                "content_type": "audio/mpeg",
                "content": b"fake-audio",
                "text": None,
            },
        ]
        return [attachment], [response], processed

    async def _fake_preprocess_media_attachments_with_gemini(**kwargs: object):
        nonlocal preprocess_call_count
        preprocess_call_count += 1
        assert kwargs["actual_model"] == "gpt-5.4"
        return ([cached_output], False)

    async def _fail_preprocess_media_attachments_with_gemini(
        **_kwargs: object,
    ) -> tuple[list[str], bool]:
        msg = "cached media preprocessing should be reused after restart"
        raise AssertionError(msg)

    async def _noop_set_parent_message(**kwargs: object) -> None:
        return None

    bot = _DummyBot()
    msg = FakeMessage(
        id=313,
        content="at ai summarize this audio",
        author=FakeUser(1234),
        attachments=[attachment],
    )

    def _make_context(*, current_msg_nodes: dict[int, object]) -> MessageBuildContext:
        return MessageBuildContext(
            new_msg=msg,  # type: ignore[arg-type]
            discord_bot=bot,  # type: ignore[arg-type]
            httpx_client=httpx_client,
            twitter_api=DummyTwitterApi(),
            msg_nodes=current_msg_nodes,  # type: ignore[arg-type]
            actual_model="gpt-5.4",
            accept_usernames=False,
            max_text=100000,
            max_images=0,
            max_messages=1,
            max_tweet_replies=50,
            enable_youtube_transcripts=True,
            youtube_transcript_method="youtube-transcript-api",
        )

    monkeypatch.setattr(
        "llmcord.logic.messages.download_and_process_attachments",
        _fake_download_and_process_attachments,
    )
    monkeypatch.setattr(
        "llmcord.logic.messages.preprocess_media_attachments_with_gemini",
        _fake_preprocess_media_attachments_with_gemini,
    )
    monkeypatch.setattr(
        "llmcord.logic.messages._set_parent_message",
        _noop_set_parent_message,
    )
    monkeypatch.setattr("llmcord.logic.messages.get_db", lambda: db)

    try:
        first_result = await build_messages(
            context=_make_context(current_msg_nodes=msg_nodes),
        )

        first_user_content = str(first_result.messages[0]["content"])
        assert "cached audio transcript" in first_user_content
        assert preprocess_call_count == 1

        cached_results, cached_failed = await db.aget_message_media_preprocessing_data(
            "313",
        )
        assert cached_results == [cached_output]
        assert cached_failed is False

        monkeypatch.setattr(
            "llmcord.logic.messages.preprocess_media_attachments_with_gemini",
            _fail_preprocess_media_attachments_with_gemini,
        )

        second_result = await build_messages(
            context=_make_context(current_msg_nodes={}),
        )

        second_user_content = str(second_result.messages[0]["content"])
        assert "cached audio transcript" in second_user_content
        assert preprocess_call_count == 1
        assert download_attachment_counts == [1, 0]
    finally:
        await db.aclose()


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
    monkeypatch.setattr("llmcord.logic.messages.get_db", lambda: _FakeDB())

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
            youtube_transcript_method="youtube-transcript-api",
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


@pytest.mark.asyncio
async def test_tiktok_query_adds_video_file_for_gemini(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: Any,
    msg_nodes: dict[int, object],
) -> None:
    async def _fake_download_and_process_attachments(**kwargs: object):
        return [], [], []

    async def _fake_maybe_download_tiktok_videos(**kwargs: object):
        return TikTokDownloadResult(
            videos=[
                DownloadedTikTokVideo(
                    content=b"fake-mp4-bytes",
                    content_type="video/mp4",
                ),
            ],
            failed_urls=[],
        )

    async def _noop_set_parent_message(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "llmcord.logic.messages.download_and_process_attachments",
        _fake_download_and_process_attachments,
    )
    monkeypatch.setattr(
        "llmcord.logic.messages.maybe_download_tiktok_videos_with_failures",
        _fake_maybe_download_tiktok_videos,
    )
    monkeypatch.setattr(
        "llmcord.logic.messages._set_parent_message",
        _noop_set_parent_message,
    )
    monkeypatch.setattr("llmcord.logic.messages.get_db", lambda: _FakeDB())

    bot = _DummyBot()
    msg = FakeMessage(
        id=33,
        content=(
            "summarize https://www.tiktok.com/@contraryian/video/7602846033332292894"
        ),
        author=FakeUser(1234),
        attachments=[],
    )

    result = await build_messages(
        context=MessageBuildContext(
            new_msg=msg,  # type: ignore[arg-type]
            discord_bot=bot,  # type: ignore[arg-type]
            httpx_client=httpx_client,
            twitter_api=DummyTwitterApi(),
            msg_nodes=msg_nodes,  # type: ignore[arg-type]
            actual_model="gemini-2.0-flash",
            accept_usernames=False,
            max_text=100000,
            max_images=5,
            max_messages=1,
            max_tweet_replies=50,
            enable_youtube_transcripts=True,
            youtube_transcript_method="youtube-transcript-api",
        ),
    )

    content = result.messages[0]["content"]
    assert isinstance(content, list)
    parts = cast("list[dict[str, object]]", content)

    text_part = next((p for p in parts if p.get("type") == "text"), None)
    assert text_part is not None
    assert str(text_part.get("text")).startswith(
        "<@1234>: summarize https://www.tiktok.com/",
    )

    file_part = next((p for p in parts if p.get("type") == "file"), None)
    assert file_part is not None
    file_obj = file_part.get("file")
    assert isinstance(file_obj, dict)
    file_data = cast("dict[str, object]", file_obj).get("file_data")
    assert isinstance(file_data, str)
    assert file_data.startswith("data:video/mp4;base64,")


@pytest.mark.asyncio
async def test_tiktok_and_facebook_download_used_for_non_gemini(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: Any,
    msg_nodes: dict[int, object],
) -> None:
    async def _fake_download_and_process_attachments(**kwargs: object):
        return [], [], []

    async def _fake_maybe_download_tiktok_videos(**kwargs: object):
        assert kwargs["force_download"] is True
        return TikTokDownloadResult(
            videos=[
                DownloadedTikTokVideo(
                    content=b"tiktok-mp4-bytes",
                    content_type="video/mp4",
                ),
            ],
            failed_urls=[],
        )

    async def _fake_maybe_download_facebook_videos(**kwargs: object):
        assert kwargs["force_download"] is True
        return FacebookDownloadResult(
            videos=[
                DownloadedFacebookVideo(
                    content=b"facebook-mp4-bytes",
                    content_type="video/mp4",
                ),
            ],
            failed_urls=[],
        )

    async def _fake_preprocess_media_attachments_with_gemini(**kwargs: object):
        processed_attachments = kwargs["processed_attachments"]
        assert isinstance(processed_attachments, list)
        attachment_contents = [
            cast("dict[str, object]", attachment)["content"]
            for attachment in processed_attachments
        ]
        assert attachment_contents == [b"tiktok-mp4-bytes", b"facebook-mp4-bytes"]
        return (
            [
                (
                    "--- Gemini preprocessing for Video attachment 1 ---\n"
                    "Video transcription per timestamp:\n\n"
                    "0s to 10s: tiktok clip"
                ),
            ],
            False,
        )

    async def _noop_set_parent_message(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "llmcord.logic.messages.download_and_process_attachments",
        _fake_download_and_process_attachments,
    )
    monkeypatch.setattr(
        "llmcord.logic.messages.maybe_download_tiktok_videos_with_failures",
        _fake_maybe_download_tiktok_videos,
    )
    monkeypatch.setattr(
        "llmcord.logic.messages.maybe_download_facebook_videos_with_failures",
        _fake_maybe_download_facebook_videos,
    )
    monkeypatch.setattr(
        "llmcord.logic.messages.preprocess_media_attachments_with_gemini",
        _fake_preprocess_media_attachments_with_gemini,
    )
    monkeypatch.setattr(
        "llmcord.logic.messages._set_parent_message",
        _noop_set_parent_message,
    )
    monkeypatch.setattr("llmcord.logic.messages.get_db", lambda: _FakeDB())

    bot = _DummyBot()
    msg = FakeMessage(
        id=34,
        content=(
            "summarize "
            "https://www.tiktok.com/@contraryian/video/7602846033332292894 "
            "and https://www.facebook.com/share/r/18ZA4xvsak/"
        ),
        author=FakeUser(1234),
        attachments=[],
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
            youtube_transcript_method="youtube-transcript-api",
        ),
    )

    user_content = str(result.messages[0]["content"])
    assert "Video transcription per timestamp" in user_content
    assert "0s to 10s: tiktok clip" in user_content
