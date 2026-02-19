from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llmcord.logic.messages import MessageBuildContext, build_messages
from llmcord.services.facebook import DownloadedFacebookVideo, FacebookDownloadResult
from llmcord.services.tiktok import DownloadedTikTokVideo, TikTokDownloadResult

from ._fakes import DummyTwitterApi, FakeMessage, FakeUser


class _FakeDB:
    def get_message_search_data(self, _message_id: str) -> tuple[None, None, None]:
        return None, None, None


class _DummyBot:
    def __init__(self, user_id: int = 999) -> None:
        self.user = FakeUser(user_id)


@pytest.mark.asyncio
async def test_x_url_extraction_appended(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: Any,
    msg_nodes: dict[int, object],
) -> None:
    async def _fake_fetch_tweet_with_replies(*args: object, **kwargs: object) -> str:
        return (
            "Grok 4.20 just dominated the Alpha Arena leaderboard. "
            "Grok delivered +34.59 percent return and the top spot overall.\n"
            "Joel Eriksson: It's competing against GPT 5.1 and Claude 4.5 Sonnet...\n"
            "Squirrel: what is this arena? And what you dominated?"
        )

    async def _fake_download_and_process_attachments(**kwargs: object):
        return [], [], []

    async def _noop_set_parent_message(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "llmcord.logic.content.fetch_tweet_with_replies",
        _fake_fetch_tweet_with_replies,
    )
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
        content="at ai summarize https://x.com/cb_doge/status/2019995898894520711",
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
            max_tweet_replies=50,
            enable_youtube_transcripts=True,
            youtube_transcript_method="youtube-transcript-api",
        ),
    )

    assert result.messages, "Expected at least one built message"
    user_content = str(result.messages[0]["content"])
    assert user_content.startswith(
        "summarize https://x.com/cb_doge/status/2019995898894520711",
    )
    assert "Grok 4.20 just dominated the Alpha Arena leaderboard" in user_content
    assert "It's competing against GPT 5.1 and Claude 4.5 Sonnet" in user_content
    assert "what is this arena? And what you dominated?" in user_content


@pytest.mark.asyncio
async def test_reddit_url_extraction_appended(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: Any,
    msg_nodes: dict[int, object],
) -> None:
    async def _fake_extract_reddit_post(*args: object, **kwargs: object) -> str:
        return (
            "We use and love both Claude Code and Codex CLI agents... "
            "So we built our own SWE-Bench!\n"
            "ClaudeAI-mod-bot: The verdict is a hard split...\n"
            "rydan: Glad to see I'm not the only Gemini Pro hater\n"
            "Other user: Codex and Opus performance discussion"
        )

    async def _fake_download_and_process_attachments(**kwargs: object):
        return [], [], []

    async def _noop_set_parent_message(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "llmcord.logic.content.extract_reddit_post",
        _fake_extract_reddit_post,
    )
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
        id=2,
        content="at ai summarize https://www.reddit.com/r/ClaudeAI/s/WdtZ84wM3m",
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
            max_tweet_replies=50,
            enable_youtube_transcripts=True,
            youtube_transcript_method="youtube-transcript-api",
        ),
    )

    user_content = str(result.messages[0]["content"])
    assert user_content.startswith(
        "summarize https://www.reddit.com/r/ClaudeAI/s/WdtZ84wM3m",
    )
    assert "We use and love both Claude Code and Codex CLI agents" in user_content
    assert "The verdict is a hard split" in user_content
    assert "Glad to see I'm not the only Gemini Pro hater" in user_content


@pytest.mark.asyncio
async def test_youtube_url_extraction_appended(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: Any,
    msg_nodes: dict[int, object],
) -> None:
    async def _fake_extract_youtube_transcript(*args: object, **kwargs: object):
        return "Winner confirmed: Age 28", None

    async def _fake_download_and_process_attachments(**kwargs: object):
        return [], [], []

    async def _noop_set_parent_message(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "llmcord.logic.content.extract_youtube_transcript_with_reason",
        _fake_extract_youtube_transcript,
    )
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
        id=3,
        content="at ai what age won https://www.youtube.com/watch?v=9WEQts7b8Pw&pp=ugUEEgJlbg%3D%3D",
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
            max_tweet_replies=50,
            enable_youtube_transcripts=True,
            youtube_transcript_method="youtube-transcript-api",
        ),
    )

    user_content = str(result.messages[0]["content"])
    assert user_content.startswith(
        "what age won https://www.youtube.com/watch?v=9WEQts7b8Pw",
    )
    assert "Age 28" in user_content


@pytest.mark.asyncio
async def test_multiple_youtube_urls_extracted_concurrently(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: Any,
    msg_nodes: dict[int, object],
) -> None:
    expected_ids = {"AAAAAAAAAAA", "BBBBBBBBBBB", "CCCCCCCCCCC"}
    minimum_concurrency = 2
    completed_ids: list[str] = []
    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def _fake_extract_youtube_transcript(
        video_id: str,
        *args: object,
        **kwargs: object,
    ):
        nonlocal in_flight, max_in_flight
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)

        await asyncio.sleep(0.02)

        async with lock:
            in_flight -= 1
        completed_ids.append(video_id)
        return f"Transcript for {video_id}", None

    async def _fake_download_and_process_attachments(**kwargs: object):
        return [], [], []

    async def _noop_set_parent_message(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "llmcord.logic.content.extract_youtube_transcript_with_reason",
        _fake_extract_youtube_transcript,
    )
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
        id=4,
        content=(
            "at ai summarize "
            "https://www.youtube.com/watch?si=one&v=AAAAAAAAAAA&pp=x "
            "https://youtu.be/BBBBBBBBBBB?t=5 "
            "https://www.youtube.com/shorts/CCCCCCCCCCC?feature=share"
        ),
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
            max_tweet_replies=50,
            enable_youtube_transcripts=True,
            youtube_transcript_method="youtube-transcript-api",
        ),
    )

    user_content = str(result.messages[0]["content"])
    assert "Transcript for AAAAAAAAAAA" in user_content
    assert "Transcript for BBBBBBBBBBB" in user_content
    assert "Transcript for CCCCCCCCCCC" in user_content
    assert set(completed_ids) == expected_ids
    assert len(completed_ids) == len(expected_ids)
    assert max_in_flight >= minimum_concurrency


@pytest.mark.asyncio
async def test_multiple_twitter_urls_extracted_concurrently(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: Any,
    msg_nodes: dict[int, object],
) -> None:
    expected_ids = {1111111111111111111, 2222222222222222222, 3333333333333333333}
    minimum_concurrency = 2
    completed_ids: list[int] = []
    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def _fake_fetch_tweet_with_replies(
        _twitter_api: object,
        tweet_id: int,
        *args: object,
        **kwargs: object,
    ) -> str:
        nonlocal in_flight, max_in_flight
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)

        await asyncio.sleep(0.02)

        async with lock:
            in_flight -= 1
        completed_ids.append(tweet_id)
        return f"Tweet payload for {tweet_id}"

    async def _fake_download_and_process_attachments(**kwargs: object):
        return [], [], []

    async def _noop_set_parent_message(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "llmcord.logic.content.fetch_tweet_with_replies",
        _fake_fetch_tweet_with_replies,
    )
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
        id=7,
        content=(
            "at ai summarize "
            "https://x.com/a/status/1111111111111111111 "
            "https://twitter.com/b/status/2222222222222222222 "
            "https://x.com/c/status/3333333333333333333"
        ),
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
            max_tweet_replies=50,
            enable_youtube_transcripts=True,
            youtube_transcript_method="youtube-transcript-api",
        ),
    )

    user_content = str(result.messages[0]["content"])
    assert "Tweet payload for 1111111111111111111" in user_content
    assert "Tweet payload for 2222222222222222222" in user_content
    assert "Tweet payload for 3333333333333333333" in user_content
    assert set(completed_ids) == expected_ids
    assert len(completed_ids) == len(expected_ids)
    assert max_in_flight >= minimum_concurrency


@pytest.mark.asyncio
async def test_external_source_collectors_run_concurrently(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: Any,
    msg_nodes: dict[int, object],
) -> None:
    minimum_concurrency = 2
    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def _collector(name: str) -> str:
        nonlocal in_flight, max_in_flight
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)

        await asyncio.sleep(0.02)

        async with lock:
            in_flight -= 1
        return name

    async def _fake_collect_youtube_transcripts(*args: object, **kwargs: object):
        await _collector("youtube")
        return ["YT transcript"], []

    async def _fake_collect_tweets(*args: object, **kwargs: object):
        await _collector("twitter")
        return ["Tweet batch"]

    async def _fake_collect_reddit_posts(*args: object, **kwargs: object):
        await _collector("reddit")
        return ["Reddit batch"]

    async def _fake_collect_generic_url_contents(*args: object, **kwargs: object):
        await _collector("generic")
        return ["--- URL Content: https://example.com ---\nExample body"], []

    async def _fake_extract_pdf_texts(*args: object, **kwargs: object):
        await _collector("pdf")
        return ["--- PDF Attachment 1 Content ---\nPDF body"]

    async def _fake_download_and_process_attachments(**kwargs: object):
        return [], [], []

    async def _noop_set_parent_message(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "llmcord.logic.content._collect_youtube_transcripts",
        _fake_collect_youtube_transcripts,
    )
    monkeypatch.setattr(
        "llmcord.logic.content._collect_tweets",
        _fake_collect_tweets,
    )
    monkeypatch.setattr(
        "llmcord.logic.content._collect_reddit_posts",
        _fake_collect_reddit_posts,
    )
    monkeypatch.setattr(
        "llmcord.logic.content._collect_generic_url_contents",
        _fake_collect_generic_url_contents,
    )
    monkeypatch.setattr(
        "llmcord.logic.content._extract_pdf_texts",
        _fake_extract_pdf_texts,
    )
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
        id=8,
        content="at ai summarize https://example.com",
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
            max_tweet_replies=50,
            enable_youtube_transcripts=True,
            youtube_transcript_method="youtube-transcript-api",
        ),
    )

    user_content = str(result.messages[0]["content"])
    assert "YT transcript" in user_content
    assert "Tweet batch" in user_content
    assert "Reddit batch" in user_content
    assert "Example body" in user_content
    assert "PDF body" in user_content
    assert max_in_flight >= minimum_concurrency


@pytest.mark.asyncio
async def test_youtube_failure_reason_is_in_warning(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: Any,
    msg_nodes: dict[int, object],
) -> None:
    async def _fake_extract_youtube_transcript(*args: object, **kwargs: object):
        return None, "Subtitles are disabled for this video"

    async def _fake_download_and_process_attachments(**kwargs: object):
        return [], [], []

    async def _noop_set_parent_message(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "llmcord.logic.content.extract_youtube_transcript_with_reason",
        _fake_extract_youtube_transcript,
    )
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
        id=6,
        content="at ai summarize https://www.youtube.com/watch?v=aD-uI63jR8c",
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
            max_tweet_replies=50,
            enable_youtube_transcripts=True,
            youtube_transcript_method="youtube-transcript-api",
        ),
    )

    failed_warning = next(
        warning
        for warning in result.user_warnings
        if warning.startswith("⚠️ failed to extract from some urls.")
    )
    assert (
        failed_warning
        == '⚠️ failed to extract from some urls. click "failed urls" to see which urls.'
    )
    assert result.failed_extractions
    assert "Subtitles are disabled for this video" in result.failed_extractions[0]
    assert "https://www.youtube.com/watch?v=aD-uI63jR8c" in result.failed_extractions[0]


@pytest.mark.asyncio
async def test_general_url_extraction_appended(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: Any,
    msg_nodes: dict[int, object],
) -> None:
    calls: list[str] = []

    async def _fake_extract_url_content(
        url: str,
        *args: object,
        **kwargs: object,
    ) -> str:
        calls.append(url)
        return "Elephants are large mammals of the family Elephantidae."

    async def _fake_download_and_process_attachments(**kwargs: object):
        return [], [], []

    async def _noop_set_parent_message(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "llmcord.logic.content.extract_url_content",
        _fake_extract_url_content,
    )
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
        id=5,
        content="at ai summarize https://en.wikipedia.org/wiki/Elephant",
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
            max_tweet_replies=50,
            enable_youtube_transcripts=True,
            youtube_transcript_method="youtube-transcript-api",
        ),
    )

    user_content = str(result.messages[0]["content"])
    assert user_content.startswith("summarize https://en.wikipedia.org/wiki/Elephant")
    assert (
        "--- URL Content: https://en.wikipedia.org/wiki/Elephant ---\n"
        "Elephants are large mammals of the family Elephantidae."
    ) in user_content
    assert "Elephants are large mammals" in user_content
    assert calls == ["https://en.wikipedia.org/wiki/Elephant"]


@pytest.mark.asyncio
async def test_tiktok_and_facebook_not_failed_when_mp4_download_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: Any,
    msg_nodes: dict[int, object],
) -> None:
    tiktok_url = "https://www.tiktok.com/@creator/video/7602846033332292894"
    facebook_url = "https://www.facebook.com/share/r/18ZA4xvsak/"

    async def _fake_download_and_process_attachments(**kwargs: object):
        return [], [], []

    async def _fake_maybe_download_tiktok_videos_with_failures(**kwargs: object):
        return TikTokDownloadResult(
            videos=[
                DownloadedTikTokVideo(
                    content=b"tiktok-mp4-bytes",
                    content_type="video/mp4",
                ),
            ],
            failed_urls=[],
        )

    async def _fake_maybe_download_facebook_videos_with_failures(**kwargs: object):
        return FacebookDownloadResult(
            videos=[
                DownloadedFacebookVideo(
                    content=b"facebook-mp4-bytes",
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
        _fake_maybe_download_tiktok_videos_with_failures,
    )
    monkeypatch.setattr(
        "llmcord.logic.messages.maybe_download_facebook_videos_with_failures",
        _fake_maybe_download_facebook_videos_with_failures,
    )
    monkeypatch.setattr(
        "llmcord.logic.messages._set_parent_message",
        _noop_set_parent_message,
    )
    monkeypatch.setattr("llmcord.logic.messages.get_db", lambda: _FakeDB())

    bot = _DummyBot()
    msg = FakeMessage(
        id=9,
        content=f"at ai summarize {tiktok_url} and {facebook_url}",
        author=FakeUser(1234),
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
            max_images=0,
            max_messages=1,
            max_tweet_replies=50,
            enable_youtube_transcripts=True,
            youtube_transcript_method="youtube-transcript-api",
        ),
    )

    assert result.failed_extractions == []
    assert not any(
        warning.startswith("⚠️ failed to extract from some urls.")
        for warning in result.user_warnings
    )


@pytest.mark.asyncio
async def test_tiktok_and_facebook_failed_only_when_mp4_download_fails(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: Any,
    msg_nodes: dict[int, object],
) -> None:
    tiktok_url = "https://www.tiktok.com/@creator/video/7602846033332292894"
    facebook_url = "https://www.facebook.com/share/r/18ZA4xvsak/"

    async def _fake_download_and_process_attachments(**kwargs: object):
        return [], [], []

    async def _fake_maybe_download_tiktok_videos_with_failures(**kwargs: object):
        return TikTokDownloadResult(videos=[], failed_urls=[tiktok_url])

    async def _fake_maybe_download_facebook_videos_with_failures(**kwargs: object):
        return FacebookDownloadResult(videos=[], failed_urls=[facebook_url])

    async def _noop_set_parent_message(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "llmcord.logic.messages.download_and_process_attachments",
        _fake_download_and_process_attachments,
    )
    monkeypatch.setattr(
        "llmcord.logic.messages.maybe_download_tiktok_videos_with_failures",
        _fake_maybe_download_tiktok_videos_with_failures,
    )
    monkeypatch.setattr(
        "llmcord.logic.messages.maybe_download_facebook_videos_with_failures",
        _fake_maybe_download_facebook_videos_with_failures,
    )
    monkeypatch.setattr(
        "llmcord.logic.messages._set_parent_message",
        _noop_set_parent_message,
    )
    monkeypatch.setattr("llmcord.logic.messages.get_db", lambda: _FakeDB())

    bot = _DummyBot()
    msg = FakeMessage(
        id=10,
        content=f"at ai summarize {tiktok_url} and {facebook_url}",
        author=FakeUser(1234),
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
            max_images=0,
            max_messages=1,
            max_tweet_replies=50,
            enable_youtube_transcripts=True,
            youtube_transcript_method="youtube-transcript-api",
        ),
    )

    assert set(result.failed_extractions) == {tiktok_url, facebook_url}
    assert any(
        warning.startswith("⚠️ failed to extract from some urls.")
        for warning in result.user_warnings
    )
