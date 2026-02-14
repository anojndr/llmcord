from __future__ import annotations

from typing import Any

import pytest

from llmcord.logic.content import _merge_reverse_image_results
from llmcord.logic.messages import MessageBuildContext, build_messages
from llmcord.services.extractors import (
    _fetch_twitter_results,
    _process_google_lens_results,
)

from ._fakes import DummyTwitterApi, FakeAttachment, FakeMessage, FakeUser


class _FakeDB:
    def get_message_search_data(self, _message_id: str) -> tuple[None, None, None]:
        return None, None, None

    def save_message_search_data(self, *_args: object, **_kwargs: object) -> None:
        return None


class _DummyBot:
    def __init__(self, user_id: int = 999) -> None:
        self.user = FakeUser(user_id)


def test_googlelens_overlap_weighting_prioritizes_shared_urls() -> None:
    yandex_results = [
        "- [Yandex Unique](https://example.com/yandex-only) (example.com)",
        "- [Shared](https://example.com/shared) (example.com)",
    ]
    google_results = [
        "- [Google Unique](https://example.com/google-only) (example.com)",
        "- [Shared Match](https://example.com/shared) (example.com)",
    ]

    merged = _merge_reverse_image_results(
        yandex_results,
        google_results,
        prefer_overlapping_matches=True,
    )

    assert merged
    assert "shared" in merged[0].lower()


@pytest.mark.asyncio
async def test_googlelens_url_content_extraction_uses_fast_fail_settings(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: Any,
) -> None:
    calls: list[tuple[float, int]] = []

    async def _fake_extract_url_content(
        _url: str,
        _httpx_client: Any,
        *,
        timeout_seconds: float = 20,
        retries: int = 2,
        **_kwargs: object,
    ) -> str | None:
        calls.append((timeout_seconds, retries))
        return None

    monkeypatch.setattr(
        "llmcord.services.extractors.extract_url_content",
        _fake_extract_url_content,
    )

    visual_matches = [
        {
            "title": "Example Result",
            "link": "https://example.com/post/1",
            "source": "example.com",
        },
    ]

    await _process_google_lens_results(visual_matches, httpx_client)

    assert calls == [(5.0, 0)]


@pytest.mark.asyncio
async def test_googlelens_twitter_url_parsing_extracts_status_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called_ids: list[int] = []

    async def _fake_fetch_tweet_with_replies(
        _twitter_api: Any,
        tweet_id: int,
        **_kwargs: object,
    ) -> str | None:
        called_ids.append(tweet_id)
        return "tweet body"

    monkeypatch.setattr(
        "llmcord.services.extractors.fetch_tweet_with_replies",
        _fake_fetch_tweet_with_replies,
    )

    twitter_content = await _fetch_twitter_results(
        ["https://x.com/cb_doge/status/2019995898894520711"],
        DummyTwitterApi(),
        max_tweet_replies=0,
    )

    assert called_ids == [2019995898894520711]
    assert twitter_content == ["tweet body"]


@pytest.mark.asyncio
async def test_googlelens_results_appended(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: Any,
    msg_nodes: dict[int, object],
) -> None:
    async def _fake_perform_yandex_lookup(*args: object, **kwargs: object):
        lens_results = [
            "Result: I Shall Survive Using Potions!",
            "Text match: potions will save me",
        ]
        return lens_results, []

    async def _fake_perform_google_lens_lookup(*args: object, **kwargs: object):
        lens_results = [
            "Result: Survival Through Potions",
            "Text match: potion anime cover",
        ]
        return lens_results, []

    async def _fake_download_and_process_attachments(**kwargs: object):
        return [], [], []

    async def _noop_set_parent_message(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "llmcord.logic.content.perform_yandex_lookup",
        _fake_perform_yandex_lookup,
    )
    monkeypatch.setattr(
        "llmcord.logic.content.perform_google_lens_lookup",
        _fake_perform_google_lens_lookup,
    )
    monkeypatch.setattr("llmcord.logic.content.get_bad_keys_db", lambda: _FakeDB())
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
        id=10,
        content="at ai googlelens anime title?",
        author=FakeUser(1234),
        attachments=[
            FakeAttachment(
                url="https://cdn.discordapp.com/attachments/1/2/chrome_jHqK1D4oUn.png",
                content_type="image/png",
                filename="chrome_jHqK1D4oUn.png",
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
            max_tweet_replies=50,
            enable_youtube_transcripts=True,
        ),
    )

    user_content = str(result.messages[0]["content"])
    assert "reverse image results" in user_content.lower()
    assert "google lens" in user_content.lower()
    assert "yandex" in user_content.lower()
    assert "potions will save me" in user_content.lower()
    assert "potion anime cover" in user_content.lower()


@pytest.mark.asyncio
async def test_googlelens_multiple_images_results_appended(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: Any,
    msg_nodes: dict[int, object],
) -> None:
    async def _fake_perform_yandex_lookup(image_url, *args, **kwargs):
        if "img1" in image_url:
            return ["Result: Image 1 Yandex"], []
        if "img2" in image_url:
            return ["Result: Image 2 Yandex"], []
        return [], []

    async def _fake_perform_google_lens_lookup(image_url, *args, **kwargs):
        if "img1" in image_url:
            return ["Result: Image 1 Google"], []
        if "img2" in image_url:
            return ["Result: Image 2 Google"], []
        return [], []

    async def _fake_download_and_process_attachments(**kwargs: object):
        return [], [], []

    async def _noop_set_parent_message(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "llmcord.logic.content.perform_yandex_lookup",
        _fake_perform_yandex_lookup,
    )
    monkeypatch.setattr(
        "llmcord.logic.content.perform_google_lens_lookup",
        _fake_perform_google_lens_lookup,
    )
    monkeypatch.setattr("llmcord.logic.content.get_bad_keys_db", lambda: _FakeDB())
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
        id=10,
        content="at ai googlelens multiple?",
        author=FakeUser(1234),
        attachments=[
            FakeAttachment(
                url="https://example.com/img1.png",
                content_type="image/png",
                filename="img1.png",
            ),
            FakeAttachment(
                url="https://example.com/img2.png",
                content_type="image/png",
                filename="img2.png",
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
            max_tweet_replies=50,
            enable_youtube_transcripts=True,
        ),
    )

    user_content = str(result.messages[0]["content"])
    assert "Image 1 (img1.png)" in user_content
    assert "Results for Image 1 (img1.png):" in user_content
    assert "Result: Image 1 Yandex" in user_content
    assert "Result: Image 1 Google" in user_content
    assert "Image 2 (img2.png)" in user_content
    assert "Results for Image 2 (img2.png):" in user_content
    assert "Result: Image 2 Yandex" in user_content
    assert "Result: Image 2 Google" in user_content
    expected_instr = (
        "provide the top 3 most frequently appearing results for each image"
    )
    assert expected_instr in user_content
    assert "confidence level" in user_content
    assert "Towa no Yuugure" in user_content
