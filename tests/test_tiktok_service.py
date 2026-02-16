from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from llmcord.services.tiktok import (
    DownloadedTikTokVideo,
    maybe_download_tiktok_video,
    maybe_download_tiktok_videos,
)


@pytest.mark.asyncio
async def test_maybe_download_tiktok_video_accepts_vt_short_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch_mock = AsyncMock(return_value="https://d.rapidcdn.app/v2?token=test&dl=1")
    download_mock = AsyncMock(
        return_value=DownloadedTikTokVideo(
            content=b"video-bytes",
            content_type="video/mp4",
        ),
    )

    monkeypatch.setattr(
        "llmcord.services.tiktok._fetch_snaptik_download_url",
        fetch_mock,
    )
    monkeypatch.setattr(
        "llmcord.services.tiktok._download_video_payload",
        download_mock,
    )

    result = await maybe_download_tiktok_video(
        cleaned_content="summarize https://vt.tiktok.com/ZSmY56WhY/",
        actual_model="gemini-2.0-flash",
        httpx_client=AsyncMock(),
    )

    assert result is not None
    assert result.content_type == "video/mp4"

    fetch_mock.assert_awaited_once()
    await_args = fetch_mock.await_args
    assert await_args is not None
    assert await_args.kwargs["tiktok_url"] == "https://vt.tiktok.com/ZSmY56WhY/"


@pytest.mark.asyncio
async def test_maybe_download_tiktok_video_rejects_non_tiktok_domains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch_mock = AsyncMock()
    download_mock = AsyncMock()

    monkeypatch.setattr(
        "llmcord.services.tiktok._fetch_snaptik_download_url",
        fetch_mock,
    )
    monkeypatch.setattr(
        "llmcord.services.tiktok._download_video_payload",
        download_mock,
    )

    result = await maybe_download_tiktok_video(
        cleaned_content="summarize https://example.com/video/123",
        actual_model="gemini-2.0-flash",
        httpx_client=AsyncMock(),
    )

    assert result is None
    fetch_mock.assert_not_called()
    download_mock.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_download_tiktok_videos_handles_multiple_urls_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_url = "https://vt.tiktok.com/ZSmY56WhY/"
    second_url = "https://www.tiktok.com/@creator/video/7602846033332292894"
    download_urls = {
        first_url: "https://d.rapidcdn.app/v2?token=first&dl=1",
        second_url: "https://d.rapidcdn.app/v2?token=second&dl=1",
    }
    seen_urls: set[str] = set()
    ready_gate = asyncio.Event()

    async def _fake_fetch_snaptik_download_url(
        *,
        tiktok_url: str,
        httpx_client: object,
    ) -> str:
        del httpx_client
        seen_urls.add(tiktok_url)
        if len(seen_urls) == len(download_urls):
            ready_gate.set()
        await ready_gate.wait()
        return download_urls[tiktok_url]

    async def _fake_download_video_payload(
        *,
        download_url: str,
        httpx_client: object,
    ) -> DownloadedTikTokVideo:
        del httpx_client
        return DownloadedTikTokVideo(
            content=download_url.encode("utf-8"),
            content_type="video/mp4",
        )

    monkeypatch.setattr(
        "llmcord.services.tiktok._fetch_snaptik_download_url",
        _fake_fetch_snaptik_download_url,
    )
    monkeypatch.setattr(
        "llmcord.services.tiktok._download_video_payload",
        _fake_download_video_payload,
    )

    results = await asyncio.wait_for(
        maybe_download_tiktok_videos(
            cleaned_content=f"summarize {first_url} and {second_url}",
            actual_model="gemini-2.0-flash",
            httpx_client=AsyncMock(),
        ),
        timeout=1,
    )

    assert seen_urls == {first_url, second_url}
    assert len(results) == len(download_urls)
    assert {result.content for result in results} == {
        download_urls[first_url].encode("utf-8"),
        download_urls[second_url].encode("utf-8"),
    }
