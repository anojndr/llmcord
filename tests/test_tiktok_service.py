from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from llmcord.services.tiktok import DownloadedTikTokVideo, maybe_download_tiktok_video


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
