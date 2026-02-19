from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from curl_cffi.requests import exceptions as curl_requests_exceptions

from llmcord.services.facebook import (
    DownloadedFacebookVideo,
    maybe_download_facebook_video,
    maybe_download_facebook_videos,
    maybe_download_facebook_videos_with_failures,
)


@pytest.mark.asyncio
async def test_maybe_download_facebook_video_accepts_share_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params_mock = AsyncMock(
        return_value=object(),
    )
    result_html_mock = AsyncMock(return_value="<html/>")

    def _fake_extract_first_snapcdn_token(_: str) -> str:
        return "irrelevant"

    def _fake_extract_download_url_from_snapcdn_token(_: str) -> str:
        return "https://video.example.com/video.mp4"

    download_mock = AsyncMock(
        return_value=DownloadedFacebookVideo(
            content=b"video-bytes",
            content_type="video/mp4",
        ),
    )

    monkeypatch.setattr(
        "llmcord.services.facebook._fetch_fdownloader_params",
        params_mock,
    )
    monkeypatch.setattr(
        "llmcord.services.facebook._fetch_fdownloader_result_html",
        result_html_mock,
    )
    monkeypatch.setattr(
        "llmcord.services.facebook._extract_first_snapcdn_token",
        _fake_extract_first_snapcdn_token,
    )
    monkeypatch.setattr(
        "llmcord.services.facebook._extract_download_url_from_snapcdn_token",
        _fake_extract_download_url_from_snapcdn_token,
    )
    monkeypatch.setattr(
        "llmcord.services.facebook._download_video_payload",
        download_mock,
    )

    result = await maybe_download_facebook_video(
        cleaned_content="summarize https://www.facebook.com/share/r/18ZA4xvsak/",
        actual_model="gemini-2.0-flash",
        httpx_client=AsyncMock(),
    )

    assert result is not None
    assert result.content_type == "video/mp4"

    params_mock.assert_awaited_once()
    result_html_mock.assert_awaited_once()
    download_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_maybe_download_facebook_video_rejects_non_facebook_domains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params_mock = AsyncMock()
    download_mock = AsyncMock()

    monkeypatch.setattr(
        "llmcord.services.facebook._fetch_fdownloader_params",
        params_mock,
    )
    monkeypatch.setattr(
        "llmcord.services.facebook._download_video_payload",
        download_mock,
    )

    result = await maybe_download_facebook_video(
        cleaned_content="summarize https://example.com/video/123",
        actual_model="gemini-2.0-flash",
        httpx_client=AsyncMock(),
    )

    assert result is None
    params_mock.assert_not_called()
    download_mock.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_download_facebook_videos_handles_multiple_urls_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_url = "https://www.facebook.com/share/r/18ZA4xvsak/"
    second_url = "https://fb.watch/abc123/"
    download_urls = {
        first_url: "https://cdn.example.com/first.mp4",
        second_url: "https://cdn.example.com/second.mp4",
    }
    seen_urls: set[str] = set()
    ready_gate = asyncio.Event()

    monkeypatch.setattr(
        "llmcord.services.facebook._fetch_fdownloader_params",
        AsyncMock(return_value=object()),
    )

    async def _fake_fetch_fdownloader_result_html(
        *,
        facebook_url: str,
        params: object,
        session: object,
    ) -> str:
        del params, session
        seen_urls.add(facebook_url)
        if len(seen_urls) == len(download_urls):
            ready_gate.set()
        await ready_gate.wait()
        return facebook_url

    def _fake_extract_first_snapcdn_token(result_html: str) -> str:
        return result_html

    def _fake_extract_download_url_from_snapcdn_token(token: str) -> str:
        return download_urls[token]

    async def _fake_download_video_payload(
        *,
        download_url: str,
        httpx_client: object,
    ) -> DownloadedFacebookVideo:
        del httpx_client
        return DownloadedFacebookVideo(
            content=download_url.encode("utf-8"),
            content_type="video/mp4",
        )

    monkeypatch.setattr(
        "llmcord.services.facebook._fetch_fdownloader_result_html",
        _fake_fetch_fdownloader_result_html,
    )
    monkeypatch.setattr(
        "llmcord.services.facebook._extract_first_snapcdn_token",
        _fake_extract_first_snapcdn_token,
    )
    monkeypatch.setattr(
        "llmcord.services.facebook._extract_download_url_from_snapcdn_token",
        _fake_extract_download_url_from_snapcdn_token,
    )
    monkeypatch.setattr(
        "llmcord.services.facebook._download_video_payload",
        _fake_download_video_payload,
    )

    results = await asyncio.wait_for(
        maybe_download_facebook_videos(
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


@pytest.mark.asyncio
async def test_maybe_download_facebook_videos_handles_fdownloader_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facebook_url = "https://www.facebook.com/share/r/18ZA4xvsak/"

    monkeypatch.setattr(
        "llmcord.services.facebook._fetch_fdownloader_params",
        AsyncMock(return_value=object()),
    )
    monkeypatch.setattr(
        "llmcord.services.facebook._fetch_fdownloader_result_html",
        AsyncMock(
            side_effect=curl_requests_exceptions.Timeout(
                "timed out",
                code=0,
            ),
        ),
    )

    result = await maybe_download_facebook_videos_with_failures(
        cleaned_content=f"summarize {facebook_url}",
        actual_model="gemini-2.0-flash",
        httpx_client=AsyncMock(),
    )

    assert result.videos == []
    assert result.failed_urls == [facebook_url]
