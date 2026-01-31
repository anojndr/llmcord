"""Tests for the YTMP3 download service."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from llmcord.services.ytmp3 import Ytmp3Service


def assert_true(*, condition: bool, message: str) -> None:
    """Raise an AssertionError when a condition is false."""
    if not condition:
        raise AssertionError(message)


@pytest.mark.asyncio
async def test_download_audio_success() -> None:
    """Download should succeed and return a path-like string."""
    with patch("llmcord.services.ytmp3.async_playwright") as mock_playwright:
        # Mock the context manager
        mock_p_context = AsyncMock()
        mock_playwright.return_value.__aenter__.return_value = mock_p_context

        # Mock browser, context, page
        mock_browser = AsyncMock()
        mock_p_context.chromium.launch.return_value = mock_browser

        mock_context = AsyncMock()
        mock_browser.new_context.return_value = mock_context

        mock_page = AsyncMock()
        mock_context.new_page.return_value = mock_page

        # Mock download
        mock_download_info = AsyncMock()
        # The expect_download context manager
        mock_page.expect_download.return_value.__aenter__.return_value = (
            mock_download_info
        )

        mock_download = AsyncMock()
        mock_download.suggested_filename = "video.mp3"
        mock_download_info.value = mock_download

        # Run service
        result = await Ytmp3Service.download_audio(
            "https://youtube.com/watch?v=123",
        )

        assert_true(
            condition=result is not None,
            message="Expected a download result",
        )
        assert_true(
            condition="video.mp3" in result,
            message="Expected mp3 filename in result",
        )

        # Verify calls
        mock_page.goto.assert_called_with("https://app.ytmp3.as/", timeout=30000)
        mock_page.get_by_placeholder("youtube.com/watch?v=YoU-tuBe_25").fill.assert_called()
        mock_page.get_by_role("button", name="Convert").click.assert_called()
        mock_page.wait_for_selector.assert_called_with("text=Download", timeout=60000)

@pytest.mark.asyncio
async def test_download_audio_failure() -> None:
    """Download should return None on browser launch failure."""
    with patch("llmcord.services.ytmp3.async_playwright") as mock_playwright:
        mock_p_context = AsyncMock()
        mock_playwright.return_value.__aenter__.return_value = mock_p_context

        mock_p_context.chromium.launch.side_effect = Exception("Browser failed")

        result = await Ytmp3Service.download_audio(
            "https://youtube.com/watch?v=123",
        )

        assert_true(
            condition=result is None,
            message="Expected no result on failure",
        )
