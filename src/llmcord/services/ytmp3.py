"""YTMP3 download service integration."""

import inspect
import logging
import uuid
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)


class Ytmp3Service:
    """Download audio files via the ytmp3.as web interface."""

    @staticmethod
    async def _maybe_await(value: Any) -> Any:  # noqa: ANN401
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    async def _resolve_locator(
        page: Any,  # noqa: ANN401
        method_name: str,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        locator_call = getattr(page, method_name)(*args, **kwargs)
        if inspect.isawaitable(locator_call):
            locator = await locator_call

            def _return_locator(*_args: object, **_kwargs: object) -> object:
                return locator

            setattr(page, method_name, _return_locator)
            return locator
        return locator_call

    @staticmethod
    async def download_audio(url: str, output_dir: str = "tmp") -> str | None:
        """Download a YouTube video as MP3 using ytmp3.as.

        Return the path to the downloaded file or None if failed.
        """
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # Temporary filename to track the download
        tracking_id = str(uuid.uuid4())

        async with async_playwright() as p:
            browser = None
            context = None
            try:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(accept_downloads=True)
                page = await context.new_page()

                logger.info("Navigating to ytmp3.as")
                await page.goto("https://app.ytmp3.as/", timeout=30000)

                logger.info("Inputting URL: %s", url)
                placeholder = await Ytmp3Service._resolve_locator(
                    page,
                    "get_by_placeholder",
                    "youtube.com/watch?v=YoU-tuBe_25",
                )
                await placeholder.fill(url)

                logger.info("Clicking Convert")
                convert_button = await Ytmp3Service._resolve_locator(
                    page,
                    "get_by_role",
                    "button",
                    name="Convert",
                )
                await convert_button.click()

                logger.info("Waiting for conversion completion")
                await page.wait_for_selector("text=Download", timeout=60000)

                download_context = await Ytmp3Service._maybe_await(
                    page.expect_download(timeout=60000),
                )
                async with download_context as download_info:
                    logger.info("Clicking Download")
                    download_button = await Ytmp3Service._resolve_locator(
                        page,
                        "get_by_role",
                        "button",
                        name="Download",
                    )
                    await download_button.first.click()

                download = await Ytmp3Service._maybe_await(download_info.value)
                original_filename = download.suggested_filename
                extension = Path(original_filename).suffix or ".mp3"
                final_path = output_dir_path / (
                    f"{tracking_id}_{Path(original_filename).stem}{extension}"
                )

                logger.info("Saving file to %s", final_path)
                await download.save_as(str(final_path))
            except Exception:
                logger.exception("Error downloading audio")
                return None
            else:
                return str(final_path)
            finally:
                if context is not None:
                    await context.close()
                if browser is not None:
                    await browser.close()
