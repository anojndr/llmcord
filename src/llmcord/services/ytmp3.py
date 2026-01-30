import asyncio
import logging
import os
import uuid
from typing import Optional

from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

class Ytmp3Service:
    @staticmethod
    async def download_audio(url: str, output_dir: str = "tmp") -> Optional[str]:
        """
        Downloads a YouTube video as MP3 using ytmp3.as.
        Returns the path to the downloaded file or None if failed.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Temporary filename to track the download
        tracking_id = str(uuid.uuid4())
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(accept_downloads=True)
            page = await context.new_page()

            try:
                logger.info("Navigating to ytmp3.as")
                await page.goto("https://app.ytmp3.as/", timeout=30000)

                logger.info(f"Inputting URL: {url}")
                await page.get_by_placeholder("youtube.com/watch?v=YoU-tuBe_25").fill(url)
                
                logger.info("Clicking Convert")
                await page.get_by_role("button", name="Convert").click()

                # Wait for the download button to appear (it shows "Download" text)
                # It might take some time for conversion
                logger.info("Waiting for conversion completion")
                
                # Wait for the specific Download button that appears after conversion
                # Based on previous snapshot analysis, it appears in a generic container
                # We can wait for the text "Download" to resolve
                await page.wait_for_selector("text=Download", timeout=60000)

                # Setup download listener before clicking
                async with page.expect_download(timeout=60000) as download_info:
                    logger.info("Clicking Download")
                    # There might be multiple "Download" texts (e.g. ad banners), we want the button
                    # In the snapshot it was: button "Download" [ref=e36]
                    await page.get_by_role("button", name="Download").first.click()

                download = await download_info.value
                original_filename = download.suggested_filename
                extension = os.path.splitext(original_filename)[1]
                if not extension:
                    extension = ".mp3"
                
                final_path = os.path.join(output_dir, f"{tracking_id}{extension}")
                
                logger.info(f"Saving file to {final_path}")
                await download.save_as(final_path)
                
                return final_path

            except Exception as e:
                logger.error(f"Error downloading audio: {e}")
                return None
            finally:
                await context.close()
                await browser.close()
