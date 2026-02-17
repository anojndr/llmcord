"""TikTok video download helpers via Snaptik HTTP endpoints."""

from __future__ import annotations

import asyncio
import html
import logging
import re
from dataclasses import dataclass

import httpx

from llmcord.core.config import DEFAULT_USER_AGENT, is_gemini_model
from llmcord.core.error_handling import log_exception

logger = logging.getLogger(__name__)

_TIKTOK_URL_RE = re.compile(
    (
        r"https?://(?:www\.)?(?:"
        r"tiktok\.com|vm\.tiktok\.com|m\.tiktok\.com|vt\.tiktok\.com"
        r")/[^\s<>()\[\]{}]+"
    ),
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(r'name="token"\s+value="([^"]+)"', re.IGNORECASE)

_PACKED_SNAPTIK_RE = re.compile(
    (
        r"eval\(function\(h,u,n,t,e,r\)\{.*?\}\("
        r'"(?P<data>.+?)",\s*(?P<u>\d+),\s*"(?P<n>[^\"]+)",\s*'
        r"(?P<t>\d+),\s*(?P<e>\d+),\s*(?P<r>\d+)\)\)"
    ),
    re.DOTALL,
)

_DOWNLOAD_URL_RE = re.compile(
    r'href=\\?"(?P<url>https://d\.rapidcdn\.app/v2\?token=[^"\\]+?&dl=1)\\?"',
    re.IGNORECASE,
)


@dataclass(slots=True)
class DownloadedTikTokVideo:
    """Downloaded TikTok video payload for Gemini file input."""

    content: bytes
    content_type: str


def _extract_tiktok_urls(text: str) -> list[str]:
    unique_urls: list[str] = []
    seen_urls: set[str] = set()
    for match in _TIKTOK_URL_RE.finditer(text):
        candidate_url = match.group(0).rstrip('.,;:!?)"')
        if candidate_url in seen_urls:
            continue
        seen_urls.add(candidate_url)
        unique_urls.append(candidate_url)

    return unique_urls


def _decode_snaptik_payload(payload: str) -> str | None:
    packed = _PACKED_SNAPTIK_RE.search(payload)
    if not packed:
        return None

    encoded = packed.group("data")
    key = packed.group("n")
    shift = int(packed.group("t"))
    base = int(packed.group("e"))

    if base <= 1 or base >= len(key):
        return None

    delimiter = key[base]
    decoded_chars: list[str] = []
    for segment in encoded.split(delimiter):
        if not segment:
            continue

        translated = segment
        for idx, ch in enumerate(key):
            translated = translated.replace(ch, str(idx))

        try:
            char_code = int(translated, base) - shift
        except ValueError:
            continue

        if char_code < 0:
            continue
        decoded_chars.append(chr(char_code))

    if not decoded_chars:
        return None

    decoded = "".join(decoded_chars)
    try:
        return decoded.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return decoded


def _extract_download_url(decoded_payload: str) -> str | None:
    match = _DOWNLOAD_URL_RE.search(decoded_payload)
    if not match:
        return None
    return html.unescape(match.group("url"))


async def _fetch_snaptik_download_url(
    *,
    tiktok_url: str,
    httpx_client: httpx.AsyncClient,
) -> str | None:
    page_response = await httpx_client.get("https://snaptik.app/en2")
    page_response.raise_for_status()

    token_match = _TOKEN_RE.search(page_response.text)
    if not token_match:
        logger.warning("Snaptik token not found in page response")
        return None

    convert_request_data = {
        "url": tiktok_url,
        "lang": "en2",
        "token": token_match.group(1),
    }
    convert_headers = {
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
    }
    convert_response = await httpx_client.post(
        "https://snaptik.app/abc2.php",
        data=convert_request_data,
        headers=convert_headers,
    )
    convert_response.raise_for_status()

    decoded_payload = _decode_snaptik_payload(convert_response.text)
    if not decoded_payload:
        logger.warning("Snaptik payload decode failed for url: %s", tiktok_url)
        return None

    download_url = _extract_download_url(decoded_payload)
    if not download_url:
        logger.warning("Snaptik download url not found for url: %s", tiktok_url)
        return None

    return download_url


async def _download_video_payload(
    *,
    download_url: str,
    httpx_client: httpx.AsyncClient,
) -> DownloadedTikTokVideo | None:
    request_headers = {"User-Agent": DEFAULT_USER_AGENT}
    video_response = await httpx_client.get(
        download_url,
        headers=request_headers,
        follow_redirects=True,
    )
    video_response.raise_for_status()

    video_bytes = video_response.content
    if not video_bytes:
        return None

    content_type = video_response.headers.get("content-type", "video/mp4")
    if ";" in content_type:
        content_type = content_type.split(";", 1)[0].strip()
    if not content_type.startswith("video/"):
        content_type = "video/mp4"

    return DownloadedTikTokVideo(content=video_bytes, content_type=content_type)


async def maybe_download_tiktok_videos(
    *,
    cleaned_content: str,
    actual_model: str,
    httpx_client: httpx.AsyncClient,
) -> list[DownloadedTikTokVideo]:
    """Download TikTok videos via Snaptik for Gemini requests when URLs are present."""
    if not is_gemini_model(actual_model):
        return []

    tiktok_urls = _extract_tiktok_urls(cleaned_content)
    if not tiktok_urls:
        return []

    async def _download_for_url(tiktok_url: str) -> DownloadedTikTokVideo | None:
        try:
            download_url = await _fetch_snaptik_download_url(
                tiktok_url=tiktok_url,
                httpx_client=httpx_client,
            )
            if not download_url:
                return None

            return await _download_video_payload(
                download_url=download_url,
                httpx_client=httpx_client,
            )
        except (httpx.HTTPError, ValueError) as exc:
            log_exception(
                logger=logger,
                message="Failed to download TikTok video for Gemini request",
                error=exc,
                context={"tiktok_url": tiktok_url},
            )
            return None

    downloaded_videos = await asyncio.gather(
        *(_download_for_url(url) for url in tiktok_urls),
    )
    return [video for video in downloaded_videos if video is not None]


async def maybe_download_tiktok_video(
    *,
    cleaned_content: str,
    actual_model: str,
    httpx_client: httpx.AsyncClient,
) -> DownloadedTikTokVideo | None:
    """Download the first TikTok video for Gemini requests when a URL is present."""
    downloaded_videos = await maybe_download_tiktok_videos(
        cleaned_content=cleaned_content,
        actual_model=actual_model,
        httpx_client=httpx_client,
    )
    if not downloaded_videos:
        return None
    return downloaded_videos[0]
