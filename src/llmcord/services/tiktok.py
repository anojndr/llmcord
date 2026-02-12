"""TikTok video download helpers via Snaptik HTTP endpoints."""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass

import httpx

from llmcord.core.config import is_gemini_model

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


def _extract_first_tiktok_url(text: str) -> str | None:
    match = _TIKTOK_URL_RE.search(text)
    if not match:
        return None
    return match.group(0).rstrip('.,;:!?)"')


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
    request_headers = {"user-agent": "TelegramBot (like TwitterBot)"}
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


async def maybe_download_tiktok_video(
    *,
    cleaned_content: str,
    actual_model: str,
    httpx_client: httpx.AsyncClient,
) -> DownloadedTikTokVideo | None:
    """Download TikTok video via Snaptik for Gemini requests when URL is present."""
    if not is_gemini_model(actual_model):
        return None

    tiktok_url = _extract_first_tiktok_url(cleaned_content)
    if not tiktok_url:
        return None

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
    except (httpx.HTTPError, ValueError):
        logger.exception("Failed to download TikTok video for Gemini request")
        return None
