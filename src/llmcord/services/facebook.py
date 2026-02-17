"""Facebook video download helpers via third-party HTTP endpoints."""

from __future__ import annotations

import asyncio
import base64
import html
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx
from curl_cffi.requests import AsyncSession

from llmcord.core.config import DEFAULT_USER_AGENT, is_gemini_model
from llmcord.core.error_handling import log_exception

logger = logging.getLogger(__name__)

_HTTP_STATUS_OK = 200
_JWT_PART_COUNT = 3

_FACEBOOK_URL_RE = re.compile(
    r"https?://(?:www\.|m\.)?(?:facebook\.com|fb\.watch)/[^\s<>()\[\]{}]+",
    re.IGNORECASE,
)

_FDOWNLOADER_URL = "https://fdownloader.net/en"
_K_URL_SEARCH_RE = re.compile(r'k_url_search="(?P<url>https://[^"]+/api/ajaxSearch)"')
_K_EXP_RE = re.compile(r'k_exp="(?P<exp>\d+)"')
_K_TOKEN_RE = re.compile(r'k_token="(?P<token>[0-9a-f]{64})"', re.IGNORECASE)

_SNAPCDN_TOKEN_RE = re.compile(
    r"https://dl\.snapcdn\.app/download\?token=(?P<token>[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class DownloadedFacebookVideo:
    """Downloaded Facebook video payload for Gemini file input."""

    content: bytes
    content_type: str


@dataclass(frozen=True, slots=True)
class _FDownloaderParams:
    search_url: str
    k_exp: str
    k_token: str


def _extract_facebook_urls(text: str) -> list[str]:
    unique_urls: list[str] = []
    seen_urls: set[str] = set()
    for match in _FACEBOOK_URL_RE.finditer(text):
        candidate_url = match.group(0).rstrip('.,;:!?)"')
        if candidate_url in seen_urls:
            continue
        seen_urls.add(candidate_url)
        unique_urls.append(candidate_url)
    return unique_urls


def _parse_fdownloader_params(page_html: str) -> _FDownloaderParams | None:
    search_match = _K_URL_SEARCH_RE.search(page_html)
    exp_match = _K_EXP_RE.search(page_html)
    token_match = _K_TOKEN_RE.search(page_html)
    if not (search_match and exp_match and token_match):
        return None

    return _FDownloaderParams(
        search_url=search_match.group("url"),
        k_exp=exp_match.group("exp"),
        k_token=token_match.group("token"),
    )


async def _fetch_fdownloader_params(
    *,
    session: AsyncSession,
) -> _FDownloaderParams | None:
    response = await session.get(
        _FDOWNLOADER_URL,
        headers={"user-agent": DEFAULT_USER_AGENT},
    )
    if response.status_code != _HTTP_STATUS_OK:
        logger.warning(
            "FDownloader homepage request failed with status=%s",
            response.status_code,
        )
        return None

    params = _parse_fdownloader_params(response.text)
    if not params:
        logger.warning("FDownloader params not found in homepage response")
    return params


async def _fetch_fdownloader_result_html(
    *,
    facebook_url: str,
    params: _FDownloaderParams,
    session: AsyncSession,
) -> str | None:
    # NOTE: FDownloader includes a Cloudflare Turnstile widget, but its API currently
    # accepts `cftoken=0` for non-render downloads.
    request_data = {
        "k_exp": params.k_exp,
        "k_token": params.k_token,
        "q": facebook_url,
        "lang": "en",
        "web": "fdownloader.net",
        "v": "v2",
        "w": "",
        "cftoken": "0",
    }
    headers = {
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "referer": "https://fdownloader.net/",
        "user-agent": DEFAULT_USER_AGENT,
    }
    response = await session.post(
        params.search_url,
        data=request_data,
        headers=headers,
    )
    if response.status_code != _HTTP_STATUS_OK:
        logger.warning(
            "FDownloader ajaxSearch failed status=%s url=%s",
            response.status_code,
            facebook_url,
        )
        return None

    try:
        payload = response.json()
    except ValueError:
        logger.warning(
            "FDownloader ajaxSearch returned non-JSON response for url=%s",
            facebook_url,
        )
        return None

    data_html = payload.get("data")
    if not isinstance(data_html, str) or not data_html.strip():
        return None

    return data_html


def _extract_first_snapcdn_token(result_html: str) -> str | None:
    match = _SNAPCDN_TOKEN_RE.search(result_html)
    if not match:
        return None
    return match.group("token")


def _decode_jwt_payload(token: str) -> dict[str, Any] | None:
    parts = token.split(".")
    if len(parts) != _JWT_PART_COUNT:
        return None

    payload_b64 = parts[1]
    payload_b64 += "=" * (-len(payload_b64) % 4)
    try:
        payload_json = base64.urlsafe_b64decode(payload_b64.encode("ascii"))
        decoded = json.loads(payload_json.decode("utf-8"))
    except (ValueError, UnicodeError):
        return None

    if not isinstance(decoded, dict):
        return None
    return decoded


def _extract_download_url_from_snapcdn_token(token: str) -> str | None:
    payload = _decode_jwt_payload(token)
    if not payload:
        return None

    url = payload.get("url")
    if not isinstance(url, str) or not url.startswith("http"):
        return None

    return html.unescape(url)


async def _download_video_payload(
    *,
    download_url: str,
    httpx_client: httpx.AsyncClient,
) -> DownloadedFacebookVideo | None:
    video_response = await httpx_client.get(
        download_url,
        headers={"User-Agent": DEFAULT_USER_AGENT},
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

    return DownloadedFacebookVideo(content=video_bytes, content_type=content_type)


async def maybe_download_facebook_videos(
    *,
    cleaned_content: str,
    actual_model: str,
    httpx_client: httpx.AsyncClient,
) -> list[DownloadedFacebookVideo]:
    """Download Facebook videos via FDownloader for Gemini requests."""
    if not is_gemini_model(actual_model):
        return []

    facebook_urls = _extract_facebook_urls(cleaned_content)
    if not facebook_urls:
        return []

    async with AsyncSession(impersonate="chrome120") as session:
        params = await _fetch_fdownloader_params(session=session)
        if not params:
            return []

        async def _download_for_url(
            facebook_url: str,
        ) -> DownloadedFacebookVideo | None:
            try:
                result_html = await _fetch_fdownloader_result_html(
                    facebook_url=facebook_url,
                    params=params,
                    session=session,
                )
                if not result_html:
                    return None

                token = _extract_first_snapcdn_token(result_html)
                if not token:
                    return None

                download_url = _extract_download_url_from_snapcdn_token(token)
                if not download_url:
                    return None

                return await _download_video_payload(
                    download_url=download_url,
                    httpx_client=httpx_client,
                )
            except (httpx.HTTPError, ValueError) as exc:
                log_exception(
                    logger=logger,
                    message="Failed to download Facebook video for Gemini request",
                    error=exc,
                    context={"facebook_url": facebook_url},
                )
                return None

        downloaded_videos = await asyncio.gather(
            *(_download_for_url(url) for url in facebook_urls),
        )
        return [video for video in downloaded_videos if video is not None]


async def maybe_download_facebook_video(
    *,
    cleaned_content: str,
    actual_model: str,
    httpx_client: httpx.AsyncClient,
) -> DownloadedFacebookVideo | None:
    """Download the first Facebook video for Gemini requests when a URL is present."""
    downloaded_videos = await maybe_download_facebook_videos(
        cleaned_content=cleaned_content,
        actual_model=actual_model,
        httpx_client=httpx_client,
    )
    if not downloaded_videos:
        return None
    return downloaded_videos[0]
