"""QuillBot humanizer integration for slash command usage."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from curl_cffi.requests import AsyncSession
from curl_cffi.requests import exceptions as curl_requests_exceptions

from llmcord.core.config import DEFAULT_USER_AGENT

logger = logging.getLogger(__name__)

_QUILLBOT_HUMANIZER_URL = "https://quillbot.com/ai-humanizer"
_WORD_RE = re.compile(r"\S+")

_LOCALE_PARAPHRASER_URL_RE = re.compile(
    r"(?:https?://quillbot\.com)?/locales/prod/[0-9a-z]+/[a-z-]+/Paraphraser",
    re.IGNORECASE,
)
_WORDS_AT_A_TIME_RE = re.compile(
    r"\b(?P<limit>\d{2,4})\s+words\s+at\s+a\s+time\b",
    re.IGNORECASE,
)
_WORDS_RE = re.compile(r"\b(?P<limit>\d{2,4})\s+words\b", re.IGNORECASE)

_HTTP_OK = 200
_HUMANIZE_STRENGTH = 13
_DEFAULT_INPUT_LANGUAGE = "en"
_DEFAULT_DIALECT = "US"
_MIN_PLAUSIBLE_WORD_LIMIT = 10
_MAX_PLAUSIBLE_WORD_LIMIT = 2_000


class QuillBotHumanizerError(RuntimeError):
    """Raised when QuillBot humanization fails."""


@dataclass(frozen=True, slots=True)
class QuillBotHumanizeResult:
    """Result payload for a humanize request."""

    text: str
    word_limit: int
    segment_count: int
    mode: str


def split_text_for_word_limit(text: str, word_limit: int) -> list[str]:
    """Split text into ordered segments that each fit the given word limit."""
    if word_limit <= 0:
        msg = "word_limit must be greater than 0"
        raise ValueError(msg)

    matches = list(_WORD_RE.finditer(text))
    if not matches:
        return []

    segments: list[str] = []
    for start_index in range(0, len(matches), word_limit):
        end_index = min(start_index + word_limit, len(matches)) - 1
        start_char = matches[start_index].start()
        end_char = matches[end_index].end()
        segment = text[start_char:end_char].strip()
        if segment:
            segments.append(segment)

    return segments


async def humanize_text_with_quillbot(text: str) -> QuillBotHumanizeResult:
    """Humanize input text through QuillBot and merge all processed segments."""
    stripped_text = text.strip()
    if not stripped_text:
        msg = "Input text is empty"
        raise QuillBotHumanizerError(msg)

    async with AsyncSession(impersonate="chrome120") as session:
        await _prime_quillbot_with_curl_cffi(session=session)
        word_limit = await _fetch_word_limit_http(session=session)

        segments = split_text_for_word_limit(stripped_text, word_limit)
        if not segments:
            msg = "No words found in input text"
            raise QuillBotHumanizerError(msg)

        outputs = [
            await _humanize_segment_http(session=session, segment=segment)
            for segment in segments
        ]

    merged_text = " ".join(outputs).strip()
    if not merged_text:
        msg = "QuillBot returned an empty response"
        raise QuillBotHumanizerError(msg)

    return QuillBotHumanizeResult(
        text=merged_text,
        word_limit=word_limit,
        segment_count=len(segments),
        mode="humanize",
    )


async def _prime_quillbot_with_curl_cffi(*, session: AsyncSession) -> None:
    """Warm up cookies and anti-bot checks through curl_cffi before browser use."""
    await session.get(
        _QUILLBOT_HUMANIZER_URL,
        headers={"user-agent": DEFAULT_USER_AGENT},
        timeout=30,
    )
    await session.get(
        "https://quillbot.com/api/auth/spam-check",
        headers={"user-agent": DEFAULT_USER_AGENT},
        timeout=30,
    )


async def _fetch_humanizer_html(*, session: AsyncSession) -> str:
    try:
        response = await session.get(
            _QUILLBOT_HUMANIZER_URL,
            headers={"user-agent": DEFAULT_USER_AGENT},
            timeout=30,
        )
    except curl_requests_exceptions.Timeout as exc:
        msg = "QuillBot humanizer page request timed out"
        raise QuillBotHumanizerError(msg) from exc
    except curl_requests_exceptions.RequestException as exc:
        msg = "Failed to fetch QuillBot humanizer page"
        raise QuillBotHumanizerError(msg) from exc

    if response.status_code != _HTTP_OK:
        msg = f"QuillBot page fetch failed with status={response.status_code}"
        raise QuillBotHumanizerError(msg)

    return response.text


def _extract_locale_url_from_html(html: str) -> str:
    normalized = html.replace("\\u002F", "/").replace("\\/", "/")
    locale_url_match = _LOCALE_PARAPHRASER_URL_RE.search(normalized)
    if not locale_url_match:
        msg = "Failed to find QuillBot locale URL in page HTML"
        raise QuillBotHumanizerError(msg)

    locale_url = locale_url_match.group(0)
    if locale_url.startswith("/"):
        return f"https://quillbot.com{locale_url}"
    return locale_url


async def _fetch_locale_payload(
    *,
    session: AsyncSession,
    locale_url: str,
) -> dict[str, object]:
    try:
        locale_response = await session.get(
            locale_url,
            headers={
                "user-agent": DEFAULT_USER_AGENT,
                "referer": _QUILLBOT_HUMANIZER_URL,
            },
            timeout=30,
        )
    except curl_requests_exceptions.Timeout as exc:
        msg = "QuillBot locale request timed out"
        raise QuillBotHumanizerError(msg) from exc
    except curl_requests_exceptions.RequestException as exc:
        msg = "Failed to fetch QuillBot locale payload"
        raise QuillBotHumanizerError(msg) from exc

    if locale_response.status_code != _HTTP_OK:
        msg = (
            "QuillBot locale fetch failed "
            f"status={locale_response.status_code} url={locale_url}"
        )
        raise QuillBotHumanizerError(msg)

    try:
        locale_payload = locale_response.json()
    except ValueError as exc:
        msg = "QuillBot locale payload was not JSON"
        raise QuillBotHumanizerError(msg) from exc

    if not isinstance(locale_payload, dict):
        msg = "QuillBot locale payload had unexpected shape"
        raise QuillBotHumanizerError(msg)

    return locale_payload


def _parse_word_limit_from_locale_payload(locale_payload: dict[str, object]) -> int:
    candidates: list[int] = []
    for value in locale_payload.values():
        if not isinstance(value, str):
            continue

        match = _WORDS_AT_A_TIME_RE.search(value)
        if match:
            candidates.append(int(match.group("limit")))
            continue

        if "word" in value.lower():
            candidates.extend(
                int(fallback.group("limit")) for fallback in _WORDS_RE.finditer(value)
            )

    plausible = [
        n
        for n in candidates
        if _MIN_PLAUSIBLE_WORD_LIMIT <= n <= _MAX_PLAUSIBLE_WORD_LIMIT
    ]
    if not plausible:
        msg = "Failed to parse QuillBot word limit from locale payload"
        raise QuillBotHumanizerError(msg)

    # Prefer the highest plausible limit (handles variants like 125/150).
    return max(plausible)


async def _fetch_word_limit_http(*, session: AsyncSession) -> int:
    """Fetch QuillBot's current word limit via HTTP by parsing their locale payload."""
    html = await _fetch_humanizer_html(session=session)
    locale_url = _extract_locale_url_from_html(html)
    locale_payload = await _fetch_locale_payload(session=session, locale_url=locale_url)
    return _parse_word_limit_from_locale_payload(locale_payload)


async def _humanize_segment_http(*, session: AsyncSession, segment: str) -> str:
    """Humanize a single segment using QuillBot's HTTP paraphraser endpoint."""
    payload = {
        "fthresh": -1,
        "autoflip": False,
        "wikify": False,
        "inputLang": _DEFAULT_INPUT_LANGUAGE,
        "strength": _HUMANIZE_STRENGTH,
        "quoteIndex": -1,
        "text": segment,
        "frozenWords": [],
        "nBeams": 4,
        "freezeQuotes": True,
        "preferActive": False,
        "dialect": _DEFAULT_DIALECT,
        "prevSentence": "",
        "nextSentence": "",
        "promptVersion": "v2",
        "multilingualModelVersion": "v2",
    }

    try:
        response = await session.post(
            f"https://quillbot.com/api/paraphraser/single-paraphrase/{_HUMANIZE_STRENGTH}",
            json=payload,
            headers={
                "content-type": "application/json",
                "referer": _QUILLBOT_HUMANIZER_URL,
                "origin": "https://quillbot.com",
                "user-agent": DEFAULT_USER_AGENT,
            },
            timeout=60,
        )
    except curl_requests_exceptions.Timeout as exc:
        logger.warning("QuillBot humanize request timed out")
        msg = "QuillBot request timed out"
        raise QuillBotHumanizerError(msg) from exc
    except curl_requests_exceptions.RequestException as exc:
        logger.warning("QuillBot humanize request failed: %s", exc)
        msg = "QuillBot request failed"
        raise QuillBotHumanizerError(msg) from exc

    if response.status_code != _HTTP_OK:
        msg = f"QuillBot request failed with status={response.status_code}"
        raise QuillBotHumanizerError(msg)

    try:
        payload_json = response.json()
    except ValueError as exc:
        msg = "QuillBot returned non-JSON response"
        raise QuillBotHumanizerError(msg) from exc

    try:
        data = payload_json["data"]
        first_row = data[0]
        # Prefer the exact key used for humanize strength, but be tolerant of changes.
        paras_key = "paras_14"
        if paras_key not in first_row:
            paras_keys = [
                key
                for key in first_row
                if isinstance(key, str) and key.startswith("paras_")
            ]
            if not paras_keys:
                msg = "QuillBot response missing paraphrase alternatives"
                raise QuillBotHumanizerError(msg)
            paras_key = sorted(paras_keys)[0]

        alts = first_row[paras_key]
        first_alt = alts[0]["alt"]
    except (KeyError, IndexError, TypeError) as exc:
        msg = "Unexpected QuillBot response shape"
        raise QuillBotHumanizerError(msg) from exc

    if not isinstance(first_alt, str) or not first_alt.strip():
        msg = "QuillBot returned an empty paraphrase"
        raise QuillBotHumanizerError(msg)

    return first_alt.strip()
