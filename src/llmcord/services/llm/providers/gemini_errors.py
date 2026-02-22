"""Gemini error classification helpers.

These utilities normalize Gemini backend/client errors into a small set of
categories so higher-level logic (generation loop, search decider) can make
stable retry / key-rotation decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

GeminiErrorAction = Literal[
    "remove_key",  # treat as key/quota specific; try next key
    "skip_provider",  # treat as provider/request specific; skip remaining keys
]


@dataclass(slots=True)
class GeminiErrorClassification:
    """Normalized view of a Gemini error."""

    http_status: int | None
    api_status: str | None
    message: str
    action: GeminiErrorAction


_STATUS_PATTERN = re.compile(r'"status"\s*:\s*"([A-Z_]+)"')
_CODE_PATTERN = re.compile(r'"code"\s*:\s*(\d{3})')
_PAREN_CODE_PATTERN = re.compile(r"\((\d{3})\)")

HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_TOO_MANY_REQUESTS = 429
HTTP_INTERNAL_SERVER_ERROR = 500
HTTP_SERVICE_UNAVAILABLE = 503
HTTP_GATEWAY_TIMEOUT = 504

_SERVER_ERROR_MIN = 500


def _get_exception_status_code(error: BaseException) -> int | None:
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(error, "response", None)
    if response is not None:
        resp_code = getattr(response, "status_code", None)
        if isinstance(resp_code, int):
            return resp_code

    return None


def _extract_code_from_text(text: str) -> int | None:
    if match := _CODE_PATTERN.search(text):
        return int(match.group(1))
    if match := _PAREN_CODE_PATTERN.search(text):
        return int(match.group(1))
    return None


def _extract_status_from_text(text: str) -> str | None:
    match = _STATUS_PATTERN.search(text)
    if not match:
        return None
    return match.group(1)


def _is_leaked_api_key_error(lowered: str) -> bool:
    return "reported as leaked" in lowered and "api key" in lowered


def _build_classification(
    *,
    http_status: int | None,
    api_status: str | None,
    message: str,
    action: GeminiErrorAction,
) -> GeminiErrorClassification:
    return GeminiErrorClassification(
        http_status=http_status,
        api_status=api_status,
        message=message,
        action=action,
    )


def _classify_by_api_status(
    *,
    http_status: int | None,
    api_status: str,
    lowered: str,
) -> GeminiErrorClassification | None:
    if api_status == "FAILED_PRECONDITION" or "enable billing" in lowered:
        return _build_classification(
            http_status=http_status or HTTP_BAD_REQUEST,
            api_status="FAILED_PRECONDITION",
            message=(
                "Gemini request failed precondition (billing/region). "
                "If you are using the free tier, enable billing in Google AI Studio."
            ),
            action="skip_provider",
        )

    api_rules: dict[str, tuple[GeminiErrorAction, str]] = {
        "INVALID_ARGUMENT": (
            "skip_provider",
            "Gemini rejected the request as invalid (INVALID_ARGUMENT).",
        ),
        "PERMISSION_DENIED": (
            "remove_key",
            "Gemini permission denied (check API key permissions/model access).",
        ),
        "NOT_FOUND": (
            "skip_provider",
            "Gemini resource not found (NOT_FOUND).",
        ),
        "RESOURCE_EXHAUSTED": (
            "remove_key",
            "Gemini rate limit/quota exceeded (RESOURCE_EXHAUSTED).",
        ),
        "DEADLINE_EXCEEDED": (
            "skip_provider",
            "Gemini request timed out (DEADLINE_EXCEEDED).",
        ),
        "UNAVAILABLE": (
            "skip_provider",
            "Gemini service temporarily unavailable (UNAVAILABLE).",
        ),
        "INTERNAL": (
            "skip_provider",
            "Gemini internal error (INTERNAL).",
        ),
    }

    rule = api_rules.get(api_status)
    if rule is None:
        return None

    action, message = rule
    return _build_classification(
        http_status=http_status,
        api_status=api_status,
        message=message,
        action=action,
    )


def _classify_by_http_status(
    *,
    http_status: int,
    api_status: str | None,
) -> GeminiErrorClassification:
    if http_status in {HTTP_UNAUTHORIZED, HTTP_FORBIDDEN}:
        return _build_classification(
            http_status=http_status,
            api_status=api_status,
            message="Gemini authentication/permission error.",
            action="remove_key",
        )

    if http_status == HTTP_TOO_MANY_REQUESTS:
        return _build_classification(
            http_status=http_status,
            api_status=api_status,
            message="Gemini rate limit/quota exceeded.",
            action="remove_key",
        )

    if http_status >= _SERVER_ERROR_MIN:
        return _build_classification(
            http_status=http_status,
            api_status=api_status,
            message="Gemini server error.",
            action="skip_provider",
        )

    if http_status == HTTP_BAD_REQUEST:
        return _build_classification(
            http_status=http_status,
            api_status=api_status,
            message="Gemini rejected the request (400).",
            action="skip_provider",
        )

    if http_status == HTTP_NOT_FOUND:
        return _build_classification(
            http_status=http_status,
            api_status=api_status,
            message="Gemini resource not found (404).",
            action="skip_provider",
        )

    return _build_classification(
        http_status=http_status,
        api_status=api_status,
        message="Gemini request failed.",
        action="skip_provider",
    )


def classify_gemini_error(error: Exception) -> GeminiErrorClassification | None:
    """Classify a Gemini-related exception.

    Returns:
        A GeminiErrorClassification when the error looks like a Gemini backend
        error, otherwise None.

    """
    raw_text = str(error).strip()
    lowered = raw_text.lower()

    classification: GeminiErrorClassification | None = None

    # Leaked key blocking is a special-case string not always accompanied by
    # stable JSON fields.
    if _is_leaked_api_key_error(lowered):
        classification = _build_classification(
            http_status=_get_exception_status_code(error) or HTTP_FORBIDDEN,
            api_status="PERMISSION_DENIED",
            message=(
                "Gemini API key was reported as leaked. Please use another API key."
            ),
            action="remove_key",
        )
    else:
        http_status = _get_exception_status_code(error) or _extract_code_from_text(
            raw_text,
        )
        api_status = _extract_status_from_text(raw_text)

        if http_status is None and api_status is None:
            return None

        if api_status is not None:
            classification = _classify_by_api_status(
                http_status=http_status,
                api_status=api_status,
                lowered=lowered,
            )

        if classification is None and isinstance(http_status, int):
            classification = _classify_by_http_status(
                http_status=http_status,
                api_status=api_status,
            )

    return classification
