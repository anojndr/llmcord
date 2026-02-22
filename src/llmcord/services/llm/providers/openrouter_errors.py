"""OpenRouter error parsing + classification helpers.

OpenRouter returns a JSON error envelope:

    {"error": {"code": <int>, "message": <str>, "metadata": {...}?}}

In streaming mode, errors that occur mid-stream are sent as SSE "data" events
with an `error` field at the top level and `choices[0].finish_reason == "error"`.

LiteLLM may surface these in a few different ways (exceptions with embedded JSON,
objects with `.error`, or dict-like payloads). This module provides small,
defensive utilities to:

- Extract OpenRouter error details from objects/dicts/text
- Raise a stable exception for mid-stream and 200-with-error-body cases
- Classify errors into actions used by llmcord's key-rotation and fallback logic
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, cast

OpenRouterErrorAction = Literal[
    "remove_key",  # key/credits/rate-limit specific; try next key
    "skip_provider",  # request/provider specific; skip remaining keys
]


@dataclass(slots=True)
class OpenRouterErrorDetails:
    """Normalized view of an OpenRouter error."""

    http_status: int | None
    code: int | str | None
    message: str
    metadata: dict[str, Any] | None
    provider: str | None
    model: str | None


@dataclass(slots=True)
class OpenRouterErrorClassification:
    """Classification used by generation/search loops."""

    http_status: int | None
    code: int | str | None
    message: str
    action: OpenRouterErrorAction


class OpenRouterAPIError(RuntimeError):
    """Raised when OpenRouter returns an error payload (including mid-stream)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: int | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create an OpenRouterAPIError.

        Args:
            message: Human-readable error message.
            status_code: Optional HTTP status code when known.
            code: OpenRouter error code (numeric HTTP code or string code).
            metadata: Optional OpenRouter error metadata payload.

        """
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.metadata = metadata


_STREAM_FINISH_REASON_ERROR_MESSAGE = (
    "OpenRouter stream terminated with finish_reason=error"
)


_JSON_SUBSTRING_RE = re.compile(r"\{.*\}", flags=re.DOTALL)

HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_PAYMENT_REQUIRED = 402
HTTP_FORBIDDEN = 403
HTTP_REQUEST_TIMEOUT = 408
HTTP_TOO_MANY_REQUESTS = 429
HTTP_BAD_GATEWAY = 502
HTTP_SERVICE_UNAVAILABLE = 503

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


def _try_parse_json(value: str) -> object | None:
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _extract_json_from_text(text: str) -> object | None:
    """Best-effort JSON extraction from an exception string."""
    raw = text.strip()
    if not raw:
        return None

    if (parsed := _try_parse_json(raw)) is not None:
        return parsed

    match = _JSON_SUBSTRING_RE.search(raw)
    if not match:
        return None

    return _try_parse_json(match.group(0))


def _cast_mapping(value: dict[Any, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", value)


def _try_pydantic_dump(obj: object) -> dict[str, Any] | None:
    dumped = getattr(obj, "model_dump", None)
    if not callable(dumped):
        return None

    try:
        out = dumped()
    except TypeError:
        out = dumped(mode="python")

    if isinstance(out, dict):
        return _cast_mapping(out)
    return None


def _try_dict_method(obj: object) -> dict[str, Any] | None:
    dumped = getattr(obj, "dict", None)
    if not callable(dumped):
        return None

    try:
        out = dumped()
    except TypeError:
        out = dumped(exclude_none=True)

    if isinstance(out, dict):
        return _cast_mapping(out)
    return None


def _try_model_extra(obj: object) -> dict[str, Any] | None:
    model_extra = getattr(obj, "model_extra", None)
    if isinstance(model_extra, dict):
        return _cast_mapping(model_extra)
    return None


def _to_mapping(obj: object) -> dict[str, Any] | None:
    if isinstance(obj, dict):
        return _cast_mapping(obj)

    for extractor in (_try_pydantic_dump, _try_dict_method, _try_model_extra):
        extracted = extractor(obj)
        if extracted is not None:
            return extracted

    return None


def _coerce_error_details(
    *,
    error_obj: object,
    provider: str | None = None,
    model: str | None = None,
) -> OpenRouterErrorDetails | None:
    """Coerce either a standard error envelope or top-level error object."""
    if error_obj is None:
        return None

    error_map = _to_mapping(error_obj)
    if error_map is None:
        code = getattr(error_obj, "code", None)
        message = getattr(error_obj, "message", None)
        if isinstance(message, str) and message.strip():
            return OpenRouterErrorDetails(
                http_status=code if isinstance(code, int) else None,
                code=code if isinstance(code, (int, str)) else None,
                message=message.strip(),
                metadata=None,
                provider=provider,
                model=model,
            )
        return None

    code = error_map.get("code")
    message = error_map.get("message")
    metadata = error_map.get("metadata")
    if not isinstance(message, str) or not message.strip():
        return None

    if metadata is not None and not isinstance(metadata, dict):
        metadata = None

    http_status: int | None = code if isinstance(code, int) else None
    return OpenRouterErrorDetails(
        http_status=http_status,
        code=code if isinstance(code, (int, str)) else None,
        message=message.strip(),
        metadata=metadata,
        provider=provider,
        model=model,
    )


def extract_openrouter_error_details(obj: object) -> OpenRouterErrorDetails | None:
    """Extract OpenRouter error details from a dict/object payload."""
    payload = _to_mapping(obj)
    provider = None
    model = None

    if payload is not None:
        provider_value = payload.get("provider")
        if isinstance(provider_value, str):
            provider = provider_value
        model_value = payload.get("model")
        if isinstance(model_value, str):
            model = model_value

        if "error" in payload and isinstance(payload.get("error"), dict):
            # Standard error response or mid-stream chunk with top-level error.
            details = _coerce_error_details(
                error_obj=payload.get("error"),
                provider=provider,
                model=model,
            )
            if details is not None:
                return details

    error_attr = getattr(obj, "error", None)
    details = _coerce_error_details(
        error_obj=error_attr,
        provider=provider,
        model=model,
    )
    if details is not None:
        return details

    return None


def _format_metadata(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""

    # Moderation errors
    reasons = metadata.get("reasons")
    if (
        isinstance(reasons, list)
        and reasons
        and all(isinstance(item, str) for item in reasons)
    ):
        joined = ", ".join(reasons[:5])
        return f" | reasons={joined}"

    # Provider errors
    provider_name = metadata.get("provider_name")
    if isinstance(provider_name, str) and provider_name.strip():
        return f" | provider={provider_name.strip()}"

    return ""


def format_openrouter_error(details: OpenRouterErrorDetails) -> str:
    """Format OpenRouter error details into a stable, human-readable string."""
    code = details.code
    status = details.http_status
    meta_suffix = _format_metadata(details.metadata)

    if isinstance(status, int):
        return f"OpenRouter HTTP {status}: {details.message}{meta_suffix}"

    if code is not None:
        return f"OpenRouter error ({code}): {details.message}{meta_suffix}"

    return f"OpenRouter error: {details.message}{meta_suffix}"


def raise_for_openrouter_payload_error(
    *,
    payload_obj: object,
) -> None:
    """Raise when a response/chunk includes an OpenRouter error.

    This checks both:
    - Standard OpenRouter error envelopes (`{"error": ...}`)
    - OpenRouter's "finish_reason=error" cases where an error is embedded in
      `choices` (sometimes without a top-level `error` field)

    """
    details = extract_openrouter_error_details(payload_obj)
    if details is None:
        # Fall back to detecting OpenRouter's "error" finish reason.
        payload = _to_mapping(payload_obj)
        choices = None
        if payload is not None:
            choices = payload.get("choices")
        if choices is None:
            choices = getattr(payload_obj, "choices", None)

        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            finish_reason = None
            if isinstance(first_choice, dict):
                finish_reason = first_choice.get("finish_reason")
            else:
                finish_reason = getattr(first_choice, "finish_reason", None)

            if (
                finish_reason is not None
                and str(finish_reason).strip().lower() == "error"
            ):
                raise_for_openrouter_stream_finish_reason_error()

        return

    raise OpenRouterAPIError(
        format_openrouter_error(details),
        status_code=details.http_status,
        code=details.code,
        metadata=details.metadata,
    )


def raise_for_openrouter_stream_finish_reason_error() -> None:
    """Raise a stable error when OpenRouter stream ends with `finish_reason=error`."""
    raise OpenRouterAPIError(
        _STREAM_FINISH_REASON_ERROR_MESSAGE,
        code="server_error",
    )


def _infer_http_status_from_code(
    *,
    http_status: int | None,
    code: int | str | None,
) -> int | None:
    inferred: int | None = http_status if isinstance(http_status, int) else None
    if inferred is None and isinstance(code, int):
        inferred = code
    if inferred is not None:
        return inferred

    if not isinstance(code, str) or not code:
        return None

    code_lower = code.lower()
    if "rate_limit" in code_lower or "too_many_requests" in code_lower:
        inferred = HTTP_TOO_MANY_REQUESTS
    elif "invalid_api_key" in code_lower or "unauthorized" in code_lower:
        inferred = HTTP_UNAUTHORIZED
    elif "insufficient" in code_lower or "payment" in code_lower:
        inferred = HTTP_PAYMENT_REQUIRED
    elif "moderation" in code_lower or "content_policy" in code_lower:
        inferred = HTTP_FORBIDDEN
    elif "server_error" in code_lower or "provider" in code_lower:
        inferred = HTTP_BAD_GATEWAY

    return inferred


def _extract_openrouter_details_from_exception(
    error: Exception,
) -> tuple[int | None, int | str | None, str] | None:
    raw_text = str(error).strip()

    if isinstance(error, OpenRouterAPIError):
        return error.status_code, error.code, (raw_text or "OpenRouter error")

    http_status = _get_exception_status_code(error)
    parsed = _extract_json_from_text(raw_text)
    details = (
        extract_openrouter_error_details(parsed) if isinstance(parsed, dict) else None
    )

    if details is not None and details.http_status is not None:
        http_status = details.http_status

    if http_status is None and details is None:
        return None

    message = format_openrouter_error(details) if details is not None else raw_text
    code = details.code if details is not None else None
    return http_status, code, message


def _classify_openrouter_status(
    *,
    http_status: int | None,
    code: int | str | None,
    message: str,
) -> OpenRouterErrorClassification:
    action: OpenRouterErrorAction

    if http_status in {
        HTTP_UNAUTHORIZED,
        HTTP_PAYMENT_REQUIRED,
        HTTP_TOO_MANY_REQUESTS,
    }:
        action = "remove_key"
    else:
        action = "skip_provider"

    # Unknown-but-OpenRouter-ish: default to skipping provider so fallbacks can run.
    return OpenRouterErrorClassification(
        http_status=http_status,
        code=code,
        message=message,
        action=action,
    )


def classify_openrouter_error(error: Exception) -> OpenRouterErrorClassification | None:
    """Classify an OpenRouter-related exception.

    This aims to provide stable messages/actions regardless of how LiteLLM
    formats the underlying error.
    """
    extracted = _extract_openrouter_details_from_exception(error)
    if extracted is None:
        return None

    http_status, code, message = extracted
    inferred_status = _infer_http_status_from_code(http_status=http_status, code=code)
    if inferred_status is not None:
        http_status = inferred_status

    # Keep some explicit status codes visible in the classification even though
    # they all currently map to skip/remove decisions.
    return _classify_openrouter_status(
        http_status=http_status,
        code=code,
        message=message,
    )
