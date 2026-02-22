from __future__ import annotations

import httpx
import pytest

from llmcord.services.search.core import _search_single_query_tavily
from llmcord.services.search.tavily import tavily_search

HTTP_TOO_MANY_REQUESTS = 429


class _TestReadTimeout(httpx.ReadTimeout):
    def __init__(self, *, request: httpx.Request) -> None:
        super().__init__("timed out", request=request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "kind"),
    [
        (400, "bad_request"),
        (401, "unauthorized"),
        (429, "rate_limited"),
        (432, "plan_limit_exceeded"),
        (433, "paygo_limit_exceeded"),
        (500, "server_error"),
    ],
)
async def test_tavily_search_parses_detail_error_schema(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    kind: str,
) -> None:
    request = httpx.Request("POST", "https://api.tavily.com/search")
    response = httpx.Response(
        status_code,
        json={"detail": {"error": "simulated error"}},
        request=request,
        headers={"retry-after": "7"} if status_code == HTTP_TOO_MANY_REQUESTS else {},
    )

    async def _fake_request_with_retries(
        *_args: object,
        **_kwargs: object,
    ) -> httpx.Response:
        return response

    # Avoid creating real clients in this unit test.
    monkeypatch.setattr(
        "llmcord.services.search.tavily._get_client_from_package",
        lambda: object(),
    )
    monkeypatch.setattr(
        "llmcord.services.search.tavily.request_with_retries",
        _fake_request_with_retries,
    )

    result = await tavily_search(
        query="hello world",
        tavily_api_key="tvly-test",
        max_results=5,
        search_depth="basic",
    )

    assert "error" in result
    assert result["query"] == "hello world"
    assert result["status_code"] == status_code
    assert isinstance(result.get("tavily_error"), dict)
    assert result["tavily_error"]["http_status"] == status_code
    assert result["tavily_error"]["kind"] == kind
    if status_code == HTTP_TOO_MANY_REQUESTS:
        assert result["tavily_error"]["retry_after"] == "7"


@pytest.mark.asyncio
async def test_tavily_search_handles_malformed_json_with_unexpected_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("POST", "https://api.tavily.com/search")
    response = httpx.Response(
        200,
        content=b"not-json",
        request=request,
    )

    async def _fake_request_with_retries(
        *_args: object,
        **_kwargs: object,
    ) -> httpx.Response:
        return response

    monkeypatch.setattr(
        "llmcord.services.search.tavily._get_client_from_package",
        lambda: object(),
    )
    monkeypatch.setattr(
        "llmcord.services.search.tavily.request_with_retries",
        _fake_request_with_retries,
    )

    result = await tavily_search(
        query="hello world",
        tavily_api_key="tvly-test",
    )

    assert "error" in result
    assert result["query"] == "hello world"
    assert result["tavily_error"]["kind"] == "unexpected_response"


@pytest.mark.asyncio
async def test_tavily_search_timeout_is_captured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("POST", "https://api.tavily.com/search")

    async def _fake_request_with_retries(
        *_args: object,
        **_kwargs: object,
    ) -> httpx.Response:
        raise _TestReadTimeout(request=request)

    monkeypatch.setattr(
        "llmcord.services.search.tavily._get_client_from_package",
        lambda: object(),
    )
    monkeypatch.setattr(
        "llmcord.services.search.tavily.request_with_retries",
        _fake_request_with_retries,
    )

    result = await tavily_search(
        query="hello world",
        tavily_api_key="tvly-test",
    )

    assert "error" in result
    assert result["query"] == "hello world"
    assert result["tavily_error"]["kind"] == "timeout"


@pytest.mark.asyncio
async def test_tavily_key_rotation_short_circuits_on_bad_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def _fake_tavily_search(
        query: str,
        tavily_api_key: str,
        max_results: int = 5,
        search_depth: str = "basic",
    ) -> dict:
        _ = (max_results, search_depth)
        calls.append(tavily_api_key)
        return {
            "error": "HTTP 400: simulated",
            "query": query,
            "status_code": 400,
            "tavily_error": {"http_status": 400, "kind": "bad_request"},
        }

    monkeypatch.setattr(
        "llmcord.services.search.core.tavily_search",
        _fake_tavily_search,
    )

    result = await _search_single_query_tavily(
        query="bad input",
        depth="basic",
        keys=["tvly-1", "tvly-2"],
        max_results_per_query=5,
    )

    assert "error" in result
    assert calls == ["tvly-1"]
