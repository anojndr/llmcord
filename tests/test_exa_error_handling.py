from __future__ import annotations

# ruff: noqa: SLF001
import json

from llmcord.services.search import exa as exa_mod


def test_extract_exa_results_surfaces_tag_and_request_id() -> None:
    payload = {
        "requestId": "67207943fab9832d162b5317f4cca830",
        "error": "Invalid request body",
        "tag": "INVALID_REQUEST_BODY",
    }
    mcp_result = {
        "result": {
            "isError": False,
            "content": [{"text": json.dumps(payload)}],
        },
    }

    out = exa_mod._extract_exa_results(mcp_result, query="latest news")

    assert "error" in out
    assert "INVALID_REQUEST_BODY" in str(out["error"])
    assert "requestId=67207943fab9832d162b5317f4cca830" in str(out["error"])


def test_extract_exa_results_formats_tool_errors_with_payload() -> None:
    payload = {
        "requestId": "req-123",
        "error": "Missing or invalid API key",
        "tag": "INVALID_API_KEY",
    }
    mcp_result = {
        "result": {
            "isError": True,
            "content": [{"text": json.dumps(payload)}],
        },
    }

    out = exa_mod._extract_exa_results(mcp_result, query="latest news")

    assert "error" in out
    assert "INVALID_API_KEY" in str(out["error"])
    assert "requestId=req-123" in str(out["error"])


def test_extract_exa_results_surfaces_statuses_errors_when_no_results() -> None:
    payload = {
        "results": [],
        "statuses": [
            {
                "id": "https://example.com",
                "status": "error",
                "error": {"tag": "CRAWL_NOT_FOUND", "httpStatusCode": 404},
            },
        ],
    }
    mcp_result = {
        "result": {
            "isError": False,
            "content": [{"text": json.dumps(payload)}],
        },
    }

    out = exa_mod._extract_exa_results(mcp_result, query="example")

    assert out.get("query") == "example"
    assert "error" in out
    assert "CRAWL_NOT_FOUND" in str(out["error"])
    assert "exa_status_errors" in out


def test_extract_exa_results_handles_429_simple_error_shape() -> None:
    payload = {
        "error": "You've exceeded your Exa rate limit of 10 requests per second.",
    }
    mcp_result = {
        "result": {
            "isError": False,
            "content": [{"text": json.dumps(payload)}],
        },
    }

    out = exa_mod._extract_exa_results(mcp_result, query="rate limit")

    assert "error" in out
    assert "rate limit" in str(out["error"]).lower()
