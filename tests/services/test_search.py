"""Tests for search service helpers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, TypeVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmcord.services.search import (
    convert_messages_to_openai_format,
    decide_web_search,
    get_current_datetime_strings,
    tavily_search,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable

pytestmark = pytest.mark.usefixtures("mock_dependencies")

T = TypeVar("T")


def run_async(coro: Awaitable[T]) -> T:
    """Run an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


def assert_true(*, condition: bool, message: str) -> None:
    """Raise an AssertionError when a condition is false."""
    if not condition:
        raise AssertionError(message)


def test_get_current_datetime_strings() -> None:
    """Return values should contain non-empty date and time strings."""
    date_str, time_str = get_current_datetime_strings()
    assert_true(condition=bool(date_str), message="Expected non-empty date string")
    assert_true(condition=bool(time_str), message="Expected non-empty time string")


def test_convert_messages_to_openai_format() -> None:
    """Ensure message format conversion preserves roles order."""
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    formatted = convert_messages_to_openai_format(messages, reverse=False)
    assert_true(
        condition=formatted[0]["role"] == "user",
        message="Expected user role first",
    )
    assert_true(
        condition=formatted[1]["role"] == "assistant",
        message="Expected assistant role second",
    )

    formatted_reversed = convert_messages_to_openai_format(messages, reverse=True)
    assert_true(
        condition=formatted_reversed[0]["role"] == "assistant",
        message="Expected assistant role first when reversed",
    )
    assert_true(
        condition=formatted_reversed[1]["role"] == "user",
        message="Expected user role second when reversed",
    )


def test_convert_messages_with_system_prompt() -> None:
    """System prompt should be inserted as a system role entry."""
    messages = [{"role": "user", "content": "hello"}]
    formatted = convert_messages_to_openai_format(
        messages,
        system_prompt="Sys",
        reverse=False,
    )
    assert_true(
        condition=formatted[0]["content"] == "Sys",
        message="Expected system prompt",
    )
    assert_true(
        condition=formatted[0]["role"] == "system",
        message="Expected system role",
    )


def test_decide_web_search_no_search_needed() -> None:
    """Return should indicate no search when decider says so."""
    # Mock _run_decider_once to return no search
    with patch(
        "llmcord.services.search._run_decider_once",
        new_callable=AsyncMock,
    ) as mock_run:
        mock_run.return_value = ({"needs_search": False}, False)

        result = run_async(
            decide_web_search(
                [],
                {"provider": "gemini", "model": "test", "api_keys": ["k"]},
            ),
        )
        assert_true(
            condition=result["needs_search"] is False,
            message="Expected needs_search False",
        )


def test_decide_web_search_needs_search() -> None:
    """Return should include queries when a search is needed."""
    with patch(
        "llmcord.services.search._run_decider_once",
        new_callable=AsyncMock,
    ) as mock_run:
        mock_run.return_value = ({"needs_search": True, "queries": ["test"]}, False)

        result = run_async(
            decide_web_search(
                [],
                {"provider": "gemini", "model": "test", "api_keys": ["k"]},
            ),
        )
        assert_true(
            condition=result["needs_search"] is True,
            message="Expected needs_search True",
        )
        assert_true(
            condition=result["queries"] == ["test"],
            message="Expected query list",
        )


def test_tavily_search_success() -> None:
    """Return search results from Tavily client response."""
    with patch("llmcord.services.search._get_tavily_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [{"title": "test", "url": "http://test.com"}]}
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = run_async(tavily_search("query", "key"))
        assert_true(
            condition=len(result["results"]) == 1,
            message="Expected single result",
        )
        assert_true(
            condition=result["results"][0]["title"] == "test",
            message="Expected result title 'test'",
        )
