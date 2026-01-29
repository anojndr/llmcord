import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from llmcord.services.search import (
    get_current_datetime_strings,
    convert_messages_to_openai_format,
    decide_web_search,
    _run_decider_once,
    tavily_search,
)

def run_async(coro):
    return asyncio.run(coro)

def test_get_current_datetime_strings():
    date_str, time_str = get_current_datetime_strings()
    assert date_str
    assert time_str

def test_convert_messages_to_openai_format():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"}
    ]
    formatted = convert_messages_to_openai_format(messages, reverse=False)
    assert formatted[0]["role"] == "user"
    assert formatted[1]["role"] == "assistant"
    
    formatted_reversed = convert_messages_to_openai_format(messages, reverse=True)
    assert formatted_reversed[0]["role"] == "assistant"
    assert formatted_reversed[1]["role"] == "user"

def test_convert_messages_with_system_prompt():
    messages = [{"role": "user", "content": "hello"}]
    formatted = convert_messages_to_openai_format(messages, system_prompt="Sys", reverse=False)
    assert formatted[0]["content"] == "Sys"
    assert formatted[0]["role"] == "system"

def test_decide_web_search_no_search_needed(mock_dependencies):
    # Mock _run_decider_once to return no search
    with patch("llmcord.services.search._run_decider_once", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = ({"needs_search": False}, False)
        
        result = run_async(decide_web_search([], {"provider": "gemini", "model": "test", "api_keys": ["k"]}))
        assert result["needs_search"] is False

def test_decide_web_search_needs_search(mock_dependencies):
    with patch("llmcord.services.search._run_decider_once", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = ({"needs_search": True, "queries": ["test"]}, False)
        
        result = run_async(decide_web_search([], {"provider": "gemini", "model": "test", "api_keys": ["k"]}))
        assert result["needs_search"] is True
        assert result["queries"] == ["test"]

def test_tavily_search_success():
    with patch("llmcord.services.search._get_tavily_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [{"title": "test", "url": "http://test.com"}]}
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        result = run_async(tavily_search("query", "key"))
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "test"
