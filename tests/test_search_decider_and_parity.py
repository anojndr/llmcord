from __future__ import annotations

import pytest

from llmcord.logic.search_logic import SearchResolutionContext, resolve_search_metadata
from llmcord.services.search.decider import DeciderRunConfig, _run_decider_once
from llmcord.services.search.utils import convert_messages_to_openai_format

from ._fakes import FakeMessage, FakeUser

EXPECTED_WEB_SEARCH_MAX_CHARS = 4000


def _assert_with_trailing_newline_tolerance(actual: str, expected: str) -> None:
    """Assert exact formatting while allowing only trailing newline variance."""
    normalized = actual.rstrip("\n")
    assert normalized == expected
    trailing = actual[len(normalized) :]
    assert set(trailing).issubset({"\n"})


class _FakeDB:
    def get_message_search_data(self, _message_id: str) -> tuple[None, None, None]:
        return None, None, None

    def get_user_search_decider_model(self, _user_id: str) -> None:
        return None

    def save_message_search_data(self, *_args: object, **_kwargs: object) -> None:
        return None


class _DummyBot:
    def __init__(self, user_id: int = 999) -> None:
        self.user = FakeUser(user_id)


@pytest.mark.asyncio
async def test_web_search_decider_appends_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_decide_web_search(
        messages: list[dict[str, object]],
        _decider_config: dict,
    ) -> dict[str, object]:
        assert messages, "Expected messages passed to decider"
        return {"needs_search": True, "queries": ["latest news"]}

    async def _fake_perform_web_search(
        queries: list[str],
        *args: object,
        **kwargs: object,
    ):
        assert queries == ["latest news"]
        options = kwargs.get("options")
        assert options is not None
        assert (
            getattr(options, "max_chars_per_url", None) == EXPECTED_WEB_SEARCH_MAX_CHARS
        )
        return "--- Search Results ---\nHeadline: Example", {"provider": "mock"}

    def _fake_is_googlelens_query(*args: object, **kwargs: object) -> bool:
        return False

    monkeypatch.setattr(
        "llmcord.logic.search_logic.decide_web_search",
        _fake_decide_web_search,
    )
    monkeypatch.setattr(
        "llmcord.logic.search_logic.perform_web_search",
        _fake_perform_web_search,
    )
    monkeypatch.setattr("llmcord.logic.search_logic.get_bad_keys_db", lambda: _FakeDB())

    bot = _DummyBot()
    msg = FakeMessage(id=20, content="at ai latest news", author=FakeUser(1234))
    messages: list[dict[str, object]] = [{"role": "user", "content": "latest news"}]

    search_metadata = await resolve_search_metadata(
        SearchResolutionContext(
            new_msg=msg,  # type: ignore[arg-type]
            discord_bot=bot,  # type: ignore[arg-type]
            msg_nodes={},
            messages=messages,
            user_warnings=set(),
            tavily_api_keys=["tvly-TEST"],
            config={
                "web_search_provider": "tavily",
                "web_search_max_chars_per_url": EXPECTED_WEB_SEARCH_MAX_CHARS,
                "web_search_decider_model": "gemini/gemini-3-flash-preview",
                "providers": {"gemini": {"api_key": ["k"]}},
                "models": {},
            },
            web_search_available=True,
            web_search_provider="tavily",
            exa_mcp_url="",
            actual_model="gpt-4o",
        ),
        is_googlelens_query_func=_fake_is_googlelens_query,
    )

    assert search_metadata == {"provider": "mock"}
    _assert_with_trailing_newline_tolerance(
        str(messages[0]["content"]),
        "latest news\n\n--- Search Results ---\nHeadline: Example",
    )


def test_search_formatting_tolerates_only_trailing_newline() -> None:
    expected = "latest news\n\n--- Search Results ---\nHeadline: Example"

    # Allowed: exact output or output with one or more trailing newlines.
    _assert_with_trailing_newline_tolerance(expected, expected)
    _assert_with_trailing_newline_tolerance(f"{expected}\n", expected)
    _assert_with_trailing_newline_tolerance(f"{expected}\n\n", expected)

    # Not allowed: whitespace mutation that changes formatting semantics.
    with pytest.raises(AssertionError):
        _assert_with_trailing_newline_tolerance(
            "latest news\n\n--- Search Results ---\nHeadline:  Example",
            expected,
        )


def test_decider_message_format_preserves_or_describes_files() -> None:
    # Non-Gemini decider: file parts are converted to a text description.
    base_messages: list[dict[str, object]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "summarize this"},
                {
                    "type": "file",
                    "file": {
                        "file_data": "data:application/pdf;base64,AA==",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="},
                },
            ],
        },
    ]

    openai_non_gemini = convert_messages_to_openai_format(
        base_messages,
        system_prompt=None,
        reverse=False,
        is_gemini=False,
    )
    assert openai_non_gemini
    content = openai_non_gemini[0]["content"]
    assert isinstance(content, list)
    assert any(
        part.get("type") == "text" and "pdf" in str(part.get("text", "")).lower()
        for part in content
    )

    # Gemini decider: file parts remain as file parts.
    openai_gemini = convert_messages_to_openai_format(
        base_messages,
        system_prompt=None,
        reverse=False,
        is_gemini=True,
    )
    assert openai_gemini
    gemini_content = openai_gemini[0]["content"]
    assert isinstance(gemini_content, list)
    assert any(part.get("type") == "file" for part in gemini_content)


@pytest.mark.asyncio
async def test_web_search_decider_normalizes_mapping_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_config: dict[str, object] = {}

    async def _fake_decide_web_search(
        _messages: list[dict[str, object]],
        decider_config: dict,
    ) -> dict[str, object]:
        captured_config.update(decider_config)
        return {"needs_search": False}

    def _fake_is_googlelens_query(*args: object, **kwargs: object) -> bool:
        return False

    monkeypatch.setattr(
        "llmcord.logic.search_logic.decide_web_search",
        _fake_decide_web_search,
    )
    monkeypatch.setattr("llmcord.logic.search_logic.get_bad_keys_db", lambda: _FakeDB())

    bot = _DummyBot()
    msg = FakeMessage(id=21, content="at ai hi", author=FakeUser(1234))
    messages: list[dict[str, object]] = [{"role": "user", "content": "hi"}]

    await resolve_search_metadata(
        SearchResolutionContext(
            new_msg=msg,  # type: ignore[arg-type]
            discord_bot=bot,  # type: ignore[arg-type]
            msg_nodes={},
            messages=messages,
            user_warnings=set(),
            tavily_api_keys=["tvly-TEST"],
            config={
                "web_search_provider": "tavily",
                "web_search_decider_model": (
                    "google-gemini-cli/gemini-3-flash-preview-minimal"
                ),
                "providers": {
                    "google-gemini-cli": {
                        "api_key": {
                            "refresh": "refresh-token",
                            "projectId": "project-123",
                        },
                    },
                },
                "models": {
                    "google-gemini-cli/gemini-3-flash-preview-minimal": {},
                },
            },
            web_search_available=True,
            web_search_provider="tavily",
            exa_mcp_url="",
            actual_model="gpt-4o",
        ),
        is_googlelens_query_func=_fake_is_googlelens_query,
    )

    api_keys = captured_config.get("api_keys")
    assert isinstance(api_keys, list)
    assert len(api_keys) == 1
    assert '"refresh":"refresh-token"' in str(api_keys[0])


@pytest.mark.asyncio
async def test_decider_google_gemini_cli_uses_native_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_stream_google_gemini_cli(**kwargs: object):
        assert kwargs.get("model") == "gemini-3-flash-preview-minimal"
        yield '{"needs_search":false}', None, False

    async def _fail_litellm_call(**_kwargs: object) -> object:
        msg = "litellm path should not be used for google-gemini-cli decider"
        raise AssertionError(msg)

    monkeypatch.setattr(
        "llmcord.services.search.decider.stream_google_gemini_cli",
        _fake_stream_google_gemini_cli,
    )
    monkeypatch.setattr(
        "llmcord.services.search.decider.litellm.acompletion",
        _fail_litellm_call,
    )

    result, exhausted = await _run_decider_once(
        [{"role": "user", "content": "hello"}],
        DeciderRunConfig(
            provider="google-gemini-cli",
            model="gemini-3-flash-preview-minimal",
            api_keys=["refresh-token"],
            base_url="https://cloudcode-pa.googleapis.com",
            extra_headers=None,
            model_parameters=None,
        ),
    )

    assert exhausted is False
    assert result == {"needs_search": False}
